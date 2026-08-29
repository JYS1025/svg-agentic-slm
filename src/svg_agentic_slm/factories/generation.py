"""Assembly helpers for the text-to-SVG generation runtime.

This module centralizes config loading, component wiring, and artifact path
planning so the CLI stays thin and collaborators can extend the pipeline
without rewriting command code. Durable artifact publication is delegated to
``svg_agentic_slm.artifacts.writer``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.generator import GeneratorAgent
from svg_agentic_slm.agents.llm_critic import LLMCritic
from svg_agentic_slm.agents.orchestrator import SVGGenerationOrchestrator
from svg_agentic_slm.agents.rag_agent import RAGAgent
from svg_agentic_slm.agents.rule_critic import RuleBasedCritic
from svg_agentic_slm.agents.schemas import (
    CriticFeedback,
    CriticInput,
    GenerationRequest,
    validate_critic_feedback,
)
from svg_agentic_slm.agents.vlm_critic import VLMCritic
from svg_agentic_slm.artifacts.writer import (
    GenerationArtifacts as GenerationArtifacts,
)
from svg_agentic_slm.artifacts.writer import (
    build_bundle_token,
)
from svg_agentic_slm.artifacts.writer import (
    persist_generation_artifacts as persist_generation_artifacts,
)
from svg_agentic_slm.cli.overrides import merge_nested_dicts
from svg_agentic_slm.models.base import BaseModelBackend
from svg_agentic_slm.models.gemma_loader import GemmaModelBackend
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.llama_cpp_backend import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_ID,
    LlamaCppModelBackend,
)
from svg_agentic_slm.models.openai_compatible_backend import OpenAICompatibleBackend
from svg_agentic_slm.models.transformers_text_backend import TransformersTextBackend
from svg_agentic_slm.models.transformers_vlm_backend import TransformersVLMBackend
from svg_agentic_slm.rag.base import BaseRetriever
from svg_agentic_slm.rag.chroma_store import ChromaRetriever
from svg_agentic_slm.rag.document_loader import load_svg_corpus
from svg_agentic_slm.rag.qdrant_store import QdrantRetriever
from svg_agentic_slm.svg.gates import SmokeRenderGate
from svg_agentic_slm.svg.labeler import CriticLabeler
from svg_agentic_slm.svg.renderer import CairoSVGRenderer
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.paths import get_config_dir
from svg_agentic_slm.utils.seed import set_seed

logger = logging.getLogger(__name__)

GROUNDED_VISUAL_CRITIC_TYPES = frozenset({"vlm", "rule_vlm", "critic_v1"})


@dataclass
class GenerationRuntime:
    """Assembled runtime dependencies and resolved settings for generation."""

    orchestrator: SVGGenerationOrchestrator
    model_backend: BaseModelBackend
    request: GenerationRequest
    output_dir: Path
    render_output_dir: Path
    svg_output_path: Path
    metadata_output_path: Path
    render_output_path: Path | None
    generation_config: dict[str, Any]
    model_config: dict[str, Any]
    rag_config: dict[str, Any]
    paths_config: dict[str, Any]
    config_paths: dict[str, str]
    enable_rag: bool
    enable_critic: bool
    enable_render: bool
    critic_type: str | None
    render_config: dict[str, Any]
    run_id: str
    critic_model_config: dict[str, Any] | None = None
    critic_model_backend: BaseModelBackend | None = None
    execution_command: list[str] | None = None
    benchmark_hash: str | None = None


class CompositeCritic(BaseCritic):
    """Aggregate multiple critics behind the single-critic orchestrator contract."""

    def __init__(self, critics: list[BaseCritic]) -> None:
        if not critics:
            raise ValueError("CompositeCritic requires at least one critic.")
        self._critics = critics

    @property
    def name(self) -> str:
        critic_names = ",".join(critic.name for critic in self._critics)
        return f"CompositeCritic[{critic_names}]"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return self._combine([
            validate_critic_feedback(critic.critique(instruction, svg_content))
            for critic in self._critics
        ])

    def critique_attempt(self, value: CriticInput) -> CriticFeedback:
        return self._combine([
            validate_critic_feedback(critic.critique_attempt(value))
            for critic in self._critics
        ])

    def _combine(self, feedback_items: list[CriticFeedback]) -> CriticFeedback:
        raw_sections: list[str] = []
        for item in feedback_items:
            section = (
                f"[{item.critic_type}] score={item.score:.1f} valid={item.is_valid} "
                f"matches_instruction={item.matches_instruction} "
                f"issues={item.issues} suggestions={item.suggestions}"
            )
            if item.raw_response is not None:
                section += f"\n[{item.critic_type}] raw_response:\n{item.raw_response}"
            raw_sections.append(section)

        statuses = {item.status for item in feedback_items if item.status is not None}
        if "invalid" in statuses:
            status = "invalid"
        elif "revise" in statuses:
            status = "revise"
        elif statuses == {"pass"}:
            status = "pass"
        else:
            status = None

        structured_issues = []
        for item in feedback_items:
            for issue in item.structured_issues:
                if issue not in structured_issues:
                    structured_issues.append(issue)
        preserve = _unique_strings(
            entry for item in feedback_items for entry in item.preserve
        )
        model_calls = []
        seen_call_ids: set[str] = set()
        evidence_provenance: list[dict[str, Any]] = []
        for item in feedback_items:
            for call in item.model_calls:
                if call.critic_call_id not in seen_call_ids:
                    seen_call_ids.add(call.critic_call_id)
                    model_calls.append(call)
            raw_provenance = item.metadata.get("evidence_provenance", [])
            if isinstance(raw_provenance, dict):
                raw_provenance = [raw_provenance]
            if isinstance(raw_provenance, list):
                for record in raw_provenance:
                    if isinstance(record, dict) and record not in evidence_provenance:
                        evidence_provenance.append(dict(record))

        return validate_critic_feedback(
            CriticFeedback(
                score=sum(item.score for item in feedback_items) / len(feedback_items),
                is_valid=all(item.is_valid for item in feedback_items),
                matches_instruction=all(item.matches_instruction for item in feedback_items),
                issues=_unique_strings(issue for item in feedback_items for issue in item.issues),
                suggestions=_unique_strings(
                    suggestion for item in feedback_items for suggestion in item.suggestions
                ),
                critic_type="+".join(item.critic_type for item in feedback_items),
                raw_response="\n".join(raw_sections),
                critic_version="+".join(
                    item.critic_version or "unversioned" for item in feedback_items
                ),
                model_id=_join_optional_strings(
                    item.model_id for item in feedback_items
                ),
                model_revision=_join_optional_strings(
                    item.model_revision for item in feedback_items
                ),
                prompt_version=_join_optional_strings(
                    item.prompt_version for item in feedback_items
                ),
                status=status,
                structured_issues=structured_issues,
                preserve=preserve,
                schema_version=max(item.schema_version for item in feedback_items),
                metadata={
                    "children": [dict(item.metadata) for item in feedback_items],
                    "evidence_provenance": evidence_provenance,
                },
                model_calls=model_calls,
            ),
        )


def build_generation_runtime(
    config_path: str | Path,
    *,
    prompt: str,
    enable_rag: bool = False,
    enable_critic: bool = False,
    output_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> GenerationRuntime:
    """Build the config-driven runtime for the generate command."""
    config_path = Path(config_path)
    generation_wrapper = load_yaml_config(config_path)
    if overrides:
        generation_wrapper = merge_nested_dicts(generation_wrapper, overrides)
    generation_config = generation_wrapper.get("generation", {})

    model_path = (
        Path(model_config_path)
        if model_config_path is not None
        else _resolve_related_config(config_path, "model.yaml")
    )
    rag_path = _resolve_related_config(config_path, "rag.yaml")
    paths_path = _resolve_related_config(config_path, "paths.yaml")

    model_wrapper = load_yaml_config(model_path)
    rag_wrapper = load_yaml_config(rag_path)
    paths_wrapper = load_yaml_config(paths_path)
    if overrides:
        model_overrides = {
            key: overrides[key]
            for key in ("model", "critic_model")
            if key in overrides
        }
        if model_overrides:
            model_wrapper = merge_nested_dicts(model_wrapper, model_overrides)
        rag_wrapper = merge_nested_dicts(rag_wrapper, {"rag": overrides.get("rag", {})})
        paths_wrapper = merge_nested_dicts(paths_wrapper, {"paths": overrides.get("paths", {})})

    model_config = model_wrapper.get("model", {})
    critic_model_config = model_wrapper.get("critic_model")
    if not isinstance(model_config, dict):
        raise ValueError("model must be a mapping.")
    if critic_model_config is not None and not isinstance(critic_model_config, dict):
        raise ValueError("critic_model must be a mapping when provided.")
    if critic_model_config == {}:
        raise ValueError("critic_model must not be empty when provided.")
    validate_model_config_security(model_config)
    if critic_model_config is not None:
        validate_model_config_security(critic_model_config)
    rag_config = rag_wrapper.get("rag", {})
    validate_rag_config_security(rag_config)
    paths_config = paths_wrapper.get("paths", {})

    generation_params = GenerationConfig.from_dict(generation_config)
    if generation_params.seed is not None:
        set_seed(generation_params.seed)

    orchestration_config = generation_config.get("orchestration", {})
    rag_enabled = enable_rag or orchestration_config.get("enable_rag", False)
    enable_revision_rag = orchestration_config.get("enable_revision_rag", False)
    if not isinstance(enable_revision_rag, bool):
        raise TypeError("enable_revision_rag must be a boolean.")
    critic_enabled = enable_critic or orchestration_config.get("enable_critic", False)
    critic_type = (
        orchestration_config.get("critic_type", "critic_v1")
        if critic_enabled
        else None
    )
    critic_score_threshold = orchestration_config.get("critic_score_threshold", 3.0)
    if (
        not isinstance(critic_score_threshold, (int, float))
        or isinstance(critic_score_threshold, bool)
        or not 0.0 <= float(critic_score_threshold) <= 4.0
    ):
        raise ValueError("critic_score_threshold must be between 0 and 4.")
    critic_score_threshold = float(critic_score_threshold)
    if critic_type not in {
        None,
        "rule",
        "llm",
        "both",
        "vlm",
        "rule_vlm",
        "critic_v1",
    }:
        raise ValueError(f"Unsupported critic_type: {critic_type}")
    grounded_visual_critic = critic_type in GROUNDED_VISUAL_CRITIC_TYPES
    if grounded_visual_critic:
        if critic_model_config is None:
            raise ValueError(f"critic_type={critic_type!r} requires a nonempty critic_model.")
        critic_backend_type = str(critic_model_config.get("backend_type", "")).strip().lower()
        if critic_backend_type != "transformers_vlm":
            raise ValueError(
                f"critic_type={critic_type!r} requires "
                "critic_model.backend_type='transformers_vlm'."
            )

    output_dir = Path(paths_config.get("outputs", {}).get("generations", "./outputs/generations"))
    render_output_dir = Path(paths_config.get("outputs", {}).get("renders", "./outputs/renders"))
    render_config = _resolve_render_config(generation_config)
    svg_settings = generation_config.get("svg", {})
    run_id = f"run_{uuid4().hex}"
    artifact_paths = _resolve_artifact_paths(
        generations_dir=output_dir,
        renders_dir=render_output_dir,
        instruction=prompt,
        output_path=output_path,
        render_enabled=render_config["enabled"],
        render_format=render_config["output_format"],
        run_id=run_id,
    )
    svg_output_path = artifact_paths["svg"]
    metadata_output_path = artifact_paths["metadata"]
    if svg_output_path is None or metadata_output_path is None:
        raise RuntimeError("Generation artifact paths were not resolved.")

    model_backend = _build_model_backend(model_config, generation_params)
    # Load the native llama.cpp/CUDA runtime before RAG corpus validation can
    # import Conda's lxml and pin an older libstdc++ into this process.
    model_backend.load_model()
    rag_agent = _build_rag_agent(rag_config) if rag_enabled else None

    critic_model_backend = model_backend
    if (
        critic_enabled
        and critic_type in {"llm", "both", *GROUNDED_VISUAL_CRITIC_TYPES}
        and critic_model_config is not None
    ):
        separate_critic_backend: BaseModelBackend | None = None
        try:
            separate_critic_backend = _build_model_backend(
                critic_model_config,
                generation_params,
            )
            separate_critic_backend.load_model()
        except Exception:
            for backend in (separate_critic_backend, model_backend):
                if backend is None:
                    continue
                try:
                    backend.unload_model()
                except Exception:
                    logger.exception("Failed to unload a model after runtime setup failure.")
            raise
        critic_model_backend = separate_critic_backend

    generator = GeneratorAgent(
        model_backend,
        max_svg_length=svg_settings.get("max_svg_length", 8192),
        max_context_characters=svg_settings.get(
            "max_context_characters",
            12000,
        ),
        max_context_tokens=svg_settings.get("max_context_tokens"),
        enable_revision_rag=enable_revision_rag,
    )
    validator = SVGValidator()
    critic = (
        _build_critic(
            critic_type,
            validator,
            critic_model_backend,
            critic_model_config=critic_model_config,
            score_threshold=critic_score_threshold,
        )
        if critic_enabled
        else None
    )
    renderer = _build_renderer(render_config) if render_config["enabled"] else None
    critic_evidence_settings = critic_model_config or {}
    critic_render_width = critic_evidence_settings.get("render_width", 256)
    critic_render_height = critic_evidence_settings.get("render_height", 256)
    if (
        not isinstance(critic_render_width, int)
        or isinstance(critic_render_width, bool)
        or critic_render_width <= 0
        or not isinstance(critic_render_height, int)
        or isinstance(critic_render_height, bool)
        or critic_render_height <= 0
    ):
        raise ValueError("Critic evidence render dimensions must be positive integers.")
    critic_labeler = CriticLabeler() if grounded_visual_critic else None
    smoke_render_gate = (
        SmokeRenderGate(width=critic_render_width, height=critic_render_height)
        if grounded_visual_critic
        else None
    )

    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=validator,
        renderer=renderer,
        critic=critic,
        rag_agent=rag_agent,
        max_revisions=orchestration_config.get("max_revision_rounds", 2),
        output_dir=output_dir,
        render_output_path=artifact_paths["render"],
        render_width=render_config["width"] or svg_settings.get("target_width", 256),
        render_height=render_config["height"] or svg_settings.get("target_height", 256),
        render_format=render_config["output_format"],
        critic_acceptance_score=orchestration_config.get(
            "critic_acceptance_score",
            8.0,
        ),
        critic_score_threshold=critic_score_threshold,
        critic_labeler=critic_labeler,
        smoke_render_gate=smoke_render_gate,
        require_visual_evidence=grounded_visual_critic,
        max_no_improvement_rounds=orchestration_config.get(
            "max_no_improvement_rounds",
            1,
        ),
        min_critic_score_improvement=orchestration_config.get(
            "min_critic_score_improvement",
            0.1,
        ),
    )

    request = GenerationRequest(
        instruction=prompt,
        config_overrides=generation_params.to_dict(),
        run_id=run_id,
    )

    return GenerationRuntime(
        orchestrator=orchestrator,
        model_backend=model_backend,
        request=request,
        output_dir=output_dir,
        render_output_dir=render_output_dir,
        svg_output_path=svg_output_path,
        metadata_output_path=metadata_output_path,
        render_output_path=artifact_paths["render"],
        generation_config=generation_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        critic_model_backend=(
            critic_model_backend if critic_model_backend is not model_backend else None
        ),
        rag_config=_redact_sensitive_config(rag_config),
        paths_config=paths_config,
        config_paths={
            "generation": str(config_path),
            "model": str(model_path),
            "rag": str(rag_path),
            "paths": str(paths_path),
        },
        enable_rag=rag_enabled,
        enable_critic=critic_enabled,
        enable_render=render_config["enabled"],
        critic_type=critic_type,
        render_config=render_config,
        run_id=run_id,
    )


def close_generation_runtime(runtime: GenerationRuntime) -> None:
    """Release model resources owned by an assembled generation runtime."""
    if (
        runtime.critic_model_backend is not None
        and runtime.critic_model_backend is not runtime.model_backend
    ):
        critic_unload = getattr(runtime.critic_model_backend, "unload_model", None)
        if callable(critic_unload):
            critic_unload()
    unload = getattr(runtime.model_backend, "unload_model", None)
    if callable(unload):
        unload()


def _resolve_related_config(config_path: Path, filename: str) -> Path:
    sibling = config_path.parent / filename
    if sibling.exists():
        return sibling
    fallback = get_config_dir() / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Related config file not found: {filename}")


def _build_critic(
    critic_type: str | None,
    validator: SVGValidator,
    model_backend: BaseModelBackend,
    *,
    critic_model_config: dict[str, Any] | None = None,
    score_threshold: float = 3.0,
) -> BaseCritic:
    if critic_type == "rule":
        return RuleBasedCritic(validator)
    if critic_type == "llm":
        return LLMCritic(model_backend)
    if critic_type == "both":
        return CompositeCritic(
            [
                RuleBasedCritic(validator),
                LLMCritic(model_backend),
            ]
        )
    if critic_type in GROUNDED_VISUAL_CRITIC_TYPES:
        if not isinstance(model_backend, TransformersVLMBackend):
            raise TypeError("VLM critics require a TransformersVLMBackend.")
        settings = critic_model_config or {}
        vlm_critic = VLMCritic(
            model_backend,
            CairoSVGRenderer(),
            render_width=settings.get("render_width", 512),
            render_height=settings.get("render_height", 512),
            background_color=settings.get("background_color", "#ffffff"),
            max_new_tokens=settings.get("max_new_tokens", 2048),
            score_threshold=score_threshold,
        )
        return vlm_critic
    raise ValueError(f"Unsupported critic_type: {critic_type}")


def _build_model_backend(
    model_config: dict[str, Any],
    generation_config: GenerationConfig,
) -> BaseModelBackend:
    if not all(isinstance(key, str) for key in model_config):
        raise ValueError("Model config keys must be strings.")
    backend_type = str(model_config.get("backend_type", "llama_cpp")).strip().lower()
    llama_cpp_keys = {
        "backend_type",
        "chat_format",
        "conversion_runtime",
        "filename",
        "flash_attn",
        "model_id",
        "model_path",
        "measure_streaming_metrics",
        "main_gpu",
        "n_batch",
        "n_ctx",
        "n_gpu_layers",
        "quantization",
        "quantization_provider",
        "revision",
        "split_mode",
        "upstream_model_id",
        "use_mmap",
        "verbose",
    }
    openai_compatible_keys = {
        "allow_insecure_http",
        "api_key_env",
        "backend_type",
        "base_url",
        "engine",
        "max_retries",
        "model_id",
        "revision",
        "timeout_seconds",
    }
    transformers_vlm_keys = {
        "attn_implementation",
        "auto_model_class",
        "backend_type",
        "background_color",
        "device",
        "do_sample",
        "dtype",
        "enable_thinking",
        "local_files_only",
        "max_new_tokens",
        "model_id",
        "render_height",
        "render_width",
        "revision",
        "token_env",
        "trust_remote_code",
    }
    transformers_text_keys = {
        "adapter_path",
        "attn_implementation",
        "auto_model_class",
        "backend_type",
        "codec_grid_size",
        "codec_manifest_path",
        "device",
        "dtype",
        "enable_thinking",
        "local_files_only",
        "model_id",
        "model_path",
        "output_format",
        "revision",
        "token_env",
        "tokenizer_path",
        "trust_remote_code",
    }
    if backend_type in {"llama_cpp", "gemma"}:
        supported_keys = llama_cpp_keys
    elif backend_type == "openai_compatible":
        supported_keys = openai_compatible_keys
    elif backend_type == "transformers_vlm":
        supported_keys = transformers_vlm_keys
    elif backend_type == "transformers_text":
        supported_keys = transformers_text_keys
    else:
        raise ValueError(f"Unsupported model backend_type: {backend_type}")

    unknown_keys = set(model_config) - supported_keys
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown model config option(s): {names}")

    if backend_type == "openai_compatible":
        return OpenAICompatibleBackend(
            base_url=model_config.get("base_url", ""),
            model_id=model_config.get("model_id", ""),
            api_key_env=model_config.get("api_key_env"),
            model_revision=model_config.get("revision"),
            engine=model_config.get("engine", ""),
            timeout_seconds=model_config.get("timeout_seconds", 120.0),
            max_retries=model_config.get("max_retries", 0),
            allow_insecure_http=model_config.get("allow_insecure_http", False),
            generation_config=generation_config,
        )
    if backend_type == "transformers_vlm":
        backend_kwargs: dict[str, Any] = {}
        for config_key, constructor_key in (
            ("model_id", "model_id"),
            ("revision", "model_revision"),
            ("device", "device"),
            ("dtype", "dtype"),
            ("attn_implementation", "attn_implementation"),
            ("auto_model_class", "auto_model_class"),
            ("max_new_tokens", "max_new_tokens"),
            ("do_sample", "do_sample"),
            ("enable_thinking", "enable_thinking"),
            ("local_files_only", "local_files_only"),
            ("trust_remote_code", "trust_remote_code"),
            ("token_env", "token_env"),
        ):
            if config_key in model_config:
                backend_kwargs[constructor_key] = model_config[config_key]
        return TransformersVLMBackend(**backend_kwargs)
    if backend_type == "transformers_text":
        backend_kwargs = {"generation_config": generation_config}
        for config_key, constructor_key in (
            ("model_id", "model_id"),
            ("revision", "model_revision"),
            ("model_path", "model_path"),
            ("adapter_path", "adapter_path"),
            ("tokenizer_path", "tokenizer_path"),
            ("output_format", "output_format"),
            ("codec_manifest_path", "codec_manifest_path"),
            ("codec_grid_size", "codec_grid_size"),
            ("auto_model_class", "auto_model_class"),
            ("device", "device"),
            ("dtype", "dtype"),
            ("attn_implementation", "attn_implementation"),
            ("enable_thinking", "enable_thinking"),
            ("local_files_only", "local_files_only"),
            ("trust_remote_code", "trust_remote_code"),
            ("token_env", "token_env"),
        ):
            if config_key in model_config:
                backend_kwargs[constructor_key] = model_config[config_key]
        return TransformersTextBackend(**backend_kwargs)

    backend_class = LlamaCppModelBackend if backend_type == "llama_cpp" else GemmaModelBackend
    return backend_class(
        model_id=model_config.get("model_id", DEFAULT_MODEL_ID),
        filename=model_config.get("filename", DEFAULT_MODEL_FILE),
        model_revision=model_config.get("revision"),
        model_path=model_config.get("model_path"),
        upstream_model_id=model_config.get("upstream_model_id"),
        quantization=model_config.get("quantization"),
        quantization_provider=model_config.get("quantization_provider"),
        conversion_runtime=model_config.get("conversion_runtime"),
        n_ctx=model_config.get("n_ctx", 8192),
        n_gpu_layers=model_config.get("n_gpu_layers", -1),
        main_gpu=model_config.get("main_gpu", 0),
        split_mode=model_config.get("split_mode", "layer"),
        n_batch=model_config.get("n_batch", 512),
        flash_attn=model_config.get("flash_attn", True),
        use_mmap=model_config.get("use_mmap", True),
        verbose=model_config.get("verbose", False),
        chat_format=model_config.get("chat_format"),
        measure_streaming_metrics=model_config.get("measure_streaming_metrics", False),
        generation_config=generation_config,
    )


def validate_model_config_security(model_config: dict[str, Any]) -> None:
    """Reject inline credentials before config is logged or persisted."""
    if not all(isinstance(key, str) for key in model_config):
        raise ValueError("Model config keys must be strings.")
    sensitive_paths = _find_sensitive_config_paths(model_config, prefix="model")
    if sensitive_paths:
        names = ", ".join(sensitive_paths)
        raise ValueError(
            f"Inline model credentials are not allowed ({names}); use api_key_env."
        )


def build_rag_retriever(
    rag_config: dict[str, Any],
    *,
    index_chroma_corpus: bool = True,
) -> BaseRetriever:
    """Build the configured retrieval backend without changing consumers."""
    validate_rag_config_security(rag_config)
    backend = str(rag_config.get("backend", "chromadb")).strip().lower()
    if backend in {"chroma", "chromadb"}:
        settings = _merge_backend_settings(rag_config, "chromadb")
        dataset_roots = settings.get("dataset_roots", {})
        if not isinstance(dataset_roots, dict):
            raise ValueError("rag.chromadb.dataset_roots must be a mapping.")
        embedding_dimension = settings.get("embedding_dimension")
        retriever = ChromaRetriever(
            collection_name=settings.get("collection_name", "svg_patterns"),
            persist_directory=settings.get(
                "persist_directory",
                "./data/chroma_db",
            ),
            embedding_model=settings.get(
                "embedding_model",
                "all-MiniLM-L6-v2",
            ),
            similarity_threshold=settings.get("similarity_threshold", 0.0),
            embedding_revision=settings.get("embedding_revision"),
            embedding_dimension=(
                int(embedding_dimension) if embedding_dimension is not None else None
            ),
            query_instruction=settings.get("query_instruction"),
            device=settings.get("device", "cuda:0"),
            dataset_roots=dataset_roots,
            precomputed_embeddings=bool(settings.get("precomputed_embeddings", False)),
            overfetch_factor=int(settings.get("overfetch_factor", 5)),
            document_field=settings.get("document_field", "description"),
        )
        corpus_path = settings.get("corpus_path")
        if index_chroma_corpus and corpus_path:
            load_svg_corpus(corpus_path, retriever)
        return retriever

    if backend == "qdrant":
        settings = _merge_backend_settings(rag_config, "qdrant")
        return QdrantRetriever(
            collection_name=settings.get(
                "collection_name",
                "svg_text2svg_stack_minilm384_v1",
            ),
            embedding_model=settings.get(
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            similarity_threshold=settings.get("similarity_threshold", 0.0),
            url_env=settings.get("url_env", "QDRANT_URL"),
            api_key_env=settings.get("api_key_env", "QDRANT_API_KEY"),
            timeout_seconds=settings.get("timeout_seconds", 120.0),
            upload_batch_size=settings.get("upload_batch_size", 64),
            compress_svg=settings.get("compress_svg", True),
            on_disk_vectors=settings.get("on_disk_vectors", True),
            on_disk_payload=settings.get("on_disk_payload", True),
            on_disk_hnsw=settings.get("on_disk_hnsw", True),
            scalar_quantization=settings.get("scalar_quantization", True),
        )

    raise ValueError(f"Unsupported RAG backend: {backend!r}. Choose 'chromadb' or 'qdrant'.")


def _build_rag_agent(rag_config: dict[str, Any]) -> RAGAgent:
    retriever = build_rag_retriever(rag_config)
    if isinstance(retriever, QdrantRetriever):
        retriever.preflight()
    return RAGAgent(
        retriever=retriever,
        top_k=rag_config.get("top_k", 3),
    )


def _merge_backend_settings(
    rag_config: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    nested = rag_config.get(backend, {})
    if not isinstance(nested, dict):
        raise ValueError(f"rag.{backend} must be a mapping.")
    return {**rag_config, **nested}


def validate_rag_config_security(rag_config: dict[str, Any]) -> None:
    """Reject inline credentials that could be persisted in run artifacts."""
    sensitive_paths = list(_find_sensitive_config_paths(rag_config))
    if sensitive_paths:
        joined = ", ".join(sensitive_paths)
        raise ValueError(
            "RAG secrets must be supplied through environment variables, "
            f"not config values: {joined}"
        )


def _find_sensitive_config_paths(
    value: Any,
    *,
    prefix: str = "rag",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if _is_sensitive_config_key(key_text):
                paths.append(path)
            else:
                paths.extend(_find_sensitive_config_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(
                _find_sensitive_config_paths(
                    nested,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return paths


def _redact_sensitive_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _is_sensitive_config_key(str(key))
                else _redact_sensitive_config(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    exact_names = {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "token",
        "password",
        "secret",
        "private_key",
    }
    suffixes = (
        "_api_key",
        "_credential",
        "_credentials",
        "_token",
        "_password",
        "_secret",
        "_private_key",
    )
    return normalized in exact_names or normalized.endswith(suffixes)


def _build_renderer(render_config: dict[str, Any]) -> CairoSVGRenderer:
    backend = render_config["backend"]
    if backend == "cairosvg":
        return CairoSVGRenderer()
    raise ValueError(f"Unsupported render backend: {backend}")


def _resolve_render_config(generation_config: dict[str, Any]) -> dict[str, Any]:
    render_config = generation_config.get("render", {})
    return {
        "enabled": render_config.get("enabled", True),
        "backend": render_config.get("backend", "cairosvg"),
        "output_format": render_config.get("output_format", "png"),
        "width": render_config.get("width"),
        "height": render_config.get("height"),
    }


def _resolve_artifact_paths(
    generations_dir: Path,
    renders_dir: Path,
    instruction: str,
    output_path: str | Path | None,
    render_enabled: bool,
    render_format: str,
    run_id: str | None = None,
) -> dict[str, Path | None]:
    if output_path is not None:
        path = Path(output_path)
        if path.suffix and path.suffix.lower() != ".svg":
            raise ValueError("Generation output path must use the .svg extension.")
        svg_path = path.with_suffix(".svg")
        metadata_path = svg_path.with_suffix(".json")
        run_suffix = build_bundle_token(run_id or f"run_{uuid4().hex}")[:12]
        render_path = (
            _build_render_output_path(
                stem=f"{svg_path.stem}.{run_suffix}",
                render_dir=svg_path.parent,
                render_format=render_format,
                svg_suffix=svg_path.suffix,
            )
            if render_enabled
            else None
        )
        return {
            "svg": svg_path,
            "metadata": metadata_path,
            "render": render_path,
        }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    slug = _slugify_instruction(instruction)
    run_suffix = (run_id or f"run_{uuid4().hex}").removeprefix("run_")[:8]
    stem = f"{timestamp}_{slug}_{run_suffix}"
    return {
        "svg": generations_dir / f"{stem}.svg",
        "metadata": generations_dir / f"{stem}.json",
        "render": (
            _build_render_output_path(
                stem=stem,
                render_dir=renders_dir,
                render_format=render_format,
                svg_suffix=".svg",
            )
            if render_enabled
            else None
        ),
    }


def _build_render_output_path(
    stem: str,
    render_dir: Path,
    render_format: str,
    svg_suffix: str,
) -> Path:
    suffix = f".{render_format.lower()}"
    filename = f"{stem}{suffix}"
    if suffix == svg_suffix:
        filename = f"{stem}.rendered{suffix}"
    return render_dir / filename


def _slugify_instruction(instruction: str, max_length: int = 48) -> str:
    lowered = instruction.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or "generation"


def _unique_strings(items: Any) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def _join_optional_strings(items: Any) -> str | None:
    values = _unique_strings(item for item in items if item)
    return "+".join(values) or None
