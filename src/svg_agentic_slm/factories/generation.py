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
    GenerationRequest,
    validate_critic_feedback,
)
from svg_agentic_slm.artifacts.writer import (
    GenerationArtifacts,
    build_bundle_token,
    persist_generation_artifacts,
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
from svg_agentic_slm.rag.chroma_store import ChromaRetriever
from svg_agentic_slm.svg.renderer import CairoSVGRenderer
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.paths import get_config_dir
from svg_agentic_slm.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class GenerationRuntime:
    """Assembled runtime dependencies and resolved settings for generation."""

    orchestrator: SVGGenerationOrchestrator
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
        feedback_items = [
            validate_critic_feedback(critic.critique(instruction, svg_content))
            for critic in self._critics
        ]
        raw_sections = [
            f"[{item.critic_type}] score={item.score:.1f} valid={item.is_valid} "
            f"matches_instruction={item.matches_instruction} "
            f"issues={item.issues} suggestions={item.suggestions}"
            for item in feedback_items
        ]
        return validate_critic_feedback(
            CriticFeedback(
                score=sum(item.score for item in feedback_items) / len(feedback_items),
                is_valid=all(item.is_valid for item in feedback_items),
                matches_instruction=all(
                    item.matches_instruction for item in feedback_items
                ),
                issues=_unique_strings(
                    issue for item in feedback_items for issue in item.issues
                ),
                suggestions=_unique_strings(
                    suggestion
                    for item in feedback_items
                    for suggestion in item.suggestions
                ),
                critic_type="+".join(item.critic_type for item in feedback_items),
                raw_response="\n".join(raw_sections),
                critic_version="+".join(
                    item.critic_version or "unversioned" for item in feedback_items
                ),
            ),
        )


def build_generation_runtime(
    config_path: str | Path,
    *,
    prompt: str,
    enable_rag: bool = False,
    enable_critic: bool = False,
    output_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> GenerationRuntime:
    """Build the config-driven runtime for the generate command."""
    config_path = Path(config_path)
    generation_wrapper = load_yaml_config(config_path)
    if overrides:
        generation_wrapper = merge_nested_dicts(generation_wrapper, overrides)
    generation_config = generation_wrapper.get("generation", {})

    model_path = _resolve_related_config(config_path, "model.yaml")
    rag_path = _resolve_related_config(config_path, "rag.yaml")
    paths_path = _resolve_related_config(config_path, "paths.yaml")

    model_wrapper = load_yaml_config(model_path)
    rag_wrapper = load_yaml_config(rag_path)
    paths_wrapper = load_yaml_config(paths_path)
    if overrides:
        model_wrapper = merge_nested_dicts(model_wrapper, {"model": overrides.get("model", {})})
        rag_wrapper = merge_nested_dicts(rag_wrapper, {"rag": overrides.get("rag", {})})
        paths_wrapper = merge_nested_dicts(paths_wrapper, {"paths": overrides.get("paths", {})})

    model_config = model_wrapper.get("model", {})
    rag_config = rag_wrapper.get("rag", {})
    paths_config = paths_wrapper.get("paths", {})

    generation_params = GenerationConfig.from_dict(generation_config)
    if generation_params.seed is not None:
        set_seed(generation_params.seed)

    orchestration_config = generation_config.get("orchestration", {})
    rag_enabled = enable_rag or orchestration_config.get("enable_rag", False)
    critic_enabled = enable_critic or orchestration_config.get("enable_critic", False)
    critic_type = orchestration_config.get("critic_type", "rule") if critic_enabled else None

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
    model_backend.load_model()

    generator = GeneratorAgent(
        model_backend,
        max_svg_length=svg_settings.get("max_svg_length", 8192),
        max_context_characters=svg_settings.get(
            "max_context_characters",
            12000,
        ),
    )
    validator = SVGValidator()
    critic = _build_critic(critic_type, validator, model_backend) if critic_enabled else None
    rag_agent = _build_rag_agent(rag_config) if rag_enabled else None
    renderer = _build_renderer(render_config) if render_config["enabled"] else None

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
    )

    request = GenerationRequest(
        instruction=prompt,
        config_overrides=generation_params.to_dict(),
        run_id=run_id,
    )

    return GenerationRuntime(
        orchestrator=orchestrator,
        request=request,
        output_dir=output_dir,
        render_output_dir=render_output_dir,
        svg_output_path=svg_output_path,
        metadata_output_path=metadata_output_path,
        render_output_path=artifact_paths["render"],
        generation_config=generation_config,
        model_config=model_config,
        rag_config=rag_config,
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
    raise ValueError(f"Unsupported critic_type: {critic_type}")


def _build_model_backend(
    model_config: dict[str, Any],
    generation_config: GenerationConfig,
) -> BaseModelBackend:
    supported_keys = {
        "backend_type",
        "chat_format",
        "conversion_runtime",
        "filename",
        "flash_attn",
        "model_id",
        "model_path",
        "n_batch",
        "n_ctx",
        "n_gpu_layers",
        "quantization",
        "quantization_provider",
        "revision",
        "upstream_model_id",
        "use_mmap",
        "verbose",
    }
    unknown_keys = set(model_config) - supported_keys
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown model config option(s): {names}")

    backend_type = model_config.get("backend_type", "llama_cpp")
    if backend_type not in ("llama_cpp", "gemma"):
        raise ValueError(f"Unsupported model backend_type: {backend_type}")
    backend_class = (
        LlamaCppModelBackend if backend_type == "llama_cpp" else GemmaModelBackend
    )
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
        n_batch=model_config.get("n_batch", 512),
        flash_attn=model_config.get("flash_attn", True),
        use_mmap=model_config.get("use_mmap", True),
        verbose=model_config.get("verbose", False),
        chat_format=model_config.get("chat_format"),
        generation_config=generation_config,
    )


def _build_rag_agent(rag_config: dict[str, Any]) -> RAGAgent:
    retriever = ChromaRetriever(
        collection_name=rag_config.get("collection_name", "svg_patterns"),
        persist_directory=rag_config.get("persist_directory", "./data/chroma_db"),
        embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
    )
    return RAGAgent(
        retriever=retriever,
        top_k=rag_config.get("top_k", 3),
    )


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
