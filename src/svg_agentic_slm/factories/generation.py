"""Assembly helpers for the text-to-SVG generation runtime.

This module centralizes config loading, component wiring, and artifact
management so the CLI stays thin and collaborators can extend the
pipeline without rewriting command code.
"""

from __future__ import annotations

import json
import os
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
from svg_agentic_slm.agents.schemas import CriticFeedback, GenerationRequest, GenerationResult
from svg_agentic_slm.cli.overrides import merge_nested_dicts
from svg_agentic_slm.models.gemma_loader import GemmaModelBackend
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.rag.chroma_store import ChromaRetriever
from svg_agentic_slm.svg.renderer import CairoSVGRenderer
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.paths import get_config_dir
from svg_agentic_slm.utils.seed import set_seed


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


@dataclass
class GenerationArtifacts:
    """Paths to files produced by a generation run."""

    svg_path: Path
    metadata_path: Path
    render_path: Path | None


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
            critic.critique(instruction, svg_content)
            for critic in self._critics
        ]
        raw_sections = [
            f"[{item.critic_type}] score={item.score:.1f} valid={item.is_valid} "
            f"matches_instruction={item.matches_instruction} "
            f"issues={item.issues} suggestions={item.suggestions}"
            for item in feedback_items
        ]
        return CriticFeedback(
            score=sum(item.score for item in feedback_items) / len(feedback_items),
            is_valid=all(item.is_valid for item in feedback_items),
            matches_instruction=all(item.matches_instruction for item in feedback_items),
            issues=_unique_strings(
                issue
                for item in feedback_items
                for issue in item.issues
            ),
            suggestions=_unique_strings(
                suggestion
                for item in feedback_items
                for suggestion in item.suggestions
            ),
            critic_type="+".join(item.critic_type for item in feedback_items),
            raw_response="\n".join(raw_sections),
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

    output_dir = Path(
        paths_config.get("outputs", {}).get("generations", "./outputs/generations")
    )
    render_output_dir = Path(
        paths_config.get("outputs", {}).get("renders", "./outputs/renders")
    )
    render_config = _resolve_render_config(generation_config)
    svg_settings = generation_config.get("svg", {})
    artifact_paths = _resolve_artifact_paths(
        generations_dir=output_dir,
        renders_dir=render_output_dir,
        instruction=prompt,
        output_path=output_path,
        render_enabled=render_config["enabled"],
        render_format=render_config["output_format"],
    )
    svg_output_path = artifact_paths["svg"]
    metadata_output_path = artifact_paths["metadata"]
    if svg_output_path is None or metadata_output_path is None:
        raise RuntimeError("Generation artifact paths were not resolved.")

    model_backend = GemmaModelBackend(
        model_id=model_config.get("model_id", "google/gemma-3-4b-it"),
        device_map=model_config.get("device_map", "auto"),
        torch_dtype=model_config.get("torch_dtype", "bfloat16"),
        generation_config=generation_params,
        trust_remote_code=model_config.get("trust_remote_code", False),
        load_in_4bit=model_config.get("load_in_4bit", False),
        load_in_8bit=model_config.get("load_in_8bit", False),
        attn_implementation=model_config.get("attn_implementation"),
        max_memory=model_config.get("max_memory"),
    )
    model_backend.load_model()

    generator = GeneratorAgent(model_backend)
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
    )

    request = GenerationRequest(
        instruction=prompt,
        config_overrides=generation_params.to_dict(),
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
    )


def persist_generation_artifacts(
    result: GenerationResult,
    runtime: GenerationRuntime,
) -> GenerationArtifacts:
    """Save the generated SVG and a JSON sidecar with run metadata."""
    runtime.svg_output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.svg_output_path.write_text(result.generated_svg, encoding="utf-8")

    metadata_dir = runtime.metadata_output_path.parent.resolve()
    svg_reference = _relative_artifact_reference(runtime.svg_output_path, metadata_dir)
    render_reference = (
        _relative_artifact_reference(Path(result.render_path), metadata_dir)
        if result.render_path
        else None
    )
    metadata_payload = {
        "instruction": result.instruction,
        "svg_path": svg_reference,
        "is_valid": result.is_valid,
        "render_path": render_reference,
        "revision_count": result.revision_count,
        "critic_feedback": [
            {
                "score": feedback.score,
                "is_valid": feedback.is_valid,
                "matches_instruction": feedback.matches_instruction,
                "issues": feedback.issues,
                "suggestions": feedback.suggestions,
                "critic_type": feedback.critic_type,
                "raw_response": feedback.raw_response,
            }
            for feedback in result.critic_feedback
        ],
        "runtime": {
            "enable_rag": runtime.enable_rag,
            "enable_critic": runtime.enable_critic,
            "enable_render": runtime.enable_render,
            "critic_type": runtime.critic_type,
            "config_paths": runtime.config_paths,
            "generation_config": runtime.generation_config,
            "model_config": runtime.model_config,
            "rag_config": runtime.rag_config if runtime.enable_rag else {},
            "render_config": runtime.render_config,
            "planned_artifacts": {
                "svg_path": str(runtime.svg_output_path),
                "metadata_path": str(runtime.metadata_output_path),
                "render_path": (
                    str(runtime.render_output_path) if runtime.render_output_path else None
                ),
            },
        },
        "metadata": result.metadata,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    runtime.metadata_output_path.write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return GenerationArtifacts(
        svg_path=runtime.svg_output_path,
        metadata_path=runtime.metadata_output_path,
        render_path=Path(result.render_path) if result.render_path else None,
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
    model_backend: GemmaModelBackend,
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
) -> dict[str, Path | None]:
    if output_path is not None:
        path = Path(output_path)
        if path.suffix and path.suffix.lower() != ".svg":
            raise ValueError("Generation output path must use the .svg extension.")
        svg_path = path.with_suffix(".svg")
        metadata_path = svg_path.with_suffix(".json")
        render_path = (
            _build_render_output_path(
                stem=svg_path.stem,
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
    stem = f"{timestamp}_{slug}_{uuid4().hex[:8]}"
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


def _relative_artifact_reference(path: Path, metadata_dir: Path) -> str:
    """Return a sidecar-relative path so artifact bundles remain portable."""
    return os.path.relpath(path.resolve(), start=metadata_dir)
