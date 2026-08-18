"""Retrieval-only CLI for inspecting the RAG-to-Generator handoff."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from svg_agentic_slm.agents.generator import select_retrieval_context
from svg_agentic_slm.agents.rag_agent import RAGAgent
from svg_agentic_slm.agents.schemas import GenerationRequest
from svg_agentic_slm.factories.generation import build_rag_retriever
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.prompts.system_prompts import get_svg_generator_system_prompt
from svg_agentic_slm.prompts.text_to_svg import (
    INITIAL_PROMPT_VERSION,
    build_text_to_svg_prompt,
)
from svg_agentic_slm.utils.config import load_yaml_config


def rag_preview(
    prompt: str = typer.Argument(
        ...,
        help="Natural language SVG description used as the retrieval query.",
    ),
    rag_config_path: Path = typer.Option(
        "configs/rag.yaml",
        "--rag-config",
        "--config",
        "-c",
        help="Active RAG YAML configuration.",
    ),
    generation_config_path: Path = typer.Option(
        "configs/generation.yaml",
        "--generation-config",
        help="Generation YAML used for parameters and the context budget.",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        min=1,
        help="Override rag.top_k for this query.",
    ),
    prompt_preview: bool = typer.Option(
        False,
        "--prompt-preview",
        help="Include the exact post-context-budget Generator prompt.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run ID included in the typed GenerationRequest.",
    ),
) -> None:
    """Retrieve RAG context and print it without loading or calling the SLM."""
    try:
        payload = build_rag_preview_payload(
            prompt=prompt,
            rag_config_path=rag_config_path,
            generation_config_path=generation_config_path,
            top_k=top_k,
            include_prompt_preview=prompt_preview,
            run_id=run_id,
        )
    except Exception as exc:
        typer.echo(f"RAG preview failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_rag_preview_payload(
    *,
    prompt: str,
    rag_config_path: str | Path,
    generation_config_path: str | Path,
    top_k: int | None = None,
    include_prompt_preview: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the exact typed pre-Generator payload using retrieval only."""
    instruction = prompt.strip()
    if not instruction:
        raise ValueError("prompt must be non-empty.")

    rag_config = _load_mapping_section(rag_config_path, "rag")
    generation_config = _load_mapping_section(generation_config_path, "generation")
    effective_top_k = _resolve_top_k(top_k, rag_config.get("top_k", 3))
    max_context_characters = _resolve_max_context_characters(generation_config)

    retriever = build_rag_retriever(
        rag_config,
        index_chroma_corpus=False,
    )
    context = RAGAgent(retriever=retriever, top_k=effective_top_k).retrieve(instruction)
    generation_parameters = GenerationConfig.from_dict(generation_config).to_dict()
    request = GenerationRequest(
        instruction=instruction,
        config_overrides=generation_parameters,
        run_id=run_id,
    )

    payload: dict[str, Any] = {
        "request": asdict(request),
        "context": [asdict(item) for item in context],
    }
    if include_prompt_preview:
        selected_context, truncated_ids = select_retrieval_context(
            context,
            max_context_characters=max_context_characters,
        )
        payload["prompt_preview"] = {
            "prompt_version": INITIAL_PROMPT_VERSION,
            "max_context_characters": max_context_characters,
            "context_item_ids": [item.item_id for item in selected_context],
            "truncated_context_item_ids": truncated_ids,
            "system_prompt": get_svg_generator_system_prompt(),
            "prompt": build_text_to_svg_prompt(
                instruction=instruction,
                retrieved_examples=selected_context,
            ),
        }
    return payload


def _load_mapping_section(config_path: str | Path, section_name: str) -> dict[str, Any]:
    wrapper = load_yaml_config(config_path)
    section = wrapper.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"{config_path} must contain a '{section_name}' mapping.")
    return section


def _resolve_top_k(cli_value: int | None, configured_value: Any) -> int:
    value = cli_value if cli_value is not None else configured_value
    if isinstance(value, bool):
        raise ValueError("rag.top_k must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("rag.top_k must be a positive integer.") from exc
    if result <= 0:
        raise ValueError("rag.top_k must be a positive integer.")
    return result


def _resolve_max_context_characters(generation_config: dict[str, Any]) -> int:
    svg_config = generation_config.get("svg", {})
    if not isinstance(svg_config, dict):
        raise ValueError("generation.svg must be a mapping.")
    value = svg_config.get("max_context_characters", 12000)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("generation.svg.max_context_characters must be a non-negative integer.")
    return value
