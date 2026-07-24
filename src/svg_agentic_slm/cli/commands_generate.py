"""CLI command for SVG generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from svg_agentic_slm.cli.overrides import parse_override_items, set_nested_override
from svg_agentic_slm.factories.generation import (
    build_generation_runtime,
    persist_generation_artifacts,
)

console = Console()


def generate(
    prompt: str = typer.Argument(
        ...,
        help="Natural language description of the SVG to generate.",
    ),
    config: Path = typer.Option(
        "configs/generation.yaml",
        "--config", "-c",
        help="Path to generation config file.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save the generated SVG file.",
    ),
    enable_rag: bool = typer.Option(
        False,
        "--rag",
        help="Enable RAG retrieval for context.",
    ),
    enable_critic: bool = typer.Option(
        False,
        "--critic",
        help="Enable critic feedback.",
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        "--max-new-tokens",
        help="Override generation.max_new_tokens for this run.",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Override generation.temperature for this run.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Override generation.seed for this run.",
    ),
    render_enabled: Optional[bool] = typer.Option(
        None,
        "--render/--no-render",
        help="Override generation.render.enabled for this run.",
    ),
    overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help=(
            "Nested config override in dotted.path=value form. "
            "Example: --set generation.top_p=0.8"
        ),
    ),
) -> None:
    """Generate an SVG from a text description.

    This command loads the generation config, builds the pipeline
    components, and runs the orchestrator.
    """
    console.print(f"[bold blue]SVG Generation[/bold blue]")
    console.print(f"Prompt: {prompt}")
    console.print(f"Config: {config}")
    console.print(f"RAG: {'enabled' if enable_rag else 'disabled'}")
    console.print(f"Critic: {'enabled' if enable_critic else 'disabled'}")

    try:
        cli_overrides = parse_override_items(overrides)
        if max_new_tokens is not None:
            set_nested_override(cli_overrides, "generation.max_new_tokens", max_new_tokens)
        if temperature is not None:
            set_nested_override(cli_overrides, "generation.temperature", temperature)
        if seed is not None:
            set_nested_override(cli_overrides, "generation.seed", seed)
        if render_enabled is not None:
            set_nested_override(cli_overrides, "generation.render.enabled", render_enabled)

        runtime = build_generation_runtime(
            config_path=config,
            prompt=prompt,
            enable_rag=enable_rag,
            enable_critic=enable_critic,
            output_path=output,
            overrides=cli_overrides,
        )
        result = runtime.orchestrator.run(runtime.request)
        artifacts = persist_generation_artifacts(result=result, runtime=runtime)
    except Exception as e:
        console.print(f"[bold red]Generation failed: {e}[/bold red]")
        raise typer.Exit(code=1)

    console.print("\n[bold green]Generated SVG:[/bold green]")
    console.print(result.generated_svg)
    console.print(f"\nValid SVG: {result.is_valid}")
    final_attempt = result.attempts[-1] if result.attempts else None
    if final_attempt is not None:
        console.print(f"Outcome: {final_attempt.metadata.get('outcome', 'unknown')}")
        console.print(f"Stop reason: {final_attempt.metadata.get('stop_reason', 'unknown')}")

    if result.critic_feedback:
        latest_feedback = result.critic_feedback[-1]
        console.print(
            "Critic feedback: "
            f"{latest_feedback.critic_type} score={latest_feedback.score:.1f}"
        )

    console.print(f"\nSVG saved to: {artifacts.svg_path}")
    if artifacts.render_path is not None:
        console.print(f"Render saved to: {artifacts.render_path}")
    elif result.metadata.get("render", {}).get("enabled"):
        render_error = result.metadata.get("render", {}).get("error")
        console.print(f"[yellow]Render not produced: {render_error}[/yellow]")
    console.print(f"Metadata saved to: {artifacts.metadata_path}")
