"""CLI command for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from svg_agentic_slm.cli.overrides import parse_override_items, set_nested_override

console = Console()


def evaluate(
    config: Path = typer.Option(
        "configs/eval.yaml",
        "--config", "-c",
        help="Path to evaluation config file.",
    ),
    report_dir: Optional[Path] = typer.Option(
        None,
        "--report-dir", "-r",
        help="Override the configured directory for evaluation reports.",
    ),
    artifact_path: Optional[Path] = typer.Option(
        None,
        "--artifact-path", "-a",
        help="Artifact directory, sidecar JSON, or SVG artifact file to evaluate.",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        help="Override eval.max_samples for this run.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Override eval.seed for this run.",
    ),
    overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Nested config override in dotted.path=value form. Example: --set eval.max_samples=10",
    ),
) -> None:
    """Run evaluation on the SVG generation pipeline.

    Loads the evaluation configuration, runs the evaluator,
    and generates a report.
    """
    console.print(f"[bold blue]Evaluation[/bold blue]")
    console.print(f"Config: {config}")
    if report_dir is not None:
        console.print(f"Report directory override: {report_dir}")

    from svg_agentic_slm.eval.run_eval import run_evaluation
    from svg_agentic_slm.eval.report import generate_report

    try:
        cli_overrides = parse_override_items(overrides)
        if report_dir is not None:
            set_nested_override(cli_overrides, "eval.output_dir", str(report_dir))
        if artifact_path is not None:
            set_nested_override(cli_overrides, "eval.artifact_path", str(artifact_path))
        if max_samples is not None:
            set_nested_override(cli_overrides, "eval.max_samples", max_samples)
        if seed is not None:
            set_nested_override(cli_overrides, "eval.seed", seed)

        result = run_evaluation(config_path=config, overrides=cli_overrides)
        console.print(f"\n{result.summary()}")

        resolved_report_dir = Path(result.metadata["output_dir"])
        report_path = generate_report(result, output_dir=resolved_report_dir)
        console.print(f"\n[green]Report saved to: {report_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(code=1)
