"""CLI command for evaluation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def evaluate(
    config: Path = typer.Option(
        "configs/eval.yaml",
        "--config", "-c",
        help="Path to evaluation config file.",
    ),
    report_dir: Path = typer.Option(
        "outputs/eval_reports",
        "--report-dir", "-r",
        help="Directory for evaluation reports.",
    ),
) -> None:
    """Run evaluation on the SVG generation pipeline.

    Loads the evaluation configuration, runs the evaluator,
    and generates a report.
    """
    console.print(f"[bold blue]Evaluation[/bold blue]")
    console.print(f"Config: {config}")
    console.print(f"Report directory: {report_dir}")

    from svg_agentic_slm.eval.run_eval import run_evaluation
    from svg_agentic_slm.eval.report import generate_report

    try:
        result = run_evaluation(config_path=config)
        console.print(f"\n{result.summary()}")

        report_path = generate_report(result, output_dir=report_dir)
        console.print(f"\n[green]Report saved to: {report_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(code=1)
