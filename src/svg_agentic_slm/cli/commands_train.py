"""CLI command for LoRA training."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def train(
    config: Path = typer.Option(
        "configs/train_lora.yaml",
        "--config", "-c",
        help="Path to training config file.",
    ),
) -> None:
    """Run LoRA fine-tuning for text-to-SVG.

    Loads the training configuration and launches the SFT trainer.
    """
    console.print(f"[bold blue]LoRA Training[/bold blue]")
    console.print(f"Config: {config}")

    from svg_agentic_slm.train.train_text_to_svg import run_training

    try:
        run_training(config_path=config)
        console.print("[bold green]Training complete.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(code=1)
