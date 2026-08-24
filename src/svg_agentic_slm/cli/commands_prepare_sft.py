"""CLI command for preparing the auditable MMSVG SFT split."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from svg_agentic_slm.data.mmsvg_sft import (
    load_preparation_config,
    prepare_mmsvg_sft_dataset,
)

console = Console()


def prepare_sft(
    config: Path = typer.Option(
        Path("configs/data_mmsvg_sft.yaml"),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="MMSVG SFT preparation YAML configuration.",
    ),
) -> None:
    """Filter, balance, split, and serialize MMSVG records for SFT."""
    try:
        preparation_config = load_preparation_config(config)
        manifest = prepare_mmsvg_sft_dataset(preparation_config)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[red]SFT preparation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(json.dumps(manifest, ensure_ascii=True, indent=2, default=str))
