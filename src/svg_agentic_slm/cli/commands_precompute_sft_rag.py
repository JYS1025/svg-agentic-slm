"""CLI for resumable MMSVG SFT hybrid-RAG precomputation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console

from svg_agentic_slm.data.mmsvg_sft_rag_precompute import (
    MMSVGSFTRAGPrecomputeConfig,
    precompute_mmsvg_sft_rag,
)

console = Console()


def precompute_sft_rag(
    input_dir: Path = typer.Option(
        Path("data/processed/mmsvg_sft_20k"),
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing prepared train/validation/test JSONL.",
    ),
    output_dir: Path = typer.Option(
        Path("data/processed/mmsvg_sft_20k_rag"),
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Atomic publication directory for normalized Parquet artifacts.",
    ),
    rag_config: Path = typer.Option(
        Path("configs/rag.yaml"),
        "--rag-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Hybrid MMSVG Chroma/SigLIP2 RAG configuration.",
    ),
    selection_manifest: Path = typer.Option(
        Path("data/sft_sample_ids/manifest.json"),
        "--selection-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Manifest for the fixed Icon 10K + Illustration 10K IDs.",
    ),
    instruction_policy: str = typer.Option(
        "r1",
        "--instruction-policy",
        help="prepared, r0, r1 (detail 60/description 40), or r2.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Process only the first N fixed samples for a smoke test.",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size",
        min=1,
        help=(
            "Number of SFT prompts encoded and reranked together. "
            "CUDA OOM automatically halves a failing batch."
        ),
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume compatible per-sample atomic checkpoints.",
    ),
) -> None:
    """Precompute self-leakage-safe hybrid RAG Top-3 for fixed SFT samples."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        manifest = precompute_mmsvg_sft_rag(
            MMSVGSFTRAGPrecomputeConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                rag_config_path=rag_config,
                selection_manifest_path=selection_manifest,
                instruction_policy=instruction_policy,
                limit=limit,
                resume=resume,
                retrieval_batch_size=batch_size,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]SFT RAG precompute failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[bold green]SFT RAG precompute completed.[/bold green]")
    console.print(json.dumps(manifest, ensure_ascii=True, indent=2))
