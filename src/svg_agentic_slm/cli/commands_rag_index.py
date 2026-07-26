"""CLI command for streaming an SVG dataset into Qdrant."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from svg_agentic_slm.factories.generation import build_rag_retriever
from svg_agentic_slm.rag.hf_indexer import (
    DEFAULT_DATASET_REVISION,
    index_huggingface_svg_dataset,
)
from svg_agentic_slm.rag.qdrant_store import QdrantRetriever
from svg_agentic_slm.utils.config import load_yaml_config

console = Console()


def rag_index(
    config: Path = typer.Option(
        "configs/rag.yaml",
        "--config",
        "-c",
        help="Path to the RAG config file.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help=(
            "Target Qdrant collection size. Existing stable IDs are upserted, so reruns are safe."
        ),
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Embedding and upload batch size override.",
    ),
) -> None:
    """Stream caption/SVG pairs to Qdrant without a full local download."""
    console.print("[bold blue]Qdrant SVG Dataset Indexing[/bold blue]")
    console.print(f"Config: {config}")

    try:
        wrapper = load_yaml_config(config)
        rag_config = wrapper.get("rag", {})
        if not isinstance(rag_config, dict):
            raise ValueError("The 'rag' config must be a mapping.")

        indexing = rag_config.get("indexing", {})
        if not isinstance(indexing, dict):
            raise ValueError("rag.indexing must be a mapping.")
        qdrant_config = rag_config.get("qdrant", {})
        if not isinstance(qdrant_config, dict):
            raise ValueError("rag.qdrant must be a mapping.")

        target = int(limit if limit is not None else indexing.get("index_limit", 100_000))
        upload_batch = int(batch_size if batch_size is not None else indexing.get("batch_size", 64))
        if target <= 0 or upload_batch <= 0:
            raise ValueError("index limit and batch size must be positive.")
        rag_config = {
            **rag_config,
            "backend": "qdrant",
            "qdrant": {
                **qdrant_config,
                "upload_batch_size": upload_batch,
            },
        }

        retriever = build_rag_retriever(
            rag_config,
            index_chroma_corpus=False,
        )
        if not isinstance(retriever, QdrantRetriever):
            raise TypeError("Configured retriever is not Qdrant.")
        if qdrant_config.get("optimize_existing_collection", True):
            retriever.optimize_storage()

        console.print(f"Dataset: {indexing.get('dataset_id', 'starvector/text2svg-stack')}")
        console.print(f"Qdrant collection: {retriever.collection_name}")
        console.print(f"Target collection size: {target:,}")
        console.print("Streaming mode is enabled; the full dataset is not saved locally.")
        result = index_huggingface_svg_dataset(
            retriever,
            dataset_id=indexing.get(
                "dataset_id",
                "starvector/text2svg-stack",
            ),
            dataset_split=indexing.get("dataset_split", "train"),
            dataset_revision=indexing.get(
                "dataset_revision",
                DEFAULT_DATASET_REVISION,
            ),
            index_limit=target,
            batch_size=upload_batch,
            max_svg_chars=int(indexing.get("max_svg_chars", 24_000)),
            max_caption_chars=int(indexing.get("max_caption_chars", 1_200)),
            shuffle_buffer=int(indexing.get("shuffle_buffer", 2_000)),
            seed=int(indexing.get("seed", 42)),
        )
    except Exception as exc:
        console.print(f"[bold red]RAG indexing failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Qdrant indexing completed.[/bold green]")
    console.print(
        f"Collection count: {result.collection_count_before:,} -> {result.collection_count_after:,}"
    )
    console.print(
        f"Newly uploaded: {result.uploaded_this_run:,}; "
        f"scanned: {result.scanned_this_run:,}; "
        f"skipped: {result.skipped_this_run:,}"
    )
