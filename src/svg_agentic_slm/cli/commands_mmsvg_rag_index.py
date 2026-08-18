"""CLI command for local MMSVG description-only Chroma indexing."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from svg_agentic_slm.rag.embedding import (
    DEFAULT_QWEN3_EMBEDDING_MODEL,
    DEFAULT_QWEN3_EMBEDDING_REVISION,
    DEFAULT_SVG_QUERY_INSTRUCTION,
)
from svg_agentic_slm.rag.mmsvg_chroma_indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MMSVG_SOURCES,
    index_mmsvg_descriptions,
)
from svg_agentic_slm.utils.config import load_yaml_config

console = Console()


def mmsvg_rag_index(
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
        help="Index only the first N deterministic rows (for smoke tests).",
    ),
    batch_size: int = typer.Option(
        256,
        "--batch-size",
        min=1,
        help="GPU embedding and Chroma upload batch size.",
    ),
    device: str = typer.Option(
        "cuda:0",
        "--device",
        help="Torch device, for example cuda:0 or cpu.",
    ),
    persist_directory: Path | None = typer.Option(
        None,
        "--persist-directory",
        help="Override the Chroma persistence directory.",
    ),
    collection: str | None = typer.Option(
        None,
        "--collection",
        help="Override the Chroma collection name.",
    ),
) -> None:
    """Build a resumable Chroma index from raw MMSVG descriptions only."""
    console.print("[bold blue]MMSVG Description-only Chroma Indexing[/bold blue]")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        wrapper = load_yaml_config(config)
        rag_config = _mapping(wrapper.get("rag", {}), "rag")
        chroma_config = _mapping(rag_config.get("chromadb", {}), "rag.chromadb")
        indexing = _mapping(
            rag_config.get("mmsvg_indexing", {}),
            "rag.mmsvg_indexing",
        )
        resolved_persist = Path(
            persist_directory or chroma_config.get("persist_directory", "./data/chroma_db")
        )
        resolved_collection = str(
            collection or chroma_config.get("collection_name", DEFAULT_COLLECTION_NAME)
        )
        sources = tuple(
            replace(
                source,
                data_dir=Path(
                    indexing.get(
                        f"{source.dataset_type}_data_dir",
                        source.data_dir,
                    )
                ),
            )
            for source in DEFAULT_MMSVG_SOURCES
        )
        model_name = str(chroma_config.get("embedding_model", DEFAULT_QWEN3_EMBEDDING_MODEL))
        model_revision = str(
            chroma_config.get(
                "embedding_revision",
                DEFAULT_QWEN3_EMBEDDING_REVISION,
            )
        )
        query_instruction = str(
            chroma_config.get("query_instruction", DEFAULT_SVG_QUERY_INSTRUCTION)
        )
        console.print(f"Collection: {resolved_collection}")
        console.print(f"Persistence: {resolved_persist}")
        console.print("Embedding field: description only")
        console.print(f"Model revision: {model_revision}")
        console.print(f"Device / batch: {device} / {batch_size}")
        console.print(f"Target: {limit if limit is not None else 'all 1,159,423 rows'}")
        result = index_mmsvg_descriptions(
            sources=sources,
            persist_directory=resolved_persist,
            collection_name=resolved_collection,
            model_name=model_name,
            model_revision=model_revision,
            device=device,
            embedding_batch_size=batch_size,
            read_batch_size=int(indexing.get("read_batch_size", 4096)),
            limit=limit,
            query_instruction=query_instruction,
            log_every_batches=int(indexing.get("log_every_batches", 10)),
        )
    except Exception as exc:
        console.print(f"[bold red]MMSVG indexing failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]MMSVG Chroma indexing completed.[/bold green]")
    console.print(
        f"Collection count: {result.collection_count_before:,} -> {result.collection_count_after:,}"
    )
    console.print(
        f"Scanned: {result.scanned_this_run:,}; "
        f"newly indexed: {result.indexed_this_run:,}; "
        f"already present: {result.existing_this_run:,}; "
        f"elapsed: {result.elapsed_seconds / 60:.1f} min"
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


if __name__ == "__main__":
    typer.run(mmsvg_rag_index)
