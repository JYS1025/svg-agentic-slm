"""CLI command for resumable local MMSVG Chroma indexing."""

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
    DEFAULT_COLLECTION_NAMES,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MMSVG_SOURCES,
    DESCRIPTION_FIELD,
    DETAIL_FIELD,
    index_mmsvg_documents,
    normalize_document_field,
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
    document_field: str | None = typer.Option(
        None,
        "--document-field",
        help="MMSVG text field to embed: description or detail.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Index only the first N deterministic rows (for smoke tests).",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="GPU embedding and Chroma upload batch size override.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Torch device override, for example cuda:0 or cpu.",
    ),
    max_seq_length: int | None = typer.Option(
        None,
        "--max-seq-length",
        min=1,
        help="Embedding-token truncation limit override.",
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
    """Build a resumable Chroma index from one raw MMSVG text field."""
    console.print("[bold blue]MMSVG Field-specific Chroma Indexing[/bold blue]")
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
        field = normalize_document_field(
            document_field or str(indexing.get("document_field", DESCRIPTION_FIELD))
        )
        resolved_persist = _resolve_persist_directory(
            field=field,
            override=persist_directory,
            chroma_config=chroma_config,
            indexing=indexing,
        )
        resolved_collection = _resolve_collection_name(
            field=field,
            override=collection,
            chroma_config=chroma_config,
            indexing=indexing,
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
        resolved_device = str(
            device or indexing.get("device", chroma_config.get("device", "cuda:0"))
        )
        resolved_batch_size = int(batch_size or indexing.get("batch_size", 256))
        resolved_max_seq_length = int(
            max_seq_length or indexing.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        )
        console.print(f"Collection: {resolved_collection}")
        console.print(f"Persistence: {resolved_persist}")
        console.print(f"Embedding field: {field} only")
        console.print(f"Document template: {{{field}}}")
        console.print(f"Model revision: {model_revision}")
        console.print(f"Device / batch: {resolved_device} / {resolved_batch_size}")
        console.print(f"Max sequence length: {resolved_max_seq_length}")
        console.print(f"Target: {limit if limit is not None else 'all 1,159,423 rows'}")
        result = index_mmsvg_documents(
            document_field=field,
            sources=sources,
            persist_directory=resolved_persist,
            collection_name=resolved_collection,
            model_name=model_name,
            model_revision=model_revision,
            device=resolved_device,
            embedding_batch_size=resolved_batch_size,
            read_batch_size=int(indexing.get("read_batch_size", 4096)),
            max_seq_length=resolved_max_seq_length,
            limit=limit,
            query_instruction=query_instruction,
            log_every_batches=int(indexing.get("log_every_batches", 10)),
        )
    except Exception as exc:
        console.print(f"[bold red]MMSVG indexing failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]MMSVG Chroma indexing completed.[/bold green]")
    console.print(
        f"Collection count: {result.collection_count_before:,} "
        f"-> {result.collection_count_after:,}"
    )
    console.print(
        f"Field: {result.document_field}; scanned: {result.scanned_this_run:,}; "
        f"newly indexed: {result.indexed_this_run:,}; "
        f"already present: {result.existing_this_run:,}; "
        f"elapsed: {result.elapsed_seconds / 60:.1f} min"
    )


def _resolve_persist_directory(
    *,
    field: str,
    override: Path | None,
    chroma_config: dict[str, Any],
    indexing: dict[str, Any],
) -> Path:
    if override is not None:
        return override
    if "persist_directory" in indexing:
        return Path(str(indexing["persist_directory"]))
    if field == DETAIL_FIELD:
        # Never place a detail run in the active description directory merely
        # because configs/rag.yaml was used with a CLI field override.
        return Path("./data/chroma_db_detail")
    return Path(str(chroma_config.get("persist_directory", "./data/chroma_db")))


def _resolve_collection_name(
    *,
    field: str,
    override: str | None,
    chroma_config: dict[str, Any],
    indexing: dict[str, Any],
) -> str:
    if override:
        return override
    if "collection_name" in indexing:
        return str(indexing["collection_name"])
    if field == DETAIL_FIELD:
        return DEFAULT_COLLECTION_NAMES[DETAIL_FIELD]
    return str(chroma_config.get("collection_name", DEFAULT_COLLECTION_NAMES[DESCRIPTION_FIELD]))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


if __name__ == "__main__":
    typer.run(mmsvg_rag_index)
