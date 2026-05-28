"""Document loading utilities for the RAG corpus.

Provides functions to load SVG pattern documents from JSONL files
into the retriever's index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from svg_agentic_slm.data.jsonl import read_jsonl
from svg_agentic_slm.rag.base import BaseRetriever

logger = logging.getLogger(__name__)


def load_svg_corpus(
    corpus_path: str | Path,
    retriever: BaseRetriever,
) -> int:
    """Load SVG pattern documents from a JSONL file into a retriever.

    Args:
        corpus_path: Path to the JSONL corpus file.
        retriever: The retriever to add documents to.

    Returns:
        Number of documents loaded.

    TODO: Add chunking support for large documents.
    TODO: Add deduplication.
    """
    records = read_jsonl(corpus_path)

    documents: list[dict[str, Any]] = []
    for record in records:
        documents.append({
            "content": record.get("svg_snippet", ""),
            "metadata": {
                "pattern_name": record.get("pattern_name", ""),
                "description": record.get("description", ""),
                "tags": record.get("tags", []),
            },
        })

    if documents:
        retriever.add_documents(documents)

    logger.info("Loaded %d documents from %s", len(documents), corpus_path)
    return len(documents)
