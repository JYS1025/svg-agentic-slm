"""Schemas for RAG retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedExample:
    """A single example retrieved from the RAG corpus.

    Attributes:
        content: The retrieved content (e.g., SVG code or snippet).
        description: Human-readable description of the content.
        score: Similarity score from the retriever.
        source: Identifier for the source document/collection.
        metadata: Additional metadata from the vector store.
    """

    content: str
    description: str = ""
    score: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)
