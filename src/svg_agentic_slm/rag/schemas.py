"""Schemas for RAG retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RetrievedItemKind = Literal[
    "reference_svg",
    "positive_experience",
    "negative_lesson",
    "correction_pair",
]


@dataclass
class RetrievedExample:
    """A single example retrieved from the RAG corpus.

    Attributes:
        content: The retrieved content (e.g., SVG code or snippet).
        item_id: Stable identifier for the same logical corpus item.
        source: Stable source document or collection identifier.
        description: Human-readable description of the content.
        score: Similarity score from the retriever.
        metadata: Whitelisted metadata crossing the shared boundary.
    """

    content: str
    item_id: str
    source: str
    description: str = ""
    score: float = 0.0
    score_kind: str = "similarity"
    rank: int | None = None
    kind: RetrievedItemKind = "reference_svg"
    corpus_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("RetrievedExample.item_id must be a stable non-empty ID.")
        if not self.source.strip():
            raise ValueError("RetrievedExample.source must be non-empty.")
        if not self.score_kind.strip():
            raise ValueError("RetrievedExample.score_kind must be non-empty.")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("RetrievedExample.rank must be positive when provided.")
