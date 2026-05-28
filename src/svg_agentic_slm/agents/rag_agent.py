"""RAG agent wrapper.

Provides a thin agent-layer wrapper around the RAG retriever,
making it easy to integrate retrieval into the orchestration
pipeline with a consistent interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svg_agentic_slm.rag.base import BaseRetriever
    from svg_agentic_slm.rag.schemas import RetrievedExample

logger = logging.getLogger(__name__)


class RAGAgent:
    """Wrapper around a retriever for use in the orchestration pipeline.

    This is not a full agent (no model backend), but a convenience
    wrapper that provides retrieval as a pipeline step.

    Args:
        retriever: The retriever backend to use.
        top_k: Number of examples to retrieve.
    """

    def __init__(self, retriever: BaseRetriever, top_k: int = 3) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedExample]:
        """Retrieve relevant SVG examples for a query.

        Args:
            query: The text query (typically the user instruction).

        Returns:
            List of retrieved examples.
        """
        logger.info("Retrieving %d examples for query: %s", self._top_k, query[:80])
        return self._retriever.retrieve(query, top_k=self._top_k)

    def format_context(self, examples: list[RetrievedExample]) -> str:
        """Format retrieved examples into a context string for the prompt.

        Args:
            examples: List of retrieved examples.

        Returns:
            Formatted context string.

        TODO: Improve formatting based on prompt template requirements.
        """
        if not examples:
            return ""

        parts: list[str] = ["Relevant SVG examples:"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"\n--- Example {i} ---")
            parts.append(f"Description: {ex.description}")
            parts.append(f"SVG: {ex.content}")
        return "\n".join(parts)
