"""RAG agent wrapper.

Provides a thin agent-layer wrapper around the RAG retriever,
making it easy to integrate retrieval into the orchestration
pipeline with a consistent interface.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from svg_agentic_slm.rag.metadata_policy import (
    MINIMAL_RETRIEVAL_METADATA_POLICY,
    RetrievalMetadataPolicy,
)
from svg_agentic_slm.svg.validator import safe_svg_element_names

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

    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 3,
        metadata_policy: RetrievalMetadataPolicy = MINIMAL_RETRIEVAL_METADATA_POLICY,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self._retriever = retriever
        self._top_k = top_k
        self._metadata_policy = metadata_policy

    def retrieve(self, query: str) -> list[RetrievedExample]:
        """Retrieve relevant SVG examples for a query.

        Args:
            query: The text query (typically the user instruction).

        Returns:
            List of retrieved examples.
        """
        logger.info("Retrieving %d examples for query: %s", self._top_k, query[:80])
        examples = self._retriever.retrieve(query, top_k=self._top_k)
        sanitized_examples: list[RetrievedExample] = []
        for index, example in enumerate(examples, 1):
            if (
                example.kind == "reference_svg"
                and safe_svg_element_names(example.content, allow_fragment=True) is None
            ):
                logger.warning(
                    "Dropped unsafe or malformed RAG SVG reference '%s'.",
                    example.item_id,
                )
                continue
            metadata, dropped_keys = self._metadata_policy.apply(example.metadata)
            if dropped_keys:
                logger.info(
                    "Dropped non-whitelisted RAG metadata for item '%s': %s",
                    example.item_id,
                    ", ".join(dropped_keys),
                )
            sanitized_examples.append(
                replace(
                    example,
                    rank=example.rank if example.rank is not None else index,
                    metadata=metadata,
                )
            )
        return sanitized_examples

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
