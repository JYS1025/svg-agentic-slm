"""Abstract interface for retrieval backends.

Defines the contract that all retriever implementations must follow.
This allows swapping ChromaDB for another vector database without
changing consumer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from svg_agentic_slm.rag.schemas import RetrievedExample


class BaseRetriever(ABC):
    """Abstract interface for a vector-based retriever."""

    @abstractmethod
    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """Add documents to the retrieval index.

        Args:
            documents: List of document dictionaries to index.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedExample]:
        """Retrieve similar documents for a query.

        Args:
            query: The text query to search for.
            top_k: Number of results to return.

        Returns:
            List of retrieved examples, ordered by relevance.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the index."""
        ...
