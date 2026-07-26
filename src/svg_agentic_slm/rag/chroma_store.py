"""ChromaDB-backed semantic retrieval for SVG examples."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from svg_agentic_slm.rag.base import BaseRetriever
from svg_agentic_slm.rag.schemas import RetrievedExample

logger = logging.getLogger(__name__)


class ChromaRetriever(BaseRetriever):
    """Store and retrieve SVG examples from one persistent Chroma collection.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for ChromaDB persistence.
        embedding_model: Chroma's built-in MiniLM model, or a model supported by
            ``SentenceTransformerEmbeddingFunction``.
        similarity_threshold: Minimum cosine similarity to return.
    """

    _BATCH_SIZE = 256

    def __init__(
        self,
        collection_name: str = "svg_patterns",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._client: Any = None
        self._collection: Any = None
        self._embedding_function: Any = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize the local client, embedding model, and collection."""
        if self._collection is not None:
            return

        if self._client is None:
            try:
                import chromadb
                from chromadb.utils.embedding_functions import (
                    DefaultEmbeddingFunction,
                    SentenceTransformerEmbeddingFunction,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is required for RAG. Install project dependencies "
                    "with `pip install -e .`."
                ) from exc

            if self._embedding_model in {
                "all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
            }:
                self._embedding_function = DefaultEmbeddingFunction()
            else:
                try:
                    self._embedding_function = SentenceTransformerEmbeddingFunction(
                        model_name=self._embedding_model,
                        normalize_embeddings=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "A custom embedding_model requires the sentence-transformers "
                        "package and a valid model name."
                    ) from exc

            self._client = chromadb.PersistentClient(path=self._persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Upsert SVG examples, embedding their descriptions and structure."""
        if not documents:
            return

        self._ensure_initialized()

        records: list[tuple[str, str, dict[str, str | int | float | bool]]] = []
        for document in documents:
            content = str(document.get("content", "")).strip()
            if not content:
                continue

            raw_metadata = dict(document.get("metadata") or {})
            search_text = _build_search_text(content, raw_metadata)
            metadata = _normalize_metadata(raw_metadata)
            metadata["svg_content"] = content
            record_id = hashlib.sha256(f"{search_text}\n{content}".encode()).hexdigest()
            records.append((record_id, search_text, metadata))

        for start in range(0, len(records), self._BATCH_SIZE):
            batch = records[start : start + self._BATCH_SIZE]
            self._collection.upsert(
                ids=[record[0] for record in batch],
                documents=[record[1] for record in batch],
                metadatas=[record[2] for record in batch],
            )

        logger.info("Indexed %d SVG examples in '%s'.", len(records), self._collection_name)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedExample]:
        """Return the SVG examples most semantically similar to ``query``."""
        if not query.strip() or top_k <= 0:
            return []

        self._ensure_initialized()
        result_count = min(top_k, self._collection.count())
        if result_count == 0:
            return []

        results = self._collection.query(
            query_texts=[query.strip()],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_result_row(results, "ids")
        documents = _first_result_row(results, "documents")
        metadatas = _first_result_row(results, "metadatas")
        distances = _first_result_row(results, "distances")

        examples: list[RetrievedExample] = []
        for index, record_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))
            if score < self._similarity_threshold:
                continue

            content = str(metadata.pop("svg_content", ""))
            if not content and index < len(documents):
                candidate = str(documents[index])
                content = candidate if candidate.lstrip().startswith("<") else ""

            description = str(metadata.get("description", ""))
            source = str(metadata.get("pattern_name", "") or record_id)
            examples.append(
                RetrievedExample(
                    content=content,
                    item_id=str(record_id),
                    source=source,
                    description=description,
                    score=score,
                    score_kind="cosine_similarity",
                    corpus_version=_optional_text(metadata.get("corpus_version")),
                    metadata=_restore_metadata(metadata),
                )
            )

        logger.info("Retrieved %d SVG examples for: %s", len(examples), query[:80])
        return examples

    def clear(self) -> None:
        """Delete all indexed examples while keeping the retriever reusable."""
        self._ensure_initialized()
        self._client.delete_collection(name=self._collection_name)
        self._collection = None
        self._ensure_initialized()
        logger.info("Cleared collection '%s'.", self._collection_name)


def _build_search_text(content: str, metadata: dict[str, Any]) -> str:
    """Build the text embedded by Chroma while preserving SVG as payload."""
    name = str(metadata.get("pattern_name", "")).replace("_", " ")
    description = str(metadata.get("description", ""))
    tags = metadata.get("tags", [])
    if isinstance(tags, list):
        tags_text = ", ".join(str(tag) for tag in tags)
    else:
        tags_text = str(tags)
    return "\n".join(
        part
        for part in (
            f"Name: {name}" if name else "",
            f"Description: {description}" if description else "",
            f"Tags: {tags_text}" if tags_text else "",
            f"SVG: {content}",
        )
        if part
    )


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata to scalar values accepted by all supported Chroma versions."""
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalized


def _restore_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore JSON-encoded list/dict metadata for downstream consumers."""
    restored = dict(metadata)
    for key, value in restored.items():
        if not isinstance(value, str) or not value.startswith(("[", "{")):
            continue
        try:
            restored[key] = json.loads(value)
        except json.JSONDecodeError:
            pass
    return restored


def _first_result_row(results: dict[str, Any], key: str) -> list[Any]:
    """Extract the first query row from Chroma's column-oriented response."""
    rows = results.get(key) or []
    return list(rows[0]) if rows and rows[0] is not None else []


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
