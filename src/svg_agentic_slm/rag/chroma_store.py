"""ChromaDB-backed semantic retrieval for SVG examples."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from svg_agentic_slm.rag.base import BaseRetriever
from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.svg.validator import safe_svg_element_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromaQueryHit:
    """Unhydrated result from a precomputed MMSVG Chroma collection."""

    item_id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    score: float
    raw_rank: int
    document_field: str


@dataclass(frozen=True)
class MMSVGCorpusRecord:
    """Trusted MMSVG payload resolved from a manifest-validated Parquet pointer."""

    item_id: str
    description: str
    detail: str
    svg: str
    image_png: bytes
    source: str
    corpus_version: str | None
    metadata: dict[str, Any]


class ChromaRetriever(BaseRetriever):
    """Store and retrieve SVG examples from one persistent Chroma collection.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for ChromaDB persistence.
        embedding_model: Chroma's built-in MiniLM model, or a model supported by
            ``SentenceTransformerEmbeddingFunction``.
        similarity_threshold: Minimum cosine similarity to return.
        document_field: MMSVG text field encoded by a precomputed collection.
            ``description`` remains the default; ``detail`` is also supported.
    """

    _BATCH_SIZE = 256
    _MAX_SVG_CHARACTERS = 1_000_000

    def __init__(
        self,
        collection_name: str = "svg_patterns",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.0,
        embedding_revision: str | None = None,
        embedding_dimension: int | None = None,
        query_instruction: str | None = None,
        device: str = "cuda:0",
        query_encoder_batch_size: int = 1,
        dataset_roots: Mapping[str, str | Path] | None = None,
        precomputed_embeddings: bool = False,
        overfetch_factor: int = 5,
        document_field: str = "description",
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        if embedding_dimension is not None and embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive when provided.")
        if overfetch_factor <= 0:
            raise ValueError("overfetch_factor must be positive.")
        if (
            isinstance(query_encoder_batch_size, bool)
            or not isinstance(query_encoder_batch_size, int)
            or query_encoder_batch_size <= 0
        ):
            raise ValueError("query_encoder_batch_size must be a positive integer.")
        if precomputed_embeddings and not embedding_revision:
            raise ValueError("precomputed embeddings require embedding_revision.")
        if precomputed_embeddings and not (query_instruction or "").strip():
            raise ValueError("precomputed embeddings require query_instruction.")
        normalized_document_field = str(document_field).strip().casefold()
        if normalized_document_field not in {"description", "detail"}:
            raise ValueError("document_field must be 'description' or 'detail'.")
        if not precomputed_embeddings and normalized_document_field != "description":
            raise ValueError(
                "document_field='detail' is only supported for precomputed embeddings."
            )
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._embedding_revision = embedding_revision
        self._embedding_dimension = embedding_dimension
        self._query_instruction = (query_instruction or "").strip()
        self._device = device
        self._query_encoder_batch_size = query_encoder_batch_size
        self._precomputed_embeddings = precomputed_embeddings
        self._overfetch_factor = overfetch_factor
        self._document_field = normalized_document_field
        self._dataset_roots = {
            str(dataset_type).strip().casefold(): Path(root).resolve()
            for dataset_type, root in (dataset_roots or {}).items()
        }
        self._client: Any = None
        self._collection: Any = None
        self._embedding_function: Any = None
        self._query_encoder: Any = None
        self._manifest: dict[str, Any] = {}
        self._source_info: dict[str, dict[str, Any]] = {}
        self._allowed_shards: dict[str, dict[str, Path]] = {}
        self._parquet_files: dict[Path, Any] = {}
        self._row_group_cache: OrderedDict[tuple[Path, int, tuple[str, ...]], Any] = OrderedDict()

    def _ensure_initialized(self) -> None:
        """Lazily initialize the local client, embedding model, and collection."""
        if self._collection is not None:
            return

        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is required for RAG. Install project dependencies "
                    "with `pip install -e .`."
                ) from exc

            self._client = chromadb.PersistentClient(path=self._persist_directory)

        if self._precomputed_embeddings:
            collection_names = {
                item if isinstance(item, str) else str(item.name)
                for item in self._client.list_collections()
            }
            if self._collection_name not in collection_names:
                raise RuntimeError(
                    f"Precomputed Chroma collection '{self._collection_name}' does not exist. "
                    "Run the MMSVG indexer first."
                )
            collection = self._client.get_collection(
                name=self._collection_name,
                embedding_function=None,
            )
            self._load_and_validate_manifest(collection)
            self._collection = collection
            return

        try:
            from chromadb.utils.embedding_functions import (
                DefaultEmbeddingFunction,
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB embedding functions are unavailable. Reinstall chromadb."
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

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Upsert SVG examples, embedding their descriptions and structure."""
        if not documents:
            return
        if self._precomputed_embeddings:
            raise RuntimeError(
                "The MMSVG collection is managed by the resumable precomputed indexer."
            )

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
        collection_count = self._collection.count()
        result_count = min(
            top_k * self._overfetch_factor if self._precomputed_embeddings else top_k,
            collection_count,
        )
        if result_count == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "n_results": result_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if self._precomputed_embeddings:
            query_vector = self._ensure_query_encoder().encode_queries([query.strip()])
            query_kwargs["query_embeddings"] = _embedding_rows(query_vector)
        else:
            query_kwargs["query_texts"] = [query.strip()]
        results = self._collection.query(**query_kwargs)
        ids = _first_result_row(results, "ids")
        documents = _first_result_row(results, "documents")
        metadatas = _first_result_row(results, "metadatas")
        distances = _first_result_row(results, "distances")

        examples: list[RetrievedExample] = []
        for index, record_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(-1.0, min(1.0, 1.0 - distance))
            if score < self._similarity_threshold:
                continue

            raw_document = documents[index] if index < len(documents) else None
            document = "" if raw_document is None else str(raw_document).strip()
            description = "" if self._precomputed_embeddings else str(
                metadata.get("description", "")
            )
            content = str(metadata.pop("svg_content", ""))
            if self._precomputed_embeddings:
                dataset_type = str(metadata.get("dataset_type", "")).strip().casefold()
                metadata_record_id = str(metadata.get("record_id", "")).strip()
                if str(record_id) != f"{dataset_type}:{metadata_record_id}":
                    logger.warning("Skipped mismatched MMSVG Chroma ID: %s", record_id)
                    continue
                try:
                    description, content = self._resolve_svg_pointer(
                        metadata,
                        expected_document=document,
                    )
                except Exception as exc:
                    logger.warning("Skipped invalid MMSVG pointer for %s: %s", record_id, exc)
                    continue
            elif not content:
                content = document if document.lstrip().startswith("<") else ""
            if not content:
                continue
            if safe_svg_element_names(content, allow_fragment=True) is None:
                logger.warning("Skipped unsafe or malformed retrieved SVG: %s", record_id)
                continue

            source, corpus_version = self._source_identity(metadata, str(record_id))
            examples.append(
                RetrievedExample(
                    content=content,
                    item_id=str(record_id),
                    source=source,
                    description=description,
                    score=score,
                    score_kind="cosine_similarity",
                    corpus_version=corpus_version,
                    metadata=_restore_metadata(metadata),
                )
            )
            if len(examples) >= top_k:
                break

        logger.info("Retrieved %d SVG examples for: %s", len(examples), query[:80])
        return examples

    def encode_queries(self, queries: Sequence[str]) -> list[list[float]]:
        """Encode an ordered query batch for compatible precomputed indexes."""
        if isinstance(queries, (str, bytes, bytearray)) or not isinstance(
            queries, Sequence
        ):
            raise TypeError("queries must be a sequence of strings.")
        normalized: list[str] = []
        for query in queries:
            if not isinstance(query, str):
                raise TypeError("queries must contain only strings.")
            value = query.strip()
            if not value:
                raise ValueError("queries must contain only non-empty strings.")
            normalized.append(value)
        if not normalized:
            raise ValueError("queries must be non-empty.")
        if not self._precomputed_embeddings:
            raise RuntimeError("Shared query encoding requires a precomputed collection.")
        self._ensure_initialized()
        rows = _embedding_rows(self._ensure_query_encoder().encode_queries(normalized))
        if len(rows) != len(normalized):
            raise RuntimeError("Query encoder returned an unexpected number of rows.")
        if self._embedding_dimension is not None and any(
            len(row) != self._embedding_dimension for row in rows
        ):
            raise RuntimeError("Query encoder returned an unexpected embedding dimension.")
        return rows

    def encode_query(self, query: str) -> list[float]:
        """Encode one query once for reuse across compatible precomputed indexes."""
        return self.encode_queries([query])[0]

    def query_hits_by_embedding(
        self,
        query_embedding: Sequence[float],
        *,
        n_results: int,
    ) -> list[ChromaQueryHit]:
        """Query a precomputed collection without hydrating Parquet payloads."""
        return self.query_hits_many_by_embeddings(
            [query_embedding],
            n_results=n_results,
        )[0]

    def query_hits_many_by_embeddings(
        self,
        query_embeddings: Sequence[Sequence[float]],
        *,
        n_results: int,
    ) -> list[list[ChromaQueryHit]]:
        """Query an ordered embedding batch without hydrating Parquet payloads."""
        if not self._precomputed_embeddings:
            raise RuntimeError(
                "query_hits_many_by_embeddings requires precomputed embeddings."
            )
        if isinstance(query_embeddings, (str, bytes, bytearray)) or not isinstance(
            query_embeddings, Sequence
        ):
            raise TypeError("query_embeddings must be a sequence of embedding rows.")
        if n_results <= 0:
            return [[] for _ in query_embeddings]
        vectors: list[list[float]] = []
        for embedding in query_embeddings:
            if isinstance(embedding, (str, bytes, bytearray)) or not isinstance(
                embedding, Sequence
            ):
                raise TypeError("Every query embedding must be a numeric sequence.")
            vector = [float(component) for component in embedding]
            if not vector:
                raise ValueError("Query embeddings must be non-empty.")
            if (
                self._embedding_dimension is not None
                and len(vector) != self._embedding_dimension
            ):
                raise ValueError("query_embedding has an unexpected dimension.")
            vectors.append(vector)
        if not vectors:
            return []
        self._ensure_initialized()
        result_count = min(n_results, int(self._collection.count()))
        if result_count == 0:
            return [[] for _ in vectors]
        results = self._collection.query(
            query_embeddings=vectors,
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
        query_count = len(vectors)
        ids_rows = _result_rows(results, "ids", expected_rows=query_count)
        document_rows = _result_rows(results, "documents", expected_rows=query_count)
        metadata_rows = _result_rows(results, "metadatas", expected_rows=query_count)
        distance_rows = _result_rows(results, "distances", expected_rows=query_count)
        result_batches: list[list[ChromaQueryHit]] = []
        for query_index, ids in enumerate(ids_rows):
            documents = document_rows[query_index]
            metadatas = metadata_rows[query_index]
            distances = distance_rows[query_index]
            hits: list[ChromaQueryHit] = []
            for index, record_id in enumerate(ids):
                metadata = (
                    dict(metadatas[index] or {}) if index < len(metadatas) else {}
                )
                dataset_type = str(metadata.get("dataset_type", "")).strip().casefold()
                metadata_record_id = str(metadata.get("record_id", "")).strip()
                if str(record_id) != f"{dataset_type}:{metadata_record_id}":
                    logger.warning("Skipped mismatched MMSVG Chroma ID: %s", record_id)
                    continue
                distance = float(distances[index]) if index < len(distances) else 1.0
                score = max(-1.0, min(1.0, 1.0 - distance))
                if score < self._similarity_threshold:
                    continue
                raw_document = documents[index] if index < len(documents) else None
                document = "" if raw_document is None else str(raw_document).strip()
                if not document:
                    logger.warning(
                        "Skipped MMSVG hit with an empty indexed document: %s",
                        record_id,
                    )
                    continue
                hits.append(
                    ChromaQueryHit(
                        item_id=str(record_id),
                        document=document,
                        metadata=metadata,
                        distance=distance,
                        score=score,
                        raw_rank=index + 1,
                        document_field=self._document_field,
                    )
                )
            result_batches.append(hits)
        return result_batches

    def index_contract(self) -> dict[str, Any]:
        """Return the manifest fields that must match across dual MMSVG indexes."""
        if not self._precomputed_embeddings:
            raise RuntimeError("Index contracts exist only for precomputed collections.")
        self._ensure_initialized()
        return {
            key: self._manifest.get(key)
            for key in (
                "index_schema_version",
                "embedding_model",
                "embedding_revision",
                "embedding_dimension",
                "normalized_embeddings",
                "query_instruction",
                "distance",
                "sources",
            )
        }

    def hydrate_mmsvg_hit(self, hit: ChromaQueryHit) -> MMSVGCorpusRecord:
        """Resolve description, detail, SVG, and image.bytes for one trusted hit."""
        if hit.document_field != self._document_field:
            raise ValueError("Chroma hit belongs to a different document field.")
        self._ensure_initialized()
        payload = self._resolve_mmsvg_pointer(
            hit.metadata,
            expected_document=hit.document,
            include_image=True,
        )
        source, corpus_version = self._source_identity(hit.metadata, hit.item_id)
        return MMSVGCorpusRecord(
            item_id=hit.item_id,
            description=payload["description"],
            detail=payload["detail"],
            svg=payload["svg"],
            image_png=payload["image_png"],
            source=source,
            corpus_version=corpus_version,
            metadata=_restore_metadata(hit.metadata),
        )

    def clear(self) -> None:
        """Delete all indexed examples while keeping the retriever reusable."""
        if self._precomputed_embeddings:
            raise RuntimeError("Refusing to clear the managed MMSVG collection.")
        self._ensure_initialized()
        self._client.delete_collection(name=self._collection_name)
        self._collection = None
        self._ensure_initialized()
        logger.info("Cleared collection '%s'.", self._collection_name)

    def _ensure_query_encoder(self) -> Any:
        if self._query_encoder is not None:
            return self._query_encoder
        from svg_agentic_slm.rag.embedding import Qwen3EmbeddingEncoder

        self._query_encoder = Qwen3EmbeddingEncoder(
            model_name=self._embedding_model,
            revision=str(self._embedding_revision),
            device=self._device,
            batch_size=self._query_encoder_batch_size,
            expected_dimension=int(self._embedding_dimension or 1024),
            query_instruction=self._query_instruction,
        )
        return self._query_encoder

    def _load_and_validate_manifest(self, collection: Any) -> None:
        manifest_path = Path(self._persist_directory) / f"{self._collection_name}.manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing Chroma index manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid Chroma index manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Chroma index manifest must be a JSON object: {manifest_path}")

        expected = {
            "index_schema_version": 1,
            "collection_name": self._collection_name,
            "document_field": self._document_field,
            "document_template": f"{{{self._document_field}}}",
            "embedding_model": self._embedding_model,
            "embedding_revision": self._embedding_revision,
            "embedding_dimension": self._embedding_dimension,
            "normalized_embeddings": True,
            "query_instruction": self._query_instruction,
            "distance": "cosine",
        }
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise RuntimeError("Chroma index manifest mismatch: " + ", ".join(sorted(mismatches)))
        collection_metadata = collection.metadata or {}
        collection_expected = {
            "index_schema_version": 1,
            "document_field": self._document_field,
            "embedding_model": self._embedding_model,
            "embedding_revision": self._embedding_revision,
            "embedding_dimension": self._embedding_dimension,
            "hnsw:space": "cosine",
        }
        collection_mismatches = [
            key
            for key, value in collection_expected.items()
            if collection_metadata.get(key) != value
        ]
        if collection_mismatches:
            raise RuntimeError(
                "Chroma collection metadata mismatch: " + ", ".join(sorted(collection_mismatches))
            )
        raw_sources = manifest.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise RuntimeError("Chroma index manifest sources must be a non-empty list.")
        source_info: dict[str, dict[str, Any]] = {}
        for source in raw_sources:
            if not isinstance(source, dict):
                raise RuntimeError("Every Chroma index manifest source must be a mapping.")
            dataset_type = str(source.get("dataset_type", "")).strip().casefold()
            if not dataset_type:
                raise RuntimeError("Chroma index manifest source has a blank dataset_type.")
            if dataset_type in source_info:
                raise RuntimeError(
                    f"Duplicate dataset_type in Chroma index manifest: {dataset_type}"
                )
            for key in ("dataset_id", "dataset_revision"):
                if not str(source.get(key, "")).strip():
                    raise RuntimeError(
                        f"Chroma index manifest source {dataset_type} has a blank {key}."
                    )
            for key in ("expected_shards", "expected_rows"):
                value = source.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise RuntimeError(
                        f"Chroma index manifest source {dataset_type} has invalid {key}."
                    )
            source_info[dataset_type] = dict(source)
        missing_roots = sorted(set(source_info) - set(self._dataset_roots))
        if missing_roots:
            raise RuntimeError("Missing trusted dataset_roots for: " + ", ".join(missing_roots))
        extra_roots = sorted(set(self._dataset_roots) - set(source_info))
        if extra_roots:
            raise RuntimeError("Unexpected trusted dataset_roots for: " + ", ".join(extra_roots))
        expected_count = sum(int(source["expected_rows"]) for source in source_info.values())
        actual_count = int(collection.count())
        if actual_count != expected_count:
            raise RuntimeError(
                f"Incomplete MMSVG Chroma collection: {actual_count} rows; "
                f"expected {expected_count}."
            )
        allowed_shards: dict[str, dict[str, Path]] = {}
        for dataset_type, source in source_info.items():
            root = self._dataset_roots[dataset_type]
            if not root.is_dir():
                raise RuntimeError(f"MMSVG dataset root does not exist: {root}")
            shard_paths = sorted(root.glob("*.parquet"))
            expected_shards = source.get("expected_shards")
            if expected_shards is not None and len(shard_paths) != int(expected_shards):
                raise RuntimeError(
                    f"MMSVG root {root} has {len(shard_paths)} shards; expected {expected_shards}."
                )
            resolved_shards: dict[str, Path] = {}
            for shard_path in shard_paths:
                resolved = shard_path.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"MMSVG shard escapes its trusted root: {shard_path}"
                    ) from exc
                resolved_shards[shard_path.name] = resolved
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("MMSVG source validation requires pyarrow.") from exc
            actual_source_rows = sum(
                int(pq.read_metadata(shard_path).num_rows)
                for shard_path in resolved_shards.values()
            )
            expected_source_rows = int(source["expected_rows"])
            if actual_source_rows != expected_source_rows:
                raise RuntimeError(
                    f"MMSVG source {dataset_type} has {actual_source_rows} rows; "
                    f"expected {expected_source_rows}."
                )
            allowed_shards[dataset_type] = resolved_shards
        self._manifest = manifest
        self._source_info = source_info
        self._allowed_shards = allowed_shards

    def _resolve_svg_pointer(
        self,
        metadata: dict[str, Any],
        *,
        expected_document: str,
    ) -> tuple[str, str]:
        payload = self._resolve_mmsvg_pointer(
            metadata,
            expected_document=expected_document,
            include_image=False,
        )
        return payload["description"], payload["svg"]

    def _resolve_mmsvg_pointer(
        self,
        metadata: dict[str, Any],
        *,
        expected_document: str,
        include_image: bool,
    ) -> dict[str, Any]:
        dataset_type = str(metadata.get("dataset_type", "")).strip().casefold()
        shard = str(metadata.get("shard", "")).strip()
        record_id = str(metadata.get("record_id", "")).strip()
        row_index = metadata.get("row_index_in_shard")
        if dataset_type not in self._dataset_roots:
            raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
        if not shard or Path(shard).name != shard or Path(shard).suffix != ".parquet":
            raise ValueError("Invalid Parquet shard name.")
        if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
            raise ValueError("Invalid row_index_in_shard.")
        if not record_id:
            raise ValueError("Missing record_id.")

        shard_path = self._allowed_shards.get(dataset_type, {}).get(shard)
        if shard_path is None:
            raise ValueError("Parquet shard is not in the trusted manifest root.")

        parquet_file = self._parquet_files.get(shard_path)
        if parquet_file is None:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("MMSVG pointer hydration requires pyarrow.") from exc
            parquet_file = pq.ParquetFile(shard_path)
            self._parquet_files[shard_path] = parquet_file
        if row_index >= parquet_file.metadata.num_rows:
            raise ValueError("Parquet row pointer is out of bounds.")

        row_start = 0
        row_group_index = -1
        for candidate_index in range(parquet_file.num_row_groups):
            row_count = parquet_file.metadata.row_group(candidate_index).num_rows
            if row_index < row_start + row_count:
                row_group_index = candidate_index
                break
            row_start += row_count
        if row_group_index < 0:
            raise ValueError("Unable to locate Parquet row group.")

        columns = list(
            dict.fromkeys(
                ["id", "description", "detail", self._document_field, "svg"]
                + (["image"] if include_image else [])
            )
        )
        cache_key = (shard_path, row_group_index, tuple(columns))
        table = self._row_group_cache.get(cache_key)
        if table is None:
            table = parquet_file.read_row_group(
                row_group_index,
                columns=columns,
                use_threads=True,
            )
            self._row_group_cache[cache_key] = table
            while len(self._row_group_cache) > 8:
                self._row_group_cache.popitem(last=False)
        else:
            self._row_group_cache.move_to_end(cache_key)

        local_index = row_index - row_start
        actual_id = str(table.column("id")[local_index].as_py()).strip()
        description = str(table.column("description")[local_index].as_py()).strip()
        detail = str(table.column("detail")[local_index].as_py()).strip()
        actual_document = str(table.column(self._document_field)[local_index].as_py()).strip()
        svg_content = str(table.column("svg")[local_index].as_py()).strip()
        if actual_id != record_id:
            raise ValueError("Parquet record ID does not match Chroma metadata.")
        if actual_document != expected_document.strip():
            raise ValueError(
                f"Parquet {self._document_field} does not match the indexed document."
            )
        if not description or not detail:
            raise ValueError("Parquet MMSVG text payload is incomplete.")
        if not svg_content:
            raise ValueError("Parquet SVG payload is empty.")
        if len(svg_content) > self._MAX_SVG_CHARACTERS:
            raise ValueError("Parquet SVG payload exceeds the safety limit.")

        image_png = b""
        if include_image:
            raw_image = table.column("image")[local_index].as_py()
            if not isinstance(raw_image, dict):
                raise ValueError("Parquet image payload must be a struct.")
            raw_bytes = raw_image.get("bytes")
            if not isinstance(raw_bytes, bytes) or not raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Parquet image.bytes must contain PNG bytes.")
            image_png = raw_bytes
        return {
            "description": description,
            "detail": detail,
            "svg": svg_content,
            "image_png": image_png,
        }

    def _source_identity(
        self,
        metadata: dict[str, Any],
        fallback_id: str,
    ) -> tuple[str, str | None]:
        if not self._precomputed_embeddings:
            source = str(metadata.get("pattern_name", "") or fallback_id)
            return source, _optional_text(metadata.get("corpus_version"))
        dataset_type = str(metadata.get("dataset_type", "")).strip().casefold()
        record_id = str(metadata.get("record_id", "") or fallback_id)
        source_info = self._source_info.get(dataset_type, {})
        dataset_id = str(source_info.get("dataset_id", dataset_type))
        revision = _optional_text(source_info.get("dataset_revision"))
        revision_suffix = f"@{revision}" if revision else ""
        return f"hf://{dataset_id}{revision_suffix}/train/{record_id}", revision


def _build_search_text(content: str, metadata: dict[str, Any]) -> str:
    """Build the legacy generic-corpus text while preserving stable IDs."""
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


def _result_rows(
    results: dict[str, Any],
    key: str,
    *,
    expected_rows: int,
) -> list[list[Any]]:
    """Extract every query row while rejecting ambiguous Chroma cardinality."""
    raw_rows = results.get(key)
    if raw_rows is None:
        raise RuntimeError(f"Chroma query response is missing {key!r} rows.")
    rows = list(raw_rows)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Chroma query response returned {len(rows)} {key!r} rows; "
            f"expected {expected_rows}."
        )
    return [list(row) if row is not None else [] for row in rows]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _embedding_rows(value: Any) -> list[list[float]]:
    rows = value.tolist() if hasattr(value, "tolist") else value
    return [[float(component) for component in row] for row in rows]
