"""Qdrant-backed semantic retrieval for SVG examples.

The implementation keeps Qdrant and sentence-transformers imports lazy so
Chroma-only installations continue to work without the optional dependencies.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
import os
import uuid
from typing import Any

from svg_agentic_slm.rag.base import BaseRetriever
from svg_agentic_slm.rag.schemas import RetrievedExample

logger = logging.getLogger(__name__)


class QdrantRetriever(BaseRetriever):
    """Store and retrieve SVG references from Qdrant.

    Credentials are read from environment variables rather than YAML because
    the resolved RAG configuration is persisted in generation metadata.
    """

    def __init__(
        self,
        *,
        collection_name: str = "svg_text2svg_stack_minilm384_v1",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.0,
        url: str | None = None,
        api_key: str | None = None,
        url_env: str = "QDRANT_URL",
        api_key_env: str = "QDRANT_API_KEY",
        timeout_seconds: float = 120.0,
        upload_batch_size: int = 64,
        compress_svg: bool = True,
        on_disk_vectors: bool = True,
        on_disk_payload: bool = True,
        on_disk_hnsw: bool = True,
        scalar_quantization: bool = True,
        client: Any = None,
        encoder: Any = None,
        models_api: Any = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        if upload_batch_size <= 0:
            raise ValueError("upload_batch_size must be positive.")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._url = url
        self._api_key = api_key
        self._url_env = url_env
        self._api_key_env = api_key_env
        self._timeout_seconds = timeout_seconds
        self._upload_batch_size = upload_batch_size
        self._compress_svg = compress_svg
        self._on_disk_vectors = on_disk_vectors
        self._on_disk_payload = on_disk_payload
        self._on_disk_hnsw = on_disk_hnsw
        self._scalar_quantization = scalar_quantization
        self._client = client
        self._encoder = encoder
        self._models = models_api
        self._collection_checked = False

    @property
    def collection_name(self) -> str:
        """Return the configured Qdrant collection name."""
        return self._collection_name

    def preflight(self) -> None:
        """Verify connectivity, collection schema, and embedding compatibility."""
        self._ensure_collection(create_if_missing=False)

    def optimize_storage(self) -> None:
        """Apply configured on-disk and quantization settings to an old collection."""
        self._ensure_client_and_models()
        existed = self._client.collection_exists(self._collection_name)
        self._ensure_collection(create_if_missing=True)
        if not existed:
            return

        options: dict[str, Any] = {}
        if self._on_disk_vectors:
            options["vectors_config"] = {"": self._models.VectorParamsDiff(on_disk=True)}
        if self._on_disk_payload:
            options["collection_params"] = self._models.CollectionParamsDiff(on_disk_payload=True)
        if self._on_disk_hnsw:
            options["hnsw_config"] = self._models.HnswConfigDiff(on_disk=True)
        if self._scalar_quantization:
            options["quantization_config"] = self._models.ScalarQuantization(
                scalar=self._models.ScalarQuantizationConfig(
                    type=self._models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        if not options:
            return

        accepted = self._client.update_collection(
            collection_name=self._collection_name,
            **options,
        )
        if accepted is False:
            logger.warning(
                "Qdrant did not apply storage updates to '%s'. This is "
                "expected in Qdrant local-memory mode; server collections "
                "should be checked in the dashboard.",
                self._collection_name,
            )
        else:
            logger.info(
                "Requested storage optimization for Qdrant collection '%s'.",
                self._collection_name,
            )

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Embed and idempotently upload SVG documents."""
        if not documents:
            return

        self._ensure_collection(create_if_missing=True)
        records = [
            record
            for document in documents
            if (record := self._prepare_document(document)) is not None
        ]
        for start in range(0, len(records), self._upload_batch_size):
            batch = records[start : start + self._upload_batch_size]
            vectors = self._encode(
                [record["search_text"] for record in batch],
                batch_size=self._upload_batch_size,
            )
            points = [
                self._models.PointStruct(
                    id=record["point_id"],
                    vector=_vector_to_list(vector),
                    payload=record["payload"],
                )
                for record, vector in zip(batch, vectors, strict=True)
            ]
            self._client.upload_points(
                collection_name=self._collection_name,
                points=points,
                parallel=1,
                max_retries=5,
                wait=True,
            )

        logger.info(
            "Uploaded %d SVG examples to Qdrant collection '%s'.",
            len(records),
            self._collection_name,
        )

    def missing_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only documents whose stable point IDs are not stored yet."""
        if not documents:
            return []

        self._ensure_collection(create_if_missing=True)
        documents_by_id: dict[str, dict[str, Any]] = {}
        for document in documents:
            record = self._prepare_document(document)
            if record is not None:
                documents_by_id.setdefault(record["point_id"], document)
        if not documents_by_id:
            return []

        existing = self._client.retrieve(
            collection_name=self._collection_name,
            ids=list(documents_by_id),
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(point.id) for point in existing}
        return [
            document
            for point_id, document in documents_by_id.items()
            if point_id not in existing_ids
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedExample]:
        """Return SVG examples ordered by Qdrant cosine similarity."""
        if not query.strip() or top_k <= 0:
            return []

        self._ensure_collection(create_if_missing=False)
        query_vector = _vector_to_list(self._encode(query.strip()))
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=self._similarity_threshold,
            with_payload=True,
            with_vectors=False,
        )

        examples: list[RetrievedExample] = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            content = _decode_svg_payload(payload)
            if not content:
                logger.warning("Skipping Qdrant point %s without SVG content.", hit.id)
                continue

            description = str(payload.get("description", ""))
            item_id = str(payload.get("record_id") or hit.id)
            source = str(payload.get("source") or payload.get("record_id") or hit.id)
            metadata = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "description",
                    "source",
                    "svg_code",
                    "svg_content",
                    "content",
                    "svg_data",
                    "svg_codec",
                }
            }
            metadata["qdrant_point_id"] = str(hit.id)
            metadata["qdrant_collection"] = self._collection_name
            examples.append(
                RetrievedExample(
                    content=content,
                    item_id=item_id,
                    source=source,
                    description=description,
                    score=float(hit.score),
                    score_kind="cosine_similarity",
                    corpus_version=_optional_text(payload.get("dataset_revision")),
                    metadata=metadata,
                )
            )

        logger.info(
            "Retrieved %d Qdrant SVG examples for: %s",
            len(examples),
            query[:80],
        )
        return examples

    def clear(self) -> None:
        """Delete and recreate this collection with the configured schema."""
        self._ensure_dependencies()
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._collection_checked = False
        self._ensure_collection(create_if_missing=True)
        logger.info("Cleared Qdrant collection '%s'.", self._collection_name)

    def count(self) -> int:
        """Return the exact number of points, or zero if collection is absent."""
        self._ensure_client_and_models()
        if not self._client.collection_exists(self._collection_name):
            return 0
        return int(
            self._client.count(
                collection_name=self._collection_name,
                exact=True,
            ).count
        )

    def _prepare_document(
        self,
        document: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw_content = document.get("content")
        content = "" if raw_content is None else str(raw_content).strip()
        if not content:
            return None

        raw_metadata = dict(document.get("metadata") or {})
        raw_search_text = raw_metadata.pop("search_text", "")
        search_text = "" if raw_search_text is None else str(raw_search_text).strip()
        if not search_text:
            search_text = _build_search_text(raw_metadata)
        if not search_text:
            return None

        identity = str(
            raw_metadata.get("point_identity")
            or raw_metadata.get("source")
            or _fallback_identity(search_text, content)
        )
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        payload = _json_safe_mapping(raw_metadata)
        payload.pop("point_identity", None)
        raw_description = raw_metadata.get("description")
        payload.setdefault(
            "description",
            "" if raw_description is None else str(raw_description),
        )
        payload.setdefault("source", identity)
        payload["embedding_model"] = self._embedding_model
        payload.update(_encode_svg_payload(content, compress=self._compress_svg))
        return {
            "point_id": point_id,
            "search_text": search_text,
            "payload": payload,
        }

    def _ensure_client_and_models(self) -> None:
        if self._models is None or self._client is None:
            try:
                from dotenv import load_dotenv
                from qdrant_client import QdrantClient, models
            except ImportError as exc:
                raise RuntimeError(
                    'Qdrant support is not installed. Run `pip install -e ".[qdrant]"`.'
                ) from exc

            load_dotenv()
            self._models = self._models or models
            if self._client is None:
                url = (self._url or os.getenv(self._url_env, "")).strip()
                api_key = (
                    self._api_key if self._api_key is not None else os.getenv(self._api_key_env, "")
                )
                api_key = api_key.strip() if api_key else None
                if not url:
                    raise RuntimeError(
                        f"Qdrant URL is missing. Set {self._url_env} in the "
                        "environment or .env file."
                    )
                if "cloud.qdrant.io" in url and not api_key:
                    raise RuntimeError(
                        f"Qdrant Cloud API key is missing. Set "
                        f"{self._api_key_env}; do not place it in rag.yaml."
                    )
                self._client = QdrantClient(
                    url=url.rstrip("/"),
                    api_key=api_key,
                    timeout=self._timeout_seconds,
                    prefer_grpc=False,
                )

    def _ensure_encoder(self) -> None:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for Qdrant embeddings. "
                    'Run `pip install -e ".[qdrant]"`.'
                ) from exc
            self._encoder = SentenceTransformer(self._embedding_model)

    def _ensure_dependencies(self) -> None:
        self._ensure_client_and_models()
        self._ensure_encoder()

    def _ensure_collection(self, *, create_if_missing: bool) -> None:
        if self._collection_checked:
            return

        self._ensure_client_and_models()
        collection_exists = self._client.collection_exists(self._collection_name)
        if not collection_exists and not create_if_missing:
            raise RuntimeError(
                f"Qdrant collection '{self._collection_name}' does not "
                "exist. Run `svg-agentic-slm rag-index` first."
            )

        self._ensure_encoder()
        dimension = self._encoder.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine embedding dimension.")

        if not collection_exists:
            self._create_collection(int(dimension))
            self._collection_checked = True
            return

        collection = self._client.get_collection(self._collection_name)
        vectors_config = collection.config.params.vectors
        if isinstance(vectors_config, dict):
            raise RuntimeError(
                "Named-vector Qdrant collections are not supported by this retriever."
            )
        existing_dimension = getattr(vectors_config, "size", None)
        existing_distance = getattr(vectors_config, "distance", None)
        if existing_distance is not None and existing_distance != self._models.Distance.COSINE:
            raise RuntimeError("Qdrant collection must use cosine distance for this retriever.")
        if existing_dimension is not None and int(existing_dimension) != int(dimension):
            raise RuntimeError(
                f"Qdrant collection dimension is {existing_dimension}, but "
                f"'{self._embedding_model}' produces {dimension}. Use the "
                "original model or a new collection."
            )
        self._validate_embedding_model()
        self._collection_checked = True

    def _validate_embedding_model(self) -> None:
        if not hasattr(self._client, "scroll"):
            return
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=1,
            with_payload=["embedding_model"],
            with_vectors=False,
        )
        if not points:
            return
        stored_model = str((points[0].payload or {}).get("embedding_model", ""))
        if stored_model and stored_model != self._embedding_model:
            raise RuntimeError(
                f"Qdrant collection uses embedding model '{stored_model}', "
                f"not '{self._embedding_model}'. Use the stored model or a "
                "new collection."
            )

    def _create_collection(self, dimension: int) -> None:
        create_options: dict[str, Any] = {
            "collection_name": self._collection_name,
            "vectors_config": self._models.VectorParams(
                size=dimension,
                distance=self._models.Distance.COSINE,
                on_disk=self._on_disk_vectors,
            ),
            "on_disk_payload": self._on_disk_payload,
            "shard_number": 1,
        }
        if self._on_disk_hnsw:
            create_options["hnsw_config"] = self._models.HnswConfigDiff(on_disk=True)
        if self._scalar_quantization:
            create_options["quantization_config"] = self._models.ScalarQuantization(
                scalar=self._models.ScalarQuantizationConfig(
                    type=self._models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        self._client.create_collection(**create_options)
        logger.info(
            "Created storage-optimized Qdrant collection '%s' (%d dimensions).",
            self._collection_name,
            dimension,
        )

    def _encode(
        self,
        texts: str | list[str],
        *,
        batch_size: int | None = None,
    ) -> Any:
        self._ensure_encoder()
        return self._encoder.encode(
            texts,
            batch_size=batch_size or self._upload_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def _build_search_text(metadata: dict[str, Any]) -> str:
    name = str(metadata.get("pattern_name") or metadata.get("record_id") or "").replace("_", " ")
    description = str(metadata.get("description", ""))
    tags = metadata.get("tags", [])
    elements = metadata.get("svg_elements", [])
    return "\n".join(
        part
        for part in (
            f"Name: {name}" if name else "",
            f"Description: {description}" if description else "",
            f"Tags: {_join_values(tags)}" if tags else "",
            f"SVG elements: {_join_values(elements)}" if elements else "",
        )
        if part
    )


def _join_values(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _fallback_identity(search_text: str, content: str) -> str:
    digest = hashlib.sha256(f"{search_text}\n{content}".encode()).hexdigest()
    return f"sha256:{digest}"


def _encode_svg_payload(content: str, *, compress: bool) -> dict[str, str]:
    if not compress:
        return {"svg_codec": "utf8", "svg_data": content}

    compressed = gzip.compress(
        content.encode("utf-8"),
        compresslevel=6,
        mtime=0,
    )
    encoded = base64.b64encode(compressed).decode("ascii")
    if len(encoded) < len(content.encode("utf-8")):
        return {"svg_codec": "gzip+base64", "svg_data": encoded}
    return {"svg_codec": "utf8", "svg_data": content}


def _decode_svg_payload(payload: dict[str, Any]) -> str:
    for key in ("svg_code", "svg_content", "content", "svg_snippet"):
        value = payload.get(key)
        if value:
            return str(value)

    data = payload.get("svg_data")
    if not data:
        return ""
    codec = str(payload.get("svg_codec", "utf8"))
    if codec == "utf8":
        return str(data)
    if codec != "gzip+base64":
        logger.warning("Unsupported SVG payload codec: %s", codec)
        return ""

    try:
        compressed = base64.b64decode(str(data), validate=True)
        return gzip.decompress(compressed).decode("utf-8")
    except (EOFError, OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Could not decode compressed SVG payload: %s", exc)
        return ""


def _json_safe_mapping(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        try:
            json.dumps(value, ensure_ascii=False)
            safe[str(key)] = value
        except (TypeError, ValueError):
            safe[str(key)] = str(value)
    return safe


def _vector_to_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return list(vector.tolist())
    return [float(value) for value in vector]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
