"""Resumable field-specific Chroma indexing for the local MMSVG v2 corpus."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from svg_agentic_slm.rag.embedding import (
    DEFAULT_QWEN3_EMBEDDING_DIMENSION,
    DEFAULT_QWEN3_EMBEDDING_MODEL,
    DEFAULT_QWEN3_EMBEDDING_REVISION,
    DEFAULT_SVG_QUERY_INSTRUCTION,
    Qwen3EmbeddingEncoder,
)

logger = logging.getLogger(__name__)

DESCRIPTION_FIELD = "description"
DETAIL_FIELD = "detail"
SUPPORTED_DOCUMENT_FIELDS = (DESCRIPTION_FIELD, DETAIL_FIELD)
DEFAULT_COLLECTION_NAMES = {
    DESCRIPTION_FIELD: "mmsvg_description_qwen3_0_6b_1024_v1",
    DETAIL_FIELD: "mmsvg_detail_qwen3_0_6b_1024_v1",
}
# Kept for compatibility with the original description-only indexer.
DEFAULT_COLLECTION_NAME = DEFAULT_COLLECTION_NAMES[DESCRIPTION_FIELD]
DEFAULT_MAX_SEQ_LENGTH = 512
EXPECTED_MMSVG_ROWS = 1_159_423
INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MMSVGSource:
    """One audited MMSVG v2 source stored as local Parquet shards."""

    dataset_type: str
    dataset_id: str
    dataset_revision: str
    data_dir: Path
    expected_shards: int | None = None
    expected_rows: int | None = None


@dataclass(frozen=True)
class MMSVGRecord:
    """One field-specific text record plus its stable Parquet payload pointer."""

    record_id: str
    document_text: str
    document_field: str
    dataset_type: str
    dataset_id: str
    dataset_revision: str
    shard: str
    row_index_in_shard: int


@dataclass(frozen=True)
class MMSVGIndexingResult:
    """Summary of a resumable local Chroma indexing run."""

    document_field: str
    target_count: int
    collection_count_before: int
    collection_count_after: int
    scanned_this_run: int
    indexed_this_run: int
    existing_this_run: int
    elapsed_seconds: float


DEFAULT_MMSVG_SOURCES = (
    MMSVGSource(
        dataset_type="icon",
        dataset_id="OmniSVG/MMSVG-Icon",
        dataset_revision="8b1980d64000138d9fd14c3bfbd592edcc4b0be9",
        data_dir=Path("data/raw/mmsvg_icon_v2/data"),
        expected_shards=91,
        expected_rows=904_011,
    ),
    MMSVGSource(
        dataset_type="illustration",
        dataset_id="OmniSVG/MMSVG-Illustration",
        dataset_revision="6d81c98ae9bc1f4e1fca80cea496a73cb7f150c1",
        data_dir=Path("data/raw/mmsvg_illustration_v2/data"),
        expected_shards=26,
        expected_rows=255_412,
    ),
)


def normalize_document_field(document_field: str) -> str:
    """Validate and normalize the only MMSVG text fields approved for indexing."""
    normalized = str(document_field).strip().casefold()
    if normalized not in SUPPORTED_DOCUMENT_FIELDS:
        allowed = ", ".join(SUPPORTED_DOCUMENT_FIELDS)
        raise ValueError(f"document_field must be one of: {allowed}.")
    return normalized


def document_template(document_field: str) -> str:
    """Return the exact, label-free document template pinned in the manifest."""
    return "{" + normalize_document_field(document_field) + "}"


def stable_record_id(dataset_type: str, record_id: str) -> str:
    """Return an ID shared by description/detail collections for exact joins."""
    normalized_type = dataset_type.strip().casefold()
    normalized_id = record_id.strip()
    if not normalized_type or not normalized_id:
        raise ValueError("dataset_type and record_id must be non-empty.")
    # Do not include document_field: separate collections must expose identical
    # item IDs so their Top-K results can be intersected without another lookup.
    return f"{normalized_type}:{normalized_id}"


def iter_mmsvg_records(
    sources: Sequence[MMSVGSource],
    *,
    document_field: str = DESCRIPTION_FIELD,
    read_batch_size: int = 4096,
    limit: int | None = None,
) -> Iterator[MMSVGRecord]:
    """Yield deterministic rows while projecting only ``id`` and one text field."""
    field = normalize_document_field(document_field)
    if read_batch_size <= 0:
        raise ValueError("read_batch_size must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MMSVG indexing requires pyarrow.") from exc

    emitted = 0
    for source in sources:
        data_dir = Path(source.data_dir)
        shards = sorted(data_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards found in {data_dir}.")
        if source.expected_shards is not None and len(shards) != source.expected_shards:
            raise RuntimeError(
                f"{source.dataset_id} has {len(shards)} shards; expected "
                f"{source.expected_shards}."
            )

        source_rows = 0
        for shard_path in shards:
            parquet_file = pq.ParquetFile(shard_path)
            missing_columns = {"id", field} - set(parquet_file.schema_arrow.names)
            if missing_columns:
                names = ", ".join(sorted(missing_columns))
                raise RuntimeError(f"{shard_path} is missing required columns: {names}.")

            row_index = 0
            for arrow_batch in parquet_file.iter_batches(
                batch_size=read_batch_size,
                columns=["id", field],
                use_threads=True,
            ):
                batch_values = arrow_batch.to_pydict()
                for raw_id, raw_text in zip(
                    batch_values["id"],
                    batch_values[field],
                    strict=True,
                ):
                    record_id = "" if raw_id is None else str(raw_id).strip()
                    text = "" if raw_text is None else str(raw_text).strip()
                    if not record_id or not text:
                        raise ValueError(
                            f"Blank id/{field} in {shard_path.name} at row {row_index}."
                        )
                    yield MMSVGRecord(
                        record_id=record_id,
                        document_text=text,
                        document_field=field,
                        dataset_type=source.dataset_type,
                        dataset_id=source.dataset_id,
                        dataset_revision=source.dataset_revision,
                        shard=shard_path.name,
                        row_index_in_shard=row_index,
                    )
                    row_index += 1
                    source_rows += 1
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return

        if source.expected_rows is not None and source_rows != source.expected_rows:
            raise RuntimeError(
                f"{source.dataset_id} has {source_rows} rows; expected {source.expected_rows}."
            )


def index_mmsvg_documents(
    *,
    document_field: str = DESCRIPTION_FIELD,
    sources: Sequence[MMSVGSource] = DEFAULT_MMSVG_SOURCES,
    persist_directory: str | Path = "./data/chroma_db",
    collection_name: str | None = None,
    model_name: str = DEFAULT_QWEN3_EMBEDDING_MODEL,
    model_revision: str = DEFAULT_QWEN3_EMBEDDING_REVISION,
    device: str = "cuda:0",
    embedding_batch_size: int = 256,
    read_batch_size: int = 4096,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    limit: int | None = None,
    query_instruction: str = DEFAULT_SVG_QUERY_INSTRUCTION,
    log_every_batches: int = 10,
    encoder: Qwen3EmbeddingEncoder | None = None,
    client: Any = None,
) -> MMSVGIndexingResult:
    """Embed one raw MMSVG text field and upsert only missing stable IDs."""
    field = normalize_document_field(document_field)
    resolved_collection = collection_name or DEFAULT_COLLECTION_NAMES[field]
    if embedding_batch_size <= 0 or read_batch_size <= 0:
        raise ValueError("embedding_batch_size and read_batch_size must be positive.")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    if log_every_batches <= 0:
        raise ValueError("log_every_batches must be positive.")
    target_count = (
        limit if limit is not None else sum(source.expected_rows or 0 for source in sources)
    )
    if target_count <= 0:
        raise ValueError("A positive limit or expected source row counts are required.")

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(
        collection_name=resolved_collection,
        document_field=field,
        sources=sources,
        model_name=model_name,
        model_revision=model_revision,
        max_seq_length=max_seq_length,
        query_instruction=query_instruction,
    )
    _ensure_manifest(persist_path / f"{resolved_collection}.manifest.json", manifest)

    if client is None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("MMSVG indexing requires chromadb.") from exc
        client = chromadb.PersistentClient(path=str(persist_path))

    collection = client.get_or_create_collection(
        name=resolved_collection,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "document_field": field,
            "embedding_model": model_name,
            "embedding_revision": model_revision,
            "embedding_dimension": DEFAULT_QWEN3_EMBEDDING_DIMENSION,
        },
    )
    _validate_collection_metadata(collection.metadata or {}, manifest)

    before = int(collection.count())
    if before > target_count:
        raise RuntimeError(
            f"Chroma collection has {before} rows, above target {target_count}; "
            "refusing to treat a mismatched collection as complete."
        )
    if before == target_count:
        logger.info(
            "Chroma collection already has %d rows (target=%d); nothing to do.",
            before,
            target_count,
        )
        return MMSVGIndexingResult(
            document_field=field,
            target_count=target_count,
            collection_count_before=before,
            collection_count_after=before,
            scanned_this_run=0,
            indexed_this_run=0,
            existing_this_run=0,
            elapsed_seconds=0.0,
        )

    if encoder is None:
        encoder = Qwen3EmbeddingEncoder(
            model_name=model_name,
            revision=model_revision,
            device=device,
            batch_size=embedding_batch_size,
            max_seq_length=max_seq_length,
            expected_dimension=DEFAULT_QWEN3_EMBEDDING_DIMENSION,
            query_instruction=query_instruction,
        )

    started = time.monotonic()
    scanned = 0
    indexed = 0
    existing = 0
    batch_number = 0
    record_batch: list[MMSVGRecord] = []
    record_iterator = iter_mmsvg_records(
        sources,
        document_field=field,
        read_batch_size=read_batch_size,
        limit=target_count,
    )

    for record in record_iterator:
        record_batch.append(record)
        if len(record_batch) < embedding_batch_size:
            continue
        batch_number += 1
        batch_indexed, batch_existing = _index_record_batch(
            collection,
            encoder,
            record_batch,
        )
        scanned += len(record_batch)
        indexed += batch_indexed
        existing += batch_existing
        record_batch.clear()
        if batch_number % log_every_batches == 0:
            _log_progress(
                document_field=field,
                scanned=scanned,
                indexed=indexed,
                existing=existing,
                target=target_count,
                started=started,
            )

    if record_batch:
        batch_indexed, batch_existing = _index_record_batch(
            collection,
            encoder,
            record_batch,
        )
        scanned += len(record_batch)
        indexed += batch_indexed
        existing += batch_existing

    after = int(collection.count())
    elapsed = time.monotonic() - started
    if after < target_count:
        raise RuntimeError(f"Indexing ended with {after} rows, below target {target_count}.")
    if limit is None and target_count == EXPECTED_MMSVG_ROWS and after != target_count:
        raise RuntimeError(f"Full collection count is {after}; expected exactly {target_count}.")
    _log_progress(
        document_field=field,
        scanned=scanned,
        indexed=indexed,
        existing=existing,
        target=target_count,
        started=started,
    )
    return MMSVGIndexingResult(
        document_field=field,
        target_count=target_count,
        collection_count_before=before,
        collection_count_after=after,
        scanned_this_run=scanned,
        indexed_this_run=indexed,
        existing_this_run=existing,
        elapsed_seconds=elapsed,
    )


def index_mmsvg_descriptions(**kwargs: Any) -> MMSVGIndexingResult:
    """Backward-compatible entry point for the original description indexer."""
    if "document_field" in kwargs:
        raise TypeError("index_mmsvg_descriptions does not accept document_field.")
    return index_mmsvg_documents(document_field=DESCRIPTION_FIELD, **kwargs)


def _index_record_batch(
    collection: Any,
    encoder: Qwen3EmbeddingEncoder,
    records: Sequence[MMSVGRecord],
) -> tuple[int, int]:
    fields = {record.document_field for record in records}
    if len(fields) != 1:
        raise RuntimeError("An indexing batch cannot mix MMSVG document fields.")
    ids = [stable_record_id(record.dataset_type, record.record_id) for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate stable IDs occurred within an indexing batch.")
    existing_ids = _existing_ids(collection, ids)
    missing = [
        record for record, item_id in zip(records, ids, strict=True) if item_id not in existing_ids
    ]
    if not missing:
        return 0, len(records)

    missing_ids = [stable_record_id(record.dataset_type, record.record_id) for record in missing]
    documents = [record.document_text for record in missing]
    vectors = encoder.encode_documents(documents)
    vector_rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
    collection.upsert(
        ids=missing_ids,
        embeddings=vector_rows,
        documents=documents,
        metadatas=[
            {
                "dataset_type": record.dataset_type,
                "record_id": record.record_id,
                "document_field": record.document_field,
                "shard": record.shard,
                "row_index_in_shard": record.row_index_in_shard,
            }
            for record in missing
        ],
    )
    return len(missing), len(records) - len(missing)


def _existing_ids(collection: Any, ids: Sequence[str]) -> set[str]:
    try:
        result = collection.get(ids=list(ids), include=[])
    except (TypeError, ValueError):
        result = collection.get(ids=list(ids), include=["metadatas"])
    return {str(item_id) for item_id in (result.get("ids") or [])}


def _build_manifest(
    *,
    collection_name: str,
    document_field: str,
    sources: Sequence[MMSVGSource],
    model_name: str,
    model_revision: str,
    max_seq_length: int,
    query_instruction: str,
) -> dict[str, Any]:
    field = normalize_document_field(document_field)
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "collection_name": collection_name,
        "document_field": field,
        "document_template": document_template(field),
        "embedding_model": model_name,
        "embedding_revision": model_revision,
        "embedding_dimension": DEFAULT_QWEN3_EMBEDDING_DIMENSION,
        "max_seq_length": max_seq_length,
        "normalized_embeddings": True,
        "distance": "cosine",
        "query_instruction": query_instruction,
        "sources": [
            {
                **asdict(source),
                "data_dir": str(source.data_dir),
            }
            for source in sources
        ],
    }


def _ensure_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current == expected or _matches_legacy_description_manifest(current, expected):
            return
        raise RuntimeError(
            f"Index manifest mismatch at {path}; use a new collection or "
            "restore the original indexing configuration."
        )
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _matches_legacy_description_manifest(
    current: Any,
    expected: dict[str, Any],
) -> bool:
    """Accept, but never rewrite, the deployed manifest that predates max_seq_length."""
    if not isinstance(current, dict) or expected.get("document_field") != DESCRIPTION_FIELD:
        return False
    if "max_seq_length" in current or expected.get("max_seq_length") != DEFAULT_MAX_SEQ_LENGTH:
        return False
    legacy_expected = dict(expected)
    legacy_expected.pop("max_seq_length", None)
    return current == legacy_expected


def _validate_collection_metadata(
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    keys = (
        "index_schema_version",
        "document_field",
        "embedding_model",
        "embedding_revision",
        "embedding_dimension",
    )
    mismatches = [key for key in keys if metadata.get(key) != manifest.get(key)]
    if mismatches:
        raise RuntimeError(
            "Chroma collection metadata does not match the index manifest: "
            + ", ".join(mismatches)
        )


def _log_progress(
    *,
    document_field: str,
    scanned: int,
    indexed: int,
    existing: int,
    target: int,
    started: float,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = scanned / elapsed
    remaining = max(target - scanned, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    logger.info(
        "MMSVG %s indexing: scanned=%d/%d indexed=%d existing=%d "
        "rate=%.1f rows/s eta=%.1f min",
        document_field,
        scanned,
        target,
        indexed,
        existing,
        rate,
        eta_seconds / 60,
    )
