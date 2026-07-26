"""Streaming Hugging Face dataset indexing for the Qdrant SVG corpus."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

from svg_agentic_slm.rag.qdrant_store import QdrantRetriever

logger = logging.getLogger(__name__)

DEFAULT_DATASET_REVISION = "c6f2bf0fffd8c1b69fcf748c97f4b0e7de6f2687"

_SVG_FIELDS = ("Svg", "svg", "svg_code", "svg_snippet", "content")
_ID_FIELDS = ("Filename", "filename", "Id", "id", "record_id")
_CAPTION_FIELDS = (
    "caption_cogvlm",
    "caption_llava",
    "caption_blip2",
    "Caption",
    "caption",
    "description",
)
_FORBIDDEN_ELEMENTS = {"script", "foreignobject", "iframe", "object", "embed"}
_UNSAFE_REFERENCE_MARKERS = (
    "http:",
    "https:",
    "file:",
    "ftp:",
    "data:",
    "javascript:",
)
_CSS_URL_PATTERN = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class IndexingResult:
    """Summary of a bounded, idempotent dataset indexing run."""

    target_count: int
    collection_count_before: int
    collection_count_after: int
    uploaded_this_run: int
    scanned_this_run: int
    skipped_this_run: int


def index_huggingface_svg_dataset(
    retriever: QdrantRetriever,
    *,
    dataset_id: str = "starvector/text2svg-stack",
    dataset_split: str = "train",
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    index_limit: int = 100_000,
    batch_size: int = 64,
    max_svg_chars: int = 24_000,
    max_caption_chars: int = 1_200,
    shuffle_buffer: int = 2_000,
    seed: int = 42,
) -> IndexingResult:
    """Stream valid caption/SVG pairs and upsert a target-sized Qdrant corpus.

    ``index_limit`` is the desired collection size, not an amount appended to
    whatever is already stored. The deterministic stream and stable point IDs
    make reruns safe; existing IDs are skipped before embedding and upload.
    """
    _validate_index_settings(
        index_limit=index_limit,
        batch_size=batch_size,
        max_svg_chars=max_svg_chars,
        max_caption_chars=max_caption_chars,
        shuffle_buffer=shuffle_buffer,
    )

    before = retriever.count()
    if before >= index_limit:
        logger.info(
            "Qdrant collection already has %d points (target=%d); skipping.",
            before,
            index_limit,
        )
        return IndexingResult(
            target_count=index_limit,
            collection_count_before=before,
            collection_count_after=before,
            uploaded_this_run=0,
            scanned_this_run=0,
            skipped_this_run=0,
        )

    dataset = _load_streaming_dataset(
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
    return index_svg_rows(
        retriever,
        dataset,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        index_limit=index_limit,
        batch_size=batch_size,
        max_svg_chars=max_svg_chars,
        max_caption_chars=max_caption_chars,
        collection_count_before=before,
    )


def index_svg_rows(
    retriever: QdrantRetriever,
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    dataset_split: str,
    dataset_revision: str,
    index_limit: int,
    batch_size: int,
    max_svg_chars: int,
    max_caption_chars: int,
    collection_count_before: int | None = None,
) -> IndexingResult:
    """Index an iterable of dataset rows; public to support offline tests."""
    before = retriever.count() if collection_count_before is None else collection_count_before
    batch: list[dict[str, Any]] = []
    uploaded = 0
    current_count = before
    scanned = 0
    skipped = 0
    max_scanned = max(index_limit * 20, index_limit)

    for raw_row in rows:
        if current_count >= index_limit or scanned >= max_scanned:
            break
        scanned += 1
        document = prepare_svg_document(
            raw_row,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            dataset_revision=dataset_revision,
            max_svg_chars=max_svg_chars,
            max_caption_chars=max_caption_chars,
        )
        if document is None:
            skipped += 1
            continue

        batch.append(document)
        if len(batch) >= batch_size:
            added = _add_missing_documents(
                retriever,
                batch,
                remaining=index_limit - current_count,
            )
            uploaded += added
            current_count += added
            batch.clear()
            logger.info(
                "Qdrant SVG indexing progress: count=%d/%d new=%d scanned=%d skipped=%d",
                current_count,
                index_limit,
                uploaded,
                scanned,
                skipped,
            )

    if batch and current_count < index_limit:
        added = _add_missing_documents(
            retriever,
            batch,
            remaining=index_limit - current_count,
        )
        uploaded += added

    after = retriever.count()
    if after == 0:
        raise RuntimeError(
            "No valid SVG records were indexed. Check the dataset fields and validation limits."
        )
    if after < index_limit:
        logger.warning(
            "Dataset stream ended at %d collection points (target=%d, scanned=%d).",
            after,
            index_limit,
            scanned,
        )

    return IndexingResult(
        target_count=index_limit,
        collection_count_before=before,
        collection_count_after=after,
        uploaded_this_run=uploaded,
        scanned_this_run=scanned,
        skipped_this_run=skipped,
    )


def _add_missing_documents(
    retriever: QdrantRetriever,
    documents: list[dict[str, Any]],
    *,
    remaining: int,
) -> int:
    """Upload at most ``remaining`` new stable IDs from one input batch."""
    if remaining <= 0:
        return 0
    missing = retriever.missing_documents(documents)
    selected = missing[:remaining]
    if selected:
        retriever.add_documents(selected)
    return len(selected)


def prepare_svg_document(
    raw_row: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_split: str,
    dataset_revision: str,
    max_svg_chars: int,
    max_caption_chars: int,
) -> dict[str, Any] | None:
    """Convert one dataset row into the backend-neutral retriever schema."""
    svg_code = _first_text(raw_row, _SVG_FIELDS)
    if not svg_code or len(svg_code) > max_svg_chars:
        return None

    element_names = validate_svg_for_reference(svg_code)
    if element_names is None:
        return None

    description, caption_field = _first_text_with_field(
        raw_row,
        _CAPTION_FIELDS,
    )
    description = description[:max_caption_chars].strip()
    tags = _normalize_tags(raw_row.get("Label") or raw_row.get("label") or raw_row.get("tags"))
    if not description:
        description = " ".join(tags)[:max_caption_chars].strip()
    if not description:
        return None

    record_id = _first_text(raw_row, _ID_FIELDS)
    if not record_id:
        record_id = str(uuid.uuid5(uuid.NAMESPACE_URL, svg_code))
    source = f"hf://{dataset_id}/{dataset_split}/{record_id}"
    search_text = "\n".join(
        part
        for part in (
            f"Description: {description}",
            f"Tags: {', '.join(tags)}" if tags else "",
            (f"SVG elements: {', '.join(element_names)}" if element_names else ""),
        )
        if part
    )
    return {
        "content": svg_code,
        "metadata": {
            "point_identity": source,
            "source": source,
            "dataset_id": dataset_id,
            "dataset_split": dataset_split,
            "dataset_revision": dataset_revision,
            "record_id": record_id,
            "description": description,
            "caption_field": caption_field,
            "tags": tags,
            "svg_elements": element_names,
            "search_text": search_text,
        },
    }


def validate_svg_for_reference(svg_code: str) -> list[str] | None:
    """Reject malformed or active/external-content SVG references."""
    lowered = svg_code.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        return None

    try:
        root = ElementTree.fromstring(svg_code)
    except ElementTree.ParseError:
        return None
    if _local_name(root.tag).lower() != "svg":
        return None

    names: list[str] = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        name = _local_name(element.tag).lower()
        if name in _FORBIDDEN_ELEMENTS:
            return None
        if name != "svg" and name not in names:
            names.append(name)

        for raw_name, raw_value in element.attrib.items():
            attribute_name = _local_name(raw_name).lower()
            attribute_value = str(raw_value).strip().lower()
            if attribute_name.startswith("on"):
                return None
            if (
                attribute_name in {"href", "src"}
                and attribute_value
                and not attribute_value.startswith("#")
            ):
                return None
            if _contains_external_reference(attribute_value):
                return None

        if name == "style" and _contains_external_reference(element.text or ""):
            return None

    return names[:32]


def _contains_external_reference(value: str) -> bool:
    lowered = value.strip().lower()
    if any(marker in lowered for marker in _UNSAFE_REFERENCE_MARKERS):
        return True
    if lowered.startswith("//"):
        return True
    for match in _CSS_URL_PATTERN.finditer(lowered):
        target = match.group(2).strip().strip("'\"")
        if target and not target.startswith("#"):
            return True
    return False


def _load_streaming_dataset(
    *,
    dataset_id: str,
    dataset_split: str,
    dataset_revision: str,
    shuffle_buffer: int,
    seed: int,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for indexing. Install project "
            "dependencies with `pip install -e .`."
        ) from exc

    dataset = load_dataset(
        dataset_id,
        split=dataset_split,
        revision=dataset_revision,
        streaming=True,
    )
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def _validate_index_settings(
    *,
    index_limit: int,
    batch_size: int,
    max_svg_chars: int,
    max_caption_chars: int,
    shuffle_buffer: int,
) -> None:
    values = {
        "index_limit": index_limit,
        "batch_size": batch_size,
        "max_svg_chars": max_svg_chars,
        "max_caption_chars": max_caption_chars,
        "shuffle_buffer": shuffle_buffer,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be positive.")


def _first_text(row: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    value, _ = _first_text_with_field(row, fields)
    return value


def _first_text_with_field(
    row: Mapping[str, Any],
    fields: tuple[str, ...],
) -> tuple[str, str]:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip(), field
    return "", ""


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    separator = "|" if "|" in text else ","
    return [part.strip() for part in text.split(separator) if part.strip()]


def _local_name(value: Any) -> str:
    return str(value).rsplit("}", 1)[-1]
