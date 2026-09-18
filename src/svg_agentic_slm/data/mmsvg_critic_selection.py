"""Deterministic, leakage-safe MMSVG sampling for critic rollouts.

The selector deliberately combines two independent coverage objectives:

* explicit SVG structural edge strata; and
* semantic buckets built from the existing MMSVG description/detail Qwen
  embeddings stored in ChromaDB.

It never embeds text again.  Chroma rows are fetched by their stable
``{domain}:{raw_id}`` IDs and the returned IDs are joined explicitly because
Chroma does not promise to preserve requested-ID order.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from svg_agentic_slm.svg.renderer import CairoSVGRenderer
from svg_agentic_slm.svg.validator import SVGValidator

logger = logging.getLogger(__name__)
ALGORITHM_VERSION = "mmsvg-critic-diverse-v1"
EXPECTED_EMBEDDING_DIMENSION = 1_024
DOMAINS = ("icon", "illustration")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)
PATH_COMMAND_RE = re.compile(r"(?<![A-Za-z])([MmLlHhVvCcSsQqTtAaZz])(?![A-Za-z])")
HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
REQUIRED_COLUMNS = {
    "id",
    "svg",
    "description",
    "keywords",
    "detail",
    "token_len",
}

DATASET_CONTRACTS: dict[str, dict[str, Any]] = {
    "icon": {
        "dataset_id": "OmniSVG/MMSVG-Icon",
        "revision": "8b1980d64000138d9fd14c3bfbd592edcc4b0be9",
        "relative_source": "mmsvg_icon_v2/data",
        "expected_rows": 904_011,
        "expected_shards": 91,
    },
    "illustration": {
        "dataset_id": "OmniSVG/MMSVG-Illustration",
        "revision": "6d81c98ae9bc1f4e1fca80cea496a73cb7f150c1",
        "relative_source": "mmsvg_illustration_v2/data",
        "expected_rows": 255_412,
        "expected_shards": 26,
    },
}


class ChromaCollection(Protocol):
    """Small part of the Chroma collection API needed by this module."""

    def get(self, *, ids: list[str], include: list[str]) -> Mapping[str, Any]: ...

    def count(self) -> int: ...


@dataclass(frozen=True)
class SelectionConfig:
    """Runtime contract for a critic rollout ID export."""

    data_root: Path
    generator_id_dir: Path
    generator_sft_dir: Path
    final_references_path: Path
    benchmark_exclusions: tuple[Path, ...]
    description_chroma_dir: Path
    detail_chroma_dir: Path
    output_dir: Path
    sample_size_per_domain: int = 5_000
    edge_quota_per_domain: int = 1_000
    seed: int = 42
    scan_batch_size: int = 8_192
    chroma_batch_size: int = 8_192
    projection_dimension: int = 64
    lsh_bits: int = 16
    render_size: int = 64
    enforce_source_contract: bool = True


@dataclass(frozen=True)
class Exclusions:
    """Generator and benchmark leakage denylist."""

    generator_raw_ids: frozenset[str]
    generator_raw_ids_by_domain: Mapping[str, frozenset[str]]
    generator_svg_hashes: frozenset[str]
    final_reference_item_ids: frozenset[str]
    final_reference_svg_hashes: frozenset[str]
    benchmark_raw_ids: frozenset[str]
    benchmark_text_hashes: frozenset[str]
    benchmark_svg_hashes: frozenset[str]
    provenance: tuple[Mapping[str, Any], ...] = ()
    final_references_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    """Compact source pointer plus diversity features for one eligible row."""

    domain: str
    record_id: str
    item_id: str
    shard: str
    row_index_in_shard: int
    canonical_svg_sha256: str
    caption_sha256: str
    detail_sha256: str
    token_len: int
    svg_char_length: int
    description_words: int
    detail_words: int
    keyword_count: int
    element_count: int
    path_count: int
    path_command_count: int
    color_count: int
    arc_command_count: int
    curve_command_count: int
    rare_command_count: int
    style_feature_count: int
    advanced_element_count: int
    gradient_count: int
    clip_path_count: int
    mask_count: int
    filter_count: int
    text_element_count: int
    use_element_count: int
    image_element_count: int
    transform_count: int
    stroke_count: int
    opacity_count: int
    viewbox_aspect_ratio: float
    command_signature: str
    tie_break: int
    semantic_bucket: int = -1
    description_detail_cosine: float = math.nan
    selection_reason: str = ""
    selection_rank_in_domain: int = -1


@dataclass
class DomainScan:
    candidates: list[Candidate]
    statistics: Counter[str]
    source_files: list[dict[str, Any]]


@dataclass
class DomainSelection:
    selected: list[Candidate]
    quota_targets: dict[str, int]
    quota_actual: dict[str, int]
    statistics: Counter[str]
    semantic_bucket_count: int


@dataclass(frozen=True)
class EdgeStratum:
    name: str
    weight: int
    score: Callable[[Candidate], tuple[float, ...]]
    eligible: Callable[[Candidate], bool] = lambda _candidate: True


EDGE_STRATA: tuple[EdgeStratum, ...] = (
    EdgeStratum("long_token", 10, lambda c: (float(c.token_len),)),
    EdgeStratum("long_svg", 6, lambda c: (float(c.svg_char_length),)),
    EdgeStratum("many_elements", 6, lambda c: (float(c.element_count),)),
    EdgeStratum("many_paths", 6, lambda c: (float(c.path_count),)),
    EdgeStratum("many_path_commands", 6, lambda c: (float(c.path_command_count),)),
    EdgeStratum("many_colors", 6, lambda c: (float(c.color_count),)),
    EdgeStratum(
        "arc_heavy",
        6,
        lambda c: (float(c.arc_command_count), float(c.path_command_count)),
        lambda c: c.arc_command_count > 0,
    ),
    EdgeStratum(
        "curve_heavy",
        6,
        lambda c: (float(c.curve_command_count), float(c.path_command_count)),
        lambda c: c.curve_command_count > 0,
    ),
    EdgeStratum(
        "rare_commands",
        5,
        lambda c: (float(c.rare_command_count), float(len(c.command_signature))),
        lambda c: c.rare_command_count > 0,
    ),
    EdgeStratum(
        "gradient",
        3,
        lambda c: (float(c.gradient_count),),
        lambda c: c.gradient_count > 0,
    ),
    EdgeStratum(
        "clip_path",
        3,
        lambda c: (float(c.clip_path_count),),
        lambda c: c.clip_path_count > 0,
    ),
    EdgeStratum(
        "mask",
        3,
        lambda c: (float(c.mask_count),),
        lambda c: c.mask_count > 0,
    ),
    EdgeStratum(
        "filter",
        3,
        lambda c: (float(c.filter_count),),
        lambda c: c.filter_count > 0,
    ),
    EdgeStratum(
        "text_element",
        3,
        lambda c: (float(c.text_element_count),),
        lambda c: c.text_element_count > 0,
    ),
    EdgeStratum(
        "use_element",
        3,
        lambda c: (float(c.use_element_count),),
        lambda c: c.use_element_count > 0,
    ),
    EdgeStratum(
        "image_element",
        3,
        lambda c: (float(c.image_element_count),),
        lambda c: c.image_element_count > 0,
    ),
    EdgeStratum(
        "transform",
        3,
        lambda c: (float(c.transform_count),),
        lambda c: c.transform_count > 0,
    ),
    EdgeStratum(
        "stroke",
        3,
        lambda c: (float(c.stroke_count),),
        lambda c: c.stroke_count > 0,
    ),
    EdgeStratum(
        "opacity",
        3,
        lambda c: (float(c.opacity_count),),
        lambda c: c.opacity_count > 0,
    ),
    EdgeStratum(
        "viewbox_aspect_extreme",
        3,
        lambda c: (float(c.viewbox_aspect_ratio),),
        lambda c: c.viewbox_aspect_ratio > 1.0,
    ),
    EdgeStratum(
        "minimal",
        10,
        lambda c: (
            -float(c.path_command_count),
            -float(c.element_count),
            -float(c.color_count),
        ),
    ),
)


def caption_sha256(value: str) -> str:
    """Use the same casefold + whitespace-collapse contract as Generator SFT."""
    normalized = " ".join(value.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_svg(svg: str) -> tuple[str, Any]:
    """Return strict XML C14N and its parsed root without network/entity access."""
    from lxml import etree  # type: ignore[import-untyped]

    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        recover=False,
    )
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    canonical = etree.tostring(root, method="c14n", with_comments=False).decode("utf-8")
    return canonical, root


def canonical_svg_sha256(svg: str) -> str:
    canonical, _root = canonicalize_svg(svg)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_uint64(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _required_text(value: Any) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return text.strip()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    """Read physical lines so U+2028/U+2029 inside JSON strings remain intact."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_id_file(path: Path) -> frozenset[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value:
            raise ValueError(f"Blank ID at {path}:{line_number}.")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}.")
    return frozenset(values)


def load_exclusions(
    generator_id_files: Mapping[str, Path],
    generator_sft_dir: Path,
    benchmark_paths: Sequence[Path],
    final_references_path: Path,
) -> Exclusions:
    """Load and cross-check raw Generator IDs, prepared hashes, and benchmarks."""
    generator_by_domain = {domain: _load_id_file(generator_id_files[domain]) for domain in DOMAINS}
    generator_ids = frozenset().union(*generator_by_domain.values())
    if sum(len(values) for values in generator_by_domain.values()) != len(generator_ids):
        raise ValueError("Generator raw IDs overlap across MMSVG domains.")

    prepared_by_domain: dict[str, set[str]] = {domain: set() for domain in DOMAINS}
    generator_svg_hashes: set[str] = set()
    prepared_files = sorted(generator_sft_dir.glob("*.jsonl"))
    if not prepared_files:
        raise FileNotFoundError(f"No prepared Generator SFT JSONL in {generator_sft_dir}.")
    for path in prepared_files:
        for line_number, payload in _iter_jsonl(path):
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid prepared SFT row at {path}:{line_number}.")
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"Missing SFT metadata at {path}:{line_number}.")
            domain = str(metadata.get("dataset_type", "")).casefold()
            record_id = _required_text(metadata.get("record_id"))
            declared_hash = _required_text(metadata.get("canonical_svg_sha256")).lower()
            output_svg = _required_text(payload.get("output_svg"))
            if domain not in DOMAINS or not record_id or not declared_hash or not output_svg:
                raise ValueError(f"Incomplete SFT identity at {path}:{line_number}.")
            actual_hash = canonical_svg_sha256(output_svg)
            if actual_hash != declared_hash:
                raise ValueError(f"SFT canonical SVG hash mismatch at {path}:{line_number}.")
            if record_id in prepared_by_domain[domain]:
                raise ValueError(f"Duplicate prepared SFT ID {domain}:{record_id}.")
            prepared_by_domain[domain].add(record_id)
            generator_svg_hashes.add(actual_hash)
    for domain in DOMAINS:
        if prepared_by_domain[domain] != set(generator_by_domain[domain]):
            missing = generator_by_domain[domain] - prepared_by_domain[domain]
            extra = prepared_by_domain[domain] - generator_by_domain[domain]
            raise ValueError(
                f"Prepared Generator SFT IDs disagree for {domain}: "
                f"missing={len(missing)}, extra={len(extra)}."
            )

    if not final_references_path.is_file():
        raise FileNotFoundError(final_references_path)
    final_table = pq.read_table(
        final_references_path,
        columns=["item_id", "canonical_svg_sha256"],
    )
    final_item_ids: set[str] = set()
    final_svg_hashes: set[str] = set()
    item_to_hash: dict[str, str] = {}
    for row_index, row in enumerate(final_table.to_pylist()):
        item_id = _required_text(row.get("item_id"))
        svg_hash = _required_text(row.get("canonical_svg_sha256")).lower()
        if not item_id or not svg_hash:
            raise ValueError(
                f"Blank final-reference identity at {final_references_path}:{row_index}."
            )
        if ":" not in item_id or item_id.split(":", 1)[0].casefold() not in DOMAINS:
            raise ValueError(f"Invalid final-reference item_id: {item_id!r}.")
        previous = item_to_hash.setdefault(item_id, svg_hash)
        if previous != svg_hash:
            raise ValueError(f"Conflicting final-reference hash for {item_id}.")
        final_item_ids.add(item_id)
        final_svg_hashes.add(svg_hash)
    final_provenance = {
        "path": str(final_references_path.resolve()),
        "rows": final_table.num_rows,
        "unique_item_ids": len(final_item_ids),
        "unique_canonical_svg_hashes": len(final_svg_hashes),
        "size_bytes": final_references_path.stat().st_size,
        "sha256": file_sha256(final_references_path),
    }

    benchmark_ids: set[str] = set()
    benchmark_text_hashes: set[str] = set()
    benchmark_svg_hashes: set[str] = set()
    provenance: list[Mapping[str, Any]] = []
    for path in benchmark_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = 0
        for line_number, payload in _iter_jsonl(path):
            rows += 1
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid benchmark row at {path}:{line_number}.")
            for key in ("id", "item_id", "record_id"):
                value = _required_text(payload.get(key))
                if value:
                    benchmark_ids.add(value)
            for key in (
                "caption",
                "description",
                "detail",
                "detailed_description",
                "instruction",
                "text",
            ):
                value = _required_text(payload.get(key))
                if value:
                    benchmark_text_hashes.add(caption_sha256(value))
            for key in ("canonical_svg_sha256", "svg_sha256", "svg_hash"):
                value = _required_text(payload.get(key)).lower()
                if value:
                    benchmark_svg_hashes.add(value)
            for key in ("caption_hashes", "detail_hashes"):
                values = payload.get(key, [])
                if isinstance(values, list):
                    benchmark_text_hashes.update(str(value).lower() for value in values)
            values = payload.get("svg_hashes", [])
            if isinstance(values, list):
                benchmark_svg_hashes.update(str(value).lower() for value in values)
        provenance.append(
            {
                "path": str(path.resolve()),
                "rows": rows,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return Exclusions(
        generator_raw_ids=generator_ids,
        generator_raw_ids_by_domain={
            domain: frozenset(values) for domain, values in generator_by_domain.items()
        },
        generator_svg_hashes=frozenset(generator_svg_hashes),
        final_reference_item_ids=frozenset(final_item_ids),
        final_reference_svg_hashes=frozenset(final_svg_hashes),
        benchmark_raw_ids=frozenset(benchmark_ids),
        benchmark_text_hashes=frozenset(benchmark_text_hashes),
        benchmark_svg_hashes=frozenset(benchmark_svg_hashes),
        provenance=tuple(provenance),
        final_references_provenance=final_provenance,
    )


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    # lxml exposes parsed names in Clark notation. Avoid constructing millions
    # of QName objects during the full-corpus structural scan.
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _structure_features(
    *,
    canonical_svg: str,
    root: Any,
    description: str,
    detail: str,
    keywords: str,
    token_len: int,
) -> dict[str, Any]:
    elements = [element for element in root.iter() if isinstance(element.tag, str)]
    names = [_local_name(element.tag) for element in elements]
    path_data = " ".join(
        str(element.get("d", "")) for element in elements if _local_name(element.tag) == "path"
    )
    commands = [match.group(1).upper() for match in PATH_COMMAND_RE.finditer(path_data)]
    command_counts = Counter(commands)
    rare_commands = sum(command_counts[name] for name in ("H", "V", "Q", "S", "T", "A"))
    curve_commands = sum(command_counts[name] for name in ("C", "S", "Q", "T"))
    advanced_names = {
        "defs",
        "linearGradient",
        "radialGradient",
        "clipPath",
        "mask",
        "filter",
        "text",
        "use",
        "image",
    }
    advanced_count = sum(name in advanced_names for name in names)
    gradient_count = names.count("linearGradient") + names.count("radialGradient")
    clip_path_count = names.count("clipPath")
    mask_count = names.count("mask")
    filter_count = names.count("filter")
    text_element_count = names.count("text")
    use_element_count = names.count("use")
    image_element_count = names.count("image")
    transform_count = sum(element.get("transform") is not None for element in elements)
    stroke_count = sum(
        any(_local_name(name).casefold().startswith("stroke") for name in element.attrib)
        for element in elements
    )
    opacity_count = sum(
        any("opacity" in _local_name(name).casefold() for name in element.attrib)
        for element in elements
    )
    viewbox_aspect_ratio = 1.0
    viewbox = root.get("viewBox") or root.get("viewbox")
    if viewbox:
        try:
            values = [float(value) for value in re.split(r"[\s,]+", viewbox.strip())]
            width, height = abs(values[2]), abs(values[3])
            if len(values) == 4 and width > 0.0 and height > 0.0:
                viewbox_aspect_ratio = max(width / height, height / width)
        except (IndexError, ValueError, ZeroDivisionError):
            viewbox_aspect_ratio = 1.0
    style_needles = (
        "stroke=",
        "opacity=",
        'fill="none"',
        "currentColor",
        "transform=",
        "url(#",
    )
    colors = {color.casefold() for color in HEX_COLOR_RE.findall(canonical_svg)}
    keywords_list = [part.strip() for part in keywords.split(",") if part.strip()]
    return {
        "token_len": token_len,
        "svg_char_length": len(canonical_svg),
        "description_words": len(WORD_RE.findall(description)),
        "detail_words": len(WORD_RE.findall(detail)),
        "keyword_count": len(keywords_list),
        "element_count": len(elements),
        "path_count": names.count("path"),
        "path_command_count": len(commands),
        "color_count": len(colors),
        "arc_command_count": command_counts["A"],
        "curve_command_count": curve_commands,
        "rare_command_count": rare_commands,
        "style_feature_count": sum(needle in canonical_svg for needle in style_needles),
        "advanced_element_count": advanced_count,
        "gradient_count": gradient_count,
        "clip_path_count": clip_path_count,
        "mask_count": mask_count,
        "filter_count": filter_count,
        "text_element_count": text_element_count,
        "use_element_count": use_element_count,
        "image_element_count": image_element_count,
        "transform_count": transform_count,
        "stroke_count": stroke_count,
        "opacity_count": opacity_count,
        "viewbox_aspect_ratio": viewbox_aspect_ratio,
        "command_signature": "".join(sorted(command_counts)),
    }


def scan_domain(
    *,
    domain: str,
    shards: Sequence[Path],
    exclusions: Exclusions,
    seed: int,
    batch_size: int,
    validator: SVGValidator | None = None,
) -> DomainScan:
    """Stream a domain and retain only strict, leakage-safe candidate metadata."""
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported MMSVG domain: {domain}")
    if not shards:
        raise FileNotFoundError(f"No Parquet shards for {domain}.")
    validator = validator or SVGValidator()
    statistics: Counter[str] = Counter()
    candidates: list[Candidate] = []
    seen_source_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []

    ordered_shards = sorted(Path(path) for path in shards)
    for shard_number, shard in enumerate(ordered_shards, 1):
        parquet_file = pq.ParquetFile(shard)
        missing = REQUIRED_COLUMNS - set(parquet_file.schema_arrow.names)
        if missing:
            raise RuntimeError(f"{shard} is missing columns: {', '.join(sorted(missing))}.")
        source_files.append(
            {
                "path": str(shard.resolve()),
                "rows": parquet_file.metadata.num_rows,
                "size_bytes": shard.stat().st_size,
                "mtime_ns": shard.stat().st_mtime_ns,
            }
        )
        row_index = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=sorted(REQUIRED_COLUMNS),
            use_threads=True,
        ):
            for row in batch.to_pylist():
                statistics["rows_seen"] += 1
                record_id = _required_text(row.get("id"))
                if not record_id:
                    statistics["rejected_missing_field"] += 1
                    row_index += 1
                    continue
                if record_id in seen_source_ids:
                    raise RuntimeError(f"Duplicate raw MMSVG ID: {domain}:{record_id}.")
                seen_source_ids.add(record_id)
                if record_id in exclusions.generator_raw_ids:
                    statistics["rejected_generator_id"] += 1
                    row_index += 1
                    continue
                if record_id in exclusions.benchmark_raw_ids:
                    statistics["rejected_benchmark_id"] += 1
                    row_index += 1
                    continue
                item_id = f"{domain}:{record_id}"
                if item_id in exclusions.final_reference_item_ids:
                    statistics["rejected_final_reference_item_id"] += 1
                    row_index += 1
                    continue
                svg = _required_text(row.get("svg"))
                description = _required_text(row.get("description"))
                detail = _required_text(row.get("detail"))
                keywords = _required_text(row.get("keywords"))
                if not svg or not description or not detail or not keywords:
                    statistics["rejected_missing_field"] += 1
                    row_index += 1
                    continue
                description_hash = caption_sha256(description)
                detail_hash = caption_sha256(detail)
                if (
                    description_hash in exclusions.benchmark_text_hashes
                    or detail_hash in exclusions.benchmark_text_hashes
                ):
                    statistics["rejected_benchmark_text"] += 1
                    row_index += 1
                    continue
                try:
                    validation = validator.validate(svg)
                except Exception:
                    statistics["rejected_svg_validation"] += 1
                    row_index += 1
                    continue
                if not validation.is_valid:
                    statistics["rejected_svg_validation"] += 1
                    row_index += 1
                    continue
                try:
                    canonical_svg, root = canonicalize_svg(svg)
                except Exception:
                    statistics["rejected_svg_canonicalization"] += 1
                    row_index += 1
                    continue
                canonical_hash = hashlib.sha256(canonical_svg.encode("utf-8")).hexdigest()
                if canonical_hash in exclusions.generator_svg_hashes:
                    statistics["rejected_generator_svg_hash"] += 1
                    row_index += 1
                    continue
                if canonical_hash in exclusions.benchmark_svg_hashes:
                    statistics["rejected_benchmark_svg_hash"] += 1
                    row_index += 1
                    continue
                if canonical_hash in exclusions.final_reference_svg_hashes:
                    statistics["rejected_final_reference_svg_hash"] += 1
                    row_index += 1
                    continue
                raw_token_len = row.get("token_len")
                token_len = int(raw_token_len) if raw_token_len is not None else 0
                features = _structure_features(
                    canonical_svg=canonical_svg,
                    root=root,
                    description=description,
                    detail=detail,
                    keywords=keywords,
                    token_len=token_len,
                )
                candidates.append(
                    Candidate(
                        domain=domain,
                        record_id=record_id,
                        item_id=item_id,
                        shard=str(shard.resolve()),
                        row_index_in_shard=row_index,
                        canonical_svg_sha256=canonical_hash,
                        caption_sha256=description_hash,
                        detail_sha256=detail_hash,
                        tie_break=stable_uint64(seed, domain, record_id),
                        **features,
                    )
                )
                statistics["eligible_before_render"] += 1
                row_index += 1
        if row_index != parquet_file.metadata.num_rows:
            raise RuntimeError(f"Parquet row count changed while scanning {shard}.")
        logger.info(
            "critic-select scan domain=%s shard=%d/%d rows=%d eligible=%d",
            domain,
            shard_number,
            len(ordered_shards),
            statistics["rows_seen"],
            statistics["eligible_before_render"],
        )
    statistics["unique_source_ids"] = len(seen_source_ids)
    return DomainScan(candidates, statistics, source_files)


def _fetch_embeddings(
    collection: ChromaCollection,
    item_ids: Sequence[str],
    *,
    expected_domain: str,
) -> np.ndarray:
    result = collection.get(ids=list(item_ids), include=["embeddings", "metadatas"])
    returned_ids = [str(value) for value in result.get("ids", [])]
    raw_embeddings = result.get("embeddings")
    raw_metadatas = result.get("metadatas")
    if raw_embeddings is None or raw_metadatas is None:
        raise RuntimeError("Chroma get did not return embeddings and metadatas.")
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    metadatas = list(raw_metadatas)
    if embeddings.ndim != 2 or len(returned_ids) != embeddings.shape[0]:
        raise RuntimeError("Malformed Chroma embedding batch.")
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Expected {EXPECTED_EMBEDDING_DIMENSION}D Qwen embeddings; got {embeddings.shape[1]}D."
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Chroma returned a non-finite embedding.")
    if np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise RuntimeError("Chroma returned a zero-norm embedding.")
    if len(returned_ids) != len(set(returned_ids)):
        raise RuntimeError("Chroma returned duplicate IDs.")
    positions = {item_id: index for index, item_id in enumerate(returned_ids)}
    if set(positions) != set(item_ids):
        missing = set(item_ids) - set(positions)
        extra = set(positions) - set(item_ids)
        raise RuntimeError(f"Chroma ID mismatch: missing={len(missing)}, extra={len(extra)}.")
    for index, item_id in enumerate(returned_ids):
        metadata = metadatas[index] if index < len(metadatas) else None
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"Missing Chroma metadata for {item_id}.")
        domain = str(metadata.get("dataset_type", "")).casefold()
        record_id = str(metadata.get("record_id", ""))
        if domain != expected_domain or item_id != f"{domain}:{record_id}":
            raise RuntimeError(f"Mismatched Chroma pointer metadata for {item_id}.")
    return np.stack([embeddings[positions[item_id]] for item_id in item_ids])


def annotate_semantic_buckets(
    candidates: Sequence[Candidate],
    *,
    description_collection: ChromaCollection,
    detail_collection: ChromaCollection,
    seed: int,
    batch_size: int,
    projection_dimension: int,
    lsh_bits: int,
) -> dict[str, Any]:
    """Attach dual-embedding cosine agreement and deterministic semantic LSH."""
    if not candidates:
        raise ValueError("Cannot annotate an empty candidate pool.")
    if batch_size <= 0 or projection_dimension <= 0:
        raise ValueError("Embedding batch and projection dimensions must be positive.")
    if not 1 <= lsh_bits <= min(projection_dimension, 20):
        raise ValueError("lsh_bits must be between 1 and min(projection_dimension, 20).")
    projection: np.ndarray | None = None
    dimension: int | None = None
    rng = np.random.default_rng(seed)
    total_batches = math.ceil(len(candidates) / batch_size)
    for batch_number, start in enumerate(range(0, len(candidates), batch_size), 1):
        batch = candidates[start : start + batch_size]
        item_ids = [candidate.item_id for candidate in batch]
        domain = batch[0].domain
        if any(candidate.domain != domain for candidate in batch):
            raise ValueError("A semantic batch cannot mix MMSVG domains.")
        description = _fetch_embeddings(
            description_collection,
            item_ids,
            expected_domain=domain,
        )
        detail = _fetch_embeddings(detail_collection, item_ids, expected_domain=domain)
        if description.shape != detail.shape:
            raise RuntimeError("Description/detail Chroma dimensions differ.")
        if dimension is None:
            dimension = description.shape[1]
            projection = rng.standard_normal(
                (dimension, projection_dimension), dtype=np.float32
            ) / math.sqrt(projection_dimension)
        if description.shape[1] != dimension or projection is None:
            raise RuntimeError("Chroma embedding dimension changed during export.")
        description /= np.maximum(np.linalg.norm(description, axis=1, keepdims=True), 1e-12)
        detail /= np.maximum(np.linalg.norm(detail, axis=1, keepdims=True), 1e-12)
        agreement = np.sum(description * detail, axis=1)
        # The 60/40 policy mirrors the project's prompt-source mixture.
        combined = 0.4 * description + 0.6 * detail
        combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
        projected = combined @ projection
        projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
        bit_matrix = projected[:, :lsh_bits] >= 0
        powers = np.left_shift(np.uint32(1), np.arange(lsh_bits, dtype=np.uint32))
        buckets = bit_matrix.astype(np.uint32) @ powers
        for candidate, cosine, bucket in zip(batch, agreement, buckets, strict=True):
            candidate.description_detail_cosine = float(cosine)
            candidate.semantic_bucket = int(bucket)
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            logger.info(
                "critic-select embeddings domain=%s batch=%d/%d rows=%d/%d",
                domain,
                batch_number,
                total_batches,
                min(start + len(batch), len(candidates)),
                len(candidates),
            )
    assert projection is not None and dimension is not None
    return {
        "embedding_dimension": dimension,
        "projection_dimension": projection_dimension,
        "lsh_bits": lsh_bits,
        "projection_seed": seed,
        "projection_sha256": hashlib.sha256(projection.tobytes(order="C")).hexdigest(),
        "combination": "l2_normalize(0.4*description + 0.6*detail)",
    }


def _allocate_edge_quotas(total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("Edge quota cannot be negative.")
    weights = sum(stratum.weight for stratum in EDGE_STRATA)
    raw = [total * stratum.weight / weights for stratum in EDGE_STRATA]
    allocated = [math.floor(value) for value in raw]
    remainder = total - sum(allocated)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - allocated[index]), index),
    )
    for index in order[:remainder]:
        allocated[index] += 1
    return {stratum.name: allocated[index] for index, stratum in enumerate(EDGE_STRATA)}


def _rank_for_stratum(
    candidates: Sequence[Candidate],
    stratum: EdgeStratum,
    desired: int,
) -> list[Candidate]:
    eligible = (candidate for candidate in candidates if stratum.eligible(candidate))
    capacity = max(2_048, desired * 32)
    return heapq.nsmallest(
        capacity,
        eligible,
        key=lambda candidate: (
            *(-value for value in stratum.score(candidate)),
            candidate.tie_break,
            candidate.item_id,
        ),
    )


def _composite_edge_score(candidate: Candidate) -> tuple[float, ...]:
    return (
        math.log1p(max(candidate.token_len, 0))
        + math.log1p(candidate.path_command_count)
        + math.log1p(candidate.element_count)
        + math.log1p(candidate.color_count),
        float(candidate.advanced_element_count + candidate.style_feature_count),
        float(candidate.rare_command_count),
    )


def _bucket_order(
    buckets: Mapping[int, Sequence[Candidate]],
    *,
    anchor_codes: Sequence[int],
    target: int,
    lsh_bits: int,
    seed: int,
    domain: str,
) -> list[int]:
    codes = np.asarray(sorted(buckets), dtype=np.uint32)
    if target <= 0 or codes.size == 0:
        return []
    alive = np.ones(codes.shape[0], dtype=bool)
    lookup = np.asarray([value.bit_count() for value in range(1 << lsh_bits)], dtype=np.uint8)
    minimum = np.full(codes.shape[0], lsh_bits + 1, dtype=np.uint8)
    for anchor in sorted(set(anchor_codes)):
        minimum = np.minimum(minimum, lookup[np.bitwise_xor(codes, np.uint32(anchor))])
    sizes = np.asarray([len(buckets[int(code)]) for code in codes], dtype=np.int64)
    ties = np.asarray(
        [stable_uint64(seed, "bucket", domain, int(code)) for code in codes],
        dtype=np.uint64,
    )
    output: list[int] = []
    for _ in range(min(target, len(codes))):
        alive_indices = np.flatnonzero(alive)
        best_distance = int(minimum[alive_indices].max())
        choices = alive_indices[minimum[alive_indices] == best_distance]
        smallest_bucket = int(sizes[choices].min())
        choices = choices[sizes[choices] == smallest_bucket]
        chosen_index = int(choices[np.argmin(ties[choices])])
        chosen = int(codes[chosen_index])
        output.append(chosen)
        alive[chosen_index] = False
        minimum = np.minimum(minimum, lookup[np.bitwise_xor(codes, np.uint32(chosen))])
    return output


def select_domain_candidates(
    candidates: Sequence[Candidate],
    *,
    domain: str,
    sample_size: int,
    edge_quota: int,
    lsh_bits: int,
    seed: int,
    accept: Callable[[Candidate], bool],
    globally_selected_hashes: set[str] | None = None,
    globally_selected_text_pairs: set[tuple[str, str]] | None = None,
) -> DomainSelection:
    """Choose one domain, enforcing edge quotas before semantic coverage."""
    if sample_size <= 0 or not 0 <= edge_quota <= sample_size:
        raise ValueError("Require 0 <= edge_quota <= positive sample_size.")
    if any(candidate.domain != domain for candidate in candidates):
        raise ValueError("Candidate domain mismatch.")
    if any(candidate.semantic_bucket < 0 for candidate in candidates):
        raise ValueError("Semantic buckets must be annotated before selection.")
    selected_hashes = globally_selected_hashes if globally_selected_hashes is not None else set()
    selected_text_pairs = (
        globally_selected_text_pairs if globally_selected_text_pairs is not None else set()
    )
    selected_ids: set[str] = set()
    selected: list[Candidate] = []
    stats: Counter[str] = Counter()

    def try_accept(candidate: Candidate, reason: str) -> bool:
        if candidate.item_id in selected_ids:
            return False
        if candidate.canonical_svg_sha256 in selected_hashes:
            stats["rejected_selected_canonical_duplicate"] += 1
            return False
        text_pair = (candidate.caption_sha256, candidate.detail_sha256)
        if text_pair in selected_text_pairs:
            stats["rejected_selected_text_pair_duplicate"] += 1
            return False
        if not accept(candidate):
            stats["rejected_render"] += 1
            return False
        candidate.selection_reason = reason
        candidate.selection_rank_in_domain = len(selected)
        selected.append(candidate)
        selected_ids.add(candidate.item_id)
        selected_hashes.add(candidate.canonical_svg_sha256)
        selected_text_pairs.add(text_pair)
        return True

    quota_targets = _allocate_edge_quotas(edge_quota)
    quota_actual: dict[str, int] = {name: 0 for name in quota_targets}
    for stratum in EDGE_STRATA:
        desired = quota_targets[stratum.name]
        for candidate in _rank_for_stratum(candidates, stratum, desired):
            if quota_actual[stratum.name] >= desired:
                break
            if try_accept(candidate, f"edge:{stratum.name}"):
                quota_actual[stratum.name] += 1

    deficit = edge_quota - len(selected)
    if deficit > 0:
        fallback = heapq.nsmallest(
            max(2_048, deficit * 64),
            candidates,
            key=lambda candidate: (
                *(-value for value in _composite_edge_score(candidate)),
                candidate.tie_break,
                candidate.item_id,
            ),
        )
        for candidate in fallback:
            if len(selected) >= edge_quota:
                break
            try_accept(candidate, "edge:fallback")
    if len(selected) != edge_quota:
        raise RuntimeError(
            f"Could not satisfy {domain} structural edge quota: {len(selected)}/{edge_quota}."
        )

    semantic_target = sample_size - edge_quota
    buckets: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.item_id not in selected_ids:
            buckets[candidate.semantic_bucket].append(candidate)
    for values in buckets.values():
        values.sort(key=lambda candidate: (candidate.tie_break, candidate.item_id))
    bucket_order = _bucket_order(
        buckets,
        anchor_codes=[candidate.semantic_bucket for candidate in selected],
        target=min(len(buckets), semantic_target + 2_048),
        lsh_bits=lsh_bits,
        seed=seed,
        domain=domain,
    )
    semantic_added = 0
    next_index: dict[int, int] = defaultdict(int)
    for code in bucket_order:
        while next_index[code] < len(buckets[code]):
            candidate = buckets[code][next_index[code]]
            next_index[code] += 1
            if try_accept(candidate, "semantic:lsh_maximin"):
                semantic_added += 1
                break
        if semantic_added >= semantic_target:
            break

    # If there are fewer usable buckets than requested, round-robin through
    # additional members before falling back to a global deterministic order.
    while semantic_added < semantic_target:
        made_progress = False
        for code in bucket_order:
            while next_index[code] < len(buckets[code]):
                candidate = buckets[code][next_index[code]]
                next_index[code] += 1
                if try_accept(candidate, "semantic:lsh_round_robin"):
                    semantic_added += 1
                    made_progress = True
                    break
            if semantic_added >= semantic_target:
                break
        if not made_progress:
            break
    if semantic_added < semantic_target:
        for candidate in sorted(candidates, key=lambda value: (value.tie_break, value.item_id)):
            if try_accept(candidate, "semantic:deterministic_backfill"):
                semantic_added += 1
            if semantic_added >= semantic_target:
                break
    if len(selected) != sample_size:
        raise RuntimeError(
            f"Insufficient strict {domain} candidates: selected={len(selected)}, "
            f"required={sample_size}."
        )
    stats["selected_edge"] = edge_quota
    stats["selected_semantic"] = semantic_target
    stats["selected_total"] = len(selected)
    stats["selected_semantic_buckets"] = len(
        {
            candidate.semantic_bucket
            for candidate in selected
            if candidate.selection_reason.startswith("semantic:")
        }
    )
    return DomainSelection(
        selected=selected,
        quota_targets=quota_targets,
        quota_actual=quota_actual,
        statistics=stats,
        semantic_bucket_count=len(buckets),
    )


class SVGPointerLoader:
    """Hydrate only selected SVG rows from their immutable Parquet pointers."""

    def __init__(self, *, cache_size: int = 64) -> None:
        self._files: dict[str, pq.ParquetFile] = {}
        self._cache: OrderedDict[tuple[str, int], pa.Table] = OrderedDict()
        self._cache_size = cache_size

    def svg(self, candidate: Candidate) -> str:
        parquet_file = self._files.setdefault(candidate.shard, pq.ParquetFile(candidate.shard))
        row_start = 0
        row_group = -1
        for index in range(parquet_file.num_row_groups):
            count = parquet_file.metadata.row_group(index).num_rows
            if candidate.row_index_in_shard < row_start + count:
                row_group = index
                break
            row_start += count
        if row_group < 0:
            raise RuntimeError(f"Invalid Parquet pointer for {candidate.item_id}.")
        key = (candidate.shard, row_group)
        table = self._cache.get(key)
        if table is None:
            table = parquet_file.read_row_group(row_group, columns=["id", "svg"])
            self._cache[key] = table
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        local_index = candidate.row_index_in_shard - row_start
        row = table.slice(local_index, 1).to_pylist()[0]
        if _required_text(row.get("id")) != candidate.record_id:
            raise RuntimeError(f"Parquet pointer ID mismatch for {candidate.item_id}.")
        return _required_text(row.get("svg"))


def make_strict_acceptor(
    *,
    render_size: int,
    pointer_loader: SVGPointerLoader | None = None,
    validator: SVGValidator | None = None,
    renderer: CairoSVGRenderer | None = None,
) -> Callable[[Candidate], bool]:
    """Build a cached final strict-validation/C14N/CairoSVG acceptance gate."""
    loader = pointer_loader or SVGPointerLoader()
    validator = validator or SVGValidator()
    renderer = renderer or CairoSVGRenderer()
    cache: dict[str, bool] = {}

    def accept(candidate: Candidate) -> bool:
        cached = cache.get(candidate.canonical_svg_sha256)
        if cached is not None:
            return cached
        try:
            svg = loader.svg(candidate)
            validation = validator.validate(svg)
            if not validation.is_valid:
                cache[candidate.canonical_svg_sha256] = False
                return False
            canonical, _root = canonicalize_svg(svg)
            canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if canonical_hash != candidate.canonical_svg_sha256:
                cache[candidate.canonical_svg_sha256] = False
                return False
            png = renderer.render_bytes(
                canonical,
                output_width=render_size,
                output_height=render_size,
            )
            result = bool(png.startswith(PNG_MAGIC))
        except Exception:
            result = False
        cache[candidate.canonical_svg_sha256] = result
        return result

    return accept


def _open_chroma_collection(
    directory: Path,
    *,
    expected_field: str,
) -> tuple[ChromaCollection, dict[str, Any]]:
    manifests = sorted(directory.glob("*.manifest.json"))
    matching: list[tuple[Path, dict[str, Any]]] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("document_field") == expected_field:
            matching.append((path, payload))
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one {expected_field} Chroma manifest in {directory}; "
            f"found {len(matching)}."
        )
    manifest_path, manifest = matching[0]
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required to reuse MMSVG embeddings.") from exc
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_collection(str(manifest["collection_name"]), embedding_function=None)
    if collection.count() != sum(
        int(source["expected_rows"]) for source in manifest.get("sources", [])
    ):
        raise RuntimeError(f"Chroma row-count contract failed for {directory}.")
    output = dict(manifest)
    output["manifest_path"] = str(manifest_path.resolve())
    output["manifest_sha256"] = file_sha256(manifest_path)
    return collection, output


def _validate_dual_chroma_contracts(
    description: Mapping[str, Any], detail: Mapping[str, Any]
) -> None:
    for key in ("embedding_model", "embedding_revision", "embedding_dimension", "sources"):
        if description.get(key) != detail.get(key):
            raise RuntimeError(f"Description/detail Chroma contract differs at {key}.")
    if description.get("document_field") != "description":
        raise RuntimeError("Description Chroma manifest has the wrong document field.")
    if detail.get("document_field") != "detail":
        raise RuntimeError("Detail Chroma manifest has the wrong document field.")
    if int(description.get("embedding_dimension", -1)) != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Critic selection requires {EXPECTED_EMBEDDING_DIMENSION}D Qwen embeddings."
        )


def _audit_row(candidate: Candidate) -> dict[str, Any]:
    row = asdict(candidate)
    # ``stable_uint64`` intentionally spans the full unsigned 64-bit range.
    # Arrow's inferred Python-int type is signed int64, so preserve the value
    # losslessly as a fixed-width hexadecimal string in the audit artifact.
    row["tie_break"] = f"{candidate.tie_break:016x}"
    row["validator_pass"] = True
    row["canonicalization_pass"] = True
    row["renderable_pass"] = True
    row["generator_id_overlap"] = False
    row["generator_svg_hash_overlap"] = False
    row["benchmark_overlap"] = False
    return row


def _selected_feature_coverage(candidates: Sequence[Candidate]) -> dict[str, int]:
    """Report final coverage for every explicitly targeted rare SVG feature."""
    predicates: dict[str, Callable[[Candidate], bool]] = {
        "gradient": lambda c: c.gradient_count > 0,
        "clip_path": lambda c: c.clip_path_count > 0,
        "mask": lambda c: c.mask_count > 0,
        "filter": lambda c: c.filter_count > 0,
        "text_element": lambda c: c.text_element_count > 0,
        "use_element": lambda c: c.use_element_count > 0,
        "image_element": lambda c: c.image_element_count > 0,
        "transform": lambda c: c.transform_count > 0,
        "stroke": lambda c: c.stroke_count > 0,
        "opacity": lambda c: c.opacity_count > 0,
        "viewbox_aspect_extreme_ge_2": lambda c: c.viewbox_aspect_ratio >= 2.0,
        "arc_command": lambda c: c.arc_command_count > 0,
        "curve_command": lambda c: c.curve_command_count > 0,
    }
    return {
        name: sum(predicate(candidate) for candidate in candidates)
        for name, predicate in predicates.items()
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_outputs(
    *,
    output_dir: Path,
    selections: Mapping[str, DomainSelection],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        outputs: dict[str, Any] = {}
        all_rows: list[dict[str, Any]] = []
        for domain in DOMAINS:
            selection = selections[domain]
            output_name = f"mmsvg_{domain}_critic_rollout_{len(selection.selected)}_ids.txt"
            path = stage / output_name
            path.write_text(
                "\n".join(candidate.record_id for candidate in selection.selected) + "\n",
                encoding="utf-8",
            )
            outputs[domain] = {
                "file": output_name,
                "rows": len(selection.selected),
                "unique_ids": len({candidate.record_id for candidate in selection.selected}),
                "sha256": file_sha256(path),
            }
            all_rows.extend(_audit_row(candidate) for candidate in selection.selected)
        audit_path = stage / "selection_audit.parquet"
        pq.write_table(
            pa.Table.from_pylist(all_rows),
            audit_path,
            compression="zstd",
            use_dictionary=True,
        )
        outputs["selection_audit"] = {
            "file": audit_path.name,
            "rows": len(all_rows),
            "sha256": file_sha256(audit_path),
        }
        manifest["outputs"] = outputs
        manifest_path = stage / "manifest.json"
        _write_json(manifest_path, manifest)
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def export_critic_rollout_ids(config: SelectionConfig) -> dict[str, Any]:
    """Execute the full deterministic export and atomically publish artifacts."""
    if config.sample_size_per_domain <= 0:
        raise ValueError("sample_size_per_domain must be positive.")
    if not 0 <= config.edge_quota_per_domain <= config.sample_size_per_domain:
        raise ValueError("edge_quota_per_domain must be within the sample size.")
    generator_id_files = {
        "icon": config.generator_id_dir / "mmsvg_icon_sft_10000_ids.txt",
        "illustration": (config.generator_id_dir / "mmsvg_illustration_sft_10000_ids.txt"),
    }
    exclusions = load_exclusions(
        generator_id_files,
        config.generator_sft_dir,
        config.benchmark_exclusions,
        config.final_references_path,
    )
    logger.info(
        "critic-select exclusions generator=%d final_refs=%d benchmark_ids=%d",
        len(exclusions.generator_raw_ids),
        len(exclusions.final_reference_item_ids),
        len(exclusions.benchmark_raw_ids),
    )
    if config.enforce_source_contract:
        final_provenance = exclusions.final_references_provenance
        if (
            final_provenance.get("rows") != 60_000
            or final_provenance.get("unique_item_ids") != 52_999
            or final_provenance.get("unique_canonical_svg_hashes") != 52_999
        ):
            raise RuntimeError(
                f"Pinned Generator final-reference contract failed: {dict(final_provenance)}"
            )
    description_collection, description_contract = _open_chroma_collection(
        config.description_chroma_dir,
        expected_field="description",
    )
    detail_collection, detail_contract = _open_chroma_collection(
        config.detail_chroma_dir,
        expected_field="detail",
    )
    _validate_dual_chroma_contracts(description_contract, detail_contract)

    selections: dict[str, DomainSelection] = {}
    scans: dict[str, DomainScan] = {}
    semantic_contract: dict[str, Any] | None = None
    globally_selected_hashes: set[str] = set()
    globally_selected_text_pairs: set[tuple[str, str]] = set()
    accept = make_strict_acceptor(render_size=config.render_size)
    for domain in DOMAINS:
        logger.info("critic-select starting domain=%s", domain)
        source_contract = DATASET_CONTRACTS[domain]
        source_dir = config.data_root / str(source_contract["relative_source"])
        shards = sorted(source_dir.glob("*.parquet"))
        scan = scan_domain(
            domain=domain,
            shards=shards,
            exclusions=exclusions,
            seed=config.seed,
            batch_size=config.scan_batch_size,
        )
        if config.enforce_source_contract:
            rows = sum(int(item["rows"]) for item in scan.source_files)
            if (
                len(shards) != source_contract["expected_shards"]
                or rows != source_contract["expected_rows"]
            ):
                raise RuntimeError(
                    f"Pinned MMSVG source contract failed for {domain}: "
                    f"shards={len(shards)}, rows={rows}."
                )
        current_semantic = annotate_semantic_buckets(
            scan.candidates,
            description_collection=description_collection,
            detail_collection=detail_collection,
            seed=config.seed,
            batch_size=config.chroma_batch_size,
            projection_dimension=config.projection_dimension,
            lsh_bits=config.lsh_bits,
        )
        if semantic_contract is None:
            semantic_contract = current_semantic
        elif current_semantic != semantic_contract:
            raise RuntimeError("Semantic projection contract changed across domains.")
        selection = select_domain_candidates(
            scan.candidates,
            domain=domain,
            sample_size=config.sample_size_per_domain,
            edge_quota=config.edge_quota_per_domain,
            lsh_bits=config.lsh_bits,
            seed=config.seed,
            accept=accept,
            globally_selected_hashes=globally_selected_hashes,
            globally_selected_text_pairs=globally_selected_text_pairs,
        )
        scans[domain] = scan
        selections[domain] = selection
        logger.info(
            "critic-select selected domain=%s rows=%d edge=%d semantic=%d",
            domain,
            len(selection.selected),
            selection.statistics["selected_edge"],
            selection.statistics["selected_semantic"],
        )

    selected_all = [candidate for domain in DOMAINS for candidate in selections[domain].selected]
    selected_ids = {candidate.record_id for candidate in selected_all}
    selected_hashes = {candidate.canonical_svg_sha256 for candidate in selected_all}
    selected_text_pairs = {
        (candidate.caption_sha256, candidate.detail_sha256) for candidate in selected_all
    }
    if selected_ids & exclusions.generator_raw_ids:
        raise RuntimeError("Generator raw ID leakage survived final audit.")
    if selected_hashes & exclusions.generator_svg_hashes:
        raise RuntimeError("Generator SVG hash leakage survived final audit.")
    selected_item_ids = {candidate.item_id for candidate in selected_all}
    if selected_item_ids & exclusions.final_reference_item_ids:
        raise RuntimeError("Generator final-reference item ID leakage survived final audit.")
    if selected_hashes & exclusions.final_reference_svg_hashes:
        raise RuntimeError("Generator final-reference SVG hash leakage survived final audit.")
    if len(selected_hashes) != len(selected_all):
        raise RuntimeError("Selected canonical SVG hashes are not globally unique.")
    if len(selected_text_pairs) != len(selected_all):
        raise RuntimeError("Selected caption/detail hash pairs are not globally unique.")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "MMSVG critic rollout diverse holdout IDs",
        "selection": {
            "seed": config.seed,
            "domain_order": list(DOMAINS),
            "sample_size_per_domain": config.sample_size_per_domain,
            "edge_quota_per_domain": config.edge_quota_per_domain,
            "semantic_quota_per_domain": (
                config.sample_size_per_domain - config.edge_quota_per_domain
            ),
            "edge_strata_weights": {stratum.name: stratum.weight for stratum in EDGE_STRATA},
            "semantic_method": (
                "dual-Qwen weighted embedding, seeded random projection, "
                "LSH bucket maximin Hamming coverage"
            ),
        },
        "leakage_policy": {
            "generator": ["raw_id", "canonical_svg_sha256"],
            "generator_final_top3": ["item_id", "canonical_svg_sha256"],
            "benchmark": ["raw_id", "caption_or_detail_sha256", "canonical_svg_sha256"],
            "selected_deduplication": "global canonical_svg_sha256",
            "selected_text_deduplication": "global exact (caption_sha256, detail_sha256)",
            "caption_hash_contract": "SHA-256(casefold + whitespace collapse)",
            "svg_hash_contract": "SHA-256(strict lxml XML C14N, comments removed)",
        },
        "exclusions": {
            "generator_raw_ids": len(exclusions.generator_raw_ids),
            "generator_svg_hashes": len(exclusions.generator_svg_hashes),
            "final_reference_item_ids": len(exclusions.final_reference_item_ids),
            "final_reference_svg_hashes": len(exclusions.final_reference_svg_hashes),
            "final_references": dict(exclusions.final_references_provenance),
            "benchmark_raw_ids": len(exclusions.benchmark_raw_ids),
            "benchmark_text_hashes": len(exclusions.benchmark_text_hashes),
            "benchmark_svg_hashes": len(exclusions.benchmark_svg_hashes),
            "generator_id_files": {
                domain: {
                    "path": str(path.resolve()),
                    "rows": len(exclusions.generator_raw_ids_by_domain[domain]),
                    "sha256": file_sha256(path),
                }
                for domain, path in generator_id_files.items()
            },
            "benchmark_files": list(exclusions.provenance),
        },
        "chroma": {
            "description": description_contract,
            "detail": detail_contract,
            "export_api": (
                "collection.get(ids=batch, include=['embeddings','metadatas']); "
                "explicit returned-ID join"
            ),
            "batch_size": config.chroma_batch_size,
            "semantic": semantic_contract,
        },
        "sources": {
            domain: {
                **DATASET_CONTRACTS[domain],
                "files": scans[domain].source_files,
                "scan_statistics": dict(scans[domain].statistics),
            }
            for domain in DOMAINS
        },
        "domain_results": {
            domain: {
                "quota_targets": selections[domain].quota_targets,
                "quota_actual": selections[domain].quota_actual,
                "statistics": dict(selections[domain].statistics),
                "eligible_semantic_buckets": selections[domain].semantic_bucket_count,
                "selected_feature_coverage": _selected_feature_coverage(
                    selections[domain].selected
                ),
            }
            for domain in DOMAINS
        },
        "render_gate": {
            "validator": "SVGValidator strict static policy",
            "renderer": "CairoSVGRenderer.render_bytes",
            "width": config.render_size,
            "height": config.render_size,
            "scope": "ranked final candidates and deterministic backfills",
        },
        "final_audit": {
            "rows": len(selected_all),
            "unique_raw_ids": len(selected_ids),
            "unique_item_ids": len(selected_item_ids),
            "unique_canonical_svg_hashes": len(selected_hashes),
            "unique_caption_detail_hash_pairs": len(selected_text_pairs),
            "generator_raw_id_overlap": len(selected_ids & exclusions.generator_raw_ids),
            "generator_svg_hash_overlap": len(selected_hashes & exclusions.generator_svg_hashes),
            "final_reference_item_id_overlap": len(
                selected_item_ids & exclusions.final_reference_item_ids
            ),
            "final_reference_svg_hash_overlap": len(
                selected_hashes & exclusions.final_reference_svg_hashes
            ),
            "benchmark_raw_id_overlap": len(selected_ids & exclusions.benchmark_raw_ids),
            "benchmark_text_hash_overlap": len(
                {
                    value
                    for candidate in selected_all
                    for value in (candidate.caption_sha256, candidate.detail_sha256)
                }
                & exclusions.benchmark_text_hashes
            ),
            "benchmark_svg_hash_overlap": len(selected_hashes & exclusions.benchmark_svg_hashes),
        },
    }
    return _publish_outputs(
        output_dir=config.output_dir,
        selections=selections,
        manifest=manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic 5K Icon + 5K Illustration MMSVG critic rollout "
            "holdout using existing dual Chroma embeddings and structural edge quotas."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--generator-id-dir", type=Path, default=Path("data/sft_sample_ids"))
    parser.add_argument(
        "--generator-sft-dir", type=Path, default=Path("data/processed/mmsvg_sft_20k")
    )
    parser.add_argument(
        "--final-references-path",
        type=Path,
        default=Path("data/processed/mmsvg_sft_20k_rag/final_references.parquet"),
        help="Generator RAG final Top-3 artifact; all item IDs and SVG hashes are denied.",
    )
    parser.add_argument(
        "--benchmark-exclusion",
        action="append",
        type=Path,
        dest="benchmark_exclusions",
        help="Repeat for each benchmark JSONL; defaults to the two local benchmark files.",
    )
    parser.add_argument("--description-chroma-dir", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--detail-chroma-dir", type=Path, default=Path("data/chroma_db_detail"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/critic_rollout_sample_ids"))
    parser.add_argument("--sample-size-per-domain", type=int, default=5_000)
    parser.add_argument("--edge-quota-per-domain", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-batch-size", type=int, default=8_192)
    parser.add_argument("--chroma-batch-size", type=int, default=8_192)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--lsh-bits", type=int, default=16)
    parser.add_argument("--render-size", type=int, default=64)
    return parser


def config_from_args(args: argparse.Namespace) -> SelectionConfig:
    benchmark_exclusions = args.benchmark_exclusions or [
        Path("data/processed/benchmark_exclusions/mmsvgbench.jsonl"),
        Path("data/processed/benchmark_exclusions/vectorgym_vg_text.jsonl"),
    ]
    return SelectionConfig(
        data_root=args.data_root,
        generator_id_dir=args.generator_id_dir,
        generator_sft_dir=args.generator_sft_dir,
        final_references_path=args.final_references_path,
        benchmark_exclusions=tuple(benchmark_exclusions),
        description_chroma_dir=args.description_chroma_dir,
        detail_chroma_dir=args.detail_chroma_dir,
        output_dir=args.output_dir,
        sample_size_per_domain=args.sample_size_per_domain,
        edge_quota_per_domain=args.edge_quota_per_domain,
        seed=args.seed,
        scan_batch_size=args.scan_batch_size,
        chroma_batch_size=args.chroma_batch_size,
        projection_dimension=args.projection_dimension,
        lsh_bits=args.lsh_bits,
        render_size=args.render_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    manifest = export_critic_rollout_ids(config_from_args(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALGORITHM_VERSION",
    "Candidate",
    "DomainSelection",
    "Exclusions",
    "SelectionConfig",
    "annotate_semantic_buckets",
    "build_parser",
    "canonical_svg_sha256",
    "caption_sha256",
    "export_critic_rollout_ids",
    "load_exclusions",
    "main",
    "make_strict_acceptor",
    "scan_domain",
    "select_domain_candidates",
]
