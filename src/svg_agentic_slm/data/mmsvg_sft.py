"""Prepare a deterministic, balanced MMSVG dataset for Generator SFT."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import yaml

from svg_agentic_slm.svg.validator import SVGValidator

_PATH_COMMAND_RE = re.compile(r"(?<![A-Za-z])[MmLlHhVvCcSsQqTtAaZz](?![A-Za-z])")
_WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)
_REJECTION_REASONS = (
    "missing_required_field",
    "missing_detail",
    "description_length",
    "benchmark_exclusion",
    "svg_char_length",
    "svg_validation",
    "svg_canonicalization",
    "duplicate_svg",
    "render",
    "svg_tokenization",
    "svg_token_length",
)


@dataclass(frozen=True)
class SourceConfig:
    paths: tuple[str, ...]
    dataset_id: str
    dataset_revision: str | None = None


@dataclass(frozen=True)
class SplitConfig:
    train_per_domain: int = 9000
    validation_per_domain: int = 500
    test_per_domain: int = 500

    @property
    def per_domain(self) -> int:
        return self.train_per_domain + self.validation_per_domain + self.test_per_domain


@dataclass(frozen=True)
class FilterConfig:
    min_description_words: int = 2
    max_description_words: int = 80
    require_detail: bool = True
    min_svg_char_length: int = 32
    max_svg_char_length: int = 8192
    max_svg_token_length: int | None = 8192
    require_renderable: bool = False


@dataclass(frozen=True)
class FieldConfig:
    id: tuple[str, ...] = ("id", "item_id", "row_id", "index")
    description: tuple[str, ...] = ("description", "caption", "text")
    detail: tuple[str, ...] = ("detail", "detailed_description")
    svg: tuple[str, ...] = ("svg", "svg_code", "output_svg")


@dataclass(frozen=True)
class TokenizerConfig:
    model_id: str | None = None
    revision: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False


@dataclass(frozen=True)
class MMSVGPreparationConfig:
    icon: SourceConfig
    illustration: SourceConfig
    output_dir: Path
    seed: int = 42
    batch_size: int = 1024
    split: SplitConfig = field(default_factory=SplitConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    exclusion_manifests: tuple[Path, ...] = ()
    rag_results_path: Path | None = None


@dataclass
class _Exclusions:
    ids: set[str] = field(default_factory=set)
    svg_hashes: set[str] = field(default_factory=set)
    caption_hashes: set[str] = field(default_factory=set)


def load_preparation_config(path: str | Path) -> MMSVGPreparationConfig:
    """Load and validate the YAML contract used by the preparation script."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = payload.get("prepare_mmsvg_sft", payload)
    if not isinstance(root, dict):
        raise ValueError("prepare_mmsvg_sft configuration must be a mapping.")
    sources = _mapping(root.get("sources"), "sources")
    split = _mapping(root.get("split", {}), "split")
    filters = _mapping(root.get("filters", {}), "filters")
    fields = _mapping(root.get("fields", {}), "fields")
    tokenizer = _mapping(root.get("tokenizer", {}), "tokenizer")
    return MMSVGPreparationConfig(
        icon=_source_config(sources.get("icon"), "icon"),
        illustration=_source_config(sources.get("illustration"), "illustration"),
        output_dir=Path(_nonempty(root.get("output_dir"), "output_dir")),
        seed=int(root.get("seed", 42)),
        batch_size=_positive_int(root.get("batch_size", 1024), "batch_size"),
        split=SplitConfig(
            train_per_domain=_positive_int(split.get("train_per_domain", 9000), "train_per_domain"),
            validation_per_domain=_positive_int(
                split.get("validation_per_domain", 500), "validation_per_domain"
            ),
            test_per_domain=_positive_int(split.get("test_per_domain", 500), "test_per_domain"),
        ),
        filters=FilterConfig(
            min_description_words=int(filters.get("min_description_words", 2)),
            max_description_words=int(filters.get("max_description_words", 80)),
            require_detail=bool(filters.get("require_detail", True)),
            min_svg_char_length=int(filters.get("min_svg_char_length", 32)),
            max_svg_char_length=int(filters.get("max_svg_char_length", 8192)),
            max_svg_token_length=(
                int(filters["max_svg_token_length"])
                if filters.get("max_svg_token_length") is not None
                else None
            ),
            require_renderable=bool(filters.get("require_renderable", False)),
        ),
        fields=FieldConfig(
            id=_string_tuple(fields.get("id", FieldConfig.id), "fields.id"),
            description=_string_tuple(
                fields.get("description", FieldConfig.description), "fields.description"
            ),
            detail=_string_tuple(fields.get("detail", FieldConfig.detail), "fields.detail"),
            svg=_string_tuple(fields.get("svg", FieldConfig.svg), "fields.svg"),
        ),
        tokenizer=TokenizerConfig(
            model_id=_optional_string(tokenizer.get("model_id"), "tokenizer.model_id"),
            revision=_optional_string(tokenizer.get("revision"), "tokenizer.revision"),
            local_files_only=bool(tokenizer.get("local_files_only", True)),
            trust_remote_code=bool(tokenizer.get("trust_remote_code", False)),
        ),
        exclusion_manifests=tuple(Path(item) for item in root.get("exclusion_manifests", [])),
        rag_results_path=(Path(root["rag_results_path"]) if root.get("rag_results_path") else None),
    )


def prepare_mmsvg_sft_dataset(config: MMSVGPreparationConfig) -> dict[str, Any]:
    """Stream, clean, deduplicate, balance, split, and publish MMSVG records."""
    _validate_config(config)
    validator = SVGValidator()
    token_counter = _build_token_counter(config.tokenizer)
    exclusions = _load_exclusions(config.exclusion_manifests)
    rag_results = _load_rag_results(config.rag_results_path)
    seen_svg_hashes = set(exclusions.svg_hashes)
    stats: dict[str, dict[str, int]] = {}
    selected: dict[str, list[dict[str, Any]]] = {}

    for domain, source in (("icon", config.icon), ("illustration", config.illustration)):
        rng = random.Random(f"{config.seed}:{domain}")
        reservoir: list[dict[str, Any]] = []
        domain_stats: dict[str, int] = {
            "rows_seen": 0,
            "eligible": 0,
            "selected": 0,
            "eligible_not_selected": 0,
            **{f"rejected_{reason}": 0 for reason in _REJECTION_REASONS},
        }
        for row, pointer in _iter_source_rows(source, config.batch_size):
            domain_stats["rows_seen"] += 1
            prepared, rejection = _prepare_row(
                row=row,
                pointer=pointer,
                domain=domain,
                source=source,
                fields=config.fields,
                filters=config.filters,
                exclusions=exclusions,
                seen_svg_hashes=seen_svg_hashes,
                validator=validator,
                token_counter=token_counter,
                rag_results=rag_results,
            )
            if prepared is None:
                key = f"rejected_{rejection or 'unknown'}"
                domain_stats[key] = domain_stats.get(key, 0) + 1
                continue
            seen_svg_hashes.add(prepared["metadata"]["canonical_svg_sha256"])
            domain_stats["eligible"] += 1
            eligible_index = domain_stats["eligible"]
            if len(reservoir) < config.split.per_domain:
                reservoir.append(prepared)
            else:
                replacement = rng.randrange(eligible_index)
                if replacement < config.split.per_domain:
                    reservoir[replacement] = prepared
        if len(reservoir) != config.split.per_domain:
            raise RuntimeError(
                f"MMSVG {domain} produced {len(reservoir)} eligible selected rows; "
                f"{config.split.per_domain} are required."
            )
        rng.shuffle(reservoir)
        domain_stats["selected"] = len(reservoir)
        domain_stats["eligible_not_selected"] = domain_stats["eligible"] - len(reservoir)
        stats[domain] = domain_stats
        selected[domain] = reservoir

    split_records = _assign_splits(selected, config)
    manifest = _publish_dataset(config, split_records, stats)
    return manifest


def _prepare_row(
    *,
    row: dict[str, Any],
    pointer: dict[str, Any],
    domain: str,
    source: SourceConfig,
    fields: FieldConfig,
    filters: FilterConfig,
    exclusions: _Exclusions,
    seen_svg_hashes: set[str],
    validator: SVGValidator,
    token_counter: Callable[[str], int] | None,
    rag_results: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, str | None]:
    record_id = _first_text(row, fields.id)
    description = _first_text(row, fields.description)
    detail = _first_text(row, fields.detail)
    svg = _first_text(row, fields.svg)
    if not record_id or not description or not svg:
        return None, "missing_required_field"
    if filters.require_detail and not detail:
        return None, "missing_detail"
    description_words = len(_WORD_RE.findall(description))
    if not filters.min_description_words <= description_words <= filters.max_description_words:
        return None, "description_length"
    if (
        record_id in exclusions.ids
        or _caption_hash(description) in exclusions.caption_hashes
        or (detail and _caption_hash(detail) in exclusions.caption_hashes)
    ):
        return None, "benchmark_exclusion"
    if not filters.min_svg_char_length <= len(svg) <= filters.max_svg_char_length:
        return None, "svg_char_length"
    try:
        validation = validator.validate(svg)
    except Exception:
        return None, "svg_validation"
    if not validation.is_valid:
        return None, "svg_validation"
    try:
        canonical_svg, element_count = _canonicalize_svg(svg)
    except Exception:
        return None, "svg_canonicalization"
    canonical_hash = hashlib.sha256(canonical_svg.encode("utf-8")).hexdigest()
    if canonical_hash in seen_svg_hashes:
        return None, "duplicate_svg"
    if filters.require_renderable and not _is_renderable(canonical_svg):
        return None, "render"
    try:
        svg_token_length = token_counter(canonical_svg) if token_counter is not None else None
    except Exception:
        return None, "svg_tokenization"
    if (
        filters.max_svg_token_length is not None
        and svg_token_length is not None
        and svg_token_length > filters.max_svg_token_length
    ):
        return None, "svg_token_length"
    stable_record_key = hashlib.sha256(
        "\x1f".join(
            (
                source.dataset_id,
                source.dataset_revision or "",
                domain,
                record_id,
                canonical_hash,
            )
        ).encode("utf-8")
    ).hexdigest()
    mixed_instruction_source = (
        "detail" if int(stable_record_key[:16], 16) % 100 < 60 else "description"
    )
    retrieved_item_ids = rag_results.get(
        f"{domain}:{record_id}", rag_results.get(record_id, [])
    )
    metadata = {
        "dataset_id": source.dataset_id,
        "dataset_revision": source.dataset_revision,
        "dataset_type": domain,
        "record_id": record_id,
        "source_pointer": pointer,
        "description_words": description_words,
        "detail_words": len(_WORD_RE.findall(detail)) if detail else 0,
        "svg_char_length": len(canonical_svg),
        "svg_token_length": svg_token_length,
        "svg_element_count": element_count,
        "svg_path_count": canonical_svg.count("<path"),
        "svg_path_command_count": len(_PATH_COMMAND_RE.findall(canonical_svg)),
        "canonical_svg_sha256": canonical_hash,
        "caption_sha256": _caption_hash(description),
        "detail_sha256": _caption_hash(detail),
        "stable_record_key_sha256": stable_record_key,
        "instruction_policy": {
            "r0_description_only": "description",
            "r1_detail_60_description_40": mixed_instruction_source,
            "r2_detail_only": "detail",
        },
        "retrieved_item_ids": retrieved_item_ids,
    }
    return (
        {
            "task": "text_to_svg",
            "instruction": description,
            "instruction_source": "description",
            "description": description,
            "detail": detail,
            "output_svg": canonical_svg,
            "metadata": metadata,
        },
        None,
    )


def _assign_splits(
    selected: dict[str, list[dict[str, Any]]], config: MMSVGPreparationConfig
) -> dict[str, list[dict[str, Any]]]:
    output = {"train": [], "validation": [], "test": []}
    for domain in ("icon", "illustration"):
        rows = selected[domain]
        boundaries = (
            ("train", 0, config.split.train_per_domain),
            (
                "validation",
                config.split.train_per_domain,
                config.split.train_per_domain + config.split.validation_per_domain,
            ),
            (
                "test",
                config.split.train_per_domain + config.split.validation_per_domain,
                config.split.per_domain,
            ),
        )
        for split_name, start, end in boundaries:
            for row in rows[start:end]:
                output[split_name].append(
                    {
                        **row,
                        "metadata": {**row["metadata"], "split": split_name},
                    }
                )
    for index, split_name in enumerate(("train", "validation", "test")):
        random.Random(config.seed + index).shuffle(output[split_name])
    return output


def _publish_dataset(
    config: MMSVGPreparationConfig,
    splits: dict[str, list[dict[str, Any]]],
    stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / f".prepare-{os.getpid()}-{config.seed}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    try:
        split_manifest: dict[str, Any] = {}
        for split_name, records in splits.items():
            path = temp_dir / f"{split_name}.jsonl"
            _write_jsonl(path, records)
            split_manifest[split_name] = {
                "file": path.name,
                "rows": len(records),
                "sha256": _file_sha256(path),
                "icon_rows": sum(
                    row["metadata"]["dataset_type"] == "icon" for row in records
                ),
                "illustration_rows": sum(
                    row["metadata"]["dataset_type"] == "illustration" for row in records
                ),
            }
        manifest = {
            "schema_version": 1,
            "dataset": "mmsvg-generator-sft-balanced-20k",
            "seed": config.seed,
            "config_sha256": hashlib.sha256(
                json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "sources": {
                "icon": _source_provenance(config.icon),
                "illustration": _source_provenance(config.illustration),
            },
            "benchmark_exclusions": [_file_provenance(path) for path in config.exclusion_manifests],
            "rag_results": (
                _file_provenance(config.rag_results_path)
                if config.rag_results_path is not None
                else None
            ),
            "filters": asdict(config.filters),
            "splits": split_manifest,
            "scan_statistics": stats,
            "instruction_fields": ["description", "detail"],
            "default_instruction_source": "description",
            "instruction_policies": {
                "r0": "description-only",
                "r1": "stable-record-hash detail 60% / description 40%",
                "r2": "detail-only",
            },
            "rag_context_in_training": False,
            "rag_top3_storage": "metadata.retrieved_item_ids only",
            "deduplication": {
                "exact": "SHA-256 of canonical SVG XML across both domains",
                "perceptual_or_near_duplicate": "not applied",
                "group_aware_split": "not applied; no verified MMSVG grouping field configured",
            },
            "domain_scan_order": ["icon", "illustration"],
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        for path in temp_dir.iterdir():
            os.replace(path, output_dir / path.name)
        temp_dir.rmdir()
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _iter_source_rows(
    source: SourceConfig, batch_size: int
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    paths = _expand_paths(source.paths)
    if not paths:
        raise FileNotFoundError(f"No input files matched MMSVG source {source.dataset_id!r}.")
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("pyarrow is required to read MMSVG Parquet files.") from exc
            parquet = pq.ParquetFile(path)
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=batch_size):
                rows = batch.to_pylist()
                for index, row in enumerate(rows):
                    yield dict(row), {"file": str(path), "row": row_offset + index}
                row_offset += len(rows)
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(f"Expected JSON object at {path}:{row_index + 1}.")
                        yield row, {"file": str(path), "row": row_index}
        else:
            raise ValueError(f"Unsupported MMSVG input format: {path}")


def _load_exclusions(paths: tuple[Path, ...]) -> _Exclusions:
    exclusions = _Exclusions()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Benchmark exclusion manifest not found: {path}")
        payloads: Iterable[Any]
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            payloads = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payloads = raw if isinstance(raw, list) else [raw]
        for payload in payloads:
            _merge_exclusion_payload(exclusions, payload)
    return exclusions


def _merge_exclusion_payload(exclusions: _Exclusions, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    for key in ("id", "item_id", "record_id"):
        if payload.get(key) is not None:
            exclusions.ids.add(str(payload[key]))
    for key in ("canonical_svg_sha256", "svg_sha256", "svg_hash"):
        if payload.get(key):
            exclusions.svg_hashes.add(str(payload[key]).lower())
    for key in (
        "instruction",
        "caption",
        "description",
        "detail",
        "detailed_description",
        "text",
    ):
        if payload.get(key):
            exclusions.caption_hashes.add(_caption_hash(str(payload[key])))
    for key, target in (
        ("ids", exclusions.ids),
        ("svg_hashes", exclusions.svg_hashes),
        ("caption_hashes", exclusions.caption_hashes),
    ):
        values = payload.get(key, [])
        if isinstance(values, list):
            target.update(str(value).lower() if "hash" in key else str(value) for value in values)


def _load_rag_results(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"RAG result file not found: {path}")
    output: dict[str, list[str]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = row.get("record_id", row.get("id"))
        values = row.get("retrieved_item_ids", row.get("top3", row.get("neighbors", [])))
        if record_id is None or not isinstance(values, list):
            raise ValueError(f"Invalid RAG result at {path}:{line_index}.")
        item_ids = [str(value) for value in values]
        if len(item_ids) > 3 or len(item_ids) != len(set(item_ids)):
            raise ValueError(
                f"RAG result at {path}:{line_index} must contain at most three unique item IDs."
            )
        dataset_type = row.get("dataset_type", row.get("domain"))
        key = f"{dataset_type}:{record_id}" if dataset_type else str(record_id)
        if key in output and output[key] != item_ids:
            raise ValueError(f"Conflicting RAG result key {key!r} at {path}:{line_index}.")
        output[key] = item_ids
    return output


def _build_token_counter(config: TokenizerConfig) -> Callable[[str], int] | None:
    if config.model_id is None:
        return None
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenizer-aware filtering.") from exc
    kwargs: dict[str, Any] = {
        "revision": config.revision,
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        processor = AutoProcessor.from_pretrained(config.model_id, **kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **kwargs)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _canonicalize_svg(svg: str) -> tuple[str, int]:
    from lxml import etree  # type: ignore[import-untyped]

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    root = etree.fromstring(svg.encode("utf-8"), parser=parser)
    canonical = etree.tostring(root, method="c14n", with_comments=False).decode("utf-8")
    return canonical, sum(1 for _ in root.iter())


def _is_renderable(svg: str) -> bool:
    try:
        import cairosvg

        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=64, output_height=64)
        return bool(png and png.startswith(b"\x89PNG\r\n\x1a\n"))
    except Exception:
        return False


def _source_config(value: Any, name: str) -> SourceConfig:
    mapping = _mapping(value, f"sources.{name}")
    paths = _string_tuple(mapping.get("paths", ()), f"sources.{name}.paths")
    return SourceConfig(
        paths=paths,
        dataset_id=_nonempty(mapping.get("dataset_id"), f"sources.{name}.dataset_id"),
        dataset_revision=_optional_string(
            mapping.get("dataset_revision"), f"sources.{name}.dataset_revision"
        ),
    )


def _expand_paths(patterns: tuple[str, ...]) -> list[Path]:
    resolved: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(os.path.expanduser(pattern), recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        resolved.update(Path(match).resolve() for match in matches if Path(match).is_file())
    return sorted(resolved)


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            if text.strip():
                return text.strip()
    return ""


def _caption_hash(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(resolved),
    }


def _source_provenance(source: SourceConfig) -> dict[str, Any]:
    return {
        **asdict(source),
        "files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in _expand_paths(source.paths)
        ],
    }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain at least one string.")
    output = tuple(str(item).strip() for item in value)
    if any(not item for item in output):
        raise ValueError(f"{name} cannot contain empty strings.")
    return output


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _validate_config(config: MMSVGPreparationConfig) -> None:
    if (
        config.split.train_per_domain,
        config.split.validation_per_domain,
        config.split.test_per_domain,
    ) != (9000, 500, 500):
        raise ValueError(
            "The approved MMSVG split requires train/validation/test = 9000/500/500 "
            "for each domain."
        )
    if not config.filters.require_detail:
        raise ValueError("Detail is required so the immutable pool supports R1 and R2 ablations.")
    if not config.filters.require_renderable:
        raise ValueError("Renderable SVG filtering must remain enabled for fail-closed preparation.")
    if config.filters.max_svg_token_length is None or config.tokenizer.model_id is None:
        raise ValueError("Tokenizer-aware SVG length filtering must remain enabled.")
    if not config.exclusion_manifests:
        raise ValueError("At least one benchmark exclusion manifest is required.")
    if config.filters.min_description_words > config.filters.max_description_words:
        raise ValueError("Description word bounds are inverted.")
    if config.filters.min_svg_char_length > config.filters.max_svg_char_length:
        raise ValueError("SVG character bounds are inverted.")
