"""Offline Gemma-tokenizer audit for pinned official OpenVGLab cached targets.

This module loads only a tokenizer. It never loads a model and forces Hugging Face
offline mode before resolving the pinned local tokenizer snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from svg_agentic_slm.svg.discrete_runtime import register_named_codec_tokens
from svg_agentic_slm.svg.official_discrete_cache import (
    CACHED_TARGET_FIELD,
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    OPENVGLAB_NAMED_VOCABULARY_SIZE,
    CachedOpenVGLabGemmaDialect,
    OpenVGLabCacheConfig,
    load_all_cached_records,
)
from svg_agentic_slm.train.sft_trainer import _ResponseOnlyDataset
from svg_agentic_slm.utils.atomic import atomic_write_text
from svg_agentic_slm.utils.config import load_yaml_config

_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_LENGTH_METRICS = ("codec_tokens", "labeled_tokens", "full_chat_tokens")


@dataclass(frozen=True)
class TokenizerAuditConfig:
    path: Path
    revision: str
    local_files_only: bool = True
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("tokenizer.path must be an absolute local snapshot path")
        if not _PINNED_REVISION.fullmatch(self.revision):
            raise ValueError("tokenizer.revision must be a pinned 40-hex commit revision")
        if not self.local_files_only:
            raise ValueError("local_discrete_audit requires tokenizer.local_files_only=true")
        if self.trust_remote_code:
            raise ValueError("local_discrete_audit requires tokenizer.trust_remote_code=false")


@dataclass(frozen=True)
class AuditDataConfig:
    train_path: Path
    validation_path: Path
    test_path: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("train", self.train_path),
            ("validation", self.validation_path),
            ("test", self.test_path),
        )


@dataclass(frozen=True)
class AuditOutputConfig:
    report_path: Path
    longest_record_path: Path | None = None


@dataclass(frozen=True)
class AuditThresholdConfig:
    codec_tokens: tuple[int, ...] = (1024, 2048, 4096, 8192)
    labeled_tokens: tuple[int, ...] = (1024, 2048, 4096, 8192)
    full_chat_tokens: tuple[int, ...] = (2048, 2304, 4096, 8192)
    maximum_codec_tokens: int = 8192
    maximum_labeled_tokens: int = 8192
    maximum_full_chat_tokens: int = 8192

    def __post_init__(self) -> None:
        for name in _LENGTH_METRICS:
            values = getattr(self, name)
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            ):
                raise ValueError(f"thresholds.{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"thresholds.{name} must be sorted and unique")
        for name in (
            "maximum_codec_tokens",
            "maximum_labeled_tokens",
            "maximum_full_chat_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"thresholds.{name} must be a positive integer")

    def reporting_thresholds(self, metric: str) -> tuple[int, ...]:
        return getattr(self, metric)

    def maximum(self, metric: str) -> int:
        return getattr(self, f"maximum_{metric}")


@dataclass(frozen=True)
class LocalDiscreteAuditConfig:
    tokenizer: TokenizerAuditConfig
    data: AuditDataConfig
    output: AuditOutputConfig
    thresholds: AuditThresholdConfig = field(default_factory=AuditThresholdConfig)
    cache: OpenVGLabCacheConfig | None = None
    grid_size: int = 200
    dataset_max_seq_length: int = 8192
    expected_total_records: int | None = 20000
    maximum_failure_details: int = 100

    def __post_init__(self) -> None:
        for name in ("grid_size", "dataset_max_seq_length", "maximum_failure_details"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.expected_total_records is not None and (
            isinstance(self.expected_total_records, bool)
            or not isinstance(self.expected_total_records, int)
            or self.expected_total_records <= 0
        ):
            raise ValueError("expected_total_records must be a positive integer or null")
        if self.dataset_max_seq_length != 8192:
            raise ValueError("official discrete audit requires dataset_max_seq_length=8192")
        if self.cache is not None:
            if self.cache.allow_live_reencode:
                raise ValueError("official discrete audit requires allow_live_reencode=false")
            configured_paths = dict(self.data.items())
            for split, path in configured_paths.items():
                expected = self.cache.prepared_root / f"{split}.jsonl"
                if path.resolve() != expected:
                    raise ValueError(f"data.{split}_path must be pinned to {expected}")


@dataclass
class _LengthAccumulator:
    codec_tokens: list[int] = field(default_factory=list)
    labeled_tokens: list[int] = field(default_factory=list)
    full_chat_tokens: list[int] = field(default_factory=list)
    maximum_observed_id: int | None = None

    def add(self, measurements: Mapping[str, Any]) -> None:
        for name in _LENGTH_METRICS:
            getattr(self, name).append(int(measurements[name]))
        observed_id = measurements.get("maximum_observed_id")
        if observed_id is not None:
            self.maximum_observed_id = (
                observed_id
                if self.maximum_observed_id is None
                else max(self.maximum_observed_id, observed_id)
            )

    def extend(self, other: _LengthAccumulator) -> None:
        for name in _LENGTH_METRICS:
            getattr(self, name).extend(getattr(other, name))
        if other.maximum_observed_id is not None:
            self.maximum_observed_id = (
                other.maximum_observed_id
                if self.maximum_observed_id is None
                else max(self.maximum_observed_id, other.maximum_observed_id)
            )


@dataclass
class _FailureLog:
    maximum_details: int
    count: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        split: str,
        line_number: int | None,
        record_id: str | None,
        code: str,
        message: str,
    ) -> None:
        self.count += 1
        if len(self.details) < self.maximum_details:
            self.details.append(
                {
                    "split": split,
                    "line_number": line_number,
                    "record_id": record_id,
                    "code": code,
                    "message": message,
                }
            )


class _RecordValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _resolve_path(value: Any, base_dir: Path, *, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _integer_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def load_local_discrete_audit_config(
    config_path: str | Path,
) -> LocalDiscreteAuditConfig:
    """Load and strictly validate a local discrete audit YAML file."""

    path = Path(config_path).expanduser().resolve()
    payload = load_yaml_config(path)
    root = _mapping(payload.get("local_discrete_audit", payload), name="local_discrete_audit")
    tokenizer = _mapping(root.get("tokenizer"), name="tokenizer")
    data = _mapping(root.get("data"), name="data")
    output = _mapping(root.get("output"), name="output")
    threshold_payload = _mapping(root.get("thresholds", {}), name="thresholds")
    cache_payload = _mapping(root.get("official_cache"), name="official_cache")
    base_dir = path.parent

    tokenizer_config = TokenizerAuditConfig(
        path=_resolve_path(tokenizer.get("path"), base_dir, name="tokenizer.path"),
        revision=str(tokenizer.get("revision", "")),
        local_files_only=tokenizer.get("local_files_only", True),
        trust_remote_code=tokenizer.get("trust_remote_code", False),
    )
    data_config = AuditDataConfig(
        train_path=_resolve_path(data.get("train_path"), base_dir, name="data.train_path"),
        validation_path=_resolve_path(
            data.get("validation_path"), base_dir, name="data.validation_path"
        ),
        test_path=_resolve_path(data.get("test_path"), base_dir, name="data.test_path"),
    )
    longest_value = output.get("longest_record_path")
    output_config = AuditOutputConfig(
        report_path=_resolve_path(
            output.get("report_path"), base_dir, name="output.report_path"
        ),
        longest_record_path=(
            _resolve_path(longest_value, base_dir, name="output.longest_record_path")
            if longest_value is not None
            else None
        ),
    )
    defaults = AuditThresholdConfig()
    thresholds = AuditThresholdConfig(
        codec_tokens=_integer_tuple(
            threshold_payload.get("codec_tokens", defaults.codec_tokens),
            name="thresholds.codec_tokens",
        ),
        labeled_tokens=_integer_tuple(
            threshold_payload.get("labeled_tokens", defaults.labeled_tokens),
            name="thresholds.labeled_tokens",
        ),
        full_chat_tokens=_integer_tuple(
            threshold_payload.get("full_chat_tokens", defaults.full_chat_tokens),
            name="thresholds.full_chat_tokens",
        ),
        maximum_codec_tokens=threshold_payload.get(
            "maximum_codec_tokens", defaults.maximum_codec_tokens
        ),
        maximum_labeled_tokens=threshold_payload.get(
            "maximum_labeled_tokens", defaults.maximum_labeled_tokens
        ),
        maximum_full_chat_tokens=threshold_payload.get(
            "maximum_full_chat_tokens", defaults.maximum_full_chat_tokens
        ),
    )
    cache_config = OpenVGLabCacheConfig(
        cache_path=_resolve_path(
            cache_payload.get("cache_path"), base_dir, name="official_cache.cache_path"
        ),
        audit_manifest_path=_resolve_path(
            cache_payload.get("audit_manifest_path"),
            base_dir,
            name="official_cache.audit_manifest_path",
        ),
        prepared_root=_resolve_path(
            cache_payload.get("prepared_root"),
            base_dir,
            name="official_cache.prepared_root",
        ),
        expected_cache_sha256=str(cache_payload.get("expected_cache_sha256", "")),
        expected_audit_sha256=str(cache_payload.get("expected_audit_sha256", "")),
        expected_input_sha256=dict(
            _mapping(
                cache_payload.get("expected_input_sha256"),
                name="official_cache.expected_input_sha256",
            )
        ),
        expected_split_counts=dict(
            _mapping(
                cache_payload.get("expected_split_counts"),
                name="official_cache.expected_split_counts",
            )
        ),
        allow_live_reencode=cache_payload.get("allow_live_reencode", False),
    )
    return LocalDiscreteAuditConfig(
        tokenizer=tokenizer_config,
        data=data_config,
        output=output_config,
        thresholds=thresholds,
        cache=cache_config,
        grid_size=root.get("grid_size", 200),
        dataset_max_seq_length=root.get("dataset_max_seq_length", 8192),
        expected_total_records=root.get("expected_total_records", 20000),
        maximum_failure_details=root.get("maximum_failure_details", 100),
    )


def _load_offline_tokenizer(config: TokenizerAuditConfig) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not config.path.is_dir():
        raise FileNotFoundError(f"Pinned tokenizer snapshot does not exist: {config.path}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(config.path),
        revision=config.revision,
        local_files_only=True,
        trust_remote_code=config.trust_remote_code,
    )


def _create_official_codec(grid_size: int) -> Any:
    del grid_size
    return CachedOpenVGLabGemmaDialect()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=isinstance(value, Mapping),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_vocabulary_manifest(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping) or any(
        not isinstance(token, str)
        or isinstance(token_id, bool)
        or not isinstance(token_id, int)
        for token, token_id in vocabulary.items()
    ):
        raise TypeError("Tokenizer get_vocab() must return a string-to-integer mapping")
    ordered = sorted(vocabulary.items(), key=lambda item: (item[1], item[0]))
    return {
        "size": len(ordered),
        "maximum_id": max((token_id for _token, token_id in ordered), default=None),
        "sha256": _json_sha256(ordered),
    }


def _integer_list(value: Any, *, name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _RecordValidationError("invalid_token_sequence", f"{name} is not a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise _RecordValidationError(
            "invalid_token_sequence", f"{name} contains non-integer values"
        )
    return list(value)


def _record_id(record: Mapping[str, Any]) -> str | None:
    for container in (record, record.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        for key in ("id", "sample_id", "source_id", "record_id", "example_id", "svg_id"):
            value = container.get(key)
            if value is not None:
                return str(value)
    return None


def _cached_discrete_target(
    codec: Any,
    record: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    tokens = codec.target_tokens_for_record(record)
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes, bytearray)):
        raise _RecordValidationError(
            "invalid_codec_target", "Cached codec target is not a token sequence"
        )
    tokens = tuple(tokens)
    if not tokens or any(not isinstance(token, str) or not token for token in tokens):
        raise _RecordValidationError(
            "invalid_codec_target", "Cached codec target contains invalid token strings"
        )
    target = codec.target_for_record(record)
    if not isinstance(target, str) or not target or target != "".join(tokens):
        raise _RecordValidationError(
            "cached_target_serialization_mismatch",
            "Cached target text does not equal its ordered framed token strings",
        )
    return target, tokens


def _audit_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    codec: Any,
    codec_id_set: frozenset[int],
    dataset_max_seq_length: int,
    dataset_factory: Callable[..., Any],
) -> dict[str, Any]:
    target, framed_tokens = _cached_discrete_target(codec, record)
    target_ids = _integer_list(
        tokenizer.encode(target, add_special_tokens=False), name="codec target IDs"
    )
    if not target_ids:
        raise _RecordValidationError("empty_codec_target", "Codec target has no token IDs")
    if len(target_ids) != len(framed_tokens):
        raise _RecordValidationError(
            "framed_target_count_mismatch",
            f"Tokenizer produced {len(target_ids)} IDs for {len(framed_tokens)} framed tokens",
        )
    cache_context = record.get(CACHED_TARGET_FIELD)
    if isinstance(cache_context, Mapping):
        framed_count = cache_context.get("framed_count")
        if framed_count != len(framed_tokens):
            raise _RecordValidationError(
                "cached_framed_count_mismatch",
                f"Cache declares {framed_count!r} framed IDs but target has {len(framed_tokens)}",
            )
    unexpected_target_ids = sorted(set(target_ids).difference(codec_id_set))
    if unexpected_target_ids:
        raise _RecordValidationError(
            "target_contains_non_codec_ids",
            f"Codec target contains non-codec IDs: {unexpected_target_ids[:10]}",
        )

    dataset = dataset_factory(
        [dict(record)],
        tokenizer=tokenizer,
        instruction_mode="description_only",
        target_representation="omnisvg_discrete",
        max_seq_length=dataset_max_seq_length,
        seed=42,
        codec=codec,
    )
    sample = dataset[0]
    if not isinstance(sample, Mapping):
        raise _RecordValidationError("invalid_dataset_sample", "Dataset sample is not a mapping")
    input_ids = _integer_list(sample.get("input_ids"), name="dataset input_ids")
    labels = _integer_list(sample.get("labels"), name="dataset labels")
    if len(input_ids) != len(labels):
        raise _RecordValidationError(
            "input_label_length_mismatch", "Dataset input_ids and labels have different lengths"
        )
    labeled_ids = [label for label in labels if label != -100]
    if len(labeled_ids) != len(target_ids):
        raise _RecordValidationError(
            "labeled_target_count_mismatch",
            f"Labeled count {len(labeled_ids)} differs from target count {len(target_ids)}",
        )
    if labeled_ids != target_ids:
        raise _RecordValidationError(
            "labeled_target_id_mismatch", "Labeled IDs do not exactly match codec target IDs"
        )
    unexpected_labeled_ids = sorted(set(labeled_ids).difference(codec_id_set))
    if unexpected_labeled_ids:
        raise _RecordValidationError(
            "labels_contain_non_codec_ids",
            f"Labels contain non-codec IDs: {unexpected_labeled_ids[:10]}",
        )
    return {
        "codec_tokens": len(target_ids),
        "labeled_tokens": len(labeled_ids),
        "full_chat_tokens": len(input_ids),
        "framed_target_tokens": len(framed_tokens),
        "maximum_observed_id": max((*target_ids, *labeled_ids)),
        "target_ids_sha256": _json_sha256(target_ids),
    }


def _nearest_rank(sorted_values: Sequence[int], quantile: float) -> int | None:
    if not sorted_values:
        return None
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _length_summary(
    values: Sequence[int],
    *,
    reporting_thresholds: Sequence[int],
    maximum: int,
) -> dict[str, Any]:
    ordered = sorted(values)
    count = len(ordered)
    exceedances = {
        str(threshold): {
            "count": sum(value > threshold for value in ordered),
            "fraction": (
                sum(value > threshold for value in ordered) / count if count else 0.0
            ),
        }
        for threshold in reporting_thresholds
    }
    maximum_exceeded = sum(value > maximum for value in ordered)
    return {
        "count": count,
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1] if ordered else None,
        "reporting_threshold_comparison": "greater_than",
        "reporting_thresholds": exceedances,
        "configured_maximum": maximum,
        "configured_maximum_exceeded_count": maximum_exceeded,
    }


def _accumulator_manifest(
    accumulator: _LengthAccumulator, thresholds: AuditThresholdConfig
) -> dict[str, Any]:
    return {
        "codec_token_length": _length_summary(
            accumulator.codec_tokens,
            reporting_thresholds=thresholds.codec_tokens,
            maximum=thresholds.maximum_codec_tokens,
        ),
        "labeled_token_length": _length_summary(
            accumulator.labeled_tokens,
            reporting_thresholds=thresholds.labeled_tokens,
            maximum=thresholds.maximum_labeled_tokens,
        ),
        "full_chat_token_length": _length_summary(
            accumulator.full_chat_tokens,
            reporting_thresholds=thresholds.full_chat_tokens,
            maximum=thresholds.maximum_full_chat_tokens,
        ),
        "maximum_observed_id": accumulator.maximum_observed_id,
    }


def _process_split(
    split: str,
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    codec: Any,
    codec_id_set: frozenset[int],
    config: LocalDiscreteAuditConfig,
    dataset_factory: Callable[..., Any],
    failures: _FailureLog,
) -> tuple[dict[str, Any], _LengthAccumulator, dict[str, Any] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"{split} JSONL does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    accumulator = _LengthAccumulator()
    physical_lines = 0
    blank_lines = 0
    decoded_records = 0
    failed_records = 0
    threshold_violation_count = 0
    longest: dict[str, Any] | None = None

    physical_lines = len(records)
    for record_offset, decoded in enumerate(records, start=1):
        decoded_records += 1
        record_id = _record_id(decoded)
        cache_context = decoded.get(CACHED_TARGET_FIELD)
        line_number = (
            cache_context.get("source_line", record_offset)
            if isinstance(cache_context, Mapping)
            else record_offset
        )
        try:
            measurements = _audit_record(
                decoded,
                tokenizer=tokenizer,
                codec=codec,
                codec_id_set=codec_id_set,
                dataset_max_seq_length=config.dataset_max_seq_length,
                dataset_factory=dataset_factory,
            )
        except _RecordValidationError as exc:
            failed_records += 1
            failures.add(
                split=split,
                line_number=line_number,
                record_id=record_id,
                code=exc.code,
                message=str(exc),
            )
            continue
        except Exception as exc:
            failed_records += 1
            failures.add(
                split=split,
                line_number=line_number,
                record_id=record_id,
                code="dataset_or_codec_error",
                message=f"{type(exc).__name__}: {exc}",
            )
            continue

        accumulator.add(measurements)
        for metric in _LENGTH_METRICS:
            maximum = config.thresholds.maximum(metric)
            if measurements[metric] > maximum:
                threshold_violation_count += 1
                failures.add(
                    split=split,
                    line_number=line_number,
                    record_id=record_id,
                    code=f"{metric}_maximum_exceeded",
                    message=f"{measurements[metric]} exceeds configured maximum {maximum}",
                )
        candidate = {
            "split": split,
            "line_number": line_number,
            "record_id": record_id,
            "measurements": measurements,
            "record": dict(decoded),
        }
        if longest is None or measurements["full_chat_tokens"] > longest["measurements"][
            "full_chat_tokens"
        ]:
            longest = candidate

    split_manifest = {
        "path": str(path),
        "input_sha256": digest,
        "input_bytes": path.stat().st_size,
        "physical_lines": physical_lines,
        "blank_lines": blank_lines,
        "decoded_records": decoded_records,
        "successful_records": len(accumulator.full_chat_tokens),
        "failed_records": failed_records,
        "threshold_violation_count": threshold_violation_count,
        "statistics": _accumulator_manifest(accumulator, config.thresholds),
    }
    return split_manifest, accumulator, longest


def _has_maximum_exceedance(statistics: Mapping[str, Any]) -> bool:
    return any(
        statistics[f"{metric[:-1] if metric.endswith('s') else metric}_length"][
            "configured_maximum_exceeded_count"
        ]
        > 0
        for metric in _LENGTH_METRICS
    )


def run_local_discrete_audit(
    config: LocalDiscreteAuditConfig,
    *,
    tokenizer_loader: Callable[[TokenizerAuditConfig], Any] = _load_offline_tokenizer,
    codec_factory: Callable[[int], Any] = _create_official_codec,
    registration_function: Callable[[Any, Any], Any] = register_named_codec_tokens,
    dataset_factory: Callable[..., Any] = _ResponseOnlyDataset,
    record_loader: Callable[[LocalDiscreteAuditConfig], Mapping[str, Sequence[Mapping[str, Any]]]]
    | None = None,
) -> dict[str, Any]:
    """Run the pinned-cache audit and atomically publish its artifacts."""

    tokenizer = tokenizer_loader(config.tokenizer)
    tokenizer_before = _tokenizer_vocabulary_manifest(tokenizer)
    codec = codec_factory(config.grid_size)
    ordered_tokens = tuple(codec.vocabulary_tokens())
    codec_manifest = codec.codec_manifest()
    declared_vocabulary_size = codec_manifest.get("vocabulary_size")
    if declared_vocabulary_size != len(ordered_tokens):
        raise RuntimeError(
            f"Codec returned {len(ordered_tokens)} tokens but declared "
            f"{declared_vocabulary_size!r}"
        )
    if (
        codec.metadata.backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
        and declared_vocabulary_size != OPENVGLAB_NAMED_VOCABULARY_SIZE
    ):
        raise RuntimeError("Official cached codec must declare exactly 45,062 tokens")
    registration = registration_function(tokenizer, codec)
    token_ids = tuple(registration.token_ids)
    if len(token_ids) != declared_vocabulary_size or len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Registered codec IDs do not match the declared unique-token count")
    tokenizer_after = _tokenizer_vocabulary_manifest(tokenizer)
    codec_id_set = frozenset(token_ids)
    failures = _FailureLog(config.maximum_failure_details)
    overall = _LengthAccumulator()
    split_manifests: dict[str, Any] = {}
    longest: dict[str, Any] | None = None
    if record_loader is None:
        if config.cache is None:
            raise ValueError("Official audit requires a pinned cache configuration")
        records_by_split = load_all_cached_records(config.cache)
    else:
        records_by_split = record_loader(config)
    if set(records_by_split) != {"train", "validation", "test"}:
        raise ValueError("Record loader must return train/validation/test splits")

    for split, path in config.data.items():
        split_manifest, accumulator, split_longest = _process_split(
            split,
            path,
            records_by_split[split],
            tokenizer=tokenizer,
            codec=codec,
            codec_id_set=codec_id_set,
            config=config,
            dataset_factory=dataset_factory,
            failures=failures,
        )
        split_manifests[split] = split_manifest
        overall.extend(accumulator)
        if split_longest is not None and (
            longest is None
            or split_longest["measurements"]["full_chat_tokens"]
            > longest["measurements"]["full_chat_tokens"]
        ):
            longest = split_longest

    observed_total = sum(item["decoded_records"] for item in split_manifests.values())
    if config.expected_total_records is not None and observed_total != config.expected_total_records:
        failures.add(
            split="all",
            line_number=None,
            record_id=None,
            code="unexpected_total_record_count",
            message=(
                f"Decoded {observed_total} records; expected {config.expected_total_records}"
            ),
        )
    overall_statistics = _accumulator_manifest(overall, config.thresholds)
    maximum_exceeded = any(
        overall_statistics[name]["configured_maximum_exceeded_count"] > 0
        for name in (
            "codec_token_length",
            "labeled_token_length",
            "full_chat_token_length",
        )
    )
    status = "fail" if failures.count or maximum_exceeded else "pass"
    report = {
        "schema_version": 1,
        "audit": "official_cached_openvglab_gemma_discrete_dataset",
        "status": status,
        "offline": True,
        "model_loaded": False,
        "configuration": {
            "tokenizer_path": str(config.tokenizer.path),
            "tokenizer_revision": config.tokenizer.revision,
            "local_files_only": True,
            "trust_remote_code": config.tokenizer.trust_remote_code,
            "codec_backend_id": codec.metadata.backend_id,
            "dataset_class": (
                f"{dataset_factory.__module__}.{dataset_factory.__qualname__}"
                if hasattr(dataset_factory, "__module__")
                else str(dataset_factory)
            ),
            "instruction_mode": "description_only",
            "target_representation": "omnisvg_discrete",
            "seed": 42,
            "dataset_max_seq_length": config.dataset_max_seq_length,
            "expected_total_records": config.expected_total_records,
            "quantile_method": "nearest-rank",
        },
        "provenance": (
            {
                "cache_path": str(config.cache.cache_path),
                "cache_sha256": config.cache.expected_cache_sha256,
                "audit_manifest_path": str(config.cache.audit_manifest_path),
                "audit_manifest_sha256": config.cache.expected_audit_sha256,
                "prepared_root": str(config.cache.prepared_root),
                "input_sha256": dict(config.cache.expected_input_sha256),
                "expected_split_counts": dict(config.cache.expected_split_counts),
                "allow_live_reencode": config.cache.allow_live_reencode,
            }
            if config.cache is not None
            else None
        ),
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocabulary_before_registration": tokenizer_before,
            "vocabulary_after_registration": tokenizer_after,
        },
        "codec": {
            "manifest": codec_manifest,
            "token_count": len(ordered_tokens),
            "ordered_token_strings_sha256": _json_sha256(ordered_tokens),
            "ordered_token_ids_sha256": registration.token_ids_sha256,
            "registration_vocabulary_sha256": getattr(
                registration, "vocabulary_sha256", None
            ),
            "minimum_registered_id": min(token_ids),
            "maximum_registered_id": max(token_ids),
            "added_token_count": registration.added_token_count,
        },
        "inputs": split_manifests,
        "overall": {
            "decoded_records": observed_total,
            "successful_records": len(overall.full_chat_tokens),
            "statistics": overall_statistics,
        },
        "failures": {
            "count": failures.count,
            "details_truncated": failures.count > len(failures.details),
            "details": failures.details,
        },
        "longest_record": (
            {
                "split": longest["split"],
                "line_number": longest["line_number"],
                "record_id": longest["record_id"],
                "measurements": longest["measurements"],
            }
            if longest is not None
            else None
        ),
    }

    if config.output.longest_record_path is not None and longest is not None:
        longest_payload = dict(longest["record"])
        longest_payload.pop(CACHED_TARGET_FIELD, None)
        if any(isinstance(value, (bytes, bytearray)) for value in longest_payload.values()):
            raise RuntimeError("Longest public record still contains binary cache payloads")
        longest_payload["official_discrete_audit"] = {
            "split": longest["split"],
            "line_number": longest["line_number"],
            "record_id": longest["record_id"],
            **longest["measurements"],
        }
        atomic_write_text(
            config.output.longest_record_path,
            json.dumps(longest_payload, ensure_ascii=False, sort_keys=True) + "\n",
        )
        report["longest_record"]["output_path"] = str(
            config.output.longest_record_path
        )
    atomic_write_text(
        config.output.report_path,
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit official cached OpenVGLab targets with a local Gemma tokenizer"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local_discrete_audit.yaml"),
    )
    args = parser.parse_args(argv)
    try:
        config = load_local_discrete_audit_config(args.config)
        report = run_local_discrete_audit(config)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
