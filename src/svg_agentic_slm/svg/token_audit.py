"""Streaming, deterministic MMSVG token-length audit for pinned codec backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import os
import sqlite3
import struct
import sys
import tempfile
import uuid
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from svg_agentic_slm.svg.codec_backends import (
    OPENVGLAB_TRAIN_4B_BACKEND_ID,
    CodecEncodeResult,
    CodecError,
    SVGCodec,
    create_svg_codec,
)

AUDIT_SCHEMA_VERSION = 1
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_THRESHOLDS = (2048, 8192, 32768)


class TokenAuditError(RuntimeError):
    """Base error for audit configuration or record handling."""


class AuditConfigurationError(TokenAuditError):
    pass


class AuditRecordError(TokenAuditError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ChatPrefixCounter(Protocol):
    def count_prefix_tokens(self, row: Mapping[str, Any]) -> int: ...

    def to_manifest(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ConstantChatPrefixCounter:
    prefix_tokens: int

    def __post_init__(self) -> None:
        if self.prefix_tokens < 0:
            raise AuditConfigurationError("constant chat prefix must be non-negative")

    def count_prefix_tokens(self, row: Mapping[str, Any]) -> int:
        del row
        return self.prefix_tokens

    def to_manifest(self) -> dict[str, Any]:
        return {"mode": "constant", "prefix_tokens": self.prefix_tokens}


@dataclass(frozen=True, slots=True)
class FieldChatPrefixCounter:
    field: str

    def count_prefix_tokens(self, row: Mapping[str, Any]) -> int:
        if self.field not in row:
            raise AuditRecordError(
                f"Missing chat-prefix length field: {self.field}",
                code="missing_chat_prefix_field",
            )
        value = row[self.field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditRecordError(
                f"Chat-prefix field {self.field!r} must be a non-negative integer",
                code="invalid_chat_prefix_length",
            )
        return value

    def to_manifest(self) -> dict[str, Any]:
        return {"mode": "row_field", "field": self.field}


def _normalize_chat_token_ids(value: Any) -> tuple[int, ...]:
    """Normalize one tokenizer result without accepting ambiguous batches."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise AuditRecordError(
                "Chat tokenizer mapping does not contain input_ids",
                code="invalid_chat_token_sequence",
            )
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception as exc:
            raise AuditRecordError(
                f"Chat tokenizer tensor conversion failed: {exc}",
                code="invalid_chat_token_sequence",
            ) from exc
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AuditRecordError(
            "Chat tokenizer returned a non-sequence",
            code="invalid_chat_token_sequence",
        )

    items = list(value)
    if len(items) == 1:
        nested = items[0]
        if hasattr(nested, "tolist"):
            try:
                nested = nested.tolist()
            except Exception as exc:
                raise AuditRecordError(
                    f"Chat tokenizer nested tensor conversion failed: {exc}",
                    code="invalid_chat_token_sequence",
                ) from exc
        if isinstance(nested, Sequence) and not isinstance(
            nested, (str, bytes, bytearray)
        ):
            items = list(nested)

    if not items:
        raise AuditRecordError(
            "Chat tokenizer returned an empty input_ids sequence",
            code="invalid_chat_token_sequence",
        )

    token_ids: list[int] = []
    for index, token in enumerate(items):
        if hasattr(token, "tolist"):
            try:
                token = token.tolist()
            except Exception as exc:
                raise AuditRecordError(
                    f"Chat token {index} tensor conversion failed: {exc}",
                    code="invalid_chat_token_sequence",
                ) from exc
        if isinstance(token, bool) or isinstance(token, (Mapping, Sequence)):
            raise AuditRecordError(
                "Chat tokenizer returned multiple batches or a nested token sequence",
                code="invalid_chat_token_sequence",
            )
        try:
            token_id = operator.index(token)
        except TypeError as exc:
            raise AuditRecordError(
                f"Chat token {index} is not an integer ID",
                code="invalid_chat_token_sequence",
            ) from exc
        if token_id < 0:
            raise AuditRecordError(
                f"Chat token {index} is negative",
                code="invalid_chat_token_sequence",
            )
        token_ids.append(token_id)
    return tuple(token_ids)


class LocalTokenizerChatPrefixCounter:
    """Count a text-only assistant prefix with a strictly local HF tokenizer."""

    def __init__(
        self,
        tokenizer_root: str | Path,
        *,
        prompt_field: str,
        system_field: str | None = None,
    ) -> None:
        root = Path(tokenizer_root).expanduser().resolve()
        if not root.is_dir():
            raise AuditConfigurationError(f"Local tokenizer root does not exist: {root}")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise AuditConfigurationError(
                f"Unable to load local chat tokenizer without network access: {exc}"
            ) from exc

        artifact_hashes: dict[str, str] = {}
        allowed_suffixes = {".json", ".model", ".tiktoken"}
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in allowed_suffixes:
                artifact_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not artifact_hashes:
            raise AuditConfigurationError(
                f"No hashable tokenizer artifacts found under local root: {root}"
            )

        self._root = root
        self._prompt_field = prompt_field
        self._system_field = system_field
        self._tokenizer = tokenizer
        self._artifact_hashes = artifact_hashes

    def count_prefix_tokens(self, row: Mapping[str, Any]) -> int:
        prompt = row.get(self._prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise AuditRecordError(
                f"Prompt field {self._prompt_field!r} must be a non-empty string",
                code="invalid_chat_prompt",
            )
        messages: list[dict[str, str]] = []
        if self._system_field is not None:
            system = row.get(self._system_field)
            if not isinstance(system, str):
                raise AuditRecordError(
                    f"System field {self._system_field!r} must be a string",
                    code="invalid_chat_system_prompt",
                )
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            token_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise AuditRecordError(
                f"Local chat-template tokenization failed: {exc}",
                code="chat_template_failed",
            ) from exc
        return len(_normalize_chat_token_ids(token_ids))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mode": "local_hf_tokenizer",
            "tokenizer_root": str(self._root),
            "prompt_field": self._prompt_field,
            "system_field": self._system_field,
            "local_files_only": True,
            "trust_remote_code": False,
            "artifact_sha256": self._artifact_hashes,
            "definition": "chat assistant prefix + framed SVG IDs",
            "multimodal_image_tokens_included": False,
        }


@dataclass(frozen=True, slots=True)
class TokenAuditConfig:
    prepared_root: Path
    output_path: Path
    token_cache_path: Path | None = None
    svg_field: str = "output_svg"
    id_field: str = "id"
    domain_field: str = "domain"
    source_field: str = "source"
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "prepared_root", Path(self.prepared_root).expanduser().resolve())
        object.__setattr__(self, "output_path", Path(self.output_path).expanduser().resolve())
        if self.token_cache_path is not None:
            object.__setattr__(
                self,
                "token_cache_path",
                Path(self.token_cache_path).expanduser().resolve(),
            )
        thresholds = tuple(sorted(set(self.thresholds)))
        if not thresholds or any(value <= 0 for value in thresholds):
            raise AuditConfigurationError("thresholds must be positive integers")
        object.__setattr__(self, "thresholds", thresholds)
        if not self.svg_field or not self.id_field or not self.domain_field or not self.source_field:
            raise AuditConfigurationError("row field names must be non-empty")

    @property
    def split_paths(self) -> tuple[tuple[str, Path], ...]:
        return tuple((split, self.prepared_root / f"{split}.jsonl") for split in DEFAULT_SPLITS)


class _LengthAccumulator:
    def __init__(self, thresholds: tuple[int, ...]) -> None:
        self._thresholds = thresholds
        self._values: list[int] = []
        self._over = {threshold: 0 for threshold in thresholds}
        self._max_value: int | None = None
        self._max_sample: dict[str, Any] | None = None

    def add(self, value: int, sample: Mapping[str, Any]) -> None:
        if value < 0:
            raise ValueError("length must be non-negative")
        self._values.append(value)
        for threshold in self._thresholds:
            if value > threshold:
                self._over[threshold] += 1
        if self._max_value is None or value > self._max_value:
            self._max_value = value
            self._max_sample = dict(sample)

    @staticmethod
    def _nearest_rank(sorted_values: Sequence[int], quantile: float) -> int | None:
        if not sorted_values:
            return None
        index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
        return sorted_values[index]

    def to_manifest(self) -> dict[str, Any]:
        ordered = sorted(self._values)
        return {
            "count": len(ordered),
            "min": ordered[0] if ordered else None,
            "p50": self._nearest_rank(ordered, 0.50),
            "p95": self._nearest_rank(ordered, 0.95),
            "p99": self._nearest_rank(ordered, 0.99),
            "max": self._max_value,
            "max_sample": self._max_sample,
            "greater_than_threshold": {
                str(threshold): self._over[threshold] for threshold in self._thresholds
            },
            "quantile_method": "nearest-rank",
        }


def _mapping_path_value(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _identity_value(
    row: Mapping[str, Any],
    configured_field: str,
    aliases: Sequence[str],
) -> str | None:
    candidates = tuple(dict.fromkeys((configured_field.split(".")[-1], *aliases)))
    direct = _mapping_path_value(row, configured_field)
    if direct is not None and not isinstance(direct, (Mapping, Sequence)):
        return str(direct)
    if isinstance(direct, str):
        return direct

    containers: list[Mapping[str, Any]] = [row]
    for key in ("metadata", "source_metadata", "provenance"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
            source_nested = nested.get("source")
            if isinstance(source_nested, Mapping):
                containers.append(source_nested)
    for container in containers:
        for key in candidates:
            value = container.get(key)
            if isinstance(value, str):
                return value
            if value is not None and not isinstance(value, (Mapping, Sequence, bool)):
                return str(value)
    return None


def _pack_uint32(tokens: Sequence[int]) -> bytes:
    if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
        raise AuditRecordError(
            "Token cache accepts integer IDs only",
            code="non_integer_cache_token",
        )
    if any(token < 0 or token > 0xFFFFFFFF for token in tokens):
        raise AuditRecordError(
            "Token cache ID is outside uint32 range",
            code="cache_token_out_of_range",
        )
    packed = struct.pack(f"<{len(tokens)}I", *tokens) if tokens else b""
    return zlib.compress(packed, level=9)


class _AtomicSqliteTokenCache:
    def __init__(self, target: Path, codec_manifest: Mapping[str, Any]) -> None:
        self._target = target
        target.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        self._connection = sqlite3.connect(self._temporary)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE sequences (
                sample_index INTEGER PRIMARY KEY,
                split TEXT NOT NULL,
                source_line INTEGER NOT NULL,
                sample_id TEXT,
                domain TEXT,
                source TEXT,
                body_count INTEGER NOT NULL,
                framed_count INTEGER NOT NULL,
                body_uint32_le_zlib BLOB NOT NULL,
                framed_uint32_le_zlib BLOB NOT NULL
            );
            CREATE INDEX sequences_split_sample_id
                ON sequences(split, sample_id);
            """
        )
        metadata = {
            "cache_schema_version": 1,
            "integer_encoding": "uint32-little-endian-zlib",
            "codec": dict(codec_manifest),
        }
        for key, value in metadata.items():
            self._connection.execute(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True, separators=(",", ":"))),
            )

    def add(
        self,
        *,
        sample_index: int,
        split: str,
        source_line: int,
        sample_id: str | None,
        domain: str | None,
        source: str | None,
        result: CodecEncodeResult,
    ) -> None:
        if result.body_tokens is None:
            raise AuditRecordError(
                "Audit cache requires a backend that exposes body tokens",
                code="body_tokens_unavailable",
            )
        body = tuple(result.body_tokens)
        framed = tuple(result.tokens)
        self._connection.execute(
            """
            INSERT INTO sequences(
                sample_index, split, source_line, sample_id, domain, source,
                body_count, framed_count, body_uint32_le_zlib, framed_uint32_le_zlib
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_index,
                split,
                source_line,
                sample_id,
                domain,
                source,
                len(body),
                len(framed),
                sqlite3.Binary(_pack_uint32(body)),
                sqlite3.Binary(_pack_uint32(framed)),
            ),
        )

    def finalize(self) -> None:
        self._connection.commit()
        self._connection.close()
        descriptor = os.open(self._temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(self._temporary, self._target)

    def abort(self) -> None:
        try:
            self._connection.close()
        finally:
            self._temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class TokenAuditRunner:
    """Stream three fixed JSONL splits and write a reproducible audit manifest."""

    def __init__(
        self,
        codec: SVGCodec,
        chat_prefix_counter: ChatPrefixCounter,
        config: TokenAuditConfig,
    ) -> None:
        if not codec.metadata.can_encode:
            raise AuditConfigurationError("selected codec backend cannot encode")
        if codec.metadata.backend_id != OPENVGLAB_TRAIN_4B_BACKEND_ID:
            raise AuditConfigurationError(
                "phase-1 MMSVG audit accepts only the pinned OpenVGLab 4B analysis backend"
            )
        self._codec = codec
        self._chat_prefix_counter = chat_prefix_counter
        self._config = config

    def run(self) -> dict[str, Any]:
        for split, path in self._config.split_paths:
            if not path.is_file():
                raise AuditConfigurationError(f"Missing required {split} split: {path}")

        metrics = {
            "body": _LengthAccumulator(self._config.thresholds),
            "bos_eos": _LengthAccumulator(self._config.thresholds),
            "full_chat": _LengthAccumulator(self._config.thresholds),
        }
        failures: list[dict[str, Any]] = []
        inputs: dict[str, Any] = {}
        split_counts: dict[str, dict[str, int]] = {}
        total_records = 0
        total_successes = 0
        cache = (
            _AtomicSqliteTokenCache(
                self._config.token_cache_path,
                self._codec.metadata.to_manifest(),
            )
            if self._config.token_cache_path is not None
            else None
        )

        try:
            for split, path in self._config.split_paths:
                digest = hashlib.sha256()
                byte_count = 0
                split_total = 0
                split_success = 0
                with path.open("rb") as handle:
                    for source_line, raw_line in enumerate(handle, start=1):
                        digest.update(raw_line)
                        byte_count += len(raw_line)
                        sample_index = total_records
                        total_records += 1
                        split_total += 1
                        sample_id: str | None = None
                        domain: str | None = None
                        source: str | None = None
                        try:
                            if not raw_line.strip():
                                raise AuditRecordError("Blank JSONL line", code="blank_jsonl_line")
                            try:
                                row = json.loads(raw_line.decode("utf-8"))
                            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                raise AuditRecordError(
                                    f"Invalid UTF-8 JSON object: {exc}",
                                    code="invalid_jsonl_record",
                                ) from exc
                            if not isinstance(row, dict):
                                raise AuditRecordError(
                                    "JSONL record must be an object",
                                    code="invalid_jsonl_record_type",
                                )
                            sample_id = _identity_value(
                                row,
                                self._config.id_field,
                                (
                                    "id",
                                    "sample_id",
                                    "source_id",
                                    "source_index",
                                    "record_id",
                                    "upstream_id",
                                ),
                            )
                            domain = _identity_value(
                                row,
                                self._config.domain_field,
                                ("domain", "subset", "mmsvg_domain", "source_subset"),
                            )
                            source = _identity_value(
                                row,
                                self._config.source_field,
                                (
                                    "source",
                                    "source_dataset",
                                    "dataset",
                                    "dataset_name",
                                    "source_repo",
                                ),
                            )
                            svg = row.get(self._config.svg_field)
                            if not isinstance(svg, str) or not svg.strip():
                                raise AuditRecordError(
                                    f"Missing non-empty SVG field: {self._config.svg_field}",
                                    code="invalid_svg_field",
                                )
                            result = self._codec.encode(svg)
                            if result.body_tokens is None:
                                raise AuditRecordError(
                                    "Codec result does not expose body tokens",
                                    code="body_tokens_unavailable",
                                )
                            body_length = len(result.body_tokens)
                            framed_length = len(result.tokens)
                            if framed_length != body_length + 2:
                                raise AuditRecordError(
                                    "Pinned encoder result is not BOS + body + EOS",
                                    code="unexpected_framed_length",
                                )
                            prefix_length = self._chat_prefix_counter.count_prefix_tokens(row)
                            full_chat_length = prefix_length + framed_length
                            sample = {
                                "sample_index": sample_index,
                                "split": split,
                                "source_line": source_line,
                                "id": sample_id,
                                "domain": domain,
                                "source": source,
                            }
                            metrics["body"].add(body_length, sample)
                            metrics["bos_eos"].add(framed_length, sample)
                            metrics["full_chat"].add(full_chat_length, sample)
                            if cache is not None:
                                cache.add(
                                    sample_index=sample_index,
                                    split=split,
                                    source_line=source_line,
                                    sample_id=sample_id,
                                    domain=domain,
                                    source=source,
                                    result=result,
                                )
                            split_success += 1
                            total_successes += 1
                        except CodecError as exc:
                            failures.append(
                                {
                                    "sample_index": sample_index,
                                    "split": split,
                                    "source_line": source_line,
                                    "id": sample_id,
                                    "domain": domain,
                                    "source": source,
                                    "error": exc.to_manifest(),
                                }
                            )
                        except AuditRecordError as exc:
                            failures.append(
                                {
                                    "sample_index": sample_index,
                                    "split": split,
                                    "source_line": source_line,
                                    "id": sample_id,
                                    "domain": domain,
                                    "source": source,
                                    "error": {
                                        "type": type(exc).__name__,
                                        "code": exc.code,
                                        "message": str(exc),
                                    },
                                }
                            )
                inputs[split] = {
                    "path": str(path),
                    "sha256": digest.hexdigest(),
                    "bytes": byte_count,
                    "records": split_total,
                }
                split_counts[split] = {
                    "records": split_total,
                    "successes": split_success,
                    "failures": split_total - split_success,
                }
            if cache is not None:
                cache.finalize()
        except Exception:
            if cache is not None:
                cache.abort()
            raise

        manifest: dict[str, Any] = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "codec": self._codec.metadata.to_manifest(),
            "determinism": {
                "masking_enabled": False,
                "dataset_shuffling": False,
                "split_order": list(DEFAULT_SPLITS),
            },
            "inputs": inputs,
            "row_fields": {
                "svg": self._config.svg_field,
                "id": self._config.id_field,
                "domain": self._config.domain_field,
                "source": self._config.source_field,
            },
            "chat_prefix": self._chat_prefix_counter.to_manifest(),
            "length_definitions": {
                "body": "OpenVGLab SVG IDs before BOS/EOS",
                "bos_eos": "BOS + OpenVGLab SVG body + EOS",
                "full_chat": "configured chat assistant prefix + BOS/EOS-framed SVG IDs",
            },
            "thresholds": list(self._config.thresholds),
            "metrics": {name: accumulator.to_manifest() for name, accumulator in metrics.items()},
            "counts": {
                "records": total_records,
                "successes": total_successes,
                "failures": len(failures),
                "by_split": split_counts,
            },
            "failures": failures,
            "token_cache": (
                {
                    "path": str(self._config.token_cache_path),
                    "format": "sqlite3/uint32-little-endian-zlib",
                    "indexed_by": ["sample_index", "split+sample_id"],
                    "successful_records_only": True,
                }
                if self._config.token_cache_path is not None
                else None
            ),
        }
        _atomic_write_json(self._config.output_path, manifest)
        return manifest


def _expanded_yaml(path: Path) -> dict[str, Any]:
    try:
        expanded = os.path.expandvars(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuditConfigurationError(f"Unable to read audit config: {exc}") from exc
    if "${" in expanded:
        raise AuditConfigurationError("Audit config contains unresolved ${ENVIRONMENT_VARIABLE}")
    try:
        payload = yaml.safe_load(expanded)
    except yaml.YAMLError as exc:
        raise AuditConfigurationError(f"Invalid audit YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditConfigurationError("Audit YAML root must be a mapping")
    return payload


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AuditConfigurationError(f"Audit config field {key!r} must be a non-empty string")
    return value


def _chat_counter_from_config(payload: Mapping[str, Any]) -> ChatPrefixCounter:
    chat = payload.get("chat")
    if not isinstance(chat, dict):
        raise AuditConfigurationError("Audit config requires a chat mapping")
    mode = chat.get("mode")
    if mode == "constant":
        value = chat.get("prefix_tokens")
        if isinstance(value, bool) or not isinstance(value, int):
            raise AuditConfigurationError("chat.prefix_tokens must be an integer")
        return ConstantChatPrefixCounter(value)
    if mode == "row_field":
        return FieldChatPrefixCounter(_required_string(chat, "field"))
    if mode == "local_hf_tokenizer":
        system_field = chat.get("system_field")
        if system_field is not None and not isinstance(system_field, str):
            raise AuditConfigurationError("chat.system_field must be a string or null")
        return LocalTokenizerChatPrefixCounter(
            _required_string(chat, "tokenizer_root"),
            prompt_field=_required_string(chat, "prompt_field"),
            system_field=system_field,
        )
    raise AuditConfigurationError(
        "chat.mode must be one of: local_hf_tokenizer, row_field, constant"
    )


def _optional_path(value: Any, *, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AuditConfigurationError(f"{field_name} must be a path string or null")
    return Path(value)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit pinned OpenVGLab 4B SVG token lengths over train/validation/test JSONL"
    )
    parser.add_argument("--config", type=Path, required=True, help="Audit YAML configuration")
    parser.add_argument("--upstream-root", type=Path, help="Override external OmniSVG-train root")
    parser.add_argument("--prepared-root", type=Path, help="Override prepared MMSVG 20K root")
    parser.add_argument("--output", type=Path, help="Override atomic JSON manifest path")
    parser.add_argument("--token-cache", type=Path, help="Override optional SQLite cache path")
    parser.add_argument(
        "--no-token-cache",
        action="store_true",
        help="Disable the cache even when configured",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        payload = _expanded_yaml(args.config.expanduser().resolve())
        backend = payload.get("backend")
        if backend != OPENVGLAB_TRAIN_4B_BACKEND_ID:
            raise AuditConfigurationError(
                f"backend must be exactly {OPENVGLAB_TRAIN_4B_BACKEND_ID!r}"
            )
        upstream_root = args.upstream_root or Path(_required_string(payload, "upstream_root"))
        prepared_root = args.prepared_root or Path(_required_string(payload, "prepared_root"))
        output_path = args.output or Path(_required_string(payload, "output"))
        configured_cache = _optional_path(payload.get("token_cache"), field_name="token_cache")
        cache_path = None if args.no_token_cache else (args.token_cache or configured_cache)
        fields = payload.get("fields", {})
        if not isinstance(fields, dict):
            raise AuditConfigurationError("fields must be a mapping")
        raw_thresholds = payload.get("thresholds", list(DEFAULT_THRESHOLDS))
        if not isinstance(raw_thresholds, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_thresholds
        ):
            raise AuditConfigurationError("thresholds must be a list of integers")

        codec = create_svg_codec(
            OPENVGLAB_TRAIN_4B_BACKEND_ID,
            upstream_root=upstream_root,
        )
        runner = TokenAuditRunner(
            codec=codec,
            chat_prefix_counter=_chat_counter_from_config(payload),
            config=TokenAuditConfig(
                prepared_root=prepared_root,
                output_path=output_path,
                token_cache_path=cache_path,
                svg_field=str(fields.get("svg", "output_svg")),
                id_field=str(fields.get("id", "id")),
                domain_field=str(fields.get("domain", "domain")),
                source_field=str(fields.get("source", "source")),
                thresholds=tuple(raw_thresholds),
            ),
        )
        manifest = runner.run()
    except (CodecError, TokenAuditError, OSError, ValueError) as exc:
        print(f"token audit failed: {exc}", file=sys.stderr)
        return 2
    failure_count = manifest["counts"]["failures"]
    print(
        json.dumps(
            {
                "manifest": str(output_path.expanduser().resolve()),
                "records": manifest["counts"]["records"],
                "successes": manifest["counts"]["successes"],
                "failures": failure_count,
                "max": {
                    key: value["max"] for key, value in manifest["metrics"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 2 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
