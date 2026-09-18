"""Pinned OpenVGLab 4B cache mapped into a model-neutral Gemma namespace.

The official released training encoder is encode-only and emits absolute Qwen
vocabulary IDs. This module does not claim checkpoint compatibility and never
silently re-encodes SVG. It consumes only the audited immutable SQLite cache,
then maps each upstream integer ID to one newly registered named token.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

OFFICIAL_CACHED_GEMMA_BACKEND_ID = (
    "openvglab-omnisvg-train-4b-cached-gemma-named-v1"
)
OPENVGLAB_TRAIN_COMMIT = "812489fd9d191e39fe94bc0c4027e5d0121e0fc6"
OPENVGLAB_TOKENIZATION_CONFIG_SHA256 = (
    "bd56bc2bb9b39f614a9d553d2319332a18a075c4239bb71ec437822b20d97dd4"
)
PINNED_CACHE_SHA256 = "da25dc3db8739e846d952b3044e5f7977eaf8820f4ea0c45c46ed8111f91a3fa"
PINNED_AUDIT_SHA256 = "30e3deb8a8cf443a1c32a58bc48e41b35b4adc615c14eb7fb9dee1f48d4e35c7"
PINNED_INPUT_SHA256 = {
    "train": "7920c33235ede566657003ac63166b01b3e594b6ccc521d34d9fbe937e5642bd",
    "validation": "ab4a24667b19160a3c3337ad9b25a8e1c4c4ba58116ec690c3829d25b621ef2a",
    "test": "381d28d516273698a1c4e0b793376fe90aa7135ad89b69a04ec6cbda57ab9d3b",
}
PINNED_SPLIT_COUNTS = {"train": 18000, "validation": 1000, "test": 1000}
OPENVGLAB_MIN_PRODUCIBLE_ID = 151938
OPENVGLAB_MAX_PRODUCIBLE_ID = 196999
OPENVGLAB_BOS_ID = 196998
OPENVGLAB_EOS_ID = 196999
OPENVGLAB_NAMED_VOCABULARY_SIZE = (
    OPENVGLAB_MAX_PRODUCIBLE_ID - OPENVGLAB_MIN_PRODUCIBLE_ID + 1
)
CACHED_TARGET_FIELD = "_openvglab_4b_cached_target_v1"
_PREPARED_SPLITS = ("train", "validation", "test")


def _normalize_requested_splits(
    requested_splits: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested_splits is None:
        return _PREPARED_SPLITS
    if isinstance(requested_splits, (str, bytes)):
        raise TypeError("requested_splits must be a sequence of split names.")
    requested = tuple(requested_splits)
    if not requested:
        raise ValueError("requested_splits must contain at least one split.")
    unknown = [split for split in requested if split not in _PREPARED_SPLITS]
    if unknown:
        raise ValueError(f"Unknown prepared split: {unknown[0]}")
    if len(set(requested)) != len(requested):
        raise ValueError("requested_splits must not contain duplicates.")
    requested_set = set(requested)
    return tuple(split for split in _PREPARED_SPLITS if split in requested_set)

_EXPECTED_CODEC_METADATA = {
    "backend_id": (
        "openvglab-omnisvg-train-4b-812489fd9d191e39fe94bc0c4027e5d0121e0fc6"
    ),
    "codec_version": OPENVGLAB_TRAIN_COMMIT,
    "dialect": "openvglab-training-code-4b",
    "directions": ["encode"],
    "model_family": "Qwen2.5-VL",
    "model_size": "4B",
    "official_checkpoint_compatible": False,
    "provenance": "official-training-source",
    "source_revision": OPENVGLAB_TRAIN_COMMIT,
    "token_kind": "absolute-integer-id",
}


class OfficialDiscreteCacheError(RuntimeError):
    """Raised when pinned cache provenance or record identity cannot be proven."""


class UnsupportedOfficialDiscreteOperation(OfficialDiscreteCacheError):
    """Raised for live encoding or decoding, which this dialect cannot prove."""


@dataclass(frozen=True)
class CachedDialectMetadata:
    backend_id: str = OFFICIAL_CACHED_GEMMA_BACKEND_ID
    codec_name: str = "OpenVGLab 4B cached IDs in a Gemma named-token namespace"
    codec_version: str = "cache-v1"
    dialect: str = "openvglab-training-code-4b-to-gemma-named-v1"
    directions: tuple[str, ...] = ("cached-target",)
    lossy: bool = True
    official_checkpoint_compatible: bool = False
    provenance: str = "official-training-source+pinned-audited-cache"
    token_kind: str = "named-special-token-mapped-from-absolute-integer-id"

    @property
    def can_encode(self) -> bool:
        return False

    @property
    def can_decode(self) -> bool:
        return False

    def to_manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "codec_name": self.codec_name,
            "codec_version": self.codec_version,
            "dialect": self.dialect,
            "directions": list(self.directions),
            "lossy": self.lossy,
            "official_checkpoint_compatible": self.official_checkpoint_compatible,
            "provenance": self.provenance,
            "token_kind": self.token_kind,
            "source_revision": OPENVGLAB_TRAIN_COMMIT,
            "tokenization_config_sha256": OPENVGLAB_TOKENIZATION_CONFIG_SHA256,
            "training_capability": "pinned-cache-only",
            "live_encode_supported": False,
            "decode_supported": False,
            "generation_supported": False,
        }


@dataclass(frozen=True)
class OpenVGLabCacheConfig:
    cache_path: Path
    audit_manifest_path: Path
    prepared_root: Path
    expected_cache_sha256: str = PINNED_CACHE_SHA256
    expected_audit_sha256: str = PINNED_AUDIT_SHA256
    expected_input_sha256: Mapping[str, str] = field(
        default_factory=lambda: dict(PINNED_INPUT_SHA256)
    )
    expected_split_counts: Mapping[str, int] = field(
        default_factory=lambda: dict(PINNED_SPLIT_COUNTS)
    )
    allow_live_reencode: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_path", Path(self.cache_path).expanduser().resolve())
        object.__setattr__(
            self,
            "audit_manifest_path",
            Path(self.audit_manifest_path).expanduser().resolve(),
        )
        object.__setattr__(
            self, "prepared_root", Path(self.prepared_root).expanduser().resolve()
        )
        for name in ("expected_cache_sha256", "expected_audit_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if set(self.expected_input_sha256) != set(PINNED_INPUT_SHA256):
            raise ValueError("expected_input_sha256 must define train/validation/test")
        if set(self.expected_split_counts) != set(PINNED_SPLIT_COUNTS):
            raise ValueError("expected_split_counts must define train/validation/test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_file_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise OfficialDiscreteCacheError(f"Pinned {label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise OfficialDiscreteCacheError(
            f"Pinned {label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )


def _unpack_framed_ids(blob: bytes, expected_count: int) -> tuple[int, ...]:
    try:
        raw = zlib.decompress(blob)
    except zlib.error as exc:
        raise OfficialDiscreteCacheError(f"Invalid cached zlib sequence: {exc}") from exc
    if len(raw) % 4:
        raise OfficialDiscreteCacheError("Cached uint32 payload byte length is not divisible by four")
    count = len(raw) // 4
    if count != expected_count:
        raise OfficialDiscreteCacheError(
            f"Cached framed count mismatch: metadata {expected_count}, payload {count}"
        )
    return struct.unpack(f"<{count}I", raw)


def _prepared_record_identity(record: Mapping[str, Any]) -> str | None:
    candidates = ("id", "sample_id", "source_id", "source_index", "record_id", "upstream_id")
    for container in (record, record.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        for key in candidates:
            value = container.get(key)
            if value is not None:
                return str(value)
    return None


class CachedOpenVGLabGemmaDialect:
    """Map pinned upstream 4B IDs to deterministic named special-token strings."""

    metadata = CachedDialectMetadata()

    @staticmethod
    def token_for_upstream_id(upstream_id: int) -> str:
        if (
            isinstance(upstream_id, bool)
            or not isinstance(upstream_id, int)
            or not OPENVGLAB_MIN_PRODUCIBLE_ID
            <= upstream_id
            <= OPENVGLAB_MAX_PRODUCIBLE_ID
        ):
            raise OfficialDiscreteCacheError(
                f"Upstream ID {upstream_id!r} is outside the declared 4B namespace"
            )
        return f"<svgovg4b:{upstream_id}>"

    @cached_property
    def _vocabulary(self) -> tuple[str, ...]:
        return tuple(
            self.token_for_upstream_id(upstream_id)
            for upstream_id in range(
                OPENVGLAB_MIN_PRODUCIBLE_ID, OPENVGLAB_MAX_PRODUCIBLE_ID + 1
            )
        )

    def vocabulary_tokens(self) -> tuple[str, ...]:
        return self._vocabulary

    @property
    def eos_token(self) -> str:
        return self.token_for_upstream_id(OPENVGLAB_EOS_ID)

    def codec_manifest(self) -> dict[str, Any]:
        return {
            **self.metadata.to_manifest(),
            "namespace": "<svgovg4b:UPSTREAM_ID>",
            "ordered_upstream_id_min": OPENVGLAB_MIN_PRODUCIBLE_ID,
            "ordered_upstream_id_max": OPENVGLAB_MAX_PRODUCIBLE_ID,
            "vocabulary_size": OPENVGLAB_NAMED_VOCABULARY_SIZE,
            "bos_upstream_id": OPENVGLAB_BOS_ID,
            "eos_upstream_id": OPENVGLAB_EOS_ID,
            "bos_token": self.token_for_upstream_id(OPENVGLAB_BOS_ID),
            "eos_token": self.eos_token,
            "ordered_token_strings_sha256": _json_sha256(self.vocabulary_tokens()),
            "namespace_scope": (
                "conservative contiguous source interval covering every audited and "
                "configuration-producible 4B training ID"
            ),
            "qwen_checkpoint_compatibility_claimed": False,
        }

    def target_tokens_for_record(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        payload = record.get(CACHED_TARGET_FIELD)
        if not isinstance(payload, Mapping):
            raise OfficialDiscreteCacheError(
                "Official discrete target lacks pinned cache record context; live re-encode is disabled"
            )
        if payload.get("backend_id") != OFFICIAL_CACHED_GEMMA_BACKEND_ID:
            raise OfficialDiscreteCacheError("Cached target backend provenance mismatch")
        if payload.get("cache_sha256") != payload.get("expected_cache_sha256"):
            raise OfficialDiscreteCacheError("Cached target cache SHA-256 provenance mismatch")
        framed_blob = payload.get("framed_uint32_le_zlib")
        framed_count = payload.get("framed_count")
        if not isinstance(framed_blob, bytes) or not isinstance(framed_count, int):
            raise OfficialDiscreteCacheError("Cached target payload is malformed")
        ids = _unpack_framed_ids(framed_blob, framed_count)
        if len(ids) < 2 or ids[0] != OPENVGLAB_BOS_ID or ids[-1] != OPENVGLAB_EOS_ID:
            raise OfficialDiscreteCacheError("Cached target lacks exact OpenVGLab BOS/EOS framing")
        return tuple(self.token_for_upstream_id(upstream_id) for upstream_id in ids)

    def target_for_record(self, record: Mapping[str, Any]) -> str:
        return "".join(self.target_tokens_for_record(record))

    def encode(self, svg: str) -> Any:
        raise UnsupportedOfficialDiscreteOperation(
            "Official Gemma discrete training is cache-only; live SVG encoding is disabled"
        )

    def decode(self, tokens: Sequence[Any]) -> Any:
        raise UnsupportedOfficialDiscreteOperation(
            "No audited decoder or generation grammar exists for the OpenVGLab 4B +1 training dialect"
        )


class PinnedOpenVGLabTokenCache:
    """Read and validate the immutable 20K token cache without write access."""

    def __init__(
        self,
        config: OpenVGLabCacheConfig,
        requested_splits: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.requested_splits = _normalize_requested_splits(requested_splits)
        _require_file_hash(
            config.cache_path, config.expected_cache_sha256, label="SQLite token cache"
        )
        _require_file_hash(
            config.audit_manifest_path,
            config.expected_audit_sha256,
            label="token audit manifest",
        )
        self.audit_manifest = self._load_audit_manifest()
        uri = f"file:{config.cache_path.as_posix()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.execute("PRAGMA query_only=ON")
        self._validate_database()

    def __enter__(self) -> PinnedOpenVGLabTokenCache:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _load_audit_manifest(self) -> Mapping[str, Any]:
        try:
            manifest = json.loads(self.config.audit_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OfficialDiscreteCacheError(f"Invalid pinned audit manifest: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise OfficialDiscreteCacheError("Pinned audit manifest root is not an object")
        codec = manifest.get("codec")
        if not isinstance(codec, Mapping):
            raise OfficialDiscreteCacheError("Pinned audit manifest lacks codec provenance")
        for key, expected in _EXPECTED_CODEC_METADATA.items():
            if codec.get(key) != expected:
                raise OfficialDiscreteCacheError(
                    f"Pinned audit codec provenance mismatch for {key}: {codec.get(key)!r}"
                )
        inputs = manifest.get("inputs")
        if not isinstance(inputs, Mapping):
            raise OfficialDiscreteCacheError("Pinned audit manifest lacks input hashes")
        for split, expected_hash in self.config.expected_input_sha256.items():
            entry = inputs.get(split)
            if not isinstance(entry, Mapping) or entry.get("sha256") != expected_hash:
                raise OfficialDiscreteCacheError(
                    f"Pinned audit input SHA-256 mismatch for {split}"
                )
        counts = manifest.get("counts")
        if not isinstance(counts, Mapping) or counts.get("failures") != 0:
            raise OfficialDiscreteCacheError("Pinned audit did not record a failure-free cache")
        return manifest

    def _validate_database(self) -> None:
        metadata_rows = self.connection.execute(
            "SELECT key, value_json FROM metadata ORDER BY key"
        ).fetchall()
        try:
            metadata = {key: json.loads(value) for key, value in metadata_rows}
        except json.JSONDecodeError as exc:
            raise OfficialDiscreteCacheError(f"Invalid SQLite metadata JSON: {exc}") from exc
        if metadata.get("cache_schema_version") != 1:
            raise OfficialDiscreteCacheError("Unsupported SQLite token cache schema")
        if metadata.get("integer_encoding") != "uint32-little-endian-zlib":
            raise OfficialDiscreteCacheError("Unexpected SQLite token integer encoding")
        codec = metadata.get("codec")
        if not isinstance(codec, Mapping):
            raise OfficialDiscreteCacheError("SQLite cache lacks codec provenance")
        for key, expected in _EXPECTED_CODEC_METADATA.items():
            if codec.get(key) != expected:
                raise OfficialDiscreteCacheError(
                    f"SQLite codec provenance mismatch for {key}: {codec.get(key)!r}"
                )
        for split in self.requested_splits:
            expected_count = self.config.expected_split_counts[split]
            observed = self.connection.execute(
                "SELECT COUNT(*) FROM sequences WHERE split = ?", (split,)
            ).fetchone()[0]
            if observed != expected_count:
                raise OfficialDiscreteCacheError(
                    f"SQLite {split} row count mismatch: expected {expected_count}, observed {observed}"
                )

    def load_split(self, split: str) -> list[dict[str, Any]]:
        if split not in self.config.expected_input_sha256:
            raise ValueError(f"Unknown prepared split: {split}")
        if split not in self.requested_splits:
            raise ValueError(f"Prepared split was not requested when opening cache: {split}")
        path = self.config.prepared_root / f"{split}.jsonl"
        if not path.is_file():
            raise OfficialDiscreteCacheError(f"Pinned prepared split does not exist: {path}")
        cursor = self.connection.execute(
            """
            SELECT source_line, sample_id, body_count, framed_count, framed_uint32_le_zlib
            FROM sequences WHERE split = ? ORDER BY source_line
            """,
            (split,),
        )
        rows: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        cache_row = cursor.fetchone()
        with path.open("rb") as handle:
            for source_line, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} input contains blank line {source_line}"
                    )
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} input line {source_line} is invalid: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} input line {source_line} is not an object"
                    )
                if cache_row is None:
                    raise OfficialDiscreteCacheError(
                        f"SQLite cache ended before {split} input line {source_line}"
                    )
                (
                    cached_line,
                    cached_sample_id,
                    body_count,
                    framed_count,
                    framed_blob,
                ) = cache_row
                sample_id = _prepared_record_identity(record)
                if cached_line != source_line or sample_id != cached_sample_id:
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} identity mismatch at line {source_line}: "
                        f"record {sample_id!r}, cache {cached_sample_id!r}"
                    )
                if framed_count != body_count + 2:
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} cache framing count mismatch at line {source_line}"
                    )
                svg = record.get("output_svg")
                if not isinstance(svg, str) or not svg:
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} record {sample_id!r} lacks output_svg"
                    )
                svg_sha256 = hashlib.sha256(svg.encode("utf-8")).hexdigest()
                record_metadata = record.get("metadata")
                canonical_svg_sha256 = (
                    record_metadata.get("canonical_svg_sha256")
                    if isinstance(record_metadata, Mapping)
                    else None
                )
                if canonical_svg_sha256 is not None and canonical_svg_sha256 != svg_sha256:
                    raise OfficialDiscreteCacheError(
                        f"Pinned {split} record {sample_id!r} canonical SVG SHA-256 mismatch"
                    )
                record[CACHED_TARGET_FIELD] = {
                    "backend_id": OFFICIAL_CACHED_GEMMA_BACKEND_ID,
                    "cache_sha256": self.config.expected_cache_sha256,
                    "expected_cache_sha256": self.config.expected_cache_sha256,
                    "input_sha256": self.config.expected_input_sha256[split],
                    "split": split,
                    "source_line": source_line,
                    "sample_id": sample_id,
                    "svg_sha256": svg_sha256,
                    "canonical_svg_sha256": canonical_svg_sha256,
                    "body_count": body_count,
                    "framed_count": framed_count,
                    "framed_uint32_le_zlib": bytes(framed_blob),
                }
                rows.append(record)
                cache_row = cursor.fetchone()
        if cache_row is not None:
            raise OfficialDiscreteCacheError(f"SQLite cache has extra {split} rows")
        observed_hash = digest.hexdigest()
        expected_hash = self.config.expected_input_sha256[split]
        if observed_hash != expected_hash:
            raise OfficialDiscreteCacheError(
                f"Pinned {split} input SHA-256 mismatch: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        expected_count = self.config.expected_split_counts[split]
        if len(rows) != expected_count:
            raise OfficialDiscreteCacheError(
                f"Pinned {split} record count mismatch: expected {expected_count}, observed {len(rows)}"
            )
        return rows

    def load_all(self) -> dict[str, list[dict[str, Any]]]:
        if self.requested_splits != _PREPARED_SPLITS:
            raise ValueError("load_all requires a cache opened for every prepared split.")
        return {split: self.load_split(split) for split in _PREPARED_SPLITS}


def load_all_cached_records(
    config: OpenVGLabCacheConfig,
) -> dict[str, list[dict[str, Any]]]:
    with PinnedOpenVGLabTokenCache(config) as cache:
        return cache.load_all()


def load_cached_splits_records(
    config: OpenVGLabCacheConfig,
    splits: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    requested_splits = _normalize_requested_splits(splits)
    with PinnedOpenVGLabTokenCache(
        config,
        requested_splits=requested_splits,
    ) as cache:
        return {split: cache.load_split(split) for split in requested_splits}


def load_cached_split_records(
    config: OpenVGLabCacheConfig,
    split: str,
) -> list[dict[str, Any]]:
    return load_cached_splits_records(config, (split,))[split]
