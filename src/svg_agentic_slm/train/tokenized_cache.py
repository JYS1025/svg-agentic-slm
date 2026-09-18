"""Immutable, fingerprinted tokenization caches with bounded-memory mmap reads.

Only the cache builder calls the original dataset's ``__getitem__``. Each row
stores numeric vectors in little-endian binary form and a small JSON descriptor;
no executable serialization format is used. Readers validate the committed file
hashes before opening read-only mappings. Persisted sampling/provenance snapshots
preserve the original training order without retaining its tokenizer or records.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import shutil
import struct
import sys
import tempfile
import time
from array import array
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA_VERSION = 1
_ROWS_MAGIC = b"SVGTOKR1"
_INDEX_MAGIC = b"SVGTOKI1"
_INDEX_ENTRY = struct.Struct("<QQQ")  # row offset, row byte length, token length
_DESCRIPTOR_SIZE = struct.Struct("<I")
_MAX_DESCRIPTOR_BYTES = 1024 * 1024
_VECTOR_DTYPES = {"b": 1, "i": 4, "q": 8, "d": 8}
_NUMPY_DTYPES = {"b": "i1", "i": "<i4", "q": "<i8", "d": "<f8"}
_REQUIRED_VECTORS = frozenset({"input_ids", "attention_mask", "labels"})
_OPTIONAL_VECTORS = frozenset({"loss_weights", "response_loss_weights", "response_token_roles"})


class TokenizedCacheError(RuntimeError):
    """A cache is incomplete, incompatible, or corrupt; it must not be consumed."""


class TokenizedCacheLockTimeoutError(TokenizedCacheError, TimeoutError):
    """Another process did not release the cache construction lock in time."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _integer_dtype(values: list[int]) -> str:
    minimum, maximum = min(values), max(values)
    if -128 <= minimum and maximum <= 127:
        return "b"
    if -(2**31) <= minimum and maximum < 2**31:
        return "i"
    return "q"


def _encode_row(feature: Any) -> tuple[bytes, int]:
    if not isinstance(feature, Mapping) or not all(isinstance(key, str) for key in feature):
        raise TokenizedCacheError("Tokenized cache rows must be mappings with string keys.")
    if not _REQUIRED_VECTORS.issubset(feature):
        raise TokenizedCacheError("Tokenized cache rows require input_ids, attention_mask, labels.")
    input_ids = feature["input_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise TokenizedCacheError("Tokenized cache input_ids must be a non-empty integer list.")
    token_length = len(input_ids)
    descriptors: list[list[Any]] = []
    vectors: list[bytes] = []
    for key, value in feature.items():
        if key in _REQUIRED_VECTORS | _OPTIONAL_VECTORS:
            if not isinstance(value, list) or len(value) != token_length:
                raise TokenizedCacheError(f"Tokenized cache vector {key!r} has the wrong length.")
            if key in _REQUIRED_VECTORS or key == "response_token_roles":
                if not all(_integer(item) for item in value):
                    raise TokenizedCacheError(f"Tokenized cache vector {key!r} must be integer.")
                dtype = _integer_dtype(value)
            elif all(_integer(item) for item in value):
                dtype = _integer_dtype(value)
            elif all(isinstance(item, float) and math.isfinite(item) for item in value):
                dtype = "d"
            elif all(
                (_integer(item) or isinstance(item, float)) and math.isfinite(item)
                for item in value
            ):
                # Preserve mixed int/float types exactly instead of coercing integers.
                descriptors.append([key, "json", value])
                continue
            else:
                raise TokenizedCacheError(f"Tokenized cache vector {key!r} is not finite numeric.")
            try:
                packed = array(dtype, value)
            except (OverflowError, TypeError) as exc:
                raise TokenizedCacheError(
                    f"Tokenized cache vector {key!r} exceeds its dtype."
                ) from exc
            if packed.itemsize != _VECTOR_DTYPES[dtype]:
                raise TokenizedCacheError("Platform array element width differs from cache format.")
            if sys.byteorder != "little":
                packed.byteswap()
            descriptors.append([key, dtype, token_length])
            vectors.append(packed.tobytes())
        else:
            # Role counts and other small JSON provenance fields retain their values.
            try:
                encoded_extra = _json_bytes(value)
            except (TypeError, ValueError) as exc:
                raise TokenizedCacheError(
                    f"Tokenized cache field {key!r} is not JSON data."
                ) from exc
            if json.loads(encoded_extra) != value:
                raise TokenizedCacheError(f"Tokenized cache field {key!r} is not JSON-preserving.")
            descriptors.append([key, "json", value])
    descriptor = _json_bytes(descriptors)
    if len(descriptor) > _MAX_DESCRIPTOR_BYTES:
        raise TokenizedCacheError("Tokenized cache row descriptor is too large.")
    return _DESCRIPTOR_SIZE.pack(len(descriptor)) + descriptor + b"".join(vectors), token_length


@contextmanager
def _build_lock(path: Path, *, timeout_seconds: float) -> Iterator[None]:
    started = time.monotonic()
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError as exc:
            if time.monotonic() - started >= timeout_seconds:
                raise TokenizedCacheLockTimeoutError(
                    f"Timed out waiting for tokenized cache lock {path}. "
                    "If a builder crashed, remove its stale lock only after confirming it stopped."
                ) from exc
            time.sleep(min(0.1, max(0.001, timeout_seconds - (time.monotonic() - started))))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes({"pid": os.getpid(), "created_unix": time.time()}))
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        path.unlink()


def _build_cache(dataset: Any, destination: Path, fingerprint: dict[str, Any]) -> None:
    row_count = len(dataset)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent))
    try:
        rows_path = temporary / "rows.bin"
        index_path = temporary / "index.bin"
        rows_digest = hashlib.sha256(_ROWS_MAGIC)
        index_digest = hashlib.sha256(_INDEX_MAGIC)
        with rows_path.open("wb") as rows, index_path.open("wb") as index:
            rows.write(_ROWS_MAGIC)
            index.write(_INDEX_MAGIC)
            for row_index in range(row_count):
                encoded, token_length = _encode_row(dataset[row_index])
                entry = _INDEX_ENTRY.pack(rows.tell(), len(encoded), token_length)
                index.write(entry)
                index_digest.update(entry)
                rows.write(encoded)
                rows_digest.update(encoded)
            if len(dataset) != row_count:
                raise TokenizedCacheError("Dataset length changed while building tokenized cache.")
            for handle in (rows, index):
                handle.flush()
                os.fsync(handle.fileno())
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_sha256": hashlib.sha256(_json_bytes(fingerprint)).hexdigest(),
            "row_count": row_count,
            "source_metadata": {
                "verified_lengths": list(dataset.verified_lengths),
                "verified_length_manifest": dataset.verified_length_manifest(),
                "instruction_selection_manifest": dataset.instruction_selection_manifest(),
                "structural_response_role_manifest": dataset.structural_response_role_manifest(),
            },
            "files": {
                "rows.bin": {
                    "size_bytes": rows_path.stat().st_size,
                    "sha256": rows_digest.hexdigest(),
                },
                "index.bin": {
                    "size_bytes": index_path.stat().st_size,
                    "sha256": index_digest.hexdigest(),
                },
            },
        }
        manifest["source_metadata_sha256"] = hashlib.sha256(
            _json_bytes(manifest["source_metadata"])
        ).hexdigest()
        with (temporary / "manifest.json").open("wb") as handle:
            handle.write(_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        # The directory becomes visible only after every file and the manifest exist.
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class TokenizedDatasetCache:
    """Map cached rows, retaining only snapshots of the source sampling contract."""

    def __init__(
        self,
        dataset: Any,
        *,
        path: Path,
        fingerprint: dict[str, Any],
        status: str,
        batched_array_fetch: bool = True,
    ) -> None:
        if not isinstance(batched_array_fetch, bool):
            raise TypeError("Tokenized cache batched_array_fetch must be boolean.")
        self._path = path
        self._status = status
        self._batched_array_fetch = batched_array_fetch
        self._rows_map: mmap.mmap | None = None
        self._index_map: mmap.mmap | None = None
        self._rows_file: Any = None
        self._index_file: Any = None
        try:
            manifest_path = path / "manifest.json"
            if path.is_symlink() or manifest_path.is_symlink():
                raise TokenizedCacheError("Tokenized cache may not use symbolic links.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(_json_bytes(fingerprint)).hexdigest()
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != _SCHEMA_VERSION
                or manifest.get("fingerprint") != fingerprint
                or manifest.get("fingerprint_sha256") != expected_hash
                or not _integer(manifest.get("row_count"))
                or manifest["row_count"] < 0
                or manifest["row_count"] != len(dataset)
            ):
                raise TokenizedCacheError(
                    "Tokenized cache manifest/fingerprint/row count mismatch."
                )
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != {"rows.bin", "index.bin"}:
                raise TokenizedCacheError("Tokenized cache manifest has invalid file entries.")
            for name, entry in files.items():
                file_path = path / name
                if (
                    not isinstance(entry, dict)
                    or file_path.is_symlink()
                    or not file_path.is_file()
                    or not _integer(entry.get("size_bytes"))
                    or file_path.stat().st_size != entry["size_bytes"]
                    or _sha256_file(file_path) != entry.get("sha256")
                ):
                    raise TokenizedCacheError(
                        f"Tokenized cache file {name!r} failed integrity checks."
                    )
            self._manifest = manifest
            self._open_mappings()
            self._runtime_lengths = self._validate_index()
            # Persisted snapshots let a warm cache avoid constructing the original
            # dataset/tokenizer and keep those objects out of spawn worker state.
            metadata = manifest.get("source_metadata")
            if not isinstance(metadata, dict):
                raise TokenizedCacheError("Tokenized cache source metadata is missing.")
            if hashlib.sha256(_json_bytes(metadata)).hexdigest() != manifest.get(
                "source_metadata_sha256"
            ):
                raise TokenizedCacheError(
                    "Tokenized cache source metadata failed integrity checks."
                )
            verified_lengths = metadata.get("verified_lengths")
            if (
                not isinstance(verified_lengths, list)
                or len(verified_lengths) != len(self)
                or any(not _integer(value) or value <= 0 for value in verified_lengths)
                or not isinstance(metadata.get("verified_length_manifest"), dict)
                or not isinstance(metadata.get("instruction_selection_manifest"), dict)
                or not isinstance(
                    metadata.get("structural_response_role_manifest"), (dict, type(None))
                )
            ):
                raise TokenizedCacheError("Tokenized cache source metadata is invalid.")
            self._verified_lengths = tuple(verified_lengths)
            self._verified_length_manifest = metadata["verified_length_manifest"]
            self._instruction_selection_manifest = metadata["instruction_selection_manifest"]
            self._structural_role_manifest = metadata["structural_response_role_manifest"]
        except Exception as exc:
            self.close()
            if isinstance(exc, TokenizedCacheError):
                raise
            raise TokenizedCacheError(
                f"Cannot read complete tokenized cache {path}: {exc}"
            ) from exc

    def _open_mappings(self) -> None:
        if self._rows_map is None:
            try:
                self._rows_file = (self._path / "rows.bin").open("rb")
                self._rows_map = mmap.mmap(self._rows_file.fileno(), 0, access=mmap.ACCESS_READ)
                self._index_file = (self._path / "index.bin").open("rb")
                self._index_map = mmap.mmap(self._index_file.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                self.close()
                raise

    def _validate_index(self) -> tuple[int, ...]:
        rows, index = self._rows_map, self._index_map
        if rows is None or index is None:
            raise TokenizedCacheError("Tokenized cache mappings are unavailable.")
        if (
            rows[: len(_ROWS_MAGIC)] != _ROWS_MAGIC
            or index[: len(_INDEX_MAGIC)] != _INDEX_MAGIC
            or len(index) != len(_INDEX_MAGIC) + len(self) * _INDEX_ENTRY.size
        ):
            raise TokenizedCacheError("Tokenized cache binary header or index size is invalid.")
        next_offset = len(_ROWS_MAGIC)
        lengths: list[int] = []
        for row_index in range(len(self)):
            offset, byte_length, token_length = self._entry(row_index)
            if (
                offset != next_offset
                or byte_length < _DESCRIPTOR_SIZE.size
                or token_length <= 0
                or offset + byte_length > len(rows)
            ):
                raise TokenizedCacheError("Tokenized cache index contains invalid row bounds.")
            # Check descriptors and vector lengths without materializing token vectors.
            self._row_descriptor(offset, byte_length, token_length)
            next_offset += byte_length
            lengths.append(token_length)
        if next_offset != len(rows):
            raise TokenizedCacheError("Tokenized cache contains unindexed trailing data.")
        return tuple(lengths)

    def _entry(self, index: int) -> tuple[int, int, int]:
        if self._index_map is None:
            raise TokenizedCacheError("Tokenized cache index is unavailable.")
        return _INDEX_ENTRY.unpack_from(
            self._index_map, len(_INDEX_MAGIC) + index * _INDEX_ENTRY.size
        )

    def _row_descriptor(
        self, offset: int, byte_length: int, token_length: int
    ) -> tuple[list[list[Any]], int]:
        rows = self._rows_map
        if rows is None:
            raise TokenizedCacheError("Tokenized cache rows are unavailable.")
        descriptor_length = _DESCRIPTOR_SIZE.unpack_from(rows, offset)[0]
        data_start = offset + _DESCRIPTOR_SIZE.size + descriptor_length
        row_end = offset + byte_length
        if not 0 < descriptor_length <= _MAX_DESCRIPTOR_BYTES or data_start > row_end:
            raise TokenizedCacheError("Tokenized cache row descriptor bounds are invalid.")
        descriptors = json.loads(rows[offset + _DESCRIPTOR_SIZE.size : data_start])
        if not isinstance(descriptors, list):
            raise TokenizedCacheError("Tokenized cache row descriptor must be an array.")
        keys: set[str] = set()
        vector_bytes = 0
        for item in descriptors:
            if not isinstance(item, list) or len(item) != 3 or not isinstance(item[0], str):
                raise TokenizedCacheError("Tokenized cache row field descriptor is invalid.")
            key, dtype, value = item
            if key in keys or dtype not in (*_VECTOR_DTYPES, "json"):
                raise TokenizedCacheError("Tokenized cache row field is duplicated or invalid.")
            keys.add(key)
            if dtype != "json":
                if not _integer(value) or value != token_length:
                    raise TokenizedCacheError("Tokenized cache row vector length is invalid.")
                vector_bytes += value * _VECTOR_DTYPES[dtype]
            if key in _REQUIRED_VECTORS and dtype not in ("b", "i", "q"):
                raise TokenizedCacheError("Required tokenized cache vectors must be integer.")
        if not _REQUIRED_VECTORS.issubset(keys) or data_start + vector_bytes != row_end:
            raise TokenizedCacheError("Tokenized cache row schema or vector byte size is invalid.")
        return descriptors, data_start

    def __len__(self) -> int:
        return int(self._manifest["row_count"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._read_row(index, as_arrays=False)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Fetch a DataLoader batch without creating Python scalar lists.

        Numeric arrays own writable storage, independent of the read-only mmap.
        They remain safe after ``close()`` and cannot mutate the cache. Keeping
        this one copy also avoids exporting mmap buffers into spawned workers.
        The scalar ``__getitem__`` API continues returning ordinary lists.
        """
        return [self._read_row(index, as_arrays=self._batched_array_fetch) for index in indices]

    def _read_row(self, index: int, *, as_arrays: bool) -> dict[str, Any]:
        if not _integer(index):
            raise TypeError("Tokenized cache index must be an integer.")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._open_mappings()
        offset, byte_length, token_length = self._entry(index)
        descriptors, cursor = self._row_descriptor(offset, byte_length, token_length)
        feature: dict[str, Any] = {}
        for key, dtype, value in descriptors:
            if dtype == "json":
                feature[key] = value
            else:
                vector_bytes = value * _VECTOR_DTYPES[dtype]
                if as_arrays:
                    # frombuffer is temporary: the returned copy owns its data.
                    # astype normalizes endianness for CPUs with either byte order.
                    feature[key] = np.frombuffer(
                        self._rows_map, dtype=_NUMPY_DTYPES[dtype], count=value, offset=cursor
                    ).astype(np.dtype(_NUMPY_DTYPES[dtype]).newbyteorder("="), copy=True)
                else:
                    packed = array(dtype)
                    packed.frombytes(self._rows_map[cursor : cursor + vector_bytes])
                    if sys.byteorder != "little":
                        packed.byteswap()
                    feature[key] = packed.tolist()
                cursor += vector_bytes
        return feature

    @property
    def runtime_lengths(self) -> tuple[int, ...]:
        return self._runtime_lengths

    @property
    def verified_lengths(self) -> Sequence[int]:
        return self._verified_lengths

    def verified_length_manifest(self) -> dict[str, Any]:
        return json.loads(_json_bytes(self._verified_length_manifest))

    def instruction_selection_manifest(self) -> dict[str, Any]:
        return json.loads(_json_bytes(self._instruction_selection_manifest))

    def structural_response_role_manifest(self) -> dict[str, Any] | None:
        return json.loads(_json_bytes(self._structural_role_manifest))

    def cache_manifest(self) -> dict[str, Any]:
        lengths = self._runtime_lengths
        manifest = {key: value for key, value in self._manifest.items() if key != "source_metadata"}
        return {
            **json.loads(_json_bytes(manifest)),
            "source_metadata_sha256": hashlib.sha256(
                _json_bytes(self._manifest["source_metadata"])
            ).hexdigest(),
            "status": self._status,
            "cache_path": str(self._path),
            "storage": "read_only_mmap_little_endian_int8_int32_int64_float64_json",
            "batched_array_fetch": self._batched_array_fetch,
            "runtime_lengths": {
                "count": len(lengths),
                "minimum": min(lengths, default=0),
                "maximum": max(lengths, default=0),
                "sum": sum(lengths),
                "ordered_values_sha256": hashlib.sha256(_json_bytes(lengths)).hexdigest(),
            },
            "sampling_lengths": "snapshot_of_original_dataset",
        }

    def close(self) -> None:
        for name in ("_rows_map", "_index_map", "_rows_file", "_index_file"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader spawn workers reopen mappings locally; on-disk data stays binary/JSON.
        state = dict(self.__dict__)
        for name in ("_rows_map", "_index_map", "_rows_file", "_index_file"):
            state[name] = None
        return state

    def __del__(self) -> None:
        self.close()


def load_or_build_tokenized_cache(
    dataset: Any,
    *,
    cache_dir: str | Path,
    fingerprint: Mapping[str, Any],
    lock_timeout_seconds: float = 600.0,
    batched_array_fetch: bool = True,
) -> TokenizedDatasetCache:
    """Build once per fingerprint, or verify and reuse an immutable cache.

    The caller must fingerprint every input that affects serialization (source
    bytes, tokenizer/template, prompt, instruction/target policy, length and loss
    configuration, and serialization version). Existing corrupt caches are never
    silently rebuilt. A crashed builder leaves an explicit lock to investigate.
    """
    if not isinstance(batched_array_fetch, bool):
        raise TypeError("Tokenized cache batched_array_fetch must be boolean.")
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        raise ValueError("Tokenized cache fingerprint must be a non-empty mapping.")
    if not all(isinstance(key, str) for key in fingerprint):
        raise ValueError("Tokenized cache fingerprint keys must be strings.")
    try:
        normalized_fingerprint = json.loads(_json_bytes(dict(fingerprint)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Tokenized cache fingerprint must contain finite JSON values.") from exc
    if (
        isinstance(lock_timeout_seconds, bool)
        or not isinstance(lock_timeout_seconds, (int, float))
        or not math.isfinite(lock_timeout_seconds)
        or lock_timeout_seconds < 0
    ):
        raise ValueError("Tokenized cache lock timeout must be finite and non-negative.")
    fingerprint_hash = hashlib.sha256(_json_bytes(normalized_fingerprint)).hexdigest()
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"v{_SCHEMA_VERSION}-{fingerprint_hash}"
    status = "reused"
    if not destination.exists():
        with _build_lock(root / f".{destination.name}.lock", timeout_seconds=lock_timeout_seconds):
            if not destination.exists():
                _build_cache(dataset, destination, normalized_fingerprint)
                status = "built"
    return TokenizedDatasetCache(
        dataset,
        path=destination,
        fingerprint=normalized_fingerprint,
        status=status,
        batched_array_fetch=batched_array_fetch,
    )
