from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import struct
import weakref
from pathlib import Path
from typing import Any

import pytest

from svg_agentic_slm.train.tokenized_cache import (
    TokenizedCacheError,
    TokenizedCacheLockTimeoutError,
    TokenizedDatasetCache,
    load_or_build_tokenized_cache,
)


class _Dataset:
    def __init__(self, rows: list[dict[str, Any]], *, fail_at: int | None = None) -> None:
        self.rows = rows
        self.calls: list[int] = []
        self.fail_at = fail_at
        self.verified_lengths = tuple(100 + index for index in range(len(rows)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.calls.append(index)
        if self.fail_at == index:
            raise ValueError("serialization failure")
        return copy.deepcopy(self.rows[index])

    def verified_length_manifest(self) -> dict[str, Any]:
        return {"source": "prepared_metadata", "count": len(self)}

    def instruction_selection_manifest(self) -> dict[str, Any]:
        return {"instruction_mode": "detail_only", "selected_instruction_count": len(self)}

    def structural_response_role_manifest(self) -> dict[str, Any]:
        return {"sample_count": len(self)}


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "input_ids": [10, 300001, 306999],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 300001, 306999],
        },
        {
            "input_ids": [7, 8, 9, 10],
            "attention_mask": [1, 1, 1, 1],
            "labels": [-100, -100, 9, 10],
            "loss_weights": [1.0, 1.0, 1.5, 8.0],
            "response_token_roles": [0, 0, 2, 1],
            "response_token_role_counts": {"eos_count": 1, "labeled_token_count": 2},
        },
    ]


def _load(dataset: _Dataset, cache_dir: Path, *, revision: str = "a") -> TokenizedDatasetCache:
    return load_or_build_tokenized_cache(
        dataset,
        cache_dir=cache_dir,
        fingerprint={"tokenizer_revision": revision, "source_sha256": "fixture"},
    )


def test_cache_preserves_rows_and_sampling_without_retokenizing(tmp_path: Path) -> None:
    original = _rows()
    source = _Dataset(original)
    cache = _load(source, tmp_path)
    try:
        assert source.calls == [0, 1]
        assert cache.cache_manifest()["status"] == "built"
        for _ in range(3):
            assert [cache[index] for index in range(len(cache))] == original
        assert source.calls == [0, 1]
        assert cache.runtime_lengths == (3, 4)
        assert cache.verified_lengths == (100, 101)
        assert cache.verified_length_manifest() == source.verified_length_manifest()
        assert cache.instruction_selection_manifest() == source.instruction_selection_manifest()
        assert (
            cache.structural_response_role_manifest() == source.structural_response_role_manifest()
        )
        assert cache[-1] == original[-1]
        mutated = cache[0]
        mutated["input_ids"][0] = 999
        assert cache[0] == original[0]
        with pytest.raises(IndexError):
            cache[2]
        with pytest.raises(TypeError):
            cache[True]
    finally:
        cache.close()

    reused_source = _Dataset(original, fail_at=0)
    reused = _load(reused_source, tmp_path)
    try:
        assert reused.cache_manifest()["status"] == "reused"
        assert reused[0] == original[0]
        assert reused_source.calls == []
    finally:
        reused.close()


def test_cache_preserves_mixed_weight_types_and_worker_reopen(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["response_loss_weights"] = [1.0, 2, 3.5]
    cache = _load(_Dataset(rows), tmp_path)
    worker = object.__new__(TokenizedDatasetCache)
    worker.__dict__.update(cache.__getstate__())
    cache.close()
    try:
        weights = worker[0]["response_loss_weights"]
        assert weights == rows[0]["response_loss_weights"]
        assert [type(value) for value in weights] == [float, int, float]
        assert worker[1] == rows[1]
    finally:
        worker.close()


def test_batched_fetch_preserves_order_and_owns_storage_after_cache_close(tmp_path: Path) -> None:
    import numpy as np

    source = _Dataset(_rows())
    cache = _load(source, tmp_path)
    batch = cache.__getitems__([1, 0, 1, -1])
    assert cache.__getitems__([]) == []
    assert isinstance(cache[0]["input_ids"], list)
    assert cache.cache_manifest()["batched_array_fetch"] is True
    cache.close()
    for row, original in zip(batch, [_rows()[1], _rows()[0], _rows()[1], _rows()[1]]):
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                assert value.flags.owndata and value.flags.writeable and value.dtype.isnative
                assert value.tolist() == original[key]
            else:
                assert value == original[key]
    batch[0]["input_ids"][0] = 99
    assert batch[2]["input_ids"][0] == _rows()[1]["input_ids"][0]
    assert cache[1] == _rows()[1]
    cache.close()
    assert source.calls == [0, 1]


def test_batched_fetch_opt_out_keeps_list_api_and_reuses_identical_cache(tmp_path: Path) -> None:
    cache = _load(_Dataset(_rows()), tmp_path)
    cache_path = cache.cache_manifest()["cache_path"]
    cache.close()
    cache = load_or_build_tokenized_cache(
        _Dataset(_rows(), fail_at=0),
        cache_dir=tmp_path,
        fingerprint={"tokenizer_revision": "a", "source_sha256": "fixture"},
        batched_array_fetch=False,
    )
    try:
        assert cache.__getitems__([1, 0]) == [_rows()[1], _rows()[0]]
        assert cache.cache_manifest()["cache_path"] == cache_path
        assert cache.cache_manifest()["batched_array_fetch"] is False
    finally:
        cache.close()


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_batched_fetch_flag_requires_boolean(tmp_path: Path, invalid: Any) -> None:
    with pytest.raises(TypeError, match="batched_array_fetch"):
        load_or_build_tokenized_cache(
            _Dataset(_rows()), cache_dir=tmp_path, fingerprint={"v": 1}, batched_array_fetch=invalid
        )


def test_warm_cache_needs_only_source_count_and_retains_no_dataset(tmp_path: Path) -> None:
    source = _Dataset(_rows())
    reference = weakref.ref(source)
    cache = _load(source, tmp_path)
    del source
    assert reference() is None
    cache.close()

    class LengthOnlyDataset:
        def __len__(self) -> int:
            return 2

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"Warm cache must not request source metadata: {name}")

    warm = _load(LengthOnlyDataset(), tmp_path)  # type: ignore[arg-type]
    try:
        assert warm[1] == _rows()[1]
        assert warm.verified_lengths == (100, 101)
        assert warm.instruction_selection_manifest()["instruction_mode"] == "detail_only"
    finally:
        warm.close()


def test_fingerprint_changes_isolate_caches_and_count_mismatch_is_rejected(tmp_path: Path) -> None:
    source = _Dataset(_rows())
    first = _load(source, tmp_path, revision="a")
    second = _load(source, tmp_path, revision="b")
    try:
        assert first.cache_manifest()["cache_path"] != second.cache_manifest()["cache_path"]
        assert source.calls == [0, 1, 0, 1]
    finally:
        first.close()
        second.close()
    with pytest.raises(TokenizedCacheError, match="row count mismatch"):
        _load(_Dataset(_rows()[:1]), tmp_path)


@pytest.mark.parametrize(
    "corruption", ["truncate", "flip", "missing_manifest", "bad_index", "source_metadata"]
)
def test_corrupt_or_incomplete_cache_is_not_silently_rebuilt(
    tmp_path: Path, corruption: str
) -> None:
    cache = _load(_Dataset(_rows()), tmp_path)
    path = Path(cache.cache_manifest()["cache_path"])
    cache.close()
    if corruption == "missing_manifest":
        (path / "manifest.json").unlink()
    elif corruption == "source_metadata":
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_metadata"]["verified_lengths"][0] += 1
        manifest_path.write_text(json.dumps(manifest))
    elif corruption == "bad_index":
        # Even if hashes agree, a malformed offset must not become an mmap read.
        index_path = path / "index.bin"
        data = bytearray(index_path.read_bytes())
        struct.pack_into("<Q", data, 8, 2**63)
        index_path.write_bytes(data)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["index.bin"]["sha256"] = hashlib.sha256(data).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    else:
        rows_path = path / "rows.bin"
        data = bytearray(rows_path.read_bytes())
        if corruption == "truncate":
            del data[-8:]
        else:
            data[-1] ^= 1
        rows_path.write_bytes(data)
    source = _Dataset(_rows())
    with pytest.raises(TokenizedCacheError):
        _load(source, tmp_path)
    assert source.calls == []


def test_failed_build_releases_lock_and_never_publishes_partial_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="serialization failure"):
        _load(_Dataset(_rows(), fail_at=1), tmp_path)
    assert list(tmp_path.iterdir()) == []
    cache = _load(_Dataset(_rows()), tmp_path)
    try:
        assert cache.cache_manifest()["status"] == "built"
        assert cache[1] == _rows()[1]
    finally:
        cache.close()


@pytest.mark.parametrize(
    "bad_feature",
    [
        {"input_ids": [1], "attention_mask": [1]},
        {"input_ids": [True], "attention_mask": [1], "labels": [-100]},
        {"input_ids": [1], "attention_mask": [1, 1], "labels": [-100]},
        {"input_ids": [1], "attention_mask": [1], "labels": [-100], "loss_weights": [float("nan")]},
    ],
)
def test_invalid_rows_do_not_produce_a_cache(tmp_path: Path, bad_feature: dict[str, Any]) -> None:
    with pytest.raises(TokenizedCacheError):
        _load(_Dataset([bad_feature]), tmp_path)
    assert list(tmp_path.iterdir()) == []


class _LoggedDataset(_Dataset):
    def __init__(self, log_path: str) -> None:
        super().__init__(_rows())
        self.log_path = log_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}:{index}\n")
        return super().__getitem__(index)


def _cache_process(cache_dir: str, log_path: str, start: Any, queue: Any) -> None:
    start.wait(20)
    try:
        cache = _load(_LoggedDataset(log_path), Path(cache_dir))
        try:
            queue.put((cache.cache_manifest()["status"], cache[1] == _rows()[1]))
        finally:
            cache.close()
    except Exception as exc:
        queue.put(("error", repr(exc)))


def test_concurrent_processes_build_once_and_read_identical_rows(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    start = context.Event()
    log_path = tmp_path / "calls.txt"
    processes = [
        context.Process(
            target=_cache_process,
            args=(str(tmp_path / "cache"), str(log_path), start, queue),
        )
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        results = [queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert sorted(results) == [("built", True), ("reused", True), ("reused", True)]
        calls = log_path.read_text().splitlines()
        assert len(calls) == 2
        assert len({call.split(":")[0] for call in calls}) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()
        queue.join_thread()


def test_existing_builder_lock_times_out_without_modifying_it(tmp_path: Path) -> None:
    fingerprint = {"source": "locked"}
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()
    lock = tmp_path / f".v1-{key}.lock"
    lock.write_text("existing owner")
    with pytest.raises(TokenizedCacheLockTimeoutError, match="Timed out"):
        load_or_build_tokenized_cache(
            _Dataset(_rows()),
            cache_dir=tmp_path,
            fingerprint=fingerprint,
            lock_timeout_seconds=0.02,
        )
    assert lock.read_text() == "existing owner"
    assert list(tmp_path.iterdir()) == [lock]
