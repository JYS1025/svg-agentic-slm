from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_critic_distill_pipeline import (
    _append_event,
    _assemble_stable_snapshots,
    _copy_initial_snapshot,
    _latest_complete_checkpoint,
    _resume_compatible_optimizer,
)


def _row(index: int, *, passes: bool = False) -> dict[str, object]:
    return {
        "sample_id": f"icon:{index}",
        "candidate_svg_sha256": f"{index:064x}",
        "sample_index": index,
        "passes_threshold": passes,
    }


def _write_snapshot(path: Path, splits: dict[str, list[dict[str, object]]]) -> None:
    path.mkdir(parents=True)
    for split, rows in splits.items():
        (path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_append_event_allows_path_metadata(tmp_path: Path) -> None:
    event_log = tmp_path / "events.jsonl"

    _append_event(event_log, "snapshot_created", path="/tmp/snapshot")

    event = json.loads(event_log.read_text())
    assert event["event"] == "snapshot_created"
    assert event["path"] == "/tmp/snapshot"


def _write_checkpoint(path: Path, step: int, *, complete: bool = True) -> None:
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_0.pth",
    ):
        (path / name).write_bytes(b"checkpoint")
    if not complete:
        (path / "optimizer.pt").unlink()


def test_latest_complete_checkpoint_skips_incomplete_and_mismatched_steps(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    _write_checkpoint(checkpoints / "checkpoint-50", 50)
    _write_checkpoint(checkpoints / "checkpoint-75", 75, complete=False)
    _write_checkpoint(checkpoints / "checkpoint-100", 99)

    assert _latest_complete_checkpoint(tmp_path) == (
        checkpoints / "checkpoint-50"
    ).resolve()


def test_paged_adamw_8bit_uses_nonpaged_compatible_optimizer_on_resume() -> None:
    assert _resume_compatible_optimizer("paged_adamw_8bit") == "adamw_bnb_8bit"
    assert _resume_compatible_optimizer("adamw_torch") == "adamw_torch"


def test_staged_snapshots_train_only_on_new_rows_then_full_with_fixed_holdouts(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial"
    final_raw = tmp_path / "final-raw"
    delta = tmp_path / "delta"
    full = tmp_path / "full"
    _write_snapshot(
        initial,
        {
            "train": [_row(1), _row(2)],
            "validation": [_row(3, passes=True)],
            "test": [_row(4, passes=True)],
        },
    )
    # The raw final split may move old rows; stable assembly must ignore that movement.
    _write_snapshot(
        final_raw,
        {
            "train": [_row(1), _row(3, passes=True), _row(5), _row(6)],
            "validation": [_row(2)],
            "test": [_row(4, passes=True)],
        },
    )

    result = _assemble_stable_snapshots(
        initial_dir=initial,
        final_raw_dir=final_raw,
        delta_dir=delta,
        full_dir=full,
    )
    delta_train = [json.loads(line) for line in (delta / "train.jsonl").read_text().splitlines()]
    full_train = [json.loads(line) for line in (full / "train.jsonl").read_text().splitlines()]
    stable_validation = [
        json.loads(line) for line in (full / "validation.jsonl").read_text().splitlines()
    ]
    stable_test = [json.loads(line) for line in (full / "test.jsonl").read_text().splitlines()]

    assert [row["sample_index"] for row in delta_train] == [5, 6]
    assert [row["sample_index"] for row in full_train] == [1, 2, 5, 6]
    assert [row["sample_index"] for row in stable_validation] == [3]
    assert [row["sample_index"] for row in stable_test] == [4]
    assert result["new_records"] == 2


@pytest.mark.parametrize("overlap", ["candidate_hash", "sample_id"])
def test_staged_snapshots_reject_new_rows_overlapping_fixed_holdouts(
    tmp_path: Path, overlap: str
) -> None:
    held_out = _row(3)
    new_row = _row(5)
    key = "candidate_svg_sha256" if overlap == "candidate_hash" else "sample_id"
    new_row[key] = held_out[key]
    initial = {"train": [_row(1)], "validation": [held_out], "test": [_row(4)]}
    final = {**initial, "train": [_row(1), new_row]}
    _write_snapshot(tmp_path / "initial", initial)
    _write_snapshot(tmp_path / "final", final)

    with pytest.raises(ValueError, match="split boundaries"):
        _assemble_stable_snapshots(
            initial_dir=tmp_path / "initial",
            final_raw_dir=tmp_path / "final",
            delta_dir=tmp_path / "delta",
            full_dir=tmp_path / "full",
        )

    assert not (tmp_path / "delta").exists()
    assert not (tmp_path / "full").exists()


def test_staged_snapshots_allow_new_copy_of_training_candidate(tmp_path: Path) -> None:
    new_row = {**_row(5), "candidate_svg_sha256": _row(1)["candidate_svg_sha256"]}
    initial = {"train": [_row(1)], "validation": [_row(3)], "test": [_row(4)]}
    _write_snapshot(tmp_path / "initial", initial)
    _write_snapshot(tmp_path / "final", {**initial, "train": [_row(1), new_row]})

    result = _assemble_stable_snapshots(
        initial_dir=tmp_path / "initial",
        final_raw_dir=tmp_path / "final",
        delta_dir=tmp_path / "delta",
        full_dir=tmp_path / "full",
    )

    assert result["new_records"] == 1
    assert result["snapshots"]["full"]["counts"]["train"] == 2


def _snapshot_with_manifest(path: Path) -> dict[str, object]:
    _write_snapshot(path, {"train": [_row(1)], "validation": [_row(2)], "test": [_row(3)]})
    manifest = {
        "similarity_contract": {"metric": "siglip2_pair_probability"},
        "split_sha256": {
            split: hashlib.sha256((path / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in ("train", "validation", "test")
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_initial_snapshot_hash_failure_can_be_retried(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "initial"
    manifest = _snapshot_with_manifest(source)
    original = (source / "validation.jsonl").read_bytes()
    (source / "validation.jsonl").write_text("corrupt\n")

    with pytest.raises(ValueError, match="changed while being copied: validation"):
        _copy_initial_snapshot(source, destination)

    assert not (destination / "manifest.json").exists()
    assert not list(destination.glob("*.tmp"))
    (source / "validation.jsonl").write_bytes(original)
    _copy_initial_snapshot(source, destination)
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    for split in ("train", "validation", "test"):
        assert (destination / f"{split}.jsonl").read_bytes() == (
            source / f"{split}.jsonl"
        ).read_bytes()


def test_initial_snapshot_resumes_interrupted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    source = tmp_path / "source"
    destination = tmp_path / "initial"
    _snapshot_with_manifest(source)
    real_copy = shutil.copy2

    def interrupt_copy(source_path: Path, destination_path: Path) -> None:
        real_copy(source_path, destination_path)
        if source_path.name == "validation.jsonl":
            raise OSError("interrupted")

    with monkeypatch.context() as context:
        context.setattr("scripts.run_critic_distill_pipeline.shutil.copy2", interrupt_copy)
        with pytest.raises(OSError, match="interrupted"):
            _copy_initial_snapshot(source, destination)

    assert not (destination / "manifest.json").exists()
    _copy_initial_snapshot(source, destination)
    assert (destination / "manifest.json").is_file()
