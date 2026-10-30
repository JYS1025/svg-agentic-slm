from __future__ import annotations

import json
from pathlib import Path

from scripts.run_critic_distill_pipeline import (
    _append_event,
    _assemble_stable_snapshots,
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
