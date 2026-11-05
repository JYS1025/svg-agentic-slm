#!/usr/bin/env python3
"""Run initial, new-only, and full critic-distillation QLoRA stages."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from svg_agentic_slm.train.critic_distill_data import (
    _validate_candidate_hash_split_disjointness,
    prepare_critic_distill_dataset,
)
from svg_agentic_slm.train.critic_similarity_data import precompute_critic_similarity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / ".work/omnisvg_critic_distill/teacher_runs/luna_critic_v3_nonblank"
DEFAULT_SOURCE = ROOT / ".work/omnisvg_critic_distill/runs/full_10k"
DEFAULT_SIMILARITY = ROOT / ".work/omnisvg_critic_distill/similarity/siglip2"
DEFAULT_INITIAL_SNAPSHOT = ROOT / "data/processed/omnisvg_critic_distill"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _append_event(event_log_path: Path, event: str, **fields: Any) -> None:
    value = {"timestamp": _utc_now(), "event": event, **fields}
    with event_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(value, sort_keys=True, default=str), flush=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["sample_id"], row["candidate_svg_sha256"]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _load_snapshot(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: _read_jsonl(path / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }


def _copy_initial_snapshot(source: Path, destination: Path) -> None:
    """Publish the completion manifest only after every copied split is verified."""
    manifest = json.loads((source / "manifest.json").read_text())
    if "similarity_contract" not in manifest:
        raise ValueError("Initial snapshot does not contain production SigLIP2 evidence.")
    splits = ("train", "validation", "test")
    expected_hashes = {split: manifest["split_sha256"][split] for split in splits}
    destination.mkdir(parents=True, exist_ok=True)
    for split, expected in expected_hashes.items():
        temporary = destination / f".{split}.{os.getpid()}.tmp"
        try:
            shutil.copy2(source / f"{split}.jsonl", temporary)
            if _sha256(temporary) != expected:
                raise ValueError(f"Initial snapshot changed while being copied: {split}")
            os.replace(temporary, destination / f"{split}.jsonl")
        finally:
            temporary.unlink(missing_ok=True)
    _atomic_json(destination / "manifest.json", manifest)


def _assemble_stable_snapshots(
    *, initial_dir: Path, final_raw_dir: Path, delta_dir: Path, full_dir: Path
) -> dict[str, Any]:
    initial = _load_snapshot(initial_dir)
    final_raw = _load_snapshot(final_raw_dir)
    initial_keys = {
        _record_key(row) for rows in initial.values() for row in rows
    }
    final_by_key = {
        _record_key(row): row for rows in final_raw.values() for row in rows
    }
    missing = initial_keys - set(final_by_key)
    if missing:
        raise ValueError(f"Final teacher snapshot lost {len(missing)} initial records.")
    new_rows = [
        row
        for key, row in sorted(
            final_by_key.items(), key=lambda item: item[1]["sample_index"]
        )
        if key not in initial_keys
    ]
    initial_train = initial["train"]
    fixed_validation = initial["validation"]
    fixed_test = initial["test"]
    layouts = {
        delta_dir: {
            "train": new_rows,
            "validation": fixed_validation,
            "test": fixed_test,
            "kind": "new_validated_records_only_with_fixed_initial_holdouts",
        },
        full_dir: {
            "train": sorted(
                [*initial_train, *new_rows], key=lambda row: row["sample_index"]
            ),
            "validation": fixed_validation,
            "test": fixed_test,
            "kind": "initial_train_plus_all_new_with_fixed_initial_holdouts",
        },
    }
    # New candidates may belong to a held-out input or repeat a held-out SVG.
    # Check the assembled layouts before writing any completion manifests.
    for layout in layouts.values():
        split_rows = {split: layout[split] for split in ("train", "validation", "test")}
        _validate_candidate_hash_split_disjointness(split_rows)
        owners: dict[str, str] = {}
        for split, rows in split_rows.items():
            for row in rows:
                sample_id = row["sample_id"]
                owner = owners.setdefault(sample_id, split)
                if owner != split:
                    raise ValueError(
                        f"Sample ID crosses dataset split boundaries: {sample_id} "
                        f"({owner}/{split})"
                    )
    manifests = {}
    for destination, layout in layouts.items():
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {
            split: _write_jsonl(destination / f"{split}.jsonl", layout[split])
            for split in ("train", "validation", "test")
        }
        manifest = {
            "schema_version": 1,
            "kind": layout["kind"],
            "initial_snapshot": str(initial_dir),
            "final_raw_snapshot": str(final_raw_dir),
            "counts": {
                split: len(layout[split]) for split in ("train", "validation", "test")
            },
            "pass_counts": {
                split: sum(bool(row["passes_threshold"]) for row in layout[split])
                for split in ("train", "validation", "test")
            },
            "split_sha256": hashes,
        }
        _atomic_json(destination / "manifest.json", manifest)
        manifests[destination.name] = manifest
    train_key_sets = [
        {_record_key(row) for row in initial_train},
        {_record_key(row) for row in new_rows},
    ]
    if train_key_sets[0] & train_key_sets[1]:
        raise RuntimeError("Initial and new-only training cohorts overlap.")
    return {
        "new_records": len(new_rows),
        "initial_train_records": len(initial_train),
        "stable_validation_records": len(fixed_validation),
        "stable_test_records": len(fixed_test),
        "snapshots": manifests,
    }


def _teacher_complete(teacher_dir: Path) -> tuple[bool, dict[str, Any]]:
    progress = json.loads((teacher_dir / "progress.json").read_text())
    complete = bool(
        progress.get("source_run_finished")
        and int(progress.get("awaiting_teacher", -1)) == 0
    )
    return complete, progress


def _wait_for_teacher(
    teacher_dir: Path, events_path: Path, *, poll_seconds: float
) -> dict[str, Any]:
    last_awaiting = None
    while True:
        complete, progress = _teacher_complete(teacher_dir)
        awaiting = progress.get("awaiting_teacher")
        if awaiting != last_awaiting:
            _append_event(
                events_path,
                "teacher_wait_progress",
                awaiting_teacher=awaiting,
                validated=progress.get("validated"),
                skipped_blank_candidate=progress.get("skipped_blank_candidate"),
                halted_reason=progress.get("halted_reason"),
            )
            last_awaiting = awaiting
        if complete:
            return progress
        time.sleep(poll_seconds)


def _stage_config(
    *,
    base_config: Path,
    pipeline_id: str,
    stage_id: str,
    stage_kind: str,
    parent_stage_id: str | None,
    snapshot_dir: Path,
    output_dir: Path,
    epochs: float,
    init_adapter_from: Path | None,
) -> dict[str, Any]:
    config = yaml.safe_load(base_config.read_text())
    train = config["train"]
    train["run"] = {
        "pipeline_id": pipeline_id,
        "stage_id": stage_id,
        "stage_kind": stage_kind,
        "parent_stage_id": parent_stage_id,
    }
    train["dataset"] = {
        "train_path": str(snapshot_dir / "train.jsonl"),
        "validation_path": str(snapshot_dir / "validation.jsonl"),
        "test_path": str(snapshot_dir / "test.jsonl"),
    }
    train["sft"].update(
        {
            "output_dir": str(output_dir),
            "num_train_epochs": epochs,
            "max_steps": -1,
            "early_stopping_patience": None,
            "resume_from_checkpoint": None,
            "init_adapter_from": (
                str(init_adapter_from) if init_adapter_from is not None else None
            ),
        }
    )
    return config


def _latest_complete_checkpoint(output_dir: Path) -> Path | None:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for checkpoint in checkpoints_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required = (
            checkpoint / "trainer_state.json",
            checkpoint / "adapter_config.json",
            checkpoint / "adapter_model.safetensors",
            checkpoint / "optimizer.pt",
            checkpoint / "scheduler.pt",
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            continue
        if not any(checkpoint.glob("rng_state_*.pth")):
            continue
        try:
            trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if int(trainer_state.get("global_step", -1)) != step:
            continue
        candidates.append((step, checkpoint.resolve()))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _resume_compatible_optimizer(optimizer_name: str) -> str:
    # bitsandbytes paged optimizer state cannot be restored reliably with DDP:
    # the first optimizer step exits in its C++ UVM interface. The non-paged
    # AdamW8bit implementation uses the same serialized 8-bit state layout.
    if optimizer_name == "paged_adamw_8bit":
        return "adamw_bnb_8bit"
    return optimizer_name


def _launch_stage(
    *,
    stage_dir: Path,
    config: dict[str, Any],
    num_processes: int,
    events_path: Path,
) -> Path:
    output_dir = Path(config["train"]["sft"]["output_dir"])
    manifest_path = output_dir / "training_manifest.json"
    stage_id = config["train"]["run"]["stage_id"]
    if manifest_path.exists():
        _append_event(events_path, "stage_reused", stage_id=stage_id)
        return output_dir / "adapter"
    resume_checkpoint = _latest_complete_checkpoint(output_dir)
    config["train"]["sft"]["resume_from_checkpoint"] = (
        str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    if resume_checkpoint is not None:
        configured_optimizer = str(config["train"]["sft"].get("optim", ""))
        resume_optimizer = _resume_compatible_optimizer(configured_optimizer)
        config["train"]["sft"]["optim"] = resume_optimizer
        _append_event(
            events_path,
            "stage_checkpoint_resume_selected",
            stage_id=stage_id,
            checkpoint=str(resume_checkpoint),
            configured_optimizer=configured_optimizer,
            resume_optimizer=resume_optimizer,
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    config_path = stage_dir / "stage_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [
        str(ROOT / ".venv/bin/accelerate"),
        "launch",
        "--num_processes",
        str(num_processes),
    ]
    if num_processes > 1:
        command.append("--multi_gpu")
    command.extend(
        [
            "--module",
            "svg_agentic_slm.train.train_critic_distill",
            "--config",
            str(config_path),
        ]
    )
    _append_event(
        events_path,
        "stage_started",
        stage_id=stage_id,
        command=command,
        config_sha256=_sha256(config_path),
    )
    console_path = stage_dir / "console.log"
    with console_path.open("a", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            console.write(line)
            console.flush()
        return_code = process.wait()
    if return_code:
        _append_event(
            events_path, "stage_failed", stage_id=stage_id, return_code=return_code
        )
        raise RuntimeError(f"Critic SFT stage failed: {stage_id} ({return_code})")
    _append_event(events_path, "stage_completed", stage_id=stage_id)
    return output_dir / "adapter"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = args.pipeline_dir.resolve()
    pipeline_root.mkdir(parents=True, exist_ok=True)
    pipeline_id = pipeline_root.name
    events_path = pipeline_root / "pipeline_events.jsonl"
    lock_path = pipeline_root / "pipeline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns pipeline {pipeline_root}") from exc
        manifest = {
            "schema_version": 1,
            "pipeline_id": pipeline_id,
            "status": "running",
            "phase": "initial_snapshot_and_stage",
            "created_or_resumed_at": _utc_now(),
            "base_config": str(args.config.resolve()),
            "teacher_run": str(args.teacher_run.resolve()),
            "similarity_dir": str(args.similarity_dir.resolve()),
            "initial_snapshot_source": str(args.initial_snapshot.resolve()),
            "final_epochs": args.final_epochs,
            "num_processes": args.num_processes,
            "stage_order": [
                "01_initial_1epoch",
                "02_new_only_1epoch",
                "03_full_Nepoch",
            ],
        }
        _atomic_json(pipeline_root / "pipeline_manifest.json", manifest)
        _append_event(
            events_path,
            "pipeline_started_or_resumed",
            pipeline_id=pipeline_id,
            pipeline_root=str(pipeline_root),
        )
        snapshots = pipeline_root / "snapshots"
        stages = pipeline_root / "stages"
        initial = snapshots / "01_initial"
        final_raw = snapshots / "final_raw"
        delta = snapshots / "02_new_only"
        full = snapshots / "03_full_stable"

        if not (initial / "manifest.json").exists():
            source_snapshot = args.initial_snapshot.resolve()
            if (source_snapshot / "manifest.json").exists():
                _copy_initial_snapshot(source_snapshot, initial)
                _append_event(
                    events_path,
                    "initial_snapshot_copied",
                    source=str(source_snapshot),
                    path=str(initial),
                )
            else:
                _append_event(events_path, "initial_similarity_started")
                precompute_critic_similarity(
                    teacher_run=args.teacher_run,
                    output_dir=args.similarity_dir,
                    device=args.similarity_device,
                    dtype=args.similarity_dtype,
                    batch_size=args.similarity_batch_size,
                    local_files_only=True,
                )
                prepare_critic_distill_dataset(
                    source_run=args.source_run,
                    teacher_run=args.teacher_run,
                    similarity_scores_path=args.similarity_dir / "scores.jsonl",
                    output_dir=initial,
                    seed=42,
                )
            _append_event(events_path, "initial_snapshot_created", path=str(initial))
        stage1_dir = stages / "01_initial_1epoch"
        stage1_output = stage1_dir / "output"
        stage1_adapter = _launch_stage(
            stage_dir=stage1_dir,
            config=_stage_config(
                base_config=args.config,
                pipeline_id=pipeline_id,
                stage_id="01_initial_1epoch",
                stage_kind="initial_snapshot_one_epoch",
                parent_stage_id=None,
                snapshot_dir=initial,
                output_dir=stage1_output,
                epochs=1.0,
                init_adapter_from=None,
            ),
            num_processes=args.num_processes,
            events_path=events_path,
        )

        manifest["phase"] = "waiting_for_teacher_completion"
        _atomic_json(pipeline_root / "pipeline_manifest.json", manifest)
        _append_event(events_path, "teacher_wait_started")
        teacher_final = _wait_for_teacher(
            args.teacher_run, events_path, poll_seconds=args.poll_seconds
        )
        _append_event(events_path, "teacher_complete", progress=teacher_final)

        manifest["phase"] = "final_similarity_and_snapshot"
        _atomic_json(pipeline_root / "pipeline_manifest.json", manifest)
        if not (full / "manifest.json").exists():
            _append_event(events_path, "final_similarity_started")
            precompute_critic_similarity(
                teacher_run=args.teacher_run,
                output_dir=args.similarity_dir,
                device=args.similarity_device,
                dtype=args.similarity_dtype,
                batch_size=args.similarity_batch_size,
                local_files_only=True,
            )
            prepare_critic_distill_dataset(
                source_run=args.source_run,
                teacher_run=args.teacher_run,
                similarity_scores_path=args.similarity_dir / "scores.jsonl",
                output_dir=final_raw,
                seed=42,
            )
            assembly = _assemble_stable_snapshots(
                initial_dir=initial,
                final_raw_dir=final_raw,
                delta_dir=delta,
                full_dir=full,
            )
            _append_event(events_path, "stable_snapshots_created", **assembly)
        delta_manifest = json.loads((delta / "manifest.json").read_text())
        parent_adapter = stage1_adapter
        parent_stage = "01_initial_1epoch"
        if int(delta_manifest["counts"]["train"]) > 0:
            manifest["phase"] = "new_only_stage"
            _atomic_json(pipeline_root / "pipeline_manifest.json", manifest)
            stage2_dir = stages / "02_new_only_1epoch"
            parent_adapter = _launch_stage(
                stage_dir=stage2_dir,
                config=_stage_config(
                    base_config=args.config,
                    pipeline_id=pipeline_id,
                    stage_id="02_new_only_1epoch",
                    stage_kind="new_records_only_one_epoch",
                    parent_stage_id=parent_stage,
                    snapshot_dir=delta,
                    output_dir=stage2_dir / "output",
                    epochs=1.0,
                    init_adapter_from=stage1_adapter,
                ),
                num_processes=args.num_processes,
                events_path=events_path,
            )
            parent_stage = "02_new_only_1epoch"
        else:
            _append_event(events_path, "stage_skipped_empty_delta", stage_id="02_new_only_1epoch")

        stage3_dir = stages / "03_full_Nepoch"
        manifest["phase"] = "full_dataset_stage"
        _atomic_json(pipeline_root / "pipeline_manifest.json", manifest)
        final_adapter = _launch_stage(
            stage_dir=stage3_dir,
            config=_stage_config(
                base_config=args.config,
                pipeline_id=pipeline_id,
                stage_id="03_full_Nepoch",
                stage_kind="full_stable_dataset",
                parent_stage_id=parent_stage,
                snapshot_dir=full,
                output_dir=stage3_dir / "output",
                epochs=float(args.final_epochs),
                init_adapter_from=parent_adapter,
            ),
            num_processes=args.num_processes,
            events_path=events_path,
        )
        manifest.update(
            {
                "status": "completed",
                "phase": "completed",
                "completed_at": _utc_now(),
                "final_adapter": str(final_adapter),
                "fixed_holdout_snapshot": str(initial),
            }
        )
        _atomic_json(pipeline_root / "pipeline_manifest.json", manifest)
        _append_event(events_path, "pipeline_completed", final_adapter=str(final_adapter))
        return manifest


def main() -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_critic_distill.yaml")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--teacher-run", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--similarity-dir", type=Path, default=DEFAULT_SIMILARITY)
    parser.add_argument(
        "--initial-snapshot", type=Path, default=DEFAULT_INITIAL_SNAPSHOT
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=ROOT / f"outputs/sft/critic_distill_pipeline_{timestamp}",
    )
    parser.add_argument("--final-epochs", type=float, default=3.0)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--similarity-device", default="cuda:0")
    parser.add_argument("--similarity-dtype", default="bfloat16")
    parser.add_argument("--similarity-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.final_epochs <= 0 or args.num_processes <= 0 or args.poll_seconds <= 0:
        parser.error("epochs, process count, and polling interval must be positive")
    try:
        result = run_pipeline(args)
    except Exception as exc:
        manifest_path = args.pipeline_dir.resolve() / "pipeline_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _atomic_json(manifest_path, manifest)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
