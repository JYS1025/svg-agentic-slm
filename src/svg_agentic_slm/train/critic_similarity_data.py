"""Durably precompute production SigLIP2 evidence for critic distillation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from svg_agentic_slm.models.siglip2_similarity import (
    DEFAULT_SIGLIP2_MODEL_ID,
    DEFAULT_SIGLIP2_MODEL_REVISION,
    DEFAULT_SIGLIP2_TEXT_TEMPLATE,
    Siglip2SimilarityScorer,
)
from svg_agentic_slm.train.critic_distill_data import _jsonl_snapshot, _latest_by


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def precompute_critic_similarity(
    *,
    teacher_run: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    batch_size: int = 64,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Score the complete validated-teacher snapshot that exists at invocation time."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    teacher_dir = Path(teacher_run).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    teacher_manifest = json.loads((teacher_dir / "run_manifest.json").read_text())
    source_dir = Path(teacher_manifest["identity"]["source_run"]).resolve()
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text())
    input_path = Path(source_manifest["settings"]["input_manifest"]).resolve()
    input_rows, input_sha = _jsonl_snapshot(input_path)
    inputs = _latest_by(input_rows, lambda row: row["sample_id"])
    teacher_rows, teacher_snapshot_sha = _jsonl_snapshot(teacher_dir / "results.jsonl")
    teacher_latest = _latest_by(
        teacher_rows, lambda row: (row["sample_id"], row["candidate_svg_sha256"])
    )
    selected = sorted(
        (row for row in teacher_latest.values() if row.get("status") == "validated"),
        key=lambda row: (row.get("sample_index", 0), row["candidate_svg_sha256"]),
    )

    scores_path = destination / "scores.jsonl"
    existing_rows, _ = _jsonl_snapshot(scores_path) if scores_path.exists() else ([], None)
    existing = _latest_by(
        existing_rows, lambda row: (row["sample_id"], row["candidate_svg_sha256"])
    )
    pending = [
        row
        for row in selected
        if (row["sample_id"], row["candidate_svg_sha256"]) not in existing
    ]
    identity = {
        "schema_version": 1,
        "metric": "siglip2_pair_probability",
        "model_id": DEFAULT_SIGLIP2_MODEL_ID,
        "model_revision": DEFAULT_SIGLIP2_MODEL_REVISION,
        "text_template": DEFAULT_SIGLIP2_TEXT_TEMPLATE,
        "inference_dtype": dtype,
        "teacher_run": str(teacher_dir),
        "source_input_path": str(input_path),
        "source_input_sha256": input_sha,
    }
    manifest_path = destination / "manifest.json"
    lock_path = destination / "run.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another SigLIP2 scorer owns {destination}") from exc
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("identity") != identity:
                raise ValueError("SigLIP2 score-store identity changed; use a new directory.")
        else:
            _atomic_json(manifest_path, {"identity": identity})
        if pending:
            scorer = Siglip2SimilarityScorer(
                model_id=DEFAULT_SIGLIP2_MODEL_ID,
                model_revision=DEFAULT_SIGLIP2_MODEL_REVISION,
                device=device,
                dtype=dtype,
                local_files_only=local_files_only,
                trust_remote_code=False,
                text_template=DEFAULT_SIGLIP2_TEXT_TEMPLATE,
                max_pair_batch_size=batch_size,
            )
            scorer.load_model()
            try:
                for offset in range(0, len(pending), batch_size):
                    rows = pending[offset : offset + batch_size]
                    instructions: list[str] = []
                    image_bytes: list[bytes] = []
                    attempt_ids: list[str] = []
                    for row in rows:
                        record = inputs.get(row["sample_id"])
                        if record is None:
                            raise ValueError(f"Missing source instruction: {row['sample_id']}")
                        image_path = teacher_dir / row["request_dir"] / "candidate.rendered.png"
                        if _sha256_file(image_path) != row["candidate_rendered_png_sha256"]:
                            raise ValueError(f"Candidate render hash mismatch: {image_path}")
                        instructions.append(record["instruction"])
                        image_bytes.append(image_path.read_bytes())
                        attempt_ids.append(row["sample_id"])
                    evidence = scorer.score_pairs_batch(
                        instructions,
                        image_bytes,
                        attempt_ids=attempt_ids,
                    )
                    output_rows = []
                    for teacher, item in zip(rows, evidence, strict=True):
                        output_rows.append(
                            {
                                "schema_version": 1,
                                "sample_id": teacher["sample_id"],
                                "sample_index": teacher["sample_index"],
                                "candidate_svg_sha256": teacher["candidate_svg_sha256"],
                                "candidate_image_sha256": item.image_sha256,
                                "instruction_sha256": hashlib.sha256(
                                    inputs[teacher["sample_id"]]["instruction"].encode()
                                ).hexdigest(),
                                "metric": item.metric,
                                "score": item.score,
                                "raw_logit": item.raw_logit,
                                "model_id": item.model_id,
                                "model_revision": item.model_revision,
                                "text_template": item.text_template,
                                "text_input": item.text_input,
                                "dtype": item.dtype,
                                "device": item.device,
                            }
                        )
                    _append_jsonl(scores_path, output_rows)
                    completed = len(existing) + offset + len(rows)
                    progress = {
                        "validated_in_invocation_snapshot": len(selected),
                        "previously_scored": len(existing),
                        "newly_scored": offset + len(rows),
                        "completed": completed,
                        "remaining_in_snapshot": len(pending) - offset - len(rows),
                        "teacher_results_snapshot_sha256": teacher_snapshot_sha,
                    }
                    _atomic_json(destination / "progress.json", progress)
                    print(json.dumps(progress, sort_keys=True), flush=True)
            finally:
                scorer.unload_model()

        all_scores, scores_sha = _jsonl_snapshot(scores_path)
        latest_scores = _latest_by(
            all_scores, lambda row: (row["sample_id"], row["candidate_svg_sha256"])
        )
        status_counts = Counter(
            row.get("status", "missing") for row in teacher_latest.values()
        )
        result = {
            "identity": identity,
            "teacher_results_snapshot_sha256": teacher_snapshot_sha,
            "teacher_status_counts": dict(sorted(status_counts.items())),
            "validated_in_invocation_snapshot": len(selected),
            "scores_total": len(latest_scores),
            "scores_jsonl_sha256": scores_sha,
            "newly_scored": len(pending),
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
        }
        _atomic_json(destination / "progress.json", result)
        return result


__all__ = ["precompute_critic_similarity"]
