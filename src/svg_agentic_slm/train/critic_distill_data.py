"""Freeze validated OmniSVG teacher judgments into critic SFT splits."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from svg_agentic_slm.agents.vlm_critic import _parse_critic_contract
from svg_agentic_slm.models.image_text_similarity import SIGLIP2_PAIR_PROBABILITY
from svg_agentic_slm.models.siglip2_similarity import (
    DEFAULT_SIGLIP2_MODEL_ID,
    DEFAULT_SIGLIP2_MODEL_REVISION,
    DEFAULT_SIGLIP2_TEXT_TEMPLATE,
)
from svg_agentic_slm.svg.labeler import CriticLabeler


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if _sha256(data) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return data


def _jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read only newline-terminated rows so a live writer cannot leak a partial row."""
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1] if b"\n" in data else b""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows, _sha256(complete)


def _latest_by(rows: list[dict[str, Any]], key: Any) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}
    for row in rows:
        latest[key(row)] = row
    return latest


def _split_rows(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically preserve the rare pass class without duplicating it."""
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for passes in (False, True):
        group = [row for row in rows if bool(row["passes_threshold"]) is passes]
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['candidate_svg_sha256']}".encode()
            ).digest()
        )
        holdout = max(1, round(len(group) * 0.05)) if len(group) >= 3 else 0
        train_end = len(group) - 2 * holdout
        validation_end = len(group) - holdout
        splits["train"].extend(group[:train_end])
        splits["validation"].extend(group[train_end:validation_end])
        splits["test"].extend(group[validation_end:])
    for rows_for_split in splits.values():
        rows_for_split.sort(key=lambda row: row["sample_index"])
    return splits


def _validate_candidate_hash_split_disjointness(
    split_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """Allow repeated candidates only when every copy stays in one split."""
    owners: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for split, rows in split_rows.items():
        for row in rows:
            candidate_hash = row["candidate_svg_sha256"]
            owner = owners.setdefault(candidate_hash, split)
            if owner != split:
                conflicts.setdefault(candidate_hash, {owner}).add(split)
    if conflicts:
        details = ", ".join(
            f"{candidate_hash}:{'/'.join(sorted(splits))}"
            for candidate_hash, splits in sorted(conflicts.items())
        )
        raise ValueError(
            "Candidate SVG hashes cross dataset split boundaries: " + details
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare_critic_distill_dataset(
    *,
    source_run: str | Path,
    teacher_run: str | Path,
    similarity_scores_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Join only current, validated teacher labels and atomically write 90/5/5 splits."""
    source_dir = Path(source_run).resolve()
    teacher_dir = Path(teacher_run).resolve()
    destination = Path(output_dir).resolve()
    teacher_manifest = json.loads((teacher_dir / "run_manifest.json").read_text())
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text())
    if Path(teacher_manifest["identity"]["source_run"]).resolve() != source_dir:
        raise ValueError("Teacher run does not point to the selected source run.")
    input_path = Path(source_manifest["settings"]["input_manifest"]).resolve()
    input_rows, input_snapshot_sha = _jsonl_snapshot(input_path)
    if input_snapshot_sha != source_manifest["settings"]["input_sha256"]:
        raise ValueError("Source input JSONL differs from the pinned OmniSVG input.")
    inputs = _latest_by(input_rows, lambda row: row["sample_id"])
    if len(inputs) != len(input_rows):
        raise ValueError("Source input contains duplicate sample IDs.")

    source_rows: list[dict[str, Any]] = []
    source_snapshot_hashes: dict[str, str] = {}
    for gpu in source_manifest["settings"]["gpus"]:
        path = source_dir / f"results.gpu{gpu}.jsonl"
        rows, digest = _jsonl_snapshot(path)
        source_rows.extend(rows)
        source_snapshot_hashes[path.name] = digest
    source_latest = _latest_by(source_rows, lambda row: row["sample_id"])

    teacher_rows, teacher_snapshot_sha = _jsonl_snapshot(teacher_dir / "results.jsonl")
    teacher_latest = _latest_by(
        teacher_rows, lambda row: (row["sample_id"], row["candidate_svg_sha256"])
    )
    status_counts = Counter(row.get("status", "missing") for row in teacher_latest.values())
    similarity_path = Path(similarity_scores_path).resolve()
    similarity_rows, similarity_snapshot_sha = _jsonl_snapshot(similarity_path)
    similarities = _latest_by(
        similarity_rows, lambda row: (row["sample_id"], row["candidate_svg_sha256"])
    )
    prepared_rows: list[dict[str, Any]] = []
    stale_teacher = 0
    rejected_blank = 0
    missing_similarity = 0
    for sample_id, source in sorted(
        source_latest.items(), key=lambda item: item[1].get("sample_index", 0)
    ):
        if source.get("status") != "generated":
            continue
        candidate_sha = source["candidate_svg_sha256"]
        teacher = teacher_latest.get((sample_id, candidate_sha))
        if teacher is None:
            if any(key[0] == sample_id for key in teacher_latest):
                stale_teacher += 1
            continue
        if teacher.get("status") == "skipped_blank_candidate":
            rejected_blank += 1
            continue
        if teacher.get("status") != "validated":
            continue
        record = inputs.get(sample_id)
        if record is None:
            raise ValueError(f"Teacher sample is absent from source inputs: {sample_id}")
        similarity = similarities.get((sample_id, candidate_sha))
        if similarity is None:
            missing_similarity += 1
            continue
        request_dir = teacher_dir / teacher["request_dir"]
        request_input = json.loads((request_dir / "input.json").read_text())
        if request_input.get("candidate_svg_sha256") != candidate_sha:
            raise ValueError(f"Teacher request candidate mismatch: {sample_id}")
        image_path = request_dir / "candidate.rendered.png"
        _checked_bytes(image_path, teacher["candidate_rendered_png_sha256"])
        if similarity.get("candidate_image_sha256") != teacher["candidate_rendered_png_sha256"]:
            raise ValueError(f"SigLIP2 image provenance mismatch: {sample_id}")
        if similarity.get("instruction_sha256") != _sha256(record["instruction"].encode()):
            raise ValueError(f"SigLIP2 instruction provenance mismatch: {sample_id}")
        expected_similarity = {
            "metric": SIGLIP2_PAIR_PROBABILITY,
            "model_id": DEFAULT_SIGLIP2_MODEL_ID,
            "model_revision": DEFAULT_SIGLIP2_MODEL_REVISION,
            "text_template": DEFAULT_SIGLIP2_TEXT_TEMPLATE,
        }
        for field, expected in expected_similarity.items():
            if similarity.get(field) != expected:
                raise ValueError(f"SigLIP2 {field} mismatch for {sample_id}")
        score = similarity.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"Invalid SigLIP2 score for {sample_id}")
        candidate_path = source_dir / source["candidate_svg_path"]
        candidate_svg = _checked_bytes(candidate_path, candidate_sha).decode("utf-8")
        labeling = CriticLabeler().label(candidate_svg, attempt_id=sample_id)
        allowed_ids = sorted(labeling.elements)
        if allowed_ids != request_input.get("allowed_target_ids"):
            raise ValueError(f"Candidate labeling differs from teacher request: {sample_id}")
        feedback = teacher.get("feedback")
        if not isinstance(feedback, dict):
            raise ValueError(f"Validated teacher row has no feedback: {sample_id}")
        _parse_critic_contract(
            feedback,
            allowed_target_ids=set(allowed_ids),
            score_threshold=3.0,
        )
        applicable_scores = [
            item["score"] for item in feedback["evaluations"] if item["applicable"]
        ]
        prepared_rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "sample_index": source["sample_index"],
                "domain": record["domain"],
                "selected_text_field": record["selected_text_field"],
                "instruction": record["instruction"],
                "candidate_svg_sha256": candidate_sha,
                "candidate_image_path": str(image_path.resolve()),
                "candidate_image_sha256": teacher["candidate_rendered_png_sha256"],
                "labeled_svg": labeling.labeled_svg,
                "allowed_target_ids": allowed_ids,
                "similarity_score": round(float(score), 6),
                "similarity_evidence": {
                    field: similarity[field]
                    for field in (
                        "metric",
                        "raw_logit",
                        "model_id",
                        "model_revision",
                        "text_template",
                        "text_input",
                    )
                },
                "target_json": _canonical_json(feedback),
                "passes_threshold": min(applicable_scores) >= 3,
                "teacher_created_at": teacher.get("created_at"),
            }
        )

    split_rows = _split_rows(prepared_rows, seed=seed)
    all_rows = sum(len(rows) for rows in split_rows.values())
    if all_rows == 0:
        raise ValueError("No current validated teacher records were available.")
    _validate_candidate_hash_split_disjointness(split_rows)

    identity = {
        "schema_version": 1,
        "source_run": str(source_dir),
        "teacher_run": str(teacher_dir),
        "input_jsonl_sha256": input_snapshot_sha,
        "source_results_sha256": source_snapshot_hashes,
        "teacher_results_complete_lines_sha256": teacher_snapshot_sha,
        "similarity_scores_path": str(similarity_path),
        "similarity_scores_complete_lines_sha256": similarity_snapshot_sha,
        "similarity_contract": {
            "metric": SIGLIP2_PAIR_PROBABILITY,
            "model_id": DEFAULT_SIGLIP2_MODEL_ID,
            "model_revision": DEFAULT_SIGLIP2_MODEL_REVISION,
            "text_template": DEFAULT_SIGLIP2_TEXT_TEMPLATE,
            "prompt_precision_decimals": 6,
        },
        "teacher_prompt_version": teacher_manifest["identity"].get("prompt_version"),
        "critic_prompt_version": teacher_manifest["identity"].get(
            "critic_prompt_version"
        ),
        "split_policy": (
            "pass-stratified sha256(seed:candidate_svg_sha256) rank; exact approximately "
            "90/5/5 without duplication"
        ),
        "split_seed": seed,
        "student_evidence": (
            "candidate_render_plus_instruction_labeled_svg_allowed_ids_siglip2_score"
        ),
        "reference_evidence_in_student": False,
    }
    fingerprint = _sha256(_canonical_json(identity).encode())
    destination.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[tuple[Path, Path]] = []
    split_hashes: dict[str, str] = {}
    for split, rows in split_rows.items():
        final_path = destination / f"{split}.jsonl"
        temporary = destination / f".{split}.{os.getpid()}.tmp"
        payload = "".join(_canonical_json(row) + "\n" for row in rows).encode()
        temporary.write_bytes(payload)
        temporary_paths.append((temporary, final_path))
        split_hashes[split] = _sha256(payload)
    pass_counts = {
        split: sum(bool(row["passes_threshold"]) for row in rows)
        for split, rows in split_rows.items()
    }
    manifest = {
        **identity,
        "fingerprint": fingerprint,
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "pass_counts": pass_counts,
        "acceptance_calibration_insufficient": sum(pass_counts.values()) < 100,
        "teacher_status_counts": dict(sorted(status_counts.items())),
        "excluded_blank": rejected_blank,
        "excluded_missing_similarity": missing_similarity,
        "stale_teacher_candidates": stale_teacher,
        "split_sha256": split_hashes,
    }
    manifest_temp = destination / f".manifest.{os.getpid()}.tmp"
    manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for temporary, final_path in temporary_paths:
        os.replace(temporary, final_path)
    os.replace(manifest_temp, destination / "manifest.json")
    return manifest


__all__ = ["prepare_critic_distill_dataset"]
