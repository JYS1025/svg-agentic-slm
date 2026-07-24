"""Tests for generation artifact readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svg_agentic_slm.artifacts.generation import (
    list_generation_artifacts,
    load_generation_artifact,
)


def test_load_generation_artifact_from_svg_path(tmp_path: Path) -> None:
    """Loading by SVG path should resolve the matching metadata sidecar."""
    svg_path = tmp_path / "sample.svg"
    metadata_path = tmp_path / "sample.json"
    svg_path.write_text("<svg></svg>", encoding="utf-8")

    payload = {
        "instruction": "Draw a square.",
        "svg_path": str(svg_path),
        "render_path": str(tmp_path / "sample.png"),
        "is_valid": True,
        "revision_count": 0,
        "critic_feedback": [],
        "runtime": {"enable_render": True},
        "metadata": {"validation": {"is_valid": True}},
        "generated_at_utc": "2026-07-10T12:00:00Z",
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    record = load_generation_artifact(svg_path)

    assert record.instruction == "Draw a square."
    assert record.svg_path == svg_path
    assert record.metadata_path == metadata_path
    assert record.render_path == tmp_path / "sample.png"
    assert record.runtime["enable_render"] is True
    assert record.schema_version == 0


def test_list_generation_artifacts_returns_sorted_records(tmp_path: Path) -> None:
    """Artifact listing should read all JSON sidecars in sorted order."""
    for name in ["b", "a"]:
        svg_path = tmp_path / f"{name}.svg"
        metadata_path = tmp_path / f"{name}.json"
        svg_path.write_text("<svg></svg>", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "instruction": f"draw {name}",
                    "svg_path": str(svg_path),
                    "is_valid": True,
                    "revision_count": 0,
                    "critic_feedback": [],
                    "runtime": {},
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

    records = list_generation_artifacts(tmp_path)

    assert [record.instruction for record in records] == ["draw a", "draw b"]


def test_load_generation_artifact_resolves_paths_relative_to_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Portable sidecars should resolve artifact paths independently of cwd."""
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    svg_path = artifact_dir / "sample.svg"
    render_path = artifact_dir / "sample.png"
    metadata_path = artifact_dir / "sample.json"
    svg_path.write_text("<svg></svg>", encoding="utf-8")
    render_path.write_bytes(b"render")
    metadata_path.write_text(
        json.dumps(
            {
                "instruction": "Draw a square.",
                "svg_path": "sample.svg",
                "render_path": "sample.png",
            }
        ),
        encoding="utf-8",
    )
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    record = load_generation_artifact(metadata_path)

    assert record.svg_path == svg_path
    assert record.render_path == render_path


def test_load_generation_artifact_reads_version_and_run_id(tmp_path: Path) -> None:
    svg_path = tmp_path / "sample.svg"
    metadata_path = tmp_path / "sample.json"
    svg_path.write_text("<svg></svg>", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(_v1_payload("sample.svg")),
        encoding="utf-8",
    )

    record = load_generation_artifact(metadata_path)

    assert record.schema_version == 1
    assert record.run_id == "run-1"


def test_v1_reader_rejects_final_svg_content_mismatch(tmp_path: Path) -> None:
    svg_path = tmp_path / "sample.svg"
    attempt_path = tmp_path / "attempt.svg"
    svg_path.write_text('<svg id="canonical"/>', encoding="utf-8")
    attempt_path.write_text('<svg id="attempt"/>', encoding="utf-8")
    payload = _v1_payload("sample.svg")
    attempts = payload["metadata"]["generator"]["attempts"]  # type: ignore[index]
    attempts[0]["svg_ref"] = "attempt.svg"  # type: ignore[index]
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must match the final successful attempt"):
        load_generation_artifact(metadata_path)


def test_load_generation_artifact_reads_v1_attempt_lineage(tmp_path: Path) -> None:
    svg_path = tmp_path / "sample.svg"
    attempt_dir = tmp_path / "sample.attempts"
    attempt_dir.mkdir()
    attempt_svg = attempt_dir / "attempt-000.svg"
    raw_output = attempt_dir / "attempt-000.call-000.raw.txt"
    prompt = attempt_dir / "attempt-000.call-000.prompt.txt"
    svg_path.write_text("<svg></svg>", encoding="utf-8")
    attempt_svg.write_text("<svg></svg>", encoding="utf-8")
    raw_output.write_text("raw", encoding="utf-8")
    prompt.write_text("prompt", encoding="utf-8")
    metadata_path = tmp_path / "sample.json"
    payload = _v1_payload("sample.svg")
    payload["stop_reason"] = "critic_acceptance_threshold_met"
    payload["metadata"] = {
        "generator": {
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "mode": "initial",
                    "svg_ref": "sample.attempts/attempt-000.svg",
                    "raw_output_ref": (
                        "sample.attempts/attempt-000.call-000.raw.txt"
                    ),
                    "status": "succeeded",
                    "error": None,
                    "outcome": "accepted",
                    "stop_reason": "critic_acceptance_threshold_met",
                    "prompt_version": "test-v1",
                    "parent_attempt_id": None,
                    "trigger_feedback_id": None,
                    "context_item_ids": [],
                    "truncated_context_item_ids": [],
                    "metadata": {},
                    "model_calls": [
                        {
                            "model_call_id": "call-1",
                            "prompt_ref": (
                                "sample.attempts/attempt-000.call-000.prompt.txt"
                            ),
                            "system_prompt_ref": None,
                            "raw_output_ref": (
                                "sample.attempts/attempt-000.call-000.raw.txt"
                            ),
                            "generation_parameters": {},
                            "model_id": "test-model",
                            "model_revision": None,
                            "finish_reason": "stop",
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "latency_seconds": 0.1,
                            "metadata": {},
                        }
                    ],
                }
            ]
        }
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    record = load_generation_artifact(metadata_path)

    assert record.outcome == "accepted"
    assert record.stop_reason == "critic_acceptance_threshold_met"
    assert len(record.attempts) == 1
    assert record.attempts[0].svg_path == attempt_svg
    assert record.attempts[0].raw_output_path == raw_output
    assert record.attempts[0].model_calls[0].prompt_path == prompt


def test_v1_reader_rejects_missing_cwd_fallback_and_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar_dir = tmp_path / "sidecar"
    cwd_dir = tmp_path / "cwd"
    sidecar_dir.mkdir()
    cwd_dir.mkdir()
    (cwd_dir / "missing.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    missing_metadata = sidecar_dir / "missing.json"
    missing_metadata.write_text(
        json.dumps(_v1_payload("missing.svg")),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_generation_artifact(missing_metadata)

    escaped_svg = tmp_path / "escaped.svg"
    escaped_svg.write_text("<svg/>", encoding="utf-8")
    escaped_metadata = sidecar_dir / "escaped.json"
    escaped_metadata.write_text(
        json.dumps(_v1_payload("../escaped.svg")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_generation_artifact(escaped_metadata)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("is_valid", "yes", "is_valid must be a boolean"),
        ("revision_count", "many", "revision_count must be"),
        ("critic_feedback", [42], r"critic_feedback\[0\] must be an object"),
    ],
)
def test_v1_reader_rejects_invalid_field_types(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    payload[field_name] = invalid_value
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_generation_artifact(metadata_path)


def test_v1_reader_rejects_non_object_attempt(tmp_path: Path) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    payload["metadata"] = {"generator": {"attempts": [42]}}
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"attempts\[0\] must be an object"):
        load_generation_artifact(metadata_path)


def test_v1_reader_rejects_duplicate_and_broken_revision_lineage(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    attempts = payload["metadata"]["generator"]["attempts"]  # type: ignore[index]
    revision = dict(attempts[0])  # type: ignore[index]
    revision.update(
        {
            "mode": "revision",
            "parent_attempt_id": "missing-attempt",
            "trigger_feedback_id": "missing-feedback",
        }
    )
    attempts.append(revision)  # type: ignore[union-attr]
    payload["revision_count"] = 1
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="attempt_id values must be unique"):
        load_generation_artifact(metadata_path)


def test_v1_reader_rejects_feedback_targeting_unknown_attempt(tmp_path: Path) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    payload["critic_feedback"] = [
        {
            "feedback_id": "feedback-1",
            "target_attempt_id": "missing-attempt",
            "score": 8.0,
            "is_valid": True,
            "matches_instruction": True,
            "issues": [],
            "suggestions": [],
            "critic_type": "test",
        }
    ]
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="target_attempt_id"):
        load_generation_artifact(metadata_path)


def test_v1_reader_accepts_correlated_revision_lineage(tmp_path: Path) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    attempts = payload["metadata"]["generator"]["attempts"]  # type: ignore[index]
    initial = attempts[0]  # type: ignore[index]
    initial["outcome"] = "rejected"
    initial["stop_reason"] = "critic_revision_requested"
    revision = dict(initial)
    revision.update(
        {
            "attempt_id": "attempt-2",
            "mode": "revision",
            "parent_attempt_id": "attempt-1",
            "trigger_feedback_id": "feedback-1",
            "outcome": "accepted",
            "stop_reason": "critic_acceptance_threshold_met",
        }
    )
    attempts.append(revision)  # type: ignore[union-attr]
    payload["revision_count"] = 1
    payload["stop_reason"] = "critic_acceptance_threshold_met"
    payload["critic_feedback"] = [
        _feedback_payload("feedback-1", "attempt-1", score=4.0),
        _feedback_payload("feedback-2", "attempt-2", score=9.0),
    ]
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    record = load_generation_artifact(metadata_path)

    assert record.revision_count == 1
    assert record.attempts[-1].parent_attempt_id == "attempt-1"
    assert record.attempts[-1].trigger_feedback_id == "feedback-1"


@pytest.mark.parametrize(
    ("feedback_updates", "message"),
    [
        ({"is_valid": False}, "is_valid=true"),
        ({"matches_instruction": False}, "matches_instruction=true"),
        ({"score": 7.9}, "configured acceptance threshold"),
    ],
)
def test_v1_reader_rejects_inconsistent_accepted_critic_feedback(
    tmp_path: Path,
    feedback_updates: dict[str, object],
    message: str,
) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    payload["stop_reason"] = "critic_acceptance_threshold_met"
    payload["runtime"] = {
        "enable_critic": True,
        "generation_config": {
            "orchestration": {
                "critic_acceptance_score": 8.0,
            }
        },
    }
    attempts = payload["metadata"]["generator"]["attempts"]  # type: ignore[index]
    attempts[0]["stop_reason"] = "critic_acceptance_threshold_met"  # type: ignore[index]
    feedback = _feedback_payload("feedback-1", "attempt-1", score=9.0)
    feedback.update(feedback_updates)
    payload["critic_feedback"] = [feedback]
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_generation_artifact(metadata_path)


def test_v1_reader_requires_final_feedback_to_target_final_attempt(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    attempts = payload["metadata"]["generator"]["attempts"]  # type: ignore[index]
    initial = attempts[0]  # type: ignore[index]
    initial["outcome"] = "rejected"
    initial["stop_reason"] = "critic_revision_requested"
    revision = dict(initial)
    revision.update(
        {
            "attempt_id": "attempt-2",
            "mode": "revision",
            "parent_attempt_id": "attempt-1",
            "trigger_feedback_id": "feedback-1",
            "outcome": "accepted",
            "stop_reason": "critic_acceptance_threshold_met",
        }
    )
    attempts.append(revision)  # type: ignore[union-attr]
    payload["revision_count"] = 1
    payload["stop_reason"] = "critic_acceptance_threshold_met"
    payload["runtime"] = {"enable_critic": True}
    payload["critic_feedback"] = [
        _feedback_payload("feedback-1", "attempt-1", score=4.0),
        _feedback_payload("feedback-2", "attempt-1", score=9.0),
    ]
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="final attempt"):
        load_generation_artifact(metadata_path)


def test_v1_reader_requires_feedback_when_accepted_with_critic_enabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    payload["runtime"] = {"enable_critic": True}
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires final feedback"):
        load_generation_artifact(metadata_path)


def test_v1_reader_uses_configured_critic_acceptance_threshold(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.svg").write_text("<svg/>", encoding="utf-8")
    payload = _v1_payload("sample.svg")
    payload["stop_reason"] = "critic_acceptance_threshold_met"
    payload["runtime"] = {
        "enable_critic": True,
        "generation_config": {
            "orchestration": {
                "critic_acceptance_score": 7.0,
            }
        },
    }
    attempts = payload["metadata"]["generator"]["attempts"]  # type: ignore[index]
    attempts[0]["stop_reason"] = "critic_acceptance_threshold_met"  # type: ignore[index]
    payload["critic_feedback"] = [
        _feedback_payload("feedback-1", "attempt-1", score=7.5)
    ]
    metadata_path = tmp_path / "sample.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    record = load_generation_artifact(metadata_path)

    assert record.outcome == "accepted"


def _v1_payload(svg_path: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "instruction": "Draw a square.",
        "svg_path": svg_path,
        "render_path": None,
        "is_valid": True,
        "outcome": "accepted",
        "stop_reason": "generator_only_complete",
        "revision_count": 0,
        "critic_feedback": [],
        "runtime": {},
        "metadata": {
            "generator": {
                "attempts": [
                    {
                        "attempt_id": "attempt-1",
                        "mode": "initial",
                        "parent_attempt_id": None,
                        "trigger_feedback_id": None,
                        "svg_ref": svg_path,
                        "raw_output_ref": None,
                        "status": "succeeded",
                        "error": None,
                        "outcome": "accepted",
                        "stop_reason": "generator_only_complete",
                        "prompt_version": "test-v1",
                        "context_item_ids": [],
                        "truncated_context_item_ids": [],
                        "model_calls": [],
                        "metadata": {},
                    }
                ]
            }
        },
        "generated_at_utc": "2026-07-10T12:00:00Z",
    }


def _feedback_payload(
    feedback_id: str,
    target_attempt_id: str,
    *,
    score: float,
) -> dict[str, object]:
    return {
        "feedback_id": feedback_id,
        "target_attempt_id": target_attempt_id,
        "score": score,
        "is_valid": True,
        "matches_instruction": True,
        "issues": [],
        "suggestions": [],
        "critic_type": "test",
    }
