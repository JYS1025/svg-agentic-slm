"""Focused checks for the standalone OmniSVG teacher runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_teacher():
    path = ROOT / "scripts/run_omnisvg_critic_teacher.py"
    spec = importlib.util.spec_from_file_location("omnisvg_teacher_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_input(tmp_path: Path, candidate_svg: str) -> tuple[Path, Path, Path, dict, dict]:
    source = tmp_path / "source"
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    (source / "candidate").mkdir(parents=True)
    (inputs / "reference").mkdir(parents=True)
    candidate = source / "candidate/sample.svg"
    reference = inputs / "reference/sample.svg"
    candidate.write_text(candidate_svg)
    reference.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="30" fill=""/></svg>'
    )
    row = {
        "sample_id": "icon:test",
        "sample_index": 0,
        "selected_text_field": "detail",
        "candidate_svg_path": "candidate/sample.svg",
        "candidate_svg_sha256": sha(candidate),
        "reference_svg_path": "reference/sample.svg",
        "reference_svg_sha256": sha(reference),
    }
    record = {
        "sample_id": "icon:test",
        "selected_text_field": "detail",
        "instruction": "A centered black circle.",
    }
    return source, inputs, output, row, record


def test_teacher_prompt_uses_current_critic_contract_without_explicit_pair_list() -> None:
    teacher = load_teacher()
    prompt = teacher.system_prompt()
    assert prompt.startswith(teacher.build_vlm_critic_system_prompt(score_threshold=3.0))
    assert "DISTILLATION REFERENCE EVIDENCE" in prompt
    assert "Required evaluations order:" not in prompt
    assert "1. semantic / presence" not in prompt


def test_validate_feedback_canonicalizes_complete_evaluations_without_retry() -> None:
    teacher = load_teacher()
    evaluations = [
        {
            "category": category,
            "type": issue_type,
            "applicable": True,
            "score": 4,
            "reason": "The candidate satisfies this criterion.",
        }
        for category, issue_type in reversed(teacher.PAIRS)
    ]

    feedback = teacher.validate_feedback(
        json.dumps({"evaluations": evaluations, "issues": []}),
        allowed_ids=[],
    )

    assert [
        (item["category"], item["type"]) for item in feedback["evaluations"]
    ] == teacher.PAIRS


def test_validation_retry_text_changes_the_retry_input() -> None:
    teacher = load_teacher()
    error = "ValidationError: issue type does not belong to its category"

    retry_text = teacher.validation_retry_text(error)

    assert error in retry_text
    assert "complete critic JSON object" in retry_text
    assert "each of the 18 category and type pairs exactly once" in retry_text


def test_latest_validation_error_recovers_resume_context() -> None:
    teacher = load_teacher()
    result = {
        "attempts": [
            {"validation_error": "first error"},
            {"validation_error": "last error"},
        ]
    }

    assert teacher.latest_validation_error(result) == "last error"


def test_prepare_request_rerenders_reference_and_puts_candidate_image_first(tmp_path: Path) -> None:
    teacher = load_teacher()
    source, inputs, output, row, record = fixture_input(
        tmp_path,
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect x="20" y="20" width="60" height="60" fill="#123456"/></svg>',
    )
    content, allowed_ids, metadata = teacher.prepare_request(
        source, inputs, output, row, record
    )
    saved = json.loads((output / metadata["request_dir"] / "input.json").read_text())
    assert content[0]["type"] == "input_image"
    assert content[0]["detail"] == "high"
    assert content[1]["type"] == "input_text"
    assert "You are a professional SVG design critic." in content[1]["text"]
    assert "Required evaluations order:" not in teacher.system_prompt()
    assert allowed_ids
    assert saved["candidate_blank_white"] is False
    assert saved["reference_blank_white"] is False
    reference_png = output / saved["reference_rendered_png_path"]
    assert not teacher.blank_white_png(reference_png.read_bytes())


def test_prepare_request_skips_blank_candidate_before_api(tmp_path: Path) -> None:
    teacher = load_teacher()
    source, inputs, output, row, record = fixture_input(
        tmp_path,
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect x="20" y="20" width="60" height="60" fill="none"/></svg>',
    )
    with pytest.raises(teacher.BlankCandidateError) as caught:
        teacher.prepare_request(source, inputs, output, row, record)
    saved = json.loads((output / caught.value.metadata["request_dir"] / "input.json").read_text())
    assert saved["status"] == "skipped_blank_candidate"
    assert saved["candidate_blank_white"] is True
    assert saved["user_content"] == []
