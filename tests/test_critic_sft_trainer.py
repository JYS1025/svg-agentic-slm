from __future__ import annotations

import json
from pathlib import Path

import pytest

from svg_agentic_slm.agents.schemas import CRITIC_ISSUE_TYPES
from svg_agentic_slm.train.critic_sft_trainer import load_critic_sft_records


def _scorecard_payload() -> dict[str, object]:
    evaluations = []
    for category, issue_types in CRITIC_ISSUE_TYPES.items():
        for issue_type in sorted(issue_types):
            evaluations.append(
                {
                    "category": category,
                    "type": issue_type,
                    "applicable": True,
                    "score": 4,
                    "reason": "The visible property satisfies the instruction.",
                }
            )
    return {"evaluations": evaluations, "issues": []}


def _write_record(tmp_path: Path, **updates) -> Path:
    image = tmp_path / "render.png"
    image.write_bytes(b"not-decoded-during-contract-validation")
    record = {
        "sample_id": "sample-1",
        "instruction": "Draw a circle.",
        "image_path": image.name,
        "labeled_svg": '<svg><circle id="s0000"/></svg>',
        "allowed_target_ids": ["s0000"],
        "critic_output": _scorecard_payload(),
    }
    record.update(updates)
    path = tmp_path / "critic.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_critic_sft_loader_builds_completion_only_vlm_record(tmp_path: Path) -> None:
    from datasets import Dataset

    path = _write_record(tmp_path)

    rows, manifest = load_critic_sft_records(path, score_threshold=3.0)

    assert manifest["record_count"] == 1
    assert manifest["label_contract"] == "vlm_critic_production_output_exact"
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["prompt"][0]["content"][0]["type"] == "text"
    assert rows[0]["prompt"][1]["content"][0] == {"type": "image"}
    completion = json.loads(rows[0]["completion"][0]["content"])
    assert len(completion["evaluations"]) == 18
    assert completion["issues"] == []
    # All message content uses the same list-of-parts shape, so Arrow can infer
    # a stable nested schema before the VLM collator decodes the image.
    assert Dataset.from_list(rows).num_rows == 1


def test_critic_sft_loader_rejects_contract_invalid_label(tmp_path: Path) -> None:
    path = _write_record(tmp_path, critic_output={"evaluations": [], "issues": []})

    with pytest.raises(ValueError):
        load_critic_sft_records(path, score_threshold=3.0)


def test_critic_sft_loader_rejects_missing_image(tmp_path: Path) -> None:
    path = _write_record(tmp_path, image_path="missing.png")

    with pytest.raises(FileNotFoundError, match="image not found"):
        load_critic_sft_records(path, score_threshold=3.0)
