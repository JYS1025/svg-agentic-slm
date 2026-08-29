"""Regression tests for grounded Critic scorecard-contract handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svg_agentic_slm.agents.schemas import (
    CRITIC_ISSUE_TYPES,
    CriticInput,
    CriticTraceError,
)
from svg_agentic_slm.agents.vlm_critic import _CRITIC_RESPONSE_FORMAT, VLMCritic
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.vlm_critic import build_vlm_critic_prompt
from svg_agentic_slm.prompts.system_prompts import get_svg_vlm_critic_system_prompt
from svg_agentic_slm.svg.labeler import CriticLabeler


class _VisionModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []
        self.images: list[bytes] = []
        self.system_prompts: list[str] = []

    def generate_with_image(
        self, prompt: str, image_bytes: bytes, **kwargs
    ) -> ModelResponse:
        self.prompts.append(prompt)
        self.images.append(image_bytes)
        self.system_prompts.append(kwargs["system_prompt"])
        return ModelResponse(text=next(self._responses), model_id="critic-model")


class _UnusedRenderer:
    def render_bytes(self, *args, **kwargs) -> bytes:
        raise AssertionError("critique_attempt must use supplied evidence")


def _critic_input() -> CriticInput:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<circle id="circle" cx="10" cy="10" r="5"/>'
        "</svg>"
    )
    return CriticInput(
        attempt_id="attempt-1",
        instruction="Draw a circle.",
        canonical_svg=svg,
        render_png=b"png-evidence",
        labeling=CriticLabeler().label(svg, "attempt-1"),
    )


def _scorecard_payload(
    *,
    scores: dict[tuple[str, str], int] | None = None,
    not_applicable: set[tuple[str, str]] | None = None,
    issues: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    scores = scores or {}
    not_applicable = not_applicable or set()
    evaluations = []
    for category, issue_types in CRITIC_ISSUE_TYPES.items():
        for issue_type in sorted(issue_types):
            pair = (category, issue_type)
            applicable = pair not in not_applicable
            evaluations.append(
                {
                    "category": category,
                    "type": issue_type,
                    "applicable": applicable,
                    "score": scores.get(pair, 4) if applicable else None,
                    "reason": (
                        "The visible property satisfies the instruction."
                        if applicable
                        else "The instruction and image do not use this property."
                    ),
                }
            )
    return {"evaluations": evaluations, "issues": list(issues or [])}


def test_vlm_retry_is_format_repair_of_previous_judgment_only() -> None:
    invalid_payload = _scorecard_payload()
    invalid_payload["explanation"] = "remove me"
    invalid_response = json.dumps(invalid_payload)
    repaired_response = json.dumps(_scorecard_payload())
    model = _VisionModel([invalid_response, repaired_response])
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    feedback = critic.critique_attempt(_critic_input())

    assert feedback.status == "pass"
    assert feedback.score == 4.0
    assert len(feedback.evaluations) == 18
    assert len(feedback.model_calls) == 2
    assert feedback.model_calls[0].validation_success is False
    assert feedback.model_calls[1].validation_success is True
    repair_prompt = model.prompts[1]
    assert "Do not re-evaluate" in repair_prompt
    assert invalid_response in json.loads(
        repair_prompt.split("<previous_response_json>\n", 1)[1].split(
            "\n</previous_response_json>", 1
        )[0]
    )
    assert "<original_instruction_json>" not in repair_prompt
    assert model.system_prompts[0] == model.system_prompts[1]
    assert model.system_prompts[0] == feedback.model_calls[0].system_prompt
    assert feedback.model_calls[0].system_prompt == feedback.model_calls[1].system_prompt
    assert model.images[0] == b"png-evidence"
    assert model.images[1] != model.images[0]
    assert model.images[1].startswith(b"\x89PNG")


def test_vlm_format_repair_cannot_change_judgment_fields() -> None:
    invalid_payload = _scorecard_payload()
    invalid_payload["explanation"] = "remove me"
    changed_payload = _scorecard_payload(scores={("layout", "scale"): 3})
    model = _VisionModel([json.dumps(invalid_payload), json.dumps(changed_payload)])
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    with pytest.raises(CriticTraceError, match="format repair"):
        critic.critique_attempt(_critic_input())


def test_vlm_contract_failure_raises_trace_error_instead_of_svg_feedback() -> None:
    model = _VisionModel(["not json", "still not json"])
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    with pytest.raises(CriticTraceError) as captured:
        critic.critique_attempt(_critic_input())

    assert len(captured.value.model_calls) == 2
    assert all(not call.validation_success for call in captured.value.model_calls)
    assert "contract retry" in str(captured.value)
    assert model.images == [b"png-evidence", b"png-evidence"]
    assert "Perform the full image-grounded evaluation again" in model.prompts[1]


def test_vlm_unusable_response_retries_full_evaluation_with_original_image() -> None:
    model = _VisionModel(["not json", json.dumps(_scorecard_payload())])
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    feedback = critic.critique_attempt(_critic_input())

    assert feedback.status == "pass"
    assert model.images == [b"png-evidence", b"png-evidence"]
    assert "<original_instruction_json>" in model.prompts[1]
    assert feedback.model_calls[1].generation_parameters["full_evaluation_retry"] is True
    assert feedback.model_calls[1].generation_parameters["format_repair_only"] is False


def test_vlm_missing_judgment_field_retries_with_visual_evidence() -> None:
    model = _VisionModel(
        [json.dumps({"issues": []}), json.dumps(_scorecard_payload())]
    )
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    feedback = critic.critique_attempt(_critic_input())

    assert feedback.status == "pass"
    assert model.images == [b"png-evidence", b"png-evidence"]
    assert feedback.model_calls[1].generation_parameters["full_evaluation_retry"] is True


def test_vlm_duplicate_issues_become_trace_failure_not_uncaught_value_error() -> None:
    target_id = next(iter(_critic_input().labeling.elements))
    issue = {
        "category": "layout",
        "type": "scale",
        "scope": "object",
        "target_ids": [target_id],
        "observed": "The circle is too small.",
        "expected": "The circle should be larger.",
        "fix": "Increase the circle size.",
    }
    duplicate = {**issue, "fix": "Scale the same circle up."}
    response = json.dumps(
        _scorecard_payload(
            scores={("layout", "scale"): 2},
            issues=[issue, duplicate],
        )
    )
    model = _VisionModel([response, response])
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    with pytest.raises(CriticTraceError, match="duplicate issue") as captured:
        critic.critique_attempt(_critic_input())

    assert len(captured.value.model_calls) == 2


def test_vlm_prompt_separates_static_contract_from_task_input() -> None:
    system_prompt = get_svg_vlm_critic_system_prompt(score_threshold=3.0)
    user_prompt = build_vlm_critic_prompt(
        "Draw a circle.",
        labeled_svg='<svg data-agent-id="g0001"/>',
        allowed_target_ids=["g0001"],
        score_threshold=3.0,
    )

    assert system_prompt.startswith("You are an expert image-grounded SVG critic.")
    assert "Rules:" in system_prompt
    assert "OUTPUT JSON FORMAT" in system_prompt
    assert '"evaluations": [' in system_prompt
    assert '"applicable": true' in system_prompt
    assert '"score": null' in system_prompt
    assert '"target_ids": []' in system_prompt
    assert "configured score threshold is 3" in system_prompt
    assert "18 valid category and type pairs" in system_prompt
    assert "ISSUE TAXONOMY" in system_prompt
    assert system_prompt.count("Meaning:") == 22
    assert system_prompt.count("Example:") == 18
    assert "Follow these output rules" not in system_prompt
    assert '"preserve"' not in system_prompt
    assert '"severity"' not in system_prompt
    assert "—" not in system_prompt
    assert "–" not in system_prompt
    assert ";" not in system_prompt

    assert "<original_instruction_json>" in user_prompt
    assert '"Draw a circle."' in user_prompt
    assert "<labeled_svg_json>" in user_prompt
    assert 'data-agent-id=\\"g0001\\"' in user_prompt
    assert '<allowed_target_ids_json>\n["g0001"]' in user_prompt
    assert "OUTPUT JSON FORMAT" not in user_prompt
    assert "ISSUE TAXONOMY" not in user_prompt
    assert "Rules:" not in user_prompt


def test_vlm_accepts_scorecard_and_targeted_issue() -> None:
    target_id = next(iter(_critic_input().labeling.elements))
    issue = {
        "category": "layout",
        "type": "scale",
        "scope": "object",
        "target_ids": [target_id],
        "observed": "The circle is too small.",
        "expected": "The circle should occupy most of the canvas.",
        "fix": f"Increase the size of {target_id}.",
    }
    response = json.dumps(
        _scorecard_payload(
            scores={("layout", "scale"): 2},
            issues=[issue],
        )
    )
    critic = VLMCritic(_VisionModel([response]), _UnusedRenderer())  # type: ignore[arg-type]

    feedback = critic.critique_attempt(_critic_input())

    assert feedback.schema_version == 3
    assert feedback.status == "revise"
    assert feedback.score == 2.0
    assert feedback.preserve == []
    assert feedback.structured_issues[0].category == "layout"
    assert feedback.structured_issues[0].type == "scale"
    assert feedback.structured_issues[0].target_ids == [target_id]


def test_vlm_allows_not_applicable_evaluation_without_score() -> None:
    response = json.dumps(
        _scorecard_payload(not_applicable={("semantic", "text_content")})
    )
    critic = VLMCritic(_VisionModel([response]), _UnusedRenderer())  # type: ignore[arg-type]

    feedback = critic.critique_attempt(_critic_input())

    text_evaluation = next(
        item
        for item in feedback.evaluations
        if (item.category, item.type) == ("semantic", "text_content")
    )
    assert text_evaluation.applicable is False
    assert text_evaluation.score is None
    assert feedback.status == "pass"


def test_runtime_response_schema_encodes_scorecard_contract() -> None:
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "svg_agentic_slm"
        / "agents"
        / "critic_output.schema.json"
    )
    packaged_schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert _CRITIC_RESPONSE_FORMAT["schema"] == packaged_schema
    assert packaged_schema["required"] == ["evaluations", "issues"]
    evaluation_schema = packaged_schema["properties"]["evaluations"]
    assert evaluation_schema["minItems"] == 18
    assert evaluation_schema["maxItems"] == 18
    assert evaluation_schema["items"]["properties"]["score"] == {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 4,
    }
    issue_schema = packaged_schema["properties"]["issues"]["items"]
    assert "severity" not in issue_schema["properties"]
    assert issue_schema["required"] == [
        "category",
        "type",
        "scope",
        "target_ids",
        "observed",
        "expected",
        "fix",
    ]
