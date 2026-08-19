"""Regression tests for grounded Critic response-contract handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svg_agentic_slm.agents.schemas import CriticInput, CriticTraceError
from svg_agentic_slm.agents.vlm_critic import _CRITIC_RESPONSE_FORMAT, VLMCritic
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.svg.labeler import CriticLabeler


class _VisionModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []
        self.images: list[bytes] = []

    def generate_with_image(
        self, prompt: str, image_bytes: bytes, **kwargs
    ) -> ModelResponse:
        self.prompts.append(prompt)
        self.images.append(image_bytes)
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


def test_vlm_retry_is_format_repair_of_previous_judgment_only() -> None:
    invalid_pass = (
        '{"status":"pass","issues":[],"preserve":[],"explanation":"remove me"}'
    )
    model = _VisionModel([invalid_pass, '{"status":"pass","issues":[],"preserve":[]}'])
    critic = VLMCritic(model, _UnusedRenderer())  # type: ignore[arg-type]

    feedback = critic.critique_attempt(_critic_input())

    assert feedback.status == "pass"
    assert len(feedback.model_calls) == 2
    assert feedback.model_calls[0].validation_success is False
    assert feedback.model_calls[1].validation_success is True
    repair_prompt = model.prompts[1]
    assert "Do not re-evaluate" in repair_prompt
    assert invalid_pass in json.loads(
        repair_prompt.split("<previous_response_json>\n", 1)[1].split(
            "\n</previous_response_json>", 1
        )[0]
    )
    assert "<user_instruction_json>" not in repair_prompt
    assert model.images[0] == b"png-evidence"
    assert model.images[1] != model.images[0]
    assert model.images[1].startswith(b"\x89PNG")


def test_vlm_format_repair_cannot_change_judgment_fields() -> None:
    contradictory = '{"status":"pass","issues":[],"preserve":["keep circle"]}'
    model = _VisionModel([contradictory, '{"status":"pass","issues":[],"preserve":[]}'])
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
    assert "format repair" in str(captured.value)


def test_runtime_response_schema_uses_status_conditions_from_packaged_schema() -> None:
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "svg_agentic_slm"
        / "agents"
        / "critic_output.schema.json"
    )
    packaged_schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert _CRITIC_RESPONSE_FORMAT["schema"] == packaged_schema
    assert len(packaged_schema["allOf"]) == 2
    assert packaged_schema["allOf"][0]["then"]["properties"]["preserve"] == {
        "maxItems": 0
    }
    assert packaged_schema["allOf"][1]["then"]["properties"]["issues"] == {
        "minItems": 1
    }
