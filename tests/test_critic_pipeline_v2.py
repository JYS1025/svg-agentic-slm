import json

import pytest

from svg_agentic_slm.agents.llm_critic import LLMCritic, compatibility_score
from svg_agentic_slm.agents.schemas import CriticFeedback, CriticFeedbackEvent, CriticInput, CriticIssue
from svg_agentic_slm.agents.orchestrator import _feedback_meets_acceptance
from svg_agentic_slm.models.base import BaseMultimodalModelBackend
from svg_agentic_slm.models.schemas import ImageInput, ModelResponse
from svg_agentic_slm.svg.gates import SmokeRenderGate
from svg_agentic_slm.svg.labeler import CriticLabeler, strip_reserved_labels
from svg_agentic_slm.svg.validator import SVGValidator


SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><defs><linearGradient id="p"><stop offset="0"/></linearGradient><linearGradient id="unused"/></defs><g><rect id="box" width="20" height="20" fill="url(#p)"/></g></svg>'


class FakeMultimodalBackend(BaseMultimodalModelBackend):
    def __init__(self, payload: dict): self.payload, self.calls = payload, []
    def load_model(self): pass
    def is_loaded(self): return True
    def generate(self, prompt: str, **kwargs): return ModelResponse("", "fake")
    def generate_multimodal(self, prompt: str, images: list[ImageInput], **kwargs):
        self.calls.append((prompt, images, kwargs))
        return ModelResponse(json.dumps(self.payload), "fake-vision", model_revision="r1")


def test_validator_emits_stable_diagnostic_code():
    result = SVGValidator().validate("<svg>")
    assert result.diagnostics[0].code == "xml_parse_error"


def test_labeler_is_deterministic_and_only_labels_reachable_resource():
    labeler = CriticLabeler()
    first = labeler.label(SVG, "attempt-1")
    second = labeler.label(SVG, "attempt-1")
    assert first == second
    assert "data-agent-id" not in SVG
    assert {ref.original_id for ref in first.elements.values()} >= {"box", "p"}
    assert "unused" not in {ref.original_id for ref in first.elements.values()}
    assert strip_reserved_labels(first.labeled_svg).find("data-agent-id") == -1


def test_smoke_render_gate_produces_decodable_png():
    result = SmokeRenderGate(timeout_seconds=5).evaluate(SVG)
    assert result.success
    assert result.png.startswith(b"\x89PNG")


def test_multimodal_critic_pass_is_schema_constrained_and_correlated():
    backend = FakeMultimodalBackend({"status": "pass", "issues": [], "preserve": []})
    labeling = CriticLabeler().label(SVG, "attempt-1")
    png = SmokeRenderGate().evaluate(SVG).png
    feedback = LLMCritic(backend).critique_attempt(CriticInput(
        "attempt-1", "Draw a box", SVG, ImageInput("image/png", png), labeling,
    ))
    assert feedback.status == "pass"
    assert feedback.score == 10.0
    assert feedback.matches_instruction
    assert backend.calls[0][1][0].data == png
    assert backend.calls[0][2]["response_format"]["type"] == "json_object"
    assert len(feedback.model_calls) == 1
    assert feedback.model_calls[0].prompt
    assert feedback.model_calls[0].system_prompt
    assert feedback.model_calls[0].response.text
    assert feedback.model_calls[0].validation_success


def test_unknown_target_retries_then_fails():
    issue = {"category": "style", "type": "fill_or_paint_server", "severity": "minor",
             "scope": "object", "target_ids": ["e9999"], "observed": "wrong color",
             "expected": "blue", "fix": "make it blue"}
    backend = FakeMultimodalBackend({"status": "revise", "issues": [issue], "preserve": []})
    labeling = CriticLabeler().label(SVG, "attempt-1")
    png = SmokeRenderGate().evaluate(SVG).png
    with pytest.raises(RuntimeError, match="Unknown target ID"):
        LLMCritic(backend).critique_attempt(CriticInput(
            "attempt-1", "Draw a blue box", SVG, ImageInput("image/png", png), labeling,
        ))
    assert len(backend.calls) == 2


def test_score_is_bounded_but_does_not_encode_acceptance():
    minor = CriticIssue("style", "fill_or_paint_server", "minor", "object", ["e0001"],
                        "wrong", "right", "fix")
    assert compatibility_score("pass", []) == 10.0
    assert compatibility_score("invalid", []) == 0.0
    assert compatibility_score("revise", [minor]) == 9.5
    many = [CriticIssue("shape", "contour_or_curve_geometry", "critical", "global", [],
                        "wrong", "right", "fix") for _ in range(3)]
    assert compatibility_score("revise", many) == 0.0
    feedback = CriticFeedback(
        score=9.5, is_valid=True, matches_instruction=False, status="revise",
        structured_issues=[minor], issues=["wrong"], suggestions=["fix"],
    )
    event = CriticFeedbackEvent("feedback-1", "attempt-1", feedback)
    assert not _feedback_meets_acceptance(event, 8.0)
