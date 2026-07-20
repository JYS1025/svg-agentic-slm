"""Tests for Generator initial and revision contracts."""

from __future__ import annotations

from svg_agentic_slm.agents.generator import GeneratorAgent
from svg_agentic_slm.agents.schemas import (
    CriticFeedback,
    CriticFeedbackEvent,
    GenerationRequest,
)
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.rag.schemas import RetrievedExample


class _RecordingBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.calls.append((prompt, kwargs))
        return ModelResponse(
            text=self.responses.pop(0),
            model_id="fake-model",
            model_revision="revision",
        )


def test_generate_extracts_svg_and_preserves_context_provenance() -> None:
    backend = _RecordingBackend(["Here is the result:\n```svg\n<svg><circle/></svg>\n```"])
    generator = GeneratorAgent(backend)
    context = [
        RetrievedExample(
            content="<svg><rect/></svg>",
            description="A square",
            source="corpus:item-1",
            item_id="rag-1",
            score=0.9,
            score_kind="cosine_similarity",
        )
    ]

    output = generator.generate(
        GenerationRequest(instruction="Draw a circle.", run_id="run-1"),
        context=context,
    )

    assert output.status == "succeeded"
    assert output.svg == "<svg><circle/></svg>"
    assert output.raw_output.startswith("Here is")
    assert output.context_item_ids == ["rag-1"]
    assert "corpus:item-1" in backend.calls[0][0]
    assert output.model_calls[0].prompt == backend.calls[0][0]
    assert output.model_calls[0].system_prompt


def test_revise_links_parent_and_feedback() -> None:
    backend = _RecordingBackend(
        [
            '<svg><circle cx="1"/></svg>',
            '<svg><circle cx="5"/></svg>',
        ]
    )
    generator = GeneratorAgent(backend)
    request = GenerationRequest(instruction="Center a circle.")
    initial = generator.generate(request)
    event = CriticFeedbackEvent(
        feedback_id="feedback-1",
        target_attempt_id=initial.attempt_id,
        feedback=CriticFeedback(
            score=4.0,
            issues=["Circle is off-center."],
            suggestions=["Move the circle to the center."],
        ),
    )

    revised = generator.revise(request, initial, event)

    assert revised.mode == "revision"
    assert revised.parent_attempt_id == initial.attempt_id
    assert revised.trigger_feedback_id == "feedback-1"
    assert "Circle is off-center." in backend.calls[1][0]


def test_generate_reports_svg_extraction_failure() -> None:
    generator = GeneratorAgent(_RecordingBackend(["No SVG was produced."]))

    output = generator.generate(GenerationRequest(instruction="Draw."))

    assert output.status == "failed"
    assert output.svg == ""
    assert output.error == "svg_extraction_failed"
