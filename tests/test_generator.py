"""Tests for Generator initial and revision contracts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from svg_agentic_slm.agents.generator import (
    GeneratorAgent,
    _format_required_changes_json,
)
from svg_agentic_slm.agents.schemas import (
    CriticFeedback,
    CriticFeedbackEvent,
    CriticIssue,
    GenerationRequest,
)
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.text_to_svg import build_retrieval_context
from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.svg.validator import SVGValidator


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

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))


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
    assert output.metadata["context_usage"][0]["status"] == "fully_used"
    assert output.metadata["context_usage"][0]["token_count"] > 0
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
            is_valid=True,
            matches_instruction=False,
            issues=["Circle is off-center."],
            suggestions=["Move the circle to the center."],
        ),
    )

    revised = generator.revise(request, initial, event)

    assert revised.mode == "revision"
    assert revised.parent_attempt_id == initial.attempt_id
    assert revised.trigger_feedback_id == "feedback-1"
    assert "Circle is off-center." in backend.calls[1][0]
    assert "<original_instruction>" in backend.calls[1][0]
    assert "<previous_labeled_svg>" in backend.calls[1][0]
    assert "<required_changes_json>" in backend.calls[1][0]
    assert "Revision mode:" not in backend.calls[0][1]["system_prompt"]
    assert "Revision mode:" in backend.calls[1][1]["system_prompt"]


def test_validity_revision_uses_structural_repair_prompt_without_target_constraints() -> None:
    backend = _RecordingBackend(
        [
            "No SVG was produced.",
            '<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
        ]
    )
    generator = GeneratorAgent(backend)
    request = GenerationRequest(instruction="Draw a circle.")
    initial = generator.generate(request)
    issue = CriticIssue(
        category="validity",
        type="generator_output_failure",
        scope="global",
        target_ids=[],
        observed="No complete SVG document was produced.",
        expected="One complete, valid SVG document.",
        fix="Generate the entire SVG document.",
    )
    event = CriticFeedbackEvent(
        feedback_id="feedback-validity",
        target_attempt_id=initial.attempt_id,
        feedback=CriticFeedback(
            score=0.0,
            is_valid=False,
            matches_instruction=False,
            critic_type="critic_evidence_gate",
            status="invalid",
            structured_issues=[issue],
            schema_version=3,
        ),
    )

    revised = generator.revise(request, initial, event)

    prompt, kwargs = backend.calls[1]
    assert initial.status == "failed"
    assert "<previous_invalid_output>" in prompt
    assert "No SVG was produced." in prompt
    assert "<validity_feedback_json>" in prompt
    assert "<previous_labeled_svg>" not in prompt
    assert "Validity repair mode:" in kwargs["system_prompt"]
    assert "Revision mode:" not in kwargs["system_prompt"]
    assert "data-agent-id values only to locate" not in kwargs["system_prompt"]
    assert revised.status == "succeeded"
    assert revised.metadata["revision_kind"] == "validity"


def test_scorecard_feedback_preserves_detailed_targeted_revision_instructions() -> None:
    feedback = CriticFeedback(
        score=2.0,
        is_valid=True,
        matches_instruction=False,
        critic_type="scorecard",
        status="revise",
        structured_issues=[
            CriticIssue(
                category="layout",
                type="scale",
                scope="object",
                target_ids=["g0002"],
                observed="The smartphone is too small.",
                expected="The smartphone should be visually comparable to the laptop.",
                fix="Increase the overall size of g0002.",
            )
        ],
        schema_version=3,
        metadata={"score_threshold": 3.0},
    )

    changes = json.loads(_format_required_changes_json(feedback))

    assert changes == [
        {
            "category": "layout",
            "type": "scale",
            "scope": "object",
            "target_ids": ["g0002"],
            "observed": "The smartphone is too small.",
            "expected": "The smartphone should be visually comparable to the laptop.",
            "fix": "Increase the overall size of g0002.",
        }
    ]


@pytest.mark.parametrize(
    ("enable_revision_rag", "expected_in_revision"),
    [(False, False), (True, True)],
)
def test_revision_rag_is_configurable(
    enable_revision_rag: bool,
    expected_in_revision: bool,
) -> None:
    backend = _RecordingBackend(["<svg><circle/></svg>", "<svg><circle/></svg>"])
    generator = GeneratorAgent(
        backend,
        enable_revision_rag=enable_revision_rag,
    )
    request = GenerationRequest(instruction="Draw a circle.")
    context = [
        RetrievedExample(
            content="<svg><rect/></svg>",
            description="RAG marker for revision",
            source="corpus:revision-rag-marker",
            item_id="rag-revision",
        )
    ]
    initial = generator.generate(request, context=context)
    event = CriticFeedbackEvent(
        feedback_id="feedback-rag",
        target_attempt_id=initial.attempt_id,
        feedback=CriticFeedback(
            score=2.0,
            is_valid=True,
            matches_instruction=False,
            issues=["The circle is too small."],
            suggestions=["Increase the circle size."],
        ),
    )

    revised = generator.revise(request, initial, event, context=context)

    assert "corpus:revision-rag-marker" in backend.calls[0][0]
    assert (
        "corpus:revision-rag-marker" in backend.calls[1][0]
    ) is expected_in_revision
    assert revised.context_item_ids == (["rag-revision"] if expected_in_revision else [])


def test_generate_reports_svg_extraction_failure() -> None:
    generator = GeneratorAgent(_RecordingBackend(["No SVG was produced."]))

    output = generator.generate(GenerationRequest(instruction="Draw."))

    assert output.status == "failed"
    assert output.svg == ""
    assert output.error == "svg_extraction_failed"


def test_context_budget_uses_whole_svg_elements_and_records_usage() -> None:
    backend = _RecordingBackend(["<svg><path d='M0 0'/></svg>"])
    item = RetrievedExample(
        content=(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect id="kept" width="10" height="10"/>'
            '<circle id="dropped" cx="20" cy="20" r="5"/>'
            "</svg>"
        ),
        description="Two shapes",
        source="corpus:item",
        item_id="rag-elements",
    )
    one_element = replace(
        item,
        content=(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect id="kept" width="10" height="10"/>'
            "</svg>"
        ),
    )
    token_budget = backend.count_tokens(build_retrieval_context([one_element]))
    generator = GeneratorAgent(backend, max_context_tokens=token_budget)

    output = generator.generate(
        GenerationRequest(instruction="Draw a path."),
        context=[item],
    )

    prompt = backend.calls[0][0]
    context_svg_start = prompt.index('<svg xmlns="http://www.w3.org/2000/svg">')
    context_svg_end = prompt.index("</svg>", context_svg_start) + len("</svg>")
    selected_svg = prompt[context_svg_start:context_svg_end]
    assert SVGValidator().validate(selected_svg).is_valid
    assert 'id="kept"' in selected_svg
    assert 'id="dropped"' not in selected_svg
    assert output.context_item_ids == ["rag-elements"]
    assert output.truncated_context_item_ids == ["rag-elements"]
    assert output.metadata["context_token_count"] <= token_budget
    assert output.metadata["context_usage"] == [
        {
            "item_id": "rag-elements",
            "status": "partially_used",
            "token_count": output.metadata["context_token_count"],
            "included_element_count": 1,
            "total_element_count": 2,
        }
    ]


def test_context_budget_drops_whole_non_svg_item_without_character_slicing() -> None:
    backend = _RecordingBackend(["<svg/>"])
    generator = GeneratorAgent(backend, max_context_tokens=1)
    item = RetrievedExample(
        content="Never slice this correction text.",
        description="Correction",
        source="corpus:correction",
        item_id="rag-correction",
        kind="correction_pair",
    )

    output = generator.generate(
        GenerationRequest(instruction="Draw."),
        context=[item],
    )

    assert "Never slice" not in backend.calls[0][0]
    assert output.context_item_ids == []
    assert output.metadata["context_usage"][0]["status"] == "dropped"
    assert output.metadata["context_usage"][0]["token_count"] == 0


def test_nonempty_context_requires_backend_tokenizer() -> None:
    class BackendWithoutTokenizer:
        def generate(self, prompt: str, **kwargs) -> ModelResponse:
            return ModelResponse(text="<svg/>", model_id="fake")

    generator = GeneratorAgent(BackendWithoutTokenizer())  # type: ignore[arg-type]
    item = RetrievedExample(
        content="Correction text",
        description="Correction",
        source="corpus:correction",
        item_id="rag-correction",
        kind="correction_pair",
    )

    with pytest.raises(RuntimeError, match="tokenizer-based count_tokens"):
        generator.generate(GenerationRequest(instruction="Draw."), context=[item])
