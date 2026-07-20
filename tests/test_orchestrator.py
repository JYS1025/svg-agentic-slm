"""Tests for the SVG generation orchestrator.

Verifies that the orchestrator can be instantiated with
stub/mock components and that the dependency injection
pattern works correctly.
"""

from __future__ import annotations

from svg_agentic_slm.agents.base import BaseCritic, BaseGenerator
from svg_agentic_slm.agents.orchestrator import SVGGenerationOrchestrator
from svg_agentic_slm.agents.schemas import (
    CriticFeedback,
    CriticFeedbackEvent,
    GenerationRequest,
    GenerationResult,
    GeneratorOutput,
)
from svg_agentic_slm.svg.base import BaseValidator
from svg_agentic_slm.svg.schemas import SVGValidationResult


class StubGenerator(BaseGenerator):
    """Stub generator for testing."""

    @property
    def name(self) -> str:
        return "StubGenerator"

    def generate(
        self,
        request: GenerationRequest,
        context: str | None = None,
    ) -> str:
        return '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'


class StubValidator(BaseValidator):
    """Stub validator for testing."""

    def validate(self, svg_content: str) -> SVGValidationResult:
        return SVGValidationResult(is_valid=True, has_svg_tag=True, has_closing_tag=True)


class StubCritic(BaseCritic):
    """Stub critic for testing."""

    @property
    def name(self) -> str:
        return "StubCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(score=8.0, is_valid=True, critic_type="stub")


class RevisingGenerator(BaseGenerator):
    """Typed Generator that records the feedback correlation it receives."""

    def __init__(self) -> None:
        self.revision_feedback: CriticFeedbackEvent | None = None

    @property
    def name(self) -> str:
        return "RevisingGenerator"

    def generate(
        self,
        request: GenerationRequest,
        context=None,
    ) -> GeneratorOutput:
        return GeneratorOutput(
            attempt_id="attempt-initial",
            mode="initial",
            svg='<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
            raw_output="<svg><rect/></svg>",
            status="succeeded",
            prompt_version="test-v1",
        )

    def revise(
        self,
        request: GenerationRequest,
        previous: GeneratorOutput,
        feedback: CriticFeedbackEvent,
        context=None,
    ) -> GeneratorOutput:
        self.revision_feedback = feedback
        return GeneratorOutput(
            attempt_id="attempt-revised",
            mode="revision",
            svg='<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
            raw_output="<svg><circle/></svg>",
            status="succeeded",
            prompt_version="test-revision-v1",
            parent_attempt_id=previous.attempt_id,
            trigger_feedback_id=feedback.feedback_id,
        )


class SequentialCritic(BaseCritic):
    """Request one revision and then accept it."""

    def __init__(self) -> None:
        self.scores = iter([4.0, 9.0])

    @property
    def name(self) -> str:
        return "SequentialCritic"

    def critique(self, instruction: str, svg_content: str) -> CriticFeedback:
        return CriticFeedback(
            score=next(self.scores),
            is_valid=True,
            matches_instruction=True,
            critic_type="sequential",
        )


def test_orchestrator_instantiation() -> None:
    """Test that the orchestrator can be created with stub components."""
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
    )
    assert orchestrator is not None


def test_orchestrator_with_all_components() -> None:
    """Test orchestrator with all optional components."""
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=StubCritic(),
    )
    assert orchestrator is not None


def test_orchestrator_run() -> None:
    """Test that the orchestrator can run a basic pipeline."""
    orchestrator = SVGGenerationOrchestrator(
        generator=StubGenerator(),
        validator=StubValidator(),
        critic=StubCritic(),
    )

    request = GenerationRequest(instruction="Draw a red square.")
    result = orchestrator.run(request)

    assert isinstance(result, GenerationResult)
    assert result.instruction == "Draw a red square."
    assert result.generated_svg != ""
    assert result.is_valid


def test_orchestrator_correlates_feedback_and_revision() -> None:
    generator = RevisingGenerator()
    orchestrator = SVGGenerationOrchestrator(
        generator=generator,
        validator=StubValidator(),
        critic=SequentialCritic(),
        max_revisions=2,
    )

    result = orchestrator.run(GenerationRequest(instruction="Draw a circle."))

    assert result.revision_count == 1
    assert len(result.attempts) == 2
    assert len(result.feedback_events) == 2
    assert generator.revision_feedback is result.feedback_events[0]
    assert result.feedback_events[0].target_attempt_id == "attempt-initial"
    assert result.attempts[1].parent_attempt_id == "attempt-initial"
    assert result.attempts[1].metadata["outcome"] == "accepted"
