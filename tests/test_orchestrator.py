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
    GenerationRequest,
    GenerationResult,
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
