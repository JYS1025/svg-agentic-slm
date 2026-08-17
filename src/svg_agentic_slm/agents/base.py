"""Abstract base classes for agents.

Defines the interfaces for generator and critic agents.
Concrete implementations depend on these interfaces, not on
specific model backends or processing logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from svg_agentic_slm.agents.schemas import (
    CriticFeedback,
    CriticInput,
    CriticFeedbackEvent,
    GenerationRequest,
    GeneratorOutput,
)

if TYPE_CHECKING:
    from svg_agentic_slm.rag.schemas import RetrievedExample


class BaseAgent(ABC):
    """Abstract base class for all agents in the pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name for logging and identification."""
        ...


class BaseGenerator(BaseAgent):
    """Abstract interface for SVG generator agents.

    A generator takes a text instruction (and optionally RAG context)
    and produces an SVG string.
    """

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        context: list[RetrievedExample] | None = None,
    ) -> GeneratorOutput:
        """Generate an initial SVG attempt.

        Args:
            request: The generation request containing the instruction.
            context: Optional typed items returned by RAG.

        Returns:
            Typed initial attempt and provenance.
        """
        ...

    def revise(
        self,
        request: GenerationRequest,
        previous: GeneratorOutput,
        feedback: CriticFeedbackEvent,
        context: list[RetrievedExample] | None = None,
    ) -> GeneratorOutput:
        """Revise a previous SVG attempt using correlated Critic feedback."""
        raise NotImplementedError(f"{self.name} does not implement revision.")


class BaseCritic(BaseAgent):
    """Abstract interface for SVG critic agents.

    A critic evaluates a generated SVG against the original instruction
    and provides structured feedback.
    """

    @abstractmethod
    def critique(
        self,
        instruction: str,
        svg_content: str,
    ) -> CriticFeedback:
        """Evaluate an SVG against an instruction.

        Args:
            instruction: The original text instruction.
            svg_content: The generated SVG string to evaluate.

        Returns:
            Structured critic feedback.
        """
        ...

    def critique_attempt(self, value: CriticInput) -> CriticFeedback:
        """Evaluate correlated evidence; legacy critics receive canonical SVG only."""
        return self.critique(value.instruction, value.canonical_svg)
