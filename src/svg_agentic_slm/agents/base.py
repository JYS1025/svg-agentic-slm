"""Abstract base classes for agents.

Defines the interfaces for generator and critic agents.
Concrete implementations depend on these interfaces, not on
specific model backends or processing logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from svg_agentic_slm.agents.schemas import CriticFeedback, GenerationRequest


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
        context: str | None = None,
    ) -> str:
        """Generate an SVG string from a generation request.

        Args:
            request: The generation request containing the instruction.
            context: Optional additional context (e.g., from RAG retrieval).

        Returns:
            Generated SVG string.
        """
        ...


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
