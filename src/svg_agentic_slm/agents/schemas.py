"""Schemas for agent communication.

Defines the data structures passed between agents and the orchestrator.
Using explicit schemas instead of raw dictionaries improves type safety
and makes the data flow self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    """A request to generate an SVG from a text instruction.

    Attributes:
        instruction: Natural language description of the desired SVG.
        task: Task identifier (default: 'text_to_svg').
        config_overrides: Optional overrides for generation config.
    """

    instruction: str
    task: str = "text_to_svg"
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """The result of the SVG generation pipeline.

    Attributes:
        instruction: The original instruction.
        generated_svg: The generated SVG string.
        is_valid: Whether the SVG passed validation.
        render_path: Path to the rendered image, if rendering was performed.
        critic_feedback: Feedback from the critic, if critic was used.
        revision_count: Number of revision rounds performed.
        metadata: Additional metadata about the generation process.
    """

    instruction: str
    generated_svg: str = ""
    is_valid: bool = False
    render_path: str | None = None
    critic_feedback: list[CriticFeedback] = field(default_factory=list)
    revision_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticFeedback:
    """Structured feedback from a critic agent.

    Attributes:
        score: Overall quality score (1-10).
        is_valid: Whether the critic considers the SVG valid.
        matches_instruction: Whether the SVG matches the instruction.
        issues: List of identified issues.
        suggestions: List of improvement suggestions.
        critic_type: Type of critic that produced this feedback.
        raw_response: Raw critic response, if from an LLM critic.
    """

    score: float = 0.0
    is_valid: bool = False
    matches_instruction: bool = False
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    critic_type: str = "unknown"
    raw_response: str | None = None
