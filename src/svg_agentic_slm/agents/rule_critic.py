"""Rule-based SVG critic.

Evaluates SVG quality using deterministic rules rather than
an LLM. This is fast, reproducible, and does not require GPU.
"""

from __future__ import annotations

import logging

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import CriticFeedback
from svg_agentic_slm.svg.base import BaseValidator

logger = logging.getLogger(__name__)


class RuleBasedCritic(BaseCritic):
    """Critic that evaluates SVGs using deterministic rules.

    Uses the SVG validator and a set of heuristic rules
    to produce structured feedback.

    Args:
        validator: SVG validator instance.
    """

    def __init__(self, validator: BaseValidator) -> None:
        self._validator = validator

    @property
    def name(self) -> str:
        return "RuleBasedCritic"

    def critique(
        self,
        instruction: str,
        svg_content: str,
    ) -> CriticFeedback:
        """Evaluate SVG using rule-based checks.

        Args:
            instruction: Original text instruction.
            svg_content: Generated SVG to evaluate.

        Returns:
            Structured feedback from rule-based analysis.

        TODO: Add heuristic checks for instruction alignment.
        TODO: Add complexity checks (too simple / too complex).
        TODO: Add color usage analysis.
        """
        validation = self._validator.validate(svg_content)

        issues: list[str] = list(validation.errors)
        suggestions: list[str] = []

        if validation.warnings:
            issues.extend(validation.warnings)

        # Basic scoring: start at 10, deduct for issues
        score = max(0.0, 10.0 - len(validation.errors) * 3 - len(validation.warnings) * 1)

        return CriticFeedback(
            score=score,
            is_valid=validation.is_valid,
            matches_instruction=True,  # TODO: implement instruction matching
            issues=issues,
            suggestions=suggestions,
            critic_type="rule",
            critic_version="rule-svg-validation-v1",
        )
