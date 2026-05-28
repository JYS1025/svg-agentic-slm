"""LLM-based SVG critic.

Uses a language model to evaluate SVG quality and instruction
alignment. More flexible than rule-based critique but requires
model inference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import CriticFeedback
from svg_agentic_slm.prompts.critic_prompts import build_critic_prompt
from svg_agentic_slm.prompts.system_prompts import get_svg_critic_system_prompt

if TYPE_CHECKING:
    from svg_agentic_slm.models.base import BaseModelBackend

logger = logging.getLogger(__name__)


class LLMCritic(BaseCritic):
    """Critic that uses an LLM to evaluate SVGs.

    Args:
        model_backend: The model backend to use for critique.
    """

    def __init__(self, model_backend: BaseModelBackend) -> None:
        self._model = model_backend

    @property
    def name(self) -> str:
        return "LLMCritic"

    def critique(
        self,
        instruction: str,
        svg_content: str,
    ) -> CriticFeedback:
        """Evaluate SVG using an LLM.

        Args:
            instruction: Original text instruction.
            svg_content: Generated SVG to evaluate.

        Returns:
            Structured feedback parsed from LLM response.

        TODO: Implement LLM response parsing into CriticFeedback.
        TODO: Add retry logic for unparseable LLM responses.
        TODO: Consider using structured output / JSON mode.
        """
        system_prompt = get_svg_critic_system_prompt()
        critic_prompt = build_critic_prompt(instruction, svg_content)
        full_prompt = f"{system_prompt}\n\n{critic_prompt}"

        logger.info("[PLACEHOLDER] Would call LLM critic for instruction: %s", instruction[:80])

        # TODO: Implement actual LLM call and response parsing
        # raw_response = self._model.generate(full_prompt)
        # return self._parse_response(raw_response)

        return CriticFeedback(
            score=5.0,
            is_valid=True,
            matches_instruction=True,
            issues=["LLM critique not yet implemented."],
            suggestions=["Implement LLM response parsing."],
            critic_type="llm",
            raw_response=None,
        )
