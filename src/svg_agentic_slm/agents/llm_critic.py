"""LLM-based SVG critic.

Uses a language model to evaluate SVG quality and instruction
alignment. More flexible than rule-based critique but requires
model inference.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from svg_agentic_slm.agents.base import BaseCritic
from svg_agentic_slm.agents.schemas import CriticFeedback, validate_critic_feedback
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.critic_prompts import (
    CRITIC_PROMPT_VERSION,
    build_critic_prompt,
)
from svg_agentic_slm.prompts.system_prompts import get_svg_critic_system_prompt

if TYPE_CHECKING:
    from svg_agentic_slm.models.base import BaseModelBackend

logger = logging.getLogger(__name__)

LLM_CRITIC_VERSION = "llm-critic-v1"


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
        """Evaluate SVG using an LLM and parse its JSON response."""
        system_prompt = get_svg_critic_system_prompt()
        critic_prompt = build_critic_prompt(instruction, svg_content)
        logger.info("Evaluating SVG with LLM critic for: %s", instruction[:80])

        response = self._model.generate(
            critic_prompt,
            system_prompt=system_prompt,
        )
        if isinstance(response, str):
            response = ModelResponse(text=response, model_id="legacy-backend")
        if not isinstance(response, ModelResponse):
            raise TypeError("Model backend must return ModelResponse.")

        payload = _parse_json_object(response.text)
        required_fields = {
            "score",
            "is_valid",
            "matches_instruction",
            "issues",
            "suggestions",
        }
        missing_fields = sorted(required_fields - payload.keys())
        if missing_fields:
            raise ValueError(
                "LLM critic response is missing required field(s): "
                + ", ".join(missing_fields)
            )

        feedback = CriticFeedback(
            score=payload["score"],
            is_valid=payload["is_valid"],
            matches_instruction=payload["matches_instruction"],
            issues=payload["issues"],
            suggestions=payload["suggestions"],
            critic_type="llm",
            raw_response=response.text,
            critic_version=LLM_CRITIC_VERSION,
            model_id=response.model_id,
            model_revision=response.model_revision,
            prompt_version=CRITIC_PROMPT_VERSION,
        )
        return validate_critic_feedback(feedback)


def _parse_json_object(raw_response: str) -> dict:
    """Parse a JSON object, tolerating a surrounding markdown code fence."""
    text = raw_response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start < 0 or object_end < object_start:
        raise ValueError("LLM critic response must contain a JSON object.")

    try:
        payload = json.loads(text[object_start : object_end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("LLM critic response contains invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM critic response JSON must be an object.")
    return payload
