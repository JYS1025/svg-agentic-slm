"""SVG generator agent implementation.

Uses a model backend to generate SVG code from text instructions.
The generator does not know about critic feedback, RAG retrieval,
or orchestration — it only generates SVG from a prompt.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from svg_agentic_slm.agents.base import BaseGenerator
from svg_agentic_slm.agents.schemas import GenerationRequest
from svg_agentic_slm.prompts.system_prompts import get_svg_generator_system_prompt
from svg_agentic_slm.prompts.text_to_svg import build_text_to_svg_prompt

if TYPE_CHECKING:
    from svg_agentic_slm.models.base import BaseModelBackend
    from svg_agentic_slm.rag.schemas import RetrievedExample

logger = logging.getLogger(__name__)


class GeneratorAgent(BaseGenerator):
    """SVG generator agent backed by a language model.

    Args:
        model_backend: The model backend to use for generation.
    """

    def __init__(self, model_backend: BaseModelBackend) -> None:
        self._model = model_backend

    @property
    def name(self) -> str:
        return "GeneratorAgent"

    def generate(
        self,
        request: GenerationRequest,
        context: str | None = None,
    ) -> str:
        """Generate SVG from a text instruction.

        Args:
            request: The generation request.
            context: Optional pre-formatted context string.

        Returns:
            Generated SVG string.

        TODO: Integrate system prompt with model chat template.
        TODO: Add SVG extraction from model output.
        TODO: Add retry logic for malformed outputs.
        """
        logger.info("Generating SVG for: %s", request.instruction[:80])

        # Build prompt
        prompt = build_text_to_svg_prompt(
            instruction=request.instruction,
            retrieved_examples=None,  # TODO: pass actual retrieved examples
        )

        # Prepend system prompt
        system_prompt = get_svg_generator_system_prompt()
        full_prompt = f"{system_prompt}\n\n{prompt}"

        # Generate
        generated_text = self._model.generate(
            full_prompt,
            **request.config_overrides,
        )

        # TODO: Extract SVG from generated text using svg.normalizer.extract_svg_from_text
        return generated_text
