"""SVG Generator implementation with initial and revision operations."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from svg_agentic_slm.agents.base import BaseGenerator
from svg_agentic_slm.agents.schemas import (
    CriticFeedback,
    CriticFeedbackEvent,
    GenerationRequest,
    GeneratorOutput,
    ModelCallTrace,
)
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.system_prompts import get_svg_generator_system_prompt
from svg_agentic_slm.prompts.text_to_svg import (
    INITIAL_PROMPT_VERSION,
    REVISION_PROMPT_VERSION,
    build_retrieval_context,
    build_revision_prompt,
    build_text_to_svg_prompt,
)
from svg_agentic_slm.svg.normalizer import extract_svg_from_text, normalize_svg

if TYPE_CHECKING:
    from svg_agentic_slm.models.base import BaseModelBackend
    from svg_agentic_slm.rag.schemas import RetrievedExample

logger = logging.getLogger(__name__)


class GeneratorAgent(BaseGenerator):
    """SVG generator agent backed by a language model.

    Args:
        model_backend: The model backend to use for generation.
    """

    def __init__(
        self,
        model_backend: BaseModelBackend,
        *,
        max_svg_length: int = 8192,
        max_context_characters: int = 12000,
    ) -> None:
        if max_svg_length <= 0:
            raise ValueError("max_svg_length must be positive.")
        if max_context_characters < 0:
            raise ValueError("max_context_characters must be non-negative.")
        self._model = model_backend
        self._max_svg_length = max_svg_length
        self._max_context_characters = max_context_characters

    @property
    def name(self) -> str:
        return "GeneratorAgent"

    def generate(
        self,
        request: GenerationRequest,
        context: list[RetrievedExample] | None = None,
    ) -> GeneratorOutput:
        """Generate an initial SVG attempt."""
        logger.info("Generating SVG for: %s", request.instruction[:80])
        selected_context, truncated_ids = self._select_context(context or [])
        prompt = build_text_to_svg_prompt(
            instruction=request.instruction,
            retrieved_examples=selected_context,
        )
        return self._invoke(
            request=request,
            prompt=prompt,
            mode="initial",
            prompt_version=INITIAL_PROMPT_VERSION,
            context=selected_context,
            truncated_context_item_ids=truncated_ids,
        )

    def revise(
        self,
        request: GenerationRequest,
        previous: GeneratorOutput,
        feedback: CriticFeedbackEvent,
        context: list[RetrievedExample] | None = None,
    ) -> GeneratorOutput:
        """Revise a previous attempt using feedback that targets it."""
        if feedback.target_attempt_id != previous.attempt_id:
            raise ValueError("Feedback target does not match the previous Generator attempt.")
        selected_context, truncated_ids = self._select_context(context or [])
        revision = build_revision_prompt(
            instruction=request.instruction,
            previous_svg=previous.svg,
            feedback=_format_feedback(feedback.feedback),
        )
        context_prefix = build_retrieval_context(selected_context)
        prompt = f"{context_prefix}\n\n{revision}" if selected_context else revision
        return self._invoke(
            request=request,
            prompt=prompt,
            mode="revision",
            prompt_version=REVISION_PROMPT_VERSION,
            context=selected_context,
            truncated_context_item_ids=truncated_ids,
            parent_attempt_id=previous.attempt_id,
            trigger_feedback_id=feedback.feedback_id,
        )

    def _invoke(
        self,
        *,
        request: GenerationRequest,
        prompt: str,
        mode: Literal["initial", "revision"],
        prompt_version: str,
        context: list[RetrievedExample],
        truncated_context_item_ids: list[str],
        parent_attempt_id: str | None = None,
        trigger_feedback_id: str | None = None,
    ) -> GeneratorOutput:
        attempt_id = f"attempt_{uuid4().hex}"
        model_call_id = f"model_call_{uuid4().hex}"
        if "system_prompt" in request.config_overrides:
            raise ValueError("system_prompt is owned by Generator and cannot be overridden.")
        system_prompt = get_svg_generator_system_prompt()
        response = self._model.generate(
            prompt,
            system_prompt=system_prompt,
            **request.config_overrides,
        )
        if isinstance(response, str):
            response = ModelResponse(text=response, model_id="legacy-backend")

        extracted = extract_svg_from_text(response.text)
        error: str | None = None
        svg = ""
        status: Literal["succeeded", "failed"] = "failed"
        if extracted is None:
            error = "svg_extraction_failed"
        else:
            svg = normalize_svg(extracted)
            if len(svg) > self._max_svg_length:
                error = "max_svg_length_exceeded"
                svg = ""
            else:
                status = "succeeded"

        return GeneratorOutput(
            attempt_id=attempt_id,
            mode=mode,
            svg=svg,
            raw_output=response.text,
            status=status,
            prompt_version=prompt_version,
            model_calls=[
                ModelCallTrace(
                    model_call_id=model_call_id,
                    response=response,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    generation_parameters=dict(request.config_overrides),
                )
            ],
            parent_attempt_id=parent_attempt_id,
            trigger_feedback_id=trigger_feedback_id,
            error=error,
            context_item_ids=[item.item_id for item in context],
            truncated_context_item_ids=truncated_context_item_ids,
            metadata={
                "run_id": request.run_id,
                "max_svg_length": self._max_svg_length,
                "max_context_characters": self._max_context_characters,
            },
        )

    def _select_context(
        self,
        items: list[RetrievedExample],
    ) -> tuple[list[RetrievedExample], list[str]]:
        selected: list[RetrievedExample] = []
        truncated_ids: list[str] = []
        remaining = self._max_context_characters

        for item in items:
            overhead = len(item.description) + len(item.source) + 128
            available_content = remaining - overhead
            if available_content <= 0:
                truncated_ids.append(item.item_id)
                continue

            if len(item.content) <= available_content:
                selected.append(item)
                remaining -= overhead + len(item.content)
                continue

            selected.append(replace(item, content=item.content[:available_content]))
            truncated_ids.append(item.item_id)
            remaining = 0

        return selected, truncated_ids


def _format_feedback(feedback: CriticFeedback) -> str:
    issues = "\n".join(f"- {issue}" for issue in feedback.issues) or "- None"
    suggestions = "\n".join(f"- {suggestion}" for suggestion in feedback.suggestions) or "- None"
    return (
        f"Score: {feedback.score}\n"
        f"Valid: {feedback.is_valid}\n"
        f"Matches instruction: {feedback.matches_instruction}\n"
        f"Issues:\n{issues}\n"
        f"Suggestions:\n{suggestions}"
    )
