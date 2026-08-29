"""SVG Generator implementation with initial and revision operations."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
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
    VALIDITY_REVISION_PROMPT_VERSION,
    build_retrieval_context,
    build_revision_prompt,
    build_text_to_svg_prompt,
    build_validity_revision_prompt,
)
from svg_agentic_slm.rag.context import ContextSelection, select_context_by_tokens
from svg_agentic_slm.svg.labeler import strip_reserved_labels
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
        max_context_tokens: int | None = None,
        enable_revision_rag: bool = False,
    ) -> None:
        if max_svg_length <= 0:
            raise ValueError("max_svg_length must be positive.")
        if max_context_characters < 0:
            raise ValueError("max_context_characters must be non-negative.")
        if max_context_tokens is not None and max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative.")
        if not isinstance(enable_revision_rag, bool):
            raise TypeError("enable_revision_rag must be a boolean.")
        self._model = model_backend
        self._max_svg_length = max_svg_length
        self._max_context_characters = max_context_characters
        self._max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else max_context_characters // 4
        )
        self._enable_revision_rag = enable_revision_rag

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
        context_selection = self._select_context(context or [])
        selected_context = context_selection.items
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
            context_selection=context_selection,
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
        revision_context = (context or []) if self._enable_revision_rag else []
        context_selection = self._select_context(revision_context)
        selected_context = context_selection.items
        validity_repair = (
            previous.status == "failed"
            or feedback.feedback.status == "invalid"
            or not feedback.feedback.is_valid
        )
        revision_input_truncated = False
        if validity_repair:
            previous_output, revision_input_truncated = _validity_revision_input(
                previous,
                maximum=self._max_svg_length,
            )
            revision = build_validity_revision_prompt(
                instruction=request.instruction,
                previous_output=previous_output,
                validity_feedback_json=_format_required_changes_json(feedback.feedback),
                previous_output_truncated=revision_input_truncated,
            )
            prompt_version = VALIDITY_REVISION_PROMPT_VERSION
        else:
            previous_svg = _revision_input_svg(previous, feedback.feedback)
            revision = build_revision_prompt(
                instruction=request.instruction,
                previous_svg=previous_svg,
                required_changes_json=_format_required_changes_json(feedback.feedback),
            )
            prompt_version = REVISION_PROMPT_VERSION
        context_prefix = build_retrieval_context(selected_context)
        prompt = f"{context_prefix}\n\n{revision}" if selected_context else revision
        return self._invoke(
            request=request,
            prompt=prompt,
            mode="revision",
            prompt_version=prompt_version,
            context=selected_context,
            context_selection=context_selection,
            parent_attempt_id=previous.attempt_id,
            trigger_feedback_id=feedback.feedback_id,
            validity_repair=validity_repair,
            revision_input_truncated=revision_input_truncated,
        )

    def _invoke(
        self,
        *,
        request: GenerationRequest,
        prompt: str,
        mode: Literal["initial", "revision"],
        prompt_version: str,
        context: list[RetrievedExample],
        context_selection: ContextSelection,
        parent_attempt_id: str | None = None,
        trigger_feedback_id: str | None = None,
        validity_repair: bool = False,
        revision_input_truncated: bool = False,
    ) -> GeneratorOutput:
        attempt_id = f"attempt_{uuid4().hex}"
        model_call_id = f"model_call_{uuid4().hex}"
        if "system_prompt" in request.config_overrides:
            raise ValueError("system_prompt is owned by Generator and cannot be overridden.")
        system_prompt = get_svg_generator_system_prompt(
            revision=mode == "revision",
            validity_repair=validity_repair,
        )
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
            try:
                cleaned_svg = strip_reserved_labels(extracted)
            except Exception:
                # Preserve the legacy normalization path for malformed model
                # output. The downstream validator remains authoritative.
                cleaned_svg = extracted
            svg = normalize_svg(cleaned_svg)
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
            truncated_context_item_ids=[
                item.item_id
                for item in context_selection.usage
                if item.status != "fully_used"
            ],
            metadata={
                "run_id": request.run_id,
                "max_svg_length": self._max_svg_length,
                "max_context_characters": self._max_context_characters,
                "max_context_tokens": self._max_context_tokens,
                "enable_revision_rag": self._enable_revision_rag,
                "revision_kind": (
                    "validity"
                    if validity_repair
                    else "targeted"
                    if mode == "revision"
                    else None
                ),
                "revision_input_truncated": revision_input_truncated,
                "context_token_count": context_selection.token_count,
                "context_usage": [asdict(item) for item in context_selection.usage],
            },
        )

    def _select_context(
        self,
        items: list[RetrievedExample],
    ) -> ContextSelection:
        if not items:
            return ContextSelection(items=[], usage=[], token_count=0)
        count_tokens = getattr(self._model, "count_tokens", None)
        if not callable(count_tokens):
            raise RuntimeError(
                "RAG context requires a model backend with tokenizer-based "
                "count_tokens support."
            )
        return select_context_by_tokens(
            items,
            max_tokens=self._max_context_tokens,
            count_tokens=count_tokens,
            format_context=build_retrieval_context,
        )


def _format_required_changes_json(feedback: CriticFeedback) -> str:
    structured_issues = list(getattr(feedback, "structured_issues", []))
    if structured_issues:
        changes = [asdict(issue) for issue in structured_issues]
    else:
        changes = _legacy_required_changes(feedback)
    return json.dumps(changes, ensure_ascii=False, indent=2)


def _legacy_required_changes(feedback: CriticFeedback) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    count = max(len(feedback.issues), len(feedback.suggestions))
    for index in range(count):
        observed = (
            feedback.issues[index]
            if index < len(feedback.issues)
            else "The reviewer requested a correction."
        )
        fix = (
            feedback.suggestions[index]
            if index < len(feedback.suggestions)
            else "Correct the observed problem."
        )
        changes.append(
            {
                "category": "general",
                "type": "review_feedback",
                "scope": "global",
                "target_ids": [],
                "observed": observed,
                "expected": "The SVG should satisfy the original instruction.",
                "fix": fix,
            }
        )
    return changes


def _revision_input_svg(previous: GeneratorOutput, feedback: CriticFeedback) -> str:
    """Select canonical or attempt-labeled SVG without changing legacy revisions."""
    structured_issues = list(getattr(feedback, "structured_issues", []))
    if getattr(feedback, "status", None) != "revise" or not structured_issues:
        return previous.svg

    evidence = getattr(previous, "critic_evidence", None)
    if evidence is None:
        raise ValueError("Structured revision feedback requires attempt Critic evidence.")
    if evidence.attempt_id != previous.attempt_id:
        raise ValueError("Critic evidence does not match the previous Generator attempt.")
    labeling = evidence.labeling
    if labeling.attempt_id != previous.attempt_id:
        raise ValueError("Critic labeling does not match the previous Generator attempt.")
    if not isinstance(labeling.labeled_svg, str) or not labeling.labeled_svg:
        raise ValueError("Critic labeling must contain a labeled SVG.")

    allowed_ids = set(labeling.elements)
    referenced_ids = {
        target_id
        for issue in structured_issues
        for target_id in issue.target_ids
    }
    unknown_ids = sorted(referenced_ids - allowed_ids)
    if unknown_ids:
        raise ValueError(
            "Structured revision feedback contains unknown target ID(s): "
            + ", ".join(unknown_ids)
        )
    return labeling.labeled_svg


def _validity_revision_input(
    previous: GeneratorOutput,
    *,
    maximum: int,
) -> tuple[str, bool]:
    """Return bounded invalid output evidence for a structural repair prompt."""
    candidate = previous.svg.strip() or previous.raw_output.strip()
    if len(candidate) <= maximum:
        return candidate, False
    return candidate[:maximum], True
