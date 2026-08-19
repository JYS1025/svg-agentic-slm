"""SVG Generator implementation with initial and revision operations."""

from __future__ import annotations

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
    build_retrieval_context,
    build_revision_prompt,
    build_text_to_svg_prompt,
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
    ) -> None:
        if max_svg_length <= 0:
            raise ValueError("max_svg_length must be positive.")
        if max_context_characters < 0:
            raise ValueError("max_context_characters must be non-negative.")
        if max_context_tokens is not None and max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative.")
        self._model = model_backend
        self._max_svg_length = max_svg_length
        self._max_context_characters = max_context_characters
        self._max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else max_context_characters // 4
        )

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
        context_selection = self._select_context(context or [])
        selected_context = context_selection.items
        previous_svg = _revision_input_svg(previous, feedback.feedback)
        revision = build_revision_prompt(
            instruction=request.instruction,
            previous_svg=previous_svg,
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
            context_selection=context_selection,
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
        context_selection: ContextSelection,
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


def _format_feedback(feedback: CriticFeedback) -> str:
    structured_issues = list(getattr(feedback, "structured_issues", []))
    if getattr(feedback, "status", None) == "revise" and structured_issues:
        issue_sections: list[str] = []
        for index, issue in enumerate(structured_issues, start=1):
            target_ids = ", ".join(issue.target_ids) or "GLOBAL_OR_MISSING_OBJECT"
            issue_sections.append(
                f"Issue {index}:\n"
                f"  Category: {issue.category}\n"
                f"  Type: {issue.type}\n"
                f"  Severity: {issue.severity}\n"
                f"  Scope: {issue.scope}\n"
                f"  Target IDs: {target_ids}\n"
                f"  Observed: {issue.observed}\n"
                f"  Expected: {issue.expected}\n"
                f"  Required fix: {issue.fix}"
            )
        preserve_items = list(getattr(feedback, "preserve", []))
        preserve = "\n".join(f"- {item}" for item in preserve_items) or "- None"
        return (
            f"Status: revise\n"
            f"Compatibility score: {feedback.score}\n"
            "Structured issues:\n"
            + "\n".join(issue_sections)
            + "\nPreserve constraints:\n"
            + preserve
            + "\nRevision constraints:\n"
            "- Target IDs refer to data-agent-id values in the labeled previous SVG.\n"
            "- Make the smallest changes necessary to address the listed issues.\n"
            "- Preserve unaffected elements, layout, styling, and all explicit preserve items.\n"
            "- Modify a target and only the adjacent or shared resource nodes required "
            "by its fix.\n"
            "- Do not emit any data-agent-id attributes in the revised SVG."
        )

    issues = "\n".join(f"- {issue}" for issue in feedback.issues) or "- None"
    suggestions = "\n".join(f"- {suggestion}" for suggestion in feedback.suggestions) or "- None"
    return (
        f"Score: {feedback.score}\n"
        f"Valid: {feedback.is_valid}\n"
        f"Matches instruction: {feedback.matches_instruction}\n"
        f"Issues:\n{issues}\n"
        f"Suggestions:\n{suggestions}"
    )


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
