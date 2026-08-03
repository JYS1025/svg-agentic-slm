"""SVG generation orchestrator.

Coordinates the end-to-end SVG generation pipeline including
RAG retrieval, generation, validation, rendering, and critique.

All dependencies are injected through the constructor, making
the orchestrator testable and backend-agnostic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from svg_agentic_slm.agents.schemas import (
    CriticEvidence,
    CriticFeedback,
    CriticFeedbackEvent,
    CriticInput,
    CriticIssue,
    GenerationRequest,
    GenerationResult,
    GeneratorOutput,
    validate_critic_feedback,
)
from svg_agentic_slm.models.schemas import ImageInput
from svg_agentic_slm.svg.gates import SmokeRenderGate
from svg_agentic_slm.svg.labeler import CriticLabeler

if TYPE_CHECKING:
    from svg_agentic_slm.agents.base import BaseCritic, BaseGenerator
    from svg_agentic_slm.agents.rag_agent import RAGAgent
    from svg_agentic_slm.rag.schemas import RetrievedExample
    from svg_agentic_slm.svg.base import BaseRenderer, BaseValidator
    from svg_agentic_slm.svg.schemas import SVGValidationResult

logger = logging.getLogger(__name__)


class SVGGenerationOrchestrator:
    """Orchestrates the SVG generation pipeline.

    Coordinates the following steps:
    1. Receive user prompt / GenerationRequest.
    2. (Optional) Retrieve similar examples via RAG.
    3. Call the generator agent to produce SVG.
    4. Validate the generated SVG.
    5. (Optional) Render SVG to raster image.
    6. (Optional) Call critic for feedback.
    7. (Optional) Revise based on critic feedback.
    8. Return GenerationResult.

    All components are injected, not instantiated internally.

    Args:
        generator: The SVG generator agent.
        validator: The SVG validator.
        renderer: The SVG renderer (optional).
        critic: The SVG critic agent (optional).
        rag_agent: The RAG retrieval agent (optional).
        max_revisions: Maximum number of critic-driven revision rounds.
        output_dir: Directory for saving outputs.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        validator: BaseValidator,
        renderer: BaseRenderer | None = None,
        critic: BaseCritic | None = None,
        rag_agent: RAGAgent | None = None,
        max_revisions: int = 2,
        output_dir: str | Path = "./outputs/generations",
        render_output_path: str | Path | None = None,
        render_width: int = 256,
        render_height: int = 256,
        render_format: str = "png",
        critic_acceptance_score: float = 8.0,
        smoke_render_gate: SmokeRenderGate | None = None,
        critic_labeler: CriticLabeler | None = None,
        critic_error_policy: str = "fail_closed",
    ) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions must be non-negative.")
        if not 0.0 <= critic_acceptance_score <= 10.0:
            raise ValueError("critic_acceptance_score must be between 0 and 10.")
        self._generator = generator
        self._validator = validator
        self._renderer = renderer
        self._critic = critic
        self._rag_agent = rag_agent
        self._max_revisions = max_revisions
        self._output_dir = Path(output_dir)
        self._render_output_path = Path(render_output_path) if render_output_path else None
        self._render_width = render_width
        self._render_height = render_height
        self._render_format = render_format
        self._critic_acceptance_score = critic_acceptance_score
        self._smoke_render_gate = smoke_render_gate or SmokeRenderGate(render_width, render_height)
        self._critic_labeler = critic_labeler or CriticLabeler()
        if critic_error_policy not in {"fail_closed", "fail_open"}:
            raise ValueError("critic_error_policy must be fail_closed or fail_open.")
        self._critic_error_policy = critic_error_policy

    def run(
        self,
        request: GenerationRequest,
        *,
        on_generator_input: Callable[[GenerationRequest, list[RetrievedExample]], None]
        | None = None,
    ) -> GenerationResult:
        """Run the full SVG generation pipeline.

        Args:
            request: The generation request.
            on_generator_input: Optional observer called with the exact typed
                request and retrieved context before the initial Generator call.

        Returns:
            The generation result with SVG, validation, and feedback.

        """
        logger.info("Starting generation pipeline for: %s", request.instruction[:80])
        started_at = time.perf_counter()
        run_id = request.run_id or f"run_{uuid4().hex}"

        result = GenerationResult(instruction=request.instruction, run_id=run_id)
        result.metadata["request"] = {
            "task": request.task,
            "config_overrides": request.config_overrides,
            "run_id": run_id,
        }

        # Step 1: RAG retrieval (optional)
        context = []
        if self._rag_agent is not None:
            context = self._rag_agent.retrieve(request.instruction)
            logger.info("Retrieved %d RAG examples.", len(context))
        result.metadata["rag"] = {
            "enabled": self._rag_agent is not None,
            "retrieved_examples": len(context),
            "items": [
                {
                    "item_id": item.item_id,
                    "source": item.source,
                    "score": item.score,
                    "score_kind": item.score_kind,
                    "rank": item.rank,
                    "kind": item.kind,
                    "corpus_version": item.corpus_version,
                    "metadata": item.metadata,
                }
                for item in context
            ],
        }

        # Step 2: Generate SVG
        if on_generator_input is not None:
            on_generator_input(request, context)
        current = self._generator.generate(request, context=context)
        current = _coerce_generator_output(current)
        result.attempts.append(current)

        validation = self._validate_attempt(current)
        latest_feedback_event: CriticFeedbackEvent | None = None

        # Step 3: Critique and revise
        while self._critic is not None and current.status == "succeeded":
            if not validation.is_valid:
                feedback = _deterministic_feedback(validation.diagnostics)
            else:
                smoke = self._smoke_render_gate.evaluate(current.svg)
                if not smoke.success:
                    feedback = _deterministic_feedback(smoke.diagnostics)
                else:
                    labeling = self._critic_labeler.label(current.svg, current.attempt_id)
                    current.critic_evidence = CriticEvidence(
                        attempt_id=current.attempt_id, png=smoke.png, labeling=labeling,
                        diagnostics=smoke.diagnostics, renderer_version=smoke.renderer_version,
                        width=self._render_width, height=self._render_height,
                    )
                    critic_input = CriticInput(
                        attempt_id=current.attempt_id, instruction=request.instruction,
                        canonical_svg=current.svg, render_png=ImageInput("image/png", smoke.png),
                        labeling=labeling, render_width=self._render_width,
                        render_height=self._render_height,
                    )
                    try:
                        feedback = validate_critic_feedback(self._critic.critique_attempt(critic_input))
                    except RuntimeError as exc:
                        current.critic_error_calls = list(getattr(exc, "model_calls", []))
                        result.metadata["critic_error"] = {"type": type(exc).__name__, "message": str(exc),
                                                           "policy": self._critic_error_policy, "score": 0.0}
                        if self._critic_error_policy == "fail_open":
                            break
                        current.metadata["outcome"] = "rejected"
                        current.metadata["stop_reason"] = "critic_error"
                        break
            latest_feedback_event = CriticFeedbackEvent(
                feedback_id=f"feedback_{uuid4().hex}",
                target_attempt_id=current.attempt_id,
                feedback=feedback,
            )
            result.critic_feedback.append(feedback)
            result.feedback_events.append(latest_feedback_event)
            logger.info("Critic feedback: score=%.1f", feedback.score)

            if (
                validation.is_valid
                and _feedback_meets_acceptance(
                    latest_feedback_event,
                    self._critic_acceptance_score,
                )
            ) or result.revision_count >= self._max_revisions:
                break

            current.metadata["outcome"] = "rejected"
            current.metadata["stop_reason"] = "critic_revision_requested"
            current = self._generator.revise(
                request,
                previous=current,
                feedback=latest_feedback_event,
                context=context,
            )
            current = _coerce_generator_output(current)
            result.attempts.append(current)
            result.revision_count += 1
            validation = self._validate_attempt(current)

        result.generated_svg = current.svg
        result.is_valid = validation.is_valid
        result.metadata["validation"] = _validation_metadata(validation)

        # Step 4: Render (optional)
        render_success = False
        render_error: str | None = None
        if self._renderer is not None and current.status == "succeeded":
            render_path = self._render_output_path or (self._output_dir / "render.png")
            render_result = self._renderer.render(
                current.svg,
                render_path,
                width=self._render_width,
                height=self._render_height,
                output_format=self._render_format,
            )
            render_success = render_result.success
            render_error = render_result.error
            if render_result.success and render_result.output_path is not None:
                result.render_path = str(render_result.output_path)
            logger.info(
                "Render result: success=%s, path=%s, error=%s",
                render_result.success,
                render_result.output_path,
                render_result.error,
            )
        result.metadata["render"] = {
            "enabled": self._renderer is not None,
            "render_path": result.render_path,
            "planned_output_path": (
                str(self._render_output_path) if self._render_output_path else None
            ),
            "success": render_success,
            "error": render_error,
            "format": self._render_format,
            "width": self._render_width,
            "height": self._render_height,
        }

        result.metadata["critic"] = {
            "enabled": self._critic is not None,
            "feedback_count": len(result.critic_feedback),
            "acceptance_score": self._critic_acceptance_score,
        }
        critic_failed_closed = "critic_error" in result.metadata and self._critic_error_policy == "fail_closed"
        if critic_failed_closed:
            current.metadata["outcome"] = "rejected"
        elif current.status == "failed":
            current.metadata["outcome"] = "failed"
        elif validation.is_valid and (
            latest_feedback_event is None
            or _feedback_meets_acceptance(
                latest_feedback_event,
                self._critic_acceptance_score,
            )
        ):
            current.metadata["outcome"] = "accepted"
        else:
            current.metadata["outcome"] = "rejected"
        current.metadata["stop_reason"] = (
            "critic_error" if critic_failed_closed else _stop_reason(
                current=current, validation_is_valid=validation.is_valid,
                feedback_event=latest_feedback_event,
                acceptance_score=self._critic_acceptance_score,
                revision_count=result.revision_count, max_revisions=self._max_revisions,
            )
        )

        result.metadata["timing"] = {
            "generation_latency_seconds": round(time.perf_counter() - started_at, 6),
        }

        logger.info(
            "Pipeline complete. Valid=%s, Revisions=%d",
            result.is_valid,
            result.revision_count,
        )
        return result

    def _validate_attempt(self, attempt: GeneratorOutput) -> SVGValidationResult:
        validation = self._validator.validate(attempt.svg)
        attempt.metadata["validation"] = _validation_metadata(validation)
        logger.info(
            "Validation result: valid=%s, errors=%s",
            validation.is_valid,
            validation.errors,
        )
        return validation


def _validation_metadata(validation: SVGValidationResult) -> dict:
    return {
        "is_valid": validation.is_valid,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "has_svg_tag": validation.has_svg_tag,
        "has_closing_tag": validation.has_closing_tag,
        "is_well_formed_xml": validation.is_well_formed_xml,
        "diagnostics": [
            {"code": item.code, "message": item.message, "severity": item.severity,
             "line": item.line, "column": item.column}
            for item in validation.diagnostics
        ],
    }


def _coerce_generator_output(value: GeneratorOutput | str) -> GeneratorOutput:
    """Adapt legacy string-returning test or third-party Generators."""
    if isinstance(value, GeneratorOutput):
        return value
    if isinstance(value, str):
        return GeneratorOutput(
            attempt_id=f"attempt_{uuid4().hex}",
            mode="initial",
            svg=value,
            raw_output=value,
            status="succeeded",
            prompt_version="legacy",
        )
    raise TypeError("Generator must return GeneratorOutput.")


def _stop_reason(
    *,
    current: GeneratorOutput,
    validation_is_valid: bool,
    feedback_event: CriticFeedbackEvent | None,
    acceptance_score: float,
    revision_count: int,
    max_revisions: int,
) -> str:
    if current.status == "failed":
        return current.error or "generator_failed"
    if not validation_is_valid:
        return "validation_failed"
    if feedback_event is None:
        return "generator_only_complete"
    if feedback_event.feedback.status == "pass":
        return "critic_passed"
    if _feedback_meets_acceptance(feedback_event, acceptance_score):
        return "critic_acceptance_threshold_met"
    if revision_count >= max_revisions:
        return "max_revisions_reached"
    if not feedback_event.feedback.is_valid:
        return "critic_marked_invalid"
    if not feedback_event.feedback.matches_instruction:
        return "critic_instruction_mismatch"
    return "revision_stopped"


def _feedback_meets_acceptance(
    feedback_event: CriticFeedbackEvent,
    acceptance_score: float,
) -> bool:
    feedback = feedback_event.feedback
    if feedback.status is not None:
        return feedback.status == "pass" and feedback.is_valid
    return feedback.is_valid and feedback.matches_instruction and feedback.score >= acceptance_score


def _deterministic_feedback(diagnostics: list[object]) -> CriticFeedback:
    messages = [str(getattr(item, "message", item)) for item in diagnostics] or ["Unknown validity failure."]
    issue = CriticIssue(
        category="validity", type="xml_or_security_validation", severity="critical",
        scope="global", target_ids=[], observed="; ".join(messages),
        expected="A well-formed, static, self-contained and renderable SVG.",
        fix="Repair the reported validity error before visual revision.",
    )
    return validate_critic_feedback(CriticFeedback(
        score=0.0, is_valid=False, matches_instruction=False, issues=messages,
        suggestions=[issue.fix], critic_type="deterministic_gate",
        critic_version="deterministic-gate-v2", status="invalid",
        structured_issues=[issue], schema_version=2,
    ))
