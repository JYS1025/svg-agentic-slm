"""SVG generation orchestrator.

Coordinates the end-to-end SVG generation pipeline including
RAG retrieval, generation, validation, rendering, and critique.

All dependencies are injected through the constructor, making
the orchestrator testable and backend-agnostic.
"""

from __future__ import annotations

import hashlib
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
    CriticTraceError,
    GenerationRequest,
    GenerationResult,
    GeneratorOutput,
    validate_critic_feedback,
)
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
        critic_labeler: CriticLabeler | None = None,
        smoke_render_gate: SmokeRenderGate | None = None,
        require_visual_evidence: bool = False,
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
        self._critic_labeler = critic_labeler or CriticLabeler()
        self._smoke_render_gate = smoke_render_gate or SmokeRenderGate(
            width=render_width,
            height=render_height,
        )
        self._require_visual_evidence = require_visual_evidence

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
            feedback = self._critique_attempt(request, current, validation)
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
        if self._renderer is not None:
            if current.status != "succeeded":
                render_error = "Render skipped because SVG generation failed."
            elif not validation.is_valid:
                render_error = "Render skipped because SVG validation failed."
            else:
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
        if current.status == "failed":
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
        current.metadata["stop_reason"] = _stop_reason(
            current=current,
            validation_is_valid=validation.is_valid,
            feedback_event=latest_feedback_event,
            acceptance_score=self._critic_acceptance_score,
            revision_count=result.revision_count,
            max_revisions=self._max_revisions,
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

    def _critique_attempt(
        self,
        request: GenerationRequest,
        attempt: GeneratorOutput,
        validation: SVGValidationResult,
    ) -> CriticFeedback:
        """Build immutable visual evidence and invoke the typed Critic boundary."""
        if self._critic is None:
            raise RuntimeError("Critic evidence requested without a configured Critic.")
        if not self._require_visual_evidence:
            return validate_critic_feedback(
                self._critic.critique(request.instruction, attempt.svg)
            )
        if not validation.is_valid:
            messages = list(validation.errors) or ["SVG validation failed."]
            return _invalid_evidence_feedback(
                attempt.attempt_id,
                stage="svg_validation_failure",
                messages=messages,
            )

        render_result = self._smoke_render_gate.evaluate(attempt.svg)
        if not render_result.success:
            messages = [
                f"{diagnostic.code}: {diagnostic.message}"
                for diagnostic in render_result.diagnostics
            ] or ["SVG smoke render failed."]
            return _invalid_evidence_feedback(
                attempt.attempt_id,
                stage="smoke_render_failure",
                messages=messages,
            )

        try:
            labeling = self._critic_labeler.label(attempt.svg, attempt.attempt_id)
        except Exception as exc:
            return _invalid_evidence_feedback(
                attempt.attempt_id,
                stage="labeling_failure",
                messages=[f"{type(exc).__name__}: {exc}"],
            )

        evidence = CriticEvidence(
            attempt_id=attempt.attempt_id,
            png=render_result.png,
            labeling=labeling,
            diagnostics=list(render_result.diagnostics),
            renderer="cairosvg",
            renderer_version=render_result.renderer_version,
            width=self._smoke_render_gate.width,
            height=self._smoke_render_gate.height,
        )
        attempt.critic_evidence = evidence
        critic_input = CriticInput(
            attempt_id=attempt.attempt_id,
            instruction=request.instruction,
            canonical_svg=attempt.svg,
            render_png=evidence.png,
            labeling=evidence.labeling,
            render_width=evidence.width,
            render_height=evidence.height,
        )
        try:
            feedback = validate_critic_feedback(self._critic.critique_attempt(critic_input))
        except CriticTraceError as exc:
            attempt.critic_error_calls.extend(exc.model_calls)
            raise
        _attach_evidence_provenance(feedback, evidence)
        return feedback

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
    }


def _invalid_evidence_feedback(
    attempt_id: str,
    *,
    stage: str,
    messages: list[str],
) -> CriticFeedback:
    observed = "; ".join(message for message in messages if message) or stage
    fix = "Produce a safe, renderable SVG before visual critique."
    issue = CriticIssue(
        category="validity",
        type=stage,
        severity="critical",
        scope="global",
        target_ids=[],
        observed=observed,
        expected="A structurally valid SVG with a successful canonical PNG smoke render.",
        fix=fix,
    )
    return validate_critic_feedback(
        CriticFeedback(
            score=0.0,
            is_valid=False,
            matches_instruction=False,
            issues=[observed],
            suggestions=[fix],
            critic_type="critic_evidence_gate",
            critic_version="critic-evidence-gate-v1",
            status="invalid",
            structured_issues=[issue],
            schema_version=2,
            metadata={
                "evidence_provenance": [
                    {
                        "attempt_id": attempt_id,
                        "stage": stage,
                        "success": False,
                        "diagnostics": list(messages),
                    }
                ]
            },
        )
    )


def _attach_evidence_provenance(
    feedback: CriticFeedback,
    evidence: CriticEvidence,
) -> None:
    record = {
        "attempt_id": evidence.attempt_id,
        "renderer": evidence.renderer,
        "renderer_version": evidence.renderer_version,
        "width": evidence.width,
        "height": evidence.height,
        "png_sha256": hashlib.sha256(evidence.png).hexdigest(),
        "labeled_svg_sha256": hashlib.sha256(
            evidence.labeling.labeled_svg.encode("utf-8")
        ).hexdigest(),
        "target_ids": sorted(evidence.labeling.elements),
    }
    raw_provenance = feedback.metadata.get("evidence_provenance", [])
    provenance = list(raw_provenance) if isinstance(raw_provenance, list) else []
    if record not in provenance:
        provenance.append(record)
    feedback.metadata["evidence_provenance"] = provenance


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
    if feedback.schema_version >= 2:
        return (
            feedback.status == "pass"
            and feedback.is_valid
            and feedback.matches_instruction
        )
    return feedback.is_valid and feedback.matches_instruction and feedback.score >= acceptance_score
