"""Evaluator class for scoring generated SVG artifact bundles."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from svg_agentic_slm.artifacts.generation import GenerationArtifactRecord
from svg_agentic_slm.eval.metrics import (
    compute_generation_latency,
    compute_render_success_rate,
    compute_simple_instruction_alignment,
    compute_svg_validity_rate,
)
from svg_agentic_slm.eval.schemas import EvaluationResult

if TYPE_CHECKING:
    from svg_agentic_slm.agents.orchestrator import SVGGenerationOrchestrator
    from svg_agentic_slm.data.schemas import TextToSVGExample
    from svg_agentic_slm.svg.base import BaseValidator

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = {
    "svg_validity_rate",
    "render_success_rate",
    "generation_latency",
    "simple_instruction_alignment",
}


class Evaluator:
    """Evaluates SVG generation quality from stored artifacts.

    Args:
        validator: SVG validator for metric computation.

    TODO: Add batch evaluation support.
    TODO: Add parallel evaluation.
    TODO: Add progress reporting.
    """

    def __init__(
        self,
        validator: BaseValidator,
        orchestrator: SVGGenerationOrchestrator | None = None,
    ) -> None:
        self._validator = validator
        self._orchestrator = orchestrator

    def evaluate_artifacts(
        self,
        artifacts: list[GenerationArtifactRecord],
        max_samples: int | None = None,
        metrics: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate previously generated artifact bundles."""
        selected_metrics = set(metrics) if metrics is not None else SUPPORTED_METRICS
        unsupported_metrics = selected_metrics - SUPPORTED_METRICS
        if unsupported_metrics:
            names = ", ".join(sorted(unsupported_metrics))
            raise ValueError(f"Unsupported evaluation metrics: {names}")
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError("eval.max_samples must be non-negative or null.")
            artifacts = artifacts[:max_samples]

        logger.info("Evaluating %d artifact bundles.", len(artifacts))
        if not artifacts:
            return EvaluationResult(
                num_samples=0,
                metadata={
                    "evaluation_mode": "artifacts",
                    "status": "empty",
                    "computed_metrics": sorted(selected_metrics),
                    "render_attempt_count": 0,
                },
            )

        instructions: list[str] = []
        svg_outputs: list[str] = []
        render_results: list[bool] = []
        latencies: list[float] = []
        per_sample_results: list[dict[str, object]] = []
        outcome_counts: dict[str, int] = {}

        for artifact in artifacts:
            svg_content = artifact.svg_path.read_text(encoding="utf-8")
            validation = self._validator.validate(svg_content)
            render_metadata = artifact.metadata.get("render", {})
            if not isinstance(render_metadata, dict):
                render_metadata = {}
            render_enabled = bool(
                render_metadata.get(
                    "enabled",
                    artifact.runtime.get(
                        "enable_render",
                        "success" in render_metadata or artifact.render_path is not None,
                    ),
                )
            )
            render_success: bool | None = None
            if render_enabled:
                recorded_success = bool(
                    render_metadata.get("success", artifact.render_path is not None)
                )
                render_success = bool(
                    recorded_success
                    and artifact.render_path is not None
                    and artifact.render_path.is_file()
                )
                render_results.append(render_success)
            latency = _coerce_latency(artifact.metadata)

            instructions.append(artifact.instruction)
            svg_outputs.append(svg_content)
            latencies.append(latency)

            critic_types = [
                feedback.get("critic_type", "unknown")
                for feedback in artifact.critic_feedback
            ]
            outcome = artifact.outcome or "unknown"
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            per_sample_results.append(
                {
                    "instruction": artifact.instruction,
                    "svg_path": str(artifact.svg_path),
                    "metadata_path": str(artifact.metadata_path),
                    "schema_version": artifact.schema_version,
                    "run_id": artifact.run_id,
                    "outcome": artifact.outcome,
                    "stop_reason": artifact.stop_reason,
                    "attempt_count": len(artifact.attempts),
                    "render_path": str(artifact.render_path) if artifact.render_path else None,
                    "is_valid": validation.is_valid,
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings,
                    "render_success": render_success,
                    "generation_latency_seconds": latency,
                    "critic_types": critic_types,
                    "generated_at_utc": artifact.generated_at_utc,
                }
            )

        return EvaluationResult(
            num_samples=len(artifacts),
            svg_validity_rate=(
                compute_svg_validity_rate(svg_outputs, self._validator)
                if "svg_validity_rate" in selected_metrics
                else 0.0
            ),
            render_success_rate=(
                compute_render_success_rate(render_results)
                if "render_success_rate" in selected_metrics
                else 0.0
            ),
            avg_generation_latency=(
                compute_generation_latency(latencies)
                if "generation_latency" in selected_metrics
                else 0.0
            ),
            avg_instruction_alignment=(
                compute_simple_instruction_alignment(instructions, svg_outputs)
                if "simple_instruction_alignment" in selected_metrics
                else 0.0
            ),
            per_sample_results=per_sample_results,
            metadata={
                "evaluation_mode": "artifacts",
                "artifact_count": len(artifacts),
                "computed_metrics": sorted(selected_metrics),
                "render_attempt_count": len(render_results),
                "outcome_counts": outcome_counts,
                "latency_source": "artifact.metadata.timing.generation_latency_seconds",
            },
        )

    def evaluate(
        self,
        examples: list[TextToSVGExample],
        max_samples: int | None = None,
    ) -> EvaluationResult:
        """Run evaluation on a list of examples.

        Args:
            examples: List of text-to-SVG examples to evaluate.
            max_samples: Maximum number of samples to evaluate.
                         None means evaluate all.

        Returns:
            Aggregated evaluation results.

        TODO: Implement actual evaluation loop with timing.
        """
        logger.warning(
            "Dataset-backed evaluation is not yet implemented in this runtime. "
            "Use evaluate_artifacts() with generated outputs instead."
        )
        return EvaluationResult(
            num_samples=len(examples[:max_samples] if max_samples is not None else examples),
            metadata={
                "status": "not_implemented",
                "evaluation_mode": "dataset",
            },
        )


def _coerce_latency(metadata: dict[str, object]) -> float:
    timing = metadata.get("timing", {})
    if not isinstance(timing, dict):
        return 0.0
    raw_value = timing.get("generation_latency_seconds", 0.0)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return 0.0
