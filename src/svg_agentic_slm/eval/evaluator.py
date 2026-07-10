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
    ) -> EvaluationResult:
        """Evaluate previously generated artifact bundles."""
        if max_samples is not None:
            artifacts = artifacts[:max_samples]

        logger.info("Evaluating %d artifact bundles.", len(artifacts))
        if not artifacts:
            return EvaluationResult(
                num_samples=0,
                metadata={"evaluation_mode": "artifacts", "status": "empty"},
            )

        instructions: list[str] = []
        svg_outputs: list[str] = []
        render_results: list[bool] = []
        latencies: list[float] = []
        per_sample_results: list[dict[str, object]] = []

        for artifact in artifacts:
            svg_content = artifact.svg_path.read_text(encoding="utf-8")
            validation = self._validator.validate(svg_content)
            render_metadata = artifact.metadata.get("render", {})
            render_success = bool(render_metadata.get("success", artifact.render_path is not None))
            latency = _coerce_latency(artifact.metadata)

            instructions.append(artifact.instruction)
            svg_outputs.append(svg_content)
            render_results.append(render_success)
            latencies.append(latency)

            critic_types = [
                feedback.get("critic_type", "unknown")
                for feedback in artifact.critic_feedback
            ]
            per_sample_results.append(
                {
                    "instruction": artifact.instruction,
                    "svg_path": str(artifact.svg_path),
                    "metadata_path": str(artifact.metadata_path),
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
            svg_validity_rate=compute_svg_validity_rate(svg_outputs, self._validator),
            render_success_rate=compute_render_success_rate(render_results),
            avg_generation_latency=compute_generation_latency(latencies),
            avg_instruction_alignment=compute_simple_instruction_alignment(instructions, svg_outputs),
            per_sample_results=per_sample_results,
            metadata={
                "evaluation_mode": "artifacts",
                "artifact_count": len(artifacts),
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
