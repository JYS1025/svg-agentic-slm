"""Evaluation loops for stored artifacts and dataset-backed generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from svg_agentic_slm.agents.schemas import GenerationRequest
from svg_agentic_slm.artifacts.generation import GenerationArtifactRecord
from svg_agentic_slm.eval.metrics import (
    compute_generation_latency,
    compute_render_success_rate,
    compute_simple_instruction_alignment,
    compute_svg_validity_rate,
)
from svg_agentic_slm.eval.policy import BenchmarkRunPolicy
from svg_agentic_slm.eval.schemas import EvaluationResult
from svg_agentic_slm.utils.atomic import atomic_write_text

if TYPE_CHECKING:
    from svg_agentic_slm.agents.orchestrator import SVGGenerationOrchestrator
    from svg_agentic_slm.agents.schemas import GenerationResult
    from svg_agentic_slm.data.schemas import TextToSVGExample
    from svg_agentic_slm.svg.base import BaseRenderer, BaseValidator

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = {
    "generation_success_rate",
    "svg_validity_rate",
    "render_success_rate",
    "generation_latency",
    "time_to_first_token",
    "tokens_per_second",
    "simple_instruction_alignment",
}


class Evaluator:
    """Evaluate stored artifacts or generate predictions for benchmark records."""

    def __init__(
        self,
        validator: BaseValidator,
        orchestrator: SVGGenerationOrchestrator | None = None,
        renderer: BaseRenderer | None = None,
        prediction_output_dir: str | Path | None = None,
        render_output_dir: str | Path | None = None,
    ) -> None:
        self._validator = validator
        self._orchestrator = orchestrator
        self._renderer = renderer
        self._prediction_output_dir = (
            Path(prediction_output_dir) if prediction_output_dir is not None else None
        )
        self._render_output_dir = (
            Path(render_output_dir) if render_output_dir is not None else None
        )

    def evaluate_artifacts(
        self,
        artifacts: list[GenerationArtifactRecord],
        max_samples: int | None = None,
        metrics: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate previously generated artifact bundles."""
        selected_metrics = _resolve_metrics(metrics)
        artifacts = _limit_samples(artifacts, max_samples)

        logger.info("Evaluating %d artifact bundles.", len(artifacts))
        if not artifacts:
            return _empty_result("artifacts", selected_metrics)

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
                feedback.get("critic_type", "unknown") for feedback in artifact.critic_feedback
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
                    "generation_succeeded": bool(svg_content.strip()),
                    "is_valid": validation.is_valid,
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings,
                    "render_success": render_success,
                    "generation_latency_seconds": latency,
                    "critic_types": critic_types,
                    "generated_at_utc": artifact.generated_at_utc,
                }
            )

        return _aggregate_result(
            mode="artifacts",
            selected_metrics=selected_metrics,
            instructions=instructions,
            svg_outputs=svg_outputs,
            render_results=render_results,
            latencies=latencies,
            ttfts=[],
            throughputs=[],
            generation_successes=[bool(svg.strip()) for svg in svg_outputs],
            per_sample_results=per_sample_results,
            validator=self._validator,
            metadata={
                "artifact_count": len(artifacts),
                "render_attempt_count": len(render_results),
                "outcome_counts": outcome_counts,
                "latency_source": "artifact.metadata.timing.generation_latency_seconds",
            },
        )

    def evaluate(
        self,
        examples: list[TextToSVGExample],
        max_samples: int | None = None,
        metrics: list[str] | None = None,
        run_policy: BenchmarkRunPolicy | None = None,
    ) -> EvaluationResult:
        """Generate and score one deterministic slice of a benchmark dataset."""
        selected_metrics = _resolve_metrics(metrics)
        examples = _limit_samples(examples, max_samples)
        policy = run_policy or BenchmarkRunPolicy()
        if not examples:
            result = _empty_result("dataset", selected_metrics)
            result.metadata["run_policy"] = policy.metadata()
            return result
        if self._orchestrator is None:
            raise RuntimeError("Dataset-backed evaluation requires an orchestrator.")

        for example in examples:
            policy.validate(example)

        instructions: list[str] = []
        svg_outputs: list[str] = []
        render_results: list[bool] = []
        latencies: list[float] = []
        ttfts: list[float] = []
        throughputs: list[float] = []
        generation_successes: list[bool] = []
        per_sample_results: list[dict[str, Any]] = []

        for index, example in enumerate(examples):
            metadata = example.metadata or {}
            sample_id = str(metadata["sample_id"])
            generation_result = self._orchestrator.run(
                GenerationRequest(
                    instruction=example.instruction,
                    task=example.task,
                    run_id=f"eval_{uuid4().hex}",
                )
            )
            succeeded = bool(
                generation_result.attempts
                and generation_result.attempts[-1].status == "succeeded"
            )
            svg_content = generation_result.generated_svg if succeeded else ""
            validation = self._validator.validate(svg_content)
            prediction_path = self._write_prediction(index, sample_id, svg_content)
            render_success, render_path, render_error = self._render_prediction(
                index, sample_id, svg_content, succeeded
            )
            if self._renderer is not None:
                render_results.append(render_success)

            latency = _coerce_latency(generation_result.metadata)
            model_metrics = _last_model_metrics(generation_result)
            if model_metrics["ttft"] is not None:
                ttfts.append(model_metrics["ttft"])
            if model_metrics["throughput"] is not None:
                throughputs.append(model_metrics["throughput"])

            instructions.append(example.instruction)
            svg_outputs.append(svg_content)
            latencies.append(latency)
            generation_successes.append(succeeded)
            per_sample_results.append(
                {
                    "benchmark_id": metadata["benchmark_id"],
                    "sample_id": sample_id,
                    "source_split": metadata.get("source_split"),
                    "difficulty": metadata.get("difficulty"),
                    "data_partition": metadata["data_partition"],
                    "memory_eligible": metadata["memory_eligible"],
                    "instruction": example.instruction,
                    "generation_succeeded": succeeded,
                    "is_valid": validation.is_valid,
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings,
                    "render_success": render_success if self._renderer is not None else None,
                    "render_error": render_error,
                    "generation_latency_seconds": latency,
                    "time_to_first_token_seconds": model_metrics["ttft"],
                    "tokens_per_second": model_metrics["throughput"],
                    "prompt_tokens": model_metrics["prompt_tokens"],
                    "completion_tokens": model_metrics["completion_tokens"],
                    "svg_path": str(prediction_path) if prediction_path else None,
                    "render_path": str(render_path) if render_path else None,
                    "run_id": generation_result.run_id,
                    "stop_reason": (
                        generation_result.attempts[-1].metadata.get("stop_reason")
                        if generation_result.attempts
                        else "no_attempt"
                    ),
                }
            )

        return _aggregate_result(
            mode="dataset",
            selected_metrics=selected_metrics,
            instructions=instructions,
            svg_outputs=svg_outputs,
            render_results=render_results,
            latencies=latencies,
            ttfts=ttfts,
            throughputs=throughputs,
            generation_successes=generation_successes,
            per_sample_results=per_sample_results,
            validator=self._validator,
            metadata={
                "run_policy": policy.metadata(),
                "render_attempt_count": len(render_results),
                "prediction_output_dir": (
                    str(self._prediction_output_dir) if self._prediction_output_dir else None
                ),
                "render_output_dir": (
                    str(self._render_output_dir) if self._render_output_dir else None
                ),
            },
        )

    def _write_prediction(self, index: int, sample_id: str, svg: str) -> Path | None:
        if self._prediction_output_dir is None or not svg:
            return None
        self._prediction_output_dir.mkdir(parents=True, exist_ok=True)
        path = self._prediction_output_dir / f"{index:04d}_{_safe_name(sample_id)}.svg"
        atomic_write_text(path, svg)
        return path

    def _render_prediction(
        self, index: int, sample_id: str, svg: str, succeeded: bool
    ) -> tuple[bool, Path | None, str | None]:
        if self._renderer is None:
            return False, None, None
        if not succeeded:
            return False, None, "generation_failed"
        if self._render_output_dir is None:
            raise RuntimeError("A render output directory is required when rendering evaluation.")
        path = self._render_output_dir / f"{index:04d}_{_safe_name(sample_id)}.png"
        render = self._renderer.render(svg, path, output_format="png")
        return render.success, render.output_path if render.success else None, render.error


def _resolve_metrics(metrics: list[str] | None) -> set[str]:
    selected = set(metrics) if metrics is not None else SUPPORTED_METRICS
    unsupported = selected - SUPPORTED_METRICS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported evaluation metrics: {names}")
    return selected


def _limit_samples(items: list[Any], max_samples: int | None) -> list[Any]:
    if max_samples is None:
        return items
    if max_samples < 0:
        raise ValueError("eval.max_samples must be non-negative or null.")
    return items[:max_samples]


def _empty_result(mode: str, selected_metrics: set[str]) -> EvaluationResult:
    return EvaluationResult(
        num_samples=0,
        metadata={
            "evaluation_mode": mode,
            "status": "empty",
            "computed_metrics": sorted(selected_metrics),
            "render_attempt_count": 0,
        },
    )


def _aggregate_result(
    *,
    mode: str,
    selected_metrics: set[str],
    instructions: list[str],
    svg_outputs: list[str],
    render_results: list[bool],
    latencies: list[float],
    ttfts: list[float],
    throughputs: list[float],
    generation_successes: list[bool],
    per_sample_results: list[dict[str, Any]],
    validator: BaseValidator,
    metadata: dict[str, Any],
) -> EvaluationResult:
    return EvaluationResult(
        num_samples=len(svg_outputs),
        generation_success_rate=(
            sum(generation_successes) / len(generation_successes)
            if "generation_success_rate" in selected_metrics and generation_successes
            else 0.0
        ),
        svg_validity_rate=(
            compute_svg_validity_rate(svg_outputs, validator)
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
        avg_time_to_first_token=(
            compute_generation_latency(ttfts)
            if "time_to_first_token" in selected_metrics
            else 0.0
        ),
        avg_tokens_per_second=(
            compute_generation_latency(throughputs)
            if "tokens_per_second" in selected_metrics
            else 0.0
        ),
        avg_instruction_alignment=(
            compute_simple_instruction_alignment(instructions, svg_outputs)
            if "simple_instruction_alignment" in selected_metrics
            else 0.0
        ),
        per_sample_results=per_sample_results,
        metadata={
            "evaluation_mode": mode,
            "computed_metrics": sorted(selected_metrics),
            **metadata,
        },
    )


def _coerce_latency(metadata: dict[str, object]) -> float:
    timing = metadata.get("timing", {})
    if not isinstance(timing, dict):
        return 0.0
    raw_value = timing.get("generation_latency_seconds", 0.0)
    return float(raw_value) if isinstance(raw_value, (int, float)) else 0.0


def _last_model_metrics(result: GenerationResult) -> dict[str, int | float | None]:
    if not result.attempts or not result.attempts[-1].model_calls:
        return {
            "ttft": None,
            "throughput": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
    response = result.attempts[-1].model_calls[-1].response
    return {
        "ttft": response.time_to_first_token_seconds,
        "throughput": response.tokens_per_second,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
    return safe.strip("-")[:96] or "sample"
