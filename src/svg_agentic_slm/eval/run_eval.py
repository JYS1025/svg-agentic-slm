"""Entry point for running evaluations.

Provides a high-level function to load config, build components,
and run evaluation. Called by the CLI eval command.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from svg_agentic_slm.artifacts.generation import (
    GenerationArtifactRecord,
    list_generation_artifacts,
    load_generation_artifact,
)
from svg_agentic_slm.cli.overrides import merge_nested_dicts
from svg_agentic_slm.data.jsonl import read_jsonl
from svg_agentic_slm.data.schemas import TextToSVGExample
from svg_agentic_slm.eval.evaluator import Evaluator
from svg_agentic_slm.eval.policy import BenchmarkRunPolicy
from svg_agentic_slm.eval.schemas import EvaluationResult
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "generation_success_rate",
    "svg_validity_rate",
    "render_success_rate",
    "generation_latency",
    "time_to_first_token",
    "tokens_per_second",
]


def run_evaluation(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Load configuration and evaluate generated artifact bundles.

    Args:
        config_path: Path to the eval.yaml config file.
        overrides: Optional dictionary of config overrides.

    Returns:
        Evaluation results.

    Supported artifact sources:
    - directory of generated `.json` sidecars
    - single `.json` sidecar
    - single `.svg` file with matching `.json` sidecar
    """
    config = load_yaml_config(config_path)
    if overrides:
        config = merge_nested_dicts(config, overrides)
    eval_config = config.get("eval", {})

    metrics = eval_config.get("metrics")
    if metrics is None:
        metrics = DEFAULT_METRICS
    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics):
        raise ValueError("eval.metrics must be a list of metric names.")
    artifact_value = eval_config.get("artifact_path")
    dataset_value = eval_config.get("dataset_path")
    if artifact_value and dataset_value:
        raise ValueError("Configure exactly one of eval.artifact_path or eval.dataset_path.")

    if dataset_value:
        source_path = Path(dataset_value)
        logger.info("Running dataset-backed evaluation from: %s", source_path)
        examples, manifest = _load_dataset_records(source_path)
        result = _run_dataset_evaluation(
            examples=examples,
            manifest=manifest,
            eval_config=eval_config,
            config_path=Path(config_path),
            metrics=metrics,
        )
        result.metadata["benchmark_manifest"] = manifest
        result.metadata["acceptance_gate"] = _evaluate_acceptance_thresholds(
            result,
            eval_config.get("acceptance_thresholds", {}),
        )
    else:
        source_path = Path(artifact_value or "./outputs/generations")
        logger.info("Running artifact-backed evaluation from: %s", source_path)
        artifacts = _load_artifact_records(source_path)
        evaluator = Evaluator(validator=SVGValidator())
        result = evaluator.evaluate_artifacts(
            artifacts,
            max_samples=eval_config.get("max_samples"),
            metrics=metrics,
        )
    result.metadata.update(
        {
            "config_path": str(config_path),
            "source": str(source_path),
            "artifact_source": (
                str(source_path) if result.metadata["evaluation_mode"] == "artifacts" else None
            ),
            "dataset_source": (
                str(source_path) if result.metadata["evaluation_mode"] == "dataset" else None
            ),
            "metrics": metrics,
            "requested_max_samples": eval_config.get("max_samples"),
            "output_dir": str(eval_config.get("output_dir", "./outputs/eval_reports")),
        }
    )
    return result


def _load_artifact_records(source_path: Path) -> list[GenerationArtifactRecord]:
    if source_path.is_dir():
        return list_generation_artifacts(source_path)
    if source_path.suffix in {".json", ".svg"}:
        return [load_generation_artifact(source_path)]
    raise ValueError(
        "artifact_path must be a directory, a .json metadata sidecar, or a .svg artifact file."
    )


def _load_dataset_records(source_path: Path) -> tuple[list[TextToSVGExample], dict[str, Any]]:
    if source_path.suffix != ".jsonl":
        raise ValueError("dataset_path must point to a prepared .jsonl file.")
    records = [TextToSVGExample.from_dict(record) for record in read_jsonl(source_path)]
    manifest_path = source_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_file") != source_path.name:
        raise ValueError("Benchmark manifest output_file does not match dataset_path.")
    if manifest.get("num_records") != len(records):
        raise ValueError("Benchmark manifest record count does not match the JSONL snapshot.")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if manifest.get("output_sha256") != digest:
        raise ValueError("Benchmark JSONL checksum does not match its manifest.")
    return records, manifest


def _run_dataset_evaluation(
    *,
    examples: list[TextToSVGExample],
    manifest: dict[str, Any],
    eval_config: dict[str, Any],
    config_path: Path,
    metrics: list[str],
) -> EvaluationResult:
    partition = eval_config.get("partition", "held_out")
    allow_memory_ingestion = eval_config.get("allow_memory_ingestion", False)
    if partition == "held_out":
        if manifest.get("benchmark_status") != "adopted":
            raise ValueError("Held-out evaluation requires an adopted benchmark manifest.")
        if manifest.get("license_review_required") is not False:
            raise ValueError("Held-out evaluation requires completed benchmark license review.")
        if manifest.get("memory_eligible") is not False:
            raise ValueError("Held-out benchmark manifests must set memory_eligible=false.")
        declared_metrics = manifest.get("evaluation_metrics")
        if not isinstance(declared_metrics, list) or not set(metrics).issubset(declared_metrics):
            raise ValueError("Evaluation metrics must be frozen in the adopted benchmark manifest.")
    policy = BenchmarkRunPolicy(
        partition=partition,
        allow_memory_ingestion=allow_memory_ingestion,
    )
    selected = examples
    max_samples = eval_config.get("max_samples")
    if max_samples is not None:
        if not isinstance(max_samples, int) or max_samples < 0:
            raise ValueError("eval.max_samples must be non-negative or null.")
        selected = examples[:max_samples]
    if not selected:
        return Evaluator(validator=SVGValidator()).evaluate(
            [], metrics=metrics, run_policy=policy
        )

    from svg_agentic_slm.factories.generation import (
        build_generation_runtime,
        close_generation_runtime,
    )
    from svg_agentic_slm.svg.renderer import CairoSVGRenderer

    generation_config_path = Path(
        eval_config.get("generation_config") or config_path.with_name("generation.yaml")
    )
    generation_overrides = eval_config.get("generation_overrides", {})
    if not isinstance(generation_overrides, dict):
        raise ValueError("eval.generation_overrides must be a mapping.")
    # Evaluation owns rendering and gives every sample a unique output path.
    generation_overrides = merge_nested_dicts(
        generation_overrides,
        {"generation": {"render": {"enabled": False}}},
    )
    runtime = build_generation_runtime(
        generation_config_path,
        prompt=selected[0].instruction,
        enable_rag=bool(eval_config.get("enable_rag", False)),
        enable_critic=bool(eval_config.get("enable_critic", False)),
        overrides=generation_overrides,
    )
    output_dir = Path(eval_config.get("output_dir", "./outputs/eval_reports"))
    render_enabled = bool(eval_config.get("render_predictions", True))
    evaluator = Evaluator(
        validator=SVGValidator(),
        orchestrator=runtime.orchestrator,
        renderer=CairoSVGRenderer() if render_enabled else None,
        prediction_output_dir=output_dir / "predictions",
        render_output_dir=Path(
            eval_config.get("render_output_dir", output_dir / "renders")
        ),
    )
    try:
        return evaluator.evaluate(
            examples,
            max_samples=max_samples,
            metrics=metrics,
            run_policy=policy,
        )
    finally:
        close_generation_runtime(runtime)


def _evaluate_acceptance_thresholds(
    result: EvaluationResult,
    raw_thresholds: object,
) -> dict[str, Any]:
    if not isinstance(raw_thresholds, dict):
        raise ValueError("eval.acceptance_thresholds must be a mapping.")
    values = {
        "generation_success_rate": result.generation_success_rate,
        "svg_validity_rate": result.svg_validity_rate,
        "render_success_rate": result.render_success_rate,
        "avg_generation_latency": result.avg_generation_latency,
        "avg_time_to_first_token": result.avg_time_to_first_token,
        "avg_tokens_per_second": result.avg_tokens_per_second,
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, threshold in raw_thresholds.items():
        if name.startswith("min_"):
            metric = name.removeprefix("min_")
            comparison = "minimum"
            passed = metric in values and values[metric] >= float(threshold)
        elif name.startswith("max_"):
            metric = name.removeprefix("max_")
            comparison = "maximum"
            passed = metric in values and values[metric] <= float(threshold)
        else:
            raise ValueError(f"Acceptance threshold must start with min_ or max_: {name}")
        if metric not in values:
            raise ValueError(f"Unknown acceptance threshold metric: {metric}")
        checks[name] = {
            "metric": metric,
            "comparison": comparison,
            "actual": values[metric],
            "threshold": float(threshold),
            "passed": passed,
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
