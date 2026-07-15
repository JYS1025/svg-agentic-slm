"""Entry point for running evaluations.

Provides a high-level function to load config, build components,
and run evaluation. Called by the CLI eval command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from svg_agentic_slm.artifacts.generation import (
    GenerationArtifactRecord,
    list_generation_artifacts,
    load_generation_artifact,
)
from svg_agentic_slm.cli.overrides import merge_nested_dicts
from svg_agentic_slm.eval.evaluator import Evaluator
from svg_agentic_slm.eval.schemas import EvaluationResult
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "svg_validity_rate",
    "render_success_rate",
    "generation_latency",
    "simple_instruction_alignment",
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

    source_path = Path(
        eval_config.get("artifact_path")
        or eval_config.get("dataset_path", "./outputs/generations")
    )
    logger.info("Running artifact-backed evaluation from: %s", source_path)

    artifacts = _load_artifact_records(source_path)
    metrics = eval_config.get("metrics")
    if metrics is None:
        metrics = DEFAULT_METRICS
    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics):
        raise ValueError("eval.metrics must be a list of metric names.")
    evaluator = Evaluator(validator=SVGValidator())
    result = evaluator.evaluate_artifacts(
        artifacts,
        max_samples=eval_config.get("max_samples"),
        metrics=metrics,
    )
    result.metadata.update(
        {
            "config_path": str(config_path),
            "artifact_source": str(source_path),
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
