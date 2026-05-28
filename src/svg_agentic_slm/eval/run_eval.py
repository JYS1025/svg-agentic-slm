"""Entry point for running evaluations.

Provides a high-level function to load config, build components,
and run evaluation. Called by the CLI eval command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from svg_agentic_slm.eval.schemas import EvaluationResult
from svg_agentic_slm.utils.config import load_yaml_config

logger = logging.getLogger(__name__)


def run_evaluation(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Load configuration and run evaluation.

    Args:
        config_path: Path to the eval.yaml config file.
        overrides: Optional dictionary of config overrides.

    Returns:
        Evaluation results.

    TODO: Implement full evaluation setup:
    1. Load eval config.
    2. Build model backend.
    3. Build orchestrator with components.
    4. Load evaluation dataset.
    5. Run evaluator.
    6. Generate report.
    """
    config = load_yaml_config(config_path)
    eval_config = config.get("eval", {})

    logger.info("[PLACEHOLDER] Would run evaluation with config from: %s", config_path)
    logger.info("Dataset: %s", eval_config.get("dataset_path", "not specified"))
    logger.info("Metrics: %s", eval_config.get("metrics", []))

    # Placeholder result
    return EvaluationResult(
        num_samples=0,
        metadata={"config_path": str(config_path), "status": "placeholder"},
    )
