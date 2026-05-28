"""Evaluation report generation.

Provides functions to format and save evaluation results
as human-readable reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from svg_agentic_slm.eval.schemas import EvaluationResult

logger = logging.getLogger(__name__)


def generate_report(
    result: EvaluationResult,
    output_dir: str | Path,
    report_name: str = "eval_report",
) -> Path:
    """Generate and save an evaluation report.

    Args:
        result: The evaluation results to report.
        output_dir: Directory to save the report.
        report_name: Base name for the report file.

    Returns:
        Path to the generated report file.

    TODO: Add HTML report generation.
    TODO: Add visualization (charts, tables).
    TODO: Add comparison reports across experiments.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = output_dir / f"{report_name}.json"
    report_data = {
        "num_samples": result.num_samples,
        "svg_validity_rate": result.svg_validity_rate,
        "render_success_rate": result.render_success_rate,
        "avg_generation_latency": result.avg_generation_latency,
        "avg_instruction_alignment": result.avg_instruction_alignment,
        "metadata": result.metadata,
    }
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Save text summary
    txt_path = output_dir / f"{report_name}.txt"
    with open(txt_path, "w") as f:
        f.write(result.summary())

    logger.info("Report saved to %s and %s", json_path, txt_path)
    return json_path
