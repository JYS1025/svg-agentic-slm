"""Evaluation report generation.

Provides functions to format and save evaluation results
as human-readable reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from svg_agentic_slm.eval.schemas import EvaluationResult
from svg_agentic_slm.utils.atomic import atomic_write_text

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
        "generation_success_rate": result.generation_success_rate,
        "svg_validity_rate": result.svg_validity_rate,
        "render_success_rate": result.render_success_rate,
        "avg_generation_latency": result.avg_generation_latency,
        "avg_instruction_alignment": result.avg_instruction_alignment,
        "avg_time_to_first_token": result.avg_time_to_first_token,
        "avg_tokens_per_second": result.avg_tokens_per_second,
        "per_sample_results": result.per_sample_results,
        "metadata": result.metadata,
    }
    atomic_write_text(json_path, json.dumps(report_data, indent=2, ensure_ascii=False))

    # Save text summary
    txt_path = output_dir / f"{report_name}.txt"
    lines = [result.summary()]
    if result.per_sample_results:
        lines.append("\nPer-sample results:")
        for sample in result.per_sample_results:
            instruction = sample.get("instruction", "")
            svg_path = sample.get("svg_path", "")
            render_success = sample.get("render_success", False)
            is_valid = sample.get("is_valid", False)
            lines.append(
                f"- valid={is_valid} render_success={render_success} "
                f"instruction={instruction} svg_path={svg_path}"
            )
    atomic_write_text(txt_path, "\n".join(lines))

    logger.info("Report saved to %s and %s", json_path, txt_path)
    return json_path
