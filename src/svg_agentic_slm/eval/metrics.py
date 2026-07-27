"""Evaluation metric functions.

Each metric function takes predictions and references and returns
a score. Metrics are designed to be composable and independently testable.
"""

from __future__ import annotations

import logging

from svg_agentic_slm.svg.base import BaseValidator

logger = logging.getLogger(__name__)


def compute_svg_validity_rate(
    svg_outputs: list[str],
    validator: BaseValidator,
) -> float:
    """Compute the fraction of SVG outputs that pass validation.

    Args:
        svg_outputs: List of generated SVG strings.
        validator: SVG validator instance.

    Returns:
        Fraction of valid SVGs (0.0 to 1.0).
    """
    if not svg_outputs:
        return 0.0

    valid_count = sum(
        1 for svg in svg_outputs
        if validator.validate(svg).is_valid
    )
    return valid_count / len(svg_outputs)


def compute_render_success_rate(
    render_results: list[bool],
) -> float:
    """Compute the fraction of successful renders.

    Args:
        render_results: List of boolean success flags.

    Returns:
        Fraction of successful renders (0.0 to 1.0).
    """
    if not render_results:
        return 0.0
    return sum(render_results) / len(render_results)


def compute_generation_latency(
    latencies: list[float],
) -> float:
    """Compute the average generation latency.

    Args:
        latencies: List of generation times in seconds.

    Returns:
        Average latency in seconds.
    """
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def compute_simple_instruction_alignment(
    instructions: list[str],
    svg_outputs: list[str],
) -> float:
    """Compute a simple instruction alignment score.

    This is a placeholder metric. A real implementation would use
    an LLM-as-judge, CLIP similarity, or other semantic comparison.

    Args:
        instructions: List of text instructions.
        svg_outputs: List of generated SVG strings.

    Returns:
        Average alignment score (0.0 to 1.0).

    TODO: Implement meaningful instruction alignment metric.
    TODO: Consider CLIP-based visual similarity.
    TODO: Consider LLM-as-judge evaluation.
    """
    if not instructions or not svg_outputs:
        return 0.0

    # Placeholder: just check that SVGs are non-empty
    scores = [
        1.0 if svg.strip() else 0.0
        for svg in svg_outputs
    ]
    return sum(scores) / len(scores)
