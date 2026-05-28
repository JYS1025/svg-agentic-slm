"""Schemas for evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Aggregated evaluation results across a dataset.

    Attributes:
        num_samples: Total number of samples evaluated.
        svg_validity_rate: Fraction of samples producing valid SVGs.
        render_success_rate: Fraction of samples that rendered successfully.
        avg_generation_latency: Average generation time in seconds.
        avg_instruction_alignment: Average instruction alignment score (0-1).
        per_sample_results: Detailed per-sample results.
        metadata: Additional metadata about the evaluation run.
    """

    num_samples: int = 0
    svg_validity_rate: float = 0.0
    render_success_rate: float = 0.0
    avg_generation_latency: float = 0.0
    avg_instruction_alignment: float = 0.0
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation results."""
        return (
            f"Evaluation Results ({self.num_samples} samples):\n"
            f"  SVG Validity Rate:        {self.svg_validity_rate:.1%}\n"
            f"  Render Success Rate:      {self.render_success_rate:.1%}\n"
            f"  Avg Generation Latency:   {self.avg_generation_latency:.3f}s\n"
            f"  Avg Instruction Alignment: {self.avg_instruction_alignment:.1%}"
        )
