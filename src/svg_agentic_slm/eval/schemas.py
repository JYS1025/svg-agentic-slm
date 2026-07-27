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
    generation_success_rate: float = 0.0
    svg_validity_rate: float = 0.0
    render_success_rate: float = 0.0
    avg_generation_latency: float = 0.0
    avg_instruction_alignment: float = 0.0
    avg_time_to_first_token: float = 0.0
    avg_tokens_per_second: float = 0.0
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the evaluation results."""
        computed_metrics = self.metadata.get("computed_metrics")
        selected = set(computed_metrics) if isinstance(computed_metrics, list) else None
        lines = [f"Evaluation Results ({self.num_samples} samples):"]
        metric_lines = [
            (
                "generation_success_rate",
                f"  Generation Success Rate: {self.generation_success_rate:.1%}",
            ),
            ("svg_validity_rate", f"  SVG Validity Rate:        {self.svg_validity_rate:.1%}"),
            ("render_success_rate", f"  Render Success Rate:      {self.render_success_rate:.1%}"),
            (
                "generation_latency",
                f"  Avg Generation Latency:   {self.avg_generation_latency:.3f}s",
            ),
            (
                "simple_instruction_alignment",
                f"  Avg Instruction Alignment: {self.avg_instruction_alignment:.1%}",
            ),
            (
                "time_to_first_token",
                f"  Avg Time to First Token:  {self.avg_time_to_first_token:.3f}s",
            ),
            (
                "tokens_per_second",
                f"  Avg Decode Throughput:    {self.avg_tokens_per_second:.2f} tok/s",
            ),
        ]
        lines.extend(line for name, line in metric_lines if selected is None or name in selected)
        return "\n".join(lines)
