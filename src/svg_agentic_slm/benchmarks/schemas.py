"""Shared output contracts for dataset-specific benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkExample:
    """One immutable text-to-SVG benchmark record."""

    benchmark_id: str
    sample_id: str
    instruction: str
    reference_svg: str
    source_split: str
    source: str
    source_revision: str
    data_partition: str
    difficulty: str | None = None
    task_source_revision: str | None = None
    memory_eligible: bool = False

    def to_jsonl_record(self, reference_svg_sha256: str) -> dict[str, object]:
        """Convert to the repository's existing text-to-SVG JSONL shape."""
        return {
            "task": "text_to_svg",
            "instruction": self.instruction,
            "output_svg": self.reference_svg,
            "metadata": {
                "benchmark_id": self.benchmark_id,
                "sample_id": self.sample_id,
                "source_split": self.source_split,
                "difficulty": self.difficulty,
                "source": self.source,
                "source_revision": self.source_revision,
                "task_source_revision": self.task_source_revision,
                "reference_svg_sha256": reference_svg_sha256,
                "data_partition": self.data_partition,
                "memory_eligible": self.memory_eligible,
            },
        }


@dataclass(frozen=True)
class BenchmarkPreparationResult:
    """Files and counts produced by one benchmark adapter."""

    adapter: str
    output_path: Path
    manifest_path: Path
    num_records: int
    records_by_split: dict[str, int]
