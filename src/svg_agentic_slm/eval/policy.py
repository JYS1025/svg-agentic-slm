"""Run policies that prevent benchmark samples from entering model memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from svg_agentic_slm.data.schemas import TextToSVGExample


@dataclass(frozen=True)
class BenchmarkRunPolicy:
    """Explicit data-use policy for a dataset-backed evaluation run."""

    partition: Literal["candidate", "development", "held_out"] = "held_out"
    allow_memory_ingestion: bool = False

    def __post_init__(self) -> None:
        if self.partition not in {"candidate", "development", "held_out"}:
            raise ValueError(f"Unsupported benchmark partition policy: {self.partition!r}")
        if not isinstance(self.allow_memory_ingestion, bool):
            raise TypeError("allow_memory_ingestion must be a boolean.")
        if self.partition == "held_out" and self.allow_memory_ingestion:
            raise ValueError("Held-out benchmark runs cannot enable memory ingestion.")

    def validate(self, example: TextToSVGExample) -> None:
        """Reject records that do not carry an auditable isolation contract."""
        metadata = example.metadata
        if not isinstance(metadata, dict):
            raise ValueError("Benchmark records must include metadata.")
        required = {
            "benchmark_id",
            "sample_id",
            "source_revision",
            "data_partition",
            "memory_eligible",
        }
        missing = required - set(metadata)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Benchmark record is missing isolation field(s): {names}")
        if self.partition == "held_out" and metadata["memory_eligible"] is not False:
            raise ValueError(
                f"Held-out sample {metadata['sample_id']!r} must set memory_eligible=false."
            )

    def metadata(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "allow_memory_ingestion": self.allow_memory_ingestion,
            "memory_write_blocked": not self.allow_memory_ingestion,
        }
