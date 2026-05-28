"""Data schemas for dataset examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextToSVGExample:
    """A single text-to-SVG training/evaluation example.

    Attributes:
        task: Task identifier (e.g., 'text_to_svg').
        instruction: Natural language description of the desired SVG.
        output_svg: The target SVG code.
        metadata: Optional additional metadata.
    """

    task: str
    instruction: str
    output_svg: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        """Convert to a dictionary suitable for JSONL serialization."""
        d = {
            "task": self.task,
            "instruction": self.instruction,
            "output_svg": self.output_svg,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TextToSVGExample:
        """Create from a dictionary (e.g., parsed from JSONL)."""
        return cls(
            task=data["task"],
            instruction=data["instruction"],
            output_svg=data["output_svg"],
            metadata=data.get("metadata"),
        )
