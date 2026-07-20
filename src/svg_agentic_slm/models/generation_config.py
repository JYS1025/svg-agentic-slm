"""Generation configuration schema.

Defines the parameters that control text generation.
This is separate from model loading configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Controls sampling parameters, length limits, and other
    generation-time settings. Values should come from YAML configs,
    not be hard-coded in agents or models.
    """

    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    num_return_sequences: int = 1
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive.")
        if self.num_return_sequences != 1:
            raise ValueError("num_return_sequences=1 is the only supported v1 contract.")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for passing to model backends."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationConfig:
        """Create a config and reject unsupported generation-time options."""
        known_fields = {item.name for item in fields(cls)}
        section_keys = {"svg", "orchestration", "render"}
        unknown_fields = set(data) - known_fields - section_keys
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown generation config option(s): {names}")
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
