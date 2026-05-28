"""Generation configuration schema.

Defines the parameters that control text generation.
This is separate from model loading configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for passing to model backends."""
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationConfig:
        """Create a GenerationConfig from a dictionary.

        Unknown keys are silently ignored.
        """
        known_fields = {f.name for f in field()} if False else set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
