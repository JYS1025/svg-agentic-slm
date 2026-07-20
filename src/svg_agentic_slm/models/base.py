"""Abstract base class for model backends.

Defines the interface that all model backends must implement.
This allows swapping the underlying model (e.g., Gemma -> another SLM)
without changing agent or orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from svg_agentic_slm.models.schemas import ModelResponse


class BaseModelBackend(ABC):
    """Abstract interface for a language model backend.

    Concrete implementations handle model loading, tokenization,
    and text generation for a specific framework or model family.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer into memory.

        This may involve downloading weights, setting up quantization,
        and moving the model to the appropriate device.
        """
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional generation parameters (e.g., temperature,
                      max_new_tokens) that override defaults.

        Returns:
            Typed generated text and backend provenance.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check whether the model is currently loaded in memory.

        Returns:
            True if the model is loaded and ready for generation.
        """
        ...

    def unload_model(self) -> None:
        """Unload the model from memory to free resources.

        Default implementation is a no-op. Override if cleanup is needed.
        """
        pass
