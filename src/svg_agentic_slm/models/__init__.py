"""Model loading and backend abstraction.

This module provides interfaces and implementations for loading
and running language model backends. Model-specific details
(loading, tokenization, generation) are encapsulated here and
not leaked into agent or orchestration code.
"""

from svg_agentic_slm.models.transformers_text_backend import TransformersTextBackend
from svg_agentic_slm.models.transformers_vlm_backend import TransformersVLMBackend

__all__ = ["TransformersTextBackend", "TransformersVLMBackend"]
