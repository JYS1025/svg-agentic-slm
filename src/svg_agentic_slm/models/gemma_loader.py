"""Backward-compatible Gemma backend name.

The local profile uses a pinned QAT Q4_0 compatibility quant through
llama.cpp. New code should depend on ``BaseModelBackend`` or instantiate
``LlamaCppModelBackend`` directly.
"""

from __future__ import annotations

from svg_agentic_slm.models.llama_cpp_backend import LlamaCppModelBackend


class GemmaModelBackend(LlamaCppModelBackend):
    """Compatibility alias for the selected local Gemma backend."""
