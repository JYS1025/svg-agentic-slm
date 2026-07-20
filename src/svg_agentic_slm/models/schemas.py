"""Typed values exchanged with model backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """A backend-neutral text generation response."""

    text: str
    model_id: str
    model_revision: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
