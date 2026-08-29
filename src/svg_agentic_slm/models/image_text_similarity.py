"""Typed image-text similarity boundary used by generation and evaluation."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

SIGLIP2_PAIR_PROBABILITY = "siglip2_pair_probability"


@dataclass(frozen=True)
class ImageTextSimilarityEvidence:
    """Attempt-correlated provenance for one image-text similarity score."""

    attempt_id: str
    metric: str
    score: float
    raw_logit: float
    model_id: str
    model_revision: str | None
    text_template: str
    text_input: str
    image_sha256: str
    device: str
    dtype: str
    latency_seconds: float


class BaseImageTextSimilarityScorer(ABC):
    """Lifecycle and inference contract for image-text similarity models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the scorer model and processor."""
        ...

    @abstractmethod
    def score(
        self,
        instruction: str,
        image_png: bytes,
        *,
        attempt_id: str,
    ) -> ImageTextSimilarityEvidence:
        """Score one instruction against one rendered PNG."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether the scorer is ready for inference."""
        ...

    def unload_model(self) -> None:
        """Release scorer resources. Concrete implementations may override."""


def validate_image_text_similarity_evidence(
    value: object,
) -> ImageTextSimilarityEvidence:
    """Validate similarity evidence at agent and artifact boundaries."""
    if not isinstance(value, ImageTextSimilarityEvidence):
        raise TypeError("Similarity evidence must be ImageTextSimilarityEvidence.")
    for field_name in (
        "attempt_id",
        "metric",
        "model_id",
        "text_template",
        "text_input",
        "image_sha256",
        "device",
        "dtype",
    ):
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"Similarity evidence {field_name} must be non-empty.")
    if value.metric != SIGLIP2_PAIR_PROBABILITY:
        raise ValueError(f"Unsupported image-text similarity metric: {value.metric!r}.")
    if value.model_revision is not None and (
        not isinstance(value.model_revision, str) or not value.model_revision.strip()
    ):
        raise ValueError("Similarity evidence model_revision must be non-empty or None.")
    if not _is_finite_number(value.score) or not 0.0 <= float(value.score) <= 1.0:
        raise ValueError("Similarity evidence score must be finite and between 0 and 1.")
    if not _is_finite_number(value.raw_logit):
        raise ValueError("Similarity evidence raw_logit must be finite.")
    if (
        not _is_finite_number(value.latency_seconds)
        or float(value.latency_seconds) < 0.0
    ):
        raise ValueError("Similarity evidence latency_seconds must be non-negative.")
    if re.fullmatch(r"[0-9a-f]{64}", value.image_sha256) is None:
        raise ValueError("Similarity evidence image_sha256 must be a lowercase SHA-256.")
    return value


def image_text_similarity_prompt_payload(
    evidence: ImageTextSimilarityEvidence,
) -> dict[str, object]:
    """Return the trusted subset of similarity evidence exposed to the Critic."""
    value = validate_image_text_similarity_evidence(evidence)
    return {
        "attempt_id": value.attempt_id,
        "metric": value.metric,
        "score": round(float(value.score), 6),
        "raw_logit": round(float(value.raw_logit), 6),
        "model_id": value.model_id,
        "model_revision": value.model_revision,
        "text_template": value.text_template,
    }


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
