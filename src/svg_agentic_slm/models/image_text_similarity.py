"""Typed image-text similarity boundary used by generation and evaluation."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
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

    def score_batch(
        self,
        instruction: str,
        image_pngs: Sequence[bytes],
        *,
        attempt_ids: Sequence[str],
    ) -> list[ImageTextSimilarityEvidence]:
        """Score an ordered image batch with a compatibility fallback."""
        if isinstance(image_pngs, (bytes, bytearray, str)) or not isinstance(
            image_pngs, Sequence
        ):
            raise TypeError("image_pngs must be a sequence of PNG byte strings.")
        if isinstance(attempt_ids, (str, bytes, bytearray)) or not isinstance(
            attempt_ids, Sequence
        ):
            raise TypeError("attempt_ids must be a sequence of strings.")
        if not image_pngs:
            raise ValueError("image_pngs and attempt_ids must be non-empty.")
        if len(image_pngs) != len(attempt_ids):
            raise ValueError("image_pngs and attempt_ids must have equal lengths.")
        return [
            self.score(instruction, image_png, attempt_id=attempt_id)
            for image_png, attempt_id in zip(image_pngs, attempt_ids, strict=True)
        ]

    def score_pairs_batch(
        self,
        instructions: Sequence[str],
        image_pngs: Sequence[bytes],
        *,
        attempt_ids: Sequence[str],
    ) -> list[ImageTextSimilarityEvidence]:
        """Score ordered text-image pairs with a compatibility fallback."""
        for name, values in (
            ("instructions", instructions),
            ("image_pngs", image_pngs),
            ("attempt_ids", attempt_ids),
        ):
            if isinstance(values, (str, bytes, bytearray)) or not isinstance(
                values, Sequence
            ):
                raise TypeError(f"{name} must be a sequence.")
        if not instructions:
            raise ValueError("Pair batches must be non-empty.")
        if len(instructions) != len(image_pngs) or len(instructions) != len(attempt_ids):
            raise ValueError("Pair batch inputs must have equal lengths.")
        return [
            self.score(instruction, image_png, attempt_id=attempt_id)
            for instruction, image_png, attempt_id in zip(
                instructions,
                image_pngs,
                attempt_ids,
                strict=True,
            )
        ]

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


def image_text_similarity_prompt_score(
    evidence: ImageTextSimilarityEvidence,
) -> float:
    """Return only the rounded 0-1 score exposed to the Critic prompt."""
    value = validate_image_text_similarity_evidence(evidence)
    return round(float(value.score), 6)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
