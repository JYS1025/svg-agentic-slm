"""Whitelist policy for metadata crossing the RAG-to-Generator boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class RetrievalMetadataPolicy:
    """Project vector-store metadata onto an explicit shared whitelist.

    Retriever implementations may keep arbitrary internal metadata. Only keys
    listed here are allowed to cross into Generator prompts and artifacts.
    """

    allowed_keys: frozenset[str] = frozenset()
    unknown_key_behavior: Literal["drop", "raise"] = "drop"

    def apply(self, metadata: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Return whitelisted metadata and the sorted keys that were excluded."""
        unknown_keys = sorted(set(metadata) - self.allowed_keys)
        if unknown_keys and self.unknown_key_behavior == "raise":
            names = ", ".join(unknown_keys)
            raise ValueError(f"RAG metadata contains non-whitelisted key(s): {names}")
        return (
            {key: metadata[key] for key in self.allowed_keys if key in metadata},
            unknown_keys,
        )


# Cycle 0 deliberately allows no free-form metadata across the shared boundary.
# Add a key here only after its owner, semantics, and artifact retention policy
# are documented in the cross-team contract.
MINIMAL_RETRIEVAL_METADATA_POLICY = RetrievalMetadataPolicy()

MMSVG_HYBRID_RETRIEVAL_METADATA_POLICY = RetrievalMetadataPolicy(
    allowed_keys=frozenset(
        {
            "retrieval_strategy",
            "row_index_in_shard",
            "shard",
            "record_id",
            "dataset_type",
            "canonical_svg_sha256",
            "description_rank",
            "description_score",
            "detail_rank",
            "detail_score",
            "is_overlap",
            "siglip2_score",
            "siglip2_raw_logit",
            "siglip2_model_id",
            "siglip2_model_revision",
            "image_sha256",
            "selection_reason",
        }
    )
)
