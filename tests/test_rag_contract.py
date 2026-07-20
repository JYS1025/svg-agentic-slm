"""Tests for the neutral RAG-to-Generator contract."""

from __future__ import annotations

import pytest

from svg_agentic_slm.agents.rag_agent import RAGAgent
from svg_agentic_slm.rag.metadata_policy import RetrievalMetadataPolicy
from svg_agentic_slm.rag.schemas import RetrievedExample


class _Retriever:
    def __init__(self, example: RetrievedExample) -> None:
        self.example = example

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedExample]:
        return [self.example]


def test_cycle0_metadata_policy_drops_every_free_form_key() -> None:
    example = RetrievedExample(
        content="<svg></svg>",
        item_id="svg:corpus:item-1",
        source="corpus/item-1.svg",
        metadata={"render_ref": "renders/item-1.png", "private_store_key": "abc"},
    )

    result = RAGAgent(_Retriever(example)).retrieve("Draw an icon.")

    assert result[0].metadata == {}
    assert result[0].rank == 1


def test_explicit_metadata_whitelist_can_be_extended_at_adapter_boundary() -> None:
    example = RetrievedExample(
        content="<svg></svg>",
        item_id="svg:corpus:item-1",
        source="corpus/item-1.svg",
        metadata={"render_ref": "renders/item-1.png", "private_store_key": "abc"},
    )
    policy = RetrievalMetadataPolicy(allowed_keys=frozenset({"render_ref"}))

    result = RAGAgent(_Retriever(example), metadata_policy=policy).retrieve("Draw.")

    assert result[0].metadata == {"render_ref": "renders/item-1.png"}


def test_retrieved_example_requires_stable_identity_fields() -> None:
    with pytest.raises(ValueError, match="item_id"):
        RetrievedExample(content="<svg></svg>", item_id="", source="corpus/item.svg")

    with pytest.raises(ValueError, match="source"):
        RetrievedExample(content="<svg></svg>", item_id="item-1", source="")
