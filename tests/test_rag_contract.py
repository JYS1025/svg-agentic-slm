"""Tests for the neutral RAG-to-Generator contract."""

from __future__ import annotations

import pytest

from svg_agentic_slm.agents.rag_agent import RAGAgent
from svg_agentic_slm.data.jsonl import write_jsonl
from svg_agentic_slm.rag.document_loader import load_svg_corpus
from svg_agentic_slm.rag.metadata_policy import RetrievalMetadataPolicy
from svg_agentic_slm.rag.schemas import RetrievedExample


class _Retriever:
    def __init__(self, example: RetrievedExample) -> None:
        self.example = example

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedExample]:
        return [self.example]


class _CollectingRetriever:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    def add_documents(self, documents: list[dict]) -> None:
        self.documents.extend(documents)


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


def test_rag_agent_drops_unsafe_svg_references() -> None:
    example = RetrievedExample(
        content='<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
        item_id="unsafe-svg",
        source="external-corpus",
    )

    assert RAGAgent(_Retriever(example)).retrieve("Draw an icon.") == []


def test_rag_agent_preserves_safe_fragments_and_non_svg_experiences() -> None:
    safe_fragment = RetrievedExample(
        content='<circle cx="4" cy="4" r="2"/>',
        item_id="safe-fragment",
        source="local-corpus",
    )
    experience = RetrievedExample(
        content="Prefer a high-contrast foreground.",
        item_id="experience-1",
        source="curated-memory",
        kind="positive_experience",
    )

    assert RAGAgent(_Retriever(safe_fragment)).retrieve("Draw.")[0].rank == 1
    assert RAGAgent(_Retriever(experience)).retrieve("Draw.")[0].content == experience.content


def test_chroma_corpus_loader_indexes_only_safe_svg_fragments(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(
        [
            {
                "pattern_name": "safe-circle",
                "description": "A circle",
                "svg_snippet": '<circle cx="4" cy="4" r="2"/>',
            },
            {
                "pattern_name": "unsafe-script",
                "description": "A script",
                "svg_snippet": "<script>alert(1)</script>",
            },
        ],
        corpus_path,
    )
    retriever = _CollectingRetriever()

    loaded = load_svg_corpus(corpus_path, retriever)

    assert loaded == 1
    assert [item["metadata"]["pattern_name"] for item in retriever.documents] == [
        "safe-circle"
    ]
