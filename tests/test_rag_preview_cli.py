"""Tests for the retrieval-only RAG preview CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from svg_agentic_slm.cli import commands_rag_preview
from svg_agentic_slm.cli.app import app
from svg_agentic_slm.rag.schemas import RetrievedExample

runner = CliRunner()


class _FakeRetriever:
    def __init__(self, examples: list[RetrievedExample]) -> None:
        self.examples = examples
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedExample]:
        self.calls.append((query, top_k))
        return self.examples[:top_k]


def _write_configs(tmp_path: Path, *, max_context_characters: int = 12000) -> tuple[Path, Path]:
    rag_path = tmp_path / "rag.yaml"
    rag_path.write_text(
        "rag:\n  backend: chromadb\n  top_k: 3\n  chromadb:\n    collection_name: test\n",
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.yaml"
    generation_path.write_text(
        "generation:\n"
        "  max_new_tokens: 128\n"
        "  temperature: 0.2\n"
        "  svg:\n"
        f"    max_context_characters: {max_context_characters}\n",
        encoding="utf-8",
    )
    return rag_path, generation_path


def test_rag_preview_prints_exact_typed_generator_input_without_slm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rag_path, generation_path = _write_configs(tmp_path)
    retriever = _FakeRetriever(
        [
            RetrievedExample(
                content="<svg><circle/></svg>",
                item_id="mmsvg-icon-1",
                source="MMSVG/Icon",
                description="A blue circle",
                score=0.93,
                score_kind="cosine_similarity",
                metadata={"parquet_path": "must-not-cross-boundary"},
            )
        ]
    )
    factory_calls: list[tuple[dict[str, Any], bool]] = []

    def fake_build_rag_retriever(
        config: dict[str, Any],
        *,
        index_chroma_corpus: bool,
    ) -> _FakeRetriever:
        factory_calls.append((config, index_chroma_corpus))
        return retriever

    monkeypatch.setattr(
        commands_rag_preview,
        "build_rag_retriever",
        fake_build_rag_retriever,
    )

    result = runner.invoke(
        app,
        [
            "rag-preview",
            "파란 원 아이콘",
            "--rag-config",
            str(rag_path),
            "--generation-config",
            str(generation_path),
            "--top-k",
            "1",
            "--run-id",
            "preview-1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"request", "context"}
    assert payload["request"] == {
        "instruction": "파란 원 아이콘",
        "task": "text_to_svg",
        "config_overrides": {
            "max_new_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "num_return_sequences": 1,
            "seed": 42,
        },
        "run_id": "preview-1",
    }
    assert payload["context"] == [
        {
            "content": "<svg><circle/></svg>",
            "item_id": "mmsvg-icon-1",
            "source": "MMSVG/Icon",
            "description": "A blue circle",
            "score": 0.93,
            "score_kind": "cosine_similarity",
            "rank": 1,
            "kind": "reference_svg",
            "corpus_version": None,
            "metadata": {},
        }
    ]
    assert retriever.calls == [("파란 원 아이콘", 1)]
    assert factory_calls[0][1] is False


def test_rag_preview_prompt_is_post_generator_context_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rag_path, generation_path = _write_configs(
        tmp_path,
        max_context_characters=133,
    )
    retriever = _FakeRetriever(
        [
            RetrievedExample(
                content="<svg/>",
                item_id="item-1",
                source="s",
                description="D",
                score=0.8,
            )
        ]
    )
    monkeypatch.setattr(
        commands_rag_preview,
        "build_rag_retriever",
        lambda config, *, index_chroma_corpus: retriever,
    )

    result = runner.invoke(
        app,
        [
            "rag-preview",
            "Draw.",
            "--config",
            str(rag_path),
            "--generation-config",
            str(generation_path),
            "--prompt-preview",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["context"][0]["content"] == "<svg/>"
    preview = payload["prompt_preview"]
    assert preview["max_context_characters"] == 133
    assert preview["context_item_ids"] == []
    assert preview["truncated_context_item_ids"] == ["item-1"]
    assert "Content:" not in preview["prompt"]
    assert "Now generate an SVG for the following description:\nDraw." in preview["prompt"]
    assert preview["system_prompt"].startswith("You are an expert SVG code generator.")
