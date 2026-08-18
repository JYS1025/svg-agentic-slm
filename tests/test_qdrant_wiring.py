"""Network-free tests for RAG backend and CLI wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from svg_agentic_slm.agents.schemas import GenerationRequest
from svg_agentic_slm.cli import commands_generate, commands_rag_index
from svg_agentic_slm.cli.app import app
from svg_agentic_slm.factories import generation
from svg_agentic_slm.rag.hf_indexer import IndexingResult
from svg_agentic_slm.rag.schemas import RetrievedExample
from svg_agentic_slm.utils.config import load_yaml_config

runner = CliRunner()


class _FactoryProduct:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_backend_factory_builds_qdrant_from_merged_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "QdrantRetriever", _FactoryProduct)

    retriever = generation.build_rag_retriever(
        {
            "backend": "qdrant",
            "embedding_model": "top-level-model",
            "similarity_threshold": 0.4,
            "upload_batch_size": 32,
            "qdrant": {
                "collection_name": "cloud-svg",
                "embedding_model": "nested-model",
                "url_env": "CUSTOM_URL",
                "api_key_env": "CUSTOM_KEY",
                "timeout_seconds": 9,
                "upload_batch_size": 7,
                "compress_svg": False,
                "on_disk_vectors": False,
                "on_disk_payload": False,
                "on_disk_hnsw": False,
                "scalar_quantization": False,
            },
        }
    )

    assert isinstance(retriever, _FactoryProduct)
    assert retriever.kwargs == {
        "collection_name": "cloud-svg",
        "embedding_model": "nested-model",
        "similarity_threshold": 0.4,
        "url_env": "CUSTOM_URL",
        "api_key_env": "CUSTOM_KEY",
        "timeout_seconds": 9,
        "upload_batch_size": 7,
        "compress_svg": False,
        "on_disk_vectors": False,
        "on_disk_payload": False,
        "on_disk_hnsw": False,
        "scalar_quantization": False,
    }


@pytest.mark.parametrize("backend", [None, "chroma", "chromadb"])
def test_backend_factory_preserves_chroma_default_and_corpus_loading(
    monkeypatch: pytest.MonkeyPatch,
    backend: str | None,
) -> None:
    loaded: list[tuple[str, Any]] = []
    monkeypatch.setattr(generation, "ChromaRetriever", _FactoryProduct)
    monkeypatch.setattr(
        generation,
        "load_svg_corpus",
        lambda path, retriever: loaded.append((path, retriever)),
    )
    config: dict[str, Any] = {
        "collection_name": "top-level",
        "corpus_path": "fallback.jsonl",
        "chromadb": {
            "collection_name": "nested",
            "persist_directory": "db",
            "corpus_path": "corpus.jsonl",
        },
    }
    if backend is not None:
        config["backend"] = backend

    retriever = generation.build_rag_retriever(config)

    assert retriever.kwargs["collection_name"] == "nested"
    assert retriever.kwargs["persist_directory"] == "db"
    assert loaded == [("corpus.jsonl", retriever)]


def test_backend_factory_skips_chroma_corpus_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "ChromaRetriever", _FactoryProduct)

    def fail_if_loaded(path: str, retriever: Any) -> None:
        raise AssertionError("corpus loader must not run")

    monkeypatch.setattr(generation, "load_svg_corpus", fail_if_loaded)

    generation.build_rag_retriever(
        {"backend": "chromadb", "corpus_path": "corpus.jsonl"},
        index_chroma_corpus=False,
    )


def test_backend_factory_forwards_precomputed_mmsvg_chroma_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "ChromaRetriever", _FactoryProduct)
    dataset_roots = {
        "icon": "./data/raw/mmsvg_icon_v2/data",
        "illustration": "./data/raw/mmsvg_illustration_v2/data",
    }

    retriever = generation.build_rag_retriever(
        {
            "backend": "chromadb",
            "chromadb": {
                "collection_name": "mmsvg-description-qwen",
                "persist_directory": "./data/chroma_db",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_revision": "revision-123",
                "embedding_dimension": 1024,
                "similarity_threshold": 0.25,
                "query_instruction": "Retrieve matching SVG examples.",
                "device": "cuda:1",
                "dataset_roots": dataset_roots,
                "precomputed_embeddings": True,
                "overfetch_factor": 7,
            },
        },
        index_chroma_corpus=False,
    )

    assert isinstance(retriever, _FactoryProduct)
    assert retriever.kwargs == {
        "collection_name": "mmsvg-description-qwen",
        "persist_directory": "./data/chroma_db",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_revision": "revision-123",
        "embedding_dimension": 1024,
        "similarity_threshold": 0.25,
        "query_instruction": "Retrieve matching SVG examples.",
        "device": "cuda:1",
        "dataset_roots": dataset_roots,
        "precomputed_embeddings": True,
        "overfetch_factor": 7,
    }


def test_active_rag_config_keeps_chroma_and_qdrant_embedding_dimensions_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_config = load_yaml_config(Path(__file__).parents[1] / "configs" / "rag.yaml")["rag"]
    monkeypatch.setattr(generation, "ChromaRetriever", _FactoryProduct)
    monkeypatch.setattr(generation, "QdrantRetriever", _FactoryProduct)

    chroma = generation.build_rag_retriever(
        {**rag_config, "backend": "chromadb"},
        index_chroma_corpus=False,
    )
    qdrant = generation.build_rag_retriever(
        {**rag_config, "backend": "qdrant"},
        index_chroma_corpus=False,
    )

    assert chroma.kwargs["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert chroma.kwargs["embedding_dimension"] == 1024
    assert qdrant.kwargs["embedding_model"] == ("sentence-transformers/all-MiniLM-L6-v2")


def test_backend_factory_rejects_unknown_backend_and_invalid_nested_config() -> None:
    with pytest.raises(ValueError, match="Unsupported RAG backend"):
        generation.build_rag_retriever({"backend": "pinecone"})

    with pytest.raises(ValueError, match=r"rag\.qdrant must be a mapping"):
        generation.build_rag_retriever({"backend": "qdrant", "qdrant": "not-a-mapping"})


@pytest.mark.parametrize("top_k", [True, 0, -1, 3.5, "3"])
def test_rag_agent_factory_rejects_non_positive_integer_top_k(top_k: Any) -> None:
    with pytest.raises(ValueError, match=r"rag\.top_k must be a positive integer"):
        generation._build_rag_agent({"top_k": top_k})


@pytest.mark.parametrize("secret_key", ["api_key", "token", "service_password"])
def test_backend_factory_rejects_inline_secrets(secret_key: str) -> None:
    with pytest.raises(ValueError, match="environment variables"):
        generation.build_rag_retriever(
            {
                "backend": "qdrant",
                "qdrant": {secret_key: "must-not-be-persisted"},
            }
        )


def test_generate_rag_backend_option_enables_rag_and_sets_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    result_value = SimpleNamespace(
        generated_svg="<svg/>",
        is_valid=True,
        critic_feedback=[],
        metadata={},
        render_path=None,
        attempts=[],
    )
    runtime = SimpleNamespace(
        orchestrator=SimpleNamespace(run=lambda request: result_value),
        request=object(),
    )

    def fake_build_generation_runtime(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(
        commands_generate,
        "build_generation_runtime",
        fake_build_generation_runtime,
    )
    monkeypatch.setattr(
        commands_generate,
        "persist_generation_artifacts",
        lambda **kwargs: SimpleNamespace(
            svg_path=tmp_path / "out.svg",
            metadata_path=tmp_path / "out.json",
            render_path=None,
        ),
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "Draw a circle",
            "--config",
            str(tmp_path / "generation.yaml"),
            "--rag-backend",
            "QDRANT",
        ],
    )

    assert result.exit_code == 0
    assert captured["enable_rag"] is True
    assert captured["overrides"] == {"rag": {"backend": "qdrant"}}


def test_generate_prints_exact_generator_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parameters = {
        "request_id": "request-123",
        "task": "text_to_svg",
        "prompt": "파란 원 안에 흰색 별을 그려줘",
        "rag_context": [
            {
                "rank": 1,
                "description": "A white star inside a blue circle",
                "svg_code": "<svg><circle/><path/></svg>",
            }
        ],
        "rag_metadata": {
            "enabled": True,
            "retrieved_examples": 1,
            "references": [{"rank": 1, "score": 0.91}],
        },
    }
    result_value = SimpleNamespace(
        generated_svg="<svg/>",
        is_valid=True,
        critic_feedback=[],
        metadata={},
        render_path=None,
        attempts=[],
    )

    class FakeOrchestrator:
        def run(
            self,
            request: object,
            *,
            on_generator_input: Any = None,
        ) -> Any:
            assert on_generator_input is not None
            on_generator_input(
                GenerationRequest(
                    instruction=parameters["prompt"],
                    task="text_to_svg",
                    run_id="run-123",
                ),
                [
                    RetrievedExample(
                        content="<svg><circle/><path/></svg>",
                        item_id="item-1",
                        source="test-corpus",
                        description="A white star inside a blue circle",
                        score=0.91,
                        rank=1,
                    )
                ],
            )
            return result_value

    runtime = SimpleNamespace(
        orchestrator=FakeOrchestrator(),
        request=object(),
    )
    monkeypatch.setattr(
        commands_generate,
        "build_generation_runtime",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        commands_generate,
        "persist_generation_artifacts",
        lambda **kwargs: SimpleNamespace(
            svg_path=tmp_path / "out.svg",
            metadata_path=tmp_path / "out.json",
            render_path=None,
        ),
    )

    result = runner.invoke(
        app,
        [
            "generate",
            parameters["prompt"],
            "--rag-backend",
            "qdrant",
            "--print-generator-parameters",
        ],
    )

    assert result.exit_code == 0
    assert "Generator input (typed contract):" in result.stdout
    assert '"run_id": "run-123"' in result.stdout
    assert '"context": [' in result.stdout
    assert '"item_id": "item-1"' in result.stdout
    assert "<svg><circle/><path/></svg>" in result.stdout


def test_generate_rejects_unknown_rag_backend_before_runtime_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_built(**kwargs: Any) -> Any:
        raise AssertionError("runtime must not be built")

    monkeypatch.setattr(
        commands_generate,
        "build_generation_runtime",
        fail_if_built,
    )

    result = runner.invoke(
        app,
        ["generate", "Draw a circle", "--rag-backend", "unknown"],
    )

    assert result.exit_code == 1
    assert "--rag-backend must be" in result.stdout


def test_rag_index_command_is_registered_and_applies_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeQdrantRetriever:
        collection_name = "cloud-svg"
        optimized = False

        def optimize_storage(self) -> None:
            self.optimized = True

    fake_retriever = FakeQdrantRetriever()
    monkeypatch.setattr(
        commands_rag_index,
        "QdrantRetriever",
        FakeQdrantRetriever,
    )
    monkeypatch.setattr(
        commands_rag_index,
        "load_yaml_config",
        lambda path: {
            "rag": {
                "backend": "chromadb",
                "indexing": {
                    "dataset_id": "org/data",
                    "dataset_split": "validation",
                    "dataset_revision": "commit",
                    "index_limit": 100,
                    "batch_size": 20,
                    "max_svg_chars": 900,
                    "max_caption_chars": 80,
                    "shuffle_buffer": 12,
                    "seed": 3,
                },
            }
        },
    )

    def fake_build(config: dict[str, Any], **kwargs: Any) -> Any:
        captured["factory_config"] = config
        captured["factory_kwargs"] = kwargs
        return fake_retriever

    def fake_index(retriever: Any, **kwargs: Any) -> IndexingResult:
        captured["index_retriever"] = retriever
        captured["index_kwargs"] = kwargs
        return IndexingResult(
            target_count=kwargs["index_limit"],
            collection_count_before=2,
            collection_count_after=8,
            uploaded_this_run=6,
            scanned_this_run=7,
            skipped_this_run=1,
        )

    monkeypatch.setattr(commands_rag_index, "build_rag_retriever", fake_build)
    monkeypatch.setattr(
        commands_rag_index,
        "index_huggingface_svg_dataset",
        fake_index,
    )

    result = runner.invoke(
        app,
        [
            "rag-index",
            "--config",
            "ignored.yaml",
            "--limit",
            "8",
            "--batch-size",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert "Qdrant indexing completed" in result.stdout
    assert "2 -> 8" in result.stdout
    assert captured["factory_config"]["backend"] == "qdrant"
    assert captured["factory_config"]["qdrant"] == {"upload_batch_size": 4}
    assert fake_retriever.optimized is True
    assert captured["factory_kwargs"] == {"index_chroma_corpus": False}
    assert captured["index_retriever"] is fake_retriever
    assert captured["index_kwargs"] == {
        "dataset_id": "org/data",
        "dataset_split": "validation",
        "dataset_revision": "commit",
        "index_limit": 8,
        "batch_size": 4,
        "max_svg_chars": 900,
        "max_caption_chars": 80,
        "shuffle_buffer": 12,
        "seed": 3,
    }


def test_rag_index_help_is_available_without_optional_dependencies() -> None:
    result = runner.invoke(app, ["rag-index", "--help"])

    assert result.exit_code == 0
    assert "--limit" in result.stdout
    assert "--batch-size" in result.stdout
