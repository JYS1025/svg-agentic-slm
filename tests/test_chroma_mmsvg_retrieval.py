"""Offline contracts for precomputed MMSVG Chroma retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from svg_agentic_slm.agents.rag_agent import RAGAgent  # noqa: E402
from svg_agentic_slm.rag.chroma_store import (  # noqa: E402
    ChromaRetriever,
    _build_search_text,
)
from svg_agentic_slm.rag.embedding import Qwen3EmbeddingEncoder  # noqa: E402
from svg_agentic_slm.rag.schemas import RetrievedExample  # noqa: E402


class _RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_queries(self, texts: list[str]) -> Any:
        self.calls.append(list(texts))
        return np.asarray([[0.1, 0.2, 0.3]], dtype="float32")


class _Collection:
    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def count(self) -> int:
        return len(self.results["ids"][0])

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        assert "query_texts" not in kwargs
        return self.results


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": ["icon-0", "icon-1", "icon-2"],
                "description": [
                    "white star in blue circle",
                    "red rounded square",
                    "thin green leaf",
                ],
                "svg": [
                    '<svg xmlns="http://www.w3.org/2000/svg"><circle/><path/></svg>',
                    '<svg xmlns="http://www.w3.org/2000/svg"><rect rx="4"/></svg>',
                    '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
                ],
                "keywords": ["must-not-cross"] * 3,
                "detail": ["must-not-cross"] * 3,
            }
        ),
        path,
    )


def _results(row_indexes: list[int]) -> dict[str, Any]:
    descriptions = [
        "white star in blue circle",
        "red rounded square",
        "thin green leaf",
    ]
    return {
        "ids": [[f"icon:icon-{index}" for index in range(len(row_indexes))]],
        "documents": [[descriptions[index] for index in range(len(row_indexes))]],
        "metadatas": [
            [
                {
                    "dataset_type": "icon",
                    "record_id": f"icon-{index}",
                    "shard": "part.parquet",
                    "row_index_in_shard": row_index,
                }
                for index, row_index in enumerate(row_indexes)
            ]
        ],
        "distances": [[0.05, 0.2, 0.35][: len(row_indexes)]],
    }


def _retriever(
    root: Path,
    collection: _Collection,
    encoder: _RecordingEncoder,
) -> ChromaRetriever:
    retriever = ChromaRetriever(
        collection_name="mmsvg-test",
        persist_directory=str(root.parent / "chroma"),
        embedding_model="offline/fake-qwen",
        embedding_revision="revision-123",
        embedding_dimension=3,
        query_instruction="Retrieve matching SVG references.",
        device="cpu",
        dataset_roots={"icon": str(root)},
        precomputed_embeddings=True,
        overfetch_factor=1,
    )
    retriever._query_encoder = encoder
    retriever._collection = collection
    retriever._allowed_shards = {"icon": {"part.parquet": (root / "part.parquet").resolve()}}
    retriever._source_info = {
        "icon": {
            "dataset_id": "OmniSVG/MMSVG-Icon",
            "dataset_revision": "test-rev",
        }
    }
    return retriever


def test_query_embeddings_hydrates_top3_typed_generator_context(tmp_path: Path) -> None:
    root = tmp_path / "icons"
    _write_parquet(root / "part.parquet")
    encoder = _RecordingEncoder()
    collection = _Collection(_results([0, 1, 2]))

    examples = RAGAgent(_retriever(root, collection, encoder), top_k=3).retrieve(
        "  blue circle with white star  "
    )

    assert encoder.calls == [["blue circle with white star"]]
    assert len(collection.calls) == 1
    query_call = collection.calls[0]
    assert np.asarray(query_call["query_embeddings"]) == pytest.approx(
        np.asarray([[0.1, 0.2, 0.3]])
    )
    assert query_call["n_results"] == 3
    assert query_call["include"] == ["documents", "metadatas", "distances"]
    assert all(isinstance(item, RetrievedExample) for item in examples)
    assert [item.item_id for item in examples] == [
        "icon:icon-0",
        "icon:icon-1",
        "icon:icon-2",
    ]
    assert [item.rank for item in examples] == [1, 2, 3]
    assert [item.description for item in examples] == [
        "white star in blue circle",
        "red rounded square",
        "thin green leaf",
    ]
    assert [item.score for item in examples] == pytest.approx([0.95, 0.8, 0.65])
    assert all(item.score_kind == "cosine_similarity" for item in examples)
    assert all(item.kind == "reference_svg" and item.source for item in examples)
    assert examples[0].content.endswith("<circle/><path/></svg>")
    assert examples[1].content.endswith('<rect rx="4"/></svg>')
    assert examples[2].content.endswith("<path/></svg>")
    assert all(item.metadata == {} for item in examples)
    assert all("must-not-cross" not in item.description for item in examples)


def test_description_only_mmsvg_index_does_not_change_legacy_search_template() -> None:
    content = "<svg><circle/></svg>"
    search_text = _build_search_text(
        content,
        {
            "pattern_name": "blue_circle",
            "description": "A blue circle",
            "tags": ["blue", "round"],
        },
    )

    assert search_text == (
        f"Name: blue circle\nDescription: A blue circle\nTags: blue, round\nSVG: {content}"
    )


def test_parquet_pointer_must_resolve_to_matching_record_id(tmp_path: Path) -> None:
    root = tmp_path / "icons"
    _write_parquet(root / "part.parquet")
    retriever = _retriever(root, _Collection(_results([1])), _RecordingEncoder())

    assert retriever.retrieve("blue circle", top_k=1) == []


def test_zero_similarity_threshold_excludes_negative_cosine_hits(tmp_path: Path) -> None:
    root = tmp_path / "icons"
    _write_parquet(root / "part.parquet")
    results = _results([0, 1])
    results["distances"] = [[1.2, 0.2]]
    retriever = _retriever(root, _Collection(results), _RecordingEncoder())

    examples = retriever.retrieve("blue circle", top_k=2)

    assert [example.item_id for example in examples] == ["icon:icon-1"]
    assert examples[0].score == pytest.approx(0.8)


def test_overfetch_skips_unsafe_svg_and_still_returns_three(tmp_path: Path) -> None:
    root = tmp_path / "icons"
    parquet_path = root / "part.parquet"
    parquet_path.parent.mkdir(parents=True)
    descriptions = ["unsafe", "circle", "square", "triangle"]
    pq.write_table(
        pa.table(
            {
                "id": [f"icon-{index}" for index in range(4)],
                "description": descriptions,
                "svg": [
                    '<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
                    '<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>',
                    '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
                    '<svg xmlns="http://www.w3.org/2000/svg"><path/></svg>',
                ],
            }
        ),
        parquet_path,
    )
    collection = _Collection(
        {
            "ids": [[f"icon:icon-{index}" for index in range(4)]],
            "documents": [descriptions],
            "metadatas": [
                [
                    {
                        "dataset_type": "icon",
                        "record_id": f"icon-{index}",
                        "shard": "part.parquet",
                        "row_index_in_shard": index,
                    }
                    for index in range(4)
                ]
            ],
            "distances": [[0.01, 0.1, 0.2, 0.3]],
        }
    )
    retriever = _retriever(root, collection, _RecordingEncoder())
    retriever._overfetch_factor = 2

    examples = RAGAgent(retriever, top_k=3).retrieve("geometric shapes")

    assert collection.calls[0]["n_results"] == 4
    assert [example.item_id for example in examples] == [
        "icon:icon-1",
        "icon:icon-2",
        "icon:icon-3",
    ]
    assert [example.rank for example in examples] == [1, 2, 3]


def test_query_instruction_reaches_sentence_transformer_prompt_without_loading_model() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class _Model:
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            calls.append((list(texts), kwargs))
            return np.asarray([[0.25, 0.75]], dtype="float32")

    encoder = Qwen3EmbeddingEncoder(
        model_name="offline/fake",
        revision="rev",
        device="cpu",
        expected_dimension=2,
        query_instruction="Retrieve SVGs with matching geometry and colors.",
    )
    encoder._model = _Model()

    assert encoder.encode_queries(["  파란 원 안의 흰색 별  "]).tolist() == [[0.25, 0.75]]
    texts, kwargs = calls[0]
    assert texts == ["파란 원 안의 흰색 별"]
    assert kwargs["prompt"] == (
        "Instruct: Retrieve SVGs with matching geometry and colors.\nQuery:"
    )
    assert kwargs["normalize_embeddings"] is True


def test_failed_manifest_validation_cannot_be_bypassed_on_retry(tmp_path: Path) -> None:
    class _Client:
        def __init__(self) -> None:
            self.get_calls = 0
            self.collection = object()

        def list_collections(self) -> list[str]:
            return ["mmsvg-test"]

        def get_collection(self, **kwargs: Any) -> object:
            self.get_calls += 1
            return self.collection

    retriever = ChromaRetriever(
        collection_name="mmsvg-test",
        persist_directory=str(tmp_path / "missing-manifest"),
        embedding_model="offline/fake-qwen",
        embedding_revision="revision-123",
        embedding_dimension=3,
        query_instruction="Retrieve matching SVG references.",
        device="cpu",
        dataset_roots={"icon": tmp_path},
        precomputed_embeddings=True,
    )
    client = _Client()
    retriever._client = client

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Missing Chroma index manifest"):
            retriever._ensure_initialized()
        assert retriever._collection is None
    manifest_dir = tmp_path / "missing-manifest"
    manifest_dir.mkdir()
    (manifest_dir / "mmsvg-test.manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest must be a JSON object"):
        retriever._ensure_initialized()
    assert retriever._collection is None
    assert client.get_calls == 3
