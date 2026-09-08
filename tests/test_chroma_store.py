"""Offline tests for precomputed MMSVG Chroma retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from svg_agentic_slm.rag.chroma_store import ChromaRetriever

_COLLECTION_NAME = "mmsvg-test"
_DESCRIPTION = "A white star centered inside a blue circle"
_DETAIL = "A flat blue circular badge with a crisp five-point white star in the center"
_SVG = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4"/></svg>'


class _FakeQueryEncoder:
    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        assert queries == ["draw a blue star badge"]
        return [[0.1, 0.2, 0.3, 0.4]]


class _FakeCollection:
    def __init__(self, *, document_field: str, indexed_document: str) -> None:
        self.metadata = {
            "index_schema_version": 1,
            "document_field": document_field,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": "model-revision",
            "embedding_dimension": 4,
            "hnsw:space": "cosine",
        }
        self._indexed_document = indexed_document
        self.last_query: dict[str, Any] | None = None

    def count(self) -> int:
        return 1

    def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
        self.last_query = kwargs
        return {
            "ids": [["icon:record-1"]],
            "documents": [[self._indexed_document]],
            "metadatas": [
                [
                    {
                        "dataset_type": "icon",
                        "record_id": "record-1",
                        "shard": "part-00000.parquet",
                        "row_index_in_shard": 0,
                    }
                ]
            ],
            "distances": [[0.2]],
        }


def _build_retriever(
    tmp_path: Path,
    *,
    document_field: str,
    indexed_document: str,
) -> tuple[ChromaRetriever, _FakeCollection]:
    persist_directory = tmp_path / "chroma"
    dataset_root = tmp_path / "icon"
    persist_directory.mkdir()
    dataset_root.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["record-1"],
                "description": [_DESCRIPTION],
                "detail": [_DETAIL],
                "svg": [_SVG],
            }
        ),
        dataset_root / "part-00000.parquet",
    )
    manifest = {
        "index_schema_version": 1,
        "collection_name": _COLLECTION_NAME,
        "document_field": document_field,
        "document_template": f"{{{document_field}}}",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_revision": "model-revision",
        "embedding_dimension": 4,
        "normalized_embeddings": True,
        "query_instruction": "Represent this query for retrieving relevant SVGs",
        "distance": "cosine",
        "sources": [
            {
                "dataset_type": "icon",
                "dataset_id": "OmniSVG/MMSVG-Icon",
                "dataset_revision": "dataset-revision",
                "expected_shards": 1,
                "expected_rows": 1,
            }
        ],
    }
    (persist_directory / f"{_COLLECTION_NAME}.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    collection = _FakeCollection(
        document_field=document_field,
        indexed_document=indexed_document,
    )
    retriever = ChromaRetriever(
        collection_name=_COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_revision="model-revision",
        embedding_dimension=4,
        query_instruction="Represent this query for retrieving relevant SVGs",
        dataset_roots={"icon": dataset_root},
        precomputed_embeddings=True,
        document_field=document_field,
        device="cpu",
    )
    retriever._load_and_validate_manifest(collection)
    retriever._collection = collection
    retriever._query_encoder = _FakeQueryEncoder()
    return retriever, collection


@pytest.mark.parametrize(
    ("document_field", "indexed_document"),
    [("description", _DESCRIPTION), ("detail", _DETAIL)],
)
def test_precomputed_document_field_validates_index_and_returns_description_svg(
    tmp_path: Path,
    document_field: str,
    indexed_document: str,
) -> None:
    retriever, collection = _build_retriever(
        tmp_path,
        document_field=document_field,
        indexed_document=indexed_document,
    )

    examples = retriever.retrieve("draw a blue star badge", top_k=1)

    assert len(examples) == 1
    assert examples[0].description == _DESCRIPTION
    assert examples[0].content == _SVG
    assert examples[0].item_id == "icon:record-1"
    assert examples[0].score == pytest.approx(0.8)
    assert collection.last_query is not None
    assert collection.last_query["query_embeddings"] == [[0.1, 0.2, 0.3, 0.4]]
    assert "query_texts" not in collection.last_query


def test_detail_pointer_rejects_document_that_does_not_match_source_parquet(
    tmp_path: Path,
) -> None:
    retriever, _ = _build_retriever(
        tmp_path,
        document_field="detail",
        indexed_document="tampered detail",
    )

    examples = retriever.retrieve("draw a blue star badge", top_k=1)

    assert examples == []


@pytest.mark.parametrize("document_field", ["", "svg", "description+detail"])
def test_precomputed_retriever_rejects_unknown_document_field(
    document_field: str,
) -> None:
    with pytest.raises(ValueError, match="document_field"):
        ChromaRetriever(
            precomputed_embeddings=True,
            embedding_revision="revision",
            query_instruction="query instruction",
            document_field=document_field,
        )

