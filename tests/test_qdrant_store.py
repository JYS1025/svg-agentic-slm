"""Network-free tests for the Qdrant retriever."""

from __future__ import annotations

import base64
import gzip
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from svg_agentic_slm.rag.qdrant_store import QdrantRetriever


class _ModelValue:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeModels:
    PointStruct = _ModelValue
    VectorParams = _ModelValue
    VectorParamsDiff = _ModelValue
    HnswConfigDiff = _ModelValue
    CollectionParamsDiff = _ModelValue
    ScalarQuantization = _ModelValue
    ScalarQuantizationConfig = _ModelValue
    Distance = SimpleNamespace(COSINE="cosine")
    ScalarType = SimpleNamespace(INT8="int8")


class _FakeEncoder:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: Any, **kwargs: Any) -> Any:
        self.calls.append((texts, kwargs))
        if isinstance(texts, str):
            return [0.1] * self.dimension
        return [[float(index + 1)] * self.dimension for index, _ in enumerate(texts)]


class _FakeClient:
    def __init__(self, *, exists: bool = False, dimension: int = 3) -> None:
        self.exists = exists
        self.dimension = dimension
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.query_hits: list[Any] = []
        self.deleted: list[str] = []
        self.point_ids: set[str] = set()

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: Any) -> None:
        self.created.append(kwargs)
        self.exists = True

    def update_collection(self, **kwargs: Any) -> bool:
        self.updated.append(kwargs)
        return True

    def get_collection(self, collection_name: str) -> Any:
        vectors = SimpleNamespace(size=self.dimension)
        params = SimpleNamespace(vectors=vectors)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def upload_points(self, **kwargs: Any) -> None:
        self.uploads.append(kwargs)
        self.point_ids.update(str(point.id) for point in kwargs["points"])

    def query_points(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return SimpleNamespace(points=self.query_hits)

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool,
        with_vectors: bool,
    ) -> list[Any]:
        return [SimpleNamespace(id=point_id) for point_id in ids if point_id in self.point_ids]

    def count(self, *, collection_name: str, exact: bool) -> Any:
        return SimpleNamespace(count=len(self.point_ids))

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        self.exists = False
        self.point_ids.clear()


def _retriever(
    client: _FakeClient,
    encoder: _FakeEncoder | None = None,
    **kwargs: Any,
) -> QdrantRetriever:
    return QdrantRetriever(
        client=client,
        encoder=encoder or _FakeEncoder(),
        models_api=_FakeModels,
        **kwargs,
    )


def test_empty_inputs_do_not_initialize_optional_dependencies() -> None:
    retriever = QdrantRetriever()

    retriever.add_documents([])

    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("query", top_k=0) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"similarity_threshold": -0.1}, "similarity_threshold"),
        ({"similarity_threshold": 1.1}, "similarity_threshold"),
        ({"upload_batch_size": 0}, "upload_batch_size"),
        ({"collection_name": "  "}, "collection_name"),
    ],
)
def test_constructor_rejects_invalid_settings(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        QdrantRetriever(**kwargs)


def test_add_documents_creates_collection_batches_and_uses_stable_ids() -> None:
    client = _FakeClient()
    encoder = _FakeEncoder(dimension=3)
    retriever = _retriever(
        client,
        encoder,
        collection_name="svg-test",
        embedding_model="fake-encoder",
        upload_batch_size=1,
    )
    long_svg = "<svg>" + ("<rect width='1' height='1'/>" * 100) + "</svg>"
    documents = [
        {
            "content": long_svg,
            "metadata": {
                "point_identity": "hf://dataset/main/train/example.svg",
                "description": "many rectangles",
                "tags": ["grid", "red"],
                "search_text": "Description: many rectangles",
            },
        },
        {
            "content": "<svg><circle r='4'/></svg>",
            "metadata": {"description": "a circle", "source": "circle.svg"},
        },
        {"content": "   ", "metadata": {"description": "ignored"}},
    ]

    retriever.add_documents(documents)

    assert len(client.created) == 1
    create = client.created[0]
    assert create["collection_name"] == "svg-test"
    assert create["vectors_config"].size == 3
    assert create["vectors_config"].distance == "cosine"
    assert create["vectors_config"].on_disk is True
    assert create["on_disk_payload"] is True
    assert create["hnsw_config"].on_disk is True
    assert create["quantization_config"].scalar.type == "int8"

    assert len(client.uploads) == 2
    first_point = client.uploads[0]["points"][0]
    expected_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "hf://dataset/main/train/example.svg",
        )
    )
    assert first_point.id == expected_id
    assert first_point.vector == [1.0, 1.0, 1.0]
    assert first_point.payload["source"] == "hf://dataset/main/train/example.svg"
    assert first_point.payload["embedding_model"] == "fake-encoder"
    assert first_point.payload["svg_codec"] == "gzip+base64"
    assert "point_identity" not in first_point.payload

    assert encoder.calls[0][0] == ["Description: many rectangles"]
    assert encoder.calls[1][0] == [
        "Description: a circle",
    ]
    assert all(call[1]["normalize_embeddings"] for call in encoder.calls)

    retriever.add_documents([documents[0]])
    assert client.uploads[-1]["points"][0].id == expected_id
    assert len(client.point_ids) == 2


def test_optimize_storage_updates_an_existing_collection() -> None:
    client = _FakeClient(exists=True)
    retriever = _retriever(client, collection_name="svg-test")

    retriever.optimize_storage()

    assert len(client.updated) == 1
    update = client.updated[0]
    assert update["collection_name"] == "svg-test"
    assert update["vectors_config"][""].on_disk is True
    assert update["collection_params"].on_disk_payload is True
    assert update["hnsw_config"].on_disk is True
    assert update["quantization_config"].scalar.type == "int8"


def test_missing_documents_filters_stable_ids_without_embedding() -> None:
    client = _FakeClient()
    encoder = _FakeEncoder()
    retriever = _retriever(client, encoder)
    documents = [
        {
            "content": "<svg><circle/></svg>",
            "metadata": {
                "point_identity": "circle",
                "description": "circle",
            },
        },
        {
            "content": "<svg><rect/></svg>",
            "metadata": {
                "point_identity": "rectangle",
                "description": "rectangle",
            },
        },
    ]
    existing_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "circle"))
    client.point_ids.add(existing_id)

    missing = retriever.missing_documents(documents)

    assert missing == [documents[1]]
    assert encoder.calls == []


def test_none_content_and_none_only_search_text_are_not_indexed() -> None:
    client = _FakeClient()
    encoder = _FakeEncoder()
    retriever = _retriever(client, encoder)

    retriever.add_documents(
        [
            {
                "content": None,
                "metadata": {"search_text": "must still be skipped"},
            },
            {
                "content": "<svg/>",
                "metadata": {"search_text": None},
            },
        ]
    )

    assert client.uploads == []
    assert encoder.calls == []


def test_retrieve_maps_payload_and_forwards_query_options() -> None:
    client = _FakeClient(exists=True)
    encoder = _FakeEncoder()
    retriever = _retriever(
        client,
        encoder,
        collection_name="svg-test",
        similarity_threshold=0.55,
    )
    client.query_hits = [
        SimpleNamespace(
            id="point-1",
            score=0.91,
            payload={
                "svg_data": "<svg><path d='M0 0'/></svg>",
                "svg_codec": "utf8",
                "description": "a path",
                "source": "sample.svg",
                "tags": ["line"],
            },
        ),
        SimpleNamespace(
            id="point-without-svg",
            score=0.8,
            payload={"description": "skip me"},
        ),
    ]

    examples = retriever.retrieve("  draw a path  ", top_k=7)

    assert len(examples) == 1
    example = examples[0]
    assert example.content == "<svg><path d='M0 0'/></svg>"
    assert example.description == "a path"
    assert example.score == pytest.approx(0.91)
    assert example.source == "sample.svg"
    assert example.metadata == {
        "tags": ["line"],
        "qdrant_point_id": "point-1",
        "qdrant_collection": "svg-test",
    }
    assert client.queries == [
        {
            "collection_name": "svg-test",
            "query": [0.1, 0.1, 0.1],
            "limit": 7,
            "score_threshold": 0.55,
            "with_payload": True,
            "with_vectors": False,
        }
    ]
    assert encoder.calls[0][0] == "draw a path"


def test_truncated_gzip_payload_is_skipped_instead_of_raising() -> None:
    client = _FakeClient(exists=True)
    retriever = _retriever(client)
    truncated = gzip.compress(b"<svg/>")[:-2]
    client.query_hits = [
        SimpleNamespace(
            id="broken-point",
            score=0.8,
            payload={
                "svg_codec": "gzip+base64",
                "svg_data": base64.b64encode(truncated).decode("ascii"),
            },
        )
    ]

    assert retriever.retrieve("query") == []


def test_collection_dimension_mismatch_fails_before_query() -> None:
    client = _FakeClient(exists=True, dimension=5)
    retriever = _retriever(client, _FakeEncoder(dimension=3))

    with pytest.raises(RuntimeError, match="dimension is 5"):
        retriever.retrieve("query")

    assert client.queries == []


def test_missing_collection_gives_indexing_instruction() -> None:
    retriever = _retriever(_FakeClient(exists=False))

    with pytest.raises(RuntimeError, match="rag-index"):
        retriever.retrieve("query")


def test_count_and_clear_use_injected_client_without_optional_packages() -> None:
    client = _FakeClient(exists=True)
    client.point_ids.update({"one", "two"})
    retriever = _retriever(client, collection_name="svg-test")

    assert retriever.count() == 2

    retriever.clear()

    assert client.deleted == ["svg-test"]
    assert len(client.created) == 1
    assert retriever.count() == 0
