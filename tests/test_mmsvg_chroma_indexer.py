"""Focused offline tests for field-specific MMSVG Chroma indexing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from svg_agentic_slm.cli.commands_mmsvg_rag_index import (  # noqa: E402
    _resolve_collection_name,
    _resolve_persist_directory,
)
from svg_agentic_slm.rag.mmsvg_chroma_indexer import (  # noqa: E402
    DEFAULT_MAX_SEQ_LENGTH,
    MMSVGRecord,
    MMSVGSource,
    _build_manifest,
    _ensure_manifest,
    _index_record_batch,
    index_mmsvg_documents,
    iter_mmsvg_records,
    normalize_document_field,
    stable_record_id,
)
from svg_agentic_slm.utils.config import load_yaml_config  # noqa: E402


def _write_parquet(path: Path, **columns: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _source(data_dir: Path, dataset_type: str = "icon", rows: int | None = None) -> MMSVGSource:
    return MMSVGSource(
        dataset_type=dataset_type,
        dataset_id=f"org/mmsvg-{dataset_type}",
        dataset_revision=f"{dataset_type}-revision",
        data_dir=data_dir,
        expected_shards=1,
        expected_rows=rows,
    )


def _record(record_id: str, text: str, field: str = "detail") -> MMSVGRecord:
    return MMSVGRecord(
        record_id=record_id,
        document_text=text,
        document_field=field,
        dataset_type="icon",
        dataset_id="org/mmsvg-icon",
        dataset_revision="revision",
        shard="part.parquet",
        row_index_in_shard=int(record_id.rsplit("-", maxsplit=1)[-1]),
    )


def test_stable_record_id_is_shared_across_document_fields() -> None:
    assert stable_record_id("icon", "record-7") == "icon:record-7"
    assert stable_record_id("icon", "record-7") != stable_record_id(
        "illustration", "record-7"
    )
    # document_field is deliberately not an input, so description/detail joins
    # use exactly this same item ID in their separate collections.
    assert "description" not in stable_record_id("icon", "record-7")
    assert "detail" not in stable_record_id("icon", "record-7")


@pytest.mark.parametrize("field", ["description", "detail", " DETAIL "])
def test_normalize_document_field_accepts_only_supported_fields(field: str) -> None:
    assert normalize_document_field(field) in {"description", "detail"}


def test_normalize_document_field_rejects_unpinned_compositions() -> None:
    with pytest.raises(ValueError, match="description, detail"):
        normalize_document_field("description+detail")


@pytest.mark.parametrize(
    ("field", "expected"),
    [("description", ["short one", "short two"]), ("detail", ["long one", "long two"])],
)
def test_iter_records_projects_exactly_one_configured_text_field(
    tmp_path: Path,
    field: str,
    expected: list[str],
) -> None:
    data_dir = tmp_path / "icons"
    _write_parquet(
        data_dir / "part.parquet",
        id=["icon-0", "icon-1"],
        description=["short one", "short two"],
        detail=["long one", "long two"],
        keywords=["excluded one", "excluded two"],
        svg=["<svg id='excluded-0'/>", "<svg id='excluded-1'/>"]
    )

    records = list(
        iter_mmsvg_records(
            [_source(data_dir, rows=2)],
            document_field=field,
            read_batch_size=1,
        )
    )

    assert [record.document_text for record in records] == expected
    assert {record.document_field for record in records} == {field}
    assert [record.row_index_in_shard for record in records] == [0, 1]
    assert all(not hasattr(record, "svg") for record in records)
    assert all(not hasattr(record, "keywords") for record in records)


def test_resumable_batch_encodes_only_missing_detail_documents() -> None:
    records = [_record("icon-0", "existing detail"), _record("icon-1", "new detail")]

    class _Collection:
        def __init__(self) -> None:
            self.upserts: list[dict[str, Any]] = []

        def get(self, *, ids: list[str], include: list[str]) -> dict[str, Any]:
            assert ids == ["icon:icon-0", "icon:icon-1"]
            assert include == []
            return {"ids": ["icon:icon-0"]}

        def upsert(self, **kwargs: Any) -> None:
            self.upserts.append(kwargs)

    class _Encoder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[0.1, 0.2, 0.3]]

    collection = _Collection()
    encoder = _Encoder()

    indexed, existing = _index_record_batch(collection, encoder, records)  # type: ignore[arg-type]

    assert (indexed, existing) == (1, 1)
    assert encoder.calls == [["new detail"]]
    assert collection.upserts[0]["ids"] == ["icon:icon-1"]
    assert collection.upserts[0]["documents"] == ["new detail"]
    assert collection.upserts[0]["metadatas"] == [
        {
            "dataset_type": "icon",
            "record_id": "icon-1",
            "document_field": "detail",
            "shard": "part.parquet",
            "row_index_in_shard": 1,
        }
    ]


def test_detail_manifest_pins_field_template_model_context_and_sources(tmp_path: Path) -> None:
    source = _source(tmp_path / "icons", rows=2)
    manifest = _build_manifest(
        collection_name="mmsvg-detail-test",
        document_field="detail",
        sources=[source],
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_revision="model-commit",
        max_seq_length=768,
        query_instruction="retrieve SVG examples",
    )

    assert manifest["document_field"] == "detail"
    assert manifest["document_template"] == "{detail}"
    assert manifest["max_seq_length"] == 768
    assert manifest["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert manifest["embedding_revision"] == "model-commit"
    assert manifest["sources"][0]["dataset_revision"] == "icon-revision"

    path = tmp_path / "detail.manifest.json"
    _ensure_manifest(path, manifest)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        _ensure_manifest(path, {**manifest, "max_seq_length": 512})


def test_existing_description_manifest_is_accepted_without_rewriting(tmp_path: Path) -> None:
    expected = _build_manifest(
        collection_name="description-existing",
        document_field="description",
        sources=[_source(tmp_path / "icons", rows=1)],
        model_name="model",
        model_revision="revision",
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        query_instruction="instruction",
    )
    legacy = dict(expected)
    legacy.pop("max_seq_length")
    path = tmp_path / "description.manifest.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    _ensure_manifest(path, expected)

    assert json.loads(path.read_text(encoding="utf-8")) == legacy


def test_detail_defaults_are_physically_separate_from_active_description_config() -> None:
    description_config = {
        "collection_name": "mmsvg_description_qwen3_0_6b_1024_v1",
        "persist_directory": "./data/chroma_db",
    }

    assert _resolve_persist_directory(
        field="detail",
        override=None,
        chroma_config=description_config,
        indexing={},
    ) == Path("data/chroma_db_detail")
    assert _resolve_collection_name(
        field="detail",
        override=None,
        chroma_config=description_config,
        indexing={},
    ) == "mmsvg_detail_qwen3_0_6b_1024_v1"


def test_detail_config_is_queryable_and_physically_separate() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "rag_mmsvg_detail.yaml"
    rag = load_yaml_config(config_path)["rag"]

    assert rag["chromadb"]["document_field"] == "detail"
    assert rag["mmsvg_indexing"]["document_field"] == "detail"
    assert rag["chromadb"]["persist_directory"] == "./data/chroma_db_detail"
    assert rag["mmsvg_indexing"]["persist_directory"] == "./data/chroma_db_detail"
    assert rag["chromadb"]["collection_name"] == "mmsvg_detail_qwen3_0_6b_1024_v1"
    assert rag["mmsvg_indexing"]["collection_name"] == (
        "mmsvg_detail_qwen3_0_6b_1024_v1"
    )
    assert rag["chromadb"]["device"] == "cuda:1"
    assert rag["mmsvg_indexing"]["device"] == "cuda:1"


def test_index_mmsvg_documents_resumes_with_stable_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "icons"
    _write_parquet(
        data_dir / "part.parquet",
        id=["icon-0", "icon-1"],
        description=["short zero", "short one"],
        detail=["long zero", "long one"],
    )

    class _Collection:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}
            self.ids = {"icon:icon-0"}
            self.upserts: list[dict[str, Any]] = []

        def count(self) -> int:
            return len(self.ids)

        def get(self, *, ids: list[str], include: list[str]) -> dict[str, Any]:
            return {"ids": [item_id for item_id in ids if item_id in self.ids]}

        def upsert(self, **kwargs: Any) -> None:
            self.upserts.append(kwargs)
            self.ids.update(kwargs["ids"])

    class _Client:
        def __init__(self, collection: _Collection) -> None:
            self.collection = collection

        def get_or_create_collection(self, **kwargs: Any) -> _Collection:
            self.collection.metadata = dict(kwargs["metadata"])
            return self.collection

    class _Encoder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

    collection = _Collection()
    encoder = _Encoder()
    result = index_mmsvg_documents(
        document_field="detail",
        sources=[_source(data_dir, rows=2)],
        persist_directory=tmp_path / "chroma-detail",
        collection_name="detail-test",
        model_name="model",
        model_revision="revision",
        embedding_batch_size=2,
        read_batch_size=1,
        max_seq_length=256,
        query_instruction="instruction",
        encoder=encoder,  # type: ignore[arg-type]
        client=_Client(collection),
    )

    assert result.document_field == "detail"
    assert result.collection_count_before == 1
    assert result.collection_count_after == 2
    assert result.indexed_this_run == 1
    assert result.existing_this_run == 1
    assert encoder.calls == [["long one"]]
    assert collection.upserts[0]["ids"] == ["icon:icon-1"]
    manifest = json.loads(
        (tmp_path / "chroma-detail" / "detail-test.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["document_field"] == "detail"
    assert manifest["document_template"] == "{detail}"
    assert manifest["max_seq_length"] == 256
