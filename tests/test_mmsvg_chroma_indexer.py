"""Offline tests for streaming MMSVG description records from Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from svg_agentic_slm.rag.mmsvg_chroma_indexer import (  # noqa: E402
    MMSVGRecord,
    MMSVGSource,
    _index_record_batch,
    iter_mmsvg_records,
    stable_record_id,
)


def _write_parquet(path: Path, **columns: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _source(data_dir: Path, dataset_type: str) -> MMSVGSource:
    return MMSVGSource(
        dataset_type=dataset_type,
        dataset_id=f"org/mmsvg-{dataset_type}",
        dataset_revision=f"{dataset_type}-revision",
        data_dir=data_dir,
    )


def test_stable_record_id_is_deterministic_and_dataset_type_namespaced() -> None:
    assert stable_record_id("icon", "record-7") == "icon:record-7"
    assert stable_record_id("icon", "record-7") == stable_record_id("icon", "record-7")
    assert stable_record_id("icon", "record-7") != stable_record_id("illustration", "record-7")


def test_iter_records_uses_only_id_and_description_and_preserves_order(
    tmp_path: Path,
) -> None:
    icon_dir = tmp_path / "icons"
    illustration_dir = tmp_path / "illustrations"
    _write_parquet(
        icon_dir / "z-last.parquet",
        id=["icon-z"],
        description=["last icon shard"],
    )
    _write_parquet(
        icon_dir / "a-first.parquet",
        id=["icon-a0", "icon-a1"],
        description=["first icon", "second icon"],
        keyword=["must not be embedded", "also excluded"],
        detail=["must not be appended", "also excluded"],
        svg=["<svg id='ignored-0'/>", "<svg id='ignored-1'/>"],
    )
    _write_parquet(
        illustration_dir / "part.parquet",
        id=["illustration-0"],
        description=["one illustration"],
        keyword=["excluded illustration keyword"],
    )

    records = list(
        iter_mmsvg_records(
            [_source(icon_dir, "icon"), _source(illustration_dir, "illustration")],
            read_batch_size=1,
            limit=None,
        )
    )

    assert records == [
        MMSVGRecord(
            record_id="icon-a0",
            description="first icon",
            dataset_type="icon",
            dataset_id="org/mmsvg-icon",
            dataset_revision="icon-revision",
            shard="a-first.parquet",
            row_index_in_shard=0,
        ),
        MMSVGRecord(
            record_id="icon-a1",
            description="second icon",
            dataset_type="icon",
            dataset_id="org/mmsvg-icon",
            dataset_revision="icon-revision",
            shard="a-first.parquet",
            row_index_in_shard=1,
        ),
        MMSVGRecord(
            record_id="icon-z",
            description="last icon shard",
            dataset_type="icon",
            dataset_id="org/mmsvg-icon",
            dataset_revision="icon-revision",
            shard="z-last.parquet",
            row_index_in_shard=0,
        ),
        MMSVGRecord(
            record_id="illustration-0",
            description="one illustration",
            dataset_type="illustration",
            dataset_id="org/mmsvg-illustration",
            dataset_revision="illustration-revision",
            shard="part.parquet",
            row_index_in_shard=0,
        ),
    ]
    assert all(not hasattr(record, "keyword") for record in records)
    assert all(not hasattr(record, "detail") for record in records)
    assert all(not hasattr(record, "svg") for record in records)


def test_iter_records_applies_one_global_limit_across_sources(tmp_path: Path) -> None:
    icon_dir = tmp_path / "icons"
    illustration_dir = tmp_path / "illustrations"
    _write_parquet(
        icon_dir / "part.parquet",
        id=["icon-0", "icon-1"],
        description=["zero", "one"],
    )
    _write_parquet(
        illustration_dir / "part.parquet",
        id=["illustration-0", "illustration-1"],
        description=["two", "three"],
    )

    records = list(
        iter_mmsvg_records(
            [_source(icon_dir, "icon"), _source(illustration_dir, "illustration")],
            read_batch_size=1,
            limit=3,
        )
    )

    assert [record.record_id for record in records] == [
        "icon-0",
        "icon-1",
        "illustration-0",
    ]


@pytest.mark.parametrize("description", ["", "   ", None])
def test_iter_records_rejects_blank_description(
    tmp_path: Path,
    description: str | None,
) -> None:
    data_dir = tmp_path / "icons"
    _write_parquet(
        data_dir / "part.parquet",
        id=["icon-0"],
        description=[description],
    )

    with pytest.raises(ValueError, match="description"):
        list(
            iter_mmsvg_records(
                [_source(data_dir, "icon")],
                read_batch_size=1,
                limit=None,
            )
        )


@pytest.mark.parametrize("record_id", ["", "   ", None])
def test_iter_records_rejects_blank_record_id(
    tmp_path: Path,
    record_id: str | None,
) -> None:
    data_dir = tmp_path / "icons"
    _write_parquet(
        data_dir / "part.parquet",
        id=[record_id],
        description=["valid description"],
    )

    with pytest.raises(ValueError, match="id"):
        list(
            iter_mmsvg_records(
                [_source(data_dir, "icon")],
                read_batch_size=1,
                limit=None,
            )
        )


def test_resumable_batch_encodes_only_missing_raw_descriptions() -> None:
    records = [
        MMSVGRecord(
            record_id="icon-0",
            description="already indexed description",
            dataset_type="icon",
            dataset_id="org/mmsvg-icon",
            dataset_revision="revision",
            shard="part.parquet",
            row_index_in_shard=0,
        ),
        MMSVGRecord(
            record_id="icon-1",
            description="new raw description",
            dataset_type="icon",
            dataset_id="org/mmsvg-icon",
            dataset_revision="revision",
            shard="part.parquet",
            row_index_in_shard=1,
        ),
    ]

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

    indexed, existing = _index_record_batch(collection, encoder, records)

    assert (indexed, existing) == (1, 1)
    assert encoder.calls == [["new raw description"]]
    assert collection.upserts == [
        {
            "ids": ["icon:icon-1"],
            "embeddings": [[0.1, 0.2, 0.3]],
            "documents": ["new raw description"],
            "metadatas": [
                {
                    "dataset_type": "icon",
                    "record_id": "icon-1",
                    "shard": "part.parquet",
                    "row_index_in_shard": 1,
                }
            ],
        }
    ]
