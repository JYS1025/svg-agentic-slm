from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from svg_agentic_slm.data.mmsvg_critic_selection import (
    Candidate,
    Exclusions,
    _audit_row,
    annotate_semantic_buckets,
    canonical_svg_sha256,
    caption_sha256,
    load_exclusions,
    scan_domain,
    select_domain_candidates,
)


def _svg(body: str = '<path d="M0 0 L10 10 Z" fill="#112233"/>') -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">{body}</svg>'


def _candidate(index: int, *, domain: str = "icon", bucket: int | None = None) -> Candidate:
    value = index + 1
    return Candidate(
        domain=domain,
        record_id=f"id-{index}",
        item_id=f"{domain}:id-{index}",
        shard=f"/tmp/{domain}.parquet",
        row_index_in_shard=index,
        canonical_svg_sha256=f"{value:064x}",
        caption_sha256=f"{value + 100:064x}",
        detail_sha256=f"{value + 200:064x}",
        token_len=100 + index,
        svg_char_length=200 + index,
        description_words=5,
        detail_words=10,
        keyword_count=3,
        element_count=2 + index,
        path_count=1 + index,
        path_command_count=2 + index,
        color_count=1 + index,
        arc_command_count=index % 2,
        curve_command_count=index % 3,
        rare_command_count=index % 4,
        style_feature_count=index % 2,
        advanced_element_count=0,
        gradient_count=0,
        clip_path_count=0,
        mask_count=0,
        filter_count=0,
        text_element_count=0,
        use_element_count=0,
        image_element_count=0,
        transform_count=index % 2,
        stroke_count=index % 2,
        opacity_count=index % 2,
        viewbox_aspect_ratio=1.0 + index,
        command_signature="AL",
        tie_break=index,
        semantic_bucket=index if bucket is None else bucket,
    )


class _FakeCollection:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def count(self) -> int:
        return len(self.vectors)

    def get(self, *, ids: list[str], include: list[str]) -> dict[str, Any]:
        assert include == ["embeddings", "metadatas"]
        self.calls.append(ids)
        returned = list(reversed(ids))
        return {
            "ids": returned,
            "embeddings": np.stack([self.vectors[item_id] for item_id in returned]),
            "metadatas": [
                {
                    "dataset_type": item_id.split(":", 1)[0],
                    "record_id": item_id.split(":", 1)[1],
                }
                for item_id in returned
            ],
        }


def test_dual_chroma_embeddings_are_rejoined_and_deterministic() -> None:
    first = _candidate(0)
    second = _candidate(1)
    vectors = {
        first.item_id: np.eye(1, 1024, 0, dtype=np.float32)[0],
        second.item_id: np.eye(1, 1024, 1, dtype=np.float32)[0],
    }
    detail_vectors = {
        first.item_id: vectors[first.item_id].copy(),
        second.item_id: vectors[second.item_id].copy(),
    }
    description = _FakeCollection(vectors)
    detail = _FakeCollection(detail_vectors)

    contract = annotate_semantic_buckets(
        [first, second],
        description_collection=description,
        detail_collection=detail,
        seed=42,
        batch_size=2,
        projection_dimension=16,
        lsh_bits=8,
    )

    assert contract["embedding_dimension"] == 1024
    assert first.description_detail_cosine == pytest.approx(1.0)
    assert second.description_detail_cosine == pytest.approx(1.0)
    assert first.semantic_bucket >= 0
    assert second.semantic_bucket >= 0
    assert description.calls == [[first.item_id, second.item_id]]


def test_chroma_embedding_validation_fails_closed() -> None:
    candidate = _candidate(0)
    bad = np.ones(1024, dtype=np.float32)
    bad[10] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        annotate_semantic_buckets(
            [candidate],
            description_collection=_FakeCollection({candidate.item_id: bad}),
            detail_collection=_FakeCollection({candidate.item_id: np.ones(1024, dtype=np.float32)}),
            seed=42,
            batch_size=1,
            projection_dimension=8,
            lsh_bits=4,
        )


def test_audit_row_preserves_unsigned_tie_break_for_arrow() -> None:
    candidate = _candidate(0)
    candidate.tie_break = (1 << 64) - 1

    row = _audit_row(candidate)
    table = pa.Table.from_pylist([row])

    assert row["tie_break"] == "ffffffffffffffff"
    assert table.schema.field("tie_break").type == pa.string()


def test_scan_excludes_generator_and_benchmark_leakage(tmp_path: Path) -> None:
    generator_svg = _svg('<rect height="10" width="10"/>')
    benchmark_svg = _svg('<circle cx="5" cy="5" r="4"/>')
    keep_svg = _svg()
    rows = [
        ("keep", keep_svg, "keep caption", "keep detail"),
        ("generator-id", _svg('<path d="M1 1 L2 2"/>'), "x", "x detail"),
        ("generator-hash-copy", generator_svg, "y", "y detail"),
        ("benchmark-text", _svg('<path d="M2 2 L3 3"/>'), "benchmark caption", "z"),
        ("benchmark-detail", _svg('<path d="M3 3 L4 4"/>'), "z", "benchmark detail"),
        ("benchmark-hash-copy", benchmark_svg, "z2", "z2 detail"),
        ("final-reference-id", _svg('<path d="M4 4 L5 5"/>'), "f", "f detail"),
        ("final-reference-hash-copy", _svg('<ellipse rx="2" ry="3"/>'), "g", "g detail"),
        (
            "unsafe",
            '<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
            "unsafe",
            "unsafe detail",
        ),
    ]
    table = pa.table(
        {
            "id": [row[0] for row in rows],
            "svg": [row[1] for row in rows],
            "description": [row[2] for row in rows],
            "keywords": ["a, b"] * len(rows),
            "detail": [row[3] for row in rows],
            "token_len": list(range(10, 10 + len(rows))),
        }
    )
    shard = tmp_path / "part.parquet"
    pq.write_table(table, shard)
    exclusions = Exclusions(
        generator_raw_ids=frozenset({"generator-id"}),
        generator_raw_ids_by_domain={"icon": frozenset({"generator-id"})},
        generator_svg_hashes=frozenset({canonical_svg_sha256(generator_svg)}),
        final_reference_item_ids=frozenset({"icon:final-reference-id"}),
        final_reference_svg_hashes=frozenset(
            {canonical_svg_sha256(_svg('<ellipse rx="2" ry="3"/>'))}
        ),
        benchmark_raw_ids=frozenset(),
        benchmark_text_hashes=frozenset(
            {caption_sha256("benchmark caption"), caption_sha256("benchmark detail")}
        ),
        benchmark_svg_hashes=frozenset({canonical_svg_sha256(benchmark_svg)}),
    )

    scan = scan_domain(
        domain="icon",
        shards=[shard],
        exclusions=exclusions,
        seed=42,
        batch_size=3,
    )

    assert [candidate.record_id for candidate in scan.candidates] == ["keep"]
    assert not (
        {candidate.item_id for candidate in scan.candidates} & exclusions.final_reference_item_ids
    )
    assert not (
        {candidate.canonical_svg_sha256 for candidate in scan.candidates}
        & exclusions.final_reference_svg_hashes
    )
    assert scan.statistics["rejected_generator_id"] == 1
    assert scan.statistics["rejected_generator_svg_hash"] == 1
    assert scan.statistics["rejected_benchmark_text"] == 2
    assert scan.statistics["rejected_benchmark_svg_hash"] == 1
    assert scan.statistics["rejected_final_reference_item_id"] == 1
    assert scan.statistics["rejected_final_reference_svg_hash"] == 1
    assert scan.statistics["rejected_svg_validation"] == 1


def test_selection_is_deterministic_and_deduplicates_svg_and_text_pairs() -> None:
    base = [_candidate(index, bucket=index) for index in range(12)]
    base[1].canonical_svg_sha256 = base[0].canonical_svg_sha256
    base[3].caption_sha256 = base[2].caption_sha256
    base[3].detail_sha256 = base[2].detail_sha256
    rejected_id = base[-1].item_id

    def run(values: list[Candidate]) -> tuple[list[str], Any]:
        copied = [
            replace(candidate, selection_reason="", selection_rank_in_domain=-1)
            for candidate in values
        ]
        selection = select_domain_candidates(
            copied,
            domain="icon",
            sample_size=6,
            edge_quota=2,
            lsh_bits=8,
            seed=42,
            accept=lambda candidate: candidate.item_id != rejected_id,
        )
        return [candidate.item_id for candidate in selection.selected], selection

    first_ids, first = run(base)
    second_ids, second = run(list(reversed(base)))
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids)) == 6
    hashes = [candidate.canonical_svg_sha256 for candidate in first.selected]
    text_pairs = [
        (candidate.caption_sha256, candidate.detail_sha256) for candidate in first.selected
    ]
    assert len(hashes) == len(set(hashes))
    assert len(text_pairs) == len(set(text_pairs))
    assert first.statistics["selected_edge"] == 2
    assert first.statistics["selected_semantic"] == 4
    assert second.quota_targets == first.quota_targets


def test_generator_prepared_hashes_and_benchmark_fields_are_loaded(tmp_path: Path) -> None:
    ids_dir = tmp_path / "ids"
    ids_dir.mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    benchmark = tmp_path / "benchmark.jsonl"
    final_references = tmp_path / "final_references.parquet"
    svg_by_id = {
        "icon-a": _svg('<rect width="2" height="2"/>'),
        "illustration-a": _svg('<circle r="2"/>'),
    }
    id_files: dict[str, Path] = {}
    prepared_rows = []
    for domain, record_id in (("icon", "icon-a"), ("illustration", "illustration-a")):
        path = ids_dir / f"{domain}.txt"
        path.write_text(record_id + "\n", encoding="utf-8")
        id_files[domain] = path
        svg = svg_by_id[record_id]
        prepared_rows.append(
            {
                "output_svg": svg,
                "metadata": {
                    "dataset_type": domain,
                    "record_id": record_id,
                    "canonical_svg_sha256": canonical_svg_sha256(svg),
                },
            }
        )
    (prepared / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prepared_rows),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "id": "bench-id",
                "caption": "Benchmark caption",
                "detail": "Benchmark detail",
                "svg_hashes": ["a" * 64],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "item_id": ["icon:rag-reference", "icon:rag-reference"],
                "canonical_svg_sha256": ["b" * 64, "b" * 64],
            }
        ),
        final_references,
    )

    exclusions = load_exclusions(id_files, prepared, [benchmark], final_references)

    assert exclusions.generator_raw_ids == {"icon-a", "illustration-a"}
    assert len(exclusions.generator_svg_hashes) == 2
    assert exclusions.benchmark_raw_ids == {"bench-id"}
    assert caption_sha256("Benchmark caption") in exclusions.benchmark_text_hashes
    assert caption_sha256("Benchmark detail") in exclusions.benchmark_text_hashes
    assert exclusions.benchmark_svg_hashes == {"a" * 64}
    assert exclusions.final_reference_item_ids == {"icon:rag-reference"}
    assert exclusions.final_reference_svg_hashes == {"b" * 64}
    assert exclusions.final_references_provenance["rows"] == 2
    assert exclusions.final_references_provenance["unique_item_ids"] == 1
