"""Tests for resumable, leakage-safe SFT RAG artifact generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from svg_agentic_slm.cli import commands_precompute_sft_rag
from svg_agentic_slm.cli.app import app
from svg_agentic_slm.data import mmsvg_sft_rag_precompute as precompute_module
from svg_agentic_slm.data.mmsvg_sft_rag_precompute import (
    MMSVGSFTRAGPrecomputeConfig,
    _resolve_instruction,
    _retrieve_sample_batch,
    _validate_fixed_selection,
    precompute_mmsvg_sft_rag,
)
from svg_agentic_slm.rag.mmsvg_hybrid import (
    HybridCandidateTrace,
    HybridHitTrace,
    HybridRetrievalRequest,
    HybridRetrievalTrace,
    canonical_svg_sha256,
)
from svg_agentic_slm.rag.schemas import RetrievedExample

runner = CliRunner()


class _FakeHybridRetriever:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        leak_target: bool = False,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.leak_target = leak_target
        self.calls: list[tuple[str, set[str], set[str]]] = []

    def retrieve_with_trace(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_item_ids: set[str] | frozenset[str] | None = None,
        exclude_svg_sha256s: set[str] | frozenset[str] | None = None,
    ) -> tuple[list[RetrievedExample], HybridRetrievalTrace]:
        excluded_ids = set(exclude_item_ids or ())
        excluded_hashes = set(exclude_svg_sha256s or ())
        self.calls.append((query, excluded_ids, excluded_hashes))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("injected interruption")
        assert top_k == 3
        target_id = next(iter(excluded_ids))
        item_ids = (
            [target_id, "icon:reference-b", "icon:reference-c"]
            if self.leak_target
            else ["icon:reference-a", "icon:reference-b", "icon:reference-c"]
        )
        colors = ["red", "green", "blue"]
        examples: list[RetrievedExample] = []
        candidates: list[HybridCandidateTrace] = []
        description_hits: list[HybridHitTrace] = [
            HybridHitTrace(
                branch="description",
                raw_rank=1,
                item_id=target_id,
                embedding_score=0.99,
                eligible_rank=None,
                canonical_svg_sha256=None,
                excluded=True,
                exclusion_reason="target_item_id",
            )
        ]
        for rank, (item_id, color) in enumerate(zip(item_ids, colors, strict=True), 1):
            svg = _svg(color)
            svg_hash = canonical_svg_sha256(svg)
            image_hash = hashlib.sha256(f"image:{item_id}".encode()).hexdigest()
            examples.append(
                RetrievedExample(
                    content=svg,
                    item_id=item_id,
                    source=f"hf://mmsvg/{item_id}",
                    description=f"description {item_id}",
                    detail=f"detail {item_id}",
                    score=1.0 - rank / 10,
                    score_kind="siglip2_pair_probability",
                    rank=rank,
                    corpus_version="revision",
                    metadata={
                        "dataset_type": "icon",
                        "record_id": item_id.split(":", 1)[-1],
                        "shard": "train-00000-of-00001.parquet",
                        "row_index_in_shard": rank,
                        "canonical_svg_sha256": svg_hash,
                        "image_sha256": image_hash,
                        "selection_reason": "siglip2",
                    },
                )
            )
            description_hits.append(
                HybridHitTrace(
                    branch="description",
                    raw_rank=rank + 1,
                    item_id=item_id,
                    embedding_score=1.0 - rank / 10,
                    eligible_rank=rank,
                    canonical_svg_sha256=svg_hash,
                    excluded=False,
                    exclusion_reason=None,
                )
            )
            candidates.append(
                HybridCandidateTrace(
                    item_id=item_id,
                    canonical_svg_sha256=svg_hash,
                    description_rank=rank + 1,
                    description_score=1.0 - rank / 10,
                    detail_rank=None,
                    detail_score=None,
                    is_overlap=False,
                    siglip2_score=1.0 - rank / 10,
                    siglip2_raw_logit=float(rank),
                    image_sha256=image_hash,
                    eligible=True,
                    exclusion_reason=None,
                    selected=True,
                    final_rank=rank,
                    selection_reason="siglip2",
                )
            )
        trace = HybridRetrievalTrace(
            query=query,
            search_k=32,
            branch_top_k=3,
            final_top_k=3,
            excluded_item_ids=tuple(sorted(excluded_ids)),
            excluded_svg_sha256s=tuple(sorted(excluded_hashes)),
            description_hits=tuple(description_hits),
            detail_hits=(),
            candidates=tuple(candidates),
            final_item_ids=tuple(item_ids),
        )
        return examples, trace


class _FakeBatchHybridRetriever(_FakeHybridRetriever):
    def __init__(
        self,
        *,
        fail_on_batch_call: int | None = None,
        oom_above: int | None = None,
        leak_target: bool = False,
    ) -> None:
        super().__init__(leak_target=leak_target)
        self.fail_on_batch_call = fail_on_batch_call
        self.oom_above = oom_above
        self.batch_calls: list[tuple[HybridRetrievalRequest, ...]] = []

    def retrieve_many_with_trace(
        self,
        requests: Sequence[HybridRetrievalRequest],
    ) -> list[tuple[list[RetrievedExample], HybridRetrievalTrace]]:
        ordered = tuple(requests)
        self.batch_calls.append(ordered)
        if (
            self.fail_on_batch_call is not None
            and len(self.batch_calls) == self.fail_on_batch_call
        ):
            raise RuntimeError("injected interruption")
        if self.oom_above is not None and len(ordered) > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        return [
            self.retrieve_with_trace(
                request.query,
                top_k=request.top_k,
                exclude_item_ids=request.exclude_item_ids,
                exclude_svg_sha256s=request.exclude_svg_sha256s,
            )
            for request in ordered
        ]


def test_precompute_resumes_and_publishes_normalized_artifacts(
    tmp_path: Path,
) -> None:
    config = _fixture_config(
        tmp_path,
        output_name="result",
        retrieval_batch_size=1,
    )
    interrupted = _FakeBatchHybridRetriever(fail_on_batch_call=2)

    with pytest.raises(RuntimeError, match="injected interruption"):
        precompute_mmsvg_sft_rag(config, retriever=interrupted)

    checkpoints = sorted(
        (tmp_path / ".result.resume" / "checkpoints").glob("*.json")
    )
    assert len(checkpoints) == 1

    resumed = _FakeBatchHybridRetriever()
    manifest = precompute_mmsvg_sft_rag(config, retriever=resumed)

    assert len(resumed.batch_calls) == 1
    assert len(resumed.calls) == 1
    assert resumed.calls[0][0] == "illustration detail"
    assert resumed.calls[0][1] == {"illustration:illustration-target"}
    assert len(resumed.calls[0][2]) == 1
    assert manifest["sample_rows"] == 2
    assert manifest["final_reference_rows"] == 6

    output = tmp_path / "result"
    expected = {
        "samples",
        "retrieval_hits",
        "candidate_scores",
        "final_references",
        "corpus_items",
    }
    assert set(manifest["artifacts"]) == expected
    for entry in manifest["artifacts"].values():
        path = output / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert pq.ParquetFile(path).metadata.num_rows == entry["rows"]

    samples = pq.read_table(output / "samples.parquet").to_pylist()
    assert [row["instruction_source"] for row in samples] == [
        "description",
        "detail",
    ]
    final = pq.read_table(output / "final_references.parquet").to_pylist()
    assert len(final) == 6
    assert all(
        row["item_id"]
        not in {"icon:icon-target", "illustration:illustration-target"}
        for row in final
    )
    assert [row["final_rank"] for row in final[:3]] == [1, 2, 3]

    corpus = pq.read_table(output / "corpus_items.parquet").to_pylist()
    assert len(corpus) == 3
    assert {
        "item_id",
        "dataset_type",
        "record_id",
        "shard",
        "row_index_in_shard",
    }.issubset(corpus[0])

    already_complete = _FakeBatchHybridRetriever(fail_on_batch_call=1)
    second_manifest = precompute_mmsvg_sft_rag(
        config,
        retriever=already_complete,
    )
    assert already_complete.batch_calls == []
    assert already_complete.calls == []
    assert second_manifest["run_contract_sha256"] == manifest["run_contract_sha256"]


def test_precompute_uses_ordered_batch_requests_and_records_batch_size(
    tmp_path: Path,
) -> None:
    config = _fixture_config(
        tmp_path,
        output_name="batched",
        retrieval_batch_size=2,
    )
    retriever = _FakeBatchHybridRetriever()

    manifest = precompute_mmsvg_sft_rag(config, retriever=retriever)

    assert len(retriever.batch_calls) == 1
    requests = retriever.batch_calls[0]
    assert [request.query for request in requests] == [
        "icon description",
        "illustration detail",
    ]
    assert [request.top_k for request in requests] == [3, 3]
    assert [request.exclude_item_ids for request in requests] == [
        frozenset({"icon:icon-target"}),
        frozenset({"illustration:illustration-target"}),
    ]
    assert [request.exclude_svg_sha256s for request in requests] == [
        frozenset({canonical_svg_sha256(_svg("black"))}),
        frozenset({canonical_svg_sha256(_svg("white"))}),
    ]
    assert manifest["run_contract"]["execution"] == {
        "retrieval_batch_size": 2,
        "cuda_oom_backoff": "halve failing batch until one sample",
    }
    sample_rows = pq.read_table(
        tmp_path / "batched" / "samples.parquet"
    ).to_pylist()
    assert [row["instruction"] for row in sample_rows] == [
        "icon description",
        "illustration detail",
    ]


def test_cuda_oom_recursively_halves_batch_and_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        SimpleNamespace(
            instruction=f"prompt-{index}",
            target_item_id=f"icon:target-{index}",
            target_svg_sha256=f"{index + 1:064x}",
        )
        for index in range(4)
    ]
    retriever = _FakeBatchHybridRetriever(oom_above=1)
    cache_releases: list[None] = []
    monkeypatch.setattr(
        precompute_module,
        "_release_cuda_cache",
        lambda: cache_releases.append(None),
    )

    results = _retrieve_sample_batch(retriever, samples)  # type: ignore[arg-type]

    assert [len(requests) for requests in retriever.batch_calls] == [
        4,
        2,
        1,
        1,
        2,
        1,
        1,
    ]
    assert len(cache_releases) == 3
    assert [trace.query for _examples, trace in results] == [
        "prompt-0",
        "prompt-1",
        "prompt-2",
        "prompt-3",
    ]
    singleton_requests = [
        requests[0] for requests in retriever.batch_calls if len(requests) == 1
    ]
    assert [request.query for request in singleton_requests] == [
        "prompt-0",
        "prompt-1",
        "prompt-2",
        "prompt-3",
    ]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_precompute_config_rejects_invalid_retrieval_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(ValueError, match="retrieval_batch_size"):
        MMSVGSFTRAGPrecomputeConfig(
            input_dir=Path("input"),
            output_dir=Path("output"),
            rag_config_path=Path("rag.yaml"),
            selection_manifest_path=Path("manifest.json"),
            retrieval_batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_precompute_cli_forwards_batch_size_and_resume_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "prepared"
    input_dir.mkdir()
    rag_config = tmp_path / "rag.yaml"
    rag_config.write_text("rag: {}\n", encoding="utf-8")
    selection_manifest = tmp_path / "selection.json"
    selection_manifest.write_text("{}\n", encoding="utf-8")
    captured: dict[str, MMSVGSFTRAGPrecomputeConfig] = {}

    def fake_precompute(config: MMSVGSFTRAGPrecomputeConfig) -> dict[str, Any]:
        captured["config"] = config
        return {"sample_rows": 0}

    monkeypatch.setattr(
        commands_precompute_sft_rag,
        "precompute_mmsvg_sft_rag",
        fake_precompute,
    )

    result = runner.invoke(
        app,
        [
            "precompute-sft-rag",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--rag-config",
            str(rag_config),
            "--selection-manifest",
            str(selection_manifest),
            "--batch-size",
            "257",
            "--no-resume",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].retrieval_batch_size == 257
    assert captured["config"].resume is False


def test_precompute_fails_closed_if_target_reaches_final_references(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path, output_name="leaky", limit=1)

    with pytest.raises(RuntimeError, match="Target leakage"):
        precompute_mmsvg_sft_rag(
            config,
            retriever=_FakeHybridRetriever(leak_target=True),
        )

    assert not (tmp_path / "leaky").exists()


def test_r1_accepts_boolean_and_legacy_string_policy(tmp_path: Path) -> None:
    row = {"description": "short description", "detail": "long detail"}
    source, instruction = _resolve_instruction(
        row,
        {"instruction_policy": {"r1_detail_60_description_40": True}},
        "r1",
        path=tmp_path / "train.jsonl",
        line_number=1,
    )
    assert (source, instruction) == ("detail", "long detail")

    source, instruction = _resolve_instruction(
        row,
        {"instruction_policy": {"r1_detail_60_description_40": "description"}},
        "r1",
        path=tmp_path / "train.jsonl",
        line_number=2,
    )
    assert (source, instruction) == ("description", "short description")


def test_precompute_rejects_prepared_ids_outside_fixed_selection(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path, output_name="wrong-selection")
    icon_path = tmp_path / "selection" / "icon.txt"
    icon_path.write_text("some-other-icon\n", encoding="utf-8")
    manifest_path = tmp_path / "selection" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["datasets"]["icon"]["output_sha256"] = _file_hash(icon_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not match the fixed selection"):
        precompute_mmsvg_sft_rag(config, retriever=_FakeHybridRetriever())


def test_fixed_20k_selection_validates_id_file_split_boundaries() -> None:
    selection_ids = {
        domain: tuple(f"{domain}-{index:05d}" for index in range(10_000))
        for domain in ("icon", "illustration")
    }
    samples = []
    for domain, record_ids in selection_ids.items():
        for index, record_id in enumerate(record_ids):
            split = (
                "train"
                if index < 9_000
                else "validation"
                if index < 9_500
                else "test"
            )
            samples.append(
                SimpleNamespace(
                    dataset_type=domain,
                    record_id=record_id,
                    split=split,
                )
            )

    samples[0].split = "validation"
    samples[9_000].split = "train"
    with pytest.raises(RuntimeError, match="ID-file boundary"):
        _validate_fixed_selection(samples, selection_ids)


def _fixture_config(
    tmp_path: Path,
    *,
    output_name: str,
    limit: int | None = None,
    retrieval_batch_size: int = 128,
) -> MMSVGSFTRAGPrecomputeConfig:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    icon = _prepared_row(
        domain="icon",
        record_id="icon-target",
        color="black",
        description="icon description",
        detail="icon detail",
        r1_uses_detail=False,
    )
    illustration = _prepared_row(
        domain="illustration",
        record_id="illustration-target",
        color="white",
        description="illustration description",
        detail="illustration detail",
        r1_uses_detail=True,
    )
    (input_dir / "train.jsonl").write_text(
        json.dumps(icon) + "\n" + json.dumps(illustration) + "\n",
        encoding="utf-8",
    )
    (input_dir / "validation.jsonl").write_text("", encoding="utf-8")
    (input_dir / "test.jsonl").write_text("", encoding="utf-8")

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    icon_ids = selection_dir / "icon.txt"
    illustration_ids = selection_dir / "illustration.txt"
    icon_ids.write_text("icon-target\n", encoding="utf-8")
    illustration_ids.write_text("illustration-target\n", encoding="utf-8")
    selection_manifest = selection_dir / "manifest.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "icon": {
                        "output_file": icon_ids.name,
                        "output_sha256": _file_hash(icon_ids),
                        "sample_unique_ids": 1,
                    },
                    "illustration": {
                        "output_file": illustration_ids.name,
                        "output_sha256": _file_hash(illustration_ids),
                        "sample_unique_ids": 1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rag_config = tmp_path / "rag.yaml"
    rag_config.write_text("rag:\n  backend: chromadb\n", encoding="utf-8")
    return MMSVGSFTRAGPrecomputeConfig(
        input_dir=input_dir,
        output_dir=tmp_path / output_name,
        rag_config_path=rag_config,
        selection_manifest_path=selection_manifest,
        instruction_policy="r1",
        limit=limit,
        expected_sample_count=2,
        retrieval_batch_size=retrieval_batch_size,
    )


def _prepared_row(
    *,
    domain: str,
    record_id: str,
    color: str,
    description: str,
    detail: str,
    r1_uses_detail: bool,
) -> dict[str, Any]:
    svg = _svg(color)
    return {
        "task": "text_to_svg",
        "instruction": description,
        "instruction_source": "description",
        "description": description,
        "detail": detail,
        "output_svg": svg,
        "metadata": {
            "dataset_type": domain,
            "record_id": record_id,
            "split": "train",
            "canonical_svg_sha256": canonical_svg_sha256(svg),
            "stable_record_key_sha256": hashlib.sha256(
                f"{domain}:{record_id}".encode()
            ).hexdigest(),
            "instruction_policy": {
                "r1_detail_60_description_40": r1_uses_detail,
            },
        },
    }


def _svg(color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="10" height="10" fill="{color}"></rect>'
        "</svg>"
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
