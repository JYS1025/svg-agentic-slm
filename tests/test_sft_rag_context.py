"""Focused tests for leakage-clean precomputed RAG conditioning in SFT."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import svg_agentic_slm.train.sft_trainer as sft_module
from svg_agentic_slm.agents.generator import GeneratorAgent
from svg_agentic_slm.agents.schemas import GenerationRequest
from svg_agentic_slm.data.mmsvg_sft_rag_precompute import (
    _corpus_items_schema,
    _file_sha256,
    _final_references_schema,
    _json_sha256,
)
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.prompts.text_to_svg import build_retrieval_context, build_text_to_svg_prompt
from svg_agentic_slm.rag.context import _svg_element_variants
from svg_agentic_slm.rag.mmsvg_hybrid import canonical_svg_sha256
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.train.rag_context import PrecomputedRAGContextStore
from svg_agentic_slm.train.sft_trainer import (
    ModelTrainingConfig,
    SFTConfig,
    TextToSVGSFTTrainer,
    _ResponseOnlyDataset,
    _select_instruction,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        text = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return [ord(character) for character in text] if tokenize else text

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[Any]]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }


class _RecordingBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text)

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(text="<svg/>", model_id="fake")


def _svg(color: str, *, second_color: str | None = None) -> str:
    second = (
        f'<circle cx="12" cy="12" r="3" fill="{second_color}"/>'
        if second_color is not None
        else ""
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        f'<rect width="8" height="8" fill="{color}"/>{second}</svg>'
    )


def _sample_key() -> str:
    return hashlib.sha256(b"fixed-sft-sample").hexdigest()


def _sft_record(*, policy: bool | str = True) -> dict[str, Any]:
    target_svg = _svg("black")
    return {
        "task": "text_to_svg",
        "description": "a black square",
        "detail": "a centered black square on a transparent canvas",
        "output_svg": target_svg,
        "metadata": {
            "stable_record_key_sha256": _sample_key(),
            "dataset_type": "icon",
            "record_id": "target",
            "canonical_svg_sha256": canonical_svg_sha256(target_svg),
            "full_chat_token_length": 1,
            "instruction_policy": {
                "r1_detail_60_description_40": policy,
            },
        },
    }


def _write_artifacts(
    tmp_path: Path,
    *,
    candidates: list[tuple[str, str]],
    links: list[tuple[int, str, float, str, str]],
) -> tuple[Path, Path]:
    corpus_rows: list[dict[str, Any]] = []
    corpus_by_id: dict[str, dict[str, Any]] = {}
    for row_index, (item_id, svg) in enumerate(candidates):
        dataset_type, record_id = item_id.split(":", maxsplit=1)
        row = {
            "item_id": item_id,
            "dataset_type": dataset_type,
            "record_id": record_id,
            "shard": "train-00000.parquet",
            "row_index_in_shard": row_index,
            "source": f"hf://mmsvg/train/{item_id}",
            "corpus_version": "revision-1",
            "description": f"description for {item_id}",
            "detail": f"detail for {item_id}",
            "svg": svg,
            "canonical_svg_sha256": canonical_svg_sha256(svg),
            "image_sha256": f"{row_index + 1:064x}",
        }
        corpus_rows.append(row)
        corpus_by_id[item_id] = row

    final_rows = [
        {
            "sample_key": _sample_key(),
            "final_rank": rank,
            "item_id": item_id,
            "canonical_svg_sha256": corpus_by_id[item_id][
                "canonical_svg_sha256"
            ],
            "retrieval_score": score,
            "score_kind": score_kind,
            "selection_reason": selection_reason,
        }
        for rank, item_id, score, score_kind, selection_reason in links
    ]
    final_path = tmp_path / "final_references.parquet"
    corpus_path = tmp_path / "corpus_items.parquet"
    pq.write_table(
        pa.Table.from_pylist(final_rows, schema=_final_references_schema()),
        final_path,
    )
    pq.write_table(
        pa.Table.from_pylist(corpus_rows, schema=_corpus_items_schema()),
        corpus_path,
    )
    input_files: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        input_path = tmp_path / f"{split}.jsonl"
        input_path.write_text(f'{{"split":"{split}"}}\n', encoding="utf-8")
        input_files[split] = {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "rows": 1,
        }
    run_contract = {
        "instruction_policy": "r1",
        "input": {"files": input_files},
    }
    manifest = {
        "schema_version": 1,
        "dataset": "mmsvg-generator-sft-hybrid-rag-top3",
        "run_contract": run_contract,
        "run_contract_sha256": _json_sha256(run_contract),
        "sample_rows": 1,
        "final_reference_rows": len(final_rows),
        "artifacts": {
            "final_references": {
                "file": final_path.name,
                "rows": len(final_rows),
                "sha256": _file_sha256(final_path),
            },
            "corpus_items": {
                "file": corpus_path.name,
                "rows": len(corpus_rows),
                "sha256": _file_sha256(corpus_path),
            },
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return final_path, corpus_path


def _valid_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return _write_artifacts(
        tmp_path,
        candidates=[
            ("icon:overlap", _svg("blue", second_color="navy")),
            ("icon:siglip-a", _svg("green", second_color="lime")),
            ("illustration:siglip-b", _svg("orange", second_color="red")),
        ],
        links=[
            (1, "icon:overlap", 0.91, "hybrid_overlap_priority", "overlap"),
            (2, "icon:siglip-a", 0.82, "siglip2_pair_probability", "siglip2"),
            (
                3,
                "illustration:siglip-b",
                0.73,
                "siglip2_pair_probability",
                "siglip2",
            ),
        ],
    )


def test_authoritative_join_preserves_rank_score_and_payload(tmp_path: Path) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)

    examples = store.references_for_record(_sft_record())

    assert [example.item_id for example in examples] == [
        "icon:overlap",
        "icon:siglip-a",
        "illustration:siglip-b",
    ]
    assert [example.rank for example in examples] == [1, 2, 3]
    assert [example.score for example in examples] == [0.91, 0.82, 0.73]
    assert [example.score_kind for example in examples] == [
        "hybrid_overlap_priority",
        "siglip2_pair_probability",
        "siglip2_pair_probability",
    ]
    assert examples[0].detail == "detail for icon:overlap"
    assert examples[0].metadata["is_overlap"] is True
    assert examples[1].metadata["is_overlap"] is False


@pytest.mark.parametrize(
    ("bad_item_id", "bad_svg", "error"),
    [
        ("icon:target", _svg("purple"), "target item ID"),
        ("icon:target-copy", _svg("black"), "identical canonical SVG"),
    ],
)
def test_target_item_id_and_canonical_svg_leakage_fail_closed(
    tmp_path: Path,
    bad_item_id: str,
    bad_svg: str,
    error: str,
) -> None:
    final_path, corpus_path = _write_artifacts(
        tmp_path,
        candidates=[
            (bad_item_id, bad_svg),
            ("icon:other-a", _svg("blue")),
            ("icon:other-b", _svg("green")),
        ],
        links=[
            (1, bad_item_id, 0.9, "hybrid_overlap_priority", "overlap"),
            (2, "icon:other-a", 0.8, "siglip2_pair_probability", "siglip2"),
            (3, "icon:other-b", 0.7, "siglip2_pair_probability", "siglip2"),
        ],
    )
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)

    with pytest.raises(ValueError, match=error):
        store.references_for_record(_sft_record())


def test_requires_exact_unique_ranks_one_two_three(tmp_path: Path) -> None:
    final_path, corpus_path = _write_artifacts(
        tmp_path,
        candidates=[
            ("icon:a", _svg("red")),
            ("icon:b", _svg("blue")),
            ("icon:c", _svg("green")),
        ],
        links=[
            (1, "icon:a", 0.9, "hybrid_overlap_priority", "overlap"),
            (2, "icon:b", 0.8, "siglip2_pair_probability", "siglip2"),
            (2, "icon:c", 0.7, "siglip2_pair_probability", "siglip2"),
        ],
    )

    with pytest.raises(ValueError, match="exactly ranks 1, 2, 3"):
        PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)


def test_rejects_unsafe_svg_even_when_manifest_checksums_match(tmp_path: Path) -> None:
    unsafe_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<script>alert("unsafe")</script></svg>'
    )
    final_path, corpus_path = _write_artifacts(
        tmp_path,
        candidates=[
            ("icon:unsafe", unsafe_svg),
            ("icon:a", _svg("blue")),
            ("icon:b", _svg("green")),
        ],
        links=[
            (1, "icon:unsafe", 0.9, "hybrid_overlap_priority", "overlap"),
            (2, "icon:a", 0.8, "siglip2_pair_probability", "siglip2"),
            (3, "icon:b", 0.7, "siglip2_pair_probability", "siglip2"),
        ],
    )

    with pytest.raises(ValueError, match="unsafe or malformed SVG"):
        PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)


def test_input_jsonl_checksum_must_match_precompute_contract(tmp_path: Path) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    store.validate_input_files(train_path, validation_path)

    train_path.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum differs"):
        store.validate_input_files(train_path, validation_path)


def test_duplicate_sample_key_across_splits_fails_closed(tmp_path: Path) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)

    with pytest.raises(ValueError, match="Duplicate SFT sample_key"):
        store.validate_records([_sft_record(), _sft_record()])


@pytest.mark.parametrize("invalid_budget", [True, 1.5, -1])
def test_rag_context_budget_must_be_an_explicit_nonnegative_integer(
    tmp_path: Path,
    invalid_budget: Any,
) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)

    with pytest.raises(ValueError, match="non-negative integer"):
        TextToSVGSFTTrainer(
            model_config=ModelTrainingConfig(model_id="model", revision="revision"),
            lora_config=LoRAConfig(),
            sft_config=SFTConfig(rag_max_context_tokens=invalid_budget),
            train_data_path=tmp_path / "train.jsonl",
            eval_data_path=tmp_path / "validation.jsonl",
            instruction_mode="mixed_60_detail_40_description",
            target_representation="raw_xml",
            rag_context_store=store,
        )


def test_precompute_instruction_policy_must_match_training_mode(tmp_path: Path) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)

    with pytest.raises(ValueError, match="instruction policy does not match"):
        TextToSVGSFTTrainer(
            model_config=ModelTrainingConfig(model_id="model", revision="revision"),
            lora_config=LoRAConfig(),
            sft_config=SFTConfig(rag_max_context_tokens=100),
            train_data_path=tmp_path / "train.jsonl",
            eval_data_path=tmp_path / "validation.jsonl",
            instruction_mode="description_only",
            target_representation="raw_xml",
            rag_context_store=store,
        )


def test_dataset_uses_inference_token_budget_selector_and_same_prompt_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)
    all_references = store.references_for_record(_sft_record())
    tokenizer = _FakeTokenizer()
    first_item_budget = len(tokenizer.encode(
        build_retrieval_context(all_references[:1]),
        add_special_tokens=False,
    ))
    calls: list[tuple[str, list[Any] | None]] = []

    def _record_prompt(
        instruction: str,
        retrieved_examples: list[Any] | None = None,
    ) -> str:
        calls.append((instruction, retrieved_examples))
        return "recorded prompt"

    monkeypatch.setattr(sft_module, "build_text_to_svg_prompt", _record_prompt)
    dataset = _ResponseOnlyDataset(
        [_sft_record()],
        tokenizer=tokenizer,
        instruction_mode="mixed_60_detail_40_description",
        target_representation="raw_xml",
        max_seq_length=100_000,
        seed=42,
        codec=None,
        rag_context_store=store,
        rag_max_context_tokens=first_item_budget,
    )

    encoded = dataset[0]

    assert encoded["input_ids"]
    assert any(label == -100 for label in encoded["labels"])
    assert any(label != -100 for label in encoded["labels"])
    instruction, selected = calls[0]
    assert instruction == _sft_record()["detail"]
    assert selected is not None
    assert [example.item_id for example in selected] == ["icon:overlap"]


def test_partial_svg_context_prompt_matches_inference_generator(tmp_path: Path) -> None:
    final_path, corpus_path = _valid_artifacts(tmp_path)
    store = PrecomputedRAGContextStore.from_parquet(final_path, corpus_path)
    record = _sft_record()
    references = store.references_for_record(record)
    variants = _svg_element_variants(references[0])
    assert len(variants) == 2
    partial_reference = replace(references[0], content=variants[0])
    budget = len(build_retrieval_context([partial_reference]))

    dataset = _ResponseOnlyDataset(
        [record],
        tokenizer=_FakeTokenizer(),
        instruction_mode="mixed_60_detail_40_description",
        target_representation="raw_xml",
        max_seq_length=100_000,
        seed=42,
        codec=None,
        rag_context_store=store,
        rag_max_context_tokens=budget,
    )
    selected = dataset._select_rag_context(record)
    assert selected is not None
    assert len(selected) == 1
    assert "<rect" in selected[0].content
    assert "<circle" not in selected[0].content
    assert SVGValidator().validate(selected[0].content).is_valid
    training_prompt = build_text_to_svg_prompt(
        record["detail"],
        retrieved_examples=selected,
    )

    backend = _RecordingBackend()
    generator = GeneratorAgent(backend, max_context_tokens=budget)
    generator.generate(
        GenerationRequest(instruction=record["detail"]),
        context=references,
    )

    assert backend.prompts == [training_prompt]


def test_baseline_prompt_receives_no_rag_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[Any] | None] = []

    def _record_prompt(
        instruction: str,
        retrieved_examples: list[Any] | None = None,
    ) -> str:
        calls.append(retrieved_examples)
        return instruction

    monkeypatch.setattr(sft_module, "build_text_to_svg_prompt", _record_prompt)
    dataset = _ResponseOnlyDataset(
        [_sft_record()],
        tokenizer=_FakeTokenizer(),
        instruction_mode="description_only",
        target_representation="raw_xml",
        max_seq_length=100_000,
        seed=42,
        codec=None,
    )

    dataset[0]

    assert calls == [None]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (True, "long detail"),
        (False, "short description"),
        ("detail", "long detail"),
        ("description", "short description"),
    ],
)
def test_r1_accepts_boolean_and_legacy_string_values(
    policy: bool | str,
    expected: str,
) -> None:
    row = {
        "description": "short description",
        "detail": "long detail",
        "metadata": {
            "instruction_policy": {
                "r1_detail_60_description_40": policy,
            }
        },
    }

    assert _select_instruction(
        row,
        "mixed_60_detail_40_description",
    ) == expected
