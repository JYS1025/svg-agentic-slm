from __future__ import annotations

import copy
import hashlib
import json

import pytest

import svg_agentic_slm.train.sft_trainer as sft_trainer
from svg_agentic_slm.train.sft_trainer import (
    SFTConfig,
    _ResponseOnlyDataset,
    _select_instruction,
)


def _row(
    *,
    detail: str,
    selected_field: str = "detail",
    record_id: str = "record-1",
) -> dict[str, object]:
    return {
        "description": "A concise description.",
        "detail": detail,
        "output_svg": "<svg></svg>",
        "metadata": {
            "record_id": record_id,
            "full_chat_token_length": 100,
            "instruction_policy": {
                "r1_detail_60_description_40": selected_field,
            },
        },
    }


def test_detail_text_normalization_config_is_strict_and_opt_in() -> None:
    assert SFTConfig().detail_text_normalization == "none"
    assert (
        SFTConfig(detail_text_normalization="mmsvg_list_repr_v1").detail_text_normalization
        == "mmsvg_list_repr_v1"
    )

    for invalid in ("list_repr", "", None, True):
        with pytest.raises(ValueError, match="detail_text_normalization"):
            SFTConfig(detail_text_normalization=invalid)  # type: ignore[arg-type]


def test_mmsvg_list_repr_normalizes_selected_detail_to_plain_paragraph() -> None:
    detail = (
        "['First sentence.', \"The object's second sentence.\", "
        "'  Third sentence.  ']"
    )

    assert _select_instruction(
        _row(detail=detail),
        "detail_only",
        detail_text_normalization="mmsvg_list_repr_v1",
    ) == "First sentence. The object's second sentence. Third sentence."


def test_scalar_detail_is_unchanged_and_none_preserves_legacy_repr() -> None:
    scalar = "A normal detailed description with [internal] brackets."
    legacy = "['First sentence.', 'Second sentence.']"

    assert _select_instruction(
        _row(detail=scalar),
        "detail_only",
        detail_text_normalization="mmsvg_list_repr_v1",
    ) == scalar
    assert _select_instruction(
        _row(detail=legacy),
        "detail_only",
        detail_text_normalization="none",
    ) == legacy


@pytest.mark.parametrize(
    "detail",
    [
        "['unterminated'",
        "not a list]",
        "[]",
        "['']",
        "['valid', 3]",
        "[['nested']]",
    ],
)
def test_mmsvg_list_repr_candidate_fails_closed(detail: str) -> None:
    with pytest.raises(ValueError, match="MMSVG detail list"):
        _select_instruction(
            _row(detail=detail, record_id="bad-detail"),
            "detail_only",
            detail_text_normalization="mmsvg_list_repr_v1",
        )


def test_mixed_description_branch_does_not_invoke_literal_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_parser(_: object) -> object:
        raise AssertionError("literal parser must not run for a selected description")

    monkeypatch.setattr(sft_trainer.ast, "literal_eval", unexpected_parser)
    row = _row(detail="['unterminated'", selected_field="description")

    assert _select_instruction(
        row,
        "mixed_60_detail_40_description",
        detail_text_normalization="mmsvg_list_repr_v1",
    ) == "A concise description."


def test_mixed_detail_branch_applies_normalization() -> None:
    row = _row(
        detail="['First sentence.', 'Second sentence.']",
        selected_field="detail",
    )

    assert _select_instruction(
        row,
        "mixed_60_detail_40_description",
        detail_text_normalization="mmsvg_list_repr_v1",
    ) == "First sentence. Second sentence."


def test_dataset_manifest_hashes_selected_instructions_without_mutating_records() -> None:
    records = [
        _row(
            detail="['First sentence.', 'Second sentence.']",
            record_id="list-detail",
        ),
        _row(detail="Scalar detail.", record_id="scalar-detail"),
    ]
    before = copy.deepcopy(records)

    dataset = _ResponseOnlyDataset(
        records,  # type: ignore[arg-type]
        tokenizer=object(),
        instruction_mode="detail_only",
        target_representation="raw_xml",
        max_seq_length=256,
        seed=42,
        codec=None,
        detail_text_normalization="mmsvg_list_repr_v1",
    )

    expected_instructions = (
        "First sentence. Second sentence.",
        "Scalar detail.",
    )
    expected_sha256 = hashlib.sha256(
        json.dumps(
            expected_instructions,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = dataset.instruction_selection_manifest()

    assert records == before
    assert manifest == {
        "instruction_mode": "detail_only",
        "detail_text_normalization": "mmsvg_list_repr_v1",
        "selected_instruction_count": 2,
        "selected_instruction_ordered_sha256": expected_sha256,
        "selected_instruction_hash_serialization": (
            "utf8_json_array_ensure_ascii_false_compact"
        ),
        "selected_detail_count": 2,
        "legacy_detail_candidate_count": 1,
        "legacy_detail_normalized_count": 1,
        "normalization_scope": "selected_detail_only",
        "raw_records_mutated": False,
    }


def test_manifest_counts_only_selected_detail_candidates() -> None:
    records = [
        _row(
            detail="['Malformed but unselected'",
            selected_field="description",
            record_id="description-branch",
        ),
        _row(
            detail="['Selected detail.']",
            selected_field="detail",
            record_id="detail-branch",
        ),
    ]

    dataset = _ResponseOnlyDataset(
        records,  # type: ignore[arg-type]
        tokenizer=object(),
        instruction_mode="mixed_60_detail_40_description",
        target_representation="raw_xml",
        max_seq_length=256,
        seed=42,
        codec=None,
        detail_text_normalization="mmsvg_list_repr_v1",
    )
    manifest = dataset.instruction_selection_manifest()

    assert manifest["selected_detail_count"] == 1
    assert manifest["legacy_detail_candidate_count"] == 1
    assert manifest["legacy_detail_normalized_count"] == 1
