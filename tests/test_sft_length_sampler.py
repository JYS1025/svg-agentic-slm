from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from svg_agentic_slm.train.sft_trainer import (
    _ResponseOnlyDataset,
    _VerifiedLengthSamplerMixin,
    SFTConfig,
)


def _records(lengths: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "metadata": {
                "record_id": f"record-{index}",
                "full_chat_token_length": length,
            }
        }
        for index, length in enumerate(lengths)
    ]


class _NoTokenizationDataset(_ResponseOnlyDataset):
    def __getitem__(self, index: int) -> dict[str, list[int]]:
        raise AssertionError("length grouping must not tokenize dataset rows")


def _dataset(lengths: list[int]) -> _NoTokenizationDataset:
    return _NoTokenizationDataset(
        _records(lengths),
        tokenizer=object(),
        instruction_mode="description_only",
        target_representation="raw_xml",
        max_seq_length=65_536,
        seed=42,
        codec=None,
    )


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 307_206

    @staticmethod
    def _render(messages: list[dict[str, str]]) -> str:
        return "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert not add_generation_prompt
        rendered = self._render(messages)
        return list(range(len(rendered))) if tokenize else rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[Any]]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class _CachedTargetCodec:
    @staticmethod
    def target_for_record(row: dict[str, Any]) -> str:
        return str(row["cached_target"])


def _discrete_dataset(
    metadata_lengths: list[Any],
    *,
    max_seq_length: int = 4096,
) -> _ResponseOnlyDataset:
    records = [
        {
            "description": f"description-{index}",
            "cached_target": "<codec-token>" * (index + 1),
            "metadata": {
                "record_id": f"discrete-{index}",
                "full_chat_token_length": metadata_length,
            },
        }
        for index, metadata_length in enumerate(metadata_lengths)
    ]
    return _ResponseOnlyDataset(
        records,
        tokenizer=_CharacterTokenizer(),
        instruction_mode="description_only",
        target_representation="omnisvg_discrete",
        max_seq_length=max_seq_length,
        seed=42,
        codec=_CachedTargetCodec(),  # type: ignore[arg-type]
    )


class _FallbackTrainer:
    def _get_train_sampler(self, train_dataset: Any = None) -> Any:
        self.fallback_dataset = train_dataset
        return "base-sampler"


class _SamplerSubject(_VerifiedLengthSamplerMixin, _FallbackTrainer):
    def __init__(
        self,
        dataset: _ResponseOnlyDataset,
        strategy: str,
        *,
        gradient_accumulation_steps: int = 2,
        length_grouping_batch_size: int | None = None,
    ) -> None:
        self.train_dataset = dataset
        self.args = SimpleNamespace(
            train_sampling_strategy=strategy,
            train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            length_grouping_batch_size=length_grouping_batch_size,
        )


@pytest.mark.parametrize("invalid", [None, True, False, 0, -1, 1.5, "10"])
def test_response_only_dataset_rejects_invalid_verified_length(invalid: Any) -> None:
    with pytest.raises(ValueError, match="full_chat_token_length"):
        _dataset([invalid])  # type: ignore[list-item]


def test_verified_length_manifest_is_stable_and_auditable() -> None:
    first = _dataset([9, 3, 12]).verified_length_manifest()
    second = _dataset([9, 3, 12]).verified_length_manifest()

    assert first == second
    assert first == {
        "source": "record.metadata.full_chat_token_length",
        "validation": "positive_integer",
        "count": 3,
        "minimum": 3,
        "maximum": 12,
        "sum": 24,
        "ordered_values_sha256": first["ordered_values_sha256"],
    }
    assert len(first["ordered_values_sha256"]) == 64


def test_discrete_sampler_and_manifest_use_exact_runtime_lengths_not_metadata() -> None:
    dataset = _discrete_dataset([1, 999_999, None])
    actual_lengths = tuple(len(dataset[index]["input_ids"]) for index in range(len(dataset)))

    assert dataset.verified_lengths == actual_lengths
    assert dataset.verified_lengths != (1, 999_999, None)
    sampler = _SamplerSubject(dataset, "group_by_length")._get_train_sampler()
    assert tuple(sampler.lengths) == actual_lengths

    manifest = dataset.verified_length_manifest()
    assert manifest["source"] == "runtime_serialization.input_ids_length"
    assert manifest["validation"] == "exact_runtime_serialization_without_truncation"
    assert manifest["maximum"] == manifest["actual_maximum"] == max(actual_lengths)
    assert manifest["contract"] == {
        "target_representation": "omnisvg_discrete",
        "instruction_mode": "description_only",
        "max_seq_length": 4096,
        "tokenizer_class": "_CharacterTokenizer",
        "tokenizer_vocabulary_size": 307_206,
        "serialization": "apply_chat_template(tokenize=True, add_generation_prompt=False)",
        "truncation": False,
    }
    assert manifest["metadata_diagnostic"] == {
        "source": "record.metadata.full_chat_token_length",
        "used_for_sampling": False,
        "present_positive_integer_count": 2,
        "missing_or_invalid_count": 1,
        "match_count": 0,
        "mismatch_count": 2,
    }


def test_discrete_length_precomputation_fails_fast_above_context() -> None:
    with pytest.raises(ValueError, match="exceeding max_seq_length=1"):
        _discrete_dataset([1], max_seq_length=1)


def test_length_grouped_sampler_uses_supplied_lengths_without_getitem() -> None:
    dataset = _dataset([40, 3, 17, 8, 25, 12])
    subject = _SamplerSubject(dataset, "group_by_length")

    torch.manual_seed(42)
    first = list(subject._get_train_sampler())
    torch.manual_seed(42)
    second = list(subject._get_train_sampler())

    assert first == second
    assert sorted(first) == list(range(len(dataset)))
    assert len(set(first)) == len(dataset)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "16"])
def test_length_grouping_batch_size_rejects_non_positive_integers(invalid: Any) -> None:
    with pytest.raises(ValueError, match="length_grouping_batch_size"):
        SFTConfig(
            train_sampling_strategy="group_by_length",
            length_grouping_batch_size=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid", [True, False, 0, -2, 1.5, "1"])
def test_max_steps_rejects_values_other_than_minus_one_or_positive_int(invalid: Any) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        SFTConfig(max_steps=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("valid", [-1, 1, 25])
def test_max_steps_accepts_disabled_or_positive_int(valid: int) -> None:
    assert SFTConfig(max_steps=valid).max_steps == valid


def test_length_grouping_batch_size_requires_grouped_sampling() -> None:
    with pytest.raises(ValueError, match="requires"):
        SFTConfig(
            train_sampling_strategy="random",
            length_grouping_batch_size=16,
        )


def _optimizer_window_unions(
    order: list[int],
    *,
    world_size: int,
    gradient_accumulation_steps: int,
) -> list[frozenset[int]]:
    rank_streams = [order[rank::world_size] for rank in range(world_size)]
    steps = len(order) // (world_size * gradient_accumulation_steps)
    return [
        frozenset(
            sample_index
            for rank_stream in rank_streams
            for sample_index in rank_stream[
                step * gradient_accumulation_steps : (step + 1)
                * gradient_accumulation_steps
            ]
        )
        for step in range(steps)
    ]


def test_fixed_grouping_batch_preserves_global_order_and_effective_windows() -> None:
    dataset = _dataset([((index * 37) % 251) + 1 for index in range(96)])
    three_gpu = _SamplerSubject(
        dataset,
        "group_by_length",
        gradient_accumulation_steps=16,
        length_grouping_batch_size=16,
    )
    two_gpu = _SamplerSubject(
        dataset,
        "group_by_length",
        gradient_accumulation_steps=24,
        length_grouping_batch_size=16,
    )

    torch.manual_seed(42)
    three_gpu_order = list(three_gpu._get_train_sampler())
    torch.manual_seed(42)
    two_gpu_order = list(two_gpu._get_train_sampler())

    assert three_gpu_order == two_gpu_order
    assert three_gpu._get_train_sampler().batch_size == 16
    assert two_gpu._get_train_sampler().batch_size == 16

    three_gpu_windows = _optimizer_window_unions(
        three_gpu_order,
        world_size=3,
        gradient_accumulation_steps=16,
    )
    two_gpu_windows = _optimizer_window_unions(
        two_gpu_order,
        world_size=2,
        gradient_accumulation_steps=24,
    )
    expected_windows = [
        frozenset(three_gpu_order[start : start + 48])
        for start in range(0, len(three_gpu_order), 48)
    ]

    assert three_gpu_windows == two_gpu_windows == expected_windows
    assert all(len(window) == 48 for window in expected_windows)


@pytest.mark.parametrize("strategy", ["random", "sequential"])
def test_non_grouped_sampling_delegates_to_transformers_default(strategy: str) -> None:
    dataset = _dataset([4, 7])
    subject = _SamplerSubject(dataset, strategy)

    assert subject._get_train_sampler(dataset) == "base-sampler"
    assert subject.fallback_dataset is dataset
