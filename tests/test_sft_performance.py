from __future__ import annotations

import copy
import struct
import zlib
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, TrainingArguments

from svg_agentic_slm.svg.official_discrete_cache import (
    CACHED_TARGET_FIELD,
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    OPENVGLAB_BOS_ID,
    OPENVGLAB_EOS_ID,
    CachedOpenVGLabGemmaDialect,
)
from svg_agentic_slm.train.sft_trainer import (
    SFTConfig,
    _cache_record_identity,
    _prepare_sft_dataset,
    _ResponseOnlyCollator,
    _ResponseOnlyDataset,
    _tokenization_identity,
    _VerifiedLengthSamplerMixin,
)


def _tokenizer():
    tokens = ["<unk>", "<pad>", "<eos>", "<svg>", "</svg>", "circle", "square"]
    backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<' + message['role'] + '> ' + message['content'] + ' </' + message['role'] + '> ' }}"
        "{% endfor %}"
    )
    return tokenizer


def _records():
    return [
        {
            "description": f"Draw a {shape}",
            "output_svg": f"<svg> {shape} </svg>",
            "metadata": {"record_id": str(index), "full_chat_token_length": 91 - index},
        }
        for index, shape in enumerate(("circle", "square"))
    ]


def _kwargs(tokenizer, codec=None):
    return {
        "tokenizer": tokenizer,
        "instruction_mode": "description_only",
        "target_representation": "omnisvg_discrete" if codec else "raw_xml",
        "max_seq_length": 4096,
        "seed": 42,
        "codec": codec,
    }


def _cache(tmp_path, records, kwargs):
    return _prepare_sft_dataset(
        records,
        split="train",
        config=SFTConfig(output_dir=str(tmp_path)),
        dataset_kwargs=kwargs,
        tokenization_identity=_tokenization_identity(kwargs["tokenizer"], kwargs["codec"]),
    )


class _Sampler(_VerifiedLengthSamplerMixin):
    def __init__(self, dataset):
        self.train_dataset = dataset
        self.args = SimpleNamespace(
            train_sampling_strategy="group_by_length",
            length_grouping_batch_size=2,
        )


@pytest.mark.parametrize("discrete", [False, True])
def test_cache_preserves_exact_training_rows_and_warm_start_skips_serialization(
    tmp_path,
    monkeypatch,
    discrete,
):
    records, tokenizer = _records(), _tokenizer()
    codec = CachedOpenVGLabGemmaDialect() if discrete else None
    if discrete:
        for index, row in enumerate(records):
            ids = [OPENVGLAB_BOS_ID, 151938, *([151943] * (index + 1)), OPENVGLAB_EOS_ID]
            row[CACHED_TARGET_FIELD] = {
                "backend_id": OFFICIAL_CACHED_GEMMA_BACKEND_ID,
                "cache_sha256": "fixture",
                "expected_cache_sha256": "fixture",
                "framed_uint32_le_zlib": zlib.compress(struct.pack(f"<{len(ids)}I", *ids)),
                "framed_count": len(ids),
            }
    kwargs = _kwargs(tokenizer, codec)
    reference = _ResponseOnlyDataset(records, **kwargs)
    rows = [reference[i] for i in range(len(reference))]
    original = _ResponseOnlyDataset._serialize_record
    calls = []

    def record_call(self, index):
        calls.append(index)
        return original(self, index)

    monkeypatch.setattr(_ResponseOnlyDataset, "_serialize_record", record_call)
    cached = _cache(tmp_path, records, kwargs)
    try:
        assert calls == [0, 1]  # Discrete length validation must not serialize twice.
        assert [cached[i] for i in range(len(cached))] == rows
        assert cached.verified_lengths == reference.verified_lengths
        assert cached.verified_length_manifest() == reference.verified_length_manifest()
        assert cached.instruction_selection_manifest() == reference.instruction_selection_manifest()
        collator = _ResponseOnlyCollator(tokenizer)
        for key, expected in collator(rows).items():
            assert torch.equal(collator([cached[0], cached[1]])[key], expected)
        torch.manual_seed(71)
        expected_order = list(_Sampler(reference)._get_train_sampler())
        torch.manual_seed(71)
        assert list(_Sampler(cached)._get_train_sampler()) == expected_order
    finally:
        cached.close()

    def forbidden_init(*args, **kwargs):
        raise AssertionError("Warm cache must not construct or tokenize the source dataset")

    monkeypatch.setattr(_ResponseOnlyDataset, "__init__", forbidden_init)
    warm = _cache(tmp_path, records, kwargs)
    try:
        assert warm.cache_manifest()["status"] == "reused"
        assert [warm[i] for i in range(len(warm))] == rows
        assert calls == [0, 1]
    finally:
        warm.close()


@pytest.mark.parametrize("change", ["content", "order", "template", "vocabulary", "length"])
def test_serialization_changes_cannot_reuse_stale_cache(tmp_path, change):
    records, tokenizer = _records(), _tokenizer()
    kwargs = _kwargs(tokenizer)
    first = _cache(tmp_path, records, kwargs)
    original_key = first.cache_manifest()["fingerprint_sha256"]
    first.close()
    if change == "content":
        records[0]["output_svg"] = "<svg> square square </svg>"
    elif change == "order":
        records.reverse()
    elif change == "template":
        tokenizer.chat_template += " <eos>"
    elif change == "vocabulary":
        tokenizer.add_tokens(["new-token"])
    else:
        kwargs["max_seq_length"] += 1
    changed = _cache(tmp_path, records, kwargs)
    try:
        assert changed.cache_manifest()["status"] == "built"
        assert changed.cache_manifest()["fingerprint_sha256"] != original_key
    finally:
        changed.close()


def test_binary_identity_is_stable_and_distinct_from_json_lookalikes():
    payload = {"compressed": b"\x00\xff\x01"}
    identity = _cache_record_identity(payload)
    assert identity == _cache_record_identity(copy.deepcopy(payload))
    assert identity != _cache_record_identity({"compressed": b"\x00\xff\x02"})
    assert identity != _cache_record_identity(identity)


def test_deferred_discrete_validation_preserves_structural_roles(tmp_path, monkeypatch):
    records, tokenizer = _records(), _tokenizer()
    kwargs = _kwargs(tokenizer, CachedOpenVGLabGemmaDialect())
    kwargs["response_eos_loss_weight"] = 2.0
    features = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 3],
            "loss_weights": [1.0, 1.0, 2.0],
            "response_token_roles": [0, 2, 1],
            "response_token_role_counts": {
                "labeled_token_count": 2,
                "eos_count": 1,
                "path_color_terminator_count": 1,
            },
        },
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, 2],
            "loss_weights": [1.0, 2.0],
            "response_token_roles": [0, 1],
            "response_token_role_counts": {
                "labeled_token_count": 1,
                "eos_count": 1,
                "path_color_terminator_count": 0,
            },
        },
    ]
    calls = []

    def serialized(self, index):
        calls.append(index)
        return copy.deepcopy(features[index])

    monkeypatch.setattr(_ResponseOnlyDataset, "_serialize_record", serialized)
    cache = _cache(tmp_path, records, kwargs)
    try:
        assert calls == [0, 1]
        assert cache.verified_lengths == (3, 2)
        counts = cache.structural_response_role_manifest()
        assert counts["labeled_token_count"] == 3
        assert counts["eos_count"] == 2
        assert counts["path_color_terminator_count"] == 1
        assert [cache[i] for i in range(2)] == features
        collator = _ResponseOnlyCollator(tokenizer)
        for key, value in collator(features).items():
            assert torch.equal(collator([cache[0], cache[1]])[key], value)
    finally:
        cache.close()


def test_cached_dataset_reopens_in_spawn_dataloader_worker(tmp_path):
    tokenizer = _tokenizer()
    cached = _cache(tmp_path, _records(), _kwargs(tokenizer))
    try:
        collator = _ResponseOnlyCollator(tokenizer)
        expected = collator([cached[0], cached[1]])
        loader = torch.utils.data.DataLoader(
            cached,
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
            collate_fn=collator,
        )
        batches = list(loader)
        assert len(batches) == 1
        for key, value in expected.items():
            assert torch.equal(batches[0][key], value)
    finally:
        cached.close()


@pytest.mark.parametrize(
    "options",
    [
        {"dataloader_num_workers": -1},
        {"dataloader_num_workers": True},
        {"dataloader_num_workers": 0, "dataloader_persistent_workers": True},
        {"dataloader_num_workers": 0, "dataloader_prefetch_factor": 2},
        {"dataloader_prefetch_factor": 0},
        {"dataloader_non_blocking": True, "dataloader_pin_memory": False},
        {"tokenized_cache": "true"},
        {"tokenized_cache_dir": ""},
        {"tokenized_cache_lock_timeout_seconds": 0},
        {"tokenized_cache_batched_fetch": 1},
        {"collator_array_fast_path": "true"},
        {"logging_nan_inf_filter": 1},
        {"lm_head_loss_chunk_size": True},
        {"lm_head_loss_chunk_size": 0},
        {"lm_head_loss_backend": "fused"},
    ],
)
def test_invalid_performance_configuration_is_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        SFTConfig(**options)


def test_cache_remains_optional_and_loader_arguments_are_supported(tmp_path):
    config = SFTConfig(tokenized_cache=False, dataloader_num_workers=0)
    dataset = _prepare_sft_dataset(
        _records(),
        split="train",
        config=config,
        dataset_kwargs=_kwargs(_tokenizer()),
        tokenization_identity=None,
    )
    assert isinstance(dataset, _ResponseOnlyDataset)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        accelerator_config={"non_blocking": True},
        logging_nan_inf_filter=False,
    )
    assert args.accelerator_config.non_blocking
    assert not args.logging_nan_inf_filter
