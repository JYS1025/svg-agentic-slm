from __future__ import annotations

import copy
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from svg_agentic_slm.train.response_collator import ResponseOnlyCollator
from svg_agentic_slm.train.tokenized_cache import load_or_build_tokenized_cache


def _features(*, weighted: bool) -> list[dict[str, Any]]:
    rows = [
        {"input_ids": [2, 300001, 9], "attention_mask": [1, 1, 1], "labels": [-100, 300001, 9]},
        {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [-100, 4]},
    ]
    if weighted:
        rows[0].update(
            loss_weights=[1.0, 1.25, 8.0],
            response_token_roles=[0, 2, 1],
            response_token_role_counts={"eos_count": 1},
        )
        rows[1].update(loss_weights=[1, 2.5], response_token_roles=[0, 1])
    return rows


def _converted(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    result = copy.deepcopy(rows)
    for row in result:
        for key, value in row.items():
            if not isinstance(value, list):
                continue
            if kind == "array":
                row[key] = array("d" if key == "loss_weights" else "q", value)
            elif kind != "list":
                dtype = np.float64 if key == "loss_weights" else np.int32
                row[key] = np.asarray(value, dtype=dtype)
                if kind == "readonly":
                    row[key].flags.writeable = False
                elif kind == "big_endian":
                    row[key] = row[key].astype(row[key].dtype.newbyteorder(">"))
    return result


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", ["list", "numpy", "array", "readonly", "big_endian"])
def test_array_and_list_collators_have_identical_padding_dtype_and_values(
    weighted: bool, kind: str
) -> None:
    tokenizer = SimpleNamespace(pad_token_id=42, eos_token_id=9)
    features = _converted(_features(weighted=weighted), kind)
    reference = ResponseOnlyCollator(tokenizer, array_fast_path=False)(features)
    actual = ResponseOnlyCollator(tokenizer)(features)

    assert actual.keys() == reference.keys()
    assert torch.equal(actual["input_ids"], torch.tensor([[2, 300001, 9], [3, 4, 42]]))
    assert torch.equal(actual["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(actual["labels"], torch.tensor([[-100, 300001, 9], [-100, 4, -100]]))
    for key, tensor in actual.items():
        assert tensor.dtype == reference[key].dtype
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, reference[key])
        assert tensor.is_contiguous()
    if weighted:
        assert actual["loss_weights"].dtype == torch.float32
        assert actual["loss_weights"][1, 2].item() == 1.0
        assert actual["response_token_roles"][1, 2].item() == 0


def test_array_collator_avoids_per_row_torch_tensor_and_does_not_alias_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _converted(_features(weighted=True), "numpy")

    def unexpected_tensor(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Array collator must not construct per-row torch.tensor copies")

    monkeypatch.setattr(torch, "tensor", unexpected_tensor)
    actual = ResponseOnlyCollator(SimpleNamespace(pad_token_id=0))(features)
    features[0]["input_ids"][0] = 999
    features[0]["loss_weights"][0] = 7.0
    assert actual["input_ids"][0, 0].item() == 2
    assert actual["loss_weights"][0, 0].item() == 1.0
    actual["input_ids"][1, 0] = 777
    assert features[1]["input_ids"][0] == 3


@pytest.mark.parametrize("array_fast_path", [False, True])
@pytest.mark.parametrize("bad_weight", [True, False, 0, -1.0, float("inf"), float("nan"), "1"])
def test_invalid_weights_remain_rejected(array_fast_path: bool, bad_weight: Any) -> None:
    features = _features(weighted=True)
    features[0]["loss_weights"][1] = bad_weight
    with pytest.raises(ValueError, match="weights must be finite and positive"):
        ResponseOnlyCollator(SimpleNamespace(pad_token_id=0), array_fast_path=array_fast_path)(
            features
        )


@pytest.mark.parametrize("array_fast_path", [False, True])
def test_mixed_weighted_rows_and_bad_shapes_are_rejected(array_fast_path: bool) -> None:
    collator = ResponseOnlyCollator(
        SimpleNamespace(pad_token_id=0), array_fast_path=array_fast_path
    )
    features = _features(weighted=True)
    del features[1]["loss_weights"]
    with pytest.raises(ValueError, match="cannot mix"):
        collator(features)
    features = _converted(_features(weighted=True), "numpy")
    features[1]["response_token_roles"] = np.asarray([1])
    with pytest.raises(ValueError, match="shape differs"):
        collator(features)


def test_numpy_boolean_weights_are_rejected() -> None:
    features = _converted(_features(weighted=True), "numpy")
    features[0]["loss_weights"] = np.asarray([True, True, True])
    with pytest.raises(ValueError, match="weights must be finite and positive"):
        ResponseOnlyCollator(SimpleNamespace(pad_token_id=0))(features)


def test_padding_fallback_and_flag_validation() -> None:
    collator = ResponseOnlyCollator(SimpleNamespace(pad_token_id=None, eos_token_id=19))
    assert collator(_features(weighted=False))["input_ids"][1, -1].item() == 19
    with pytest.raises(ValueError, match="must define"):
        ResponseOnlyCollator(SimpleNamespace(pad_token_id=None, eos_token_id=None))
    with pytest.raises(TypeError, match="must be boolean"):
        ResponseOnlyCollator(SimpleNamespace(pad_token_id=0), array_fast_path=1)


class _Source:
    verified_lengths = (30, 20)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _features(weighted=True)[index]

    def verified_length_manifest(self) -> dict[str, Any]:
        return {"source": "fixture"}

    def instruction_selection_manifest(self) -> dict[str, Any]:
        return {"source": "fixture"}

    def structural_response_role_manifest(self) -> dict[str, Any]:
        return {"sample_count": 2}


@pytest.mark.parametrize("workers", [0, 1])
def test_dataloader_uses_batched_fetch_with_portable_spawn_and_safe_close(
    tmp_path: Path, workers: int
) -> None:
    cache = load_or_build_tokenized_cache(_Source(), cache_dir=tmp_path, fingerprint={"v": 1})
    tokenizer = SimpleNamespace(pad_token_id=0)
    expected = ResponseOnlyCollator(tokenizer, array_fast_path=False)(_features(weighted=True))
    options = {"multiprocessing_context": "spawn"} if workers else {}
    loader = torch.utils.data.DataLoader(
        cache,
        batch_size=2,
        num_workers=workers,
        collate_fn=ResponseOnlyCollator(tokenizer),
        **options,
    )
    try:
        batches = list(loader)
    finally:
        cache.close()
    assert len(batches) == 1
    for key, tensor in expected.items():
        assert torch.equal(batches[0][key], tensor)
