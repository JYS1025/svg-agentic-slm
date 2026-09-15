"""Response-only padding with a portable array path and a list baseline."""

from __future__ import annotations

import math
from array import array
from typing import Any

import numpy as np


class ResponseOnlyCollator:
    """Produce independent padded CPU tensors from lists or owned arrays.

    The array path writes directly into NumPy views of the batch tensors. It
    avoids allocating a temporary torch tensor for every field in every row.
    No mmap view escapes into the result, and no input array is ever written.
    ``array_fast_path=False`` retains the original list/torch.tensor algorithm.
    """

    def __init__(self, tokenizer: Any, *, array_fast_path: bool = True) -> None:
        if not isinstance(array_fast_path, bool):
            raise TypeError("Response collator array_fast_path must be boolean.")
        self._array_fast_path = array_fast_path
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = tokenizer.eos_token_id
        if self._pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Response collator requires a non-empty batch.")
        if not self._array_fast_path:
            return self._collate_lists(features)
        return self._collate_arrays(features)

    @staticmethod
    def _weighted(features: list[dict[str, Any]]) -> bool:
        weighted = ["loss_weights" in feature for feature in features]
        if any(weighted) and not all(weighted):
            raise ValueError("A batch cannot mix structural-weighted and unweighted records.")
        return all(weighted)

    def _allocate(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        shape = (len(features), max_length)
        batch = {
            "input_ids": torch.full(shape, self._pad_token_id, dtype=torch.long, device="cpu"),
            "attention_mask": torch.zeros(shape, dtype=torch.long, device="cpu"),
            "labels": torch.full(shape, -100, dtype=torch.long, device="cpu"),
        }
        if self._weighted(features):
            batch["loss_weights"] = torch.ones(shape, dtype=torch.float32, device="cpu")
            batch["response_token_roles"] = torch.zeros(shape, dtype=torch.long, device="cpu")
        return batch

    def _collate_arrays(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._allocate(features)
        views = {key: tensor.numpy() for key, tensor in batch.items()}
        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            for key in ("input_ids", "labels"):
                source = np.asarray(feature[key])
                if source.ndim != 1 or len(source) != length:
                    raise ValueError(f"Response {key} shape differs from input IDs.")
                if source.dtype.kind not in "biuf":
                    raise ValueError(f"Response {key} must contain numeric values.")
                np.copyto(views[key][row_index, :length], source, casting="unsafe")
            views["attention_mask"][row_index, :length] = 1
            if "loss_weights" in batch:
                raw_weights = feature["loss_weights"]
                # NumPy coercion would erase a bool inside a mixed Python list.
                if isinstance(raw_weights, (list, tuple)) and any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in raw_weights
                ):
                    raise ValueError("Structural response weights must be finite and positive.")
                weights = np.asarray(raw_weights)
                roles = np.asarray(feature.get("response_token_roles"))
                if (
                    weights.ndim != 1
                    or len(weights) != length
                    or roles.ndim != 1
                    or len(roles) != length
                ):
                    raise ValueError(
                        "Structural response weight/role shape differs from input IDs."
                    )
                if (
                    weights.dtype.kind not in "iuf"
                    or not bool(np.isfinite(weights).all())
                    or bool((weights <= 0).any())
                ):
                    raise ValueError("Structural response weights must be finite and positive.")
                if roles.dtype.kind not in "biuf":
                    raise ValueError("Structural response roles must contain numeric values.")
                np.copyto(views["loss_weights"][row_index, :length], weights, casting="unsafe")
                np.copyto(
                    views["response_token_roles"][row_index, :length], roles, casting="unsafe"
                )
        return batch

    def _collate_lists(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self._allocate(features)
        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            batch["input_ids"][row_index, :length] = torch.tensor(
                _as_list(feature["input_ids"]), dtype=torch.long
            )
            batch["attention_mask"][row_index, :length] = 1
            batch["labels"][row_index, :length] = torch.tensor(
                _as_list(feature["labels"]), dtype=torch.long
            )
            if "loss_weights" in batch:
                weights = _as_list(feature["loss_weights"])
                roles = _as_list(feature.get("response_token_roles"))
                if (
                    not isinstance(weights, list)
                    or len(weights) != length
                    or not isinstance(roles, list)
                    or len(roles) != length
                ):
                    raise ValueError(
                        "Structural response weight/role shape differs from input IDs."
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in weights
                ):
                    raise ValueError("Structural response weights must be finite and positive.")
                batch["loss_weights"][row_index, :length] = torch.tensor(
                    weights, dtype=torch.float32
                )
                batch["response_token_roles"][row_index, :length] = torch.tensor(
                    roles, dtype=torch.long
                )
        return batch


def _as_list(value: Any) -> Any:
    if isinstance(value, array):
        return value.tolist()
    # A lazy import keeps the original list-only path usable without array setup.
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
