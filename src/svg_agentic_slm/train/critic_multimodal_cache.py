"""Shared-memory token and Qwen image-patch cache for multimodal critic SFT."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from array import array
from pathlib import Path
from typing import Any

import numpy as np

from svg_agentic_slm.prompts.vlm_critic import (
    VLM_CRITIC_PROMPT_VERSION,
    build_vlm_critic_prompt,
    build_vlm_critic_system_prompt,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def critic_cache_fingerprint(
    *,
    split_paths: dict[str, Path],
    model_id: str,
    revision: str,
    max_seq_length: int,
    enable_thinking: bool,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "processor_class": "AutoProcessor",
        "critic_prompt_version": VLM_CRITIC_PROMPT_VERSION,
        "score_threshold": 3.0,
        "enable_thinking": enable_thinking,
        "max_seq_length": max_seq_length,
        "split_sha256": {
            split: _file_sha256(path) for split, path in sorted(split_paths.items())
        },
        "implementation_sha256": _file_sha256(Path(__file__).resolve()),
        "pixel_storage": "torch_bfloat16_raw_uint16",
    }
    return hashlib.sha256(_canonical(identity)).hexdigest(), identity


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            result.append(row)
    return result


def _chat_texts(processor: Any, row: dict[str, Any], image: Any) -> tuple[str, str]:
    system = build_vlm_critic_system_prompt(score_threshold=3.0)
    prompt = build_vlm_critic_prompt(
        row["instruction"],
        labeled_svg=row["labeled_svg"],
        allowed_target_ids=row["allowed_target_ids"],
        score_threshold=3.0,
        similarity_score=row["similarity_score"],
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prefix = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    full = processor.apply_chat_template(
        [*messages, {"role": "assistant", "content": row["target_json"]}],
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False,
    )
    return prefix, full


def _build_split(
    *,
    processor: Any,
    source: Path,
    destination: Path,
    max_seq_length: int,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    rows = _rows(source)
    offsets = array("Q", [0])
    prefix_lengths = array("I")
    grid_values = array("i")
    sample_ids: list[str] = []
    pixel_shape: tuple[int, int] | None = None
    input_path = destination / "input_ids.i32"
    type_path = destination / "mm_token_type_ids.u8"
    pixel_path = destination / "pixel_values.bf16"
    with (
        input_path.open("wb") as input_handle,
        type_path.open("wb") as type_handle,
        pixel_path.open("wb") as pixel_handle,
    ):
        for index, row in enumerate(rows):
            image_path = Path(row["candidate_image_path"])
            if _file_sha256(image_path) != row["candidate_image_sha256"]:
                raise ValueError(f"Cached image source hash mismatch: {image_path}")
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB").copy()
            try:
                prefix_text, full_text = _chat_texts(processor, row, image)
                prefix_encoded = processor(
                    text=prefix_text,
                    images=image,
                    return_tensors="pt",
                )
                encoded = processor(
                    text=full_text,
                    images=image,
                    return_tensors="pt",
                )
            finally:
                image.close()
            input_ids = encoded["input_ids"][0].to(dtype=torch.int32, device="cpu")
            prefix_ids = prefix_encoded["input_ids"][0].to(dtype=torch.int32, device="cpu")
            if not torch.equal(input_ids[: prefix_ids.numel()], prefix_ids):
                raise ValueError(
                    f"Assistant-prefix tokenization is not a prefix for {row['sample_id']}"
                )
            length = int(input_ids.numel())
            if length > max_seq_length:
                raise ValueError(
                    f"Critic sample {row['sample_id']} has {length} tokens, exceeding "
                    f"max_seq_length={max_seq_length}; truncation is forbidden."
                )
            if prefix_ids.numel() >= length:
                raise ValueError(f"Critic sample has no supervised response: {row['sample_id']}")
            token_types = encoded.get("mm_token_type_ids")
            if token_types is None:
                token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
            else:
                token_types = token_types[0].to(dtype=torch.uint8, device="cpu")
            pixels = encoded["pixel_values"].to(dtype=torch.bfloat16, device="cpu").contiguous()
            shape = tuple(int(value) for value in pixels.shape)
            if len(shape) != 2:
                raise ValueError("Qwen pixel_values must have [patches, patch_features] shape.")
            if pixel_shape is None:
                pixel_shape = shape
            elif shape != pixel_shape:
                raise ValueError(
                    "All critic renders must produce a fixed Qwen patch shape; "
                    f"observed {shape} after {pixel_shape}."
                )
            grid = encoded["image_grid_thw"].reshape(-1).to(dtype=torch.int32, device="cpu")
            if grid.numel() != 3:
                raise ValueError("Each critic sample must contain exactly one image grid.")
            input_handle.write(input_ids.numpy().tobytes())
            type_handle.write(token_types.numpy().tobytes())
            pixel_handle.write(pixels.view(torch.uint16).numpy().tobytes())
            offsets.append(offsets[-1] + length)
            prefix_lengths.append(prefix_ids.numel())
            grid_values.extend(grid.tolist())
            sample_ids.append(row["sample_id"])
            if index and index % 100 == 0:
                input_handle.flush()
                type_handle.flush()
                pixel_handle.flush()
    if pixel_shape is None:
        raise ValueError(f"Critic split is empty: {source}")
    np.save(destination / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(destination / "prefix_lengths.npy", np.asarray(prefix_lengths, dtype=np.uint32))
    np.save(
        destination / "image_grid_thw.npy",
        np.asarray(grid_values, dtype=np.int32).reshape(-1, 3),
    )
    (destination / "sample_ids.json").write_text(json.dumps(sample_ids) + "\n")
    return {
        "count": len(rows),
        "total_tokens": int(offsets[-1]),
        "min_tokens": min(
            int(offsets[i + 1] - offsets[i]) for i in range(len(rows))
        ),
        "max_tokens": max(
            int(offsets[i + 1] - offsets[i]) for i in range(len(rows))
        ),
        "pixel_shape_per_sample": list(pixel_shape),
        "pixel_dtype": "bfloat16",
        "input_ids_dtype": "int32",
        "mm_token_type_ids_dtype": "uint8",
        "image_grid_thw_dtype": "int32",
    }


def load_or_build_critic_cache(
    *,
    processor: Any,
    split_paths: dict[str, Path],
    cache_root: str | Path,
    model_id: str,
    revision: str,
    max_seq_length: int,
    enable_thinking: bool = False,
) -> tuple[dict[str, CriticMultimodalDataset], dict[str, Any]]:
    fingerprint, identity = critic_cache_fingerprint(
        split_paths=split_paths,
        model_id=model_id,
        revision=revision,
        max_seq_length=max_seq_length,
        enable_thinking=enable_thinking,
    )
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root / fingerprint
    lock_path = root / f".{fingerprint}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = final / "manifest.json"
        if not manifest_path.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=root))
            try:
                split_manifests = {}
                for split, source in sorted(split_paths.items()):
                    split_dir = temporary / split
                    split_dir.mkdir()
                    split_manifests[split] = _build_split(
                        processor=processor,
                        source=source,
                        destination=split_dir,
                        max_seq_length=max_seq_length,
                    )
                manifest = {
                    **identity,
                    "fingerprint": fingerprint,
                    "splits": split_manifests,
                }
                (temporary / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                os.replace(temporary, final)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Critic cache manifest fingerprint mismatch.")
    datasets = {
        split: CriticMultimodalDataset(final / split, manifest["splits"][split])
        for split in split_paths
    }
    return datasets, manifest


class CriticMultimodalDataset:
    """Read-only mmap dataset; PNG decoding and image processing never occur here."""

    def __init__(self, path: Path, manifest: dict[str, Any]) -> None:
        self.path = path
        self.manifest = manifest
        count = int(manifest["count"])
        total_tokens = int(manifest["total_tokens"])
        pixel_shape = tuple(manifest["pixel_shape_per_sample"])
        self._offsets = np.load(path / "offsets.npy", mmap_mode="r")
        self._prefix_lengths = np.load(path / "prefix_lengths.npy", mmap_mode="r")
        self._grid = np.load(path / "image_grid_thw.npy", mmap_mode="r")
        self._input_ids = np.memmap(
            path / "input_ids.i32", mode="r", dtype=np.int32, shape=(total_tokens,)
        )
        self._token_types = np.memmap(
            path / "mm_token_type_ids.u8", mode="r", dtype=np.uint8, shape=(total_tokens,)
        )
        self._pixels = np.memmap(
            path / "pixel_values.bf16",
            mode="r",
            dtype=np.uint16,
            shape=(count, *pixel_shape),
        )
        self.verified_lengths = tuple(
            int(self._offsets[i + 1] - self._offsets[i]) for i in range(count)
        )

    def __len__(self) -> int:
        return len(self._prefix_lengths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        start, end = int(self._offsets[index]), int(self._offsets[index + 1])
        ids = torch.from_numpy(np.array(self._input_ids[start:end], copy=True)).long()
        labels = ids.clone()
        labels[: int(self._prefix_lengths[index])] = -100
        pixels_u16 = torch.from_numpy(np.array(self._pixels[index], copy=True))
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
            "mm_token_type_ids": torch.from_numpy(
                np.array(self._token_types[start:end], copy=True)
            ).long(),
            "pixel_values": pixels_u16.view(torch.bfloat16),
            "image_grid_thw": torch.from_numpy(np.array(self._grid[index], copy=True)).long(),
        }


class CriticMultimodalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Critic collator requires at least one feature.")
        maximum = max(len(item["input_ids"]) for item in features)
        shape = (len(features), maximum)
        result = {
            "input_ids": torch.full(shape, self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros(shape, dtype=torch.long),
            "labels": torch.full(shape, -100, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(shape, dtype=torch.long),
        }
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            for name in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
                result[name][row, :length] = feature[name]
        result["pixel_values"] = torch.cat(
            [feature["pixel_values"] for feature in features], dim=0
        )
        result["image_grid_thw"] = torch.stack(
            [feature["image_grid_thw"] for feature in features]
        )
        return result


__all__ = [
    "CriticMultimodalCollator",
    "CriticMultimodalDataset",
    "critic_cache_fingerprint",
    "load_or_build_critic_cache",
]
