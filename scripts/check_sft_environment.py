#!/usr/bin/env python3
"""Fail fast when the configured SFT stack cannot execute on visible GPUs."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def _load_training_settings(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="document")
    train = _mapping(document.get("train"), name="train")
    model = _mapping(train.get("model"), name="train.model")
    sft = _mapping(train.get("sft"), name="train.sft")
    return model, sft


def _smoke_test_gpu(
    torch: Any,
    *,
    index: int,
    dtype: Any,
    quant_type: str,
    double_quant: bool,
    test_4bit: bool,
    test_8bit_optimizer: bool,
) -> None:
    device = torch.device(f"cuda:{index}")
    probe = torch.randn(64, 64, device=device, dtype=dtype, requires_grad=True)
    probe.square().mean().backward()
    if not bool(torch.isfinite(probe.grad).all()):
        raise RuntimeError("native CUDA forward/backward produced a non-finite gradient")

    if test_4bit or test_8bit_optimizer:
        import bitsandbytes as bnb

    if test_4bit:
        layer = bnb.nn.Linear4bit(
            64,
            64,
            bias=False,
            compute_dtype=dtype,
            compress_statistics=double_quant,
            quant_type=quant_type,
        )
        with torch.no_grad():
            layer.weight.normal_()
        layer = layer.to(device)
        inputs = torch.randn(2, 64, device=device, dtype=dtype, requires_grad=True)
        layer(inputs).float().square().mean().backward()
        if not bool(torch.isfinite(inputs.grad).all()):
            raise RuntimeError("bitsandbytes 4-bit backward produced a non-finite gradient")

    if test_8bit_optimizer:
        parameter = torch.nn.Parameter(torch.randn(64, 64, device=device))
        optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=1e-4)
        parameter.square().mean().backward()
        optimizer.step()
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError("bitsandbytes 8-bit optimizer produced a non-finite parameter")

    torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Require at least this many CUDA devices after CUDA_VISIBLE_DEVICES filtering.",
    )
    parser.add_argument(
        "--skip-kernel-smoke",
        action="store_true",
        help="Only inspect availability; do not execute CUDA and bitsandbytes kernels.",
    )
    args = parser.parse_args()

    if args.expected_gpus is not None and args.expected_gpus <= 0:
        parser.error("--expected-gpus must be positive.")

    try:
        model, sft = _load_training_settings(args.config)
        import torch
    except (ImportError, OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    errors: list[str] = []
    if not torch.cuda.is_available():
        errors.append(
            "PyTorch cannot access CUDA. Install the PyTorch build matching the server driver "
            "and expose at least one NVIDIA GPU."
        )
    visible_gpus = torch.cuda.device_count()
    expected_gpus = args.expected_gpus or visible_gpus
    if visible_gpus < expected_gpus:
        errors.append(f"expected {expected_gpus} visible GPUs, but PyTorch found {visible_gpus}")

    dtype_name = str(model.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        errors.append(f"torch has no dtype named {dtype_name!r}")
    if bool(sft.get("bf16", False)) and dtype_name != "bfloat16":
        errors.append("sft.bf16=true requires train.model.dtype=bfloat16")
    if bool(sft.get("bf16", False)) and bool(sft.get("fp16", False)):
        errors.append("sft.bf16 and sft.fp16 cannot both be true")

    use_4bit = bool(model.get("load_in_4bit", False))
    optim = str(sft.get("optim", ""))
    use_8bit_optimizer = "8bit" in optim
    if use_4bit or use_8bit_optimizer:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:  # Native backend load failures vary by platform.
            errors.append(
                "bitsandbytes could not load; install a build compatible with this server: "
                f"{exc}"
            )

    devices_to_test = min(expected_gpus, visible_gpus)
    if dtype is not None and not errors:
        for index in range(devices_to_test):
            name = torch.cuda.get_device_name(index)
            properties = torch.cuda.get_device_properties(index)
            if dtype is torch.bfloat16:
                with torch.cuda.device(index):
                    if not torch.cuda.is_bf16_supported():
                        errors.append(f"GPU {index} ({name}) does not support configured BF16")
                        continue
            if not args.skip_kernel_smoke:
                try:
                    _smoke_test_gpu(
                        torch,
                        index=index,
                        dtype=dtype,
                        quant_type=str(model.get("bnb_4bit_quant_type", "nf4")),
                        double_quant=bool(model.get("bnb_4bit_use_double_quant", True)),
                        test_4bit=use_4bit,
                        test_8bit_optimizer=use_8bit_optimizer,
                    )
                except Exception as exc:  # CUDA extension failures do not share one exception type.
                    errors.append(f"GPU {index} ({name}) kernel smoke test failed: {exc}")
            memory_gib = properties.total_memory / 1024**3
            print(
                f"GPU {index}: {name}; compute capability "
                f"{properties.major}.{properties.minor}; VRAM {memory_gib:.1f} GiB"
            )

    versions = ", ".join(
        f"{name}={_package_version(distribution)}"
        for name, distribution in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("peft", "peft"),
            ("bitsandbytes", "bitsandbytes"),
        )
    )
    print(f"Python {sys.version.split()[0]}; {versions}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"SFT environment check passed for {devices_to_test} visible GPU(s).")
    print(
        "Model download/access, dataset paths, free disk space, and full-model VRAM "
        "remain runtime inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
