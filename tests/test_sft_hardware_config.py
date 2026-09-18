from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from svg_agentic_slm.train.sft_trainer import SFTConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_a100_runtime_options_round_trip() -> None:
    config = SFTConfig.from_dict(
        {
            "tf32": True,
            "float32_matmul_precision": "high",
            "optim": "adamw_torch_fused",
            "torch_compile": False,
            "dataloader_drop_last": True,
            "ddp_backend": "nccl",
            "ddp_bucket_cap_mb": 100,
            "ddp_broadcast_buffers": False,
            "ddp_static_graph": True,
            "include_num_input_tokens_seen": True,
        }
    )

    assert config.tf32 is True
    assert config.float32_matmul_precision == "high"
    assert config.optim == "adamw_torch_fused"
    assert config.dataloader_drop_last is True
    assert config.ddp_backend == "nccl"
    assert config.ddp_bucket_cap_mb == 100
    assert config.ddp_broadcast_buffers is False
    assert config.ddp_static_graph is True
    assert config.include_num_input_tokens_seen is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tf32": "yes"}, "tf32"),
        ({"float32_matmul_precision": "fast"}, "float32_matmul_precision"),
        ({"torch_compile_mode": "quick"}, "torch_compile_mode"),
        ({"ddp_bucket_cap_mb": 0}, "ddp_bucket_cap_mb"),
        ({"ddp_static_graph": 1}, "ddp_static_graph"),
        ({"dataloader_drop_last": 1}, "dataloader_drop_last"),
    ],
)
def test_hardware_runtime_options_reject_invalid_values(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SFTConfig(**kwargs)


@pytest.mark.parametrize(
    (
        "filename",
        "max_length",
        "load_in_4bit",
        "gradient_checkpointing",
        "per_device_batch_size",
        "gradient_accumulation_steps",
    ),
    [
        ("train_lora_a100_80gb.yaml", 8192, False, True, 4, 4),
        ("train_lora_a100_80gb_long.yaml", 32768, True, True, 1, 16),
        ("train_critic_a100_80gb.yaml", 8192, False, False, 2, 8),
    ],
)
def test_a100_profiles_preserve_expected_speed_memory_tradeoff(
    filename: str,
    max_length: int,
    load_in_4bit: bool,
    gradient_checkpointing: bool,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    payload = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / filename).read_text(encoding="utf-8")
    )["train"]
    sft = SFTConfig.from_dict(payload["sft"])

    assert payload["model"]["load_in_4bit"] is load_in_4bit
    assert sft.max_seq_length == max_length
    assert sft.gradient_checkpointing is gradient_checkpointing
    assert sft.per_device_train_batch_size == per_device_batch_size
    assert sft.per_device_eval_batch_size == per_device_batch_size
    assert sft.gradient_accumulation_steps == gradient_accumulation_steps
    assert sft.tf32 is True
    assert sft.optim == "adamw_torch_fused"
    assert sft.dataloader_drop_last is False
    assert sft.ddp_backend == "nccl"
    assert sft.ddp_static_graph is False
