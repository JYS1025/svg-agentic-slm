"""Configuration entry point for image-grounded Critic SFT."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from svg_agentic_slm.train.critic_sft_trainer import (
    CriticSFTTrainer,
    load_critic_sft_records,
)
from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.train.sft_trainer import ModelTrainingConfig, SFTConfig
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def build_critic_trainer(config_path: str | Path) -> CriticSFTTrainer:
    config = load_yaml_config(config_path)
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("train configuration must be a mapping.")
    model = train_config.get("model", {})
    dataset = train_config.get("dataset", {})
    critic = train_config.get("critic", {})
    for name, value in (("model", model), ("dataset", dataset), ("critic", critic)):
        if not isinstance(value, dict):
            raise ValueError(f"train.{name} must be a mapping.")
    sft_config = SFTConfig.from_dict(train_config.get("sft", {}))
    set_seed(sft_config.seed)
    model_config = ModelTrainingConfig(
        model_id=str(model.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
        revision=str(
            model.get("revision", "66285546d2b821cf421d4f5eb2576359d3770cd3")
        ),
        auto_model_class=str(model.get("auto_model_class", "image_text_to_text")),
        dtype=str(model.get("dtype", "bfloat16")),
        attn_implementation=str(model.get("attn_implementation", "sdpa")),
        load_in_4bit=bool(model.get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(model.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(model.get("bnb_4bit_use_double_quant", True)),
        local_files_only=bool(model.get("local_files_only", False)),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        token_env=model.get("token_env"),
    )
    train_path = dataset.get("train_path")
    if not isinstance(train_path, (str, Path)) or not str(train_path).strip():
        raise ValueError("train.dataset.train_path is required for Critic SFT.")
    validation_path = dataset.get("validation_path")
    return CriticSFTTrainer(
        model_config=model_config,
        lora_config=LoRAConfig.from_dict(train_config.get("lora", {})),
        sft_config=sft_config,
        train_data_path=train_path,
        eval_data_path=validation_path,
        score_threshold=float(critic.get("score_threshold", 3.0)),
    )


def run_training(config_path: str | Path) -> dict[str, Any]:
    logger.info("Starting auditable Critic SFT experiment from %s", config_path)
    return build_critic_trainer(config_path).train()


def validate_training_data(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    train_config = config.get("train", {})
    dataset = train_config.get("dataset", {})
    critic = train_config.get("critic", {})
    sft = SFTConfig.from_dict(train_config.get("sft", {}))
    threshold = float(critic.get("score_threshold", 3.0))
    _, train_manifest = load_critic_sft_records(
        dataset["train_path"],
        score_threshold=threshold,
    )
    validation_manifest = None
    if sft.do_eval:
        _, validation_manifest = load_critic_sft_records(
            dataset["validation_path"],
            score_threshold=threshold,
        )
        overlap = set(train_manifest["sample_ids"]) & set(
            validation_manifest["sample_ids"]
        )
        if overlap:
            raise ValueError("Critic train/validation sample IDs overlap.")
    return {"train": train_manifest, "validation": validation_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the image-grounded SVG Critic adapter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_critic_a100_80gb.yaml"),
        help="Auditable Critic SFT experiment YAML.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate label/image/split contracts without loading a model.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = (
        validate_training_data(args.config)
        if args.validate_only
        else run_training(args.config)
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
