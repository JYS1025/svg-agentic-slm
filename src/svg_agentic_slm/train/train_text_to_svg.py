"""Entry point for text-to-SVG fine-tuning.

Provides a high-level function to load config and run training.
This is called by the CLI train command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.train.sft_trainer import SFTConfig, TextToSVGSFTTrainer
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_training(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Load configuration and run text-to-SVG LoRA training.

    Args:
        config_path: Path to the train_lora.yaml config file.
        overrides: Optional dictionary of config overrides.

    TODO: Implement config override merging.
    """
    config = load_yaml_config(config_path)
    train_config = config.get("train", {})

    seed = train_config.get("sft", {}).get("seed", 42)
    set_seed(seed)

    lora_config = LoRAConfig.from_dict(train_config.get("lora", {}))
    sft_config = SFTConfig.from_dict(train_config.get("sft", {}))

    model_id = train_config.get("base_model_id", "google/gemma-3-4b-it")
    dataset_config = train_config.get("dataset", {})

    trainer = TextToSVGSFTTrainer(
        model_id=model_id,
        lora_config=lora_config,
        sft_config=sft_config,
        train_data_path=dataset_config.get("train_path", "./data/processed/train.jsonl"),
        eval_data_path=dataset_config.get("eval_path"),
    )

    logger.info("Starting training with config from: %s", config_path)
    trainer.train()
    logger.info("Training complete.")
