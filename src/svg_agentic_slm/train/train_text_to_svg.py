"""Configuration entry point for text-to-SVG QLoRA training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.svg.official_discrete_cache import (
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    OpenVGLabCacheConfig,
)
from svg_agentic_slm.train.sft_trainer import (
    ModelTrainingConfig,
    SFTConfig,
    TextToSVGSFTTrainer,
)
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_training(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if overrides:
        raise ValueError("Training overrides are not supported; persist an auditable YAML config.")
    config = load_yaml_config(config_path)
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("train configuration must be a mapping.")
    model = train_config.get("model", {})
    dataset = train_config.get("dataset", {})
    experiment = train_config.get("experiment", {})
    for name, value in (("model", model), ("dataset", dataset), ("experiment", experiment)):
        if not isinstance(value, dict):
            raise ValueError(f"train.{name} must be a mapping.")

    sft_config = SFTConfig.from_dict(train_config.get("sft", {}))
    set_seed(sft_config.seed)
    model_config = ModelTrainingConfig(
        model_id=str(model.get("model_id", "google/gemma-4-12B-it-qat-q4_0-unquantized")),
        revision=str(
            model.get("revision", "b6ed86275a6a5735884e208bfed95b445a684ca2")
        ),
        auto_model_class=str(model.get("auto_model_class", "multimodal_lm")),
        dtype=str(model.get("dtype", "bfloat16")),
        attn_implementation=str(model.get("attn_implementation", "sdpa")),
        load_in_4bit=bool(model.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(model.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(model.get("bnb_4bit_use_double_quant", True)),
        local_files_only=bool(model.get("local_files_only", False)),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        token_env=model.get("token_env"),
    )
    official_cache_data = experiment.get("official_cache")
    official_cache_config = None
    if official_cache_data is not None:
        if not isinstance(official_cache_data, dict):
            raise ValueError("train.experiment.official_cache must be a mapping.")
        required = ("cache_path", "audit_manifest_path", "prepared_root")
        missing = [key for key in required if not official_cache_data.get(key)]
        if missing:
            raise ValueError(
                "train.experiment.official_cache is missing: " + ", ".join(missing)
            )
        cache_kwargs: dict[str, Any] = {
            "cache_path": Path(str(official_cache_data["cache_path"])),
            "audit_manifest_path": Path(
                str(official_cache_data["audit_manifest_path"])
            ),
            "prepared_root": Path(str(official_cache_data["prepared_root"])),
            "allow_live_reencode": official_cache_data.get(
                "allow_live_reencode", False
            ),
        }
        for key in (
            "expected_cache_sha256",
            "expected_audit_sha256",
            "expected_input_sha256",
            "expected_split_counts",
        ):
            if key in official_cache_data:
                cache_kwargs[key] = official_cache_data[key]
        official_cache_config = OpenVGLabCacheConfig(**cache_kwargs)
    trainer = TextToSVGSFTTrainer(
        model_config=model_config,
        lora_config=LoRAConfig.from_dict(train_config.get("lora", {})),
        sft_config=sft_config,
        train_data_path=dataset.get("train_path", "./data/processed/mmsvg_sft_20k/train.jsonl"),
        eval_data_path=dataset.get(
            "validation_path", "./data/processed/mmsvg_sft_20k/validation.jsonl"
        ),
        test_data_path=dataset.get("test_path"),
        instruction_mode=str(experiment.get("instruction_mode", "description_only")),
        target_representation=str(experiment.get("target_representation", "raw_xml")),
        codec_backend_id=str(
            experiment.get("codec_backend_id", OFFICIAL_CACHED_GEMMA_BACKEND_ID)
        ),
        official_cache_config=official_cache_config,
        allow_legacy_toy_codec=experiment.get("allow_legacy_toy_codec", False),
        official_train_sample_ids=experiment.get("official_train_sample_ids"),
        official_validation_sample_ids=experiment.get(
            "official_validation_sample_ids"
        ),
    )
    logger.info("Starting auditable SFT experiment from %s", config_path)
    return trainer.train()


def main() -> None:
    """Run one YAML-defined SFT experiment under Python or Accelerate."""
    parser = argparse.ArgumentParser(description="Train the Gemma 4 text-to-SVG adapter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_lora.yaml"),
        help="Auditable SFT experiment YAML.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_training(args.config)
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
