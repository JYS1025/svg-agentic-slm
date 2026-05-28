"""SFT (Supervised Fine-Tuning) trainer skeleton.

Provides the trainer class for LoRA fine-tuning on text-to-SVG data.
Does not implement actual training — only defines the interface
and placeholder logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from svg_agentic_slm.train.lora_config import LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning.

    Attributes:
        output_dir: Directory for saving checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        warmup_ratio: Warmup ratio.
        lr_scheduler_type: Learning rate scheduler type.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        max_seq_length: Maximum sequence length.
        bf16: Use bfloat16 precision.
        seed: Random seed.
    """

    output_dir: str = "./checkpoints/lora_text_to_svg"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_seq_length: int = 4096
    bf16: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTConfig:
        """Create from a dictionary (e.g., from YAML config)."""
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class TextToSVGSFTTrainer:
    """Trainer for LoRA fine-tuning on text-to-SVG data.

    Args:
        model_id: HuggingFace model identifier.
        lora_config: LoRA configuration.
        sft_config: SFT training configuration.
        train_data_path: Path to training JSONL file.
        eval_data_path: Optional path to evaluation JSONL file.

    TODO: Integrate with HuggingFace TRL SFTTrainer.
    TODO: Add data collation and formatting.
    TODO: Add PEFT model wrapping.
    TODO: Add Weights & Biases / TensorBoard logging.
    """

    def __init__(
        self,
        model_id: str,
        lora_config: LoRAConfig,
        sft_config: SFTConfig,
        train_data_path: str | Path,
        eval_data_path: str | Path | None = None,
    ) -> None:
        self._model_id = model_id
        self._lora_config = lora_config
        self._sft_config = sft_config
        self._train_data_path = Path(train_data_path)
        self._eval_data_path = Path(eval_data_path) if eval_data_path else None

    def train(self) -> None:
        """Run the training loop.

        TODO: Implement the full training pipeline:
        1. Load base model with quantization config.
        2. Apply PEFT/LoRA.
        3. Load and format training dataset.
        4. Initialize TRL SFTTrainer.
        5. Run training.
        6. Save LoRA weights.
        """
        logger.info(
            "[PLACEHOLDER] Would train LoRA on model '%s' with data from '%s'",
            self._model_id,
            self._train_data_path,
        )
        logger.info("LoRA config: r=%d, alpha=%d", self._lora_config.r, self._lora_config.lora_alpha)
        logger.info("SFT config: epochs=%d, lr=%.2e", self._sft_config.num_train_epochs, self._sft_config.learning_rate)
        logger.warning("Training not yet implemented — this is a placeholder.")

    def save_model(self, output_path: str | Path) -> None:
        """Save the trained LoRA adapter weights.

        Args:
            output_path: Directory to save the adapter.

        TODO: Implement adapter saving.
        """
        logger.info("[PLACEHOLDER] Would save LoRA adapter to %s", output_path)
