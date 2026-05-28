"""LoRA configuration for fine-tuning.

Defines the configuration dataclass for LoRA parameters,
separate from model loading or training loop logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        bias: Bias type ('none', 'all', 'lora_only').
        task_type: Task type for the PEFT config.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> Any:
        """Convert to a PEFT LoraConfig object.

        Returns:
            A peft.LoraConfig instance.

        TODO: Implement when peft is available:
        # from peft import LoraConfig as PeftLoraConfig
        # return PeftLoraConfig(
        #     r=self.r,
        #     lora_alpha=self.lora_alpha,
        #     lora_dropout=self.lora_dropout,
        #     target_modules=self.target_modules,
        #     bias=self.bias,
        #     task_type=self.task_type,
        # )
        """
        raise NotImplementedError(
            "PEFT integration not yet implemented. "
            "Install peft and implement to_peft_config()."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoRAConfig:
        """Create from a dictionary (e.g., from YAML config)."""
        return cls(
            r=data.get("r", 16),
            lora_alpha=data.get("lora_alpha", 32),
            lora_dropout=data.get("lora_dropout", 0.05),
            target_modules=data.get("target_modules", ["q_proj", "v_proj"]),
            bias=data.get("bias", "none"),
            task_type=data.get("task_type", "CAUSAL_LM"),
        )
