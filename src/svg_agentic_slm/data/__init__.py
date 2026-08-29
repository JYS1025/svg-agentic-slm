"""Dataset schemas, JSONL utilities, and data loading.

This module handles data I/O and dataset abstractions.
It is independent from model, agent, and training code.
"""

from svg_agentic_slm.data.mmsvg_sft import (
    MMSVGPreparationConfig,
    load_preparation_config,
    prepare_mmsvg_sft_dataset,
)

__all__ = [
    "MMSVGPreparationConfig",
    "load_preparation_config",
    "prepare_mmsvg_sft_dataset",
]
