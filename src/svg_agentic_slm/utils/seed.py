"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # TODO: Consider torch.backends.cudnn.deterministic = True
            # for full reproducibility (may impact performance).
    except ImportError:
        pass
