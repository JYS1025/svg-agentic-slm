"""Data preprocessing utilities.

Provides functions to preprocess raw data into the standardized
text-to-SVG JSONL format.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_text_to_svg(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Preprocess raw data into text-to-SVG JSONL format.

    Args:
        input_path: Path to the raw input data.
        output_path: Path for the processed output JSONL file.

    Returns:
        Number of examples processed.

    TODO: Implement data cleaning and normalization.
    TODO: Add SVG validation as part of preprocessing.
    TODO: Add deduplication.
    TODO: Add train/eval/test splitting.
    """
    logger.info(
        "[PLACEHOLDER] Would preprocess %s -> %s",
        input_path,
        output_path,
    )
    # TODO: Implement actual preprocessing pipeline.
    return 0
