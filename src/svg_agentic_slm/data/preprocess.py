"""Shared post-adapter data preprocessing utilities.

Dataset-specific download, field mapping, joins, and split interpretation do
not belong here. Each dataset owns an isolated adapter under
``svg_agentic_slm.benchmarks``; this module may later host normalization that
is valid for every already-adapted text-to-SVG record.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_text_to_svg(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """Apply future dataset-neutral normalization to adapted JSONL records.

    Args:
        input_path: Path to the raw input data.
        output_path: Path for the processed output JSONL file.

    Returns:
        Number of examples processed.

    TODO: Define the dataset-neutral input contract before implementing.
    TODO: Keep upstream-specific mapping in one adapter per dataset.
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
