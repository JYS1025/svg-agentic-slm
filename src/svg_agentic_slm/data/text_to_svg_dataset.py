"""Text-to-SVG dataset abstraction.

Provides a dataset class for loading and iterating over
text-to-SVG examples from JSONL files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from svg_agentic_slm.data.jsonl import read_jsonl
from svg_agentic_slm.data.schemas import TextToSVGExample

logger = logging.getLogger(__name__)


class TextToSVGDataset:
    """A dataset of text-to-SVG examples loaded from JSONL.

    Args:
        file_path: Path to the JSONL file containing examples.

    TODO: Add support for HuggingFace Datasets integration.
    TODO: Add train/eval/test split support.
    TODO: Add filtering and sampling utilities.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._examples: list[TextToSVGExample] = []
        self._loaded = False

    def load(self) -> None:
        """Load examples from the JSONL file."""
        raw_records = read_jsonl(self.file_path)
        self._examples = [
            TextToSVGExample.from_dict(record)
            for record in raw_records
            if record.get("task") == "text_to_svg"
        ]
        self._loaded = True
        logger.info("Loaded %d text-to-SVG examples from %s", len(self._examples), self.file_path)

    @property
    def examples(self) -> list[TextToSVGExample]:
        """Return loaded examples. Raises if not yet loaded."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call .load() first.")
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TextToSVGExample:
        return self.examples[index]
