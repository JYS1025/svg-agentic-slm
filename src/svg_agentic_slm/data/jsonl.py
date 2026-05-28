"""JSONL read/write utilities.

Provides simple, reliable functions for reading and writing
JSON Lines files, the standard data format for this project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line contains invalid JSON.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
    return records


def write_jsonl(
    records: list[dict[str, Any]],
    file_path: str | Path,
    append: bool = False,
) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        records: List of dictionaries to write.
        file_path: Path to the output JSONL file.
        append: If True, append to existing file. Otherwise overwrite.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
