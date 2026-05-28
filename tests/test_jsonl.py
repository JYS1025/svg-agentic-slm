"""Tests for JSONL read/write utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svg_agentic_slm.data.jsonl import read_jsonl, write_jsonl


def test_write_and_read_jsonl(tmp_path: Path) -> None:
    """Test that write_jsonl and read_jsonl are symmetric."""
    records = [
        {"task": "text_to_svg", "instruction": "Draw a circle.", "output_svg": "<svg></svg>"},
        {"task": "text_to_svg", "instruction": "Draw a square.", "output_svg": "<svg></svg>"},
    ]
    file_path = tmp_path / "test.jsonl"

    write_jsonl(records, file_path)
    loaded = read_jsonl(file_path)

    assert loaded == records


def test_read_jsonl_file_not_found() -> None:
    """Test that read_jsonl raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        read_jsonl("/nonexistent/path/data.jsonl")


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    """Test that write_jsonl creates parent directories."""
    nested_path = tmp_path / "a" / "b" / "c" / "data.jsonl"
    write_jsonl([{"key": "value"}], nested_path)
    assert nested_path.exists()


def test_write_jsonl_append_mode(tmp_path: Path) -> None:
    """Test that write_jsonl can append to existing files."""
    file_path = tmp_path / "test.jsonl"

    write_jsonl([{"a": 1}], file_path)
    write_jsonl([{"b": 2}], file_path, append=True)

    loaded = read_jsonl(file_path)
    assert len(loaded) == 2
    assert loaded[0] == {"a": 1}
    assert loaded[1] == {"b": 2}


def test_read_jsonl_skips_empty_lines(tmp_path: Path) -> None:
    """Test that read_jsonl skips blank lines."""
    file_path = tmp_path / "test.jsonl"
    file_path.write_text('{"a": 1}\n\n{"b": 2}\n\n')

    loaded = read_jsonl(file_path)
    assert len(loaded) == 2
