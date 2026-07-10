"""Tests for generation artifact readers."""

from __future__ import annotations

import json
from pathlib import Path

from svg_agentic_slm.artifacts.generation import (
    load_generation_artifact,
    list_generation_artifacts,
)


def test_load_generation_artifact_from_svg_path(tmp_path: Path) -> None:
    """Loading by SVG path should resolve the matching metadata sidecar."""
    svg_path = tmp_path / "sample.svg"
    metadata_path = tmp_path / "sample.json"
    svg_path.write_text("<svg></svg>", encoding="utf-8")

    payload = {
        "instruction": "Draw a square.",
        "svg_path": str(svg_path),
        "render_path": str(tmp_path / "sample.png"),
        "is_valid": True,
        "revision_count": 0,
        "critic_feedback": [],
        "runtime": {"enable_render": True},
        "metadata": {"validation": {"is_valid": True}},
        "generated_at_utc": "2026-07-10T12:00:00Z",
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    record = load_generation_artifact(svg_path)

    assert record.instruction == "Draw a square."
    assert record.svg_path == svg_path
    assert record.metadata_path == metadata_path
    assert record.render_path == tmp_path / "sample.png"
    assert record.runtime["enable_render"] is True


def test_list_generation_artifacts_returns_sorted_records(tmp_path: Path) -> None:
    """Artifact listing should read all JSON sidecars in sorted order."""
    for name in ["b", "a"]:
        svg_path = tmp_path / f"{name}.svg"
        metadata_path = tmp_path / f"{name}.json"
        svg_path.write_text("<svg></svg>", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "instruction": f"draw {name}",
                    "svg_path": str(svg_path),
                    "is_valid": True,
                    "revision_count": 0,
                    "critic_feedback": [],
                    "runtime": {},
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

    records = list_generation_artifacts(tmp_path)

    assert [record.instruction for record in records] == ["draw a", "draw b"]
