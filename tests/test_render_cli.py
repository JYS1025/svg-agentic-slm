"""Tests for the standalone render CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from svg_agentic_slm.cli.app import app

runner = CliRunner()


def test_render_command_rejects_unsupported_backend(tmp_path: Path) -> None:
    """The render CLI should fail fast for unknown backends."""
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "render",
            str(svg_path),
            "--backend",
            "playwright",
        ],
    )

    assert result.exit_code == 1
    assert "Unsupported backend" in result.stdout


def test_render_command_writes_png_when_cairosvg_available(tmp_path: Path) -> None:
    """The render CLI should save a PNG when CairoSVG is available."""
    pytest.importorskip("cairosvg")

    svg_path = tmp_path / "sample.svg"
    png_path = tmp_path / "sample.png"
    svg_path.write_text(
        (
            '<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="128" height="128" fill="white"/>'
            '<circle cx="64" cy="64" r="24" fill="blue"/>'
            "</svg>"
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "render",
            str(svg_path),
            "--output",
            str(png_path),
            "--width",
            "128",
            "--height",
            "128",
        ],
    )

    assert result.exit_code == 0
    assert "Rendered output saved to" in result.stdout
    assert png_path.exists()
    assert png_path.stat().st_size > 0
