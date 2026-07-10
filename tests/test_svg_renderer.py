"""Tests for SVG rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from svg_agentic_slm.svg.renderer import CairoSVGRenderer


def test_cairosvg_renderer_renders_png(tmp_path: Path) -> None:
    """The CairoSVG renderer should produce a PNG file for a valid SVG."""
    pytest.importorskip("cairosvg")

    renderer = CairoSVGRenderer()
    output_path = tmp_path / "sample.png"
    svg = (
        '<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="128" height="128" fill="white"/>'
        '<circle cx="64" cy="64" r="24" fill="blue"/>'
        "</svg>"
    )

    result = renderer.render(svg, output_path, width=128, height=128)

    assert result.success is True
    assert result.output_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cairosvg_renderer_rejects_unknown_format(tmp_path: Path) -> None:
    """Unsupported formats should fail cleanly without raising."""
    renderer = CairoSVGRenderer()
    output_path = tmp_path / "sample.bmp"
    svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    result = renderer.render(svg, output_path, output_format="bmp")

    assert result.success is False
    assert result.error is not None
    assert "Unsupported render format" in result.error
