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


def test_render_command_rejects_mismatched_format_and_extension(tmp_path: Path) -> None:
    """The declared output format must agree with the explicit file extension."""
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "render",
            str(svg_path),
            "--output",
            str(tmp_path / "sample.png"),
            "--format",
            "pdf",
        ],
    )

    assert result.exit_code == 1
    assert "does not match --format" in result.stdout


def test_render_command_rejects_unsafe_svg_before_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svg_path = tmp_path / "unsafe.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>',
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("renderer must not receive unsafe SVG")

    monkeypatch.setattr("svg_agentic_slm.svg.renderer.CairoSVGRenderer.render", fail_if_called)
    result = runner.invoke(app, ["render", str(svg_path)])

    assert result.exit_code == 1
    assert "validation failed" in result.stdout


def test_render_command_uses_format_for_default_output(tmp_path: Path, monkeypatch) -> None:
    """A format override should also determine the default output suffix."""
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_render(self, svg_content, output_path, width=256, height=256, output_format="png"):
        from svg_agentic_slm.svg.schemas import SVGRenderResult

        captured["output_path"] = output_path
        captured["output_format"] = output_format
        return SVGRenderResult(success=True, output_path=output_path, format=output_format)

    monkeypatch.setattr("svg_agentic_slm.svg.renderer.CairoSVGRenderer.render", fake_render)
    result = runner.invoke(app, ["render", str(svg_path), "--format", "pdf"])

    assert result.exit_code == 0
    assert captured["output_path"] == tmp_path / "sample.pdf"
    assert captured["output_format"] == "pdf"
