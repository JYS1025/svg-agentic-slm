"""Tests for SVG validator."""

from __future__ import annotations

from svg_agentic_slm.svg.validator import SVGValidator


def test_valid_svg() -> None:
    """Test that a well-formed SVG passes validation."""
    validator = SVGValidator()
    svg = '<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
    result = validator.validate(svg)

    assert result.is_valid
    assert result.has_svg_tag
    assert result.has_closing_tag
    assert not result.errors


def test_missing_svg_tag() -> None:
    """Test that missing <svg> tag is detected."""
    validator = SVGValidator()
    result = validator.validate("<div>Not an SVG</div>")

    assert not result.is_valid
    assert not result.has_svg_tag
    assert any("<svg>" in e for e in result.errors)


def test_missing_closing_tag() -> None:
    """Test that missing </svg> tag is detected."""
    validator = SVGValidator()
    result = validator.validate('<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/>')

    assert not result.is_valid
    assert result.has_svg_tag
    assert not result.has_closing_tag


def test_empty_content() -> None:
    """Test that empty content is rejected."""
    validator = SVGValidator()
    result = validator.validate("")

    assert not result.is_valid
    assert result.errors


def test_whitespace_only() -> None:
    """Test that whitespace-only content is rejected."""
    validator = SVGValidator()
    result = validator.validate("   \n\t  ")

    assert not result.is_valid


def test_missing_xmlns_warning() -> None:
    """Test that missing xmlns produces a warning."""
    validator = SVGValidator()
    svg = '<svg width="256" height="256"><circle r="10"/></svg>'
    result = validator.validate(svg)

    assert result.is_valid  # Warnings don't cause failure
    assert result.warnings
    assert any("xmlns" in w for w in result.warnings)
