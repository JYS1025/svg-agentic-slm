"""Data schemas for SVG processing results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SVGValidationResult:
    """Result of validating an SVG string.

    Attributes:
        is_valid: Whether the SVG passed all validation checks.
        errors: List of validation error messages.
        warnings: List of non-fatal validation warnings.
        has_svg_tag: Whether the string contains an <svg> opening tag.
        has_closing_tag: Whether the string contains a </svg> closing tag.
        is_well_formed_xml: Whether the SVG parses as valid XML.
    """

    is_valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_svg_tag: bool = False
    has_closing_tag: bool = False
    is_well_formed_xml: bool = False


@dataclass
class SVGRenderResult:
    """Result of rendering an SVG to a raster image.

    Attributes:
        success: Whether the render completed successfully.
        output_path: Path to the rendered image file, if successful.
        format: Image format (e.g., 'png', 'pdf').
        width: Rendered image width in pixels.
        height: Rendered image height in pixels.
        error: Error message if rendering failed.
    """

    success: bool = False
    output_path: Path | None = None
    format: str = "png"
    width: int = 256
    height: int = 256
    error: str | None = None
