"""Data schemas for SVG processing results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ValidityErrorCode = Literal[
    "empty_svg", "xml_parse_error", "missing_svg_root", "invalid_root",
    "invalid_namespace", "unsafe_doctype", "unsafe_element", "unsafe_attribute",
    "unsafe_css", "external_reference", "render_failure", "render_timeout",
    "render_output_error", "reserved_label_dependency",
]


@dataclass(frozen=True)
class SVGDiagnostic:
    code: ValidityErrorCode
    message: str
    severity: Literal["error", "warning"] = "error"
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class SVGElementRef:
    agent_id: str
    xpath: str
    tag: str
    original_id: str | None
    parent_agent_id: str | None
    role: Literal["svg", "group", "graphics", "resource"]


@dataclass(frozen=True)
class SVGLabelingResult:
    attempt_id: str
    labeled_svg: str
    elements: dict[str, SVGElementRef]


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
    diagnostics: list[SVGDiagnostic] = field(default_factory=list)


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
