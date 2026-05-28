"""Abstract interfaces for SVG validation and rendering.

Defines the contracts that concrete validators and renderers
must implement. This enables swapping backends (e.g., CairoSVG
vs. Playwright) without changing consumer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from svg_agentic_slm.svg.schemas import SVGRenderResult, SVGValidationResult


class BaseValidator(ABC):
    """Abstract interface for SVG validation."""

    @abstractmethod
    def validate(self, svg_content: str) -> SVGValidationResult:
        """Validate an SVG string.

        Args:
            svg_content: Raw SVG string to validate.

        Returns:
            Validation result with pass/fail status and diagnostics.
        """
        ...


class BaseRenderer(ABC):
    """Abstract interface for SVG rendering.

    Implementations convert SVG strings to raster images.
    The initial backend is CairoSVG, but this interface supports
    adding Playwright/browser-based rendering later.
    """

    @abstractmethod
    def render(
        self,
        svg_content: str,
        output_path: Path,
        width: int = 256,
        height: int = 256,
        output_format: str = "png",
    ) -> SVGRenderResult:
        """Render an SVG string to a raster image.

        Args:
            svg_content: The SVG string to render.
            output_path: Path where the rendered image will be saved.
            width: Output image width in pixels.
            height: Output image height in pixels.
            output_format: Image format ('png', 'pdf', etc.).

        Returns:
            Render result with success status and output path.
        """
        ...
