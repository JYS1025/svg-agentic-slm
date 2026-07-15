"""SVG rendering implementations.

Provides SVG-to-raster rendering via CairoSVG as the initial backend.
The BaseRenderer interface allows adding alternative renderers
(e.g., Playwright/browser-based) in the future.
"""

from __future__ import annotations

import logging
from pathlib import Path

from svg_agentic_slm.svg.base import BaseRenderer
from svg_agentic_slm.svg.schemas import SVGRenderResult

logger = logging.getLogger(__name__)


class CairoSVGRenderer(BaseRenderer):
    """SVG renderer using CairoSVG.

    Converts SVG strings to PNG (or other raster formats) using
    the CairoSVG library.
    """

    def render(
        self,
        svg_content: str,
        output_path: Path,
        width: int = 256,
        height: int = 256,
        output_format: str = "png",
    ) -> SVGRenderResult:
        """Render an SVG string to a raster image using CairoSVG.

        Args:
            svg_content: The SVG string to render.
            output_path: Path where the rendered image will be saved.
            width: Output image width in pixels.
            height: Output image height in pixels.
            output_format: Image format ('png', 'pdf', 'ps', 'svg').

        Returns:
            Render result with success/failure status.

        """
        output_format = output_format.lower()
        logger.info(
            "Rendering SVG to %s at %dx%d using CairoSVG (%s)",
            output_path,
            width,
            height,
            output_format,
        )

        supported_formats = {"png", "pdf", "ps", "svg"}
        if output_format not in supported_formats:
            return SVGRenderResult(
                success=False,
                output_path=output_path,
                format=output_format,
                width=width,
                height=height,
                error=f"Unsupported render format: {output_format}",
        )

        try:
            import cairosvg  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("CairoSVG is not available: %s", exc)
            return SVGRenderResult(
                success=False,
                output_path=output_path,
                format=output_format,
                width=width,
                height=height,
                error="CairoSVG is not installed.",
            )

        renderers = {
            "png": cairosvg.svg2png,
            "pdf": cairosvg.svg2pdf,
            "ps": cairosvg.svg2ps,
            "svg": cairosvg.svg2svg,
        }
        renderer = renderers[output_format]

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            renderer(
                bytestring=svg_content.encode("utf-8"),
                write_to=str(output_path),
                output_width=width,
                output_height=height,
            )
        except Exception as exc:
            logger.exception("SVG rendering failed for %s", output_path)
            return SVGRenderResult(
                success=False,
                output_path=output_path,
                format=output_format,
                width=width,
                height=height,
                error=str(exc),
            )

        return SVGRenderResult(
            success=True,
            output_path=output_path,
            format=output_format,
            width=width,
            height=height,
        )


# TODO: Add PlaywrightRenderer implementation.
# class PlaywrightRenderer(BaseRenderer):
#     """SVG renderer using Playwright for browser-based rendering."""
#     ...
