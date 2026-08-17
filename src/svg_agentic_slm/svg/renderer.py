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
from svg_agentic_slm.svg.validator import SVGValidator
from svg_agentic_slm.utils.atomic import atomic_output_path

logger = logging.getLogger(__name__)


class CairoSVGRenderer(BaseRenderer):
    """SVG renderer using CairoSVG.

    Converts SVG strings to PNG (or other raster formats) using
    the CairoSVG library.
    """

    def render_bytes(
        self,
        svg_content: str,
        output_width: int = 512,
        output_height: int = 512,
        background_color: str = "#ffffff",
    ) -> bytes:
        """Validate and render an SVG to in-memory PNG bytes."""
        if output_width <= 0 or output_height <= 0:
            raise ValueError("Render dimensions must be positive.")

        validation = SVGValidator().validate(svg_content)
        if not validation.is_valid:
            details = "; ".join(validation.errors) or "unknown validation error"
            raise ValueError(f"Unsafe or invalid SVG: {details}")

        try:
            import cairosvg  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise RuntimeError("CairoSVG is not installed.") from exc

        try:
            rendered = cairosvg.svg2png(
                bytestring=svg_content.encode("utf-8"),
                output_width=output_width,
                output_height=output_height,
                background_color=background_color,
            )
        except Exception as exc:
            raise RuntimeError("In-memory SVG rendering failed.") from exc
        if not isinstance(rendered, (bytes, bytearray)):
            raise RuntimeError("CairoSVG did not return PNG bytes.")
        return bytes(rendered)

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

        validation = SVGValidator().validate(svg_content)
        if not validation.is_valid:
            details = "; ".join(validation.errors) or "unknown validation error"
            return SVGRenderResult(
                success=False,
                output_path=output_path,
                format=output_format,
                width=width,
                height=height,
                error=f"Unsafe or invalid SVG: {details}",
            )

        try:
            import cairosvg  # type: ignore[import-not-found,import-untyped]
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
            with atomic_output_path(output_path) as temporary_path:
                renderer(
                    bytestring=svg_content.encode("utf-8"),
                    write_to=str(temporary_path),
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
