"""Deterministic renderability gate isolated in a killable process."""

from __future__ import annotations

import io
import multiprocessing as mp
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

from PIL import Image

from svg_agentic_slm.svg.schemas import SVGDiagnostic


@dataclass(frozen=True)
class SmokeRenderResult:
    success: bool
    png: bytes = b""
    diagnostics: list[SVGDiagnostic] = field(default_factory=list)
    renderer_version: str | None = None


class SmokeRenderGate:
    def __init__(self, width: int = 256, height: int = 256, timeout_seconds: float = 10.0,
                 max_svg_bytes: int = 1_048_576, max_png_bytes: int = 16_777_216) -> None:
        self.width, self.height = width, height
        self.timeout_seconds = timeout_seconds
        self.max_svg_bytes, self.max_png_bytes = max_svg_bytes, max_png_bytes

    def evaluate(self, svg: str) -> SmokeRenderResult:
        if len(svg.encode("utf-8")) > self.max_svg_bytes:
            return self._error("render_output_error", "SVG exceeds smoke-render input limit.")
        parent, child = mp.Pipe(duplex=False)
        process = mp.Process(target=_render_worker, args=(child, svg, self.width, self.height))
        process.start(); child.close(); process.join(self.timeout_seconds)
        if process.is_alive():
            process.terminate(); process.join()
            return self._error("render_timeout", "SVG smoke render timed out.")
        if not parent.poll():
            return self._error("render_failure", "SVG renderer exited without a result.")
        ok, payload = parent.recv()
        if not ok:
            return self._error("render_failure", str(payload))
        png = bytes(payload)
        if len(png) > self.max_png_bytes or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            return self._error("render_output_error", "Renderer did not produce an acceptable PNG.")
        try:
            with Image.open(io.BytesIO(png)) as image:
                image.load()
                if image.size != (self.width, self.height):
                    return self._error("render_output_error", f"Unexpected PNG dimensions: {image.size}.")
        except Exception as exc:
            return self._error("render_output_error", f"PNG decode failed: {exc}")
        return SmokeRenderResult(True, png, renderer_version=_cairosvg_version())

    def _error(self, code: str, message: str) -> SmokeRenderResult:
        return SmokeRenderResult(False, diagnostics=[SVGDiagnostic(code, message)])  # type: ignore[arg-type]


def _render_worker(connection: object, svg: str, width: int, height: int) -> None:
    try:
        import cairosvg
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, output_height=height)
        connection.send((True, png))  # type: ignore[attr-defined]
    except BaseException as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))  # type: ignore[attr-defined]
    finally:
        connection.close()  # type: ignore[attr-defined]


def _cairosvg_version() -> str | None:
    try: return version("CairoSVG")
    except PackageNotFoundError: return None
