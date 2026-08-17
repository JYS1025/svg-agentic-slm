"""Deterministic renderability gate isolated in a killable process."""

from __future__ import annotations

import io
import multiprocessing as mp
import time
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
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_render_worker, args=(child, svg, self.width, self.height))
        try:
            process.start()
        except BaseException as exc:
            parent.close()
            child.close()
            return self._error("render_failure", f"Renderer process failed to start: {type(exc).__name__}: {exc}")
        child.close()

        deadline = time.monotonic() + self.timeout_seconds
        message_ready = False
        try:
            # Drain the pipe while the child is alive. Joining first can deadlock when
            # a PNG is larger than the OS pipe buffer and the child blocks in send().
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if parent.poll(min(0.05, remaining)):
                    message_ready = True
                    break
                if not process.is_alive():
                    message_ready = parent.poll()
                    break

            if not message_ready:
                if process.is_alive():
                    process.terminate()
                    process.join()
                    return self._error("render_timeout", "SVG smoke render timed out.")
                process.join()
                return self._error("render_failure", "SVG renderer exited without a result.")

            try:
                ok, payload = parent.recv()
            except (EOFError, OSError) as exc:
                if process.is_alive():
                    process.terminate()
                process.join()
                return self._error("render_failure", f"SVG renderer result could not be received: {exc}")

            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join()
                return self._error("render_timeout", "SVG smoke render timed out.")
        finally:
            parent.close()

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
        rendered = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=width,
            output_height=height,
            background_color="#ffffff",
        )
        with Image.open(io.BytesIO(rendered)) as image:
            image.load()
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            output = io.BytesIO()
            white.convert("RGB").save(output, format="PNG")
            png = output.getvalue()
        connection.send((True, png))  # type: ignore[attr-defined]
    except BaseException as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))  # type: ignore[attr-defined]
    finally:
        connection.close()  # type: ignore[attr-defined]


def _cairosvg_version() -> str | None:
    try: return version("CairoSVG")
    except PackageNotFoundError: return None
