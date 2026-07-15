"""CLI command for SVG rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from svg_agentic_slm.svg.renderer import CairoSVGRenderer

console = Console()


def render(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the SVG file to render.",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for the rendered image.",
    ),
    width: int = typer.Option(
        256,
        "--width", "-w",
        help="Output width in pixels.",
    ),
    height: int = typer.Option(
        256,
        "--height", "-h",
        help="Output height in pixels.",
    ),
    backend: str = typer.Option(
        "cairosvg",
        "--backend", "-b",
        help="Rendering backend (cairosvg, playwright).",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--format", "-f",
        help="Explicit output format override (png, pdf, ps, svg).",
    ),
) -> None:
    """Render an SVG file to a raster image."""
    resolved_format = (output_format or (output.suffix.lstrip(".") if output else "png")).lower()
    if output is None:
        output = input_file.with_suffix(f".{resolved_format}")
    elif output.suffix and output.suffix.lower() != f".{resolved_format}":
        console.print(
            "[bold red]Output extension "
            f"'{output.suffix}' does not match --format '{resolved_format}'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if not output.suffix:
        output = output.with_suffix(f".{resolved_format}")

    console.print(f"[bold blue]SVG Rendering[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output}")
    console.print(f"Size: {width}x{height}")
    console.print(f"Backend: {backend}")
    console.print(f"Format: {resolved_format}")

    svg_content = input_file.read_text(encoding="utf-8")

    if backend != "cairosvg":
        console.print(f"[bold red]Unsupported backend: {backend}[/bold red]")
        raise typer.Exit(code=1)

    renderer = CairoSVGRenderer()
    result = renderer.render(
        svg_content,
        output,
        width=width,
        height=height,
        output_format=resolved_format,
    )

    if not result.success:
        console.print(f"\n[bold red]Rendering failed: {result.error}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"\n[green]Rendered output saved to: {result.output_path}[/green]")
