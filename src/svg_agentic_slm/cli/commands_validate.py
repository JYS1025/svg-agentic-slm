"""CLI command for SVG validation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from svg_agentic_slm.svg.validator import SVGValidator

console = Console()


def validate(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the SVG file to validate.",
        exists=True,
    ),
) -> None:
    """Validate an SVG file.

    Runs the SVG validator on the specified file and prints
    the validation results.
    """
    svg_content = input_file.read_text(encoding="utf-8")

    validator = SVGValidator()
    result = validator.validate(svg_content)

    if result.is_valid:
        console.print("[bold green]✓ SVG is valid.[/bold green]")
    else:
        console.print("[bold red]✗ SVG validation failed.[/bold red]")

    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  • {error}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")

    console.print(f"\nHas <svg> tag: {result.has_svg_tag}")
    console.print(f"Has </svg> tag: {result.has_closing_tag}")
    console.print(f"Well-formed XML: {result.is_well_formed_xml}")
