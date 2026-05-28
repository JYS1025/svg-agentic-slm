"""SVG diff utilities.

Provides functions to compute and display differences between
two SVG strings, useful for debugging revisions and evaluating
changes made by critic feedback.
"""

from __future__ import annotations

import difflib


def compute_svg_diff(svg_a: str, svg_b: str) -> str:
    """Compute a unified diff between two SVG strings.

    Args:
        svg_a: The first (original) SVG string.
        svg_b: The second (revised) SVG string.

    Returns:
        A unified diff string.

    TODO: Add structured diff that understands SVG elements.
    TODO: Add semantic diff (e.g., detect color changes, shape additions).
    """
    lines_a = svg_a.splitlines(keepends=True)
    lines_b = svg_b.splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile="original.svg",
        tofile="revised.svg",
    )
    return "".join(diff)
