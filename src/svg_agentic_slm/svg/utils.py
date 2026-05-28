"""General SVG utility functions.

Low-level helpers for working with SVG strings that don't fit
into validation, rendering, or normalization categories.
"""

from __future__ import annotations


def wrap_svg_boilerplate(
    inner_content: str,
    width: int = 256,
    height: int = 256,
) -> str:
    """Wrap SVG inner content with standard boilerplate.

    Args:
        inner_content: SVG elements to place inside the <svg> tag.
        width: SVG canvas width.
        height: SVG canvas height.

    Returns:
        Complete SVG string with root element and namespace.
    """
    return (
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"{inner_content}"
        f"</svg>"
    )


def estimate_svg_complexity(svg_content: str) -> int:
    """Estimate the complexity of an SVG by counting elements.

    This is a rough heuristic, not a precise measure.

    Args:
        svg_content: SVG string to analyze.

    Returns:
        Estimated number of SVG elements.

    TODO: Implement more sophisticated complexity metrics
          (e.g., path data complexity, nesting depth).
    """
    # Simple count of self-closing and opening tags
    import re

    tags = re.findall(r"<(\w+)[\s/>]", svg_content)
    return len(tags)
