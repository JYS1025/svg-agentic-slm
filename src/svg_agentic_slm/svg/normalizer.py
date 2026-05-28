"""SVG normalization utilities.

Provides functions to normalize SVG strings for consistent
comparison and processing (e.g., formatting, attribute ordering,
whitespace normalization).
"""

from __future__ import annotations

import re


def normalize_svg(svg_content: str) -> str:
    """Normalize an SVG string for consistent comparison.

    Performs basic whitespace normalization. More advanced
    normalization (attribute reordering, canonical XML) is
    planned for future implementation.

    Args:
        svg_content: Raw SVG string.

    Returns:
        Normalized SVG string.

    TODO: Implement attribute ordering normalization.
    TODO: Implement canonical XML normalization using lxml.
    TODO: Implement numeric precision normalization.
    """
    # Strip leading/trailing whitespace
    normalized = svg_content.strip()

    # Collapse multiple whitespace characters
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def extract_svg_from_text(text: str) -> str | None:
    """Extract an SVG string from a larger text that may contain
    additional content (e.g., explanations, markdown code fences).

    Args:
        text: Text possibly containing SVG code.

    Returns:
        The extracted SVG string, or None if no SVG was found.

    TODO: Handle markdown code fences (```svg ... ```).
    TODO: Handle multiple SVGs in the same text.
    """
    # Try to extract SVG between <svg and </svg>
    match = re.search(r"(<svg[\s\S]*?</svg>)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None
