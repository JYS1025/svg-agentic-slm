from __future__ import annotations

from svg_agentic_slm.data.mmsvg_sft import _first_text


def test_first_text_joins_mmsvg_sentence_lists_without_python_repr_markup() -> None:
    row = {
        "detail": [
            "A blue circle is centered on the canvas.",
            "A thin white ring surrounds it.",
        ]
    }

    assert _first_text(row, ("detail",)) == (
        "A blue circle is centered on the canvas. "
        "A thin white ring surrounds it."
    )


def test_first_text_skips_empty_candidates_and_decodes_sequence_bytes() -> None:
    row = {
        "detail": [],
        "detailed_description": [b"first sentence", None, "second sentence"],
    }

    assert _first_text(row, ("detail", "detailed_description")) == (
        "first sentence second sentence"
    )
