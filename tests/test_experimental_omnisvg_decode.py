from __future__ import annotations

import pytest

from svg_agentic_slm.svg.experimental_omnisvg_decode import (
    ExperimentalOmniSVGDecodeError,
    _token_kind,
)


@pytest.mark.parametrize(
    ("token_id", "expected"),
    [
        (151939, "command"),
        (151943, "command"),
        (151944, "coordinate"),
        (191943, "coordinate"),
        (191947, "color"),
        (196044, "color"),
        (196437, "arc"),
        (196536, "arc"),
    ],
)
def test_token_kind_accepts_exact_encoder_producible_boundaries(
    token_id: int,
    expected: str,
) -> None:
    assert _token_kind(token_id) == expected


@pytest.mark.parametrize(
    "reserved_id",
    [
        191944,
        191945,
        191946,
        196045,
        196435,
        196436,
        196537,
        196998,
    ],
)
def test_token_kind_rejects_reserved_training_encoder_gaps(reserved_id: int) -> None:
    with pytest.raises(ExperimentalOmniSVGDecodeError) as caught:
        _token_kind(reserved_id)

    assert caught.value.code == "invalid_inference_token"
