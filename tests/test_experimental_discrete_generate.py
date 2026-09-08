from __future__ import annotations

from svg_agentic_slm.svg.experimental_discrete_generate import _parser


def test_experimental_generation_baseline_has_no_repetition_penalty() -> None:
    arguments = _parser().parse_args(
        [
            "--base-model",
            "base",
            "--adapter-dir",
            "adapter",
            "--tokenizer-dir",
            "tokenizer",
            "--output-dir",
            "output",
        ]
    )

    assert arguments.do_sample is False
    assert arguments.repetition_penalty == 1.0
