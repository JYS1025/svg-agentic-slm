"""Network-free tests for the pinned SigLIP2 similarity scorer."""

from __future__ import annotations

import hashlib
import math
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
import transformers
from PIL import Image

from svg_agentic_slm.models.image_text_similarity import (
    SIGLIP2_PAIR_PROBABILITY,
)
from svg_agentic_slm.models.siglip2_similarity import (
    DEFAULT_SIGLIP2_MODEL_REVISION,
    Siglip2SimilarityScorer,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        }


class _FakeModel:
    def __init__(self, logit: float = 1.0) -> None:
        self.logit = logit
        self.device: str | None = None
        self.inputs: dict[str, torch.Tensor] | None = None

    def eval(self):
        return self

    def to(self, device: str):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return SimpleNamespace(
            logits_per_image=torch.tensor([[self.logit]], dtype=torch.float32)
        )


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 3), "white").save(output, format="PNG")
    return output.getvalue()


def test_siglip2_scores_pair_with_pinned_model_and_processor_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _FakeProcessor()
    model = _FakeModel(logit=1.0)
    processor_load: dict[str, object] = {}
    model_load: dict[str, object] = {}

    def load_processor(model_id: str, **kwargs):
        processor_load.update(model_id=model_id, **kwargs)
        return processor

    def load_model(model_id: str, **kwargs):
        model_load.update(model_id=model_id, **kwargs)
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)

    scorer = Siglip2SimilarityScorer(device="cpu", dtype="float32")
    scorer.load_model()
    png = _png_bytes()
    evidence = scorer.score("Two hands hold together", png, attempt_id="attempt-1")

    expected_hub_kwargs = {
        "revision": DEFAULT_SIGLIP2_MODEL_REVISION,
        "local_files_only": False,
        "trust_remote_code": False,
    }
    assert processor_load == {
        "model_id": "google/siglip2-base-patch16-224",
        **expected_hub_kwargs,
    }
    assert model_load == {
        "model_id": "google/siglip2-base-patch16-224",
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
        **expected_hub_kwargs,
    }
    assert model.device == "cpu"
    assert model.inputs is not None
    assert model.inputs["input_ids"].dtype == torch.long
    assert model.inputs["pixel_values"].dtype == torch.float32
    assert processor.kwargs is not None
    assert processor.kwargs["text"] == [
        "This is a photo of Two hands hold together."
    ]
    assert processor.kwargs["padding"] == "max_length"
    assert processor.kwargs["truncation"] is True
    assert processor.kwargs["max_length"] == 64
    assert processor.kwargs["return_tensors"] == "pt"
    assert isinstance(processor.kwargs["images"][0], Image.Image)
    assert evidence.metric == SIGLIP2_PAIR_PROBABILITY
    assert evidence.raw_logit == pytest.approx(1.0)
    assert evidence.score == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert evidence.image_sha256 == hashlib.sha256(png).hexdigest()
    assert evidence.model_revision == DEFAULT_SIGLIP2_MODEL_REVISION
    assert evidence.attempt_id == "attempt-1"
    assert evidence.latency_seconds >= 0.0

    scorer.unload_model()
    assert scorer.is_loaded() is False


def test_siglip2_requires_loaded_model_and_valid_png() -> None:
    scorer = Siglip2SimilarityScorer(device="cpu", dtype="float32")

    with pytest.raises(ValueError, match="PNG bytes"):
        scorer.score("Draw a circle", b"not-png", attempt_id="attempt-1")
    with pytest.raises(RuntimeError, match="not loaded"):
        scorer.score("Draw a circle", _png_bytes(), attempt_id="attempt-1")


@pytest.mark.parametrize(
    "template",
    ["No placeholder", "{instruction} {instruction}", "{other}"],
)
def test_siglip2_rejects_ambiguous_text_templates(template: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Siglip2SimilarityScorer(text_template=template)
