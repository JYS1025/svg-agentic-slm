"""Tests for the local Transformers multimodal critic backend."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
from PIL import Image

from svg_agentic_slm.models.transformers_vlm_backend import TransformersVLMBackend


class _FakeProcessor:
    def __init__(self) -> None:
        self.template_kwargs = None
        self.saw_image = False

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        image = messages[0]["content"][0]["image"]
        self.saw_image = isinstance(image, Image.Image)
        return {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        }

    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [42, 43]
        assert kwargs == {"skip_special_tokens": True}
        return '{"status":"pass","issues":[],"preserve":[]}'


class _FakeModel:
    def __init__(self) -> None:
        self.device = None
        self.generate_kwargs = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        suffix = torch.tensor([[42, 43]], dtype=torch.long)
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 3), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_vlm_backend_uses_tokenized_multimodal_chat_template() -> None:
    backend = TransformersVLMBackend(
        model_id="example/gemma",
        model_revision="a" * 40,
        device="cpu",
        dtype="float32",
        auto_model_class="multimodal_lm",
        enable_thinking=False,
    )
    processor = _FakeProcessor()
    model = _FakeModel()
    backend._processor = processor
    backend._model = model
    backend._torch_dtype = torch.float32

    response = backend.generate_with_image(
        "Return critic JSON.",
        _png_bytes(),
        max_new_tokens=16,
        do_sample=False,
    )

    assert processor.saw_image is True
    assert processor.template_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert model.generate_kwargs["pixel_values"].dtype == torch.float32
    assert response.text.startswith('{"status":"pass"')
    assert response.metadata["auto_model_class"] == "multimodal_lm"
    assert response.metadata["enable_thinking"] is False


def test_vlm_backend_selects_multimodal_auto_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    processor = _FakeProcessor()
    model = _FakeModel()
    load_calls = {}

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda model_id, **kwargs: processor,
    )

    def load_model(model_id, **kwargs):
        load_calls["model_id"] = model_id
        load_calls["kwargs"] = kwargs
        return model

    monkeypatch.setattr(
        transformers.AutoModelForMultimodalLM,
        "from_pretrained",
        load_model,
    )

    backend = TransformersVLMBackend(
        model_id="example/gemma",
        model_revision="b" * 40,
        device="cpu",
        dtype="float32",
        auto_model_class="multimodal_lm",
    )
    backend.load_model()

    assert backend.is_loaded()
    assert model.device == "cpu"
    assert load_calls["model_id"] == "example/gemma"
    assert load_calls["kwargs"]["revision"] == "b" * 40
