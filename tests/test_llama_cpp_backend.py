"""Tests for the local llama.cpp GGUF backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.llama_cpp_backend import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    LlamaCppModelBackend,
)


class _FakeLlamaClient:
    def __init__(self, **kwargs) -> None:
        self.load_kwargs = kwargs
        self.call_kwargs = None

    def create_chat_completion(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "choices": [
                {
                    "message": {"content": "<svg></svg>"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
            },
        }


class _FakeStreamingLlamaClient(_FakeLlamaClient):
    n_tokens = 20

    def create_chat_completion(self, **kwargs):
        self.call_kwargs = kwargs
        assert kwargs["stream"] is True
        return iter(
            [
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "<svg>"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "</svg>"}, "finish_reason": "stop"}]},
            ]
        )

    def tokenize(self, value: bytes, **kwargs):
        assert value == b"<svg></svg>"
        return [1, 2, 3, 4]


def test_llama_cpp_backend_loads_with_full_gpu_offload(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    clients = []

    def client_factory(**kwargs):
        client = _FakeLlamaClient(**kwargs)
        clients.append(client)
        return client

    backend = LlamaCppModelBackend(
        model_path=model_path,
        model_revision="revision",
        n_ctx=8192,
        n_gpu_layers=-1,
        main_gpu=0,
        split_mode="none",
        client_factory=client_factory,
    )

    backend.load_model()
    response = backend.generate(
        "Draw a circle.",
        system_prompt="Return SVG.",
        max_new_tokens=64,
        do_sample=False,
    )

    assert backend.is_loaded()
    assert clients[0].load_kwargs["n_gpu_layers"] == -1
    assert clients[0].load_kwargs["n_ctx"] == 8192
    assert clients[0].load_kwargs["main_gpu"] == 0
    assert clients[0].load_kwargs["split_mode"] == 0
    assert clients[0].call_kwargs["temperature"] == 0.0
    assert clients[0].call_kwargs["messages"][0]["role"] == "system"
    assert response.text == "<svg></svg>"
    assert response.prompt_tokens == 12
    assert response.metadata["quantization"] == "Q4_0"
    assert response.metadata["quantization_provider"] == "LM Studio Community"
    assert response.metadata["upstream_model_id"] == ("google/gemma-4-12B-it-qat-q4_0-unquantized")
    assert response.metadata["main_gpu"] == 0
    assert response.metadata["split_mode"] == "none"


def test_llama_cpp_backend_pins_download_revision(tmp_path: Path) -> None:
    model_path = tmp_path / "downloaded.gguf"
    model_path.write_bytes(b"fake")
    download_kwargs = {}

    def download_resolver(**kwargs):
        download_kwargs.update(kwargs)
        return str(model_path)

    backend = LlamaCppModelBackend(
        model_revision="immutable-revision",
        client_factory=lambda **kwargs: _FakeLlamaClient(**kwargs),
        download_resolver=download_resolver,
    )
    backend.load_model()

    assert download_kwargs["revision"] == "immutable-revision"
    assert download_kwargs["filename"].endswith(".gguf")


def test_llama_cpp_backend_defaults_to_compatibility_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "downloaded.gguf"
    model_path.write_bytes(b"fake")
    download_kwargs = {}

    def download_resolver(**kwargs):
        download_kwargs.update(kwargs)
        return str(model_path)

    backend = LlamaCppModelBackend(
        client_factory=lambda **kwargs: _FakeLlamaClient(**kwargs),
        download_resolver=download_resolver,
    )
    backend.load_model()

    assert download_kwargs == {
        "repo_id": DEFAULT_MODEL_ID,
        "filename": DEFAULT_MODEL_FILE,
        "revision": DEFAULT_MODEL_REVISION,
    }


def test_custom_distribution_does_not_inherit_default_provenance() -> None:
    backend = LlamaCppModelBackend(
        model_id="example/custom-gguf",
        filename="custom.gguf",
        model_path="/tmp/custom.gguf",
    )

    assert backend.model_revision is None
    assert backend.upstream_model_id is None
    assert backend.quantization is None
    assert backend.quantization_provider is None
    assert backend.conversion_runtime is None


def test_llama_cpp_backend_rejects_unknown_generation_option(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    backend = LlamaCppModelBackend(
        model_path=model_path,
        generation_config=GenerationConfig(),
        client_factory=lambda **kwargs: _FakeLlamaClient(**kwargs),
    )
    backend.load_model()

    with pytest.raises(ValueError, match="Unsupported"):
        backend.generate("Draw.", unsupported=True)


def test_llama_cpp_backend_rejects_invalid_generation_override(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    backend = LlamaCppModelBackend(
        model_path=model_path,
        client_factory=lambda **kwargs: _FakeLlamaClient(**kwargs),
    )
    backend.load_model()

    with pytest.raises(ValueError, match="max_new_tokens"):
        backend.generate("Draw.", max_new_tokens=0)


def test_llama_cpp_backend_requires_explicit_load() -> None:
    backend = LlamaCppModelBackend(model_path="/tmp/not-loaded.gguf")

    with pytest.raises(RuntimeError, match="not loaded"):
        backend.generate("Draw.")


def test_llama_cpp_backend_records_streaming_ttft_and_throughput(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    backend = LlamaCppModelBackend(
        model_path=model_path,
        measure_streaming_metrics=True,
        client_factory=lambda **kwargs: _FakeStreamingLlamaClient(**kwargs),
    )
    backend.load_model()

    response = backend.generate("Draw.")

    assert response.text == "<svg></svg>"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 16
    assert response.completion_tokens == 4
    assert response.time_to_first_token_seconds is not None
    assert response.tokens_per_second is not None
    assert response.metadata["streaming_metrics_enabled"] is True


def test_native_client_is_imported_before_download_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake")
    events: list[str] = []

    def import_client():
        events.append("native_client")
        return _FakeLlamaClient

    def download_resolver(**kwargs):
        events.append("download")
        return str(model_path)

    backend = LlamaCppModelBackend(
        download_resolver=download_resolver,
    )
    monkeypatch.setattr(backend, "_import_llama_client", import_client)

    backend.load_model()

    assert events == ["native_client", "download"]
