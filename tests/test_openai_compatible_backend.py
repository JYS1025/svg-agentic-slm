"""Tests for the external OpenAI-compatible model backend."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from svg_agentic_slm.factories.generation import (
    build_generation_runtime,
    persist_generation_artifacts,
)
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.openai_compatible_backend import OpenAICompatibleBackend


class _Response:
    def __init__(self, payload: object) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return self._data[:amount]


class _RecordingURLopener:
    def __init__(self, payloads: list[object]) -> None:
        self._payloads = iter(payloads)
        self.calls: list[tuple[Request, float]] = []

    def __call__(self, request: Request, *, timeout: float) -> _Response:
        self.calls.append((request, timeout))
        return _Response(next(self._payloads))


class _FailingURLopener:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: Request, *, timeout: float) -> _Response:
        self.calls += 1
        raise URLError("connection failed")


def test_backend_verifies_model_and_normalizes_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingURLopener(
        [
            {"data": [{"id": "svg-model@abc"}]},
            {
                "model": "svg-model@abc",
                "choices": [
                    {
                        "message": {"content": "<svg/>"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        ]
    )
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        opener,
    )
    monkeypatch.setenv("MODEL_API_KEY", "test-secret")
    backend = OpenAICompatibleBackend(
        base_url="http://127.0.0.1:8000/v1/",
        model_id="svg-model@abc",
        model_revision="abc",
        api_key_env="MODEL_API_KEY",
        engine="vllm",
        generation_config=GenerationConfig(temperature=0.25, top_k=20),
    )

    backend.load_model()
    result = backend.generate("Draw.", system_prompt="Return SVG.")

    assert backend.is_loaded()
    assert result.text == "<svg/>"
    assert result.model_id == "svg-model@abc"
    assert result.model_revision == "abc"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.metadata["engine"] == "vllm"
    assert "test-secret" not in repr(result)
    assert opener.calls[0][0].full_url.endswith("/v1/models")
    assert opener.calls[1][0].full_url.endswith("/v1/chat/completions")
    assert opener.calls[1][0].get_header("Authorization") == "Bearer test-secret"
    request_payload = json.loads(opener.calls[1][0].data or b"{}")
    assert request_payload["model"] == "svg-model@abc"
    assert request_payload["top_k"] == 20
    assert request_payload["repetition_penalty"] == 1.1
    assert request_payload["messages"][0]["role"] == "system"


def test_llama_cpp_dialect_uses_repeat_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingURLopener(
        [
            {"data": [{"id": "local-model"}]},
            {
                "model": "local-model",
                "choices": [{"message": {"content": "<svg/>"}}],
            },
        ]
    )
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        opener,
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8080/v1",
        model_id="local-model",
        engine="llama_cpp",
    )

    backend.load_model()
    backend.generate("Draw.")

    request_payload = json.loads(opener.calls[1][0].data or b"{}")
    assert request_payload["repeat_penalty"] == 1.1
    assert "repetition_penalty" not in request_payload


@pytest.mark.parametrize(
    ("engine", "token_response", "expected_payload"),
    [
        (
            "vllm",
            {"count": 3},
            {"model": "local-model", "prompt": "token budget"},
        ),
        (
            "llama_cpp",
            {"tokens": [1, 2, 3]},
            {"content": "token budget", "add_special": True},
        ),
    ],
)
def test_backend_counts_tokens_with_served_model_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    token_response: dict[str, object],
    expected_payload: dict[str, object],
) -> None:
    opener = _RecordingURLopener(
        [{"data": [{"id": "local-model"}]}, token_response]
    )
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        opener,
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8080/v1",
        model_id="local-model",
        engine=engine,
    )

    backend.load_model()
    count = backend.count_tokens("token budget")

    assert count == 3
    assert opener.calls[1][0].full_url.endswith("/v1/tokenize")
    assert json.loads(opener.calls[1][0].data or b"{}") == expected_payload


def test_backend_rejects_unserved_or_mismatched_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unserved = _RecordingURLopener([{"data": [{"id": "other-model"}]}])
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        unserved,
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model_id="expected-model",
        engine="vllm",
    )

    with pytest.raises(RuntimeError, match="is not served"):
        backend.load_model()

    mismatched = _RecordingURLopener(
        [
            {"data": [{"id": "expected-model"}]},
            {
                "model": "other-model",
                "choices": [{"message": {"content": "<svg/>"}}],
            },
        ]
    )
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        mismatched,
    )
    backend.load_model()
    with pytest.raises(RuntimeError, match="does not match"):
        backend.generate("Draw.")


def test_backend_requires_response_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingURLopener(
        [
            {"data": [{"id": "expected-model"}]},
            {"choices": [{"message": {"content": "<svg/>"}}]},
        ]
    )
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        opener,
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model_id="expected-model",
        engine="vllm",
    )

    backend.load_model()
    with pytest.raises(RuntimeError, match="does not match"):
        backend.generate("Draw.")


def test_backend_retries_readiness_but_not_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = _FailingURLopener()
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        failing,
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model_id="model",
        engine="vllm",
        max_retries=2,
    )

    with pytest.raises(RuntimeError, match="request failed"):
        backend.load_model()
    assert failing.calls == 3

    backend._ready = True
    failing.calls = 0
    with pytest.raises(RuntimeError, match="request failed"):
        backend.generate("Draw.")
    assert failing.calls == 1


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com/v1",
        "https://user:password@example.com/v1",
        "file:///tmp/model-server",
    ],
)
def test_backend_rejects_unsafe_endpoint_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleBackend(
            base_url=base_url,
            model_id="model",
            engine="vllm",
        )


def test_backend_requires_configured_api_key_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_MODEL_KEY", raising=False)
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model_id="model",
        api_key_env="MISSING_MODEL_KEY",
        engine="vllm",
    )

    with pytest.raises(RuntimeError, match="MISSING_MODEL_KEY"):
        backend.load_model()


def test_backend_rejects_unknown_generation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _RecordingURLopener([{"data": [{"id": "model"}]}])
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        opener,
    )
    backend = OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
        model_id="model",
        engine="vllm",
    )
    backend.load_model()

    with pytest.raises(ValueError, match="unsupported_option"):
        backend.generate("Draw.", unsupported_option=True)


def test_openai_profile_runs_through_pipeline_and_artifact_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "integration-model@abc"
    opener = _RecordingURLopener(
        [
            {"data": [{"id": model_id}]},
            {
                "model": model_id,
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<svg xmlns="http://www.w3.org/2000/svg">'
                                "<circle cx=\"8\" cy=\"8\" r=\"4\"/>"
                                "</svg>"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9},
            },
        ]
    )
    monkeypatch.setattr(
        "svg_agentic_slm.models.openai_compatible_backend._URL_OPENER.open",
        opener,
    )
    output_dir = tmp_path / "outputs"
    configs = {
        "generation.yaml": {
            "generation": {
                "max_new_tokens": 64,
                "do_sample": False,
                "render": {"enabled": False},
            }
        },
        "model.yaml": {"model": {"backend_type": "llama_cpp"}},
        "rag.yaml": {"rag": {}},
        "paths.yaml": {
            "paths": {
                "outputs": {
                    "generations": str(output_dir),
                    "renders": str(tmp_path / "renders"),
                }
            }
        },
        "profile.yaml": {
            "model": {
                "backend_type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "model_id": model_id,
                "revision": "abc",
                "engine": "vllm",
            },
            "critic_model": {
                "backend_type": "openai_compatible",
                "base_url": "http://localhost:8001/v1",
                "model_id": "unused-critic@abc",
                "revision": "abc",
                "engine": "vllm",
            },
        },
    }
    for filename, payload in configs.items():
        (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")

    runtime = build_generation_runtime(
        config_path=tmp_path / "generation.yaml",
        model_config_path=tmp_path / "profile.yaml",
        prompt="Draw a circle.",
    )
    result = runtime.orchestrator.run(runtime.request)
    artifacts = persist_generation_artifacts(result=result, runtime=runtime)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert result.is_valid
    assert result.attempts[0].model_calls[0].response.model_id == model_id
    assert result.attempts[0].model_calls[0].response.metadata["engine"] == "vllm"
    assert metadata["runtime"]["model_config"]["model_id"] == model_id
    assert metadata["runtime"]["critic_model_config"] is None
