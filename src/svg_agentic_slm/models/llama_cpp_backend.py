"""llama.cpp backend for local GGUF inference."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from svg_agentic_slm.models.base import BaseModelBackend
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.schemas import ModelResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "lmstudio-community/gemma-4-12B-it-QAT-GGUF"
DEFAULT_MODEL_FILE = "gemma-4-12B-it-QAT-Q4_0.gguf"
DEFAULT_MODEL_REVISION = "291406f49e16eff811c85ad8884d375f34138663"
DEFAULT_UPSTREAM_MODEL_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized"
DEFAULT_QUANTIZATION = "Q4_0"
DEFAULT_QUANTIZATION_PROVIDER = "LM Studio Community"
DEFAULT_CONVERSION_RUNTIME = "llama.cpp b9518"


class LlamaCppModelBackend(BaseModelBackend):
    """Run a GGUF model through llama-cpp-python.

    The backend owns model download, chat-template application, and conversion
    of llama.cpp's response into the shared :class:`ModelResponse` contract.
    Agent code remains independent of GGUF and llama.cpp.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        filename: str = DEFAULT_MODEL_FILE,
        model_revision: str | None = None,
        model_path: str | Path | None = None,
        upstream_model_id: str | None = None,
        quantization: str | None = None,
        quantization_provider: str | None = None,
        conversion_runtime: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        flash_attn: bool = True,
        use_mmap: bool = True,
        verbose: bool = False,
        chat_format: str | None = None,
        measure_streaming_metrics: bool = False,
        generation_config: GenerationConfig | None = None,
        client_factory: Callable[..., Any] | None = None,
        download_resolver: Callable[..., str] | None = None,
    ) -> None:
        if n_ctx <= 0:
            raise ValueError("n_ctx must be positive.")
        if n_batch <= 0:
            raise ValueError("n_batch must be positive.")

        uses_default_distribution = model_id == DEFAULT_MODEL_ID and filename == DEFAULT_MODEL_FILE
        self.model_id = model_id
        self.filename = filename
        self.model_revision = model_revision or (
            DEFAULT_MODEL_REVISION if uses_default_distribution else None
        )
        self.model_path = Path(model_path) if model_path else None
        self.upstream_model_id = upstream_model_id or (
            DEFAULT_UPSTREAM_MODEL_ID if uses_default_distribution else None
        )
        self.quantization = quantization or (
            DEFAULT_QUANTIZATION if uses_default_distribution else None
        )
        self.quantization_provider = quantization_provider or (
            DEFAULT_QUANTIZATION_PROVIDER if uses_default_distribution else None
        )
        self.conversion_runtime = conversion_runtime or (
            DEFAULT_CONVERSION_RUNTIME if uses_default_distribution else None
        )
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.flash_attn = flash_attn
        self.use_mmap = use_mmap
        self.verbose = verbose
        self.chat_format = chat_format
        self.measure_streaming_metrics = measure_streaming_metrics
        self.generation_config = generation_config or GenerationConfig()
        self._client_factory = client_factory
        self._download_resolver = download_resolver
        self._model: Any = None
        self._resolved_model_path: Path | None = None

    def load_model(self) -> None:
        """Download, if needed, and load the configured GGUF model."""
        if self.is_loaded():
            return

        client_factory = self._client_factory or self._import_llama_client()
        # Load llama.cpp's native runtime before huggingface_hub. In some Conda
        # environments the downloader imports NumPy, whose bundled libstdc++ is
        # older than the CUDA-enabled llama.cpp build requires.
        model_path = self.model_path or self._download_model()
        client_kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "flash_attn": self.flash_attn,
            "use_mmap": self.use_mmap,
            "verbose": self.verbose,
        }
        if self.chat_format is not None:
            client_kwargs["chat_format"] = self.chat_format

        logger.info(
            "Loading GGUF model '%s' with n_gpu_layers=%d and n_ctx=%d.",
            self.model_id,
            self.n_gpu_layers,
            self.n_ctx,
        )
        try:
            self._model = client_factory(**client_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GGUF model '{self.model_id}' from '{model_path}'."
            ) from exc
        self._resolved_model_path = Path(model_path)

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a chat completion using GGUF metadata's chat template."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load_model() before generate().")

        system_prompt = kwargs.pop("system_prompt", None)
        stop = kwargs.pop("stop", None)
        config = self._resolve_generation_config(kwargs)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": config["max_new_tokens"],
            "temperature": config["temperature"] if config["do_sample"] else 0.0,
            "top_p": config["top_p"] if config["do_sample"] else 1.0,
            "top_k": config["top_k"],
            "repeat_penalty": config["repetition_penalty"],
        }
        if config.get("seed") is not None:
            call_kwargs["seed"] = config["seed"]
        if stop is not None:
            call_kwargs["stop"] = stop

        if self.measure_streaming_metrics:
            completion = self._generate_streaming(call_kwargs)
        else:
            completion = self._generate_buffered(call_kwargs)

        return ModelResponse(
            text=completion["text"],
            model_id=self.model_id,
            model_revision=self.model_revision,
            finish_reason=completion["finish_reason"],
            prompt_tokens=completion["prompt_tokens"],
            completion_tokens=completion["completion_tokens"],
            latency_seconds=completion["latency_seconds"],
            time_to_first_token_seconds=completion["time_to_first_token_seconds"],
            tokens_per_second=completion["tokens_per_second"],
            metadata={
                "backend": "llama_cpp",
                "backend_version": _package_version("llama-cpp-python"),
                "model_file": self.filename,
                "resolved_model_path": str(self._resolved_model_path),
                "upstream_model_id": self.upstream_model_id,
                "quantization": self.quantization,
                "quantization_provider": self.quantization_provider,
                "conversion_runtime": self.conversion_runtime,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "n_batch": self.n_batch,
                "flash_attn": self.flash_attn,
                "streaming_metrics_enabled": self.measure_streaming_metrics,
            },
        )

    def _generate_buffered(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        try:
            payload = self._model.create_chat_completion(**call_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Generation failed for model '{self.model_id}'.") from exc
        latency_seconds = time.perf_counter() - started_at

        try:
            choice = payload["choices"][0]
            text = str(choice["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama.cpp returned an unexpected completion payload.") from exc

        usage = payload.get("usage", {})
        completion_tokens = _optional_int(usage.get("completion_tokens"))
        return {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
            "completion_tokens": completion_tokens,
            "latency_seconds": latency_seconds,
            "time_to_first_token_seconds": None,
            "tokens_per_second": (
                completion_tokens / latency_seconds
                if completion_tokens is not None and latency_seconds > 0
                else None
            ),
        }

    def _generate_streaming(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        try:
            payload = self._model.create_chat_completion(**call_kwargs, stream=True)
            if isinstance(payload, Mapping):
                raise TypeError("llama.cpp returned a buffered payload for a streaming request.")
            chunks = _consume_chat_chunks(payload, started_at=started_at)
        except Exception as exc:
            raise RuntimeError(f"Generation failed for model '{self.model_id}'.") from exc

        completion_tokens = self._count_tokens(chunks["text"])
        total_tokens = getattr(self._model, "n_tokens", None)
        prompt_tokens = (
            max(0, int(total_tokens) - completion_tokens)
            if isinstance(total_tokens, int)
            else None
        )
        generation_seconds = chunks["latency_seconds"] - (
            chunks["time_to_first_token_seconds"] or 0.0
        )
        return {
            **chunks,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": (
                completion_tokens / generation_seconds if generation_seconds > 0 else None
            ),
        }

    def _count_tokens(self, text: str) -> int:
        tokenize = getattr(self._model, "tokenize", None)
        if not callable(tokenize):
            return 0
        tokens = tokenize(text.encode("utf-8"), add_bos=False, special=True)
        return len(tokens)

    def is_loaded(self) -> bool:
        """Return whether the llama.cpp client is ready."""
        return self._model is not None

    def unload_model(self) -> None:
        """Release the llama.cpp client and its CPU/GPU allocations."""
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
        self._model = None
        self._resolved_model_path = None
        gc.collect()
        logger.info("GGUF model unloaded.")

    def _download_model(self) -> Path:
        resolver = self._download_resolver
        if resolver is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface-hub is required to download the configured GGUF model."
                ) from exc
            resolver = hf_hub_download

        download_kwargs: dict[str, Any] = {
            "repo_id": self.model_id,
            "filename": self.filename,
        }
        if self.model_revision is not None:
            download_kwargs["revision"] = self.model_revision
        try:
            return Path(resolver(**download_kwargs))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{self.filename}' from '{self.model_id}'."
            ) from exc

    @staticmethod
    def _import_llama_client() -> Callable[..., Any]:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python with CUDA support is required for the local GGUF "
                "backend. Install the project's local-gpu optional dependencies."
            ) from exc
        return cast(Callable[..., Any], Llama)

    def _resolve_generation_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        supported = set(self.generation_config.to_dict())
        unknown = set(overrides) - supported
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported llama.cpp generation option(s): {names}")

        merged = {**self.generation_config.to_dict(), **overrides}
        return GenerationConfig.from_dict(merged).to_dict()


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _consume_chat_chunks(
    payload: Iterable[Mapping[str, Any]],
    *,
    started_at: float,
) -> dict[str, Any]:
    text_parts: list[str] = []
    finish_reason: str | None = None
    first_token_at: float | None = None
    for chunk in payload:
        try:
            choice = chunk["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama.cpp returned an unexpected streaming payload.") from exc
        delta = choice.get("delta", {})
        content = delta.get("content") if isinstance(delta, Mapping) else None
        if content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text_parts.append(str(content))
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])

    finished_at = time.perf_counter()
    text = "".join(text_parts)
    if not text:
        raise RuntimeError("llama.cpp streaming completion did not contain text.")
    return {
        "text": text,
        "finish_reason": finish_reason,
        "latency_seconds": finished_at - started_at,
        "time_to_first_token_seconds": (
            first_token_at - started_at if first_token_at is not None else None
        ),
    }
