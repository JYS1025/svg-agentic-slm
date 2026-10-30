"""OpenAI-compatible HTTP backend for externally served language models."""

from __future__ import annotations

import base64
import ipaddress
import json
import math
import os
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from svg_agentic_slm.models.base import BaseModelBackend
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.schemas import ModelResponse

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SUPPORTED_ENGINES = {"llama_cpp", "responses", "vllm"}
SUPPORTED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_URL_OPENER = build_opener(_NoRedirectHandler())


class OpenAICompatibleBackend(BaseModelBackend):
    """Call one explicitly identified model through an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key_env: str | None = None,
        model_revision: str | None = None,
        engine: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 0,
        allow_insecure_http: bool = False,
        reasoning_effort: str = "medium",
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if not isinstance(allow_insecure_http, bool):
            raise ValueError("allow_insecure_http must be a boolean.")
        self.base_url = _validate_base_url(base_url, allow_insecure_http)
        self.model_id = _require_nonempty_string(model_id, "model_id")
        self.api_key_env = _validate_api_key_env(api_key_env)
        self.model_revision = _validate_optional_string(model_revision, "model_revision")
        self.engine = _require_nonempty_string(engine, "engine").lower()
        if self.engine not in SUPPORTED_ENGINES:
            supported = ", ".join(sorted(SUPPORTED_ENGINES))
            raise ValueError(f"engine must be one of: {supported}.")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive.")
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be non-negative.")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.reasoning_effort = _require_nonempty_string(
            reasoning_effort, "reasoning_effort"
        ).lower()
        if self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            supported = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            raise ValueError(f"reasoning_effort must be one of: {supported}.")
        self.generation_config = generation_config or GenerationConfig()
        self._ready = False

    @property
    def supports_response_format(self) -> bool:
        """Return whether image calls can enforce a JSON schema server-side."""
        return self.engine == "responses"

    @property
    def response_format_dialect(self) -> str | None:
        """Identify the structured-output JSON Schema subset in use."""
        return "openai" if self.engine == "responses" else None

    def load_model(self) -> None:
        """Verify server readiness and that the configured model is being served."""
        self._ready = False
        payload = self._request_json("GET", "/models")
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise RuntimeError("OpenAI-compatible /models response must contain a data list.")
        model_ids = {
            item.get("id")
            for item in raw_models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if self.model_id not in model_ids:
            raise RuntimeError(
                f"Configured model '{self.model_id}' is not served by the endpoint."
            )
        self._ready = True

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate one non-streaming chat completion."""
        if not self.is_loaded():
            raise RuntimeError("Model endpoint is not ready. Call load_model() before generate().")

        system_prompt = kwargs.pop("system_prompt", None)
        stop = kwargs.pop("stop", None)
        config = self._resolve_generation_config(kwargs)
        if self.engine == "responses":
            if stop is not None:
                raise ValueError("The Responses API backend does not support stop sequences.")
            return self._generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                config=config,
            )
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})

        request_payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": config["max_new_tokens"],
            "temperature": config["temperature"] if config["do_sample"] else 0.0,
            "top_p": config["top_p"] if config["do_sample"] else 1.0,
            "top_k": config["top_k"],
            "n": 1,
            "seed": config.get("seed"),
            "stream": False,
        }
        penalty_key = "repeat_penalty" if self.engine == "llama_cpp" else "repetition_penalty"
        request_payload[penalty_key] = config["repetition_penalty"]
        if stop is not None:
            request_payload["stop"] = stop
        if request_payload["seed"] is None:
            del request_payload["seed"]

        started_at = time.perf_counter()
        payload = self._request_json("POST", "/chat/completions", request_payload)
        latency_seconds = time.perf_counter() - started_at
        choice, message = _extract_choice(payload)
        response_model = payload.get("model")
        if not isinstance(response_model, str) or response_model != self.model_id:
            raise RuntimeError(
                "OpenAI-compatible response model does not match the configured model."
            )
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        return ModelResponse(
            text=message["content"],
            model_id=self.model_id,
            model_revision=self.model_revision,
            finish_reason=_optional_string(choice.get("finish_reason")),
            prompt_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
            latency_seconds=latency_seconds,
            metadata={
                "backend": "openai_compatible",
                "client": "stdlib-urllib",
                "engine": self.engine,
                "base_url": self.base_url,
                "served_model": self.model_id,
            },
        )

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate from text and one image through the Responses API."""
        if not self.is_loaded():
            raise RuntimeError("Model endpoint is not ready. Call load_model() before generate().")
        if self.engine != "responses":
            raise ValueError("Image generation requires engine='responses'.")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must contain encoded image data.")
        normalized_mime = _require_nonempty_string(mime_type, "mime_type")
        if normalized_mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("mime_type must be image/png, image/jpeg, or image/webp.")
        system_prompt = kwargs.pop("system_prompt", None)
        response_format = kwargs.pop("response_format", None)
        config = self._resolve_generation_config(kwargs)
        image_url = (
            f"data:{normalized_mime};base64,"
            + base64.b64encode(bytes(image_bytes)).decode("ascii")
        )
        return self._generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            config=config,
            image_url=image_url,
            response_format=response_format,
        )

    def _generate_response(
        self,
        *,
        prompt: str,
        system_prompt: object,
        config: dict[str, Any],
        image_url: str | None = None,
        response_format: object = None,
    ) -> ModelResponse:
        content: str | list[dict[str, str]]
        if image_url is None:
            content = prompt
        else:
            content = [
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": prompt},
            ]
        request_payload: dict[str, Any] = {
            "model": self.model_id,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": config["max_new_tokens"],
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
        }
        if system_prompt is not None:
            request_payload["instructions"] = str(system_prompt)
        if response_format is not None:
            if not isinstance(response_format, dict):
                raise ValueError("response_format must be an object.")
            request_payload["text"] = {"format": response_format}

        started_at = time.perf_counter()
        payload = self._request_json("POST", "/responses", request_payload)
        latency_seconds = time.perf_counter() - started_at
        output_text = _extract_response_output_text(payload)
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        served_model = payload.get("model")
        return ModelResponse(
            text=output_text,
            model_id=self.model_id,
            model_revision=self.model_revision,
            finish_reason=_optional_string(payload.get("status")),
            prompt_tokens=_optional_nonnegative_int(usage.get("input_tokens")),
            completion_tokens=_optional_nonnegative_int(usage.get("output_tokens")),
            latency_seconds=latency_seconds,
            metadata={
                "backend": "openai_compatible",
                "client": "stdlib-urllib",
                "engine": self.engine,
                "base_url": self.base_url,
                "served_model": served_model if isinstance(served_model, str) else None,
                "response_id": _optional_string(payload.get("id")),
                "reasoning_effort": self.reasoning_effort,
                "usage": usage,
            },
        )

    def is_loaded(self) -> bool:
        """Return whether endpoint and model readiness were verified."""
        return self._ready

    def unload_model(self) -> None:
        """Clear local readiness state without controlling the external server."""
        self._ready = False

    def count_tokens(self, text: str) -> int:
        """Count tokens through the serving engine's tokenizer endpoint."""
        if not self.is_loaded():
            raise RuntimeError("Model endpoint is not ready. Call load_model() first.")
        if self.engine == "responses":
            # Responses endpoints do not expose the serving tokenizer. Match the
            # repository's established four-characters-per-token RAG fallback.
            return math.ceil(len(text) / 4)
        request_payload: dict[str, Any] = (
            {"model": self.model_id, "prompt": text}
            if self.engine == "vllm"
            else {"content": text, "add_special": True}
        )
        payload = self._request_json("POST", "/tokenize", request_payload)
        count = payload.get("count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            return count
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise RuntimeError("Tokenizer response must contain a non-negative count or tokens list.")

    def _resolve_generation_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        supported = set(self.generation_config.to_dict())
        unknown = set(overrides) - supported
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported generation option(s): {names}")
        merged = {**self.generation_config.to_dict(), **overrides}
        return GenerationConfig.from_dict(merged).to_dict()

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        api_key = self._resolve_api_key()
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, allow_nan=False).encode("utf-8")

        # Retrying a generation POST could duplicate an expensive inference request.
        attempts = self.max_retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            request = Request(url, data=body, headers=headers, method=method)
            try:
                with _URL_OPENER.open(request, timeout=self.timeout_seconds) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("OpenAI-compatible response exceeded the size limit.")
                return _decode_json_object(raw)
            except HTTPError as exc:
                try:
                    error_detail = exc.read(4096).decode("utf-8", errors="replace").strip()
                except Exception:
                    error_detail = ""
                raise RuntimeError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}"
                    + (f": {error_detail}" if error_detail else ".")
                ) from exc
            except (TimeoutError, URLError) as exc:
                if attempt + 1 >= attempts:
                    raise RuntimeError("OpenAI-compatible endpoint request failed.") from exc
        raise AssertionError("unreachable")

    def _resolve_api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(
                f"Required API key environment variable is not set: {self.api_key_env}."
            )
        return value


def _validate_base_url(value: str, allow_insecure_http: bool) -> str:
    base_url = _require_nonempty_string(value, "base_url").rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment.")
    if parsed.scheme == "http" and not allow_insecure_http and not _is_loopback(parsed.hostname):
        raise ValueError(
            "Remote HTTP endpoints require allow_insecure_http=true or HTTPS."
        )
    return base_url


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_api_key_env(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("api_key_env must be a string.")
    name = value.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValueError("api_key_env must be a valid environment variable name.")
    return name


def _require_nonempty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _validate_optional_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field_name)


def _decode_json_object(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("OpenAI-compatible endpoint returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI-compatible endpoint response must be a JSON object.")
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _extract_choice(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("OpenAI-compatible response must contain exactly one choice.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("OpenAI-compatible response choice must contain text content.")
    return choice, message


def _extract_response_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = payload.get("output")
    if not isinstance(output, list):
        raise RuntimeError("Responses API response must contain an output array.")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") in {"output_text", "text"}
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
    if not parts:
        raise RuntimeError("Responses API response did not contain output text.")
    return "".join(parts)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
