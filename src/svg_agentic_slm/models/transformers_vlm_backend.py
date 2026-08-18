"""Local Hugging Face Transformers backend for image-conditioned critique."""

from __future__ import annotations

import gc
import os
import re
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import Any

from svg_agentic_slm.models.base import BaseModelBackend
from svg_agentic_slm.models.schemas import ModelResponse

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"

_ALLOWED_DTYPES = {"bfloat16", "float16", "float32"}
_ALLOWED_ATTENTION_IMPLEMENTATIONS = {"sdpa", "eager", "flash_attention_2"}
_AUTO_MODEL_CLASSES = {
    "image_text_to_text": "AutoModelForImageTextToText",
    "multimodal_lm": "AutoModelForMultimodalLM",
}
_MIME_TO_PIL_FORMATS = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
}
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_PIXELS = 4096 * 4096
_MAX_PROMPT_CHARACTERS = 100_000


class TransformersVLMBackend(BaseModelBackend):
    """Run a pinned image-text-to-text model with Transformers.

    The backend deliberately has no text-only fallback. Callers must provide
    encoded image bytes through :meth:`generate_with_image` so a VLM critic
    cannot silently degrade into a text-only critic.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model_revision: str = DEFAULT_MODEL_REVISION,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        auto_model_class: str = "image_text_to_text",
        max_new_tokens: int = 384,
        do_sample: bool = False,
        enable_thinking: bool | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        token_env: str | None = None,
    ) -> None:
        self._model_id = _require_nonempty_string(model_id, "model_id")
        self._model_revision = _require_commit_revision(model_revision)
        self._device = _validate_device(device)
        self._dtype = _validate_choice(dtype, "dtype", _ALLOWED_DTYPES)
        self._attn_implementation = _validate_choice(
            attn_implementation,
            "attn_implementation",
            _ALLOWED_ATTENTION_IMPLEMENTATIONS,
        )
        self._auto_model_class = _validate_choice(
            auto_model_class, "auto_model_class", set(_AUTO_MODEL_CLASSES)
        )
        self._max_new_tokens = _require_positive_int(max_new_tokens, "max_new_tokens")
        self._do_sample = _require_bool(do_sample, "do_sample")
        if enable_thinking is not None:
            enable_thinking = _require_bool(enable_thinking, "enable_thinking")
        self._enable_thinking = enable_thinking
        self._local_files_only = _require_bool(local_files_only, "local_files_only")
        self._trust_remote_code = _require_bool(trust_remote_code, "trust_remote_code")
        self._token_env = _validate_token_env(token_env)

        self._processor: Any = None
        self._model: Any = None
        self._torch_dtype: Any = None
        self._lock = threading.RLock()

    @property
    def model_id(self) -> str:
        """Return the immutable Hugging Face repository identifier."""
        return self._model_id

    @property
    def model_revision(self) -> str:
        """Return the pinned Hugging Face commit revision."""
        return self._model_revision

    def load_model(self) -> None:
        """Load the pinned processor and VLM on the configured device."""
        with self._lock:
            if self.is_loaded():
                return

            try:
                import torch
                import transformers
                from transformers import AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "The 'vlm' dependencies are required for TransformersVLMBackend."
                ) from exc
            model_class_name = _AUTO_MODEL_CLASSES[self._auto_model_class]
            model_loader = getattr(transformers, model_class_name, None)
            if model_loader is None:
                raise RuntimeError(
                    f"Configured VLM loader {model_class_name} is unavailable in Transformers."
                )

            _validate_runtime_device(torch, self._device, self._dtype)
            torch_dtype = getattr(torch, self._dtype)
            token = self._resolve_token()
            hub_kwargs: dict[str, Any] = {
                "revision": self._model_revision,
                "local_files_only": self._local_files_only,
                "trust_remote_code": self._trust_remote_code,
            }
            if token is not None:
                hub_kwargs["token"] = token

            try:
                processor = AutoProcessor.from_pretrained(self._model_id, **hub_kwargs)
                model = model_loader.from_pretrained(
                    self._model_id,
                    dtype=torch_dtype,
                    attn_implementation=self._attn_implementation,
                    **hub_kwargs,
                )
                model = model.eval().to(self._device)
            except Exception as exc:
                self._processor = None
                self._model = None
                self._torch_dtype = None
                raise RuntimeError(
                    f"Failed to load Transformers VLM '{self._model_id}' at pinned revision."
                ) from exc

            self._processor = processor
            self._model = model
            self._torch_dtype = torch_dtype

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Reject text-only inference because this backend requires an image."""
        del prompt, kwargs
        raise ValueError(
            "TransformersVLMBackend requires vision input; call generate_with_image()."
        )

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response conditioned on one validated in-memory image."""
        with self._lock:
            if not self.is_loaded():
                raise RuntimeError("Model is not loaded. Call load_model() before generation.")

            normalized_prompt = _validate_prompt(prompt)
            normalized_mime = _validate_mime_type(mime_type)
            system_prompt = kwargs.pop("system_prompt", None)
            if system_prompt is not None:
                system_prompt = _require_nonempty_string(system_prompt, "system_prompt")
                normalized_prompt = f"{system_prompt}\n\n{normalized_prompt}"
                normalized_prompt = _validate_prompt(normalized_prompt)
            options = self._resolve_generation_options(kwargs)
            image = _decode_image(image_bytes, normalized_mime)
            image_width, image_height = image.size

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": normalized_prompt},
                    ],
                }
            ]
            template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            try:
                inputs = self._processor.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except Exception as exc:
                raise RuntimeError("Failed to preprocess VLM image and prompt inputs.") from exc
            finally:
                image.close()

            moved_inputs = _move_inputs(
                inputs,
                device=self._device,
                floating_dtype=self._torch_dtype,
            )
            input_ids = moved_inputs.get("input_ids")
            if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
                raise RuntimeError("VLM processor did not return two-dimensional input_ids.")
            prompt_tokens = int(input_ids.shape[-1])

            import torch

            if options["seed"] is not None:
                torch.manual_seed(options["seed"])
                if self._device.startswith("cuda"):
                    torch.cuda.manual_seed_all(options["seed"])

            call_kwargs: dict[str, Any] = {
                "max_new_tokens": options["max_new_tokens"],
                "do_sample": options["do_sample"],
            }
            if options["do_sample"]:
                call_kwargs["temperature"] = options["temperature"]
                call_kwargs["top_p"] = options["top_p"]

            started_at = time.perf_counter()
            try:
                with torch.inference_mode():
                    generated_ids = self._model.generate(**moved_inputs, **call_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Vision-language generation failed for model '{self._model_id}'."
                ) from exc
            latency_seconds = time.perf_counter() - started_at

            if getattr(generated_ids, "ndim", 0) != 2 or generated_ids.shape[0] != 1:
                raise RuntimeError("VLM returned an unexpected generated token tensor.")
            completion_ids = generated_ids[0, prompt_tokens:]
            completion_tokens = int(completion_ids.numel())
            text = self._processor.decode(completion_ids, skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError("VLM generation returned an empty response.")

            return ModelResponse(
                text=text,
                model_id=self._model_id,
                model_revision=self._model_revision,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency_seconds,
                tokens_per_second=(
                    completion_tokens / latency_seconds if latency_seconds > 0 else None
                ),
                metadata={
                    "backend": "transformers_vlm",
                    "backend_version": _package_version("transformers"),
                    "device": self._device,
                    "dtype": self._dtype,
                    "attn_implementation": self._attn_implementation,
                    "auto_model_class": self._auto_model_class,
                    "do_sample": options["do_sample"],
                    "enable_thinking": self._enable_thinking,
                    "image_mime_type": normalized_mime,
                    "image_width": image_width,
                    "image_height": image_height,
                    "local_files_only": self._local_files_only,
                    "trust_remote_code": self._trust_remote_code,
                },
            )

    def is_loaded(self) -> bool:
        """Return whether both processor and model are available."""
        return self._processor is not None and self._model is not None

    def unload_model(self) -> None:
        """Release model references and cached CUDA allocations."""
        with self._lock:
            self._model = None
            self._processor = None
            self._torch_dtype = None
            gc.collect()
            if self._device.startswith("cuda"):
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    def _resolve_token(self) -> str | None:
        if self._token_env is None:
            return None
        token = os.environ.get(self._token_env)
        if not token:
            raise RuntimeError(
                f"Required Hugging Face token environment variable is not set: "
                f"{self._token_env}."
            )
        return token

    def _resolve_generation_options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        supported = {"max_new_tokens", "do_sample", "temperature", "top_p", "seed"}
        unknown = set(overrides) - supported
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported VLM generation option(s): {names}")

        max_new_tokens = _require_positive_int(
            overrides.get("max_new_tokens", self._max_new_tokens),
            "max_new_tokens",
        )
        do_sample = _require_bool(overrides.get("do_sample", self._do_sample), "do_sample")
        temperature = _require_probability_like(
            overrides.get("temperature", 1.0),
            "temperature",
            include_zero=False,
        )
        top_p = _require_probability_like(
            overrides.get("top_p", 1.0),
            "top_p",
            include_zero=False,
        )
        seed = overrides.get("seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None.")
        if not do_sample and ({"temperature", "top_p"} & set(overrides)):
            raise ValueError("temperature and top_p require do_sample=true.")
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }


def _decode_image(image_bytes: bytes, mime_type: str) -> Any:
    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes.")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty.")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image_bytes exceeds {_MAX_IMAGE_BYTES} bytes.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for VLM image decoding.") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            detected_format = (source.format or "").upper()
            if detected_format not in _MIME_TO_PIL_FORMATS[mime_type]:
                raise ValueError(
                    f"Image content format {detected_format!r} does not match {mime_type!r}."
                )
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are empty or exceed the safety limit.")
            source.load()
            return source.convert("RGB").copy()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("image_bytes does not contain a decodable supported image.") from exc


def _move_inputs(inputs: Any, *, device: str, floating_dtype: Any) -> dict[str, Any]:
    if not hasattr(inputs, "items"):
        raise RuntimeError("VLM processor returned an unexpected input container.")
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(device)
        is_floating_point = getattr(value, "is_floating_point", None)
        if callable(is_floating_point) and is_floating_point():
            value = value.to(dtype=floating_dtype)
        moved[key] = value
    return moved


def _validate_runtime_device(torch: Any, device: str, dtype: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device}.")
    index = 0 if device == "cuda" else int(device.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device index is unavailable: {device}.")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured CUDA runtime does not support bfloat16.")


def _validate_prompt(value: str) -> str:
    prompt = _require_nonempty_string(value, "prompt")
    if len(prompt) > _MAX_PROMPT_CHARACTERS:
        raise ValueError(f"prompt exceeds {_MAX_PROMPT_CHARACTERS} characters.")
    return prompt


def _validate_mime_type(value: str) -> str:
    mime_type = _require_nonempty_string(value, "mime_type").lower()
    if mime_type not in _MIME_TO_PIL_FORMATS:
        names = ", ".join(sorted(_MIME_TO_PIL_FORMATS))
        raise ValueError(f"Unsupported mime_type {mime_type!r}; choose one of: {names}.")
    return mime_type


def _validate_device(value: str) -> str:
    device = _require_nonempty_string(value, "device").lower()
    if re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", device) is None:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<non-negative index>'.")
    return device


def _require_commit_revision(value: str) -> str:
    revision = _require_nonempty_string(value, "model_revision").lower()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model_revision must be an immutable 40-character commit SHA.")
    return revision


def _validate_token_env(value: str | None) -> str | None:
    if value is None:
        return None
    name = _require_nonempty_string(value, "token_env")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValueError("token_env must be a valid environment variable name.")
    return name


def _validate_choice(value: str, name: str, choices: set[str]) -> str:
    normalized = _require_nonempty_string(value, name).lower()
    if normalized not in choices:
        names = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {names}.")
    return normalized


def _require_nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _require_probability_like(value: Any, name: str, *, include_zero: bool) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    number = float(value)
    lower_bound_ok = number >= 0.0 if include_zero else number > 0.0
    if not lower_bound_ok or number > 1.0:
        lower = "0" if include_zero else "greater than 0"
        raise ValueError(f"{name} must be {lower} and at most 1.")
    return number


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "TransformersVLMBackend",
]
