"""Pinned Hugging Face SigLIP2 image-text similarity scorer."""

from __future__ import annotations

import gc
import hashlib
import io
import math
import os
import re
import string
import threading
import time
from typing import Any

from PIL import Image

from svg_agentic_slm.models.image_text_similarity import (
    SIGLIP2_PAIR_PROBABILITY,
    BaseImageTextSimilarityScorer,
    ImageTextSimilarityEvidence,
    validate_image_text_similarity_evidence,
)

DEFAULT_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_SIGLIP2_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DEFAULT_SIGLIP2_TEXT_TEMPLATE = "This is a photo of {instruction}."

_ALLOWED_DTYPES = {"bfloat16", "float16", "float32"}
_ALLOWED_ATTENTION_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2"}


class Siglip2SimilarityScorer(BaseImageTextSimilarityScorer):
    """Compute a sigmoid image-text pair probability with a pinned SigLIP2 model."""

    def __init__(
        self,
        model_id: str = DEFAULT_SIGLIP2_MODEL_ID,
        *,
        model_revision: str = DEFAULT_SIGLIP2_MODEL_REVISION,
        device: str = "cpu",
        dtype: str = "float32",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        token_env: str | None = None,
        text_template: str = DEFAULT_SIGLIP2_TEXT_TEMPLATE,
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
        self._local_files_only = _require_bool(local_files_only, "local_files_only")
        self._trust_remote_code = _require_bool(trust_remote_code, "trust_remote_code")
        self._token_env = _validate_token_env(token_env)
        self._text_template = _validate_text_template(text_template)

        self._processor: Any = None
        self._model: Any = None
        self._torch_dtype: Any = None
        self._lock = threading.RLock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def text_template(self) -> str:
        return self._text_template

    def load_model(self) -> None:
        """Load the pinned processor and dual encoder on the configured device."""
        with self._lock:
            if self.is_loaded():
                return
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch and Transformers are required for SigLIP2 similarity."
                ) from exc

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
                model = AutoModel.from_pretrained(
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
                    f"Failed to load SigLIP2 model '{self._model_id}' at pinned revision."
                ) from exc

            self._processor = processor
            self._model = model
            self._torch_dtype = torch_dtype

    def score(
        self,
        instruction: str,
        image_png: bytes,
        *,
        attempt_id: str,
    ) -> ImageTextSimilarityEvidence:
        """Return the independent sigmoid probability for one text-image pair."""
        normalized_instruction = _require_nonempty_string(instruction, "instruction")
        normalized_attempt_id = _require_nonempty_string(attempt_id, "attempt_id")
        image_bytes = _validate_png_bytes(image_png)
        text_input = self._text_template.format(instruction=normalized_instruction)

        with self._lock:
            if not self.is_loaded():
                raise RuntimeError("SigLIP2 similarity model is not loaded.")
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required for SigLIP2 similarity.") from exc

            started_at = time.perf_counter()
            try:
                with Image.open(io.BytesIO(image_bytes)) as source:
                    source.load()
                    image = source.convert("RGB")
                inputs = self._processor(
                    text=[text_input],
                    images=[image],
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                moved_inputs = _move_inputs(
                    inputs,
                    device=self._device,
                    floating_dtype=self._torch_dtype,
                )
                with torch.inference_mode():
                    outputs = self._model(**moved_inputs)
                logits = getattr(outputs, "logits_per_image", None)
                if logits is None or not hasattr(logits, "numel") or logits.numel() != 1:
                    raise ValueError(
                        "SigLIP2 output logits_per_image must contain exactly one value."
                    )
                raw_logit = float(logits.detach().float().reshape(-1)[0].item())
                probability = float(torch.sigmoid(torch.tensor(raw_logit)).item())
            except Exception as exc:
                raise RuntimeError("SigLIP2 image-text similarity inference failed.") from exc
            latency_seconds = time.perf_counter() - started_at

        if not math.isfinite(raw_logit) or not math.isfinite(probability):
            raise RuntimeError("SigLIP2 returned a non-finite similarity value.")
        return validate_image_text_similarity_evidence(
            ImageTextSimilarityEvidence(
                attempt_id=normalized_attempt_id,
                metric=SIGLIP2_PAIR_PROBABILITY,
                score=probability,
                raw_logit=raw_logit,
                model_id=self._model_id,
                model_revision=self._model_revision,
                text_template=self._text_template,
                text_input=text_input,
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                device=self._device,
                dtype=self._dtype,
                latency_seconds=latency_seconds,
            )
        )

    def is_loaded(self) -> bool:
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
                "Required Hugging Face token environment variable is not set: "
                f"{self._token_env}."
            )
        return token


def _move_inputs(inputs: Any, *, device: str, floating_dtype: Any) -> dict[str, Any]:
    if not hasattr(inputs, "items"):
        raise TypeError("SigLIP2 processor output must be a mapping.")
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(device)
        if getattr(value, "is_floating_point", lambda: False)():
            value = value.to(dtype=floating_dtype)
        moved[key] = value
    return moved


def _validate_png_bytes(value: bytes) -> bytes:
    if not isinstance(value, bytes) or not value.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("image_png must contain PNG bytes.")
    return value


def _validate_text_template(value: str) -> str:
    template = _require_nonempty_string(value, "text_template")
    fields = [
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    ]
    if fields != ["instruction"]:
        raise ValueError(
            "text_template must contain exactly one {instruction} placeholder."
        )
    return template


def _validate_runtime_device(torch: Any, device: str, dtype: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device}.")
    index = 0 if device == "cuda" else int(device.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device index is unavailable: {device}.")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured CUDA device does not support bfloat16.")


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


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value
