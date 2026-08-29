"""Trainable Hugging Face checkpoint backend for SVG generation.

This backend is the inference bridge for Generator SFT experiments.  It keeps
the production ``BaseModelBackend`` contract while supporting either a local
merged checkpoint or a PEFT adapter over an immutable Hugging Face base model.
The raw-XML and discrete-SVG target representations are intentionally isolated:
discrete output is decoded from generated token IDs and is never interpreted as
ordinary text.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

from svg_agentic_slm.models.base import BaseModelBackend
from svg_agentic_slm.models.generation_config import GenerationConfig
from svg_agentic_slm.models.schemas import ModelResponse
from svg_agentic_slm.svg.discrete_codec import OmniSVGDiscreteCodec

DEFAULT_MODEL_ID: Final = "google/gemma-4-12B-it-qat-q4_0-unquantized"
DEFAULT_MODEL_REVISION: Final = "b6ed86275a6a5735884e208bfed95b445a684ca2"
DISCRETE_MANIFEST_SCHEMA_VERSION: Final = 1

_ALLOWED_DTYPES = {"bfloat16", "float16", "float32"}
_ALLOWED_ATTENTION_IMPLEMENTATIONS = {"sdpa", "eager", "flash_attention_2"}
_AUTO_MODEL_CLASSES = {
    "multimodal_lm": "AutoModelForMultimodalLM",
    "image_text_to_text": "AutoModelForImageTextToText",
}
_OUTPUT_FORMATS = {"raw_xml", "discrete_svg"}
_MAX_PROMPT_CHARACTERS = 200_000


class TransformersTextBackend(BaseModelBackend):
    """Generate SVG with a trainable Transformers checkpoint.

    ``model_path`` selects a fully merged local model.  ``adapter_path`` loads
    a local PEFT adapter over the pinned ``model_id``/``model_revision`` base.
    The two modes are mutually exclusive.  ``tokenizer_path`` can select the
    processor/tokenizer saved with an SFT run.

    In ``discrete_svg`` mode, a checkpoint produced by discrete-token SFT and a
    matching manifest are mandatory.  The manifest binds the codec contract to
    the tokenizer's complete ordered codec-token ID vector, preventing an
    accidentally reordered tokenizer from producing plausible but wrong SVG.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model_revision: str = DEFAULT_MODEL_REVISION,
        model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        output_format: str = "raw_xml",
        codec_manifest_path: str | Path | None = None,
        codec_grid_size: int = 200,
        auto_model_class: str = "multimodal_lm",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        enable_thinking: bool | None = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        token_env: str | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self._model_id = _require_nonempty_string(model_id, "model_id")
        self._model_revision = _require_commit_revision(model_revision)
        self._model_path = _optional_path(model_path, "model_path")
        self._adapter_path = _optional_path(adapter_path, "adapter_path")
        self._tokenizer_path = _optional_path(tokenizer_path, "tokenizer_path")
        self._codec_manifest_path = _optional_path(
            codec_manifest_path, "codec_manifest_path"
        )
        self._output_format = _validate_choice(
            output_format, "output_format", _OUTPUT_FORMATS
        )
        self._auto_model_class = _validate_choice(
            auto_model_class, "auto_model_class", set(_AUTO_MODEL_CLASSES)
        )
        self._device = _validate_device(device)
        self._dtype = _validate_choice(dtype, "dtype", _ALLOWED_DTYPES)
        self._attn_implementation = _validate_choice(
            attn_implementation,
            "attn_implementation",
            _ALLOWED_ATTENTION_IMPLEMENTATIONS,
        )
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError("enable_thinking must be a boolean or None.")
        self._enable_thinking = enable_thinking
        self._local_files_only = _require_bool(local_files_only, "local_files_only")
        self._trust_remote_code = _require_bool(
            trust_remote_code, "trust_remote_code"
        )
        self._token_env = _validate_token_env(token_env)
        self._generation_config = generation_config or GenerationConfig()

        if self._model_path is not None and self._adapter_path is not None:
            raise ValueError("model_path and adapter_path are mutually exclusive.")
        if self._tokenizer_path is not None and (
            self._model_path is None and self._adapter_path is None
        ):
            raise ValueError(
                "tokenizer_path requires model_path or adapter_path so an untrained "
                "tokenizer cannot be attached to the base model."
            )
        if self._output_format == "raw_xml" and self._codec_manifest_path is not None:
            raise ValueError("codec_manifest_path is only valid for discrete_svg output.")
        if self._output_format == "discrete_svg":
            if self._model_path is None and self._adapter_path is None:
                raise ValueError(
                    "discrete_svg requires a trained model_path or adapter_path."
                )
            if self._tokenizer_path is None:
                raise ValueError(
                    "discrete_svg requires tokenizer_path from the training checkpoint."
                )
            if self._codec_manifest_path is None:
                raise ValueError("discrete_svg requires codec_manifest_path.")
        if not isinstance(codec_grid_size, int) or isinstance(codec_grid_size, bool):
            raise TypeError("codec_grid_size must be an integer.")
        self._codec = (
            OmniSVGDiscreteCodec(grid_size=codec_grid_size)
            if self._output_format == "discrete_svg"
            else None
        )

        self._processor: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch_dtype: Any = None
        self._codec_id_to_token: dict[int, str] = {}
        self._codec_manifest_sha256: str | None = None
        self._lock = threading.RLock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def load_model(self) -> None:
        """Load and validate processor, tokenizer, model, and optional adapter."""
        with self._lock:
            if self.is_loaded():
                return

            processor: Any = None
            tokenizer: Any = None
            model: Any = None
            codec_id_to_token: dict[int, str] = {}
            codec_manifest_sha256: str | None = None
            try:
                import torch
                import transformers
                from transformers import AutoProcessor, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "TransformersTextBackend requires PyTorch and Transformers."
                ) from exc

            model_class_name = _AUTO_MODEL_CLASSES[self._auto_model_class]
            model_loader = getattr(transformers, model_class_name, None)
            if model_loader is None:
                raise RuntimeError(
                    f"Configured model loader {model_class_name} is unavailable in "
                    "the installed Transformers version."
                )
            _validate_runtime_device(torch, self._device, self._dtype)
            torch_dtype = getattr(torch, self._dtype)
            token = self._resolve_token()

            try:
                for path, name in (
                    (self._model_path, "model_path"),
                    (self._adapter_path, "adapter_path"),
                    (self._tokenizer_path, "tokenizer_path"),
                ):
                    if path is not None:
                        _require_local_directory(path, name)
                if self._codec_manifest_path is not None:
                    _require_local_file(
                        self._codec_manifest_path, "codec_manifest_path"
                    )

                model_source: str | Path = self._model_path or self._model_id
                model_hub_kwargs = self._source_kwargs(
                    local=self._model_path is not None,
                    token=token,
                )
                processor = AutoProcessor.from_pretrained(
                    model_source, **model_hub_kwargs
                )

                if self._tokenizer_path is not None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self._tokenizer_path,
                        local_files_only=True,
                        trust_remote_code=self._trust_remote_code,
                    )
                    if not hasattr(processor, "tokenizer"):
                        raise RuntimeError(
                            "Loaded AutoProcessor does not expose a replaceable tokenizer."
                        )
                    processor.tokenizer = tokenizer
                else:
                    tokenizer = getattr(processor, "tokenizer", None)
                    if tokenizer is None:
                        raise RuntimeError("Loaded AutoProcessor does not expose a tokenizer.")

                if self._codec is not None:
                    if self._codec_manifest_path is None:
                        raise RuntimeError("Discrete codec manifest path was not configured.")
                    codec_id_to_token, codec_manifest_sha256 = (
                        _validate_discrete_checkpoint_contract(
                            codec=self._codec,
                            tokenizer=tokenizer,
                            manifest_path=self._codec_manifest_path,
                        )
                    )

                model = model_loader.from_pretrained(
                    model_source,
                    dtype=torch_dtype,
                    attn_implementation=self._attn_implementation,
                    **model_hub_kwargs,
                )
                if self._adapter_path is not None:
                    try:
                        from peft import PeftModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "Loading adapter_path requires the optional 'peft' package."
                        ) from exc
                    _resize_base_embeddings_for_adapter(model, tokenizer)
                    model = PeftModel.from_pretrained(
                        model,
                        str(self._adapter_path),
                        is_trainable=False,
                        local_files_only=True,
                    )
                else:
                    _require_matching_embedding_size(model, tokenizer)

                model = model.eval().to(self._device)
            except Exception as exc:
                processor = None
                tokenizer = None
                model = None
                gc.collect()
                if self._device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "Failed to load the Transformers SVG Generator checkpoint "
                    f"for '{self._model_id}'."
                ) from exc

            self._processor = processor
            self._tokenizer = tokenizer
            self._model = model
            self._torch_dtype = torch_dtype
            self._codec_id_to_token = codec_id_to_token
            self._codec_manifest_sha256 = codec_manifest_sha256

    def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate raw SVG XML or decode a discrete SVG token-ID sequence."""
        with self._lock:
            if not self.is_loaded():
                raise RuntimeError("Model is not loaded. Call load_model() first.")
            normalized_prompt = _validate_prompt(prompt)
            system_prompt = kwargs.pop("system_prompt", None)
            if system_prompt is not None:
                system_prompt = _validate_prompt(system_prompt, name="system_prompt")
            options = self._resolve_generation_options(kwargs)
            messages: list[dict[str, Any]] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": normalized_prompt})

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
                    messages, **template_kwargs
                )
            except Exception as exc:
                raise RuntimeError("Failed to apply the model chat template.") from exc
            moved_inputs = _move_inputs(inputs, self._device)
            input_ids = moved_inputs.get("input_ids")
            if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
                raise RuntimeError("Processor did not return two-dimensional input_ids.")
            if int(input_ids.shape[0]) != 1:
                raise RuntimeError("Only one prompt per generation call is supported.")
            prompt_tokens = int(input_ids.shape[-1])

            import torch

            if options.seed is not None:
                torch.manual_seed(options.seed)
                if self._device.startswith("cuda"):
                    torch.cuda.manual_seed_all(options.seed)
            call_kwargs = _generation_call_kwargs(options)
            started_at = time.perf_counter()
            try:
                with torch.inference_mode():
                    generated_ids = self._model.generate(**moved_inputs, **call_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Text generation failed for model '{self._model_id}'."
                ) from exc
            latency_seconds = time.perf_counter() - started_at

            if getattr(generated_ids, "ndim", 0) != 2 or int(generated_ids.shape[0]) != 1:
                raise RuntimeError("Model returned an unexpected generated token tensor.")
            if int(generated_ids.shape[-1]) < prompt_tokens:
                raise RuntimeError("Generated sequence is shorter than its input prompt.")
            completion_ids_tensor = generated_ids[0, prompt_tokens:]
            completion_ids = [int(value) for value in completion_ids_tensor.tolist()]
            if not completion_ids:
                raise RuntimeError("Model generation returned no completion tokens.")

            if self._codec is None:
                text = self._tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                ).strip()
                if not text:
                    raise RuntimeError("Model generation returned an empty response.")
            else:
                text = _decode_discrete_completion(
                    completion_ids,
                    codec=self._codec,
                    id_to_token=self._codec_id_to_token,
                )

            completion_tokens = len(completion_ids)
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
            finish_reason = (
                "stop"
                if eos_token_id is not None and completion_ids[-1] == int(eos_token_id)
                else "length"
            )
            return ModelResponse(
                text=text,
                model_id=self._model_id,
                model_revision=self._model_revision,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency_seconds,
                tokens_per_second=(
                    completion_tokens / latency_seconds if latency_seconds > 0 else None
                ),
                metadata={
                    "backend": "transformers_text",
                    "backend_version": _package_version("transformers"),
                    "peft_version": (
                        _package_version("peft")
                        if self._adapter_path is not None
                        else None
                    ),
                    "base_model_id": self._model_id,
                    "base_model_revision": self._model_revision,
                    "checkpoint_kind": (
                        "merged" if self._model_path is not None else
                        "peft_adapter" if self._adapter_path is not None else
                        "base"
                    ),
                    "model_path": _path_string(self._model_path),
                    "adapter_path": _path_string(self._adapter_path),
                    "tokenizer_path": _path_string(self._tokenizer_path),
                    "output_format": self._output_format,
                    "codec_manifest_sha256": self._codec_manifest_sha256,
                    "device": self._device,
                    "dtype": self._dtype,
                    "attn_implementation": self._attn_implementation,
                    "auto_model_class": self._auto_model_class,
                    "enable_thinking": self._enable_thinking,
                    "local_files_only": self._local_files_only,
                    "trust_remote_code": self._trust_remote_code,
                    "generation_parameters": options.to_dict(),
                },
            )

    def count_tokens(self, text: str) -> int:
        """Count text with the exact tokenizer used by the loaded checkpoint."""
        with self._lock:
            if not self.is_loaded():
                raise RuntimeError("Model is not loaded. Call load_model() first.")
            value = _validate_prompt(text, name="text", allow_empty=True)
            try:
                token_ids = self._tokenizer.encode(
                    value, add_special_tokens=False
                )
            except Exception as exc:
                raise RuntimeError("Failed to tokenize text for context budgeting.") from exc
            if not isinstance(token_ids, list) or any(
                not isinstance(item, int) for item in token_ids
            ):
                raise RuntimeError("Tokenizer returned an unexpected token sequence.")
            return len(token_ids)

    def is_loaded(self) -> bool:
        base_loaded = (
            self._processor is not None
            and self._tokenizer is not None
            and self._model is not None
        )
        if self._codec is not None:
            return (
                base_loaded
                and bool(self._codec_id_to_token)
                and self._codec_manifest_sha256 is not None
            )
        return base_loaded

    def unload_model(self) -> None:
        """Drop all model state and clear the selected CUDA allocator cache."""
        with self._lock:
            self._model = None
            self._processor = None
            self._tokenizer = None
            self._torch_dtype = None
            self._codec_id_to_token = {}
            self._codec_manifest_sha256 = None
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

    def _source_kwargs(self, *, local: bool, token: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "local_files_only": True if local else self._local_files_only,
            "trust_remote_code": self._trust_remote_code,
        }
        if not local:
            kwargs["revision"] = self._model_revision
            if token is not None:
                kwargs["token"] = token
        return kwargs

    def _resolve_generation_options(self, overrides: dict[str, Any]) -> GenerationConfig:
        supported = set(self._generation_config.to_dict())
        unknown = set(overrides) - supported
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unsupported Transformers text generation option(s): {names}"
            )
        values = self._generation_config.to_dict()
        values.update(overrides)
        options = GenerationConfig.from_dict(values)
        if options.do_sample and options.temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=true.")
        if options.seed is not None and (
            not isinstance(options.seed, int)
            or isinstance(options.seed, bool)
            or options.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None.")
        return options


def _generation_call_kwargs(options: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": options.max_new_tokens,
        "do_sample": options.do_sample,
        "repetition_penalty": options.repetition_penalty,
        "num_return_sequences": options.num_return_sequences,
    }
    if options.do_sample:
        kwargs.update(
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
        )
    return kwargs


def _move_inputs(inputs: Any, device: str) -> dict[str, Any]:
    if not hasattr(inputs, "items"):
        raise RuntimeError("Processor returned an unexpected input container.")
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _resize_base_embeddings_for_adapter(model: Any, tokenizer: Any) -> None:
    expected = _tokenizer_length(tokenizer)
    actual = _embedding_count(model)
    if actual == expected:
        return
    resize = getattr(model, "resize_token_embeddings", None)
    if not callable(resize):
        raise RuntimeError(
            "Base model embeddings do not match the SFT tokenizer and cannot be resized."
        )
    resize(expected)
    if _embedding_count(model) != expected:
        raise RuntimeError("Base model embedding resize did not produce the expected size.")


def _require_matching_embedding_size(model: Any, tokenizer: Any) -> None:
    expected = _tokenizer_length(tokenizer)
    actual = _embedding_count(model)
    if actual != expected:
        raise RuntimeError(
            "Merged model embedding count does not match its tokenizer: "
            f"model={actual}, tokenizer={expected}."
        )


def _embedding_count(model: Any) -> int:
    getter = getattr(model, "get_input_embeddings", None)
    embeddings = getter() if callable(getter) else None
    count = getattr(embeddings, "num_embeddings", None)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError("Model does not expose a valid input embedding count.")
    return count


def _tokenizer_length(tokenizer: Any) -> int:
    try:
        size = len(tokenizer)
    except Exception as exc:
        raise RuntimeError("Tokenizer does not expose its vocabulary length.") from exc
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise RuntimeError("Tokenizer vocabulary length is invalid.")
    return size


def _validate_discrete_checkpoint_contract(
    *,
    codec: OmniSVGDiscreteCodec,
    tokenizer: Any,
    manifest_path: Path,
) -> tuple[dict[int, str], str]:
    try:
        manifest_bytes = manifest_path.read_bytes()
        payload = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Failed to read the discrete checkpoint manifest.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Discrete checkpoint manifest must be a JSON object.")
    if payload.get("schema_version") != DISCRETE_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Unsupported discrete checkpoint manifest schema_version.")
    if payload.get("codec") != codec.manifest():
        raise RuntimeError("Discrete checkpoint codec contract does not match runtime codec.")
    tokenizer_contract = payload.get("tokenizer")
    if not isinstance(tokenizer_contract, dict):
        raise RuntimeError("Discrete manifest tokenizer contract is missing.")

    expected_size = tokenizer_contract.get("vocabulary_size")
    if expected_size != _tokenizer_length(tokenizer):
        raise RuntimeError("Discrete tokenizer vocabulary size differs from its manifest.")
    expected_ids = tokenizer_contract.get("codec_token_ids")
    if not isinstance(expected_ids, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in expected_ids
    ):
        raise RuntimeError("Discrete manifest codec_token_ids must be non-negative integers.")
    codec_tokens = codec.vocabulary_tokens()
    if len(expected_ids) != len(codec_tokens):
        raise RuntimeError("Discrete manifest codec_token_ids length is invalid.")
    expected_ids_sha256 = tokenizer_contract.get("codec_token_ids_sha256")
    if expected_ids_sha256 != _sha256_json(expected_ids):
        raise RuntimeError("Discrete manifest token-ID hash is invalid.")

    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if callable(get_vocab) else None
    if not isinstance(vocabulary, dict):
        raise RuntimeError("Tokenizer does not expose an auditable vocabulary mapping.")
    actual_ids: list[int] = []
    for token in codec_tokens:
        token_id = vocabulary.get(token)
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise RuntimeError(f"Discrete codec token is missing from tokenizer: {token}")
        actual_ids.append(token_id)
    if len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("Discrete codec tokens do not map to unique tokenizer IDs.")
    if actual_ids != expected_ids:
        raise RuntimeError("Discrete tokenizer token IDs differ from training manifest.")

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return dict(zip(actual_ids, codec_tokens, strict=True)), manifest_sha256


def _decode_discrete_completion(
    completion_ids: list[int],
    *,
    codec: OmniSVGDiscreteCodec,
    id_to_token: dict[int, str],
) -> str:
    sop_id = _id_for_token(codec.sop_token, id_to_token)
    eos_id = _id_for_token(codec.eos_token, id_to_token)
    try:
        start = completion_ids.index(sop_id)
    except ValueError as exc:
        raise RuntimeError("Discrete generation did not emit an SOP token.") from exc
    tokens: list[str] = []
    for token_id in completion_ids[start:]:
        token = id_to_token.get(token_id)
        if token is None:
            raise RuntimeError(
                "Discrete generation emitted a non-codec token inside its SVG sequence."
            )
        tokens.append(token)
        if token_id == eos_id:
            break
    if not tokens or tokens[-1] != codec.eos_token:
        raise RuntimeError("Discrete generation did not emit the codec EOS token.")
    try:
        return codec.decode_tokens(tokens)
    except Exception as exc:
        raise RuntimeError("Generated discrete token IDs do not form a valid SVG.") from exc


def _id_for_token(token: str, id_to_token: dict[int, str]) -> int:
    matches = [token_id for token_id, value in id_to_token.items() if value == token]
    if len(matches) != 1:
        raise RuntimeError(f"Discrete tokenizer mapping is invalid for token {token!r}.")
    return matches[0]


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


def _validate_prompt(
    value: str,
    *,
    name: str = "prompt",
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must be non-empty.")
    if len(value) > _MAX_PROMPT_CHARACTERS:
        raise ValueError(f"{name} exceeds {_MAX_PROMPT_CHARACTERS} characters.")
    return value if allow_empty else normalized


def _validate_device(value: str) -> str:
    device = _require_nonempty_string(value, "device").lower()
    if re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", device) is None:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'.")
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


def _optional_path(value: str | Path | None, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a filesystem path or None.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty when provided.")
    return Path(normalized).expanduser()


def _require_local_directory(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} is not an existing directory: {path}")


def _require_local_file(path: Path, name: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} is not an existing file: {path}")


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


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _path_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DISCRETE_MANIFEST_SCHEMA_VERSION",
    "TransformersTextBackend",
]
