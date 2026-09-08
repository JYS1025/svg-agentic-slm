"""Test-only constrained generation for the experimental OmniSVG 4B dialect.

This module intentionally does not enable the production inference backend. It loads the
completed Gemma 4 PEFT adapter on an isolated GPU, reproduces the exact SFT chat boundary,
and permits only the registered codec vocabulary in the completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SYSTEM_PROMPT = """You are an SVG generator trained with a registered discrete SVG codec.
Plan the composition internally, but return only the registered discrete SVG tokens.
Do not output XML, prose, Markdown, code fences, or reasoning. The token sequence must be
complete and decodable, beginning with the codec start token and ending with its end token.
Follow the user's requested objects, layout, geometry, colors, and style exactly."""

_USER_TEMPLATE = """Create the requested SVG composition and encode it with the registered svgd1 discrete vocabulary.

Instruction:
{instruction}"""

_CODEC_ID_MIN = 262144
_CODEC_BODY_ID_MAX = 307203
_CODEC_BOS_ID = 307204
_CODEC_EOS_ID = 307205
_EXPECTED_VOCAB_SIZE = 307206


class ExperimentalGenerationError(RuntimeError):
    """Raised when a test-only generation contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ExperimentalGenerationError("Chat template returned multiple examples")
        value = value[0]
    if not isinstance(value, list) or not value or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ExperimentalGenerationError("Chat template returned invalid token IDs")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _build_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    user_content = _USER_TEMPLATE.format(instruction=prompt.strip())
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "<svgovg4b:196998>"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    ids = _flat_ids(rendered)
    positions = [index for index, token_id in enumerate(ids) if token_id == _CODEC_BOS_ID]
    if len(positions) != 1:
        raise ExperimentalGenerationError(
            f"Expected exactly one codec BOS at the SFT boundary, found {len(positions)}"
        )
    prompt_ids = ids[: positions[0]]
    if not prompt_ids:
        raise ExperimentalGenerationError("SFT prompt boundary is empty")
    if any(_CODEC_ID_MIN <= token_id <= _CODEC_EOS_ID for token_id in prompt_ids):
        raise ExperimentalGenerationError("Codec token leaked into the model prompt")
    return prompt_ids


def _validate_tokenizer(tokenizer: Any) -> None:
    if len(tokenizer) != _EXPECTED_VOCAB_SIZE:
        raise ExperimentalGenerationError(
            f"Tokenizer size mismatch: expected {_EXPECTED_VOCAB_SIZE}, got {len(tokenizer)}"
        )
    expected = {
        "<svgovg4b:151938>": _CODEC_ID_MIN,
        "<svgovg4b:196998>": _CODEC_BOS_ID,
        "<svgovg4b:196999>": _CODEC_EOS_ID,
    }
    actual = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in expected
    }
    if actual != expected:
        raise ExperimentalGenerationError(
            f"Saved tokenizer codec mapping mismatch: expected {expected}, got {actual}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.experimental_enable:
        raise ExperimentalGenerationError(
            "Refusing to run without --experimental-enable-official-4b-generation"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise ExperimentalGenerationError(
            "This one-case runner requires CUDA_VISIBLE_DEVICES=2 to isolate the live job"
        )

    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForMultimodalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LogitsProcessor,
        LogitsProcessorList,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentalGenerationError(
            "Expected exactly one visible CUDA device for isolated generation"
        )

    class CodecOnlyLogitsProcessor(LogitsProcessor):
        def __init__(self, prompt_length: int) -> None:
            self.prompt_length = prompt_length

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            completion_length = int(input_ids.shape[1]) - self.prompt_length
            constrained = torch.full_like(scores, float("-inf"))
            if completion_length == 0:
                constrained[:, _CODEC_BOS_ID] = scores[:, _CODEC_BOS_ID]
                return constrained
            constrained[:, _CODEC_ID_MIN : _CODEC_BODY_ID_MAX + 1] = scores[
                :, _CODEC_ID_MIN : _CODEC_BODY_ID_MAX + 1
            ]
            constrained[:, _CODEC_EOS_ID] = scores[:, _CODEC_EOS_ID]
            return constrained

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    _validate_tokenizer(tokenizer)
    prompt_ids = _build_prompt_ids(tokenizer, args.prompt)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    started = time.monotonic()
    base = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map={"": 0},
    )
    base.resize_token_embeddings(len(tokenizer))
    base.tie_weights()
    model = PeftModel.from_pretrained(
        base,
        args.adapter_dir,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.monotonic() - started

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.reset_peak_memory_stats()
    generation_started = time.monotonic()
    with torch.inference_mode():
        sampling_kwargs = (
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            if args.do_sample
            else {}
        )
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=2,
            eos_token_id=_CODEC_EOS_ID,
            forced_eos_token_id=(
                _CODEC_EOS_ID if args.force_eos_at_limit else None
            ),
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else _CODEC_EOS_ID
            ),
            use_cache=True,
            logits_processor=LogitsProcessorList(
                [CodecOnlyLogitsProcessor(len(prompt_ids))]
            ),
            repetition_penalty=args.repetition_penalty,
            **sampling_kwargs,
        )
    generation_seconds = time.monotonic() - generation_started
    completion = [int(value) for value in generated[0, len(prompt_ids) :].tolist()]
    if not completion or completion[0] != _CODEC_BOS_ID:
        raise ExperimentalGenerationError("Generated completion is missing the codec BOS")
    if completion[-1] != _CODEC_EOS_ID:
        raise ExperimentalGenerationError(
            f"Generated completion did not reach codec EOS within {args.max_new_tokens} tokens"
        )
    if _CODEC_BOS_ID in completion[1:]:
        raise ExperimentalGenerationError("Generated completion contains an embedded codec BOS")
    invalid = [
        token_id
        for token_id in completion
        if not _CODEC_ID_MIN <= token_id <= _CODEC_EOS_ID
    ]
    if invalid:
        raise ExperimentalGenerationError(
            f"Generated completion contains IDs outside the codec vocabulary: {invalid[:8]}"
        )

    output_dir = Path(args.output_dir)
    ids_path = output_dir / "candidate_ids.json"
    adapter_path = Path(args.adapter_dir) / "adapter_model.safetensors"
    tokenizer_path = Path(args.tokenizer_dir) / "tokenizer.json"
    _atomic_json(ids_path, completion)
    manifest = {
        "schema_version": 1,
        "experimental": True,
        "production_inference_enabled": False,
        "benchmark_case": "text_to_svg_icon_2",
        "prompt": args.prompt,
        "rendered_user_content": _USER_TEMPLATE.format(
            instruction=args.prompt.strip()
        ),
        "system_prompt": _SYSTEM_PROMPT,
        "seed": args.seed,
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "repetition_penalty": args.repetition_penalty,
            "force_eos_at_limit": args.force_eos_at_limit,
            "eos_at_generation_limit": len(completion) == args.max_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "completion_token_count": len(completion),
            "codec_bos_id": _CODEC_BOS_ID,
            "codec_eos_id": _CODEC_EOS_ID,
            "allowed_id_min": _CODEC_ID_MIN,
            "allowed_id_max": _CODEC_EOS_ID,
            "prompt_token_count": len(prompt_ids),
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "base_model": str(Path(args.base_model).resolve()),
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "adapter_sha256": _sha256(adapter_path),
        "tokenizer_dir": str(Path(args.tokenizer_dir).resolve()),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "candidate_ids": str(ids_path.resolve()),
        "candidate_ids_sha256": _sha256(ids_path),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_version": torch.__version__,
    }
    _atomic_json(output_dir / "generation_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experimental-enable-official-4b-generation",
        action="store_true",
        dest="experimental_enable",
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="A calendar with checkmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.88)
    parser.add_argument("--top-k", type=int, default=50)
    # Repeated commands and coordinates are valid, common SVG structure.  Keep
    # the deterministic baseline truly greedy; repetition penalties are an
    # explicit ablation rather than a hidden default.
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--force-eos-at-limit", action="store_true")
    return parser


def main() -> int:
    manifest = run(_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
