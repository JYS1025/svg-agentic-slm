"""Response-only QLoRA training for the text-to-SVG Generator."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from svg_agentic_slm.data.jsonl import read_jsonl
from svg_agentic_slm.prompts.system_prompts import get_svg_generator_system_prompt
from svg_agentic_slm.prompts.text_to_svg import build_text_to_svg_prompt
from svg_agentic_slm.svg.discrete_codec import OmniSVGDiscreteCodec
from svg_agentic_slm.train.lora_config import LoRAConfig

logger = logging.getLogger(__name__)

InstructionMode = Literal["description_only", "mixed_60_detail_40_description", "detail_only"]
TargetRepresentation = Literal["raw_xml", "omnisvg_discrete"]

_AUTO_MODEL_CLASSES = {
    "causal_lm": "AutoModelForCausalLM",
    "multimodal_lm": "AutoModelForMultimodalLM",
}

_DISCRETE_SYSTEM_PROMPT = """You are an SVG generator trained with the svgd1 discrete-codec ablation.
Plan the composition internally, but return only the registered svgd1 discrete SVG tokens.
Do not output XML, prose, Markdown, code fences, or reasoning. The token sequence must be
complete and decodable, beginning with the codec start token and ending with its end token.
Follow the user's requested objects, layout, geometry, colors, and style exactly."""


@dataclass
class SFTConfig:
    output_dir: str = "./outputs/sft/gemma4_raw"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    max_seq_length: int = 8192
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 42
    dataloader_num_workers: int = 2
    resume_from_checkpoint: str | None = None
    merge_adapter: bool = False
    report_to: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class ModelTrainingConfig:
    model_id: str
    revision: str
    auto_model_class: str = "multimodal_lm"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    local_files_only: bool = False
    trust_remote_code: bool = False
    token_env: str | None = None


class _ResponseOnlyDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        tokenizer: Any,
        instruction_mode: InstructionMode,
        target_representation: TargetRepresentation,
        max_seq_length: int,
        seed: int,
        codec: OmniSVGDiscreteCodec | None,
    ) -> None:
        self._records = records
        self._tokenizer = tokenizer
        self._instruction_mode = instruction_mode
        self._target_representation = target_representation
        self._max_seq_length = max_seq_length
        self._codec = codec

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self._records[index]
        instruction = _select_instruction(row, self._instruction_mode)
        target = _select_target(row, self._target_representation, self._codec)
        if self._target_representation == "raw_xml":
            system_prompt = get_svg_generator_system_prompt()
            user_prompt = build_text_to_svg_prompt(instruction)
        else:
            system_prompt = _DISCRETE_SYSTEM_PROMPT
            user_prompt = (
                "Create the requested SVG composition and encode it with the registered "
                f"svgd1 discrete vocabulary.\n\nInstruction:\n{instruction}"
            )
        prefix_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_ids = _template_ids(
            self._tokenizer, prefix_messages, add_generation_prompt=True
        )
        full_ids = _template_ids(
            self._tokenizer,
            prefix_messages + [{"role": "assistant", "content": target}],
            add_generation_prompt=False,
        )
        if len(full_ids) > self._max_seq_length:
            record_id = row.get("metadata", {}).get("record_id", index)
            raise ValueError(
                f"SFT record {record_id!r} has {len(full_ids)} tokens, exceeding "
                f"max_seq_length={self._max_seq_length}; do not truncate SVG targets."
            )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("The chat template does not preserve the generation prefix.")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if not any(label != -100 for label in labels):
            raise RuntimeError("SFT record has no assistant target tokens.")
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class _ResponseOnlyCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = tokenizer.eos_token_id
        if self._pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, max_length), self._pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row_index, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row_index, :length] = 1
            labels[row_index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TextToSVGSFTTrainer:
    """Load Gemma4 with QLoRA and optimize only the assistant SVG suffix."""

    def __init__(
        self,
        *,
        model_config: ModelTrainingConfig,
        lora_config: LoRAConfig,
        sft_config: SFTConfig,
        train_data_path: str | Path,
        eval_data_path: str | Path | None,
        instruction_mode: InstructionMode,
        target_representation: TargetRepresentation,
    ) -> None:
        if instruction_mode not in (
            "description_only",
            "mixed_60_detail_40_description",
            "detail_only",
        ):
            raise ValueError(f"Unsupported instruction_mode: {instruction_mode}")
        if target_representation not in ("raw_xml", "omnisvg_discrete"):
            raise ValueError(f"Unsupported target_representation: {target_representation}")
        if model_config.auto_model_class not in _AUTO_MODEL_CLASSES:
            raise ValueError(f"Unsupported auto_model_class: {model_config.auto_model_class}")
        self._model_config = model_config
        self._lora_config = lora_config
        self._sft_config = sft_config
        self._train_data_path = Path(train_data_path)
        self._eval_data_path = Path(eval_data_path) if eval_data_path else None
        self._instruction_mode = instruction_mode
        self._target_representation = target_representation

    def train(self) -> dict[str, Any]:
        try:
            import torch
            import transformers
            from peft import get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "Install training dependencies with `pip install -e '.[train]'`."
            ) from exc

        model_config = self._model_config
        token = os.environ.get(model_config.token_env) if model_config.token_env else None
        hub_kwargs: dict[str, Any] = {
            "revision": model_config.revision,
            "local_files_only": model_config.local_files_only,
            "trust_remote_code": model_config.trust_remote_code,
        }
        if token:
            hub_kwargs["token"] = token
        processor = AutoProcessor.from_pretrained(model_config.model_id, **hub_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Gemma4 AutoProcessor did not expose a tokenizer.")

        codec = OmniSVGDiscreteCodec() if self._target_representation == "omnisvg_discrete" else None
        if codec is not None:
            tokenizer.add_special_tokens({"additional_special_tokens": codec.vocabulary_tokens()})

        dtype = getattr(torch, model_config.dtype)
        quantization_config = None
        if model_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
            )
        model_loader = getattr(transformers, _AUTO_MODEL_CLASSES[model_config.auto_model_class], None)
        if model_loader is None:
            raise RuntimeError(
                f"Transformers does not provide {_AUTO_MODEL_CLASSES[model_config.auto_model_class]}."
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs: dict[str, Any] = {
            **hub_kwargs,
            "dtype": dtype,
            "attn_implementation": model_config.attn_implementation,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = {"": local_rank}
        model = model_loader.from_pretrained(model_config.model_id, **model_kwargs)
        if codec is not None:
            model.resize_token_embeddings(len(tokenizer))
        if model_config.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self._sft_config.gradient_checkpointing
            )
        lora_config = self._lora_config
        language_targets = _resolve_language_model_lora_targets(
            model, lora_config.target_modules
        )
        lora_config = LoRAConfig(
            **{**asdict(lora_config), "target_modules": language_targets}
        )
        if codec is not None:
            modules = list(dict.fromkeys(lora_config.modules_to_save + ["embed_tokens", "lm_head"]))
            lora_config = LoRAConfig(**{**asdict(lora_config), "modules_to_save": modules})
        model = get_peft_model(model, lora_config.to_peft_config())
        if self._sft_config.gradient_checkpointing and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        train_records = _load_records(self._train_data_path)
        eval_records = _load_records(self._eval_data_path) if self._eval_data_path else []
        train_dataset = _ResponseOnlyDataset(
            train_records,
            tokenizer=tokenizer,
            instruction_mode=self._instruction_mode,
            target_representation=self._target_representation,
            max_seq_length=self._sft_config.max_seq_length,
            seed=self._sft_config.seed,
            codec=codec,
        )
        eval_dataset = (
            _ResponseOnlyDataset(
                eval_records,
                tokenizer=tokenizer,
                instruction_mode=self._instruction_mode,
                target_representation=self._target_representation,
                max_seq_length=self._sft_config.max_seq_length,
                seed=self._sft_config.seed,
                codec=codec,
            )
            if eval_records
            else None
        )
        output_dir = Path(self._sft_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=self._sft_config.num_train_epochs,
            per_device_train_batch_size=self._sft_config.per_device_train_batch_size,
            per_device_eval_batch_size=self._sft_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self._sft_config.gradient_accumulation_steps,
            learning_rate=self._sft_config.learning_rate,
            weight_decay=self._sft_config.weight_decay,
            warmup_ratio=self._sft_config.warmup_ratio,
            lr_scheduler_type=self._sft_config.lr_scheduler_type,
            logging_steps=self._sft_config.logging_steps,
            save_strategy="steps",
            save_steps=self._sft_config.save_steps,
            save_total_limit=self._sft_config.save_total_limit,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=self._sft_config.eval_steps if eval_dataset is not None else None,
            bf16=self._sft_config.bf16,
            fp16=self._sft_config.fp16,
            gradient_checkpointing=self._sft_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=self._sft_config.optim,
            seed=self._sft_config.seed,
            data_seed=self._sft_config.seed,
            dataloader_num_workers=self._sft_config.dataloader_num_workers,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            report_to=self._sft_config.report_to or [],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_ResponseOnlyCollator(tokenizer),
        )
        train_result = trainer.train(resume_from_checkpoint=self._sft_config.resume_from_checkpoint)
        trainer.save_state()

        adapter_dir = output_dir / "adapter"
        tokenizer_dir = output_dir / "tokenizer"
        trainer.accelerator.wait_for_everyone()
        trainer.save_model(str(adapter_dir))
        trainer.accelerator.wait_for_everyone()
        codec_manifest = _build_codec_manifest(codec, tokenizer) if codec else None
        training_manifest = {
            "schema_version": 1,
            "base_model_id": model_config.model_id,
            "base_model_revision": model_config.revision,
            "auto_model_class": model_config.auto_model_class,
            "instruction_mode": self._instruction_mode,
            "target_representation": self._target_representation,
            "response_only_loss": True,
            "rag_context_in_training": False,
            "train_data_path": str(self._train_data_path.resolve()),
            "train_data_sha256": _file_sha256(self._train_data_path),
            "eval_data_path": str(self._eval_data_path.resolve()) if self._eval_data_path else None,
            "eval_data_sha256": _file_sha256(self._eval_data_path) if self._eval_data_path else None,
            "lora": asdict(lora_config),
            "sft": asdict(self._sft_config),
            "codec_manifest": codec_manifest,
            "versions": {
                name: _package_version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
            },
            "train_metrics": dict(train_result.metrics),
        }
        if trainer.is_world_process_zero():
            try:
                processor.save_pretrained(tokenizer_dir)
            except Exception:
                tokenizer.save_pretrained(tokenizer_dir)
            if codec_manifest is not None:
                _write_json(output_dir / "codec_manifest.json", codec_manifest)
            _write_json(output_dir / "training_manifest.json", training_manifest, default=str)
            if self._sft_config.merge_adapter:
                _merge_adapter(
                    output_dir=output_dir,
                    model_config=model_config,
                    adapter_dir=adapter_dir,
                    tokenizer_size=len(tokenizer),
                    dtype=dtype,
                    hub_kwargs=hub_kwargs,
                    processor=processor,
                    codec_manifest=codec_manifest,
                )
        trainer.accelerator.wait_for_everyone()
        return training_manifest


def _select_instruction(row: dict[str, Any], mode: InstructionMode) -> str:
    description = str(row.get("description", row.get("instruction", ""))).strip()
    detail = str(row.get("detail", "")).strip()
    if not description:
        raise ValueError("SFT record is missing description.")
    if mode == "description_only":
        return description
    if not detail:
        raise ValueError(f"SFT mode {mode!r} requires a non-empty detail field.")
    if mode == "detail_only":
        return detail
    metadata = row.get("metadata")
    instruction_policy = metadata.get("instruction_policy") if isinstance(metadata, dict) else None
    use_detail = (
        instruction_policy.get("r1_detail_60_description_40")
        if isinstance(instruction_policy, dict)
        else None
    )
    if not isinstance(use_detail, bool):
        raise ValueError(
            "Mixed-instruction SFT requires boolean metadata.instruction_policy."
            "r1_detail_60_description_40 from the deterministic data-preparation contract."
        )
    return detail if use_detail else description


def _select_target(
    row: dict[str, Any],
    representation: TargetRepresentation,
    codec: OmniSVGDiscreteCodec | None,
) -> str:
    svg = str(row.get("output_svg", "")).strip()
    if not svg:
        raise ValueError("SFT record is missing output_svg.")
    if representation == "raw_xml":
        return svg
    if codec is None:
        raise RuntimeError("Discrete target representation requires a codec.")
    return "".join(codec.encode_svg(svg))


def _template_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(value, int) for value in encoded):
        raise RuntimeError("Tokenizer chat template returned invalid token IDs.")
    return encoded


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SFT data file not found: {path}")
    records = [row for row in read_jsonl(path) if row.get("task") == "text_to_svg"]
    if not records:
        raise ValueError(f"SFT data file contains no text_to_svg records: {path}")
    return records


def _build_codec_manifest(codec: OmniSVGDiscreteCodec, tokenizer: Any) -> dict[str, Any]:
    tokens = codec.vocabulary_tokens()
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    if len(set(token_ids)) != len(token_ids) or any(value < 0 for value in token_ids):
        raise RuntimeError("Codec vocabulary did not map to unique tokenizer IDs.")
    compact = json.dumps(token_ids, separators=(",", ":"))
    manifest = {
        "schema_version": 1,
        "codec": codec.manifest(),
        "tokenizer": {
            "vocabulary_size": len(tokenizer),
            "codec_token_ids": token_ids,
            "codec_token_ids_sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        },
    }
    return manifest


def _resolve_language_model_lora_targets(model: Any, requested: list[str]) -> list[str]:
    """Resolve LoRA leaves only below Gemma4's language_model subtree.

    Gemma4 is multimodal and its vision tower contains projection names such as q_proj.
    Suffix-only PEFT matching would otherwise create trainable vision adapters that are
    unused by this text-only SFT path and break DDP with find_unused_parameters=False.
    """
    suffixes = set(requested)
    targets = [
        name
        for name, _module in model.named_modules()
        if "language_model" in name.split(".") and name.rsplit(".", 1)[-1] in suffixes
    ]
    if not targets:
        raise RuntimeError(
            "No requested LoRA modules were found under Gemma4's language_model subtree; "
            "refusing to attach adapters to the multimodal vision tower."
        )
    return targets


def _write_json(path: Path, payload: dict[str, Any], *, default: Any = None) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=default) + "\n",
        encoding="utf-8",
    )


def _merge_adapter(
    *,
    output_dir: Path,
    model_config: ModelTrainingConfig,
    adapter_dir: Path,
    tokenizer_size: int,
    dtype: Any,
    hub_kwargs: dict[str, Any],
    processor: Any,
    codec_manifest: dict[str, Any] | None,
) -> None:
    import transformers
    from peft import PeftModel

    loader = getattr(transformers, _AUTO_MODEL_CLASSES[model_config.auto_model_class])
    base = loader.from_pretrained(model_config.model_id, dtype=dtype, **hub_kwargs)
    if base.get_input_embeddings().num_embeddings != tokenizer_size:
        base.resize_token_embeddings(tokenizer_size)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged_dir = output_dir / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    if codec_manifest is not None:
        _write_json(merged_dir / "codec_manifest.json", codec_manifest)


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
