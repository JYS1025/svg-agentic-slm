"""Strict completion-only SFT for the image-grounded SVG Critic."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from svg_agentic_slm.agents.vlm_critic import (
    VLM_CRITIC_VERSION,
    validate_critic_output_payload,
)
from svg_agentic_slm.prompts.system_prompts import get_svg_vlm_critic_system_prompt
from svg_agentic_slm.prompts.vlm_critic import (
    VLM_CRITIC_PROMPT_VERSION,
    build_vlm_critic_prompt,
)
from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.train.sft_trainer import (
    ModelTrainingConfig,
    SFTConfig,
    _early_stopping_resume_contract,
    _resolve_language_model_lora_targets,
)

logger = logging.getLogger(__name__)

_AUTO_MODEL_CLASSES = {
    "multimodal_lm": "AutoModelForMultimodalLM",
    "image_text_to_text": "AutoModelForImageTextToText",
}


class StrictVisionCompletionCollator:
    """Run the TRL VLM collator without truncation and reject oversized batches."""

    def __init__(self, processor: Any, *, max_length: int) -> None:
        from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

        self._max_length = max_length
        self._collator = DataCollatorForVisionLanguageModeling(
            processor=processor,
            max_length=None,
            completion_only_loss=True,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._collator(examples)
        sequence_length = int(batch["input_ids"].shape[1])
        if sequence_length > self._max_length:
            sample_ids = [str(example.get("sample_id", "unknown")) for example in examples]
            raise ValueError(
                "Critic SFT batch exceeds max_seq_length without safe truncation: "
                f"{sequence_length} > {self._max_length}; samples={sample_ids}."
            )
        if not batch["labels"].ne(-100).any():
            raise RuntimeError("Critic SFT batch has no completion tokens.")
        return batch


class CriticSFTTrainer:
    """Fine-tune a VLM Critic while preserving its production JSON contract."""

    def __init__(
        self,
        *,
        model_config: ModelTrainingConfig,
        lora_config: LoRAConfig,
        sft_config: SFTConfig,
        train_data_path: str | Path,
        eval_data_path: str | Path | None,
        score_threshold: float = 3.0,
    ) -> None:
        if model_config.auto_model_class not in _AUTO_MODEL_CLASSES:
            raise ValueError(
                "Critic auto_model_class must be multimodal_lm or image_text_to_text."
            )
        if not sft_config.do_train:
            raise ValueError("CriticSFTTrainer requires sft.do_train=true.")
        if sft_config.do_eval and eval_data_path is None:
            raise ValueError("sft.do_eval=true requires dataset.validation_path.")
        if sft_config.do_predict:
            raise ValueError(
                "Critic SFT does not implement do_predict; use the evaluation pipeline."
            )
        if not 0.0 <= float(score_threshold) <= 4.0:
            raise ValueError("critic.score_threshold must be between 0 and 4.")
        if sft_config.merge_adapter:
            raise ValueError("Critic adapter merging is intentionally separate from SFT.")
        self._model_config = model_config
        self._lora_config = lora_config
        self._sft_config = sft_config
        self._train_data_path = Path(train_data_path)
        self._eval_data_path = Path(eval_data_path) if eval_data_path is not None else None
        self._score_threshold = float(score_threshold)

    def train(self) -> dict[str, Any]:
        try:
            import torch
            import transformers
            from datasets import Dataset, Image
            from peft import get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback
            from trl import SFTConfig as TRLSFTConfig
            from trl import SFTTrainer
        except ImportError as exc:
            raise RuntimeError(
                "Install training dependencies with `pip install -e '.[train]'`."
            ) from exc

        if self._sft_config.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(
                self._sft_config.float32_matmul_precision
            )

        train_rows, train_manifest = load_critic_sft_records(
            self._train_data_path,
            score_threshold=self._score_threshold,
        )
        eval_rows: list[dict[str, Any]] = []
        eval_manifest = None
        if self._sft_config.do_eval:
            if self._eval_data_path is None:
                raise RuntimeError("Critic validation path was not retained.")
            eval_rows, eval_manifest = load_critic_sft_records(
                self._eval_data_path,
                score_threshold=self._score_threshold,
            )
            overlap = set(train_manifest["sample_ids"]) & set(eval_manifest["sample_ids"])
            if overlap:
                raise ValueError(
                    "Critic train/validation sample IDs overlap: "
                    + ", ".join(sorted(overlap)[:10])
                )

        train_dataset = Dataset.from_list(train_rows).cast_column("image", Image())
        eval_dataset = (
            Dataset.from_list(eval_rows).cast_column("image", Image())
            if eval_rows
            else None
        )

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
        dtype = getattr(torch, model_config.dtype)
        quantization_config = None
        if model_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
            )
        loader = getattr(
            transformers,
            _AUTO_MODEL_CLASSES[model_config.auto_model_class],
            None,
        )
        if loader is None:
            raise RuntimeError("Installed Transformers lacks the configured VLM auto class.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs: dict[str, Any] = {
            **hub_kwargs,
            "dtype": dtype,
            "attn_implementation": model_config.attn_implementation,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = {"": local_rank}
        model = loader.from_pretrained(model_config.model_id, **model_kwargs)
        if model_config.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self._sft_config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        resolved_targets = _resolve_language_model_lora_targets(
            model,
            self._lora_config.target_modules,
        )
        lora_config = LoRAConfig(
            **{
                **asdict(self._lora_config),
                "target_modules": resolved_targets,
            }
        )
        model = get_peft_model(model, lora_config.to_peft_config())
        if self._sft_config.gradient_checkpointing and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        output_dir = Path(self._sft_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        early_stopping_resume = _early_stopping_resume_contract(
            resume_from_checkpoint=self._sft_config.resume_from_checkpoint,
            patience=self._sft_config.early_stopping_patience,
            threshold=self._sft_config.early_stopping_threshold,
        )
        args = TRLSFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            do_train=True,
            do_eval=eval_dataset is not None,
            num_train_epochs=self._sft_config.num_train_epochs,
            max_steps=self._sft_config.max_steps,
            per_device_train_batch_size=self._sft_config.per_device_train_batch_size,
            per_device_eval_batch_size=self._sft_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self._sft_config.gradient_accumulation_steps,
            learning_rate=self._sft_config.learning_rate,
            weight_decay=self._sft_config.weight_decay,
            # Transformers 5 represents both a step count and a ratio through
            # warmup_steps; values in [0, 1) are interpreted as a ratio.
            warmup_steps=self._sft_config.warmup_ratio,
            lr_scheduler_type=self._sft_config.lr_scheduler_type,
            logging_steps=self._sft_config.logging_steps,
            logging_nan_inf_filter=self._sft_config.logging_nan_inf_filter,
            save_strategy=self._sft_config.save_strategy,
            save_steps=self._sft_config.save_steps,
            save_total_limit=self._sft_config.save_total_limit,
            eval_strategy=(
                self._sft_config.eval_strategy if eval_dataset is not None else "no"
            ),
            eval_steps=(self._sft_config.eval_steps if eval_dataset is not None else None),
            prediction_loss_only=True,
            load_best_model_at_end=(
                self._sft_config.load_best_model_at_end and eval_dataset is not None
            ),
            restore_callback_states_from_checkpoint=early_stopping_resume[
                "restore_callback_states_from_checkpoint"
            ],
            metric_for_best_model=(
                self._sft_config.metric_for_best_model if eval_dataset is not None else None
            ),
            greater_is_better=(
                self._sft_config.greater_is_better if eval_dataset is not None else None
            ),
            bf16=self._sft_config.bf16,
            fp16=self._sft_config.fp16,
            tf32=self._sft_config.tf32,
            gradient_checkpointing=self._sft_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=self._sft_config.optim,
            torch_compile=self._sft_config.torch_compile,
            torch_compile_backend=self._sft_config.torch_compile_backend,
            torch_compile_mode=self._sft_config.torch_compile_mode,
            seed=self._sft_config.seed,
            data_seed=self._sft_config.seed,
            dataloader_num_workers=self._sft_config.dataloader_num_workers,
            dataloader_pin_memory=self._sft_config.dataloader_pin_memory,
            dataloader_persistent_workers=self._sft_config.dataloader_persistent_workers,
            dataloader_prefetch_factor=self._sft_config.dataloader_prefetch_factor,
            dataloader_drop_last=self._sft_config.dataloader_drop_last,
            accelerator_config={"non_blocking": self._sft_config.dataloader_non_blocking},
            torch_empty_cache_steps=self._sft_config.torch_empty_cache_steps,
            remove_unused_columns=True,
            ddp_find_unused_parameters=False,
            ddp_backend=(
                self._sft_config.ddp_backend
                if int(os.environ.get("WORLD_SIZE", "1")) > 1
                else None
            ),
            ddp_bucket_cap_mb=self._sft_config.ddp_bucket_cap_mb,
            ddp_broadcast_buffers=self._sft_config.ddp_broadcast_buffers,
            ddp_static_graph=self._sft_config.ddp_static_graph,
            include_num_input_tokens_seen=self._sft_config.include_num_input_tokens_seen,
            report_to=self._sft_config.report_to or [],
            max_length=None,
            packing=False,
            padding_free=False,
            completion_only_loss=True,
            assistant_only_loss=False,
            loss_type=(
                "chunked_nll" if self._sft_config.chunked_lm_head_loss else None
            ),
        )
        callbacks = []
        if self._sft_config.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self._sft_config.early_stopping_patience,
                    early_stopping_threshold=self._sft_config.early_stopping_threshold,
                )
            )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            data_collator=StrictVisionCompletionCollator(
                processor,
                max_length=self._sft_config.max_seq_length,
            ),
            callbacks=callbacks,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_result = trainer.train(
            resume_from_checkpoint=self._sft_config.resume_from_checkpoint
        )
        trainer.save_state()
        adapter_dir = output_dir / "adapter"
        processor_dir = output_dir / "processor"
        trainer.accelerator.wait_for_everyone()
        trainer.save_model(str(adapter_dir))
        trainer.accelerator.wait_for_everyone()

        manifest = {
            "schema_version": 1,
            "task": "image_grounded_svg_critic_sft",
            "base_model_id": model_config.model_id,
            "base_model_revision": model_config.revision,
            "critic_runtime_version": VLM_CRITIC_VERSION,
            "critic_prompt_version": VLM_CRITIC_PROMPT_VERSION,
            "score_threshold": self._score_threshold,
            "response_only_loss": True,
            "vision_tower_trainable": False,
            "loss_implementation": (
                "trl_chunked_nll_active_completion_tokens"
                if self._sft_config.chunked_lm_head_loss
                else "transformers_default_causal_lm"
            ),
            "strict_no_truncation": True,
            "train_data": train_manifest,
            "validation_data": eval_manifest,
            "lora": asdict(lora_config),
            "sft": asdict(self._sft_config),
            "versions": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "trl",
                    "peft",
                    "bitsandbytes",
                    "accelerate",
                )
            },
            "runtime": {
                "world_size": trainer.accelerator.num_processes,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "tf32_matmul_effective": bool(
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "native_sdpa_flash_available": bool(
                    torch.backends.cuda.is_flash_attention_available()
                ),
                "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "train_metrics": dict(train_result.metrics),
            "training_control": {
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "global_step": trainer.state.global_step,
                "early_stopping_resume": early_stopping_resume,
            },
        }
        if trainer.is_world_process_zero():
            processor.save_pretrained(processor_dir)
            _write_json(output_dir / "training_manifest.json", manifest)
        trainer.accelerator.wait_for_everyone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return manifest


def load_critic_sft_records(
    path: str | Path,
    *,
    score_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate image-grounded Critic labels from an auditable JSONL."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Critic SFT JSONL not found: {source_path}")
    rows: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    seen: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"Critic record {source_path}:{line_number} must be an object.")
            sample_id = str(raw.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen:
                raise ValueError(
                    f"Critic record {source_path}:{line_number} has a missing/duplicate sample_id."
                )
            instruction = str(raw.get("instruction", "")).strip()
            labeled_svg = str(raw.get("labeled_svg", "")).strip()
            allowed = raw.get("allowed_target_ids")
            if (
                not instruction
                or not labeled_svg
                or not isinstance(allowed, list)
                or not all(isinstance(value, str) and value for value in allowed)
                or len(set(allowed)) != len(allowed)
            ):
                raise ValueError(
                    f"Critic record {sample_id!r} has invalid instruction/labeled SVG/target IDs."
                )
            image_value = raw.get("image_path")
            if not isinstance(image_value, str) or not image_value.strip():
                raise ValueError(f"Critic record {sample_id!r} requires image_path.")
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = source_path.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Critic record {sample_id!r} image not found: {image_path}"
                )
            payload = raw.get("critic_output")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise ValueError(f"Critic record {sample_id!r} requires critic_output object.")
            validate_critic_output_payload(
                payload,
                allowed_target_ids=set(allowed),
                score_threshold=score_threshold,
            )
            similarity_score = raw.get("similarity_score")
            if similarity_score is not None:
                similarity_score = float(similarity_score)
            system_prompt = get_svg_vlm_critic_system_prompt(
                score_threshold=score_threshold
            )
            user_prompt = build_vlm_critic_prompt(
                instruction,
                labeled_svg=labeled_svg,
                allowed_target_ids=list(allowed),
                score_threshold=score_threshold,
                similarity_score=similarity_score,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "prompt": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": user_prompt},
                            ],
                        },
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                }
            )
            sample_ids.append(sample_id)
            seen.add(sample_id)
    if not rows:
        raise ValueError(f"Critic SFT JSONL is empty: {source_path}")
    return rows, {
        "path": str(source_path),
        "sha256": _sha256_file(source_path),
        "record_count": len(rows),
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(
            json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "label_contract": "vlm_critic_production_output_exact",
    }


def _sha256_file(path: Path) -> str:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
