"""Qwen3.5 multimodal critic distillation with response-only QLoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import types
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from svg_agentic_slm.train.chunked_causal_lm_loss import install_chunked_causal_lm_loss
from svg_agentic_slm.train.critic_multimodal_cache import (
    CriticMultimodalCollator,
    load_or_build_critic_cache,
)
from svg_agentic_slm.train.critic_training_logging import DetailedLoggingCallback
from svg_agentic_slm.utils.config import load_yaml_config
from svg_agentic_slm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def resolve_qwen_language_lora_targets(model: Any) -> list[str]:
    """Resolve every linear projection under language_model and exclude vision/head."""
    import torch

    targets = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if "language_model" not in parts:
            continue
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in {
            "Linear4bit",
            "Linear8bitLt",
        }:
            targets.append(name)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("No linear modules were found under Qwen language_model.")
    forbidden = [
        name
        for name in targets
        if "visual" in name.split(".") or name.endswith("lm_head")
    ]
    if forbidden:
        raise RuntimeError(f"Unsafe critic LoRA target resolution: {forbidden[:3]}")
    leaves = {name.rsplit(".", 1)[-1] for name in targets}
    expected_any = [
        {"q_proj", "k_proj", "v_proj", "o_proj"},
        {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"},
        {"gate_proj", "up_proj", "down_proj"},
    ]
    missing_groups = [sorted(group) for group in expected_any if not (leaves & group)]
    if missing_groups:
        raise RuntimeError(
            "Qwen language LoRA topology is incomplete; missing projection groups: "
            + repr(missing_groups)
        )
    return targets


def _adapter_targets_match(
    source_targets: set[str], resolved_targets: list[str]
) -> bool:
    """Accept PEFT's saved leaf-name form or the original full module paths."""
    resolved = set(resolved_targets)
    resolved_leaves = {target.rsplit(".", 1)[-1] for target in resolved_targets}
    return source_targets == resolved or source_targets == resolved_leaves


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _freeze_and_validate_vision(model: Any) -> dict[str, Any]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual = getattr(getattr(base, "model", None), "visual", None)
    if visual is None:
        raise RuntimeError("Qwen multimodal model does not expose model.visual.")
    visual.eval()
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    # Trainer calls model.train(), which recursively toggles every child. Keep the
    # frozen tower in inference mode so its behavior also stays fixed.
    visual.train = types.MethodType(lambda self, mode=True: self, visual)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unsafe = [name for name in trainable if "lora_" not in name or "language_model" not in name]
    if unsafe:
        raise RuntimeError(f"Only language-model LoRA weights may be trainable: {unsafe[:8]}")
    return {
        "vision_training": visual.training,
        "trainable_parameter_names": trainable,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "all_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _load_processor_and_datasets(
    *,
    model: dict[str, Any],
    dataset: dict[str, Any],
    sft: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Path]]:
    from transformers import AutoProcessor

    token = os.environ.get(model["token_env"]) if model.get("token_env") else None
    hub_kwargs: dict[str, Any] = {
        "revision": model["revision"],
        "local_files_only": bool(model.get("local_files_only", False)),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    if token:
        hub_kwargs["token"] = token
    processor = AutoProcessor.from_pretrained(model["model_id"], **hub_kwargs)
    split_paths = {
        "train": Path(dataset["train_path"]).resolve(),
        "validation": Path(dataset["validation_path"]).resolve(),
        "test": Path(dataset["test_path"]).resolve(),
    }
    for split, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Critic {split} split does not exist: {path}")
    datasets, cache_manifest = load_or_build_critic_cache(
        processor=processor,
        split_paths=split_paths,
        cache_root=sft.get(
            "multimodal_cache_dir", "/dev/shm/svg_agentic_slm/critic_sft"
        ),
        model_id=model["model_id"],
        revision=model["revision"],
        max_seq_length=int(sft.get("max_seq_length", 32768)),
        enable_thinking=False,
    )
    return processor, datasets, cache_manifest, split_paths


def run_training(config_path: str | Path) -> dict[str, Any]:
    import torch
    import transformers
    from peft import (
        LoraConfig,
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_pt_utils import LengthGroupedSampler

    config_path = Path(config_path).resolve()
    config = load_yaml_config(config_path)
    train = _require_mapping(config.get("train"), "train")
    model_config = _require_mapping(train.get("model"), "train.model")
    dataset_config = _require_mapping(train.get("dataset"), "train.dataset")
    lora = _require_mapping(train.get("lora"), "train.lora")
    sft = _require_mapping(train.get("sft"), "train.sft")
    run_context = _require_mapping(train.get("run", {}), "train.run")
    required_model = ("model_id", "revision")
    if any(not model_config.get(key) for key in required_model):
        raise ValueError("train.model requires model_id and revision.")
    seed = int(sft.get("seed", 42))
    set_seed(seed)
    processor, datasets, cache_manifest, split_paths = _load_processor_and_datasets(
        model=model_config,
        dataset=dataset_config,
        sft=sft,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = str(model_config.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)
    load_in_4bit = bool(model_config.get("load_in_4bit", True))
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(
                model_config.get("bnb_4bit_use_double_quant", True)
            ),
        )
    token = os.environ.get(model_config["token_env"]) if model_config.get("token_env") else None
    model_kwargs: dict[str, Any] = {
        "revision": model_config["revision"],
        "local_files_only": bool(model_config.get("local_files_only", False)),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "dtype": dtype,
        "attn_implementation": str(model_config.get("attn_implementation", "sdpa")),
    }
    if token:
        model_kwargs["token"] = token
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": local_rank}
    loader = transformers.AutoModelForMultimodalLM
    model = loader.from_pretrained(model_config["model_id"], **model_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(sft.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    targets = resolve_qwen_language_lora_targets(model)
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("lora_alpha", 32)),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias=str(lora.get("bias", "none")),
        task_type="CAUSAL_LM",
    )
    init_adapter_from = sft.get("init_adapter_from")
    adapter_initialization: dict[str, Any] = {"mode": "fresh", "source": None}
    if init_adapter_from is None:
        model = get_peft_model(model, peft_config)
    else:
        adapter_path = Path(str(init_adapter_from)).resolve()
        if not adapter_path.is_dir():
            raise ValueError(f"sft.init_adapter_from is not a directory: {adapter_path}")
        source_config = PeftConfig.from_pretrained(adapter_path)
        source_targets = set(source_config.target_modules or [])
        if not _adapter_targets_match(source_targets, targets):
            raise ValueError("Initial critic adapter LoRA target topology does not match Qwen.")
        expected_adapter_values = {
            "r": int(lora.get("r", 16)),
            "lora_alpha": int(lora.get("lora_alpha", 32)),
            "lora_dropout": float(lora.get("lora_dropout", 0.05)),
            "bias": str(lora.get("bias", "none")),
        }
        for field, expected in expected_adapter_values.items():
            if getattr(source_config, field, None) != expected:
                raise ValueError(
                    f"Initial critic adapter {field} does not match: "
                    f"{getattr(source_config, field, None)!r} != {expected!r}"
                )
        source_manifest_path = adapter_path.parent / "training_manifest.json"
        if source_manifest_path.exists():
            source_manifest = json.loads(source_manifest_path.read_text())
            if (
                source_manifest.get("base_model_id") != model_config["model_id"]
                or source_manifest.get("base_model_revision") != model_config["revision"]
            ):
                raise ValueError("Initial critic adapter base-model provenance mismatch.")
            manifest_targets = source_manifest.get("lora", {}).get(
                "resolved_target_modules"
            )
            if manifest_targets is not None and set(manifest_targets) != set(targets):
                raise ValueError(
                    "Initial critic adapter manifest target topology does not match Qwen."
                )
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        adapter_model_path = adapter_path / "adapter_model.safetensors"
        adapter_initialization = {
            "mode": "adapter_weight_only",
            "source": str(adapter_path),
            "adapter_model_sha256": (
                _sha256(adapter_model_path) if adapter_model_path.is_file() else None
            ),
            "optimizer_state_loaded": False,
            "scheduler_state_loaded": False,
        }
    trainability = _freeze_and_validate_vision(model)
    loss_manifest = install_chunked_causal_lm_loss(
        model,
        chunk_size=int(sft.get("lm_head_loss_chunk_size", 64)),
        project_only_active_tokens=True,
        loss_backend=str(sft.get("lm_head_loss_backend", "auto")),
        allow_multimodal_inputs=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    output_dir = Path(sft.get("output_dir", "outputs/sft/qwen35_critic_distill")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_strategy = str(sft.get("eval_strategy", "steps"))
    save_strategy = str(sft.get("save_strategy", eval_strategy))
    patience = sft.get("early_stopping_patience", 2)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(sft.get("num_train_epochs", 3.0)),
        max_steps=int(sft.get("max_steps", -1)),
        per_device_train_batch_size=int(sft.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(sft.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(sft.get("gradient_accumulation_steps", 16)),
        learning_rate=float(sft.get("learning_rate", 1e-4)),
        weight_decay=float(sft.get("weight_decay", 0.01)),
        # Transformers 5 accepts a fractional warmup through warmup_steps.
        warmup_steps=float(sft.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(sft.get("lr_scheduler_type", "cosine")),
        logging_steps=int(sft.get("logging_steps", 10)),
        logging_first_step=True,
        logging_nan_inf_filter=bool(sft.get("logging_nan_inf_filter", False)),
        save_strategy=save_strategy,
        save_steps=int(sft.get("save_steps", 50)),
        save_total_limit=int(sft.get("save_total_limit", 2)),
        eval_strategy=eval_strategy,
        eval_steps=int(sft.get("eval_steps", 50)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        bf16=bool(sft.get("bf16", True)),
        fp16=bool(sft.get("fp16", False)),
        gradient_checkpointing=bool(sft.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=str(sft.get("optim", "paged_adamw_8bit")),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(sft.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=bool(sft.get("dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(
            sft.get("dataloader_persistent_workers", True)
        ),
        dataloader_prefetch_factor=int(sft.get("dataloader_prefetch_factor", 2)),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=list(sft.get("report_to", [])),
    )
    setattr(
        args,
        "length_grouping_batch_size",
        int(
            sft.get(
                "length_grouping_batch_size",
                args.per_device_train_batch_size * args.gradient_accumulation_steps,
            )
        ),
    )

    class CriticTrainer(Trainer):
        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            selected = train_dataset if train_dataset is not None else self.train_dataset
            return LengthGroupedSampler(
                self.args.length_grouping_batch_size,
                lengths=selected.verified_lengths,
            )

    logging_callback = DetailedLoggingCallback(
        output_dir / "logs",
        {
            "pipeline_id": run_context.get("pipeline_id"),
            "stage_id": run_context.get("stage_id", output_dir.name),
            "stage_kind": run_context.get("stage_kind"),
            "parent_stage_id": run_context.get("parent_stage_id"),
        },
    )
    callbacks = [logging_callback]
    if patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(patience),
                early_stopping_threshold=float(sft.get("early_stopping_threshold", 0.0)),
            )
        )
    trainer = CriticTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=CriticMultimodalCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.model_accepts_loss_kwargs = True
    train_result = trainer.train(resume_from_checkpoint=sft.get("resume_from_checkpoint"))
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"], metric_key_prefix="test"
    )
    logging_callback.finalize(trainer.state, test_metrics=dict(test_metrics))
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.accelerator.wait_for_everyone()
    manifest = {
        "schema_version": 1,
        "base_model_id": model_config["model_id"],
        "base_model_revision": model_config["revision"],
        "response_only_loss": True,
        "vision_frozen": True,
        "reference_evidence_in_student": False,
        "similarity_evidence_in_student": True,
        "run": run_context,
        "lora": {
            "r": int(lora.get("r", 16)),
            "lora_alpha": int(lora.get("lora_alpha", 32)),
            "lora_dropout": float(lora.get("lora_dropout", 0.05)),
            "resolved_target_modules": targets,
            "resolved_target_modules_sha256": hashlib.sha256(
                "\n".join(targets).encode()
            ).hexdigest(),
        },
        "trainability": trainability,
        "adapter_initialization": adapter_initialization,
        "loss_implementation": loss_manifest,
        "cache": cache_manifest,
        "dataset": {
            split: {"path": str(path), "sha256": _sha256(path), "count": len(datasets[split])}
            for split, path in split_paths.items()
        },
        "acceptance_calibration_insufficient": True,
        "required_deployment_gate": "measure pass-rate/reject-collapse on held-out test",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "train_metrics": dict(train_result.metrics),
        "test_metrics": dict(test_metrics),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
        },
    }
    if trainer.is_world_process_zero():
        processor_dir = output_dir / "processor"
        processor.save_pretrained(processor_dir)
        (output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
    trainer.accelerator.wait_for_everyone()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_critic_distill.yaml"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_training(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()


__all__ = ["resolve_qwen_language_lora_targets", "run_training"]
