"""Response-only QLoRA training for the text-to-SVG Generator."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from svg_agentic_slm.data.jsonl import read_jsonl
from svg_agentic_slm.prompts.system_prompts import get_svg_generator_system_prompt
from svg_agentic_slm.prompts.text_to_svg import build_text_to_svg_prompt
from svg_agentic_slm.svg.codec_backends import (
    NamedSpecialTokenSVGCodec,
)
from svg_agentic_slm.svg.discrete_runtime import (
    NamedTokenRegistration,
    create_gemma_named_codec,
    register_named_codec_tokens,
    validate_gemma_codec_backend,
)
from svg_agentic_slm.svg.official_discrete_cache import (
    OFFICIAL_CACHED_GEMMA_BACKEND_ID,
    OpenVGLabCacheConfig,
    load_cached_splits_records,
)
from svg_agentic_slm.train.chunked_causal_lm_loss import install_chunked_causal_lm_loss
from svg_agentic_slm.train.lora_config import LoRAConfig
from svg_agentic_slm.train.paged_optimizer_resume import (
    PagedOptimizerResumeMixin,
    initial_paged_optimizer_resume_manifest,
)

logger = logging.getLogger(__name__)

InstructionMode = Literal["description_only", "mixed_60_detail_40_description", "detail_only"]
TargetRepresentation = Literal["raw_xml", "omnisvg_discrete"]
TrainSamplingStrategy = Literal["random", "sequential", "group_by_length"]
StructuralResponseLossReduction = Literal["token", "sample"]
DetailTextNormalization = Literal["none", "mmsvg_list_repr_v1"]

_VERIFIED_LENGTH_SOURCE = "record.metadata.full_chat_token_length"
_RUNTIME_SERIALIZATION_LENGTH_SOURCE = "runtime_serialization.input_ids_length"

_AUTO_MODEL_CLASSES = {
    "causal_lm": "AutoModelForCausalLM",
    "multimodal_lm": "AutoModelForMultimodalLM",
}

_DISCRETE_SYSTEM_PROMPT = """You are an SVG generator trained with a registered discrete SVG codec.
Plan the composition internally, but return only the registered discrete SVG tokens.
Do not output XML, prose, Markdown, code fences, or reasoning. The token sequence must be
complete and decodable, beginning with the codec start token and ending with its end token.
Follow the user's requested objects, layout, geometry, colors, and style exactly."""

_OFFICIAL_NAMED_TOKEN = re.compile(r"<svgovg4b:([0-9]+)>")
_OFFICIAL_TRAINING_ID_MIN = 151938
_OFFICIAL_TRAINING_BOS_ID = 196998
_OFFICIAL_TRAINING_EOS_ID = 196999
_OFFICIAL_REGISTERED_ID_MIN = 262144
_OFFICIAL_REGISTERED_COORDINATE_MIN = 262149
_OFFICIAL_REGISTERED_COORDINATE_MAX = 302148
_OFFICIAL_REGISTERED_PRE_COLOR_GAP_MIN = 302149
_OFFICIAL_REGISTERED_PRE_COLOR_GAP_MAX = 302151
_OFFICIAL_REGISTERED_COLOR_MIN = 302152
_OFFICIAL_REGISTERED_COLOR_MAX = 306249
_OFFICIAL_REGISTERED_PRE_ARC_GAP_MIN = 306250
_OFFICIAL_REGISTERED_PRE_ARC_GAP_MAX = 306641
_OFFICIAL_REGISTERED_ARC_MIN = 306642
_OFFICIAL_REGISTERED_ARC_MAX = 306741
_RESPONSE_ROLE_OTHER = 0
_RESPONSE_ROLE_EOS = 1
_RESPONSE_ROLE_PATH_COLOR_TERMINATOR = 2
_OFFICIAL_INFERENCE_COMMAND_ARGUMENTS: Mapping[int, tuple[str, ...]] = {
    151939: ("coordinate", "coordinate"),
    151940: ("coordinate",),
    151941: ("coordinate", "coordinate", "coordinate"),
    151942: ("coordinate", "arc", "arc", "arc", "coordinate"),
    151943: ("coordinate",),
}
_STRUCTURAL_RESPONSE_LOSS_CONTRACT = {
    "schema_version": 2,
    "scope": "official_cached_omnisvg_discrete_response_labels_only",
    "registered_to_training": "strict ordered <svgovg4b:N> mapping",
    "grammar": "official inference body dialect with exactly +1 applied to training body IDs",
    "roles": {
        str(_RESPONSE_ROLE_OTHER): "other_or_ignored",
        str(_RESPONSE_ROLE_EOS): "outer_labeled_eos",
        str(_RESPONSE_ROLE_PATH_COLOR_TERMINATOR): (
            "color token that closes a grammar-complete path containing at least one command"
        ),
    },
    "shift_alignment": "weight at labels[:,t] applies to logits[:,t-1,:]",
    "per_sample_training_objectives": {
        "eos": {
            "numerator": "sum(weight * exact_token_cross_entropy)",
            "denominator": "sum(active_weight)",
        },
        "path_terminator_class_mass": {
            "numerator": (
                "sum(base_exact_token_ce) + "
                "(weight-1) * sum(-log p(valid_color_class))"
            ),
            "denominator": (
                "active_token_count + (weight-1) * completed_path_count"
            ),
        },
    },
    "reductions": {
        "token": (
            "sum all sample numerators / sum all sample denominators in the "
            "accumulation window"
        ),
        "sample": (
            "mean each sample numerator/denominator across all samples in the "
            "accumulation window"
        ),
    },
    "recommended_reduction": "sample",
    "valid_registered_color_class": {
        "minimum": _OFFICIAL_REGISTERED_COLOR_MIN,
        "maximum": _OFFICIAL_REGISTERED_COLOR_MAX,
        "count": _OFFICIAL_REGISTERED_COLOR_MAX - _OFFICIAL_REGISTERED_COLOR_MIN + 1,
        "adjacent_coordinate_range_excluded": [
            _OFFICIAL_REGISTERED_COORDINATE_MIN,
            _OFFICIAL_REGISTERED_COORDINATE_MAX,
        ],
        "pre_color_gap_excluded": [
            _OFFICIAL_REGISTERED_PRE_COLOR_GAP_MIN,
            _OFFICIAL_REGISTERED_PRE_COLOR_GAP_MAX,
        ],
        "pre_arc_gap_excluded": [
            _OFFICIAL_REGISTERED_PRE_ARC_GAP_MIN,
            _OFFICIAL_REGISTERED_PRE_ARC_GAP_MAX,
        ],
        "adjacent_arc_range_excluded": [
            _OFFICIAL_REGISTERED_ARC_MIN,
            _OFFICIAL_REGISTERED_ARC_MAX,
        ],
    },
    "evaluation_reduction": "existing unweighted model causal-LM loss",
    "gradient_accumulation": {
        "normalization_scope": "entire accumulation window for the configured reduction",
        "microbatch_value": "additive numerator / shared window denominator",
        "transformers_contract": "Trainer prefetches one complete accumulation window",
        "accelerate_contract": "Trainer configures Accelerator gradient accumulation to one",
        "extra_gradient_or_reporting_scale": "forbidden",
    },
    "ignore_index": -100,
    "truncation": "forbidden",
}
_STRUCTURAL_RESPONSE_LOSS_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _STRUCTURAL_RESPONSE_LOSS_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass
class SFTConfig:
    output_dir: str = "./outputs/sft/gemma4_raw"
    num_train_epochs: float = 1.0
    max_steps: int = -1
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
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    prediction_loss_only: bool = True
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.0
    max_seq_length: int = 8192
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 42
    dataloader_num_workers: int = 2
    torch_empty_cache_steps: int | None = None
    init_adapter_from: str | None = None
    resume_from_checkpoint: str | None = None
    rehydrate_paged_optimizer_state_on_resume: bool = True
    merge_adapter: bool = False
    report_to: list[str] | None = None
    chunked_lm_head_loss: bool = False
    lm_head_loss_chunk_size: int = 256
    response_eos_loss_weight: float = 1.0
    response_path_terminator_class_mass_weight: float = 1.0
    structural_response_loss_reduction: StructuralResponseLossReduction = "token"
    detail_text_normalization: DetailTextNormalization = "none"
    train_sampling_strategy: TrainSamplingStrategy = "random"
    length_grouping_batch_size: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("do_train", "do_eval", "do_predict"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"sft.{field_name} must be boolean.")
        self.response_eos_loss_weight = _validated_response_loss_weight(
            self.response_eos_loss_weight,
            field_name="response_eos_loss_weight",
            maximum=8.0,
        )
        self.response_path_terminator_class_mass_weight = (
            _validated_response_loss_weight(
                self.response_path_terminator_class_mass_weight,
                field_name="response_path_terminator_class_mass_weight",
                maximum=4.0,
            )
        )
        if (
            self.response_eos_loss_weight > 1.0
            and self.response_path_terminator_class_mass_weight > 1.0
        ):
            raise ValueError(
                "sft.response_eos_loss_weight and "
                "sft.response_path_terminator_class_mass_weight cannot both exceed 1.0 "
                "before the follow-up EP arms are approved."
            )
        self.structural_response_loss_reduction = (
            _validated_structural_response_loss_reduction(
                self.structural_response_loss_reduction
            )
        )
        self.detail_text_normalization = _validated_detail_text_normalization(
            self.detail_text_normalization
        )
        if (
            self.structural_response_loss_reduction == "sample"
            and not _structural_response_loss_enabled(self)
        ):
            raise ValueError(
                "sft.structural_response_loss_reduction=sample requires an enabled "
                "structural response-loss weight."
            )
        for field_name in ("init_adapter_from", "resume_from_checkpoint"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"sft.{field_name} must be a non-empty path or null.")
        if self.init_adapter_from is not None and self.resume_from_checkpoint is not None:
            raise ValueError(
                "sft.init_adapter_from and sft.resume_from_checkpoint are mutually exclusive."
            )
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or (self.max_steps != -1 and self.max_steps <= 0)
        ):
            raise ValueError("sft.max_steps must be -1 or a positive integer.")
        grouping_batch_size = self.length_grouping_batch_size
        if grouping_batch_size is not None and (
            isinstance(grouping_batch_size, bool)
            or not isinstance(grouping_batch_size, int)
            or grouping_batch_size <= 0
        ):
            raise ValueError("sft.length_grouping_batch_size must be a positive integer or null.")
        if grouping_batch_size is not None and self.train_sampling_strategy != "group_by_length":
            raise ValueError(
                "sft.length_grouping_batch_size requires "
                "sft.train_sampling_strategy=group_by_length."
            )
        if not isinstance(self.rehydrate_paged_optimizer_state_on_resume, bool):
            raise ValueError(
                "sft.rehydrate_paged_optimizer_state_on_resume must be boolean."
            )
        if self.torch_empty_cache_steps is not None and (
            isinstance(self.torch_empty_cache_steps, bool)
            or not isinstance(self.torch_empty_cache_steps, int)
            or self.torch_empty_cache_steps <= 0
        ):
            raise ValueError(
                "sft.torch_empty_cache_steps must be a positive integer or null."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTConfig:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


def _validated_response_loss_weight(
    value: Any,
    *,
    field_name: str,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sft.{field_name} must be a finite number in [1.0, {maximum}].")
    normalized = float(value)
    if not math.isfinite(normalized) or not 1.0 <= normalized <= maximum:
        raise ValueError(f"sft.{field_name} must be in [1.0, {maximum}].")
    return normalized


def _validated_structural_response_loss_reduction(
    value: Any,
) -> StructuralResponseLossReduction:
    if not isinstance(value, str) or value not in ("token", "sample"):
        raise ValueError("sft.structural_response_loss_reduction must be token or sample.")
    return value


def _validated_detail_text_normalization(value: Any) -> DetailTextNormalization:
    if not isinstance(value, str) or value not in ("none", "mmsvg_list_repr_v1"):
        raise ValueError(
            "sft.detail_text_normalization must be none or mmsvg_list_repr_v1."
        )
    return value


def _structural_response_loss_enabled(config: SFTConfig) -> bool:
    return bool(
        config.response_eos_loss_weight > 1.0
        or config.response_path_terminator_class_mass_weight > 1.0
    )


def _official_inference_token_kind(token_id: int) -> str:
    if 151939 <= token_id < 151944:
        return "command"
    if 151944 <= token_id < 191944:
        return "coordinate"
    if 191947 <= token_id < 196045:
        return "path_color_terminator"
    if 196437 <= token_id < 196537:
        return "arc"
    raise ValueError(f"Official discrete response contains invalid body token ID {token_id}.")


def _official_discrete_response_loss_annotations(
    labels: Sequence[int],
    *,
    tokenizer: Any,
    eos_weight: float,
    path_terminator_class_mass_weight: float,
) -> tuple[list[float], list[int], dict[str, int]]:
    eos_weight = _validated_response_loss_weight(
        eos_weight,
        field_name="response_eos_loss_weight",
        maximum=8.0,
    )
    path_terminator_class_mass_weight = _validated_response_loss_weight(
        path_terminator_class_mass_weight,
        field_name="response_path_terminator_class_mass_weight",
        maximum=4.0,
    )
    if eos_weight > 1.0 and path_terminator_class_mass_weight > 1.0:
        raise ValueError("EOS and path-terminator class-mass arms cannot be enabled together.")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)):
        raise TypeError("Response labels must be a sequence of integers.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in labels):
        raise TypeError("Response labels must contain only integers.")
    labeled_positions = [index for index, value in enumerate(labels) if value != -100]
    if not labeled_positions:
        raise ValueError("Structural response loss requires labeled response tokens.")
    if labeled_positions != list(range(labeled_positions[0], labeled_positions[-1] + 1)):
        raise ValueError("Structural response labels must form one contiguous target span.")
    registered_ids = [int(labels[index]) for index in labeled_positions]
    converted = tokenizer.convert_ids_to_tokens(registered_ids)
    if isinstance(converted, str):
        converted = [converted]
    if not isinstance(converted, list) or len(converted) != len(registered_ids):
        raise RuntimeError("Tokenizer did not return one named token per labeled response ID.")
    training_ids: list[int] = []
    for registered_id, token in zip(registered_ids, converted):
        if not isinstance(token, str):
            raise RuntimeError("Official discrete response token is not a string.")
        match = _OFFICIAL_NAMED_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"Labeled response token {token!r} is outside the OmniSVG namespace.")
        training_id = int(match.group(1))
        expected_registered_id = _OFFICIAL_REGISTERED_ID_MIN + (
            training_id - _OFFICIAL_TRAINING_ID_MIN
        )
        if registered_id != expected_registered_id:
            raise ValueError("Registered OmniSVG token order differs from the training contract.")
        training_ids.append(training_id)
    if (
        len(training_ids) < 3
        or training_ids[0] != _OFFICIAL_TRAINING_BOS_ID
        or training_ids[-1] != _OFFICIAL_TRAINING_EOS_ID
    ):
        raise ValueError("Official discrete response lacks exact BOS/EOS framing.")
    if (
        _OFFICIAL_TRAINING_BOS_ID in training_ids[1:]
        or _OFFICIAL_TRAINING_EOS_ID in training_ids[:-1]
    ):
        raise ValueError("Official discrete response contains embedded BOS/EOS framing.")

    roles = [_RESPONSE_ROLE_OTHER] * len(labels)
    roles[labeled_positions[-1]] = _RESPONSE_ROLE_EOS
    body = training_ids[1:-1]
    index = 0
    commands_in_path = 0
    path_terminator_count = 0
    while index < len(body):
        inference_id = body[index] + 1
        kind = _official_inference_token_kind(inference_id)
        if kind == "path_color_terminator":
            if commands_in_path == 0:
                raise ValueError("Official discrete response contains an orphan color terminator.")
            target_offset = index + 1
            roles[labeled_positions[target_offset]] = _RESPONSE_ROLE_PATH_COLOR_TERMINATOR
            path_terminator_count += 1
            commands_in_path = 0
            index += 1
            continue
        if kind != "command":
            raise ValueError(
                "Official discrete response expected a command or completed-path color terminator."
            )
        expected_arguments = _OFFICIAL_INFERENCE_COMMAND_ARGUMENTS[inference_id]
        if index + len(expected_arguments) >= len(body):
            raise ValueError("Official discrete response contains a truncated command.")
        for offset, expected_kind in enumerate(expected_arguments, start=1):
            observed_kind = _official_inference_token_kind(body[index + offset] + 1)
            if observed_kind != expected_kind:
                raise ValueError(
                    "Official discrete response command argument has the wrong token class."
                )
        commands_in_path += 1
        index += 1 + len(expected_arguments)
    if commands_in_path:
        raise ValueError("Official discrete response ends with an unterminated path.")
    if path_terminator_count == 0:
        raise ValueError("Official discrete response contains no completed path.")

    weights = [1.0] * len(labels)
    for index, role in enumerate(roles):
        if role == _RESPONSE_ROLE_EOS:
            weights[index] = eos_weight
    return weights, roles, {
        "labeled_token_count": len(labeled_positions),
        "eos_count": 1,
        "path_color_terminator_count": path_terminator_count,
    }


def _structural_weighted_causal_lm_loss(
    logits: Any,
    labels: Any,
    loss_weights: Any,
    response_token_roles: Any = None,
    path_terminator_class_mass_weight: float = 1.0,
    reduction: StructuralResponseLossReduction = "token",
    normalization_denominator: Any = None,
) -> Any:
    import torch
    import torch.nn.functional as functional

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("Structural loss logits must have shape [batch, sequence, vocabulary].")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("Structural loss labels must have shape [batch, sequence].")
    if not isinstance(loss_weights, torch.Tensor) or loss_weights.ndim != 2:
        raise ValueError("Structural loss weights must have shape [batch, sequence].")
    if tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError("Structural loss label shape does not match logits.")
    if tuple(loss_weights.shape) != tuple(labels.shape):
        raise ValueError("Structural loss weight shape does not match labels.")
    path_terminator_class_mass_weight = _validated_response_loss_weight(
        path_terminator_class_mass_weight,
        field_name="response_path_terminator_class_mass_weight",
        maximum=4.0,
    )
    reduction = _validated_structural_response_loss_reduction(reduction)
    if response_token_roles is not None and (
        not isinstance(response_token_roles, torch.Tensor)
        or tuple(response_token_roles.shape) != tuple(labels.shape)
    ):
        raise ValueError("Structural response role shape does not match labels.")
    if path_terminator_class_mass_weight > 1.0 and response_token_roles is None:
        raise ValueError("Path-terminator class-mass loss requires response token roles.")
    if labels.shape[1] < 2:
        raise ValueError("Structural shifted loss requires at least two sequence positions.")
    if not bool(torch.isfinite(loss_weights).all()) or bool((loss_weights <= 0).any()):
        raise ValueError("Structural loss weights must be finite and strictly positive.")

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous().to(
        device=shift_logits.device,
        dtype=shift_logits.dtype,
    )
    active = shift_labels.ne(-100)
    if not bool(active.any()):
        raise ValueError("Structural shifted loss has no active response labels.")
    flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.view(-1)
    active_weights = shift_weights[active]
    if (
        reduction == "token"
        and torch.equal(active_weights, torch.ones_like(active_weights))
        and path_terminator_class_mass_weight == 1.0
        and normalization_denominator is None
    ):
        return functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    per_token = functional.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    token_denominator = active_weights.sum()
    if not bool(torch.isfinite(token_denominator)) or bool(token_denominator <= 0):
        raise ValueError("Structural loss active-weight denominator is invalid.")
    exact_numerator = (per_token[active] * active_weights).sum()
    numerator = exact_numerator
    local_denominator = token_denominator
    sample_numerators = None
    sample_denominators = None
    if reduction == "sample":
        sample_numerators = (per_token * shift_weights * active).sum(dim=1)
        sample_denominators = (shift_weights * active).sum(dim=1)
    if path_terminator_class_mass_weight > 1.0:
        if not torch.equal(active_weights, torch.ones_like(active_weights)):
            raise ValueError(
                "EOS exact weighting and path class-mass weighting cannot be combined."
            )
        if shift_logits.shape[-1] <= _OFFICIAL_REGISTERED_COLOR_MAX:
            raise ValueError("Model vocabulary does not contain the full registered color class.")
        shift_roles = response_token_roles[:, 1:].to(device=shift_labels.device)
        if bool(
            (
                (shift_roles < _RESPONSE_ROLE_OTHER)
                | (shift_roles > _RESPONSE_ROLE_PATH_COLOR_TERMINATOR)
            ).any()
        ):
            raise ValueError("Structural response role tensor contains an unknown role ID.")
        path_positions = active & shift_roles.eq(_RESPONSE_ROLE_PATH_COLOR_TERMINATOR)
        if not bool(path_positions.any()):
            raise ValueError("Path class-mass arm has no completed-path positions.")
        path_targets = shift_labels[path_positions]
        if bool(
            (
                (path_targets < _OFFICIAL_REGISTERED_COLOR_MIN)
                | (path_targets > _OFFICIAL_REGISTERED_COLOR_MAX)
            ).any()
        ):
            raise ValueError("Completed-path role does not point to a registered color target.")
        path_logits = shift_logits[path_positions]
        log_all_mass = torch.logsumexp(path_logits, dim=-1)
        log_color_mass = torch.logsumexp(
            path_logits[
                :,
                _OFFICIAL_REGISTERED_COLOR_MIN : _OFFICIAL_REGISTERED_COLOR_MAX + 1,
            ],
            dim=-1,
        )
        class_mass_nll = log_all_mass - log_color_mass
        auxiliary_multiplier = path_terminator_class_mass_weight - 1.0
        numerator = exact_numerator + auxiliary_multiplier * class_mass_nll.sum()
        local_denominator = token_denominator + auxiliary_multiplier * path_positions.sum().to(
            dtype=token_denominator.dtype
        )
        if reduction == "sample":
            if sample_numerators is None or sample_denominators is None:
                raise RuntimeError("Sample structural loss statistics were not initialized.")
            class_mass_by_position = torch.zeros_like(per_token)
            class_mass_by_position[path_positions] = class_mass_nll
            sample_numerators = sample_numerators + auxiliary_multiplier * (
                class_mass_by_position.sum(dim=1)
            )
            sample_denominators = sample_denominators + auxiliary_multiplier * (
                path_positions.sum(dim=1).to(dtype=sample_denominators.dtype)
            )
    if reduction == "sample":
        if sample_numerators is None or sample_denominators is None:
            raise RuntimeError("Sample structural loss statistics were not initialized.")
        if bool((sample_denominators <= 0).any()):
            raise ValueError(
                "Sample-reduced structural loss requires active response labels in every sample."
            )
        numerator = (sample_numerators / sample_denominators).sum()
        local_denominator = torch.tensor(
            labels.shape[0],
            device=numerator.device,
            dtype=numerator.dtype,
        )
    if normalization_denominator is None:
        normalization_denominator = local_denominator
    elif torch.is_tensor(normalization_denominator):
        normalization_denominator = normalization_denominator.to(
            device=numerator.device,
            dtype=numerator.dtype,
        )
    else:
        normalization_denominator = torch.tensor(
            float(normalization_denominator),
            device=numerator.device,
            dtype=numerator.dtype,
        )
    if (
        not bool(torch.isfinite(normalization_denominator))
        or bool(normalization_denominator <= 0)
        or bool(normalization_denominator < local_denominator)
    ):
        raise ValueError("Structural accumulation-window denominator is invalid.")
    return numerator / normalization_denominator


def _structural_accumulation_window_denominator(
    batch_samples: Sequence[Mapping[str, Any]],
    *,
    path_terminator_class_mass_weight: float,
    reduction: StructuralResponseLossReduction = "token",
) -> Any:
    import torch

    path_terminator_class_mass_weight = _validated_response_loss_weight(
        path_terminator_class_mass_weight,
        field_name="response_path_terminator_class_mass_weight",
        maximum=4.0,
    )
    reduction = _validated_structural_response_loss_reduction(reduction)
    denominator: Any = None
    for batch in batch_samples:
        labels = batch.get("labels")
        loss_weights = batch.get("loss_weights")
        roles = batch.get("response_token_roles")
        if not all(
            isinstance(value, torch.Tensor) for value in (labels, loss_weights, roles)
        ):
            raise ValueError(
                "Structural accumulation batches require tensor labels, weights, and roles."
            )
        if tuple(labels.shape) != tuple(loss_weights.shape) or tuple(
            labels.shape
        ) != tuple(roles.shape):
            raise ValueError("Structural accumulation batch shapes do not match.")
        active = labels[:, 1:].ne(-100)
        if not bool(active.any()):
            raise ValueError("Structural accumulation microbatch has no shifted response labels.")
        if reduction == "sample" and bool(active.sum(dim=1).eq(0).any()):
            raise ValueError(
                "Sample-reduced structural loss requires active response labels in every sample."
            )
        active_weights = loss_weights[:, 1:][active]
        if not bool(torch.isfinite(active_weights).all()) or bool((active_weights <= 0).any()):
            raise ValueError("Structural accumulation weights must be finite and positive.")
        weighted_local = active_weights.sum(dtype=torch.float32)
        if path_terminator_class_mass_weight > 1.0:
            if not torch.equal(active_weights, torch.ones_like(active_weights)):
                raise ValueError(
                    "EOS and path structural arms cannot share an accumulation window."
                )
            shifted_roles = roles[:, 1:]
            path_positions = active & shifted_roles.eq(_RESPONSE_ROLE_PATH_COLOR_TERMINATOR)
            if not bool(path_positions.any()):
                raise ValueError("Path structural microbatch has no completed-path role.")
            path_targets = labels[:, 1:][path_positions]
            if bool(
                (
                    (path_targets < _OFFICIAL_REGISTERED_COLOR_MIN)
                    | (path_targets > _OFFICIAL_REGISTERED_COLOR_MAX)
                ).any()
            ):
                raise ValueError("Path structural role points outside the valid color class.")
            weighted_local = weighted_local + (
                path_terminator_class_mass_weight - 1.0
            ) * path_positions.sum().to(dtype=torch.float32)
        local = (
            weighted_local
            if reduction == "token"
            else torch.tensor(labels.shape[0], dtype=torch.float32, device=labels.device)
        )
        denominator = local if denominator is None else denominator + local
    if denominator is None or not bool(torch.isfinite(denominator)) or bool(denominator <= 0):
        raise ValueError("Structural accumulation window has no valid denominator.")
    return denominator


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


@dataclass(frozen=True)
class _InstructionSelection:
    text: str
    selected_field: Literal["description", "detail"]
    legacy_detail_candidate: bool = False
    legacy_detail_normalized: bool = False


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
        codec: NamedSpecialTokenSVGCodec | None,
        response_eos_loss_weight: float = 1.0,
        response_path_terminator_class_mass_weight: float = 1.0,
        detail_text_normalization: DetailTextNormalization = "none",
    ) -> None:
        self._records = records
        self._tokenizer = tokenizer
        self._instruction_mode = instruction_mode
        self._target_representation = target_representation
        self._max_seq_length = max_seq_length
        self._codec = codec
        self._detail_text_normalization = _validated_detail_text_normalization(
            detail_text_normalization
        )
        self._selected_instructions: tuple[str, ...] | None = None
        self._instruction_selection_manifest: dict[str, Any] | None = None
        self._response_eos_loss_weight = _validated_response_loss_weight(
            response_eos_loss_weight,
            field_name="response_eos_loss_weight",
            maximum=8.0,
        )
        self._response_path_terminator_class_mass_weight = (
            _validated_response_loss_weight(
                response_path_terminator_class_mass_weight,
                field_name="response_path_terminator_class_mass_weight",
                maximum=4.0,
            )
        )
        self._structural_response_loss_enabled = bool(
            self._response_eos_loss_weight > 1.0
            or self._response_path_terminator_class_mass_weight > 1.0
        )
        if (
            self._response_eos_loss_weight > 1.0
            and self._response_path_terminator_class_mass_weight > 1.0
        ):
            raise ValueError("EOS and path/color terminator weights cannot be enabled together.")
        if self._structural_response_loss_enabled and target_representation != "omnisvg_discrete":
            raise ValueError("Structural response weights require omnisvg_discrete targets.")
        self._structural_role_counts: Counter[str] = Counter()
        self._verified_length_contract: dict[str, Any] | None = None
        self._metadata_length_diagnostic: dict[str, Any] | None = None
        if target_representation == "raw_xml":
            self._verified_length_source = _VERIFIED_LENGTH_SOURCE
            self._verified_length_validation = "positive_integer"
            self._verified_lengths = tuple(
                _verified_full_chat_token_length(row, index=index)
                for index, row in enumerate(records)
            )
        else:
            self._verified_length_source = _RUNTIME_SERIALIZATION_LENGTH_SOURCE
            self._verified_length_validation = (
                "exact_runtime_serialization_without_truncation"
            )
            verified_lengths: list[int] = []
            for index in range(len(records)):
                serialized = self._serialize_record(index)
                verified_lengths.append(len(serialized["input_ids"]))
                if self._structural_response_loss_enabled:
                    self._structural_role_counts.update(
                        serialized["response_token_role_counts"]
                    )
            self._verified_lengths = tuple(verified_lengths)
            try:
                tokenizer_vocabulary_size = len(tokenizer)
            except TypeError:
                tokenizer_vocabulary_size = None
            self._verified_length_contract = {
                "target_representation": target_representation,
                "instruction_mode": instruction_mode,
                "max_seq_length": max_seq_length,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
                "serialization": "apply_chat_template(tokenize=True, add_generation_prompt=False)",
                "truncation": False,
            }
            metadata_values: list[int | None] = []
            for row in records:
                metadata = row.get("metadata")
                value = (
                    metadata.get("full_chat_token_length")
                    if isinstance(metadata, Mapping)
                    else None
                )
                metadata_values.append(
                    value
                    if isinstance(value, int) and not isinstance(value, bool) and value > 0
                    else None
                )
            self._metadata_length_diagnostic = {
                "source": _VERIFIED_LENGTH_SOURCE,
                "used_for_sampling": False,
                "present_positive_integer_count": sum(
                    value is not None for value in metadata_values
                ),
                "missing_or_invalid_count": sum(
                    value is None for value in metadata_values
                ),
                "match_count": sum(
                    value == actual
                    for value, actual in zip(metadata_values, self._verified_lengths)
                    if value is not None
                ),
                "mismatch_count": sum(
                    value != actual
                    for value, actual in zip(metadata_values, self._verified_lengths)
                    if value is not None
                ),
            }
        compact_lengths = json.dumps(self._verified_lengths, separators=(",", ":"))
        self._verified_lengths_sha256 = hashlib.sha256(
            compact_lengths.encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def verified_lengths(self) -> tuple[int, ...]:
        """Return immutable, pre-validated lengths without tokenizing dataset rows."""

        return self._verified_lengths

    def verified_length_manifest(self) -> dict[str, Any]:
        manifest = {
            "source": self._verified_length_source,
            "validation": self._verified_length_validation,
            "count": len(self._verified_lengths),
            "minimum": min(self._verified_lengths),
            "maximum": max(self._verified_lengths),
            "sum": sum(self._verified_lengths),
            "ordered_values_sha256": self._verified_lengths_sha256,
        }
        if self._verified_length_contract is not None:
            manifest["contract"] = self._verified_length_contract
            manifest["actual_maximum"] = max(self._verified_lengths)
            manifest["metadata_diagnostic"] = self._metadata_length_diagnostic
        return manifest

    def structural_response_role_manifest(self) -> dict[str, Any] | None:
        if not self._structural_response_loss_enabled:
            return None
        return {
            "sample_count": len(self._records),
            "labeled_token_count": int(self._structural_role_counts["labeled_token_count"]),
            "eos_count": int(self._structural_role_counts["eos_count"]),
            "path_color_terminator_count": int(
                self._structural_role_counts["path_color_terminator_count"]
            ),
            "contract_sha256": _STRUCTURAL_RESPONSE_LOSS_CONTRACT_SHA256,
        }

    def instruction_selection_manifest(self) -> dict[str, Any]:
        self._ensure_instruction_selections()
        if self._instruction_selection_manifest is None:
            raise RuntimeError("Instruction selection manifest was not initialized.")
        return dict(self._instruction_selection_manifest)

    def _ensure_instruction_selections(self) -> None:
        if self._selected_instructions is not None:
            return
        selections = tuple(
            _select_instruction_with_provenance(
                row,
                self._instruction_mode,
                detail_text_normalization=self._detail_text_normalization,
            )
            for row in self._records
        )
        selected_instructions = tuple(selection.text for selection in selections)
        compact_instructions = json.dumps(
            selected_instructions,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._selected_instructions = selected_instructions
        self._instruction_selection_manifest = {
            "instruction_mode": self._instruction_mode,
            "detail_text_normalization": self._detail_text_normalization,
            "selected_instruction_count": len(selections),
            "selected_instruction_ordered_sha256": hashlib.sha256(
                compact_instructions.encode("utf-8")
            ).hexdigest(),
            "selected_instruction_hash_serialization": (
                "utf8_json_array_ensure_ascii_false_compact"
            ),
            "selected_detail_count": sum(
                selection.selected_field == "detail" for selection in selections
            ),
            "legacy_detail_candidate_count": sum(
                selection.legacy_detail_candidate for selection in selections
            ),
            "legacy_detail_normalized_count": sum(
                selection.legacy_detail_normalized for selection in selections
            ),
            "normalization_scope": "selected_detail_only",
            "raw_records_mutated": False,
        }

    def _serialize_record(self, index: int) -> dict[str, Any]:
        row = self._records[index]
        self._ensure_instruction_selections()
        if self._selected_instructions is None:
            raise RuntimeError("Instruction selection was not initialized.")
        instruction = self._selected_instructions[index]
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
        full_messages = prefix_messages + [{"role": "assistant", "content": target}]
        rendered = self._tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("Tokenizer chat template returned invalid rendered text.")
        target_start = rendered.find(target)
        if target_start < 0 or rendered.find(target, target_start + len(target)) >= 0:
            raise RuntimeError(
                "Assistant target must occur exactly once in the rendered chat template."
            )
        target_end = target_start + len(target)
        try:
            offset_encoded = self._tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Tokenizer must support offset mapping for response-only SFT."
            ) from exc
        if not isinstance(offset_encoded, Mapping):
            raise RuntimeError("Tokenizer offset result must be a mapping.")
        if "input_ids" not in offset_encoded or "offset_mapping" not in offset_encoded:
            raise RuntimeError("Tokenizer offset result is missing required fields.")
        offset_ids = offset_encoded["input_ids"]
        offsets = offset_encoded["offset_mapping"]
        if hasattr(offset_ids, "tolist"):
            offset_ids = offset_ids.tolist()
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offset_ids and isinstance(offset_ids[0], list):
            if len(offset_ids) != 1:
                raise RuntimeError("Tokenizer returned multiple offset-tokenized examples.")
            offset_ids = offset_ids[0]
        if (
            offsets
            and isinstance(offsets[0], (list, tuple))
            and offsets[0]
            and isinstance(offsets[0][0], (list, tuple))
        ):
            if len(offsets) != 1:
                raise RuntimeError("Tokenizer returned multiple offset mappings.")
            offsets = offsets[0]
        if (
            not isinstance(offset_ids, list)
            or not offset_ids
            or not all(isinstance(value, int) for value in offset_ids)
            or not isinstance(offsets, list)
            or len(offsets) != len(offset_ids)
            or not all(
                isinstance(span, (list, tuple))
                and len(span) == 2
                and all(isinstance(value, int) for value in span)
                for span in offsets
            )
        ):
            raise RuntimeError("Tokenizer returned invalid offset mapping data.")
        full_ids = _template_ids(
            self._tokenizer,
            full_messages,
            add_generation_prompt=False,
        )
        if offset_ids != full_ids:
            raise RuntimeError(
                "Offset tokenization does not reproduce the chat template token IDs."
            )
        if len(full_ids) > self._max_seq_length:
            record_id = row.get("metadata", {}).get("record_id", index)
            raise ValueError(
                f"SFT record {record_id!r} has {len(full_ids)} tokens, exceeding "
                f"max_seq_length={self._max_seq_length}; do not truncate SVG targets."
            )
        labels = [-100] * len(full_ids)
        for token_index, (span_start, span_end) in enumerate(offsets):
            if span_end > target_start and span_start < target_end and span_start != span_end:
                labels[token_index] = full_ids[token_index]
        if not any(label != -100 for label in labels):
            raise RuntimeError("SFT record has no assistant target tokens.")
        result: dict[str, Any] = {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }
        if self._structural_response_loss_enabled:
            loss_weights, roles, counts = _official_discrete_response_loss_annotations(
                labels,
                tokenizer=self._tokenizer,
                eos_weight=self._response_eos_loss_weight,
                path_terminator_class_mass_weight=(
                    self._response_path_terminator_class_mass_weight
                ),
            )
            result["loss_weights"] = loss_weights
            result["response_token_roles"] = roles
            result["response_token_role_counts"] = counts
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._serialize_record(index)


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
        weighted_features = ["loss_weights" in feature for feature in features]
        if any(weighted_features) and not all(weighted_features):
            raise ValueError("A batch cannot mix structural-weighted and unweighted records.")
        loss_weights = (
            torch.ones((batch_size, max_length), dtype=torch.float32)
            if all(weighted_features)
            else None
        )
        response_token_roles = (
            torch.zeros((batch_size, max_length), dtype=torch.long)
            if all(weighted_features)
            else None
        )
        for row_index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row_index, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row_index, :length] = 1
            labels[row_index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            if loss_weights is not None:
                feature_weights = feature["loss_weights"]
                feature_roles = feature.get("response_token_roles")
                if (
                    not isinstance(feature_weights, list)
                    or len(feature_weights) != length
                    or not isinstance(feature_roles, list)
                    or len(feature_roles) != length
                ):
                    raise ValueError(
                        "Structural response weight/role shape differs from input IDs."
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in feature_weights
                ):
                    raise ValueError("Structural response weights must be finite and positive.")
                loss_weights[row_index, :length] = torch.tensor(
                    feature_weights,
                    dtype=torch.float32,
                )
                response_token_roles[row_index, :length] = torch.tensor(
                    feature_roles,
                    dtype=torch.long,
                )
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if loss_weights is not None:
            batch["loss_weights"] = loss_weights
            batch["response_token_roles"] = response_token_roles
        return batch


class _StructuralResponseLossMixin:
    _svg_structural_response_loss_enabled = False
    _svg_path_terminator_class_mass_weight = 1.0
    _svg_structural_response_loss_reduction: StructuralResponseLossReduction = "token"

    def get_batch_samples(self, epoch_iterator: Any, num_batches: int, device: Any) -> Any:
        batch_samples, num_items_in_batch = super().get_batch_samples(
            epoch_iterator,
            num_batches,
            device,
        )
        if not self._svg_structural_response_loss_enabled:
            return batch_samples, num_items_in_batch
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "Structural weighted-loss pilot is single-process only until its "
                "accumulation denominator is all-reduced across DDP ranks."
            )
        if self.accelerator.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "Structural weighted-loss requires the installed Trainer-managed "
                "gradient-accumulation contract with Accelerator GAS=1."
            )
        denominator = _structural_accumulation_window_denominator(
            batch_samples,
            path_terminator_class_mass_weight=(
                self._svg_path_terminator_class_mass_weight
            ),
            reduction=self._svg_structural_response_loss_reduction,
        )
        return batch_samples, denominator.to(device)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        if not self._svg_structural_response_loss_enabled:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        prepared = dict(inputs)
        loss_weights = prepared.pop("loss_weights", None)
        response_token_roles = prepared.pop("response_token_roles", None)
        labels = prepared.get("labels")
        if loss_weights is None or labels is None or response_token_roles is None:
            raise ValueError(
                "Structural response loss requires labels, loss_weights, and response_token_roles."
            )
        if tuple(loss_weights.shape) != tuple(labels.shape):
            raise ValueError("Structural response loss weight shape differs from labels.")
        if tuple(response_token_roles.shape) != tuple(labels.shape):
            raise ValueError("Structural response role shape differs from labels.")
        if not model.training:
            return super().compute_loss(
                model,
                prepared,
                return_outputs=return_outputs,
                num_items_in_batch=None,
            )
        if num_items_in_batch is None:
            raise ValueError(
                "Structural training requires one normalization denominator for the "
                "complete gradient-accumulation window."
            )
        prepared.pop("labels")
        outputs = model(**prepared)
        logits = outputs.get("logits") if isinstance(outputs, Mapping) else outputs.logits
        loss = _structural_weighted_causal_lm_loss(
            logits,
            labels,
            loss_weights,
            response_token_roles=response_token_roles,
            path_terminator_class_mass_weight=(
                self._svg_path_terminator_class_mass_weight
            ),
            reduction=self._svg_structural_response_loss_reduction,
            normalization_denominator=num_items_in_batch,
        )
        return (loss, outputs) if return_outputs else loss


class _VerifiedLengthSamplerMixin:
    """Use audited record lengths without invoking dataset tokenization for sampling."""

    def _get_train_sampler(self, train_dataset: Any = None) -> Any:
        if self.args.train_sampling_strategy != "group_by_length":
            return super()._get_train_sampler(train_dataset)
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not isinstance(dataset, _ResponseOnlyDataset):
            raise TypeError(
                "Verified length grouping requires _ResponseOnlyDataset with "
                f"{_VERIFIED_LENGTH_SOURCE}."
            )
        from transformers.trainer_pt_utils import LengthGroupedSampler

        grouping_batch_size = getattr(self.args, "length_grouping_batch_size", None)
        if grouping_batch_size is None:
            grouping_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps
            )
        return LengthGroupedSampler(
            grouping_batch_size,
            lengths=dataset.verified_lengths,
        )


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
        test_data_path: str | Path | None = None,
        codec_backend_id: str = OFFICIAL_CACHED_GEMMA_BACKEND_ID,
        official_cache_config: OpenVGLabCacheConfig | None = None,
        allow_legacy_toy_codec: bool = False,
        official_train_sample_ids: Sequence[str] | None = None,
        official_validation_sample_ids: Sequence[str] | None = None,
    ) -> None:
        if instruction_mode not in (
            "description_only",
            "mixed_60_detail_40_description",
            "detail_only",
        ):
            raise ValueError(f"Unsupported instruction_mode: {instruction_mode}")
        if target_representation not in ("raw_xml", "omnisvg_discrete"):
            raise ValueError(f"Unsupported target_representation: {target_representation}")
        if not sft_config.do_train:
            raise ValueError("TextToSVGSFTTrainer requires sft.do_train=true.")
        if sft_config.do_eval and eval_data_path is None:
            raise ValueError("sft.do_eval=true requires dataset.validation_path.")
        if sft_config.do_predict and test_data_path is None:
            raise ValueError("sft.do_predict=true requires dataset.test_path.")
        required_data_splits = tuple(
            split
            for split, enabled in (
                ("train", sft_config.do_train),
                ("validation", sft_config.do_eval),
                ("test", sft_config.do_predict),
            )
            if enabled
        )
        structural_response_loss_enabled = _structural_response_loss_enabled(sft_config)
        if structural_response_loss_enabled and not (
            target_representation == "omnisvg_discrete"
            and codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
        ):
            raise ValueError(
                "Structural response weights are restricted to official cached "
                "omnisvg_discrete training."
            )
        if not isinstance(allow_legacy_toy_codec, bool):
            raise TypeError("allow_legacy_toy_codec must be boolean.")
        normalized_train_sample_ids = _normalize_official_train_sample_ids(
            official_train_sample_ids
        )
        normalized_validation_sample_ids = _normalize_official_train_sample_ids(
            official_validation_sample_ids
        )
        if normalized_train_sample_ids is not None and not (
            target_representation == "omnisvg_discrete"
            and codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
        ):
            raise ValueError(
                "official_train_sample_ids is only valid for official cached discrete SFT."
            )
        if normalized_validation_sample_ids is not None and not (
            target_representation == "omnisvg_discrete"
            and codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
        ):
            raise ValueError(
                "official_validation_sample_ids is only valid for official cached discrete SFT."
            )
        if target_representation == "omnisvg_discrete":
            validate_gemma_codec_backend(
                codec_backend_id,
                allow_legacy_toy_codec=allow_legacy_toy_codec,
            )
            if codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID:
                if official_cache_config is None:
                    raise ValueError(
                        "Official cached discrete SFT requires official_cache_config."
                    )
                if official_cache_config.allow_live_reencode:
                    raise ValueError(
                        "Official cached discrete SFT requires allow_live_reencode=false."
                    )
                configured_paths = {
                    "train": Path(train_data_path),
                    "validation": Path(eval_data_path) if eval_data_path else None,
                    "test": Path(test_data_path) if test_data_path else None,
                }
                for split in required_data_splits:
                    configured_path = configured_paths[split]
                    expected_path = official_cache_config.prepared_root / f"{split}.jsonl"
                    if configured_path is None or configured_path.resolve() != expected_path:
                        raise ValueError(
                            f"Official cached {split} path must be {expected_path}."
                        )
            elif official_cache_config is not None:
                raise ValueError(
                    "official_cache_config is only valid for the official cached backend."
                )
        if model_config.auto_model_class not in _AUTO_MODEL_CLASSES:
            raise ValueError(f"Unsupported auto_model_class: {model_config.auto_model_class}")
        if (
            sft_config.init_adapter_from is not None
            and target_representation != "omnisvg_discrete"
        ):
            raise ValueError(
                "sft.init_adapter_from is restricted to omnisvg_discrete training; "
                "cross-representation adapter transfer requires an explicit mapping policy."
            )
        if sft_config.lm_head_loss_chunk_size <= 0:
            raise ValueError("sft.lm_head_loss_chunk_size must be a positive integer.")
        if sft_config.train_sampling_strategy not in (
            "random",
            "sequential",
            "group_by_length",
        ):
            raise ValueError(
                "sft.train_sampling_strategy must be random, sequential, or group_by_length."
            )
        if sft_config.early_stopping_patience is not None:
            if sft_config.early_stopping_patience <= 0:
                raise ValueError("sft.early_stopping_patience must be positive or null.")
            if not sft_config.do_eval:
                raise ValueError("Early stopping requires sft.do_eval=true.")
            if not sft_config.load_best_model_at_end:
                raise ValueError("Early stopping requires sft.load_best_model_at_end=true.")
            if sft_config.eval_strategy == "no":
                raise ValueError("Early stopping requires an evaluation strategy.")
            if not sft_config.metric_for_best_model:
                raise ValueError("Early stopping requires sft.metric_for_best_model.")
        if sft_config.load_best_model_at_end and (
            sft_config.eval_strategy != sft_config.save_strategy
        ):
            raise ValueError(
                "Best-model restoration requires matching sft.eval_strategy and sft.save_strategy."
            )
        self._model_config = model_config
        self._lora_config = lora_config
        self._sft_config = sft_config
        self._train_data_path = Path(train_data_path)
        self._eval_data_path = Path(eval_data_path) if eval_data_path else None
        self._test_data_path = Path(test_data_path) if test_data_path else None
        self._instruction_mode = instruction_mode
        self._target_representation = target_representation
        self._codec_backend_id = codec_backend_id
        self._official_cache_config = official_cache_config
        self._required_data_splits = required_data_splits
        self._allow_legacy_toy_codec = allow_legacy_toy_codec
        self._official_train_sample_ids = normalized_train_sample_ids
        self._official_validation_sample_ids = normalized_validation_sample_ids

    def train(self) -> dict[str, Any]:
        cached_records_by_split: dict[str, list[dict[str, Any]]] | None = None
        if (
            self._target_representation == "omnisvg_discrete"
            and self._codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
        ):
            if self._official_cache_config is None:
                raise RuntimeError("Official cache configuration was not retained.")
            cached_records_by_split = load_cached_splits_records(
                self._official_cache_config,
                self._required_data_splits,
            )
            if self._official_train_sample_ids is not None:
                cached_records_by_split = {
                    **cached_records_by_split,
                    "train": _filter_official_train_records(
                        cached_records_by_split["train"],
                        self._official_train_sample_ids,
                    ),
                }
            if self._official_validation_sample_ids is not None:
                cached_records_by_split = {
                    **cached_records_by_split,
                    "validation": _filter_official_train_records(
                        cached_records_by_split["validation"],
                        self._official_validation_sample_ids,
                    ),
                }
        try:
            import torch
            import transformers
            from peft import get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                EarlyStoppingCallback,
                Trainer,
                TrainingArguments,
            )
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

        codec = (
            create_gemma_named_codec(
                self._codec_backend_id,
                allow_legacy_toy_codec=self._allow_legacy_toy_codec,
            )
            if self._target_representation == "omnisvg_discrete"
            else None
        )
        codec_registration: NamedTokenRegistration | None = None
        if codec is not None:
            codec_registration = register_named_codec_tokens(tokenizer, codec)

        dtype = getattr(torch, model_config.dtype)
        quantization_config = None
        if model_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
            )
        model_loader = getattr(
            transformers,
            _AUTO_MODEL_CLASSES[model_config.auto_model_class],
            None,
        )
        if model_loader is None:
            raise RuntimeError(
                "Transformers does not provide "
                f"{_AUTO_MODEL_CLASSES[model_config.auto_model_class]}."
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
            base_tying = _require_tied_input_output_embeddings(model)
        else:
            base_tying = None
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
            if codec_registration is None:
                raise RuntimeError("Discrete codec registration is unavailable.")
            if self._sft_config.chunked_lm_head_loss:
                raise ValueError(
                    "omnisvg_discrete selected-row training is incompatible with "
                    "chunked_lm_head_loss: PEFT rebuilds the full tied vocabulary "
                    "weight for every chunk. Disable chunked_lm_head_loss."
                )
            lora_config = _configure_discrete_lora(
                lora_config,
                codec_registration.token_ids,
            )
        peft_config = lora_config.to_peft_config()
        adapter_initialization: dict[str, Any] = {
            "mode": "fresh",
            "source_path": None,
            "adapter_config_sha256": None,
            "adapter_model_sha256": None,
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "trainer_state_loaded": False,
        }
        if self._sft_config.init_adapter_from is None:
            model = get_peft_model(model, peft_config)
        else:
            if codec_registration is None:
                raise RuntimeError(
                    "Adapter weight-only initialization requires discrete codec registration."
                )
            model, adapter_initialization = _load_adapter_weight_only(
                model,
                adapter_path=self._sft_config.init_adapter_from,
                expected_peft_config=peft_config,
                expected_base_model_id=model_config.model_id,
                expected_base_model_revision=model_config.revision,
                codec_backend_id=self._codec_backend_id,
                expected_codec_token_ids=codec_registration.token_ids,
            )
        trainability = (
            _discrete_trainability_manifest(
                model,
                codec_registration,
                base_tying=base_tying,
            )
            if codec_registration is not None
            else None
        )
        loss_implementation: dict[str, Any] = {
            "name": "transformers_default_causal_lm",
            "returns_labeled_logits": True,
        }
        if _structural_response_loss_enabled(self._sft_config):
            loss_implementation = {
                "name": "structural_response_token_weighted_causal_lm",
                "per_sample_training_objectives": _STRUCTURAL_RESPONSE_LOSS_CONTRACT[
                    "per_sample_training_objectives"
                ],
                "reduction": self._sft_config.structural_response_loss_reduction,
                "evaluation_reduction": _STRUCTURAL_RESPONSE_LOSS_CONTRACT[
                    "evaluation_reduction"
                ],
                "contract_sha256": _STRUCTURAL_RESPONSE_LOSS_CONTRACT_SHA256,
                "returns_labeled_logits": True,
            }
        if self._sft_config.chunked_lm_head_loss:
            loss_implementation = install_chunked_causal_lm_loss(
                model,
                chunk_size=self._sft_config.lm_head_loss_chunk_size,
            )
        if self._sft_config.gradient_checkpointing and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        if cached_records_by_split is None:
            train_records = _load_records(self._train_data_path)
            eval_records = (
                _load_records(self._eval_data_path)
                if "validation" in self._required_data_splits
                else []
            )
            test_records = (
                _load_records(self._test_data_path)
                if "test" in self._required_data_splits
                else []
            )
        else:
            train_records = cached_records_by_split["train"]
            eval_records = (
                cached_records_by_split["validation"]
                if "validation" in self._required_data_splits
                else []
            )
            test_records = (
                cached_records_by_split["test"]
                if "test" in self._required_data_splits
                else []
            )
        train_dataset = _ResponseOnlyDataset(
            train_records,
            tokenizer=tokenizer,
            instruction_mode=self._instruction_mode,
            target_representation=self._target_representation,
            max_seq_length=self._sft_config.max_seq_length,
            seed=self._sft_config.seed,
            codec=codec,
            response_eos_loss_weight=self._sft_config.response_eos_loss_weight,
            response_path_terminator_class_mass_weight=(
                self._sft_config.response_path_terminator_class_mass_weight
            ),
            detail_text_normalization=self._sft_config.detail_text_normalization,
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
                response_eos_loss_weight=self._sft_config.response_eos_loss_weight,
                response_path_terminator_class_mass_weight=(
                    self._sft_config.response_path_terminator_class_mass_weight
                ),
                detail_text_normalization=self._sft_config.detail_text_normalization,
            )
            if eval_records
            else None
        )
        test_dataset = (
            _ResponseOnlyDataset(
                test_records,
                tokenizer=tokenizer,
                instruction_mode=self._instruction_mode,
                target_representation=self._target_representation,
                max_seq_length=self._sft_config.max_seq_length,
                seed=self._sft_config.seed,
                codec=codec,
                response_eos_loss_weight=self._sft_config.response_eos_loss_weight,
                response_path_terminator_class_mass_weight=(
                    self._sft_config.response_path_terminator_class_mass_weight
                ),
                detail_text_normalization=self._sft_config.detail_text_normalization,
            )
            if test_records
            else None
        )
        output_dir = Path(self._sft_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            do_train=self._sft_config.do_train,
            do_eval=self._sft_config.do_eval,
            do_predict=self._sft_config.do_predict,
            num_train_epochs=self._sft_config.num_train_epochs,
            max_steps=self._sft_config.max_steps,
            per_device_train_batch_size=self._sft_config.per_device_train_batch_size,
            per_device_eval_batch_size=self._sft_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self._sft_config.gradient_accumulation_steps,
            learning_rate=self._sft_config.learning_rate,
            weight_decay=self._sft_config.weight_decay,
            warmup_steps=self._sft_config.warmup_ratio,
            lr_scheduler_type=self._sft_config.lr_scheduler_type,
            logging_steps=self._sft_config.logging_steps,
            save_strategy=self._sft_config.save_strategy,
            save_steps=self._sft_config.save_steps,
            save_total_limit=self._sft_config.save_total_limit,
            eval_strategy=self._sft_config.eval_strategy if eval_dataset is not None else "no",
            eval_steps=self._sft_config.eval_steps if eval_dataset is not None else None,
            bf16=self._sft_config.bf16,
            fp16=self._sft_config.fp16,
            gradient_checkpointing=self._sft_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=self._sft_config.optim,
            seed=self._sft_config.seed,
            data_seed=self._sft_config.seed,
            dataloader_num_workers=self._sft_config.dataloader_num_workers,
            torch_empty_cache_steps=self._sft_config.torch_empty_cache_steps,
            train_sampling_strategy=self._sft_config.train_sampling_strategy,
            remove_unused_columns=False,
            prediction_loss_only=self._sft_config.prediction_loss_only,
            load_best_model_at_end=(
                self._sft_config.load_best_model_at_end and eval_dataset is not None
            ),
            metric_for_best_model=(
                self._sft_config.metric_for_best_model if eval_dataset is not None else None
            ),
            greater_is_better=(
                self._sft_config.greater_is_better if eval_dataset is not None else None
            ),
            ddp_find_unused_parameters=False,
            report_to=self._sft_config.report_to or [],
        )
        setattr(
            args,
            "length_grouping_batch_size",
            self._sft_config.length_grouping_batch_size,
        )
        callbacks = []
        if self._sft_config.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self._sft_config.early_stopping_patience,
                    early_stopping_threshold=self._sft_config.early_stopping_threshold,
                )
            )
        class VerifiedLengthTrainer(
            PagedOptimizerResumeMixin,
            _RowDeltaAdapterSaveMixin,
            _VerifiedLengthSamplerMixin,
            _StructuralResponseLossMixin,
            Trainer,
        ):
            pass

        trainer = VerifiedLengthTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_ResponseOnlyCollator(tokenizer),
            callbacks=callbacks,
        )
        trainer._save_row_deltas_only = codec is not None
        trainer._svg_structural_response_loss_enabled = _structural_response_loss_enabled(
            self._sft_config
        )
        trainer._svg_path_terminator_class_mass_weight = (
            self._sft_config.response_path_terminator_class_mass_weight
        )
        trainer._svg_structural_response_loss_reduction = (
            self._sft_config.structural_response_loss_reduction
        )
        if trainer._svg_structural_response_loss_enabled:
            trainer.model_accepts_loss_kwargs = True
            if trainer.accelerator.num_processes != 1:
                raise RuntimeError(
                    "Structural weighted-loss pilot requires a single training process."
                )
        trainer._svg_enable_paged_optimizer_resume_rehydration = (
            self._sft_config.rehydrate_paged_optimizer_state_on_resume
        )
        trainer._svg_paged_optimizer_resume_audit = initial_paged_optimizer_resume_manifest(
            enabled=self._sft_config.rehydrate_paged_optimizer_state_on_resume,
            checkpoint=self._sft_config.resume_from_checkpoint,
        )
        train_result = trainer.train(resume_from_checkpoint=self._sft_config.resume_from_checkpoint)
        if self._sft_config.early_stopping_patience is not None:
            trainer.remove_callback(EarlyStoppingCallback)
        test_metrics = (
            trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            if test_dataset is not None
            else None
        )
        trainer.save_state()

        adapter_dir = output_dir / "adapter"
        tokenizer_dir = output_dir / "tokenizer"
        trainer.accelerator.wait_for_everyone()
        trainer.save_model(str(adapter_dir))
        trainer.accelerator.wait_for_everyone()
        codec_manifest = (
            _build_codec_manifest(codec, tokenizer, codec_registration)
            if codec is not None and codec_registration is not None
            else None
        )
        training_manifest = {
            "schema_version": 1,
            "base_model_id": model_config.model_id,
            "base_model_revision": model_config.revision,
            "auto_model_class": model_config.auto_model_class,
            "instruction_mode": self._instruction_mode,
            "target_representation": self._target_representation,
            "codec_backend_id": codec.metadata.backend_id if codec is not None else None,
            "official_discrete_cache": (
                _official_cache_manifest(
                    self._official_cache_config,
                    requested_splits=self._required_data_splits,
                )
                if self._official_cache_config is not None
                else None
            ),
            "official_train_selection": (
                _official_train_selection_manifest(
                    train_records,
                    requested_sample_ids=self._official_train_sample_ids,
                )
                if self._codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
                and self._target_representation == "omnisvg_discrete"
                else None
            ),
            "official_validation_selection": (
                _official_train_selection_manifest(
                    eval_records,
                    requested_sample_ids=self._official_validation_sample_ids,
                )
                if self._codec_backend_id == OFFICIAL_CACHED_GEMMA_BACKEND_ID
                and self._target_representation == "omnisvg_discrete"
                and "validation" in self._required_data_splits
                else None
            ),
            "response_only_loss": True,
            "instruction_selection": {
                "train": train_dataset.instruction_selection_manifest(),
                "validation": (
                    eval_dataset.instruction_selection_manifest()
                    if eval_dataset is not None
                    else None
                ),
                "test": (
                    test_dataset.instruction_selection_manifest()
                    if test_dataset is not None
                    else None
                ),
            },
            "loss_implementation": loss_implementation,
            "structural_response_loss": {
                "enabled": _structural_response_loss_enabled(self._sft_config),
                "reduction": self._sft_config.structural_response_loss_reduction,
                "weights": {
                    "eos": self._sft_config.response_eos_loss_weight,
                    "path_terminator_class_mass": (
                        self._sft_config.response_path_terminator_class_mass_weight
                    ),
                },
                "simultaneous_eos_and_path_weighting_allowed": False,
                "contract": _STRUCTURAL_RESPONSE_LOSS_CONTRACT,
                "contract_sha256": _STRUCTURAL_RESPONSE_LOSS_CONTRACT_SHA256,
                "role_counts": {
                    "train": train_dataset.structural_response_role_manifest(),
                    "validation": (
                        eval_dataset.structural_response_role_manifest()
                        if eval_dataset is not None
                        else None
                    ),
                    "test": (
                        test_dataset.structural_response_role_manifest()
                        if test_dataset is not None
                        else None
                    ),
                },
                "required_followup_collapse_gates": {
                    "generated_output_length": True,
                    "strict_decoded_path_count": True,
                },
                "gradient_accumulation": {
                    "normalization_scope": "entire_accumulation_window",
                    "denominator_unit": (
                        "active_weight_or_augmented_class_mass"
                        if self._sft_config.structural_response_loss_reduction == "token"
                        else "samples_after_per_sample_weighted_mean"
                    ),
                    "microbatch_local_denominator_forbidden": True,
                    "trainer_global_denominator_channel": "num_items_in_batch",
                    "microbatch_return_value": (
                        "additive numerator/global-window-denominator contribution"
                    ),
                    "transformers_manages_accumulation": True,
                    "required_accelerator_gradient_accumulation_steps": 1,
                    "extra_gradient_or_reporting_scale": False,
                },
                "distributed_scaling": {
                    "pilot_process_count": 1,
                    "multi_process_behavior": "fail_fast",
                    "future_ddp_requirement": (
                        "all-reduce numerator-equivalent gradients and the configured "
                        "reduction denominator across data-parallel ranks"
                    ),
                },
            },
            "adapter_serialization": _adapter_serialization_manifest(args),
            "rag_context_in_training": False,
            "requested_splits": list(self._required_data_splits),
            "train_data_path": str(self._train_data_path.resolve()),
            "train_data_sha256": _file_sha256(self._train_data_path),
            "eval_data_path": (
                str(self._eval_data_path.resolve())
                if "validation" in self._required_data_splits
                else None
            ),
            "eval_data_sha256": (
                _file_sha256(self._eval_data_path)
                if "validation" in self._required_data_splits
                else None
            ),
            "test_data_path": (
                str(self._test_data_path.resolve())
                if "test" in self._required_data_splits
                else None
            ),
            "test_data_sha256": (
                _file_sha256(self._test_data_path)
                if "test" in self._required_data_splits
                else None
            ),
            "sampling": {
                "strategy": self._sft_config.train_sampling_strategy,
                "implementation": (
                    "transformers.LengthGroupedSampler"
                    if self._sft_config.train_sampling_strategy == "group_by_length"
                    else "transformers.Trainer.default"
                ),
                "grouping_batch_size": (
                    self._sft_config.length_grouping_batch_size
                    or (
                        self._sft_config.per_device_train_batch_size
                        * self._sft_config.gradient_accumulation_steps
                    )
                    if self._sft_config.train_sampling_strategy == "group_by_length"
                    else None
                ),
                "grouping_batch_size_source": (
                    "sft.length_grouping_batch_size"
                    if self._sft_config.length_grouping_batch_size is not None
                    else "per_device_train_batch_size_x_gradient_accumulation_steps"
                )
                if self._sft_config.train_sampling_strategy == "group_by_length"
                else None,
                "verified_lengths": {
                    "train": train_dataset.verified_length_manifest(),
                    "validation": (
                        eval_dataset.verified_length_manifest()
                        if eval_dataset is not None
                        else None
                    ),
                    "test": (
                        test_dataset.verified_length_manifest()
                        if test_dataset is not None
                        else None
                    ),
                },
            },
            "optimizer_resume": trainer._svg_paged_optimizer_resume_audit,
            "adapter_initialization": adapter_initialization,
            "lora": _lora_manifest(lora_config),
            "trainability": trainability,
            "sft": asdict(self._sft_config),
            "codec_manifest": codec_manifest,
            "versions": {
                name: _package_version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
            },
            "train_metrics": dict(train_result.metrics),
            "test_metrics": dict(test_metrics) if test_metrics is not None else None,
            "training_control": {
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "stopped_epoch": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "early_stopping_patience": self._sft_config.early_stopping_patience,
                "early_stopping_threshold": self._sft_config.early_stopping_threshold,
            },
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


def _select_instruction(
    row: dict[str, Any],
    mode: InstructionMode,
    *,
    detail_text_normalization: DetailTextNormalization = "none",
) -> str:
    return _select_instruction_with_provenance(
        row,
        mode,
        detail_text_normalization=_validated_detail_text_normalization(
            detail_text_normalization
        ),
    ).text


def _select_instruction_with_provenance(
    row: dict[str, Any],
    mode: InstructionMode,
    *,
    detail_text_normalization: DetailTextNormalization,
) -> _InstructionSelection:
    description = str(row.get("description", row.get("instruction", ""))).strip()
    if not description:
        raise ValueError("SFT record is missing description.")
    if mode == "description_only":
        return _InstructionSelection(text=description, selected_field="description")
    detail = str(row.get("detail", "")).strip()
    if not detail:
        raise ValueError(f"SFT mode {mode!r} requires a non-empty detail field.")
    if mode == "detail_only":
        return _normalize_selected_detail(
            detail,
            detail_text_normalization=detail_text_normalization,
            record_id=_instruction_record_id(row),
        )
    metadata = row.get("metadata")
    instruction_policy = metadata.get("instruction_policy") if isinstance(metadata, dict) else None
    selected_field = (
        instruction_policy.get("r1_detail_60_description_40")
        if isinstance(instruction_policy, dict)
        else None
    )
    if selected_field not in ("detail", "description"):
        raise ValueError(
            "Mixed-instruction SFT requires metadata.instruction_policy."
            "r1_detail_60_description_40 to be exactly 'detail' or 'description'."
        )
    if selected_field == "description":
        return _InstructionSelection(text=description, selected_field="description")
    return _normalize_selected_detail(
        detail,
        detail_text_normalization=detail_text_normalization,
        record_id=_instruction_record_id(row),
    )


def _normalize_selected_detail(
    detail: str,
    *,
    detail_text_normalization: DetailTextNormalization,
    record_id: Any,
) -> _InstructionSelection:
    legacy_candidate = detail.startswith("[") or detail.endswith("]")
    if detail_text_normalization == "none" or not legacy_candidate:
        return _InstructionSelection(
            text=detail,
            selected_field="detail",
            legacy_detail_candidate=legacy_candidate,
        )
    try:
        parsed = ast.literal_eval(detail)
    except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(
            f"SFT record {record_id!r} has a malformed MMSVG detail list representation."
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"SFT record {record_id!r} MMSVG detail list must be a non-empty list."
        )
    if any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise ValueError(
            f"SFT record {record_id!r} MMSVG detail list must contain only "
            "non-empty strings."
        )
    normalized = " ".join(item.strip() for item in parsed)
    return _InstructionSelection(
        text=normalized,
        selected_field="detail",
        legacy_detail_candidate=True,
        legacy_detail_normalized=True,
    )


def _instruction_record_id(row: Mapping[str, Any]) -> Any:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        record_id = metadata.get("record_id")
        if record_id is not None:
            return record_id
    return "unknown"


def _verified_full_chat_token_length(row: dict[str, Any], *, index: int) -> int:
    metadata = row.get("metadata")
    record_id = metadata.get("record_id", index) if isinstance(metadata, Mapping) else index
    value = metadata.get("full_chat_token_length") if isinstance(metadata, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"SFT record {record_id!r} requires {_VERIFIED_LENGTH_SOURCE} "
            "to be a positive integer."
        )
    return value


def _select_target(
    row: dict[str, Any],
    representation: TargetRepresentation,
    codec: NamedSpecialTokenSVGCodec | None,
) -> str:
    if representation == "raw_xml":
        svg = str(row.get("output_svg", "")).strip()
        if not svg:
            raise ValueError("SFT record is missing output_svg.")
        return svg
    if codec is None:
        raise RuntimeError("Discrete target representation requires a codec.")
    target_for_record = getattr(codec, "target_for_record", None)
    if callable(target_for_record):
        target = target_for_record(row)
        if not isinstance(target, str) or not target:
            raise RuntimeError("Cached discrete codec returned an invalid target.")
        return target
    svg = str(row.get("output_svg", "")).strip()
    if not svg:
        raise ValueError("SFT record is missing output_svg.")
    encoded = codec.encode(svg)
    if not encoded.tokens or any(not isinstance(token, str) for token in encoded.tokens):
        raise RuntimeError("Named discrete codec returned invalid target tokens.")
    return "".join(encoded.tokens)


def _template_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise RuntimeError("Tokenizer chat template result is missing input_ids.")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError("Tokenizer chat template returned multiple examples.")
        encoded = encoded[0]
    if (
        not isinstance(encoded, list)
        or not encoded
        or not all(isinstance(value, int) for value in encoded)
    ):
        raise RuntimeError("Tokenizer chat template returned invalid token IDs.")
    return encoded


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SFT data file not found: {path}")
    records = [row for row in read_jsonl(path) if row.get("task") == "text_to_svg"]
    if not records:
        raise ValueError(f"SFT data file contains no text_to_svg records: {path}")
    return records


def _normalize_official_train_sample_ids(
    value: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("official_train_sample_ids must be a sequence of strings or null.")
    normalized = tuple(value)
    if not normalized or any(
        not isinstance(sample_id, str) or not sample_id.strip()
        for sample_id in normalized
    ):
        raise ValueError("official_train_sample_ids must contain non-empty strings.")
    normalized = tuple(sample_id.strip() for sample_id in normalized)
    if len(set(normalized)) != len(normalized):
        raise ValueError("official_train_sample_ids contains duplicate sample IDs.")
    return normalized


def _prepared_record_id(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    nested_value = metadata.get("record_id") if isinstance(metadata, Mapping) else None
    top_level_value = record.get("id")
    nested_id = str(nested_value).strip() if nested_value is not None else ""
    top_level_id = str(top_level_value).strip() if top_level_value is not None else ""
    if nested_id and top_level_id and nested_id != top_level_id:
        raise ValueError(
            "Prepared record identity mismatch between metadata.record_id and id: "
            f"{nested_id!r} != {top_level_id!r}."
        )
    record_id = nested_id or top_level_id
    if not record_id:
        raise ValueError("Prepared official record lacks metadata.record_id and id.")
    return record_id


def _filter_official_train_records(
    records: Sequence[dict[str, Any]],
    requested_sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    requested = _normalize_official_train_sample_ids(requested_sample_ids)
    if requested is None:
        raise ValueError("Official train filter requires explicit sample IDs.")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = _prepared_record_id(record)
        if record_id in indexed:
            raise ValueError(
                f"Pinned official train records contain duplicate ID {record_id!r}."
            )
        indexed[record_id] = record
    missing = [sample_id for sample_id in requested if sample_id not in indexed]
    if missing:
        raise ValueError(
            "Requested official train sample IDs are missing from the pinned cache join: "
            + ", ".join(missing)
        )
    return [indexed[sample_id] for sample_id in requested]


def _official_train_selection_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    requested_sample_ids: Sequence[str] | None,
) -> dict[str, Any]:
    selected_ids = tuple(_prepared_record_id(record) for record in records)
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("Selected official train records contain duplicate identities.")
    compact = json.dumps(selected_ids, separators=(",", ":"))
    return {
        "mode": (
            "explicit_sample_ids"
            if requested_sample_ids is not None
            else "all_pinned_train_records"
        ),
        "selected_count": len(selected_ids),
        "ordered_selected_sample_ids_sha256": hashlib.sha256(
            compact.encode("utf-8")
        ).hexdigest(),
        "requested_sample_ids": (
            list(requested_sample_ids) if requested_sample_ids is not None else None
        ),
        "identity_source": "metadata.record_id_with_id_consistency_check",
    }


def _build_codec_manifest(
    codec: NamedSpecialTokenSVGCodec,
    tokenizer: Any,
    registration: NamedTokenRegistration,
) -> dict[str, Any]:
    tokens = codec.vocabulary_tokens()
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    if len(set(token_ids)) != len(token_ids) or any(value < 0 for value in token_ids):
        raise RuntimeError("Codec vocabulary did not map to unique tokenizer IDs.")
    compact = json.dumps(token_ids, separators=(",", ":"))
    manifest = {
        "schema_version": 1,
        "codec": codec.codec_manifest(),
        "backend": codec.metadata.to_manifest(),
        "registration": registration.to_manifest(),
        "tokenizer": {
            "vocabulary_size": len(tokenizer),
            "codec_token_ids": token_ids,
            "codec_token_ids_sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        },
    }
    return manifest


def _configure_discrete_lora(
    lora_config: LoRAConfig,
    codec_token_ids: Sequence[int],
) -> LoRAConfig:
    token_ids = list(codec_token_ids)
    if not token_ids or len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Discrete LoRA requires a non-empty set of unique codec token IDs.")
    if lora_config.trainable_token_indices not in (None, token_ids):
        raise ValueError("Discrete trainable_token_indices are derived and cannot be overridden.")
    modules_to_save = [
        name
        for name in lora_config.modules_to_save
        if name not in {"embed_tokens", "lm_head"}
    ]
    return LoRAConfig(
        **{
            **asdict(lora_config),
            "modules_to_save": modules_to_save,
            "trainable_token_indices": token_ids,
            "ensure_weight_tying": True,
        }
    )


def _official_cache_manifest(
    config: OpenVGLabCacheConfig,
    *,
    requested_splits: Sequence[str],
) -> dict[str, Any]:
    return {
        "cache_path": str(config.cache_path),
        "audit_manifest_path": str(config.audit_manifest_path),
        "prepared_root": str(config.prepared_root),
        "expected_cache_sha256": config.expected_cache_sha256,
        "expected_audit_sha256": config.expected_audit_sha256,
        "expected_input_sha256": dict(config.expected_input_sha256),
        "expected_split_counts": dict(config.expected_split_counts),
        "requested_splits": list(requested_splits),
        "allow_live_reencode": config.allow_live_reencode,
    }


def _require_tied_input_output_embeddings(model: Any) -> dict[str, Any]:
    if getattr(model.config, "tie_word_embeddings", None) is not True:
        raise RuntimeError(
            "Discrete selected-row training requires model.config.tie_word_embeddings=True."
        )
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if input_weight is None or output_weight is None:
        raise RuntimeError("Model does not expose input/output embedding weights.")
    if input_weight is not output_weight and input_weight.data_ptr() != output_weight.data_ptr():
        raise RuntimeError("Model input embeddings and LM head are not physically tied.")
    return {
        "config_tie_word_embeddings": True,
        "base_weight_storage_tied_before_peft": True,
    }


def _discrete_trainability_manifest(
    model: Any,
    registration: NamedTokenRegistration,
    *,
    base_tying: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name, _parameter in trainable
        if "lora_" not in name and "trainable_tokens_delta" not in name
    ]
    if unexpected:
        raise RuntimeError(
            "Discrete PEFT exposed unexpected trainable parameters: " + ", ".join(unexpected[:10])
        )
    delta_parameters = {
        id(parameter): (name, parameter)
        for name, parameter in trainable
        if "trainable_tokens_delta" in name
    }
    if len(delta_parameters) != 1:
        raise RuntimeError("Tied discrete input/output adapters must share exactly one row delta.")
    delta_name, delta_parameter = next(iter(delta_parameters.values()))
    if tuple(delta_parameter.shape)[0] != len(registration.token_ids):
        raise RuntimeError("Trainable-token delta row count differs from codec token count.")

    input_wrapper = model.get_input_embeddings()
    output_wrapper = model.get_output_embeddings()
    if type(input_wrapper).__name__ != "TrainableTokensWrapper" or type(
        output_wrapper
    ).__name__ != "TrainableTokensWrapper":
        raise RuntimeError("PEFT did not wrap both tied input embeddings and LM head.")
    input_adapter = getattr(input_wrapper, "token_adapter", None)
    output_adapter = getattr(output_wrapper, "token_adapter", None)
    if input_adapter is None or output_adapter is None:
        raise RuntimeError("PEFT trainable-token wrappers do not expose token adapters.")
    input_delta_ids = {
        id(parameter) for parameter in input_adapter.trainable_tokens_delta.values()
    }
    output_delta_ids = {
        id(parameter) for parameter in output_adapter.trainable_tokens_delta.values()
    }
    if len(input_delta_ids) != 1 or input_delta_ids != output_delta_ids:
        raise RuntimeError("PEFT trainable-token wrappers do not share one row delta.")
    if not (
        getattr(input_adapter, "tied_adapter", None) is output_adapter
        or getattr(output_adapter, "tied_adapter", None) is input_adapter
    ):
        raise RuntimeError("PEFT trainable-token wrappers are not adapter-tied.")
    for adapter in (input_adapter, output_adapter):
        base_weight = getattr(getattr(adapter, "base_layer", None), "weight", None)
        if base_weight is None or base_weight.requires_grad:
            raise RuntimeError("Base embedding/LM-head matrices must remain frozen.")

    lora_parameters = [parameter for name, parameter in trainable if "lora_" in name]
    if not lora_parameters:
        raise RuntimeError("Discrete SFT did not create any trainable LoRA parameters.")
    return {
        **dict(base_tying or {}),
        "adapter_weight_storage_tied": True,
        "selected_row_strategy": "peft-0.20-trainable_token_indices",
        "ensure_weight_tying": True,
        "save_embedding_layers": False,
        "codec_token_count": len(registration.token_ids),
        "codec_token_ids_sha256": registration.token_ids_sha256,
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_token_parameter_count": delta_parameter.numel(),
        "lora_parameter_count": sum(parameter.numel() for parameter in lora_parameters),
        "trainable_token_parameter_name": delta_name,
        "unexpected_trainable_parameters": [],
    }


def _lora_manifest(lora_config: LoRAConfig) -> dict[str, Any]:
    payload = asdict(lora_config)
    token_indices = payload.get("trainable_token_indices")
    if isinstance(token_indices, list):
        compact = json.dumps(token_indices, separators=(",", ":"))
        payload["trainable_token_indices"] = {
            "count": len(token_indices),
            "sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        }
    return payload


_ADAPTER_TOPOLOGY_FIELDS = (
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "target_modules",
    "modules_to_save",
    "trainable_token_indices",
    "ensure_weight_tying",
)


def _load_adapter_weight_only(
    model: Any,
    *,
    adapter_path: str | Path,
    expected_peft_config: Any,
    expected_base_model_id: str,
    expected_base_model_revision: str,
    codec_backend_id: str,
    expected_codec_token_ids: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Load only audited discrete PEFT weights, never Trainer state."""

    try:
        import torch
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "Adapter weight-only initialization requires peft, torch, and safetensors."
        ) from exc

    source_dir = Path(adapter_path).expanduser()
    if not source_dir.is_dir():
        raise ValueError(f"sft.init_adapter_from is not a directory: {source_dir}")
    source_dir = source_dir.resolve()
    config_path = source_dir / "adapter_config.json"
    weights_path = source_dir / "adapter_model.safetensors"
    unsafe_weights_path = source_dir / "adapter_model.bin"
    if not config_path.is_file() or config_path.stat().st_size <= 0:
        raise ValueError("Adapter initialization requires a non-empty adapter_config.json.")
    if not weights_path.is_file() or weights_path.stat().st_size <= 0:
        raise ValueError(
            "Adapter initialization requires a non-empty adapter_model.safetensors."
        )
    if unsafe_weights_path.exists():
        raise ValueError(
            "Adapter initialization refuses ambiguous/unsafe adapter_model.bin artifacts."
        )

    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("adapter_config.json is unreadable or invalid JSON.") from exc
    if not isinstance(source_config, dict):
        raise ValueError("adapter_config.json must contain a JSON object.")

    expected_config = expected_peft_config.to_dict()
    topology = _require_matching_adapter_topology(source_config, expected_config)
    source_base = _canonical_model_reference(
        source_config.get("base_model_name_or_path"),
        field_name="adapter_config.base_model_name_or_path",
    )
    expected_base = _canonical_model_reference(
        expected_base_model_id,
        field_name="model.model_id",
    )
    if source_base != expected_base:
        raise ValueError(
            "Adapter base_model_name_or_path does not match the configured base model: "
            f"{source_base!r} != {expected_base!r}."
        )

    expected_token_ids = list(expected_codec_token_ids)
    if not expected_token_ids or len(set(expected_token_ids)) != len(expected_token_ids):
        raise RuntimeError("Current discrete codec token IDs must be non-empty and unique.")
    if topology["trainable_token_indices"] != expected_token_ids:
        raise ValueError(
            "Adapter trainable_token_indices do not match the current ordered codec token IDs."
        )
    compact_token_ids = json.dumps(expected_token_ids, separators=(",", ":"))
    codec_token_ids_sha256 = hashlib.sha256(compact_token_ids.encode("utf-8")).hexdigest()

    try:
        source_state = load_file(str(weights_path), device="cpu")
    except Exception as exc:
        raise ValueError("adapter_model.safetensors could not be decoded.") from exc
    if not source_state:
        raise ValueError("adapter_model.safetensors contains no tensors.")
    tensor_schema: dict[str, dict[str, Any]] = {}
    for name, tensor in source_state.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Adapter tensor {name!r} contains non-finite values.")
        tensor_schema[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }

    loaded_model = PeftModel.from_pretrained(
        model,
        str(source_dir),
        is_trainable=True,
        local_files_only=True,
    )
    loaded_state = get_peft_model_state_dict(loaded_model)
    source_keys = set(source_state)
    loaded_keys = set(loaded_state)
    if source_keys != loaded_keys:
        missing = sorted(source_keys - loaded_keys)
        unexpected = sorted(loaded_keys - source_keys)
        raise RuntimeError(
            "Loaded adapter state keys differ from the source artifact; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}."
        )
    for name, source_tensor in source_state.items():
        loaded_tensor = loaded_state[name].detach().cpu()
        if tuple(loaded_tensor.shape) != tuple(source_tensor.shape):
            raise RuntimeError(f"Loaded adapter tensor {name!r} has the wrong shape.")
        comparable_source = source_tensor.to(dtype=loaded_tensor.dtype)
        if not torch.equal(loaded_tensor, comparable_source):
            raise RuntimeError(f"Loaded adapter tensor {name!r} differs from its source value.")
    if not any(parameter.requires_grad for parameter in loaded_model.parameters()):
        raise RuntimeError("Weight-only adapter initialization produced no trainable parameters.")

    topology_json = json.dumps(topology, sort_keys=True, separators=(",", ":"))
    schema_json = json.dumps(tensor_schema, sort_keys=True, separators=(",", ":"))
    target_topology = topology["target_modules"]
    return loaded_model, {
        "mode": "adapter_weights_only",
        "source_path": str(source_dir),
        "adapter_config_sha256": _file_sha256(config_path),
        "adapter_model_sha256": _file_sha256(weights_path),
        "adapter_tensor_count": len(source_state),
        "adapter_tensor_schema_sha256": hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
        "adapter_topology_sha256": hashlib.sha256(topology_json.encode("utf-8")).hexdigest(),
        "resolved_target_module_count": target_topology["resolved_module_count"],
        "resolved_target_modules_sha256": target_topology["resolved_modules_sha256"],
        "canonical_target_module_suffixes": target_topology["canonical_suffixes"],
        "base_model_id": expected_base_model_id,
        "base_model_revision": expected_base_model_revision,
        "codec_backend_id": codec_backend_id,
        "codec_token_count": len(expected_token_ids),
        "codec_token_ids_sha256": codec_token_ids_sha256,
        "post_load_tensor_equality_verified": True,
        "optimizer_loaded": False,
        "scheduler_loaded": False,
        "trainer_state_loaded": False,
    }


def _require_matching_adapter_topology(
    source_config: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    source_topology = {
        field: _normalize_adapter_topology_value(field, source_config.get(field))
        for field in _ADAPTER_TOPOLOGY_FIELDS
    }
    expected_topology = {
        field: _normalize_adapter_topology_value(field, expected_config.get(field))
        for field in _ADAPTER_TOPOLOGY_FIELDS
    }
    target_topology = _require_matching_adapter_target_modules(
        source_topology["target_modules"],
        expected_topology["target_modules"],
    )
    source_topology["target_modules"] = target_topology
    expected_topology["target_modules"] = target_topology
    mismatches = [
        field
        for field in _ADAPTER_TOPOLOGY_FIELDS
        if source_topology[field] != expected_topology[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: {source_topology[field]!r} != {expected_topology[field]!r}"
            for field in mismatches
        )
        raise ValueError(f"Adapter PEFT topology mismatch ({details}).")
    return source_topology


def _require_matching_adapter_target_modules(
    source_targets: Sequence[str],
    expected_targets: Sequence[str],
) -> dict[str, Any]:
    source = list(source_targets)
    expected = list(expected_targets)
    if not source or not expected:
        raise ValueError("Adapter target_modules must be non-empty.")
    if not all(isinstance(name, str) and name for name in (*source, *expected)):
        raise ValueError("Adapter target_modules must contain non-empty strings.")
    if len(source) != len(set(source)) or len(expected) != len(set(expected)):
        raise ValueError("Adapter target_modules must not contain duplicates.")

    source_set = set(source)
    expected_set = set(expected)
    if source_set == expected_set:
        canonical_suffixes = sorted({name.rsplit(".", 1)[-1] for name in source})
        resolved_modules = sorted(expected)
    else:
        if any("." in name for name in source):
            raise ValueError(
                "Adapter target_modules mismatch: non-suffix source targets require exact equality."
            )
        coverage = {suffix: [] for suffix in source}
        for module_name in expected:
            matches = [
                suffix
                for suffix in source
                if module_name == suffix or module_name.endswith(f".{suffix}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Adapter target_modules mismatch: each resolved module must map to "
                    f"exactly one saved suffix ({module_name!r} matched {matches!r})."
                )
            coverage[matches[0]].append(module_name)
        missing_coverage = sorted(suffix for suffix, modules in coverage.items() if not modules)
        if missing_coverage:
            raise ValueError(
                "Adapter target_modules mismatch: saved suffixes have no resolved coverage: "
                + ", ".join(missing_coverage)
            )
        resolved_suffixes = {name.rsplit(".", 1)[-1] for name in expected}
        if resolved_suffixes != source_set:
            raise ValueError(
                "Adapter target_modules mismatch: resolved leaf suffixes differ from the saved set."
            )
        canonical_suffixes = sorted(source_set)
        resolved_modules = sorted(expected)

    compact_modules = json.dumps(resolved_modules, separators=(",", ":"))
    return {
        "canonical_suffixes": canonical_suffixes,
        "resolved_module_count": len(resolved_modules),
        "resolved_modules_sha256": hashlib.sha256(
            compact_modules.encode("utf-8")
        ).hexdigest(),
    }


def _normalize_adapter_topology_value(field: str, value: Any) -> Any:
    if field in {"target_modules", "modules_to_save"}:
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            raise ValueError(f"Adapter topology field {field} must be a sequence.")
        return sorted(value)
    if field == "trainable_token_indices":
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return {key: list(value[key]) for key in sorted(value)}
        if value is not None:
            raise ValueError("Adapter trainable_token_indices must be a list, mapping, or null.")
    return value


def _canonical_model_reference(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    candidate = Path(normalized).expanduser()
    return str(candidate.resolve()) if candidate.exists() else normalized.rstrip("/")


def _adapter_serialization_manifest(training_args: Any) -> dict[str, Any]:
    missing = object()
    configured = getattr(training_args, "save_safetensors", missing)
    if configured is missing:
        safe_serialization = True
        source = "peft.save_pretrained_default"
    else:
        if not isinstance(configured, bool):
            raise RuntimeError(
                "TrainingArguments.save_safetensors must be a bool when provided."
            )
        safe_serialization = configured
        source = "training_arguments.save_safetensors"

    if safe_serialization:
        try:
            import safetensors.torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Safe adapter serialization is required, but safetensors is unavailable."
            ) from exc
    return {
        "safe_serialization": safe_serialization,
        "source": source,
    }


class _RowDeltaAdapterSaveMixin:
    _save_row_deltas_only = False

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict[str, Any] | None = None,
    ) -> None:
        if not self._save_row_deltas_only:
            return super()._save(output_dir=output_dir, state_dict=state_dict)
        import torch
        from transformers.trainer import TRAINING_ARGS_NAME

        destination = output_dir or self.args.output_dir
        os.makedirs(destination, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        save_pretrained = getattr(model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise RuntimeError("Discrete PEFT model does not expose save_pretrained().")
        serialization = _adapter_serialization_manifest(self.args)
        save_pretrained(
            destination,
            state_dict=state_dict,
            safe_serialization=serialization["safe_serialization"],
            save_embedding_layers=False,
        )
        torch.save(self.args, os.path.join(destination, TRAINING_ARGS_NAME))


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
