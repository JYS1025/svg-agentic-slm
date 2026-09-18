from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

import svg_agentic_slm.train.sft_trainer as sft_module
from svg_agentic_slm.train.sft_trainer import (
    ModelTrainingConfig,
    SFTConfig,
    TextToSVGSFTTrainer,
    _official_discrete_response_loss_annotations,
    _ResponseOnlyCollator,
    _structural_accumulation_window_denominator,
    _structural_response_loss_enabled,
    _structural_weighted_causal_lm_loss,
    _StructuralResponseLossMixin,
)

_TRAINING_TO_REGISTERED_OFFSET = 262144 - 151938


def _registered(training_id: int) -> int:
    return training_id + _TRAINING_TO_REGISTERED_OFFSET


class _NamedTokenTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [
            f"<svgovg4b:{token_id - _TRAINING_TO_REGISTERED_OFFSET}>"
            for token_id in token_ids
        ]


def _one_path_labels() -> list[int]:
    training_ids = [
        196998,
        151938,
        151943,
        151943,
        191946,
        196999,
    ]
    return [-100, -100, *[_registered(token_id) for token_id in training_ids]]


def test_all_one_weights_equal_existing_shifted_cross_entropy_exactly() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 11, dtype=torch.float32)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5], [-100, 2, 1, -100, 7]],
        dtype=torch.long,
    )
    weights = torch.ones_like(labels, dtype=torch.float32)

    actual = _structural_weighted_causal_lm_loss(logits, labels, weights)
    expected = functional.cross_entropy(
        logits[:, :-1, :].contiguous().float().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )

    assert torch.equal(actual, expected)


def test_prompt_ignore_index_shift_alignment_and_weighted_normalization() -> None:
    torch.manual_seed(11)
    logits = torch.randn(1, 5, 4, dtype=torch.float32)
    labels = torch.tensor([[-100, -100, 1, 2, -100]], dtype=torch.long)
    weights = torch.tensor([[99.0, 99.0, 2.0, 3.0, 77.0]], dtype=torch.float32)

    actual = _structural_weighted_causal_lm_loss(logits, labels, weights)
    per_token = functional.cross_entropy(
        logits[:, :-1, :].contiguous().float().view(-1, 4),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(1, 4)
    expected = (per_token[0, 1] * 2.0 + per_token[0, 2] * 3.0) / 5.0

    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)


def test_sample_reduction_means_each_weighted_response_then_means_across_samples() -> None:
    torch.manual_seed(13)
    logits = torch.randn(2, 5, 7, dtype=torch.float32)
    labels = torch.tensor(
        [[-100, -100, -100, 1, 2], [-100, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 5.0], [1.0, 1.0, 1.0, 1.0, 5.0]],
        dtype=torch.float32,
    )

    actual = _structural_weighted_causal_lm_loss(
        logits,
        labels,
        weights,
        reduction="sample",
    )
    per_token = functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(2, 4)
    first = (per_token[0, 2] + 5.0 * per_token[0, 3]) / 6.0
    second = (
        per_token[1, 0]
        + per_token[1, 1]
        + per_token[1, 2]
        + 5.0 * per_token[1, 3]
    ) / 8.0
    expected = (first + second) / 2.0
    token_reduced = _structural_weighted_causal_lm_loss(logits, labels, weights)

    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.allclose(actual, token_reduced, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("eos_weight", "path_weight", "expected_eos", "expected_path"),
    [(5.0, 1.0, 5.0, 1.0), (1.0, 3.0, 1.0, 3.0)],
)
def test_only_exact_eos_or_completed_path_terminator_positions_are_weighted(
    eos_weight: float,
    path_weight: float,
    expected_eos: float,
    expected_path: float,
) -> None:
    weights, roles, counts = _official_discrete_response_loss_annotations(
        _one_path_labels(),
    tokenizer=_NamedTokenTokenizer(),
        eos_weight=eos_weight,
        path_terminator_class_mass_weight=path_weight,
    )

    assert roles == [0, 0, 0, 0, 0, 0, 2, 1]
    assert weights == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, expected_eos]
    assert counts == {
        "labeled_token_count": 6,
        "eos_count": 1,
        "path_color_terminator_count": 1,
    }


def test_default_raw_collator_contract_is_unchanged() -> None:
    config = SFTConfig()
    assert not _structural_response_loss_enabled(config)
    features = [
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        {"input_ids": [6], "attention_mask": [1], "labels": [6]},
    ]

    batch = _ResponseOnlyCollator(_NamedTokenTokenizer())(features)

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert torch.equal(batch["input_ids"], torch.tensor([[4, 5], [6, 0]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, 5], [6, -100]]))


def test_structural_weight_shape_mismatch_fails_fast() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, 1, 2]])
    with pytest.raises(ValueError, match="weight shape"):
        _structural_weighted_causal_lm_loss(
            logits,
            labels,
            torch.ones(1, 2),
        )


def test_path_terminator_uses_color_class_mass_auxiliary_and_keeps_base_ce() -> None:
    vocabulary_size = 306742
    logits = torch.zeros(1, 3, vocabulary_size, dtype=torch.float32)
    labels = torch.tensor([[-100, 17, 302152]], dtype=torch.long)
    weights = torch.ones_like(labels, dtype=torch.float32)
    roles = torch.tensor([[0, 0, 2]], dtype=torch.long)

    actual = _structural_weighted_causal_lm_loss(
        logits,
        labels,
        weights,
        response_token_roles=roles,
        path_terminator_class_mass_weight=3.0,
    )
    per_token = functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocabulary_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    class_mass_nll = torch.log(torch.tensor(vocabulary_size / (306249 - 302152 + 1)))
    expected = (per_token.sum() + 2.0 * class_mass_nll) / 4.0

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_official_structural_token_ranges_exclude_coordinate_color_and_arc_gaps() -> None:
    kind = sft_module._official_inference_token_kind

    assert kind(191943) == "coordinate"
    assert kind(191947) == "path_color_terminator"
    assert kind(196044) == "path_color_terminator"
    assert kind(196437) == "arc"
    assert kind(196536) == "arc"
    for invalid_id in (191944, 191945, 191946, 196045, 196436):
        with pytest.raises(ValueError, match="invalid body token"):
            kind(invalid_id)

    color_contract = sft_module._STRUCTURAL_RESPONSE_LOSS_CONTRACT[
        "valid_registered_color_class"
    ]
    assert color_contract == {
        "minimum": 302152,
        "maximum": 306249,
        "count": 4098,
        "adjacent_coordinate_range_excluded": [262149, 302148],
        "pre_color_gap_excluded": [302149, 302151],
        "pre_arc_gap_excluded": [306250, 306641],
        "adjacent_arc_range_excluded": [306642, 306741],
    }


@pytest.mark.parametrize("reduction", ["token", "sample"])
def test_accumulation_window_normalization_has_loss_and_gradient_parity(
    reduction: str,
) -> None:
    torch.manual_seed(29)
    full_logits = torch.randn(4, 5, 13, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2, 3],
            [-100, 4, 5, 6, -100],
            [-100, -100, 7, 8, 9],
            [-100, 3, 2, -100, -100],
        ],
        dtype=torch.long,
    )
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, -1] = torch.tensor([4.0, 1.0, 4.0, 1.0])
    active = labels[:, 1:].ne(-100)
    global_denominator = (
        weights[:, 1:][active].sum()
        if reduction == "token"
        else torch.tensor(labels.shape[0], dtype=torch.float32)
    )

    full_loss = _structural_weighted_causal_lm_loss(
        full_logits,
        labels,
        weights,
        reduction=reduction,
        normalization_denominator=global_denominator,
    )
    full_gradient = torch.autograd.grad(full_loss, full_logits)[0]

    split_logits = full_logits.detach().clone().requires_grad_(True)
    split_loss = sum(
        _structural_weighted_causal_lm_loss(
            split_logits[start:end],
            labels[start:end],
            weights[start:end],
            reduction=reduction,
            normalization_denominator=global_denominator,
        )
        for start, end in ((0, 1), (1, 3), (3, 4))
    )
    split_gradient = torch.autograd.grad(split_loss, split_logits)[0]

    assert torch.allclose(split_loss, full_loss, rtol=1e-6, atol=1e-7)
    assert torch.allclose(split_gradient, full_gradient, rtol=1e-6, atol=1e-7)


class _TinyStructuralDataset(torch.utils.data.Dataset):
    def __init__(self, *, eos_weight: float, path_weight: float) -> None:
        role = 2 if path_weight > 1.0 else 1
        target = 3 if path_weight > 1.0 else 5
        self.features = []
        for index in range(48):
            labels = torch.tensor([-100, 1 + index % 2, 2, target], dtype=torch.long)
            labels[1 : 3 - index % 3] = -100
            weights = torch.ones(4, dtype=torch.float32)
            if eos_weight > 1.0:
                weights[-1] = eos_weight
            roles = torch.zeros(4, dtype=torch.long)
            roles[-1] = role
            self.features.append(
                {
                    "input_ids": torch.tensor(
                        [index % 7, (index + 1) % 7, (index + 2) % 7, (index + 3) % 7],
                        dtype=torch.long,
                    ),
                    "attention_mask": torch.ones(4, dtype=torch.long),
                    "labels": labels,
                    "loss_weights": weights,
                    "response_token_roles": roles,
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.features[index].items()}


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, initial_logits: torch.Tensor) -> None:
        super().__init__()
        self.logit_table = torch.nn.Parameter(initial_logits.clone())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del attention_mask
        vocabulary_offset = torch.linspace(
            -0.03,
            0.03,
            self.logit_table.shape[-1],
            device=input_ids.device,
            dtype=self.logit_table.dtype,
        )
        logits = self.logit_table.unsqueeze(0) + (
            input_ids.to(self.logit_table.dtype).unsqueeze(-1) * vocabulary_offset
        )
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return outputs


class _CapturingSGD(torch.optim.SGD):
    captured_gradient: torch.Tensor | None = None

    def step(self, closure=None):
        parameter = self.param_groups[0]["params"][0]
        self.captured_gradient = parameter.grad.detach().clone()
        return super().step(closure)


def _stack_tiny_features(dataset: _TinyStructuralDataset) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([feature[key] for feature in dataset.features])
        for key in dataset.features[0]
    }


def _collate_tiny_features(
    features: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}


def _run_real_trainer_update(
    *,
    tmp_path: Path,
    case_name: str,
    dataset: _TinyStructuralDataset,
    initial_logits: torch.Tensor,
    eos_weight: float,
    path_weight: float,
    reduction: str,
    gradient_accumulation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    from transformers import Trainer, TrainingArguments

    class TinyTrainer(_StructuralResponseLossMixin, Trainer):
        pass

    batch_size = 48 // gradient_accumulation_steps
    model = _TinyCausalLM(initial_logits)
    optimizer = _CapturingSGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    arguments = TrainingArguments(
        output_dir=str(
            tmp_path / f"{case_name}_{reduction}_ga{gradient_accumulation_steps}"
        ),
        max_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.05,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        max_grad_norm=0.0,
    )
    trainer = TinyTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=_collate_tiny_features,
        optimizers=(optimizer, scheduler),
    )
    trainer._svg_structural_response_loss_enabled = True
    trainer._svg_path_terminator_class_mass_weight = path_weight
    trainer._svg_structural_response_loss_reduction = reduction
    trainer.model_accepts_loss_kwargs = True
    train_output = trainer.train()
    assert trainer.accelerator.gradient_accumulation_steps == 1
    if optimizer.captured_gradient is None:
        raise AssertionError("Real Trainer optimizer did not capture an accumulated gradient.")
    logged_losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    assert len(logged_losses) == 1
    eval_loss = float(trainer.evaluate()["eval_loss"])
    return (
        model.logit_table.detach().clone(),
        optimizer.captured_gradient,
        float(train_output.training_loss),
        logged_losses[0],
        eval_loss,
    )


@pytest.mark.parametrize(
    ("case_name", "eos_weight", "path_weight", "reduction"),
    [
        ("eos8", 8.0, 1.0, "token"),
        ("path4", 1.0, 4.0, "token"),
        ("eos8", 8.0, 1.0, "sample"),
        ("path4", 1.0, 4.0, "sample"),
    ],
)
def test_real_trainer_ga_partitions_match_manual_reference_loss_gradient_delta_and_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    eos_weight: float,
    path_weight: float,
    reduction: str,
) -> None:
    monkeypatch.setattr(sft_module, "_OFFICIAL_REGISTERED_COLOR_MIN", 3)
    monkeypatch.setattr(sft_module, "_OFFICIAL_REGISTERED_COLOR_MAX", 4)
    torch.manual_seed(101)
    initial_logits = torch.randn(4, 7, dtype=torch.float32) * 0.2
    dataset = _TinyStructuralDataset(eos_weight=eos_weight, path_weight=path_weight)
    full_batch = _stack_tiny_features(dataset)
    reference_model = _TinyCausalLM(initial_logits)
    reference_outputs = reference_model(
        input_ids=full_batch["input_ids"],
        attention_mask=full_batch["attention_mask"],
    )
    denominator = _structural_accumulation_window_denominator(
        [full_batch],
        path_terminator_class_mass_weight=path_weight,
        reduction=reduction,
    )
    reference_loss = _structural_weighted_causal_lm_loss(
        reference_outputs["logits"],
        full_batch["labels"],
        full_batch["loss_weights"],
        response_token_roles=full_batch["response_token_roles"],
        path_terminator_class_mass_weight=path_weight,
        reduction=reduction,
        normalization_denominator=denominator,
    )
    reference_gradient = torch.autograd.grad(
        reference_loss,
        reference_model.logit_table,
    )[0]
    reference_parameter = initial_logits - 0.05 * reference_gradient
    reference_eval_model = _TinyCausalLM(reference_parameter)

    for gradient_accumulation_steps in (1, 6, 48):
        parameter, gradient, training_loss, logged_loss, eval_loss = (
            _run_real_trainer_update(
                tmp_path=tmp_path,
                case_name=case_name,
                dataset=dataset,
                initial_logits=initial_logits,
                eos_weight=eos_weight,
                path_weight=path_weight,
                reduction=reduction,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        )
        assert torch.allclose(gradient, reference_gradient, rtol=2e-6, atol=2e-7)
        assert torch.allclose(parameter, reference_parameter, rtol=2e-6, atol=2e-7)
        reference_loss_value = float(reference_loss.detach())
        assert training_loss == pytest.approx(reference_loss_value, rel=2e-6, abs=2e-7)
        assert logged_loss == pytest.approx(reference_loss_value, rel=5e-4, abs=5e-4)
        eval_batch_size = 48 // gradient_accumulation_steps
        reference_eval_loss = 0.0
        with torch.no_grad():
            for start in range(0, 48, eval_batch_size):
                end = start + eval_batch_size
                batch_loss = reference_eval_model(
                    input_ids=full_batch["input_ids"][start:end],
                    attention_mask=full_batch["attention_mask"][start:end],
                    labels=full_batch["labels"][start:end],
                )["loss"]
                reference_eval_loss += float(batch_loss) * eval_batch_size / 48
        assert eval_loss == pytest.approx(reference_eval_loss, rel=2e-6, abs=2e-7)


class _BatchSampleBase:
    def get_batch_samples(self, *_args, **_kwargs):
        return [], None


class _StructuralGuardProbe(_StructuralResponseLossMixin, _BatchSampleBase):
    pass


@pytest.mark.parametrize(
    ("processes", "accelerator_ga", "message"),
    [(2, 1, "single-process"), (1, 2, "Accelerator GAS=1")],
)
def test_structural_trainer_fails_fast_outside_supported_runtime_contract(
    processes: int,
    accelerator_ga: int,
    message: str,
) -> None:
    probe = _StructuralGuardProbe()
    probe._svg_structural_response_loss_enabled = True
    probe.accelerator = SimpleNamespace(
        num_processes=processes,
        gradient_accumulation_steps=accelerator_ga,
    )
    with pytest.raises(RuntimeError, match=message):
        probe.get_batch_samples(iter(()), 1, torch.device("cpu"))


@pytest.mark.parametrize(
    "labels",
    [
        [
            -100,
            _registered(196998),
            _registered(151938),
            _registered(151943),
            _registered(151943),
            _registered(196999),
        ],
        [
            -100,
            _registered(196998),
            _registered(151938),
            _registered(151943),
            _registered(151943),
            _registered(191946),
        ],
    ],
)
def test_malformed_or_truncated_discrete_grammar_fails_fast(labels: list[int]) -> None:
    with pytest.raises(ValueError):
        _official_discrete_response_loss_annotations(
            labels,
            tokenizer=_NamedTokenTokenizer(),
            eos_weight=2.0,
            path_terminator_class_mass_weight=1.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_eos_loss_weight": 8.01},
        {"response_path_terminator_class_mass_weight": 4.01},
        {"response_eos_loss_weight": 0.99},
        {"response_path_terminator_class_mass_weight": True},
        {"response_eos_loss_weight": 2.0, "response_path_terminator_class_mass_weight": 2.0},
        {"structural_response_loss_reduction": "per_sample"},
        {"structural_response_loss_reduction": None},
        {"structural_response_loss_reduction": True},
        {"structural_response_loss_reduction": "sample"},
    ],
)
def test_structural_weight_caps_and_simultaneous_arms_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SFTConfig(**kwargs)


def test_sample_structural_reduction_is_an_explicit_valid_opt_in() -> None:
    config = SFTConfig(
        response_eos_loss_weight=2.0,
        structural_response_loss_reduction="sample",
    )

    assert config.structural_response_loss_reduction == "sample"


def test_raw_xml_rejects_structural_weight_opt_in_before_loading_data() -> None:
    with pytest.raises(ValueError, match="official cached omnisvg_discrete"):
        TextToSVGSFTTrainer(
            model_config=ModelTrainingConfig(model_id="unused", revision="unused"),
            lora_config=object(),  # type: ignore[arg-type]
            sft_config=SFTConfig(response_eos_loss_weight=2.0, do_eval=False),
            train_data_path="unused.jsonl",
            eval_data_path=None,
            instruction_mode="description_only",
            target_representation="raw_xml",
        )
