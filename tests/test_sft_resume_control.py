from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from svg_agentic_slm.train.sft_trainer import (  # noqa: E402
    SFTConfig,
    _early_stopping_resume_contract,
)


def _write_trainer_state(
    checkpoint: Path,
    *,
    patience: int = 2,
    threshold: float = 0.01,
    counter: int = 1,
) -> None:
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "stateful_callbacks": {
                    "EarlyStoppingCallback": {
                        "args": {
                            "early_stopping_patience": patience,
                            "early_stopping_threshold": threshold,
                        },
                        "attributes": {
                            "early_stopping_patience_counter": counter,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_early_stopping_resume_contract_restores_matching_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    _write_trainer_state(checkpoint)

    result = _early_stopping_resume_contract(
        resume_from_checkpoint=str(checkpoint),
        patience=2,
        threshold=0.01,
    )

    assert result["restore_callback_states_from_checkpoint"] is True
    assert result["restored_patience_counter"] == 1
    assert result["checkpoint"] == str(checkpoint.resolve())


@pytest.mark.parametrize(
    ("patience", "threshold"),
    [(3, 0.01), (2, 0.02)],
)
def test_early_stopping_resume_rejects_changed_settings(
    tmp_path: Path,
    patience: int,
    threshold: float,
) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    _write_trainer_state(checkpoint)

    with pytest.raises(ValueError, match="settings differ"):
        _early_stopping_resume_contract(
            resume_from_checkpoint=str(checkpoint),
            patience=patience,
            threshold=threshold,
        )


def test_early_stopping_resume_rejects_missing_callback_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="restorable"):
        _early_stopping_resume_contract(
            resume_from_checkpoint=str(checkpoint),
            patience=2,
            threshold=0.01,
        )


def test_resume_without_early_stopping_does_not_require_callback_state() -> None:
    result = _early_stopping_resume_contract(
        resume_from_checkpoint="checkpoint-without-callback-state",
        patience=None,
        threshold=0.0,
    )

    assert result["restore_callback_states_from_checkpoint"] is False


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "2"])
def test_early_stopping_patience_rejects_non_positive_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive or null"):
        SFTConfig(early_stopping_patience=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [True, -0.1, float("inf"), float("nan"), "0.1"])
def test_early_stopping_threshold_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        SFTConfig(early_stopping_threshold=invalid)  # type: ignore[arg-type]


class _ConstantLossModel(torch.nn.Module):
    accepts_loss_kwargs = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del input_ids, labels
        return {"loss": self.weight.square() + 1.0}


def _constant_records() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([-100, 2]),
        }
        for _ in range(4)
    ]


def _run_early_stopping_trainer(
    output_dir: Path,
    *,
    resume_from_checkpoint: str | None = None,
    restore_callback_state: bool = False,
) -> int:
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    trainer = Trainer(
        model=_ConstantLossModel(),
        args=TrainingArguments(
            output_dir=str(output_dir),
            max_steps=6,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=0.0,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=4,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            restore_callback_states_from_checkpoint=restore_callback_state,
            remove_unused_columns=False,
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=_constant_records(),
        eval_dataset=_constant_records(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer.state.global_step


def test_restored_early_stopping_resume_matches_continuous_stop(tmp_path: Path) -> None:
    continuous_dir = tmp_path / "continuous"
    assert _run_early_stopping_trainer(continuous_dir) == 3
    checkpoint = continuous_dir / "checkpoint-2"
    contract = _early_stopping_resume_contract(
        resume_from_checkpoint=str(checkpoint),
        patience=2,
        threshold=0.0,
    )

    resumed_step = _run_early_stopping_trainer(
        tmp_path / "resumed",
        resume_from_checkpoint=str(checkpoint),
        restore_callback_state=contract["restore_callback_states_from_checkpoint"],
    )

    assert resumed_step == 3
