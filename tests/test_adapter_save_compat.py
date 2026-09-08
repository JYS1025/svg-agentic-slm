from types import SimpleNamespace

import pytest
from transformers.trainer import TRAINING_ARGS_NAME

from svg_agentic_slm.train.sft_trainer import (
    _RowDeltaAdapterSaveMixin,
    _adapter_serialization_manifest,
)


class _CapturingModel:
    def __init__(self) -> None:
        self.destination = None
        self.kwargs = None

    def save_pretrained(self, destination, **kwargs) -> None:
        self.destination = destination
        self.kwargs = kwargs


class _Accelerator:
    @staticmethod
    def unwrap_model(model):
        return model


class _RowDeltaTrainer(_RowDeltaAdapterSaveMixin):
    _save_row_deltas_only = True

    def __init__(self, args) -> None:
        self.args = args
        self.model = _CapturingModel()
        self.accelerator = _Accelerator()


@pytest.mark.parametrize(
    ("args", "expected_safe", "expected_source"),
    [
        (SimpleNamespace(), True, "peft.save_pretrained_default"),
        (
            SimpleNamespace(save_safetensors=False),
            False,
            "training_arguments.save_safetensors",
        ),
    ],
)
def test_row_delta_save_supports_training_args_with_or_without_save_safetensors(
    tmp_path,
    args,
    expected_safe,
    expected_source,
) -> None:
    trainer = _RowDeltaTrainer(args)
    state_dict = {"adapter.delta": object()}

    trainer._save(str(tmp_path), state_dict=state_dict)

    assert trainer.model.destination == str(tmp_path)
    assert trainer.model.kwargs == {
        "state_dict": state_dict,
        "safe_serialization": expected_safe,
        "save_embedding_layers": False,
    }
    assert (tmp_path / TRAINING_ARGS_NAME).is_file()
    assert _adapter_serialization_manifest(args) == {
        "safe_serialization": expected_safe,
        "source": expected_source,
    }
