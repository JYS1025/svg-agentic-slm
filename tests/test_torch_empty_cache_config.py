import pytest
from transformers import TrainingArguments

from svg_agentic_slm.train.sft_trainer import SFTConfig


def test_torch_empty_cache_steps_round_trips_to_training_arguments(tmp_path) -> None:
    config = SFTConfig.from_dict({"torch_empty_cache_steps": 1})

    args = TrainingArguments(
        output_dir=str(tmp_path),
        torch_empty_cache_steps=config.torch_empty_cache_steps,
    )

    assert config.torch_empty_cache_steps == 1
    assert args.torch_empty_cache_steps == 1


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "1"])
def test_torch_empty_cache_steps_rejects_invalid_values(invalid) -> None:
    with pytest.raises(ValueError, match="torch_empty_cache_steps"):
        SFTConfig(torch_empty_cache_steps=invalid)
