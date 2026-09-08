from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from svg_agentic_slm.train.chunked_causal_lm_loss import install_chunked_causal_lm_loss


def _tiny_peft_gemma() -> object:
    config = Gemma4TextConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=32,
        layer_types=["full_attention", "sliding_attention"],
        final_logit_softcapping=12.0,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=67,
        tie_word_embeddings=False,
    )
    model = Gemma4ForCausalLM(config)
    return get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            task_type="CAUSAL_LM",
        ),
    )


def test_chunked_loss_matches_gemma_loss_and_trainable_gradients() -> None:
    torch.manual_seed(7)
    reference = _tiny_peft_gemma()
    chunked = copy.deepcopy(reference)
    install_chunked_causal_lm_loss(chunked, chunk_size=3)
    reference.train()
    chunked.train()

    input_ids = torch.randint(3, 67, (2, 13))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :5] = -100

    reference_loss = reference(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss
    chunked_loss = chunked(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss
    torch.testing.assert_close(chunked_loss, reference_loss, rtol=1e-5, atol=1e-6)

    reference_loss.backward()
    chunked_loss.backward()
    reference_gradients = {
        name: parameter.grad
        for name, parameter in reference.named_parameters()
        if parameter.requires_grad
    }
    chunked_gradients = {
        name: parameter.grad
        for name, parameter in chunked.named_parameters()
        if parameter.requires_grad
    }
    assert reference_gradients.keys() == chunked_gradients.keys()
    assert any("lora_" in name and gradient is not None for name, gradient in chunked_gradients.items())
    assert any("lm_head" in name and gradient is not None for name, gradient in chunked_gradients.items())
    for name, reference_gradient in reference_gradients.items():
        chunked_gradient = chunked_gradients[name]
        assert reference_gradient is not None, name
        assert chunked_gradient is not None, name
        torch.testing.assert_close(chunked_gradient, reference_gradient, rtol=2e-5, atol=2e-6)


def test_unlabeled_forward_preserves_standard_logits() -> None:
    model = _tiny_peft_gemma()
    expected = model(input_ids=torch.tensor([[2, 3, 4]])).logits
    install_chunked_causal_lm_loss(model, chunk_size=2)
    actual = model(input_ids=torch.tensor([[2, 3, 4]])).logits
    torch.testing.assert_close(actual, expected)


def test_tiny_train_validation_then_single_held_out_test(tmp_path) -> None:
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    model = _tiny_peft_gemma()
    install_chunked_causal_lm_loss(model, chunk_size=3)

    def record(offset: int) -> dict[str, object]:
        input_ids = (torch.arange(12) + offset).remainder(64).add(3)
        labels = input_ids.clone()
        labels[:4] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "checkpoints"),
            num_train_epochs=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=0.0,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,
            remove_unused_columns=False,
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=[record(0), record(1), record(2), record(3)],
        eval_dataset=[record(4), record(5)],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)],
    )
    trainer.train()
    assert trainer.state.epoch == 2.0
    assert trainer.state.best_model_checkpoint is not None

    test_metrics = trainer.evaluate(
        eval_dataset=[record(6), record(7)],
        metric_key_prefix="test",
    )
    assert "test_loss" in test_metrics
