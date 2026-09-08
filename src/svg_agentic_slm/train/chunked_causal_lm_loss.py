"""Memory-bounded causal-LM loss for long response-only SVG targets."""

from __future__ import annotations

import types
from functools import wraps
from typing import Any


def install_chunked_causal_lm_loss(model: Any, *, chunk_size: int) -> dict[str, Any]:
    """Patch a PEFT base model to avoid materializing full sequence logits.

    The complete Gemma backbone forward is unchanged. For labeled train/eval calls,
    only the vocabulary projection and cross entropy are evaluated in checkpointed
    target-token chunks. Calls without labels retain the original generation path.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not hasattr(model, "get_base_model"):
        raise TypeError("Chunked causal-LM loss requires a PEFT model.")
    peft_config = getattr(model, "active_peft_config", None)
    if peft_config is None or getattr(peft_config, "is_prompt_learning", False):
        raise TypeError("Chunked causal-LM loss supports non-prompt PEFT adapters only.")

    base_model = model.get_base_model()
    if getattr(base_model, "_svg_chunked_causal_lm_loss", None) is not None:
        raise RuntimeError("Chunked causal-LM loss is already installed on this model.")
    if not hasattr(base_model, "model") or not hasattr(base_model, "lm_head"):
        raise TypeError("Expected a causal-LM base model exposing `model` and `lm_head`.")
    config = getattr(base_model, "config", None)
    if config is None:
        raise TypeError("Expected a causal-LM base model exposing `config`.")
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    softcap = getattr(text_config, "final_logit_softcapping", None)
    original_forward = base_model.forward

    @wraps(original_forward)
    def chunked_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        labels = kwargs.get("labels")
        if labels is None:
            return original_forward(*args, **kwargs)
        if args:
            raise ValueError("Labeled chunked-loss forwards require keyword arguments.")

        unsupported = {
            name
            for name in (
                "pixel_values",
                "pixel_values_videos",
                "input_features",
                "input_features_mask",
            )
            if kwargs.get(name) is not None
        }
        if unsupported:
            raise ValueError(
                "Chunked SVG SFT loss is text-only; unsupported multimodal inputs: "
                + ", ".join(sorted(unsupported))
            )
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Chunked SVG SFT loss does not support cached training forwards.")

        backbone_kwargs = dict(kwargs)
        labels = backbone_kwargs.pop("labels")
        return_dict = backbone_kwargs.pop("return_dict", None)
        backbone_kwargs.pop("logits_to_keep", None)
        backbone_kwargs.pop("num_items_in_batch", None)
        backbone_kwargs["use_cache"] = False
        backbone_kwargs["return_dict"] = True
        outputs = self.model(**backbone_kwargs)
        loss = chunked_causal_lm_loss(
            hidden_states=outputs.last_hidden_state,
            labels=labels,
            lm_head=self.lm_head,
            final_logit_softcapping=softcap,
            chunk_size=chunk_size,
            use_checkpoint=self.training,
        )

        if return_dict is False:
            return (loss,)
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    base_model.forward = types.MethodType(chunked_forward, base_model)
    provenance = {
        "name": "checkpointed_chunked_causal_lm",
        "chunk_size": chunk_size,
        "shift": "causal_next_token",
        "selection": "labels_not_equal_ignore_index",
        "reduction": "sum_over_chunks_then_active_token_mean",
        "logit_softcapping": softcap,
        "cross_entropy_dtype": "float32",
        "checkpoint": "torch.utils.checkpoint.use_reentrant_false_during_training",
        "returns_labeled_logits": False,
    }
    base_model._svg_chunked_causal_lm_loss = provenance
    return provenance


def chunked_causal_lm_loss(
    *,
    hidden_states: Any,
    labels: Any,
    lm_head: Any,
    final_logit_softcapping: float | None,
    chunk_size: int,
    use_checkpoint: bool,
    ignore_index: int = -100,
) -> Any:
    """Compute standard shifted causal-LM mean loss without full logits."""
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected hidden_states [batch, sequence, hidden] and labels [batch, sequence].")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must have identical batch/sequence dimensions.")
    if hidden_states.shape[1] < 2:
        raise ValueError("Causal-LM loss requires at least two sequence positions.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    shifted_labels = labels[:, 1:]
    active_count = int(shifted_labels.ne(ignore_index).sum().item())
    if active_count == 0:
        raise ValueError("Chunked causal-LM loss received no active shifted labels.")

    def loss_sum_for_chunk(hidden_chunk: Any, target_labels: Any) -> Any:
        logits = lm_head(hidden_chunk.reshape(-1, hidden_states.shape[-1]))
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        return functional.cross_entropy(
            logits.float(),
            target_labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )

    total_loss = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    shifted_length = hidden_states.shape[1] - 1
    for start in range(0, shifted_length, chunk_size):
        end = min(start + chunk_size, shifted_length)
        hidden_chunk = hidden_states[:, start:end, :]
        target_chunk = shifted_labels[:, start:end]
        if use_checkpoint and torch.is_grad_enabled():
            chunk_loss = checkpoint(
                loss_sum_for_chunk,
                hidden_chunk,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_loss = loss_sum_for_chunk(hidden_chunk, target_chunk)
        total_loss = total_loss + chunk_loss
    return total_loss / active_count
