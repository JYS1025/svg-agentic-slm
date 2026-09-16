"""Memory-bounded causal-LM loss for long response-only SVG targets."""

from __future__ import annotations

import math
import types
from functools import wraps
from typing import Any


def install_chunked_causal_lm_loss(
    model: Any,
    *,
    chunk_size: int,
    project_only_active_tokens: bool = True,
    loss_backend: str = "auto",
) -> dict[str, Any]:
    """Patch a PEFT base model to avoid materializing full sequence logits.

    The complete Gemma backbone forward is unchanged. For labeled train/eval calls,
    only supervised next-token positions reach the vocabulary projection and cross
    entropy, in chunks of at most ``chunk_size`` tokens across the batch. A
    supported frozen linear head precomputes each chunk's hidden-state gradient;
    other heads retain the checkpointed autograd path. The complete backbone still
    sees prompt and padding positions. Calls
    without labels retain the original generation path. Set
    ``project_only_active_tokens=False`` to benchmark the original projection of
    all shifted positions, with ``chunk_size`` sequence positions per example.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(project_only_active_tokens, bool):
        raise TypeError("project_only_active_tokens must be boolean.")
    if loss_backend not in {"auto", "checkpoint", "streamed"}:
        raise ValueError("loss_backend must be auto, checkpoint, or streamed.")
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
        num_items_in_batch = backbone_kwargs.pop("num_items_in_batch", None)
        backbone_kwargs["use_cache"] = False
        backbone_kwargs["return_dict"] = True
        outputs = self.model(**backbone_kwargs)
        selected_backend = _resolve_loss_backend(
            outputs.last_hidden_state,
            self.lm_head,
            loss_backend,
        )
        self._svg_chunked_causal_lm_loss["last_backend"] = selected_backend
        loss = chunked_causal_lm_loss(
            hidden_states=outputs.last_hidden_state,
            labels=labels,
            lm_head=self.lm_head,
            final_logit_softcapping=softcap,
            chunk_size=chunk_size,
            use_checkpoint=self.training,
            normalization_denominator=num_items_in_batch,
            project_only_active_tokens=project_only_active_tokens,
            loss_backend=selected_backend,
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
        "name": "chunked_causal_lm",
        "backend_requested": loss_backend,
        "last_backend": None,
        "backend_policy": "streamed_for_supported_frozen_linear_otherwise_checkpoint",
        "chunk_size": chunk_size,
        "shift": "causal_next_token",
        "selection": "labels_not_equal_ignore_index",
        "projection": (
            "active_shifted_labels_only" if project_only_active_tokens else "all_shifted_positions"
        ),
        "chunk_unit": (
            "active_tokens_across_batch"
            if project_only_active_tokens
            else "sequence_positions_per_example"
        ),
        "reduction": "sum_over_chunks_then_active_token_mean",
        "accumulation_normalization": (
            "trainer_num_items_in_batch_when_provided_otherwise_local_active_tokens"
        ),
        "logit_softcapping": softcap,
        "cross_entropy_dtype": "float32",
        "checkpoint": "non_reentrant_during_training_when_checkpoint_backend_selected",
        "returns_labeled_logits": False,
    }
    base_model._svg_chunked_causal_lm_loss = provenance
    return provenance


def _resolve_loss_backend(hidden_states: Any, lm_head: Any, requested: str) -> str:
    if requested not in {"auto", "checkpoint", "streamed"}:
        raise ValueError("loss_backend must be auto, checkpoint, or streamed.")
    if requested == "checkpoint":
        return requested
    from svg_agentic_slm.train.streamed_linear_cross_entropy import (
        supports_streamed_linear_cross_entropy,
    )

    supported = supports_streamed_linear_cross_entropy(hidden_states, lm_head)
    if not supported and requested == "streamed":
        raise ValueError(
            "Streamed loss requires a supported frozen plain Linear head and BF16/FP32/FP64; "
            "use loss_backend=auto or checkpoint for this model."
        )
    return "streamed" if supported else "checkpoint"


def chunked_causal_lm_loss(
    *,
    hidden_states: Any,
    labels: Any,
    lm_head: Any,
    final_logit_softcapping: float | None,
    chunk_size: int,
    use_checkpoint: bool,
    normalization_denominator: Any = None,
    ignore_index: int = -100,
    project_only_active_tokens: bool = True,
    loss_backend: str = "checkpoint",
) -> Any:
    """Compute shifted causal-LM loss, projecting only supervised tokens by default.

    Prompt and padding rows with ignored next-token labels have exactly zero loss
    gradient. Gathering the remaining hidden rows before ``lm_head`` avoids their
    expensive vocabulary projection, while autograd scatters gradients back to
    the original backbone positions. ``False`` preserves the dense chunk baseline.
    """
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "Expected hidden_states [batch, sequence, hidden] and labels [batch, sequence]."
        )
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must have identical batch/sequence dimensions.")
    if hidden_states.shape[1] < 2:
        raise ValueError("Causal-LM loss requires at least two sequence positions.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(project_only_active_tokens, bool):
        raise TypeError("project_only_active_tokens must be boolean.")

    shifted_labels = labels[:, 1:]
    shifted_length = hidden_states.shape[1] - 1
    if project_only_active_tokens:
        # nonzero synchronizes once on CUDA to determine the dynamic output size;
        # reuse those indices rather than synchronizing for a separate token sum.
        active_indices = shifted_labels.reshape(-1).ne(ignore_index).nonzero(as_tuple=True)[0]
        active_count = active_indices.numel()
    else:
        active_count = int(shifted_labels.ne(ignore_index).sum().item())
    if active_count == 0:
        raise ValueError("Chunked causal-LM loss received no active shifted labels.")

    if normalization_denominator is None:
        denominator = torch.tensor(active_count, dtype=torch.float32, device=hidden_states.device)
    elif torch.is_tensor(normalization_denominator):
        if normalization_denominator.numel() != 1:
            raise ValueError("Chunked causal-LM normalization denominator must be scalar.")
        denominator = normalization_denominator.to(dtype=torch.float32).reshape(())
        # One device-to-host check for externally supplied CUDA tensors. Validate
        # CPU tensors before transfer so they do not introduce an extra GPU sync.
        # The local active-count denominator above is already valid on the host.
        if not bool(torch.isfinite(denominator) & (denominator >= active_count)):
            raise ValueError("Chunked causal-LM normalization denominator is invalid.")
        denominator = denominator.to(device=hidden_states.device)
    else:
        try:
            denominator_value = float(normalization_denominator)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Chunked causal-LM normalization denominator must be a finite number."
            ) from exc
        if not math.isfinite(denominator_value) or denominator_value < active_count:
            raise ValueError("Chunked causal-LM normalization denominator is invalid.")
        denominator = torch.tensor(denominator_value, dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(denominator) & (denominator >= active_count)):
            raise ValueError("Chunked causal-LM normalization denominator is invalid.")
        denominator = denominator.to(device=hidden_states.device)

    if project_only_active_tokens:
        # Two-dimensional indexing avoids copying the entire non-contiguous
        # hidden_states[:, :-1] view merely to flatten its batch/sequence axes.
        selected_hidden_states = hidden_states[
            active_indices // shifted_length, active_indices % shifted_length
        ]
        selected_labels = shifted_labels.reshape(-1).index_select(0, active_indices)
        chunked_length = active_count
    else:
        chunked_length = shifted_length

    selected_backend = _resolve_loss_backend(hidden_states, lm_head, loss_backend)
    if selected_backend == "streamed":
        from svg_agentic_slm.train.streamed_linear_cross_entropy import (
            _streamed_linear_cross_entropy_prevalidated,
        )

        return _streamed_linear_cross_entropy_prevalidated(
            hidden_states=(
                selected_hidden_states
                if project_only_active_tokens
                else hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
            ),
            labels=(selected_labels if project_only_active_tokens else shifted_labels.reshape(-1)),
            lm_head=lm_head,
            final_logit_softcapping=final_logit_softcapping,
            chunk_size=chunk_size,
            normalization_denominator=denominator,
            ignore_index=ignore_index,
        )

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
    for start in range(0, chunked_length, chunk_size):
        end = min(start + chunk_size, chunked_length)
        if project_only_active_tokens:
            hidden_chunk = selected_hidden_states[start:end]
            target_chunk = selected_labels[start:end]
        else:
            hidden_chunk = hidden_states[:, start:end, :]
            target_chunk = shifted_labels[:, start:end]
        if use_checkpoint and torch.is_grad_enabled():
            chunk_loss = checkpoint(
                loss_sum_for_chunk,
                hidden_chunk,
                target_chunk,
                use_reentrant=False,
                # A trainable head can contain adapter dropout. Recompute it
                # with the same mask so checkpointing preserves its gradients.
                preserve_rng_state=True,
            )
        else:
            chunk_loss = loss_sum_for_chunk(hidden_chunk, target_chunk)
        total_loss = total_loss + chunk_loss
    return total_loss / denominator
