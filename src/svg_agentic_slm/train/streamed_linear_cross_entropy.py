"""Portable streamed cross entropy for a frozen, ordinary linear vocabulary head.

The forward computes each chunk's first derivative immediately, then retains only
the hidden-state gradient. Backward scales that saved derivative; it does not
recompute the vocabulary projection. This follows the chunked/precomputed-gradient
schedule used by Liger's fused linear CE, using native PyTorch operations instead
of a device-specific fused kernel. No vocabulary classes or gradients are pruned.

References:
    https://arxiv.org/abs/2410.10989 (Liger Kernel, 2024/2025)
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/
        fused_linear_cross_entropy.py
    https://arxiv.org/abs/2411.09009 (Cut Cross Entropy, 2024/2025)

This is not CCE's on-chip log-sum-exp implementation or its gradient filtering.
Only first-order derivatives are supported. FP16 is deliberately unsupported:
precomputing unscaled gradients can lose values that later GradScaler scaling
would otherwise preserve. BF16, FP32, and FP64 work on CPU and CUDA.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as functional


def supports_streamed_linear_cross_entropy(hidden_states: Any, lm_head: Any) -> bool:
    """Return whether bypassing ``lm_head.forward`` preserves its supported contract.

    Unsupported heads/dtypes must use the ordinary autograd/checkpointed path.
    Forward/backward hooks, tensor subclasses, and monkey-patched modules also use
    that fallback rather than silently bypassing their behavior.
    """
    tensor_types = (torch.Tensor, torch.nn.Parameter)
    allowed_dtypes = (torch.float32, torch.bfloat16, torch.float64)
    if (
        type(hidden_states) not in tensor_types
        or hidden_states.device.type not in ("cpu", "cuda")
        or hidden_states.layout != torch.strided
        or hidden_states.dtype not in allowed_dtypes
        or type(lm_head) is not torch.nn.Linear
        or getattr(lm_head.forward, "__func__", None) is not torch.nn.Linear.forward
        or getattr(lm_head, "_compiled_call_impl", None) is not None
    ):
        return False
    for parameter in (lm_head.weight, lm_head.bias):
        if parameter is not None and (
            type(parameter) not in tensor_types
            or parameter.requires_grad
            or parameter.dtype not in allowed_dtypes
            or parameter.device != hidden_states.device
            or parameter.layout != torch.strided
        ):
            return False
    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
        if getattr(lm_head, name, None):
            return False
        if getattr(torch.nn.modules.module, "_global" + name, None):
            return False
    if torch.is_autocast_enabled(hidden_states.device.type):
        if torch.get_autocast_dtype(hidden_states.device.type) != torch.bfloat16:
            return False
    elif lm_head.weight.dtype != hidden_states.dtype or (
        lm_head.bias is not None and lm_head.bias.dtype != hidden_states.dtype
    ):
        return False
    return True


def _denominator(value: Any, *, active_count: int, device: Any, dtype: Any) -> torch.Tensor:
    if value is None:
        return torch.tensor(active_count, dtype=dtype, device=device)
    if torch.is_tensor(value):
        if value.numel() != 1 or value.requires_grad:
            raise ValueError("Normalization denominator must be a constant scalar.")
        result = value.to(dtype=dtype).reshape(())
    else:
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Normalization denominator must be finite.") from exc
        if not math.isfinite(scalar):
            raise ValueError("Normalization denominator must be finite.")
        result = torch.tensor(scalar, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(result) & (result >= active_count)):
        raise ValueError("Normalization denominator must be finite and >= active token count.")
    return result.to(device=device)


class _StreamedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        labels: torch.Tensor,
        denominator: torch.Tensor,
        softcap: float | None,
        chunk_size: int,
        ignore_index: int,
        compute_gradient: bool,
    ) -> torch.Tensor:
        total_loss = torch.zeros((), dtype=denominator.dtype, device=hidden_states.device)
        hidden_gradient = torch.empty_like(hidden_states) if compute_gradient else None
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            # Function.forward normally runs without grad. Build and discard one
            # local graph at a time; the surrounding autocast context is retained.
            with torch.set_grad_enabled(compute_gradient):
                hidden_chunk = hidden_states[start:end].detach()
                hidden_chunk.requires_grad_(compute_gradient)
                logits = functional.linear(hidden_chunk, weight, bias)
                if softcap is not None:
                    logits = torch.tanh(logits / softcap) * softcap
                chunk_loss = functional.cross_entropy(
                    logits.to(dtype=denominator.dtype),
                    labels[start:end],
                    ignore_index=ignore_index,
                    reduction="sum",
                )
                if compute_gradient:
                    chunk_gradient = torch.autograd.grad(
                        chunk_loss / denominator,
                        hidden_chunk,
                        create_graph=False,
                        retain_graph=False,
                    )[0]
            total_loss.add_(chunk_loss.detach())
            if hidden_gradient is not None:
                hidden_gradient[start:end].copy_(chunk_gradient)
                del chunk_gradient
            # Do not overlap the preceding chunk's logits with the next GEMM.
            del hidden_chunk, logits, chunk_loss
        if hidden_gradient is not None:
            ctx.save_for_backward(hidden_gradient)
        return total_loss / denominator

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("Streamed linear cross entropy supports first-order gradients only.")
        (hidden_gradient,) = ctx.saved_tensors
        # Do not modify the saved gradient: retain_graph/repeated backward and
        # multiple scaled consumers must each receive the original derivative.
        return (hidden_gradient * grad_output,) + (None,) * 8


def streamed_linear_cross_entropy(
    *,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Linear,
    final_logit_softcapping: float | None,
    chunk_size: int,
    normalization_denominator: Any = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return mean CE for aligned ``[tokens, hidden]`` states and ``[tokens]`` labels.

    The caller performs the causal shift. Ignored labels contribute zero gradient;
    gathering active rows before this function avoids their projection as well.
    An external denominator preserves Trainer's accumulation-window token mean.
    For BF16/FP32, logits are promoted to FP32 for CE; FP64 is retained for accurate
    reference/gradcheck use. Ordinary scalar loss scaling is applied in backward.
    Evaluation under no_grad/inference_mode does not precompute unused gradients.
    """
    if not supports_streamed_linear_cross_entropy(hidden_states, lm_head):
        raise TypeError(
            "Streamed CE requires a frozen plain Linear head and CPU/CUDA BF16/FP32/FP64 "
            "without custom hooks; use the checkpointed loss fallback otherwise."
        )
    if (
        hidden_states.ndim != 2
        or labels.ndim != 1
        or labels.shape[0] != hidden_states.shape[0]
        or hidden_states.shape[1] != lm_head.in_features
        or hidden_states.shape[0] == 0
    ):
        raise ValueError("Expected nonempty hidden_states [tokens, hidden] and labels [tokens].")
    if labels.dtype != torch.long or labels.device != hidden_states.device:
        raise ValueError("Labels must be int64 on the hidden-state device.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if final_logit_softcapping is not None and (
        not math.isfinite(final_logit_softcapping) or final_logit_softcapping <= 0
    ):
        raise ValueError("final_logit_softcapping must be finite and positive, or None.")
    active_count = int(labels.ne(ignore_index).sum().item())
    if active_count == 0:
        raise ValueError("Streamed linear cross entropy requires active labels.")
    loss_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
    denominator = _denominator(
        normalization_denominator,
        active_count=active_count,
        device=hidden_states.device,
        dtype=loss_dtype,
    )
    return _streamed_linear_cross_entropy_prevalidated(
        hidden_states=hidden_states,
        labels=labels,
        lm_head=lm_head,
        final_logit_softcapping=final_logit_softcapping,
        chunk_size=chunk_size,
        normalization_denominator=denominator,
        ignore_index=ignore_index,
    )


def _streamed_linear_cross_entropy_prevalidated(
    *,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Linear,
    final_logit_softcapping: float | None,
    chunk_size: int,
    normalization_denominator: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Apply streamed CE after the caller validated shape, support, and denominator."""
    return _StreamedLinearCrossEntropy.apply(
        hidden_states,
        lm_head.weight,
        lm_head.bias,
        labels,
        normalization_denominator,
        final_logit_softcapping,
        chunk_size,
        ignore_index,
        torch.is_grad_enabled() and hidden_states.requires_grad,
    )
