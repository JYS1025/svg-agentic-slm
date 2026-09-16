"""Independent first-order references for the portable streamed vocabulary loss."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from svg_agentic_slm.train.streamed_linear_cross_entropy import (  # noqa: E402
    streamed_linear_cross_entropy,
    supports_streamed_linear_cross_entropy,
)


def _reference(hidden, labels, head, softcap, denominator=None):
    logits = head(hidden)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    if logits.dtype != torch.float64:
        logits = logits.float()
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    return loss / (labels.ne(-100).sum() if denominator is None else denominator)


def _fixture(dtype=torch.float32, device="cpu"):
    torch.manual_seed(91)
    hidden = torch.randn(13, 7, dtype=dtype, device=device, requires_grad=True)
    head = torch.nn.Linear(7, 19, dtype=dtype, device=device).requires_grad_(False)
    labels = torch.tensor([-100, -100, -100, 4, 1, 6, -100, 3, 8, -100, 2, 0, -100], device=device)
    return hidden, labels, head


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("softcap", [None, 0.75, 30.0])
@pytest.mark.parametrize("chunk_size", [1, 5, 50])
def test_matches_independent_dense_loss_and_hidden_gradient(dtype, softcap, chunk_size):
    hidden, labels, head = _fixture(dtype)
    expected_hidden = hidden.detach().clone().requires_grad_()
    actual = streamed_linear_cross_entropy(
        hidden_states=hidden,
        labels=labels,
        lm_head=head,
        final_logit_softcapping=softcap,
        chunk_size=chunk_size,
    )
    expected = _reference(expected_hidden, labels, head, softcap)
    tolerance = dict(rtol=2e-2, atol=3e-4) if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(actual, expected, **tolerance)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, **tolerance)
    assert torch.count_nonzero(hidden.grad[labels.eq(-100)]) == 0
    assert head.weight.grad is None
    assert head.bias.grad is None


def test_fp64_gradcheck_with_softcap_and_noncontiguous_hidden():
    hidden = torch.randn(3, 6, dtype=torch.float64)[:, ::2].requires_grad_()
    assert not hidden.is_contiguous()
    head = torch.nn.Linear(3, 5, dtype=torch.float64).requires_grad_(False)
    labels = torch.tensor([2, -100, 4])
    assert torch.autograd.gradcheck(
        lambda value: streamed_linear_cross_entropy(
            hidden_states=value,
            labels=labels,
            lm_head=head,
            final_logit_softcapping=0.7,
            chunk_size=2,
        ),
        (hidden,),
    )


@pytest.mark.parametrize("scale", [0.37, 128.0, -2.0])
def test_external_denominator_and_repeated_scaled_backward(scale):
    hidden, labels, head = _fixture(torch.float64)
    expected_hidden = hidden.detach().clone().requires_grad_()
    denominator = torch.tensor(31)
    actual = streamed_linear_cross_entropy(
        hidden_states=hidden,
        labels=labels,
        lm_head=head,
        final_logit_softcapping=1.0,
        chunk_size=4,
        normalization_denominator=denominator,
    )
    expected = _reference(expected_hidden, labels, head, 1.0, denominator)
    expected_gradient = torch.autograd.grad(expected * scale, expected_hidden)[0]
    for _ in range(2):
        actual_gradient = torch.autograd.grad(actual * scale, hidden, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_saves_only_hidden_gradient_and_does_not_project_in_backward(monkeypatch):
    hidden, labels, head = _fixture()
    original_linear = torch.nn.functional.linear
    projected_rows = []

    def track_linear(values, weight, bias=None):
        projected_rows.append(values.shape[0])
        return original_linear(values, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", track_linear)
    actual = streamed_linear_cross_entropy(
        hidden_states=hidden,
        labels=labels,
        lm_head=head,
        final_logit_softcapping=30.0,
        chunk_size=5,
    )
    assert projected_rows == [5, 5, 3]
    assert len(actual.grad_fn.saved_tensors) == 1
    assert actual.grad_fn.saved_tensors[0].shape == hidden.shape
    actual.backward()
    assert projected_rows == [5, 5, 3]


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_evaluation_does_not_precompute_gradients(monkeypatch, mode):
    hidden, labels, head = _fixture()

    def unexpected_grad(*args, **kwargs):
        raise AssertionError("evaluation must not compute a gradient")

    monkeypatch.setattr(torch.autograd, "grad", unexpected_grad)
    with mode():
        actual = streamed_linear_cross_entropy(
            hidden_states=hidden,
            labels=labels,
            lm_head=head,
            final_logit_softcapping=None,
            chunk_size=5,
        )
        expected = _reference(hidden, labels, head, None)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_accumulated_optimizer_step_matches_dense_effective_batch():
    torch.manual_seed(23)
    inputs = torch.randn(3, 7, 5)
    labels = torch.tensor(
        [
            [-100, -100, -100, 1, 3, 5, 7],
            [-100, 3, 2, 1, 6, 8, -100],
            [-100, -100, -100, -100, -100, -100, -100],
        ]
    )
    backbone = torch.nn.Linear(5, 7)
    reference_backbone = copy.deepcopy(backbone)
    head = torch.nn.Linear(7, 19).requires_grad_(False)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference_backbone.parameters(), lr=0.1)
    _reference(reference_backbone(inputs).reshape(-1, 7), labels.reshape(-1), head, 0.75).backward()
    reference_optimizer.step()
    for batch_slice in (slice(0, 1), slice(1, 3)):
        streamed_linear_cross_entropy(
            hidden_states=backbone(inputs[batch_slice]).reshape(-1, 7),
            labels=labels[batch_slice].reshape(-1),
            lm_head=head,
            final_logit_softcapping=0.75,
            chunk_size=3,
            normalization_denominator=labels.ne(-100).sum(),
        ).backward()
    optimizer.step()
    for actual, expected in zip(backbone.parameters(), reference_backbone.parameters()):
        torch.testing.assert_close(actual, expected)


def test_cpu_bf16_autocast_and_grad_scaler_optimizer_update():
    torch.manual_seed(19)
    inputs = torch.randn(13, 5)
    _, labels, head = _fixture()
    initial_backbone = torch.nn.Linear(5, 7)
    results = []
    for streamed in (False, True):
        backbone = copy.deepcopy(initial_backbone)
        optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            hidden = backbone(inputs)
            assert supports_streamed_linear_cross_entropy(hidden, head)
            loss = (
                streamed_linear_cross_entropy(
                    hidden_states=hidden,
                    labels=labels,
                    lm_head=head,
                    final_logit_softcapping=30.0,
                    chunk_size=5,
                )
                if streamed
                else _reference(hidden, labels, head, 30.0)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        results.append([parameter.detach().clone() for parameter in backbone.parameters()])
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)


def test_unsupported_heads_and_dtypes_use_fallback_contract():
    hidden, _, head = _fixture()
    assert supports_streamed_linear_cross_entropy(hidden, head)
    head.weight.requires_grad_(True)
    assert not supports_streamed_linear_cross_entropy(hidden, head)
    head.requires_grad_(False)
    head.bias.requires_grad_(True)
    assert not supports_streamed_linear_cross_entropy(hidden, head)
    head.requires_grad_(False)
    hook = head.register_forward_hook(lambda module, args, output: output)
    assert not supports_streamed_linear_cross_entropy(hidden, head)
    hook.remove()
    assert not supports_streamed_linear_cross_entropy(hidden.half(), head.half())
    head.float()
    with torch.autocast("cpu", dtype=torch.float16):
        assert not supports_streamed_linear_cross_entropy(hidden, head)

    class CustomLinear(torch.nn.Linear):
        pass

    assert not supports_streamed_linear_cross_entropy(
        hidden, CustomLinear(7, 19).requires_grad_(False)
    )
    head.forward = lambda value: torch.nn.functional.linear(value, head.weight, head.bias)
    assert not supports_streamed_linear_cross_entropy(hidden, head)


def test_higher_order_derivatives_fail_explicitly():
    hidden, labels, head = _fixture()
    loss = streamed_linear_cross_entropy(
        hidden_states=hidden,
        labels=labels,
        lm_head=head,
        final_logit_softcapping=None,
        chunk_size=5,
    )
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(loss, hidden, create_graph=True)


@pytest.mark.parametrize("denominator", [0, 6, float("inf"), float("nan"), 1e100])
def test_invalid_denominator_is_rejected(denominator):
    hidden, labels, head = _fixture()
    with pytest.raises(ValueError, match="denominator"):
        streamed_linear_cross_entropy(
            hidden_states=hidden,
            labels=labels,
            lm_head=head,
            final_logit_softcapping=None,
            chunk_size=5,
            normalization_denominator=denominator,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_bf16_autocast_matches_dense_reference():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU does not support BF16")
    hidden, labels, head = _fixture(device="cuda")
    reference_hidden = hidden.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = streamed_linear_cross_entropy(
            hidden_states=hidden,
            labels=labels,
            lm_head=head,
            final_logit_softcapping=0.75,
            chunk_size=5,
        )
        expected = _reference(reference_hidden, labels, head, 0.75)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-2, atol=3e-4)
