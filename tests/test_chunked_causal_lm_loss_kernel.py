"""Loss-kernel equivalence checks that require PyTorch, without model downloads."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from svg_agentic_slm.train.chunked_causal_lm_loss import (  # noqa: E402
    chunked_causal_lm_loss,
)


def _labels() -> torch.Tensor:
    # Uneven prompts, right padding, an entirely ignored example, and a masked
    # interior token exercise the causal shift and gather across batch boundaries.
    return torch.tensor(
        [
            [4, -100, -100, 3, 5, -100, 7, 2, -100],
            [-100, 2, 8, 1, 6, 4, 3, 5, 9],
            [-100, -100, -100, -100, -100, -100, -100, -100, -100],
        ]
    )


def _full_loss(hidden, labels, head, softcap, denominator=None):
    logits = head(hidden[:, :-1]).float() if softcap is None else head(hidden[:, :-1])
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    targets = labels[:, 1:].reshape(-1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets, reduction="sum"
    )
    return loss / (targets.ne(-100).sum() if denominator is None else denominator)


@pytest.mark.parametrize("active_only", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("softcap", [None, 12.0])
@pytest.mark.parametrize("denominator", [None, 31, torch.tensor(31)])
def test_loss_and_all_gradients_match_dense_projection(
    active_only, checkpoint, softcap, denominator
) -> None:
    torch.manual_seed(29)
    hidden = torch.randn(3, 9, 7, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(7, 17, bias=True)
    expected_head = copy.deepcopy(head)
    labels = _labels()

    actual = chunked_causal_lm_loss(
        hidden_states=hidden,
        labels=labels,
        lm_head=head,
        final_logit_softcapping=softcap,
        chunk_size=5,
        use_checkpoint=checkpoint,
        normalization_denominator=denominator,
        project_only_active_tokens=active_only,
    )
    expected = _full_loss(expected_hidden, labels, expected_head, softcap, denominator)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    for actual_param, expected_param in zip(head.parameters(), expected_head.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)
    assert torch.count_nonzero(hidden.grad[:, -1]) == 0
    assert torch.count_nonzero(hidden.grad[:, :-1][labels[:, 1:].eq(-100)]) == 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_vocabulary_projection_receives_only_supervised_rows(checkpoint) -> None:
    hidden = torch.randn(3, 9, 7, requires_grad=True)
    labels = _labels()
    head = torch.nn.Linear(7, 17)
    projected = []
    hook = head.register_forward_pre_hook(
        lambda _module, args: projected.append(args[0].detach().clone())
    )
    loss = chunked_causal_lm_loss(
        hidden_states=hidden,
        labels=labels,
        lm_head=head,
        final_logit_softcapping=None,
        chunk_size=5,
        use_checkpoint=checkpoint,
    )
    active_mask = labels[:, 1:].ne(-100)
    assert len(projected) == 3
    assert all(rows.shape[0] <= 5 for rows in projected)
    torch.testing.assert_close(torch.cat(projected), hidden[:, :-1][active_mask])
    hook.remove()
    loss.backward()
    assert head.weight.grad is not None


def test_checkpoint_preserves_dropout_and_head_gradients_with_frozen_hidden() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(3, 9, 7)
    initial_head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(7, 17))
    results = []
    for checkpoint in (False, True):
        head = copy.deepcopy(initial_head)
        torch.manual_seed(13)
        loss = chunked_causal_lm_loss(
            hidden_states=hidden,
            labels=_labels(),
            lm_head=head,
            final_logit_softcapping=None,
            chunk_size=5,
            use_checkpoint=checkpoint,
        )
        loss.backward()
        results.append((loss.detach(), [p.grad.clone() for p in head.parameters()]))
    torch.testing.assert_close(results[0][0], results[1][0])
    for expected, actual in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(actual, expected)


def test_accumulated_optimizer_update_matches_dense_effective_batch() -> None:
    torch.manual_seed(41)
    inputs = torch.randn(3, 9, 5)
    labels = _labels()
    active_count = labels[:, 1:].ne(-100).sum()
    dense_model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Linear(7, 17))
    chunked_model = copy.deepcopy(dense_model)
    dense_optimizer = torch.optim.SGD(dense_model.parameters(), lr=0.1)
    chunked_optimizer = torch.optim.SGD(chunked_model.parameters(), lr=0.1)
    _full_loss(dense_model[0](inputs), labels, dense_model[1], 12.0).backward()
    dense_optimizer.step()

    # Different supervised-token counts per microbatch must share the full
    # accumulation-window denominator, as supplied by Trainer.
    for batch_slice in (slice(0, 1), slice(1, 3)):
        chunked_causal_lm_loss(
            hidden_states=chunked_model[0](inputs[batch_slice]),
            labels=labels[batch_slice],
            lm_head=chunked_model[1],
            final_logit_softcapping=12.0,
            chunk_size=5,
            use_checkpoint=True,
            normalization_denominator=active_count,
        ).backward()
    chunked_optimizer.step()
    for actual, expected in zip(chunked_model.parameters(), dense_model.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "denominator",
    [
        0,
        -1,
        11,
        float("nan"),
        float("inf"),
        1e100,
        "invalid",
        torch.tensor([12, 13]),
        torch.tensor(float("nan")),
        torch.tensor(11),
    ],
)
def test_rejects_invalid_normalization_denominator(denominator) -> None:
    with pytest.raises(ValueError, match="normalization denominator"):
        chunked_causal_lm_loss(
            hidden_states=torch.randn(3, 9, 7),
            labels=_labels(),
            lm_head=torch.nn.Linear(7, 17),
            final_logit_softcapping=None,
            chunk_size=5,
            use_checkpoint=False,
            normalization_denominator=denominator,
        )


@pytest.mark.parametrize("active_only", [False, True])
def test_rejects_entirely_ignored_shifted_labels(active_only) -> None:
    labels = torch.full((2, 3), -100)
    labels[:, 0] = 4  # Position zero is never a supervised causal target.
    with pytest.raises(ValueError, match="no active shifted labels"):
        chunked_causal_lm_loss(
            hidden_states=torch.randn(2, 3, 7),
            labels=labels,
            lm_head=torch.nn.Linear(7, 17),
            final_logit_softcapping=None,
            chunk_size=5,
            use_checkpoint=False,
            project_only_active_tokens=active_only,
        )
