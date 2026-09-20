"""Fused linear cross-entropy: the logits are never materialized at ``[B, L, V]``.

The plan's line is *"use a fused linear cross-entropy (Liger-style) so logits are never
materialized at 8K x vocab"*, and the arithmetic behind it is the whole reason this module
exists. At 8192 positions x ~80,000 vocabulary x 2 bytes, **one sequence's logit tensor is
1.3 GB** -- and the naive path holds at least two of them at once, because the softmax
needs a second buffer the same shape. The projection and the reduction are therefore done
together, one slice of positions at a time, and no tensor of shape ``[B, L, V]`` is ever
allocated.

## Why an autograd.Function rather than a Python loop

A chunked loop under ordinary autograd **saves no memory at all**. Every chunk's logits are
retained in the graph until backward runs, so peak allocation is the same
``sum(chunks) == B * L * V`` it was before, merely reached more slowly. The saving comes
from computing ``dL/dhidden`` and ``dL/dweight`` *inside* the forward pass, while each
chunk's logits are still live, and then dropping them. What survives the loop is
``grad_hidden`` (the shape of ``hidden``) and ``grad_weight`` (the shape of ``weight``) --
neither of which scales with sequence length times vocabulary.

Peak allocation is therefore ``O(chunk * V)`` rather than ``O(B * L * V)``, and
[`test_fused_ce.py`] asserts exactly that against an instrumented allocator, with the naive
path measured in the same test so a broken instrument fails the test rather than passing it.

## Numerics

The reduction is done in float32 regardless of the parameter dtype: ``logsumexp`` over
~80,000 bf16 logits loses far more than the memory saving is worth, and a fused kernel that
is quietly less accurate than the thing it replaces is worse than the thing it replaces.
``grad_weight`` accumulates across chunks in float32 for the same reason -- an accumulator
in bf16 loses the small contributions of most chunks -- and is cast back to the parameter
dtype once, at the end. That costs one float32 buffer the size of the weight, which is a
fixed cost, not a per-sequence one.

## Masking

``mask`` says which positions contribute. It is not an optimisation: ``qd_train.trainer``
computes it from ``Batch.lengths``, and a position that is padding must contribute neither
to the loss nor to either gradient. Masked-out positions gather index 0 rather than their
own target, so a shard whose padding slots hold arbitrary ids cannot index out of bounds --
and their contribution is then multiplied by zero, so the choice of 0 has no effect on the
result.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import torch

__all__ = [
    "TARGET_LOGIT_BYTES",
    "fused_linear_cross_entropy",
    "naive_linear_cross_entropy",
    "resolve_chunk_size",
]

#: The logit slab one chunk is allowed to occupy, in bytes, when ``chunk_size`` is not given.
#: 128 MiB is a compromise: small enough that the slab is a rounding error next to the model
#: and its optimizer state, large enough that the matmul is still worth launching. It is a
#: default, not a limit -- pass ``chunk_size`` to pin it.
TARGET_LOGIT_BYTES: Final[int] = 128 * 1024 * 1024

_REDUCTIONS: Final[frozenset[str]] = frozenset({"mean", "sum"})


def _compute_dtype(hidden: torch.Tensor, weight: torch.Tensor) -> torch.dtype:
    """The dtype the reduction runs in: at least float32, and never below the inputs'.

    ``float32`` is a **floor**, not a choice. Reducing ~80,000 bf16 logits in bf16 loses far
    more than the memory saving is worth. But hardcoding ``float32`` would silently *demote*
    a float64 model -- which is exactly what this function was added to fix, after the
    agreement test caught the fused path disagreeing with the reference at 1e-7 on inputs
    that were float64 on both sides.
    """
    return torch.promote_types(torch.float32, torch.promote_types(hidden.dtype, weight.dtype))


def resolve_chunk_size(n_positions: int, vocab_size: int, chunk_size: int | None) -> int:
    """Rows per chunk: either what the caller pinned, or what fits [`TARGET_LOGIT_BYTES`]."""
    if chunk_size is not None:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError(f"chunk_size must be int or None, got {type(chunk_size).__name__}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        return min(chunk_size, n_positions)
    # float32 is the compute dtype for the logit slab, hence 4 bytes.
    fits = TARGET_LOGIT_BYTES // max(1, vocab_size * 4)
    return max(1, min(n_positions, fits))


class _FusedLinearCE(torch.autograd.Function):
    """Projection and reduction per chunk, with the gradients formed before the logits die."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        targets: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
        scale: float,
        need_grad: bool,
    ) -> torch.Tensor:
        n_positions, hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        compute_dtype = _compute_dtype(hidden, weight)

        loss_sum = torch.zeros((), dtype=compute_dtype, device=hidden.device)
        grad_hidden = torch.zeros_like(hidden) if need_grad else None
        grad_weight = (
            torch.zeros((vocab_size, hidden_size), dtype=compute_dtype, device=hidden.device)
            if need_grad
            else None
        )
        grad_bias = (
            torch.zeros(vocab_size, dtype=compute_dtype, device=hidden.device)
            if need_grad and bias is not None
            else None
        )

        for start in range(0, n_positions, chunk_size):
            stop = min(start + chunk_size, n_positions)
            hidden_c = hidden[start:stop]
            target_c = targets[start:stop].unsqueeze(1)
            mask_c = mask[start:stop].unsqueeze(1)

            # [chunk, V] in float32 -- the only tensor in this function that scales with V
            # times anything, and `chunk` is what keeps it bounded.
            logits = torch.matmul(hidden_c, weight.t()).to(compute_dtype)
            if bias is not None:
                logits = logits + bias.to(compute_dtype)

            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            picked = logits.gather(1, target_c)
            loss_sum += torch.where(mask_c, lse - picked, torch.zeros_like(picked)).sum()

            if need_grad:
                # dL/dlogits for the *summed* loss: softmax(logits) - onehot(target).
                # Computed in place on the logit slab so no second [chunk, V] is allocated.
                probs = logits.sub_(lse).exp_()
                probs.scatter_add_(
                    1, target_c, torch.full_like(target_c, -1, dtype=probs.dtype)
                )
                probs.mul_(mask_c)
                probs.mul_(scale)
                assert grad_hidden is not None and grad_weight is not None
                grad_hidden[start:stop] = torch.matmul(probs.to(weight.dtype), weight)
                grad_weight += torch.matmul(probs.t(), hidden_c.to(compute_dtype))
                if grad_bias is not None:
                    grad_bias += probs.sum(0)
                del probs
            del logits

        ctx.save_for_backward(
            grad_hidden,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if (grad_bias is not None and bias is not None) else None,
        )
        ctx.need_grad = need_grad
        return loss_sum * scale

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        if not ctx.need_grad:
            raise RuntimeError(
                "backward on a fused_linear_cross_entropy computed without gradients. The "
                "forward decided no input required grad; that decision and this call disagree."
            )
        return (
            grad_hidden * grad_output,
            grad_weight * grad_output,
            grad_bias * grad_output if grad_bias is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


def _validate(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    bias: torch.Tensor | None,
    reduction: str,
) -> None:
    if hidden.dim() not in (2, 3):
        raise ValueError(f"hidden must be [N, H] or [B, L, H], got {tuple(hidden.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [V, H], got {tuple(weight.shape)}")
    if weight.shape[1] != hidden.shape[-1]:
        raise ValueError(
            f"weight is [V={weight.shape[0]}, H={weight.shape[1]}] but hidden's last dim is "
            f"{hidden.shape[-1]}"
        )
    if tuple(targets.shape) != tuple(hidden.shape[:-1]):
        raise ValueError(
            f"targets must be {tuple(hidden.shape[:-1])} to match hidden, got "
            f"{tuple(targets.shape)}"
        )
    if mask is not None:
        if tuple(mask.shape) != tuple(targets.shape):
            raise ValueError(
                f"mask must match targets {tuple(targets.shape)}, got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool, got {mask.dtype}")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError(f"bias must be [V={weight.shape[0]}], got {tuple(bias.shape)}")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}, got {reduction!r}")


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    chunk_size: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Cross-entropy of ``hidden @ weight.T`` against ``targets``, chunked over positions.

    Args:
        hidden: ``[N, H]`` or ``[B, L, H]`` -- the last hidden states.
        weight: ``[V, H]`` -- the language-model head. **Not** transposed.
        targets: ``[N]`` or ``[B, L]`` -- token ids. Integer dtype; ``int32`` is accepted
            because that is what ``Batch.tokens`` carries.
        mask: ``[N]`` or ``[B, L]`` bool -- which positions contribute. ``None`` means all
            of them, which for real training data means *you have not masked padding*.
        bias: ``[V]`` or ``None``.
        chunk_size: positions per chunk. ``None`` derives one from [`TARGET_LOGIT_BYTES`].
        reduction: ``"mean"`` over supervised positions, or ``"sum"``.

    Returns:
        A scalar tensor. Differentiable with respect to ``hidden``, ``weight`` and ``bias``.

    Raises:
        ValueError: on a shape mismatch, an out-of-range target, or an empty mask. An empty
            mask is a refusal rather than a zero: ``"mean"`` over nothing is ``0/0``, and
            reporting it as a zero loss would let a batch that trained on nothing look like
            a batch that fit perfectly.
    """
    _validate(hidden, weight, targets, mask, bias, reduction)

    hidden_size = hidden.shape[-1]
    vocab_size = weight.shape[0]
    flat_hidden = hidden.reshape(-1, hidden_size)
    flat_targets = targets.reshape(-1).to(torch.int64)
    n_positions = flat_hidden.shape[0]

    flat_mask = (
        torch.ones(n_positions, dtype=torch.bool, device=hidden.device)
        if mask is None
        else mask.reshape(-1)
    )
    n_supervised = int(flat_mask.sum().item())
    if n_supervised == 0:
        raise ValueError(
            "no position is supervised: the mask is empty. A loss over zero positions is 0/0, "
            "and a zero returned here would be indistinguishable from a perfect fit."
        )

    live = flat_targets[flat_mask]
    if live.numel():
        lo, hi = int(live.min().item()), int(live.max().item())
        if lo < 0 or hi >= vocab_size:
            raise ValueError(
                f"supervised targets span [{lo}, {hi}] but the head has {vocab_size} rows. A "
                "token id outside the head is the remap failure qd_train.artifacts refuses at "
                "the shard boundary; it must not be clamped into range here."
            )
    # Masked-out positions gather row 0 and are then multiplied by zero, so an arbitrary id
    # in a padding slot can neither index out of bounds nor affect the result.
    safe_targets = torch.where(flat_mask, flat_targets, torch.zeros_like(flat_targets))

    resolved_chunk = resolve_chunk_size(n_positions, vocab_size, chunk_size)
    scale = 1.0 / n_supervised if reduction == "mean" else 1.0
    need_grad = torch.is_grad_enabled() and (
        hidden.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )

    return _FusedLinearCE.apply(
        flat_hidden, weight, bias, safe_targets, flat_mask, resolved_chunk, scale, need_grad
    )


def naive_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """The reference: materialize ``[B, L, V]`` and call the stock cross-entropy.

    This is what [`fused_linear_cross_entropy`] must agree with numerically and must beat on
    memory. It is shipped rather than left in the test file because both claims are only
    meaningful against a reference that is *this* reference -- a test that compares the fused
    path against a second hand-written reduction is comparing two of our own opinions.
    """
    _validate(hidden, weight, targets, mask, bias, reduction)

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1).to(torch.int64)
    n_positions = flat_hidden.shape[0]
    flat_mask = (
        torch.ones(n_positions, dtype=torch.bool, device=hidden.device)
        if mask is None
        else mask.reshape(-1)
    )
    n_supervised = int(flat_mask.sum().item())
    if n_supervised == 0:
        raise ValueError("no position is supervised: the mask is empty")

    logits = torch.nn.functional.linear(flat_hidden, weight, bias).to(
        _compute_dtype(hidden, weight)
    )
    per_position = torch.nn.functional.cross_entropy(
        logits,
        torch.where(flat_mask, flat_targets, torch.zeros_like(flat_targets)),
        reduction="none",
    )
    total = (per_position * flat_mask).sum()
    return total / n_supervised if reduction == "mean" else total
