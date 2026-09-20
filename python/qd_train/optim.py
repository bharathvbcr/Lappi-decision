"""AdamW that keeps its master weights and its moments in fp32.

**Why this exists, measured rather than assumed.** ``torch.optim.AdamW`` keeps ``exp_avg``
and ``exp_avg_sq`` in the *parameter's* dtype. Over a bf16 tower that makes the second
moment a bf16 exponential moving average, and at the torch default ``beta2 = 0.999`` the
per-step relative increment is ``1e-3`` while bfloat16's relative spacing is ``2**-8 =
3.9e-3``. The increment is below half the spacing, so round-to-nearest returns the previous
value and **the second moment stops moving**.

``tools/moment_precision.py`` measures this against ``torch.optim.AdamW`` itself, not
against a model of it::

    bfloat16  exp_avg_sq settled    0.5000 (target 1.0) after 383 steps,
              after the gradient magnitude rises 100x:   32.0000 (target 100.0)
    float32   exp_avg_sq settled    1.0000 (target 1.0),
              after the same change:                     99.9959 (target 100.0)

A second moment wrong by 2x mis-scales every AdamW step by ``1/sqrt(2) = 1.41x``, and the
error is not transient: the EMA is at a fixed point it cannot leave, because the increment
that would move it is itself unrepresentable. It is invisible in any single step, and it
sets in after **383** of them -- fewer than the 400 optimizer steps each of this
repository's first real-tower runs took.

**What this class does.** It holds an fp32 master copy of every trainable parameter, runs
``torch.optim.AdamW`` over the masters -- so the update math is torch's, not a second
implementation of it -- and copies the masters back into the live low-precision parameters
after each step. Gradients arrive in the parameter's dtype and are cast up on the way in.

**What it costs.** 4 B/param for the master plus 8 B/param for fp32 moments, against 4
B/param of bf16 moments and no master: ``OptimizerSpec("AdamW+master", 2, 4,
keeps_fp32_master=True)`` is 16 B/param all-in against ``ADAMW_BF16``'s 8. That is the
trade, and ``qd_train.memory`` has described it from the start -- this module is what makes
the description true of something.

**What it is not.** It is not `bf16 optimizer state with stochastic rounding`, and it is not
Kahan summation over bf16 moments. Both are cheaper and both are real techniques; neither is
implemented here, and neither should be assumed from the presence of this one.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MasterWeightAdamW", "build_optimizer"]

#: Refused above this, rather than discovered as an allocation failure part way through a
#: run. An fp32 master plus fp32 moments is 12 B/param on top of the live parameters; at
#: this bound that is 96 GB of optimizer-side memory, which no single device here has.
MAX_MASTER_PARAMS: int = 8_000_000_000


class MasterWeightAdamW:
    """``torch.optim.AdamW`` over fp32 master copies of low-precision parameters.

    Exposes ``param_groups``, ``step``, ``zero_grad``, ``state_dict`` and
    ``load_state_dict``, which is the whole surface ``TrainStep.apply`` and the checkpoint
    path use -- ``param_groups`` is delegated to the inner optimizer, so the existing
    ``for group in optimizer.param_groups: group["lr"] = lr`` drives this one unchanged.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        import torch

        self._torch = torch
        live = [p for p in params if p.requires_grad]
        if not live:
            raise ValueError(
                "MasterWeightAdamW was given no parameters with requires_grad=True. An "
                "optimizer over nothing takes silent no-op steps and a run would report "
                "optimizer_steps>0 having trained nothing."
            )
        total = sum(p.numel() for p in live)
        if total > MAX_MASTER_PARAMS:
            raise ValueError(
                f"{total:,} trainable parameters exceeds MAX_MASTER_PARAMS "
                f"({MAX_MASTER_PARAMS:,}). The fp32 master and fp32 moments would need "
                f"{total * 12 / 1024 ** 3:.1f} GiB on top of the live parameters. Refusing "
                "up front rather than failing part way through a run."
            )
        if all(p.dtype == torch.float32 for p in live):
            raise ValueError(
                "MasterWeightAdamW was given only fp32 parameters. torch.optim.AdamW "
                "already keeps fp32 moments for fp32 parameters, so a master copy would "
                "double the memory and change nothing. Use torch.optim.AdamW directly, or "
                "pass the bf16/fp16 parameters this class exists for."
            )

        # A mixed-dtype model is the normal case here, not an edge one: QwenDecisionStep
        # optimises a bf16 tower together with an fp32 SpanPointerHead. A parameter that is
        # ALREADY fp32 is its own master -- copying it would double its memory to store the
        # same bits -- so only the low-precision ones get a copy. `_copy_back` is the pairs
        # that actually need one, and it is what `step` walks rather than re-testing dtypes.
        self._live = live
        self._masters = [
            p if p.dtype == torch.float32 else p.detach().clone().to(torch.float32)
            for p in live
        ]
        for m in self._masters:
            m.requires_grad_(True)
        self._copy_back = [
            (live_p, master)
            for live_p, master in zip(live, self._masters, strict=True)
            if master is not live_p
        ]
        self._inner = torch.optim.AdamW(
            self._masters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    # -- the torch.optim.Optimizer surface the callers actually use ------------------------

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """The inner optimizer's groups: the inner optimizer is the one that steps, so an
        ``lr`` written here is the ``lr`` that is used."""
        return self._inner.param_groups

    @property
    def state(self) -> dict[Any, Any]:
        """The inner optimizer's per-parameter state, keyed by the *master* tensors."""
        return self._inner.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._live:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
        self._inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Cast grads up, step on the masters, cast the masters back down.

        Only the parameters that HAVE a separate master are touched on either side: where
        the master is the live parameter itself, its gradient is already the one backward
        wrote and copying it to itself would be a no-op.
        """
        torch = self._torch
        for live, master in self._copy_back:
            master.grad = None if live.grad is None else live.grad.detach().to(torch.float32)
        self._inner.step()
        with torch.no_grad():
            for live, master in self._copy_back:
                live.copy_(master)

    # -- checkpointing --------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """The inner state **and** the masters.

        Both, because they are two halves of one thing: a resume that restores the moments
        but rebuilds the masters from the bf16 parameters throws away every bit of precision
        this class exists to keep, and would do it silently -- the run would continue, the
        loss would look ordinary, and the master would be exactly the rounded copy the
        design is meant to avoid.
        """
        return {"inner": self._inner.state_dict(), "masters": list(self._masters)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        missing = {"inner", "masters"} - set(state)
        if missing:
            raise ValueError(
                f"optimizer state is missing {sorted(missing)}. Refusing to load a partial "
                "state: restoring moments without masters silently discards the fp32 "
                "precision this optimizer exists to keep."
            )
        masters = state["masters"]
        if len(masters) != len(self._masters):
            raise ValueError(
                f"optimizer state carries {len(masters)} master tensor(s) but this optimizer "
                f"has {len(self._masters)}. The checkpoint describes a different model."
            )
        with self._torch.no_grad():
            for mine, saved in zip(self._masters, masters, strict=True):
                if tuple(mine.shape) != tuple(saved.shape):
                    raise ValueError(
                        f"master shape {tuple(saved.shape)} in the checkpoint does not match "
                        f"{tuple(mine.shape)} in this model. The checkpoint describes a "
                        "different model."
                    )
                mine.copy_(saved.to(mine.dtype))
        self._inner.load_state_dict(state["inner"])
        # The live parameters follow the restored masters, so a resumed run continues from
        # the state that was saved rather than from whatever the bf16 copy had rounded to.
        with self._torch.no_grad():
            for live, master in self._copy_back:
                live.copy_(master)


def build_optimizer(params: Any, *, spec: Any, lr: float) -> Any:
    """The one place that turns an [`qd_train.memory.OptimizerSpec`] into an optimizer.

    ``spec`` is the budget's description of the recipe; this returns the thing the budget
    describes. Keeping the two together is the point -- a footprint that does not describe
    the run cannot decide whether the next run fits, and that divergence is exactly how
    ``ADAMW_FP32`` came to be budgeted for a bf16 tower.
    """
    import torch

    live = list(params)
    if spec.keeps_fp32_master:
        return MasterWeightAdamW(live, lr=lr)
    if spec.states_per_param != 2:
        raise ValueError(
            f"optimizer spec {spec.name!r} describes {spec.states_per_param} state tensor(s) "
            "per parameter, but this builder constructs torch.optim.AdamW, which keeps "
            "exactly two (exp_avg and exp_avg_sq). Refusing to return an optimizer the "
            "budget does not describe."
        )
    trainable = [p for p in live if p.requires_grad]
    if trainable:
        # Checked against the dtype holding the MOST parameters, not against "all the same
        # dtype". A real step here is mixed -- a bf16 tower plus an fp32 SpanPointerHead of
        # 16.8 MB against 2.8 GB -- so requiring uniformity would skip the check on exactly
        # the model that matters, and that silent skip is how ADAMW_FP32 came to be
        # budgeted for a bf16 tower in the first place.
        by_dtype: dict[Any, int] = {}
        for p in trainable:
            by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel()
        dominant = max(by_dtype, key=lambda d: by_dtype[d])
        if dominant.itemsize != spec.state_bytes:
            share = by_dtype[dominant] / sum(by_dtype.values())
            raise ValueError(
                f"optimizer spec {spec.name!r} says {spec.state_bytes}-byte states, but "
                f"{share:.1%} of these parameters are {dominant} ({dominant.itemsize} "
                "bytes) and torch.optim.AdamW keeps its states in the parameter's dtype. "
                "The budget would describe a layout nothing builds."
            )
    return torch.optim.AdamW(trainable, lr=lr)
