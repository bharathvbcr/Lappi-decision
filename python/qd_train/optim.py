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

import functools
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_MOMENT_RELATIVE_ERROR",
    "MasterWeightAdamW",
    "MomentSettling",
    "build_optimizer",
    "moment_settling",
]

#: Refused above this, rather than discovered as an allocation failure part way through a
#: run. An fp32 master plus fp32 moments is 12 B/param on top of the live parameters; at
#: this bound that is 96 GB of optimizer-side memory, which no single device here has.
MAX_MASTER_PARAMS: int = 8_000_000_000

#: How far the settled second moment may sit from the value it is chasing before the run is
#: refused. A second moment wrong by this fraction mis-scales every AdamW step by
#: ``1/sqrt(1-e)``; at 1% that is 0.5%, which is inside the noise of a learning-rate choice.
#: Measured at ``beta2=0.999``: bf16 settles **50.00%** low, fp16 **26.76%** low and fp32
#: **0.003%** low, so the three land far on either side of this and the exact bar is not
#: load-bearing. fp16 is listed because it is the one that looks survivable and is not.
MAX_MOMENT_RELATIVE_ERROR: float = 0.01

#: Simulation bound for :func:`moment_settling`. fp32 at ``beta2=0.999`` stops moving at
#: step 10,301 -- at the RIGHT value -- so a bound has to clear that comfortably; fp64 never
#: settles within any practical one and reports ``None``, which is the honest answer.
MAX_SETTLING_STEPS: int = 200_000


@dataclass(frozen=True)
class MomentSettling:
    """Where AdamW's second moment comes to rest in a given dtype, and when.

    **Freezing is not the defect; freezing at the wrong value is.** Both fp32 and bf16 stop
    moving eventually, because the EMA's increment shrinks as it approaches its target until
    it falls under half the dtype's spacing. fp32 stops at 0.999970 of the target and bf16
    stops at 0.500000 of it, and only one of those is a broken optimizer. A check that keyed
    "does it stop" would refuse both; this one carries the value it stopped at.
    """

    dtype_name: str
    beta2: float
    #: The first step at which the update rounded to no change, or ``None`` if it was still
    #: moving at ``MAX_SETTLING_STEPS``.
    settled_at_step: int | None
    #: What it settled on, against a constant ``g**2 == 1.0``. The target is therefore 1.0.
    settled_value: float

    @property
    def relative_error(self) -> float:
        """How far below 1.0 the settled value sits, as a fraction."""
        return abs(1.0 - self.settled_value)

    @property
    def is_faithful(self) -> bool:
        """Whether a run longer than ``settled_at_step`` still has a usable second moment."""
        return self.relative_error <= MAX_MOMENT_RELATIVE_ERROR

    def survives(self, total_steps: int) -> bool:
        """Whether a schedule of ``total_steps`` finishes before the moment goes wrong.

        Faithful settling survives any length -- fp32 comes to rest on the right answer and
        staying there is correct, not a failure. An unfaithful one survives only a schedule
        that ends before it sets in.
        """
        if self.is_faithful:
            return True
        return self.settled_at_step is None or total_steps < self.settled_at_step

    def describe(self) -> str:
        when = (
            f"step {self.settled_at_step}"
            if self.settled_at_step is not None
            else f"no step below {MAX_SETTLING_STEPS}"
        )
        return (
            f"{self.dtype_name} at beta2={self.beta2}: exp_avg_sq stops moving at {when}, "
            f"settling at {self.settled_value:.6f} against a target of 1.0 "
            f"({self.relative_error:.2%} low)"
        )


@functools.lru_cache(maxsize=32)
def _settling(dtype_name: str, beta2: float) -> MomentSettling:
    import torch

    dtype = getattr(torch, dtype_name)
    # Simulated in the REAL dtype rather than derived. The closed form gets the condition
    # right -- the EMA freezes once ``(1-beta2)*(1-v)`` falls under half the spacing at
    # ``v`` -- and the answer wrong, because the spacing doubles at a binade boundary and
    # bf16 lands exactly on one: it settles at 0.5000, not at the 0.3386 the smooth
    # algebra predicts. One scalar EMA is microseconds, and torch's own rounding is the
    # thing under test.
    value = torch.zeros((), dtype=dtype)
    unit = torch.ones((), dtype=dtype)
    for step in range(1, MAX_SETTLING_STEPS + 1):
        nxt = (beta2 * value + (1.0 - beta2) * unit).to(dtype)
        if bool(nxt == value):
            return MomentSettling(dtype_name, beta2, step, float(value))
        value = nxt
    return MomentSettling(dtype_name, beta2, None, float(value))


def moment_settling(*, dtype: Any, beta2: float = 0.999) -> MomentSettling:
    """Measure where a second moment in ``dtype`` comes to rest, and after how many steps.

    ``tools/moment_precision.py`` measures the same thing against ``torch.optim.AdamW``
    itself and is the authority; this reproduces its scalar EMA so the answer is available
    to a guard without allocating an optimizer. The two agree on the value -- 0.500000 for
    bf16 -- and count the step differently by one: that tool reports the last update that
    MOVED the value (383), this returns the first that did NOT (384). Same event. The guard
    compares with ``<``, so a 383-step run is still admitted.

    Cached per ``(dtype, beta2)`` -- it is a property of the number format, not of a run.
    """
    if not 0.0 < beta2 < 1.0:
        raise ValueError(f"beta2 must be in (0, 1), got {beta2}")
    name = getattr(dtype, "name", None) or str(dtype).rsplit(".", 1)[-1]
    return _settling(name, float(beta2))


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


def build_optimizer(
    params: Any,
    *,
    spec: Any,
    lr: float,
    total_steps: int,
    allow_frozen_moments: bool = False,
) -> Any:
    """The one place that turns an [`qd_train.memory.OptimizerSpec`] into an optimizer.

    ``spec`` is the budget's description of the recipe; this returns the thing the budget
    describes. Keeping the two together is the point -- a footprint that does not describe
    the run cannot decide whether the next run fits, and that divergence is exactly how
    ``ADAMW_FP32`` came to be budgeted for a bf16 tower.

    **``total_steps`` is required, and that is the hardening.** Until it was, this module
    could measure that a bf16 second moment stops moving at step 384 and settles 50% low,
    say so at length in its own docstring, and still hand back the optimizer that does it --
    because nothing here knew how long the run would be, and the caller that did know was
    never asked. A schedule is not an optional detail of an optimizer whose correctness has
    a step count in it. Every caller already computes this number before it builds a step.

    ``allow_frozen_moments`` is the deliberate exception, named after
    ``shards.write_shards(allow_contradictions=...)`` and for the same reason: there is a
    real case for it -- a device on which 16 B/param does not fit -- and it must be said out
    loud, land in the recipe, and never be reachable by forgetting to pass something.

    Raises:
        ValueError: if the schedule outlives the second moment's fidelity, if the spec does
            not describe these parameters, or if ``total_steps`` is not positive.
    """
    import torch

    if total_steps < 1:
        raise ValueError(
            f"total_steps must be at least 1, got {total_steps}; an optimizer for a "
            "schedule of no steps is a budget for a run that does not happen"
        )
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
        # The moments live in the dominant dtype, so that is where the fidelity question is
        # asked. Checked BEFORE the byte-width check below, because the two can both be
        # unhappy and this is the one that silently produces a wrong model: the byte-width
        # mismatch is a budget that over-reports and never crashes, while a frozen second
        # moment mis-scales every step past 384 and shows up nowhere.
        settling = moment_settling(dtype=dominant)
        if not settling.survives(total_steps) and not allow_frozen_moments:
            raise ValueError(
                f"this run is {total_steps} optimizer step(s) long and its moments would "
                f"be {dominant}, which does not survive it. {settling.describe()}. "
                "torch.optim.AdamW keeps exp_avg_sq in the parameter's own dtype, so past "
                "that step the second moment is a fixed point it cannot leave -- the "
                "increment that would move it is itself unrepresentable -- and every "
                f"subsequent step is mis-scaled by 1/sqrt(1-{settling.relative_error:.4f}). "
                "It is invisible in any single step and in the loss curve. Pass a spec with "
                "keeps_fp32_master=True (qd_train.optim.MasterWeightAdamW implements it, at "
                "16 B/param against 8), shorten the schedule below "
                f"{settling.settled_at_step}, or pass allow_frozen_moments=True to say out "
                "loud that this run accepts a degraded optimizer."
            )
        if dominant.itemsize != spec.state_bytes:
            share = by_dtype[dominant] / sum(by_dtype.values())
            raise ValueError(
                f"optimizer spec {spec.name!r} says {spec.state_bytes}-byte states, but "
                f"{share:.1%} of these parameters are {dominant} ({dominant.itemsize} "
                "bytes) and torch.optim.AdamW keeps its states in the parameter's dtype. "
                "The budget would describe a layout nothing builds."
            )
    return torch.optim.AdamW(trainable, lr=lr)
