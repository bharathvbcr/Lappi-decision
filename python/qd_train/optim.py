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

**The cheaper alternative, [`KahanBf16AdamW`].** It keeps the weights in bf16 with a bf16
Kahan compensation buffer beside them (the rounding residual of each weight) and the moments
in fp32, and no master. That is 14 B/param against the master recipe's 20. Updates smaller
than a weight's bf16 spacing land in the compensation and accumulate, so they are not lost;
plain bf16 AdamW loses them (``tests/test_optim.py`` measures both trajectories against the
master's). Stochastic rounding is not implemented: it is unbiased but not deterministic, and
this repository pins resumed trajectories bit for bit.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_BETA2",
    "KAHAN_CHUNK_ELEMS",
    "KAHAN_RECIPE",
    "MAX_KAHAN_PARAMS",
    "MAX_MOMENT_RELATIVE_ERROR",
    "KahanBf16AdamW",
    "MasterWeightAdamW",
    "MomentSettling",
    "apply_lr",
    "build_optimizer",
    "layerwise_param_groups",
    "moment_settling",
]

#: torch's own default, and what every optimizer this repository built used until beta2
#: became a parameter. A default that moved would make every new row incomparable to the
#: old ones without anything saying so.
DEFAULT_BETA2: float = 0.999

#: The key a parameter group carries its learning-rate multiplier under. `apply_lr` is the
#: only writer of ``group["lr"]``; this is what it multiplies by.
LR_SCALE_KEY: str = "lr_scale"

#: Refused above this, rather than discovered as an allocation failure part way through a
#: run. An fp32 master plus fp32 moments is 12 B/param on top of the live parameters; at
#: this bound that is 96 GB of optimizer-side memory, which no single device here has.
MAX_MASTER_PARAMS: int = 8_000_000_000

#: Refused above this, for the same reason. [`KahanBf16AdamW`] keeps 10 B/param beside the
#: live weights (a 2-byte compensation and two fp32 moments), 14 B/param all-in with the bf16
#: weight and grad; at this bound that is 224 GB, more than any single device rented here.
#: It allocates all of it at construction, so a model that cannot fit fails before the first
#: batch is read, not at the first step.
MAX_KAHAN_PARAMS: int = 16_000_000_000

#: Elements per slice of [`KahanBf16AdamW.step`]. Each slice casts its grad, weight and
#: compensation to fp32 and builds a denominator: about four fp32 temporaries, 256 MiB at this
#: size. Without the bound, the 248,320 x 2,048 embedding alone would make 2 GiB temporaries,
#: four at once: memory the 14 B/param budget does not count.
KAHAN_CHUNK_ELEMS: int = 1 << 24

#: What [`KahanBf16AdamW.state_dict`] records itself as. [`KahanBf16AdamW.load_state_dict`]
#: refuses any other optimizer's state, so a resume cannot silently change the recipe.
KAHAN_RECIPE: str = "kahan-bf16"

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


def _normalise_groups(params: Any) -> list[dict[str, Any]]:
    """``params`` as torch accepts it -- tensors, or dicts with a ``"params"`` list -- as a
    list of groups holding only the trainable tensors, in their given order.

    One group per input dict, and a bare iterable of tensors is one group with no extra
    keys, so a caller that never asked for groups gets exactly the optimizer it got before.
    A group that ends up with no trainable tensor is dropped, except that its absence is
    checked by [`layerwise_param_groups`] where it would mean a silently ignored scale.
    """
    items = list(params)
    if items and all(isinstance(item, dict) for item in items):
        groups: list[dict[str, Any]] = []
        for item in items:
            if "params" not in item:
                raise ValueError(
                    f"a parameter group needs a 'params' key; got keys {sorted(item)}"
                )
            trainable = [p for p in item["params"] if p.requires_grad]
            if trainable:
                groups.append({**item, "params": trainable})
        return groups
    if any(isinstance(item, dict) for item in items):
        raise ValueError(
            "params mixes parameter-group dicts with bare tensors; torch.optim refuses "
            "that too, and guessing which group a bare tensor belongs to would assign it "
            "a learning rate nobody chose"
        )
    trainable = [p for p in items if p.requires_grad]
    return [{"params": trainable}] if trainable else []


def _check_lr_scale(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: lr_scale must be a number, got {type(value).__name__}")
    scale = float(value)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(
            f"{where}: lr_scale must be finite and positive, got {scale!r}. Zero would "
            "freeze the group while it still paid for moments; a group that should not "
            "train belongs outside the optimizer."
        )
    return scale


def apply_lr(optimizer: Any, lr: float) -> None:
    """Set every group's ``lr`` to ``lr * group["lr_scale"]``. The one writer of ``lr``.

    **Why this exists.** Four drivers (``backbone.QwenDecisionStep.apply``,
    ``byte_train.Rung0Step.apply``, ``tools/real_ft_run.RealFtStep.apply``,
    ``tools/ft_toy_run.ToyFtStep.apply``) each wrote ``group["lr"] = lr`` for every group
    before every step. That is correct for one group and quietly wrong for any other: a
    group built at 0.1x would train at 1x from the first step, and nothing would say so.
    The schedule is still ``RunControl.lr_at``'s; this only distributes it.

    A group without ``lr_scale`` runs at 1.0x -- that is what every optimizer built before
    this function existed means. A present value is validated every time, because a NaN
    written into a group after construction would otherwise reach ``step``.
    """
    if not (isinstance(lr, (int, float)) and math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"lr must be positive and finite, got {lr!r}")
    for i, group in enumerate(optimizer.param_groups):
        scale = _check_lr_scale(group.get(LR_SCALE_KEY, 1.0), where=f"param group {i}")
        group["lr"] = float(lr) * scale


#: Decoder-layer index in a parameter name. RSI-Jev ``rsijev/fit.py:127`` uses the same
#: pattern; it matches ``layers.3.mlp...`` and ``model.layers.3.mlp...`` alike.
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layerwise_param_groups(
    named: Iterable[tuple[str, Any]],
    *,
    lower_layers_n: int,
    lower_lr_scale: float,
    extra: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Split parameters into a ``base`` group (1.0x) and a ``base_lower`` group.

    Ported from RSI-Jev ``rsijev/fit.py:116-153`` (``_param_groups``), MIT licence,
    Copyright (c) 2026 Shanghua Gao, at commit 8f34a4f. The rule is theirs: decoder layers
    ``0 .. lower_layers_n - 1`` train at ``lower_lr_scale`` times the schedule, on the same
    schedule as the rest, because fitting a decision objective through every layer at one
    rate overwrites the representation general knowledge sits in -- and freezing those
    layers instead costs decision accuracy. Their scorer/mix groups have no counterpart
    here; ``extra`` (the span head) joins the base group.

    What differs from the source, and why: the rate is carried as ``lr_scale`` rather than
    baked into ``lr``, because every driver here rewrites ``lr`` each step and a baked value
    would be overwritten at the first one -- [`apply_lr`] is what honours it. And the split
    fails closed: RSI prints the lower group's size and carries on when it is empty; here an
    empty lower group, or ``lower_layers_n`` past the deepest layer, is a refusal, because
    both mean the scale a recipe records was applied to nothing.
    """
    if isinstance(lower_layers_n, bool) or not isinstance(lower_layers_n, int):
        raise ValueError(f"lower_layers_n must be an int, got {lower_layers_n!r}")
    if lower_layers_n < 1:
        raise ValueError(
            f"lower_layers_n must be at least 1, got {lower_layers_n}; zero means no "
            "layer-wise split, which is the plain optimizer -- do not build groups for it"
        )
    scale = _check_lr_scale(lower_lr_scale, where="lower_lr_scale")
    base: list[Any] = []
    lower: list[Any] = []
    deepest = -1
    for name, param in named:
        if not param.requires_grad:
            continue
        match = _LAYER_INDEX.search(name)
        # `visual` excluded as in the source: a vision tower's layers are not the text
        # tower's, and this repository only ever loads the text tower anyway.
        if match and "visual" not in name:
            index = int(match.group(1))
            deepest = max(deepest, index)
            if index < lower_layers_n:
                lower.append(param)
                continue
        base.append(param)
    base.extend(p for p in extra if p.requires_grad)
    if not lower:
        raise ValueError(
            f"lower_layers_n={lower_layers_n} matched no trainable parameter: no name "
            "carries a 'layers.<i>.' index. The recipe would record a 0.1x split that "
            "scaled nothing."
        )
    if lower_layers_n > deepest + 1:
        raise ValueError(
            f"lower_layers_n={lower_layers_n} but the deepest layer is {deepest}: the "
            "lower group would be the whole tower, which is a global learning rate "
            "recorded as a layer-wise one"
        )
    if not base:
        raise ValueError("every trainable parameter fell in the lower group")
    return [
        {"params": base, LR_SCALE_KEY: 1.0, "name": "base"},
        {"params": lower, LR_SCALE_KEY: scale, "name": "base_lower"},
    ]


class MasterWeightAdamW:
    """``torch.optim.AdamW`` over fp32 master copies of low-precision parameters.

    Exposes ``param_groups``, ``step``, ``zero_grad``, ``state_dict`` and
    ``load_state_dict``, which is the whole surface ``TrainStep.apply`` and the checkpoint
    path use -- ``param_groups`` is delegated to the inner optimizer, so [`apply_lr`] drives
    this one unchanged, ``lr_scale`` included.

    ``params`` may be tensors or parameter-group dicts; each group's extra keys
    (``lr_scale``, ``name``) are carried onto the inner group over its masters.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, DEFAULT_BETA2),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        fused: bool = False,
    ) -> None:
        import torch

        self._torch = torch
        groups = _normalise_groups(params)
        live = [p for g in groups for p in g["params"]]
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
        # The same groups over the masters, in the same order `_masters` holds them, so
        # `state_dict`'s parameter indices and `load_state_dict`'s length check still line
        # up with the flat list.
        master_of = {id(p): m for p, m in zip(live, self._masters, strict=True)}
        inner_groups = []
        for g in groups:
            extra = {k: v for k, v in g.items() if k != "params"}
            if LR_SCALE_KEY in extra:
                extra["lr"] = float(lr) * _check_lr_scale(
                    extra[LR_SCALE_KEY], where=f"group {extra.get('name', '?')!r}"
                )
            inner_groups.append({**extra, "params": [master_of[id(p)] for p in g["params"]]})
        # `fused=True` is torch's single-kernel AdamW over the masters: the same update rule,
        # one launch instead of a dozen foreach passes, and no full-size fp32 temporaries
        # (measured on the GH200, 2026-10-01: optimizer step 70 -> 40 ms at 4 x 8,441, peak
        # 42.2 -> 35.2 GiB). Its rounding is not foreach's, so it is a numerics change and
        # rides in the recipe; `None` is torch's own default, exactly what was built before.
        self._inner = torch.optim.AdamW(
            inner_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            fused=True if fused else None,
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


class KahanBf16AdamW:
    """AdamW over bf16 weights with a bf16 Kahan compensation buffer and fp32 moments.

    **The defect it fixes without a master.** A bf16 weight cannot record an update smaller
    than half its spacing: at the real tower's lr of 1e-5 and a weight of magnitude 0.02, an
    Adam step of ~1e-5 is below bf16's half-spacing of ~6e-5, so plain ``torch.optim.AdamW``
    over bf16 rounds most steps away. On top of that, its bf16 second moment freezes 50% low
    (:func:`moment_settling`). [`MasterWeightAdamW`] fixes both with an fp32 master and fp32
    moments, at 20 B/param. This class fixes both at 14:
    - the moments are fp32, 8 B/param, the same as the master recipe's;
    - each bf16 weight ``p`` keeps a bf16 compensation ``c``, its rounding residual. The
      weight the optimizer steps is ``p + c``, which is accurate to about 16 bits of
      mantissa. After each update the new value is split back into ``p = bf16(w)`` and
      ``c = bf16(w - p)``.

    **The update rule is torch's AdamW** (decoupled weight decay, bias-corrected moments,
    ``eps`` added after the bias-corrected root), written out because the master-free layout
    has no fp32 tensor for ``torch.optim.AdamW`` to own. ``tests/test_optim.py`` pins it to
    ``torch.optim.AdamW`` on fp32 parameters and to [`MasterWeightAdamW`]'s trajectory on bf16
    ones.

    **fp32 parameters** (the span head) get the same rule with no compensation: fp32 is
    already its own master. **fp16 is refused**: a compensated fp16 weight has fp16's range,
    which this repository has never measured for a tower.

    **Memory.** Every state tensor is allocated at construction. The step walks each tensor in
    slices of ``chunk_elems``, so its fp32 temporaries are bounded (:data:`KAHAN_CHUNK_ELEMS`).

    **Checkpoints.** :meth:`state_dict` carries the exact compensation, so a resume continues
    the trajectory bit for bit. It also carries ``masters``: ``p + c`` in fp32 on the host, the
    same list ``MasterWeightAdamW`` writes, so ``tools/ckpt_average.py --from masters``
    averages a compensated run's full-precision weights unchanged. Where ``p + c`` is an exact
    bf16 tie, the master is ``p`` itself, so ``bf16(master) == p`` holds on every element; that
    identity is how ``ckpt_average`` matches a master to its weight.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, DEFAULT_BETA2),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        chunk_elems: int = KAHAN_CHUNK_ELEMS,
    ) -> None:
        import torch

        self._torch = torch
        _check_hyper(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if isinstance(chunk_elems, bool) or not isinstance(chunk_elems, int) or chunk_elems < 1:
            raise ValueError(f"chunk_elems must be a positive int, got {chunk_elems!r}")
        self.chunk_elems = chunk_elems
        groups = _normalise_groups(params)
        live = [p for g in groups for p in g["params"]]
        if not live:
            raise ValueError(
                "KahanBf16AdamW was given no parameters with requires_grad=True. An optimizer "
                "over nothing takes silent no-op steps and a run would report "
                "optimizer_steps>0 having trained nothing."
            )
        if len({id(p) for p in live}) != len(live):
            raise ValueError(
                "a parameter appears in more than one group: it would be stepped twice per "
                "step, at two learning rates"
            )
        total = sum(p.numel() for p in live)
        if total > MAX_KAHAN_PARAMS:
            raise ValueError(
                f"{total:,} trainable parameters exceeds MAX_KAHAN_PARAMS "
                f"({MAX_KAHAN_PARAMS:,}). The compensation and fp32 moments would need "
                f"{total * 10 / 1024 ** 3:.1f} GiB beside the live weights. Refusing up front "
                "rather than failing part way through a run."
            )
        bad = sorted({str(p.dtype) for p in live} - {"torch.bfloat16", "torch.float32"})
        if bad:
            raise ValueError(
                f"KahanBf16AdamW takes bf16 and fp32 parameters only, got {bad}. A compensated "
                "fp16 weight keeps fp16's range, which no run here has measured."
            )
        if all(p.dtype == torch.float32 for p in live):
            raise ValueError(
                "KahanBf16AdamW was given only fp32 parameters, which need no compensation. "
                "Use torch.optim.AdamW directly, or pass the bf16 parameters this class "
                "exists for."
            )
        for p in live:
            if not p.is_contiguous():
                raise ValueError(
                    f"a parameter of shape {tuple(p.shape)} is not contiguous; the step walks "
                    "each parameter as one flat view, and a copy would be a second weight "
                    "nothing writes back"
                )

        self._params = live
        self.param_groups: list[dict[str, Any]] = []
        for g in groups:
            extra = {k: v for k, v in g.items() if k != "params"}
            group = {
                "lr": float(lr), "betas": tuple(betas), "eps": float(eps),
                "weight_decay": float(weight_decay), **extra, "params": list(g["params"]),
            }
            if LR_SCALE_KEY in extra:
                group["lr"] = float(lr) * _check_lr_scale(
                    extra[LR_SCALE_KEY], where=f"group {extra.get('name', '?')!r}"
                )
            self.param_groups.append(group)
        self.state: dict[Any, dict[str, Any]] = {}
        for p in live:
            entry: dict[str, Any] = {
                "step": 0,
                "exp_avg": torch.zeros_like(p, dtype=torch.float32),
                "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
            }
            if p.dtype != torch.float32:
                entry["kahan_comp"] = torch.zeros_like(p)
            self.state[p] = entry

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        """One AdamW step on every parameter that has a gradient, in bounded slices."""
        torch = self._torch
        f32 = torch.float32
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps, weight_decay = group["eps"], group["weight_decay"]
                _check_hyper(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise ValueError("KahanBf16AdamW does not take sparse gradients")
                    if grad.shape != p.shape:
                        raise ValueError(
                            f"gradient shape {tuple(grad.shape)} does not match its parameter's "
                            f"{tuple(p.shape)}"
                        )
                    st = self.state[p]
                    st["step"] += 1
                    t = st["step"]
                    # As torch computes them: Python floats, pow for the root.
                    step_size = lr / (1 - beta1**t)
                    bias_correction2_sqrt = (1 - beta2**t) ** 0.5
                    decay = 1 - lr * weight_decay
                    flat_p = p.view(-1)
                    flat_g = grad.reshape(-1)
                    flat_m = st["exp_avg"].view(-1)
                    flat_v = st["exp_avg_sq"].view(-1)
                    comp = st.get("kahan_comp")
                    flat_c = None if comp is None else comp.view(-1)
                    n = flat_p.numel()
                    for start in range(0, n, self.chunk_elems):
                        end = min(n, start + self.chunk_elems)
                        g = flat_g[start:end].to(f32)
                        m = flat_m[start:end]
                        v = flat_v[start:end]
                        m.lerp_(g, 1 - beta1)
                        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                        denom = (v.sqrt() / bias_correction2_sqrt).add_(eps)
                        if flat_c is None:
                            w = flat_p[start:end]
                            w.mul_(decay)
                            w.addcdiv_(m, denom, value=-step_size)
                            continue
                        w = flat_p[start:end].to(f32).add_(flat_c[start:end].to(f32))
                        w.mul_(decay)
                        w.addcdiv_(m, denom, value=-step_size)
                        hi = w.to(p.dtype)
                        flat_p[start:end].copy_(hi)
                        # The residual is exact in fp32; copy_ rounds it to the buffer's bf16.
                        flat_c[start:end].copy_(w.sub_(hi.to(f32)))

    # -- checkpointing --------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Moments, compensation and step counts by parameter index, plus host ``masters``.

        Shaped like ``torch.optim.Optimizer.state_dict`` (``state`` keyed by index,
        ``param_groups`` with index lists), because the checkpoint walker in
        ``qd_train.backbone`` stringifies exactly the ``state`` sub-tree's integer keys.
        """
        torch = self._torch
        index = {id(p): i for i, p in enumerate(self._params)}
        state = {}
        for p in self._params:
            st = self.state[p]
            state[index[id(p)]] = {
                "step": st["step"],
                **{k: v for k, v in st.items() if k != "step"},
            }
        groups = []
        for g in self.param_groups:
            groups.append(
                {**{k: v for k, v in g.items() if k != "params"},
                 "params": [index[id(p)] for p in g["params"]]}
            )
        masters = []
        with torch.no_grad():
            for p in self._params:
                host = p.detach().to("cpu")
                comp = self.state[p].get("kahan_comp")
                if comp is None:
                    masters.append(host.clone())
                    continue
                exact = host.to(torch.float32) + comp.detach().to("cpu", torch.float32)
                tie = exact.to(p.dtype) != host
                masters.append(torch.where(tie, host.to(torch.float32), exact))
        return {"recipe": KAHAN_RECIPE, "state": state, "param_groups": groups,
                "masters": masters}

    def load_state_dict(self, saved: dict[str, Any]) -> None:
        """Restore moments, compensation, step counts and hyper-parameters, or refuse.

        Refused, before anything is written:
        - a state another optimizer wrote (no ``recipe``, or a different one), because
          resuming a master or plain run here would silently change the recipe mid-run;
        - a partial state;
        - a state for a different model (parameter count, group layout or shape);
        - a state whose ``masters`` disagree with the live weights, which means the weights
          and the optimizer came from different checkpoints.
        """
        torch = self._torch
        missing = {"recipe", "state", "param_groups", "masters"} - set(saved)
        if missing:
            raise ValueError(
                f"optimizer state is missing {sorted(missing)}. Refusing to load a partial "
                "state, or one another optimizer wrote: a resume would change the recipe or "
                "lose the compensation without saying so."
            )
        if saved["recipe"] != KAHAN_RECIPE:
            raise ValueError(
                f"optimizer state was written by recipe {saved['recipe']!r}, not "
                f"{KAHAN_RECIPE!r}. A resume continues the recipe it was cut from."
            )
        state, groups, masters = saved["state"], saved["param_groups"], saved["masters"]
        n = len(self._params)
        if len(state) != n or len(masters) != n:
            raise ValueError(
                f"optimizer state carries {len(state)} state entries and {len(masters)} "
                f"masters, but this optimizer has {n} parameters. The checkpoint describes a "
                "different model."
            )
        if len(groups) != len(self.param_groups) or any(
            len(s["params"]) != len(g["params"])
            for s, g in zip(groups, self.param_groups, strict=True)
        ):
            raise ValueError(
                "optimizer state's parameter groups do not match this optimizer's. The "
                "checkpoint describes a different model or a different group split."
            )
        staged: list[tuple[Any, dict[str, Any]]] = []
        for i, p in enumerate(self._params):
            entry = state.get(i)
            if entry is None:
                raise ValueError(f"optimizer state has no entry for parameter {i}")
            want = {"step", "exp_avg", "exp_avg_sq"} | (
                set() if p.dtype == torch.float32 else {"kahan_comp"}
            )
            if set(entry) != want:
                raise ValueError(
                    f"parameter {i}'s state holds {sorted(entry)}, expected {sorted(want)}"
                )
            for key in want - {"step"}:
                if tuple(entry[key].shape) != tuple(p.shape):
                    raise ValueError(
                        f"parameter {i}'s {key} has shape {tuple(entry[key].shape)}, this "
                        f"model's is {tuple(p.shape)}. The checkpoint describes a different "
                        "model."
                    )
            if tuple(masters[i].shape) != tuple(p.shape):
                raise ValueError(f"master {i} has shape {tuple(masters[i].shape)}")
            host = p.detach().to("cpu")
            if not torch.equal(masters[i].to(p.dtype), host):
                raise ValueError(
                    f"master {i} does not round to the live parameter: the weights and the "
                    "optimizer state come from different checkpoints"
                )
            step = entry["step"]
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"parameter {i}'s step must be a non-negative int, got {step!r}")
            staged.append((p, entry))
        with torch.no_grad():
            for p, entry in staged:
                st = self.state[p]
                st["step"] = entry["step"]
                st["exp_avg"].copy_(entry["exp_avg"].to(st["exp_avg"].device, torch.float32))
                st["exp_avg_sq"].copy_(
                    entry["exp_avg_sq"].to(st["exp_avg_sq"].device, torch.float32)
                )
                if "kahan_comp" in st:
                    st["kahan_comp"].copy_(
                        entry["kahan_comp"].to(st["kahan_comp"].device, p.dtype)
                    )
        for mine, saved_group in zip(self.param_groups, groups, strict=True):
            for key, value in saved_group.items():
                if key != "params":
                    mine[key] = tuple(value) if key == "betas" else value


def _check_hyper(
    *, lr: float, betas: tuple[float, float], eps: float, weight_decay: float
) -> None:
    """Refuse a hyper-parameter that would write NaN or a wrong step into every weight."""
    if not (isinstance(lr, (int, float)) and math.isfinite(lr) and lr >= 0.0):
        raise ValueError(f"lr must be finite and non-negative, got {lr!r}")
    if len(betas) != 2 or not all(
        isinstance(b, (int, float)) and 0.0 <= b < 1.0 for b in betas
    ):
        raise ValueError(f"betas must be two numbers in [0, 1), got {betas!r}")
    if not (isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be finite and positive, got {eps!r}")
    if not (
        isinstance(weight_decay, (int, float))
        and math.isfinite(weight_decay)
        and weight_decay >= 0.0
    ):
        raise ValueError(f"weight_decay must be finite and non-negative, got {weight_decay!r}")


def build_optimizer(
    params: Any,
    *,
    spec: Any,
    lr: float,
    total_steps: int,
    allow_frozen_moments: bool = False,
    beta2: float = DEFAULT_BETA2,
    fused: bool = False,
) -> Any:
    """The one place that turns an [`qd_train.memory.OptimizerSpec`] into an optimizer.

    ``fused`` builds the master recipe's inner AdamW as torch's fused kernel (see
    [`MasterWeightAdamW`]). The master recipe only: it is the one every long run uses and
    the one measured, and a flag that silently did nothing on the other would be recorded
    as a recipe it did not run.

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

    ``beta2`` is AdamW's second-moment decay, default :data:`DEFAULT_BETA2` (unchanged).
    decider (Mapika/decider, Apache-2.0, @23579f7) trains at 0.95. The fidelity check below
    is asked **at the beta2 the optimizer is built with** -- the settling point is a
    property of ``(dtype, beta2)``, and checking 0.999's answer for a 0.95 optimizer would
    admit or refuse the wrong run. Measured by :func:`moment_settling` on this host: bf16 at
    0.95 settles 1.95% low after 64 steps (against 50% low after 384 at 0.999), which is
    still outside ``MAX_MOMENT_RELATIVE_ERROR``; fp32 at 0.95 settles 0.0001% low.

    ``params`` may be tensors or parameter-group dicts (see [`layerwise_param_groups`]).

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
    if not (isinstance(beta2, float) and 0.0 < beta2 < 1.0):
        raise ValueError(f"beta2 must be a float in (0, 1), got {beta2!r}")
    groups = _normalise_groups(params)
    betas = (0.9, beta2)
    if spec.keeps_fp32_master:
        return MasterWeightAdamW(groups, lr=lr, betas=betas, fused=fused)
    if fused:
        raise ValueError(
            f"fused=True is built for the fp32-master recipe only, and spec {spec.name!r} "
            "keeps no master. Refusing rather than recording a fused optimizer that is not"
        )
    if spec.compensation_bytes:
        # KahanBf16AdamW keeps exactly two fp32 moments and one bf16 compensation; a spec
        # that budgets anything else describes a layout this does not build. Its moments are
        # fp32, so the frozen-moment guard below has nothing to refuse.
        if (spec.states_per_param, spec.state_bytes, spec.compensation_bytes) != (2, 4, 2):
            raise ValueError(
                f"optimizer spec {spec.name!r} budgets {spec.states_per_param} state(s) of "
                f"{spec.state_bytes} bytes and a {spec.compensation_bytes}-byte compensation, "
                "but KahanBf16AdamW keeps two fp32 moments and one bf16 compensation. The "
                "budget would describe a layout nothing builds."
            )
        return KahanBf16AdamW(groups, lr=lr, betas=betas)
    if spec.states_per_param != 2:
        raise ValueError(
            f"optimizer spec {spec.name!r} describes {spec.states_per_param} state tensor(s) "
            "per parameter, but this builder constructs torch.optim.AdamW, which keeps "
            "exactly two (exp_avg and exp_avg_sq). Refusing to return an optimizer the "
            "budget does not describe."
        )
    trainable = [p for g in groups for p in g["params"]]
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
        settling = moment_settling(dtype=dominant, beta2=beta2)
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
    for g in groups:
        if LR_SCALE_KEY in g:
            g["lr"] = float(lr) * _check_lr_scale(
                g[LR_SCALE_KEY], where=f"group {g.get('name', '?')!r}"
            )
    return torch.optim.AdamW(groups, lr=lr, betas=betas)
