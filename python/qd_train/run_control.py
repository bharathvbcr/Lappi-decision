"""How long a run may take, how fast it learns, what it costs, and where it is.

Everything here is **torch-free on purpose**. ``docs/training-contract.md``: *"Format,
ordering, bucketing, coverage policy, schedules, the wall-clock cap and checkpoint
bookkeeping are torch-free, and run in CI on any machine. Model loading, the fused
cross-entropy and the optimizer step need torch, and their tests ``importorskip``."*

That split is not tidiness. The schedule, the cap and the resume position are the parts
whose bugs are **silent** — a warmup off by one step, a cap that is checked but never
fires, a resume that restarts two batches early — and they are also the parts that need
no GPU to test. Keeping them here means they are covered by the suite that actually runs
on this host, rather than by the suite that is reported as "not run".

## What each piece is defending against

**[`LRSchedule`] is a pure function of the step.** Not a stateful object that is advanced,
because a stateful schedule and a resumed run disagree about the step count exactly once —
at the resume — and the loss curve afterwards looks plausible. ``lr_at`` **raises** past the
end of the schedule rather than clamping: a loop that asks for the LR of a step the schedule
does not cover has a different idea of the run length than the schedule does, and clamping
hides that until the run ends early or late.

**[`WallClockCap`] is mandatory and bounded above.** Rule 4: *"Every 8xH100 job carries a
wall-clock cap, auto-terminate and a cost estimate."* There is no "no cap" value; the type
cannot express one. ``docs/plan-corrections.md`` DESIGN-4 makes [`MAX_CAP_S`] the program's
own 40 h cap, and rule 2 makes it read-only — an agent may report that the cap was hit, it
may not raise the cap to avoid hitting it.

**A cap is only a cap if something enforces it without being asked.** ``expired()`` is a poll
at an optimizer-step boundary; a step that blocks never reaches one. [`hard_exit_on_cap`] and
the watchdog ``RunControl.start()`` arms are the half that does not need the loop's help, and
they are mandatory for exactly the runs rule 4 names.

**[`CostEstimate`] is the cap priced.** DESIGN-4's finding was precisely that the plan's cap
was *not costed*: 40 h x $31.92 = $1,277 for the block alone against a "$1,085 at ~34 h"
estimate, so "we hit the cap" was also a silent 15% budget overrun. The estimate here is
computed from the cap, never from the hoped-for duration, so the number on the launch line
is the worst case rather than the happy case. DESIGN-4's *other* half — the per-GPU $3.99
against the per-node $31.92 — is why a multi-GPU estimate must state both columns: nothing
about a bare rate says which one it came from, and they differ by exactly ``n_gpus``.

**[`Position`] is `(epoch, index)`, and it is not the whole claim.** ``Batch.index`` is the
batch's position in an order that is a pure function of ``(seed, epoch, batch_tokens)`` and
the shard set -- ``ShardReader._plan`` mixes all four into its ``SeedSequence`` -- never of
hidden iterator state. So the data side is *rebuilt* rather than restored, and two integers
say where to rebuild to. They do not say which of the many orders passing through that index
was the one this run ate, which is why [`Checkpoint`] also carries a [`ConsumedPrefix`]
digest: measured on 2026-09-20, a resume whose ``batch_tokens`` differed by one token hit the
same index, passed every check, and trained on a different corpus order to completion.
Anything else -- an iterator to pickle, a file offset, a shuffle buffer -- would be state that
has to be restored rather than recomputed, and restored state is what drifts.

**[`Checkpoint`] is JSON plus a safetensors sidecar, and it is the sidecar that makes it
usable at this program's size.** JSON stores a float as decimal text, and ``t.tolist()``
widens a float32 to a float64 whose shortest round-tripping decimal is 17 significant
figures, so a trained weight costs about 21 bytes and a raw one costs 4.

Measured 2026-09-20 on the real rung-0 step with the AdamW moments present -- the same
checkpoint content encoded both ways, which is the only comparison that means anything:

===========================  =================  ====================
per model parameter          JSON               JSON + sidecar
===========================  =================  ====================
rung 0, width 64             68.84 B            12.35 B  (5.6x)
rung 0, width 128            68.86 B            12.10 B  (5.7x)
===========================  =================  ====================

A resumable checkpoint holds roughly **three** stored numbers per model parameter, not one:
the weight and AdamW's two moments. At 12.1 bytes per parameter the 1,401,501,504-parameter
remapped model is about **17 GB per checkpoint** as a pair against **97 GB** as JSON. Both
numbers matter -- the second is a rented disk full before a two-hour run ends, and the first
is why [`MAX_SIDECAR_BYTES`] is set where it is.

**A caution about the 2.80 GB figure that circulates with this work:** that is
1,401,501,504 bf16 *weights* and nothing else. It is not a checkpoint. A resume from
weights alone restarts Adam's moments from zero and does not reproduce the trajectory,
which is the property S5 rests on.

[`TensorRef`] is why ``TrainStep.state()`` no longer promises to be JSON-serialisable. Two
files are two chances to have half a checkpoint; [`Checkpoint.write`] binds them with an
ordering, a content-addressed name and a digest, and [`Checkpoint.read`] refuses any pair
that does not agree.

**[`LossLog`] stores floats as ``float.hex()``.** S5 asks that a resumed run reproduce the
loss trajectory *exactly*. ``repr``/``%g`` round-trips are not exact at every magnitude, and
a digest over lossy text compares the rendering rather than the number. ``float.hex`` is
exact and total, so the digest is a statement about the run and not about formatting.

## What "bit-exact resume" can and cannot mean here, by device

Everything in this module is integer and float bookkeeping, and is bit-exact everywhere. The
*trajectory* it records is only as reproducible as the step that produced it, and that is a
property of the device, not of the seed:

* **cpu** -- measured bit-exact across processes: six of six paired runs in the REALFT lane
  agreed on ``train.loss_log_digest``.
* **mps** -- measured **not** bit-exact: three of three paired runs of the identical command
  disagreed, and on one seed the drift moved a reported answer from 21/39 to 20/39. A seed
  does not pin an ``mps`` run, so neither does a checkpoint.
* **cuda** -- **not run**. There is no CUDA device on this host; nothing here has been
  measured on one, and the repo states no result for it.

A resume claim that does not name its device is not a claim this repository can support.
"""

from __future__ import annotations

import _thread
import ctypes
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Self

__all__ = [
    "APPROVAL_FREE_USD",
    "CAP_EXIT_CODE",
    "EMPTY_PREFIX_DIGEST",
    "MAX_CAP_S",
    "MAX_CHECKPOINT_BYTES",
    "MAX_GRAD_ACCUM",
    "MAX_LOSS_POINTS",
    "MAX_SIDECAR_BYTES",
    "MAX_SIDECAR_TENSORS",
    "TERMINATE_GRACE_S",
    "AccumulationGroup",
    "Checkpoint",
    "ConsumedPrefix",
    "CostEstimate",
    "LRSchedule",
    "LaunchRefused",
    "LossLog",
    "LossPoint",
    "Position",
    "RunControl",
    "TensorRef",
    "TerminationReason",
    "WallClockCap",
    "hard_exit_on_cap",
]

#: Why a loop stopped. Every exit from a training loop is one of these three; there is no
#: "just ended" case, because a loop that cannot say why it stopped cannot be told apart
#: from one that stopped by accident.
TerminationReason = Literal["data_exhausted", "steps_exhausted", "wall_clock_cap"]

_TERMINATION_REASONS: frozenset[str] = frozenset(TerminationReason.__args__)  # type: ignore[attr-defined]

#: The program's own wall-clock cap, from ``docs/plan-corrections.md`` DESIGN-4 ("block cost
#: $1,085 at ~34 h, wall-clock cap 40 h"). A cap above the program's cap is not a cap, so a
#: larger value is refused rather than accepted. Rule 2: read-only. An agent may report that
#: this was hit; moving it to make a run fit is a refused change.
MAX_CAP_S: Final[float] = 40 * 3600.0

#: Rule 4: *"Every 8xH100 job carries a wall-clock cap, auto-terminate and a cost estimate,
#: and needs a human yes before launch. Single-GPU jobs under $20 do not."* This is the
#: "under $20" half of that sentence. Read-only for the same reason.
APPROVAL_FREE_USD: Final[float] = 20.0

#: Gradient accumulation is a loop, so it is bounded. The number is far above any real
#: recipe; its job is to turn a mis-parsed config into a refusal instead of a hang.
MAX_GRAD_ACCUM: Final[int] = 4096

#: One point per optimizer step, so this bounds the schedule length too.
MAX_LOSS_POINTS: Final[int] = 5_000_000

#: The largest **JSON body** [`Checkpoint.write`] will produce. The number has not moved
#: since it was set; what it bounds has, and that is stated here rather than left to be
#: inferred from an unchanged constant.
#:
#: It was set against a body that carried the weights as decimal text, measured on
#: 2026-09-20 at **68.86 bytes per model parameter** for a rung-0 checkpoint with its AdamW
#: moments -- so one gibibyte was about 15.6M parameters and the 2B backbone this program is
#: built around did not fit by a factor of 90. [`TensorRef`] moved the weights out to a
#: safetensors sidecar, so this now bounds the position, the schedule, the loss log and the
#: per-tensor digests, and nothing whose size scales with the model. That makes it a
#: **tighter** bound on what it measures, not a looser one: everything it used to permit it
#: still permits, and the payload that used to dominate it is no longer in it. Rule 2 --
#: the number is unchanged and no gate was widened to let anything through.
#:
#: The tensor payload is bounded separately and explicitly by [`MAX_SIDECAR_BYTES`].
MAX_CHECKPOINT_BYTES: Final[int] = 1 << 30

#: The largest tensor payload [`Checkpoint.write`] will put in a sidecar. This is a **new
#: bound on a new payload**, not a relaxation of [`MAX_CHECKPOINT_BYTES`]: before this
#: existed, a model state of this size could not be written at all.
#:
#: The number is derived from the largest checkpoint this repository can construct, not
#: guessed. ``qd_train.memory.QWEN3_5_2B_TEXT.trainable_params()`` is 1,881,825,088 and
#: ``ADAMW_FP32`` keeps two fp32 moments per parameter, so weights and optimizer state in
#: fp32 -- the widest combination any recipe here can ask for -- is 1,881,825,088 x 12 =
#: 22,581,901,056 bytes, **21.03 GiB**. 32 GiB leaves 1.52x headroom over that and still
#: refuses a runaway: a checkpoint larger than the whole model in its widest dtype is a bug
#: in the step's ``state()``, not a recipe. ``test_the_sidecar_bound_is_derived_from_the_
#: model_it_has_to_hold`` pins the derivation against ``qd_train.memory`` so the two cannot
#: drift apart silently. Rule 2 applies to it from here on.
MAX_SIDECAR_BYTES: Final[int] = 32 << 30

#: How many tensors one checkpoint may carry. Every fan-out here is bounded; this one turns
#: a state dict built in a loop that never terminates into a refusal instead of a write that
#: never returns. Qwen3.5-2B has fewer than 1,000 parameter tensors and AdamW doubles that,
#: so a million is four orders of magnitude of headroom.
MAX_SIDECAR_TENSORS: Final[int] = 1 << 20

#: How long each rung of [`hard_exit_on_cap`]'s escalation ladder waits before the next. The
#: ladder is bounded on purpose: rule 4's "auto-terminate" is worthless if the terminate step
#: can itself block. Total worst case from expiry to process death is two of these.
TERMINATE_GRACE_S: Final[float] = 30.0

#: The status [`hard_exit_on_cap`] exits with. 124 is the convention GNU ``timeout`` uses for
#: "the thing ran past its limit", so a wrapper script reads it without being taught a new
#: number. It is deliberately **not** 3, which this repo's gates mean by ``NotRun``.
CAP_EXIT_CODE: Final[int] = 124

#: The watchdog wakes at most this often. It is a lag bound on noticing the cap, not a second
#: cap, and it is clamped against the cap below so a short cap is still noticed promptly.
_WATCHDOG_POLL_S: Final[float] = 1.0
_MIN_WATCHDOG_POLL_S: Final[float] = 0.005

#: Float slack for the per-GPU/per-instance identity check below. This is representation
#: noise in ``rate * n``, not a policy tolerance: $4.19 x 2 must equal $8.38 and $2.79 x 8
#: must equal $22.32 after binary rounding, and nothing looser is intended.
_RATE_REL_TOL: Final[float] = 1e-9


class LaunchRefused(RuntimeError):
    """A run was configured in a way rule 4 does not permit it to start in."""


def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _require_elapsed(name: str, value: float) -> float:
    """Finite **and** non-negative. Time does not run backwards.

    ``cost_for(-3600.0)`` used to return ``-4.29``: a negative bill, which reads as a run that
    earned money and which ``trainer.py`` writes straight into the append-only ledger row.
    ``remaining_s(-1e9)`` used to answer with a billion seconds of headroom against a ten
    second cap. Neither is a cheap run; both are a clock that is wrong, and a clock that is
    wrong is the one thing a wall-clock cap cannot survive quietly.
    """
    v = _require_finite(name, value)
    if v < 0.0:
        raise ValueError(
            f"{name} must be non-negative, got {v!r}. Elapsed time that is negative is not a "
            "short run; it is a broken clock, and pricing it yields a negative bill."
        )
    return v


def hard_exit_on_cap(reason: str) -> None:
    """Rule 4's "auto-terminate", as the last thing that happens on a rented box.

    Three rungs, each bounded by [`TERMINATE_GRACE_S`], because each one fails in a way the
    next one covers:

    1. ``interrupt_main()`` raises ``KeyboardInterrupt`` at the main thread's next bytecode
       boundary. A cooperative loop unwinds, writes its final checkpoint and stops -- the
       cheapest possible ending, and the only rung that keeps the work done so far.
    2. ``SIGTERM``, for a main thread that never reaches a bytecode boundary because it is
       blocked inside a C call: a CUDA kernel, a dataloader join, an NFS read.
    3. ``os._exit``, for a process that ignores ``SIGTERM`` -- a handler that swallows it, or
       an interpreter wedged below the signal machinery. This rung does not run ``atexit``
       hooks or flush buffers, which is the point: the alternative is billing until a human
       notices.

    Pass something else as ``RunControl.auto_terminate`` if a run needs a different ending --
    releasing a cloud lease, say. This one is the default because it is the one that always
    works.
    """
    print(f"\nWALL-CLOCK CAP: {reason}", file=sys.stderr, flush=True)
    print(
        "  auto-terminate (rule 4): interrupting the main thread; "
        f"SIGTERM in {TERMINATE_GRACE_S:.0f}s; hard exit {TERMINATE_GRACE_S:.0f}s after that.",
        file=sys.stderr,
        flush=True,
    )
    _thread.interrupt_main()
    time.sleep(TERMINATE_GRACE_S)
    print("WALL-CLOCK CAP: still alive after the interrupt; SIGTERM.", file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(TERMINATE_GRACE_S)
    print(
        f"WALL-CLOCK CAP: still alive after SIGTERM; exiting {CAP_EXIT_CODE} without unwinding.",
        file=sys.stderr,
        flush=True,
    )
    os._exit(CAP_EXIT_CODE)


# --- the schedule -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LRSchedule:
    """Linear warmup then cosine decay, as a pure function of the optimizer step.

    ``lr_at(step)`` depends on nothing but ``step`` and this object's fields, which is what
    lets a resumed run land on the same learning rate as the run it resumes: the step
    number is restored from the checkpoint and the rate is *recomputed*, never restored.
    """

    peak_lr: float
    total_steps: int
    warmup_steps: int = 0
    min_lr: float = 0.0

    def __post_init__(self) -> None:
        peak = _require_finite("peak_lr", self.peak_lr)
        floor = _require_finite("min_lr", self.min_lr)
        if peak <= 0.0:
            raise ValueError(f"peak_lr must be positive, got {peak!r}")
        if floor < 0.0:
            raise ValueError(f"min_lr must be non-negative, got {floor!r}")
        if floor > peak:
            raise ValueError(
                f"min_lr {floor!r} exceeds peak_lr {peak!r}: the schedule would decay upwards"
            )
        if not isinstance(self.total_steps, int) or isinstance(self.total_steps, bool):
            raise TypeError(f"total_steps must be int, got {type(self.total_steps).__name__}")
        if not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool):
            raise TypeError(f"warmup_steps must be int, got {type(self.warmup_steps).__name__}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.total_steps > MAX_LOSS_POINTS:
            raise ValueError(
                f"total_steps {self.total_steps} exceeds MAX_LOSS_POINTS {MAX_LOSS_POINTS}: "
                "the loss log this schedule implies would not be bounded"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps {self.warmup_steps} >= total_steps {self.total_steps}: the run "
                "would end before the warmup does, so the peak rate is never reached and the "
                "recipe's stated peak_lr describes a rate this run never uses"
            )

    def lr_at(self, step: int) -> float:
        """The rate for ``step``, 0-based over optimizer steps.

        Raises past the end of the schedule. A loop that asks for step ``total_steps`` has a
        different run length in mind than this object does, and a clamped value would let the
        two disagree quietly for the rest of the run.
        """
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError(f"step must be int, got {type(step).__name__}")
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if step >= self.total_steps:
            raise ValueError(
                f"step {step} is past the end of a {self.total_steps}-step schedule. The loop "
                "and the schedule disagree about how long this run is; that is a bug in one of "
                "them, not a rate to clamp."
            )
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        span = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / span
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * decay

    def to_json(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return cls(
            peak_lr=float(raw["peak_lr"]),
            total_steps=int(raw["total_steps"]),
            warmup_steps=int(raw["warmup_steps"]),
            min_lr=float(raw["min_lr"]),
        )


# --- the cap and its price --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WallClockCap:
    """The hard stop. Mandatory: there is no value of this type meaning "no cap"."""

    cap_s: float

    def __post_init__(self) -> None:
        cap = _require_finite("cap_s", self.cap_s)
        if cap <= 0.0:
            raise ValueError(f"cap_s must be positive, got {cap!r}")
        if cap > MAX_CAP_S:
            raise ValueError(
                f"cap_s {cap!r}s exceeds MAX_CAP_S {MAX_CAP_S!r}s ({MAX_CAP_S / 3600:.0f} h), the "
                "program's own cap from docs/plan-corrections.md DESIGN-4. Rule 2: a kill "
                "criterion is read-only. Report that the cap was hit; do not raise it."
            )

    @property
    def cap_hours(self) -> float:
        return self.cap_s / 3600.0

    def expired(self, elapsed_s: float) -> bool:
        return _require_elapsed("elapsed_s", elapsed_s) >= self.cap_s

    def remaining_s(self, elapsed_s: float) -> float:
        return max(0.0, self.cap_s - _require_elapsed("elapsed_s", elapsed_s))


def _no_false_zero(value: float, places: int = 2) -> str:
    """Two decimals, unless two decimals would render something real as nothing.

    ``f"{0.004:.2f}"`` is ``"0.00"``, so a rate of $0.004/h printed on a launch line reads as
    a machine that is free, and a 30 s cap prints as ``0.00 h``. Two decimals are right for
    money and for hours; silently rounding a nonzero quantity to zero is not.
    """
    rounded = round(value, places)
    if rounded == 0.0 and value != 0.0:
        return repr(float(value))
    return f"{rounded:.{places}f}"


def _usd(value: float) -> str:
    """A dollar figure that never renders as something it is not.

    ``:.2f`` rounds to nearest, so the printed projection and the *decision* were computed
    from different numbers: measured on this class, $19.9998 printed as ``$20.00`` on the
    same line as *"no approval required"*. Two cents is a rounding error; a line that argues
    with its own verdict is not, because the human approves against the line.

    So: two decimals when two decimals are the truth, and enough digits to stay on the right
    side of zero and of [`APPROVAL_FREE_USD`] when they are not.
    """
    rounded = round(value, 2)
    if (rounded >= APPROVAL_FREE_USD) != (value >= APPROVAL_FREE_USD):
        # ``repr`` and not ``%g``: six significant figures turn 19.999999998 back into "20",
        # which is the contradiction this branch exists to avoid.
        return repr(float(value))
    return _no_false_zero(value)


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The cap, priced. DESIGN-4: state the cap as what it costs.

    ``projected_usd`` is computed from the **cap**, not from the hoped-for duration, so
    hitting the cap is not also a budget surprise.

    ``usd_per_hour`` is the **whole instance's** rate, and on a multi-GPU box the price list
    prints two columns that differ by exactly ``n_gpus``. DESIGN-4's own finding was a rate
    read off the wrong one ("the Lambda page lists *per-GPU* prices, $3.99", against $31.92
    for the node). Nothing about a bare number says which column it came from, and the error
    is invisible: $2.79 x 8 GPU-hours and $2.79 an hour for the node are both plausible rates
    for the same machine, eight times apart. So a multi-GPU estimate must state **both**
    columns and they must agree -- the one check that catches a swap without a price table,
    because a swap is precisely what makes them disagree. At ``n_gpus <= 1`` the columns are
    the same number by definition and nothing is required.
    """

    usd_per_hour: float
    cap: WallClockCap
    n_gpus: int
    instance: str
    usd_per_gpu_hour: float | None = None

    def __post_init__(self) -> None:
        rate = _require_finite("usd_per_hour", self.usd_per_hour)
        if rate < 0.0:
            raise ValueError(f"usd_per_hour must be non-negative, got {rate!r}")
        if not isinstance(self.n_gpus, int) or isinstance(self.n_gpus, bool):
            raise TypeError(f"n_gpus must be int, got {type(self.n_gpus).__name__}")
        if self.n_gpus < 0:
            raise ValueError(f"n_gpus must be non-negative, got {self.n_gpus}")
        if not isinstance(self.instance, str) or not self.instance.strip():
            raise ValueError(
                "instance must name the machine being priced: a rate with no instance cannot "
                "be checked against a price list, and DESIGN-4's error was exactly a rate read "
                "off the wrong row (per-GPU $3.99 vs per-node $31.92)"
            )
        if self.n_gpus > 1:
            if self.usd_per_gpu_hour is None:
                raise ValueError(
                    f"{self.instance} has {self.n_gpus} GPUs, so its price list prints a "
                    "per-instance and a per-GPU column that differ by exactly "
                    f"{self.n_gpus}x. Pass usd_per_gpu_hour as well: reading the wrong column "
                    f"under-reports this run's cost {self.n_gpus}-fold and nothing else in "
                    "this object can tell the two apart."
                )
            per_gpu = _require_finite("usd_per_gpu_hour", self.usd_per_gpu_hour)
            if per_gpu < 0.0:
                raise ValueError(f"usd_per_gpu_hour must be non-negative, got {per_gpu!r}")
            if not math.isclose(rate, per_gpu * self.n_gpus, rel_tol=_RATE_REL_TOL):
                raise ValueError(
                    f"{self.instance}: usd_per_hour {rate!r} is not usd_per_gpu_hour "
                    f"{per_gpu!r} x {self.n_gpus} GPUs (= {per_gpu * self.n_gpus!r}). One of "
                    "the two came off the wrong column of the price list. This is DESIGN-4's "
                    "error and it is worth exactly the ratio between them."
                )
        elif self.usd_per_gpu_hour is not None:
            per_gpu = _require_finite("usd_per_gpu_hour", self.usd_per_gpu_hour)
            if not math.isclose(rate, per_gpu * max(self.n_gpus, 1), rel_tol=_RATE_REL_TOL):
                raise ValueError(
                    f"{self.instance}: usd_per_hour {rate!r} and usd_per_gpu_hour {per_gpu!r} "
                    f"disagree at n_gpus={self.n_gpus}"
                )

    #: Devices this repository runs on that cost nothing marginal: they are the machine the
    #: work is already being done on. Anything else is hardware that is being paid for by
    #: the hour, and :meth:`for_device` will not price it at zero on a caller's behalf.
    LOCAL_DEVICES: ClassVar[frozenset[str]] = frozenset({"cpu", "mps"})

    @classmethod
    def for_device(
        cls,
        *,
        cap: WallClockCap,
        device: str,
        n_gpus: int | None = None,
        usd_per_hour: float | None = None,
        usd_per_gpu_hour: float | None = None,
        instance: str | None = None,
    ) -> CostEstimate:
        """The price of a run on ``device``, refusing to invent one for rented hardware.

        Four tools reached this class through the same literal --
        ``usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}"`` -- and ``_control``'s own
        docstring said what was supposed to happen instead: *"a rented machine sets a real
        rate here"*. Nothing made it. Every GH200 run this project has made recorded itself
        as a local run on zero GPUs at zero dollars an hour.

        **Those two zeros disable both halves of rule 4 at once**, which is why this is a
        refusal and not a warning:

        * :attr:`requires_human_approval` is ``n_gpus > 1 or projected_usd >= 20``. At
          ``(0, 0.0)`` both disjuncts are False **for any cap**, so the human-yes gate
          cannot fire -- on an 8xH100 job either.
        * the per-GPU/per-instance column check is gated on ``if self.n_gpus > 1``, so
          DESIGN-4's own error, a rate read off the wrong column, is unguarded.

        ``cpu`` and ``mps`` are priced at zero because that is a measured fact about a Mac
        that is already bought, not a way around the rule. ``cuda`` is not local by
        definition: it is a rented box, and the caller states its rate, its instance and its
        GPU count or gets a refusal naming all three.
        """
        if device in cls.LOCAL_DEVICES:
            return cls(
                cap=cap,
                usd_per_hour=0.0 if usd_per_hour is None else usd_per_hour,
                n_gpus=0 if n_gpus is None else n_gpus,
                instance=instance or f"local-{device}",
                usd_per_gpu_hour=usd_per_gpu_hour,
            )
        missing = [
            name
            for name, value in (
                ("instance", instance),
                ("usd_per_hour", usd_per_hour),
                ("n_gpus", n_gpus),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"device {device!r} is not one of {sorted(cls.LOCAL_DEVICES)}, so it is "
                f"hardware being paid for by the hour, and {', '.join(missing)} cannot be "
                "defaulted here. Rule 4 needs a cost estimate before an 8xH100 job launches, "
                "and the zero-rate/zero-GPU default that used to stand in for one makes "
                "requires_human_approval False for ANY cap and skips the per-GPU column "
                "check entirely -- so the two defaults that look harmless are exactly the "
                "two that turn the rule off. State what the machine is and what it costs."
            )
        return cls(
            cap=cap,
            usd_per_hour=usd_per_hour,
            n_gpus=n_gpus,
            instance=instance,
            usd_per_gpu_hour=usd_per_gpu_hour,
        )

    @property
    def projected_usd(self) -> float:
        """What this run costs **if it runs to the cap**. The number that goes on the line."""
        return self.usd_per_hour * self.cap.cap_hours

    def cost_for(self, elapsed_s: float) -> float:
        """What it cost in fact. Measured, for the ledger row."""
        return self.usd_per_hour * _require_elapsed("elapsed_s", elapsed_s) / 3600.0

    @property
    def requires_human_approval(self) -> bool:
        """Rule 4. Multi-GPU always; single-GPU only once the capped cost reaches $20."""
        return self.n_gpus > 1 or self.projected_usd >= APPROVAL_FREE_USD

    def approval_line(self) -> str:
        need = "NEEDS A HUMAN YES" if self.requires_human_approval else "no approval required"
        per_gpu = ""
        if self.n_gpus > 1 and self.usd_per_gpu_hour is not None:
            per_gpu = f" (${_usd(self.usd_per_gpu_hour)}/GPU/h x {self.n_gpus})"
        return (
            f"{self.instance}: {self.n_gpus} GPU(s) at ${_usd(self.usd_per_hour)}/h per "
            f"instance{per_gpu}, capped at {_no_false_zero(self.cap.cap_hours)} h -> "
            f"${_usd(self.projected_usd)} at the cap -- {need}"
        )


# --- where the run is -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Position:
    """``(epoch, index)``: the next batch this run should consume.

    ``index`` is the *next* ``Batch.index``, not the last one consumed, so a checkpoint reads
    as "start here" rather than "you have done up to here" -- the off-by-one that makes a
    resumed run repeat or skip one batch lives exactly in that distinction.
    """

    epoch: int
    index: int

    def __post_init__(self) -> None:
        for name in ("epoch", "index"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"Position.{name} must be int, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"Position.{name} must be non-negative, got {v}")

    def to_json(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "index": self.index}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return cls(epoch=int(raw["epoch"]), index=int(raw["index"]))


@dataclass(frozen=True, slots=True)
class LossPoint:
    """One optimizer step's loss, tagged with the batch it came from."""

    optimizer_step: int
    epoch: int
    batch_index: int
    loss: float

    def __post_init__(self) -> None:
        for name in ("optimizer_step", "epoch", "batch_index"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"LossPoint.{name} must be int, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"LossPoint.{name} must be non-negative, got {v}")
        loss = _require_finite("LossPoint.loss", self.loss)
        if loss < 0.0:
            raise ValueError(
                f"LossPoint.loss must be non-negative, got {loss!r}. A negative cross-entropy "
                "is not a small loss; it is a masking or normalisation bug."
            )

    def to_json(self) -> dict[str, Any]:
        # float.hex round-trips exactly. A digest over %g compares the rendering.
        return {
            "step": self.optimizer_step,
            "epoch": self.epoch,
            "index": self.batch_index,
            "loss_hex": float(self.loss).hex(),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return cls(
            optimizer_step=int(raw["step"]),
            epoch=int(raw["epoch"]),
            batch_index=int(raw["index"]),
            loss=float.fromhex(raw["loss_hex"]),
        )


class LossLog:
    """The loss trajectory, append-only and strictly ordered by optimizer step.

    S5's resume test compares two of these for equality. The ordering invariant is what
    makes that comparison meaningful: a resumed run that replays a step, or skips one,
    fails on append rather than producing a log that is merely the wrong length.
    """

    __slots__ = ("_points",)

    def __init__(self, points: Iterable[LossPoint] = ()) -> None:
        self._points: list[LossPoint] = []
        for p in points:
            self.append(p)

    def append(self, point: LossPoint) -> None:
        if not isinstance(point, LossPoint):
            raise TypeError(f"LossLog holds LossPoint, got {type(point).__name__}")
        if len(self._points) >= MAX_LOSS_POINTS:
            raise ValueError(
                f"loss log reached MAX_LOSS_POINTS ({MAX_LOSS_POINTS}); a training loop that "
                "logs more steps than any schedule declares is not terminating"
            )
        if self._points and point.optimizer_step <= self._points[-1].optimizer_step:
            raise ValueError(
                f"optimizer step {point.optimizer_step} follows {self._points[-1].optimizer_step}: "
                "the loss log is strictly increasing in step. A repeated or rewound step means a "
                "resume landed in the wrong place, which is precisely what S5 forbids."
            )
        self._points.append(point)

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LossLog):
            return NotImplemented
        return self._points == other._points

    def __repr__(self) -> str:
        return f"LossLog({len(self._points)} points, digest={self.digest()[:12]})"

    @property
    def last(self) -> LossPoint | None:
        return self._points[-1] if self._points else None

    def as_tuple(self) -> tuple[LossPoint, ...]:
        return tuple(self._points)

    def losses(self) -> tuple[float, ...]:
        return tuple(p.loss for p in self._points)

    def snapshot(self) -> LossLog:
        """A copy, so a checkpoint is not a live view of a log that keeps growing."""
        out = LossLog()
        out._points = list(self._points)
        return out

    def digest(self) -> str:
        """sha256 over the exact numbers. Two runs agree iff this agrees."""
        body = json.dumps(
            [p.to_json() for p in self._points], sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def to_json(self) -> list[dict[str, Any]]:
        return [p.to_json() for p in self._points]

    @classmethod
    def from_json(cls, raw: list[dict[str, Any]]) -> Self:
        return cls(LossPoint.from_json(r) for r in raw)


class ConsumedPrefix:
    """A running sha256 over the batches a run has actually consumed, in order.

    ``Position`` says *where* a resume should start; this says *what the run that stopped
    there had eaten*. The two are not the same claim, and only the second one can be
    checked against the source a resumed run is handed.

    The gap it closes was measured on 2026-09-20 against the pre-fix code. The batch order
    is a pure function of ``(seed, epoch, batch_tokens)`` **and the shard set** --
    ``ShardReader._plan`` mixes all three into its ``SeedSequence``. ``Checkpoint`` carried
    the seed and the schedule, "so a resume against a differently-seeded source is refused
    rather than silently training on a different order", and carried neither
    ``batch_tokens`` nor any identity of the shard set. A resume whose ``batch_tokens``
    differed by one token produced a source with the same number of batches and the same
    indices, so ``trainer._skip_to``'s "did we land on index N" check passed, and the run
    trained to completion on an entirely different sequence of batches while reporting
    ``termination='steps_exhausted'``.

    Enumerating the order's parameters would fix that case. Hashing what was consumed fixes
    the class: any difference in the data -- a different budget, a different shard set, a
    rebucketing, a rewritten corpus -- lands in this digest, including ones nobody has
    thought of yet.

    Every part is length-prefixed, so ``fold(b"ab")`` and ``fold(b"a", b"b")`` cannot
    collide with each other or with ``fold(b"a"), fold(b"b")``.
    """

    __slots__ = ("_h", "_n")

    #: Bumped if what is folded ever changes, so an old digest cannot be compared against a
    #: new one and read as a mismatch of *data*.
    DOMAIN: Final[bytes] = b"qd-consumed-prefix-v1"

    def __init__(self) -> None:
        self._h = hashlib.sha256(self.DOMAIN)
        self._n = 0

    def fold(self, *parts: bytes | bytearray | memoryview) -> None:
        """Add one consumed batch. Call exactly once per batch, in consumption order."""
        self._h.update(b"\x00" + len(parts).to_bytes(4, "big"))
        for part in parts:
            view = memoryview(part)
            self._h.update(view.nbytes.to_bytes(8, "big"))
            self._h.update(view)
        self._n += 1

    @property
    def n_folded(self) -> int:
        return self._n

    def hexdigest(self) -> str:
        return self._h.hexdigest()


#: What a run that has consumed nothing has to show for itself. A real value rather than
#: ``None``: "no batches yet" and "not recorded" must not be the same string, or the second
#: becomes a way to skip the check.
EMPTY_PREFIX_DIGEST: Final[str] = ConsumedPrefix().hexdigest()

_HEX64 = frozenset("0123456789abcdef")


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hex sha256 str, got {type(value).__name__}")
    if len(value) != 64 or not set(value) <= _HEX64:
        raise ValueError(f"{name} must be 64 lowercase hex characters, got {value!r}")
    return value


def _payload_digest(body: dict[str, Any]) -> str:
    """sha256 over a checkpoint's whole body, canonically. The same shape the ledger uses."""
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fsync_dir(directory: Path) -> None:
    """Make a rename durable. The rename is metadata; the directory holds it."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    """tmp in the same directory, fsync, rename, fsync the directory; debris removed on failure.

    One owner for the sequence, because a checkpoint is now two files and a sidecar written
    with a weaker write than the metadata beside it reintroduces exactly what the metadata's
    write was hardened against. ``Path.write_text`` truncates the target and then fills it,
    so a process killed part-way through left a file that was neither checkpoint -- measured
    on 2026-09-20, 84,731 bytes of a half-written successor over a 582-byte predecessor,
    with neither readable afterwards.
    """
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            written = 0
            while written < len(body):
                written += os.write(fd, body[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


# --- tensors: the half of a checkpoint that JSON could not hold ---------------------------
#
# Measured 2026-09-20, same content encoded both ways on the real rung-0 step with the
# AdamW moments present: 68.86 bytes per model parameter as JSON against 12.10 as a
# JSON+sidecar pair, a ratio of 5.7. The per-number costs behind that are 22.89 bytes of
# decimal text against 4.02 raw. Scaled to this program's 1,401,501,504 trainable
# parameters, about 97 GB per checkpoint against about 17 GB.
#
# So the bulk goes to a sidecar and the JSON keeps the bookkeeping. What the JSON keeps is
# the part that makes the pair safe: every tensor's digest, and a digest over the whole
# tensor set, both inside the body that `payload_digest` already covers.

#: safetensors' storage dtypes, by the name its Python API takes, with the bytes one element
#: occupies. Checked against ``safetensors`` 0.8.0 by round-tripping each name through
#: ``serialize``/``deserialize``. A dtype not in this table is **refused** rather than passed
#: to the serializer: an element size this module has not confirmed is an element size that
#: cannot be checked against ``len(data)``, and an unchecked length is a silently truncated
#: tensor.
_SIDECAR_DTYPES: Final[dict[str, int]] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
}

#: The names ``safetensors.deserialize`` gives back, mapped to the names its ``TensorSpec``
#: takes. The format's wire names are not its API's names, and a round trip that guessed
#: would revive a tensor under a dtype nobody wrote.
_SIDECAR_DTYPE_FROM_WIRE: Final[dict[str, str]] = {
    "BOOL": "bool",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "BF16": "bfloat16",
}

def _wire_dtype(name: str, wire: object) -> str:
    """The dtype ``safetensors.deserialize`` reports, as the name its ``TensorSpec`` takes."""
    if not isinstance(wire, str) or wire not in _SIDECAR_DTYPE_FROM_WIRE:
        raise ValueError(
            f"the sidecar stores {name!r} as {wire!r}, which is not a dtype this checkpoint "
            f"writes. Known: {', '.join(sorted(_SIDECAR_DTYPE_FROM_WIRE))}. Refusing rather "
            "than guessing an element size for somebody's weights."
        )
    return _SIDECAR_DTYPE_FROM_WIRE[wire]


#: How to find a non-finite element in a raw little-endian buffer without unpacking it:
#: ``(itemsize, hi_off, hi_mask, hi_val, lo_off, lo_mask, lo_val)``. An IEEE float is
#: non-finite exactly when every exponent bit is set, and the exponent never spans more than
#: two bytes, so one byte-slice plus a ``bytes.translate`` answers it at C speed for the
#: whole buffer. ``lo_off is None`` means the high byte decides alone.
#:
#: Derived per format rather than copied: f16 is 1-5-10, so its five exponent bits sit
#: entirely in the high byte at mask 0x7C; bf16 is 1-8-7 and f32 is 1-8-23, so seven
#: exponent bits are in the high byte (mask 0x7F) and the eighth is the top bit of the next
#: byte down; f64 is 1-11-52, so seven are in the high byte and four more are the top nibble
#: of the next. A dtype absent from this table has no non-finite value to find.
_NON_FINITE_PROBE: Final[dict[str, tuple[int, int, int, int, int | None, int, int]]] = {
    "float16": (2, 1, 0x7C, 0x7C, None, 0, 0),
    "bfloat16": (2, 1, 0x7F, 0x7F, 0, 0x80, 0x80),
    "float32": (4, 3, 0x7F, 0x7F, 2, 0x80, 0x80),
    "float64": (8, 7, 0x7F, 0x7F, 6, 0xF0, 0xF0),
}

#: Element-slice size for the scan above. It bounds the transient copies ``[hi::size]`` and
#: ``translate`` make: without it, probing a 2.80 GB payload allocates 2.80 GB more.
_NON_FINITE_CHUNK_ELEMS: Final[int] = 1 << 22

#: Domain separation for a tensor's digest, like [`ConsumedPrefix.DOMAIN`]. Bumped if what
#: is hashed ever changes, so an old digest cannot be compared against a new one and read as
#: corrupted data. Module-level rather than a class attribute because ``TensorRef`` is a
#: slotted dataclass and an annotated class attribute there would become a fourth field.
_TENSOR_REF_DOMAIN: Final[bytes] = b"qd-tensor-ref-v1"

#: Domain separation for the digest over a whole tensor set. A different constant from the
#: per-tensor one so a set of one cannot hash to its only member.
_SIDECAR_DOMAIN: Final[bytes] = b"qd-checkpoint-sidecar-v1"

#: The key under which a JSON body carries a tensor that lives in the sidecar. Explicit, in
#: the idiom ``byte_train`` already uses for its own tags: a checkpoint that guessed would
#: revive somebody's dict-of-four-keys as a tensor.
_TENSOR_REF_TAG: Final[str] = "__tensor_ref__"

#: What a sidecar file is called: the JSON checkpoint's stem, the first 16 hex characters of
#: the tensor set's digest, and ``.safetensors``. **Content-addressed on purpose** -- it is
#: what makes the pair atomic together.
#:
#: Writing metadata and tensors as two files has one failure mode that matters: a fresh
#: metadata file pairing with a stale sidecar, which reads as a valid checkpoint and is not.
#: A fixed sidecar name has exactly that shape -- rewriting a checkpoint replaces the
#: tensors first, and a kill before the metadata lands leaves the old metadata pointing at
#: the new tensors. That is the "neither checkpoint survived" defect the atomic write closed
#: for one file, reintroduced across two.
#:
#: With the digest in the name, different content is a different file, so the predecessor's
#: tensors are still there, still named by the predecessor's metadata, still readable. The
#: stem is included as well so that two checkpoint slots in one directory never share a
#: sidecar: sharing is safe to read and unsafe to retire, and retention is simpler when
#: each slot owns its own files.
_SIDECAR_SUFFIX: Final[str] = ".safetensors"
_SIDECAR_DIGEST_CHARS: Final[int] = 16

#: ``bytes.translate`` tables that map "this byte has the exponent bits the mask names" to
#: 0xFF and everything else to 0x00, so one ``in`` answers a whole chunk. Built once.
_FLAG_TABLES: Final[dict[tuple[int, int], bytes]] = {
    (mask, val): bytes(0xFF if (b & mask) == val else 0x00 for b in range(256))
    for mask, val in {(p[2], p[3]) for p in _NON_FINITE_PROBE.values()}
}


def _first_non_finite(data: bytes, dtype: str) -> int | None:
    """The first NaN or infinity in a raw buffer, by index, or ``None`` if all are finite.

    Exact, not sampled. ``_refuse_unwritable`` refuses a non-finite float in the JSON body
    -- partly because ``json.dumps`` writes it as a bare ``NaN``/``Infinity`` token that is
    not JSON, and partly because a non-finite parameter is a run that has already diverged.
    Only the first reason stops applying when the weights move to a sidecar, where the bit
    pattern round-trips exactly. Dropping the check along with the format would retire a
    refusal without saying so, so it is reimplemented here against the bytes.
    """
    probe = _NON_FINITE_PROBE.get(dtype)
    if probe is None:
        return None
    size, hi_off, hi_mask, hi_val, lo_off, lo_mask, lo_val = probe
    table = _FLAG_TABLES[(hi_mask, hi_val)]
    n_elems = len(data) // size
    for base in range(0, n_elems, _NON_FINITE_CHUNK_ELEMS):
        stop = min(base + _NON_FINITE_CHUNK_ELEMS, n_elems)
        chunk = data[base * size + hi_off : stop * size : size]
        flagged = chunk.translate(table)
        at = flagged.find(b"\xff")
        while at >= 0:
            index = base + at
            if lo_off is None or data[index * size + lo_off] & lo_mask == lo_val:
                return index
            at = flagged.find(b"\xff", at + 1)
    return None


@dataclass(frozen=True, slots=True)
class TensorRef:
    """One tensor, as raw little-endian bytes plus the dtype and shape that read them back.

    This is the contract change. ``TrainStep.state()`` was documented as returning
    "JSON-serialisable state for a checkpoint", and ``Checkpoint`` enforced that literally:
    a tensor had to become a list of Python floats to be stored at all, at 22.89 bytes per
    number against 4. A step now returns this instead for anything of size, and the bytes
    never pass through a decimal rendering in either direction.

    **Raw bytes rather than numbers on purpose.** This module imports no torch, so it cannot
    hold a tensor; and a round trip through ``float`` is not the identity on every bit
    pattern. Measured on torch 2.12.1 over the bf16 corner cases -- +0, -0, both smallest
    subnormals, both largest finite values, +-inf, a NaN with payload 0x41, 1.0: the raw-byte
    path returns all eleven bit-identical, and the ``tolist()`` path returns the nine finite
    ones bit-identical, refuses +-inf and NaN before writing them, and would quietly
    normalise the NaN payload 0x7FC1 to 0x7FC0 if it did not.

    ``data`` is little-endian because that is what the safetensors format specifies. This
    module refuses to run on a big-endian host rather than write native-endian bytes under a
    little-endian header.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes
    #: Filled on first use by [`digest`]. Not an input and not part of equality: it is a
    #: function of the other three fields, and a checkpoint boundary that hashed 2.80 GB
    #: three times -- once for the manifest, once for the set digest, once to write it --
    #: spends 0.9 s of a rented hour on the same answer. Measured on this host at
    #: 3.05 GB/s, so the second pass is not free and not noticeable either; it is removed
    #: because it is removable, not because it was the bottleneck.
    _digest: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str):
            raise TypeError(f"dtype must be str, got {type(self.dtype).__name__}")
        if self.dtype not in _SIDECAR_DTYPES:
            raise ValueError(
                f"{self.dtype!r} is not a dtype this checkpoint can store. Known: "
                f"{', '.join(sorted(_SIDECAR_DTYPES))}. Refusing rather than handing an "
                "unverified element size to the serializer."
            )
        if not isinstance(self.shape, tuple):
            raise TypeError(f"shape must be a tuple, got {type(self.shape).__name__}")
        for axis in self.shape:
            if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
                raise ValueError(f"shape must be non-negative ints, got {self.shape!r}")
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError(f"data must be bytes, got {type(self.data).__name__}")
        object.__setattr__(self, "data", bytes(self.data))
        elems = 1
        for axis in self.shape:
            elems *= axis
        expected = elems * _SIDECAR_DTYPES[self.dtype]
        if len(self.data) != expected:
            raise ValueError(
                f"a {self.dtype} tensor of shape {self.shape} is {expected:,} bytes and this "
                f"one carries {len(self.data):,}. A length that does not match the shape is a "
                "truncated tensor, and reviving it would silently reshape somebody's weights."
            )
        at = _first_non_finite(self.data, self.dtype)
        if at is not None:
            raise ValueError(
                f"element {at:,} of this {self.dtype}{list(self.shape)} tensor is not finite. "
                "A non-finite parameter is a run that has already diverged, and a checkpoint "
                "of it is an hour spent saving the divergence. Refusing here rather than at "
                "the resume, which is where it would otherwise be found."
            )

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def digest(self) -> str:
        """sha256 over the dtype, the shape and the bytes -- everything needed to revive it.

        Length-prefixed like [`ConsumedPrefix`], so a tensor's shape cannot be shifted into
        its data or its name into its dtype and still hash the same.
        """
        if self._digest is not None:
            return self._digest
        h = hashlib.sha256(_TENSOR_REF_DOMAIN)
        name = self.dtype.encode("ascii")
        h.update(len(name).to_bytes(4, "big"))
        h.update(name)
        h.update(len(self.shape).to_bytes(4, "big"))
        for axis in self.shape:
            h.update(axis.to_bytes(8, "big"))
        h.update(len(self.data).to_bytes(8, "big"))
        h.update(self.data)
        digest = h.hexdigest()
        object.__setattr__(self, "_digest", digest)
        return digest

    def to_manifest(self) -> dict[str, Any]:
        """What the JSON body records about this tensor. Never the data."""
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "digest": self.digest(),
        }


def _refuse_unwritable(value: Any, *, path: str = "model_state") -> None:
    """Refuse state a checkpoint could not write, where it is built rather than where it is used.

    ``Checkpoint`` is JSON on purpose -- this module is torch-free. Before this check, a
    step whose ``state()`` returned tensors built a ``Checkpoint`` happily and a
    ``to_json()`` happily, and only ``write()`` raised ``TypeError: Object of type Tensor
    is not JSON serializable`` (measured 2026-09-20 against ``Rung0Step`` on torch 2.12.1).
    That is the worst possible place to find out: the first time anything tries to persist
    a checkpoint is typically the first checkpoint boundary of a rented run, and the
    failure is identical at the last one.

    It is also the device check. A tensor carries the device it was trained on, so state
    that survives this is state that is already off the accelerator and back in plain
    Python -- a checkpoint written on a rented box can be read on a laptop. [`TensorRef`]
    does not weaken that: it holds raw little-endian bytes and names no device either, which
    is why it is accepted here and a live tensor still is not.
    """
    if isinstance(value, TensorRef):
        # Already validated in full by `TensorRef.__post_init__` -- dtype, shape-vs-length
        # and the non-finite scan. Re-scanning here would walk the whole payload a second
        # time for the same answer, and the bound this module cares about is not free.
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: JSON object keys must be str, got {type(key).__name__} "
                    f"({key!r}). A checkpoint whose keys change type on the way through a "
                    "file does not round-trip."
                )
            _refuse_unwritable(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _refuse_unwritable(item, path=f"{path}[{i}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{path} is {value!r}, which JSON cannot represent: json.dumps writes it as "
                "the bare token NaN/Infinity, which is not JSON and which a strict reader "
                "refuses. A non-finite parameter is also a run that has already diverged."
            )
        return
    raise TypeError(
        f"{path} is a {type(value).__name__}, which this checkpoint cannot write: "
        "qd_train.run_control imports no torch, so a checkpoint holds JSON values and "
        "TensorRef and nothing else. Convert the step's state off the accelerator -- for a "
        "tensor, `TensorRef(dtype=..., shape=tuple(t.shape), data=<its raw little-endian "
        "bytes>)`, which goes to the safetensors sidecar rather than into this body. A "
        "checkpoint holding device tensors cannot be written at all, and would be readable "
        "only on the machine that wrote it if it could."
    )


def _split_tensors(value: Any, *, path: str = "model_state") -> tuple[Any, dict[str, TensorRef]]:
    """Separate a state tree into a JSON body and the tensors the sidecar will hold.

    Each tensor's key is the path it sits at in the tree, in the same notation
    [`_refuse_unwritable`] names offenders with, so a digest mismatch on read points at the
    same place a type refusal on write would have. The key is injective over the tree: two
    distinct positions cannot produce one string, because every step appends a bracketed
    ``repr`` of a key or an index.
    """
    if isinstance(value, TensorRef):
        return {_TENSOR_REF_TAG: {"key": path, **value.to_manifest()}}, {path: value}
    if isinstance(value, dict):
        body: dict[str, Any] = {}
        found: dict[str, TensorRef] = {}
        for key, item in value.items():
            sub, subfound = _split_tensors(item, path=f"{path}[{key!r}]")
            body[key] = sub
            found.update(subfound)
        return body, found
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        found = {}
        for i, item in enumerate(value):
            sub, subfound = _split_tensors(item, path=f"{path}[{i}]")
            items.append(sub)
            found.update(subfound)
        return items, found
    return value, {}


def _join_tensors(value: Any, tensors: Mapping[str, TensorRef]) -> Any:
    """The inverse of [`_split_tensors`], checking every tensor against what the body claims.

    The manifest in the body and the bytes in the sidecar are two independent statements,
    and this is where they are made to agree. A sidecar holding the right *number* of
    tensors under the right *names* but the wrong content is the failure that reads as a
    successful resume, so the per-tensor digest is compared, not just the shape.
    """
    if isinstance(value, dict):
        entry = value.get(_TENSOR_REF_TAG)
        if entry is not None and len(value) == 1:
            key = entry["key"]
            ref = tensors.get(key)
            if ref is None:
                raise ValueError(
                    f"the checkpoint body names a tensor at {key} and the sidecar does not "
                    f"carry it. The sidecar holds {sorted(tensors)!r}. A checkpoint missing "
                    "one of its tensors is not a checkpoint with one fewer."
                )
            if (
                ref.dtype != entry["dtype"]
                or list(ref.shape) != list(entry["shape"])
                or ref.nbytes != entry["nbytes"]
            ):
                raise ValueError(
                    f"the tensor at {key} is {ref.dtype}{list(ref.shape)} "
                    f"({ref.nbytes:,} bytes) in the sidecar and "
                    f"{entry['dtype']}{list(entry['shape'])} ({entry['nbytes']:,} bytes) in "
                    "the checkpoint body: the two files describe different tensors"
                )
            actual = ref.digest()
            if actual != entry["digest"]:
                raise ValueError(
                    f"the tensor at {key} hashes to {actual!r} and the checkpoint body "
                    f"records {entry['digest']!r}. The sidecar does not hold the weights "
                    "this checkpoint was written from -- refusing rather than resuming into "
                    "parameters nothing vouches for."
                )
            return ref
        return {k: _join_tensors(v, tensors) for k, v in value.items()}
    if isinstance(value, list):
        return [_join_tensors(v, tensors) for v in value]
    return value


def _sidecar_digest(tensors: Mapping[str, TensorRef]) -> str:
    """One digest over a whole tensor set: its names, in sorted order, and each one's digest.

    Over the per-tensor digests rather than the bytes again, so the payload is hashed once
    per checkpoint rather than twice -- at the 2.7-3.0 GB/s this host measures, a second
    pass over 2.80 GB is a second of every checkpoint boundary for no new information.
    """
    h = hashlib.sha256(_SIDECAR_DOMAIN)
    h.update(len(tensors).to_bytes(8, "big"))
    for key in sorted(tensors):
        name = key.encode("utf-8")
        h.update(len(name).to_bytes(8, "big"))
        h.update(name)
        h.update(bytes.fromhex(tensors[key].digest()))
    return h.hexdigest()


def _require_little_endian(what: str) -> None:
    """safetensors stores little-endian; torch's raw buffer is native-endian.

    On this host (arm64 macOS) they are the same and the sidecar is exact. On a big-endian
    host they are not, and writing native bytes under a little-endian header would produce a
    file that reads back byte-swapped on every other machine -- silently, because nothing in
    the format records which end it came from. Refusing is the only honest option: this has
    never been run on such a host, so it is **not run**, not "probably fine".
    """
    if sys.byteorder != "little":
        raise RuntimeError(
            f"cannot {what}: the safetensors format stores tensors little-endian and this "
            f"host is {sys.byteorder}-endian. Nothing in this repository has been run on a "
            "big-endian host, so the byte-swap this would need is unwritten and untested. "
            "Refusing rather than producing a file that reads back swapped elsewhere."
        )


def _safetensors_module() -> Any:
    """Import safetensors, or say exactly why a checkpoint with tensors cannot be written.

    Imported here rather than at module scope because ``run_control`` is the torch-free
    core: it is imported by the suite that runs on a machine with neither torch nor
    safetensors, and a checkpoint whose ``model_state`` holds no tensor never reaches this.
    ``safetensors==0.8.0`` is already in ``stack/train.lock``; this adds no dependency.
    """
    try:
        import safetensors
    except ImportError as exc:  # pragma: no cover - exercised by the torch-free suite's skip
        raise RuntimeError(
            "this checkpoint carries tensors and needs `safetensors`, which is not "
            "importable here. It is pinned at 0.8.0 in stack/train.lock and present in the "
            "training environment; the torch-free venv deliberately has neither it nor "
            "torch. A checkpoint whose model_state holds no TensorRef needs no sidecar and "
            "does not reach this."
        ) from exc
    return safetensors


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything a resume needs that it cannot recompute.

    The data position is two integers because the batch order is a pure function of
    ``(seed, epoch, batch_tokens)`` and the shard set -- so the iterator is *rebuilt*, not
    restored. ``seed`` and ``schedule`` are carried so a resume against a differently-seeded
    or differently-scheduled run is refused rather than silently training on a different
    order; ``consumed_digest`` is carried because the seed is not the whole of that order,
    and it is the only field that can be checked against the source a resume is handed
    rather than against what the resuming caller says about it.

    ``model_state`` is opaque here on purpose: this module does not import torch, and the
    step implementation is the only thing that knows what its own state means. Opaque is
    not unchecked -- see [`_refuse_unwritable`]. What is **not** here, and what a step must
    therefore put in ``model_state`` itself if its forward pass draws from an RNG: the
    framework RNG state. ``ByteDeciderConfig.dropout`` is configurable and
    ``Rung0Step.accumulate`` calls ``model.train()``, so a run with dropout above zero
    resumes into a different random stream unless its ``state()`` says otherwise.

    ## On disk: one file, or two

    ``model_state`` may hold [`TensorRef`] values. Those do **not** go in the JSON -- they
    go in a safetensors sidecar beside it, and the JSON keeps each one's dtype, shape, byte
    count and sha256, plus one digest over the whole set. A checkpoint with no ``TensorRef``
    writes exactly one file, as it always did.

    Two files are two chances to have half a checkpoint, so they are bound together three
    ways and none of them is the filesystem:

    * **the sidecar is written first and the JSON last.** Until the JSON lands, nothing
      names the sidecar and it is debris; once the JSON lands, both are already fsynced.
      The JSON is the commit record, exactly as the directory fsync is for one file.
    * **the sidecar's filename contains the tensor set's digest**, so writing a new
      checkpoint over an old one at the same path cannot overwrite the old tensors. The
      predecessor stays readable through an interrupted write of its successor -- the same
      property the single-file atomic write has, extended across the pair.
    * **the JSON records every digest and ``payload_digest`` covers the JSON.** A sidecar
      that is stale, truncated, swapped or edited fails a comparison rather than resuming.

    A uuid was considered for the pairing and rejected: it would distinguish two checkpoints
    whose tensors are byte-identical, which is the one case where sharing the bytes is both
    correct and free, and it would add an identifier that says nothing about whether the
    file's *contents* are the ones the metadata was written from. The digest answers the
    question that matters; a uuid answers a question nobody asked.
    """

    position: Position
    optimizer_step: int
    seed: int
    schedule: LRSchedule
    loss_log: LossLog
    consumed_digest: str
    model_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer_step, int) or isinstance(self.optimizer_step, bool):
            raise TypeError(
                f"optimizer_step must be int, got {type(self.optimizer_step).__name__}"
            )
        if self.optimizer_step < 0:
            raise ValueError(f"optimizer_step must be non-negative, got {self.optimizer_step}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"seed must be int, got {type(self.seed).__name__}")
        if self.optimizer_step > self.schedule.total_steps:
            raise ValueError(
                f"optimizer_step {self.optimizer_step} is past the schedule's "
                f"{self.schedule.total_steps} steps: the checkpoint and the schedule describe "
                "different runs"
            )
        last = self.loss_log.last
        if last is not None and last.optimizer_step >= self.optimizer_step:
            raise ValueError(
                f"the loss log's last step is {last.optimizer_step} but the checkpoint says the "
                f"next step is {self.optimizer_step}: resuming here would replay a logged step"
            )
        _require_digest("consumed_digest", self.consumed_digest)
        # The position and the evidence are two statements of one fact, so they are checked
        # against each other. `ConsumedPrefix` folds at least one part per batch, so no
        # non-empty prefix can hash to the empty one.
        if (self.position.index == 0) != (self.consumed_digest == EMPTY_PREFIX_DIGEST):
            raise ValueError(
                f"position.index is {self.position.index} but consumed_digest "
                f"{'is' if self.consumed_digest == EMPTY_PREFIX_DIGEST else 'is not'} the "
                "empty-prefix digest: the checkpoint's position and its record of what was "
                "consumed describe different runs"
            )
        _refuse_unwritable(self.model_state)

    def tensors(self) -> dict[str, TensorRef]:
        """Every [`TensorRef`] in ``model_state``, keyed by where it sits in the tree."""
        return _split_tensors(self.model_state)[1]

    def to_json(self) -> dict[str, Any]:
        state, tensors = _split_tensors(self.model_state)
        body = {
            "position": self.position.to_json(),
            "optimizer_step": self.optimizer_step,
            "seed": self.seed,
            "schedule": self.schedule.to_json(),
            "loss_log": self.loss_log.to_json(),
            "loss_digest": self.loss_log.digest(),
            "consumed_digest": self.consumed_digest,
            # The sidecar's own summary, inside the body so `payload_digest` covers it.
            # `None` rather than an empty record when there are no tensors, so "this
            # checkpoint has no sidecar" and "this checkpoint's sidecar is empty" stay
            # different statements -- the second would be a sidecar that failed to write.
            "sidecar": (
                None
                if not tensors
                else {
                    "digest": _sidecar_digest(tensors),
                    "n_tensors": len(tensors),
                    "nbytes": sum(t.nbytes for t in tensors.values()),
                    "format": "safetensors",
                }
            ),
            "model_state": state,
        }
        # Over the **whole** body, not just the loss log. `loss_digest` covered the
        # trajectory and left the weights, the position and the schedule unprotected, so a
        # checkpoint whose model_state had been altered -- by an editor, a half-finished
        # write on a filesystem that reorders, a bad disk -- loaded silently and resumed
        # into parameters nobody trained. With the weights in a sidecar it covers their
        # digests instead, which is the same guarantee reaching one file further.
        return {**body, "payload_digest": _payload_digest(body)}

    @staticmethod
    def sidecar_path(path: str | Path, sidecar_digest: str) -> Path:
        """Where the tensors for the checkpoint at ``path`` live. A pure function of both.

        Derived rather than stored so that the name cannot disagree with the digest beside
        it, and so a checkpoint carries no absolute path that breaks when the directory
        moves. Moving a checkpoint means moving both files; the read refuses by name if you
        move only one.
        """
        p = Path(path)
        short = _require_digest("sidecar_digest", sidecar_digest)[:_SIDECAR_DIGEST_CHARS]
        return p.with_name(f"{p.stem}.{short}{_SIDECAR_SUFFIX}")

    @classmethod
    def from_json(
        cls, raw: dict[str, Any], *, tensors: Mapping[str, TensorRef] | None = None
    ) -> Self:
        """Rebuild a checkpoint from its JSON body and, when it has one, its sidecar's tensors.

        ``tensors`` is a keyword because it is not optional in the sense "leave it out and
        get less" -- a body that declares a sidecar and is handed none is **refused**. An
        absent sidecar is not an empty one, the same rule an absent digest follows.
        """
        stated = raw.get("payload_digest")
        if stated is None:
            raise ValueError(
                "checkpoint has no payload_digest. Every checkpoint this repo writes has "
                "one, so a file without it was either truncated, written by something else, "
                "or edited -- and an absent checksum is not a passed checksum. Refusing "
                "rather than loading weights nothing vouches for."
            )
        body = {k: v for k, v in raw.items() if k != "payload_digest"}
        actual = _payload_digest(body)
        if stated != actual:
            raise ValueError(
                f"checkpoint payload_digest is {stated!r} but the body hashes to {actual!r}; "
                "the checkpoint was modified after it was written, and a resume from it "
                "cannot claim to reproduce the trajectory"
            )
        log = LossLog.from_json(raw["loss_log"])
        # Kept as well as the payload digest: it names *which* part disagrees, and it is
        # what `test_a_tampered_checkpoint_is_refused_on_read` has always asserted.
        loss_stated = raw.get("loss_digest")
        if loss_stated is None:
            raise ValueError(
                "checkpoint has no loss_digest; an absent digest is not a verified one"
            )
        if loss_stated != log.digest():
            raise ValueError(
                f"checkpoint loss_digest is {loss_stated!r} but the log hashes to "
                f"{log.digest()!r}; the checkpoint was modified after it was written, and a "
                "resume from it cannot claim to reproduce the trajectory"
            )
        # The sidecar, before anything is rebuilt from it. `sidecar` is absent only in a
        # body this repo did not write, and that is a refusal for the same reason a missing
        # `payload_digest` is: a field that can be deleted to skip a check is not a check.
        if "sidecar" not in raw:
            raise ValueError(
                "checkpoint has no `sidecar` field, not even a null one. Every checkpoint "
                "this repo writes records whether it has a tensor sidecar; a body without "
                "the field was written by something else or edited, and 'no field' must not "
                "be read as 'no sidecar'."
            )
        declared = raw["sidecar"]
        supplied = dict(tensors) if tensors is not None else None
        if declared is None:
            if supplied:
                raise ValueError(
                    f"this checkpoint declares no sidecar and {len(supplied)} tensor(s) were "
                    "supplied for it. A sidecar nobody wrote is a sidecar from somewhere "
                    "else."
                )
            supplied = {}
        else:
            if supplied is None:
                raise ValueError(
                    f"this checkpoint's {declared['n_tensors']} tensor(s) live in a "
                    f"{declared['format']} sidecar and none was supplied. Use "
                    "`Checkpoint.read`, which resolves the sidecar beside the JSON; an "
                    "absent sidecar is not an empty one."
                )
            actual = _sidecar_digest(supplied)
            if actual != declared["digest"]:
                raise ValueError(
                    f"the sidecar hashes to {actual!r} and this checkpoint records "
                    f"{declared['digest']!r}. The two files are not a matched pair -- the "
                    "sidecar is stale, truncated or from another checkpoint, and resuming "
                    "would train on weights the metadata never described."
                )
            if len(supplied) != declared["n_tensors"]:
                raise ValueError(
                    f"the sidecar holds {len(supplied)} tensor(s) and this checkpoint "
                    f"records {declared['n_tensors']}"
                )
        return cls(
            position=Position.from_json(raw["position"]),
            optimizer_step=int(raw["optimizer_step"]),
            seed=int(raw["seed"]),
            schedule=LRSchedule.from_json(raw["schedule"]),
            loss_log=log,
            consumed_digest=_require_digest("consumed_digest", raw["consumed_digest"]),
            model_state=_join_tensors(dict(raw.get("model_state", {})), supplied),
        )

    def write(self, path: str | Path) -> Path:
        """Write so that an interrupted write cannot destroy the checkpoint already there.

        ``Path.write_text`` truncates the target and then writes it, so a process killed
        part-way through left a file that was neither the old checkpoint nor the new one --
        on a rented box that is the hour you were paying to protect, deleted by the act of
        protecting it. Same lesson, same fix as ``qd_train.ledger``: the bytes reach the
        disk before anything points at them.

        * a temporary file in the target's own directory, so ``replace`` is a rename within
          one filesystem and therefore atomic;
        * ``fsync`` on it before the rename, because a rename that wins the race to the
          directory entry while the data is still in the page cache gives you an intact
          pointer to nothing after a power cut or an instance kill;
        * ``fsync`` on the directory after it, because the rename itself is metadata and is
          not durable until the directory is;
        * the temporary file removed on any failure, so a full disk leaves the previous
          checkpoint and no debris.

        A ``SIGKILL`` mid-write leaves a ``.tmp-<pid>`` beside the checkpoint. That is the
        intended outcome: harmless, visibly incomplete, and not the file anything reads.

        **With a sidecar, the same property has to hold across two files.** The order is
        tensors, then metadata, and it is the order that makes the pair atomic: the JSON is
        the only thing that names a sidecar, so until the JSON lands the new sidecar is an
        unreferenced file and the *previous* checkpoint -- previous JSON, previous sidecar,
        both still present because the sidecar's name carries its own digest -- is intact
        and readable. A kill anywhere in this method leaves a whole checkpoint or the whole
        previous one, never a metadata file whose tensors are from somewhere else.

        Once the new JSON is durable, the sidecar that the **previous JSON at this exact
        path** named is removed, if it is a different file. That is the one deletion here,
        it happens strictly after the commit, and it is derived from the predecessor's own
        record rather than from a glob -- so a crash during it leaves a stale sidecar, which
        is wasted disk, and never a live checkpoint missing its tensors.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = self.to_json()
        tensors = self.tensors()
        declared = raw["sidecar"]
        body = json.dumps(raw, sort_keys=True).encode("utf-8")
        if len(body) > MAX_CHECKPOINT_BYTES:
            raise ValueError(
                f"this checkpoint's JSON body is {len(body):,} bytes, over "
                f"MAX_CHECKPOINT_BYTES ({MAX_CHECKPOINT_BYTES:,}). The weights are not in "
                "it -- a TensorRef costs one manifest entry here and its bytes go to the "
                "sidecar -- so a body this large is a loss log or a state tree that has "
                "grown without bound, not a big model. Refusing rather than writing it: a "
                "checkpoint that takes longer than the interval between checkpoints is not "
                "a checkpoint."
            )
        if declared is not None:
            if len(tensors) > MAX_SIDECAR_TENSORS:
                raise ValueError(
                    f"this checkpoint carries {len(tensors):,} tensors, over "
                    f"MAX_SIDECAR_TENSORS ({MAX_SIDECAR_TENSORS:,})"
                )
            if declared["nbytes"] > MAX_SIDECAR_BYTES:
                raise ValueError(
                    f"this checkpoint's tensors come to {declared['nbytes']:,} bytes, over "
                    f"MAX_SIDECAR_BYTES ({MAX_SIDECAR_BYTES:,}). That is larger than the "
                    "full Qwen3.5-2B text tower with fp32 weights and two fp32 optimizer "
                    "moments (22,581,901,056 bytes), so it is a bug in the step's state(), "
                    "not a recipe. Refusing rather than filling a rented disk."
                )
            _require_little_endian("write a checkpoint sidecar")

        # What the previous checkpoint at this path pointed at, read before anything is
        # replaced. Unreadable or absent means there is nothing to retire, which is the
        # safe answer: leaving a file costs disk, deleting the wrong one costs an hour.
        stale = self._recorded_sidecar(p)

        sidecar = None if declared is None else self.sidecar_path(p, declared["digest"])
        if sidecar is not None:
            self._write_sidecar(sidecar, tensors)
        _atomic_write_bytes(p, body)
        if stale is not None and stale != sidecar and stale.parent == p.parent:
            stale.unlink(missing_ok=True)
        return p

    @staticmethod
    def _recorded_sidecar(path: Path) -> Path | None:
        """The sidecar named by whatever checkpoint is at ``path`` now, or ``None``."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            declared = raw["sidecar"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(declared, dict) or not isinstance(declared.get("digest"), str):
            return None
        try:
            return Checkpoint.sidecar_path(path, declared["digest"])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _write_sidecar(sidecar: Path, tensors: Mapping[str, TensorRef]) -> None:
        """The tensors, atomically, through the same tmp/fsync/rename/dir-fsync as the JSON.

        ``safetensors.serialize_file`` writes the container straight to a path rather than
        building it in memory first, which for a 2.80 GB payload is 2.80 GB not allocated
        twice. It does not fsync, so that is done here on a re-opened descriptor before the
        rename -- a rename that beats its own data to the disk is the defect the single-file
        write already fixed once.

        ``TensorSpec`` takes a raw address, so every buffer is held in ``pinned`` for the
        duration of the call: the library's own documented requirement, and a use-after-free
        if it is not met.
        """
        safetensors = _safetensors_module()
        pinned = [tensors[key].data for key in sorted(tensors)]
        specs = {
            key: safetensors.TensorSpec(
                dtype=tensors[key].dtype,
                shape=list(tensors[key].shape),
                data_ptr=ctypes.cast(
                    ctypes.cast(buf, ctypes.c_char_p), ctypes.c_void_p
                ).value,
                data_len=len(buf),
            )
            for key, buf in zip(sorted(tensors), pinned, strict=True)
        }
        tmp = sidecar.with_name(f"{sidecar.name}.tmp-{os.getpid()}")
        try:
            safetensors.serialize_file(specs, str(tmp))
            fd = os.open(tmp, os.O_WRONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            tmp.replace(sidecar)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            del pinned, specs
        _fsync_dir(sidecar.parent)

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """The checkpoint at ``path``, with its sidecar if it has one.

        The sidecar is resolved beside the JSON by [`sidecar_path`], never by a path stored
        in the file: a checkpoint that carried an absolute path would break the moment the
        directory moved, and a checkpoint that carried a relative one would be a way to
        point a read at a file outside the directory it was handed.
        """
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        declared = raw.get("sidecar") if isinstance(raw, dict) else None
        if declared is None:
            return cls.from_json(raw)
        _require_little_endian("read a checkpoint sidecar")
        sidecar = cls.sidecar_path(p, declared["digest"])
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{p} declares {declared['n_tensors']} tensor(s) in {sidecar.name} and that "
                f"file is not in {p.parent}. The name carries the tensors' digest, so this "
                "is a checkpoint whose sidecar was moved, deleted or never finished being "
                "written -- not one that can be resumed with the weights it has."
            )
        safetensors = _safetensors_module()
        loaded = {
            name: TensorRef(
                dtype=_wire_dtype(name, info["dtype"]),
                shape=tuple(int(axis) for axis in info["shape"]),
                data=bytes(info["data"]),
            )
            for name, info in safetensors.deserialize(sidecar.read_bytes())
        }
        return cls.from_json(raw, tensors=loaded)


# --- the three of them, driving one loop --------------------------------------------------


@dataclass(slots=True)
class RunControl:
    """The schedule, the cap and the price, plus the clock they are measured against.

    ``clock`` is injected so the cap is testable in milliseconds rather than in hours. It
    defaults to ``time.monotonic`` because a wall-clock cap measured against a clock that
    can step backwards is not a cap -- and because ``clock`` is injectable, the default is
    an argument rather than a guarantee, so every reading it returns is checked. It was not:
    a clock that stepped back 109 s took ``elapsed_s()`` from 9.0 to 0.0 and ``cost_so_far()``
    from $0.0107 to $0.0000, and the ``max(0.0, ...)`` that used to sit here is what made that
    silent rather than loud.

    ``auto_terminate`` is rule 4's fourth requirement. The rule names four things -- *"a
    wall-clock cap, auto-terminate and a cost estimate, and needs a human yes"* -- and this
    class enforced three of them: the cap is mandatory and bounded by [`MAX_CAP_S`], the
    estimate is mandatory and priced from the cap, and a job over the threshold is refused
    without ``approved_by``. Auto-terminate was not enforced anywhere. ``expired()`` is a
    *poll*: it answers when a cooperative loop asks, at an optimizer-step boundary. A step
    that blocks -- a wedged dataloader, a CUDA call that never returns, a loop that simply
    forgets to ask -- runs past the cap until a human notices, and the meter runs with it.
    So for exactly the runs rule 4 names, a terminate action is mandatory; [`hard_exit_on_cap`]
    is the ready-made one, and ``start()`` arms a watchdog that fires without the loop's help.
    """

    schedule: LRSchedule
    cap: WallClockCap
    cost: CostEstimate
    grad_accum: int = 1
    checkpoint_every: int = 0
    approved_by: str = ""
    clock: Callable[[], float] = time.monotonic
    auto_terminate: Callable[[str], None] | None = None
    _t0: float | None = field(default=None, init=False, repr=False)
    _high_water_s: float = field(default=0.0, init=False, repr=False)
    _disarm: threading.Event | None = field(default=None, init=False, repr=False)
    _fired: bool = field(default=False, init=False, repr=False)
    # The watchdog thread and the loop both read the clock. Without this, two readings that
    # are both perfectly monotonic can interleave so that the older one is compared against
    # the newer one's high-water mark, and the regression check fires on a healthy clock.
    _clock_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grad_accum, int) or isinstance(self.grad_accum, bool):
            raise TypeError(f"grad_accum must be int, got {type(self.grad_accum).__name__}")
        if not 1 <= self.grad_accum <= MAX_GRAD_ACCUM:
            raise ValueError(
                f"grad_accum must be in [1, {MAX_GRAD_ACCUM}], got {self.grad_accum}"
            )
        if not isinstance(self.checkpoint_every, int) or isinstance(self.checkpoint_every, bool):
            raise TypeError(
                f"checkpoint_every must be int, got {type(self.checkpoint_every).__name__}"
            )
        if self.checkpoint_every < 0:
            raise ValueError(f"checkpoint_every must be non-negative, got {self.checkpoint_every}")
        if self.cost.cap != self.cap:
            raise ValueError(
                f"the cost estimate prices a {self.cost.cap.cap_s}s cap but this run's cap is "
                f"{self.cap.cap_s}s. DESIGN-4: the estimate is the cap priced; two different "
                "caps means the number on the launch line is not this run's worst case."
            )
        if self.cost.requires_human_approval and not self.approved_by.strip():
            raise LaunchRefused(
                "rule 4: this run needs a human yes before launch and has none. "
                f"{self.cost.approval_line()}. Pass approved_by='<who said yes>' once a human "
                "has. An agent cannot supply this on a human's behalf."
            )
        if self.cost.requires_human_approval and self.auto_terminate is None:
            raise LaunchRefused(
                "rule 4: this run needs auto-terminate and has none. "
                f"{self.cost.approval_line()}. expired() is a poll a cooperative loop makes at "
                "optimizer-step boundaries; a step that blocks never reaches one, and the cap "
                "then bounds nothing. Pass auto_terminate=hard_exit_on_cap (or your own "
                "terminate action) so start() can arm a watchdog that does not need the loop."
            )
        if self.auto_terminate is not None and not callable(self.auto_terminate):
            raise TypeError(
                f"auto_terminate must be callable, got {type(self.auto_terminate).__name__}"
            )

    # -- the clock -------------------------------------------------------

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("RunControl.start() called twice; one control drives one run")
        self._t0 = _require_finite("clock() reading at start", self.clock())
        self._high_water_s = 0.0
        if self.auto_terminate is not None:
            self._disarm = threading.Event()
            watchdog = threading.Thread(
                target=self._watch, name="qd-wall-clock-cap", daemon=True
            )
            watchdog.start()

    def stop(self) -> None:
        """Disarm the watchdog. Idempotent, and safe on a control that never armed one.

        A finished run must not leave a thread polling a cap it has already respected. The
        thread is a daemon as well, so forgetting this cannot wedge an exit -- but forgetting
        it on a long-lived process would leave the previous run's cap live over the next one.
        """
        if self._disarm is not None:
            self._disarm.set()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def elapsed_s(self) -> float:
        if self._t0 is None:
            raise RuntimeError(
                "RunControl.elapsed_s() before start(): an unstarted run has no elapsed time, "
                "and returning 0.0 would read as a run that has only just begun"
            )
        with self._clock_lock:
            now = _require_finite("clock() reading", self.clock())
            elapsed = now - self._t0
            if elapsed < self._high_water_s:
                raise RuntimeError(
                    f"the clock went backwards: {elapsed!r}s elapsed now against "
                    f"{self._high_water_s!r}s already observed. time.monotonic does not do "
                    "this; time.time does, under NTP and under a DST transition, and an "
                    "injected clock does whatever it does. Clamping the difference away -- "
                    "which is what this method used to do -- extends the cap by the size of "
                    "the step back and lowers the cost the ledger records, both silently. "
                    "Fix the clock; the cap cannot be trusted until you do."
                )
            self._high_water_s = elapsed
            return elapsed

    def expired(self) -> bool:
        """Has the cap been reached? Polled at optimizer-step boundaries by the loop.

        This is the *cooperative* half of the cap. The half that does not need the loop is
        the watchdog ``start()`` arms; see ``auto_terminate``.
        """
        return self.cap.expired(self.elapsed_s())

    def cost_so_far(self) -> float:
        return self.cost.cost_for(self.elapsed_s())

    # -- auto-terminate: the half of the cap that does not need the loop --

    def _watch(self) -> None:
        """Poll the cap on a thread, and fire once. Runs only when ``auto_terminate`` is set.

        Measured against the **injected** clock, not against real time, so a test driving a
        fake clock arms a watchdog that fires exactly when that clock says the cap is reached
        and never on the wall. The poll interval is real time, and is a bound on the lag
        between expiry and the terminate action -- never a second cap.
        """
        disarm = self._disarm
        if disarm is None:  # pragma: no cover - start() always sets it before the thread runs
            return
        interval = min(_WATCHDOG_POLL_S, max(_MIN_WATCHDOG_POLL_S, self.cap.cap_s / 10.0))
        while not disarm.wait(interval):
            try:
                over = self.expired()
            except Exception as exc:  # a blind except is the fail-closed choice here
                # A watchdog that cannot read the clock is a watchdog that has stopped
                # watching. Reporting nothing here is how "the cap held" comes to mean "the
                # cap was never evaluated", so this terminates instead.
                self._fire(f"the cap could not be evaluated: {type(exc).__name__}: {exc}")
                return
            if over:
                self._fire(
                    f"{self.cap.cap_s:.0f}s cap reached with the loop still running. "
                    f"{self.cost.approval_line()}"
                )
                return

    def _fire(self, reason: str) -> None:
        if self._fired or self.auto_terminate is None:
            return
        self._fired = True
        self.auto_terminate(reason)

    @property
    def terminated_for_cap(self) -> bool:
        """True once the watchdog has fired. For a caller that survives its terminate action."""
        return self._fired

    # -- the schedule ----------------------------------------------------

    def lr_at(self, step: int) -> float:
        return self.schedule.lr_at(step)

    @property
    def total_steps(self) -> int:
        return self.schedule.total_steps

    @property
    def max_micro_batches(self) -> int:
        """The most batches a full run can consume. Bounds an unbounded source."""
        return self.schedule.total_steps * self.grad_accum

    def should_checkpoint(self, completed_steps: int) -> bool:
        """After ``completed_steps`` optimizer steps, is this a checkpoint boundary?

        ``checkpoint_every == 0`` means "only at the end", which the loop handles itself --
        it always writes a final checkpoint, so this returning False never means "no
        checkpoint was taken".
        """
        if completed_steps <= 0 or self.checkpoint_every == 0:
            return False
        return completed_steps % self.checkpoint_every == 0


class AccumulationGroup:
    """One optimizer step's worth of micro-batches, and the four rules for closing it.

    Gradient accumulation looks like bookkeeping and is actually policy. Four rules decide
    whether a run's loss curve means anything, and every one of them fails *quietly* when
    it is got wrong:

    1. **A partial group is never applied.** A step taken on fewer micro-batches than
       ``grad_accum`` is a step at a different effective batch size, and the loss log records
       it as though it were the same. The source ending mid-group is a contract violation,
       not a rounding error.
    2. **The cap is checked at a group boundary only.** Checking mid-group discards a
       half-accumulated gradient -- work done and thrown away, with nothing saying so.
       Overshoot is bounded by one optimizer step, which is the cheaper error.
    3. **A non-finite loss stops the run.** Continuing trains on NaN gradients and produces
       a loss curve that simply stops meaning anything, while still printing numbers.
    4. **The group's loss is its mean**, over exactly the micro-batches in it.

    This class exists because two loops need these four rules and they are not the kind of
    thing that survives being written twice: ``qd_train.byte_train`` re-implemented them and
    got 1, 2 and 3 wrong. It owns the *rules and the messages*; the caller supplies the
    exception **type** it raises for a contract violation, because "what my loop calls a
    broken contract" is domain vocabulary and ``TrainerContractViolation`` is already pinned
    by ``test_trainer.py``.

    ``extras`` ride along under the same discipline: a rung-0 batch carries a choice loss
    and a span loss alongside the total that is actually differentiated, and averaging those
    outside the group would put them on a different denominator than the number beside them.
    """

    __slots__ = ("_control", "_extras", "_losses", "_violation")

    def __init__(self, control: RunControl, *, violation: type[Exception] = ValueError) -> None:
        if not isinstance(control, RunControl):
            raise TypeError(f"control must be a RunControl, got {type(control).__name__}")
        if not (isinstance(violation, type) and issubclass(violation, Exception)):
            raise TypeError("violation must be an exception class")
        self._control = control
        self._violation = violation
        self._losses: list[float] = []
        self._extras: dict[str, list[float]] = {}

    @property
    def grad_accum(self) -> int:
        return self._control.grad_accum

    @property
    def size(self) -> int:
        """Micro-batches accumulated so far in the open group."""
        return len(self._losses)

    @property
    def open(self) -> bool:
        """True when a partial group is in progress. Rule 1 and rule 2 both turn on this."""
        return bool(self._losses)

    @property
    def ready(self) -> bool:
        return len(self._losses) >= self._control.grad_accum

    def add(self, loss: float, *, where: str, **extras: float) -> None:
        """Record one micro-batch's loss. Rule 3: a non-finite loss stops the run here."""
        value = float(loss)
        if not math.isfinite(value):
            raise self._violation(
                f"step returned a non-finite loss {value!r} at {where}. A run that keeps "
                "going after this trains on NaN gradients and reports a loss curve that "
                "simply stops meaning anything."
            )
        for name, extra in extras.items():
            extra_value = float(extra)
            if not math.isfinite(extra_value):
                raise self._violation(
                    f"step returned a non-finite {name} loss {extra_value!r} at {where}"
                )
            self._extras.setdefault(name, []).append(extra_value)
        self._losses.append(value)

    def should_stop_for_cap(self) -> bool:
        """Rule 2: the cap may only stop the loop when no group is half-accumulated."""
        return not self.open and self._control.expired()

    def refuse_partial(self, *, where: str) -> None:
        """Rule 1: call this when the source ends. Raises if a group was left open."""
        if not self.open:
            return
        raise self._violation(
            f"the source ended {self.size} micro-batch(es) into an accumulation group of "
            f"{self._control.grad_accum} at {where}. Applying a partial group would take a "
            "step at a different effective batch size than every other step in this run."
        )

    def drain(self) -> tuple[float, dict[str, float]]:
        """Rule 4: the group's mean loss and its extras' means, then reset.

        Refuses an empty group rather than returning 0.0, which would enter the loss log as
        a real measurement of a step that never happened.
        """
        if not self._losses:
            raise self._violation(
                "cannot close an empty accumulation group; a 0.0 mean would enter the loss "
                "log as a measurement of a step that was never taken"
            )
        mean = sum(self._losses) / len(self._losses)
        extras = {name: sum(vs) / len(vs) for name, vs in self._extras.items()}
        self._losses.clear()
        self._extras.clear()
        return mean, extras
