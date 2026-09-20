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

**[`CostEstimate`] is the cap priced.** DESIGN-4's finding was precisely that the plan's cap
was *not costed*: 40 h x $31.92 = $1,277 for the block alone against a "$1,085 at ~34 h"
estimate, so "we hit the cap" was also a silent 15% budget overrun. The estimate here is
computed from the cap, never from the hoped-for duration, so the number on the launch line
is the worst case rather than the happy case.

**[`Position`] is `(epoch, index)` and nothing else.** ``Batch.index`` is documented as *"the
batch's deterministic position in the epoch ... the order is a pure function of ``(seed,
epoch)``, never of hidden iterator state"*. So the entire resume state of the data side is
two integers, and S5's bit-exact resume is reconstructible from them. Anything else -- an
iterator to pickle, a file offset, a shuffle buffer -- would be state that has to be
restored rather than recomputed, and restored state is what drifts.

**[`LossLog`] stores floats as ``float.hex()``.** S5 asks that a resumed run reproduce the
loss trajectory *exactly*. ``repr``/``%g`` round-trips are not exact at every magnitude, and
a digest over lossy text compares the rendering rather than the number. ``float.hex`` is
exact and total, so the digest is a statement about the run and not about formatting.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Self

__all__ = [
    "APPROVAL_FREE_USD",
    "MAX_CAP_S",
    "MAX_GRAD_ACCUM",
    "MAX_LOSS_POINTS",
    "AccumulationGroup",
    "Checkpoint",
    "CostEstimate",
    "LRSchedule",
    "LaunchRefused",
    "LossLog",
    "LossPoint",
    "Position",
    "RunControl",
    "TerminationReason",
    "WallClockCap",
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


class LaunchRefused(RuntimeError):
    """A run was configured in a way rule 4 does not permit it to start in."""


def _require_finite(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


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
        return _require_finite("elapsed_s", elapsed_s) >= self.cap_s

    def remaining_s(self, elapsed_s: float) -> float:
        return max(0.0, self.cap_s - _require_finite("elapsed_s", elapsed_s))


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The cap, priced. DESIGN-4: state the cap as what it costs.

    ``projected_usd`` is computed from the **cap**, not from the hoped-for duration, so
    hitting the cap is not also a budget surprise.
    """

    usd_per_hour: float
    cap: WallClockCap
    n_gpus: int
    instance: str = "unspecified"

    def __post_init__(self) -> None:
        rate = _require_finite("usd_per_hour", self.usd_per_hour)
        if rate < 0.0:
            raise ValueError(f"usd_per_hour must be non-negative, got {rate!r}")
        if not isinstance(self.n_gpus, int) or isinstance(self.n_gpus, bool):
            raise TypeError(f"n_gpus must be int, got {type(self.n_gpus).__name__}")
        if self.n_gpus < 0:
            raise ValueError(f"n_gpus must be non-negative, got {self.n_gpus}")
        if not self.instance.strip():
            raise ValueError(
                "instance must name the machine being priced: a rate with no instance cannot "
                "be checked against a price list, and DESIGN-4's error was exactly a rate read "
                "off the wrong row (per-GPU $3.99 vs per-node $31.92)"
            )

    @property
    def projected_usd(self) -> float:
        """What this run costs **if it runs to the cap**. The number that goes on the line."""
        return self.usd_per_hour * self.cap.cap_hours

    def cost_for(self, elapsed_s: float) -> float:
        """What it cost in fact. Measured, for the ledger row."""
        return self.usd_per_hour * _require_finite("elapsed_s", elapsed_s) / 3600.0

    @property
    def requires_human_approval(self) -> bool:
        """Rule 4. Multi-GPU always; single-GPU only once the capped cost reaches $20."""
        return self.n_gpus > 1 or self.projected_usd >= APPROVAL_FREE_USD

    def approval_line(self) -> str:
        need = "NEEDS A HUMAN YES" if self.requires_human_approval else "no approval required"
        return (
            f"{self.instance}: {self.n_gpus} GPU(s) at ${self.usd_per_hour:.2f}/h, capped at "
            f"{self.cap.cap_hours:.2f} h -> ${self.projected_usd:.2f} at the cap -- {need}"
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


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything a resume needs that it cannot recompute.

    The data position is two integers because the batch order is a pure function of
    ``(seed, epoch)`` -- so the iterator is *rebuilt*, not restored. ``seed`` is carried so
    a resume against a differently-seeded source is refused rather than silently training
    on a different order; ``schedule`` is carried for the same reason.

    ``model_state`` is opaque here on purpose: this module does not import torch, and the
    step implementation is the only thing that knows what its own state means.
    """

    position: Position
    optimizer_step: int
    seed: int
    schedule: LRSchedule
    loss_log: LossLog
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

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_json(),
            "optimizer_step": self.optimizer_step,
            "seed": self.seed,
            "schedule": self.schedule.to_json(),
            "loss_log": self.loss_log.to_json(),
            "loss_digest": self.loss_log.digest(),
            "model_state": self.model_state,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        log = LossLog.from_json(raw["loss_log"])
        stated = raw.get("loss_digest")
        if stated is not None and stated != log.digest():
            raise ValueError(
                f"checkpoint loss_digest is {stated!r} but the log hashes to {log.digest()!r}; "
                "the checkpoint was modified after it was written, and a resume from it cannot "
                "claim to reproduce the trajectory"
            )
        return cls(
            position=Position.from_json(raw["position"]),
            optimizer_step=int(raw["optimizer_step"]),
            seed=int(raw["seed"]),
            schedule=LRSchedule.from_json(raw["schedule"]),
            loss_log=log,
            model_state=dict(raw.get("model_state", {})),
        )

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> Self:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


# --- the three of them, driving one loop --------------------------------------------------


@dataclass(slots=True)
class RunControl:
    """The schedule, the cap and the price, plus the clock they are measured against.

    ``clock`` is injected so the cap is testable in milliseconds rather than in hours. It
    defaults to ``time.monotonic`` because a wall-clock cap measured against a clock that
    can step backwards is not a cap.
    """

    schedule: LRSchedule
    cap: WallClockCap
    cost: CostEstimate
    grad_accum: int = 1
    checkpoint_every: int = 0
    approved_by: str = ""
    clock: Callable[[], float] = time.monotonic
    _t0: float | None = field(default=None, init=False, repr=False)

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

    # -- the clock -------------------------------------------------------

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("RunControl.start() called twice; one control drives one run")
        self._t0 = self.clock()

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def elapsed_s(self) -> float:
        if self._t0 is None:
            raise RuntimeError(
                "RunControl.elapsed_s() before start(): an unstarted run has no elapsed time, "
                "and returning 0.0 would read as a run that has only just begun"
            )
        return max(0.0, self.clock() - self._t0)

    def expired(self) -> bool:
        """Has the cap been reached? Checked at optimizer-step boundaries by the loop."""
        return self.cap.expired(self.elapsed_s())

    def cost_so_far(self) -> float:
        return self.cost.cost_for(self.elapsed_s())

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
