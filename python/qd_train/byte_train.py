"""Rung 0's training step and loop: the byte decider, the pointer head, one encode.

``docs/lappi.md`` puts rung 0 at the bottom of the ladder -- a byte-level decider trained
from scratch, no pretrained backbone -- and this is the module that trains it. It is
torch-gated, so its tests ``importorskip``; every index decision it relies on was made in
:mod:`qd_train.byte_batch`, which is torch-free and runs in CI.

## One encode, two heads

The two slots read the *same* context states:

* the option head scores each option against the context (:meth:`ByteDecider.score_from_hidden`),
* the span head points at a line start (:class:`qd_train.heads.SpanPointerHead`).

:func:`losses` encodes once and hands the same ``hidden`` to both. Encoding twice would not
merely be slow: with dropout on, the two passes differ, and the span head would be pointing
into states the option head never saw. ``ByteDecider.score_from_hidden`` exists precisely so
this is expressible.

## What is reused rather than restated

* The abstain row's position is decided **once**, in :func:`qd_train.heads.plan_span_batch`,
  which is the function pinned against ``crates/qd-runtime/src/answer.rs``. This module never
  computes it.
* The cap, the LR schedule, the checkpoint cadence, the cost and the approval gate are
  :class:`qd_train.run_control.RunControl`'s. This module reads them and never re-decides
  them -- rule 2: gates are read-only.
* The ordinal-to-offset conversion is :func:`qd_train.byte_batch.span_supervision`'s.

## Why this loop is not ``trainer._train``

``qd_train.trainer`` runs the same shape of loop -- accumulate, step, cap, checkpoint -- but
its batch type is the token-LM ``Batch`` from S4 and its supervision is letter-channel
shaped. Rung 0's batch is a :class:`~qd_train.byte_batch.BatchPlan`: byte ids, an option
grid and a line-start candidate set. Making one loop serve both means generalising
``trainer._train`` over its batch type, which is a change to a module the whole S2-S4 lane
rests on. That is a judgement call rather than a drive-by, so it is recorded as
``GAP-RUNG0-LOOP-DUPLICATES-TRAINER-MECHANICS`` and the policy objects above are shared in
the meantime, which is where the drift would actually hurt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Final

import torch
from torch import nn

from .byte_batch import BatchPlan, span_supervision
from .byte_context import ID_PAD
from .byte_decider import ByteDecider, ByteDeciderConfig
from .heads import SpanPointerHead, plan_span_batch
from .run_control import Checkpoint, LossLog, LossPoint, Position, RunControl, TerminationReason

__all__ = [
    "MAX_BATCHES_PER_CALL",
    "BatchTensors",
    "Rung0Losses",
    "Rung0Model",
    "Rung0Result",
    "Rung0Step",
    "plan_to_tensors",
    "train_rung0",
]

#: A hard ceiling on micro-batches consumed by one call, independent of the schedule. The
#: same bound ``qd_train.trainer`` carries, and for the same reason: the schedule bounds a
#: well-behaved source, this bounds a source that yields for ever.
MAX_BATCHES_PER_CALL: Final[int] = 10_000_000


@dataclass(frozen=True, slots=True)
class BatchTensors:
    """A :class:`BatchPlan` on a device. Carries no index decisions of its own."""

    context_ids: torch.Tensor
    context_mask: torch.Tensor
    option_ids: torch.Tensor
    option_mask: torch.Tensor
    n_live_options: torch.Tensor
    choice_target: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.context_ids.shape[0])


def plan_to_tensors(plan: BatchPlan, *, device: torch.device | str = "cpu") -> BatchTensors:
    """Move a plan onto ``device``. Pure transport: nothing here decides a row or a column."""
    if plan.batch_size == 0:
        raise ValueError("an empty batch has nothing to train on")

    def long(rows: Any) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=device)

    def boolean(rows: Any) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.bool, device=device)

    return BatchTensors(
        context_ids=long(plan.context_ids),
        context_mask=boolean(plan.context_mask),
        option_ids=long(plan.option_ids),
        option_mask=boolean(plan.option_mask),
        n_live_options=long(plan.n_live_options),
        choice_target=long(plan.choice_target),
    )


@dataclass(frozen=True, slots=True)
class Rung0Losses:
    """The two slot losses and the sum that is actually differentiated.

    Both are carried, never only the total: a total that falls while the span term is flat
    is a model that learned the mutation class and nothing about *where*, and one number
    cannot say so.
    """

    choice: float
    span: float
    total: float


class Rung0Model(nn.Module):
    """The byte decider and the span pointer head, sharing one encoder."""

    def __init__(self, config: ByteDeciderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ByteDeciderConfig()
        self.decider = ByteDecider(self.config)
        # The pointer head reads the decider's hidden states, so its width is not a free
        # parameter -- a second width here would fail at the gather rather than at build.
        self.span_head = SpanPointerHead(self.config.width)

    def losses(self, plan: BatchPlan) -> tuple[torch.Tensor, torch.Tensor]:
        """``(choice_loss, span_loss)`` for one plan, with the context encoded **once**."""
        device = next(self.parameters()).device
        tensors = plan_to_tensors(plan, device=device)

        hidden = self.decider.encode_context(tensors.context_ids, tensors.context_mask)
        choice_logits = self.decider.score_from_hidden(
            hidden,
            tensors.context_mask,
            tensors.option_ids,
            tensors.option_mask,
            tensors.n_live_options,
        )
        choice = nn.functional.cross_entropy(choice_logits, tensors.choice_target)

        # `plan_span_batch` owns the abstain row; `span_supervision` owns the ordinal-to-
        # offset conversion. Neither is re-derived here.
        span_plan = plan_span_batch(span_supervision(plan), device=device)
        span = self.span_head.loss(hidden, span_plan)
        return choice, span


class Rung0Step:
    """Forward, backward and the optimizer step for :class:`Rung0Model`.

    ``accumulate`` and ``apply`` are separate for the same reason they are in
    :class:`qd_train.trainer.TrainStep`: gradient accumulation asks two questions, and one
    method with a flag makes the caller responsible for a distinction the callee knows.
    """

    def __init__(
        self,
        model: Rung0Model,
        *,
        weight_decay: float = 0.01,
        span_weight: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        if not span_weight > 0.0:
            raise ValueError(
                f"span_weight must be positive, got {span_weight}; zero would train the span "
                "head on nothing while its loss still appeared in the log"
            )
        if not max_grad_norm > 0.0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
        self.model = model
        self.span_weight = float(span_weight)
        self.max_grad_norm = float(max_grad_norm)
        # lr is supplied per step by RunControl.lr_at, so the value here is a placeholder
        # that `apply` overwrites before every step; it is never the schedule.
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0, weight_decay=weight_decay
        )

    def accumulate(self, plan: BatchPlan) -> Rung0Losses:
        """Forward and backward one micro-batch. Returns both slot losses."""
        self.model.train()
        choice, span = self.model.losses(plan)
        total = choice + self.span_weight * span
        total.backward()
        return Rung0Losses(
            choice=float(choice.detach()),
            span=float(span.detach()),
            total=float(total.detach()),
        )

    def apply(self, *, lr: float) -> None:
        """Take the optimizer step at ``lr`` and clear the accumulated gradient."""
        if not lr > 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        # Bounded before the step: an unclipped gradient on a from-scratch model is how a
        # run ends with NaN parameters and a loss log that stops rather than says why.
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "span_weight": self.span_weight,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        missing = {"model", "optimizer", "span_weight"} - set(state)
        if missing:
            raise ValueError(f"checkpoint state is missing {sorted(missing)}")
        if float(state["span_weight"]) != self.span_weight:
            raise ValueError(
                f"this step weights the span loss at {self.span_weight} but the checkpoint "
                f"was written at {state['span_weight']}; resuming would change the objective "
                "mid-run and the loss curve would not say so"
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


@dataclass(frozen=True, slots=True)
class Rung0Result:
    """What the loop did, and why it stopped."""

    termination: TerminationReason
    optimizer_steps: int
    micro_batches: int
    rows: int
    loss_log: LossLog
    choice_log: tuple[float, ...]
    span_log: tuple[float, ...]
    checkpoint: Checkpoint
    wall_clock_s: float
    cost_usd: float
    padding_waste: float = 0.0


@dataclass
class _Accumulator:
    """One optimizer step's worth of micro-batches."""

    grad_accum: int
    choice: float = 0.0
    span: float = 0.0
    total: float = 0.0
    n: int = 0
    waste_num: float = 0.0
    waste_den: int = 0
    seen: list[int] = field(default_factory=list)

    def add(self, losses: Rung0Losses) -> None:
        self.choice += losses.choice
        self.span += losses.span
        self.total += losses.total
        self.n += 1

    @property
    def ready(self) -> bool:
        return self.n >= self.grad_accum

    def drain(self) -> tuple[float, float, float]:
        means = (self.choice / self.n, self.span / self.n, self.total / self.n)
        self.choice = self.span = self.total = 0.0
        self.n = 0
        return means


def train_rung0(
    plans: Iterable[BatchPlan],
    *,
    step: Rung0Step,
    control: RunControl,
    epoch: int = 0,
    seed: int = 0,
    on_checkpoint: Callable[[Checkpoint], None] | None = None,
    max_batches: int = MAX_BATCHES_PER_CALL,
) -> Rung0Result:
    """Train until the data, the schedule or the cap runs out -- and say which.

    There is no "just ended" exit. A loop that cannot name why it stopped cannot be told
    apart from one that stopped by accident, which is the distinction
    :data:`qd_train.run_control.TerminationReason` exists to force.

    The cap is checked **before** each micro-batch, so an expired run does no further work;
    ``control.expired`` measures against the injected clock, never against ``time.time``.
    """
    if not isinstance(step, Rung0Step):
        raise TypeError(f"step must be a Rung0Step, got {type(step).__name__}")
    if max_batches < 1 or max_batches > MAX_BATCHES_PER_CALL:
        raise ValueError(f"max_batches must be in [1, {MAX_BATCHES_PER_CALL}], got {max_batches}")

    control.start()
    accumulator = _Accumulator(grad_accum=control.grad_accum)
    loss_log = LossLog()
    choice_log: list[float] = []
    span_log: list[float] = []
    optimizer_steps = 0
    micro_batches = 0
    rows = 0
    waste_num = 0.0
    waste_den = 0

    source: Iterator[BatchPlan] = iter(plans)
    termination: TerminationReason = "data_exhausted"

    while True:
        if control.expired():
            termination = "wall_clock_cap"
            break
        if optimizer_steps >= control.total_steps:
            termination = "steps_exhausted"
            break
        if micro_batches >= max_batches:
            termination = "steps_exhausted"
            break

        try:
            plan = next(source)
        except StopIteration:
            termination = "data_exhausted"
            break

        if not isinstance(plan, BatchPlan):
            raise TypeError(f"expected a BatchPlan, got {type(plan).__name__}")

        accumulator.add(step.accumulate(plan))
        micro_batches += 1
        rows += plan.batch_size
        width = plan.context_width
        waste_den += plan.batch_size * width
        waste_num += sum(
            1 for row in plan.context_mask for live in row if not live
        )

        if not accumulator.ready:
            continue

        mean_choice, mean_span, mean_total = accumulator.drain()
        # Logged at the step's own 0-based index, then taken. `Checkpoint` reads
        # `optimizer_step` as the *next* step -- the same "start here" convention
        # `Position.index` uses -- so the increment has to land after the log or a resume
        # would replay the step just recorded.
        loss_log.append(
            LossPoint(
                optimizer_step=optimizer_steps,
                epoch=epoch,
                batch_index=micro_batches - 1,
                loss=mean_total,
            )
        )
        step.apply(lr=control.lr_at(optimizer_steps))
        optimizer_steps += 1
        choice_log.append(mean_choice)
        span_log.append(mean_span)

        if on_checkpoint is not None and control.should_checkpoint(optimizer_steps):
            on_checkpoint(
                _checkpoint(step, control, optimizer_steps, micro_batches, epoch, seed, loss_log)
            )

    # A partial accumulation is deliberately NOT applied: a step taken on fewer micro-batches
    # than `grad_accum` is a different effective batch size, and the loss log would carry it
    # as though it were the same. It is dropped, and `micro_batches` still counts it.
    checkpoint = _checkpoint(
        step, control, optimizer_steps, micro_batches, epoch, seed, loss_log
    )
    return Rung0Result(
        termination=termination,
        optimizer_steps=optimizer_steps,
        micro_batches=micro_batches,
        rows=rows,
        loss_log=loss_log.snapshot(),
        choice_log=tuple(choice_log),
        span_log=tuple(span_log),
        checkpoint=checkpoint,
        wall_clock_s=control.elapsed_s(),
        cost_usd=control.cost_so_far(),
        padding_waste=(waste_num / waste_den) if waste_den else 0.0,
    )


def _checkpoint(
    step: Rung0Step,
    control: RunControl,
    optimizer_steps: int,
    micro_batches: int,
    epoch: int,
    seed: int,
    loss_log: LossLog,
) -> Checkpoint:
    return Checkpoint(
        position=Position(epoch=epoch, index=micro_batches),
        optimizer_step=optimizer_steps,
        seed=seed,
        schedule=control.schedule,
        loss_log=loss_log.snapshot(),
        model_state=step.state(),
    )


# `ID_PAD` is re-exported so a caller building plans for this module takes the pad id from
# the same place the embedding's padding row was sized from.
PAD_ID: Final[int] = ID_PAD
