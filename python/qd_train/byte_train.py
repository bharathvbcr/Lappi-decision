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
* The four rules of gradient accumulation are
  :class:`qd_train.run_control.AccumulationGroup`'s, shared with ``qd_train.trainer``.
* The ordinal-to-offset conversion is :func:`qd_train.byte_batch.span_supervision`'s.

## Why this loop is not ``trainer._train``, and what it shares anyway

``qd_train.trainer`` runs the same *shape* of loop, but its batch type is the token-LM
``Batch`` from S4 and its supervision is letter-channel shaped. Rung 0's batch is a
:class:`~qd_train.byte_batch.BatchPlan`: byte ids, an option grid and a line-start candidate
set. One loop serving both would need four injection points -- index, supervise, accumulate,
padding accounting -- which is a framework, not a simplification.

What the two genuinely share is **policy**, and that has one owner:
:class:`qd_train.run_control.AccumulationGroup` holds the four rules of gradient
accumulation and both loops drive it. That split is not a preference: the first version of
this module re-implemented those rules and got three of the four wrong. See
``GAP-RUNG0-LOOP-DUPLICATES-TRAINER-MECHANICS`` for which, and for the tests that now pin
each one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import nn

from .byte_batch import MAX_PADDING_WASTE, BatchPlan, span_supervision
from .byte_decider import ByteDecider, ByteDeciderConfig
from .heads import SpanPointerHead, plan_span_batch
from .run_control import (
    AccumulationGroup,
    Checkpoint,
    ConsumedPrefix,
    LossLog,
    LossPoint,
    Position,
    RunControl,
    TensorRef,
    TerminationReason,
)
from .tristate import NotRun, Ran, TriState

__all__ = [
    "MAX_BATCHES_PER_CALL",
    "BatchTensors",
    "Rung0ContractViolation",
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

#: The tag for a mapping whose keys are integers. ``torch.optim``'s ``state_dict`` keys its
#: per-parameter state by integer index, and JSON object keys are strings, so without this
#: the keys come back as ``"0"`` and ``load_state_dict`` raises on a parameter it cannot find.
_INTKEYS_TAG: Final[str] = "__intkeys__"


def _portable(obj: Any) -> Any:
    """A torch state dict as something a checkpoint can hold, off whatever device it was on.

    ``run_control`` imports no torch, so this is the boundary where a step's state stops
    being a live object and becomes something a file can hold. Every tensor is detached,
    moved to the CPU and handed over as a [`TensorRef`] -- its raw little-endian bytes plus
    its dtype and shape -- so the revival is exact rather than "whatever ``torch.tensor``
    infers from a list of floats".

    **Raw bytes rather than ``tolist()``, since 2026-09-20.** Measured on this step with its
    AdamW moments present, the same content both ways: 68.86 bytes per model parameter as
    decimal text against 12.10 as a JSON+sidecar pair, a ratio of 5.7. The bytes go to the
    safetensors sidecar ``Checkpoint.write`` puts beside the JSON; nothing about the state
    tree's shape changes, only where the numbers live.

    Refuses anything it does not recognise instead of passing it through: a value this
    function does not understand is a value the checkpoint cannot promise to restore.
    """
    if isinstance(obj, torch.Tensor):
        cpu = obj.detach().to("cpu").contiguous()
        return TensorRef(
            dtype=str(obj.dtype).removeprefix("torch."),
            shape=tuple(obj.shape),
            # `.view(torch.uint8)` reinterprets without converting, so no value passes
            # through a Python float in either direction. `flatten()` first because `view`
            # needs a last dimension to widen and a 0-d tensor has none.
            data=cpu.flatten().view(torch.uint8).numpy().tobytes(),
        )
    if isinstance(obj, dict):
        if any(isinstance(k, bool) or not isinstance(k, (int, str)) for k in obj):
            raise TypeError(
                f"a state dict key is {[type(k).__name__ for k in obj]}; only str and int "
                "keys survive a JSON round trip"
            )
        if all(isinstance(k, int) and not isinstance(k, bool) for k in obj) and obj:
            return {
                _INTKEYS_TAG: [[int(k), _portable(v)] for k, v in obj.items()],
            }
        return {str(k): _portable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_portable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    raise TypeError(
        f"{type(obj).__name__} is not something a checkpoint can carry. Add a case here "
        "rather than letting it through: state that cannot be written is state a resume "
        "does not have."
    )


def _revive(obj: Any) -> Any:
    """The inverse of [`_portable`]. Tensors come back on the CPU, with their own dtype.

    On the CPU deliberately: ``Module.load_state_dict`` copies into parameters that already
    live on this run's device, and ``Optimizer.load_state_dict`` casts its state to each
    parameter's device and dtype. Reviving onto a device here would pin the checkpoint to
    the machine that wrote it, which is the thing this codec exists to prevent.

    ``frombuffer`` over a ``bytearray`` copy rather than over the ``TensorRef``'s own
    ``bytes``: ``torch.frombuffer`` aliases the buffer it is given, and aliasing an
    immutable object into a tensor torch believes it may write to is how a "read-only"
    warning becomes a corrupted checkpoint two steps later.
    """
    if isinstance(obj, TensorRef):
        dtype = getattr(torch, obj.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{obj.dtype!r} does not name a torch dtype")
        flat = torch.frombuffer(bytearray(obj.data), dtype=dtype)
        return flat.reshape(obj.shape)
    if isinstance(obj, dict):
        if set(obj) == {_INTKEYS_TAG}:
            return {int(k): _revive(v) for k, v in obj[_INTKEYS_TAG]}
        return {k: _revive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_revive(v) for v in obj]
    return obj


def _rng_state() -> dict[str, Any]:
    """The framework RNG, per device kind, through the same codec as everything else.

    A dropout draw comes from here. ``cuda`` is recorded when this process has a CUDA
    device and is absent otherwise -- absent means "this run had none", which is a
    different fact from "it had one and we did not save it", and [`_restore_rng_state`]
    tells them apart.

    ``_portable`` rather than ``.tolist()``: torch's RNG state is a uint8 tensor of a few
    thousand bytes, and writing it as a few thousand decimal integers cost about four bytes
    each for no reason. One codec for every tensor in the state tree, and the sidecar holds
    it exactly.
    """
    state: dict[str, Any] = {"cpu": _portable(torch.get_rng_state())}
    if torch.cuda.is_available():
        state["cuda"] = [_portable(t) for t in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(_revive(state["cpu"]))
    saved = state.get("cuda")
    if saved is None:
        if torch.cuda.is_available():
            raise ValueError(
                "this run has CUDA devices but the checkpoint carries no CUDA RNG state: "
                "it was written on a host without them. The CPU stream can be restored and "
                "the device stream cannot, so a dropout draw on the GPU will not reproduce. "
                "Refusing rather than resuming and calling the result bit-exact."
            )
        return
    if not torch.cuda.is_available():
        raise ValueError(
            "the checkpoint carries CUDA RNG state and this host has no CUDA device; a "
            "resume here cannot reproduce the draws the original made"
        )
    if len(saved) != torch.cuda.device_count():
        raise ValueError(
            f"the checkpoint carries CUDA RNG state for {len(saved)} device(s) and this "
            f"host has {torch.cuda.device_count()}; the streams do not line up"
        )
    torch.cuda.set_rng_state_all([_revive(s) for s in saved])


@dataclass(frozen=True, slots=True)
class BatchTensors:
    """A :class:`BatchPlan` on a device. Carries no index decisions of its own."""

    context_ids: torch.Tensor
    context_mask: torch.Tensor
    option_ids: torch.Tensor
    option_mask: torch.Tensor
    n_live_options: torch.Tensor
    choice_target: torch.Tensor


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


class Rung0ContractViolation(Exception):
    """A plan, a loss or a source violated what this loop requires to be true.

    Rung 0's counterpart to :class:`qd_train.trainer.TrainerContractViolation`. The two
    loops share the rules -- :class:`qd_train.run_control.AccumulationGroup` owns them and
    writes the messages -- and each raises in its own vocabulary.
    """


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
        """Plain Python, off the accelerator. The one thing a checkpoint must be able to hold.

        Until 2026-09-20 this returned the two ``state_dict()``s as they came, which put
        live tensors into ``Checkpoint.model_state``. Measured against that code on torch
        2.12.1: the ``Checkpoint`` was built, ``to_json()`` succeeded, and ``write()`` died
        with ``TypeError: Object of type Tensor is not JSON serializable``. Every checkpoint
        this loop took was unwritable, and would have been discovered so at whichever
        boundary first tried to persist one.

        Tensors also carry the device they were trained on. Converting here means a
        checkpoint taken on a rented GPU is readable on the machine that reviews it, which
        is the difference between an interrupted rental and a lost one.

        The framework RNG state travels too. ``ByteDeciderConfig.dropout`` is configurable
        and ``accumulate`` calls ``model.train()``, so with dropout above zero the forward
        pass draws from the global generator: without this, a resumed run would continue
        from a *fresh* stream and the trajectory would diverge for a reason the loss log
        does not show. It is recorded per device kind, because a CUDA stream cannot be
        restored onto a CPU and pretending otherwise is the silent half of the same bug.
        """
        return {
            "model": _portable(self.model.state_dict()),
            "optimizer": _portable(self.optimizer.state_dict()),
            "span_weight": self.span_weight,
            "rng": _rng_state(),
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
        self.model.load_state_dict(_revive(state["model"]))
        self.optimizer.load_state_dict(_revive(state["optimizer"]))
        if "rng" not in state:
            raise ValueError(
                "checkpoint state carries no 'rng'. Written before the RNG travelled with "
                "the step, so a resume from it cannot reproduce a dropout draw. Refusing "
                "rather than resuming into a different random stream and calling it "
                "bit-exact."
            )
        _restore_rng_state(state["rng"])


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
    #: The padding gate, against :data:`qd_train.byte_batch.MAX_PADDING_WASTE`. A tri-state
    #: rather than a float: a run that consumed no batches has no waste to report, and 0.0
    #: would read as perfect bucketing.
    padding: TriState


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
    group = AccumulationGroup(control, violation=Rung0ContractViolation)
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
    consumed = ConsumedPrefix()

    while True:
        if optimizer_steps >= control.total_steps:
            termination = "steps_exhausted"
            break
        # Rule 2: only at a group boundary, so no half-accumulated gradient is discarded.
        if group.should_stop_for_cap():
            termination = "wall_clock_cap"
            break
        if micro_batches >= max_batches:
            # Not a termination reason. A run that consumed its batch ceiling without
            # completing the schedule did not finish; reporting "steps_exhausted" would
            # make a truncated run indistinguishable from a complete one.
            raise Rung0ContractViolation(
                f"consumed {micro_batches} batches without completing the schedule's "
                f"{control.total_steps} steps; the source is not the one this run planned for"
            )

        plan = next(source, None)
        if plan is None:
            # Rule 1: a group left open when the source ends is a contract violation.
            group.refuse_partial(where=f"epoch {epoch}, batch {micro_batches}")
            termination = "data_exhausted"
            break

        if not isinstance(plan, BatchPlan):
            raise TypeError(f"expected a BatchPlan, got {type(plan).__name__}")

        _fold_plan(consumed, plan)
        losses = step.accumulate(plan)
        # Rule 3: a non-finite loss stops the run here rather than training on NaN.
        group.add(
            losses.total,
            where=f"epoch {epoch}, batch {micro_batches}",
            choice=losses.choice,
            span=losses.span,
        )
        micro_batches += 1
        rows += plan.batch_size
        waste_den += plan.batch_size * plan.context_width
        waste_num += sum(1 for row in plan.context_mask for live in row if not live)

        if not group.ready:
            continue

        mean_total, means = group.drain()
        mean_choice, mean_span = means["choice"], means["span"]
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
                _checkpoint(
                    step, control, optimizer_steps, micro_batches, epoch, seed, loss_log,
                    consumed,
                )
            )

    checkpoint = _checkpoint(
        step, control, optimizer_steps, micro_batches, epoch, seed, loss_log, consumed
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
        padding=_padding_gate(waste_num, waste_den),
    )


def _padding_gate(dead: float, total: int) -> TriState:
    """The fraction of context positions that were padding, against the S4 bar.

    ``NotRun`` on an empty run rather than 0.0: no batches means no waste figure, and a 0%
    reading would clear the gate on a run that padded nothing because it trained on nothing.
    """
    if total == 0:
        return NotRun(
            reason="no micro-batches were consumed, so there is no padding to measure; "
            "0.0 would read as perfect bucketing on a run that trained on nothing"
        )
    waste = dead / total
    return Ran(
        passed=waste <= MAX_PADDING_WASTE,
        value=waste,
        n=total - int(dead),
        n_total=total,
        detail=(
            f"{waste:.1%} of context positions were padding across {total} of them; "
            f"gate is <= {MAX_PADDING_WASTE:.0%}. Rung 0 pads to the batch maximum, so this "
            "measures the batch's raggedness, not a bucket plan."
        ),
    )


def _fold_plan(prefix: ConsumedPrefix, plan: BatchPlan) -> None:
    """Record one consumed ``BatchPlan`` in the running digest.

    The rows' identities and every supervision target: which examples this micro-batch held
    and what each one was taught. ``context_ids`` is deliberately *not* folded -- it is the
    padded byte grid, it is large, and ``example_ids`` already names the rows it was built
    from. That is a stated limit rather than an oversight: this digest answers "were these
    the same examples, supervised the same way", not "were these the same bytes".
    """
    prefix.fold(
        "\x00".join(plan.example_ids).encode("utf-8"),
        repr(plan.choice_target).encode("utf-8"),
        repr(plan.span_target).encode("utf-8"),
        repr(plan.span_end_target).encode("utf-8"),
        repr(plan.n_live_options).encode("utf-8"),
    )


def _checkpoint(
    step: Rung0Step,
    control: RunControl,
    optimizer_steps: int,
    micro_batches: int,
    epoch: int,
    seed: int,
    loss_log: LossLog,
    consumed: ConsumedPrefix,
) -> Checkpoint:
    return Checkpoint(
        position=Position(epoch=epoch, index=micro_batches),
        optimizer_step=optimizer_steps,
        seed=seed,
        schedule=control.schedule,
        loss_log=loss_log.snapshot(),
        consumed_digest=consumed.hexdigest(),
        model_state=step.state(),
    )
