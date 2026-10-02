"""The CPT and FT loops.

Takes ``Iterable[Batch]`` rather than a reader. ``docs/training-contract.md``: *"that is
dependency inversion for its own sake only in part; the practical reason is that it makes
the trainer testable against synthetic batches, and it let the three lanes be built
concurrently without one waiting on another's file to exist."* It also means this module
is torch-free: the optimizer step arrives through [`TrainStep`], and the only thing that
needs a GPU is the object implementing it.

## The two objectives, and why they are not one loop with a flag

CPT and FT ask different questions of the same rows, so they get different functions:

* **CPT** is next-token prediction over every real token pair in the row. Every position
  ``p`` with ``p < lengths[i] - 1`` is supervised, and all three of ``Batch``'s supervision
  fields are ``None``.
* **FT** supervises **one answer per row**, at the position ``Batch.target_index`` names --
  read from the batch, never inferred. ``docs/schema-api.md`` answering procedure: *"Answer
  each slot as a 1-token query."*

What they share is the machinery around the objective -- the cap, the step accounting, the
checkpoint, the ledger row -- and that lives in one private loop which takes the supervision
function as a *value*. A boolean would have been a third thing to keep in sync with the two
call sites, and the failure mode of getting it backwards is a run that trains the wrong
objective and still produces a falling loss curve.

## Spans are a second kind of answer, and they are not letters

``docs/schema-api.md`` gives ``span`` the decode *"pointer head over line-start tokens"*.
``qd_data.render._slot_suffix`` gives a ``SpanSlot`` an **empty letter alphabet** on purpose,
so the only letter a span row has is ``noul``. Supervising a span row through the letter
channel therefore trains the span head to **abstain always** -- silently, with a falling
loss curve, on the slot the plan's headline ``evidence`` example depends on. That is
``GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL``, and this module does not reintroduce it:
``SLOT_SPAN`` rows are excluded from the letter mask and routed to [`SpanSupervision`].

**The pointer-head loss itself is not computed here, and that is deliberate.** Two
independent reasons, either alone sufficient:

1. *The head is model parameters.* Scoring positions needs start/end projections over the
   hidden states. This module holds no parameters and imports no torch -- the optimizer step
   arrives through [`TrainStep`]. A trainer that owned the pointer head would own part of
   the model, which is the boundary ``docs/training-contract.md`` draws.
2. *The candidate set is not in the contract.* The head scores **line-start tokens**, and
   ``Batch`` carries no channel saying which token positions are line starts -- nor the
   newline id one would be derived from. Deriving them here would be a second implementation
   of line-start determination alongside ``qd_data.render``'s, and ``render.py``'s own
   docstring records what that costs: *"a span label derived one way pointed at a different
   line than the same label derived the other."*

So the seam is [`SpanScoringStep`]: a batch with span rows goes to ``accumulate_span``, and
a step without that method makes the loop **refuse the batch**. The refusal is the design.
See ``GAP-TRAINER-SPAN-POINTER-HEAD-NOT-IMPLEMENTABLE-HERE`` for what a lane owning the
heads still has to build, and what it needs added to ``Batch`` to build it.

## Masking is this module's job, not the model's

``Batch.lengths`` exists because *"the trainer masks loss rather than training on padding"*.
So the mask is computed here and handed to the step as [`Supervision`]. A step
implementation that ignores it is wrong, but it cannot be wrong *by omission* -- there is no
path through this loop that calls a step without one. Training on padding is the single
easiest way to get a loss curve that looks fine and a model that is worse, and it is
invisible in every metric that is itself averaged over padded positions.

## The cap, and where it is checked

Rule 4 requires auto-termination. [`RunControl.expired`] is checked at **optimizer-step
boundaries**: at the top of each accumulation group, never mid-group. A cap enforced
mid-group would throw away a half-accumulated gradient, so the run would stop having done
arithmetic it never applied. The cost is that a run overshoots the cap by at most one
optimizer step, which is the natural granularity of "stop".

Termination is a normal return, not an exception, so ``RunRecorder`` writes ``completed``
and the row carries the reason. The row is written either way -- that is the recorder's
guarantee, on every exit path including SIGTERM -- but a capped run that exited through the
exception path would be recorded as ``failed``, and a cap firing is not a failure.

## Resume

The checkpoint records ``(epoch, index)`` of the **next** batch. On resume the caller
rebuilds the source for the same ``(seed, epoch)`` and this loop skips forward until it
reaches that index, then asserts it landed exactly on it. Anything else -- a source that
starts past the index, or one whose indices are not strictly increasing -- is refused,
because S5's guarantee is that the order is a pure function of ``(seed, epoch,
batch_tokens)`` and the shard set, and a source that does not honour that cannot deliver a
bit-exact resume.

**Landing on the index is not the same as landing on the batch.** Until 2026-09-20 that was
the whole of the check, and it was measured passing on a source that shared the indices and
none of the content: a reader asked for the same ``(seed, epoch)`` with ``batch_tokens``
one token larger produced the same number of batches, so the skip found index N, the run
completed, and ``termination`` read ``steps_exhausted`` on a corpus order the checkpoint had
never seen. So every consumed batch is folded into a [`ConsumedPrefix`] digest that the
checkpoint carries, and the skip *rehashes the prefix it walks* and compares.

The cost is one sha256 per batch, measured on this host at 2.7-3.0 GB/s over the batch's
own bytes: 0.024 ms for 8x2048, 0.75 ms for 64x8192 (524,288 tokens, 2.10 MB). Against an
optimizer step on an H100 that is well under one percent, and it is paid on the training
path, not only on the resume path. The alternative was a resumed run whose loss curve looks
fine.

What this check cannot see: a source that is identical up to the resume point and differs
after it. That is a real limit and it is narrow -- for a shard set, bucket membership makes
the prefix depend on the whole set -- but it is a limit, not a proof.

## Device

Nothing in this module names a device; the device belongs to the ``TrainStep``. The
consequence for "bit-exact" is stated in ``qd_train.run_control``: **cpu** reproduces
bit-exactly across processes (measured), **mps** does not (measured -- a seed does not pin
an ``mps`` run), and **cuda** is **not run** on this host. A resume reproduces the
trajectory only as exactly as the step's device does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np

from .artifacts import NO_SPAN, SLOT_LM, SLOT_SPAN, SPAN_ABSTAIN, Batch
from .ledger import RunRecorder
from .run_control import (
    AccumulationGroup,
    Checkpoint,
    ConsumedPrefix,
    LossLog,
    LossPoint,
    Position,
    RunControl,
    TerminationReason,
)
from .tristate import Ran

__all__ = [
    "CPT",
    "FT",
    "MAX_BATCHES_PER_CALL",
    "ChoicePermutation",
    "Objective",
    "PermutationRefusal",
    "Progress",
    "SpanScoringStep",
    "SpanSupervision",
    "Supervision",
    "TrainResult",
    "TrainStep",
    "TrainerContractViolation",
    "cpt_supervision",
    "ft_supervision",
    "locate_option_lines",
    "newline_terminated_ids",
    "permute_choice_row",
    "train_cpt",
    "train_ft",
]

#: A hard ceiling on batches consumed by one call, independent of the schedule. The schedule
#: already bounds a well-behaved source (``RunControl.max_micro_batches``); this bounds a
#: source that yields for ever while the schedule waits for steps that never complete.
MAX_BATCHES_PER_CALL: Final[int] = 10_000_000

#: What a training step's forward may attend with (``QwenDecisionStep.train_attention_mask``,
#: ``real_ft_run --train-attention-mask``). ``"padding"`` -- the mask from ``Batch.lengths``
#: -- is every row so far; ``"none"`` (``is_causal`` only) is the Tier-B candidate of Fable's
#: no-mask ruling (campaign/f-j7prime-preregistered.json ``no_mask``). Here rather than in
#: ``backbone`` so the argv layer can name it without importing the tower's dependencies.
TRAIN_ATTENTION_MASKS: Final[tuple[str, ...]] = ("padding", "none")


class TrainerContractViolation(Exception):
    """A batch, a step or a resume violated what this loop requires to be true."""


# --- what a batch supervises --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpanSupervision:
    """The span rows of a batch and their gold token positions.

    A span answer is a *position pair*, not a vocabulary token, so it cannot travel in
    ``targets``/``mask`` with the letter answers. It travels here instead, and the loop
    hands it to a step that has declared it can score positions -- see [`SpanScoringStep`].

    The per-row arrays are ``[K]``, where ``K`` is the number of ``SLOT_SPAN`` rows;
    ``rows`` says which rows of the batch those are. ``line_starts`` is ``[K, L]``: the
    pointer head's **candidate set**, carried through verbatim from ``Batch`` rather than
    re-derived, because ``docs/schema-api.md`` decodes a span over line-start tokens and a
    second derivation of "which token starts a line" is how one gold comes to name two lines.

    ``abstaining`` marks the rows whose gold is ``SPAN_ABSTAIN``: not a position, but the
    head's extra row. Those rows carry ``SPAN_ABSTAIN`` in ``start`` and ``end`` and must
    not be read as positions.

    This dataclass deliberately carries **no head-layout convention** -- not the row count,
    not where the abstain row sits. Those belong to ``qd_train.heads``, which is the module
    pinned against the runtime's definition.
    """

    rows: np.ndarray
    query_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    line_starts: np.ndarray
    abstaining: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "rows": self.rows.shape,
            "query_index": self.query_index.shape,
            "start": self.start.shape,
            "end": self.end.shape,
            "abstaining": self.abstaining.shape,
        }
        if len(set(shapes.values())) != 1:
            raise TrainerContractViolation(f"span arrays must share one shape, got {shapes}")
        if self.rows.ndim != 1:
            raise TrainerContractViolation(f"span arrays must be 1-D, got {self.rows.shape}")
        if self.rows.size == 0:
            raise TrainerContractViolation(
                "SpanSupervision with no rows; absent and empty are different facts, so a "
                "batch with no span rows carries span=None rather than an empty channel"
            )
        if self.abstaining.dtype != np.bool_:
            raise TrainerContractViolation(
                f"abstaining must be bool, got {self.abstaining.dtype}"
            )
        if self.line_starts.ndim != 2 or self.line_starts.shape[0] != self.rows.size:
            raise TrainerContractViolation(
                f"line_starts must be [K={self.rows.size}, L], got {self.line_starts.shape}"
            )
        if self.line_starts.dtype != np.bool_:
            raise TrainerContractViolation(
                f"line_starts must be a bool mask, got {self.line_starts.dtype}"
            )
        declared = (self.start == SPAN_ABSTAIN) & (self.end == SPAN_ABSTAIN)
        if not bool(np.array_equal(declared, self.abstaining)):
            raise TrainerContractViolation(
                "`abstaining` disagrees with the SPAN_ABSTAIN sentinels in start/end; the flag "
                "and the sentinel are two statements of one fact"
            )
        pointing = ~self.abstaining
        if bool(np.any(self.start[pointing] > self.end[pointing])):
            raise TrainerContractViolation("a span's start is after its end")
        if bool(np.any(self.start[pointing] < 0)) or bool(np.any(self.end[pointing] < 0)):
            raise TrainerContractViolation(
                "a non-abstaining span row carries a negative position"
            )

    @property
    def n_spans(self) -> int:
        return int(self.rows.size)

    @property
    def n_abstaining(self) -> int:
        return int(self.abstaining.sum())

    def candidate_counts(self) -> np.ndarray:
        """Line-start candidates per span row. The head adds its own abstain row to this."""
        return self.line_starts.sum(axis=1).astype(np.int64)


@dataclass(frozen=True, slots=True)
class Supervision:
    """What of one batch is supervised, split by the kind of answer it has.

    ``targets``/``mask`` are the **letter** channel: both ``[B, L-1]``, because position
    ``p`` of a row predicts the token at ``p + 1``. ``mask`` is what ``Batch.lengths`` is
    for -- and, under FT, also what ``Batch.target_index`` is for.

    ``span`` is the **position** channel, present only when the batch has ``SLOT_SPAN``
    rows. Those rows are absent from ``mask`` on purpose: their only available letter is
    ``noul``, so supervising them as letters trains the span head to abstain always, which
    is ``GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`` exactly.
    """

    targets: np.ndarray
    mask: np.ndarray
    n_supervised: int
    span: SpanSupervision | None = None

    def __post_init__(self) -> None:
        if self.targets.shape != self.mask.shape:
            raise TrainerContractViolation(
                f"targets {self.targets.shape} and mask {self.mask.shape} must have one shape"
            )
        if self.mask.dtype != np.bool_:
            raise TrainerContractViolation(f"mask must be bool, got {self.mask.dtype}")
        counted = int(self.mask.sum())
        if counted != self.n_supervised:
            raise TrainerContractViolation(
                f"n_supervised says {self.n_supervised} but the mask has {counted} true "
                "positions; the count and the mask are two statements of one fact"
            )
        if self.n_supervised <= 0 and self.span is None:
            raise TrainerContractViolation(
                "no position in this batch is supervised, by letter or by span. A loss "
                "averaged over zero positions is 0/0, and a batch that contributes nothing "
                "was still paid for."
            )

    @property
    def n_answers(self) -> int:
        """Supervised letter positions plus span rows -- what the batch actually teaches."""
        return self.n_supervised + (self.span.n_spans if self.span is not None else 0)


def _prediction_grid(batch: Batch) -> tuple[np.ndarray, np.ndarray, int]:
    """``(positions[1, L-1], lengths[B, 1], width)`` -- the shared arithmetic of both masks."""
    _, width = batch.tokens.shape
    if width < 2:  # pragma: no cover - Batch.__post_init__ refuses this
        raise TrainerContractViolation(
            f"a batch padded to width {width} has no next-token pair: position p predicts "
            "token p+1, so a row needs at least two columns"
        )
    positions = np.arange(width - 1, dtype=np.int64)[None, :]
    lengths = batch.lengths.astype(np.int64)[:, None]
    return positions, lengths, width


def _refuse_unsupervised_rows(
    mask: np.ndarray, batch: Batch, what: str, *, must_be_supervised: np.ndarray | None = None
) -> None:
    """Refuse any row that was required to contribute and did not.

    ``must_be_supervised`` excludes rows whose answer lives in another channel -- a span
    row is not "unsupervised", it is supervised elsewhere.

    **No ``Batch`` the contract now accepts can reach this.** It used to fire on a
    one-token row under CPT, which is what ``GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS`` was
    opened about: the constructor permitted the row and this loop refused it, so the two
    disagreed about what a batch is. That is settled in ``artifacts._MIN_ROW_TOKENS``,
    which refuses the row where the writer can still be named. This guard stays as a
    postcondition on the mask arithmetic itself -- a mask that loses a row is the failure
    it names, and the check costs one reduction per batch -- but the *contract* statement
    is now upstream, and ``test_artifacts.py`` is where it is tested.
    """
    empty = ~mask.any(axis=1)
    if must_be_supervised is not None:
        empty &= must_be_supervised
    rows = np.flatnonzero(empty)
    if rows.size:  # pragma: no cover - Batch refuses every row that could reach this
        raise TrainerContractViolation(
            f"{what}: rows {rows.tolist()[:16]} (of {mask.shape[0]}) have no supervised "
            f"position, with lengths {batch.lengths[rows].tolist()[:16]}. A row that "
            "contributes nothing to the loss was loaded, padded and paid for; dropping it "
            "quietly makes the effective batch smaller than the recipe says it is."
        )


def cpt_supervision(batch: Batch) -> Supervision:
    """Next-token prediction over every real token pair.

    Position ``p`` is supervised when its target ``p + 1`` is a real token, i.e. when
    ``p + 1 <= lengths[i] - 1``. Padding is never a target and never an input.

    A batch carrying the FT supervision channel is refused: ``Batch``'s contract is that
    all three slot fields are ``None`` for CPT, and a batch that carries slot answers is
    one the caller meant to fine-tune on.

    Every row of an acceptable ``Batch`` contributes at least one supervised position,
    because ``artifacts._MIN_ROW_TOKENS`` is 2 and position 0 is supervised whenever
    ``lengths > 1``. That floor is the settlement of
    ``GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS``: this loop used to be the only thing
    refusing a one-token row, which made it stricter than the constructor it consumes from.
    """
    if batch.slot_kind is not None or batch.target_index is not None:
        raise TrainerContractViolation(
            "CPT was handed a batch carrying slot_kind/target_index. The contract's CPT case "
            "is all three supervision fields None; this batch has per-row slot answers, so it "
            "is an FT batch and training it as CPT would supervise the prompt as well as the "
            "answer."
        )
    positions, lengths, _ = _prediction_grid(batch)
    mask = positions < (lengths - 1)
    _refuse_unsupervised_rows(mask, batch, "CPT")
    return Supervision(targets=batch.tokens[:, 1:], mask=mask, n_supervised=int(mask.sum()))


def ft_supervision(batch: Batch) -> Supervision:
    """Supervise each row's declared answer, by the kind of answer it is.

    ``Batch.target_index[i]`` is *"the position whose next token is the gold letter"*, so
    the letter channel is ``positions == target_index`` -- read from the batch, never
    inferred as ``lengths - 2``. A batch mixing slot kinds has no single offset, and the
    inferred version was wrong for every row whose answer was not last.

    ``SLOT_SPAN`` rows are routed to [`SpanSupervision`] and are **excluded** from the
    letter channel. ``qd_data.render._slot_suffix`` gives a span slot an empty letter
    alphabet, so its only gold letter is ``noul``: supervising it as a letter would train
    the span head to abstain always, with a loss curve that looked fine. That is the bug
    ``GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`` records, and it is not reintroduced here.

    A ``SLOT_LM`` row cannot reach this function: ``Batch`` refuses one in a batch that
    carries the slot channel, which is the settlement of
    ``GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED``. The guard below stays as defence in depth
    against a ``Batch`` built around its own constructor, alongside the three other
    ``Batch``-refuses-this guards in this function.
    """
    if batch.slot_kind is None or batch.target_index is None:
        raise TrainerContractViolation(
            "FT needs slot_kind and target_index; this batch carries neither. Under the "
            "contract a batch with all three supervision fields None is a CPT batch, and "
            "guessing the answer position from lengths is what "
            "GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL was raised about."
        )

    positions, _, _ = _prediction_grid(batch)
    kinds = batch.slot_kind.astype(np.int64)
    is_span = kinds == SLOT_SPAN

    lm_rows = np.flatnonzero(kinds == SLOT_LM)
    if lm_rows.size:  # pragma: no cover - Batch.__post_init__ refuses this
        raise TrainerContractViolation(
            f"rows {lm_rows.tolist()[:16]} are SLOT_LM inside an FT batch. SLOT_LM means "
            "next-token over the whole sequence and carries no gold letter, so its "
            "target_index names nothing; this loop will not guess whether the row was meant "
            "as an LM row mixed into fine-tuning or as a mislabelled slot."
        )

    query = batch.target_index.astype(np.int64)[:, None]
    mask = (positions == query) & (~is_span)[:, None]
    _refuse_unsupervised_rows(
        mask,
        batch,
        "FT (the letter channel supervises exactly Batch.target_index)",
        must_be_supervised=~is_span,
    )

    span: SpanSupervision | None = None
    if bool(is_span.any()):
        if batch.span_target is None:  # pragma: no cover - Batch.__post_init__ refuses this
            raise TrainerContractViolation(
                "a SLOT_SPAN row without span_target reached the trainer; Batch refuses this"
            )
        if batch.line_starts is None:  # pragma: no cover - Batch.__post_init__ refuses this
            raise TrainerContractViolation(
                "a SLOT_SPAN row without line_starts reached the trainer; Batch refuses this"
            )
        rows = np.flatnonzero(is_span)
        starts = batch.span_target[rows, 0].astype(np.int64)
        ends = batch.span_target[rows, 1].astype(np.int64)
        span = SpanSupervision(
            rows=rows,
            query_index=batch.target_index.astype(np.int64)[rows],
            start=starts,
            end=ends,
            line_starts=batch.line_starts[rows],
            abstaining=(starts == SPAN_ABSTAIN) & (ends == SPAN_ABSTAIN),
        )
        if bool(np.any(span.start == NO_SPAN)):  # pragma: no cover - Batch refuses this
            raise TrainerContractViolation("a SLOT_SPAN row carries NO_SPAN as its gold span")

    return Supervision(
        targets=batch.tokens[:, 1:], mask=mask, n_supervised=int(mask.sum()), span=span
    )


@dataclass(frozen=True, slots=True)
class Objective:
    """A name, the supervision it implies, and the ledger ``run_kind`` it must be recorded as."""

    name: Literal["cpt", "ft"]
    supervise: Callable[[Batch], Supervision]
    run_kind: Literal["cpt", "ft"]


CPT: Final[Objective] = Objective(name="cpt", supervise=cpt_supervision, run_kind="cpt")
FT: Final[Objective] = Objective(name="ft", supervise=ft_supervision, run_kind="ft")


# --- the seam to torch ---------------------------------------------------------------------


@runtime_checkable
class TrainStep(Protocol):
    """One model's forward, backward and optimizer step. The only torch-shaped thing here.

    ``accumulate`` and ``apply`` are separate because gradient accumulation asks two
    different questions -- "add this batch's gradient" and "take a step" -- and a single
    method with a flag would make the caller responsible for a distinction the callee
    already knows.
    """

    def accumulate(self, batch: Batch, supervision: Supervision) -> float:
        """Forward and backward one micro-batch. Returns its mean loss over supervised positions."""
        ...

    def apply(self, *, lr: float) -> None:
        """Take the optimizer step at ``lr`` and clear the accumulated gradient."""
        ...

    def state(self) -> dict[str, Any]:
        """Checkpoint-serialisable state. Opaque to this module.

        **This contract changed on 2026-09-20 and the old wording was the defect.** It read
        "JSON-serialisable state for a checkpoint", and ``Checkpoint`` enforced that
        literally: a tensor had to be rendered as decimal text to be stored at all. Measured
        on the rung-0 step with its AdamW moments, the same content encoded both ways, 68.86
        bytes per model parameter as JSON against 12.10 as a JSON+sidecar pair -- about
        97 GB per checkpoint against 17 GB for this program's 1.4B trainable parameters. A
        rented box checkpointing periodically fills its disk before the run ends.

        What a step may now return: JSON values, and :class:`qd_train.run_control.TensorRef`
        for anything of size. A ``TensorRef`` is dtype, shape and raw little-endian bytes;
        ``Checkpoint.write`` puts the bytes in a safetensors sidecar beside the JSON and
        keeps each one's digest in the body. Nothing here holds a live tensor, this module
        still imports no torch, and a checkpoint still names no device -- ``TensorRef``
        carries bytes off the accelerator, exactly as the list of floats did.
        """
        ...

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore what ``state`` produced, including any ``TensorRef`` it returned."""
        ...


@runtime_checkable
class SpanScoringStep(TrainStep, Protocol):
    """A step that can also score *positions*, not only vocabulary tokens.

    A batch containing ``SLOT_SPAN`` rows is handed to ``accumulate_span`` instead of
    ``accumulate``, and a step that does not define it makes the loop **refuse the batch**.
    That refusal is the point: the alternative designs both fail quietly. Passing the span
    rows through ``accumulate`` would let a step ignore ``supervision.span`` and train
    nothing on them; folding them into the letter channel would train them on ``noul``.

    The implementation owns the pointer head, because the head is model parameters and this
    module holds none -- see the module docstring for why the loss itself is not computed
    here, and ``GAP-TRAINER-SPAN-POINTER-HEAD-NOT-IMPLEMENTABLE-HERE`` for what is blocked.

    ``accumulate_span`` receives the **whole** [`Supervision`], letter channel included,
    because a batch may mix span and choice rows and the relative weight of the two losses
    is a recipe decision belonging to whoever owns the heads.
    """

    def accumulate_span(self, batch: Batch, supervision: Supervision) -> float:
        """Forward and backward one micro-batch that contains span rows."""
        ...


@dataclass(frozen=True, slots=True)
class TrainResult:
    """What a loop did, and why it stopped."""

    objective: str
    termination: TerminationReason
    optimizer_steps: int
    micro_batches: int
    supervised_tokens: int
    span_rows: int
    padded_positions: int
    total_positions: int
    loss_log: LossLog
    checkpoint: Checkpoint
    wall_clock_s: float
    cost_usd: float
    row_id: str | None

    @property
    def padding_fraction(self) -> float:
        return self.padded_positions / self.total_positions if self.total_positions else 0.0


@dataclass(frozen=True, slots=True)
class Progress:
    """Where a live loop is, handed to ``on_progress`` after every optimizer step.

    Counts are cumulative over this call; ``elapsed_s`` is ``RunControl.elapsed_s()``, the
    clock the cap is checked on. The loop reports and never throttles: a caller that prints
    decides how often, so a slow sink costs the run nothing it did not choose.
    """

    optimizer_step: int
    total_steps: int
    micro_batches: int
    supervised_tokens: int
    total_positions: int
    elapsed_s: float
    loss: float


# --- the loop ------------------------------------------------------------------------------


def _fold(prefix: ConsumedPrefix, batch: Batch) -> None:
    """Record one consumed batch in the running digest.

    Everything that says *what was trained on*: where the batch sat in the epoch, how wide
    it was, the tokens themselves, and the supervision channels that decide what each row
    teaches. Two sources that agree on all of these are the same training data under a
    different name; two that do not are not a resume.

    ``np.ascontiguousarray`` because the buffer of a non-contiguous array is not the array,
    and a digest over the wrong bytes is worse than no digest.
    """
    parts: list[bytes] = [
        int(batch.index).to_bytes(8, "big"),
        int(batch.bucket).to_bytes(8, "big"),
        np.ascontiguousarray(batch.tokens).tobytes(),
        np.ascontiguousarray(batch.lengths).tobytes(),
    ]
    for optional in (
        batch.slot_kind,
        batch.target_index,
        batch.span_target,
        batch.line_starts,
    ):
        parts.append(b"" if optional is None else np.ascontiguousarray(optional).tobytes())
    prefix.fold(*parts)


def _skip_to(
    it: Iterable[Batch], *, start_index: int, budget: int
) -> tuple[Batch | None, int, ConsumedPrefix]:
    """Advance to the batch whose ``index`` is ``start_index``, hashing what it walks past.

    Returns the batch, how many were skipped, and the digest of the skipped prefix. The
    digest is built here rather than by the caller because this loop is the only thing that
    sees those batches, and because the resumed run must carry on folding into *this*
    hasher: a sha256 cannot be restored from its own output.

    Refuses a source that runs past the index without hitting it: that means the order this
    source produces is not the order the checkpoint was taken from, and resuming anyway
    would train on a different sequence of batches while claiming a bit-exact resume. The
    *content* comparison belongs to the caller, which is the one holding the checkpoint.
    """
    skipped = 0
    prefix = ConsumedPrefix()
    for batch in it:
        if skipped >= budget:
            raise TrainerContractViolation(
                f"skipped {skipped} batches without reaching index {start_index}; the source "
                "is longer than this run's batch budget"
            )
        if batch.index < start_index:
            _fold(prefix, batch)
            skipped += 1
            continue
        if batch.index != start_index:
            raise TrainerContractViolation(
                f"resume wanted batch index {start_index} but the source jumped to "
                f"{batch.index}. S5 requires the batch order to be a pure function of "
                "(seed, epoch); a source that skips the resume point is not that function."
            )
        return batch, skipped, prefix
    return None, skipped, prefix


def _train(
    batches: Iterable[Batch],
    *,
    objective: Objective,
    epoch: int,
    step: TrainStep,
    control: RunControl,
    recorder: RunRecorder,
    on_checkpoint: Callable[[Checkpoint], None] | None = None,
    resume_from: Checkpoint | None = None,
    max_batches: int = MAX_BATCHES_PER_CALL,
    on_progress: Callable[[Progress], None] | None = None,
) -> TrainResult:
    """The mechanics both objectives share. The objective itself arrives as a value."""
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError(f"epoch must be int, got {type(epoch).__name__}")
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if not 0 < max_batches <= MAX_BATCHES_PER_CALL:
        raise ValueError(
            f"max_batches must be in (0, {MAX_BATCHES_PER_CALL}], got {max_batches}"
        )
    if recorder.run_kind != objective.run_kind:
        raise TrainerContractViolation(
            f"the {objective.name.upper()} loop was handed a recorder with "
            f"run_kind={recorder.run_kind!r}. The ledger row would name a different run than "
            "the one that produced it, and every comparison against that row would be wrong."
        )

    seed = recorder.protocol.seed
    start_index = 0
    first_step = 0
    log = LossLog()
    consumed = ConsumedPrefix()

    if resume_from is not None:
        if resume_from.seed != seed:
            raise TrainerContractViolation(
                f"checkpoint was taken at seed {resume_from.seed} but this run's protocol seed "
                f"is {seed}. The seed fixes the training stream and, where the caller plans "
                "with it (tools/real_ft_run.py --batch-order seed), the batch order, so this "
                "would resume a different run, possibly into a different sequence of batches. "
                "The batch order itself is guarded by the consumed-digest check below, "
                "whatever seed planned it."
            )
        if resume_from.position.epoch != epoch:
            raise TrainerContractViolation(
                f"checkpoint is at epoch {resume_from.position.epoch} but this call was given "
                f"epoch {epoch}. Rebuild the source for the checkpoint's epoch, or pass that "
                "epoch; the loop will not guess which of the two is meant."
            )
        if resume_from.schedule != control.schedule:
            raise TrainerContractViolation(
                "the checkpoint's schedule differs from this run's. The learning rate is "
                "recomputed from the step number, so two schedules means the resumed run uses "
                "rates the original never would have."
            )
        step.load_state(resume_from.model_state)
        start_index = resume_from.position.index
        first_step = resume_from.optimizer_step
        log = resume_from.loss_log.snapshot()

    control.start()
    # Disarmed on every way out. `start()` arms a watchdog when the control carries an
    # auto_terminate, and nothing else ever called `stop()`: after this function returned,
    # the thread kept polling the cap of a run that had finished, and would fire the
    # terminate action -- a hard exit -- in the middle of whatever the process did next: the
    # next seed's arm, or scoring the model this run just trained.
    try:
        return _train_loop(
            batches, objective=objective, epoch=epoch, step=step, control=control,
            recorder=recorder, on_checkpoint=on_checkpoint, resume_from=resume_from,
            on_progress=on_progress, max_batches=max_batches, seed=seed,
            start_index=start_index,
            first_step=first_step, log=log, consumed=consumed,
        )
    finally:
        control.stop()


def _train_loop(
    batches: Iterable[Batch],
    *,
    objective: Objective,
    epoch: int,
    step: TrainStep,
    control: RunControl,
    recorder: RunRecorder,
    on_checkpoint: Callable[[Checkpoint], None] | None,
    resume_from: Checkpoint | None,
    on_progress: Callable[[Progress], None] | None,
    max_batches: int,
    seed: int,
    start_index: int,
    first_step: int,
    log: LossLog,
    consumed: ConsumedPrefix,
) -> TrainResult:
    """The loop ``_train`` runs between ``control.start()`` and ``control.stop()``."""
    optimizer_step = first_step
    micro_batches = 0
    supervised_tokens = 0
    span_rows = 0
    padded_positions = 0
    total_positions = 0
    last_index = start_index - 1
    termination: TerminationReason = "data_exhausted"

    group = AccumulationGroup(control, violation=TrainerContractViolation)

    with recorder:
        recorder.metric(
            "train.projected_usd_at_cap",
            Ran(passed=True, value=control.cost.projected_usd, detail=control.cost.approval_line()),
        )

        source = iter(batches)
        pending: Batch | None = None
        if start_index > 0:
            pending, _, consumed = _skip_to(
                source, start_index=start_index, budget=max_batches
            )
            if pending is None:
                raise TrainerContractViolation(
                    f"resume wanted batch index {start_index} in epoch {epoch} but the source "
                    "ended first; this is not the source the checkpoint was taken from"
                )
        if resume_from is not None and consumed.hexdigest() != resume_from.consumed_digest:
            # Landing on the index is not landing on the batch. Measured on 2026-09-20:
            # the same (seed, epoch) at a batch_tokens one larger yields the same number of
            # batches with the same indices and entirely different contents, and everything
            # above this line accepted it.
            raise TrainerContractViolation(
                f"the {start_index} batch(es) before the resume point hash to "
                f"{consumed.hexdigest()} but the checkpoint recorded "
                f"{resume_from.consumed_digest}. The source is not the one the checkpoint "
                "was taken from -- same seed and epoch, different batches. S5's order is a "
                "pure function of (seed, epoch, batch_tokens) and the shard set; rebuild "
                "the source with the batch_tokens and the shard set the checkpoint was "
                "written under."
            )

        while True:
            if optimizer_step >= control.total_steps:
                termination = "steps_exhausted"
                break
            # The cap is checked at a group boundary -- rule 2 of `AccumulationGroup` --
            # so no half-accumulated gradient is ever discarded. Overshoot is bounded by
            # one optimizer step.
            if group.should_stop_for_cap():
                termination = "wall_clock_cap"
                break
            if micro_batches >= max_batches:
                raise TrainerContractViolation(
                    f"consumed {micro_batches} batches without completing the schedule's "
                    f"{control.total_steps} steps; the source is not the one this run planned for"
                )

            if pending is not None:
                batch, pending = pending, None
            else:
                batch = next(source, None)
            if batch is None:
                group.refuse_partial(where=f"epoch {epoch}, after batch index {last_index}")
                termination = "data_exhausted"
                break

            if batch.index <= last_index:
                raise TrainerContractViolation(
                    f"batch index {batch.index} follows {last_index}; indices within an epoch "
                    "are strictly increasing, because S5's resume is reconstructed from them"
                )
            last_index = batch.index
            # Folded where the batch is accepted, so the digest covers exactly the batches
            # this run consumed -- the same set `Position.index` is one past the end of.
            _fold(consumed, batch)

            supervision = objective.supervise(batch)
            if supervision.span is None:
                loss = step.accumulate(batch, supervision)
            else:
                if not isinstance(step, SpanScoringStep):
                    raise TrainerContractViolation(
                        f"batch {batch.index} has {supervision.span.n_spans} SLOT_SPAN row(s), "
                        f"but {type(step).__name__} defines no accumulate_span. A span answer "
                        "is a pair of token positions scored by a pointer head over line-start "
                        "tokens (docs/schema-api.md), not a vocabulary token. This loop refuses "
                        "the batch rather than folding the span rows into the letter channel, "
                        "where their only gold is the noul letter and the span head would learn "
                        "to abstain always -- GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL."
                    )
                loss = step.accumulate_span(batch, supervision)
            # Rules 3 and 4 live in `AccumulationGroup`: a non-finite loss stops the run
            # here, and the group's recorded loss is the mean over exactly its own
            # micro-batches.
            group.add(float(loss), where=f"epoch {epoch}, batch {batch.index}")

            micro_batches += 1
            supervised_tokens += supervision.n_supervised
            span_rows += supervision.span.n_spans if supervision.span is not None else 0
            total_positions += int(batch.tokens.size)
            padded_positions += int(batch.tokens.size) - int(batch.lengths.sum())
            if not group.ready:
                continue

            group_loss, _ = group.drain()
            lr = control.lr_at(optimizer_step)
            step.apply(lr=lr)
            log.append(
                LossPoint(
                    optimizer_step=optimizer_step,
                    epoch=epoch,
                    batch_index=batch.index,
                    loss=group_loss,
                )
            )
            optimizer_step += 1
            if on_progress is not None:
                on_progress(
                    Progress(
                        optimizer_step=optimizer_step,
                        total_steps=control.total_steps,
                        micro_batches=micro_batches,
                        supervised_tokens=supervised_tokens,
                        total_positions=total_positions,
                        elapsed_s=control.elapsed_s(),
                        loss=group_loss,
                    )
                )

            if on_checkpoint is not None and control.should_checkpoint(optimizer_step - first_step):
                on_checkpoint(
                    Checkpoint(
                        position=Position(epoch=epoch, index=last_index + 1),
                        optimizer_step=optimizer_step,
                        seed=seed,
                        schedule=control.schedule,
                        loss_log=log.snapshot(),
                        consumed_digest=consumed.hexdigest(),
                        model_state=step.state(),
                    )
                )

        wall = control.elapsed_s()
        checkpoint = Checkpoint(
            position=Position(epoch=epoch, index=last_index + 1),
            optimizer_step=optimizer_step,
            seed=seed,
            schedule=control.schedule,
            loss_log=log.snapshot(),
            consumed_digest=consumed.hexdigest(),
            model_state=step.state(),
        )
        last_point = log.last
        recorder.metric(
            "train.termination",
            Ran(
                passed=termination != "wall_clock_cap",
                value=termination,
                detail=f"{objective.name} stopped after {optimizer_step} optimizer step(s)",
            ),
        )
        recorder.metric("train.optimizer_steps", Ran(passed=True, value=optimizer_step))
        recorder.metric("train.micro_batches", Ran(passed=True, value=micro_batches))
        recorder.metric(
            "train.supervised_tokens",
            Ran(passed=True, value=supervised_tokens, n=supervised_tokens, n_total=total_positions),
        )
        # Carried separately: a span answer is one row, not one token, so folding it into
        # supervised_tokens would understate what a span-heavy run actually trained on.
        recorder.metric("train.span_rows", Ran(passed=True, value=span_rows))
        recorder.metric(
            "train.padding_fraction",
            Ran(
                passed=True,
                value=(padded_positions / total_positions) if total_positions else 0.0,
                n=padded_positions,
                n_total=total_positions,
            ),
        )
        recorder.metric(
            "train.final_loss",
            Ran(passed=True, value=last_point.loss)
            if last_point is not None
            else Ran(passed=False, value=None, detail="no optimizer step completed"),
        )
        recorder.metric("train.loss_log_digest", Ran(passed=True, value=log.digest()))
        # The final checkpoint's: what identifies the batch order this run consumed (an index
        # identifies a position within an order, not the order -- artifacts.Batch). Fable's
        # seed-order ruling (AUDIT/post-f-2026-10-02/fable-seed-order-ruling.md section 1):
        # F's four seeds were found to share one order only by reading their checkpoints.
        recorder.metric(
            "train.consumed_digest",
            Ran(
                passed=True, value=checkpoint.consumed_digest,
                detail="the final checkpoint's consumed_digest: the order of the batches consumed",
            ),
        )

    return TrainResult(
        objective=objective.name,
        termination=termination,
        optimizer_steps=optimizer_step,
        micro_batches=micro_batches,
        supervised_tokens=supervised_tokens,
        span_rows=span_rows,
        padded_positions=padded_positions,
        total_positions=total_positions,
        loss_log=log,
        checkpoint=checkpoint,
        wall_clock_s=wall,
        cost_usd=control.cost.cost_for(wall),
        row_id=recorder.row.row_id if recorder.row is not None else None,
    )


def train_cpt(
    batches: Iterable[Batch],
    *,
    epoch: int,
    step: TrainStep,
    control: RunControl,
    recorder: RunRecorder,
    on_checkpoint: Callable[[Checkpoint], None] | None = None,
    resume_from: Checkpoint | None = None,
    max_batches: int = MAX_BATCHES_PER_CALL,
    on_progress: Callable[[Progress], None] | None = None,
) -> TrainResult:
    """Continued pre-training: next-token prediction over every real token of every row.

    One call covers one epoch, because ``(epoch, index)`` is the resume coordinate and the
    batch order is a function of ``(seed, epoch)``. A multi-epoch run calls this once per
    epoch, carrying the returned checkpoint forward.
    """
    return _train(
        batches,
        objective=CPT,
        epoch=epoch,
        step=step,
        control=control,
        recorder=recorder,
        on_checkpoint=on_checkpoint,
        resume_from=resume_from,
        max_batches=max_batches,
        on_progress=on_progress,
    )


def train_ft(
    batches: Iterable[Batch],
    *,
    epoch: int,
    step: TrainStep,
    control: RunControl,
    recorder: RunRecorder,
    on_checkpoint: Callable[[Checkpoint], None] | None = None,
    resume_from: Checkpoint | None = None,
    max_batches: int = MAX_BATCHES_PER_CALL,
    on_progress: Callable[[Progress], None] | None = None,
) -> TrainResult:
    """Fine-tuning: one supervised position per row, the single answer token at its end.

    The prompt is context, not target. Supervising it as well would train the model to
    reproduce agent-authored diffs -- which ``docs/hardening.md`` section 3 treats as
    hostile input -- and would drown the one token the decision actually turns on.
    """
    return _train(
        batches,
        objective=FT,
        epoch=epoch,
        step=step,
        control=control,
        recorder=recorder,
        on_checkpoint=on_checkpoint,
        resume_from=resume_from,
        max_batches=max_batches,
        on_progress=on_progress,
    )


# --- option permutation at train time ---------------------------------------------------
#
# The idea is RSI-Jev's (``rsijev/encode.py:72-80,201-221``, MIT, Copyright (c) 2026 Shanghua
# Gao, @8f34a4f): present a choice question's options in a different order each time it is
# seen, and keep the gold on the option rather than on the position. Their code permutes
# before tokenizing and un-permutes the logits; nothing of it is copied here, because this
# trainer never sees text. ``qd_data.render.shuffle_options`` already shuffles once per
# example at shard-write time, so every pass over a shard set sees that ONE order. This is
# the second, per-pass permutation, done on the token rows the reader hands over.
#
# **Why a token-space swap is exact, and what it rests on.** A choice slot renders its options
# as lines ``"<L>. <value>\n"`` (``qd_data.render._render_option_lines``) followed by
# ``"Z. noul\n"``, as the last thing before the answer, and ``escape_inline`` guarantees no
# value contains a newline. Under Qwen's byte-level pre-tokenizer a letter can only share a
# pre-token with a preceding non-letter, non-digit, non-newline character, so a letter at
# the start of a line always begins a pre-token, and the ``"."`` after it begins the next.
# Each option line therefore starts at a token boundary on its letter, and the tokens from
# there to the next line's letter are exactly that option's ``". value\n"``, whatever the
# value's last character merged with. Swapping those spans and leaving the letters where
# they are is the token sequence of the same prompt rendered with the options reordered.
#
# A bare letter followed by ``"."`` is NOT enough to find a line: a value ``"1B. trick"``
# tokenizes ``"1", "B", "."`` because a digit is its own pre-token -- measured against the
# real tokenizer, and it is what the first version of this search matched. What makes a
# position a line start is that the token before it ENDS IN A NEWLINE, and since no value
# contains one, inside the option block that holds for the line starts and nothing else.
# Which ids end in a newline is a fact about the vocabulary, not the row, so the caller
# supplies it (``newline_terminated_ids``). Both claims are checked against the real Qwen
# tokenizer by ``python/tests/test_trainer.py``, not asserted from this paragraph.


class PermutationRefusal(TrainerContractViolation):
    """A choice row whose option lines could not be located exactly.

    A refusal rather than a skip: failing to find the structure on one row means the
    tokenization assumption above is wrong for this shard set, and then every row that DID
    match is suspect too.
    """


#: Bounds the backward search for the option block. A choice slot is at most 17 lines of at
#: most ``RenderCaps.max_option_bytes`` (512) bytes; 16 * 1024 tokens is far past any of it.
MAX_OPTION_BLOCK_TOKENS: Final[int] = 16_384

#: How a byte-level BPE vocabulary spells the newline byte (GPT-2's byte-to-unicode table).
_BYTE_LEVEL_NEWLINE: Final[str] = "Ċ"


def newline_terminated_ids(vocab: Mapping[str, int]) -> frozenset[int]:
    """The ids of every byte-level token whose last byte is a newline.

    ``vocab`` is a ``tokenizer.json``'s ``model.vocab`` (token string -> id). Read off the
    vocabulary rather than guessed from a row, because the line end before an option can be
    ``"\n"``, ``")\n"``, ``").\n"`` or ``"|>\n"`` -- measured -- and the set of such tokens is the
    tokenizer's, not the corpus's.
    """
    ids = frozenset(int(i) for tok, i in vocab.items() if tok.endswith(_BYTE_LEVEL_NEWLINE))
    if not ids:
        raise PermutationRefusal(
            "no token in this vocabulary ends in the byte-level newline; it is not a "
            "byte-level BPE vocabulary and the option-line search does not apply to it"
        )
    return ids


def locate_option_lines(
    row: np.ndarray,
    *,
    answer_at: int,
    letter_ids: tuple[int, ...],
    noul_id: int,
    line_end_ids: frozenset[int],
) -> tuple[int, ...]:
    """Token positions of the option lines' letters, ``A..`` then ``noul``, in order.

    ``letter_ids`` are this row's named options' letter ids in rendered order (``A, B, ...``,
    without ``noul``). A line start is a letter whose predecessor is in ``line_end_ids`` and
    whose successor is the ``noul`` line's separator (the ``"."``). Searched backwards from
    the answer, because the option block is the last thing rendered and a context can hold
    lines of its own that start ``A.``.
    """
    if len(letter_ids) < 2:
        raise PermutationRefusal(
            f"a choice row with {len(letter_ids)} named option(s) has nothing to permute"
        )
    if len(set(letter_ids)) != len(letter_ids) or noul_id in letter_ids:
        raise PermutationRefusal(f"letter ids {letter_ids} (noul {noul_id}) are not distinct")
    floor = max(1, answer_at - MAX_OPTION_BLOCK_TOKENS)

    def line_start(q: int, letter: int) -> bool:
        return int(row[q]) == letter and int(row[q - 1]) in line_end_ids

    z = next((p for p in range(answer_at - 2, floor - 1, -1) if line_start(p, noul_id)), None)
    if z is None:
        raise PermutationRefusal(
            f"no noul letter (id {noul_id}) at a line start within "
            f"{MAX_OPTION_BLOCK_TOKENS} tokens before the answer at {answer_at}"
        )
    dot = int(row[z + 1])
    positions = [z]
    upper = z
    for letter in reversed(letter_ids):
        p = next(
            (
                q for q in range(upper - 2, floor - 1, -1)
                if line_start(q, letter) and int(row[q + 1]) == dot
            ),
            None,
        )
        if p is None:
            raise PermutationRefusal(
                f"option letter id {letter} at a line start and followed by the noul line's "
                f"separator {dot} was not found before position {upper}"
            )
        positions.append(p)
        upper = p
    positions.reverse()
    return tuple(positions)


def permute_choice_row(
    row: np.ndarray,
    *,
    answer_at: int,
    letter_ids: tuple[int, ...],
    noul_id: int,
    line_end_ids: frozenset[int],
    perm: tuple[int, ...],
) -> np.ndarray:
    """``row`` with its options shown in order ``perm`` and its gold moved with them.

    ``perm[j]`` is the ORIGINAL option shown at new position ``j`` -- RSI's
    ``option_permutation`` convention, "presented position -> canonical option index".
    Letters stay where they are; the ``". value\\n"`` spans move. A gold on option ``k``
    becomes the letter at ``perm.index(k)``; a ``noul`` gold is unchanged. The length is
    unchanged, so ``target_index`` still names the answer.
    """
    m = len(letter_ids)
    if sorted(perm) != list(range(m)):
        raise PermutationRefusal(f"{perm} is not a permutation of range({m})")
    gold = int(row[answer_at])
    if gold == noul_id:
        new_gold = gold
    elif gold in letter_ids:
        new_gold = letter_ids[perm.index(letter_ids.index(gold))]
    else:
        raise PermutationRefusal(
            f"the gold token {gold} is neither one of this row's option letters "
            f"{letter_ids} nor noul {noul_id}"
        )
    starts = locate_option_lines(
        row, answer_at=answer_at, letter_ids=letter_ids, noul_id=noul_id,
        line_end_ids=line_end_ids,
    )
    spans = [row[starts[k] + 1 : starts[k + 1]] for k in range(m)]
    body = np.concatenate(
        [
            np.concatenate([np.asarray([letter_ids[j]], dtype=row.dtype), spans[perm[j]]])
            for j in range(m)
        ]
    )
    out = row.copy()
    out[starts[0] : starts[m]] = body
    out[answer_at] = new_gold
    return out


@dataclass(frozen=True, slots=True)
class ChoicePermutation:
    """The per-pass option permutation, as a recipe: a seed and the letters' token ids.

    ``letter_ids`` maps a letter (``"A"``..``"P"``) to its post-remap token id and ``noul_id``
    is ``Z``'s, all post-remap; ``line_end_ids`` is [`newline_terminated_ids`] mapped the same
    way. The permutation of batch ``index`` row ``r`` is a pure function of
    ``(seed, index, r)`` through ``qd_data.render.DeterministicRng`` -- the stream the
    renderer's own shuffle uses -- so a resumed run regenerates the batches it skipped
    exactly, and ``Checkpoint.consumed_digest`` still matches.
    """

    seed: int
    letter_ids: Mapping[str, int]
    noul_id: int
    line_end_ids: frozenset[int]

    def permutation(self, index: int, row: int, m: int) -> tuple[int, ...]:
        from qd_data.render import DeterministicRng

        return DeterministicRng(
            "qd_train.trainer.ChoicePermutation.v1", self.seed, index, row
        ).permutation(m)

    def apply(
        self,
        batch: Batch,
        alphabets: list[tuple[str, ...] | None],
        *,
        permutation_for: Callable[[int, int, int], tuple[int, ...]] | None = None,
    ) -> tuple[Batch, int]:
        """``(batch with every choice row permuted, rows permuted)``.

        ``alphabets[r]`` is row ``r``'s named option letters in rendered order (``noul``
        excluded) for a ``SLOT_CHOICE`` row, and ``None`` for any other: score bins are
        ordinal and never shuffled (``qd_data.render.shuffle_options``), and a span row has
        no named option. A mismatch between the two is a refusal.

        ``permutation_for(index, row, m)`` replaces :meth:`permutation` as the source of each
        row's order -- the eval's second pass draws a derangement per example, where training
        draws a uniform shuffle per pass -- and every refusal here applies to it unchanged.
        """
        draw = self.permutation if permutation_for is None else permutation_for
        from .artifacts import SLOT_CHOICE

        if batch.slot_kind is None or batch.target_index is None:
            raise PermutationRefusal("option permutation applies to FT batches only")
        if len(alphabets) != batch.tokens.shape[0]:
            raise PermutationRefusal(
                f"{len(alphabets)} alphabets for a batch of {batch.tokens.shape[0]} rows"
            )
        tokens = batch.tokens.copy()
        permuted = 0
        for r, letters in enumerate(alphabets):
            is_choice = int(batch.slot_kind[r]) == SLOT_CHOICE
            if is_choice != (letters is not None):
                raise PermutationRefusal(
                    f"batch {batch.index} row {r}: slot_kind {int(batch.slot_kind[r])} "
                    f"with alphabet {letters}"
                )
            if letters is None:
                continue
            unknown = [x for x in letters if x not in self.letter_ids]
            if unknown:
                raise PermutationRefusal(
                    f"batch {batch.index} row {r}: letters {unknown} have no known token id "
                    f"(known: {sorted(self.letter_ids)}). Permuting without them would move "
                    "a line whose letter was never located."
                )
            if batch.line_starts is not None and bool(batch.line_starts[r].any()):
                raise PermutationRefusal(
                    f"batch {batch.index} row {r}: a choice row carrying line-start "
                    "candidates, whose positions this permutation would move"
                )
            answer_at = int(batch.target_index[r]) + 1
            if answer_at != int(batch.lengths[r]) - 1:
                raise PermutationRefusal(
                    f"batch {batch.index} row {r}: the answer is at {answer_at}, not the "
                    f"last real token {int(batch.lengths[r]) - 1}"
                )
            ids = tuple(self.letter_ids[x] for x in letters)
            tokens[r] = permute_choice_row(
                tokens[r],
                answer_at=answer_at,
                letter_ids=ids,
                noul_id=self.noul_id,
                line_end_ids=self.line_end_ids,
                perm=draw(batch.index, r, len(ids)),
            )
            permuted += 1
        return _replace_tokens(batch, tokens), permuted


def _replace_tokens(batch: Batch, tokens: np.ndarray) -> Batch:
    import dataclasses

    return dataclasses.replace(batch, tokens=tokens)
