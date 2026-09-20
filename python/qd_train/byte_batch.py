"""Collation planning for rung 0. Torch-free, so the index arithmetic runs in CI.

``docs/training-contract.md`` draws the line this module sits on: *format, ordering,
bucketing, coverage policy ... are torch-free and run in CI; model loading, the fused
cross-entropy and the optimizer step need torch, and their tests ``importorskip``.*

Everything here is padding widths, masks and index arithmetic — which is where the bugs
actually live. :mod:`qd_train.byte_train` turns a :class:`BatchPlan` into tensors and takes
a step; it holds no index arithmetic of its own.

## The two abstain columns are different numbers

A batch carries two slots and each has its own reserved abstention, at a *per-example*
position:

* **choice** — the abstention sits at column ``n_live_options`` for that row, matching
  ``byte_decider.abstain_column`` and therefore ``answer.rs``.
* **span** — the abstention sits at column ``n_lines`` for *that context*, matching
  ``answer.rs``'s ``span_rows = line_count + RESERVED_NOUL_ROWS``.

They are equal only by coincidence. Storing one and reusing it for the other is the shape of
defect this repo has produced five times, so :class:`BatchPlan` carries both explicitly and
:func:`plan_batch` refuses a plan where either target lands outside its own row count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .artifacts import MAX_PADDING_WASTE, SPAN_ABSTAIN
from .mutate_adapter import ByteDecision
from .schema_mirror import RESERVED_NOUL_ROWS
from .trainer import SpanSupervision

__all__ = [
    "MAX_PADDING_WASTE",
    "BatchPlan",
    "padding_waste",
    "plan_batch",
    "span_supervision",
]

# `MAX_PADDING_WASTE` is imported from `qd_train.artifacts`, which owns it and gates S4's
# shards on it. A second copy here would be a second bar: the same name meaning "the S4
# gate" in one module and "a number rung 0 happens to use" in the other, free to drift the
# first time one of them is tuned.


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """One collated batch, as plain Python. No tensors, no torch.

    Lists are row-aligned: index ``i`` of every field describes the same example.
    """

    example_ids: tuple[str, ...]
    #: Byte ids per row, already padded to ``context_width`` with ``ID_PAD``.
    context_ids: tuple[tuple[int, ...], ...]
    context_mask: tuple[tuple[bool, ...], ...]
    #: Option bytes per row per option, padded to ``option_width``.
    option_ids: tuple[tuple[tuple[int, ...], ...], ...]
    option_mask: tuple[tuple[tuple[bool, ...], ...], ...]
    n_live_options: tuple[int, ...]
    #: Target column in the choice softmax. Always a real option for a mutate corpus.
    choice_target: tuple[int, ...]
    #: Line-start offsets into the padded context, per row.
    line_starts: tuple[tuple[int, ...], ...]
    #: Target row in the span softmax: a line index, or the abstain row.
    #:
    #: A line **ordinal** -- the j-th entry of ``line_starts[i]`` -- not a byte offset.
    #: :func:`span_supervision` is the one place that converts between the two, because the
    #: head's contract type carries offsets and mixing the units is this repo's recurring
    #: defect.
    span_target: tuple[int, ...]
    #: The end pointer's target, same units. Equal to ``span_target`` for a single-line
    #: span, and the abstain row exactly when ``span_target`` is.
    span_end_target: tuple[int, ...]
    span_is_noul: tuple[bool, ...]

    @property
    def batch_size(self) -> int:
        return len(self.example_ids)

    @property
    def context_width(self) -> int:
        return len(self.context_ids[0]) if self.context_ids else 0

    def span_rows(self, i: int) -> int:
        """Rows in row ``i``'s span softmax: its line count plus the reserved abstention."""
        return len(self.line_starts[i]) + RESERVED_NOUL_ROWS

    def span_abstain_row(self, i: int) -> int:
        return self.span_rows(i) - RESERVED_NOUL_ROWS

    def choice_rows(self, i: int) -> int:
        return self.n_live_options[i] + RESERVED_NOUL_ROWS

    def choice_abstain_column(self, i: int) -> int:
        return self.choice_rows(i) - RESERVED_NOUL_ROWS


def padding_waste(plan: BatchPlan) -> float:
    """Fraction of context positions that are padding.

    An empty plan has no waste to report and raises, rather than returning 0.0 — a
    0% figure from an empty batch is the ``all([]) is True`` failure one level down.
    """
    if plan.batch_size == 0:
        raise ValueError("an empty batch has no padding waste; 0.0 would read as 'no waste'")
    total = plan.batch_size * plan.context_width
    live = sum(sum(1 for m in row if m) for row in plan.context_mask)
    return (total - live) / total


def plan_batch(
    decisions: Sequence[ByteDecision],
    *,
    pad_id: int,
    max_context_bytes: int,
    max_option_bytes: int,
) -> BatchPlan:
    """Collate decisions into one padded batch.

    Widths are the batch maxima, not the configured maxima: padding every batch to the model
    ceiling would waste most of the compute on short contexts, and ``padding_waste`` exists to
    measure what this choice costs.
    """
    if not decisions:
        raise ValueError("cannot plan an empty batch")

    context_width = max(d.context.n_bytes_kept for d in decisions)
    if context_width > max_context_bytes:
        raise ValueError(
            f"a context of {context_width} bytes exceeds the model's {max_context_bytes}-byte "
            "window; encode with a matching max_bytes rather than truncating here"
        )
    if context_width == 0:
        raise ValueError("every context in this batch is empty; there is nothing to attend to")

    n_options = max(len(d.options) for d in decisions)
    option_width = max((len(o) for d in decisions for o in d.options), default=0)
    if option_width > max_option_bytes:
        raise ValueError(
            f"an option of {option_width} bytes exceeds the {max_option_bytes}-byte budget"
        )
    if option_width == 0:
        raise ValueError("every option in this batch is empty; nothing distinguishes them")

    ctx_ids: list[tuple[int, ...]] = []
    ctx_mask: list[tuple[bool, ...]] = []
    opt_ids: list[tuple[tuple[int, ...], ...]] = []
    opt_mask: list[tuple[tuple[bool, ...], ...]] = []
    live: list[int] = []
    choice_target: list[int] = []
    starts: list[tuple[int, ...]] = []
    span_target: list[int] = []
    span_end_target: list[int] = []
    span_noul: list[bool] = []

    for d in decisions:
        ids = list(d.context.ids)
        pad = context_width - len(ids)
        ctx_ids.append(tuple(ids + [pad_id] * pad))
        ctx_mask.append(tuple([True] * len(ids) + [False] * pad))

        row_ids: list[tuple[int, ...]] = []
        row_mask: list[tuple[bool, ...]] = []
        for option in d.options:
            body = list(option)
            opad = option_width - len(body)
            row_ids.append(tuple(body + [pad_id] * opad))
            row_mask.append(tuple([True] * len(body) + [False] * opad))
        # Dead option columns are padded with a fully masked option rather than dropped, so
        # every row has the same shape; `n_live_options` is what the model masks by.
        for _ in range(n_options - len(d.options)):
            row_ids.append(tuple([pad_id] * option_width))
            row_mask.append(tuple([False] * option_width))
        opt_ids.append(tuple(row_ids))
        opt_mask.append(tuple(row_mask))

        live.append(len(d.options))
        choice_target.append(d.gold_option)
        starts.append(d.context.starts)

        if d.gold_span_line is None:
            abstain_row = len(d.context.starts)  # the reserved abstain row, last
            span_target.append(abstain_row)
            span_end_target.append(abstain_row)
            span_noul.append(True)
        else:
            span_target.append(d.gold_span_line)
            # `ByteDecision.__post_init__` refuses a decision where exactly one end is None,
            # so an end is present whenever a start is.
            assert d.gold_span_end_line is not None
            span_end_target.append(d.gold_span_end_line)
            span_noul.append(False)

    plan = BatchPlan(
        example_ids=tuple(d.example_id for d in decisions),
        context_ids=tuple(ctx_ids),
        context_mask=tuple(ctx_mask),
        option_ids=tuple(opt_ids),
        option_mask=tuple(opt_mask),
        n_live_options=tuple(live),
        choice_target=tuple(choice_target),
        line_starts=tuple(starts),
        span_target=tuple(span_target),
        span_end_target=tuple(span_end_target),
        span_is_noul=tuple(span_noul),
    )
    _validate(plan)
    return plan


def _validate(plan: BatchPlan) -> None:
    """Every target lands inside its own slot's rows. Checked per row, per slot."""
    for i, example_id in enumerate(plan.example_ids):
        if not 1 <= plan.n_live_options[i] <= len(plan.option_ids[i]):
            raise ValueError(
                f"{example_id}: {plan.n_live_options[i]} live options against "
                f"{len(plan.option_ids[i])} columns"
            )
        if not 0 <= plan.choice_target[i] < plan.n_live_options[i]:
            raise ValueError(
                f"{example_id}: choice target {plan.choice_target[i]} is not a live option "
                f"(0..{plan.n_live_options[i] - 1}); the abstain column is "
                f"{plan.choice_abstain_column(i)} and a mutate corpus never targets it"
            )
        rows = plan.span_rows(i)
        for name, target in (
            ("span target", plan.span_target[i]),
            ("span end target", plan.span_end_target[i]),
        ):
            if not 0 <= target < rows:
                raise ValueError(f"{example_id}: {name} {target} is outside {rows} rows")
        if plan.span_end_target[i] < plan.span_target[i]:
            raise ValueError(
                f"{example_id}: the span ends at row {plan.span_end_target[i]} and starts at "
                f"{plan.span_target[i]}; an end before its start is not a span"
            )
        abstaining = plan.span_target[i] == plan.span_abstain_row(i)
        if abstaining != (plan.span_end_target[i] == plan.span_abstain_row(i)):
            raise ValueError(
                f"{example_id}: one pointer abstains and the other does not. A span answer "
                "is one decision; half an abstention is not a state the runtime can render."
            )
        if abstaining != plan.span_is_noul[i]:
            raise ValueError(
                f"{example_id}: span_is_noul is {plan.span_is_noul[i]} but the target "
                f"{plan.span_target[i]} {'is' if abstaining else 'is not'} the abstain row "
                f"{plan.span_abstain_row(i)}. Those are one fact, and they disagree."
            )
        for offset in plan.line_starts[i]:
            if not plan.context_mask[i][offset]:
                raise ValueError(
                    f"{example_id}: line start {offset} falls on padding; a span candidate "
                    "outside the live context can still win the softmax"
                )


def span_supervision(plan: BatchPlan) -> SpanSupervision:
    """Bridge a :class:`BatchPlan` to the span channel :mod:`qd_train.heads` consumes.

    **This is the unit change.** ``BatchPlan.span_target`` is a line *ordinal*; the
    ``SpanSupervision.start``/``end`` that ``plan_span_batch`` reads are token *positions*.
    Handing an ordinal to a field that means a position is the "one name, two quantities"
    defect, so the conversion happens here, once, and nowhere else.

    The round trip through :func:`qd_train.heads.plan_span_batch` -- ordinal to offset here,
    offset back to ordinal there -- is deliberate. ``plan_span_batch`` is the only place the
    abstain row's position is decided and the only place pinned against the runtime; going
    around it to save the conversion would create a second such place.
    """
    if plan.batch_size == 0:
        raise ValueError("an empty batch has no span channel; absent and empty differ")

    k, width = plan.batch_size, plan.context_width
    starts_mask = np.zeros((k, width), dtype=bool)
    start = np.zeros(k, dtype=np.int64)
    end = np.zeros(k, dtype=np.int64)
    query_index = np.zeros(k, dtype=np.int64)

    for i in range(k):
        offsets = plan.line_starts[i]
        starts_mask[i, list(offsets)] = True
        # The pointer query is the last live context byte: the whole sequence has been
        # attended by then, and it is a position every row is guaranteed to have.
        query_index[i] = sum(1 for live in plan.context_mask[i] if live) - 1
        if plan.span_is_noul[i]:
            start[i] = end[i] = SPAN_ABSTAIN
        else:
            start[i] = offsets[plan.span_target[i]]
            end[i] = offsets[plan.span_end_target[i]]

    return SpanSupervision(
        rows=np.arange(k, dtype=np.int64),
        query_index=query_index,
        start=start,
        end=end,
        line_starts=starts_mask,
        abstaining=np.array(plan.span_is_noul, dtype=bool),
    )
