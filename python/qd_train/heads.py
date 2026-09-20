"""The span pointer head: it scores line-start tokens, plus one row for "no evidence".

``qd_train.trainer`` is torch-free and holds no parameters, so the span objective could not
live there -- see ``GAP-TRAINER-SPAN-POINTER-HEAD-NOT-IMPLEMENTABLE-HERE``. This is the
module that owns the parameters, and with them the one thing the trainer deliberately
refused to own: **the head's row layout**.

## The row count is a shape the Rust side already fixes

``crates/qd-runtime/src/answer.rs``::

    pub fn span_rows(context: &Context) -> usize {
        context.line_count() + RESERVED_NOUL_ROWS
    }

and ``crates/qd-runtime/src/schema.rs`` sets ``RESERVED_NOUL_ROWS = 1``. So a span slot's
head ranges over one row per line in *this* context plus one reserved abstain row. This
module re-states that shape in Python, which makes it a **second implementation of a shape
the runtime already fixes** -- exactly the setup that produced
``GAP-RT-WIRE-CONTEXT-ENCODING`` and ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``, both with
green suites on each side. So [`RESERVED_NOUL_ROWS`] and [`span_head_rows`] are pinned
against the Rust source itself by ``test_heads.py``, which reads
``crates/qd-runtime/src/`` and fails if either the constant or the formula moves.

**The abstain row is last.** ``answer.rs`` computes ``noul_row = plan.rows -
RESERVED_NOUL_ROWS`` in four places, so row ``n_candidates`` is the abstention and rows
``0..n_candidates-1`` are the line starts *in ascending token order*. A head trained with
the abstention first and served by a runtime that reads it last would put the model's
"no evidence" mass on the last line of the context, and nothing in the loss would say so.

## Why a bilinear score against the query position

``Batch.target_index`` is documented for a span row as *"the query position the pointer head
reads from"*. So the score of candidate ``j`` is ``<W · h[query], h[j]>`` -- the query's
projected state against each candidate's state. Abstention is scored the same way against a
learned vector, which is what makes it a genuine competitor in one softmax rather than a
separate threshold bolted on afterwards. A separate threshold is the design where "abstain"
and "pick a line" are never actually compared, and the model can be confident in both.

## What is *not* verified here

The gold positions this head trains on arrive from a line -> token mapping several
conversions deep (gold line -> line start in the escaped region -> character offset -> token
index). ``Batch`` checks the gold coincides with a candidate, which catches a mapping that
drifts. It cannot catch a mapping that is *consistently* wrong on a real tokenizer, because
both sides of the check come from the same mapping. See the report and
``GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import torch
from torch import nn

from .trainer import SpanSupervision

__all__ = [
    "RESERVED_NOUL_ROWS",
    "SpanPlan",
    "SpanPointerHead",
    "plan_span_batch",
    "span_head_rows",
]

#: Mirrors ``pub const RESERVED_NOUL_ROWS: usize = 1`` in
#: ``crates/qd-runtime/src/schema.rs``. Pinned against that file by ``test_heads.py``: this
#: is a duplicated constant, and a duplicated constant that nothing compares is a constant
#: that drifts.
RESERVED_NOUL_ROWS: Final[int] = 1


def span_head_rows(n_candidates: int) -> int:
    """Rows a span slot's pointer head ranges over: one per line start, plus the abstention.

    The Python counterpart of ``qd_runtime::answer::span_rows``. ``n_candidates`` is this
    context's line count, which the trainer takes from ``Batch.line_starts`` rather than
    counting newlines itself.
    """
    if not isinstance(n_candidates, (int, np.integer)) or isinstance(n_candidates, bool):
        raise TypeError(f"n_candidates must be int, got {type(n_candidates).__name__}")
    if n_candidates < 1:
        raise ValueError(
            f"n_candidates must be at least 1, got {n_candidates}: a sequence has at least "
            "one line, so an empty candidate set is a mapping failure, not an empty answer"
        )
    return int(n_candidates) + RESERVED_NOUL_ROWS


@dataclass(frozen=True, slots=True)
class SpanPlan:
    """One batch's span rows, laid out in the head's row order.

    ``candidate_pos[k, j]`` is the token position of row ``k``'s ``j``-th line start, in
    ascending order. Columns past ``n_candidates[k]`` are padding and are masked to ``-inf``
    before the softmax, never merely ignored.

    ``gold_start``/``gold_end`` are indices into the head's rows: an ordinal in
    ``[0, n_candidates)`` for a pointing row, and exactly ``n_candidates[k]`` -- the last
    row -- for an abstaining one.
    """

    candidate_pos: torch.Tensor
    candidate_valid: torch.Tensor
    n_candidates: torch.Tensor
    query_index: torch.Tensor
    gold_start: torch.Tensor
    gold_end: torch.Tensor
    abstaining: torch.Tensor

    @property
    def n_spans(self) -> int:
        return int(self.candidate_pos.shape[0])

    @property
    def max_rows(self) -> int:
        """Columns in the score matrix: the widest row's candidates plus the abstention."""
        return int(self.candidate_pos.shape[1]) + RESERVED_NOUL_ROWS


def plan_span_batch(span: SpanSupervision, *, device: torch.device | str = "cpu") -> SpanPlan:
    """Turn the trainer's contract-shaped span channel into this head's row layout.

    This is the only place the abstain row's position is decided, which is why it is also the
    only place ``test_heads.py`` has to pin against the runtime.
    """
    if not isinstance(span, SpanSupervision):
        raise TypeError(f"expected SpanSupervision, got {type(span).__name__}")

    counts = span.candidate_counts()
    if int(counts.min()) < 1:  # pragma: no cover - Batch.__post_init__ refuses this
        raise ValueError("a span row with no line-start candidate reached the head")
    n_spans, max_cand = len(counts), int(counts.max())

    candidate_pos = np.zeros((n_spans, max_cand), dtype=np.int64)
    candidate_valid = np.zeros((n_spans, max_cand), dtype=bool)
    gold_start = np.zeros(n_spans, dtype=np.int64)
    gold_end = np.zeros(n_spans, dtype=np.int64)

    for k in range(n_spans):
        positions = np.flatnonzero(span.line_starts[k])
        candidate_pos[k, : positions.size] = positions
        candidate_valid[k, : positions.size] = True
        if span.abstaining[k]:
            # The reserved row, which `answer.rs` reads as `plan.rows - RESERVED_NOUL_ROWS`.
            gold_start[k] = gold_end[k] = positions.size
            continue
        for name, gold, out in (
            ("start", int(span.start[k]), gold_start),
            ("end", int(span.end[k]), gold_end),
        ):
            hit = np.flatnonzero(positions == gold)
            if hit.size != 1:  # pragma: no cover - Batch.__post_init__ refuses this
                raise ValueError(
                    f"span row {k}: {name} gold at token {gold} is not one of this row's "
                    f"{positions.size} line-start candidates"
                )
            out[k] = int(hit[0])

    def to(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    return SpanPlan(
        candidate_pos=to(candidate_pos, torch.int64),
        candidate_valid=to(candidate_valid, torch.bool),
        n_candidates=to(counts, torch.int64),
        query_index=to(span.query_index, torch.int64),
        gold_start=to(gold_start, torch.int64),
        gold_end=to(gold_end, torch.int64),
        abstaining=to(span.abstaining, torch.bool),
    )


class SpanPointerHead(nn.Module):
    """Two bilinear pointers -- start and end -- over line-start tokens plus an abstain row.

    Parameters are deliberately small: two ``[H, H]`` projections and two ``[H]`` abstain
    vectors. The head's cost is in the gather, not in the parameters, because the candidate
    set is a handful of line starts rather than a vocabulary.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = hidden_size
        self.start_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.end_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Initialised at zero so an untrained head scores abstention exactly at the origin
        # rather than at a random offset that reads as a prior nobody chose.
        self.abstain_start = nn.Parameter(torch.zeros(hidden_size))
        self.abstain_end = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self, hidden: torch.Tensor, plan: SpanPlan
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate and the abstention, for the start and the end pointer.

        Args:
            hidden: ``[K, L, H]`` -- the hidden states of the batch's **span rows only**, in
                the order ``SpanSupervision.rows`` gives.
            plan: the row layout from [`plan_span_batch`].

        Returns:
            ``(start_scores, end_scores)``, each ``[K, max_cand + 1]``. Column ``j`` is the
            ``j``-th line start of that row; column ``n_candidates[k]`` is the abstention;
            columns past it are ``-inf`` and cannot be selected.
        """
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be [K, L, H], got {tuple(hidden.shape)}")
        n_spans, _, hidden_size = hidden.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden's last dim is {hidden_size}, head was built for {self.hidden_size}"
            )
        if n_spans != plan.n_spans:
            raise ValueError(
                f"hidden has {n_spans} span rows but the plan has {plan.n_spans}"
            )

        index = plan.candidate_pos.unsqueeze(-1).expand(-1, -1, hidden_size)
        h_cand = hidden.gather(1, index)  # [K, max_cand, H]
        h_query = hidden.gather(
            1, plan.query_index.view(n_spans, 1, 1).expand(-1, 1, hidden_size)
        ).squeeze(1)  # [K, H]

        out = []
        for proj, abstain in (
            (self.start_proj, self.abstain_start),
            (self.end_proj, self.abstain_end),
        ):
            query = proj(h_query)  # [K, H]
            candidate = torch.einsum("kh,kch->kc", query, h_cand)  # [K, max_cand]
            candidate = candidate.masked_fill(~plan.candidate_valid, float("-inf"))
            abstain_score = query @ abstain.to(query.dtype)  # [K]
            scores = torch.cat(
                [candidate, candidate.new_full((n_spans, RESERVED_NOUL_ROWS), float("-inf"))],
                dim=1,
            )
            # The abstention goes at column n_candidates[k] -- last for that row, matching
            # `noul_row = plan.rows - RESERVED_NOUL_ROWS` in qd-runtime/src/answer.rs.
            scores = scores.scatter(
                1, plan.n_candidates.view(n_spans, 1), abstain_score.view(n_spans, 1)
            )
            out.append(scores)
        return out[0], out[1]

    def loss(
        self,
        hidden: torch.Tensor,
        plan: SpanPlan,
        *,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """Cross-entropy of the start and end pointers against the gold rows.

        ``"mean"`` averages over the ``2K`` pointer decisions, so a span row costs the same
        whether its context has three lines or three hundred -- the alternative weights long
        contexts more simply because their softmax is wider.
        """
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        start_scores, end_scores = self(hidden, plan)
        total = torch.nn.functional.cross_entropy(
            start_scores, plan.gold_start, reduction="sum"
        ) + torch.nn.functional.cross_entropy(end_scores, plan.gold_end, reduction="sum")
        if reduction == "sum":
            return total
        return total / (2 * plan.n_spans)
