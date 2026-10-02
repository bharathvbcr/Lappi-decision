"""Does S4's ``train_ft`` execute, and does its letter channel abstain where the runtime looks?

``tools/rung0_toy_run.py`` answered that question for rung 0 and then named the next one:
*"Before any 8xH100 job it needs what rung 0 just got: a toy run proving ``train_ft``
executes, its loss falls on a memorisable batch, and its letter channel puts the abstention
where ``answer.rs`` reads it."* This tool is that run. It costs nothing on this Mac and it
rents nothing -- see **Rule 4** below.

Three claims, in the order they are established:

1. **It executes.** :func:`qd_train.trainer.train_ft` -- the real loop, not a reimplementation
   -- over a mixed ``choice``/``score``/``span`` batch, on ``mps`` and on ``cpu``, three seeds
   each, with a real optimizer step and a ledger row per run written by the trainer itself.
2. **The loss reaches the floor the corpus admits**, not merely "a lower number". The letter
   channel's floor is the conditional entropy of the gold letter given the prefix the model
   conditions on; the span head's is the same for its two pointers. Both are computed, not
   assumed -- and the computation is *checked* by :func:`_floor_is_exact`, which trains a
   deliberately colliding corpus and shows the run stops at its **non-zero** floor. That check
   is the whole reason "reached the floor" means more than "fell".
3. **The abstention decodes where ``answer.rs`` reads it, for every slot kind** -- and the
   assertion is about the decoded answer, never about the loss.

## Why the third claim cannot be made from a loss curve, and here not even in principle

For a span slot the abstention is a *row* of the pointer head, and
:func:`qd_train.heads.plan_span_batch` places it last. rung 0 demonstrated the counterfactual:
a span head trained abstain-first reaches loss **0.0**, a better floor than the correct
1.43e-06, and the runtime then answers "line ordinal 0" for every no-evidence case.

For a ``choice`` or ``score`` slot it is worse than that, because
:func:`qd_train.trainer.ft_supervision` routes both through the **letter channel**, which is
plain next-token cross-entropy over the vocabulary. That channel never sees a row at all. The
row set is assembled at serve time: ``answer.rs`` issues a ``QueryKind::Letters`` query of
``spec.answer_rows()`` rows and reads the abstention back as
``noul_row = plan.rows - RESERVED_NOUL_ROWS``, and the rows are the slot's rendered letters in
rendered order -- ``qd_data.render._slot_suffix`` appends ``(NOUL_LETTER, NOUL)`` **last**.

So :func:`_counterfactual` does not need two training runs the way rung 0's did. It reads
**one** trained model under two row orders, noul-last and noul-first. The loss is not merely
uninformative about the difference: it is the *same number*, because it is the same run. Under
noul-first every no-evidence answer decodes to the first option and nothing anywhere says so.

## Where the row order comes from

Not from this file. :func:`_render_row_order` builds a real :class:`qd_data.schema.Request`
and calls :func:`qd_data.render.render`, then reads the letters out of the
:class:`qd_data.render.SlotRender` that renderer produced. Restating "A..P then Z" here would
be a second copy of an ordering the renderer already owns, free to drift from it.

## What this tool is NOT

It is not a gate and it promotes nothing. Every row is ``quick`` with a stated reason: a
handful of synthetic rows on a truncated schedule, one batch repeated, against a
randomly-initialised toy transformer. Rule 8 excludes it from every decision.

It does **not** load the 2B backbone and does not touch ``qd_train.shards``: the shard writer
needs the real tokenizer and is another lane's live file. What it proves is that the FT *loop*,
its supervision routing, its span head and its letter channel are wired to the rows the runtime
reads. A wiring bug found here costs 30 seconds; the same bug found on 8xH100 costs the run.

## One metric was meant to be red, and is now green

``rule3_shard_door_checks_the_path`` is :func:`_rule3_door`'s standing measurement of
``GAP-FT-SHARD-DOOR-DOES-NOT-CHECK-THE-PATH``, re-taken on every run so the finding cannot
quietly stop being true in either direction. It recorded ``passed=False`` on every verdict
row this tool wrote before commit ``e089d27``, and that was not a regression signal; it is
deliberately **not** part of the exit status, because conflating "a known open gap is still
open" with "the FT path regressed" would make the exit status useless for the job this tool
exists to do.

``e089d27`` closed the gap, and the prediction that the metric would then "flip to
``passed=True`` by itself" was wrong: the same commit made ``repo_root`` a required keyword,
so this tool died on a ``TypeError`` before writing any row at all. :func:`_rule3_door` says
what had to move with it. The metric is green now because the door refuses, not because the
measurement was relaxed.

RUN
---
    /Users/bharath/.venvs/ml/bin/python tools/ft_toy_run.py

The repo venv deliberately carries no torch, so this refuses there with that instruction
rather than reporting a vacuous pass. Exit status is non-zero if any claim above fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

# Inline for the reason tools/bpe_line_start_collapse.py states: ruff's E402 exemption covers
# `sys.path` modification before the imports, but an ordinary assignment in between is not one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - the repo venv has no torch by design
    raise SystemExit(
        "torch is not importable from this interpreter. The repo venv carries none on "
        "purpose (pyproject: torch is the optional `mac` extra). Run this with\n"
        "    /Users/bharath/.venvs/ml/bin/python tools/ft_toy_run.py\n"
        "Refusing rather than reporting a vacuous pass."
    ) from exc

import numpy as np
from torch import nn

from qd_data.config import DataConfig
from qd_data.render import render
from qd_data.schema import (
    NOUL_LETTER,
    OPTION_LETTERS,
    ChoiceSlot,
    Request,
    ScoreSlot,
    SpanSlot,
)
from qd_train import data_access
from qd_train.artifacts import (
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
    ShardContractViolation,
    ShardHeader,
    assert_shard_trainable,
)
from qd_train.fused_ce import fused_linear_cross_entropy
from qd_train.heads import RESERVED_NOUL_ROWS, SpanPointerHead, plan_span_batch, serving_scores
from qd_train.ledger import (
    DEFAULT_LEDGER_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.optim import apply_lr
from qd_train.run_control import CostEstimate, LRSchedule, RunControl, WallClockCap
from qd_train.trainer import SpanScoringStep, ft_supervision, train_ft
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[1]

#: Hard caps. This tool answers "does it run"; a corpus or schedule large enough to be
#: interesting is large enough to hide a wiring bug behind a plausible-looking curve.
MAX_STEPS = 2_000
MAX_SEEDS = 8

#: How far above a computed floor a converged run may sit. Not a tuned threshold -- the floor
#: is exact and this is the slack a finite schedule leaves. :func:`_floor_is_exact` measures
#: the same slack on a corpus whose floor is non-zero, so the number is calibrated against a
#: case where "reached the floor" and "reached zero" are different statements.
FLOOR_SLACK = 0.05

#: The toy slot shapes. Deliberately *not* maximal: a maximal choice slot is 17 rows
#: (16 named + 1 reserved) and rung 0 already pinned that count. What is under test here is
#: the abstention's POSITION, which is the same question at 4 options as at 16.
N_OPTIONS, N_BINS, N_LINES = 4, 5, 3

# --- the toy vocabulary ------------------------------------------------------------------
#
# A stand-in for the remapped Qwen vocabulary. The only structure that matters is that each
# rendered letter has its own token id, because that is what makes a `QueryKind::Letters`
# decode a slice of `lm_head` -- docs/schema-api.md:106, "1 token over a 17-row lm_head slice".

ID_PAD = 0
ID_KIND = {SLOT_CHOICE: 1, SLOT_SCORE: 2, SLOT_SPAN: 3}
ID_ANSWER_MARK = 4  # stands for `<|qd_answer|>`: the position whose next token is the gold
ID_LINE = 5  # stands for a line-start token in the rendered context
ID_ROW_BASE = 8  # one distinct token per corpus row, so the rows are separable at all
ID_LETTER_BASE = 64
LETTER_ID: dict[str, int] = {
    letter: ID_LETTER_BASE + i for i, letter in enumerate([*OPTION_LETTERS, NOUL_LETTER])
}
VOCAB = ID_LETTER_BASE + len(LETTER_ID)

HIDDEN, N_HEADS, WIDTH = 32, 4, 12


# --- the row order the runtime reads -----------------------------------------------------


def _render_row_order() -> dict[str, list[str]]:
    """The letter rows of a choice and a score slot, taken from the renderer itself.

    ``answer.rs`` sets a choice slot's ``query.rows`` to ``options.len() + RESERVED_NOUL_ROWS``
    and reads the abstention at ``plan.rows - RESERVED_NOUL_ROWS``. Which letter sits at which
    row is decided by ``qd_data.render._slot_suffix``, which appends ``(NOUL_LETTER, NOUL)``
    after the named options -- so ``list(SlotRender.letter_to_value)`` **is** the row order.

    Read rather than restated. A second copy of "A..P then Z" in this file would be free to
    drift from the renderer, and the drift would look exactly like a correct tool.
    """
    request = Request(
        task="ft-toy",
        context=b"line one\nline two\nline three\n",
        question="which one",
        slots=(
            ChoiceSlot(name="choice", options=tuple(f"opt{i}" for i in range(N_OPTIONS))),
            ScoreSlot(name="score", bins=N_BINS),
            SpanSlot(name="span"),
        ),
        example_id="ft-toy-0",
    )
    rendered = render(request)
    return {slot.name: list(slot.letter_to_value) for slot in rendered.slots}


def _letter_rows(letters: list[str], *, noul_first: bool) -> list[int]:
    """Token ids of one slot's decode rows, in the order the runtime will read them.

    ``noul_first`` builds the defect :func:`_counterfactual` serves: the same rows in the order
    a renderer that listed ``noul`` first would produce. Nothing else about the run changes.
    """
    named = [LETTER_ID[x] for x in letters if x != NOUL_LETTER]
    noul = LETTER_ID[NOUL_LETTER]
    return [noul, *named] if noul_first else [*named, noul]


# --- the model -----------------------------------------------------------------------------


class _Block(nn.Module):
    """One pre-norm causal transformer block. Causal because the letter channel reads
    ``hidden[target_index]`` and the span head reads ``hidden[line_start]``: a position that
    could see its own gold would memorise nothing and report a floor of zero anyway."""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden {hidden} is not divisible by heads {heads}")
        self.heads = heads
        self.ln_attn = nn.LayerNorm(hidden)
        self.ln_mlp = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        shape = (b, t, self.heads, h // self.heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(b, t, h))
        return x + self.mlp(self.ln_mlp(x))


class ToyFtStep:
    """A :class:`qd_train.trainer.SpanScoringStep` over a one-block causal transformer.

    The two losses are the repo's own: :func:`qd_train.fused_ce.fused_linear_cross_entropy`
    for the letter channel and :meth:`qd_train.heads.SpanPointerHead.loss` for the pointers.
    Writing either one here would test this file instead of the code under test.

    ``accumulate_span`` receives the **whole** supervision -- letter channel included -- which
    is the seam's contract, so a mixed batch trains both channels in one backward.
    """

    def __init__(self, *, seed: int, device: str) -> None:
        torch.manual_seed(seed)
        self.device = device
        self.embed = nn.Embedding(VOCAB, HIDDEN).to(device)
        self.position = nn.Parameter(torch.zeros(WIDTH, HIDDEN, device=device))
        self.block = _Block(HIDDEN, N_HEADS).to(device)
        self.ln_f = nn.LayerNorm(HIDDEN).to(device)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(device)
        self.span_head = SpanPointerHead(HIDDEN).to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        #: Per-micro-batch component losses. `TrainResult.loss_log` carries only the combined
        #: number, and a falling combined loss with a flat span term is a model that learned
        #: the letter and nothing about *where* -- the failure rung 0 carried two logs for.
        self.letter_log: list[float] = []
        self.span_log: list[float] = []

    def parameters(self) -> list[torch.nn.Parameter]:
        return [
            *self.embed.parameters(),
            self.position,
            *self.block.parameters(),
            *self.ln_f.parameters(),
            *self.lm_head.parameters(),
            *self.span_head.parameters(),
        ]

    def hidden(self, tokens: np.ndarray) -> torch.Tensor:
        ids = torch.as_tensor(tokens.astype(np.int64), device=self.device)
        return self.ln_f(self.block(self.embed(ids) + self.position[: ids.shape[1]]))

    def _letter_loss(self, hidden: torch.Tensor, supervision) -> torch.Tensor | None:
        """``None`` when the batch is all-span: its letter channel is legitimately empty."""
        if supervision.n_supervised == 0:
            return None
        return fused_linear_cross_entropy(
            hidden[:, :-1],
            self.lm_head.weight,
            torch.as_tensor(supervision.targets.astype(np.int64), device=self.device),
            mask=torch.as_tensor(supervision.mask.copy(), device=self.device),
            chunk_size=8,
        )

    def accumulate(self, batch: Batch, supervision) -> float:
        loss = self._letter_loss(self.hidden(batch.tokens), supervision)
        if loss is None:  # pragma: no cover - `_refuse_unsupervised_rows` refuses this first
            raise RuntimeError("a span-free batch reached accumulate with no supervised token")
        loss.backward()
        self.letter_log.append(float(loss.item()))
        self.span_log.append(0.0)
        return float(loss.item())

    def accumulate_span(self, batch: Batch, supervision) -> float:
        span = supervision.span
        if span is None:  # pragma: no cover - `_train` only routes here when span is not None
            raise RuntimeError("accumulate_span was handed a supervision with no span channel")
        # One forward over the full width. Causal attention makes `hidden[:, :-1]` identical to
        # a forward over `tokens[:, :-1]`, and the span head indexes the full-width positions
        # that `Batch.line_starts` marks -- so two passes would differ only in cost.
        hidden = self.hidden(batch.tokens)
        plan = plan_span_batch(span, device=self.device)
        rows = torch.as_tensor(span.rows, device=self.device)
        span_loss = self.span_head.loss(hidden[rows], plan)
        letter = self._letter_loss(hidden, supervision)
        total = span_loss if letter is None else span_loss + letter
        total.backward()
        self.letter_log.append(0.0 if letter is None else float(letter.item()))
        self.span_log.append(float(span_loss.item()))
        return float(total.item())

    def apply(self, *, lr: float) -> None:
        # `qd_train.optim.apply_lr`, the one writer of group["lr"]; see its docstring.
        apply_lr(self.optimizer, lr)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state(self) -> dict[str, object]:
        return {"micro_batches": len(self.letter_log)}

    def load_state(self, state) -> None:
        """This tool never resumes; a checkpoint it cannot restore would be a silent lie."""
        raise NotImplementedError(
            "ToyFtStep does not implement resume. tools/ft_toy_run.py runs one epoch from "
            "scratch, and a load_state that quietly did nothing would let a resumed run "
            "report the checkpoint's loss log against freshly-initialised parameters."
        )


# --- the corpus -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _Row:
    """One toy example: which slot kind, what it answers, and whether it abstains."""

    kind: int
    n_named: int
    row_token: int
    gold_letter: str
    abstains: bool
    span_lines: tuple[int, int] | None


def _rows(*, collide: bool) -> list[_Row]:
    """Twelve rows: four of each slot kind, the last of each abstaining.

    A corpus of only-abstaining or only-answering rows would let a head that always abstains,
    or never does, reach a floor that reads as learning. One abstention per kind is the
    minimum that makes the abstain row a *competitor* rather than the only answer.

    ``collide`` gives two choice rows the same row token, so they share a prefix and carry
    different golds. That corpus is not memorisable and its floor is above zero by exactly
    its conditional label entropy -- which is what :func:`_floor_is_exact` measures.
    """
    out: list[_Row] = []
    index = 0
    for kind, n_named in ((SLOT_CHOICE, N_OPTIONS), (SLOT_SCORE, N_BINS)):
        for j in range(4):
            abstains = j == 3
            token = ID_ROW_BASE + index
            if collide and kind == SLOT_CHOICE and j in (0, 1):
                token = ID_ROW_BASE
            out.append(
                _Row(
                    kind=kind,
                    n_named=n_named,
                    row_token=token,
                    gold_letter=NOUL_LETTER if abstains else OPTION_LETTERS[j],
                    abstains=abstains,
                    span_lines=None,
                )
            )
            index += 1
    for j in range(4):
        abstains = j == 3
        out.append(
            _Row(
                kind=SLOT_SPAN,
                n_named=0,
                row_token=ID_ROW_BASE + index,
                # `qd_train.shards` gives every span row the noul letter as its trailing
                # token and relies on `slot_kind` to keep it out of the letter channel;
                # mirrored here so `ft_supervision`'s routing is under test, not bypassed.
                gold_letter=NOUL_LETTER,
                abstains=abstains,
                span_lines=None if abstains else (j % N_LINES, min(j % N_LINES + 1, N_LINES - 1)),
            )
        )
        index += 1
    return out


def _batch(rows: list[_Row], *, index: int = 0) -> Batch:
    """Collate to the contract shape. Layout per row::

        [row_token, kind, LINE, l0, LINE, l1, LINE, l2, ANSWER_MARK, gold_letter]

    ``target_index`` names the ``ANSWER_MARK`` position, because ``Batch`` defines it as
    *"the position whose next token is the gold letter"* -- read from the batch, never
    inferred as ``lengths - 2``.
    """
    n = len(rows)
    tokens = np.zeros((n, WIDTH), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)
    kinds = np.zeros(n, dtype=np.uint8)
    target_index = np.zeros(n, dtype=np.int32)
    span_target = np.full((n, 2), NO_SPAN, dtype=np.int32)
    line_starts = np.zeros((n, WIDTH), dtype=bool)

    for i, row in enumerate(rows):
        sequence = [row.row_token, ID_KIND[row.kind]]
        starts: list[int] = []
        for line in range(N_LINES):
            starts.append(len(sequence))
            sequence += [ID_LINE, ID_LINE + 1 + line]
        sequence += [ID_ANSWER_MARK, LETTER_ID[row.gold_letter]]
        if len(sequence) > WIDTH:  # pragma: no cover - WIDTH is sized for this layout
            raise ValueError(f"row {i} needs {len(sequence)} tokens, WIDTH is {WIDTH}")
        tokens[i, : len(sequence)] = sequence
        lengths[i] = len(sequence)
        kinds[i] = row.kind
        target_index[i] = len(sequence) - 2
        if row.kind == SLOT_SPAN:
            line_starts[i, starts] = True
            if row.abstains:
                span_target[i] = (SPAN_ABSTAIN, SPAN_ABSTAIN)
            else:
                first, last = row.span_lines  # type: ignore[misc]
                span_target[i] = (starts[first], starts[last])

    return Batch(
        tokens=tokens,
        lengths=lengths,
        bucket=0,
        index=index,
        slot_kind=kinds,
        target_index=target_index,
        span_target=span_target,
        line_starts=line_starts,
    )


# --- the floors ------------------------------------------------------------------------------


def _entropy(golds: list[object]) -> float:
    """Sum of ``-log p`` over a group; the optimum of cross-entropy on identical conditions."""
    counts: dict[object, int] = {}
    for gold in golds:
        counts[gold] = counts.get(gold, 0) + 1
    total = len(golds)
    return -sum(c * math.log(c / total) for c in counts.values())


def _letter_floor(batch: Batch) -> float:
    """The lowest letter-channel loss this corpus admits, given what the model conditions on.

    Grouped by the **exact prefix** ``tokens[i, :target_index[i] + 1]``, because that is what a
    causal model sees before it must emit the gold. Two rows with identical prefixes and
    different golds cannot both be fitted, and the optimum over such a group is its empirical
    label distribution. Divided by the supervised count to match
    :func:`qd_train.fused_ce.fused_linear_cross_entropy`'s default ``reduction="mean"``.
    """
    supervision = ft_supervision(batch)
    groups: dict[tuple[int, ...], list[object]] = {}
    supervised = 0
    for i in range(batch.tokens.shape[0]):
        if not supervision.mask[i].any():
            continue
        supervised += 1
        at = int(batch.target_index[i])  # type: ignore[index]
        key = tuple(int(x) for x in batch.tokens[i, : at + 1])
        groups.setdefault(key, []).append(int(batch.tokens[i, at + 1]))
    if supervised == 0:
        return 0.0
    return sum(_entropy(g) for g in groups.values()) / supervised


def _span_floor(batch: Batch) -> float:
    """The same, for the pointer head, over ``2K`` decisions to match its ``"mean"``.

    The two pointers are grouped separately: a corpus can pin a span's start and be ambiguous
    about its end, and averaging the pair first would hide that.
    """
    supervision = ft_supervision(batch)
    if supervision.span is None:
        return 0.0
    plan = plan_span_batch(supervision.span)
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for k, row in enumerate(supervision.span.rows):
        at = int(supervision.span.query_index[k])
        key = tuple(int(x) for x in batch.tokens[row, : at + 1])
        groups.setdefault(key, []).append((int(plan.gold_start[k]), int(plan.gold_end[k])))
    total = sum(
        _entropy([g[0] for g in group]) + _entropy([g[1] for g in group])
        for group in groups.values()
    )
    return total / (2 * plan.n_spans)


# --- reading the trained model the way the runtime does ----------------------------------------


def _decode(step: ToyFtStep, batch: Batch, rows: list[_Row], order: dict[str, list[str]],
            *, noul_first: bool = False) -> dict[str, object]:
    """Decode every row the way ``crates/qd-runtime/src/answer.rs`` decodes it.

    * ``choice`` and ``score``: a ``QueryKind::Letters`` query is a slice of ``lm_head`` at this
      slot's rendered letters, so the rows are the letter logits at ``target_index`` and the
      verdict is ``argmax``. Abstention iff ``top == rows - RESERVED_NOUL_ROWS``.
    * ``span``: two ``PointerStart``/``PointerEnd`` reads, taken through
      :func:`qd_train.heads.serving_scores` -- the slice the runtime will actually accept, so
      no padded column the runtime never sees can win.

    Returns one verdict per row plus the counts, and never consults the loss.
    """
    with torch.no_grad():
        hidden = step.hidden(batch.tokens)
        logits = step.lm_head(hidden)
        supervision = ft_supervision(batch)
        plan = plan_span_batch(supervision.span, device=step.device)
        start_scores, end_scores = step.span_head(
            hidden[torch.as_tensor(supervision.span.rows, device=step.device)], plan
        )
        start_rows = serving_scores(start_scores, plan)
        end_rows = serving_scores(end_scores, plan)

    verdicts: list[dict[str, object]] = []
    span_k = 0
    for i, row in enumerate(rows):
        if row.kind == SLOT_SPAN:
            n_rows = int(plan.runtime_rows[span_k])
            noul_row = n_rows - RESERVED_NOUL_ROWS
            top_start = int(start_rows[span_k].argmax())
            top_end = int(end_rows[span_k].argmax())
            abstained = top_start == noul_row or top_end == noul_row
            verdicts.append({
                "row": i,
                "kind": "span",
                "expected_abstain": row.abstains,
                "rows": n_rows,
                "noul_row": noul_row,
                "top": [top_start, top_end],
                "runtime_verdict": "abstain" if abstained else f"lines {top_start}..{top_end}",
                "correct": abstained == row.abstains,
            })
            span_k += 1
            continue
        letters = order["choice" if row.kind == SLOT_CHOICE else "score"]
        row_tokens = _letter_rows(letters, noul_first=noul_first)
        noul_row = len(row_tokens) - RESERVED_NOUL_ROWS
        at = int(batch.target_index[i])  # type: ignore[index]
        top = int(logits[i, at, torch.as_tensor(row_tokens, device=step.device)].argmax())
        abstained = top == noul_row
        verdicts.append({
            "row": i,
            "kind": "choice" if row.kind == SLOT_CHOICE else "score",
            "expected_abstain": row.abstains,
            "rows": len(row_tokens),
            "noul_row": noul_row,
            "top": top,
            "runtime_verdict": "abstain" if abstained else f"option ordinal {top}",
            "correct": abstained == row.abstains,
        })

    by_kind: dict[str, dict[str, int]] = {}
    for verdict in verdicts:
        bucket = by_kind.setdefault(
            str(verdict["kind"]),
            {"n": 0, "correct": 0, "abstaining": 0, "abstaining_correct": 0},
        )
        bucket["n"] += 1
        bucket["correct"] += int(bool(verdict["correct"]))
        bucket["abstaining"] += int(bool(verdict["expected_abstain"]))
        bucket["abstaining_correct"] += int(
            bool(verdict["expected_abstain"]) and bool(verdict["correct"])
        )

    # Split by channel, because `noul_first` reorders the LETTER rows and nothing else. A span
    # slot has no letter alphabet beyond noul (`_slot_suffix`: its `pairs` are empty), so its
    # abstention is a pointer row and is untouched by the defect. Counting all three kinds
    # together would make the correct run look like a partial failure -- which is what the
    # first version of this tool recorded: `passed=False` on a healthy run, in a ledger row,
    # while the tool itself exited 0.
    letter_kinds = ("choice", "score")
    return {
        "noul_first": noul_first,
        "verdicts": verdicts,
        "by_kind": by_kind,
        "abstaining_rows": sum(1 for v in verdicts if v["expected_abstain"]),
        "abstaining_decoded_as_abstain": sum(
            1 for v in verdicts if v["expected_abstain"] and v["correct"]
        ),
        "letter_abstaining_rows": sum(
            by_kind.get(k, {}).get("abstaining", 0) for k in letter_kinds
        ),
        "letter_abstaining_decoded_as_abstain": sum(
            by_kind.get(k, {}).get("abstaining_correct", 0) for k in letter_kinds
        ),
        "span_abstaining_rows": by_kind.get("span", {}).get("abstaining", 0),
        "span_abstaining_decoded_as_abstain": by_kind.get("span", {}).get(
            "abstaining_correct", 0
        ),
        "all_correct": all(bool(v["correct"]) for v in verdicts),
    }


# --- the runs ------------------------------------------------------------------------------


def _counterfactual_holds(shipped: dict[str, object], defect: dict[str, object]) -> bool:
    """Did reading one trained model under the noul-first row order flip every letter answer?

    Three conditions, and all three matter. Under the shipped order every row must decode as
    the runtime reads it. Under noul-first **no** choice or score abstention may survive --
    one that did would mean the row order is not what decided the answer. And the span
    abstention must be *unchanged*, because it is a pointer row that the letter reordering
    does not touch; if it moved, the two arms differ in more than the thing under test and the
    comparison proves nothing.

    The single owner of this verdict. :func:`main` and :func:`_record_verdict` both call it, so
    the exit status and the ledger row cannot disagree -- which they did in the first version
    of this tool, where the row said ``passed=False`` on a run the tool exited 0 on.
    """
    if not shipped["all_correct"]:
        return False
    if int(shipped["letter_abstaining_rows"]) < 1:  # type: ignore[arg-type]
        return False  # nothing to flip: the corpus never exercises a letter abstention
    if shipped["letter_abstaining_decoded_as_abstain"] != shipped["letter_abstaining_rows"]:
        return False
    if int(defect["letter_abstaining_decoded_as_abstain"]) != 0:  # type: ignore[arg-type]
        return False
    return (
        defect["span_abstaining_decoded_as_abstain"]
        == shipped["span_abstaining_decoded_as_abstain"]
    )


def _control(steps: int, *, device: str) -> RunControl:
    """The cap, the schedule and the price of a local run.

    Zero is a measured fact about this host, not a way around rule 4: a Mac that is already
    bought costs nothing per hour, so ``CostEstimate.requires_human_approval`` is False and
    nothing is rented. This used to say *"a run on a rented machine sets a real rate here"*
    and nothing made it -- ``device`` is a parameter, and the literal priced whatever
    arrived as a Mac. :meth:`CostEstimate.for_device` is what makes it true: cpu and mps are
    priced exactly as before, and anything else is refused until a caller states the rate.
    """
    cap = WallClockCap(cap_s=900.0)
    return RunControl(
        schedule=LRSchedule(
            peak_lr=3e-3, total_steps=steps, warmup_steps=max(1, steps // 20), min_lr=3e-4
        ),
        cap=cap,
        cost=CostEstimate.for_device(cap=cap, device=device),
        grad_accum=1,
    )


def _protocol(*, seed: int, device: str, steps: int, rows: list[_Row], tag: str) -> Protocol:
    """Real hashes, no markers. The corpus, the codec and the architecture are all synthetic
    but all identifiable, so each is hashed rather than marked ``n/a``."""

    def digest(obj: object) -> str:
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return Protocol(
        data_snapshot_hash=digest([dataclasses.asdict(r) for r in rows]),
        tokenizer_hash=f"toy:{digest({'letters': LETTER_ID, 'pad': ID_PAD, 'vocab': VOCAB})}",
        backbone_commit=f"scratch:{digest({'h': HIDDEN, 'heads': N_HEADS, 'width': WIDTH})}",
        recipe_hash=digest(_recipe(steps=steps, device=device, tag=tag)),
        seed=seed,
    )


def _recipe(*, steps: int, device: str, tag: str) -> dict[str, object]:
    """The settings that identify one toy ft run from another, in one place.

    One owner because two callers need it: the hash above, which makes two recipes
    incomparable, and the row below, which says what they were. Built twice they could
    drift, and a row whose stored recipe does not hash to its own `recipe_hash` is worse
    than a row with no recipe at all.
    """
    return {"steps": steps, "device": device, "tool": "ft_toy", "tag": tag}


def _recorder(
    ledger: Ledger, *, seed: int, device: str, steps: int, rows: list[_Row], tag: str,
    run_kind: str, notes: str, wall_clock_s: float | None,
) -> RunRecorder:
    """Both row kinds go through here and they need different answers.

    ``None`` from the ``ft`` path, whose ``with`` block contains ``train_ft``. A measured
    decode elapsed from :func:`_record_verdict`, which describes work that finished before
    its recorder existed. 72 of this tool's 156 rows recorded the time taken to write the
    row, and all 72 were verdict rows.
    """
    return RunRecorder(
        ledger,
        entry_point=Path(__file__),
        protocol=_protocol(seed=seed, device=device, steps=steps, rows=rows, tag=tag),
        run_kind=run_kind,  # type: ignore[arg-type]
        repo=REPO,
        # The device this run chose, not the best one this host offers.
        env=Environment.detect(device=device),
        wall_clock_s=wall_clock_s,
        # Through `_control` rather than a second `for_device` call beside it, so there is
        # one spelling of what a run on this device costs and the row cannot disagree with
        # the estimate that gated it. Construction is pure; nothing is armed until start().
        cost=_control(steps, device=device).cost,
        recipe=_recipe(steps=steps, device=device, tag=tag),
        quick=True,
        quick_reason=(
            f"toy FT run: {steps} optimizer steps over {len(rows)} synthetic rows on {device}, "
            "one batch repeated, randomly-initialised toy transformer. Rule 8: excluded from "
            "every decision."
        ),
        notes=notes,
    )


def _train_once(*, device: str, seed: int, steps: int, rows: list[_Row], ledger: Ledger,
                tag: str) -> dict[str, object]:
    """One run of the real ``train_ft`` on one device with one seed."""
    batch = _batch(rows)
    step = ToyFtStep(seed=seed, device=device)
    if not isinstance(step, SpanScoringStep):  # pragma: no cover - the protocol is structural
        raise TypeError(
            "ToyFtStep does not satisfy SpanScoringStep, so train_ft would refuse every batch "
            "carrying a span row rather than training it -- and the refusal would read as a "
            "corpus problem rather than a step that lost a method"
        )
    recorder = _recorder(
        ledger, seed=seed, device=device, steps=steps, rows=rows, tag=tag, run_kind="ft",
        # None: the block below contains train_ft, so the recorder's lifetime IS the run.
        wall_clock_s=None,
        notes=(
            f"tools/ft_toy_run.py [{tag}] -- does qd_train.trainer.train_ft execute, and does "
            "its loss reach the floor this corpus admits. Not an evaluation of any model."
        ),
    )
    letter_floor, span_floor = _letter_floor(batch), _span_floor(batch)
    for name, value, detail in (
        ("corpus.letter_floor", letter_floor, "conditional entropy of the gold letter"),
        ("corpus.span_floor", span_floor, "conditional entropy of the two gold pointers"),
        ("corpus.rows", len(rows), "synthetic rows, one batch repeated every step"),
        ("corpus.span_rows", sum(1 for r in rows if r.kind == SLOT_SPAN), "SLOT_SPAN rows"),
        ("corpus.abstaining_rows", sum(1 for r in rows if r.abstains), "rows whose gold is noul"),
    ):
        recorder.metric(name, Ran(passed=True, value=value, detail=detail))

    def source():
        for i in range(steps):
            yield dataclasses.replace(batch, index=i)

    started = time.monotonic()
    result = train_ft(source(), epoch=0, step=step, control=_control(steps, device=device),
                      recorder=recorder)
    wall = time.monotonic() - started
    losses = result.loss_log.losses()
    return {
        "tag": tag,
        "device": device,
        "seed": seed,
        "steps_requested": steps,
        "optimizer_steps": result.optimizer_steps,
        "termination": result.termination,
        "supervised_tokens": result.supervised_tokens,
        "span_rows": result.span_rows,
        "params": sum(p.numel() for p in step.parameters()),
        "letter_first": step.letter_log[0],
        "letter_last": step.letter_log[-1],
        "letter_floor": letter_floor,
        "span_first": step.span_log[0],
        "span_last": step.span_log[-1],
        "span_floor": span_floor,
        "total_first": losses[0],
        "total_last": losses[-1],
        "wall_clock_s": wall,
        "steps_per_s": result.optimizer_steps / wall if wall > 0 else float("inf"),
        "ft_row_id": result.row_id,
        "_step": step,
        "_batch": batch,
    }


def _floor_is_exact(*, steps: int, ledger: Ledger) -> dict[str, object]:
    """Train a corpus whose floor is **not** zero, and check the run stops there.

    Without this, "the loss reached its floor" is a claim about a number that happens to be
    zero, and a floor computed wrongly would agree with a run that simply converged. rung 0
    found this the hard way: its first corpus reused one body across rows while cycling the
    gold, the loss stopped at 0.5199, and the computed conditional entropy was 0.519860 -- the
    head was right and the data was unlearnable. Here the collision is built on purpose.
    """
    rows = _rows(collide=True)
    run = _train_once(device="cpu", seed=0, steps=steps, rows=rows, ledger=ledger,
                      tag="floor-check")
    gap = run["letter_last"] - run["letter_floor"]  # type: ignore[operator]
    return {
        "letter_floor": run["letter_floor"],
        "letter_last": run["letter_last"],
        "gap": gap,
        "floor_is_above_zero": run["letter_floor"] > 0.0,  # type: ignore[operator]
        "reached": gap <= FLOOR_SLACK,
        "ft_row_id": run["ft_row_id"],
    }


def _rule3_door() -> dict[str, object]:
    """Rule 3 at the door the FT path actually opens. Measured, not read.

    ``qd_train.shards.ShardReader.__init__`` calls
    :func:`qd_train.artifacts.assert_shard_trainable`, whose docstring says it exists because
    *"without this the held-out check is satisfied on paper and bypassed by indirection"*.

    When this was written that door checked the header's **declared** split and not the shard
    set's **path**, and ``ShardReader`` took a ``repo_root`` it never read. Commit ``e089d27``
    closed that -- ``repo_root`` is now a **required** keyword and the first thing the door
    does is call ``assert_path_not_held_out``. Both halves of this measurement had to move
    with it, and neither is cosmetic:

    * the call gained ``repo_root``, without which the whole tool died on a ``TypeError``
      before any ledger row was written;
    * the ``except`` gained ``HeldOutViolation``, because the door now refuses **through the
      path check**, which raises that and not ``ShardContractViolation``. With only the
      keyword added the refusal escapes this function and the tool still exits 1.

    So the metric does not "flip to ``passed=True`` by itself" when the gap closes, as this
    tool's own docstring predicted -- a required-keyword change takes the measurement with
    it. That is why ``python/tests/test_tool_call_sites.py`` exists.
    """
    config = DataConfig()
    header = ShardHeader(
        split="train",
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        remap_hash="r" * 64,
        vocab_size=VOCAB,
        n_sequences=4,
        total_tokens=64,
        max_seq_len=16,
        buckets=(16,),
    )
    under_holdout = REPO / "data" / "heldout" / "shards-train"

    path_check_refuses = False
    try:
        data_access.assert_path_not_held_out(under_holdout, config=config, repo_root=REPO)
    except data_access.HeldOutViolation:
        path_check_refuses = True

    shard_door_refuses = False
    try:
        assert_shard_trainable(
            header, config=config, path=under_holdout, repo_root=REPO
        )
    except (ShardContractViolation, data_access.HeldOutViolation):
        shard_door_refuses = True

    return {
        "path": str(under_holdout),
        "header_split": header.split,
        "assert_path_not_held_out": "REFUSES" if path_check_refuses else "ADMITS",
        "assert_shard_trainable": "REFUSES" if shard_door_refuses else "ADMITS",
        "agree": path_check_refuses == shard_door_refuses,
    }


def _record_verdict(
    run: dict[str, object], *, ledger: Ledger, shipped: dict[str, object],
    defect: dict[str, object], floor_check: dict[str, object], door: dict[str, object],
    rows: list[_Row], steps: int, decode_s: float,
) -> str:
    """One row per run for what happened **after** the last optimizer step.

    Separate from the ``ft`` row because ``train_ft`` owns its recorder's context manager and
    writes on exit -- so a fact that only exists once training has finished cannot be in it.
    This row names that row's id, and carries no number the ``ft`` row already carries.
    """
    recorder = _recorder(
        ledger, seed=int(run["seed"]), device=str(run["device"]), steps=steps, rows=rows,
        tag="verdict", run_kind="smoke",
        # The decode this row reports on, not the parent run's duration: an ft row and a
        # verdict row each claiming the same seconds would double-count the time spent.
        wall_clock_s=decode_s,
        notes=(
            f"tools/ft_toy_run.py verdict for ft row {run['ft_row_id']} "
            f"({run['device']} seed={run['seed']}): where the abstention decoded, read the way "
            "crates/qd-runtime/src/answer.rs reads it. Not an evaluation of any model."
        ),
    )
    with recorder:
        recorder.metric(
            "ft_run_row_id",
            Ran(passed=True, value=run["ft_row_id"], detail="the train_ft row this verdict is of"),
        )
        recorder.metric(
            "letter_loss_reached_its_floor",
            Ran(
                passed=run["letter_last"] <= run["letter_floor"] + FLOOR_SLACK,  # type: ignore[operator]
                value=run["letter_last"] - run["letter_floor"],  # type: ignore[operator]
                detail=(
                    f"letter {run['letter_last']:.6f} against this corpus's conditional label "
                    f"entropy {run['letter_floor']:.6f}; bar is floor + {FLOOR_SLACK}"
                ),
            ),
        )
        recorder.metric(
            "span_loss_reached_its_floor",
            Ran(
                passed=run["span_last"] <= run["span_floor"] + FLOOR_SLACK,  # type: ignore[operator]
                value=run["span_last"] - run["span_floor"],  # type: ignore[operator]
                detail=(
                    f"span {run['span_last']:.6f} against its pointer entropy "
                    f"{run['span_floor']:.6f}; bar is floor + {FLOOR_SLACK}"
                ),
            ),
        )
        recorder.metric(
            "both_channels_fell_separately",
            Ran(
                passed=(
                    run["letter_last"] < run["letter_first"]  # type: ignore[operator]
                    and run["span_last"] < run["span_first"]  # type: ignore[operator]
                ),
                value=(
                    f"letter {run['letter_first']:.4f}->{run['letter_last']:.6f}, "
                    f"span {run['span_first']:.4f}->{run['span_last']:.6f}"
                ),
                detail="a falling total with a flat span term learned the letter, not the place",
            ),
        )
        for kind, counts in shipped["by_kind"].items():  # type: ignore[union-attr,index]
            recorder.metric(
                f"decoded_as_the_runtime_reads_it.{kind}",
                Ran(
                    passed=counts["correct"] == counts["n"],
                    value=counts["correct"],
                    n=counts["correct"],
                    n_total=counts["n"],
                    detail=(
                        f"{counts['abstaining']} of {counts['n']} {kind} rows abstain; verdict "
                        "read at rows - RESERVED_NOUL_ROWS, never from the loss"
                    ),
                ),
            )
        recorder.metric(
            "abstain_is_last_or_the_runtime_answers_a_line",
            Ran(
                passed=_counterfactual_holds(shipped, defect),
                value=(
                    f"letter abstentions: noul-last "
                    f"{shipped['letter_abstaining_decoded_as_abstain']}/"
                    f"{shipped['letter_abstaining_rows']}, noul-first "
                    f"{defect['letter_abstaining_decoded_as_abstain']}/"
                    f"{defect['letter_abstaining_rows']}; span abstention unchanged at "
                    f"{shipped['span_abstaining_decoded_as_abstain']}/"
                    f"{shipped['span_abstaining_rows']}"
                ),
                detail=(
                    "one trained model read under two row orders, so the loss is not merely "
                    "uninformative about the difference -- it is the same number. The span "
                    "abstention is a pointer row, not a letter row, so noul-first must leave "
                    "it alone; if it moved, the two orders are not differing only in what "
                    "this measures"
                ),
            ),
        )
        recorder.metric(
            "floor_formula_is_exact",
            Ran(
                passed=bool(floor_check["floor_is_above_zero"] and floor_check["reached"]),
                value=floor_check["gap"],
                detail=(
                    f"a deliberately colliding corpus has floor "
                    f"{floor_check['letter_floor']:.6f} and its run stopped at "
                    f"{floor_check['letter_last']:.6f} (row {floor_check['ft_row_id']})"
                ),
            ),
        )
        recorder.metric(
            "rule3_shard_door_checks_the_path",
            Ran(
                passed=bool(door["agree"]),
                value=f"{door['assert_shard_trainable']} where the path check "
                      f"{door['assert_path_not_held_out']}",
                detail=(
                    f"{door['path']} with a header declaring split={door['header_split']!r}. "
                    "GAP-FT-SHARD-DOOR-DOES-NOT-CHECK-THE-PATH"
                ),
            ),
        )
        recorder.noul_rate = NotRun(
            reason="a toy memorisation run has no held-out population to compute a noul rate over"
        )
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    args = parser.parse_args(argv)

    if not 1 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [1, {MAX_STEPS}]")
    if not 1 <= len(args.seeds) <= MAX_SEEDS:
        parser.error(f"--seeds must name between 1 and {MAX_SEEDS} seeds")

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    else:
        print("mps: NOT RUN -- torch.backends.mps.is_available() is False on this host")

    ledger = Ledger(args.ledger)
    rows = _rows(collide=False)
    order = _render_row_order()
    door = _rule3_door()
    floor_check = _floor_is_exact(steps=args.steps, ledger=ledger)

    print(
        f"row order from qd_data.render: choice={order['choice']} score={order['score']} "
        f"span={order['span']}"
    )
    print(
        f"rule 3 at the shard door: assert_path_not_held_out "
        f"{door['assert_path_not_held_out']}, assert_shard_trainable "
        f"{door['assert_shard_trainable']} -- agree={door['agree']}"
    )
    print(
        f"floor formula check: colliding corpus floor {floor_check['letter_floor']:.6f}, run "
        f"stopped at {floor_check['letter_last']:.6f} (row {floor_check['ft_row_id']})"
    )

    report: dict[str, object] = {
        "torch": torch.__version__,
        "devices_run": devices,
        "row_order": order,
        "rule3_shard_door": door,
        "floor_formula_check": floor_check,
        "runs": [],
    }
    failures: list[str] = []
    for device in devices:
        for seed in args.seeds:
            run = _train_once(device=device, seed=seed, steps=args.steps, rows=rows,
                              ledger=ledger, tag="main")
            step, batch = run.pop("_step"), run.pop("_batch")
            decode_at = time.monotonic()
            shipped = _decode(step, batch, rows, order, noul_first=False)
            defect = _decode(step, batch, rows, order, noul_first=True)
            decode_s = time.monotonic() - decode_at
            run["verdict_row_id"] = _record_verdict(
                run, ledger=ledger, shipped=shipped, defect=defect, floor_check=floor_check,
                door=door, rows=rows, steps=args.steps, decode_s=decode_s,
            )
            run["decoded_noul_last"] = shipped["by_kind"]
            run["abstaining_decoded_noul_last"] = shipped["abstaining_decoded_as_abstain"]
            run["abstaining_decoded_noul_first"] = defect["abstaining_decoded_as_abstain"]
            run["abstaining_rows"] = shipped["abstaining_rows"]
            report["runs"].append(run)  # type: ignore[union-attr]

            where = f"{device} seed={seed}"
            if not shipped["all_correct"]:
                failures.append(f"{where}: a row did not decode as the runtime reads it")
            if not _counterfactual_holds(shipped, defect):
                failures.append(
                    f"{where}: the noul-first counterfactual did not hold, so this run does "
                    "not demonstrate that the row order is what decides the answer"
                )
            if run["letter_last"] > run["letter_floor"] + FLOOR_SLACK:  # type: ignore[operator]
                failures.append(f"{where}: letter loss above its floor")
            if run["span_last"] > run["span_floor"] + FLOOR_SLACK:  # type: ignore[operator]
                failures.append(f"{where}: span loss above its floor")
            print(
                f"{device} seed={seed}: total {run['total_first']:.4f} -> "
                f"{run['total_last']:.6f}  letter {run['letter_first']:.4f} -> "
                f"{run['letter_last']:.6f} (floor {run['letter_floor']:.6f})  span "
                f"{run['span_first']:.4f} -> {run['span_last']:.6f} (floor "
                f"{run['span_floor']:.6f})  abstain decoded "
                f"{shipped['abstaining_decoded_as_abstain']}/{shipped['abstaining_rows']} "
                f"noul-last vs {defect['abstaining_decoded_as_abstain']}/"
                f"{defect['abstaining_rows']} noul-first  {run['wall_clock_s']:.2f}s "
                f"({run['steps_per_s']:.1f} step/s)  ft row {run['ft_row_id']}  verdict row "
                f"{run['verdict_row_id']}"
            )

    if not floor_check["reached"] or not floor_check["floor_is_above_zero"]:
        failures.append("the colliding corpus did not stop at its computed floor")
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if failures:
        print(f"\nFAILED: {len(failures)} claim(s) did not hold:")
        for line in failures:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
