"""The span pointer head: its row layout pinned to the runtime, and its loss.

Torch-gated: this module ``importorskip``s, so in the repo venv it reports as **skipped**
rather than passing vacuously. To run it::

    uv run --python /Users/bharath/.venvs/ml/bin/python --with pytest --no-project \\
        python -m pytest python/tests/test_heads.py

The first three tests read ``crates/qd-runtime/src/`` and fail if the Rust moves. That is
the point of them: ``RESERVED_NOUL_ROWS`` and the ``n_candidates + 1`` shape exist twice in
this repo, once in Rust and once in Python, and this repo has twice shipped a pair of
green suites that disagreed about a shared definition.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.artifacts import (  # noqa: E402
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
)
from qd_train.fused_ce import fused_linear_cross_entropy  # noqa: E402
from qd_train.heads import (  # noqa: E402
    RESERVED_NOUL_ROWS,
    SpanPlan,
    SpanPointerHead,
    plan_span_batch,
    serving_scores,
    span_head_rows,
)
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    CostEstimate,
    LRSchedule,
    RunControl,
    WallClockCap,
)
from qd_train.trainer import SpanScoringStep, ft_supervision, train_ft  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNTIME_SRC = REPO / "crates" / "qd-runtime" / "src"
VOCAB, HIDDEN = 16, 6


# --- the shape this module duplicates from Rust ------------------------------------------


def test_reserved_noul_rows_matches_the_runtime_constant():
    source = (RUNTIME_SRC / "schema.rs").read_text(encoding="utf-8")
    found = re.search(r"pub const RESERVED_NOUL_ROWS:\s*usize\s*=\s*(\d+)\s*;", source)
    assert found, "could not find RESERVED_NOUL_ROWS in qd-runtime/src/schema.rs"
    assert int(found.group(1)) == RESERVED_NOUL_ROWS, (
        f"Python says {RESERVED_NOUL_ROWS}, Rust says {found.group(1)}. The span head would "
        "range over a different number of rows than the runtime reads."
    )


def test_span_head_rows_matches_the_runtime_formula():
    source = (RUNTIME_SRC / "answer.rs").read_text(encoding="utf-8")
    body = re.search(
        r"pub fn span_rows\(context: &Context\) -> usize \{(.*?)\}", source, re.DOTALL
    )
    assert body, "could not find span_rows() in qd-runtime/src/answer.rs"
    assert "context.line_count() + RESERVED_NOUL_ROWS" in body.group(1), (
        f"qd-runtime's span_rows body is now {body.group(1)!r}; qd_train.heads.span_head_rows "
        "still computes n_candidates + RESERVED_NOUL_ROWS"
    )
    # The Python side, evaluated on the same definition.
    assert span_head_rows(7) == 7 + RESERVED_NOUL_ROWS


def test_the_abstain_row_is_last_as_the_runtime_reads_it():
    """`answer.rs` indexes the abstention as `plan.rows - RESERVED_NOUL_ROWS`.

    A head trained with abstention first and served by a runtime reading it last would put
    "no evidence" mass on the context's final line, and no loss curve would say so.
    """
    source = (RUNTIME_SRC / "answer.rs").read_text(encoding="utf-8")
    assert "plan.rows - RESERVED_NOUL_ROWS" in source, (
        "qd-runtime no longer places the noul row last; qd_train.heads.plan_span_batch does"
    )
    span = _span_supervision(line_starts=[[True, False, True, False]], start=0, end=2)
    plan = plan_span_batch(span)
    # One row, two candidates -> the abstention is column 2, the last of the three.
    assert int(plan.n_candidates[0]) == 2
    assert span_head_rows(2) == 3
    assert int(plan.gold_start[0]) == 0


def test_span_head_rows_refuses_an_empty_candidate_set():
    with pytest.raises(ValueError, match="mapping failure"):
        span_head_rows(0)


# --- the plan ------------------------------------------------------------------------------


def _ft_batch(
    lengths: list[int],
    kinds: list[int],
    target_index: list[int],
    spans: list[tuple[int, int]],
    line_starts: list[list[bool]],
    width: int = 6,
    index: int = 0,
) -> Batch:
    tokens = np.zeros((len(lengths), width), dtype=np.int32)
    for r, n in enumerate(lengths):
        tokens[r, :n] = np.arange(1, n + 1, dtype=np.int32)
    return Batch(
        tokens=tokens,
        lengths=np.array(lengths, dtype=np.int32),
        bucket=0,
        index=index,
        slot_kind=np.array(kinds, dtype=np.uint8),
        target_index=np.array(target_index, dtype=np.int32),
        span_target=np.array(spans, dtype=np.int32),
        line_starts=np.array(line_starts, dtype=bool),
    )


def _span_supervision(*, line_starts, start, end, query=1):
    batch = _ft_batch(
        lengths=[len(line_starts[0])],
        kinds=[SLOT_SPAN],
        target_index=[query],
        spans=[(start, end)],
        line_starts=line_starts,
        width=len(line_starts[0]),
    )
    supervision = ft_supervision(batch)
    assert supervision.span is not None
    return supervision.span


def test_the_plan_lists_candidates_in_ascending_token_order():
    span = _span_supervision(line_starts=[[True, False, True, True]], start=2, end=3)
    plan = plan_span_batch(span)
    assert plan.candidate_pos[0].tolist() == [0, 2, 3]
    assert plan.candidate_valid[0].tolist() == [True, True, True]
    # gold token 2 is the SECOND candidate, so ordinal 1; token 3 is ordinal 2.
    assert int(plan.gold_start[0]) == 1
    assert int(plan.gold_end[0]) == 2


def test_an_abstaining_row_golds_the_reserved_row_not_a_position():
    span = _span_supervision(
        line_starts=[[True, False, True, False]], start=SPAN_ABSTAIN, end=SPAN_ABSTAIN
    )
    plan = plan_span_batch(span)
    assert plan.abstaining.tolist() == [True]
    assert int(plan.gold_start[0]) == 2 == int(plan.n_candidates[0])
    assert int(plan.gold_end[0]) == 2


def test_ragged_candidate_counts_are_padded_and_the_padding_is_unselectable():
    """The widest row must exceed the narrowest by **more than one**.

    With counts [2, 1] the narrow row's only invalid candidate column is the one the
    abstention is scattered into, so the candidate mask has no observable effect and a
    build with the mask deleted passes. Mutation testing caught exactly that. Counts
    [3, 1] leave column 2 invalid *and* unwritten, which is the column that goes stale.
    """
    batch = _ft_batch(
        lengths=[6, 2],
        kinds=[SLOT_SPAN, SLOT_SPAN],
        target_index=[1, 0],
        spans=[(0, 4), (0, 0)],
        line_starts=[
            [True, False, True, False, True, False],
            [True, False, False, False, False, False],
        ],
        width=6,
    )
    supervision = ft_supervision(batch)
    assert supervision.span is not None
    plan = plan_span_batch(supervision.span)
    assert plan.n_candidates.tolist() == [3, 1]
    assert plan.candidate_valid.tolist() == [[True, True, True], [True, False, False]]

    head = SpanPointerHead(HIDDEN).double()
    hidden = torch.randn(2, 6, HIDDEN, dtype=torch.float64)
    start_scores, _ = head(hidden, plan)
    assert start_scores.shape == (2, 4)
    # Row 1 holds one candidate (col 0) and its abstention (col 1). Columns 2 and 3 are
    # padding and must be unreachable -- without the candidate mask, column 2 would carry a
    # real score for candidate_pos[1, 2] == 0, a phantom duplicate of line 0.
    assert torch.isfinite(start_scores[1, 0])
    assert torch.isfinite(start_scores[1, 1])
    assert start_scores[1, 2] == float("-inf"), "a padded candidate column is selectable"
    assert start_scores[1, 3] == float("-inf")
    probs = start_scores[1].softmax(-1).detach()
    assert float(probs[2]) == 0.0 and float(probs[3]) == 0.0
    assert float(probs[0] + probs[1]) == pytest.approx(1.0)


# --- GAP-RT-POINTER-HEAD-PAD-SHAPE-UNRECORDED: the head pads, the runtime does not -------
#
# The Rust lane asked whether the training side pads the pointer head and recorded it
# UNVERIFIED in both directions, because it may not read python/. It does pad -- to this
# batch's widest candidate set -- and `qd-runtime`'s `backend::validate_logits` refuses
# anything but `query.rows` values, all finite. These four tests are the agreement, by
# assertion rather than by narration.


def _ragged_plan(seed: int = 0):
    """A two-row span batch with counts [3, 1]: genuine padding, and more than one column.

    Counts that differ by exactly one are useless here -- the narrow row's only padded
    column is the one the abstention is scattered into, so a broken mask is invisible.
    That vacuity was caught once already; see the test above.
    """
    batch = _ft_batch(
        lengths=[6, 2],
        kinds=[SLOT_SPAN, SLOT_SPAN],
        target_index=[1, 0],
        spans=[(0, 4), (0, 0)],
        line_starts=[
            [True, False, True, False, True, False],
            [True, False, False, False, False, False],
        ],
        width=6,
    )
    supervision = ft_supervision(batch)
    assert supervision.span is not None
    plan = plan_span_batch(supervision.span)
    torch.manual_seed(seed)
    head = SpanPointerHead(HIDDEN).double()
    hidden = torch.randn(2, 6, HIDDEN, dtype=torch.float64)
    return head, hidden, plan


def test_the_plan_states_the_row_count_the_runtime_will_demand():
    """`runtime_rows` is per row; `max_rows` is the batch's, and they are not the same.

    This is the number `answer.rs` puts in `query.rows` for a span slot. Pinned
    elementwise to `span_head_rows`, which is itself pinned to the Rust formula above, so
    the vectorised copy cannot drift from the scalar one that is checked against the source.
    """
    _, _, plan = _ragged_plan()
    assert plan.n_candidates.tolist() == [3, 1]
    assert plan.runtime_rows.tolist() == [4, 2]
    assert plan.max_rows == 4
    # The narrow row's runtime shape is *half* the matrix it is scored in. If these two
    # were the same number there would be no gap to record.
    assert int(plan.runtime_rows[1]) != plan.max_rows
    for k, count in enumerate(plan.n_candidates.tolist()):
        assert int(plan.runtime_rows[k]) == span_head_rows(count)


def test_serving_scores_strips_the_padding_the_runtime_would_refuse():
    """What leaves for `qd-runtime` is `line_count + 1` finite values, and nothing else.

    `backend::validate_logits` refuses the padded matrix twice over: on length against
    `query.rows`, and on `is_finite` for the `-inf` fill. Slicing to `runtime_rows[k]`
    satisfies both, and the last value of each slice is that row's abstention.
    """
    head, hidden, plan = _ragged_plan()
    start_scores, end_scores = head(hidden, plan)
    assert start_scores.shape == (2, 4), "the training matrix is padded to the batch max"

    for scores in (start_scores, end_scores):
        served = serving_scores(scores, plan)
        assert [int(s.numel()) for s in served] == [4, 2]
        for k, row in enumerate(served):
            assert int(row.numel()) == span_head_rows(int(plan.n_candidates[k]))
            assert bool(torch.isfinite(row).all()), "qd-runtime refuses a non-finite logit"
            # The abstention is last, as `noul_row = plan.rows - RESERVED_NOUL_ROWS` reads it.
            abstention = scores[k, int(plan.n_candidates[k])]
            assert float(row[-1].detach()) == float(abstention.detach())
        # Row 1's slice drops exactly the padding and keeps everything that is not padding.
        assert torch.equal(served[1], scores[1, :2])
        assert bool(torch.isinf(scores[1, 2:]).all()), "the dropped columns were the padding"


def test_serving_scores_pins_itself_to_what_validate_logits_actually_checks():
    """Both halves of the Rust check, read from the source rather than remembered."""
    source = (RUNTIME_SRC / "backend.rs").read_text()
    assert "logits.values.len() != query.rows" in source, (
        "qd-runtime no longer refuses a wrong row count, so serving_scores is slicing to a "
        "number nothing enforces"
    )
    assert "if !v.is_finite()" in source, (
        "qd-runtime no longer refuses a non-finite logit, so the -inf padding would reach a "
        "softmax at serve time instead of being refused"
    )
    # Which number it compares against is pinned by
    # `test_span_head_rows_matches_the_runtime_formula` above; this test pins that the
    # comparison happens at all.
    assert "expected: query.rows," in source


def test_the_head_refuses_a_row_whose_selectable_set_is_not_the_runtimes():
    """A plan whose row count and candidate mask disagree leaves a hole, silently.

    Before the postcondition, `forward` returned a row whose finite columns were `[0, 2]`
    -- a gap at column 1 -- and trained on it. The runtime would have refused that decode
    as a `logit_shape_mismatch`, but only at serve time, on a model already trained over a
    candidate set nobody serves. This is the same failure one layer earlier.
    """
    head, hidden, plan = _ragged_plan()
    lying = SpanPlan(
        candidate_pos=plan.candidate_pos.clone(),
        candidate_valid=plan.candidate_valid.clone(),
        # Row 1 truly has one candidate. Claiming two scatters its abstention into column
        # 2 and leaves column 1 at -inf: finite columns [0, 2] against an expected [0, 1, 2].
        n_candidates=torch.tensor([3, 2]),
        query_index=plan.query_index.clone(),
        gold_start=plan.gold_start.clone(),
        gold_end=plan.gold_end.clone(),
        abstaining=plan.abstaining.clone(),
    )
    with pytest.raises(ValueError) as exc:
        head(hidden, lying)
    message = str(exc.value)
    assert "span row 1" in message
    assert "qd-runtime will demand exactly 3" in message
    assert "[0, 2]" in message, "the message must show the hole, not just that there is one"
    # serving_scores refuses it through the same check, so the two cannot drift apart.
    good_scores, _ = head(hidden, plan)
    with pytest.raises(ValueError, match="span row 1"):
        serving_scores(good_scores, lying)


# --- the loss --------------------------------------------------------------------------------


def _head_and_hidden(line_starts, start, end, *, seed=0):
    torch.manual_seed(seed)
    span = _span_supervision(line_starts=line_starts, start=start, end=end)
    plan = plan_span_batch(span)
    head = SpanPointerHead(HIDDEN).double()
    hidden = torch.randn(1, len(line_starts[0]), HIDDEN, dtype=torch.float64)
    return head, hidden, plan


def test_an_untrained_head_is_uniform_over_its_rows():
    """Abstain vectors start at zero and the projections are small, so no row is favoured."""
    head, hidden, plan = _head_and_hidden([[True, False, True, False]], 0, 2)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        start_scores, _ = head(hidden, plan)
    probs = start_scores.softmax(-1)
    assert probs.shape == (1, 3)
    torch.testing.assert_close(probs, torch.full((1, 3), 1 / 3, dtype=torch.float64))
    # -ln(1/3) per pointer, for both pointers -> the mean is exactly ln(3).
    assert float(head.loss(hidden, plan).detach()) == pytest.approx(float(np.log(3)))


def test_the_head_learns_to_point_at_the_gold_line():
    head, hidden, plan = _head_and_hidden([[True, False, True, True]], 2, 3, seed=5)
    optimiser = torch.optim.SGD(head.parameters(), lr=0.5)
    first = float(head.loss(hidden, plan).detach())
    for _ in range(200):
        optimiser.zero_grad()
        loss = head.loss(hidden, plan)
        loss.backward()
        optimiser.step()
    last = float(head.loss(hidden, plan).detach())
    assert last < first, f"span loss did not fall: {first} -> {last}"
    start_scores, end_scores = head(hidden, plan)
    assert int(start_scores.argmax(-1)[0]) == int(plan.gold_start[0])
    assert int(end_scores.argmax(-1)[0]) == int(plan.gold_end[0])


def test_a_head_trained_to_abstain_selects_the_reserved_row_not_a_line():
    """The property that matters: abstention competes in the same softmax as the lines."""
    head, hidden, plan = _head_and_hidden(
        [[True, False, True, False]], SPAN_ABSTAIN, SPAN_ABSTAIN, seed=9
    )
    optimiser = torch.optim.SGD(head.parameters(), lr=0.5)
    for _ in range(200):
        optimiser.zero_grad()
        head.loss(hidden, plan).backward()
        optimiser.step()
    start_scores, end_scores = head(hidden, plan)
    abstain_row = int(plan.n_candidates[0])
    assert int(start_scores.argmax(-1)[0]) == abstain_row
    assert int(end_scores.argmax(-1)[0]) == abstain_row


def test_training_to_abstain_does_not_train_a_position_and_vice_versa():
    """An abstaining gold and a pointing gold must move the head in different directions."""
    shared = [[True, False, True, False]]
    pointing_head, hidden, pointing_plan = _head_and_hidden(shared, 0, 0, seed=3)
    abstain_head, _, abstain_plan = _head_and_hidden(
        shared, SPAN_ABSTAIN, SPAN_ABSTAIN, seed=3
    )
    assert int(pointing_plan.gold_start[0]) == 0
    assert int(abstain_plan.gold_start[0]) == 2

    for head, plan in ((pointing_head, pointing_plan), (abstain_head, abstain_plan)):
        optimiser = torch.optim.SGD(head.parameters(), lr=0.5)
        for _ in range(200):
            optimiser.zero_grad()
            head.loss(hidden, plan).backward()
            optimiser.step()

    pointing_pick = int(pointing_head(hidden, pointing_plan)[0].argmax(-1)[0])
    abstain_pick = int(abstain_head(hidden, abstain_plan)[0].argmax(-1)[0])
    assert pointing_pick == 0, "a pointing gold must select its line"
    assert abstain_pick == 2, "an abstaining gold must select the reserved row"
    assert pointing_pick != abstain_pick


def test_the_mean_reduction_is_per_pointer_not_per_candidate():
    """A three-line context and a thirty-line context must cost the same per row."""
    short, _, short_plan = _head_and_hidden([[True, True, False, False]], 0, 0, seed=1)
    with torch.no_grad():
        for p in short.parameters():
            p.zero_()
    wide_starts = [[True] * 8]
    wide, wide_hidden, wide_plan = _head_and_hidden(wide_starts, 0, 0, seed=1)
    with torch.no_grad():
        for p in wide.parameters():
            p.zero_()
    short_hidden = torch.randn(1, 4, HIDDEN, dtype=torch.float64)
    # Uniform in both cases, so the losses are ln(rows) -- different, but only because the
    # row counts differ, never because one context contributed more pointer decisions.
    assert float(short.loss(short_hidden, short_plan).detach()) == pytest.approx(float(np.log(3)))
    assert float(wide.loss(wide_hidden, wide_plan).detach()) == pytest.approx(float(np.log(9)))
    assert float(wide.loss(wide_hidden, wide_plan, reduction="sum").detach()) == pytest.approx(
        2 * float(np.log(9))
    )


def test_the_loss_refuses_an_unknown_reduction():
    head, hidden, plan = _head_and_hidden([[True, False, True, False]], 0, 2)
    with pytest.raises(ValueError, match="reduction must be"):
        head.loss(hidden, plan, reduction="none")


def test_the_head_refuses_hidden_states_that_do_not_match_its_plan():
    head, hidden, plan = _head_and_hidden([[True, False, True, False]], 0, 2)
    with pytest.raises(ValueError, match="span rows but the plan has"):
        head(hidden.expand(3, -1, -1), plan)
    with pytest.raises(ValueError, match="head was built for"):
        SpanPointerHead(HIDDEN + 1).double()(hidden, plan)


# --- the seam: a span batch trains through the real FT loop ------------------------------


class SpanAwareStep:
    """A ``SpanScoringStep``: fused letter CE for choice/score rows, pointer CE for spans."""

    def __init__(self, *, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN).double()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False).double()
        self.span_head = SpanPointerHead(HIDDEN).double()

    def _params(self):
        return [
            *self.emb.parameters(),
            *self.lm_head.parameters(),
            *self.span_head.parameters(),
        ]

    def _letter_loss(self, batch, supervision) -> torch.Tensor | None:
        if supervision.n_supervised == 0:
            return None
        hidden = self.emb(torch.from_numpy(batch.tokens[:, :-1].astype(np.int64)))
        return fused_linear_cross_entropy(
            hidden,
            self.lm_head.weight,
            torch.from_numpy(supervision.targets.astype(np.int64)),
            mask=torch.from_numpy(supervision.mask.copy()),
            chunk_size=4,
        )

    def accumulate(self, batch, supervision) -> float:
        loss = self._letter_loss(batch, supervision)
        assert loss is not None
        loss.backward()
        return float(loss.item())

    def accumulate_span(self, batch, supervision) -> float:
        span = supervision.span
        assert span is not None
        hidden = self.emb(torch.from_numpy(batch.tokens.astype(np.int64)))
        plan = plan_span_batch(span)
        total = self.span_head.loss(hidden[torch.from_numpy(span.rows)], plan)
        letter = self._letter_loss(batch, supervision)
        if letter is not None:
            total = total + letter
        total.backward()
        return float(total.item())

    def apply(self, *, lr: float) -> None:
        with torch.no_grad():
            for p in self._params():
                if p.grad is not None:
                    p -= lr * p.grad
                    p.grad = None

    def state(self) -> dict:
        return {"n": 0}

    def load_state(self, state) -> None:
        return None


def _recorder(tmp_path: Path) -> RunRecorder:
    return RunRecorder(
        Ledger(tmp_path / "ledger.jsonl"),
        protocol=Protocol(
            data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
            backbone_commit="b" * 40, recipe_hash="r" * 64, seed=7,
        ),
        run_kind="ft",
        repo=REPO,
        wall_clock_s=None,  # the caller's `with` block contains the run under test
        env=Environment(
            torch=torch.__version__, transformers_sha="none", device="cpu", host="test",
            fla_present=NotRun(reason="no CUDA on this host"),
            causal_conv1d_present=NotRun(reason="no CUDA on this host"),
        ),
    )


def _control(total_steps: int) -> RunControl:
    cap = WallClockCap(cap_s=600.0)
    return RunControl(
        schedule=LRSchedule(
            peak_lr=0.5, total_steps=total_steps, warmup_steps=1, min_lr=0.05
        ),
        cap=cap,
        cost=CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10"),
    )


def _mixed_batches(n: int):
    for i in range(n):
        yield _ft_batch(
            lengths=[5, 4],
            kinds=[SLOT_SPAN, SLOT_CHOICE],
            target_index=[3, 2],
            spans=[(0, 2), (NO_SPAN, NO_SPAN)],
            line_starts=[
                [True, False, True, False, True, False],
                [False] * 6,
            ],
            index=i,
        )


def test_a_span_batch_now_trains_through_train_ft_instead_of_being_refused(tmp_path):
    step = SpanAwareStep()
    assert isinstance(step, SpanScoringStep)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    recorder = _recorder(tmp_path)
    assert recorder.ledger.path == ledger.path

    result = train_ft(
        _mixed_batches(12), epoch=0, step=step, control=_control(12), recorder=recorder
    )
    assert result.termination == "steps_exhausted"
    assert result.optimizer_steps == 12
    assert result.span_rows == 12, "one span row per batch, counted as a row not a token"
    assert result.supervised_tokens == 12, "one letter answer per batch"
    losses = result.loss_log.losses()
    assert losses[-1] < losses[0], f"the combined FT loss must fall: {losses}"
    assert len(ledger.rows()) == 1 and ledger.rows()[0].status == "completed"
    assert ledger.rows()[0].metrics["train.span_rows"].value == 12


def test_an_all_span_batch_trains_with_an_empty_letter_channel(tmp_path):
    def batches(n: int):
        for i in range(n):
            yield _ft_batch(
                lengths=[5, 4],
                kinds=[SLOT_SPAN, SLOT_SPAN],
                target_index=[3, 2],
                spans=[(0, 2), (SPAN_ABSTAIN, SPAN_ABSTAIN)],
                line_starts=[
                    [True, False, True, False, True, False],
                    [True, False, True, False, False, False],
                ],
                index=i,
            )

    result = train_ft(
        batches(8), epoch=0, step=SpanAwareStep(), control=_control(8),
        recorder=_recorder(tmp_path),
    )
    assert result.supervised_tokens == 0, "no letter answers at all"
    assert result.span_rows == 16
    losses = result.loss_log.losses()
    assert losses[-1] < losses[0], f"an all-span batch must still train: {losses}"
