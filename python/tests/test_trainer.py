"""The CPT and FT loops: masking, the cap, and S5's bit-exact resume.

Everything here runs in the repo venv. The model is a ~20-line numpy bigram whose only
job is to be *deterministic* and to have real gradients, which is all S5's resume claim
needs -- the claim is about the loop's bookkeeping, not about a transformer. Deferring it
to a GPU would have meant the one property most likely to be silently wrong is the one
property never checked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.artifacts import (
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_LM,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
    ShardContractViolation,
)
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
from qd_train.run_control import (
    Checkpoint,
    CostEstimate,
    LossLog,
    LRSchedule,
    RunControl,
    WallClockCap,
)
from qd_train.trainer import (
    SpanScoringStep,
    SpanSupervision,
    TrainerContractViolation,
    TrainStep,
    cpt_supervision,
    ft_supervision,
    train_cpt,
    train_ft,
)
from qd_train.tristate import NotRun

REPO = Path(__file__).resolve().parents[2]
VOCAB = 16
HIDDEN = 4


# --- fixtures the whole file shares ---------------------------------------------------------


def _protocol(seed: int = 7) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        backbone_commit="b" * 40,
        recipe_hash="r" * 64,
        seed=seed,
    )


def _env() -> Environment:
    return Environment(
        torch="not-installed",
        transformers_sha="none",
        device="cpu",
        host="test",
        fla_present=NotRun(reason="no CUDA on this host"),
        causal_conv1d_present=NotRun(reason="no CUDA on this host"),
    )


def _recorder(tmp_path: Path, *, run_kind: str = "cpt", seed: int = 7) -> RunRecorder:
    return RunRecorder(
        Ledger(tmp_path / "ledger.jsonl"),
        protocol=_protocol(seed),
        run_kind=run_kind,  # type: ignore[arg-type]
        repo=REPO,
        env=_env(),
    )


def _control(
    *, total_steps: int = 10, cap_s: float = 3600.0, clock=None, warmup_steps: int = 1, **over
) -> RunControl:
    cap = WallClockCap(cap_s=cap_s)
    kwargs = {
        "schedule": LRSchedule(
            peak_lr=0.5, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=0.05
        ),
        "cap": cap,
        "cost": CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10"),
    }
    if clock is not None:
        kwargs["clock"] = clock
    kwargs.update(over)
    return RunControl(**kwargs)  # type: ignore[arg-type]


def batches_for(seed: int, epoch: int, *, n: int, rows: int = 2, width: int = 8):
    """The batch order as S4 promises it: a pure function of ``(seed, epoch)``."""
    rng = np.random.default_rng([seed, epoch])
    for i in range(n):
        lengths = rng.integers(2, width + 1, size=rows).astype(np.int32)
        tokens = rng.integers(1, VOCAB, size=(rows, width)).astype(np.int32)
        for r in range(rows):
            tokens[r, int(lengths[r]) :] = 0  # padding is visibly padding
        yield Batch(tokens=tokens, lengths=lengths, bucket=0, index=i)


class TinyStep:
    """A deterministic bigram model: ``emb[token] @ head.T``, plain SGD.

    Small enough to be obviously correct, real enough that a wrong resume moves the loss.
    """

    def __init__(self, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.emb = rng.standard_normal((VOCAB, HIDDEN)) * 0.1
        self.head = rng.standard_normal((VOCAB, HIDDEN)) * 0.1
        self.applied = 0
        self._reset()

    def _reset(self) -> None:
        self._g_emb = np.zeros_like(self.emb)
        self._g_head = np.zeros_like(self.head)
        self._n = 0

    def accumulate(self, batch: Batch, supervision) -> float:
        inputs = batch.tokens[:, :-1].astype(np.int64)
        targets = supervision.targets.astype(np.int64)[..., None]
        mask = supervision.mask.astype(np.float64)

        h = self.emb[inputs]
        logits = h @ self.head.T
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        nll = -np.take_along_axis(logp, targets, axis=-1)[..., 0]
        loss = float((nll * mask).sum() / supervision.n_supervised)

        grad = np.exp(logp)
        np.put_along_axis(
            grad, targets, np.take_along_axis(grad, targets, axis=-1) - 1.0, axis=-1
        )
        grad *= (mask / supervision.n_supervised)[..., None]
        self._g_head += np.einsum("bpv,bph->vh", grad, h)
        np.add.at(self._g_emb, inputs, grad @ self.head)
        self._n += 1
        return loss

    def apply(self, *, lr: float) -> None:
        scale = lr / max(1, self._n)
        self.emb -= scale * self._g_emb
        self.head -= scale * self._g_head
        self.applied += 1
        self._reset()

    def state(self) -> dict:
        return {"emb": self.emb.tolist(), "head": self.head.tolist()}

    def load_state(self, state) -> None:
        self.emb = np.array(state["emb"], dtype=np.float64)
        self.head = np.array(state["head"], dtype=np.float64)
        self._reset()


class StepClock:
    """A clock that advances only when an optimizer step is taken.

    Wall time in a test must be a function of work done, not of how many times the loop
    happened to read the clock -- otherwise the cap fires at a step number that changes
    whenever the loop's internals do.
    """

    def __init__(self, step: TinyStep, *, seconds_per_step: float = 1.0) -> None:
        self._step = step
        self._per = seconds_per_step

    def __call__(self) -> float:
        return self._step.applied * self._per


def test_tinystep_satisfies_the_train_step_protocol():
    assert isinstance(TinyStep(), TrainStep)


# --- masking: the point of Batch.lengths ------------------------------------------------


def _batch(lengths: list[int], width: int = 6, *, fill: int = 0, index: int = 0) -> Batch:
    tokens = np.zeros((len(lengths), width), dtype=np.int32)
    for r, n in enumerate(lengths):
        tokens[r, :n] = np.arange(1, n + 1, dtype=np.int32)
        tokens[r, n:] = fill
    return Batch(
        tokens=tokens, lengths=np.array(lengths, dtype=np.int32), bucket=0, index=index
    )


def test_cpt_supervises_every_real_token_pair_and_no_padding():
    sup = cpt_supervision(_batch([4, 2], width=6))
    # width 6 -> 5 prediction positions. Row 0 (len 4) supervises 3; row 1 (len 2) supervises 1.
    assert sup.mask.shape == (2, 5)
    assert sup.mask[0].tolist() == [True, True, True, False, False]
    assert sup.mask[1].tolist() == [True, False, False, False, False]
    assert sup.n_supervised == 4


def _ft_batch(
    lengths: list[int],
    *,
    kinds: list[int],
    target_index: list[int],
    spans: list[tuple[int, int]] | None = None,
    line_starts: list[list[bool]] | None = None,
    width: int = 6,
    index: int = 0,
) -> Batch:
    base = _batch(lengths, width=width, index=index)
    span_target = np.array(spans, dtype=np.int32) if spans is not None else None
    if line_starts is not None:
        starts_mask = np.array(line_starts, dtype=bool)
    elif spans is not None:
        # Default candidate set: every even position inside the row's real tokens. Enough
        # line starts for a gold to land on one, and none in padding.
        starts_mask = np.zeros((len(lengths), width), dtype=bool)
        for r, n in enumerate(lengths):
            starts_mask[r, 0:n:2] = True
    else:
        starts_mask = None
    return Batch(
        tokens=base.tokens,
        lengths=base.lengths,
        bucket=0,
        index=index,
        slot_kind=np.array(kinds, dtype=np.uint8),
        target_index=np.array(target_index, dtype=np.int32),
        span_target=span_target,
        line_starts=starts_mask,
    )


def test_ft_supervises_the_position_the_batch_names_not_one_inferred_from_lengths():
    # Row 0 is length 4 but its answer is at index 1, not at lengths-2 == 2. Under the
    # pre-contract-change loop this row was supervised one position too late.
    batch = _ft_batch([4, 2], kinds=[SLOT_CHOICE, SLOT_SCORE], target_index=[1, 0])
    sup = ft_supervision(batch)
    assert sup.mask[0].tolist() == [False, True, False, False, False]
    assert sup.mask[1].tolist() == [True, False, False, False, False]
    assert sup.n_supervised == 2
    assert sup.span is None


def test_a_row_with_no_supervised_position_cannot_be_built_in_the_first_place():
    """GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS, settled upstream of this loop.

    This test used to call `cpt_supervision` and assert it refused a one-token row. It
    did -- but `Batch` happily built that row, so the constructor and its only consumer
    disagreed about what a batch is, which is exactly the two-readings failure
    `artifacts.py` exists to prevent. The floor now lives in `artifacts._MIN_ROW_TOKENS`
    and the refusal is here, one layer earlier. `_refuse_unsupervised_rows` stays as a
    postcondition on the mask arithmetic; it is no longer the contract statement, and
    `test_artifacts.py` is where the contract is tested.
    """
    with pytest.raises(ShardContractViolation, match=r"row\(s\) \[1\] carry \[1\] real token"):
        _batch([4, 1], width=6)
    # And the batch one token wider is fine, so the floor is a floor and not a ban.
    assert cpt_supervision(_batch([4, 2], width=6)).n_supervised == 4


def test_cpt_refuses_a_batch_that_carries_slot_answers():
    batch = _ft_batch([4, 3], kinds=[SLOT_CHOICE, SLOT_CHOICE], target_index=[2, 1])
    with pytest.raises(TrainerContractViolation, match="all three supervision fields None"):
        cpt_supervision(batch)


def test_ft_refuses_a_batch_with_no_supervision_channel():
    with pytest.raises(TrainerContractViolation, match="FT needs slot_kind and target_index"):
        ft_supervision(_batch([4, 2], width=6))


def test_an_lm_row_never_reaches_ft_because_the_contract_refuses_it_first():
    """GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED, settled upstream of this loop.

    `ft_supervision` still carries the refusal as defence in depth, alongside its three
    other "Batch refuses this" guards, but it is no longer the only thing standing between
    a mislabelled row and a different objective: the batch cannot be built.
    """
    with pytest.raises(ShardContractViolation, match=r"row\(s\) \[0\] carry SLOT_LM"):
        _ft_batch([4, 3], kinds=[SLOT_LM, SLOT_CHOICE], target_index=[2, 1])
    # The same batch with the row relabelled is accepted and supervises both rows.
    ok = _ft_batch([4, 3], kinds=[SLOT_CHOICE, SLOT_CHOICE], target_index=[2, 1])
    assert ft_supervision(ok).n_supervised == 2


# --- spans: a position answer, never a letter ------------------------------------------


def test_a_span_row_is_kept_out_of_the_letter_channel_entirely():
    """GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL: a span's only letter is noul."""
    batch = _ft_batch(
        [5, 4],
        kinds=[SLOT_SPAN, SLOT_CHOICE],
        target_index=[3, 2],
        spans=[(0, 2), (NO_SPAN, NO_SPAN)],
    )
    sup = ft_supervision(batch)
    assert not sup.mask[0].any(), "the span row must contribute no letter target"
    assert sup.mask[1].tolist() == [False, False, True, False, False]
    assert sup.n_supervised == 1

    assert sup.span is not None
    assert sup.span.rows.tolist() == [0]
    assert sup.span.start.tolist() == [0]
    assert sup.span.end.tolist() == [2]
    assert sup.span.query_index.tolist() == [3]
    assert sup.n_answers == 2, "one letter answer plus one span answer"
    # The candidate set is carried through from Batch, never re-derived here.
    assert sup.span.line_starts.tolist() == [[True, False, True, False, True, False]]
    assert sup.span.candidate_counts().tolist() == [3]
    assert sup.span.abstaining.tolist() == [False]


def test_an_all_span_batch_is_supervised_even_though_its_letter_channel_is_empty():
    batch = _ft_batch(
        [5, 4], kinds=[SLOT_SPAN, SLOT_SPAN], target_index=[3, 2], spans=[(0, 2), (0, 2)]
    )
    sup = ft_supervision(batch)
    assert sup.n_supervised == 0
    assert sup.span is not None and sup.span.n_spans == 2
    assert sup.n_answers == 2


def test_a_batch_with_span_rows_is_refused_by_a_step_that_cannot_score_positions(tmp_path):
    """The refusal is the design: no silent fold into the letter channel, no silent drop."""
    batch = _ft_batch(
        [5, 4],
        kinds=[SLOT_SPAN, SLOT_CHOICE],
        target_index=[3, 2],
        spans=[(0, 2), (NO_SPAN, NO_SPAN)],
    )
    assert not isinstance(TinyStep(), SpanScoringStep)
    with pytest.raises(TrainerContractViolation, match="defines no accumulate_span"):
        train_ft(
            iter([batch]), epoch=0, step=TinyStep(),
            control=_control(total_steps=1, warmup_steps=0),
            recorder=_recorder(tmp_path, run_kind="ft"),
        )


def test_a_step_that_scores_positions_receives_the_span_channel(tmp_path):
    """The seam works: a pointer-head step is handed the gold positions, not a letter."""
    seen: list[SpanSupervision] = []

    class SpanAwareStep(TinyStep):
        def accumulate_span(self, batch, supervision) -> float:
            assert supervision.span is not None
            seen.append(supervision.span)
            # The letter rows are still this step's to train; the span rows are extra.
            return super().accumulate(batch, supervision)

    batch = _ft_batch(
        [5, 4],
        kinds=[SLOT_SPAN, SLOT_CHOICE],
        target_index=[3, 2],
        spans=[(0, 2), (NO_SPAN, NO_SPAN)],
    )
    step = SpanAwareStep()
    assert isinstance(step, SpanScoringStep)
    result = train_ft(
        iter([batch]), epoch=0, step=step, control=_control(total_steps=1, warmup_steps=0),
        recorder=_recorder(tmp_path, run_kind="ft"),
    )
    assert len(seen) == 1
    assert seen[0].rows.tolist() == [0]
    assert seen[0].start.tolist() == [0] and seen[0].end.tolist() == [2]
    assert result.span_rows == 1
    assert result.supervised_tokens == 1


def test_the_ledger_row_counts_span_answers_separately_from_letter_tokens(tmp_path):
    class SpanAwareStep(TinyStep):
        def accumulate_span(self, batch, supervision) -> float:
            return super().accumulate(batch, supervision)

    batch = _ft_batch(
        [5, 4],
        kinds=[SLOT_SPAN, SLOT_CHOICE],
        target_index=[3, 2],
        spans=[(0, 2), (NO_SPAN, NO_SPAN)],
    )
    rec = _recorder(tmp_path, run_kind="ft")
    train_ft(
        iter([batch]), epoch=0, step=SpanAwareStep(),
        control=_control(total_steps=1, warmup_steps=0),
        recorder=rec,
    )
    assert rec.row is not None
    assert rec.row.metrics["train.span_rows"].value == 1
    assert rec.row.metrics["train.supervised_tokens"].value == 1


def test_an_empty_span_channel_is_refused_because_absent_and_empty_differ():
    with pytest.raises(TrainerContractViolation, match="absent and empty are different"):
        SpanSupervision(
            rows=np.array([], dtype=np.int64),
            query_index=np.array([], dtype=np.int64),
            start=np.array([], dtype=np.int64),
            end=np.array([], dtype=np.int64),
            line_starts=np.zeros((0, 4), dtype=bool),
            abstaining=np.array([], dtype=bool),
        )


def test_an_abstaining_span_is_carried_through_as_abstention_not_as_a_position():
    """SPAN_ABSTAIN is the head's extra row, never a token position."""
    batch = _ft_batch(
        [5, 4],
        kinds=[SLOT_SPAN, SLOT_CHOICE],
        target_index=[3, 2],
        spans=[(SPAN_ABSTAIN, SPAN_ABSTAIN), (NO_SPAN, NO_SPAN)],
    )
    sup = ft_supervision(batch)
    assert sup.span is not None
    assert sup.span.abstaining.tolist() == [True]
    assert sup.span.n_abstaining == 1
    assert sup.span.start.tolist() == [SPAN_ABSTAIN]


def test_the_abstaining_flag_must_agree_with_the_sentinel():
    with pytest.raises(TrainerContractViolation, match="two statements of one fact"):
        SpanSupervision(
            rows=np.array([0], dtype=np.int64),
            query_index=np.array([1], dtype=np.int64),
            start=np.array([SPAN_ABSTAIN], dtype=np.int64),
            end=np.array([SPAN_ABSTAIN], dtype=np.int64),
            line_starts=np.array([[True, False, False, False]]),
            abstaining=np.array([False]),  # the sentinel says abstain; the flag says not
        )


def test_a_batch_with_no_next_token_pair_is_refused():
    """A width-1 batch needs a length-1 row, so `Batch` now refuses it one layer earlier.

    `_prediction_grid`'s own width check stays as a guard on the arithmetic, but it is
    unreachable through the constructor: `lengths.max() <= width` and
    `lengths.min() >= _MIN_ROW_TOKENS` together put the floor under `width` too.
    """
    with pytest.raises(ShardContractViolation, match=r"row\(s\) \[0\] carry \[1\] real token"):
        _batch([1], width=1)


def test_the_loss_does_not_depend_on_what_is_in_the_padding():
    """The direct statement of 'never train on padding'.

    Two batches with identical real tokens and identical lengths, differing only in the
    junk sitting in their padded slots, must produce the same loss *and* the same gradient
    -- so the same parameters after a step.
    """
    quiet = _batch([4, 2], width=6, fill=0)
    noisy = _batch([4, 2], width=6, fill=VOCAB - 1)
    assert not np.array_equal(quiet.tokens, noisy.tokens), "the two batches must differ"

    a, b = TinyStep(seed=3), TinyStep(seed=3)
    loss_a = a.accumulate(quiet, cpt_supervision(quiet))
    loss_b = b.accumulate(noisy, cpt_supervision(noisy))
    assert loss_a == loss_b
    a.apply(lr=0.1)
    b.apply(lr=0.1)
    assert np.array_equal(a.emb, b.emb)
    assert np.array_equal(a.head, b.head)


def test_supervision_refuses_a_count_that_disagrees_with_its_mask():
    from qd_train.trainer import Supervision

    with pytest.raises(TrainerContractViolation, match="two statements of one fact"):
        Supervision(
            targets=np.zeros((1, 3), dtype=np.int32),
            mask=np.array([[True, False, False]]),
            n_supervised=3,
        )


# --- a plain run ------------------------------------------------------------------------


def test_a_cpt_run_completes_its_schedule_and_writes_one_ledger_row(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    rec = RunRecorder(
        ledger, protocol=_protocol(), run_kind="cpt", repo=REPO, env=_env()
    )
    result = train_cpt(
        batches_for(7, 0, n=40), epoch=0, step=TinyStep(), control=_control(total_steps=10),
        recorder=rec,
    )
    assert result.termination == "steps_exhausted"
    assert result.optimizer_steps == 10
    assert len(result.loss_log) == 10

    rows = ledger.rows()
    assert len(rows) == 1
    assert rows[0].status == "completed"
    assert rows[0].run_kind == "cpt"
    assert result.row_id == rows[0].row_id
    assert rows[0].metrics["train.termination"].value == "steps_exhausted"
    assert rows[0].metrics["train.loss_log_digest"].value == result.loss_log.digest()
    assert rows[0].metrics["train.padding_fraction"].value == pytest.approx(
        result.padding_fraction
    )


def ft_batches_for(seed: int, epoch: int, *, n: int, rows: int = 2, width: int = 8):
    """FT batches: choice/score rows, each naming its own answer position."""
    rng = np.random.default_rng([seed, epoch])
    for i in range(n):
        lengths = rng.integers(2, width + 1, size=rows).astype(np.int32)
        tokens = rng.integers(1, VOCAB, size=(rows, width)).astype(np.int32)
        for r in range(rows):
            tokens[r, int(lengths[r]) :] = 0
        # target_index must be in [0, lengths-1); pick it inside that window so the tests
        # exercise answers that are *not* simply the final token.
        target_index = np.array(
            [rng.integers(0, int(lengths[r]) - 1) for r in range(rows)], dtype=np.int32
        )
        kinds = np.array(
            [SLOT_CHOICE if r % 2 == 0 else SLOT_SCORE for r in range(rows)], dtype=np.uint8
        )
        yield Batch(
            tokens=tokens, lengths=lengths, bucket=0, index=i,
            slot_kind=kinds, target_index=target_index,
        )


def test_an_ft_run_takes_one_supervised_token_per_row(tmp_path):
    result = train_ft(
        ft_batches_for(7, 0, n=20), epoch=0, step=TinyStep(),
        control=_control(total_steps=5), recorder=_recorder(tmp_path, run_kind="ft"),
    )
    assert result.objective == "ft"
    # 5 steps x 1 micro-batch x 2 rows = 10 supervised positions, one per row.
    assert result.supervised_tokens == 10
    assert result.span_rows == 0


def test_a_run_that_exhausts_its_data_says_so(tmp_path):
    result = train_cpt(
        batches_for(7, 0, n=3), epoch=0, step=TinyStep(),
        control=_control(total_steps=10), recorder=_recorder(tmp_path),
    )
    assert result.termination == "data_exhausted"
    assert result.optimizer_steps == 3


def test_gradient_accumulation_takes_one_step_per_group(tmp_path):
    result = train_cpt(
        batches_for(7, 0, n=12), epoch=0, step=TinyStep(),
        control=_control(total_steps=4, grad_accum=3), recorder=_recorder(tmp_path),
    )
    assert result.optimizer_steps == 4
    assert result.micro_batches == 12


def test_a_source_ending_mid_group_is_refused_rather_than_stepped_on_a_short_group(tmp_path):
    with pytest.raises(TrainerContractViolation, match="partial group"):
        train_cpt(
            batches_for(7, 0, n=5), epoch=0, step=TinyStep(),
            control=_control(total_steps=4, grad_accum=3), recorder=_recorder(tmp_path),
        )


def test_the_loop_refuses_a_recorder_that_names_a_different_run_kind(tmp_path):
    with pytest.raises(TrainerContractViolation, match="run_kind"):
        train_cpt(
            batches_for(7, 0, n=4), epoch=0, step=TinyStep(),
            control=_control(total_steps=2), recorder=_recorder(tmp_path, run_kind="ft"),
        )


def test_out_of_order_batch_indices_are_refused(tmp_path):
    def scrambled():
        out = list(batches_for(7, 0, n=4))
        return iter([out[0], out[2], out[1], out[3]])

    with pytest.raises(TrainerContractViolation, match="strictly increasing"):
        train_cpt(
            scrambled(), epoch=0, step=TinyStep(),
            control=_control(total_steps=4), recorder=_recorder(tmp_path),
        )


def test_a_non_finite_loss_stops_the_run_loudly(tmp_path):
    class NanStep(TinyStep):
        def accumulate(self, batch, supervision) -> float:
            super().accumulate(batch, supervision)
            return float("nan")

    ledger = Ledger(tmp_path / "ledger.jsonl")
    rec = RunRecorder(ledger, protocol=_protocol(), run_kind="cpt", repo=REPO, env=_env())
    with pytest.raises(TrainerContractViolation, match="non-finite loss"):
        train_cpt(
            batches_for(7, 0, n=4), epoch=0, step=NanStep(),
            control=_control(total_steps=4), recorder=rec,
        )
    # The recorder's guarantee still holds: the failure is on the record.
    rows = ledger.rows()
    assert len(rows) == 1 and rows[0].status == "failed"


# --- rule 4: the cap terminates the run, and the row is still written ----------------------


def test_the_wall_clock_cap_terminates_the_run_and_the_row_is_still_written(tmp_path):
    step = TinyStep()
    ledger = Ledger(tmp_path / "ledger.jsonl")
    rec = RunRecorder(ledger, protocol=_protocol(), run_kind="cpt", repo=REPO, env=_env())
    result = train_cpt(
        batches_for(7, 0, n=100),
        epoch=0,
        step=step,
        control=_control(total_steps=50, cap_s=5.0, clock=StepClock(step)),
        recorder=rec,
    )
    assert result.termination == "wall_clock_cap"
    assert result.optimizer_steps == 5, "the cap fires at the group boundary after 5 steps"

    rows = ledger.rows()
    assert len(rows) == 1
    assert rows[0].status == "completed", "hitting a cap is a termination, not a failure"
    termination = rows[0].metrics["train.termination"]
    assert termination.value == "wall_clock_cap"
    assert termination.passed is False, "a capped run must not read as a run that finished"
    assert rows[0].wall_clock_s >= 0.0


def test_the_ledger_row_carries_the_capped_cost_estimate(tmp_path):
    rec = _recorder(tmp_path)
    result = train_cpt(
        batches_for(7, 0, n=8), epoch=0, step=TinyStep(),
        control=_control(total_steps=4, cap_s=7200.0), recorder=rec,
    )
    assert rec.row is not None
    projected = rec.row.metrics["train.projected_usd_at_cap"]
    assert projected.value == pytest.approx(1.0)  # $0.50/h x 2 h at the cap
    assert "no approval required" in projected.detail
    assert result.cost_usd >= 0.0


# --- S5: resume reproduces the trajectory exactly -----------------------------------------


def _uninterrupted(tmp_path: Path, n_steps: int) -> LossLog:
    return train_cpt(
        batches_for(7, 0, n=n_steps * 2),
        epoch=0,
        step=TinyStep(seed=11),
        control=_control(total_steps=n_steps),
        recorder=_recorder(tmp_path / "whole"),
    ).loss_log


def test_a_run_killed_at_half_way_and_resumed_reproduces_the_loss_trajectory(tmp_path):
    """S5. Ten steps in one go, versus ten steps across a kill at five."""
    reference = _uninterrupted(tmp_path, 10)
    assert len(reference) == 10

    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20),
        epoch=0,
        step=step,
        control=_control(total_steps=10, cap_s=5.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "first"),
    )
    assert first.termination == "wall_clock_cap"
    assert first.optimizer_steps == 5
    assert first.checkpoint.position.index == 5

    resumed = train_cpt(
        batches_for(7, 0, n=20),
        epoch=0,
        step=TinyStep(seed=999),  # deliberately the wrong weights; the checkpoint supplies them
        control=_control(total_steps=10),
        recorder=_recorder(tmp_path / "second"),
        resume_from=first.checkpoint,
    )
    assert resumed.termination == "steps_exhausted"
    assert resumed.loss_log == reference
    assert resumed.loss_log.digest() == reference.digest()
    assert resumed.loss_log.losses() == reference.losses()


def test_the_resume_survives_a_round_trip_through_a_checkpoint_file(tmp_path):
    reference = _uninterrupted(tmp_path, 8)

    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=16),
        epoch=0,
        step=step,
        control=_control(total_steps=8, cap_s=4.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "first"),
    )
    path = first.checkpoint.write(tmp_path / "ckpt" / "run.json")
    from_disk = Checkpoint.read(path)

    resumed = train_cpt(
        batches_for(7, 0, n=16),
        epoch=0,
        step=TinyStep(seed=999),
        control=_control(total_steps=8),
        recorder=_recorder(tmp_path / "second"),
        resume_from=from_disk,
    )
    assert resumed.loss_log.digest() == reference.digest()


def test_checkpoints_are_taken_at_the_configured_interval(tmp_path):
    taken: list[Checkpoint] = []
    train_cpt(
        batches_for(7, 0, n=20),
        epoch=0,
        step=TinyStep(),
        control=_control(total_steps=9, checkpoint_every=3),
        recorder=_recorder(tmp_path),
        on_checkpoint=taken.append,
    )
    assert [c.optimizer_step for c in taken] == [3, 6, 9]
    assert [c.position.index for c in taken] == [3, 6, 9]


def test_resuming_under_a_different_seed_is_refused(tmp_path):
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    with pytest.raises(TrainerContractViolation, match="different sequence of batches"):
        train_cpt(
            batches_for(8, 0, n=20), epoch=0, step=TinyStep(),
            control=_control(total_steps=10),
            recorder=_recorder(tmp_path / "b", seed=8),
            resume_from=first.checkpoint,
        )


def test_resuming_into_a_different_epoch_is_refused(tmp_path):
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    with pytest.raises(TrainerContractViolation, match="will not guess"):
        train_cpt(
            batches_for(7, 1, n=20), epoch=1, step=TinyStep(),
            control=_control(total_steps=10), recorder=_recorder(tmp_path / "b"),
            resume_from=first.checkpoint,
        )


def test_resuming_under_a_different_schedule_is_refused(tmp_path):
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    with pytest.raises(TrainerContractViolation, match="rates the original never would"):
        train_cpt(
            batches_for(7, 0, n=20), epoch=0, step=TinyStep(),
            control=_control(total_steps=20), recorder=_recorder(tmp_path / "b"),
            resume_from=first.checkpoint,
        )


def test_a_source_that_skips_the_resume_point_is_refused(tmp_path):
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    assert first.checkpoint.position.index == 3
    holed = [b for b in batches_for(7, 0, n=20) if b.index != 3]
    with pytest.raises(TrainerContractViolation, match="not that function"):
        train_cpt(
            iter(holed), epoch=0, step=TinyStep(), control=_control(total_steps=10),
            recorder=_recorder(tmp_path / "b"), resume_from=first.checkpoint,
        )


def test_a_source_with_the_right_indices_and_the_wrong_batches_is_refused(tmp_path):
    """Landing on the index is not landing on the batch.

    The dangerous source is not the one that jumps or ends early -- those were already
    refused. It is the one that agrees about every index and disagrees about every batch,
    because the resume then runs to completion and reports `steps_exhausted`.

    Measured against the pre-fix code with the real `ShardReader`: the same `(seed, epoch)`
    asked with `batch_tokens` one token larger produced 17 batches where the original
    produced 17, the skip found index 4, and the run finished on a different corpus order
    with no refusal anywhere. `batches_for(8, ...)` is that source in miniature -- same
    indices, different content -- and `seed` cannot catch it, because `seed` is compared
    against the recorder's protocol and not against the source.
    """
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    impostor = list(batches_for(8, 0, n=20))
    assert [b.index for b in impostor] == [b.index for b in batches_for(7, 0, n=20)], (
        "the impostor must share the indices, or this tests the check that already existed"
    )
    with pytest.raises(TrainerContractViolation, match="different batches"):
        train_cpt(
            iter(impostor), epoch=0, step=TinyStep(), control=_control(total_steps=10),
            recorder=_recorder(tmp_path / "b"), resume_from=first.checkpoint,
        )


def test_the_consumed_digest_is_the_evidence_and_it_travels_with_the_checkpoint(tmp_path):
    """The digest covers what was eaten, and a resume continues the same running hash.

    Without the second half, a resumed run's own final checkpoint would record only the
    batches *it* consumed, and the next resume would compare a suffix against a prefix.
    """
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    assert first.checkpoint.position.index == 3
    resumed = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=TinyStep(seed=999),
        control=_control(total_steps=10), recorder=_recorder(tmp_path / "b"),
        resume_from=first.checkpoint,
    )
    whole = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=TinyStep(seed=11),
        control=_control(total_steps=10), recorder=_recorder(tmp_path / "c"),
    )
    assert resumed.checkpoint.position == whole.checkpoint.position
    assert resumed.checkpoint.consumed_digest == whole.checkpoint.consumed_digest, (
        "an interrupted run and an uninterrupted one ate the same batches in the same order"
    )
    assert first.checkpoint.consumed_digest != whole.checkpoint.consumed_digest


def test_a_source_that_ends_before_the_resume_point_is_refused(tmp_path):
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    with pytest.raises(TrainerContractViolation, match="source ended first"):
        train_cpt(
            batches_for(7, 0, n=2), epoch=0, step=TinyStep(),
            control=_control(total_steps=10), recorder=_recorder(tmp_path / "b"),
            resume_from=first.checkpoint,
        )


def test_the_checkpoint_json_is_the_two_integers_plus_what_cannot_be_recomputed(tmp_path):
    step = TinyStep(seed=11)
    first = train_cpt(
        batches_for(7, 0, n=20), epoch=0, step=step,
        control=_control(total_steps=10, cap_s=3.0, clock=StepClock(step)),
        recorder=_recorder(tmp_path / "a"),
    )
    raw = json.loads(json.dumps(first.checkpoint.to_json()))
    assert raw["position"] == {"epoch": 0, "index": 3}
    assert raw["seed"] == 7
    assert raw["optimizer_step"] == 3
    assert raw["loss_digest"] == first.loss_log.digest()
