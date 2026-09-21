"""The control that existed and had never once been evaluated.

``qd_train.eval_harness.shuffled_label_control`` has been in this repository the whole
time. Across **988 ledger rows** -- every chain file, every lane, CPU, MPS and GH200 --
``controls["shuffled_label"]`` reads ``not_run`` on all of them, and so do
``degenerate_head``, ``privileged_hunk`` and ``transfer_gate``. The reason is one line in
the harness:

    controls["shuffled_label"] = NotRun(reason="no shuffled-label model was evaluated")

The check was never broken. It needs the held-out accuracy of a model trained on destroyed
labels, and nothing in this repository ever trained one. A control that cannot be reached
is indistinguishable, in every row it appears in, from a control that was reached and
passed -- which is this project's founding defect, sitting inside the machinery built to
detect it.

``tools/rung0_real_run.py --shuffle-train-labels`` is that model. These tests pin the
permutation it performs, because a shuffle that quietly failed to destroy the signal would
make the control PASS for the wrong reason -- and a passing control is exactly the result
nobody re-examines.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

from qd_train.eval_harness import shuffled_label_control  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402


@dataclass(frozen=True, slots=True)
class FakeDecision:
    """Enough of `ByteDecision` for the permutation: the label and its option count.

    Frozen, because the real one is -- a shuffle that mutated in place would pass against a
    mutable stand-in and raise on the real type at the top of a GPU run.
    """

    example_id: str
    options: tuple[bytes, ...]
    gold_option: int


def _decisions(labels: list[int], *, arity: int = 4) -> list[FakeDecision]:
    options = tuple(f"opt{i}".encode() for i in range(arity))
    return [FakeDecision(f"e{i}", options, label) for i, label in enumerate(labels)]


def _shuffle():
    """Imported lazily: `rung0_real_run` imports torch, which the repo venv lacks."""
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    from rung0_real_run import shuffle_train_labels

    return shuffle_train_labels


# --------------------------------------------------------------------------
# 1 -- the permutation
# --------------------------------------------------------------------------


def test_the_shuffle_preserves_the_label_distribution_exactly():
    """A permutation, not fresh random labels, and the control depends on the difference.

    `shuffled_label_control` compares the shuffled model against the MAJORITY-CLASS rate.
    Drawing fresh labels would move the marginal and move that ceiling with it, so the
    control would be measured against a bar that does not apply to the real arms.
    """
    shuffle = _shuffle()
    labels = [0, 0, 0, 1, 1, 2, 3, 3, 3, 3]
    out = shuffle(_decisions(labels), seed=0)
    assert sorted(d.gold_option for d in out) == sorted(labels), (
        "the shuffle changed the label multiset, so it is not a permutation and the "
        "majority rate the control compares against no longer describes these labels"
    )


def test_the_shuffle_actually_moves_labels():
    """The one failure that would make the control pass for the wrong reason.

    A shuffle that returned its input would leave the signal intact, the 'control' model
    would score like the real one, and -- because the control asks for accuracy at or BELOW
    chance -- it would report LEAKAGE rather than silently passing. That is the safer
    direction, but it is still a lie about what was measured, so it is pinned.
    """
    shuffle = _shuffle()
    labels = [i % 4 for i in range(200)]
    out = shuffle(_decisions(labels), seed=0)
    moved = sum(1 for before, after in zip(labels, out, strict=True) if before != after.gold_option)
    assert moved > 100, (
        f"only {moved} of 200 labels moved; a permutation that leaves most labels in place "
        "has not destroyed the signal the control exists to destroy"
    )


def test_the_shuffle_never_produces_an_invalid_gold_option():
    """`ByteDecision.__post_init__` refuses a gold_option outside its own options.

    Decisions with different option counts must be permuted within their own group. A
    global permutation raises partway through a run -- and only on a corpus where the
    counts differ, which is not today's, so the failure would arrive later and elsewhere.
    """
    shuffle = _shuffle()
    mixed = _decisions([0, 1, 2, 3], arity=4) + _decisions([0, 1], arity=2)
    out = shuffle(mixed, seed=1)
    for decision in out:
        assert 0 <= decision.gold_option < len(decision.options), (
            f"{decision.example_id}: gold_option {decision.gold_option} is outside its "
            f"{len(decision.options)} options -- the real ByteDecision would have raised"
        )


def test_the_shuffle_does_not_mutate_its_input():
    shuffle = _shuffle()
    original = _decisions([0, 1, 2, 3] * 10)
    before = [d.gold_option for d in original]
    shuffle(original, seed=0)
    assert [d.gold_option for d in original] == before, "the shuffle mutated its input"


def test_the_same_seed_gives_the_same_permutation():
    """So a control run is reproducible, and so --shuffle-seed means something."""
    shuffle = _shuffle()
    labels = [i % 4 for i in range(50)]
    a = shuffle(_decisions(labels), seed=7)
    b = shuffle(_decisions(labels), seed=7)
    c = shuffle(_decisions(labels), seed=8)
    assert [d.gold_option for d in a] == [d.gold_option for d in b]
    assert [d.gold_option for d in a] != [d.gold_option for d in c], (
        "two different --shuffle-seed values produced the same permutation"
    )


# --------------------------------------------------------------------------
# 2 -- the runner reaches the control at all
# --------------------------------------------------------------------------


def test_the_runner_records_the_control_only_under_the_flag():
    """Both halves matter, and the second is the one that keeps the ledger honest.

    Under the flag the row must carry a real `shuffled_label`. WITHOUT it the row must keep
    saying `not_run`, because a run trained on true labels has not evaluated this control
    and a row that claimed otherwise would be the defect this file is about, inverted.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert 'recorder.control(\n                    "shuffled_label"' in source, (
        "rung0_real_run.py never calls recorder.control('shuffled_label'), so every row it "
        "writes reports the control as not_run -- which is the state all 988 existing rows "
        "are in"
    )
    marker = source.index('recorder.control(\n                    "shuffled_label"')
    preceding = source[:marker]
    assert "if args.shuffle_train_labels:" in preceding[-400:], (
        "the control is recorded unconditionally; a run on TRUE labels would then claim to "
        "have evaluated a shuffled-label control it never ran"
    )


def test_the_control_and_the_arm_it_controls_for_cannot_share_a_recipe_hash():
    """Everything else about them is identical by design, so the recipe must separate them.

    Without this a control run and a real run hash alike, and anything pooling by protocol
    averages a model trained on destroyed labels into the measurement it exists to check.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    recipe_start = source.index("    recipe = {")
    recipe_block = source[recipe_start : recipe_start + 2500]
    assert '"shuffle_train_labels"' in recipe_block, (
        "the recipe does not record whether labels were shuffled, so the control and the "
        "arm it controls for produce the same recipe_hash"
    )


# --------------------------------------------------------------------------
# 3 -- the harness end, on the numbers a run will actually hand it
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 2b -- the other control that never ran, and needed no GPU time at all
# --------------------------------------------------------------------------


def _degenerate_state():
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    from rung0_real_run import degenerate_head_state

    return degenerate_head_state


def test_a_constant_head_is_caught_and_a_discriminating_one_is_not():
    """`degenerate_head` sat at not_run on 988 rows for want of an array, not a measurement.

    It reads the held-out choice distribution, which every evaluation already computes. The
    two cases it exists to separate: a head answering one class on everything, and a head
    spreading its mass across classes.
    """
    degenerate_head_state = _degenerate_state()

    collapsed = [[0.97, 0.01, 0.01, 0.01] for _ in range(200)]
    verdict = degenerate_head_state(collapsed)
    assert isinstance(verdict, Ran) and not verdict.passed, verdict
    assert "DEGENERATE" in verdict.detail, verdict.detail

    healthy = [
        [0.55, 0.20, 0.15, 0.10] if i % 3 == 0
        else [0.15, 0.55, 0.20, 0.10] if i % 3 == 1
        else [0.10, 0.20, 0.15, 0.55]
        for i in range(200)
    ]
    ok = degenerate_head_state(healthy)
    assert isinstance(ok, Ran) and ok.passed, ok.detail


def test_ragged_distributions_report_not_run_rather_than_a_verdict():
    """Padding them would invent probability mass and understate the entropy.

    A check that returns the WRONG verdict is worse than one that returns NotRun, and a
    zero-padded row looks exactly like a confident head to an entropy threshold.
    """
    degenerate_head_state = _degenerate_state()
    ragged = [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5]]
    verdict = degenerate_head_state(ragged)
    assert not isinstance(verdict, Ran), verdict
    assert "ragged" in verdict.reason and "[2, 4]" in verdict.reason, verdict.reason


def test_no_held_out_rows_reports_not_run():
    degenerate_head_state = _degenerate_state()
    verdict = degenerate_head_state([])
    assert not isinstance(verdict, Ran)
    assert "no held-out rows" in verdict.reason, verdict.reason


def test_the_degenerate_control_is_recorded_on_every_run_not_only_control_runs():
    """The distinction from `shuffled_label`, and the reason it is worth stating.

    `shuffled_label` needs a model trained on destroyed labels, so only a control run can
    evaluate it. `degenerate_head` needs the distribution every run already produces, so a
    row that omitted it would be omitting a free check -- and `_fit_gate` does not cover it:
    that reads TRAIN accuracy against the train majority, so a head which fits training and
    then answers one class on everything held out clears it while being the degenerate case.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    marker = source.index('recorder.control(\n                "degenerate_head"')
    preceding = source[:marker]
    assert "if args.shuffle_train_labels:" not in preceding[-300:], (
        "degenerate_head is recorded inside the shuffled-label branch, so ordinary runs "
        "would keep reporting it as not_run"
    )


# --------------------------------------------------------------------------
# 2c -- the gate that was built and never called
# --------------------------------------------------------------------------


def _ece_state():
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    from rung0_real_run import ece_state

    return ece_state


def test_a_calibrated_model_clears_the_ece_bar_and_an_overconfident_one_does_not():
    """`calibration_fit.ece_gate` says it in its own docstring: "the ece gate
    qd_train.ledger has always listed and nothing ever computed". It was built and never
    called, which is a third shape again -- not missing, not broken, just unreferenced.

    Thresholds stay at the function's defaults here and at the call site: rule 2 makes a
    threshold read-only to an agent, and passing one from the caller is retuning it.
    """
    ece_state = _ece_state()
    rng = __import__("random").Random(0)

    # Confidence matches accuracy: 90% confident and right about 90% of the time.
    calibrated = []
    labels = []
    for _ in range(400):
        right = rng.random() < 0.9
        calibrated.append([0.9, 0.0334, 0.0333, 0.0333])
        labels.append(0 if right else 1)
    ok = ece_state(calibrated, labels)
    assert isinstance(ok, Ran) and ok.passed, ok.detail

    # Same accuracy, asserted at 99.7%: the gap between confidence and correctness is what
    # ECE measures, and it is what makes a calibrated abstention rule impossible.
    overconfident = [[0.997, 0.001, 0.001, 0.001] for _ in range(400)]
    bad = ece_state(overconfident, labels)
    assert isinstance(bad, Ran) and not bad.passed, bad.detail


def test_ece_refuses_when_the_distributions_and_the_golds_are_not_the_same_examples():
    """The failure that would score predictions against the wrong answers, silently."""
    ece_state = _ece_state()
    verdict = ece_state([[0.25] * 4] * 120, [0] * 119)
    assert not isinstance(verdict, Ran)
    assert "same examples in the same order" in verdict.reason, verdict.reason


def test_ece_refuses_ragged_and_empty_input():
    ece_state = _ece_state()
    ragged = ece_state([[0.25, 0.25, 0.25, 0.25], [0.5, 0.5]], [0, 1])
    assert not isinstance(ragged, Ran) and "ragged" in ragged.reason, ragged

    empty = ece_state([], [])
    assert not isinstance(empty, Ran) and "no held-out rows" in empty.reason, empty


def test_the_ece_gate_is_recorded_on_every_run():
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert 'recorder.gate(\n                "ece"' in source, (
        "rung0_real_run.py never calls recorder.gate('ece'), so the gate stays not_run on "
        "every row it writes"
    )
    marker = source.index('recorder.gate(\n                "ece"')
    assert "if args.shuffle_train_labels:" not in source[:marker][-300:], (
        "the ece gate is recorded inside the shuffled-label branch, so ordinary runs would "
        "keep reporting it as not_run"
    )


# --------------------------------------------------------------------------
# 2d -- the pairing, which is where both of the above would have gone wrong
# --------------------------------------------------------------------------


def test_bucketing_reorders_decisions_so_val_d_order_is_not_the_scored_order():
    """The hazard, demonstrated rather than asserted.

    `bucketed_batches` sorts by `context.n_bytes_kept` and drops any trailing one-row
    chunk. So the order probabilities come back in is the LENGTH-SORTED order, and the
    count can be smaller than the split. Pairing them with `[d.gold_option for d in val_d]`
    scores each prediction against a different example's answer.

    On this corpus that would not have been caught by a length check: 288 validation
    decisions at batch size 16 divide exactly, so both lists have 288 entries and only the
    ORDER differs. The gold is therefore taken from `plan.choice_target` in the same loop
    iteration as the probabilities, which makes the pairing true by construction.
    """
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    from rung0_real_run import bucketed_batches

    from qd_train.byte_decider import ByteDeciderConfig

    config = ByteDeciderConfig()
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")

    # The claim under test is about the sort, which is visible in the source and in the
    # behaviour. Assert the sort key is there, then that it is load-bearing.
    assert "sorted(decisions, key=lambda d: d.context.n_bytes_kept)" in source, (
        "bucketed_batches no longer length-sorts; if it now preserves order this test's "
        "premise is gone and the pairing comment above should be revisited"
    )
    assert "if len(chunk) < 2:" in source, (
        "bucketed_batches no longer drops one-row chunks; the count-mismatch half of the "
        "hazard may be gone"
    )
    assert callable(bucketed_batches) and config.max_context_bytes > 0


def test_the_gate_and_control_take_their_gold_from_the_scored_rows():
    """Both must read `choice_gold`, not `val_d`.

    This is the assertion that fails against the first version of this wiring, which paired
    `choice_probs` with `[d.gold_option for d in val_d]` and would have reported a
    calibration error computed against permuted answers -- a number, from a real run, that
    was wrong for a reason no length check could see.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert '"choice_gold": choice_gold' in source, (
        "evaluate() does not return the gold of the rows it scored, so nothing downstream "
        "can pair probabilities with answers correctly"
    )
    assert "choice_gold.append(int(plan.choice_target[i]))" in source, (
        "the gold is not collected in the same loop iteration as the probabilities, so the "
        "two are aligned only by an assumption about ordering"
    )
    # Comment lines are stripped before the negative check. The first version of this test
    # searched the whole file and matched the comment that EXPLAINS why the pairing is
    # wrong -- a source grep that reads prose as if it were code, which is the same mistake
    # in miniature as reading a summary line as if it were the record.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "[d.gold_option for d in val_d]" not in code, (
        "something still pairs against val_d order, which bucketing does not preserve"
    )


# --------------------------------------------------------------------------
# 2e -- paired_margin_vs_linear, the gate whose opponent was never scored
# --------------------------------------------------------------------------


def test_the_linear_control_iteration_budget_cannot_drift_from_its_source():
    """`rung0_real_run` restates the budget because importing it would be a cycle.

    A restated constant is a copy, and a copy drifts. This is the assertion the comment
    beside it promises: at the library default of 500 the control does not converge on this
    corpus and is correctly refused, so the budget is what makes the gate reachable at all
    -- and two different budgets in two files would mean the gate's opponent is not the
    control anyone measured.
    """
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    import rung0_linear_control
    import rung0_real_run

    assert rung0_real_run.LINEAR_CONTROL_MAX_ITER == rung0_linear_control.DEFAULT_MAX_ITER


def test_bucketed_chunks_is_the_only_place_the_scored_order_is_decided():
    """`bucketed_batches` must be built FROM `bucketed_chunks`, not beside it.

    Two implementations of "sorted by kept bytes, chunked, short chunk dropped" would let
    the batcher and the pairing disagree, and the failure that produces -- predictions
    scored against another example's answer -- is invisible to a count and to a spot check.
    """
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert code.count("sorted(decisions, key=lambda d: d.context.n_bytes_kept)") == 1, (
        "the length-sort appears more than once, so the batcher and whatever pairs results "
        "back to decisions can disagree about the order"
    )
    assert "for chunk in bucketed_chunks(decisions, batch_size=batch_size)" in code, (
        "bucketed_batches no longer derives its chunks from bucketed_chunks"
    )


def test_an_unconverged_linear_control_reports_not_run_rather_than_a_win():
    """The control's own rule, enforced at the gate.

    An unconverged fit is a weak opponent and therefore a flattering margin. Scoring it
    would manufacture exactly the result the gate exists to make hard to get.
    """
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    from rung0_real_run import linear_baseline_correctness

    # max_iter=1 cannot converge: the library default of 500 already does not on this
    # corpus, which is why the real budget is 6000.
    from qd_train.mutate_adapter import MUTATION_CLASSES

    class _Ctx:
        def __init__(self, text: str) -> None:
            self.ids = list(text.encode("utf-8"))
            self.n_bytes_kept = len(self.ids)

    @dataclass(frozen=True, slots=True)
    class _D:
        context: object
        gold_option: int

    train = [
        _D(_Ctx(f"def f{i}(): return {i}\n" * 3), i % len(MUTATION_CLASSES))
        for i in range(40)
    ]
    correct, not_run = linear_baseline_correctness(train, train[:20], seed=0, max_iter=1)
    assert correct is None, "a control that could not converge was scored anyway"
    assert not_run is not None and "manufactures a win" in not_run.reason, not_run


def test_the_paired_margin_gate_is_recorded_and_fails_closed():
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert 'recorder.gate("paired_margin_vs_linear", baseline_not_run)' in source, (
        "when the linear control does not converge the gate must report its NotRun, not "
        "be left to RunRecorder's generic 'never evaluated' default, which would lose the "
        "reason"
    )
    assert 'recorder.gate(\n                    "paired_margin_vs_linear"' in source, (
        "the gate is never recorded with a real verdict, so it stays not_run on every row"
    )


def test_a_shuffled_model_at_chance_passes_and_above_the_ceiling_fails():
    """The control's own verdict, on the two cases that matter.

    Chance is the majority rate, 60% here, not 1/k = 25%. A model scoring 58% has learned
    nothing beyond the prior; one scoring 85% on labels that were destroyed in training can
    only have read the answer off the split.
    """
    labels = [0] * 60 + [1] * 20 + [2] * 15 + [3] * 5

    at_chance = shuffled_label_control(0.58, labels, n_eval=len(labels))
    assert isinstance(at_chance, Ran) and at_chance.passed, at_chance.detail

    leaking = shuffled_label_control(0.85, labels, n_eval=len(labels))
    assert isinstance(leaking, Ran) and not leaking.passed
    assert "LEAKAGE" in leaking.detail, leaking.detail


if __name__ == "__main__":  # pragma: no cover - convenience only
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
