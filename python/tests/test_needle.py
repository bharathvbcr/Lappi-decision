"""The needle-hunk suite: depth coverage, and refusing to average away a failure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.needle import (  # noqa: E402
    build_suite,
    score_suite,
    wilson_interval,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _suite(**kw):
    return build_suite(target_tokens=kw.pop("target_tokens", 1024), cases_per_depth=kw.pop("cpd", 6), **kw)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_every_depth_bucket_is_populated_by_construction():
    """A randomly-placed needle set leaves the early buckets thin — which is exactly
    where a recurrent model fails and where the estimate must be most certain."""
    cases = _suite()
    buckets = {c.depth_bucket for c in cases}
    assert buckets == {"0-20%", "20-40%", "40-60%", "60-80%", "80-100%"}


def test_the_haystack_reaches_the_target_size():
    cases = build_suite(target_tokens=4096, cases_per_depth=2)
    assert all(c.approx_tokens >= 4096 * 0.8 for c in cases), \
        [c.approx_tokens for c in cases[:5]]


def test_the_needle_is_where_the_case_says_it_is():
    """The span convention must match the runtime's: 1-based, inclusive both ends."""
    for case in _suite():
        lines = case.context.splitlines()
        needle_lines = lines[case.needle_start_line - 1 : case.needle_end_line]
        body = "\n".join(needle_lines)
        # The silent-stub needle always introduces a zero-returning body.
        assert any(tok in body for tok in ("Ok(0)", "return 0", "return 0, nil")), \
            f"{case.case_id}: recorded span does not contain the needle:\n{body}"


def test_the_needle_is_not_findable_by_lexical_marker():
    """A `todo!()`-shaped needle would measure the tokenizer, not recall."""
    for case in _suite():
        assert "todo!()" not in case.context
        assert "NotImplementedError" not in case.context
        assert "unimplemented!" not in case.context


def test_generation_is_deterministic():
    a = build_suite(target_tokens=1024, cases_per_depth=4, seed=3)
    b = build_suite(target_tokens=1024, cases_per_depth=4, seed=3)
    assert [c.case_id for c in a] == [c.case_id for c in b]
    assert [c.context for c in a] == [c.context for c in b]


def test_a_different_seed_gives_a_different_suite():
    a = build_suite(target_tokens=1024, cases_per_depth=4, seed=1)
    b = build_suite(target_tokens=1024, cases_per_depth=4, seed=2)
    assert [c.case_id for c in a] != [c.case_id for c in b]


def test_case_ids_are_unique():
    cases = _suite(cpd=10)
    assert len({c.case_id for c in cases}) == len(cases)


@pytest.mark.parametrize("bad", [0, 64, 255])
def test_an_undersized_target_is_refused(bad: int):
    with pytest.raises(ValueError, match="at least 256"):
        build_suite(target_tokens=bad)


def test_an_unknown_language_is_refused():
    with pytest.raises(ValueError, match="no filler hunks"):
        build_suite(target_tokens=512, cases_per_depth=2, languages=("cobol",))


# --------------------------------------------------------------------------
# Scoring — the gate is the worst bucket, not the average
# --------------------------------------------------------------------------

def test_a_perfect_model_passes():
    cases = _suite()
    preds = {c.case_id: c.needle_index for c in cases}
    report, gate = score_suite(cases, preds, min_recall=0.8)
    assert isinstance(gate, Ran) and gate.passed is True
    assert report.aggregate_recall == 1.0


def test_a_model_that_lost_early_context_fails_despite_a_strong_aggregate():
    """The central failure this suite exists to catch.

    The model finds every needle except those in the first 20% of the context. Its
    aggregate recall is ~0.8, which reads fine. The gate must still fail it.
    """
    cases = _suite(cpd=10)
    preds = {
        c.case_id: (None if c.depth_bucket == "0-20%" else c.needle_index)
        for c in cases
    }
    report, gate = score_suite(cases, preds, min_recall=0.8)

    assert report.aggregate_recall > 0.7, "aggregate should still look healthy"
    assert isinstance(gate, Ran) and gate.passed is False
    assert "0-20%" in gate.detail
    early = next(b for b in report.by_depth if b.label == "0-20%")
    assert early.recall == 0.0


def test_a_missing_prediction_is_not_run_not_a_failure():
    """An unevaluated suite is not a failing one."""
    cases = _suite()
    preds = {c.case_id: c.needle_index for c in cases}
    preds.pop(cases[0].case_id)
    _, gate = score_suite(cases, preds, min_recall=0.8)
    assert isinstance(gate, NotRun)
    assert "no prediction" in gate.reason


def test_thin_depth_buckets_are_not_run():
    cases = _suite(cpd=2)  # 2 per depth < min_per_bucket
    preds = {c.case_id: c.needle_index for c in cases}
    _, gate = score_suite(cases, preds, min_recall=0.8, min_per_bucket=5)
    assert isinstance(gate, NotRun)
    assert "too thin" in gate.reason


def test_an_empty_suite_is_not_run():
    _, gate = score_suite([], {}, min_recall=0.8)
    assert isinstance(gate, NotRun)


def test_the_threshold_is_required_and_has_no_default():
    """A gate function that defaults to passing is the fail-open shape this repo removes."""
    cases = _suite()
    preds = {c.case_id: c.needle_index for c in cases}
    with pytest.raises(TypeError):
        score_suite(cases, preds)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_an_out_of_range_threshold_is_refused(bad: float):
    with pytest.raises(ValueError, match="recall in"):
        score_suite(_suite(), {}, min_recall=bad)


def test_abstaining_everywhere_fails_rather_than_erroring():
    cases = _suite()
    _, gate = score_suite(cases, {c.case_id: None for c in cases}, min_recall=0.5)
    assert isinstance(gate, Ran) and gate.passed is False


# --------------------------------------------------------------------------
# Wilson
# --------------------------------------------------------------------------

def test_wilson_stays_inside_zero_one_at_the_extremes():
    """The reason for Wilson over the normal approximation: a passing model sits at
    recall ~1.0, where the normal interval runs past 1."""
    lo, hi = wilson_interval(20, 20)
    assert 0.0 <= lo <= hi <= 1.0 and hi == 1.0 and lo > 0.7
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25


def test_wilson_narrows_with_n():
    narrow = wilson_interval(90, 100)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_on_zero_samples_is_not_a_claim():
    assert wilson_interval(0, 0) == (0.0, 0.0)
