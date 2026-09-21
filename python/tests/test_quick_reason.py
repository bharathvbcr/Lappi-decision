"""`quick_reason` has to describe the run it is written on, not the run it was written for.

The sentence this replaced -- *"the corpus is this repository's own sources rather than the
pool the plan names"* -- was true of all 988 rows, because no run had ever been given
another corpus. It was a constant that read like a finding. The commitpackft corpus exists
now, and a run on it would have recorded that sentence anyway: a false statement, in an
append-only ledger, about the exact condition rule 8 turns on.

The flag stays `True` in every case here. That is the point of the tests: the reason becomes
accurate without the flag becoming an agent's to clear.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

pytest.importorskip("torch")

from rung0_real_run import quick_reason_for  # noqa: E402

CLEAN = {
    "examples_supplied": True,
    "seeds": 24,
    "train_subsample": 1.0,
    "shuffled": False,
    "pool_records": 61193,
    "corpus_hash": "abc123def4567890" + "0" * 48,
}


def test_a_corpus_built_from_the_repository_says_so():
    reason = quick_reason_for(**{**CLEAN, "examples_supplied": False})
    assert "this repository's own tracked sources" in reason


def test_a_supplied_corpus_does_not_claim_to_be_the_repository_s_own_sources():
    """The defect this fixes: the old sentence asserted it unconditionally."""
    assert "this repository's own tracked sources" not in quick_reason_for(**CLEAN)


def test_too_few_seeds_is_named_with_the_count():
    assert "2 seed(s), fewer than the 3" in quick_reason_for(**{**CLEAN, "seeds": 2})


def test_a_subsample_is_named_with_its_share():
    reason = quick_reason_for(**{**CLEAN, "train_subsample": 0.25})
    assert "subsampled to 25%" in reason


def test_a_shuffled_arm_says_it_is_a_control_and_not_a_measurement():
    reason = quick_reason_for(**{**CLEAN, "shuffled": True})
    assert "control and not a measurement" in reason


def test_several_conditions_are_all_reported_not_just_the_first():
    """A row that is quick for three reasons and a row that is quick for one must not
    read alike -- fixing only the named one would leave the other two in place."""
    reason = quick_reason_for(
        **{**CLEAN, "examples_supplied": False, "seeds": 1, "train_subsample": 0.5}
    )
    assert "tracked sources" in reason
    assert "1 seed(s)" in reason
    assert "subsampled to 50%" in reason


def test_the_corpus_identity_is_always_carried():
    """Whatever else it says, the row must let a human find the corpus it ran on."""
    for overrides in ({}, {"seeds": 1}, {"examples_supplied": False}, {"shuffled": True}):
        reason = quick_reason_for(**{**CLEAN, **overrides})
        assert "abc123def4567890" in reason
        assert "61193 pool record(s)" in reason


def test_a_missing_pool_record_count_is_omitted_rather_than_invented():
    reason = quick_reason_for(**{**CLEAN, "pool_records": None})
    assert "pool record(s)" not in reason
    assert "abc123def4567890" in reason


def test_a_clean_run_still_refuses_to_promote_itself():
    """The whole reason the flag is not derived: an agent must not be able to argue that
    the condition no longer applies and therefore the run may promote."""
    reason = quick_reason_for(**CLEAN)
    assert "no rule-8 condition this run can determine stands" in reason
    assert "rule 2 makes a human's" in reason


def test_a_clean_run_does_not_claim_the_corpus_is_the_plan_s_pool():
    """`--examples` accepts any file. Inferring the pool's identity from a record count
    would be a guess wearing the clothes of a check."""
    reason = quick_reason_for(**CLEAN)
    assert "is not something this run can verify" in reason
