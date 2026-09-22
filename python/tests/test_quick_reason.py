"""`quick_reason` has to describe the run it is written on, not the run it was written for.

The sentence this replaced -- *"the corpus is this repository's own sources rather than the
pool the plan names"* -- was true of all 988 rows, because no run had ever been given
another corpus. It was a constant that read like a finding. The commitpackft corpus exists
now, and a run on it would have recorded that sentence anyway: a false statement, in an
append-only ledger, about the exact condition rule 8 turns on.

The flag stays `True` in every case here. That is the point of the tests: the reason becomes
accurate without the flag becoming an agent's to clear.

The same defect then shipped one tool over. `tools/real_tokenizer_pipeline.py` gained
`--commitpackft` with its reason still a literal naming this repository's history, and its
first row from the download (0c3fd775) carries that false sentence. `tools/rung0_linear_control.py`
had the same literal and had not yet been run on another corpus. Both reasons are now
computed, and tested here beside the one that was fixed first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

pytest.importorskip("torch")

import real_tokenizer_pipeline as pipeline  # noqa: E402
import rung0_linear_control as control  # noqa: E402
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


def test_the_shard_pipeline_on_its_own_history_still_says_so():
    assert "this repository alone" in pipeline.quick_reason_for(commitpackft_rows=None)


def test_a_shard_set_from_the_download_does_not_claim_to_be_this_repository():
    """Row 0c3fd775's defect: the code rows were a 2000-row sample of the download."""
    reason = pipeline.quick_reason_for(commitpackft_rows=(2000, 69893))
    assert "this repository alone" not in reason
    assert "2000-of-69893 sha256-ordered sample" in reason
    assert "bigcode/commitpackft" in reason


def test_a_shard_set_from_the_whole_download_is_not_called_a_sample_yet_stays_quick():
    """An uncapped read is the whole code source, and the reason says so -- but one seed
    and the stand-in span prose are still true of it, so it is still a reason."""
    reason = pipeline.quick_reason_for(commitpackft_rows=(69893, 69893))
    assert "all 69893 rows" in reason
    assert "ordered sample" not in reason
    assert "a subsample is marked quick" not in reason, "rule 8 quoted as if this were one"
    assert "one seed" in reason
    assert "fewer than 3 seeds" in reason
    assert "standing in for rajpurkar/squad_v2" in reason


def test_the_linear_control_does_not_claim_where_its_corpus_came_from():
    """`--examples` accepts any file, so the control names the snapshot and says it cannot
    tell, rather than asserting the repository's corpus as its one row once truthfully did."""
    reason = control.quick_reason_for(data_snapshot_hash="9ddbae6d" + "0" * 56, n_examples=40)
    assert "drawn from this repository" not in reason
    assert "9ddbae6d00000000" in reason
    assert "40 example(s)" in reason
    assert "not checked here" in reason
