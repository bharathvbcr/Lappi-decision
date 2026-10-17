"""The needle gate's failing verdict names WHERE the failing bucket sits, not a fixed cause.

The old string appended "A recurrent model losing early context looks exactly like this" to
every failure, whichever bucket failed. v5's misses are all at the END of the context
(depth fraction 1.0 = end), which that sentence contradicts
(GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06). The wording is
free text: `passed`, `value` and `n` are the gate and must not move (rule 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.needle import DEPTH_BUCKETS, build_suite, score_suite
from qd_train.tristate import Ran

_POSITION = {
    "0-20%": "start",
    "20-40%": "early-middle",
    "40-60%": "middle",
    "60-80%": "late-middle",
    "80-100%": "end",
}


def _scored(failing_bucket: str | None):
    cases = build_suite(target_tokens=1024, cases_per_depth=6)
    preds: dict[str, int | None] = {}
    for c in cases:
        wrong = c.depth_bucket == failing_bucket
        preds[c.case_id] = (c.needle_index + 1) if wrong else c.needle_index
    return cases, preds, score_suite(cases, preds, min_recall=0.95)


@pytest.mark.parametrize("bucket", DEPTH_BUCKETS)
def test_the_failing_verdict_names_the_failing_buckets_position(bucket):
    _, _, (_, state) = _scored(bucket)
    assert isinstance(state, Ran) and not state.passed
    assert f"depth {bucket}" in state.detail
    assert f"({_POSITION[bucket]} of the context)" in state.detail


@pytest.mark.parametrize("bucket", DEPTH_BUCKETS)
def test_no_verdict_asserts_a_fixed_cause(bucket):
    _, _, (_, state) = _scored(bucket)
    assert "recurrent model" not in state.detail
    assert "looks exactly like" not in state.detail


def test_the_wording_change_moves_neither_the_gate_nor_its_counts():
    cases, _, (report, state) = _scored("80-100%")
    assert state.value == 0.0
    assert state.n == state.n_total == len(cases)
    assert report.aggregate_correct == len(cases) - report.by_depth[-1].total


def test_a_passing_gate_carries_no_failure_verdict():
    _, _, (_, state) = _scored(None)
    assert state.passed and "FAILS" not in state.detail


def test_the_thin_bucket_refusal_does_not_assert_a_cause():
    cases = build_suite(target_tokens=1024, cases_per_depth=2)
    preds = {c.case_id: c.needle_index for c in cases}
    _, state = score_suite(cases, preds, min_recall=0.95)
    assert "recurrent model fails" not in state.reason
