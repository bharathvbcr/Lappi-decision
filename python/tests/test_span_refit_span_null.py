"""Span-refit span null (pre-registration amendment 2).

Closes GAP-W4A-SPAN-REFIT-ROWS-CARRY-NO-SPAN-NULL-2026-10-18 in code (the real run is not run).

Scorer: ``tools/span_refit_score.py`` emits each pointer's own top-1 over the pointing rows and
the random-+-line null over the same rows, as the three metrics ``span_beats_null`` reads.
Recorder: ``tools/span_refit_rows.py`` writes them (and refuses results without them).
Verifier: ``LedgerRow.check_span_refit_span_null`` refuses a completed span-refit row recorded
under the amendment-2 scorer sha that lacks them. Pins: the sha the ledger names is the scorer's.

``SPAN_REFIT_SCORER`` (a path) points the scorer tests at another scorer file: the fails-before
run points it at the pre-amendment scorer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from test_ledger import _env, _protocol

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import span_refit_rows as srr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402

from qd_train import ledger as L  # noqa: E402
from qd_train.span_null import (  # noqa: E402
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
    span_beats_null_state,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

SCORER = Path(os.environ.get("SPAN_REFIT_SCORER", REPO / "tools" / "span_refit_score.py"))
PREREG = REPO / "campaign" / "span-refit-mac-preregistered.json"
THREE = (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC, SPAN_NULL_METRIC)


def _scorer():
    spec = importlib.util.spec_from_file_location("span_refit_score_under_test", SCORER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vc(mod):
    """A ValCache over hand-made rows (no cache on disk): 3 pointing rows and 1 abstaining.

    row 0: 4 candidates, '+' at 1 and 2, gold (1, 1)  -> null 1/2
    row 1: 3 candidates, '+' at 0,       gold (0, 0)  -> null 1
    row 2: 5 candidates, '+' at 3,       gold (1, 2)  -> gold is not a '+' line: null 0
    row 3: abstaining
    """
    vc = mod.ValCache.__new__(mod.ValCache)
    vc.rows = [
        {"abstaining": False, "n_candidates": 4, "plus_columns": [1, 2], "gold_start": 1,
         "gold_end": 1},
        {"abstaining": False, "n_candidates": 3, "plus_columns": [0], "gold_start": 0,
         "gold_end": 0},
        {"abstaining": False, "n_candidates": 5, "plus_columns": [3], "gold_start": 1,
         "gold_end": 2},
        {"abstaining": True, "n_candidates": 2, "plus_columns": [], "gold_start": -1,
         "gold_end": -1},
    ]
    vc.abstaining = np.array([False, False, False, True])
    vc.gold = np.array([[1, 1], [0, 0], [1, 2], [-1, -1]])
    return vc


def test_pointer_top1_counts_each_pointer_over_the_pointing_rows_only() -> None:
    vc = _vc(_scorer())
    # start hits rows 0 and 2, end hits rows 0 and 1; the abstaining row's tops are ignored
    # even when they equal its (meaningless) gold.
    tops = np.array([[1, 1], [2, 0], [1, 4], [-1, -1]])
    got = vc.pointer_top1(tops)
    assert got[SPAN_START_TOP1_METRIC] == {"n": 2, "n_total": 3}
    assert got[SPAN_END_TOP1_METRIC] == {"n": 2, "n_total": 3}


def test_a_row_whose_end_could_not_be_derived_is_an_end_miss() -> None:
    vc = _vc(_scorer())
    tops = np.array([[1, 1], [0, 0], [1, 2], [3, 3]])
    got = vc.pointer_top1(tops, bad_end=np.array([False, True, False, False]))
    assert got[SPAN_START_TOP1_METRIC]["n"] == 3
    assert got[SPAN_END_TOP1_METRIC]["n"] == 2


def test_the_null_is_plus_line_null_over_the_same_pointing_rows() -> None:
    mod = _scorer()
    vc = _vc(mod)
    null = mod.pointing_added_line_null(vc)
    assert null["n_total"] == 3 and null["n"] == 2  # rows 0 and 1 have a '+' gold
    assert null["value"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert null["metric"] == SPAN_NULL_METRIC


def test_the_scorers_numbers_feed_span_beats_null() -> None:
    mod = _scorer()
    vc = _vc(mod)
    p = vc.pointer_top1(np.array([[1, 1], [0, 0], [1, 2], [-1, -1]]))  # all pointing rows hit
    null = mod.pointing_added_line_null(vc)

    def ran(d: dict, v: float) -> Ran:
        return Ran(passed=True, value=v, n=d["n"], n_total=d["n_total"])

    state = span_beats_null_state(
        ran(p[SPAN_START_TOP1_METRIC], 1.0), ran(p[SPAN_END_TOP1_METRIC], 1.0),
        ran(null, null["value"]),
    )
    assert isinstance(state, Ran) and state.passed and state.n_total == 3


# --- the recorder ------------------------------------------------------------------------------


def _block(cache: str = "bf16_cache") -> dict:
    return {"head": {"source": "/x", "file_sha256": "0" * 64}, "bf16_cache": {},
            "span_pointers": {
                "cache": cache,
                SPAN_START_TOP1_METRIC: {"n": 30, "n_total": 50},
                SPAN_END_TOP1_METRIC: {"n": 28, "n_total": 50}}}


def _result() -> dict:
    return {"nulls": {"pointing_added_line": {"value": 0.6, "n": 40, "n_total": 50}}}


def test_recorder_writes_the_three_metrics_as_ran_over_one_row_count() -> None:
    got = srr.span_null_metrics(_result(), _block(), "c")
    assert set(got) == set(THREE)
    assert all(isinstance(m, Ran) and m.n_total == 50 for m in got.values())
    assert got[SPAN_START_TOP1_METRIC].value == pytest.approx(0.6)
    assert got[SPAN_NULL_METRIC].n == 40
    assert isinstance(span_beats_null_state(*(got[k] for k in THREE)), Ran)


def test_recorder_refuses_results_from_the_scorer_before_the_amendment() -> None:
    block = _block()
    del block["span_pointers"]
    with pytest.raises(Refused):
        srr.span_null_metrics(_result(), block, "c")
    with pytest.raises(Refused):
        srr.span_null_metrics({"nulls": {}}, _block(), "c")


# --- the verifier ------------------------------------------------------------------------------

NEW_SHA = hashlib.sha256(SCORER.read_bytes()).hexdigest() if SCORER.is_file() else ""


def _row(*, sha: str, metrics: dict | None = None, gates: dict | None = None,
         status: str = "completed", tag: str = "span-refit-c") -> L.LedgerRow:
    return L.LedgerRow(
        row_id="r", written_at="now", prev_row_hash=None, protocol=_protocol(1), run_kind="eval",
        status=status, quick=True, quick_reason="span-refit test", code_commit="c", env=_env(),
        metrics=dict(metrics or {}), noul_rate=NotRun(reason="n/a"), controls={},
        gates=dict(gates or {}), wall_clock_s=1.0, cost_usd=0.0,
        recipe={"tag": tag, "script_sha256": {"span_refit_score.py": sha}},
    )


def _three(total: int = 50) -> dict:
    return {n: Ran(passed=True, value=0.5, n=25, n_total=total) for n in THREE}


def test_a_new_sha_row_without_the_span_field_is_refused_on_both_doors(tmp_path: Path) -> None:
    sha = next(iter(L.SPAN_REFIT_SPAN_NULL_SCORER_SHAS))
    row = _row(sha=sha, metrics={"val_top1.span": Ran(passed=True, value=1, n=1, n_total=2)})
    with pytest.raises(ValueError, match="span-null scorer"):
        row.check_span_refit_span_null()
    with pytest.raises(ValueError, match="span-null scorer"):
        L.Ledger(tmp_path / "l.jsonl").append(row)
    ok = _row(sha=sha, metrics=_three())
    ledger = L.Ledger(tmp_path / "ok.jsonl")
    ledger.append(ok)
    ledger.verify_chain()  # from_json runs the check on read
    partial = _three()
    del partial[SPAN_END_TOP1_METRIC]
    with pytest.raises(ValueError, match="lacks"):
        _row(sha=sha, metrics=partial).check_span_refit_span_null()
    with pytest.raises(ValueError, match="one pointing-row count"):
        bad = _three()
        bad[SPAN_NULL_METRIC] = Ran(passed=True, value=0.5, n=25, n_total=49)
        _row(sha=sha, metrics=bad).check_span_refit_span_null()


def test_rows_under_another_scorer_sha_or_tag_or_status_are_not_checked() -> None:
    sha = next(iter(L.SPAN_REFIT_SPAN_NULL_SCORER_SHAS))
    _row(sha="f" * 64).check_span_refit_span_null()  # predates the null: void, not refused
    _row(sha="f" * 64, tag="calib-fit").check_span_refit_span_null()
    with pytest.raises(ValueError, match="span-null scorer"):  # pins the scorer, not its tag
        _row(sha=sha, tag="calib-fit").check_span_refit_span_null()
    _row(sha=sha, status="failed").check_span_refit_span_null()
    gate_failed = {L.SPAN_REFIT_VALIDITY_GATE: Ran(passed=False, value=5000)}
    _row(sha=sha, gates=gate_failed, tag="span-refit-a").check_span_refit_span_null()
    with pytest.raises(ValueError, match="span-null scorer"):  # the exemption is arm (a)'s only
        _row(sha=sha, gates=gate_failed, tag="span-refit-b").check_span_refit_span_null()


# --- the pins ----------------------------------------------------------------------------------


def test_the_ledger_names_this_scorers_sha_and_the_amendment_does_too() -> None:
    if SCORER != REPO / "tools" / "span_refit_score.py":
        pytest.skip("scorer overridden")
    assert NEW_SHA in L.SPAN_REFIT_SPAN_NULL_SCORER_SHAS
    amendments = {a["id"]: a for a in json.loads(PREREG.read_text(encoding="utf-8"))["amendments"]}
    amendment = amendments[2]
    assert amendment["scorer"]["new_sha256"] == NEW_SHA
    assert amendment["scorer"]["old_sha256"] not in L.SPAN_REFIT_SPAN_NULL_SCORER_SHAS
    # Amendment 3 (v6 shards) keeps amendment 2's scorer: it names the same sha, it moves none.
    assert amendments[3]["scorer"]["sha256"] == NEW_SHA


def test_the_recorders_prereg_pin_is_the_amended_file() -> None:
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == srr.PREREG_SHA256
