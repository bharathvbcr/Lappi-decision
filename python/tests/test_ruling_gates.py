"""The two gates the human ruled in on 2026-10-17, Q2
(AUDIT/harden-2026-10-17/human-ruling-2026-10-17.md), and the property that no subset of
not_run gates promotes.

- ``decision_record_before_results``: a promotion verdict refuses when the decision record that
  defines a gate was committed after the earliest result row it judges, or when git or the record
  cannot be read (not_run, which blocks).
- ``span_beats_null``: span start and end top-1 must beat the random-``+``-line null on the same
  rows (AUDIT/span-in-diff.md: 63.61% against 89.91%, while the as-built gate read 8.4% and
  passed on all 8 seeds).
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promotion_fixtures import record_decode_coverage, write_record

from qd_train import ledger as ledger_module
from qd_train.ledger import (
    DECISION_RECORD_BLOB,
    DECISION_RECORD_COMMIT,
    REQUIRED_CONTROLS,
    REQUIRED_GATES,
    Environment,
    Ledger,
    PromotionVerdict,
    Protocol,
    RunRecorder,
    decision_record_evidence,
)
from qd_train.span_null import (
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
    plus_line_null,
    span_beats_null_state,
)
from qd_train.tristate import NotRun, Ran, TriState

REPO = Path(__file__).resolve().parents[2]
LONG_AGO = "2020-01-01T00:00:00Z"
FAR_FUTURE = "2099-01-01T00:00:00Z"
SPAN = "span_beats_null"
ORDER = "decision_record_before_results"


def _protocol(seed: int) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64, tokenizer_hash="bytes-utf8-256",
        backbone_commit="test-backbone", recipe_hash="r" * 64, seed=seed,
    )


def _git(repo: Path, *args: str, date: str = LONG_AGO) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, check=True, capture_output=True,
        env={**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date},
    )


def _decided(binding_from: str) -> dict[str, object]:
    return {"status": "decided", "value": "yes", "decided_by": "a human (test)",
            "decided_on": "2026-10-17", "decision_ref": "test fixture",
            "binding_from": binding_from}


def _committed_record(tmp_path: Path, *, span_from: str = "2000-01-01",
                      order_from: str = "2000-01-01", date: str = LONG_AGO) -> Path:
    """A decisions record committed (at ``date``) in a git repo of its own, the two ruling
    gates decided and bound from the given days, every other question open (as built)."""
    repo = tmp_path / "record-repo"
    repo.mkdir(exist_ok=True)
    _git(repo, "init", "-q")
    path = write_record(repo, base="as_built",
                        **{SPAN: _decided(span_from), ORDER: _decided(order_from)})
    _git(repo, "add", path.name, date=date)
    _git(repo, "commit", "-q", "-m", "record", date=date)
    return path


def _family(tmp_path: Path, record: Path, *, span: tuple[float, float, float, int] | None = (
        0.95, 0.95, 0.50, 100), stamp: bool = True,
        not_run: frozenset[str] = frozenset(),
        span_states: Mapping[str, TriState] | None = None) -> Ledger:
    """Three completed ft seeds, every gate and control green, the span metrics as
    ``(start, end, null, pointing_rows)``, each row stamped with ``record``. ``not_run`` names
    gates (including the two rulings) to leave not_run, for the property test."""
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=Environment.detect(),
            wall_clock_s=None, cost=None,
            decision_record_path=record if stamp else None,
        ) as rec:
            for g in REQUIRED_GATES:
                rec.gate(g, NotRun(reason="left out by the test") if g in not_run
                         else Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
            rec.metric("train.termination", Ran(passed=True, value="steps_exhausted"))
            record_decode_coverage(rec)
            if span_states is not None:
                for name, state in span_states.items():
                    rec.metric(name, state)
            elif span is not None:
                start, end, null, n = span
                gold_added = round(n * 0.9)
                rec.metric(SPAN_START_TOP1_METRIC,
                           Ran(passed=start > null, value=start, n=round(start * n), n_total=n))
                rec.metric(SPAN_END_TOP1_METRIC,
                           Ran(passed=end > null, value=end, n=round(end * n), n_total=n))
                rec.metric(SPAN_NULL_METRIC,
                           NotRun(reason="left out by the test") if SPAN in not_run
                           else Ran(passed=True, value=null, n=gold_added, n_total=n))
            if ORDER in not_run:
                rec.metric(DECISION_RECORD_COMMIT, NotRun(reason="left out by the test"))
    return led


def _verdict(led: Ledger, record: Path) -> PromotionVerdict:
    return led.promotion_verdict(_protocol(1).hash_without_seed(), decisions_path=record)


def _says(verdict: PromotionVerdict, text: str) -> bool:
    return any(text in r for r in verdict.reasons)


# ---------------------------------------------------------------------------
# decision_record_before_results
# ---------------------------------------------------------------------------


def test_a_record_committed_before_the_results_promotes(tmp_path) -> None:
    """The positive control: without it every refusal below could be the fixture's fault."""
    record = _committed_record(tmp_path)
    verdict = _verdict(_family(tmp_path, record), record)
    assert verdict.promoted, str(verdict)
    assert any(ORDER in r and "PASS" in r for r in verdict.readings), verdict.readings


def test_a_record_committed_after_the_results_refuses(tmp_path) -> None:
    record = _committed_record(tmp_path, date=FAR_FUTURE)
    verdict = _verdict(_family(tmp_path, record), record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{ORDER}' ran and FAILED"), verdict.reasons
    assert _says(verdict, "committed")


def test_a_record_with_uncommitted_edits_is_not_run_and_blocks(tmp_path) -> None:
    record = _committed_record(tmp_path)
    record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
    evidence = decision_record_evidence(record)
    assert all(isinstance(s, NotRun) for s in evidence.values())
    assert "uncommitted" in evidence[DECISION_RECORD_BLOB].reason  # type: ignore[union-attr]
    verdict = _verdict(_family(tmp_path, record), record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{ORDER}' did not run")


def test_a_record_outside_git_is_not_run_and_blocks(tmp_path) -> None:
    record = write_record(tmp_path, base="as_built",
                          **{SPAN: _decided("2000-01-01"), ORDER: _decided("2000-01-01")})
    verdict = _verdict(_family(tmp_path, record), record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{ORDER}' did not run")


def test_rows_that_name_no_record_are_not_run_and_block(tmp_path) -> None:
    """A row stamped under the repo's own record, or with no stamp to compare, cannot show the
    record it was judged under is the one being read."""
    record = _committed_record(tmp_path)
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO,
                         env=Environment.detect(), wall_clock_s=None, cost=None,
                         decision_record_path=record) as rec:
            for name in (DECISION_RECORD_BLOB, DECISION_RECORD_COMMIT):
                rec.metrics.pop(name)  # a row written without the stamp
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
            rec.metric("train.termination", Ran(passed=True, value="steps_exhausted"))
            record_decode_coverage(rec)
            rec.metric(SPAN_START_TOP1_METRIC, Ran(passed=True, value=0.9, n=90, n_total=100))
            rec.metric(SPAN_END_TOP1_METRIC, Ran(passed=True, value=0.9, n=90, n_total=100))
            rec.metric(SPAN_NULL_METRIC, Ran(passed=True, value=0.5, n=90, n_total=100))
    verdict = _verdict(led, record)
    assert not verdict.promoted
    assert _says(verdict, "do not record the decision record they were launched under")


def test_a_record_changed_after_the_results_refuses(tmp_path) -> None:
    record = _committed_record(tmp_path)
    led = _family(tmp_path, record)
    repo = record.parent
    record.write_text(record.read_text(encoding="utf-8").replace("test fixture", "edited"),
                      encoding="utf-8")
    _git(repo, "add", record.name)
    _git(repo, "commit", "-q", "-m", "edit the record after the results")
    verdict = _verdict(led, record)
    assert not verdict.promoted
    assert _says(verdict, "changed after the results")


def test_rows_that_misstate_the_blob_refuse(tmp_path) -> None:
    record = _committed_record(tmp_path)
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO,
                         env=Environment.detect(), wall_clock_s=None, cost=None,
                         decision_record_path=record) as rec:
            rec.metric(DECISION_RECORD_BLOB, Ran(passed=True, value="0" * 40))
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
            rec.metric("train.termination", Ran(passed=True, value="steps_exhausted"))
            record_decode_coverage(rec)
    verdict = _verdict(led, record)
    assert not verdict.promoted
    assert _says(verdict, "misstate the record")


def test_a_family_that_predates_the_ruling_reads_not_run_without_being_refused(
    tmp_path,
) -> None:
    record = _committed_record(tmp_path, span_from="9999-12-31", order_from="9999-12-31")
    verdict = _verdict(_family(tmp_path, record, span=None, stamp=False), record)
    assert verdict.promoted, str(verdict)
    for gate in (SPAN, ORDER):
        assert any(gate in r and "not_run" in r and "predates the 2026-10-17 ruling" in r
                   for r in verdict.readings), (gate, verdict.readings)


def test_the_evidence_is_stamped_on_ft_and_eval_rows_only(tmp_path) -> None:
    record = _committed_record(tmp_path)
    led = Ledger(tmp_path / "runs.jsonl")
    for kind in ("ft", "eval", "smoke"):
        with RunRecorder(led, protocol=_protocol(1), run_kind=kind, repo=REPO,
                         env=Environment.detect(), wall_clock_s=None, cost=None,
                         decision_record_path=record):
            pass
    stamped = {r.run_kind: DECISION_RECORD_COMMIT in r.metrics for r in led.rows()}
    assert stamped == {"ft": True, "eval": True, "smoke": False}


# ---------------------------------------------------------------------------
# span_beats_null
# ---------------------------------------------------------------------------


def test_plus_line_null_is_gold_lines_over_candidate_plus_lines_averaged_over_rows() -> None:
    plus, other = ord("+"), ord(" ")
    rows = [
        ([plus, other, plus], 0),        # gold is a + line; 2 candidate + lines -> 1/2
        ([plus, plus, plus, other], 2),  # 3 candidate + lines -> 1/3
        ([other, plus], None),           # abstaining row: not a pointing row
        ([other, other, plus], 0),       # gold is not a + line -> 0
    ]
    null, pointing, gold_added = plus_line_null(rows)
    assert (pointing, gold_added) == (3, 2)
    assert null == pytest.approx((1 / 2 + 1 / 3 + 0) / 3)


def test_plus_line_null_refuses_a_gold_outside_the_candidates() -> None:
    with pytest.raises(ValueError, match="outside"):
        plus_line_null([([ord("+")], 3)])


def test_the_tool_and_the_gate_use_one_null() -> None:
    """tools/rung0_real_run.py writes the metric the gate reads; its function delegates here."""
    text = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert "from qd_train.span_null import plus_line_null" in text
    assert "total += 1.0 / added" not in text


def test_a_span_head_below_the_null_refuses_though_it_beats_uniform_chance(tmp_path) -> None:
    """AUDIT/span-in-diff.md: 63.61% start top-1 against the 89.91% null (and 8.4% uniform,
    which the as-built gate passed on). Everything else green: only this gate refuses."""
    record = _committed_record(tmp_path, order_from="9999-12-31")
    verdict = _verdict(_family(tmp_path, record, span=(0.6361, 0.5968, 0.8991, 10635),
                               stamp=False), record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{SPAN}' ran and FAILED"), verdict.reasons
    assert len(verdict.reasons) == 3  # once per seed, and nothing else


def test_a_span_head_above_the_null_promotes(tmp_path) -> None:
    record = _committed_record(tmp_path, order_from="9999-12-31")
    verdict = _verdict(_family(tmp_path, record, span=(0.95, 0.94, 0.8991, 10635),
                               stamp=False), record)
    assert verdict.promoted, str(verdict)


@pytest.mark.parametrize("drop", ["null", "start", "end"])
def test_a_missing_span_metric_is_not_run_and_blocks(tmp_path, drop) -> None:
    gone = {"null": SPAN_NULL_METRIC, "start": SPAN_START_TOP1_METRIC,
            "end": SPAN_END_TOP1_METRIC}[drop]
    record = _committed_record(tmp_path, order_from="9999-12-31")
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO,
                         env=Environment.detect(), wall_clock_s=None, cost=None) as rec:
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
            rec.metric("train.termination", Ran(passed=True, value="steps_exhausted"))
            record_decode_coverage(rec)
            for name, value in ((SPAN_START_TOP1_METRIC, 0.95), (SPAN_END_TOP1_METRIC, 0.95),
                                (SPAN_NULL_METRIC, 0.5)):
                if name != gone:
                    rec.metric(name, Ran(passed=True, value=value, n=95, n_total=100))
    verdict = _verdict(led, record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{SPAN}' did not run")
    assert _says(verdict, gone)


def test_a_top_1_and_a_null_over_different_rows_is_not_run() -> None:
    state = span_beats_null_state(
        Ran(passed=True, value=0.9, n=90, n_total=100),
        Ran(passed=True, value=0.9, n=90, n_total=100),
        Ran(passed=True, value=0.5, n=45, n_total=90),
    )
    assert isinstance(state, NotRun) and "same rows" in state.reason


def test_the_end_pointer_must_beat_the_null_too() -> None:
    state = span_beats_null_state(
        Ran(passed=True, value=0.9, n=90, n_total=100),
        Ran(passed=True, value=0.4, n=40, n_total=100),
        Ran(passed=True, value=0.5, n=50, n_total=100),
    )
    assert isinstance(state, Ran) and not state.passed


# ---------------------------------------------------------------------------
# docs/hardening.md section 4: every subset of not_run gates refuses
# ---------------------------------------------------------------------------

ALL_SEVEN = (*REQUIRED_GATES, SPAN, ORDER)


def test_every_subset_of_gates_set_to_not_run_refuses_promotion(tmp_path, monkeypatch) -> None:
    """2^7 subsets over the five required gates and the two ruling gates, a bound family each
    time. The empty subset is the positive control: it promotes, so a refusal is the subset's
    doing. Only the stamp is computed once (git is not the subject here)."""
    record = _committed_record(tmp_path)
    stamp = decision_record_evidence(record)
    monkeypatch.setattr(ledger_module, "decision_record_evidence", lambda path=None: stamp)

    outcomes: dict[frozenset[str], bool] = {}
    for size in range(len(ALL_SEVEN) + 1):
        for subset in itertools.combinations(ALL_SEVEN, size):
            work = tmp_path / ("s-" + "-".join(subset) if subset else "s-none")
            work.mkdir()
            led = _family(work, record, not_run=frozenset(subset))
            outcomes[frozenset(subset)] = _verdict(led, record).promoted
    assert outcomes.pop(frozenset()) is True, "the all-ran family must promote"
    assert len(outcomes) == 2 ** len(ALL_SEVEN) - 1
    promoted = [sorted(s) for s, ok in outcomes.items() if ok]
    assert promoted == [], f"promoted with these gates not_run: {promoted}"


def test_every_subset_of_the_required_gates_refuses_on_a_family_that_predates_the_ruling(
    tmp_path,
) -> None:
    """The legacy reading: the ruling gates are reported, not required; the five still are."""
    record = _committed_record(tmp_path, span_from="9999-12-31", order_from="9999-12-31")
    for size in range(len(REQUIRED_GATES) + 1):
        for subset in itertools.combinations(REQUIRED_GATES, size):
            work = tmp_path / ("l-" + "-".join(subset) if subset else "l-none")
            work.mkdir()
            led = _family(work, record, span=None, stamp=False, not_run=frozenset(subset))
            assert _verdict(led, record).promoted is (size == 0), subset
