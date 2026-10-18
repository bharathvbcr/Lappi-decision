"""GAP-FT-ROWS-CARRY-NO-SPAN-NULL-2026-10-17: an FT eval row that carries span top-1 carries the
random-``+``-line null beside it, so ``span_beats_null`` reads ``ran`` and not ``not_run``.

The metrics come from ``real_ft_run.score_states`` -- the one function every FT scoring path
(``_record_score``, ``--score-checkpoint``, ``--score-plan``, ``mac_zero_shot``) writes them
through -- and the arithmetic from ``qd_train.span_null``, the module the gate reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import real_ft_run as rft  # noqa: E402
from test_ruling_gates import SPAN, _committed_record, _family, _says, _verdict  # noqa: E402

from qd_train.artifacts import SLOT_CHOICE, SLOT_SPAN  # noqa: E402
from qd_train.shards import SequenceSpec  # noqa: E402
from qd_train.span_null import (  # noqa: E402
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

PLUS, SPACE = ord("+"), ord(" ")
#: Four candidate lines, two of them added: the null is 1/2 on a row whose gold is one of them.
CANDIDATES = (SPACE, PLUS, PLUS, SPACE)


def _label(i: int, first: tuple[int, ...] | None = CANDIDATES) -> rft.Label:
    return rft.Label(
        row_id=f"r{i}", family_id="code.defect_class", slot_name="evidence",
        slot_kind=SLOT_SPAN, gold_letter="noul", letters=("noul",), line_first_bytes=first,
    )


def _verdict_row(i: int, *, top: tuple[int, int], gold: int | None) -> dict[str, object]:
    abstain = gold is None
    return {
        "kind": "span", "row_id": f"r{i}", "slot_name": "evidence", "noul_row": 4, "rows": 5,
        "expected_abstain": abstain, "gold_start": 4 if abstain else gold,
        "gold_end": 4 if abstain else gold, "top": list(top), "correct": False,
        "runtime_verdict": f"lines {top[0]}..{top[1]}",
    }


def _scored(*, top: tuple[int, int]) -> dict[str, object]:
    """Five pointing rows (gold line 1, an added line) and one abstaining row."""
    verdicts = [_verdict_row(i, top=top, gold=1) for i in range(5)]
    verdicts.append(_verdict_row(5, top=(4, 4), gold=None))
    n_span = len(verdicts)
    return {"by_kind": {"span": {"n": n_span, "correct": 0, "abstaining": 1}},
            "verdicts": verdicts}


def _states(top: tuple[int, int]):
    return rft.score_states(_scored(top=top), [_label(i) for i in range(6)])


def test_span_top1_and_the_null_are_written_over_the_same_pointing_rows() -> None:
    states = _states((1, 1))
    start, end, null = (states[n] for n in (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC,
                                            SPAN_NULL_METRIC))
    assert isinstance(start, Ran) and isinstance(end, Ran) and isinstance(null, Ran)
    assert (start.value, start.n, start.n_total) == (1.0, 5, 5)
    assert (end.value, end.n, end.n_total) == (1.0, 5, 5)
    # Both counts, from qd_train.span_null: 5 pointing rows, all with an added-line gold.
    assert (null.value, null.n, null.n_total) == (pytest.approx(0.5), 5, 5)


def test_a_span_row_with_no_candidate_bytes_is_not_run_not_a_null_over_other_lines() -> None:
    labels = [_label(i, first=None if i == 2 else CANDIDATES) for i in range(6)]
    states = rft.score_states(_scored(top=(1, 1)), labels)
    for name in (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC, SPAN_NULL_METRIC):
        assert isinstance(states[name], NotRun), name
        assert "carries no candidate bytes" in states[name].reason


def test_a_candidate_count_that_is_not_the_heads_is_not_run() -> None:
    labels = [_label(i, first=CANDIDATES[:3]) for i in range(6)]
    state = rft.score_states(_scored(top=(1, 1)), labels)[SPAN_NULL_METRIC]
    assert isinstance(state, NotRun) and "3 candidate lines against the head's 4" in state.reason


def test_a_decode_with_no_pointing_row_is_not_run() -> None:
    scored = {"by_kind": {"span": {"n": 1, "correct": 1, "abstaining": 1}},
              "verdicts": [_verdict_row(0, top=(4, 4), gold=None)]}
    state = rft.score_states(scored, [_label(0)])[SPAN_NULL_METRIC]
    assert isinstance(state, NotRun)


def test_a_letter_only_decode_still_names_the_three_metrics_as_not_run() -> None:
    labels = [rft.Label(row_id="c", family_id="f", slot_name="s", slot_kind=SLOT_CHOICE,
                        gold_letter="A", letters=("A", "B"))]
    states = rft.score_states({"by_kind": {}, "verdicts": []}, labels)
    for name in (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC, SPAN_NULL_METRIC):
        assert isinstance(states[name], NotRun)


def test_labels_read_each_lines_first_byte_off_the_rendered_text() -> None:
    spec = SequenceSpec(slot_name="evidence", text="ab\n+x\n-y\n zz", slot_kind=SLOT_SPAN,
                        span_char_starts=(3, 3), line_char_starts=(0, 3, 6, 9))
    assert rft._line_first_bytes(spec) == (ord("a"), PLUS, ord("-"), SPACE)
    bare = SequenceSpec(slot_name="evidence", text="ab", slot_kind=SLOT_SPAN,
                        span_char_starts=None)
    with pytest.raises(ValueError, match="no line_char_starts"):
        rft._line_first_bytes(bare)


def test_the_ft_eval_row_makes_span_beats_null_ran_and_promotion_reads_it(tmp_path) -> None:
    """End to end: the metrics ``score_states`` writes, on rows the ledger stores, read by the
    promotion verdict. A head above the null promotes; one below it is refused as FAILED, which
    is the gate having RUN -- not ``did not run``."""
    record = _committed_record(tmp_path, order_from="9999-12-31")
    above = _family(tmp_path, record, stamp=False, span_states=_states((1, 1)))
    verdict = _verdict(above, record)
    assert verdict.promoted, verdict.reasons
    assert not _says(verdict, f"gate '{SPAN}' did not run")

    below_dir = tmp_path / "below"
    below_dir.mkdir()
    verdict = _verdict(_family(below_dir, record, stamp=False,
                               span_states=_states((0, 0))), record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{SPAN}' ran and FAILED"), verdict.reasons
    assert not _says(verdict, f"gate '{SPAN}' did not run")


def test_without_the_null_the_gate_reads_not_run(tmp_path) -> None:
    """The failure this lane fixes, kept as the control: top-1 with no null blocks promotion."""
    record = _committed_record(tmp_path, order_from="9999-12-31")
    states = {k: v for k, v in _states((1, 1)).items() if k != SPAN_NULL_METRIC}
    verdict = _verdict(_family(tmp_path, record, stamp=False, span_states=states), record)
    assert not verdict.promoted
    assert _says(verdict, f"gate '{SPAN}' did not run")
