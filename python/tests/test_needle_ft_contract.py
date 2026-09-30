"""``needle_hunk_recall`` on the FT eval, under the contract the human approved 2026-09-30.

GAP-NEEDLE-HUNK-RECALL-HAS-NO-FT-EVAL-CONTRACT: the gate was implemented and unwired, and
the defect val set cannot score it (every row is one hunk). These pin the contract's three
numbers, that a needle case becomes a defect row whose gold is the needle hunk, and the hit
rule: the hunk of the predicted START line, a header line or an abstention being a miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_data.defect_class import CONTEXT_HEADER_LINES  # noqa: E402
from qd_train import needle  # noqa: E402
from qd_train.needle import build_suite, hunk_of_context_line, needle_defect_row  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _cases():
    return build_suite(target_tokens=1024, cases_per_depth=2, seed=3)


def test_the_contract_is_the_one_the_human_approved():
    """Rule 2: thresholds are read-only to an agent. A change here is a human decision."""
    assert needle.NEEDLE_MIN_RECALL == 0.95
    assert needle.NEEDLE_CASES_PER_DEPTH == 60
    assert needle.NEEDLE_TARGET_TOKENS == 8192
    assert needle.NEEDLE_HIT_RULE == "hunk-of-predicted-start-line"


def test_every_case_becomes_a_defect_row_whose_gold_is_the_needle_hunk():
    for case in _cases():
        row = needle_defect_row(case, config=DataConfig())
        assert row.family_id == "code.defect_class"
        gold = {g.slot_name: g for g in row.gold}
        assert gold["defect_class"].value == "stub"
        start, end = gold["defect_span"].value
        # 1-based over the rendered context, which is the header plus the haystack.
        assert hunk_of_context_line(case, start - 1) == case.needle_index
        assert hunk_of_context_line(case, end - 1) == case.needle_index
        assert row.metadata["language"] == case.language


def test_a_header_line_belongs_to_no_hunk_and_a_line_past_the_end_is_refused():
    case = _cases()[0]
    for line in range(CONTEXT_HEADER_LINES):
        assert hunk_of_context_line(case, line) is None
    assert hunk_of_context_line(case, CONTEXT_HEADER_LINES) == 0, "the haystack opens on @@"
    n = len(case.context.split("\n"))
    with pytest.raises(ValueError, match="past the context"):
        hunk_of_context_line(case, CONTEXT_HEADER_LINES + n)
    with pytest.raises(ValueError, match="negative"):
        hunk_of_context_line(case, -1)


def test_every_line_of_a_hunk_maps_to_that_hunk():
    case = _cases()[4]
    body = case.context.split("\n")
    seen = -1
    for j, text in enumerate(body[:-1]):
        if text.startswith("@@"):
            seen += 1
        assert hunk_of_context_line(case, CONTEXT_HEADER_LINES + j) == seen
    assert seen == case.n_hunks - 1


torch = pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
sys.path.insert(0, str(REPO / "tools"))
import real_ft_run as rft  # noqa: E402


def _suite(cases):
    return rft.NeedleSuite(
        cases=cases, batches=[None] * len(cases),  # type: ignore[list-item]
        labels_for={}, token_lengths=[8000 + i for i in range(len(cases))],
    )


def _line_in_hunk(case, hunk: int) -> int:
    body = case.context.split("\n")
    seen = -1
    for j, text in enumerate(body):
        if text.startswith("@@"):
            seen += 1
            if seen == hunk:
                return CONTEXT_HEADER_LINES + j + 1  # a line inside, not the header
    raise AssertionError(hunk)


def _score(monkeypatch, cases, tops):
    verdicts = [
        {"kind": "span", "row_id": c.case_id, "top": list(t), "noul_row": 999}
        for c, t in zip(cases, tops, strict=True)
    ]
    monkeypatch.setattr(rft, "_decode", lambda *a, **k: {"verdicts": verdicts})
    return rft.score_needle(None, _suite(cases), {})  # type: ignore[arg-type]


def test_pointing_inside_the_needle_hunk_is_a_hit_whatever_the_end_line(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=5, seed=1)
    tops = [(_line_in_hunk(c, c.needle_index), 0) for c in cases]
    gate, metrics = _score(monkeypatch, cases, tops)
    assert isinstance(gate, Ran) and gate.passed and gate.value == 1.0
    assert set(metrics) >= {"needle_hunk_recall.depth.0-20%", "needle_suite_tokens"}


def test_the_next_hunk_an_abstention_and_a_header_line_are_misses(monkeypatch):
    cases = build_suite(target_tokens=1024, cases_per_depth=5, seed=1)
    tops = []
    for i, c in enumerate(cases):
        if i % 3 == 0:
            other = c.needle_index + 1 if c.needle_index + 1 < c.n_hunks else c.needle_index - 1
            tops.append((_line_in_hunk(c, other), 0))
        elif i % 3 == 1:
            tops.append((999, 999))  # the abstention row
        else:
            tops.append((0, 0))  # the `file:` header
    gate, metrics = _score(monkeypatch, cases, tops)
    assert isinstance(gate, Ran) and gate.value == 0.0 and not gate.passed
    assert "abstained" in metrics["needle_suite_tokens"].detail


def test_without_the_flag_the_gate_is_not_run_and_the_recipe_is_untouched():
    suite = rft.NeedleSuite([], [], {}, [], not_run="--needle was not given")
    gate, metrics = rft.score_needle(None, suite, {})  # type: ignore[arg-type]
    assert isinstance(gate, NotRun) and metrics == {}
    import inspect

    assert "if needle_suite is not None and needle_suite.not_run is None:" in inspect.getsource(
        rft._record_score
    )


def test_needle_argv_needs_score_val_and_the_real_backbone(tmp_path):
    with pytest.raises(SystemExit, match="--needle scores the model"):
        rft.main(["--out", str(tmp_path), "--rev", "0" * 40, "--needle"])


def test_a_span_only_batch_skips_the_vocabulary_head():
    import inspect

    src = inspect.getsource(rft._decode)
    assert "if any(label.slot_kind != SLOT_SPAN for label in labels_for[b])" in src
