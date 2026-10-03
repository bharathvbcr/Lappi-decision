"""``ood_abstain`` under the contract the human approved on 2026-09-30.

GAP-OOD-ABSTAIN-HAS-NO-CONTRACT: every FT eval row carried the gate absent. These pin the
contract's numbers, the suite (three generated categories, the prose drawn from a given
pool), the runtime's choice rule minus its unfitted margin half, and the pairing with an
in-distribution bound that stops an always-abstaining model from passing.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_data.config import DataConfig  # noqa: E402
from qd_train import ood  # noqa: E402
from qd_train.ood import build_ood_suite, ood_defect_row, ood_gate  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

PROSE = [f"Which of these statements about topic {i} is correct?" for i in range(150)]


def test_the_contract_is_the_one_the_human_approved():
    """Rule 2: thresholds are read-only to an agent. A change here is a human decision."""
    assert ood.OOD_MIN_ABSTAIN == 0.90
    assert ood.OOD_MAX_IN_DISTRIBUTION_ABSTAIN == 0.05
    assert ood.OOD_CASES_PER_CATEGORY == 60
    assert ood.OOD_CATEGORIES == ("prose", "unseen-language", "scrambled")


def test_the_suite_is_sixty_of_each_category_and_deterministic():
    suite = build_ood_suite(PROSE, seed=7)
    assert collections.Counter(c.category for c in suite) == {
        "prose": 60, "unseen-language": 60, "scrambled": 60,
    }
    assert [c.context for c in suite] == [c.context for c in build_ood_suite(PROSE, seed=7)]
    assert [c.context for c in suite] != [c.context for c in build_ood_suite(PROSE, seed=8)]
    assert len({c.case_id for c in suite}) == len(suite)
    assert {c.context for c in suite if c.category == "prose"} <= set(PROSE)


def test_unseen_languages_are_not_the_mutation_pools():
    """The pool is python, go, rust and typescript (ece.lang on row 314837f2)."""
    langs = {c.language for c in build_ood_suite(PROSE) if c.category == "unseen-language"}
    assert langs and not langs & {"python", "go", "rust", "typescript"}


def test_a_prose_pool_too_small_is_refused_rather_than_repeated():
    with pytest.raises(ValueError, match="distinct texts"):
        build_ood_suite(PROSE[:10] * 10)


def test_every_case_is_asked_as_the_trained_question():
    for case in build_ood_suite(PROSE, per_category=3):
        row = ood_defect_row(case, config=DataConfig())
        assert row.family_id == "code.defect_class"
        assert row.request.context.startswith(f"file: {case.path}\n\n".encode())
        assert row.metadata["language"] == case.language


def test_the_gate_reads_each_rate_at_its_conservative_bound():
    """57/60 is 95% OOD abstention, and its Wilson lower bound (~0.86) is under 0.90: not
    a pass. The point estimate would have passed it."""
    fine_in = dict(in_abstained=0, in_total=2332)
    assert ood_gate(ood_abstained=180, ood_total=180, **fine_in).passed
    near = ood_gate(ood_abstained=57, ood_total=60, **fine_in)
    assert isinstance(near, Ran) and near.value == pytest.approx(0.95) and not near.passed


def test_an_always_abstaining_model_fails_on_the_in_distribution_bound():
    always = ood_gate(ood_abstained=180, ood_total=180, in_abstained=2332, in_total=2332)
    assert isinstance(always, Ran) and not always.passed
    assert "in-distribution: 2332 of 2332" in always.detail


def test_an_empty_side_is_not_run_rather_than_a_rate():
    assert isinstance(ood_gate(ood_abstained=0, ood_total=0, in_abstained=0, in_total=9), NotRun)
    assert isinstance(ood_gate(ood_abstained=3, ood_total=3, in_abstained=0, in_total=0), NotRun)


torch = pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
sys.path.insert(0, str(REPO / "tools"))
import real_ft_run as rft  # noqa: E402

PERM = (1, 2, 3, 0)  # position j shows the option first shown at PERM[j]


def _v(row: str, top: int, *, expected: bool = False) -> dict[str, object]:
    # row_logits and correct, as every real letter verdict carries them: the selective-risk
    # readout (real_ft_run.selective_risk_metrics) reads both.
    return {"kind": "choice", "row_id": row, "slot_name": "defect_class", "top": top,
            "noul_row": 4, "expected_abstain": expected, "correct": False,
            "row_logits": [4.0 if i == top else 0.0 for i in range(5)]}


def test_the_runtime_rule_abstains_on_noul_in_either_pass_or_on_disagreement():
    first = {"verdicts": [_v("agree", 1), _v("noul1", 4), _v("noul2", 1), _v("moved", 1),
                          _v("gold-noul", 1, expected=True)]}
    second = {"verdicts": [_v("agree", PERM.index(1)), _v("noul1", 0), _v("noul2", 4),
                           _v("moved", PERM.index(2))]}
    perms = {(r, "defect_class"): PERM for r in ("agree", "noul1", "noul2", "moved")}
    got = rft.choice_rule_abstentions(first, second, perms)
    assert got == {"agree": False, "noul1": True, "noul2": True, "moved": True}, (
        "a row whose gold is noul is not in-distribution evidence and is left out"
    )


def test_a_row_without_a_derangement_abstains_only_on_noul():
    first = {"verdicts": [_v("single", 0), _v("single-noul", 4)]}
    got = rft.choice_rule_abstentions(first, {"verdicts": []}, {})
    assert got == {"single": False, "single-noul": True}


def test_a_row_missing_its_second_pass_is_refused():
    with pytest.raises(SystemExit, match="not the second"):
        rft.choice_rule_abstentions(
            {"verdicts": [_v("r", 1)]}, {"verdicts": []}, {("r", "defect_class"): PERM}
        )


def test_score_ood_records_the_margin_half_as_not_run_and_scores_the_rest(monkeypatch):
    cases = build_ood_suite(PROSE, per_category=2, seed=0)
    passes = iter([
        {"verdicts": [_v(c.case_id, 4) for c in cases]},  # OOD first pass: all noul
        {"verdicts": [_v(c.case_id, 0) for c in cases]},  # OOD second pass
    ])
    monkeypatch.setattr(rft, "_decode", lambda *a, **k: next(passes))
    val = rft.ValSet(reader=None, labels=[], plan=[], labels_for={}, letter_id={})  # type: ignore[arg-type]
    perms = {(c.case_id, "defect_class"): PERM for c in cases}
    suite = rft.OodSuite(cases, val, rft.SecondPass([], {}, perms))
    in_first = {"verdicts": [_v(f"v{i}", 1) for i in range(200)]}
    in_second = {"verdicts": [_v(f"v{i}", PERM.index(1)) for i in range(200)]}
    in_perms = {(f"v{i}", "defect_class"): PERM for i in range(200)}
    gate, metrics, lines = rft.score_ood(
        None, suite, scored=in_first, val_second=in_second,  # type: ignore[arg-type]
        val_second_pass=rft.SecondPass([], {}, in_perms),
    )
    assert isinstance(metrics["ood_abstain.margin"], NotRun)
    assert "calibration" in metrics["ood_abstain.margin"].reason
    for category in ood.OOD_CATEGORIES:
        assert metrics[f"ood_abstain.{category}"].value == 1.0
    assert metrics["ood_abstain.in_distribution"].value == 0.0
    assert isinstance(gate, Ran) and gate.value == 1.0
    assert not gate.passed, "6 of 6 has a Wilson lower bound (~0.61) under the 0.90 floor"
    assert [v["case_id"] for v in lines] == [c.case_id for c in cases]


def test_score_ood_keeps_both_passes_per_case(monkeypatch):
    """2026-09-30: the suite failed at 22 of 180 and only the counts survived, so whether a
    calibrated margin would have abstained on the rest could not be asked of the run. Both
    passes' distributions are now kept per case."""
    cases = build_ood_suite(PROSE, per_category=1, seed=0)
    logits = [[0.1, 2.0, 0.3, 0.2, -1.0]]

    def with_dist(row, top):
        return {**_v(row, top), "row_logits": logits[0], "noul_probability": 0.05, "rows": 5}

    passes = iter([
        {"verdicts": [with_dist(c.case_id, 1) for c in cases]},
        {"verdicts": [with_dist(c.case_id, PERM.index(1) if i else 1)
                      for i, c in enumerate(cases)]},
    ])
    monkeypatch.setattr(rft, "_decode", lambda *a, **k: next(passes))
    val = rft.ValSet(reader=None, labels=[], plan=[], labels_for={}, letter_id={})  # type: ignore[arg-type]
    perms = {(c.case_id, "defect_class"): PERM for c in cases}
    suite = rft.OodSuite(cases, val, rft.SecondPass([], {}, perms))
    in_perms = {("v0", "defect_class"): PERM}
    _, _, lines = rft.score_ood(
        None, suite, scored={"verdicts": [_v("v0", 1)]},  # type: ignore[arg-type]
        val_second={"verdicts": [_v("v0", PERM.index(1))]},
        val_second_pass=rft.SecondPass([], {}, in_perms),
    )
    assert [v["category"] for v in lines] == [c.category for c in cases]
    first = lines[0]
    assert first["abstained"], "the permuted pass named another option"
    assert (first["top1"], first["top2"], first["perm"]) == (1, 1, list(PERM))
    assert first["row_logits_1"] == logits[0] and first["row_logits_2"] == logits[0]
    assert first["noul_probability_1"] == 0.05 and first["rows"] == 5
    assert not lines[1]["abstained"] and lines[1]["top2"] == PERM.index(1)
    gate = rft.ood_suite_gate(
        (Ran(passed=False, value=0.0, n=0, n_total=1), {}, lines), suite
    )
    rows = rft.suite_verdict_lines([gate], eval_row_id="e9", seed=1)
    assert rows[0]["eval_row_id"] == "e9" and rows[0]["gate"] == "ood_abstain"
    assert rows[0]["seed"] == 1 and rows[0]["case_id"] == cases[0].case_id


def test_suite_verdicts_are_written_once_and_refused_over_an_existing_file(tmp_path):
    import json

    lines = [{"eval_row_id": "e", "seed": 0, "gate": "ood_abstain", "suite": "ood",
              "case_id": "c1", "row_logits_1": [0.5, 1.5]}]
    out = tmp_path / "s.jsonl"
    rft.write_suite_verdicts_jsonl(out, lines)
    assert [json.loads(x) for x in out.read_text().splitlines()] == lines
    with pytest.raises(SystemExit, match="--suite-verdicts-out"):
        rft.write_suite_verdicts_jsonl(out, lines)
    with pytest.raises(ValueError, match="missing \\['case_id'\\]"):
        rft.write_suite_verdicts_jsonl(
            tmp_path / "t.jsonl", [{k: v for k, v in lines[0].items() if k != "case_id"}]
        )


def test_suite_verdicts_out_needs_a_suite(tmp_path):
    with pytest.raises(
        SystemExit, match="without --needle, --ood or --composed-slice there are none"
    ):
        rft.main(["--out", str(tmp_path), "--rev", "0" * 40,
                  "--suite-verdicts-out", str(tmp_path / "s.jsonl")])


def test_without_the_flag_the_gate_is_not_run_and_the_recipe_is_untouched():
    suite = rft.OodSuite([], None, None, not_run="--ood was not given")
    gate, metrics, lines = rft.score_ood(
        None, suite, scored={"verdicts": []}, val_second=None,  # type: ignore[arg-type]
        val_second_pass=rft.SecondPass([], {}, {}),
    )
    assert isinstance(gate, NotRun) and metrics == {}
    assert rft.ood_suite_gate((gate, metrics, lines), suite).recipe is None


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--ood"], "given together"),
        (["--ood-general-record", "/r.json"], "given together"),
        (["--ood", "--ood-general-record", "/r.json"], "--ood scores the model"),
    ],
)
def test_ood_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), "--rev", "0" * 40, *extra])
