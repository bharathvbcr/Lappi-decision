"""``tools/span_refit_rows.py``: each arm of the span-refit pre-registration as a ledger row.

The pre-registered scorer only drafted rows, so the arm numbers cited a JSON, not a row
(rule 5). The recorder runs the scorer once per arm inside a ``RunRecorder`` and writes its
numbers as metrics. These tests drive it with a fake scorer that writes a fixture results JSON,
against the real committed row 6af73bef, and pin: one quick row per arm, the exact recipe keys,
never the supplement key ``eval_row_id``, the validity gate on every row, refusal of a
mismatched pre-registration, and that a failed gate stops the chain after (a)'s own row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import span_refit_rows as srr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402

from qd_train.ledger import Ledger  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

EVAL_LEDGER = REPO / "ledger" / "h100x2-v5avg-2026-10-05.jsonl"
CELLS = srr.CELLS


def _counts(n: int) -> dict[str, dict[str, int]]:
    out = {"overall": {"n": n, "n_total": 7238}}
    for cell in CELLS:
        out[cell] = {"n": n // 4, "n_total": 2000}
    return out


def _results(arm_value: int = 5596, *, prereg: str = srr.PREREG_SHA256) -> dict[str, object]:
    head = {"source": "/nonexistent/head", "file_sha256": "0" * 64, "content_digest": "1" * 64,
            "cond_W_start": 12.5, "cond_W_end": 7.0}
    arms = {
        "a": {"head": head, "gate": {"value_fp32_live": arm_value, "target": 5596,
                                      "tolerance": 15, "passed": abs(arm_value - 5596) <= 15},
              "fp32_live": _counts(arm_value), "reading": "does not"},
        "b": {"head": head, "bf16_cache": _counts(6000), "fp32_live": _counts(6000),
              "reading": "above mean head"},
        "c": {"head": head, "bf16_cache": _counts(6100), "reading": "above mean head"},
        "d": {"head": head, "fp32_live": _counts(6626), "reading": "restores"},
        "e": {"head": head, "bf16_cache": _counts(6200), "reading": "restores"},
    }
    for block in arms.values():  # amendment 2: each arm's pointer top-1s over the pointing rows
        block["span_pointers"] = {
            "cache": "bf16_cache",
            "val_span_start_top1_over_chance": {"n": 3000, "n_total": 5357},
            "val_span_end_top1_over_chance": {"n": 2900, "n_total": 5357},
        }
    tests = {"b_vs_c_all_rows (prereg)": {"n": 7238, "first_only": 10, "second_only": 30,
                                          "p_exact_two_sided": 0.002}}
    for cell in CELLS:
        tests[f"e_vs_c_{cell} (reported, not read)"] = {
            "n": 1000, "first_only": 5, "second_only": 3, "p_exact_two_sided": 0.7}
    return {
        "prereg": {"sha256": prereg}, "population_pin": srr.POPULATION_PIN, "arms": arms,
        "nulls": {"uniform_line": {"by_cell": {c: 12.5 for c in CELLS}},
                  "random_plus_line": {"expected_correct": 40.0, "rows": 1733},
                  "always_abstain": {"n": 1881, "agrees": True},
                  "pointing_added_line": {"value": 0.61, "n": 4000, "n_total": 5357}},
        "mcnemar": tests,
    }


def _scorer_dir(tmp_path: Path, results: dict[str, object], *, exit_code: int = 0) -> Path:
    scripts = tmp_path / "scorer"
    scripts.mkdir()
    for name in srr.SCRIPTS:
        (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
    (scripts / "span_refit_score.py").write_text(
        "import json, sys\n"
        "a = sys.argv\n"
        "out = a[a.index('--out') + 1]\n"
        f"open(out, 'w').write({json.dumps(json.dumps(results))})\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    return scripts / "span_refit_score.py"


def _argv(tmp_path: Path, scorer: Path, *extra: str) -> list[str]:
    return ["--scorer", str(scorer), "--eval-ledger", str(EVAL_LEDGER),
            "--ledger", str(tmp_path / "out.jsonl"), "--results-dir", str(tmp_path / "res"),
            "--head-c", str(tmp_path / "hc"), "--head-e", str(tmp_path / "he"),
            "--seed1-val-cache", str(tmp_path / "s1"), *extra]


def test_one_quick_row_per_arm_with_the_exact_recipe_the_gate_and_the_numbers(tmp_path):
    scorer = _scorer_dir(tmp_path, _results())
    assert srr.main(_argv(tmp_path, scorer, "--labels", "span-extract-avg-train-r4")) == 0
    rows = Ledger(tmp_path / "out.jsonl").rows()
    assert [r.recipe["tag"] for r in rows] == [f"span-refit-{a}" for a in "abcde"]
    eval_row = next(r for r in Ledger(EVAL_LEDGER).rows())
    for row in rows:
        assert row.quick and row.run_kind == "eval" and row.status == "completed"
        assert "eval_row_id" not in row.recipe
        assert row.recipe["reported_eval_rows"] == [eval_row.row_id]
        assert row.recipe["prereg_sha256"] == srr.PREREG_SHA256
        assert row.recipe["population_pin"] == srr.POPULATION_PIN
        assert set(row.recipe["script_sha256"]) == set(srr.SCRIPTS)
        assert row.protocol.seed == eval_row.protocol.seed
        assert row.protocol.recipe_hash != eval_row.protocol.recipe_hash
        gate = row.gates[srr.GATE_NAME]
        assert isinstance(gate, Ran) and gate.passed and gate.value == 5596
    by_tag = {r.recipe["tag"]: r for r in rows}
    a_span = by_tag["span-refit-a"].metrics["val_top1.span"]
    assert isinstance(a_span, Ran) and (a_span.n, a_span.n_total) == (5596, 7238)
    d_span = by_tag["span-refit-d"].metrics["val_top1.span"]
    assert isinstance(d_span, Ran) and d_span.n == 6626
    c_row, e_row = by_tag["span-refit-c"], by_tag["span-refit-e"]
    assert "mcnemar.b_vs_c" in c_row.metrics and "cond_w_start" in c_row.metrics
    assert "mcnemar.b_vs_c" not in e_row.metrics
    assert all(f"mcnemar.e_vs_c.{c}" in e_row.metrics for c in CELLS)
    assert "null.always_abstain.overall" in c_row.metrics
    assert all(m.passed for m in c_row.metrics.values() if isinstance(m, Ran))


def test_a_prereg_hash_that_is_not_the_amended_one_is_refused_and_writes_no_row(tmp_path):
    scorer = _scorer_dir(tmp_path, _results(prereg="f" * 64))
    with pytest.raises(Refused, match="pre-registration"):
        srr.main(_argv(tmp_path, scorer, "--arms", "a"))
    # The recorder keeps the refusal as a failed row with no metrics; no arm number is written.
    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    assert row.status != "completed" and "val_top1.span" not in row.metrics


def test_a_failed_gate_writes_only_arm_a_and_stops_the_chain(tmp_path):
    scorer = _scorer_dir(tmp_path, _results(arm_value=5500), exit_code=3)
    assert srr.main(_argv(tmp_path, scorer)) == srr.GATE_EXIT
    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    gate = row.gates[srr.GATE_NAME]
    assert row.recipe["tag"] == "span-refit-a"
    assert isinstance(gate, Ran) and not gate.passed and gate.value == 5500


def test_an_arm_without_its_head_or_cache_is_refused_before_anything_runs(tmp_path):
    scorer = _scorer_dir(tmp_path, _results())
    argv = ["--scorer", str(scorer), "--eval-ledger", str(EVAL_LEDGER),
            "--ledger", str(tmp_path / "out.jsonl"), "--results-dir", str(tmp_path / "res"),
            "--arms", "e"]
    with pytest.raises(SystemExit, match="arm \\(e\\)"):
        srr.main(argv)
    assert Ledger(tmp_path / "out.jsonl").rows() == []


def test_a_result_is_never_overwritten(tmp_path):
    scorer = _scorer_dir(tmp_path, _results())
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "span-refit-score-a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Refused, match="already exists"):
        srr.main(_argv(tmp_path, scorer, "--arms", "a"))


def test_every_script_the_rows_pin_is_a_tracked_file_beside_the_default_scorer():
    import subprocess

    scorer = REPO / "tools" / "span_refit_score.py"
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "tools"], capture_output=True,
                             text=True, check=True).stdout.split()
    for name in srr.SCRIPTS:
        assert (scorer.parent / name).is_file() and f"tools/{name}" in tracked, name
