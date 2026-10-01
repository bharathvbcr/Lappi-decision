"""``tools/gate_report_row.py``: a ``qd-gate-report`` run as a ledger row of its own (rule 5).

The numbers ``qd-gate-report`` re-reports -- every gate per family, pooled and per-language ECE,
the slot diagnostics -- must cite a ledger row. This binding runs the binary inside a
``RunRecorder`` and writes each number as a metric, on a quick row that names the eval row it
re-read (``reported_eval_rows``, never the supplement key ``eval_row_id``), the verdict files'
sha256, the decisions record's and the binary's; the commit is the row's ``code_commit``. It
computes nothing itself, so every metric must equal the report the binary wrote.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import gate_report_row as grr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402
from test_gate_report_parity import (  # noqa: E402
    _record_eval_row,
    _scoring_run,
    _write_verdicts,
)

from qd_train.ledger import DEFAULT_DECISIONS_PATH, Ledger, load_promotion_decisions  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _fixture(tmp_path: Path, *, tamper: str | None = None) -> tuple[Path, Path, str]:
    scored, second, perms = _scoring_run(np.random.default_rng(11))
    ledger = Ledger(tmp_path / "eval.jsonl")
    eval_row_id, _ = _record_eval_row(ledger, scored, second, perms, tamper=tamper)
    return ledger.path, _write_verdicts(tmp_path / "v.jsonl", scored, eval_row_id), eval_row_id


def _argv(binary: Path, tmp_path: Path, verdicts: Path, eval_ledger: Path, out_ledger: Path,
          *extra: str) -> list[str]:
    return ["--bin", str(binary), "--verdicts", str(verdicts), "--eval-ledger", str(eval_ledger),
            "--ledger", str(out_ledger), "--out-json", str(tmp_path / "report.json"), *extra]


def test_the_row_cites_the_eval_row_its_files_and_the_commit_and_carries_every_number(
    tmp_path, gate_report_bin
):
    eval_ledger, verdicts, eval_row_id = _fixture(tmp_path)
    out = tmp_path / "out.jsonl"
    assert grr.main(_argv(gate_report_bin, tmp_path, verdicts, eval_ledger, out)) == 0
    (row,) = Ledger(out).rows()
    (eval_row,) = Ledger(eval_ledger).rows()
    assert row.status == "completed" and row.quick and row.run_kind == "eval"
    assert row.recipe is not None and "eval_row_id" not in row.recipe
    assert row.recipe["reported_eval_rows"] == [eval_row_id]
    assert row.recipe["verdicts_sha256"] == [hashlib.sha256(verdicts.read_bytes()).hexdigest()]
    assert row.recipe["decisions_sha256"] == hashlib.sha256(
        DEFAULT_DECISIONS_PATH.read_bytes()).hexdigest()
    assert row.recipe["bin_sha256"] == hashlib.sha256(gate_report_bin.read_bytes()).hexdigest()
    assert row.code_commit and row.protocol.seed == eval_row.protocol.seed
    assert row.protocol.recipe_hash != eval_row.protocol.recipe_hash
    for field in ("data_snapshot_hash", "tokenizer_hash", "backbone_commit"):
        assert getattr(row.protocol, field) == getattr(eval_row.protocol, field)

    report = json.loads((tmp_path / "report.json").read_text())
    (reported,) = report["eval_rows"]
    checked = row.metrics["gate_report.cross_check"]
    assert isinstance(checked, Ran) and checked.value == reported["cross_check"]["checked"]
    g1 = reported["g1"]
    for family, state in g1["gates"]["permutation_consistency"]["per_family"].items():
        got = row.metrics[f"g1.permutation_consistency.family.{family}"]
        if state["state"] == "ran":
            assert isinstance(got, Ran) and (got.value, got.n, got.n_total) == (
                state["value"], state["n"], state["n_total"])
        else:
            assert isinstance(got, NotRun) and got.reason == state["reason"]
    heads = g1["controls"]["degenerate_head"]["per_family"]["code.defect_class"]
    got = row.metrics["g1.degenerate_head.family.code.defect_class.choice.k4"]
    assert isinstance(got, Ran) and got.value == heads["choice.k4"]["value"]
    got = row.metrics["g1.ood_abstain.family.intent.in_scope.in_distribution"]
    assert isinstance(got, Ran)
    assert got.n == g1["gates"]["ood_abstain"]["per_family"]["intent.in_scope"][
        "in_distribution"]["n"]
    assert row.metrics["g2.ece.pooled"].value == reported["g2"]["pooled_all_letter_rows"]["value"]
    assert row.metrics["g2.ece.lang.none"].value == reported["g2"]["by_language"]["none"]["value"]
    k4 = reported["g3"]["slots"]["choice.k4"]["all"]
    assert row.metrics["g3.choice.k4.all.mean_entropy"].value == k4["mean_entropy"]
    sigma = k4["classes"][3]["predicted"]["deviation_sigma"]
    assert row.metrics["g3.choice.k4.all.option_3.predicted_sigma"].value == sigma["value"]
    noul = row.metrics["g3.choice.k4.all.noul.predicted_sigma"]
    assert isinstance(noul, NotRun) and "undefined" in noul.reason
    population = row.metrics["promotion_population"]
    assert population.value == load_promotion_decisions().population.value
    # Every metric is report-only: none carries a verdict.
    assert all(isinstance(s, NotRun) or s.passed for s in row.metrics.values())


def test_the_report_row_never_joins_the_eval_rows_family(tmp_path, gate_report_bin):
    """Written into the eval row's own ledger, it still is not one of the eval row's family:
    its recipe hash is the report's, and it names the row by reported_eval_rows."""
    eval_ledger, verdicts, eval_row_id = _fixture(tmp_path)
    assert grr.main(_argv(gate_report_bin, tmp_path, verdicts, eval_ledger, eval_ledger)) == 0
    eval_row, report_row = Ledger(eval_ledger).rows()
    verdict = Ledger(eval_ledger).promotion_verdict(eval_row.protocol.hash_without_seed())
    assert verdict.rows == (eval_row_id,)
    assert report_row.row_id not in verdict.rows


def test_a_report_the_binary_refuses_writes_a_failed_row(tmp_path, gate_report_bin):
    eval_ledger, verdicts, _ = _fixture(tmp_path, tamper="permutation_count")
    out = tmp_path / "out.jsonl"
    with pytest.raises(Refused, match="not that row's verdicts"):
        grr.main(_argv(gate_report_bin, tmp_path, verdicts, eval_ledger, out))
    (row,) = Ledger(out).rows()
    assert row.status == "failed"
    assert "gates.permutation_consistency" in row.notes


def test_one_eval_row_per_report_row(tmp_path, gate_report_bin):
    eval_ledger, verdicts, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="exactly one --verdicts"):
        grr.main([*_argv(gate_report_bin, tmp_path, verdicts, eval_ledger,
                         tmp_path / "out.jsonl"), "--verdicts", str(verdicts)])
    assert not (tmp_path / "out.jsonl").exists()


def test_an_existing_report_is_not_overwritten(tmp_path, gate_report_bin):
    eval_ledger, verdicts, _ = _fixture(tmp_path)
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        grr.main(_argv(gate_report_bin, tmp_path, verdicts, eval_ledger, tmp_path / "out.jsonl"))
    assert not (tmp_path / "out.jsonl").exists()
