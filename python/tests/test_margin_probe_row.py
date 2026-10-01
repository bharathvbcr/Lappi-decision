"""``tools/margin_probe_row.py``: the Rust probe's report as a ledger row of its own.

The binary comes from conftest's ``probe`` fixture, so these tests run wherever the parity
test does and skip with it when cargo is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import margin_probe_row as mpr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402

from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

N_OOD = 30


def _eval_row(ledger_path: Path, *, ood_value: float) -> str:
    protocol = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                        backbone_commit="standin", recipe_hash="r" * 64, seed=0)
    with RunRecorder(
        Ledger(ledger_path), entry_point=REPO / "tools" / "real_ft_run.py", protocol=protocol,
        run_kind="eval", repo=REPO, env=Environment.detect(device="cpu"), wall_clock_s=1.0,
        cost=None, quick=True, quick_reason="test fixture",
        recipe={"tool": "tools/real_ft_run.py", "tag": "epoch-score-val"},
    ) as rec:
        rec.gate("ood_abstain", Ran(passed=False, value=ood_value, n=N_OOD, n_total=N_OOD))
    assert rec.row is not None
    return rec.row.row_id


def _verdicts(tmp_path: Path, eval_row_id: str, *, abstain_every: int) -> tuple[Path, Path, int]:
    rng = np.random.default_rng(3)
    val = []
    for i in range(80):
        logits = rng.normal(0.0, 3.0, size=5).astype(np.float32)
        val.append({"eval_row_id": eval_row_id, "seed": 0, "row_id": f"v{i}", "kind": "choice",
                    "correct": bool(i % 7), "top": int(logits.argmax()), "gold_row": 0,
                    "row_logits": logits.tolist(), "noul_row": 4, "rows": 5})
    suite, abstained = [], 0
    for i in range(N_OOD):
        rule = i % abstain_every == 0
        abstained += rule
        suite.append({"eval_row_id": eval_row_id, "seed": 0, "gate": "ood_abstain",
                      "suite": "ood", "case_id": f"o{i}",
                      "category": ("prose", "scrambled", "unseen-language")[i % 3],
                      "abstained": rule,
                      "row_logits_1": rng.normal(0.0, 2.0, size=5).astype(np.float32).tolist()})
    val_path, suite_path = tmp_path / "val.jsonl", tmp_path / "suite.jsonl"
    val_path.write_text("".join(json.dumps(x) + "\n" for x in val))
    suite_path.write_text("".join(json.dumps(x) + "\n" for x in suite))
    return val_path, suite_path, abstained


def _argv(probe_bin: Path, tmp_path: Path, val: Path, suite: Path, eval_ledger: Path) -> list[str]:
    return ["--probe", str(probe_bin), "--val", str(val), "--suite", str(suite),
            "--split-key", "k", "--eval-ledger", str(eval_ledger),
            "--ledger", str(tmp_path / "out.jsonl"), "--report-out", str(tmp_path / "r.json")]


def test_the_row_is_quick_in_its_own_family_and_carries_the_reports_numbers(tmp_path, probe):
    eval_ledger = tmp_path / "eval.jsonl"
    # 0, 4, 8, ..., 28: eight of thirty.
    eval_id = _eval_row(eval_ledger, ood_value=8 / N_OOD)
    val, suite, abstained = _verdicts(tmp_path, eval_id, abstain_every=4)
    assert abstained == 8
    assert mpr.main(_argv(probe, tmp_path, val, suite, eval_ledger)) == 0

    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    (probed,) = Ledger(eval_ledger).rows()
    report = json.loads((tmp_path / "r.json").read_text())
    assert row.status == "completed" and row.quick and row.run_kind == "calibration"
    assert row.recipe is not None and "eval_row_id" not in row.recipe
    assert row.recipe["probed_eval_row"] == eval_id
    assert row.protocol.recipe_hash != probed.protocol.recipe_hash
    assert row.metrics["margin_probe.fitted_noul_margin"].value == report["fitted_noul_margin"]
    total = row.metrics["margin_probe.ood_total.abstained_rule_only"]
    assert (total.value, total.n, total.n_total) == (8, N_OOD, N_OOD)
    for category, body in report["ood"].items():
        got = row.metrics[f"margin_probe.ood.{category}.below_fitted_margin"]
        assert got.value == body["below_fitted_margin"] and got.n_total == body["n"]
        assert (row.metrics[f"margin_probe.ood.{category}.margin_p50"].value
                == body["margin_quantiles"]["p50"])
    assert (row.metrics["margin_probe.val_half_b.retained_precision"].value
            == report["val_half_b"]["retained_precision"])


def test_verdicts_that_disagree_with_the_eval_rows_gate_are_refused(tmp_path, probe):
    eval_ledger = tmp_path / "eval.jsonl"
    eval_id = _eval_row(eval_ledger, ood_value=8 / N_OOD)
    val, suite, abstained = _verdicts(tmp_path, eval_id, abstain_every=3)
    assert abstained == 10
    with pytest.raises(Refused, match="not that row's run"):
        mpr.main(_argv(probe, tmp_path, val, suite, eval_ledger))
    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    assert row.status == "failed" and "not that row's run" in row.notes + str(row.metrics)


def test_an_existing_report_is_not_overwritten(tmp_path, probe):
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        mpr.main(_argv(probe, tmp_path, tmp_path / "v", tmp_path / "s", tmp_path / "e"))
    assert not (tmp_path / "out.jsonl").exists()
