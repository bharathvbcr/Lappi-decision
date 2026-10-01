"""``tools/calib_fit_row.py``: the Rust fit's report as a ledger row of its own.

The binary comes from conftest's ``calib_fit_bin`` fixture, so these tests run wherever the
parity tests do and skip with them when cargo is absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import calib_fit_row as cfr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402

from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

N_CHOICE, N_SPAN = 160, 40


def _verdicts(path: Path, eval_row_id: str, *, seed: int = 0) -> tuple[int, int]:
    """Choice rows at four options and span rows; returns (choice correct, span correct)."""
    rng = np.random.default_rng(5)
    lines, right_choice, right_span = [], 0, 0
    for i in range(N_CHOICE):
        z = rng.normal(0.0, 1.0, 5).astype(np.float32)
        gold = int(rng.integers(0, 4))
        z[gold if rng.random() < 0.8 else (gold + 1) % 4] += 3.0
        top = int(z.argmax())
        right_choice += top == gold
        lines.append({"eval_row_id": eval_row_id, "seed": seed, "row_id": f"r{i}",
                      "kind": "choice", "slot_name": "defect_class", "correct": top == gold,
                      "top": top, "gold_row": gold, "rows": 5, "noul_row": 4,
                      "row_logits": [float(v) for v in z]})
    for i in range(N_SPAN):
        ok = i % 10 != 0
        right_span += ok
        lines.append({"eval_row_id": eval_row_id, "seed": seed, "row_id": f"r{i}",
                      "kind": "span", "slot_name": "defect_span", "correct": ok, "top": [1, 1],
                      "gold_row": None, "rows": 20, "noul_row": 19})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return right_choice, right_span


def _eval_row(ledger_path: Path, *, seed: int, decoded: int, choice: tuple[int, int],
              span: tuple[int, int]) -> str:
    protocol = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                        backbone_commit="standin", recipe_hash="r" * 64, seed=seed)
    with RunRecorder(
        Ledger(ledger_path), entry_point=REPO / "tools" / "real_ft_run.py", protocol=protocol,
        run_kind="eval", repo=REPO, env=Environment.detect(device="cpu"), wall_clock_s=1.0,
        cost=None, quick=True, quick_reason="test fixture",
        recipe={"tool": "tools/real_ft_run.py", "tag": "epoch-score-val"},
    ) as rec:
        rec.metric("val_rows_decoded", Ran(passed=True, value=decoded, n=decoded,
                                           n_total=decoded))
        rec.metric("val_top1.choice", Ran(passed=True, value=choice[0] / choice[1],
                                          n=choice[0], n_total=choice[1]))
        rec.metric("val_top1.span", Ran(passed=True, value=span[0] / span[1],
                                        n=span[0], n_total=span[1]))
    assert rec.row is not None
    return rec.row.row_id


def _fixture(tmp_path: Path, *, decoded_offset: int = 0, seed: int = 0,
             name: str = "eval.jsonl") -> tuple[Path, Path]:
    eval_ledger = tmp_path / name
    verdicts = tmp_path / f"verdicts-{name}"
    # The eval row id is only known once it is written, so write the verdicts twice: the
    # first pass learns the counts, the second names the row.
    right = _verdicts(verdicts, "placeholder", seed=seed)
    eval_id = _eval_row(eval_ledger, seed=seed, decoded=N_CHOICE + N_SPAN + decoded_offset,
                        choice=(right[0], N_CHOICE), span=(right[1], N_SPAN))
    _verdicts(verdicts, eval_id, seed=seed)
    return eval_ledger, verdicts


def _argv(binary: Path, tmp_path: Path, verdicts: list[Path], eval_ledgers: list[Path],
          *extra: str) -> list[str]:
    argv = ["--bin", str(binary), "--name", "row-test-v1", "--ledger",
            str(tmp_path / "out.jsonl"), "--out-dir", str(tmp_path / "fit")]
    for v in verdicts:
        argv += ["--verdicts", str(v)]
    for e in eval_ledgers:
        argv += ["--eval-ledger", str(e)]
    return [*argv, *extra]


def test_the_row_is_quick_in_its_own_family_and_carries_the_reports_numbers(
    tmp_path, calib_fit_bin
):
    eval_ledger, verdicts = _fixture(tmp_path)
    assert cfr.main(_argv(calib_fit_bin, tmp_path, [verdicts], [eval_ledger],
                          "--population", "all")) == 0
    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    (fitted,) = Ledger(eval_ledger).rows()
    report = json.loads((tmp_path / "fit" / "report.json").read_text())
    assert row.status == "completed" and row.quick and row.run_kind == "calibration"
    assert row.recipe is not None and "eval_row_id" not in row.recipe
    assert row.recipe["fitted_eval_rows"] == [fitted.row_id]
    assert row.recipe["population"] == "all"
    assert row.protocol.recipe_hash != fitted.protocol.recipe_hash
    assert row.protocol.seed == fitted.protocol.seed
    table = (tmp_path / "fit" / "table.json").read_bytes()
    assert row.metrics["calib_fit.table_sha256"].value == hashlib.sha256(table).hexdigest()
    fit = report["entries"]["choice:5"]["fit"]
    for key in ("temperature", "noul_margin", "conformal_quantile"):
        got = row.metrics[f"calib_fit.choice.k4.{key}"]
        assert got.value == fit[key] and (got.n, got.n_total) == (N_CHOICE, N_CHOICE)
    for key in ("ece_before", "ece_after"):
        state = row.metrics[f"calib_fit.choice.k4.{key}"]
        assert isinstance(state, Ran) and state.n == N_CHOICE
    spans = row.metrics["calib_fit.population.span_rows_not_fitted"]
    assert (spans.value, spans.n_total) == (N_SPAN, N_CHOICE + N_SPAN)
    assert row.metrics["calib_fit.span"].state == "not_run"


def test_two_fold_records_both_fold_tables_and_their_parameters(tmp_path, calib_fit_bin):
    eval_ledger, verdicts = _fixture(tmp_path)
    assert cfr.main(_argv(calib_fit_bin, tmp_path, [verdicts], [eval_ledger],
                          "--population", "two-fold", "--split-key", "k")) == 0
    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    for fold in ("a", "b"):
        table = (tmp_path / "fit" / f"table.fold-{fold}.json").read_bytes()
        assert (row.metrics[f"calib_fit.table_sha256.fold_{fold}"].value
                == hashlib.sha256(table).hexdigest())
        assert f"calib_fit.choice.k4.fold_{fold}.temperature" in row.metrics
    assert "calib_fit.table_sha256" not in row.metrics
    assert "out of fold" in row.metrics["calib_fit.choice.k4.ece_after"].detail


def test_counts_that_disagree_with_the_eval_row_are_refused(tmp_path, calib_fit_bin):
    eval_ledger, verdicts = _fixture(tmp_path, decoded_offset=1)
    with pytest.raises(Refused, match="not that row's run"):
        cfr.main(_argv(calib_fit_bin, tmp_path, [verdicts], [eval_ledger],
                       "--population", "all"))
    (row,) = Ledger(tmp_path / "out.jsonl").rows()
    assert row.status == "failed"


def test_two_models_are_refused_before_anything_runs(tmp_path, calib_fit_bin):
    e0, v0 = _fixture(tmp_path, seed=0, name="s0.jsonl")
    e1, v1 = _fixture(tmp_path, seed=1, name="s1.jsonl")
    with pytest.raises(Refused, match="differ in seed"):
        cfr.main(_argv(calib_fit_bin, tmp_path, [v0, v1], [e0, e1], "--population", "all"))
    assert not (tmp_path / "out.jsonl").exists()
    assert not (tmp_path / "fit").exists()


def test_an_existing_out_dir_is_not_overwritten(tmp_path, calib_fit_bin):
    (tmp_path / "fit").mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cfr.main(_argv(calib_fit_bin, tmp_path, [tmp_path / "v"], [tmp_path / "e"],
                       "--population", "all"))
    assert not (tmp_path / "out.jsonl").exists()
