"""``qd-margin-probe`` (Rust) against the Python reference it must agree with.

The probe computes each margin with ``qd_runtime::calibration::calibrate`` and fits a
``noul_margin`` with the sweep ``python/qd_train/calibration_fit.py::fit_noul_margin`` runs.
Two implementations of one fit drift unless something pins them together; this does, on
random f32 logits, the key-split and the counts the report states.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import operator
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.calibration_fit import fit_noul_margin  # noqa: E402

CARGO = shutil.which("cargo")
pytestmark = pytest.mark.skipif(CARGO is None, reason="cargo is not on PATH: the probe was not built")


@pytest.fixture(scope="module")
def probe() -> Path:
    assert CARGO is not None
    subprocess.run(
        [CARGO, "build", "--quiet", "--manifest-path", str(REPO / "Cargo.toml"),
         "-p", "qd-runtime", "--bin", "qd-margin-probe"],
        check=True, timeout=900,
    )
    binary = REPO / "target" / "debug" / "qd-margin-probe"
    assert binary.is_file(), binary
    return binary


def _margin(logits: list[float]) -> float:
    """``calibrate`` at T=1, in the order it computes: f32 in, f64 softmax, top minus second."""
    scaled = [float(np.float32(v)) / 1.0 for v in logits]
    top_logit = max(scaled)
    exps = [math.exp(v - top_logit) for v in scaled]
    total = functools.reduce(operator.add, exps)
    probs = [e / total for e in exps]
    top = max(range(len(probs)), key=lambda i: (probs[i], -i))
    second = max((p for i, p in enumerate(probs) if i != top), default=0.0)
    return probs[top] - second


def _half_a(key: str, row_id: str) -> bool:
    return hashlib.sha256(key.encode() + b"\0" + row_id.encode()).digest()[0] % 2 == 0


def test_the_probe_fits_and_counts_what_the_python_reference_does(tmp_path, probe):
    rng = np.random.default_rng(7)
    val_lines, val_items = [], []
    for i in range(400):
        logits = rng.normal(0.0, 1.0 + 3.0 * rng.random(), size=5).astype(np.float32)
        correct = bool(rng.random() < 0.3 + 0.6 * (logits.max() - np.sort(logits)[-2]) / 4)
        row = f"v{i}"
        val_lines.append({"eval_row_id": "E", "seed": 0, "row_id": row, "kind": "choice",
                          "correct": correct, "top": int(logits.argmax()), "gold_row": 0,
                          "row_logits": logits.tolist(), "noul_row": 4, "rows": 5})
        val_items.append((row, _margin(logits.tolist()), correct))
    val_lines.append({"eval_row_id": "E", "seed": 0, "row_id": "s", "kind": "span",
                      "correct": True, "top": [1, 2], "gold_row": None})
    suite_lines, ood = [], {}
    for i in range(90):
        category = ("prose", "scrambled", "unseen-language")[i % 3]
        logits = rng.normal(0.0, 2.0, size=5).astype(np.float32)
        rule = bool(rng.random() < 0.2)
        suite_lines.append({"eval_row_id": "E", "seed": 0, "gate": "ood_abstain", "suite": "ood",
                            "case_id": f"o{i}", "category": category, "abstained": rule,
                            "row_logits_1": logits.tolist()})
        ood.setdefault(category, []).append((_margin(logits.tolist()), rule))
    val = tmp_path / "val.jsonl"
    suite = tmp_path / "suite.jsonl"
    val.write_text("".join(json.dumps(x) + "\n" for x in val_lines))
    suite.write_text("".join(json.dumps(x) + "\n" for x in suite_lines))
    out = tmp_path / "report.json"
    subprocess.run(
        [str(probe), "--val", str(val), "--suite", str(suite), "--split-key", "k",
         "--target-precision", "0.8", "--out", str(out)],
        check=True, timeout=120, capture_output=True,
    )
    report = json.loads(out.read_text())

    a = [(m, c) for row, m, c in val_items if _half_a("k", row)]
    b = [(m, c) for row, m, c in val_items if not _half_a("k", row)]
    thr = fit_noul_margin(np.array([m for m, _ in a]), np.array([c for _, c in a]),
                          target_precision=0.8)
    assert report["fitted_noul_margin"] == pytest.approx(thr, abs=1e-12)
    assert report["val_half_a"]["population"]["n"] == len(a)
    assert report["val_half_b"]["population"]["n"] == len(b)
    assert report["val_half_b"]["population"]["below_fitted_margin"] == sum(m < thr for m, _ in b)
    for category, cases in ood.items():
        got = report["ood"][category]
        assert got["n"] == len(cases)
        assert got["below_fitted_margin"] == sum(m < thr for m, _ in cases)
        assert got["abstained_rule_only"] == sum(r for _, r in cases)
        assert got["abstained_rule_or_margin"] == sum(r or m < thr for m, r in cases)


def test_the_probe_refuses_verdicts_without_logits(tmp_path, probe):
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps({"eval_row_id": "E", "row_id": "v", "kind": "choice",
                               "correct": True, "top": 0}) + "\n")
    suite = tmp_path / "suite.jsonl"
    suite.write_text(json.dumps({"eval_row_id": "E", "suite": "ood", "case_id": "o"}) + "\n")
    done = subprocess.run(
        [str(probe), "--val", str(val), "--suite", str(suite), "--split-key", "k",
         "--out", str(tmp_path / "r.json")],
        timeout=120, capture_output=True, text=True, check=False,
    )
    assert done.returncode != 0 and "older than 444bedc" in done.stderr
    assert not (tmp_path / "r.json").exists()
