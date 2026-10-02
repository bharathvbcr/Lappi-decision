"""``qd-calib-fit`` (Rust) against ``calibration_fit.py`` (the oracle), bit for bit.

The parity definition is **equality**: every fitted temperature, ``noul_margin``,
nonconformity quantile and conformal cutoff, every ECE, every count and rate, every per-row
probability and margin, and every table value, compared with ``==`` on the float64 the
binary printed and the oracle computed. No tolerance. That holds on macOS arm64, numpy 2.5,
where ``np.exp``/``np.log`` and Rust's ``f64::exp``/``ln`` are the same libm; see
``crates/qd-runtime/src/calibration_fit.rs`` for the three numpy behaviours ported to get there.

The oracle (``calib_fit_oracle.py``) calls the library functions themselves; it adds only the
composition the binary performs and the runtime's ``calibrate``.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from calib_fit_oracle import OracleRefusal, calibrate, fold_of, read, run

from qd_train.calibration_fit import probability_cutoff_from_nonconformity
from qd_train.eval_harness import split_conformal_threshold

#: The real phase-3 seed-0 verdicts the margin probe read (eval row edc99143), when on disk.
REAL_VERDICTS = Path("/Users/bharath/qd-campaign/margin-2026-09-30/verdicts-s0.jsonl")
#: That eval row's own ``ece.choice.k4`` metric (gh200-margin-rescore-2026-09-30.jsonl).
REAL_ECE_CHOICE_K4 = 0.0016595129791819982

#: ``(kind, decode rows, slot, language-ish family, accuracy, gold-noul share)``, shaped on
#: J1's mixture: defect_class at four options, CSQA at five, MMLU at four, CLINC intents at
#: ten with an out-of-scope (gold-noul) share, and a five-bin score slot.
SHAPES = (
    ("choice", 5, "defect_class", "code.defect_class", 0.997, 0.0),
    ("choice", 6, "answer", "csqa", 0.72, 0.0),
    ("choice", 5, "answer", "mmlu", 0.53, 0.0),
    ("choice", 11, "intent", "clinc", 0.60, 0.15),
    ("score", 6, "severity", "code.severity", 0.80, 0.0),
)


def _bf16(x: float) -> float:
    """Round an f32 to bf16 (nearest even) and widen back: the decoder's logits are bf16."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def synthesize(path: Path, *, letters: int, spans: int, seed: int = 0,
               eval_row: str = "E-synth", shapes=SHAPES) -> None:
    """A verdict file with the real writer's key set (``_verdict_lines``)."""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for i in range(letters):
        kind, rows, slot, family, acc, noul_share = shapes[i % len(shapes)]
        noul = rows - 1
        gold = noul if rng.random() < noul_share else int(rng.integers(0, noul))
        logits = rng.normal(12.0, 2.5, rows)
        winner = gold if rng.random() < acc else int((gold + 1 + rng.integers(0, rows - 1)) % rows)
        logits[winner] += rng.gamma(2.0, 3.0)
        z = [_bf16(float(np.float32(v))) for v in logits]
        top = max(range(rows), key=lambda j: (z[j], -j))
        exps = [math.exp(v - max(z)) for v in z]
        out.append(json.dumps({
            "eval_row_id": eval_row, "seed": 0, "row_id": f"qdm:{family}:{i:06d}",
            "kind": kind, "slot_name": slot, "correct": top == gold, "top": top,
            "gold_row": gold, "expected_abstain": gold == noul, "noul_row": noul, "rows": rows,
            "language": ("python", "go", "rust", "typescript")[i % 4],
            "noul_probability": exps[noul] / sum(exps), "row_logits": z,
            "family_id": family,
        }, sort_keys=True))
    for i in range(spans):
        rows = int(rng.integers(10, 61))
        start = int(rng.integers(0, rows - 1))
        out.append(json.dumps({
            "eval_row_id": eval_row, "seed": 0, "row_id": f"qdm:code.defect_class:{i:06d}",
            "kind": "span", "slot_name": "defect_span", "correct": bool(rng.random() < 0.99),
            "top": [start, start], "gold_row": None, "expected_abstain": False,
            "noul_row": rows - 1, "rows": rows, "family_id": "code.defect_class",
        }, sort_keys=True))
    path.write_text("".join(line + "\n" for line in out), encoding="utf-8")


def fit_bin(binary: Path, paths: list[Path], out: Path, *, population: str = "all",
            split_key: str | None = None, name: str = "parity-v1",
            alpha: float = 0.1, target: float = 0.95, check: bool = True,
            ) -> subprocess.CompletedProcess[str]:
    cmd = [str(binary)]
    for p in paths:
        cmd += ["--verdicts", str(p)]
    cmd += ["--population", population, "--name", name, "--out-dir", str(out),
            "--alpha", repr(alpha), "--target-precision", repr(target)]
    if split_key is not None:
        cmd += ["--split-key", split_key]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if check and done.returncode != 0:
        raise AssertionError(f"qd-calib-fit failed: {done.stderr}")
    return done


def _same(path: str, got: object, want: object) -> None:
    """Equality, recursively; a float compares by ``==`` and says which leaf differed."""
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: {got!r} is not an object"
        for k, v in want.items():
            assert k in got, f"{path}.{k} missing from the binary's output"
            _same(f"{path}.{k}", got[k], v)
    elif isinstance(want, float) and isinstance(got, (int, float)):
        assert float(got) == want, f"{path}: binary {got!r} oracle {want!r}"
    else:
        assert got == want, f"{path}: binary {got!r} oracle {want!r}"


def assert_parity(out: Path, ref: dict[str, object]) -> dict[str, object]:
    report = json.loads((out / "report.json").read_text())
    _same("entries", report["entries"], ref["entries"])
    assert set(report["entries"]) == set(ref["entries"])
    for i, (which, table) in enumerate(sorted(ref["tables"].items())):
        meta = report["tables"][i]
        assert meta["fitted_on"] == which
        written = json.loads((out / meta["file"]).read_text())
        _same(f"tables.{which}", written, table)
        assert written == table
    lines = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    by_id = {(x["eval_row_id"], x["row_id"], x["slot_name"]): x for x in lines}
    assert len(by_id) == len(lines) == len(ref["rows"])
    for want in ref["rows"]:
        _same(f"row {want['row_id']}", by_id[(want["eval_row_id"], want["row_id"],
                                              want["slot_name"])], want)
    counts = report["population_counts"]
    assert counts["letter_rows"] == ref["letter_rows"]
    assert counts["span_rows_not_fitted"] == ref["span_rows"]
    return report


@pytest.fixture(scope="module")
def mixed_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mixed") / "verdicts.jsonl"
    synthesize(path, letters=2500, spans=900, seed=3)
    return path


@pytest.mark.parametrize(("population", "split_key"), [("all", None), ("two-fold", "j7")])
def test_real_shaped_verdicts_fit_bit_exactly(calib_fit_bin, mixed_file, tmp_path,
                                              population, split_key):
    out = tmp_path / "fit"
    fit_bin(calib_fit_bin, [mixed_file], out, population=population, split_key=split_key)
    ref = run([mixed_file], population=population, split_key=split_key, name="parity-v1")
    report = assert_parity(out, ref)
    # Mixed row counts per kind, both kinds, and gold-noul rows were all exercised.
    assert set(report["entries"]) == {"choice:5", "choice:6", "choice:11", "score:6"}
    assert report["entries"]["choice:11"]["gold_noul"] > 0
    assert report["tables"][0]["span"] == "null: not fitted"


@pytest.mark.skipif(not REAL_VERDICTS.is_file(), reason=f"{REAL_VERDICTS} is not on disk")
@pytest.mark.parametrize(("population", "split_key"), [("all", None), ("two-fold", "j7")])
def test_the_real_phase3_verdicts_fit_bit_exactly(calib_fit_bin, tmp_path, population,
                                                  split_key):
    out = tmp_path / "fit"
    fit_bin(calib_fit_bin, [REAL_VERDICTS], out, population=population, split_key=split_key)
    ref = run([REAL_VERDICTS], population=population, split_key=split_key, name="parity-v1")
    report = assert_parity(out, ref)
    # T=1 ECE is the eval row's own ece.choice.k4, to the bit: same quantity, same rows.
    assert report["entries"]["choice:5"]["ece_before"]["value"] == REAL_ECE_CHOICE_K4
    assert report["population_counts"]["span_rows_not_fitted"] == 2053


@pytest.mark.parametrize(("accuracy", "weak"), [(0.45, True), (0.97, False)])
def test_the_conformal_cutoff_is_one_minus_the_nonconformity_quantile(
    calib_fit_bin, tmp_path, accuracy, weak
):
    """GAP-CALIB-CONFORMAL-SCALE-TWO-MEANINGS, through the binary, in both regimes.

    ``q̂`` from ``split_conformal_threshold`` is on the nonconformity scale; the table must
    hold ``probability_cutoff_from_nonconformity(q̂)``. At ``q̂ = 0.5`` the two coincide and
    writing ``q̂`` would pass unnoticed, so a weak model (``q̂ > 0.5``) and a strong one
    (``q̂ < 0.5``) are both pinned, each far from 0.5, and the served sets must cover.
    """
    # A fixed boost on the winner, so accuracy alone sets the regime.
    rng = np.random.default_rng(11)
    lines = []
    for i in range(1500):
        z = rng.normal(0.0, 1.0, 5).astype(np.float32)
        gold = int(rng.integers(0, 4))
        z[gold if rng.random() < accuracy else (gold + 1) % 4] += 6.0
        lines.append(_letter(i, [float(v) for v in z], gold))
    path = _write(tmp_path / "v.jsonl", lines)
    out = tmp_path / "fit"
    fit_bin(calib_fit_bin, [path], out)
    report = json.loads((out / "report.json").read_text())
    fit = report["entries"]["choice:5"]["fit"]
    ref = run([path], population="all", split_key=None, name="parity-v1")
    ref_fit = ref["entries"]["choice:5"]["fit"]
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]

    # The oracle's q̂ is split_conformal_threshold over calibrate()'s probabilities at T.
    letters, _ = read([path])
    probs = np.asarray([calibrate(r.logits, ref_fit["temperature"])[0] for r in letters])
    q_hat = split_conformal_threshold(probs, np.asarray([r.gold for r in letters]), alpha=0.1)
    assert fit["nonconformity_quantile"] == q_hat
    assert fit["conformal_quantile"] == probability_cutoff_from_nonconformity(q_hat)
    assert fit["conformal_quantile"] == 1.0 - q_hat
    assert (q_hat > 0.5) is weak, q_hat
    assert abs(fit["conformal_quantile"] - q_hat) > 0.2, "too near 0.5 to tell the scales apart"
    table = json.loads((out / "table.json").read_text())
    assert table["letters"]["choice:5"]["conformal_quantile"] == 1.0 - q_hat
    assert table["letters"]["choice:5"]["conformal_quantile"] != q_hat
    coverage = sum(r["gold_in_set"] for r in rows) / len(rows)
    assert coverage >= 0.9, coverage


def _write(path: Path, lines: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return path


def _letter(i: int, logits: list[float], gold: int, *, kind: str = "choice",
            slot: str = "c", eval_row: str = "E", row_id: str | None = None) -> dict:
    rows = len(logits)
    top = max(range(rows), key=lambda j: (np.float32(logits[j]), -j))
    return {"eval_row_id": eval_row, "seed": 0, "row_id": row_id or f"r{i}", "kind": kind,
            "slot_name": slot, "correct": top == gold, "top": top, "gold_row": gold,
            "rows": rows, "noul_row": rows - 1, "row_logits": logits}


def _random_lines(n: int, rows: int, seed: int, **kw) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        z = rng.normal(0, 3, rows).astype(np.float32)
        gold = int(rng.integers(0, rows))
        if rng.random() < 0.7:
            z[gold] += 4
        out.append(_letter(i, [float(v) for v in z], gold, **kw))
    return out


def _parity(binary: Path, tmp_path: Path, paths: list[Path], *, population: str = "all",
            split_key: str | None = None) -> dict[str, object]:
    out = tmp_path / f"fit-{population}"
    fit_bin(binary, paths, out, population=population, split_key=split_key)
    ref = run(paths, population=population, split_key=split_key, name="parity-v1")
    return assert_parity(out, ref)


def test_an_empty_kind_gets_no_entry_and_says_so(calib_fit_bin, tmp_path):
    path = _write(tmp_path / "v.jsonl", _random_lines(150, 4, seed=1))
    report = _parity(calib_fit_bin, tmp_path, [path])
    assert set(report["entries"]) == {"choice:4"}
    assert any("no score row" in c for c in report["caveats"])
    table = json.loads((tmp_path / "fit-all" / "table.json").read_text())
    assert set(table["letters"]) == {"choice:4"} and table["span"] is None


def test_a_single_class_entry_fits_at_the_bound_like_the_reference(calib_fit_bin, tmp_path):
    rng = np.random.default_rng(2)
    lines = []
    for i in range(120):
        z = rng.normal(0, 1, 3).astype(np.float32)
        z[0] += 6  # always right, always gold 0
        lines.append(_letter(i, [float(v) for v in z], 0))
    report = _parity(calib_fit_bin, tmp_path, [_write(tmp_path / "v.jsonl", lines)])
    fit = report["entries"]["choice:3"]["fit"]
    assert fit["temperature_at_lower_bound"] and not fit["temperature_at_upper_bound"]


def test_an_all_noul_entry_fits_like_the_reference(calib_fit_bin, tmp_path):
    lines = [dict(x, gold_row=4, expected_abstain=True)
             for x in _random_lines(130, 5, seed=4)]
    report = _parity(calib_fit_bin, tmp_path, [_write(tmp_path / "v.jsonl", lines)])
    assert report["entries"]["choice:5"]["gold_noul"] == 130


def test_under_a_hundred_rows_ece_is_not_run_with_the_gates_reason(calib_fit_bin, tmp_path):
    report = _parity(calib_fit_bin, tmp_path,
                     [_write(tmp_path / "v.jsonl", _random_lines(40, 5, seed=5))])
    for state in (report["entries"]["choice:5"]["ece_before"],
                  report["entries"]["choice:5"]["scored"]["ece_after"]):
        assert state["state"] == "not_run" and "at least 100" in state["reason"]


def test_mixed_row_counts_per_kind_fit_one_entry_each(calib_fit_bin, tmp_path):
    lines = (_random_lines(110, 3, seed=6, slot="a") + _random_lines(110, 17, seed=7, slot="b")
             + _random_lines(110, 3, seed=8, kind="score", slot="s")
             + _random_lines(110, 6, seed=9, kind="score", slot="t"))
    for i, x in enumerate(lines):
        x["row_id"] = f"m{i}"
    report = _parity(calib_fit_bin, tmp_path, [_write(tmp_path / "v.jsonl", lines)],
                     population="two-fold", split_key="mix")
    assert set(report["entries"]) == {"choice:3", "choice:17", "score:3", "score:6"}


def test_a_row_id_shared_by_slots_or_eval_rows_is_not_a_duplicate(calib_fit_bin, tmp_path):
    """A row's choice and span slots share its row_id (2053 times in the real file), and so do
    three seeds' files; both are legitimate. The same row's slots land in the same fold."""
    a = _random_lines(120, 5, seed=10, slot="class", eval_row="E1")
    b = [dict(x, slot_name="class2") for x in _random_lines(120, 5, seed=11, eval_row="E1")]
    c = _random_lines(120, 5, seed=12, slot="class", eval_row="E2")
    paths = [_write(tmp_path / "e1.jsonl", a + b), _write(tmp_path / "e2.jsonl", c)]
    report = _parity(calib_fit_bin, tmp_path, paths, population="two-fold", split_key="k")
    assert report["entries"]["choice:5"]["n"] == 360
    rows = [json.loads(x) for x in (tmp_path / "fit-two-fold" / "rows.jsonl").read_text()
            .splitlines()]
    folds: dict[str, set[str]] = {}
    for r in rows:
        folds.setdefault(r["row_id"], set()).add(r["fold"])
    assert all(len(f) == 1 for f in folds.values())


def test_an_exact_duplicate_is_refused_by_both(calib_fit_bin, tmp_path):
    lines = _random_lines(120, 5, seed=13)
    path = _write(tmp_path / "v.jsonl", [*lines, lines[7]])
    done = fit_bin(calib_fit_bin, [path], tmp_path / "out", check=False)
    assert done.returncode != 0 and "appears twice" in done.stderr
    assert not (tmp_path / "out").exists()
    with pytest.raises(OracleRefusal, match="appears twice"):
        run([path], population="all", split_key=None, name="x")


@pytest.mark.parametrize("poison", ["NaN", "Infinity", "-Infinity", "1e39"])
def test_a_non_finite_logit_is_refused_by_both(calib_fit_bin, tmp_path, poison):
    lines = _random_lines(120, 5, seed=14)
    lines[3]["row_logits"][1] = 12345.5  # a sentinel no other logit spells
    body = "".join(json.dumps(x) + "\n" for x in lines)
    assert body.count("12345.5") == 1
    body = body.replace("12345.5", poison)
    path = tmp_path / "v.jsonl"
    path.write_text(body, encoding="utf-8")
    done = fit_bin(calib_fit_bin, [path], tmp_path / "out", check=False)
    assert done.returncode != 0 and "non-finite" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="non-finite"):
        run([path], population="all", split_key=None, name="x")


def test_a_fold_with_no_row_of_an_entry_scores_nothing_out_of_fold(calib_fit_bin, tmp_path):
    """Rows whose other fold fitted nothing are counted as not scored, never as covered."""
    lines = _random_lines(200, 5, seed=15)
    lonely = [x for x in _random_lines(40, 3, seed=16, slot="z")
              if fold_of("k", x["row_id"] + "-z") == "a"]
    for x in lonely:
        x["row_id"] += "-z"
    report = _parity(calib_fit_bin, tmp_path, [_write(tmp_path / "v.jsonl", lines + lonely)],
                     population="two-fold", split_key="k")
    entry = report["entries"]["choice:3"]
    assert entry["folds"]["b"]["fitted"] is False
    assert entry["scored"]["n_scored"] == 0 and entry["scored"]["n_not_scored"] == len(lonely)
    assert entry["scored"]["conformal_coverage"] is None


#: A/B rounds of the benchmark below; each round runs the oracle, then the binary.
BENCH_ROUNDS = 5


@pytest.mark.skipif(
    os.environ.get("QD_CALIB_FIT_BENCH") != "1",
    reason="the oracle-vs-qd-calib-fit benchmark runs only with QD_CALIB_FIT_BENCH=1",
)
def test_benchmark_oracle_against_qd_calib_fit_interleaved_min_of_n(calib_fit_bin, tmp_path):
    """The committed A/B: one verdict file at J1's scale (11k letter rows, 7k span rows,
    synthesized with the real writer's key set), and the real phase-3 file where it is on
    disk, fitted by the oracle and by the release binary in alternating rounds, min of
    :data:`BENCH_ROUNDS` each. The binary's time includes the process, reading, fitting,
    and writing its report, tables and rows file.

        QD_CALIB_FIT_BENCH=1 pytest -s python/tests/test_calib_fit_parity.py -k benchmark
    """
    synth = tmp_path / "j1-scale.jsonl"
    synthesize(synth, letters=11000, spans=7000, seed=21)
    inputs = [("synthetic-j1-scale", synth)]
    if REAL_VERDICTS.is_file():
        inputs.append(("real-phase3-s0", REAL_VERDICTS))
    for label, path in inputs:
        for population, key in (("all", None), ("two-fold", "bench")):
            a: list[float] = []
            b: list[float] = []
            for r in range(BENCH_ROUNDS):
                started = time.perf_counter()
                ref = run([path], population=population, split_key=key, name="bench")
                a.append(time.perf_counter() - started)
                out = tmp_path / f"{label}-{population}-{r}"
                started = time.perf_counter()
                fit_bin(calib_fit_bin, [path], out, population=population, split_key=key,
                        name="bench")
                b.append(time.perf_counter() - started)
                if r == 0:
                    assert_parity(out, ref)
            print(json.dumps({
                "benchmark": "calib_fit", "input": label, "population": population,
                "lines": sum(1 for _ in path.open(encoding="utf-8")),
                "letter_rows": ref["letter_rows"], "span_rows": ref["span_rows"],
                "rounds": BENCH_ROUNDS, "oracle_s": [round(x, 3) for x in a],
                "qd_calib_fit_s": [round(x, 3) for x in b],
                "oracle_min_s": round(min(a), 3), "qd_calib_fit_min_s": round(min(b), 3),
                "speedup_min_over_min": round(min(a) / min(b), 1),
            }))
            assert min(b) < min(a)
