"""``qd-gate-report`` (Rust) against the Python that owns each number it re-reports.

Fable round G, G1-G3. The reporter is report-only and offline: it reads ``--verdicts-out``
files on CPU and re-reports every gate per family beside its pooled value (G1), pooled and
per-language ECE with a ``none`` bucket (G2), and the two- and four-option slot diagnostics
(G3). The oracles here are the functions the eval row's numbers come from --
``real_ft_run.calibration_states`` / ``family_eces`` / ``permutation_family_metrics`` /
``choice_rule_abstentions`` / ``in_distribution_family_metrics``, ``eval_harness`` and
``calibration_fit`` -- run on a synthetic scoring run whose eval row is written through
``RunRecorder`` exactly as ``_record_score`` writes one.

Equality is exact for every number the eval row records (each ECE, mean entropy and count):
the binary sums the way numpy does. The slot marginals' means and sigmas are compared to 1e-12.
The refusals are tested too: a verdict file whose numbers disagree with its eval row, one whose
eval row is absent, and a row that records nothing comparable are each refused, not reported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_train.calibration_fit import ece_gate  # noqa: E402
from qd_train.eval_harness import (  # noqa: E402
    DEFAULT_ENTROPY_FLOOR,
    DEFAULT_MAX_CLASS_SHARE,
    _entropy,
    degenerate_head_check,
)
from qd_train.ledger import (  # noqa: E402
    DEFAULT_DECISIONS_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
    load_promotion_decisions,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

SEED = 7
#: (family, kind, rows incl. noul, languages to cycle, count, gold-noul share)
FAMILIES: tuple[tuple[str, str, int, tuple[str | None, ...], int, float], ...] = (
    ("code.defect_class", "choice", 5, ("python", "go"), 260, 0.0),
    ("knowledge.multiple_choice", "choice", 5, (None,), 180, 0.0),
    ("intent.in_scope", "choice", 3, (None,), 220, 0.05),
    ("intent.domain", "choice", 11, (None,), 140, 0.0),
)


def _derangement(rng: np.random.Generator, k: int) -> tuple[int, ...]:
    while True:
        perm = tuple(int(x) for x in rng.permutation(k))
        if all(perm[j] != j for j in range(k)):
            return perm


def _scoring_run(rng: np.random.Generator) -> tuple[dict[str, object], dict[str, object],
                                                    dict[tuple[str, str], tuple[int, ...]]]:
    """A first pass, its permuted second pass and the derangements, as _decode would give."""
    first: list[dict[str, object]] = []
    second: list[dict[str, object]] = []
    perms: dict[tuple[str, str], tuple[int, ...]] = {}
    for family, kind, rows, languages, count, noul_share in FAMILIES:
        options = rows - 1
        for i in range(count):
            gold = options if rng.random() < noul_share else int(rng.integers(options))
            z = rng.normal(0.0, 1.5, size=rows)
            z[gold] += 2.5 if family != "knowledge.multiple_choice" else 0.6
            z[-1] -= 3.0 if gold != options else -4.0
            # bf16-like: logits on a 1/16 grid, so ties happen and argmax must take the first.
            logits = [float(np.round(x * 16) / 16) for x in z + 20.0]
            top = int(np.argmax(logits))
            row_id = f"{family}:{i}"
            slot = "answer"
            first.append({
                "kind": kind, "row_id": row_id, "slot_name": slot, "family_id": family,
                "language": languages[i % len(languages)], "rows": rows,
                "noul_row": options, "gold_row": gold, "top": top, "row_logits": logits,
                "correct": top == gold, "expected_abstain": gold == options,
                "runtime_verdict": "abstain" if top == options else "answer",
            })
            perm = _derangement(rng, options)
            perms[(row_id, slot)] = perm
            if top == options or rng.random() < 0.15:
                top2 = int(rng.integers(rows))
            else:
                top2 = perm.index(top)
            second.append({"kind": kind, "row_id": row_id, "slot_name": slot, "top": top2,
                           "noul_row": options})
    for i in range(120):
        family = "qa.answer_span" if i % 3 else "code.defect_class"
        first.append({
            "kind": "span", "row_id": f"span:{i}", "slot_name": "evidence", "family_id": family,
            "rows": 30, "noul_row": 29, "gold_row": None, "top": [3, 4],
            "correct": bool(rng.random() < 0.8), "expected_abstain": False,
            "runtime_verdict": "answer",
        })
    return {"verdicts": first}, {"verdicts": second}, perms


def _env() -> Environment:
    return Environment(
        torch="2.12.1", transformers_sha="abc123", device="cpu", host="test",
        fla_present=NotRun(reason="test"), causal_conv1d_present=NotRun(reason="test"),
    )


def _record_eval_row(ledger: Ledger, scored: dict[str, object], second: dict[str, object],
                     perms: dict[tuple[str, str], tuple[int, ...]], *,
                     tamper: str | None = None) -> tuple[str, dict[str, object]]:
    """Every number _record_score and score_ood put on an eval row, from the real functions."""
    rft.annotate_second_pass(scored, second, perms)
    gate = rft.permutation_agreement(scored, second, perms)
    metrics, ece, degenerate = rft.calibration_states(scored)
    metrics.update(rft.permutation_family_metrics(scored, gate))
    indist = rft.choice_rule_abstentions(scored, second, perms)
    gold_noul = rft.choice_rule_abstentions(scored, second, perms, gold_noul=True)
    in_k = sum(indist.values())
    metrics["ood_abstain.in_distribution"] = Ran(passed=True, value=in_k / len(indist), n=in_k,
                                                 n_total=len(indist))
    metrics.update(rft.in_distribution_family_metrics(scored, indist, gold_noul))
    verdicts: list[dict[str, object]] = scored["verdicts"]  # type: ignore[assignment]
    for kind in ("choice", "span"):
        rows = [v for v in verdicts if v["kind"] == kind]
        k = sum(1 for v in rows if v["correct"])
        metrics[f"val_top1.{kind}"] = Ran(passed=True, value=k / len(rows), n=k,
                                          n_total=len(rows))
    metrics["val_rows_decoded"] = Ran(passed=True, value=len(verdicts), n=len(verdicts),
                                      n_total=len(verdicts))
    if tamper == "permutation_count" and isinstance(gate, Ran) and gate.n is not None:
        gate = Ran(passed=gate.passed, value=gate.value, n=gate.n - 1, n_total=gate.n_total)
    with RunRecorder(
        ledger, protocol=Protocol("d" * 64, "t" * 64, "b" * 40, "r" * 64, SEED),
        run_kind="eval", repo=REPO, env=_env(), wall_clock_s=None, cost=None,
        recipe={"tool": "tools/real_ft_run.py", "tag": "epoch-score-val"},
    ) as rec:
        if tamper != "nothing_comparable":
            for name, state in metrics.items():
                rec.metric(name, state)
            rec.gate("ece", ece)
            rec.control("degenerate_head", degenerate)
            rec.gate("permutation_consistency", gate)
    row = ledger.rows()[-1]
    return row.row_id, {name: state for name, state in metrics.items()}


def _write_verdicts(path: Path, scored: dict[str, object], eval_row_id: str, *,
                    drop_family: bool = False) -> Path:
    lines = []
    for i, v in enumerate(scored["verdicts"]):  # type: ignore[union-attr]
        line = {**v, "eval_row_id": eval_row_id, "seed": SEED}
        line.pop("runtime_verdict", None)
        if drop_family and i == 0:
            line.pop("family_id")
        lines.append(json.dumps(line, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(binary: Path, tmp_path: Path, verdicts: Path, ledger: Path,
         ) -> tuple[subprocess.CompletedProcess[str], dict[str, object] | None]:
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [str(binary), "--verdicts", str(verdicts), "--eval-ledger", str(ledger),
         "--decisions", str(DEFAULT_DECISIONS_PATH), "--out-json", str(out)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return proc, report


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    scored, second, perms = _scoring_run(np.random.default_rng(20261001))
    ledger = Ledger(tmp_path / "eval.jsonl")
    eval_row_id, metrics = _record_eval_row(ledger, scored, second, perms)
    verdicts = _write_verdicts(tmp_path / "verdicts.jsonl", scored, eval_row_id)
    return scored, metrics, verdicts, ledger.path


def _family_rows(scored: dict[str, object], family: str | None = None,
                 shape: str | None = None) -> list[dict[str, object]]:
    return [
        v for v in scored["verdicts"]  # type: ignore[union-attr]
        if v["kind"] != "span"
        and (family is None or v["family_id"] == family)
        and (shape is None or f"{v['kind']}.k{int(v['rows']) - 1}" == shape)
    ]


def _padded(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """Each row's own softmax, zero-padded to the widest: max and argmax are unchanged, so
    this is the oracle for an ECE by top-1 confidence across slot shapes."""
    width = max(int(v["rows"]) for v in rows)
    probs = np.zeros((len(rows), width))
    for i, v in enumerate(rows):
        z = np.asarray([v["row_logits"]], dtype=np.float64)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        probs[i, : z.shape[1]] = (z / z.sum(axis=1, keepdims=True))[0]
    return probs, np.asarray([v["gold_row"] for v in rows], dtype=int)


def test_every_number_the_eval_row_records_is_recomputed_and_equal(gate_report_bin, tmp_path,
                                                                   fixture):
    _, metrics, verdicts, ledger = fixture
    proc, report = _run(gate_report_bin, tmp_path, verdicts, ledger)
    assert proc.returncode == 0, proc.stderr
    assert report is not None
    (row,) = report["eval_rows"]
    checked = set(row["cross_check"]["names"])
    ran = {f"metrics.{name}" for name, s in metrics.items()
           if isinstance(s, Ran) and isinstance(s.value, float | int)
           and not name.startswith("ood_abstain.in_distribution.gold_noul")}
    assert ran <= checked, sorted(ran - checked)
    assert "gates.permutation_consistency" in checked
    assert row["cross_check"]["checked"] == len(checked)


def test_g1_per_family_values_equal_the_python_breakdowns(gate_report_bin, tmp_path, fixture):
    scored, metrics, verdicts, ledger = fixture
    _, report = _run(gate_report_bin, tmp_path, verdicts, ledger)
    assert report is not None
    g1 = report["eval_rows"][0]["g1"]
    for family, *_ in FAMILIES:
        perm = metrics[f"permutation_consistency.family.{family}"]
        got = g1["gates"]["permutation_consistency"]["per_family"][family]
        assert (got["n"], got["n_total"], got["value"]) == (perm.n, perm.n_total, perm.value)
        ind = metrics[f"ood_abstain.in_distribution.family.{family}"]
        got = g1["gates"]["ood_abstain"]["per_family"][family]["in_distribution"]
        assert (got["n"], got["n_total"]) == (ind.n, ind.n_total)
        for name, state in metrics.items():
            prefix = f"ece.family.{family}."
            if name.startswith(prefix):
                shape = name[len(prefix):]
                got = g1["gates"]["ece"]["per_family"][family][shape]
                assert got["value"] == state.value, (name, got, state)
                probs, _ = rft.letter_distributions(_family_rows(scored, family, shape))[shape]
                head = degenerate_head_check(probs)
                got = g1["controls"]["degenerate_head"]["per_family"][family][shape]
                assert got["value"] == head.value
                share = np.bincount(np.argmax(probs, axis=1), minlength=probs.shape[1]).max()
                assert got["top_predicted_share"] == float(share / len(probs))
    # A span-only family is named, with the reason, never dropped.
    span_only = g1["gates"]["permutation_consistency"]["per_family"]["qa.answer_span"]
    assert span_only["state"] == "not_run" and "choice rows only" in span_only["reason"]
    # The gates no verdict file can split by family say so per family.
    for name in ("needle_hunk_recall", "paired_margin_vs_linear"):
        for state in g1["gates"][name]["per_family"].values():
            assert state["state"] == "not_run"


def test_g2_pooled_and_per_language_ece_with_a_none_bucket(gate_report_bin, tmp_path, fixture):
    scored, _, verdicts, ledger = fixture
    _, report = _run(gate_report_bin, tmp_path, verdicts, ledger)
    assert report is not None
    g2 = report["eval_rows"][0]["g2"]
    rows = _family_rows(scored)
    pooled = ece_gate(*_padded(rows))
    assert isinstance(pooled, Ran)
    assert g2["pooled_all_letter_rows"]["value"] == pooled.value
    assert g2["pooled_all_letter_rows"]["n"] == len(rows)
    for language in ("python", "go", None):
        mine = [v for v in rows if v["language"] == language]
        oracle = ece_gate(*_padded(mine))
        got = g2["by_language"]["none" if language is None else language]
        assert isinstance(oracle, Ran)
        assert (got["value"], got["n"]) == (oracle.value, len(mine)), language
    assert "passed" not in g2["pooled_all_letter_rows"], "a report value is not a gate"


def test_g3_slot_diagnostics_for_two_and_four_options(gate_report_bin, tmp_path, fixture):
    scored, _, verdicts, ledger = fixture
    _, report = _run(gate_report_bin, tmp_path, verdicts, ledger)
    assert report is not None
    slots = report["eval_rows"][0]["g3"]["slots"]
    assert sorted(slots) == ["choice.k2", "choice.k4"], "only the 2- and 4-option shapes"
    for shape, body in slots.items():
        for label, stats in [("all", body["all"]), *body["per_family"].items()]:
            rows = _family_rows(scored, None if label == "all" else label, shape)
            probs, gold = rft.letter_distributions(rows)[shape]
            n, width = probs.shape
            assert stats["n"] == n
            assert stats["mean_entropy"] == float(_entropy(probs).mean())
            assert stats["entropy_floor_stated"] == DEFAULT_ENTROPY_FLOOR
            assert stats["max_class_share_stated"] == DEFAULT_MAX_CLASS_SHARE
            pred = np.argmax(probs, axis=1)
            assert stats["accuracy"]["n"] == int((pred == gold).sum())
            gold_counts = np.bincount(gold, minlength=width)
            assert stats["majority_class_rate"]["n"] == int(gold_counts.max())
            for c, cls in enumerate(stats["classes"]):
                g = gold_counts[c] / n
                p_hat = float((pred == c).mean())
                assert cls["gold"]["count"] == int(gold_counts[c])
                assert cls["predicted"]["share"] == pytest.approx(p_hat, abs=1e-15)
                mean_p = float(np.ascontiguousarray(probs[:, c]).mean())
                assert cls["mean_probability"]["value"] == pytest.approx(mean_p, abs=1e-12)
                dev = cls["predicted"]["deviation_sigma"]
                if gold_counts[c] in (0, n):
                    assert dev["state"] == "not_run" and "undefined" in dev["reason"]
                else:
                    sigma = np.sqrt(g * (1 - g) / n)
                    assert dev["value"] == pytest.approx((p_hat - g) / sigma, abs=1e-9)


def test_the_promotion_population_is_the_records_verbatim(gate_report_bin, tmp_path, fixture):
    _, _, verdicts, ledger = fixture
    _, report = _run(gate_report_bin, tmp_path, verdicts, ledger)
    assert report is not None
    population = load_promotion_decisions().population
    stated = report["promotion_population"]
    assert (stated["value"], stated["status"], stated["gap"]) == (
        population.value, population.status, population.gap)
    assert stated["record_sha256"] == population.record_sha256
    assert {d["gap"] for d in report["open_human_decisions"]} == {
        d.gap for d in load_promotion_decisions().open()}


def test_numbers_that_disagree_with_their_eval_row_are_refused(gate_report_bin, tmp_path):
    scored, second, perms = _scoring_run(np.random.default_rng(3))
    ledger = Ledger(tmp_path / "eval.jsonl")
    eval_row_id, _ = _record_eval_row(ledger, scored, second, perms, tamper="permutation_count")
    verdicts = _write_verdicts(tmp_path / "v.jsonl", scored, eval_row_id)
    proc, report = _run(gate_report_bin, tmp_path, verdicts, ledger.path)
    assert proc.returncode == 2 and report is None
    assert "gates.permutation_consistency" in proc.stderr
    assert "not that row's verdicts" in proc.stderr


def test_verdicts_whose_eval_row_is_absent_are_refused(gate_report_bin, tmp_path, fixture):
    _, _, verdicts, _ = fixture
    empty = tmp_path / "other.jsonl"
    empty.write_text("", encoding="utf-8")
    proc, report = _run(gate_report_bin, tmp_path, verdicts, empty)
    assert proc.returncode == 2 and report is None
    assert "appears 0 time(s)" in proc.stderr


def test_a_row_that_records_nothing_comparable_is_refused(gate_report_bin, tmp_path):
    """A cross-check that could not run must not read as one that passed."""
    scored, second, perms = _scoring_run(np.random.default_rng(4))
    ledger = Ledger(tmp_path / "eval.jsonl")
    eval_row_id, _ = _record_eval_row(ledger, scored, second, perms, tamper="nothing_comparable")
    verdicts = _write_verdicts(tmp_path / "v.jsonl", scored, eval_row_id)
    proc, report = _run(gate_report_bin, tmp_path, verdicts, ledger.path)
    assert proc.returncode == 2 and report is None
    assert "no number on eval row" in proc.stderr


def test_a_verdict_without_a_family_leaves_no_per_family_number(gate_report_bin, tmp_path):
    scored, second, perms = _scoring_run(np.random.default_rng(5))
    ledger = Ledger(tmp_path / "eval.jsonl")
    eval_row_id, _ = _record_eval_row(ledger, scored, second, perms)
    verdicts = _write_verdicts(tmp_path / "v.jsonl", scored, eval_row_id, drop_family=True)
    # The eval row's per-family metrics were computed WITH the family; the reporter, missing
    # one, must report no per-family number rather than a different one. The pooled numbers
    # still cross-check.
    proc, report = _run(gate_report_bin, tmp_path, verdicts, ledger.path)
    assert proc.returncode == 0, proc.stderr
    assert report is not None
    per_family = report["eval_rows"][0]["g1"]["gates"]["permutation_consistency"]["per_family"]
    assert list(per_family) == ["family"]
    assert per_family["family"]["state"] == "not_run"
    assert "carry no family_id" in per_family["family"]["reason"]


def test_no_report_value_says_passed(gate_report_bin, tmp_path, fixture):
    """Only the eval row's own recorded verdicts, quoted verbatim, carry `passed`."""
    _, _, verdicts, ledger = fixture
    _, report = _run(gate_report_bin, tmp_path, verdicts, ledger)
    assert report is not None

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            if "passed" in node:
                assert "recorded_on_the_eval_row" in path, path
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}/{i}")

    walk(report, "")
