"""Record a ``qd-margin-probe`` run as a ledger row of its own.

The probe is Rust (``crates/qd-runtime/src/bin/qd_margin_probe.rs``): it computes each
case's margin with ``qd_runtime::calibration::calibrate``, fits a ``noul_margin`` on one val
half and says what that margin would abstain on. The ledger is Python
(``qd_train.ledger.RunRecorder`` owns the chain), and a number in a report cites a row
(rule 5). This tool is the binding between the two: it runs the binary inside the
recorder's block and writes the report's numbers as metrics. It computes nothing itself.

**Its own family, never a supplement.** The row is ``quick`` (one seed's verdicts, a fitted
margin that is reported and not installed -- selecting one is the human's, rule 2), and the
promotion join reads a row naming ``recipe.eval_row_id`` as a control of that eval row, where
one quick row refuses the whole family. So the recipe names the probed row as
``probed_eval_row``, and its own recipe hash puts the row in a family of its own.

**It must be the eval row's verdicts.** The report's ``eval_row_id`` must resolve to exactly
one ``eval`` row, and the probe's rule-only OOD abstentions must equal that row's
``ood_abstain`` value: a file paired with the wrong row would describe a model nobody scored.

RUN
---
    cargo build --release -p qd-runtime --bin qd-margin-probe
    python tools/margin_probe_row.py --probe target/release/qd-margin-probe \\
      --val VERDICTS.jsonl --suite SUITE_VERDICTS.jsonl --split-key KEY \\
      --eval-ledger ledger/<eval>.jsonl --ledger ledger/<out>.jsonl --report-out REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from ft_linear_control import Refused, find_row  # noqa: E402

from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

#: Bounds the binary's run. The probe reads two JSONL files; minutes means something is wrong.
PROBE_TIMEOUT_S: Final[int] = 600
#: The populations the report states quantiles for, in the order they are recorded.
QUANTILES: Final[tuple[str, ...]] = ("p00", "p10", "p25", "p50", "p75", "p90", "p100")
QUICK_REASON: Final[str] = (
    "a diagnostic on one seed's verdicts: the fitted noul_margin is reported, not installed "
    "(rule 2), and this is not the ood_abstain gate"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(probe: Path, *, val: Path, suite: Path, split_key: str,
              target_precision: float, report_out: Path) -> dict[str, object]:
    """Run the binary once and return its report. A non-zero exit is a refusal, verbatim."""
    done = subprocess.run(
        [str(probe), "--val", str(val), "--suite", str(suite), "--split-key", split_key,
         "--target-precision", repr(target_precision), "--out", str(report_out)],
        capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False,
    )
    if done.returncode != 0:
        raise Refused(f"{probe} exited {done.returncode}: {done.stderr.strip()}")
    report = json.loads(report_out.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise Refused(f"{report_out} is not a JSON object")
    return report


def _quantile_metrics(prefix: str, quantiles: dict[str, float], n: int) -> dict[str, Ran]:
    missing = [q for q in QUANTILES if q not in quantiles]
    if missing:
        raise Refused(f"{prefix}: the report carries no quantile(s) {missing}")
    return {
        f"{prefix}.margin_{q}": Ran(passed=True, value=float(quantiles[q]), n=n, n_total=n,
                                     detail="nearest-rank quantile of p_top - p_second at T=1")
        for q in QUANTILES
    }


def report_metrics(report: dict[str, object]) -> dict[str, Ran]:
    """Every number the report states, as a metric. Measurements, so ``passed`` is True."""
    out: dict[str, Ran] = {}
    margin = float(report["fitted_noul_margin"])  # type: ignore[arg-type]
    if not math.isfinite(margin):
        raise Refused(f"fitted_noul_margin {margin!r} is not finite")
    out["margin_probe.fitted_noul_margin"] = Ran(
        passed=True, value=margin,
        detail=(f"{report['fitted_on']}; target precision {report['target_precision']}, "
                f"temperature {report['temperature']}, split key {report['split_key']!r}"),
    )
    ood = report["ood"]
    if not isinstance(ood, dict) or not ood:
        raise Refused("the report states no OOD category")
    for category, body in sorted(ood.items()):
        n = int(body["n"])
        prefix = f"margin_probe.ood.{category}"
        for key in ("below_fitted_margin", "abstained_rule_only", "abstained_rule_or_margin"):
            out[f"{prefix}.{key}"] = Ran(passed=True, value=int(body[key]), n=n, n_total=n,
                                         detail=f"cases of {n} in the {category} category")
        out.update(_quantile_metrics(prefix, body["margin_quantiles"], n))
    total = report["ood_total"]
    for key in ("abstained_rule_only", "abstained_rule_or_margin"):
        out[f"margin_probe.ood_total.{key}"] = Ran(
            passed=True, value=int(total[key]), n=int(total["n"]), n_total=int(total["n"]),
            detail="over every OOD case")
    for half in ("val_half_a", "val_half_b"):
        body = report[half]
        population = body["population"]
        n = int(population["n"])
        prefix = f"margin_probe.{half}"
        out[f"{prefix}.below_fitted_margin"] = Ran(
            passed=True, value=int(population["below_fitted_margin"]), n=n, n_total=n,
            detail="val choice rows (gold-noul excluded) the fitted margin would abstain on")
        out[f"{prefix}.retained_precision"] = Ran(
            passed=True, value=float(body["retained_precision"]), n=n, n_total=n,
            detail="accuracy over the rows the fitted margin keeps")
        out.update(_quantile_metrics(prefix, population["margin_quantiles"], n))
    return out


def check_against_gate(report: dict[str, object], gate: object) -> None:
    """The probe's rule-only abstentions are the eval row's ``ood_abstain`` value, or the
    verdicts are some other run's."""
    if not isinstance(gate, Ran) or not isinstance(gate.value, (int, float)):
        raise Refused(f"the eval row's ood_abstain is {gate!r}: there is no count to match")
    total = report["ood_total"]
    n = int(total["n"])
    mine = int(total["abstained_rule_only"]) / n
    if not math.isclose(mine, float(gate.value), rel_tol=0.0, abs_tol=1e-12):
        raise Refused(
            f"the suite verdicts abstain on {total['abstained_rule_only']} of {n} "
            f"({mine:.6f}) and the eval row's ood_abstain is {gate.value}: not that row's run"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, required=True, help="the qd-margin-probe binary")
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--split-key", required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--eval-ledger", type=Path, required=True,
                        help="the ledger holding the eval row the verdicts are")
    parser.add_argument("--ledger", type=Path, required=True, help="the ledger to append to")
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report_out.exists():
        raise SystemExit(f"{args.report_out} already exists; refusing to overwrite it")
    if not args.probe.is_file():
        raise SystemExit(f"--probe {args.probe} is not a file")

    with args.val.open(encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    row = find_row(Ledger(args.eval_ledger).rows(), str(first["eval_row_id"]))
    if row.run_kind != "eval":
        raise Refused(f"row {row.row_id} is run_kind {row.run_kind!r}, not 'eval'")
    recipe: dict[str, object] = {
        "tool": "tools/margin_probe_row.py", "tag": "margin-probe", "probe": "qd-margin-probe",
        # Not ``eval_row_id``: that key makes a row a supplement of the eval row's family.
        "probed_eval_row": row.row_id, "split_key": args.split_key,
        "target_precision": args.target_precision, "temperature": 1.0,
        "val_sha256": sha256_file(args.val), "suite_sha256": sha256_file(args.suite),
    }
    protocol = Protocol(
        data_snapshot_hash=row.protocol.data_snapshot_hash,
        tokenizer_hash=row.protocol.tokenizer_hash,
        backbone_commit=row.protocol.backbone_commit,
        recipe_hash=hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        seed=row.protocol.seed,
    )
    with RunRecorder(
        Ledger(args.ledger), entry_point=Path(__file__), protocol=protocol,
        run_kind="calibration", repo=REPO, env=Environment.detect(device="cpu"),
        wall_clock_s=None, cost=None, recipe=recipe, quick=True, quick_reason=QUICK_REASON,
        notes=f"tools/margin_probe_row.py: qd-margin-probe over eval row {row.row_id}'s verdicts",
    ) as recorder:
        started = time.monotonic()
        report = run_probe(args.probe, val=args.val, suite=args.suite,
                           split_key=args.split_key, target_precision=args.target_precision,
                           report_out=args.report_out)
        recorder.measured(time.monotonic() - started)
        if report.get("eval_row_id") != row.row_id:
            raise Refused(f"the report is eval row {report.get('eval_row_id')!r}'s, "
                          f"not {row.row_id}")
        check_against_gate(report, row.gates.get("ood_abstain"))
        recorder.metric("probed_eval_row_id", Ran(passed=True, value=row.row_id,
                                                  detail="the eval row these verdicts are"))
        for name, state in report_metrics(report).items():
            recorder.metric(name, state)
    written = recorder.row
    print(f"wrote row {written.row_id if written else '?'} to {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
