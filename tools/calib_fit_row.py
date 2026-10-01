"""Record a ``qd-calib-fit`` run as a ledger row of its own.

The fit is Rust (``crates/qd-runtime/src/bin/qd_calib_fit.rs`` over
``qd_runtime::calibration_fit``): per ``{kind}:{rows}`` entry it fits a temperature, a
``noul_margin`` and a conformal cutoff from one model's ``--verdicts-out`` files and writes
the runtime's own ``CalibrationTable`` bytes. The ledger is Python
(``qd_train.ledger.RunRecorder`` owns the chain), and a number in a report cites a row
(rule 5). This tool is the binding between the two: it runs the binary inside the recorder's
block and writes the report's numbers as metrics. It computes nothing itself.

**Its own family, never a supplement.** The row is ``quick`` and ``run_kind`` is
``calibration``: the table is written and reported, not installed -- which table ships is the
human's call (rule 2). The recipe names the fitted eval rows as ``fitted_eval_rows``, not
``eval_row_id``: that key is the promotion join's supplement key, and one quick row joined to
an eval row's family refuses the whole family.

**It must be those eval rows' verdicts.** Every verdict file's ``eval_row_id`` must resolve to
exactly one ``eval`` row, and the report's counts must equal the eval row's own: lines against
``val_rows_decoded``, and per kind the rows and the verdicts' ``correct`` against
``val_top1.<kind>``'s ``n_total`` and ``n``. A file paired with the wrong row would fit a model
nobody scored.

**One model.** A table is bound to one model's weights, and ``Protocol`` carries one seed. The
fitted eval rows must agree on data snapshot, tokenizer, backbone and seed; three seeds are
three fits.

**The table's sha256 is a metric, not a recipe field.** The recipe is fixed when the recorder
opens and is hashed into ``Protocol.recipe_hash`` before the binary runs; the table is what the
binary produces. The sha256 recorded is of the written bytes, re-read here, and equals
``CalibrationTable::hash()`` -- what ``expect.calibration_hash`` would pin.

RUN
---
    cargo build --release -p qd-runtime --bin qd-calib-fit
    python tools/calib_fit_row.py --bin target/release/qd-calib-fit \\
      --verdicts VERDICTS.jsonl --population all --name NAME \\
      --eval-ledger ledger/<eval>.jsonl --ledger ledger/<out>.jsonl --out-dir OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from ft_linear_control import Refused, find_row  # noqa: E402

from qd_train.ledger import Environment, Ledger, LedgerRow, Protocol, RunRecorder  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

#: Bounds the binary's run. J1-scale verdicts fit in well under a second; minutes is a fault.
FIT_TIMEOUT_S: Final[int] = 1800
#: More verdict files than this is not one model's fit (the binary's own cap is the same).
MAX_VERDICT_FILES: Final[int] = 64
QUICK_REASON: Final[str] = (
    "a calibration fit on one model's verdicts: the table is written and reported, not "
    "installed -- which table ships is the human's (rule 2)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_eval_row_id(path: Path) -> str:
    """The eval row a verdict file belongs to: its first line's. The binary refuses a file
    whose lines name more than one."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                eval_row_id = json.loads(line).get("eval_row_id")
                if not isinstance(eval_row_id, str) or not eval_row_id:
                    raise Refused(f"{path}: the first verdict line names no eval_row_id")
                return eval_row_id
    raise Refused(f"{path}: no verdict lines")


def one_protocol(rows: list[LedgerRow]) -> LedgerRow:
    """The eval rows must be one model's: same data, tokenizer, backbone and seed."""
    first = rows[0]
    for row in rows[1:]:
        for field in ("data_snapshot_hash", "tokenizer_hash", "backbone_commit", "seed"):
            if getattr(row.protocol, field) != getattr(first.protocol, field):
                raise Refused(
                    f"eval rows {first.row_id} and {row.row_id} differ in {field}: a table is "
                    "one model's, so fit each model's verdicts separately"
                )
    return first


def run_fit(binary: Path, args: argparse.Namespace) -> dict[str, object]:
    """Run the binary once and return its report. A non-zero exit is a refusal, verbatim."""
    cmd = [str(binary)]
    for path in args.verdicts:
        cmd += ["--verdicts", str(path)]
    cmd += ["--population", args.population, "--name", args.name,
            "--alpha", repr(args.alpha), "--target-precision", repr(args.target_precision),
            "--out-dir", str(args.out_dir)]
    if args.split_key is not None:
        cmd += ["--split-key", args.split_key]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT_S,
                          check=False)
    if done.returncode != 0:
        raise Refused(f"{binary} exited {done.returncode}: {done.stderr.strip()}")
    report = json.loads((args.out_dir / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise Refused(f"{args.out_dir / 'report.json'} is not a JSON object")
    return report


def check_counts(report: dict[str, object], eval_rows: dict[str, LedgerRow]) -> None:
    """The report counts each eval row's verdicts the way that row's own metrics do."""
    stated = report["eval_rows"]
    if not isinstance(stated, dict) or set(stated) != set(eval_rows):
        raise Refused(f"the report fits eval rows {sorted(stated)}, the files name "
                      f"{sorted(eval_rows)}")
    for eval_row_id, body in stated.items():
        row = eval_rows[eval_row_id]
        decoded = row.metrics.get("val_rows_decoded")
        if not isinstance(decoded, Ran) or decoded.value is None:
            raise Refused(f"eval row {eval_row_id} has no val_rows_decoded count to match")
        if int(body["lines"]) != int(decoded.value):
            raise Refused(f"the verdicts of {eval_row_id} hold {body['lines']} lines; the eval "
                          f"row decoded {decoded.value}: not that row's run")
        for kind, counts in sorted(body["by_kind"].items()):
            top1 = row.metrics.get(f"val_top1.{kind}")
            if not isinstance(top1, Ran) or top1.n is None:
                raise Refused(f"eval row {eval_row_id} has no val_top1.{kind} count to match")
            if (int(counts["n"]), int(counts["verdict_correct"])) != (top1.n_total, top1.n):
                raise Refused(
                    f"{eval_row_id} {kind}: the verdicts hold {counts['verdict_correct']}/"
                    f"{counts['n']} correct, the eval row says {top1.n}/{top1.n_total}"
                )


def ece_metric(state: dict[str, object], *, when: str) -> TriState:
    """The binary's ECE state, as ``ece_gate`` would have returned it. Report-only here."""
    if state["state"] != "ran":
        return NotRun(reason=str(state["reason"]))
    value = float(state["value"])  # type: ignore[arg-type]
    n = int(state["n"])  # type: ignore[call-overload]
    return Ran(
        passed=bool(state["passed"]), value=value, n=n, n_total=n,
        detail=(f"ECE {value:.4f} against a {float(state['threshold']):.4f} bar, "  # type: ignore[arg-type]
                f"{state['bins']} bins of which {state['populated_bins']} carried mass; "
                f"{when}; report-only, not a gate"),
    )


def _ratio(value: object, num: int, den: int, detail: str) -> TriState:
    if value is None:
        return NotRun(reason=f"no row to divide by: {detail}")
    return Ran(passed=True, value=float(value), n=num, n_total=den,  # type: ignore[arg-type]
               detail=detail)


def fit_metrics(prefix: str, fit: dict[str, object], n_entry: int) -> dict[str, TriState]:
    if fit.get("fitted") is False:
        reason = str(fit["reason"])
        return {f"{prefix}.{k}": NotRun(reason=reason)
                for k in ("temperature", "noul_margin", "conformal_quantile")}
    n = int(fit["n"])  # type: ignore[call-overload]
    optimum = ("at_lower_bound" if fit["temperature_at_lower_bound"] else
               "at_upper_bound" if fit["temperature_at_upper_bound"] else "interior")
    out: dict[str, TriState] = {}
    for key, detail in (
        ("temperature", "ternary search on log T over [0.05, 20], 200 iterations"),
        ("noul_margin", "lowest calibrated margin whose retained set reaches the target"),
        ("conformal_quantile", "1 - q_hat: the probability cutoff the table holds"),
        ("nonconformity_quantile", "q_hat, nonconformity scale; not written into the table"),
    ):
        out[f"{prefix}.{key}"] = Ran(passed=True, value=float(fit[key]),  # type: ignore[arg-type]
                                     n=n, n_total=n_entry, detail=f"fitted on {n} rows; {detail}")
    out[f"{prefix}.temperature_optimum"] = Ran(
        passed=True, value=optimum,
        detail="interior, or the NLL at a search bound is no worse than at the fit")
    return out


def report_metrics(report: dict[str, object]) -> dict[str, TriState]:
    """Every number the report states, as a metric."""
    out: dict[str, TriState] = {}
    counts = report["population_counts"]
    letters = int(counts["letter_rows"])  # type: ignore[index]
    spans = int(counts["span_rows_not_fitted"])  # type: ignore[index]
    total = letters + spans
    out["calib_fit.population.letter_rows"] = Ran(
        passed=True, value=letters, n=letters, n_total=total,
        detail=f"population {report['population']}; letter rows fitted, of every verdict line")
    out["calib_fit.population.span_rows_not_fitted"] = Ran(
        passed=True, value=spans, n=spans, n_total=total,
        detail="span verdicts carry no distribution: GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS")
    out["calib_fit.span"] = NotRun(reason=str(report["span"]["reason"]))  # type: ignore[index]
    entries = report["entries"]
    if not isinstance(entries, dict) or not entries:
        raise Refused("the report states no fitted entry")
    for key, entry in sorted(entries.items()):
        kind, _ = key.split(":")
        prefix = f"calib_fit.{kind}.k{entry['options']}"
        n_entry = int(entry["n"])
        if "fit" in entry:
            out.update(fit_metrics(prefix, entry["fit"], n_entry))
        else:
            for fold, body in sorted(entry["folds"].items()):
                out.update(fit_metrics(f"{prefix}.fold_{fold}", body, n_entry))
        out[f"{prefix}.ece_before"] = ece_metric(entry["ece_before"],
                                                 when="T=1, every row, before calibration")
        scored = entry["scored"]
        how = str(scored["how"]).replace("_", " ")
        n_scored, n_not = int(scored["n_scored"]), int(scored["n_not_scored"])
        out[f"{prefix}.ece_after"] = ece_metric(scored["ece_after"],
                                                when=f"at the fitted parameters, {how}")
        out[f"{prefix}.scored"] = Ran(
            passed=True, value=n_scored, n=n_scored, n_total=n_scored + n_not,
            detail=f"rows scored {how}; the rest had no parameters from their other fold")
        out[f"{prefix}.conformal_coverage"] = _ratio(
            scored["conformal_coverage"], int(scored["covered"]), n_scored,
            f"gold in the served conformal set, {how}")
        out[f"{prefix}.retained_precision"] = _ratio(
            scored["retained_precision"], int(scored["retained_correct"]),
            int(scored["retained"]), f"accuracy over rows the noul_margin keeps, {how}")
        out[f"{prefix}.abstained_by_margin"] = Ran(
            passed=True, value=int(scored["abstained_by_margin"]),
            n=int(scored["abstained_by_margin"]), n_total=n_scored,
            detail=f"rows below the fitted noul_margin, {how}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", type=Path, required=True, help="the qd-calib-fit binary")
    parser.add_argument("--verdicts", type=Path, action="append", required=True,
                        help="a --verdicts-out file; repeat for more of the same model")
    parser.add_argument("--population", choices=("all", "two-fold"), required=True)
    parser.add_argument("--split-key", default=None)
    parser.add_argument("--name", required=True, help="the table's name; hashed with it")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--eval-ledger", type=Path, action="append", required=True,
                        help="a ledger holding the eval row(s) the verdicts are; repeatable")
    parser.add_argument("--ledger", type=Path, required=True, help="the ledger to append to")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists; refusing to overwrite it")
    if not args.bin.is_file():
        raise SystemExit(f"--bin {args.bin} is not a file")
    if len(args.verdicts) > MAX_VERDICT_FILES:
        raise SystemExit(f"{len(args.verdicts)} --verdicts files; at most {MAX_VERDICT_FILES}")

    ledger_rows = [r for path in args.eval_ledger for r in Ledger(path).rows()]
    eval_rows: dict[str, LedgerRow] = {}
    for path in args.verdicts:
        row = find_row(ledger_rows, first_eval_row_id(path))
        if row.run_kind != "eval":
            raise Refused(f"row {row.row_id} is run_kind {row.run_kind!r}, not 'eval'")
        eval_rows[row.row_id] = row
    model = one_protocol(list(eval_rows.values()))
    recipe: dict[str, object] = {
        "tool": "tools/calib_fit_row.py", "tag": "calib-fit", "bin": "qd-calib-fit",
        # Not ``eval_row_id``: that key makes a row a supplement of the eval row's family.
        "fitted_eval_rows": sorted(eval_rows), "population": args.population,
        "split_key": args.split_key, "alpha": args.alpha,
        "target_precision": args.target_precision, "name": args.name,
        "verdicts_sha256": [sha256_file(p) for p in args.verdicts],
    }
    protocol = Protocol(
        data_snapshot_hash=model.protocol.data_snapshot_hash,
        tokenizer_hash=model.protocol.tokenizer_hash,
        backbone_commit=model.protocol.backbone_commit,
        recipe_hash=hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        seed=model.protocol.seed,
    )
    with RunRecorder(
        Ledger(args.ledger), entry_point=Path(__file__), protocol=protocol,
        run_kind="calibration", repo=REPO, env=Environment.detect(device="cpu"),
        wall_clock_s=None, cost=None, recipe=recipe, quick=True, quick_reason=QUICK_REASON,
        notes=(f"tools/calib_fit_row.py: qd-calib-fit over eval row(s) "
               f"{', '.join(sorted(eval_rows))}'s verdicts, population {args.population}"),
    ) as recorder:
        started = time.monotonic()
        report = run_fit(args.bin, args)
        recorder.measured(time.monotonic() - started)
        check_counts(report, eval_rows)
        recorder.metric("fitted_eval_row_ids", Ran(
            passed=True, value=",".join(sorted(eval_rows)),
            detail="the eval rows these verdicts are"))
        for table in report["tables"]:  # type: ignore[union-attr]
            written = sha256_file(args.out_dir / str(table["file"]))
            if written != table["sha256"]:
                raise Refused(f"{table['file']} hashes to {written}, the report says "
                              f"{table['sha256']}")
            suffix = "" if table["fitted_on"] == "all" else (
                "." + str(table["fitted_on"]).replace("-", "_"))
            recorder.metric(f"calib_fit.table_sha256{suffix}", Ran(
                passed=True, value=written,
                detail=(f"{table['file']} ({table['name']}): sha256 of the written bytes, "
                        f"which is CalibrationTable::hash(); entries "
                        f"{', '.join(table['entries'])}; span null, not fitted")))
        for name, state in report_metrics(report).items():
            recorder.metric(name, state)
    written_row = recorder.row
    print(f"wrote row {written_row.row_id if written_row else '?'} to {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
