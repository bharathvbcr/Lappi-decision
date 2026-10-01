"""Record a ``qd-gate-report`` run as a ledger row of its own (rule 5).

The report is Rust (``crates/qd-runtime/src/bin/qd_gate_report.rs``): every gate and control
per val family beside its pooled value (Fable G1), ECE pooled across slot shapes and per
language with a ``none`` bucket (G2), and the two- and four-option slot diagnostics (G3), re-read
on CPU from one eval row's ``--verdicts-out`` file and cross-checked against that row. The ledger
is Python, and a number in a report cites a row. This binding runs the binary inside the
recorder's block and writes the report's numbers as metrics. It computes nothing itself.

**Its own family, never a supplement.** The row is ``quick`` and REPORT-ONLY: it measures
nothing new and promotes nothing. ``run_kind`` is ``eval``; the recipe names the re-read row as
``reported_eval_rows``, not ``eval_row_id`` -- that key is the promotion join's supplement key,
and a quick row joined to an eval row's family would refuse it. The protocol copies the eval
row's data, tokenizer, backbone and seed; the recipe hash is this report's own.

**Bound to what it read.** The recipe carries the verdict file's, every suite file's, the
decisions record's and the binary's sha256; the commit is the row's ``code_commit``. The pooled
numbers are the eval row's own and are not copied here: the binary refuses to report unless it
recomputed every one of them equal, which ``gate_report.cross_check`` records with both counts.
Every metric's ``passed`` is true: a per-family value is not a verdict (rule 2).

**No model ran.** The row's environment is the CPU's; the GDN path that produced the verdicts is
the eval row's record, not this one's.

RUN
---
    cargo build --release -p qd-runtime --bin qd-gate-report
    python tools/gate_report_row.py --bin target/release/qd-gate-report \\
      --verdicts VERDICTS.jsonl [--suite-verdicts SUITE.jsonl] \\
      --eval-ledger ledger/<eval>.jsonl --ledger ledger/<out>.jsonl --out-json REPORT.json
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

from calib_fit_row import first_eval_row_id, sha256_file  # noqa: E402
from ft_linear_control import Refused, find_row  # noqa: E402

from qd_train.ledger import (  # noqa: E402
    DEFAULT_DECISIONS_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

#: Bounds the binary's run. J4's 8 MB verdict files report in about a second.
REPORT_TIMEOUT_S: Final[int] = 600
#: The binary's own cap on input files (``MAX_FILES`` in qd_gate_report.rs).
MAX_SUITE_FILES: Final[int] = 16
QUICK_REASON: Final[str] = (
    "a report-only re-read of one eval row's verdicts on CPU: it measures nothing new and "
    "promotes nothing (Fable G, rule 2)"
)
REPORT_ONLY: Final[str] = "report-only, not a verdict"

Json = dict[str, object]


# -- narrowing: a report that is not the shape the binary writes is refused, never cast -------


def _obj(node: object, where: str) -> Json:
    if not isinstance(node, dict):
        raise Refused(f"report {where}: expected an object, got {type(node).__name__}")
    return {str(k): v for k, v in node.items()}


def _list(node: object, where: str) -> list[object]:
    if not isinstance(node, list):
        raise Refused(f"report {where}: expected a list, got {type(node).__name__}")
    return node


def _int(node: object, where: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        raise Refused(f"report {where}: expected an integer, got {node!r}")
    return node


def _float(node: object, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, int | float):
        raise Refused(f"report {where}: expected a number, got {node!r}")
    return float(node)


def _scalar(node: object, where: str) -> float | int | str | None:
    if node is None or isinstance(node, str):
        return node
    if isinstance(node, bool) or not isinstance(node, int | float):
        raise Refused(f"report {where}: expected a scalar value, got {node!r}")
    return node


def _is_tri(node: object) -> bool:
    return isinstance(node, dict) and "state" in node


# -- the binary --------------------------------------------------------------------------------


def run_report(args: argparse.Namespace) -> Json:
    """Run the binary once; return its JSON report. A non-zero exit is a refusal, verbatim."""
    cmd = [str(args.bin), "--verdicts", str(args.verdicts[0])]
    for path in args.suite_verdicts:
        cmd += ["--suite-verdicts", str(path)]
    for path in args.eval_ledger:
        cmd += ["--eval-ledger", str(path)]
    cmd += ["--decisions", str(args.decisions), "--out-json", str(args.out_json)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=REPORT_TIMEOUT_S,
                          check=False)
    if done.returncode != 0:
        raise Refused(f"{args.bin} exited {done.returncode}: {done.stderr.strip()}")
    return _obj(json.loads(args.out_json.read_text(encoding="utf-8")), str(args.out_json))


# -- the report's values as metrics ------------------------------------------------------------


def tri(node: object, where: str) -> TriState:
    """A report value as a metric: ``ran`` keeps its numbers, ``passed`` true (not a verdict)."""
    state = _obj(node, where)
    if state.get("state") == "not_run":
        return NotRun(reason=str(state.get("reason", "")))
    if state.get("state") != "ran":
        raise Refused(f"report {where}: a value with no state: {state}")
    n, n_total = state.get("n"), state.get("n_total")
    return Ran(
        passed=True, value=_scalar(state.get("value"), where),
        n=None if n is None else _int(n, f"{where}.n"),
        n_total=None if n_total is None else _int(n_total, f"{where}.n_total"),
        detail=f"{state.get('detail', '')}; {REPORT_ONLY}",
    )


def per_family_metrics(prefix: str, node: object) -> dict[str, TriState]:
    """``{prefix}.family.{family}[.{part}]`` for every family's value; ``{prefix}.family``
    alone when the verdicts carried no family to split by."""
    out: dict[str, TriState] = {}
    for family, value in _obj(node, f"{prefix}.per_family").items():
        name = f"{prefix}.family" if family == "family" else f"{prefix}.family.{family}"
        if _is_tri(value):
            out[name] = tri(value, name)
            continue
        for part, inner in _obj(value, name).items():
            out[f"{name}.{part}"] = tri(inner, f"{name}.{part}")
    return out


def slot_metrics(prefix: str, node: object) -> dict[str, TriState]:
    """G3's numbers for one slot shape and population: entropy, accuracy, majority rate, and
    each decode row's predicted and mean-probability deviation from the gold marginal."""
    if _is_tri(node):
        return {prefix: tri(node, prefix)}
    stats = _obj(node, prefix)
    n = _int(stats.get("n"), f"{prefix}.n")
    out: dict[str, TriState] = {
        f"{prefix}.mean_entropy": Ran(
            passed=True, value=_float(stats.get("mean_entropy"), f"{prefix}.mean_entropy"),
            n=n, n_total=n,
            detail=(f"mean predictive entropy, nats; the control's floor "
                    f"{stats.get('entropy_floor_stated')} is stated, not applied; "
                    f"{REPORT_ONLY}")),
        f"{prefix}.top_predicted_share": Ran(
            passed=True,
            value=_float(stats.get("top_predicted_share"), f"{prefix}.top_predicted_share"),
            detail=(f"row {stats.get('top_predicted_row')}; the control's cap "
                    f"{stats.get('max_class_share_stated')} is stated, not applied; "
                    f"{REPORT_ONLY}")),
    }
    for key in ("accuracy", "decoded_correct", "majority_class_rate"):
        out[f"{prefix}.{key}"] = tri(stats.get(key), f"{prefix}.{key}")
    for entry in _list(stats.get("classes"), f"{prefix}.classes"):
        cls = _obj(entry, f"{prefix}.classes[]")
        label = str(cls.get("label")).replace(" ", "_")
        where = f"{prefix}.{label}"
        gold = _obj(cls.get("gold"), f"{where}.gold")
        predicted = _obj(cls.get("predicted"), f"{where}.predicted")
        mean_p = _obj(cls.get("mean_probability"), f"{where}.mean_probability")
        shares = (f"gold share {_float(gold.get('share'), where):.4f} "
                  f"({_int(gold.get('count'), where)}/{n}), predicted share "
                  f"{_float(predicted.get('share'), where):.4f} "
                  f"({_int(predicted.get('count'), where)}/{n}), mean probability "
                  f"{_float(mean_p.get('value'), where):.4f}")
        for kind, deviation in (("predicted_sigma", predicted.get("deviation_sigma")),
                                ("mean_probability_sigma", mean_p.get("deviation_sigma"))):
            state = tri(deviation, f"{where}.{kind}")
            if isinstance(state, Ran):
                state = Ran(passed=True, value=state.value, n=state.n, n_total=state.n_total,
                            detail=f"{shares}; {state.detail}")
            out[f"{where}.{kind}"] = state
    return out


def report_metrics(reported: Json, report: Json) -> dict[str, TriState]:
    """Every report-only number as a metric. The pooled values are the eval row's own."""
    out: dict[str, TriState] = {}
    check = _obj(reported.get("cross_check"), "cross_check")
    checked = _int(check.get("checked"), "cross_check.checked")
    skipped = [_obj(s, "cross_check.not_checked[]")
               for s in _list(check.get("not_checked"), "cross_check.not_checked")]
    out["gate_report.cross_check"] = Ran(
        passed=True, value=checked, n=checked, n_total=checked + len(skipped),
        detail=(f"{checked} numbers the eval row records were recomputed from its verdicts and "
                f"equal; {len(skipped)} were not compared: "
                + ("; ".join(f"{s.get('name')} ({s.get('why')})" for s in skipped) or "none")))
    population = _obj(report.get("promotion_population"), "promotion_population")
    out["promotion_population"] = Ran(
        passed=True, value=str(population.get("value")),
        detail=(f"status {population.get('status')} ({population.get('gap')}); families "
                f"{population.get('families')}; record {population.get('record')} sha256 "
                f"{population.get('record_sha256')}; as the human decisions record states it"))
    g1 = _obj(reported.get("g1"), "g1")
    for section in ("gates", "controls"):
        for name, body in _obj(g1.get(section), f"g1.{section}").items():
            out.update(per_family_metrics(
                f"g1.{name}", _obj(body, f"g1.{section}.{name}").get("per_family")))
    out.update(per_family_metrics(
        "g1.accuracy", _obj(g1.get("accuracy"), "g1.accuracy").get("per_family")))
    g2 = _obj(reported.get("g2"), "g2")
    out["g2.ece.pooled"] = tri(g2.get("pooled_all_letter_rows"), "g2.pooled_all_letter_rows")
    for language, state in _obj(g2.get("by_language"), "g2.by_language").items():
        out[f"g2.ece.lang.{language}"] = tri(state, f"g2.by_language.{language}")
    g3 = _obj(reported.get("g3"), "g3")
    for shape, node in _obj(g3.get("slots"), "g3.slots").items():
        body = _obj(node, f"g3.slots.{shape}")
        out.update(slot_metrics(f"g3.{shape}.all", body.get("all")))
        per_family = _obj(body.get("per_family"), f"g3.slots.{shape}.per_family")
        if "family" in per_family and _is_tri(per_family["family"]):
            out[f"g3.{shape}.family"] = tri(per_family["family"], f"g3.{shape}.family")
            continue
        for family, stats in per_family.items():
            out.update(slot_metrics(f"g3.{shape}.{family}", stats))
    return out


def the_reported_row(report: Json, eval_row_id: str, verdicts_sha256: str) -> Json:
    """The report's one eval row, which must be the row and the file this row hashed."""
    rows = _list(report.get("eval_rows"), "eval_rows")
    if len(rows) != 1:
        raise Refused(f"the report holds {len(rows)} eval rows, not one")
    row = _obj(rows[0], "eval_rows[0]")
    verdicts = _obj(row.get("verdicts"), "eval_rows[0].verdicts")
    if row.get("eval_row_id") != eval_row_id or verdicts.get("sha256") != verdicts_sha256:
        raise Refused(f"the report is of eval row {row.get('eval_row_id')} with verdicts "
                      f"{verdicts.get('sha256')}, not eval row {eval_row_id} with the verdict "
                      f"file this row hashed ({verdicts_sha256})")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", type=Path, required=True, help="the qd-gate-report binary")
    parser.add_argument("--verdicts", type=Path, action="append", required=True,
                        help="ONE eval row's --verdicts-out file")
    parser.add_argument("--suite-verdicts", type=Path, action="append", default=[],
                        help="a --suite-verdicts-out file; repeatable")
    parser.add_argument("--eval-ledger", type=Path, action="append", required=True,
                        help="a ledger holding the eval row; repeatable")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--ledger", type=Path, required=True, help="the ledger to append to")
    parser.add_argument("--out-json", type=Path, required=True,
                        help="where the binary writes its JSON report; refused if it exists")
    args = parser.parse_args(argv)
    if len(args.verdicts) != 1:
        raise SystemExit(f"{len(args.verdicts)} --verdicts files: one report row re-reads "
                         "exactly one --verdicts file, one eval row")
    if args.out_json.exists():
        raise SystemExit(f"{args.out_json} already exists; refusing to overwrite it")
    if not args.bin.is_file():
        raise SystemExit(f"--bin {args.bin} is not a file")
    if len(args.suite_verdicts) > MAX_SUITE_FILES:
        raise SystemExit(f"{len(args.suite_verdicts)} --suite-verdicts files; at most "
                         f"{MAX_SUITE_FILES}")

    eval_row = find_row([r for p in args.eval_ledger for r in Ledger(p).rows()],
                        first_eval_row_id(args.verdicts[0]))
    if eval_row.run_kind != "eval":
        raise Refused(f"row {eval_row.row_id} is run_kind {eval_row.run_kind!r}, not 'eval'")
    verdicts_sha256 = sha256_file(args.verdicts[0])
    recipe: dict[str, object] = {
        "tool": "tools/gate_report_row.py", "tag": "gate-report", "bin": "qd-gate-report",
        "bin_sha256": sha256_file(args.bin),
        # Not ``eval_row_id``: that key makes a row a supplement of the eval row's family.
        "reported_eval_rows": [eval_row.row_id],
        "verdicts_sha256": [verdicts_sha256],
        "suite_verdicts_sha256": [sha256_file(p) for p in args.suite_verdicts],
        "decisions_sha256": sha256_file(args.decisions),
    }
    protocol = Protocol(
        data_snapshot_hash=eval_row.protocol.data_snapshot_hash,
        tokenizer_hash=eval_row.protocol.tokenizer_hash,
        backbone_commit=eval_row.protocol.backbone_commit,
        recipe_hash=hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        seed=eval_row.protocol.seed,
    )
    with RunRecorder(
        Ledger(args.ledger), entry_point=Path(__file__), protocol=protocol, run_kind="eval",
        repo=REPO, env=Environment.detect(device="cpu"), wall_clock_s=None, cost=None,
        recipe=recipe, quick=True, quick_reason=QUICK_REASON,
        notes=(f"tools/gate_report_row.py: qd-gate-report over eval row {eval_row.row_id}'s "
               "verdicts; report-only"),
    ) as recorder:
        started = time.monotonic()
        report = run_report(args)
        recorder.measured(time.monotonic() - started)
        reported = the_reported_row(report, eval_row.row_id, verdicts_sha256)
        for name, state in report_metrics(reported, report).items():
            recorder.metric(name, state)
    written = recorder.row
    print(f"wrote row {written.row_id if written else '?'} to {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
