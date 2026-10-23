"""Record each arm of ``campaign/span-refit-mac-preregistered.json`` as a ledger row (rule 5).

The pre-registered scorer (``tools/span_refit_score.py``) decodes every
arm from the frozen caches and writes one results JSON; it only drafts ledger rows because
``ledger/`` was outside its lane. This binding is the recorder: for each arm it runs the scorer
once (``--arms <arm>``, so the row's wall clock is that pass's own) inside a
``qd_train.ledger.RunRecorder`` block and writes the results JSON's numbers as metrics. It
computes nothing itself; the scorer is the tracked ``tools/span_refit_score.py``, run as a
subprocess and pinned by sha256 with its three siblings (``tools/span_refit_{common,extract,
fit}.py``), so every pinned hash resolves to a committed file.

**Its own family, never a supplement.** Rows are ``run_kind="eval"``, ``quick`` (the
pre-registration's status: one tower, one fit seed, a train subsample; rule 8) and REPORT-ONLY.
The recipe names the eval row whose population the gate reproduces as ``reported_eval_rows``,
never ``eval_row_id`` (the promotion join's supplement key). The protocol is copied from that
row; the recipe hash is this row's own. Extraction and fit jobs are named in ``notes`` by their
``mac_heavy`` labels and are not priced into the row.

**One gate on every row**, ``span_refit.validity_gate``: arm (a)'s fp32 decode within +/-15 of
5,596 (row 6af73bef). It is pre-registered; recording its outcome is allowed, moving it is not.
If it fails, only the arm-(a) row is written (gate failed) and the tool stops: every arm is
void (the pre-registration's arms.a.role). Every other metric is ``passed=True``: a reading is
not a verdict (rule 2).

Refused: a pre-registration whose sha256 is not the one recorded after amendment 2, a
population pin that is not the pre-registered one, a results JSON that is not of the arm asked
for, or a head file whose sha256 differs from the one the scorer reports.

RUN
---
    python tools/span_refit_rows.py \\
      --eval-ledger ledger/h100x2-v5avg-2026-10-05.jsonl --results-dir RESULTS \\
      [--head-c DIR --head-e DIR --seed1-val-cache DIR] [--arms a b c d e]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from calib_fit_row import sha256_file  # noqa: E402
from ft_linear_control import Refused, find_row  # noqa: E402

from qd_train.ledger import (  # noqa: E402
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.span_null import (  # noqa: E402
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
)
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

PREREG_SHA256: Final[str] = "2252f630eeda39f907d03e339b6e2eea31f61fa932184281828d502f760c2cde"
POPULATION_PIN: Final[str] = "bdd571b675d83b92b2180d900cee79c9a89553f240c07fac421bcff47eb4e626"
EVAL_ROW_PREFIX: Final[str] = "6af73bef"
GATE_TARGET: Final[int] = 5596
GATE_TOLERANCE: Final[int] = 15
GATE_NAME: Final[str] = "span_refit.validity_gate"
ARMS: Final[tuple[str, ...]] = ("a", "b", "c", "d", "e")
CELLS: Final[tuple[str, ...]] = (
    "qa.answer_span/pointing", "qa.answer_span/abstain",
    "code.defect_class/pointing", "code.defect_class/abstain",
)
SCRIPTS: Final[tuple[str, ...]] = (
    "span_refit_common.py", "span_refit_extract.py", "span_refit_fit.py", "span_refit_score.py",
)
SCORE_TIMEOUT_S: Final[int] = 3600
REPORT_ONLY: Final[str] = "report-only, not a verdict (rule 2)"
GATE_EXIT: Final[int] = 3

Json = dict[str, object]


def _obj(node: object, where: str) -> Json:
    if not isinstance(node, dict):
        raise Refused(f"results {where}: expected an object, got {type(node).__name__}")
    return {str(k): v for k, v in node.items()}


def _int(node: object, where: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        raise Refused(f"results {where}: expected an integer, got {node!r}")
    return node


def _count(node: object, where: str, detail: str) -> TriState:
    state = _obj(node, where)
    return Ran(passed=True, value=_int(state.get("n"), f"{where}.n"),
               n=_int(state.get("n"), f"{where}.n"),
               n_total=_int(state.get("n_total"), f"{where}.n_total"),
               detail=f"{detail}; {REPORT_ONLY}")


def counts_key(arm: str) -> str:
    """The cache whose count the pre-registration reads for this arm: the fp32 live decode for
    the arms that have one as their gate or reference (a, d), the bf16 cache for the others."""
    return "fp32_live" if arm in ("a", "d") else "bf16_cache"


def check_results(result: Json, arm: str) -> Json:
    """The results JSON must be the pre-registered one, and carry the arm asked for."""
    prereg = _obj(result.get("prereg"), "prereg")
    if prereg.get("sha256") != PREREG_SHA256:
        raise Refused(f"the scorer ran under pre-registration {prereg.get('sha256')}, not "
                      f"{PREREG_SHA256} (amendment 2)")
    if result.get("population_pin") != POPULATION_PIN:
        raise Refused(f"population pin {result.get('population_pin')} is not {POPULATION_PIN}")
    arms = _obj(result.get("arms"), "arms")
    if arm not in arms:
        raise Refused(f"the results JSON has no arm ({arm})")
    block = _obj(arms[arm], f"arms.{arm}")
    if "not_run" in block or "void" in block or ("head" not in block and arm != "e"):
        raise Refused(f"arm ({arm}) did not run: {block.get('not_run') or block.get('void')}")
    return block


def gate_state(result: Json) -> TriState:
    block = _obj(_obj(result.get("arms"), "arms").get("a"), "arms.a")
    gate = _obj(block.get("gate"), "arms.a.gate")
    value = _int(gate.get("value_fp32_live"), "arms.a.gate.value_fp32_live")
    target, tol = _int(gate.get("target"), "gate.target"), _int(gate.get("tolerance"), "gate.tol")
    if (target, tol) != (GATE_TARGET, GATE_TOLERANCE):
        raise Refused(f"the results' gate is {target} +/- {tol}, not the pre-registered "
                      f"{GATE_TARGET} +/- {GATE_TOLERANCE}")
    return Ran(passed=abs(value - GATE_TARGET) <= GATE_TOLERANCE, value=value,
               detail=f"arm (a) fp32 decode {value} of 7,238 against {GATE_TARGET} +/- "
                      f"{GATE_TOLERANCE} (row {EVAL_ROW_PREFIX}); the scorer says "
                      f"passed={gate.get('passed')}")


def arm_metrics(result: Json, arm: str) -> dict[str, TriState]:
    """Every number the results JSON carries for one arm, as a metric."""
    block = check_results(result, arm)
    out: dict[str, TriState] = {}
    if arm == "e" and "head" not in block:
        raise Refused("arm (e) ran without --head-e: no result to record")
    key = counts_key(arm)
    counts = _obj(block.get(key), f"arms.{arm}.{key}")
    out["val_top1.span"] = _count(
        counts.get("overall"), f"arms.{arm}.{key}.overall",
        f"span top-1 over the 7,238 val span rows, {key}; reading: {block.get('reading')}")
    for cell in CELLS:
        out[f"val_top1.span.cell.{cell}"] = _count(
            counts.get(cell), f"arms.{arm}.{key}.{cell}", f"{cell}, {key}")
    head = _obj(block.get("head"), f"arms.{arm}.head")
    for name in ("cond_W_start", "cond_W_end"):
        if name in head:
            cond = head[name]
            if isinstance(cond, bool) or not isinstance(cond, int | float):
                raise Refused(f"results arms.{arm}.head.{name}: expected a number, got {cond!r}")
            out[name.lower()] = Ran(passed=True, value=float(cond),
                                    detail=f"cond(W) of the refit; {REPORT_ONLY}")
    out.update(span_null_metrics(result, block, arm))
    out.update(null_metrics(result))
    out.update(mcnemar_metrics(result, arm))
    return out


def span_null_metrics(result: Json, block: Json, arm: str) -> dict[str, TriState]:
    """The three metrics ``span_beats_null`` reads (amendment 2), as the scorer wrote them.

    Refused, never defaulted, when the results lack them: a results JSON from the scorer before
    amendment 2 is void for this gate and the arm is rerun, not re-read.
    """
    pointers = _obj(block.get("span_pointers"), f"arms.{arm}.span_pointers")
    null = _obj(_obj(result.get("nulls"), "nulls").get("pointing_added_line"),
                "nulls.pointing_added_line")
    counts: dict[str, tuple[int, int]] = {}
    for name in (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC):
        hit = _obj(pointers.get(name), f"arms.{arm}.span_pointers.{name}")
        counts[name] = (_int(hit.get("n"), f"{name}.n"),
                        _int(hit.get("n_total"), f"{name}.n_total"))
    counts[SPAN_NULL_METRIC] = (_int(null.get("n"), "nulls.pointing_added_line.n"),
                                _int(null.get("n_total"), "nulls.pointing_added_line.n_total"))
    totals = {n_total for _, n_total in counts.values()}
    if len(totals) != 1:
        raise Refused("the start and end top-1 and the null state different pointing-row counts: "
                      f"{ {k: v[1] for k, v in counts.items()} }")
    (total,) = totals
    if total == 0:
        # No pointing row: nothing was measured. real_ft_run.span_pointer_states records NotRun
        # here, and so does this: never a ran reading over zero rows.
        gone = NotRun(reason="the scored val set holds no pointing span row, so there is nothing "
                      "to measure a pointer or the random-+-line null over")
        return dict.fromkeys(counts, gone)
    for name, (n, n_total) in counts.items():
        if not 0 <= n <= n_total:
            raise Refused(f"results {name}: {n} hits of {n_total} pointing rows")
    value = null.get("value")
    if (isinstance(value, bool) or not isinstance(value, int | float)
            or not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise Refused(
            f"results nulls.pointing_added_line.value must be a rate in [0, 1]: {value!r}")
    out: dict[str, TriState] = {}
    for name, what in ((SPAN_START_TOP1_METRIC, "start"), (SPAN_END_TOP1_METRIC, "end")):
        n, n_total = counts[name]
        out[name] = Ran(
            passed=True, value=n / n_total, n=n, n_total=n_total,
            detail=(f"span {what} top-1 {n}/{n_total} over the pointing val rows, "
                    f"{pointers.get('cache')}; {REPORT_ONLY}"))
    out[SPAN_NULL_METRIC] = Ran(
        passed=True, value=float(value), n=counts[SPAN_NULL_METRIC][0], n_total=total,
        detail=f"random-+-line null on the same {total} pointing rows; {REPORT_ONLY}")
    return out


def null_metrics(result: Json) -> dict[str, TriState]:
    nulls = result.get("nulls")
    if nulls is None:
        return {}
    body = _obj(nulls, "nulls")
    uniform = _obj(body.get("uniform_line"), "nulls.uniform_line")
    out: dict[str, TriState] = {}
    for cell, value in _obj(uniform.get("by_cell"), "nulls.uniform_line.by_cell").items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise Refused(f"results nulls.uniform_line.by_cell.{cell}: not a number: {value!r}")
        out[f"null.uniform_line.{cell}"] = Ran(
            passed=True, value=float(value),
            detail=f"exact expected correct count; {REPORT_ONLY}")
    plus = _obj(body.get("random_plus_line"), "nulls.random_plus_line")
    expected = plus.get("expected_correct")
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        raise Refused(f"results nulls.random_plus_line.expected_correct: {expected!r}")
    out["null.random_plus_line.code.defect_class/pointing"] = Ran(
        passed=True, value=float(expected),
        detail=(f"exact expected correct count over the '+' lines of "
                f"{_int(plus.get('rows'), 'plus.rows')} rows; {REPORT_ONLY}"))
    absent = _obj(body.get("always_abstain"), "nulls.always_abstain")
    out["null.always_abstain.overall"] = Ran(
        passed=True, value=_int(absent.get("n"), "always_abstain.n"),
        detail=f"pre-registration states 1881, agrees={absent.get('agrees')}; {REPORT_ONLY}")
    return out


def mcnemar_metrics(result: Json, arm: str) -> dict[str, TriState]:
    """``mcnemar.b_vs_c`` on (c)'s row and ``mcnemar.e_vs_c.<cell>`` on (e)'s: both discordant
    counts and the exact p, as the scorer wrote them."""
    tests = _obj(result.get("mcnemar"), "mcnemar")
    wanted: list[tuple[str, str]] = []
    if arm == "c":
        wanted = [("mcnemar.b_vs_c", "b_vs_c_all_rows (prereg)")]
    elif arm == "e":
        wanted = [(f"mcnemar.e_vs_c.{cell}", tag) for tag in tests
                  if tag.startswith("e_vs_c_") for cell in CELLS
                  if tag.removeprefix("e_vs_c_").startswith(cell)]
    out: dict[str, TriState] = {}
    for name, tag in wanted:
        test = _obj(tests.get(tag), f"mcnemar.{tag}")
        p = test.get("p_exact_two_sided")
        if isinstance(p, bool) or not isinstance(p, int | float):
            raise Refused(f"results mcnemar.{tag}.p_exact_two_sided: {p!r}")
        out[name] = Ran(
            passed=True, value=float(p), n=_int(test.get("n"), f"{tag}.n"),
            n_total=_int(test.get("n"), f"{tag}.n"),
            detail=(f"exact two-sided McNemar p; first_only {test.get('first_only')}, "
                    f"second_only {test.get('second_only')}; {REPORT_ONLY}"))
    return out


def head_sha256_refusal(block: Json, heads: dict[str, Path]) -> None:
    """A refit head's file must hash to what the scorer says it scored."""
    head = _obj(block.get("head"), "head")
    source = head.get("source")
    claimed = head.get("file_sha256")
    if (isinstance(source, str) and Path(source).parent in heads.values()
            and sha256_file(Path(source)) != claimed):
        raise Refused(f"{source} hashes to {sha256_file(Path(source))}, the scorer scored "
                      f"{claimed}")


def run_scorer(args: argparse.Namespace, arm: str, out_json: Path) -> tuple[int, Json | None]:
    cmd = [sys.executable, str(args.scorer), "--arms", arm, "--out", str(out_json)]
    if args.val_cache is not None:
        cmd += ["--val-cache", str(args.val_cache)]
    if args.head_c is not None:
        cmd += ["--head-c", str(args.head_c)]
    if args.head_e is not None:
        cmd += ["--head-e", str(args.head_e)]
    if args.seed1_val_cache is not None:
        cmd += ["--seed1-val-cache", str(args.seed1_val_cache)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=SCORE_TIMEOUT_S,
                          check=False)
    if done.returncode not in (0, GATE_EXIT):
        raise Refused(f"{args.scorer} exited {done.returncode}: {done.stderr.strip()[-2000:]}")
    if not out_json.is_file():
        raise Refused(f"{args.scorer} wrote no {out_json}")
    return done.returncode, _obj(json.loads(out_json.read_text(encoding="utf-8")), str(out_json))


def record_arm(args: argparse.Namespace, eval_row_protocol: Protocol, eval_row_id: str,
               arm: str) -> tuple[bool, str]:
    """One arm, one row. Returns (gate passed, row id)."""
    out_json = args.results_dir / f"span-refit-score-{arm}.json"
    if out_json.exists():
        raise Refused(f"{out_json} already exists; a result is never overwritten")
    scripts = {name: sha256_file(args.scorer.parent / name) for name in SCRIPTS}
    recipe: dict[str, object] = {
        "tool": "tools/span_refit_rows.py", "tag": f"span-refit-{arm}", "device": "mps",
        "score_dtype": "fp32", "prereg_sha256": PREREG_SHA256, "population_pin": POPULATION_PIN,
        "script_sha256": scripts, "reported_eval_rows": [eval_row_id],
        "mac_heavy_labels": list(args.labels),
    }
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol = Protocol(
        data_snapshot_hash=eval_row_protocol.data_snapshot_hash,
        tokenizer_hash=eval_row_protocol.tokenizer_hash,
        backbone_commit=eval_row_protocol.backbone_commit,
        recipe_hash=recipe_hash, seed=eval_row_protocol.seed)
    with RunRecorder(
        Ledger(args.ledger), entry_point=Path(__file__), protocol=protocol, run_kind="eval",
        repo=REPO, env=Environment.detect(device="mps"), wall_clock_s=None, cost=None,
        recipe=recipe, quick=True,
        quick_reason=("prereg status: quick, report-only (rule 8): one tower, one fit seed, a "
                      "train subsample"),
        notes=(f"tools/span_refit_rows.py: arm ({arm}) of campaign/span-refit-mac-preregistered"
               f".json, scored by span_refit_score.py --arms {arm}; extraction and fit jobs: "
               f"{', '.join(args.labels) or 'not named'}; report-only"),
    ) as recorder:
        started = time.monotonic()
        code, result = run_scorer(args, arm, out_json)
        recorder.measured(time.monotonic() - started)
        assert result is not None
        check_results(result, "a")
        gate = gate_state(result)
        recorder.gate(GATE_NAME, gate)
        recorder.metric("results_json_sha256", Ran(
            passed=True, value=sha256_file(out_json),
            detail=f"the scorer's results file for this arm; {REPORT_ONLY}"))
        if code == GATE_EXIT or (isinstance(gate, Ran) and not gate.passed):
            if arm != "a":
                raise Refused("the validity gate failed: every arm is void; no row but (a)'s")
            for name, state in gate_metrics(result).items():
                recorder.metric(name, state)
            return False, recorder.row.row_id if recorder.row else "?"
        block = check_results(result, arm)
        head_sha256_refusal(block, {k: v for k, v in (("c", args.head_c), ("e", args.head_e))
                                    if v is not None})
        for name, state in arm_metrics(result, arm).items():
            recorder.metric(name, state)
    return True, recorder.row.row_id if recorder.row else "?"


def gate_metrics(result: Json) -> dict[str, TriState]:
    block = check_results(result, "a")
    counts = _obj(block.get("fp32_live"), "arms.a.fp32_live")
    return {"val_top1.span": _count(counts.get("overall"), "arms.a.fp32_live.overall",
                                    "arm (a), fp32 decode; the validity gate FAILED")}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scorer", type=Path, default=REPO / "tools" / "span_refit_score.py",
                   help="the tracked span_refit_score.py; its siblings are pinned beside it")
    p.add_argument("--eval-ledger", type=Path, action="append", required=True,
                   help="a ledger holding row 6af73bef; repeatable")
    p.add_argument("--ledger", type=Path,
                   default=REPO / "ledger" / "mac-span-refit-2026-10-06.jsonl")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--val-cache", type=Path, default=None)
    p.add_argument("--head-c", type=Path, default=None)
    p.add_argument("--head-e", type=Path, default=None)
    p.add_argument("--seed1-val-cache", type=Path, default=None)
    p.add_argument("--labels", nargs="*", default=[], help="the mac_heavy labels of the jobs "
                   "that made the caches and heads, named in the rows' notes")
    args = p.parse_args(argv)
    if not args.scorer.is_file():
        raise SystemExit(f"--scorer {args.scorer} is not a file")
    for arm, needed in (("c", args.head_c), ("e", args.head_e), ("d", args.seed1_val_cache)):
        if arm in args.arms and needed is None:
            raise SystemExit(f"arm ({arm}) needs its head or cache argument")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    eval_row = find_row([r for path in args.eval_ledger for r in Ledger(path).rows()],
                        EVAL_ROW_PREFIX)
    if eval_row.run_kind != "eval":
        raise Refused(f"row {eval_row.row_id} is run_kind {eval_row.run_kind!r}, not 'eval'")
    order = ["a"] + [a for a in args.arms if a != "a"]
    for arm in order:
        passed, row_id = record_arm(args, eval_row.protocol, eval_row.row_id, arm)
        print(f"arm ({arm}): row {row_id}")
        if not passed:
            print("the validity gate failed: every arm is void; stopping", file=sys.stderr)
            return GATE_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
