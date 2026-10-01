"""Interleaved A/B, min of N: ``tools/real_ft_run.py``'s CPU prelude, before against after.

What a GH200 job pays with the GPU idle, measured on the Mac on the inputs J7g and J4 ran with
(the phase-4 v3 shard set, the v3 defect corpus with its noul rows, the general record, the
J4 revision). Three stages, each timed from ``main()`` to the first thing that is not CPU
preparation, in one fresh process per run:

* ``score`` -- the ``--score-checkpoint --score-val --needle --ood`` scoring process, up to
  ``_score_checkpoint`` (the tower load). Its needle worker is not started: in the after arm
  the handoff it would be given IS written (that is new work this arm pays), then stubbed.
* ``worker`` -- the ``--needle`` worker, up to ``needle_predictions`` (its decode).
  ``_checkpoint_step`` (the checkpoint and tower load, the same work in both arms) is stubbed.
  Before: the worker's own full prelude. After: it reads the handoff the after arm's
  ``score`` run of the same round wrote.
* ``train`` -- ``--epoch --no-memorise --batch-tokens 16384 --optimizer master --score-val
  --needle --ood`` (J4), up to ``_train``. ``_probe_one`` (a device subprocess per bucket) is
  stubbed in both arms, so the probes are NOT measured.

Every run also hashes what its stage hands on -- the val plan, labels and letter ids, the
second pass, the needle and OOD suites, the eval widths, and for training the epoch plan, the
train labels and the inventory -- and the two arms must hash the same, stage by stage: the
evidence that the faster prelude scores and trains on exactly the inputs the slower one did.

Gated: ``QD_PREP_BENCH_BEFORE`` names a checkout of the code before the change (with the
gitignored ``data/pool`` files the defect rebuild reads), ``QD_PREP_BIN`` the qd-prep both
arms sign MinHash with. Optional: ``QD_PREP_BENCH_ROUNDS`` (default 3), ``QD_PREP_BENCH_OUT``,
``QD_PREP_BENCH_RECORD``, ``QD_PREP_BENCH_BACKBONE``, and ``QD_PREP_BENCH_RESULTS`` (a JSON
file the rounds are written to). Run as a module, this file is one arm's one stage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
BEFORE_ENV = "QD_PREP_BENCH_BEFORE"
#: The corpus revision J4 and J7g rebuilt their labels at.
REV = "b381a03cb12e816c380915b1983f4e13cd1c843c"
STAGES = ("score", "worker", "train")
#: Seconds one stage may take; the slowest measured was ~150 s.
STAGE_TIMEOUT_S = 1800.0


def _inputs() -> dict[str, Path]:
    home = Path.home()
    snapshots = home / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots"
    backbone = os.environ.get("QD_PREP_BENCH_BACKBONE") or (
        str(next(iter(sorted(snapshots.iterdir())))) if snapshots.is_dir() else ""
    )
    return {
        "out": Path(os.environ.get(
            "QD_PREP_BENCH_OUT", "/Users/bharath/qd-campaign/phase4-v3-2026-10-01")),
        "record": Path(os.environ.get(
            "QD_PREP_BENCH_RECORD",
            str(home / ".cache/qd-decision/general/fetch-record-2026-09-29.json"))),
        "backbone": Path(backbone),
    }


def _argv(checkout: Path, stage: str, scratch: Path) -> list[str]:
    inputs = _inputs()
    pool = checkout / "data" / "pool"
    common = [
        "--out", str(inputs["out"]), "--no-repo-history",
        "--defect-class", str(pool / "commitpackft-corpus-v3"),
        "--defect-download", str(pool / "commitpackft"),
        "--defect-noul", str(pool / "defect-noul-v1"),
        "--general-record", str(inputs["record"]), "--general-max-rows", "200000",
        "--replay-partition", "--rev", REV, "--real-backbone", str(inputs["backbone"]),
        "--ledger", str(scratch / "ledger.jsonl"),
        "--score-val", "--needle", "--ood", "--ood-general-record", str(inputs["record"]),
        "--devices", "cpu",
    ]
    if stage == "train":
        return [*common, "--epoch", "--no-memorise", "--seeds", "0", "1", "2",
                "--batch-tokens", "16384", "--optimizer", "master"]
    return [*common, "--score-checkpoint", str(scratch / "AVG.safetensors"),
            "--ft-ledger", str(scratch / "ft.jsonl"), "--seeds", "0", "1", "2",
            "--score-dtype", "fp32"]


# --- digests of what a stage hands on ---------------------------------------------------------


def _sha(*parts: object) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _batches(batches: Any) -> str:
    digest = hashlib.sha256()
    for b in batches:
        digest.update(f"{int(b.bucket)} {int(b.index)}".encode())
        for name in ("tokens", "lengths", "slot_kind", "target_index", "span_target",
                     "line_starts"):
            array = getattr(b, name)
            if array is None:
                digest.update(b"none")
                continue
            digest.update(f"{array.dtype.str}{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _labels(labels: Any) -> str:
    return _sha([dataclasses.astuple(x) for x in labels])


def _labels_for(labels_for: Any) -> str:
    return _sha({str(k): [dataclasses.astuple(x) for x in v] for k, v in labels_for.items()})


def _val(val: Any) -> dict[str, str]:
    return {"plan": _batches(val.plan), "labels": _labels(val.labels),
            "labels_for": _labels_for(val.labels_for),
            "letter_id": _sha(list(val.letter_id.items()))}


def _second(sp: Any) -> dict[str, str]:
    if sp is None:
        return {"none": "none"}
    return {"batches": _batches(sp.batches), "labels_for": _labels_for(sp.labels_for),
            "perms": _sha(sorted((list(k), list(v)) for k, v in sp.perms.items())),
            "not_run": str(sp.not_run)}


def _needle(suite: Any) -> dict[str, str]:
    return {"digest": str(suite.digest), "batches": _batches(suite.batches),
            "labels_for": _labels_for(suite.labels_for),
            "cases": _sha([dataclasses.asdict(c) for c in suite.cases]),
            "lengths": _sha(suite.token_lengths), "seed": str(suite.seed),
            "not_run": str(suite.not_run)}


def _ood(suite: Any) -> dict[str, Any]:
    return {"cases": _sha([dataclasses.asdict(c) for c in suite.cases]),
            "val": _val(suite.val) if suite.val is not None else "none",
            "second": _second(suite.second_pass), "record": suite.record_sha256,
            "seed": str(suite.seed), "not_run": str(suite.not_run)}


# --- one arm, one stage (run as a module) -------------------------------------------------------


class _Stop(Exception):
    pass


def run_stage(checkout: Path, stage: str, scratch: Path) -> dict[str, Any]:
    """Import ``checkout``'s ``real_ft_run`` and run ``stage`` to its stop point."""
    sys.path.insert(0, str(checkout / "python"))
    sys.path.insert(0, str(checkout / "tools"))
    os.chdir(checkout)
    started = time.perf_counter()
    rfr = importlib.import_module("real_ft_run")
    imported = time.perf_counter() - started
    if not Path(rfr.__file__).resolve().is_relative_to(checkout.resolve()):
        raise SystemExit(f"imported {rfr.__file__}, not {checkout}'s")
    handoff = scratch / "needle-handoff.npz"
    argv = _argv(checkout, stage, scratch)
    found: dict[str, Any] = {}

    if stage == "score":
        def needle_worker(raw_argv, suite, **kwargs):
            if hasattr(rfr, "write_needle_handoff"):
                if handoff.exists():
                    handoff.unlink()
                rfr.write_needle_handoff(handoff, suite, **kwargs)
            return rfr.NeedleDecoded({}, ())

        def score(args, *, val, second_pass, needle_suite, ood_suite, **kwargs):
            found.update(val=_val(val), second=_second(second_pass), needle=_needle(needle_suite),
                         ood=_ood(ood_suite),
                         widths=_sha(rfr.suite_widths(needle_suite, ood_suite)))
            raise _Stop

        rfr.run_needle_worker = needle_worker
        rfr._score_checkpoint = score
    elif stage == "worker":
        if hasattr(rfr, "write_needle_handoff"):
            if not handoff.is_file():
                raise SystemExit(f"{handoff}: run this arm's score stage first")
            argv += ["--needle-handoff", str(handoff)]
        argv += ["--needle-predictions-out", str(scratch / "worker-predictions.json")]

        def checkpoint_step(args, *, reader, val, device, eval_widths, suite_seed):
            found.update(plan_shapes=_sha([list(b.tokens.shape) for b in val.plan]),
                         widths=_sha(list(eval_widths)), suite_seed=str(suite_seed))
            return None, {}, {}, 0, {}

        def predictions(step, suite, letter_id):
            found.update(needle=_needle(suite), letter_id=_sha(list(letter_id.items())))
            raise _Stop

        rfr._checkpoint_step = checkpoint_step
        rfr.needle_predictions = predictions
    elif stage == "train":
        def probe(device, rows, width, hidden, heads):
            return {"device": device, "rows": rows, "width": width, "ok": True, "wall_s": 0.0}

        def train(**kwargs):
            local = sys._getframe(1).f_locals
            found.update(
                plan=_batches(kwargs["plan"]), widths=_sha(list(kwargs["eval_widths"])),
                alphabets=_sha(kwargs["alphabets"]),
                letter_id=_sha(list(local["letter_id"].items())),
                labels=_labels(local["labels"]),
                inventory=_sha(local["inventory"]), val=_val(local["val_set"]),
                second=_second(local["second_pass"]), needle=_needle(local["needle_suite"]),
                ood=_ood(local["ood_suite"]),
            )
            raise _Stop

        rfr._probe_one = probe
        rfr._train = train
    else:
        raise SystemExit(f"stage {stage!r} is not one of {STAGES}")
    started = time.perf_counter()
    try:
        rfr.main(argv)
    except _Stop:
        pass
    else:
        raise SystemExit(f"{stage}: main() returned before its stop point")
    return {"stage": stage, "checkout": str(checkout), "prelude_s": time.perf_counter() - started,
            "import_s": imported, "digest": _sha(found)}


def _run(checkout: Path, stage: str, scratch: Path) -> dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(checkout), stage, str(scratch)],
        capture_output=True, text=True, timeout=STAGE_TIMEOUT_S, check=False,
    )
    lines = [x for x in done.stdout.splitlines() if x.startswith("BENCH ")]
    if done.returncode != 0 or len(lines) != 1:
        raise AssertionError(
            f"{checkout} {stage} exited {done.returncode}:\n{done.stdout[-3000:]}\n"
            f"{done.stderr[-3000:]}"
        )
    return json.loads(lines[0][len("BENCH "):])


@pytest.mark.skipif(
    not (os.environ.get(BEFORE_ENV) and os.environ.get("QD_PREP_BIN")),
    reason=f"{BEFORE_ENV} (a pre-change checkout) and QD_PREP_BIN gate this ~30-minute benchmark",
)
def test_benchmark_prelude_before_against_after_interleaved_min_of_n(tmp_path):
    before = Path(os.environ[BEFORE_ENV]).resolve()
    after = REPO
    rounds = int(os.environ.get("QD_PREP_BENCH_ROUNDS", "3"))
    if rounds < 1:
        raise ValueError(f"QD_PREP_BENCH_ROUNDS must be at least 1, got {rounds}")
    results: list[dict[str, Any]] = []
    for r in range(rounds):
        # Alternate which arm goes first, so a load drift across a round favours neither.
        arms = (("before", before), ("after", after))
        for arm, checkout in arms if r % 2 == 0 else arms[::-1]:
            for stage in STAGES:
                got = _run(checkout, stage, tmp_path / f"{arm}-r{r}")
                got.update(arm=arm, round=r)
                results.append(got)
                print(f"round {r} {arm:6s} {stage:6s} {got['prelude_s']:8.2f} s "
                      f"(import {got['import_s']:.2f} s) digest {got['digest'][:16]}")
    summary: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        rows = {arm: [x for x in results if x["arm"] == arm and x["stage"] == stage]
                for arm in ("before", "after")}
        digests = {x["digest"] for arm in rows.values() for x in arm}
        summary[stage] = {
            arm: {"runs_s": [round(x["prelude_s"], 2) for x in xs],
                  "min_s": round(min(x["prelude_s"] for x in xs), 2)}
            for arm, xs in rows.items()
        }
        summary[stage]["same_digest"] = len(digests) == 1
        summary[stage]["digest"] = sorted(digests)
        print(f"{stage}: before min {summary[stage]['before']['min_s']} s, after min "
              f"{summary[stage]['after']['min_s']} s, digests equal: {len(digests) == 1}")
    out = os.environ.get("QD_PREP_BENCH_RESULTS")
    if out:
        Path(out).write_text(
            json.dumps({"rounds": rounds, "results": results, "summary": summary}, indent=1),
            encoding="utf-8",
        )
    for stage in STAGES:
        assert summary[stage]["same_digest"], f"{stage}: the arms hand on different inputs"


if __name__ == "__main__":
    result = run_stage(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
    print("BENCH " + json.dumps(result), flush=True)
