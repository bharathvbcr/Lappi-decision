"""Time ``tools/ft_linear_control.py``'s own path to the split on v5's data, on the Mac: MISS, then
HIT, as separate processes.

    QD_PREP_BIN=<abs> python AUDIT/v6-ctlcache-2026-10-04/measure_ctl_split.py \\
        --scratch DIR --label NAME [--cache DIR] -- <the control's split flags>

v5's eval rows and verdicts exist only on the H100 box, so the control cannot run end to end
here. This script runs the control's ``main`` in-process on v5's split flags, against a STAND-IN
eval row and verdict that this script writes into ``--scratch``, and stops it right after the
split. What runs:

* argv parsing and the cache flags' checks (``check_dir``);
* the verdicts' load and ``check_against_eval_row``;
* the holdout and exclusion-list pairings against the stand-in row's recipe;
* ``check_shard_set`` against v5's shard set;
* ``QD_PREP_BIN``;
* the split, rebuilt or read from the cache (``split_cache.cached_split_rows``);
* the control's own rule-3 check.

It stops at the next call, ``request_texts``, which is patched to raise ``StopAfterSplit``. No
fit runs and no control row is written.

The stand-in row is ``quick``, scores nothing, and lives only in ``--scratch``'s ledger. It names
v5's ``data_snapshot_hash``, read from the shard set's train.json (the v5 shard row c3bd0374
names the same one), and the sha256 of v5's exclusion list, as a real v5 eval row would. Its one
verdict is a span line, so no letter row needs pairing.

Instrumentation, and nothing else, wraps ``split_cache.cached_split_rows`` to time it and read its
metric. Each run prints one ``MEASURE`` line of JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))

import ft_linear_control as ftc  # noqa: E402
import split_cache  # noqa: E402

from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

SHARDS = Path("/Users/bharath/qd-campaign/phase4-v5r-2026-10-03")
SEED = 2


class StopAfterSplit(Exception):
    """Raised where the control would start building its documents: the split is done."""


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def stand_in(scratch: Path, split_flags: list[str]) -> tuple[Path, Path]:
    """The stand-in eval row's ledger and its one-line verdicts file, written once."""
    ledger, verdicts = scratch / "standin-ledger.jsonl", scratch / "standin-verdicts.jsonl"
    if ledger.exists() and verdicts.exists():
        return ledger, verdicts
    snapshot = split_cache.read_manifests(SHARDS).data_snapshot_hash
    exclusions = Path(_flag(split_flags, "--exclude-identity-keys"))
    protocol = Protocol(
        data_snapshot_hash=snapshot, tokenizer_hash="t" * 64, recipe_hash="r" * 64,
        backbone_commit="b" * 40, seed=SEED,
    )
    with RunRecorder(
        Ledger(ledger), entry_point=Path(__file__), protocol=protocol, run_kind="eval",
        repo=WT, env=Environment.detect(device="cpu"), wall_clock_s=0.0, cost=None,
        quick=True, quick_reason="a stand-in eval row for the v6-ctlcache split measurement; "
        "it scores nothing and is read by nothing but this script",
        recipe={"tool": "AUDIT/v6-ctlcache-2026-10-04/measure_ctl_split.py",
                "exclusions_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest()},
    ) as rec:
        for kind in ftc.LETTER_KIND_NAMES:
            rec.metric(f"val_top1.{kind}", NotRun(reason="a stand-in row scores nothing"))
    assert rec.row is not None
    verdicts.write_text(json.dumps({
        "eval_row_id": rec.row.row_id, "seed": SEED, "row_id": "standin", "kind": "span",
        "correct": True, "expected_abstain": True,
    }) + "\n", encoding="utf-8")
    return ledger, verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("split_flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    split_flags = [a for a in args.split_flags if a != "--"]
    args.scratch.mkdir(parents=True, exist_ok=True)
    ledger, verdicts = stand_in(args.scratch, split_flags)
    cache_flags = ([] if args.cache is None else
                   ["--split-cache", str(args.cache), "--split-cache-shards", str(SHARDS)])

    seen: dict[str, object] = {}
    real_cached = split_cache.cached_split_rows

    def timed_cached(*a: object, **kw: object):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        train, val, metric = real_cached(*a, **kw)  # type: ignore[arg-type]
        seen["cached_split_rows_s"] = round(time.perf_counter() - t, 2)
        seen["metric"] = metric.to_json()
        return train, val, metric

    def stop(rows: object, *, seed: int) -> object:
        seen["rows_at_stop"] = len(rows)  # type: ignore[arg-type]
        raise StopAfterSplit

    ftc.split_cache.cached_split_rows = timed_cached  # type: ignore[assignment]
    ftc.request_texts = stop  # type: ignore[assignment]
    argv = ["--ledger", str(ledger), "--verdicts", str(verdicts),
            "--write-ledger", str(args.scratch / "never-written.jsonl"), *split_flags,
            *cache_flags]
    t0 = time.perf_counter()
    try:
        ftc.main(argv)
    except StopAfterSplit:
        stopped = "after the split and the control's rule-3 check, at request_texts"
    else:
        stopped = "NOT stopped: main returned"
    seen.update(label=args.label, wall_to_stop_s=round(time.perf_counter() - t0, 2),
                stopped=stopped,
                control_row_written=(args.scratch / "never-written.jsonl").exists())
    print("MEASURE " + json.dumps(seen, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
