"""Throwaway (role 3): what A7 reads on the v5 rebuild with the pre-dedupe drop list, without
the 46-minute build. Runs tools/real_tokenizer_pipeline.py's own main() on the v5 build's argv
(dryrun_drops.sh) and, at its dedupe call, takes build_mixture's rows and runs dedupe + split
twice: (1) as built, which must reproduce the first build's counts and A7's 98 missing keys
(parity), and (2) with qd_train.exclusions.drop_before_dedupe applied to the rows first. A7's
own check_split reads each split's val and held-out rows (exclusions touch train only; contrast
rows are train), against v4's manifests and the pool's val rows less those build_mixture refuses.
Stops before anything is written; writes dryrun-drops.json beside this script.

Usage (through dryrun_drops.sh): python dryrun_drops.py <drop list> <pipeline argv...>
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

WT = Path("/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt")
HERE = Path(__file__).resolve().parent
DROPS = Path(sys.argv[1])
os.chdir(WT)
sys.path[:0] = [str(WT / "tools"), str(WT / "python")]

import real_tokenizer_pipeline as rtp  # noqa: E402
import v5_a7_check as a7  # noqa: E402
from qd_data.decisions import load_decision_pool, pool_data_config, rewrite_typed_decision  # noqa: E402
from qd_data.manifest import Manifest  # noqa: E402
from qd_data.mixture import RowRefused  # noqa: E402
from qd_train.exclusions import drop_before_dedupe  # noqa: E402

POOL = Path("/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4")
V4 = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01")
PARITY = {"rows_in": 561681, "kept": 555469,
          "split": {"train": 495208, "val": 22315, "heldout": 37946}}


class _Stop(Exception):
    pass


def pool_expectation() -> a7.PoolExpectation:
    pool = load_decision_pool(POOL)
    config = pool_data_config()
    val: dict[str, Counter[str]] = {}
    for rows in pool.raw.values():
        for i, raw in enumerate(rows):
            if raw.split != "val":
                continue
            try:
                row = rewrite_typed_decision(raw, family_id=raw.family_id, index=i, config=config)
            except RowRefused:
                continue
            val.setdefault(row.family_id, Counter())[row.identity_key] += 1
    return a7.PoolExpectation(examples_sha256=pool.examples_sha256, val=val)


def read_a7(split_report, pool) -> dict[str, object]:
    out: dict[str, object] = {"failed": [], "families": []}
    for split_name in ("val", "heldout"):
        v4 = Manifest.read(V4 / a7.SPLIT_PATHS[split_name])
        v5 = SimpleNamespace(entries=list(split_report.rows_by_split.get(split_name, ())))
        for r in a7.check_split(v4, v5, split_name, a7.GATE, pool):
            out["families"].append(r)
            if not r["passed"]:
                out["failed"].append(
                    f"{split_name}/{r['family']}: {r.get('missing')} missing, "
                    f"{r.get('extra')} extra; first missing {r.get('first_missing')}"
                )
    return out


result: dict[str, object] = {"drops": str(DROPS)}
_real_dedupe, _real_split = rtp.dedupe, rtp.split


def _dedupe(rows, **kw):
    config = kw["config"]
    rows = list(rows)
    pool = pool_expectation()
    base = _real_dedupe(rows, **kw)
    base_split = _real_split(base, config=config)
    counts = base_split.counts()
    got = {"rows_in": len(rows), "kept": len(base.kept),
           "split": {k: counts.get(k, 0) for k in PARITY["split"]}}
    print(f"parity: {got} vs the build's {PARITY}", flush=True)
    if got != PARITY:
        raise SystemExit(2)
    base_a7 = read_a7(base_split, pool)
    print(f"as built, A7 fails: {base_a7['failed']}", flush=True)
    result["as_built"] = {"counts": got, "failed": base_a7["failed"]}
    kept, drops, dropped = drop_before_dedupe(rows, DROPS, config=config)
    print(f"drop list sha256 {drops.sha256}: {len(drops.keys)} keys, {len(dropped)} rows "
          f"dropped before dedupe, by family {dict(Counter(r.family_id for r in dropped))}",
          flush=True)
    after = _real_dedupe(list(kept), **kw)
    after_split = _real_split(after, config=config)
    counts2 = after_split.counts()
    after_a7 = read_a7(after_split, pool)
    result["with_drops"] = {
        "rows_in": len(kept), "kept": len(after.kept),
        "split": {k: counts2.get(k, 0) for k in PARITY["split"]},
        "failed": after_a7["failed"],
        "families": [{k: v for k, v in r.items() if k in ("split", "family", "v4", "v5",
                      "expected", "missing", "extra", "first_missing", "passed")}
                     for r in after_a7["families"]],
    }
    print(f"with drops: rows {len(kept)}, kept {len(after.kept)}, split {counts2}", flush=True)
    print(f"with drops, A7 fails: {after_a7['failed'] or 'none: A7 PASS'}", flush=True)
    raise _Stop


rtp.dedupe = _dedupe
try:
    rtp.main(sys.argv[2:])
    raise SystemExit("the pipeline returned without reaching dedupe")
except _Stop:
    pass
(HERE / "dryrun-drops.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(f"written {HERE / 'dryrun-drops.json'}")
