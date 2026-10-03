"""Throwaway (role 3): what v5_a7_check would say if its pool expectation dropped the val rows
the pipeline's build_mixture drops (``RowRefused`` from ``rewrite_typed_decision``, counted, not
raised). Prints the refused pool val rows by family and reason, then runs A7's own ``check``
with that expectation and writes the report beside this script. Changes nothing in the build.

Usage: python build/v5-build/a7_pool_refusals.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

WT = Path("/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt")
sys.path.insert(0, str(WT / "python"))
sys.path.insert(0, str(WT / "tools"))

import v5_a7_check as a7  # noqa: E402
from qd_data.decisions import load_decision_pool, pool_data_config, rewrite_typed_decision  # noqa: E402
from qd_data.mixture import RowRefused  # noqa: E402

POOL = Path("/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4")
V4 = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01")
V5 = Path("/Users/bharath/qd-campaign/phase4-v5-2026-10-03")
OUT = Path(__file__).resolve().parent / "a7-trial-pool-refusals.json"

pool = load_decision_pool(POOL)
config = pool_data_config()
val: dict[str, Counter[str]] = {}
refused: Counter[tuple[str, str, str]] = Counter()
refused_rows: list[dict[str, object]] = []
all_splits_refused: Counter[tuple[str, str]] = Counter()
for source, rows in pool.raw.items():
    for i, raw in enumerate(rows):
        try:
            row = rewrite_typed_decision(raw, family_id=raw.family_id, index=i, config=config)
        except RowRefused as exc:
            all_splits_refused[(raw.split, exc.reason_code)] += 1
            if raw.split == "val":
                refused[(source, raw.family_id, exc.reason_code)] += 1
                refused_rows.append({"source": source, "family": raw.family_id, "index": i,
                                     "reason": exc.reason_code, "actual": str(exc.actual)[:120]})
            continue
        if raw.split == "val":
            val.setdefault(row.family_id, Counter())[row.identity_key] += 1

stated = pool.manifest.get("val_by_family")
got = {f: sum(c.values()) for f, c in val.items()}
by_family = Counter()
for (_, fam, _), n in refused.items():
    by_family[fam] += n
print("pool rows refused by split and reason:", dict(sorted(all_splits_refused.items())))
print("pool val rows refused:", {f"{s} | {f} | {r}": n for (s, f, r), n in sorted(refused.items())})
recon = {f: {"stated": stated.get(f), "built": got.get(f, 0), "refused": by_family.get(f, 0)}
         for f in sorted(set(stated) | set(got) | set(by_family))}
bad = {f: v for f, v in recon.items() if v["stated"] != v["built"] + v["refused"]}
print("stated == built + refused for every pool family:", not bad, bad or "")

expectation = a7.PoolExpectation(examples_sha256=pool.examples_sha256, val=val)
a7.pool_expectation = lambda _pool_dir: expectation
report = a7.check(V4, V5, decisions_pool=POOL)
report["trial"] = {"pool_val_refused": refused_rows, "reconciliation": recon}
OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
for r in report["families"]:
    mark = "ok  " if r["passed"] else "FAIL"
    print(f"{mark} {r['split']:8s} {r['family']:34s} v4 {r['v4']:6d} v5 {r['v5']:6d} "
          f"expected {r.get('expected', '-')}")
    for reason in r["reasons"]:
        print(f"       {reason}")
print(f"A7 trial {'PASS' if report['passed'] else 'REFUSED'}: "
      f"{report['families_checked'] - len(report['failed'])}/{report['families_checked']}; "
      f"report {OUT}")
