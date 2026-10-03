"""Throwaway (role 3): the pre-dedupe drop list from knockers.json (Fable's option B, ratified
~21:52Z 2026-10-03). For every dedupe component that removed one of A7's missing eval units,
every member that is not an eval unit: the kept train twin and the train members dedupe
already dropped, so that, with all of them gone before dedupe, nothing in the component sorts
ahead of the eval unit. Eval members (the missing units, members that ended val/held-out,
repo keys pinned -val:/-heldout:) are never listed; the pipeline's drop_before_dedupe
re-checks every listed row's split and refuses a non-train one. Writes the list
(byte-sorted, unique, LF) and a summary; prints what stays unrecoverable.

Usage: python build/v5-build/derive_drops.py <out list path>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1])
k = json.loads((HERE / "knockers.json").read_text(encoding="utf-8"))
missing_units = {e["unit"] for fam in k["families"].values() for e in fam if e.get("unit")}

drop: set[str] = set()
by_family: Counter[str] = Counter()
residual: list[dict[str, object]] = []
for fam_key, entries in k["families"].items():
    for e in entries:
        members = e.get("members")
        if not members:
            residual.append({"family": fam_key, "key": e["key"], "why": "no cluster found"})
            continue
        evals_ahead = []
        for m in members:
            is_eval = (
                m["unit"] in missing_units
                or m["ended"] in ("val", "heldout")
                or any(p in (m["repo"] or "") for p in ("-val:", "-heldout:"))
            )
            if is_eval:
                if m["unit"] != e["unit"] and m["unit"] < e["unit"]:
                    evals_ahead.append(m["unit"])
                continue
            for ik in m["identity_keys"]:
                if ik not in drop:
                    by_family[m["family"]] += 1
                drop.add(ik)
        if evals_ahead:
            residual.append({"family": fam_key, "key": e["key"],
                             "why": "an eval unit sorts ahead of it in its component",
                             "ahead": evals_ahead[:3]})

keys = sorted(drop, key=lambda s: s.encode("utf-8"))
out.write_bytes(b"".join(s.encode("utf-8") + b"\n" for s in keys))
print(f"{len(keys)} identity keys -> {out}")
print(f"by family: {dict(sorted(by_family.items()))}")
print(f"residual (not recoverable by dropping train rows): {len(residual)}")
for r in residual:
    print(f"  {r}")
(HERE / "derive-drops-summary.json").write_text(json.dumps(
    {"n_keys": len(keys), "by_family": dict(by_family), "residual": residual}, indent=1
) + "\n", encoding="utf-8")
