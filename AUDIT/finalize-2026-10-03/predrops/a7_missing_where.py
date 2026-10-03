"""Throwaway (role 3): every A7 missing identity key of the trial (a7_pool_refusals.py's
expectation), and where it is in the v5 build: in v5's train manifest (a split move, which for a
held-out key would be rule 3), in no manifest (a dedupe knock-out, the DRAFT's case), or in
exclusions.txt. Reads manifests only; prints counts and the keys, writes a JSON beside it.

Usage: python build/v5-build/a7_missing_where.py
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
from qd_data.manifest import Manifest  # noqa: E402
from qd_data.mixture import RowRefused  # noqa: E402

POOL = Path("/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4")
V4 = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01")
V5 = Path("/Users/bharath/qd-campaign/phase4-v5-2026-10-03")
EXCL = Path("/Users/bharath/qd-campaign/v5-containment-scoped-2026-10-03/exclusions.txt")
OUT = Path(__file__).resolve().parent / "a7-missing-where.json"

pool = load_decision_pool(POOL)
config = pool_data_config()
want_val: dict[str, Counter[str]] = {}
for rows in pool.raw.values():
    for i, raw in enumerate(rows):
        if raw.split != "val":
            continue
        try:
            row = rewrite_typed_decision(raw, family_id=raw.family_id, index=i, config=config)
        except RowRefused:
            continue
        want_val.setdefault(row.family_id, Counter())[row.identity_key] += 1

v4 = {s: Manifest.read(V4 / p) for s, p in a7.SPLIT_PATHS.items()}
v5 = {s: Manifest.read(V5 / p) for s, p in a7.SPLIT_PATHS.items()}
v5_train = Manifest.read(V5 / "data/pool/train.json")
train_keys: dict[str, list[tuple[str, str]]] = {}
for e in v5_train.entries:
    train_keys.setdefault(e.identity_key, []).append((e.family_id, e.repo_key))
excluded = {line.strip() for line in EXCL.read_text(encoding="utf-8").splitlines() if line.strip()}

result: dict[str, object] = {}
for split in ("val", "heldout"):
    ids4 = {}
    for e in v4[split].entries:
        ids4.setdefault(e.family_id, Counter())[e.identity_key] += 1
    ids5 = {}
    for e in v5[split].entries:
        ids5.setdefault(e.family_id, Counter())[e.identity_key] += 1
    families = set(ids4) | set(ids5) | (set(want_val) if split == "val" else set())
    for fam in sorted(families):
        if fam in a7.GATE.clinc[split]:
            continue
        want = want_val.get(fam) if (split == "val" and fam in want_val and fam not in ids4) else ids4.get(fam, Counter())
        if want is None:
            want = Counter()
        missing = want - ids5.get(fam, Counter())
        if not missing:
            continue
        where = Counter()
        detail = []
        for key in sorted(missing):
            in_train = train_keys.get(key, [])
            place = "in v5 train" if in_train else "in no v5 manifest"
            if key in excluded:
                place += " + in exclusions.txt"
            where[place] += missing[key]
            detail.append({"key": key, "place": place, "train_rows": in_train[:3]})
        result[f"{split}/{fam}"] = {"missing": sum(missing.values()), "where": dict(where),
                                   "keys": detail}
        print(f"{split}/{fam}: {sum(missing.values())} missing -> {dict(where)}")
        for d in detail[:4]:
            print(f"    {d['key'][:110]} | {d['place']} | {d['train_rows'][:1]}")
OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(f"written {OUT}")
