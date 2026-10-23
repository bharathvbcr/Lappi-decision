"""Read-only: enumerate every shard header under ~/qd-campaign and its drift against HEAD.

Writes sets.json (one record per header). Never opens a shard payload, only header.json
and the sibling manifest names. No allow_stale_code anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO / "python"))
from qd_data.fingerprint import code_fingerprint, drifted_modules, shard_code_fingerprint  # noqa: E402

ROOT = Path.home() / "qd-campaign"
cur_all = code_fingerprint()
cur_v2 = shard_code_fingerprint()
rows = []
for h in sorted(ROOT.rglob("header.json")):
    d = json.loads(h.read_text())
    rec_fp = d.get("code_fingerprint")
    ver = d.get("code_fingerprint_version", 1)
    cur = cur_v2 if ver == 2 else cur_all
    if rec_fp is None:
        drift = ("no-fingerprint",)
    else:
        ch, ad, rm = drifted_modules(rec_fp, cur)
        drift = tuple(ch) + tuple("+" + a for a in ad) + tuple("-" + r for r in rm)
    # v1-rule restricted to the 16 listed modules
    if rec_fp:
        ch16 = [m for m in cur_v2 if m in rec_fp and rec_fp[m] != cur_v2[m]]
    else:
        ch16 = None
    rows.append(
        {
            "dir": str(h.parent),
            "header_keys": sorted(d.keys()),
            "fp_version": ver,
            "data_snapshot_hash": d.get("data_snapshot_hash"),
            "shard_hash": d.get("shard_hash"),
            "tokenizer_hash": d.get("tokenizer_hash"),
            "drift": list(drift),
            "drift16": ch16,
            "files": sorted(p.name for p in h.parent.iterdir()),
            "code_commit": d.get("code_commit"),
            "rule": d.get("data_rules") or d.get("rule"),
        }
    )
OUT.write_text(json.dumps(rows, indent=1))
print(len(rows), "headers;", sum(1 for r in rows if r["drift"]), "drifted")
