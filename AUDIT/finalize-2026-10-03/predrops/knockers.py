"""Throwaway (role 3): the dedupe clusters that removed A7's missing val/held-out rows.

Runs tools/real_tokenizer_pipeline.py's own main() on the v5 build's argv (knockers.sh passes
it), with ``dedupe`` and ``split`` wrapped so the run stops right after the split and writes
nothing past it. Parity first: rows in, kept and the split counts must equal the v5 build's
log (561681 / 555469 / train 495208, val 22315, heldout 37946), or the clusters are not the
build's and the script exits 2. Then, for every identity key in a7-missing-where.json, the
cluster that dropped its unit: every member, its family, source, repo, and where it ended
(kept and in which split, or dropped). Writes knockers.json beside this script.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

WT = Path("/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt")
HERE = Path(__file__).resolve().parent
os.chdir(WT)
sys.path[:0] = [str(WT / "tools"), str(WT / "python")]

import real_tokenizer_pipeline as rtp  # noqa: E402
from qd_data.dedupe import content_unit_key  # noqa: E402

PARITY = {"rows_in": 561681, "kept": 555469,
          "split": {"train": 495208, "val": 22315, "heldout": 37946}}


class _Stop(Exception):
    pass


captured: dict[str, object] = {}
_real_dedupe, _real_split = rtp.dedupe, rtp.split


def _dedupe(rows, **kw):
    captured["rows"] = list(rows)
    report = _real_dedupe(rows, **kw)
    captured["dedupe"] = report
    return report


def _split(report, **kw):
    captured["split"] = _real_split(report, **kw)
    raise _Stop


rtp.dedupe, rtp.split = _dedupe, _split
try:
    rtp.main(sys.argv[1:])
    raise SystemExit("the pipeline returned without reaching split")
except _Stop:
    pass

rows = captured["rows"]
report = captured["dedupe"]
split_report = captured["split"]
counts = split_report.counts()
got = {"rows_in": len(rows), "kept": len(report.kept),
       "split": {k: counts.get(k, 0) for k in PARITY["split"]}}
print(f"parity: got {got}, the build's log {PARITY}")
if got != PARITY:
    raise SystemExit(2)

unit_rows: dict[str, list] = {}
for r in rows:
    unit_rows.setdefault(content_unit_key(r), []).append(r)
split_of_unit: dict[str, str] = {}
for split_name, split_rows in split_report.rows_by_split.items():
    for r in split_rows:
        split_of_unit[content_unit_key(r)] = split_name

cluster_of: dict[str, tuple[str, object]] = {}
for c in report.clusters:
    for k in (c.kept_unit_key, *c.dropped_unit_keys):
        cluster_of[k] = ("minhash", c)
for c in report.exact_content_clusters:
    for k in (c.kept_unit_key, c.minhash_unit_key, *c.dropped_unit_keys):
        if k is not None:
            cluster_of.setdefault(k, ("exact", c))


def describe(unit_key: str) -> dict[str, object]:
    members = unit_rows.get(unit_key, [])
    first = members[0] if members else None
    return {
        "unit": unit_key,
        "family": None if first is None else first.family_id,
        "source": None if first is None else first.source_id,
        "repo": None if first is None else first.repo_key,
        "identity_keys": sorted({m.identity_key for m in members}),
        "n_rows": len(members),
        "ended": split_of_unit.get(unit_key, "dropped"),
        "text_head": None if first is None else first.dedupe_text[:160],
    }


missing = json.loads((HERE / "a7-missing-where.json").read_text(encoding="utf-8"))
out: dict[str, object] = {"parity": got, "families": {}}
by_key_unit: dict[str, list[str]] = {}
for unit_key in unit_rows:
    by_key_unit.setdefault(unit_key.split("|", 1)[0], []).append(unit_key)
seen_clusters: set[int] = set()
for fam_key, info in missing.items():
    fam_out = []
    for entry in info["keys"]:
        key = entry["key"]
        units = by_key_unit.get(key, [])
        for u in units:
            kind, c = cluster_of.get(u, (None, None))
            if c is None:
                fam_out.append({"key": key, "unit": u, "cluster": None,
                                "ended": split_of_unit.get(u, "dropped")})
                continue
            members = [m for m in (getattr(c, "kept_unit_key", None),
                                   getattr(c, "minhash_unit_key", None),
                                   *c.dropped_unit_keys) if m is not None]
            fam_out.append({
                "key": key, "unit": u, "cluster_kind": kind,
                "kept": c.kept_unit_key or getattr(c, "minhash_unit_key", None),
                "min_edge_jaccard": getattr(c, "min_edge_jaccard", None),
                "members": [describe(m) for m in members],
            })
            seen_clusters.add(id(c))
        if not units:
            fam_out.append({"key": key, "unit": None, "cluster": None,
                            "note": "no dedupe input unit carries this identity key"})
    out["families"][fam_key] = fam_out
    ends = Counter()
    for f in fam_out:
        for m in f.get("members", []):
            if m["unit"] != f["unit"]:
                ends[(m["family"], m["ended"])] += 1
    print(f"{fam_key}: {len(fam_out)} units; the other members by (family, ended): "
          f"{dict(ends)}")
out["n_clusters"] = len(seen_clusters)
(HERE / "knockers.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(f"{len(seen_clusters)} clusters; written {HERE / 'knockers.json'}")
