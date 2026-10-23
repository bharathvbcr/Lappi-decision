"""Read-only: for each set in sets.json, find who still reads it.

One streaming pass per file (no corpus held in memory). Per file it extracts
  - every `qd-campaign/<path>` mention (path up to the first char outside [\\w./-]),
  - every run of >=16 lowercase hex chars,
and a set is hit when a path mention equals the set's relative dir (or is a prefix-extension of
it, e.g. the set's own `/header.json`), or a hex run starts with the set's shard_hash[:16].

Kinds:
  live      campaign/*.json|sh, tools/, python/ (non-test), crates/ (non-test), data/
  test      python/tests, crates/*/tests
  ledger    ledger/*.jsonl
  external  files under ~/qd-campaign other than the set's own directory
  narrative HANDOFF/, AUDIT/, docs/, *.md, gaps.jsonl (mentions; not readers)
Corpus-level data_snapshot_hash is shared across sets and is NOT counted as a set reader.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(sys.argv[1])
SETS = REPO / "AUDIT/x4-v1-headers-2026-10-22/sets.json"
OUT = REPO / "AUDIT/x4-v1-headers-2026-10-22/readers.json"
CAMP = Path.home() / "qd-campaign"
EXCL = "AUDIT/x4-v1-headers-2026-10-22/"
PATH_RE = re.compile(r"qd-campaign/([\w./@+-]+)")
HEX_RE = re.compile(r"[0-9a-f]{16,64}")
TEXT_EXT = {".json", ".jsonl", ".md", ".sh", ".txt", ".log", ".py", ".toml", ".rs", ".yml", ".yaml", ".csv", ".tsv"}


def kind_repo(rel: str) -> str:
    if rel.startswith(("HANDOFF/", "AUDIT/", "docs/")) or rel.endswith(".md") or rel == "gaps.jsonl":
        return "narrative"
    if rel.startswith("ledger/"):
        return "ledger"
    if rel.startswith("python/tests/") or "/tests/" in rel:
        return "test"
    return "live"


sets = json.loads(SETS.read_text())
rels = {s["dir"].replace(str(CAMP) + "/", ""): s for s in sets}
h16 = defaultdict(list)
for rel, s in rels.items():
    h16[s["shard_hash"][:16]].append(rel)

hits: dict[str, dict[str, set[str]]] = {rel: defaultdict(set) for rel in rels}


def scan(kind: str, name: str, path: Path, own_dir: str | None = None) -> None:
    try:
        with path.open(errors="ignore") as fh:
            for line in fh:
                if "qd-campaign/" in line:
                    for m in PATH_RE.finditer(line):
                        p = m.group(1).rstrip("./,")
                        for rel in rels:
                            if p == rel or p.startswith(rel + "/"):
                                if kind == "external" and name.startswith("~/qd-campaign/" + rel + "/"):
                                    continue
                                hits[rel][kind].add(name + " [path]")
                if len(line) >= 16:
                    for m in HEX_RE.finditer(line):
                        for rel in h16.get(m.group(0)[:16], ()):
                            if kind == "external" and name.startswith("~/qd-campaign/" + rel + "/"):
                                continue
                            hits[rel][kind].add(name + " [hash]")
    except OSError:
        pass


tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")
for rel in tracked:
    if not rel or rel.startswith(EXCL) or Path(rel).suffix not in TEXT_EXT:
        continue
    p = REPO / rel
    if p.is_file():
        scan(kind_repo(rel), rel, p)
for p in CAMP.rglob("*"):
    if p.is_file() and p.suffix in TEXT_EXT and p.stat().st_size <= 200_000_000:
        sp = str(p)
        if "/target" in sp or "/.git/" in sp:
            continue
        if p.name == "header.json":
            continue
        scan("external", sp.replace(str(CAMP) + "/", "~/qd-campaign/"), p)

res = []
for rel, s in rels.items():
    h = {k: sorted(v) for k, v in hits[rel].items()}
    for k in ("live", "test", "ledger", "external", "narrative"):
        h.setdefault(k, [])
    res.append({"dir": s["dir"], "rel": rel, "shard_hash": s["shard_hash"], "data_snapshot_hash": s["data_snapshot_hash"], "hits": h})
OUT.write_text(json.dumps(res, indent=1))
for r in res:
    h = r["hits"]
    print(f'{r["rel"]:100s} live={len(h["live"])} test={len(h["test"])} ledger={len(h["ledger"])} ext={len(h["external"])} narr={len(h["narrative"])}')
