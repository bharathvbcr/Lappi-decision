"""File-by-file comparison of two char_build outputs.

    python compare.py <before dir> <after dir>

Byte-compares every file. For JSON files that differ, prints every differing leaf path (dotted),
so a reader can see which fields moved. Exit 1 if any non-JSON file differs or the file sets differ.
"""

import json
import sys
from pathlib import Path

a, b = Path(sys.argv[1]), Path(sys.argv[2])
fa = {str(p.relative_to(a)) for p in a.rglob("*") if p.is_file()}
fb = {str(p.relative_to(b)) for p in b.rglob("*") if p.is_file()}
print("files before:", len(fa), "after:", len(fb),
      "only-before:", sorted(fa - fb), "only-after:", sorted(fb - fa))


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from leaves(x[k], f"{prefix}.{k}" if prefix else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def diff(x, y):
    lx, ly = dict(leaves(x)), dict(leaves(y))
    out = []
    for k in sorted(set(lx) | set(ly)):
        if lx.get(k, "<absent>") != ly.get(k, "<absent>"):
            out.append((k, lx.get(k, "<absent>"), ly.get(k, "<absent>")))
    return out


identical, bad = 0, 0
for rel in sorted(fa & fb):
    ra, rb = (a / rel).read_bytes(), (b / rel).read_bytes()
    if ra == rb:
        identical += 1
        continue
    if not rel.endswith(".json"):
        bad += 1
        print("DIFFERS (non-JSON):", rel)
        continue
    d = diff(json.loads(ra), json.loads(rb))
    print(f"DIFFERS (JSON) {rel}: {len(d)} leaf fields")
    for k, x, y in d:
        sx, sy = json.dumps(x)[:70], json.dumps(y)[:70]
        print(f"    {k}: {sx} -> {sy}")
print("byte-identical:", identical, "of", len(fa & fb), "; differing non-JSON files:", bad)
sys.exit(1 if bad or fa != fb else 0)
