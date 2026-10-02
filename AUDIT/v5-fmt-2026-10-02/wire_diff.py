"""Structural diff of fixtures/wire/ before and after the prompt-format-2 regeneration.

Throwaway analysis (lane L-v5-fmt, 2026-10-02); it ships nothing.

    python AUDIT/v5-fmt-2026-10-02/wire_diff.py <old_dir> <new_dir>

Reports the file sets and, per byte-changed file: whether every object has the same keys at
every level, whether the envelope fields and slot names are identical, and which leaf paths
changed value or JSON type. For an answer file it also prints each slot's shape (answered or
abstained) and the envelope's degraded flag on both sides; for index.json, which request fields
moved per fixture, with the old and new context bytes decoded.
"""

import base64
import json
import sys
from pathlib import Path

ENVELOPE = ("status", "schema_version", "backend", "degraded")


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["key_mismatch"].append((path, sorted(set(a) ^ set(b))))
        for k in sorted(set(a) & set(b)):
            walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["changed"].append((path, a, b))
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]", out)
    elif a != b or type(a) is not type(b):
        out["changed"].append((path, a, b))


def shape(envelope):
    slots = sorted(envelope["slots"].items())
    return {k: ("abstained" if s.get("noul") else "answered") for k, s in slots}


def index_delta(a, b):
    old_e = {e["file"]: e for e in a["entries"]}
    new_e = {e["file"]: e for e in b["entries"]}
    same = set(old_e) == set(new_e)
    print(f"  index entries: old={len(old_e)} new={len(new_e)} same_files={same}")
    for f in sorted(set(old_e) & set(new_e)):
        for field in sorted(set(old_e[f]) | set(new_e[f])):
            if old_e[f].get(field) == new_e[f].get(field):
                continue
            if field != "request":
                print(f"  entry field changed: {f}: {field}")
                continue
            ra, rb = old_e[f]["request"], new_e[f]["request"]
            moved = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
            print(f"  request changed: {f}: fields {moved}")
            if "context_b64" in moved:
                print(f"    context old: {base64.b64decode(ra['context_b64'])!r}")
                print(f"    context new: {base64.b64decode(rb['context_b64'])!r}")


def main():
    old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    old_files = {p.name for p in old_dir.glob("*.json")}
    new_files = {p.name for p in new_dir.glob("*.json")}
    print(f"files: old={len(old_files)} new={len(new_files)} same_set={old_files == new_files}")
    changed_files = sorted(
        n
        for n in old_files & new_files
        if (old_dir / n).read_bytes() != (new_dir / n).read_bytes()
    )
    print(f"byte-changed files: {len(changed_files)}: {changed_files}")
    for name in changed_files:
        a = json.loads((old_dir / name).read_text())
        b = json.loads((new_dir / name).read_text())
        out = {"key_mismatch": [], "changed": []}
        walk(a, b, "$", out)
        envelope_same = all(a.get(k) == b.get(k) for k in ENVELOPE)
        slots_same = sorted(a.get("slots", {})) == sorted(b.get("slots", {}))
        print(f"\n== {name}")
        print(f"  keys identical at every level: {not out['key_mismatch']} {out['key_mismatch']}")
        print(f"  envelope fields {ENVELOPE} identical: {envelope_same}")
        print(f"  slot names identical: {slots_same} {sorted(b.get('slots', {}))}")
        for path, x, y in out["changed"]:
            kind = "value" if type(x) is type(y) else f"type {type(x).__name__}->{type(y).__name__}"
            print(f"  changed {path}: {json.dumps(x)} -> {json.dumps(y)} ({kind})")
        leaf_fields = {p.rsplit(".", 1)[-1].split("[")[0] for p, _, _ in out["changed"]}
        print(f"  changed leaf field names: {sorted(leaf_fields)}")
        if isinstance(a.get("slots"), dict) and isinstance(b.get("slots"), dict):
            same_shape = shape(a) == shape(b) and a.get("degraded") == b.get("degraded")
            print(f"  shape old: degraded={a.get('degraded')} {shape(a)}")
            print(f"  shape new: degraded={b.get('degraded')} {shape(b)}")
            print(f"  shape identical: {same_shape}")
        if name == "index.json":
            index_delta(a, b)


if __name__ == "__main__":
    main()
