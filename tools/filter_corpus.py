"""Derive a corpus subset, and a manifest that says it is one.

## Why a tool rather than a `grep -v clean`

A subset is a different corpus and must hash differently, or the ledger reads two
populations as one protocol measured twice. ``rung0_real_run.py`` takes ``corpus_hash``
from the manifest, so a subset that reused its parent's manifest would be indistinguishable
from the parent in every row it wrote. This writes a manifest carrying the filter, the
parent's digest and the surviving counts, so the hash moves for the right reason and a
reader can see what was dropped.

## The case this exists for

``--drop-class clean`` on the commitpackft corpus. ``crates/qd-mutate/src/generate.rs``
emits ``clean`` rows with ``before == after`` and an empty diff, so in
``--context-source diff`` an empty context means ``clean`` and nothing else -- measured,
8,450 of 50,178 rows, an exact biconditional (``AUDIT/after-vs-diff-leak.md``). Dropping
the class removes the leak without pretending to have fixed it.

**This is a diagnostic subset, not the shipping task.** The option list stays four wide, so
the model keeps a ``clean`` output it is never shown a gold example of, and no run over
this corpus can discharge the four-way gate. The repair that does is to give ``clean`` rows
the agent's own commit diff, which needs the pool to carry the before-image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--manifest-in", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--drop-class", action="append", default=[],
        help="a mutation class to remove. Repeatable. Recorded in the manifest, so the "
             "subset's corpus_hash moves and its rows cannot pool with the parent's",
    )
    args = parser.parse_args(argv)

    if not args.drop_class:
        raise SystemExit(
            "no --drop-class given. A subset identical to its parent would write a "
            "DIFFERENT corpus_hash for the SAME rows, which is the incomparability "
            "defect pointing the other way."
        )

    rows = [
        json.loads(line)
        for line in args.examples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{args.examples} holds no examples")

    drop = set(args.drop_class)
    before = Counter(r["class"] for r in rows)
    unknown = drop - set(before)
    if unknown:
        # Fail closed: a typo would otherwise drop nothing and produce a "filtered"
        # corpus identical to its parent but carrying a different hash and a manifest
        # claiming a filter had been applied.
        raise SystemExit(
            f"--drop-class named {sorted(unknown)}, which no row carries. Present "
            f"classes: {sorted(before)}"
        )

    kept = [r for r in rows if r["class"] not in drop]
    if not kept:
        raise SystemExit("every row was dropped; there is nothing to train on")
    after = Counter(r["class"] for r in kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for record in kept:
            sink.write(json.dumps(record, sort_keys=True) + "\n")

    parent = json.loads(args.manifest_in.read_text(encoding="utf-8"))
    manifest = {
        "derived_from": {
            "manifest": str(args.manifest_in),
            # The parent's own hash, so the lineage is checkable rather than asserted.
            "manifest_sha256": hashlib.sha256(
                json.dumps(parent, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "examples": str(args.examples),
        },
        "filter": {"dropped_classes": sorted(drop)},
        "examples": len(kept),
        "examples_before": len(rows),
        "by_class": dict(sorted(after.items())),
        "by_class_before": dict(sorted(before.items())),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        # Carried forward because `rung0_real_run.py` reads it for the row's pool size;
        # the pool did not change, only which of its examples survived.
        "pool": parent.get("pool"),
    }
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"kept {len(kept)} of {len(rows)} example(s), dropping {sorted(drop)}")
    for cls in sorted(before):
        print(f"  {cls:>9} {before[cls]:>6} -> {after.get(cls, 0):>6}")
    print(f"  wrote {args.out}")
    print(f"  sha256 {manifest['sha256']}")
    print(f"  manifest {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
