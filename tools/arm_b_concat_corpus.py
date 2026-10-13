#!/usr/bin/env python3
"""Concatenate qd-mutate corpus parts into one corpus directory (card ft-b1dfb6da273).

``single-file-mutants-v1`` is four ``qd-mutate generate --language L --limit 3000`` runs, one per
language, joined in the order given (alphabetical: go python rust typescript). A corpus the loader
(``qd_data.defect_class.load_defect_rows``) reads is a directory of ``examples.jsonl`` and a
``manifest.json`` that pins those bytes and names the pool, so this writes both: the rows
byte-for-byte as the parts hold them, and the first part's manifest with ``examples_sha256`` and
``totals`` recomputed and a ``concatenation_of`` block naming every part's sha256.

    arm_b_concat_corpus.py --out DIR PART_DIR [PART_DIR ...]

Every part must name the same pool (path, sha256, records) and the same seed; a part that does not
refuses the run. Nothing is parsed into a row type: the rows are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


class ConcatRefusal(Exception):
    """A part that cannot join the others."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def concat(parts: list[Path], out: Path) -> dict[str, Any]:
    """Write ``out/examples.jsonl`` and ``out/manifest.json`` from ``parts``; return the latter."""
    if not parts:
        raise ConcatRefusal("no part given")
    if out.exists():
        raise ConcatRefusal(f"{out} exists; refusing to overwrite it")
    body = b""
    manifests: list[dict[str, Any]] = []
    described: list[dict[str, Any]] = []
    for part in parts:
        examples = (part / "examples.jsonl").read_bytes()
        manifest = json.loads((part / "manifest.json").read_text())
        if _sha(examples) != manifest.get("examples_sha256"):
            raise ConcatRefusal(f"{part}: examples.jsonl does not hash to its manifest's sha256")
        if manifests and (
            manifest["pool"] != manifests[0]["pool"] or manifest["seed"] != manifests[0]["seed"]
        ):
            raise ConcatRefusal(f"{part}: a different pool or seed from {parts[0]}")
        manifests.append(manifest)
        body += examples
        described.append(
            {
                "part": part.name,
                "examples": manifest["totals"]["examples"],
                "examples_sha256": manifest["examples_sha256"],
                "manifest_sha256": _sha((part / "manifest.json").read_bytes()),
            }
        )
    merged = dict(manifests[0])
    totals = dict(manifests[0]["totals"])
    for key, value in totals.items():
        if isinstance(value, int):
            totals[key] = sum(int(m["totals"][key]) for m in manifests)
    merged["totals"] = totals
    merged["examples_sha256"] = _sha(body)
    merged["concatenation_of"] = described
    n_lines = body.count(b"\n")
    if n_lines != totals["examples"]:
        raise ConcatRefusal(f"{n_lines} rows written, the totals say {totals['examples']}")
    out.mkdir(parents=True)
    (out / "examples.jsonl").write_bytes(body)
    (out / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n")
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("parts", type=Path, nargs="+")
    args = p.parse_args(argv)
    try:
        m = concat(args.parts, args.out)
    except ConcatRefusal as exc:
        print(f"arm_b_concat_corpus: refused: {exc}", file=sys.stderr)
        return 2
    print(f"examples {m['totals']['examples']} sha256 {m['examples_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
