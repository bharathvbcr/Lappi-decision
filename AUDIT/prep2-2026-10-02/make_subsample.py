"""Content-hash subsamples of v4's inputs for the containment scan (lane L-prep2, 2026-10-02).

Re-creates what HANDOFF/prep-containment-2026-10-02.md describes L-prep's lost scratch script
(``partb/make_subsample.py``) doing: every v4 input sampled by content hash -- the eleven
general caches named by the fetch record, the 25,000 composed rows and the 5,004 noul rows --
with the composed corpus's base sampled by the pipeline's own ``--defect-max-rows``. L-prep's
hash was not recorded, so these subsamples are NOT L-prep's rows: a line is kept iff the first
eight bytes of sha256(line without its newline), big-endian, are below ``fraction * 2**64``.

Writes, under ``--out``:

* ``general/``: the cache tree (same relative paths), each jsonl sampled, CLINC's
  ``intent_names.json`` and the pinned ``clinc__oos-eval/<commit>/domains.json`` copied, and
  ``fetch-record.json`` with each entry's ``jsonl``, ``jsonl_sha256`` and ``rows`` rewritten;
* ``commitpackft-composed-v1/`` and ``defect-noul-v3b/``: ``examples.jsonl`` sampled and
  ``manifest.json`` with ``examples_sha256`` and ``totals.examples`` rewritten;
* ``subsample.json``: the fraction, the hash rule, every count and sha256, and the
  ``--defect-max-rows`` to pass (round(fraction x the base corpus's totals.examples)).

Run with the repo's ML python from the repository root::

    python AUDIT/prep2-2026-10-02/make_subsample.py --fraction 0.05 \\
        --out /Users/bharath/qd-campaign/prep2-2026-10-02/sub-p05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GENERAL_RECORD = Path("/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json")
COMPOSED = REPO / "data" / "pool" / "commitpackft-composed-v1"
NOUL = REPO / "data" / "pool" / "defect-noul-v3b"
BASE = REPO / "data" / "pool" / "commitpackft-corpus-v3"
CLINC_DOMAINS_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
RULE = "keep a line iff int.from_bytes(sha256(line.rstrip(b'\\n'))[:8], 'big') < fraction * 2**64"


def _keep(line: bytes, bound: int) -> bool:
    body = line[:-1] if line.endswith(b"\n") else line
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "big") < bound


def _sample(src: Path, dest: Path, bound: int) -> dict[str, object]:
    """``src`` sampled into ``dest`` line by line; its counts and both sha256s."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    read = kept = 0
    out_hash = hashlib.sha256()
    with src.open("rb") as fh, dest.open("xb") as out:
        for line in fh:
            if not line.strip():
                continue
            read += 1
            if _keep(line, bound):
                kept += 1
                out.write(line)
                out_hash.update(line)
    return {"src": str(src), "dest": str(dest), "read": read, "kept": kept,
            "sha256": out_hash.hexdigest()}


def _manifest(src_dir: Path, dest_dir: Path, sampled: dict[str, object]) -> None:
    manifest = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["examples_sha256"] = sampled["sha256"]
    manifest["totals"] = {**manifest["totals"], "examples": sampled["kept"]}
    manifest["subsample_of"] = {"dir": str(src_dir), "rule": RULE}
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise SystemExit(f"--fraction {args.fraction} is not a subsample")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite it")
    bound = int(args.fraction * 2**64)
    root = GENERAL_RECORD.parent.resolve()
    general = args.out / "general"
    entries = json.loads(GENERAL_RECORD.read_text(encoding="utf-8"))
    report: dict[str, object] = {"fraction": args.fraction, "rule": RULE, "general": []}
    for entry in entries:
        src = Path(entry["jsonl"]).resolve()
        rel = src.relative_to(root)
        sampled = _sample(src, general / rel, bound)
        entry.update(jsonl=str((general / rel).resolve()), jsonl_sha256=sampled["sha256"],
                     rows=sampled["kept"])
        if entry["dataset"] == "clinc/clinc_oos":
            names = src.parent / "intent_names.json"
            if not (general / rel).parent.joinpath(names.name).exists():
                shutil.copy2(names, (general / rel).parent / names.name)
        report["general"].append(sampled)
    domains = Path("clinc__oos-eval") / CLINC_DOMAINS_COMMIT / "domains.json"
    (general / domains).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / domains, general / domains)
    (general / "fetch-record.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    for src_dir in (COMPOSED, NOUL):
        dest_dir = args.out / src_dir.name
        sampled = _sample(src_dir / "examples.jsonl", dest_dir / "examples.jsonl", bound)
        _manifest(src_dir, dest_dir, sampled)
        report[src_dir.name] = sampled
    base_examples = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    base_total = int(base_examples["totals"]["examples"])
    report["base"] = {"dir": str(BASE), "examples": base_total,
                      "defect_max_rows": round(args.fraction * base_total)}
    (args.out / "subsample.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "defect_max_rows": report["base"]["defect_max_rows"],
                      "general_rows": sum(g["kept"] for g in report["general"]),
                      "composed": report[COMPOSED.name]["kept"],
                      "noul": report[NOUL.name]["kept"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
