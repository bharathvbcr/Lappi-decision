"""v5 noul inputs: the allowlist the Rust generators read, and the defect-noul-v3c manifest.

A thin binding over ``qd_data.defect_class`` (the canonical split, licence and composite
rules); it computes nothing of its own.

    python tools/noul_v5_inputs.py allowlist \\
        --g6-root /Users/bharath/qd-campaign/commitpackft-g6-2026-10-02 \\
        --g6-record AUDIT/v5-plan-2026-10-02/g6-commitpackft-download.md \\
        --out data/pool/noul-v5-allowlist.json
    qd-noul-rows own-prose --allowlist data/pool/noul-v5-allowlist.json ...
    qd-noul-rows g6 --allowlist data/pool/noul-v5-allowlist.json ...
    python tools/noul_v5_inputs.py v3c --pool-dir data/pool \\
        --part defect-noul-v3b:noul-rows --part own-prose-v1:own-prose \\
        --part commitpackft-g6-v1:g6 \\
        --contrast knowledge.multiple_choice=1200,commonsense.multiple_choice=800 \\
        --contrast-seed 20260919 --out data/pool/defect-noul-v3c

``allowlist`` reads each G6 file's sha256 pin from the download record's table and refuses a
file whose bytes differ. ``v3c`` pins every part's manifest and data file by sha256; the parts
are the v3c directory's siblings, which is where the loader looks for them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_data.config import DataConfig
from qd_data.defect_class import (
    CONTRAST_TWIN_FAMILIES,
    G6_LANGUAGES,
    NOUL_COMPOSITE_SCHEMA,
    composite_digest,
    noul_v5_allowlist,
)
from qd_train.replay import write_text_atomic

REPO = Path(__file__).resolve().parents[1]
#: A row of the download record's table: ``| perl | 5,102,583 | `<sha256>` | 2,288 |``.
_PIN_ROW = re.compile(r"^\|\s*([a-z]+)\s*\|\s*[\d,]+\s*\|\s*`([0-9a-f]{64})`\s*\|\s*([\d,]+)\s*\|$")
#: What each part kind's rows are read from.
_DATA_FILE = {"noul-rows": "examples.jsonl", "g6": "examples.jsonl", "own-prose": "units.jsonl"}


def g6_pins(record: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in record.read_text(encoding="utf-8").splitlines():
        m = _PIN_ROW.match(line.strip())
        if m:
            if m.group(1) in pins:
                raise SystemExit(f"{record}: {m.group(1)} is pinned twice")
            pins[m.group(1)] = m.group(2)
    if set(pins) != set(G6_LANGUAGES):
        raise SystemExit(f"{record} pins {sorted(pins)}, not the G6 languages {G6_LANGUAGES}")
    return pins


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _part_rows(kind: str, manifest: dict) -> int:
    if kind == "own-prose":
        return int(manifest["units"]["count"])
    return int(manifest["totals"]["examples"])


def cmd_allowlist(args: argparse.Namespace) -> int:
    out = noul_v5_allowlist(
        g6_root=args.g6_root, g6_pins=g6_pins(args.g6_record), config=DataConfig(),
        repo_root=REPO,
    )
    out["g6"]["record"] = str(args.g6_record)
    write_text_atomic(args.out, json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["g6"]["counts"], indent=1, sort_keys=True))
    print(f"allowlist {args.out} sha256 {_sha(args.out)}")
    return 0


def cmd_v3c(args: argparse.Namespace) -> int:
    parts = []
    by_part: dict[str, int] = {}
    for spec in args.part:
        name, _, kind = spec.partition(":")
        if kind not in _DATA_FILE or not name or "/" in name:
            raise SystemExit(f"--part {spec!r} is not <sibling dir>:<{'|'.join(_DATA_FILE)}>")
        part_dir = args.pool_dir / name
        manifest = json.loads((part_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = _part_rows(kind, manifest)
        parts.append({
            "name": name, "kind": kind,
            "manifest_sha256": _sha(part_dir / "manifest.json"),
            "data_sha256": _sha(part_dir / _DATA_FILE[kind]), "rows": rows,
        })
        by_part[name] = rows
    per_family: dict[str, int] = {}
    for item in args.contrast.split(","):
        family, _, n = item.partition("=")
        if family not in CONTRAST_TWIN_FAMILIES or not n.isdigit() or int(n) <= 0:
            raise SystemExit(f"--contrast {item!r} is not <one of {CONTRAST_TWIN_FAMILIES}>=<n>")
        per_family[family] = int(n)
    config = DataConfig()
    manifest = {
        "schema": NOUL_COMPOSITE_SCHEMA,
        "name": args.out.name,
        "split": {"seed": config.seed, "train_fraction": config.train_fraction,
                  "val_fraction": config.val_fraction},
        "parts": parts,
        "examples_sha256": composite_digest(parts),
        "examples_sha256_basis": "qd_data.defect_class.composite_digest over the parts' pins",
        "totals": {"examples": sum(by_part.values()), "by_part": by_part},
        "contrast": {"per_family": per_family, "seed": args.contrast_seed},
        "contrast_basis": (
            "derived by qd_train.contrast.apply_contrast after dedupe, split and "
            "--exclude-identity-keys, from surviving train twins, in blake2b order of identity "
            "key; never a dedupe or split unit; only on the rebuild that applies the exclusion "
            "list (campaign/v5-preregistered.DRAFT.json data.sources[4].amended, build_order[3])"
        ),
        "layout": "every part is a sibling directory of this one; the loader reads them there",
    }
    if args.note:
        manifest["note"] = args.note
    args.out.mkdir(parents=True, exist_ok=False)
    write_text_atomic(args.out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("allowlist")
    a.add_argument("--g6-root", required=True, type=Path)
    a.add_argument("--g6-record", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.set_defaults(func=cmd_allowlist)
    v = sub.add_parser("v3c")
    v.add_argument("--pool-dir", required=True, type=Path)
    v.add_argument("--part", action="append", required=True)
    v.add_argument("--contrast", required=True)
    v.add_argument("--contrast-seed", required=True, type=int)
    v.add_argument("--note", default="")
    v.add_argument("--out", required=True, type=Path)
    v.set_defaults(func=cmd_v3c)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
