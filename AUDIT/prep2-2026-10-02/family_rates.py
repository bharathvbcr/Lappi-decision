"""Per-family exclusion rates of ``tools/containment_scan.py`` runs (lane L-prep2, 2026-10-02).

Re-creates L-prep's lost ``partb/family_rates.py`` as HANDOFF/prep-containment-2026-10-02.md
reports its numbers: a family's rate is the share of its distinct TRAIN identity keys that
``exclusions.txt`` lists. CLINC is also reported as one group, the union of the intent.*
families' keys (one utterance's identity key is shared by all four, so one hit excludes it
from all four), which is L-prep's "CLINC 210 of 1,044 utterance keys".

Per family it also reports ``hit_by_own_rows``: the keys whose OWN family's train row is the
source of an enforced pair (train -> val or train -> heldout), which says which family drives
a group's exclusions; and the enforced pairs per (source family, target set).

Input: scan directories written with ``--request-out DIR.request.bin`` (the request is read
for every train row's identity key and family). Output: one JSON object on stdout.

    python AUDIT/prep2-2026-10-02/family_rates.py SCAN_DIR [SCAN_DIR ...]
"""

from __future__ import annotations

import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO

MAGIC = b"QDPCTIN1"
ENFORCED_TARGETS = ("val", "heldout")
GROUPS = {
    "CLINC (intent.*)": lambda f: f.startswith("intent."),
    "MMLU (knowledge.multiple_choice)": lambda f: f == "knowledge.multiple_choice",
    "CSQA (commonsense.multiple_choice)": lambda f: f == "commonsense.multiple_choice",
    "SQuAD answer_span": lambda f: f == "qa.answer_span",
    "code.defect_class": lambda f: f == "code.defect_class",
}


def _str(fh: BinaryIO) -> str:
    (n,) = struct.unpack("<I", fh.read(4))
    return fh.read(n).decode("utf-8")


def train_rows(request: Path) -> list[tuple[str, str, str, str]]:
    """``(key, identity, family, text)`` of every train row of a ``QDPCTIN1`` request."""
    out: list[tuple[str, str, str, str]] = []
    with request.open("rb") as fh:
        if fh.read(8) != MAGIC:
            raise SystemExit(f"{request}: not a QDPCTIN1 request")
        fh.read(12)
        _str(fh)
        _str(fh)
        (n_checks,) = struct.unpack("<I", fh.read(4))
        for _ in range(n_checks):
            _str(fh)
            fh.read(1)
            _str(fh)
        (n_sets,) = struct.unpack("<I", fh.read(4))
        names = [_str(fh) for _ in range(n_sets)]
        (n_scans,) = struct.unpack("<I", fh.read(4))
        fh.read(9 * n_scans)
        (n_rows,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n_rows):
            (i,) = struct.unpack("<I", fh.read(4))
            row = (_str(fh), _str(fh), _str(fh), _str(fh))
            if names[i] == "train":
                out.append(row)
    return out


def rates(scan: Path) -> dict[str, object]:
    request = scan.with_name(scan.name + ".request.bin")
    rows = train_rows(request)
    excluded = set((scan / "exclusions.txt").read_text(encoding="utf-8").splitlines())
    att = json.loads((scan / "attestation.json").read_text(encoding="utf-8"))
    keys_by_family: dict[str, set[str]] = defaultdict(set)
    for _key, identity, family, _text in rows:
        keys_by_family[family].add(identity)
    own_hit: dict[str, set[str]] = defaultdict(set)
    pairs: Counter[str] = Counter()
    with (scan / "pairs.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] == "train" and f[4] in ENFORCED_TARGETS:
                own_hit[f[3]].add(f[2])
                pairs[f"{f[3]} -> {f[4]} ({f[6]})"] += 1
    families = {}
    for family, keys in sorted(keys_by_family.items()):
        families[family] = {
            "train_keys": len(keys),
            "excluded": len(keys & excluded),
            "rate": round(len(keys & excluded) / len(keys), 4) if keys else None,
            "hit_by_own_rows": len(own_hit[family]),
        }
    groups = {}
    for name, member in GROUPS.items():
        keys = set().union(*(k for f, k in keys_by_family.items() if member(f)))
        groups[name] = {
            "train_keys": len(keys),
            "excluded": len(keys & excluded),
            "rate": round(len(keys & excluded) / len(keys), 4) if keys else None,
        }
    strip = att["export"].get("template_strip", {})
    return {
        "scan": str(scan),
        "template_strip_applied": strip.get("applied"),
        "n_pairs": att["n_pairs"],
        "n_exclusions": att["n_exclusions"],
        "clean": att["clean"],
        "sets": {k: {"rows": v["rows"], "too_short": v["too_short"]}
                 for k, v in att["sets"].items()},
        "groups": groups,
        "families": families,
        "enforced_pairs": dict(sorted(pairs.items())),
    }


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    print(json.dumps([rates(Path(a)) for a in argv], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
