"""Per (family, slot): what v5's template strip removed and what it left (lane L-prep2).

Reads one stripped scan and its unstripped twin over the same subsample (both written with
``--request-out``): the stripped strings and too_short counts come from the stripped scan's
attestation (``export.template_strip``); the word counts (``\\w+`` over ``str.lower``, the
containment tokenizer) are the median and 10th percentile per slot text, before and after,
read from the two requests. Output: one JSON object on stdout.

    python AUDIT/prep2-2026-10-02/slot_summary.py STRIPPED_SCAN UNSTRIPPED_SCAN
"""

from __future__ import annotations

import json
import re
import statistics
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from family_rates import MAGIC, _str

WORD = re.compile(r"\w+")


def words_by_slot(request: Path) -> dict[str, list[int]]:
    """``{family/slot: [word count per slot text]}`` over every set of a request."""
    out: dict[str, list[int]] = defaultdict(list)
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
        for _ in range(n_sets):
            _str(fh)
        (n_scans,) = struct.unpack("<I", fh.read(4))
        fh.read(9 * n_scans)
        (n_rows,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n_rows):
            fh.read(4)
            key, _identity, family, text = _str(fh), _str(fh), _str(fh), _str(fh)
            out[f"{family}/{key.rsplit('#', 1)[1]}"].append(len(WORD.findall(text.lower())))
    return out


def _stats(counts: list[int]) -> dict[str, float]:
    ordered = sorted(counts)
    return {"median": statistics.median(ordered), "p10": ordered[len(ordered) // 10]}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    stripped, plain = (Path(a) for a in argv)
    att = json.loads((stripped / "attestation.json").read_text(encoding="utf-8"))
    groups = att["export"]["template_strip"]["family_slots"]
    after = words_by_slot(stripped.with_name(stripped.name + ".request.bin"))
    before = words_by_slot(plain.with_name(plain.name + ".request.bin"))
    out = {}
    for name, group in groups.items():
        out[name] = {
            "rows": group["rows"],
            "stripped": group["stripped"],
            "option_sets": group["option_sets"],
            "too_short_after_strip_by_set": {
                s: f"{v['too_short_after_strip']}/{v['rows']}" for s, v in group["by_set"].items()
            },
            "words_before": _stats(before[name]),
            "words_after": _stats(after[name]),
        }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
