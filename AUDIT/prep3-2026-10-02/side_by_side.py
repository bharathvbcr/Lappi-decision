"""Version 1 against version 2 of the containment strip, on the 1/2/5% subsamples (lane L-prep3).

Throwaway analysis; it moves nothing. Inputs: L-prep2's scans (``scan-p0X-nostrip``,
``scan-p0X-strip``: version 1) and this lane's ``scan-p0X-v2``, all with their
``.request.bin`` kept, over byte-identical subsamples (``subsample-reproduction.json``).

Writes one JSON object: per size, per scan, ``AUDIT/prep2-2026-10-02/family_rates.py``'s
groups (CLINC per distinct identity key, the union of intent.*), pairs, keys; the version-2
attestation's key_ii_blind per set and distinct option values per (2b) family-slot; and
whether the pair rows with no intent.* row on either side are byte-identical between
version 1 and version 2 (every scan in ``pairs.tsv``, enforced or not).

    python AUDIT/prep3-2026-10-02/side_by_side.py PREP2_SCAN_ROOT PREP3_SCAN_ROOT
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "AUDIT" / "prep2-2026-10-02"))

from family_rates import rates  # noqa: E402

SIZES = ("p01", "p02", "p05")
INTENT = "intent."


def _non_intent_pairs(scan: Path) -> list[str]:
    with (scan / "pairs.tsv").open(encoding="utf-8") as fh:
        next(fh)
        return [line for line in fh
                if not (line.split("\t")[3].startswith(INTENT)
                        or line.split("\t")[6].startswith(INTENT))]


def _summary(scan: Path) -> dict[str, Any]:
    r = rates(scan)
    return {
        "template_strip_applied": r["template_strip_applied"],
        "pairs": r["n_pairs"],
        "identity_keys_excluded": r["n_exclusions"],
        "clean": r["clean"],
        "groups": {g: f"{v['excluded']}/{v['train_keys']}" for g, v in r["groups"].items()},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    prep2, prep3 = (Path(a) for a in argv)
    out: dict[str, Any] = {}
    for size in SIZES:
        v1_plain = prep2 / f"scan-{size}-nostrip"
        v1 = prep2 / f"scan-{size}-strip"
        v2 = prep3 / f"scan-{size}-v2"
        strip = json.loads((v2 / "attestation.json").read_text(encoding="utf-8"))[
            "export"]["template_strip"]
        a, b = _non_intent_pairs(v1), _non_intent_pairs(v2)
        out[size] = {
            "unstripped (L-prep2)": _summary(v1_plain),
            "version 1 (L-prep2)": _summary(v1),
            "version 2 (L-prep3)": _summary(v2),
            "version 2 key_ii_blind per set (identity keys)": {
                s: b_["n"] for s, b_ in strip["key_ii_blind"].items()},
            "version 2 too_short_after_strip per intent.* family-slot per set": {
                name: {s: v["too_short_after_strip"] for s, v in g["by_set"].items()}
                for name, g in strip["family_slots"].items() if name.startswith(INTENT)},
            "version 2 distinct option values per (2b) family-slot": {
                name: g["closed_vocabulary"]["distinct_option_values"]
                for name, g in strip["family_slots"].items() if g["closed_vocabulary"]},
            "non_intent_pair_rows": {
                "version 1": len(a), "version 2": len(b), "byte_identical": a == b,
                "sha256": hashlib.sha256("".join(b).encode("utf-8")).hexdigest(),
            },
        }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if all(v["non_intent_pair_rows"]["byte_identical"] for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
