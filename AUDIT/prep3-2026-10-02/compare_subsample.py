"""Do re-created subsamples reproduce L-prep2's? (lane L-prep3, 2026-10-02; throwaway analysis).

Compares each ``subsample.json`` written by ``AUDIT/prep2-2026-10-02/make_subsample.py`` with
L-prep2's committed record of the same fraction (``AUDIT/prep2-2026-10-02/subsample-p0X.json``)
on everything but the output paths (``dest``) and the checkout root of the in-repo corpora's
paths, which differ by design: every general file's
``src``, ``read``, ``kept`` and ``sha256``, the composed and noul samples, the base corpus and
its ``defect_max_rows``, the fraction and the hash rule. Exit 0 iff all of them are equal.

    python AUDIT/prep3-2026-10-02/compare_subsample.py NEW_SUBSAMPLE_JSON PREP2_SUBSAMPLE_JSON [...]

(pairs of arguments: new, then L-prep2's)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SAMPLED_KEYS = ("src", "read", "kept", "sha256")
#: make_subsample.py reads the composed, noul and base corpora under its own checkout
#: (``REPO / "data" / "pool"``), so their absolute paths name the worktree that ran it. They
#: are compared from ``data/pool/`` on; the general caches' paths are absolute and compared
#: whole.
IN_REPO = "/data/pool/"


def _path(value: str) -> str:
    return "data/pool/" + value.split(IN_REPO, 1)[1] if IN_REPO in value else value


def _sampled(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: _path(entry[k]) if k == "src" else entry[k] for k in SAMPLED_KEYS}


def comparable(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` without the per-run output paths."""
    return {
        "fraction": record["fraction"],
        "rule": record["rule"],
        "general": [_sampled(e) for e in record["general"]],
        "commitpackft-composed-v1": _sampled(record["commitpackft-composed-v1"]),
        "defect-noul-v3b": _sampled(record["defect-noul-v3b"]),
        "base": {**record["base"], "dir": _path(record["base"]["dir"])},
    }


def main(argv: list[str]) -> int:
    if not argv or len(argv) % 2:
        raise SystemExit(__doc__)
    out = []
    ok = True
    for new_path, old_path in zip(argv[::2], argv[1::2], strict=True):
        new = comparable(json.loads(Path(new_path).read_text(encoding="utf-8")))
        old = comparable(json.loads(Path(old_path).read_text(encoding="utf-8")))
        differ = sorted(k for k in new if new[k] != old[k])
        ok = ok and not differ
        out.append({
            "new": new_path, "prep2": old_path, "identical": not differ, "differ": differ,
            "general_files": len(new["general"]),
            "general_rows_kept": sum(e["kept"] for e in new["general"]),
            "composed_kept": new["commitpackft-composed-v1"]["kept"],
            "noul_kept": new["defect-noul-v3b"]["kept"],
            "defect_max_rows": new["base"]["defect_max_rows"],
        })
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
