"""Count the span rows a shard set carries, by family and pointing/abstain. Read-only.

Reads only supervision.npz, sequence_index.json and header.json of the shard directory given
as argv[1]: no token is opened. A span row is ``slot_kind == SLOT_SPAN`` (3); an abstaining one
has ``span_target[:, 0] == SPAN_ABSTAIN`` (-2), a pointing one has a start >= 0
(``qd_train.artifacts``). The family is the row_id's second ``:`` field.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

SLOT_SPAN = 3
SPAN_ABSTAIN = -2


def count(shard_dir: Path) -> dict[str, object]:
    header = json.loads((shard_dir / "header.json").read_text(encoding="utf-8"))
    sup = np.load(shard_dir / "supervision.npz")
    kind, target = sup["slot_kind"], sup["span_target"]
    seqs = json.loads((shard_dir / "sequence_index.json").read_text(encoding="utf-8"))["sequences"]
    if len(seqs) != kind.shape[0]:
        raise SystemExit(f"{len(seqs)} index rows against {kind.shape[0]} supervision rows")
    cells: collections.Counter[str] = collections.Counter()
    for i, row in enumerate(seqs):
        if int(kind[i]) != SLOT_SPAN:
            continue
        start = int(target[i, 0])
        if start != SPAN_ABSTAIN and start < 0:
            raise SystemExit(f"row {i}: span start {start} is neither a position nor abstain")
        family = ":".join(str(row["row_id"]).split(":")[:2])
        cells[f"{family}/{'abstain' if start == SPAN_ABSTAIN else 'pointing'}"] += 1
    span = int((kind == SLOT_SPAN).sum())
    return {
        "shard_dir": str(shard_dir),
        "shard_hash": header["shard_hash"],
        "n_sequences": header["n_sequences"],
        "span_rows": span,
        "pointing": sum(v for k, v in cells.items() if k.endswith("/pointing")),
        "abstain": sum(v for k, v in cells.items() if k.endswith("/abstain")),
        "cells": dict(sorted(cells.items())),
    }


def main() -> int:
    print(json.dumps(count(Path(sys.argv[1])), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
