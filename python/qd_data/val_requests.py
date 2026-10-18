"""Write a split's rows as the wire requests ``qd-release-check build-requests`` selects from
(GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17).

The builder itself is Rust (``crates/qd-preflight/src/release_check.rs``), beside the check that
reads what it builds; the rows exist only as Python objects, so this is the one seam that turns
them into JSONL. One line per row, ``Request.to_wire()`` and nothing else: no gold answer ever
reaches a replay request, and the held-out split is refused here as everywhere else a rows file
is opened (CLAUDE.md rule 3 is about training, but a release is replayed on val).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from .rows import DataRow

__all__ = ["write_val_requests"]


def write_val_requests(rows: Iterable[DataRow], path: Path, *, split_of: dict[str, str]) -> int:
    """Write one wire request per row to ``path`` (atomically) and return how many.

    ``split_of`` maps ``row_id`` to its split, from the manifest; a row not in ``val`` is
    refused, as is a row the map does not know, so a train or held-out row cannot be replayed
    as if it were unseen.
    """
    lines: list[str] = []
    for row in rows:
        split = split_of.get(row.row_id)
        if split != "val":
            raise ValueError(
                f"row {row.row_id!r} is in split {split!r}, not 'val'; a release is replayed on "
                "val rows only"
            )
        lines.append(json.dumps(row.request.to_wire(), sort_keys=True, separators=(",", ":")))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return len(lines)
