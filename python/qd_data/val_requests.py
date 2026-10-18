"""Write a split's rows as the wire requests ``qd-release-check build-requests`` selects from
(GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17).

The builder itself is Rust (``crates/qd-preflight/src/release_check.rs``), beside the check that
reads what it builds; the rows exist only as Python objects, so this is the one seam that turns
them into JSONL. One line per row, ``Request.to_wire()`` and nothing else: no gold answer ever
reaches a replay request, and the held-out split is refused here as everywhere else a rows file
is opened (CLAUDE.md rule 3 is about training, but a release is replayed on val).

``main`` is the command ``tools/release_replay.sh`` (``make release-replay``) runs first: it reads
a val-rows file and writes the requests file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import Request

__all__ = ["ValRow", "main", "read_val_rows", "write_val_requests"]

#: Exit codes of ``main``: ``tools/release_replay.sh`` reads 1 as FAILED and anything else as NotRun.
EXIT_REFUSED = 1
EXIT_NOT_RUN = 3
#: A bound on the rows file, so a wrong path cannot be slurped whole.
MAX_VAL_ROWS_BYTES = 1 << 30


class _HasRequest(Protocol):
    """What the writer needs of a row; ``qd_data.rows.DataRow`` and :class:`ValRow` both fit."""

    @property
    def row_id(self) -> str: ...

    @property
    def request(self) -> Request: ...


@dataclass(frozen=True, slots=True)
class ValRow:
    """One line of a val-rows file: the row's id, its split and its request, and no gold."""

    row_id: str
    split: str
    request: Request


def write_val_requests(rows: Iterable[_HasRequest], path: Path, *, split_of: dict[str, str]) -> int:
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


def read_val_rows(path: Path) -> list[ValRow]:
    """Read a val-rows file: JSONL, each line ``{"row_id", "split", "request": <wire request>}``.

    Anything malformed raises ``ValueError`` naming the line; a duplicate ``row_id`` is refused,
    because two rows under one id would make ``split_of`` ambiguous.
    """
    if path.stat().st_size > MAX_VAL_ROWS_BYTES:
        raise ValueError(f"{path} is over {MAX_VAL_ROWS_BYTES} bytes")
    rows: list[ValRow] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            row_id, split = raw["row_id"], raw["split"]
            if not isinstance(row_id, str) or not isinstance(split, str):
                raise TypeError("row_id and split must be strings")
            request = Request.from_wire(raw["request"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{n}: not a val row ({exc})") from exc
        if row_id in seen:
            raise ValueError(f"{path}:{n}: row_id {row_id!r} appears twice")
        seen.add(row_id)
        rows.append(ValRow(row_id=row_id, split=split, request=request))
    if not rows:
        raise ValueError(f"{path} holds no rows")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Turn a val-rows file into the wire requests ``qd-release-check build-requests`` reads.

    A row whose split is not ``val`` is a refusal (exit 1, nothing written); an unreadable input
    is exit 3.
    """
    parser = argparse.ArgumentParser(description=(main.__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--val-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = read_val_rows(args.val_rows)
    except (OSError, ValueError) as exc:
        print(f"NotRun: val_requests - {exc}", file=sys.stderr)
        return EXIT_NOT_RUN
    try:
        n = write_val_requests(rows, args.out, split_of={r.row_id: r.split for r in rows})
    except ValueError as exc:
        print(f"FAILED: val_requests - {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"wrote {args.out} ({n} requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
