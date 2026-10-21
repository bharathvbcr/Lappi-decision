"""Write a split's rows as the wire requests ``qd-release-check build-requests`` selects from
(GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17).

The builder itself is Rust (``crates/qd-preflight/src/release_check.rs``), beside the check that
reads what it builds; the rows exist only as Python objects, so this is the one seam that turns
them into JSONL. One line per row, ``Request.to_wire()`` and nothing else: no gold answer ever
reaches a replay request, and the held-out split is refused here as everywhere else a rows file
is opened (CLAUDE.md rule 3 is about training, but a release is replayed on val).

``main`` is the command ``tools/release_replay.sh`` (``make release-replay``) runs first: it reads
a val-rows file and writes the requests file.

The val-rows file itself is written by :func:`write_val_rows`
(GAP-W9B-NO-TRACKED-VAL-ROWS-FILE-WRITER-2026-10-18), from the rows the pipeline rebuilds and the
val manifest it wrote beside its shards. A sidecar (:func:`binding_path`) records the val
manifest's sha256, the ``data_snapshot_hash``, the train manifest's sha256 (what a release
records as ``train_manifest.ledger_binding.train_manifest_sha256``) and the rows file's own
sha256; ``main`` refuses a file whose sidecar is missing or names another shard set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qd_train.data_access import assert_path_not_held_out

from .config import DataConfig
from .manifest import Manifest
from .rows import DataRow, row_content_hash
from .schema import Request

__all__ = [
    "ValRow",
    "ValRowsNotRun",
    "binding_path",
    "check_binding",
    "main",
    "read_val_rows",
    "write_val_requests",
    "write_val_rows",
]

#: Exit codes of ``main``: ``tools/release_replay.sh`` reads 1 as FAILED, anything else as
#: NotRun.
EXIT_REFUSED = 1
EXIT_NOT_RUN = 3
#: A bound on the rows file, so a wrong path cannot be slurped whole.
MAX_VAL_ROWS_BYTES = 1 << 30
#: Bound on the rows a val-rows file may hold; a val split over it fails closed.
MAX_VAL_ROWS = 1_000_000
BINDING_SCHEMA = "qd-val-rows-binding/v1"


class ValRowsNotRun(ValueError):
    """The writer or the check could not run (an empty val split, no binding): NotRun, not a
    pass and not a refusal."""


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


def binding_path(rows_path: Path) -> Path:
    return rows_path.with_name(rows_path.name + ".binding.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_val_rows(
    rows: Iterable[DataRow],
    *,
    val_manifest: Path,
    train_manifest: Path,
    out: Path,
    config: DataConfig,
    repo_root: Path,
    max_rows: int = MAX_VAL_ROWS,
) -> int:
    """Write the VAL_ROWS file for one shard set and bind it to that set; return the row count.

    Reads the val manifest only: both manifest paths go through
    ``qd_train.data_access.assert_path_not_held_out`` before a byte is read, and a manifest that
    does not declare split ``val`` is refused. ``rows`` (the pipeline's rebuilt val rows) must be
    exactly the manifest's entries, by id and content hash; the output order is the manifest's,
    so it is a function of the shard set alone. An empty val split is :class:`ValRowsNotRun`.
    """
    assert_path_not_held_out(val_manifest, config=config, repo_root=repo_root)
    assert_path_not_held_out(train_manifest, config=config, repo_root=repo_root)
    manifest = Manifest.read(val_manifest)
    if manifest.split != "val":
        raise ValueError(f"{val_manifest} declares split {manifest.split!r}, not 'val'")
    if not manifest.entries:
        raise ValRowsNotRun(f"{val_manifest} lists no val rows: the val split is empty")
    if len(manifest.entries) > max_rows:
        raise ValueError(
            f"{val_manifest} lists {len(manifest.entries)} rows, over the cap {max_rows}"
        )
    by_id = {r.row_id: r for r in rows}
    if set(by_id) != {e.row_id for e in manifest.entries}:
        raise ValueError(f"the rows are not {val_manifest}'s entries (ids differ)")
    lines: list[str] = []
    for entry in manifest.entries:
        row = by_id[entry.row_id]
        if row_content_hash(row) != entry.content_hash:
            raise ValueError(f"row {entry.row_id!r}'s content hash is not {val_manifest}'s")
        wire = {"row_id": row.row_id, "split": "val", "request": row.request.to_wire()}
        lines.append(json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n")
    binding = {
        "schema": BINDING_SCHEMA,
        "val_manifest_sha256": _sha256_file(val_manifest),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "data_snapshot_hash": manifest.snapshot_hash(),
        "n_rows": len(lines),
        "rows_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
    }
    rows_text = "".join(lines)
    binding_text = json.dumps(binding, indent=2, sort_keys=True) + "\n"
    side = binding_path(out)
    # An existing file is overwritten only by the bytes it already holds: a rows file and its
    # binding are the evidence a release was replayed on this shard set, so a rerun that would
    # change either (another seed, another val split) names a new path instead of replacing it.
    for path, text in ((out, rows_text), (side, binding_text)):
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(
                f"{path} exists and differs from what this shard set writes: not overwritten "
                "(write to a new path, or remove it on purpose)"
            )
    # Rows first: a write cut short between the two leaves a rows file with no binding, which
    # check_binding reports as NotRun, and a rerun completes the pair.
    if not out.exists():
        _write_atomic(out, rows_text)
    if not side.exists():
        _write_atomic(side, binding_text)
    return len(lines)


def check_binding(rows_path: Path, *, train_manifest_sha256: str) -> None:
    """Refuse a val-rows file not written by :func:`write_val_rows` for the shard set whose
    train manifest has this sha256 (the one the release records). No sidecar is NotRun."""
    side = binding_path(rows_path)
    try:
        binding = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValRowsNotRun(
            f"{rows_path} has no readable binding ({side}): it was not written by the tracked "
            f"val-rows writer, so which shard set it came from is unknown ({exc})"
        ) from exc
    if not isinstance(binding, dict) or binding.get("schema") != BINDING_SCHEMA:
        raise ValRowsNotRun(f"{side} is not a {BINDING_SCHEMA} binding")
    if binding.get("train_manifest_sha256") != train_manifest_sha256:
        raise ValueError(
            f"{rows_path} is bound to train_manifest_sha256 "
            f"{binding.get('train_manifest_sha256')!r}, the release records "
            f"{train_manifest_sha256!r}: another shard set's val rows"
        )
    if _sha256_file(rows_path) != binding.get("rows_sha256"):
        raise ValueError(f"{rows_path} does not match the rows_sha256 its binding records")


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
    parser.add_argument(
        "--expect-train-manifest-sha256", required=True,
        help="the train manifest sha256 the release records; the val rows' binding must name it",
    )
    args = parser.parse_args(argv)
    try:
        check_binding(args.val_rows, train_manifest_sha256=args.expect_train_manifest_sha256)
    except ValRowsNotRun as exc:
        print(f"NotRun: val_requests - {exc}", file=sys.stderr)
        return EXIT_NOT_RUN
    except (OSError, ValueError) as exc:
        print(f"FAILED: val_requests - {exc}", file=sys.stderr)
        return EXIT_REFUSED
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
