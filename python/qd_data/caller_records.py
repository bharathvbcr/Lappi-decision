"""The content guard that keeps an app's caller records out of training (CLAUDE.md rule 3).

Apps write what they saw and chose around a decision to
``.../Lappi/heldout/caller-records/<app>/<day>.jsonl`` (``docs/caller-contract.md``). The path
layer of the door (a ``heldout`` segment) refuses that store; a *copy* placed anywhere else carries
no marker. This module reads the bytes: any line whose parsed ``"record"`` starts with
``lappi.caller_record`` refuses the whole file, wherever it lives and whatever it is called.

**One rule, three implementations, one oracle.** ``crates/qd-train/src/held_out.rs`` (layer 5) and
``crates/qd-prep/src/heldout.rs`` mirror this module, and all three run every case of
``fixtures/caller/guard-cases.json``; a behaviour change is a change to that file first. The three
verdicts:

* ``record``: a line is a caller record: a parsed object whose ``"record"`` starts with the
  prefix; or,
  for a line that does not parse or whose parse hides it (a duplicated key, a BOM, damage, a JSON
  array holding one), a line carrying both the ``"record"`` key text and the prefix. The text test
  errs toward refusing: this only ever narrows admission.
* ``unscannable``: the bytes cannot be read as lines of text, so they are refused, not passed. A
  gzip, zstd or UTF-16 (BOM) magic, or a file longer than the byte budget. A check that could not
  run is never the same result as a check that passed.
* ``clean``: everything else. A line longer than the line budget is skipped unparsed (the contract
  writes no record over 64 KiB). The budget is on the line's content, without its ``\\n`` or
  ``\\r\\n``, so a record of exactly the bound is scanned.

There is no sniff of the first byte: a file whose first line is garbage, a comment, a JSON string
or an array is scanned like any other.

Where it is called: every place a pool or source file is opened for rows on the way to shards
(``qd_data.loaders.read_jsonl``, ``qd_data.decisions.load_decision_pool``, the
``qd_data.defect_class`` readers,
``qd_train.mutate_adapter``, ``tools/real_tokenizer_pipeline.py``). A path that is not a regular
file (missing, a directory) has no bytes to scan and is left to the read that follows, which
refuses it by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO, Final

__all__ = [
    "CALLER_RECORD_PREFIX",
    "MAX_GUARD_BYTES",
    "MAX_GUARD_LINE_BYTES",
    "CallerRecordRefusal",
    "assert_no_caller_records",
    "is_caller_record_line",
    "scan_stream",
]

#: ``qd_runtime::caller_record::RECORD``. A prefix, so a later version is refused too.
CALLER_RECORD_PREFIX: Final[str] = "lappi.caller_record"
#: A longer line is skipped unparsed. The contract writes no record over 64 KiB.
MAX_GUARD_LINE_BYTES: Final[int] = 1 << 20
#: A file longer than this is refused as unscanned. Mirrors the Rust guards.
MAX_GUARD_BYTES: Final[int] = 8 << 30

_UNSCANNABLE_MAGIC: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\x1f\x8b", "gzip"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\xff\xfe", "UTF-16 (little-endian BOM)"),
    (b"\xfe\xff", "UTF-16 (big-endian BOM)"),
)
_KEY_TEXT: Final[bytes] = b'"record"'
_PREFIX_BYTES: Final[bytes] = CALLER_RECORD_PREFIX.encode("ascii")


class CallerRecordRefusal(ValueError):
    """A file the door refuses: a caller record, or bytes the guard could not scan.

    ``verdict`` is ``"record"`` or ``"unscannable"``, the oracle's two refusing words.
    """

    def __init__(self, verdict: str, detail: str) -> None:
        super().__init__(detail)
        self.verdict = verdict


def is_caller_record_line(line: bytes, line_bytes: int = MAX_GUARD_LINE_BYTES) -> bool:
    """Whether one line (without its terminator) is a caller record. See the module docstring."""
    line = line.rstrip(b"\r\n")
    if len(line) > line_bytes:
        return False
    try:
        obj = json.loads(line)
    except (ValueError, RecursionError):
        obj = None
    if isinstance(obj, dict):
        record = obj.get("record")
        if isinstance(record, str) and record.startswith(CALLER_RECORD_PREFIX):
            return True
    return _KEY_TEXT in line and _PREFIX_BYTES in line


def scan_stream(
    fh: BinaryIO,
    *,
    label: str,
    max_bytes: int = MAX_GUARD_BYTES,
    line_bytes: int = MAX_GUARD_LINE_BYTES,
) -> None:
    """Scan a binary stream, raising :class:`CallerRecordRefusal` unless it is clean."""
    start = fh.tell()
    head = fh.read(4)
    fh.seek(start)
    for magic, name in _UNSCANNABLE_MAGIC:
        if head.startswith(magic):
            raise CallerRecordRefusal(
                "unscannable",
                f"{label}: {name} bytes cannot be scanned for caller records, so the file is "
                "refused rather than passed as clean (CLAUDE.md rule 3)",
            )
    total = 0
    line_no = 0
    # Room for the content, a '\r' and a '\n': a record of exactly `line_bytes` is scanned.
    read_cap = line_bytes + 2
    while True:
        line = fh.readline(read_cap)
        if not line:
            return
        line_no += 1
        total += len(line)
        if not line.endswith(b"\n") and len(line) == read_cap:
            # Over-long: drain the rest of the line unparsed, still counting it.
            while True:
                rest = fh.readline(1 << 20)
                total += len(rest)
                if not rest or rest.endswith(b"\n") or total > max_bytes:
                    break
        elif is_caller_record_line(line, line_bytes):
            raise CallerRecordRefusal(
                "record",
                f"{label}: line {line_no} is a caller record ({CALLER_RECORD_PREFIX}). Caller "
                "records are never training data wherever the file lives; the whole file is "
                "refused (CLAUDE.md rule 3, docs/caller-contract.md)",
            )
        if total > max_bytes:
            raise CallerRecordRefusal(
                "unscannable",
                f"{label}: longer than the {max_bytes}-byte budget of the caller-record guard, "
                "so it was not fully scanned; refused rather than passed as clean",
            )


def assert_no_caller_records(
    path: Path | str,
    *,
    max_bytes: int = MAX_GUARD_BYTES,
    line_bytes: int = MAX_GUARD_LINE_BYTES,
) -> None:
    """Refuse ``path`` if it holds a caller record or cannot be scanned."""
    p = Path(path)
    if not p.is_file():
        return
    with p.open("rb") as fh:
        scan_stream(fh, label=str(p), max_bytes=max_bytes, line_bytes=line_bytes)
