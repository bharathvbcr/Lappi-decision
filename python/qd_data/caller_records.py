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

* ``record``: a line is a caller record when it is JSONL whose TOP-LEVEL ``"record"`` key
  (any of them, so a duplicated key cannot hide one) holds a string starting with the prefix, or
  a JSON array holding such an object. A line that parses and is not that is clean, however
  much text it quotes: a document that mentions the format, a nested ``record`` key, source
  code. Only a line that does NOT parse (a BOM, damage) and starts, in column 0 after an optional
  BOM, with ``{`` or ``[`` is read as text, and only the adjacency ``"record"`` ``:``
  ``"lappi.caller_record`` counts, so a Rust or Python line that merely contains both strings is
  clean. This only ever narrows admission.
* ``unscannable``: the bytes cannot be read as lines of text, so they are refused, not passed. A
  gzip, zstd or UTF-16 (BOM) magic, a file longer than the byte budget, a bare ``\\r`` (one not
  followed by ``\\n``: the readers split on it in text mode and not otherwise, so a record could
  sit behind it), any other character ``str.splitlines()`` splits on that a reader could honour
  (``\\x0b``, ``\\x0c``, ``\\x1c``-``\\x1e``, anywhere in a line: a raw control character is
  not legal JSON, so no row holds one; and ``\\x85``, U+2028 and U+2029 OUTSIDE a JSON string: a
  raw U+2028 inside a string value is legal JSON and legitimate data, so it is passed), or a path
  that is neither a regular file nor a directory (a FIFO, a device has no stable bytes to scan).
  A check that could not run is never the same result as a check that passed.
* ``clean``: everything else. A line longer than the line budget is skipped unparsed (the contract
  writes no record over 64 KiB). The budget is on the line's content, without its ``\\n`` or
  ``\\r\\n``, so a record of exactly the bound is scanned.

**A line is what ends in ``\\n``, with an optional preceding ``\\r`` stripped.** Every reader on
the way to training splits that way (:func:`jsonl_lines`, :func:`split_jsonl_text`); the
separator refusal above is defence in depth for a reader that does not.

There is no sniff of the first byte: a file whose first line is garbage, a comment, a JSON string
or an array is scanned like any other.

Where it is called: every place a pool or source file is opened for rows on the way to shards
(``qd_data.loaders.read_jsonl``, ``qd_data.decisions.load_decision_pool``, the
``qd_data.defect_class`` readers,
``qd_train.mutate_adapter``, ``tools/real_tokenizer_pipeline.py``). A missing path or a
directory has no bytes to scan and is left to the read that follows, which refuses it by name;
any other non-regular path is refused as unscannable.
"""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, Final

__all__ = [
    "CALLER_RECORD_PREFIX",
    "MAX_GUARD_BYTES",
    "MAX_GUARD_LINE_BYTES",
    "CallerRecordRefusal",
    "assert_no_caller_records",
    "is_caller_record_line",
    "jsonl_lines",
    "scan_stream",
    "split_jsonl_text",
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
_BOM: Final[bytes] = b"\xef\xbb\xbf"
#: A damaged line is a record only when the key and the prefix sit next to each other.
_ADJACENT: Final[re.Pattern[bytes]] = re.compile(
    rb'"record"[ \t\r\n]*:[ \t\r\n]*"' + re.escape(CALLER_RECORD_PREFIX.encode("ascii"))
)


class _Pairs(list[tuple[str, object]]):
    """A parsed JSON object with every key kept, duplicates included."""


class CallerRecordRefusal(ValueError):
    """A file the door refuses: a caller record, or bytes the guard could not scan.

    ``verdict`` is ``"record"`` or ``"unscannable"``, the oracle's two refusing words.
    """

    def __init__(self, verdict: str, detail: str) -> None:
        super().__init__(detail)
        self.verdict = verdict


def _pairs_record(pairs: _Pairs) -> bool:
    return any(
        key == "record" and isinstance(value, str) and value.startswith(CALLER_RECORD_PREFIX)
        for key, value in pairs
    )


def is_caller_record_line(line: bytes, line_bytes: int = MAX_GUARD_LINE_BYTES) -> bool:
    """Whether one line (without its terminator) is a caller record. See the module docstring."""
    line = line.removesuffix(b"\n").removesuffix(b"\r")
    if len(line) > line_bytes:
        return False
    try:
        obj = json.loads(line.decode("utf-8"), object_pairs_hook=_Pairs)
    except (ValueError, RecursionError):
        # Not parseable (a BOM, damage, invalid UTF-8): read as text, narrowly.
        head = line.removeprefix(_BOM)
        return head[:1] in (b"{", b"[") and _ADJACENT.search(line) is not None
    if isinstance(obj, _Pairs):
        return _pairs_record(obj)
    if isinstance(obj, list):
        return any(isinstance(e, _Pairs) and _pairs_record(e) for e in obj)
    return False


def _bare_cr(prev_cr: bool, chunk: bytes) -> tuple[bool, bool]:
    """Whether ``chunk`` holds a ``\\r`` not followed by ``\\n``, and whether it ends in one.

    ``prev_cr`` is a ``\\r`` that ended the previous chunk of the same line.
    """
    if prev_cr and chunk[:1] != b"\n":
        return True, False
    for i in range(len(chunk) - 1):
        if chunk[i] == 13 and chunk[i + 1] != 10:
            return True, False
    return False, chunk.endswith(b"\r")


def _refuse_bare_cr(label: str, line_no: int) -> CallerRecordRefusal:
    return CallerRecordRefusal(
        "unscannable",
        f"{label}: line {line_no} holds a bare carriage return; the readers split lines "
        "differently (text mode on '\\r', '\\n' only) so a record could sit behind it. Refused "
        "rather than passed as clean (CLAUDE.md rule 3)",
    )


#: A raw control character ``str.splitlines()`` splits on. Not legal inside a JSON string either,
#: so it is refused wherever it sits.
_CONTROL_BREAK: Final[re.Pattern[bytes]] = re.compile(rb"[\x0b\x0c\x1c-\x1e]")
_CONTROL_BYTES: Final[bytes] = b"\x0b\x0c\x1c\x1d\x1e"
#: U+0085, U+2028, U+2029 in UTF-8. Legal inside a JSON string, a line break outside one.
_UNICODE_BREAKS: Final[tuple[bytes, ...]] = (b"\xc2\x85", b"\xe2\x80\xa8", b"\xe2\x80\xa9")
_ESCAPED_BYTE: Final[re.Pattern[bytes]] = re.compile(rb"\\.", re.DOTALL)


class _BreakScan:
    """Finds a ``str.splitlines()`` separator the guard does not split on, in one line.

    Fed the line in chunks (an over-long line is read in pieces) and keeps the JSON string state
    across them: a backslash escapes the byte after it, a quote toggles the string. The common
    case (no candidate byte sequence in the chunk) costs a few C-level scans; the byte walk runs
    only on a chunk that holds U+0085, U+2028 or U+2029.
    """

    __slots__ = ("escaped", "in_string", "tail")

    def __init__(self) -> None:
        self.in_string = False
        self.escaped = False
        #: The last two bytes of the previous chunk, so a sequence split across chunks is seen.
        self.tail = b""

    def feed(self, chunk: bytes) -> str | None:
        """What separator ``chunk`` holds outside a JSON string (control characters: anywhere)."""
        control = _CONTROL_BREAK.search(chunk)
        if control is not None:
            return f"control character U+{control.group()[0]:04X}"
        buf = self.tail + chunk
        found = None
        if any(brk in buf for brk in _UNICODE_BREAKS):
            found = self._walk(buf, len(self.tail))
        else:
            self._advance(chunk)
        self.tail = buf[-2:]
        return found

    def _advance(self, chunk: bytes) -> None:
        if self.escaped and chunk:
            chunk = chunk[1:]
            self.escaped = False
        rest = _ESCAPED_BYTE.sub(b"", chunk)
        if rest.endswith(b"\\"):
            self.escaped = True
            rest = rest[:-1]
        if rest.count(b'"') & 1:
            self.in_string = not self.in_string

    def _walk(self, buf: bytes, start: int) -> str | None:
        for i in range(start, len(buf)):
            b = buf[i]
            if not self.in_string:
                if b == 0x85 and i >= 1 and buf[i - 1] == 0xC2:
                    return "U+0085"
                if b in (0xA8, 0xA9) and i >= 2 and buf[i - 2 : i] == b"\xe2\x80":
                    return "U+2028" if b == 0xA8 else "U+2029"
            if self.escaped:
                self.escaped = False
            elif b == 0x5C:
                self.escaped = True
            elif b == 0x22:
                self.in_string = not self.in_string
        return None


def _maybe_break(line: bytes) -> bool:
    """The cheap test a line must pass before :class:`_BreakScan` walks it (C-level scans only)."""
    return len(line.translate(None, _CONTROL_BYTES)) != len(line) or any(
        brk in line for brk in _UNICODE_BREAKS
    )


def _refuse_foreign_break(label: str, line_no: int, what: str) -> CallerRecordRefusal:
    return CallerRecordRefusal(
        "unscannable",
        f"{label}: line {line_no} holds {what}, which str.splitlines() splits on and the "
        "guard does not; a reader that did could read a record behind it. Refused rather than "
        "passed as clean (CLAUDE.md rule 3)",
    )


def split_jsonl_text(text: str) -> list[str]:
    """``text`` as the guard splits it: on ``\\n`` alone, a preceding ``\\r`` stripped.

    The canonical split for a JSONL reader that holds the whole text. Not ``str.splitlines()``
    (which also splits on ``\\r``, ``\\x0b``, ``\\x0c``, ``\\x1c``-``\\x1e``, U+0085, U+2028 and
    U+2029, so a row whose string value holds one is cut in two and a record can be hidden
    behind one) and not a universal-newline read (which splits on a bare ``\\r``). Terminators
    are dropped, and a final ``\\n`` does not open an empty last line.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def jsonl_lines(path: Path | str) -> Iterator[str]:
    """Each line of a UTF-8 JSONL file, split exactly as the guard splits it, streaming.

    A line keeps its ``\\n`` and a ``\\r\\n`` ending reads as ``\\n``, which is what a
    universal-newline text read yielded before, so a row-byte bound sees the same length. A bare
    ``\\r`` is data here (the guard refuses a file that holds one). The one iterator every
    JSONL reader on the way to training uses; a reader that opens the file itself in text mode
    splits on more than the guard does.
    """
    with Path(path).open(encoding="utf-8", newline="\n") as fh:
        for line in fh:
            yield line[:-2] + "\n" if line.endswith("\r\n") else line


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
        overlong = not line.endswith(b"\n") and len(line) == read_cap
        # A trailing CRLF's '\r' is not bare; the rest of the line may hold none.
        bare, carry = _bare_cr(False, line)
        if bare:
            raise _refuse_bare_cr(label, line_no)
        # A line is walked only when a C-level scan finds a candidate; an over-long one is always
        # walked, because the string state must carry across the reads that make it up.
        breaks = _BreakScan() if overlong else None
        foreign = (
            breaks.feed(line) if breaks is not None
            else _BreakScan().feed(line) if _maybe_break(line) else None
        )
        if foreign is not None:
            raise _refuse_foreign_break(label, line_no, foreign)
        if breaks is not None:
            # Over-long: drain the rest of the line unparsed, still counting it.
            while True:
                rest = fh.readline(1 << 20)
                total += len(rest)
                bare, carry = _bare_cr(carry, rest)
                if bare or (not rest and carry):
                    raise _refuse_bare_cr(label, line_no)
                foreign = breaks.feed(rest)
                if foreign is not None:
                    raise _refuse_foreign_break(label, line_no, foreign)
                if not rest or rest.endswith(b"\n") or total > max_bytes:
                    break
        elif carry:
            # A lone '\r' ended the last line of the stream.
            raise _refuse_bare_cr(label, line_no)
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
    try:
        mode = p.stat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        return
    if not stat.S_ISREG(mode):
        raise CallerRecordRefusal(
            "unscannable",
            f"{p}: not a regular file (a FIFO, a device or a socket has no stable bytes to "
            "scan), so it is refused rather than passed as clean (CLAUDE.md rule 3)",
        )
    with p.open("rb") as fh:
        scan_stream(fh, label=str(p), max_bytes=max_bytes, line_bytes=line_bytes)
