"""Writing `gaps.jsonl`, with the rules it is checked against applied before the write.

`python/tests/test_gaps_ledger.py` makes this file's rules executable and has done since it
was written. What did not exist was anything that *writes* a record. Every lane hand-rolled
`os.open(..., O_APPEND)` and `json.dumps`, with the schema carried in the author's head and
the vocabulary sitting in a test module no tool can sensibly import.

The result arrived on 2026-09-21, twice in one afternoon. A lane appended
``status: "closed-answered"`` -- a reasonable English phrase, and not one of the five
statuses in use. Its correction then put prose in ``supersedes``, which takes a gap id. Both
were caught by the gates, which is the gates working; but neither could be fixed by
appending, because one test scanned every line rather than the current record per id. The
only remedy left was to rewrite the file, which `CLAUDE.md` forbids for the concrete reason
that a concurrent lane's record can be lost between the read and the write.

So: a rule that says how to write, a test that says what is valid, and nothing joining them.
This module is the join. :func:`append_gap` validates against the same vocabulary the tests
import from here, refuses anything that would land malformed, and only then appends -- one
line, ``O_APPEND``, ``fsync``, no read-modify-write.

The validation is deliberately *exactly* what the tests check, neither more nor less. A
writer stricter than the checker refuses records that are fine; a writer looser than the
checker is the thing that just happened.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The ledger this module writes. Repo-level, not package-level: it is the index of what
#: every lane could not verify, not a training artifact.
DEFAULT_GAPS_PATH = Path(__file__).resolve().parents[2] / "gaps.jsonl"

#: Every record needs these three. Checked on the CURRENT record per id by the tests --
#: earlier lines are immutable history -- and on every record this function writes, because
#: a new record has no excuse for being incomplete.
REQUIRED_KEYS = frozenset({"id", "question", "status"})

#: The only statuses a record may carry, and which of them mean the question is answered.
#:
#: Pinned 2026-09-21 to the five already in use rather than to a tidier pair, because
#: rewriting 255 records' statuses is a different change from stopping a sixth appearing --
#: and this file is append-only, so the rewrite is not available anyway.
#:
#: `resolved-with-residual` and `closed-no-defect` are the two that cost real time: they
#: mean CLOSED, and a reader that treats only {resolved, closed} as closed counted 111 open
#: records where the answer was 78. That is the repository's own "one name, two quantities"
#: defect, in the file that records that defect.
OPEN_STATUSES: frozenset[str] = frozenset({"open"})
CLOSED_STATUSES: frozenset[str] = frozenset(
    {"resolved", "closed", "resolved-with-residual", "closed-no-defect"}
)
KNOWN_STATUSES: frozenset[str] = OPEN_STATUSES | CLOSED_STATUSES


class GapRecordError(ValueError):
    """A record that would land malformed. Raised before anything is written."""


def read_gaps(path: Path | None = None) -> list[dict[str, Any]]:
    """Every line, oldest first. Earlier lines are history; the last per id is current."""
    target = DEFAULT_GAPS_PATH if path is None else path
    if not target.is_file():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def current_records(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Last line wins, which is this file's convention and not an implementation detail."""
    return {rec["id"]: rec for rec in read_gaps(path)}


def validate_gap(record: Mapping[str, Any], *, known_ids: set[str]) -> str:
    """Return the single JSON line this record would become, or raise.

    ``known_ids`` is what ``supersedes`` and ``answers`` may point at. Passed in rather
    than read here so the caller decides which file is being written and so this stays a
    pure function -- the one part of this module worth testing without touching a disk.
    """
    if not isinstance(record, Mapping):
        raise GapRecordError(f"a gap record is an object, got {type(record).__name__}")

    missing = sorted(REQUIRED_KEYS - set(record))
    if missing:
        raise GapRecordError(
            f"record is missing {missing}. A record without all three reads as neither "
            "open nor closed, and a reader tallying what is left will not count it"
        )

    gid = record["id"]
    if not isinstance(gid, str) or not gid.startswith("GAP-") or not gid[4:].strip():
        raise GapRecordError(
            f"id {gid!r} does not look like a gap id: it must be a string starting 'GAP-' "
            "with something after it"
        )

    status = record["status"]
    if status not in KNOWN_STATUSES:
        raise GapRecordError(
            f"status {status!r} is not one of {sorted(KNOWN_STATUSES)}. The vocabulary is "
            "pinned to what is already in use; a sixth value makes every open-count wrong "
            "and cannot be corrected by rewriting 255 records in an append-only file"
        )

    question = record["question"]
    if not isinstance(question, str) or not question.strip():
        raise GapRecordError("question must be a non-empty string; it is what the record is")

    supersedes = record.get("supersedes")
    if supersedes is not None:
        if not isinstance(supersedes, str):
            raise GapRecordError(f"supersedes must be a gap id, got {type(supersedes).__name__}")
        if supersedes not in known_ids:
            raise GapRecordError(
                f"supersedes={supersedes!r} names no record in this ledger. It takes a gap "
                "id, not a description of what changed -- a dangling citation one level in"
            )

    answers = record.get("answers")
    if answers is not None:
        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
            raise GapRecordError("answers must be a list of gap ids")
        dangling = sorted(a for a in answers if a not in known_ids)
        if dangling:
            raise GapRecordError(f"answers names records that do not exist: {dangling}")

    try:
        line = json.dumps(dict(record), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise GapRecordError(f"record is not JSON-serialisable: {exc}") from exc
    if "\n" in line or "\r" in line:
        raise GapRecordError(
            "record serialises to more than one line; a record is one line, because a "
            "reader splits this file on newlines"
        )
    return line


def append_gap(record: Mapping[str, Any], *, path: Path | None = None) -> str:
    """Validate, then append one line with ``O_APPEND`` and ``fsync``. Returns the line.

    Never a read-modify-write. The file is read to learn which ids exist -- which a
    reference check needs -- and the write is a single ``O_APPEND`` call, so a concurrent
    lane appending between the read and the write loses nothing. That is the difference
    the rule is about: reading is safe, rewriting is not.
    """
    target = DEFAULT_GAPS_PATH if path is None else path
    line = validate_gap(record, known_ids={rec["id"] for rec in read_gaps(target)})
    fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return line
