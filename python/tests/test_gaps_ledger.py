"""`gaps.jsonl` is the index of what is open. These tests make its rules executable.

Three lanes and two concurrent sessions append to this file, and it has already grown two
different conventions for revising a record and one silent way to lose a finding:

* A lane that writes "recorded as ..." in a document and never appends the record leaves a
  citation pointing at nothing. Measured before this file existed: nine ids in
  ``AUDIT/gdn-reference-and-contracts.md`` and one in ``docs/plan-corrections.md`` were
  cited in prose and absent from the ledger. Reading the ledger to see what was open
  missed all ten.

No example id appears anywhere in this file, deliberately. A plausible-looking placeholder
in prose is indistinguishable from a real citation to any scanner, including this one --
the first draft used three and broke the sibling guard in ``test_wire_gap_pins.py``.
* Two revision conventions coexist. Some records mint a new id and name the old one in
  ``supersedes``; others append a fresh line under the *same* id (the XLANG-FINISH lane's
  re-verifications read "VERIFIED OPEN, not closed"). Both are append-only and lossless,
  but a reader that builds a dict silently keeps whichever came last.

So the semantics are stated here and checked, rather than left to whoever reads the file
next: **a record's current state is its LAST line; earlier lines are history.**
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "gaps.jsonl"

REQUIRED_KEYS = {"id", "question", "status"}

# Deliberately permissive on the tail so a line-wrapped citation is *caught* as a
# fragment here and resolved by the prefix rule below, rather than silently skipped.
ID_RE = re.compile(rb"GAP-[A-Z0-9][A-Z0-9-]{3,}")


def _lines() -> list[tuple[int, dict]]:
    out = []
    for n, raw in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if raw:
            out.append((n, json.loads(raw)))
    return out


def current_records() -> dict[str, dict]:
    """Last line wins. This function *is* the convention; import it rather than
    re-deriving it, so a second reading cannot come into existence."""
    return {r["id"]: r for _, r in _lines()}


def test_every_line_is_one_valid_record_carrying_an_id() -> None:
    """Per-LINE invariants are limited to what history can satisfy.

    Only what is true of every line ever written belongs here. Demanding a full key set on
    every line would be unsatisfiable by design: earlier lines are immutable history, so
    the only way to add a missing field to line 65 is to rewrite the file -- the very thing
    the append-only rule forbids, and the thing that made a concurrent rewrite unverifiable
    earlier in this repo's history. Completeness is asserted on the *current* record below,
    where an append can actually fix it.
    """
    for n, rec in _lines():
        assert isinstance(rec, dict), f"line {n} is not an object"
        assert isinstance(rec.get("id"), str) and rec["id"].startswith("GAP-"), (
            f"line {n}: id {rec.get('id')!r} does not look like a gap id"
        )


def test_every_current_record_carries_the_required_keys() -> None:
    """A record whose current state lacks `status` reads as neither open nor closed, and a
    reader tallying what is left will simply not count it. Fixable by appending a corrected
    record under the same id, which is what the ten S4 records needed."""
    incomplete = {
        gid: sorted(REQUIRED_KEYS - set(rec)) for gid, rec in current_records().items()
    }
    incomplete = {g: k for g, k in incomplete.items() if k}
    assert not incomplete, (
        "current records missing required keys (append a corrected record under the same "
        f"id; do not edit history): {incomplete}"
    )


def test_every_supersedes_and_answers_reference_resolves() -> None:
    """A revision that names a record which does not exist is the same dangling
    citation as a document naming one, one level further in."""
    known = {r["id"] for _, r in _lines()}
    dangling: list[str] = []
    for n, rec in _lines():
        refs: list[str] = []
        if isinstance(rec.get("supersedes"), str):
            refs.append(rec["supersedes"])
        answers = rec.get("answers")
        if isinstance(answers, list):
            refs.extend(a for a in answers if isinstance(a, str))
        for ref in refs:
            if ref not in known:
                dangling.append(f"line {n} ({rec['id']}) -> {ref}")
    assert not dangling, "references to records that do not exist:\n  " + "\n  ".join(dangling)


def test_the_last_line_for_an_id_is_the_one_a_reader_gets() -> None:
    """Pins the convention itself. Repeated ids are history, not a conflict."""
    seen = [r["id"] for _, r in _lines()]
    current = current_records()
    assert set(seen) == set(current), "current_records() dropped an id"
    assert len(current) == len(set(seen)), "current_records() must be one record per id"
    for gid, rec in current.items():
        last = [r for _, r in _lines() if r["id"] == gid][-1]
        assert rec is last or rec == last, f"{gid}: current_records() is not the last line"


def _scanned_files() -> list[Path]:
    """Tracked files AND untracked-but-not-ignored ones.

    `git ls-files` alone lists only what is already committed, which would have made this
    guard blind to exactly the newest work -- every handoff and document written in the
    session that added it. A citation should be caught before it is committed, not after.
    Ignored paths stay out: build output and the DevMap store are not prose anyone reads.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    paths = [REPO / p.decode() for p in (tracked + untracked).split(b"\0") if p]
    assert paths, "git listed no files at all; refusing to report a vacuous clean scan"
    return paths


def test_every_gap_id_cited_in_a_tracked_file_exists_in_the_ledger() -> None:
    """The defect this file was written for.

    A citation resolves if it names a record exactly. A citation that ENDS IN A HYPHEN may
    instead resolve as a prefix of at least one record, because those are the two harmless
    shapes a partial id takes in prose: an id wrapped across a line break in markdown or a
    comment, and a deliberate family glob naming several records by their shared stem.

    The hyphen condition is the whole point of the split, and it was missing at first. A
    peer session reviewing this file noticed that ``ID_RE`` carries the hyphen inside its
    character class, so prefix tolerance without that condition also absorbs a *complete*
    citation that is merely truncated or mistyped -- and a mistyped id that happens to be a
    prefix of a real record is exactly the dangling citation this test exists to catch.
    They had lost four false positives to the same greedy match using rg. A token with no
    trailing hyphen is a whole citation and must resolve exactly.

    The nine AUDIT ids this test first caught were prefixes of nothing.
    """
    known = {r["id"] for _, r in _lines()}
    dangling: dict[str, set[str]] = {}
    seen_any = False
    for path in _scanned_files():
        if path.name == "gaps.jsonl" or not path.is_file():
            continue
        try:
            blob = path.read_bytes()
        except OSError:  # pragma: no cover - a tracked path that cannot be read
            continue
        for m in ID_RE.finditer(blob):
            gid = m.group().decode()
            seen_any = True
            if gid in known:
                continue
            if gid.endswith("-") and any(k.startswith(gid) for k in known):
                continue
            dangling.setdefault(gid, set()).add(str(path.relative_to(REPO)))

    # Borrowed from the sibling guard in test_wire_gap_pins.py: a scan that matched nothing
    # passes for the wrong reason. Zero citations across the whole repo means the scan
    # broke, not that every citation resolved.
    assert seen_any, (
        "no gap id was found in any scanned file, so this test asserted nothing. Either "
        "every citation was removed or the scan stopped working."
    )

    if dangling:
        detail = "\n  ".join(
            f"{gid} cited in {', '.join(sorted(where))}" for gid, where in sorted(dangling.items())
        )
        pytest.fail(
            f"{len(dangling)} gap id(s) are cited in scanned files but are not in "
            "gaps.jsonl. A document claiming a record was written, when it never was, "
            "hides the finding from anyone reading the ledger:\n  " + detail
        )
