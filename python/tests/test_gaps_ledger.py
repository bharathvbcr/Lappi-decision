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
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train import gaps as _gaps

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "gaps.jsonl"

# The vocabulary and the required keys now live in `qd_train.gaps`, beside the function
# that WRITES a record, and are re-exported here so this module stays the place other tests
# import them from. They were defined here first, which was the defect: a test module is
# not importable by a tool, so every lane hand-rolled its append with the schema carried in
# its head -- and on 2026-09-21 two lanes' records were refused by these very tests for
# using words nobody had declared. A checker with no matching writer is half a contract.
REQUIRED_KEYS = set(_gaps.REQUIRED_KEYS)
OPEN_STATUSES = _gaps.OPEN_STATUSES
CLOSED_STATUSES = _gaps.CLOSED_STATUSES
KNOWN_STATUSES = _gaps.KNOWN_STATUSES


def statuses_outside_vocabulary(records: Mapping[str, dict]) -> dict[str, str]:
    """``{id: status}`` for every record whose status nobody declared.

    A function over records rather than a loop inside a test, so the guard can be shown to
    catch a bad value on synthetic input -- a check that has only ever been run against
    data that satisfies it has not been shown to do anything.
    """
    return {
        gid: str(rec.get("status"))
        for gid, rec in records.items()
        if str(rec.get("status")) not in KNOWN_STATUSES
    }


def open_records() -> dict[str, dict]:
    """The records whose question is still unanswered -- the single owner of that rule.

    Import this rather than writing ``status == "open"`` or ``status not in {...}`` again;
    two conventions disagreeing by 33 records is what it exists to prevent.
    """
    return {
        gid: rec
        for gid, rec in current_records().items()
        if str(rec.get("status")) in OPEN_STATUSES
    }


# Deliberately permissive on the tail: the hyphen is INSIDE the class, so a citation
# wrapped across a line break is captured *with* its trailing hyphen instead of being
# clipped. That hyphen is the signal `citation_resolves` keys on, so a pattern that
# drops it cannot tell a wrap from a truncation.
_ID_PATTERN = _gaps.GAP_ID_PATTERN
ID_RE = re.compile(_ID_PATTERN.encode())
ID_RE_TEXT = re.compile(_ID_PATTERN)


def citation_resolves(token: str, known: Collection[str]) -> bool:
    """Does a ``GAP-`` token found in prose resolve to a record? The single owner of that
    rule; import it rather than writing a second one.

    Exact match resolves. A token ending in a hyphen may also resolve as a prefix of at
    least one record, because a trailing hyphen marks the two harmless shapes: an id
    wrapped across a line break, and a family glob naming records by their stem. A token
    *without* a trailing hyphen is a complete citation and must resolve exactly —
    otherwise prefix tolerance also absorbs a truncated or mistyped id, which is the
    dangling citation the guard exists to catch.

    This exists because two checkers disagreed on the same text. ``test_wire_gap_pins.py``
    carried its own pattern that could not end in a hyphen, so a citation wrapped across a
    line break in ``test_shards.py`` was clipped at the break and reported as dangling,
    while the full record it names sat in the ledger. Two implementations of one check,
    giving opposite verdicts on valid prose, is the defect this repository keeps finding
    in its own subject matter.

    Note what is deliberately absent above: the clipped form itself. Writing it would make
    this docstring fail the very guard it serves, because a scanner cannot tell a quoted
    broken token from a citation — which is the same reason the ledger's own prose is
    excluded from the scan.
    """
    if token in known:
        return True
    return token.endswith("-") and any(k.startswith(token) for k in known)


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


def _references(rec: dict) -> list[str]:
    refs: list[str] = []
    if isinstance(rec.get("supersedes"), str):
        refs.append(rec["supersedes"])
    answers = rec.get("answers")
    if isinstance(answers, list):
        refs.extend(a for a in answers if isinstance(a, str))
    return refs


def test_every_supersedes_and_answers_reference_resolves() -> None:
    """A revision that names a record which does not exist is the same dangling
    citation as a document naming one, one level further in.

    Asserted on CURRENT records, and that is a correction rather than a relaxation. The
    sibling above states the principle this one was breaking: *"only what is true of every
    line ever written belongs"* on a per-line check, because earlier lines are immutable
    history and the only way to change one is to rewrite the file -- the thing the
    append-only rule forbids. Completeness and the status vocabulary were already asserted
    on the current record for exactly that reason; this check scanned every line anyway.

    It cost a rewrite on 2026-09-21. A lane put prose where a gap id belongs, could not fix
    it by appending because this test read the bad line forever, and removed the line --
    measuring first that the line was its own, two minutes old and uncommitted, which it
    was. The rule and the gate were in conflict and the gate was the one that could bend.

    **Both numbers**, so this is not a quiet narrowing: the historical count is printed on
    every run. A dangling reference in a superseded record is history and unfixable; one in
    a current record is a live defect an append can repair, and only that fails.
    """
    known = {r["id"] for _, r in _lines()}

    historical = [
        f"line {n} ({rec['id']}) -> {ref}"
        for n, rec in _lines()
        for ref in _references(rec)
        if ref not in known
    ]
    live = [
        f"{gid} -> {ref}"
        for gid, rec in current_records().items()
        for ref in _references(rec)
        if ref not in known
    ]
    print(
        f"gaps.jsonl references: {len(live)} dangling in current records, "
        f"{len(historical)} dangling across all {len(_lines())} lines including history"
    )
    assert not live, (
        "current records reference records that do not exist (append a corrected record "
        "under the same id; do not edit history):\n  " + "\n  ".join(live)
    )


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
        # The ledger's own prose is excluded, and that exclusion is LOAD-BEARING -- see
        # test_the_ledger_is_excluded_from_the_citation_scan_for_a_reason below before
        # removing it. A record documenting a rename, a family of records, or someone
        # else's typo has to be able to write down a string that is *not* an id, and
        # "this is not an id" cannot be said without saying it. No checker can tell that
        # from a dangling reference, because the distinction lives in the prose around it.
        if path.name == LEDGER.name or not path.is_file():
            continue
        try:
            blob = path.read_bytes()
        except OSError:  # pragma: no cover - a tracked path that cannot be read
            continue
        for m in ID_RE.finditer(blob):
            gid = m.group().decode()
            seen_any = True
            if citation_resolves(gid, known):
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


def test_the_ledger_is_excluded_from_the_citation_scan_for_a_reason() -> None:
    """A self-retiring pin on the `gaps.jsonl` skip above.

    An unexplained skip in a checker invites someone to tighten it. This measures what
    scanning the ledger's own prose *would* report, so the justification is live data
    rather than an assertion in a comment.

    Every such hit is a record being precise about id history -- a spelling that was
    renamed before the code landed, a family named by its stem, a typo made elsewhere.
    Saying "this string is not an id" requires writing the string, and no scanner can
    distinguish that from a citation, because the distinction is in the surrounding prose.

    Tightening the trailing-hyphen rule made this MORE necessary, not less: a stem like
    the qd-mutate family prefix is a prefix of many real records but carries no trailing
    hyphen, so it now reports as dangling where it used to be absorbed.

    If this ever legitimately reaches zero, the skip has no more work to do -- delete the
    skip and this test together, in that order.
    """
    known = {r["id"] for _, r in _lines()}
    would_report = set()
    for m in ID_RE.finditer(LEDGER.read_bytes()):
        gid = m.group().decode()
        if not citation_resolves(gid, known):
            would_report.add(gid)

    assert would_report, (
        "scanning gaps.jsonl now yields no would-be dangling id, so the skip in "
        "test_every_gap_id_cited_in_a_tracked_file_exists_in_the_ledger protects nothing. "
        "Remove the skip, then remove this test."
    )


# --- the status vocabulary ------------------------------------------------------------
#
# GAP-GAPS-STATUS-VOCABULARY-MAKES-THE-OPEN-COUNT-WRONG. `status` was required from the
# first version of this file and its VALUE was never checked, so five conventions grew in
# one ledger and "how many gaps are open" had two defensible answers 33 records apart.


def test_an_undeclared_status_is_caught() -> None:
    """The guard, shown to fire.

    On synthetic records, because gaps.jsonl is append-only and valid today: a check that
    has only ever run against data satisfying it has not been shown to do anything.
    """
    # These ids are deliberately too SHORT to match ID_RE above, which needs five or more
    # characters after the prefix. A readable name like the obvious one for a bad record is
    # a syntactically valid citation, and test_every_gap_id_cited_in_a_tracked_file_exists
    # scans this file like any other -- it caught exactly that while this test was written.
    good = {"GAP-A": {"status": "open"}, "GAP-B": {"status": "resolved-with-residual"}}
    assert statuses_outside_vocabulary(good) == {}

    for bad in ("resovled", "Open", "open ", "in-progress", "", None):
        records = {**good, "GAP-D": {"status": bad}}
        caught = statuses_outside_vocabulary(records)
        assert "GAP-D" in caught, f"{bad!r} was accepted as a status"
        assert set(caught) == {"GAP-D"}, "a good record was swept up with the bad one"

    assert statuses_outside_vocabulary({"GAP-C": {}}) == {"GAP-C": "None"}


def test_every_status_in_the_ledger_is_one_of_the_pinned_vocabulary() -> None:
    """And the same guard over the real file."""
    unknown = statuses_outside_vocabulary(current_records())
    assert not unknown, (
        "gap records carry a status outside the pinned vocabulary "
        f"{sorted(KNOWN_STATUSES)}:\n  "
        + "\n  ".join(f"{gid}: {status!r}" for gid, status in sorted(unknown.items()))
        + "\nAdd the value to OPEN_STATUSES or CLOSED_STATUSES deliberately, or fix the "
        "record. A status nobody declared is a record that counts as neither."
    )


def test_open_and_closed_partition_the_ledger_with_nothing_left_over() -> None:
    """Adding a value to one set and forgetting the other would create a third category
    that is neither counted nor reported."""
    assert not (OPEN_STATUSES & CLOSED_STATUSES), "a status cannot mean both"
    current = current_records()
    counted = len(open_records()) + sum(
        1 for rec in current.values() if str(rec.get("status")) in CLOSED_STATUSES
    )
    assert counted == len(current), (
        f"{len(current)} records but {counted} were classified; the vocabulary does not "
        "partition the ledger"
    )


def test_the_two_statuses_that_read_as_open_are_counted_as_closed() -> None:
    """The specific pair that cost this session an hour, asserted by name.

    Pinning the set is not enough: someone tidying this later could move
    `resolved-with-residual` into OPEN_STATUSES on the reasonable-sounding grounds that a
    residual is unfinished work. It is not -- the residual is recorded as its own gap, and
    counting the parent as open double-counts it.
    """
    assert "resolved-with-residual" in CLOSED_STATUSES
    assert "closed-no-defect" in CLOSED_STATUSES
    assert open_records().keys() <= current_records().keys()
