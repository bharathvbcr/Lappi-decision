"""What the rows say about what produced them, as opposed to what the tools say.

`test_tool_call_sites.py` asserts that every tool in `tools/` records `code_that_ran`. That
is a true statement about six files in this checkout, and on 2026-09-21 it was true while
65 of 89 `gh200-rung0-*` rows carried no digest at all -- written by `rung0_real_run.py`,
which had recorded one since `80cddb8` and was never missing from that test's scope. The
copy on the rented box predated the commit.

So there are two halves and only one of them is a property of this repository's source:

* the **tool** half -- does the code record provenance? Source-checkable, and closed.
* the **deployment** half -- did the code that actually ran record it? Not visible in any
  file here.

The second half is visible in the *rows*, at the moment they are brought back, and that is
what this file checks. It needs no ssh and trusts nothing the box says about itself: a row
either carries a digest or it does not.

## Why `code_commit` cannot stand in

Measured over `ledger/` on 2026-09-21: `8b9df39bf816d149efc108e3581ab4ba8192688f-dirty`
appears on **389 rows without a digest and 24 with one**. The same string, both sides. The
box is synced by COPYING files into a clone pinned at an old commit, so its commit is the
clone's and its `-dirty` bit is permanent -- one bit standing for 6894 insertions across 72
paths. A commit string that appears on both sides of a distinction cannot be used to make
it.

## What this pins, and what it cannot

699 of 727 training rows predate the wiring or the sync that carried it. The ledger is
append-only and those rows say what they were written with; they are not fixable and are not
failed here. What is failed is the 700th -- a training row arriving without a digest from
here on, whoever wrote it and wherever it ran.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "ledger"

#: Training/smoke rows without a closure digest, per ledger file, as of 2026-09-21.
#:
#: An inventory, not a budget. The ledger is append-only, so these counts can only be
#: matched or exceeded -- and exceeding one means a row arrived without provenance, which
#: is the event this file exists for. A file absent from this map may contribute NONE.
KNOWN_WITHOUT_DIGEST = {
    # 2026-10-18. The gate chain that branch consolidation kept whole as its own file when it
    # forked from runs.jsonl after feae39e0 (171b9c6, GAP-LEDGER-FORK-MAIN-GATES-2026-10-15).
    # Its 108 non-build rows (52 ft, 56 smoke: ft_toy_run, rung0_toy_run, real_ft_run,
    # rung0_real_run and the real-tokenizer smoke) were written on a Mac between
    # 2026-09-20T12:51Z and 2026-09-21T03:35Z, the day before `RunRecorder.__enter__` began
    # recording `code_that_ran` (2c84587, 2026-09-22). None carries the metric, as the rows of
    # that date in runs.jsonl do not. So they ARRIVED in this directory late, they did not
    # arrive without provenance late: the writer is not defective, and this is the
    # append-only ledger's record of what a pre-wiring run said. The file's 126 build rows
    # are excluded by `_training_rows`, as everywhere.
    "runs.main-gates-2026-10-15.jsonl": 108,
    # 2026-09-22. NOT a row that predates the wiring -- it came from a tool that had the
    # wiring and lost it on the way out. `rung0_real_run.py` recorded `code_that_ran` at
    # the END of its training block, and this run was killed by SIGTERM at its wall-clock
    # cap, so `RunRecorder._on_signal` wrote the row before the digest was ever set. The
    # ledger is append-only, so the row stands as the evidence it is.
    #
    # The writer is fixed rather than the inventory relaxed: `RunRecorder.__enter__` now
    # records the digest before the work and before the signal handlers are installed, so
    # completed, failed and killed runs all carry it. `test_provenance_on_every_exit.py`
    # holds that to all three exit paths and to every tool.
    # GAP-LEDGER-A-KILLED-RUN-LOSES-ITS-PROVENANCE.
    "gh200-commitpackft-2026-09-22.jsonl": 1,
    "gh200-2026-09-20.jsonl": 116,
    "gh200-2026-09-21.jsonl": 198,
    "gh200-det-2026-09-21.jsonl": 18,
    "gh200-determinism-2026-09-21.jsonl": 8,
    "gh200-overnight-2026-09-21.jsonl": 176,
    "gh200-rung0-capacity-4096-e10-2026-09-21.jsonl": 24,
    "gh200-rung0-capacity-4096-e30-2026-09-21.jsonl": 24,
    "gh200-rung0-capacity-sw005-2026-09-21.jsonl": 5,
    "gh200-rung0-capacity-sw005-nondet-2026-09-21.jsonl": 8,
    "gh200-rung0-repro-2026-09-21.jsonl": 2,
    "gh200-rung0-reprodet-2026-09-21.jsonl": 2,
    "gh200-seedcheck-2026-09-21.jsonl": 4,
    "gh200-smoke-2026-09-21.jsonl": 4,
    "runs.jsonl": 108,
    "shardset-v3-2026-09-21.jsonl": 1,
    "shardset-v4-2026-09-21.jsonl": 1,
}


def _rows() -> list[dict]:
    out: list[dict] = []
    for path in sorted(LEDGER.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_file"] = path.name
                out.append(row)
    return out


def _has_digest(row: dict) -> bool:
    """Whether the row carries a closure digest that actually says something.

    A `code_that_ran` present as `not_run` is not provenance -- it is the tool saying it
    could not compute one, which is the honest answer and still leaves the row unpinned.
    Only a `Ran` with a value counts.
    """
    state = row.get("metrics", {}).get("code_that_ran")
    return isinstance(state, dict) and bool(state.get("value"))


def _training_rows() -> list[dict]:
    """Rows a closure digest applies to.

    Build rows are excluded and the exclusion is not a convenience: a build row is
    identified by its commands and its toolchain -- both now stored on it -- and it is
    written on this Mac from a real git checkout, where `code_commit` means what it says.
    Its "closure" is the whole workspace, which no digest of `qd_train` would describe.
    """
    return _training_rows_of(_rows())


def _terminated_started_ids(rows: list[dict]) -> set[str]:
    """The ``row_id`` of every row some other row names in ``lifecycle.started_row_id``: the
    ``started`` rows that a terminal row ended."""
    ids: set[str] = set()
    for row in rows:
        state = row.get("metrics", {}).get("lifecycle.started_row_id")
        if isinstance(state, dict) and state.get("value"):
            ids.add(str(state["value"]))
    return ids


def _training_rows_of(rows: list[dict]) -> list[dict]:
    """:func:`_training_rows`'s rule over any list of rows.

    A ``started`` row is also left out once a terminal row names it. It is the lifecycle marker
    ``RunRecorder.__enter__`` appends FIRST, before the digest is computed (so that a SIGKILLed
    run still leaves a row), and it measures nothing; its digest is on the terminal row, which
    stays in and is checked. A ``started`` row nobody terminated stays in too: that is a run
    whose end nobody saw, and it has no digest to show for itself.
    """
    ended = _terminated_started_ids(rows)
    return [
        r for r in rows
        if r["run_kind"] != "build"
        and not (r.get("status") == "started" and r.get("row_id") in ended)
    ]


def test_the_ledger_is_readable_and_not_empty() -> None:
    """Scope first. Every assertion below is vacuous over an empty directory, and a
    `glob` that stopped matching would read exactly like a repository with no history."""
    rows = _rows()
    assert len(rows) >= 932, (
        f"{len(rows)} row(s) found across {LEDGER}; 932 existed when this was written and "
        "the ledger is append-only, so fewer means files moved or this stopped finding them"
    )


def test_no_new_training_row_arrives_without_saying_what_produced_it() -> None:
    """The deployment half, checked where it is visible.

    The 699 rows below predate either the wiring or the sync that carried it, and they are
    not failed: the ledger is append-only and a row says what it was written with. A row
    beyond them is a live defect -- a runner that stopped recording provenance, or, far
    more likely, a machine running sources older than the ones in this checkout, which is
    exactly what produced the 65 rows that started this.

    It fails on ARRIVAL, which is the first moment this repository can see the box's work.
    """
    missing = Counter(r["_file"] for r in _training_rows() if not _has_digest(r))

    grown = {
        name: (count, KNOWN_WITHOUT_DIGEST.get(name, 0))
        for name, count in missing.items()
        if count > KNOWN_WITHOUT_DIGEST.get(name, 0)
    }
    assert not grown, (
        "training rows arrived without a closure digest:\n  "
        + "\n  ".join(
            f"{name}: {now} without a digest, {then} known"
            for name, (now, then) in sorted(grown.items())
        )
        + "\n\nA row with no digest is pinned only by code_commit, which on the rented box "
        "is a permanently `-dirty` string from a clone pinned at an old commit -- measured "
        "on both sides of this exact distinction, so it cannot make it. Either the tool "
        "that wrote these stopped recording `code_that_ran`, or the machine that ran it "
        "was on older sources than this checkout. Check the second first: that is what "
        "happened to 65 gh200-rung0 rows on 2026-09-21, from a tool that had recorded the "
        "digest for hours."
    )


def test_the_coverage_is_reported_as_two_numbers() -> None:
    """Both numbers, because one of them alone is the failure this repository keeps finding.

    "28 rows carry provenance" reads like progress. "28 of 727" reads like what it is. The
    ratio is printed rather than asserted upward -- nothing here should fail because
    history is what it is -- but the floor is pinned so it cannot quietly fall.
    """
    training = _training_rows()
    with_digest = [r for r in training if _has_digest(r)]
    assert training, "no training rows; this test would pass vacuously"
    assert len(with_digest) >= 28, (
        f"{len(with_digest)} of {len(training)} training rows carry a closure digest, "
        "down from 28. Rows are append-only, so this number cannot fall unless a file was "
        "rewritten"
    )
    print(
        f"ledger provenance: {len(with_digest)} of {len(training)} training row(s) carry a "
        f"closure digest ({100 * len(with_digest) / len(training):.1f}%); "
        f"{len(_rows()) - len(training)} build row(s) excluded by design"
    )


def test_code_commit_does_not_distinguish_the_two_groups() -> None:
    """The measurement that rules out the obvious alternative to this whole file.

    If `code_commit` separated rows-with-provenance from rows-without, none of this would
    be needed -- the commit would already say which sources ran. It does not:
    `8b9df39…-dirty` sits on 389 rows without a digest and 24 with one. Asserted rather
    than recounted in prose, because the day this stops being true is the day a simpler
    check becomes possible, and it should be noticed rather than assumed.
    """
    training = _training_rows()
    without = {r["code_commit"] for r in training if not _has_digest(r)}
    with_d = {r["code_commit"] for r in training if _has_digest(r)}
    assert without & with_d, (
        "no code_commit appears on both sides any more. If that holds, code_commit has "
        "become able to distinguish a row written by a provenance-recording tool from one "
        "that was not, and this file's premise needs re-deriving rather than inheriting"
    )


def _row(row_id: str, *, status: str, digest: bool = False, ended: str | None = None) -> dict:
    metrics: dict[str, dict] = {}
    if digest:
        metrics["code_that_ran"] = {"state": "ran", "passed": True, "value": "d" * 64}
    if ended is not None:
        metrics["lifecycle.started_row_id"] = {"state": "ran", "passed": True, "value": ended}
    return {"row_id": row_id, "run_kind": "smoke", "status": status, "metrics": metrics}


def test_a_started_marker_is_not_a_training_row_once_a_terminal_row_ended_it() -> None:
    """``RunRecorder.__enter__`` appends the ``started`` row before it computes the digest, so
    that row can never carry one; the terminal row that names it does, and is the one checked.
    Counting the marker made the 2026-10-18 LoRA-2B shard build read as two rows that "arrived
    without saying what produced them" (``mac-v6-lora2b-shards-2026-10-18.jsonl``)."""
    rows = [
        _row("s1", status="started"),
        _row("t1", status="completed", digest=True, ended="s1"),
    ]
    assert [r["row_id"] for r in _training_rows_of(rows)] == ["t1"]


def test_an_unended_started_row_and_a_terminal_row_without_a_digest_still_count() -> None:
    """The exemption is for the marker a terminal row ended, and no wider: a run nobody saw
    end (no terminal row names it) and a terminal row that lost its digest both stay in."""
    rows = [
        _row("s2", status="started"),
        _row("s3", status="started"),
        _row("t3", status="completed", ended="s3"),
    ]
    kept = _training_rows_of(rows)
    assert [r["row_id"] for r in kept] == ["s2", "t3"]
    assert not any(_has_digest(r) for r in kept)
