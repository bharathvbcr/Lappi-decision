"""A row written on the way out still has to say what code produced it.

``RunRecorder``'s docstring promises a row on every exit path: completed, failed, killed.
It did not promise **provenance** on every exit path, and it did not deliver it. Every
tool recorded ``code_that_ran`` at the end of its training block, so a run that died never
recorded it at all -- ``_on_signal`` calls ``_finish`` immediately, and the digest was not
yet among the metrics.

That is not hypothetical. ``ledger/gh200-commitpackft-2026-09-22.jsonl`` carries an ``ft``
row with ``code_that_ran: None`` and the note ``received SIGTERM``: the 2.5-hour-capped fit
terminated on 2026-09-21. ``test_ledger_provenance`` caught it on arrival, which is what it
is for -- ``GAP-LEDGER-A-KILLED-RUN-LOSES-ITS-PROVENANCE``.

The perverse part is which runs lost it. A run that completes is the one whose provenance
is least interesting; a run that was killed, or that raised, is the one somebody will later
need to pin. `code_commit` cannot do it -- on the rented box it is a permanently ``-dirty``
string from a clone at an old commit, which ``qd_train.ledger`` says in as many words
distinguishes nothing.

The fix is not "also record it in the signal handler". The digest never depends on the
run's outcome, so it is computed in ``__enter__``, before the work and before the handlers
are installed.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from qd_train.ledger import (  # noqa: E402
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)

TOOLS = REPO / "tools"


def _protocol() -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="bytes-utf8-256",
        backbone_commit="test-backbone",
        recipe_hash="r" * 64,
        seed=0,
    )


def _recorder(ledger: Ledger, **over) -> RunRecorder:
    kwargs = {
        "protocol": _protocol(),
        "run_kind": "ft",
        "repo": REPO,
        "env": Environment.detect(),
        "wall_clock_s": None,
        # `None` is how a run on a local device says "nothing is billed here"; the
        # constructor refuses it for anything rented, which is a different test's subject.
        "cost": None,
        "entry_point": Path(__file__),
    }
    kwargs.update(over)
    return RunRecorder(ledger, **kwargs)  # type: ignore[arg-type]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _digest_of(row: dict) -> str | None:
    state = row.get("metrics", {}).get("code_that_ran")
    return state.get("value") if isinstance(state, dict) else None


# ---------------------------------------------------------------------------
# The three exit paths
# ---------------------------------------------------------------------------


def test_a_completed_run_records_the_digest(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger):
        pass
    (row,) = _rows(tmp_path / "l.jsonl")
    assert row["status"] == "completed"
    assert _digest_of(row), row.get("metrics")


def test_a_run_that_raises_still_records_the_digest(tmp_path) -> None:
    """The failure case matters more than the success one: a row that says `failed` and
    cannot say what failed is a dead end."""
    ledger = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(RuntimeError, match="boom"), _recorder(ledger):
        raise RuntimeError("boom")
    (row,) = _rows(tmp_path / "l.jsonl")
    assert row["status"] == "failed"
    assert _digest_of(row), row.get("metrics")


def test_a_run_killed_mid_block_still_records_the_digest(tmp_path) -> None:
    """The load-bearing test, and the one that fails against the pre-fix code.

    SIGTERM is delivered to this process while the block is open, exactly as it was to the
    capped fit on 2026-09-21. `_on_signal` writes the row and re-raises, so the signal
    still kills what it was aimed at -- the row is written first.
    """
    ledger = Ledger(tmp_path / "l.jsonl")
    previous = signal.getsignal(signal.SIGTERM)
    delivered = []

    def swallow(signum, frame):  # the handler `_on_signal` restores and re-raises into
        delivered.append(signum)

    signal.signal(signal.SIGTERM, swallow)
    try:
        with _recorder(ledger):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert delivered, "the signal was never delivered, so this tested nothing"
    (row,) = _rows(tmp_path / "l.jsonl")
    assert row["status"] == "killed"
    assert "SIGTERM" in row["notes"]
    assert _digest_of(row), (
        "a killed run wrote a row with no closure digest -- pinned only by code_commit, "
        "which on the rented box is a permanently -dirty string that distinguishes nothing"
    )


def test_the_digest_is_on_the_row_before_any_work_happens(tmp_path) -> None:
    """Recorded by `__enter__`, not by the caller. If it were the caller's job, every
    caller would have to remember it on every exit path, which is the arrangement that
    produced the defect."""
    ledger = Ledger(tmp_path / "l.jsonl")
    recorder = _recorder(ledger)
    assert "code_that_ran" not in recorder.metrics
    with recorder:
        assert recorder.metrics["code_that_ran"].value, "not recorded on entry"


def test_omitting_the_entry_point_is_allowed_and_records_nothing(tmp_path) -> None:
    """Test fixtures and non-tool callers construct recorders without an entry point.
    They get no digest, which is honest -- `test_ledger_provenance` is what makes it a
    failure for a row that should have had one."""
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger, entry_point=None):
        pass
    (row,) = _rows(tmp_path / "l.jsonl")
    assert _digest_of(row) is None


# ---------------------------------------------------------------------------
# No tool can forget
# ---------------------------------------------------------------------------


def _tools_constructing_a_recorder() -> dict[str, bool]:
    """Every ``tools/*.py`` that builds a ``RunRecorder``, and whether it passes
    ``entry_point``. Parsed rather than imported: most of these import torch at scope."""
    found: dict[str, bool] = {}
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "RunRecorder":
                continue
            passes = any(kw.arg == "entry_point" for kw in node.keywords)
            # A tool with several construction sites passes only if ALL of them do.
            found[path.name] = found.get(path.name, True) and passes
    return found


def test_every_tool_that_records_a_run_passes_its_entry_point() -> None:
    """The mechanism lives in `RunRecorder`, but a tool that does not pass `entry_point`
    opts out of it silently. This is the check that makes opting out visible."""
    tools = _tools_constructing_a_recorder()
    assert tools, "no tool constructs a RunRecorder; this test stopped finding them"
    missing = sorted(name for name, ok in tools.items() if not ok)
    assert not missing, (
        "these tools construct a RunRecorder without entry_point, so a run of theirs that "
        f"is killed or raises writes a row that cannot say what produced it: {missing}"
    )


def test_no_tool_still_records_the_digest_by_hand_after_the_work() -> None:
    """The old arrangement, removed rather than left beside the new one.

    A tool doing both would be harmless today and would quietly become the only mechanism
    the moment somebody dropped `entry_point`, which is how a replaced path survives its
    replacement.
    """
    offenders = []
    for path in sorted(TOOLS.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if '"code_that_ran"' in source or "'code_that_ran'" in source:
            offenders.append(path.name)
    assert not offenders, (
        "these tools still name code_that_ran themselves; RunRecorder.__enter__ records "
        f"it now, and two mechanisms for one fact drift: {offenders}"
    )
