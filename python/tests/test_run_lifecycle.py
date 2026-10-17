"""A run SIGKILLed past its handlers still leaves a row, and the ledger can say so.

``RunRecorder`` wrote a row on every exit a process can observe, and SIGKILL is not one:
memguard SIGTERMs a process over 32 GiB and SIGKILLs it ten seconds later, leaving no log, so
a run busy in a native call never reached ``_on_signal`` and left nothing
(GAP-MAC-A-SIGKILLED-RUN-LEAVES-NO-ROW, GAP-A-KILLED-RUNG0-RUN-LEAVES-NO-ROW-AND-NO-COST).

The contract under test (docs/ledger-schema.md, "Run lifecycle"): ``__enter__`` appends a
``started`` row before any work; every row that ends the run names it; and
``reconcile_orphan_started_rows`` appends a ``killed`` row for a started row whose process is
gone, once.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from qd_train import ledger as ledger_module
from qd_train.ledger import (
    LIFECYCLE_STARTED_ROW,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)

REPO = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO / "python"


def _protocol(seed: int = 0) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="bytes-utf8-256",
        backbone_commit="test-backbone",
        recipe_hash="r" * 64,
        seed=seed,
    )


def _recorder(ledger: Ledger, **over) -> RunRecorder:
    kwargs = {
        "protocol": _protocol(),
        "run_kind": "ft",
        "repo": REPO,
        "env": Environment.detect(),
        "wall_clock_s": None,
        "cost": None,
    }
    kwargs.update(over)
    return RunRecorder(ledger, **kwargs)  # type: ignore[arg-type]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metric(row: dict, name: str):
    state = row["metrics"].get(name)
    return None if state is None or state["state"] != "ran" else state["value"]


#: A child that opens a recorder on the ledger it is given, reports its pid, and then blocks.
#: It never leaves the ``with`` block, so only SIGKILL's absence of a handler is in play.
_CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {python!r})
    from pathlib import Path
    from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
    proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="bytes-utf8-256",
                     backbone_commit="test-backbone", recipe_hash="r" * 64, seed=0)
    rec = RunRecorder(Ledger(sys.argv[1]), protocol=proto, run_kind="ft", repo={repo!r},
                      env=Environment.detect(), wall_clock_s=None, cost=None,
                      registration_id="reg-test", run_cap_s=600.0)
    with rec:
        print("ready", flush=True)
        time.sleep(600)
    """
)


def _sigkilled_run(ledger_path: Path) -> int:
    """Run the child, SIGKILL it once it is inside the block, and return its (dead) pid."""
    child = subprocess.Popen(
        [sys.executable, "-I", "-c",
         _CHILD.format(python=str(PYTHON_DIR), repo=str(REPO)), str(ledger_path)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        if child.stdout is not None:
            child.stdout.close()
    assert child.returncode == -signal.SIGKILL
    return child.pid


# ---------------------------------------------------------------------------
# Task 1: the started row, and the terminal row that names it
# ---------------------------------------------------------------------------


def test_sigkilled_run_leaves_a_started_row(tmp_path) -> None:
    """The one that fails against the pre-fix code: SIGKILL, and the ledger is empty."""
    path = tmp_path / "l.jsonl"
    pid = _sigkilled_run(path)
    (row,) = _rows(path)
    assert row["status"] == "started"
    assert _metric(row, "lifecycle.pid") == pid
    assert _metric(row, "lifecycle.host")
    assert _metric(row, "lifecycle.started_at")
    assert _metric(row, "lifecycle.registration_id") == "reg-test"
    assert _metric(row, "lifecycle.cap_s") == 600.0


def test_started_row_is_in_the_ledger_before_any_work(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger):
        (row,) = _rows(tmp_path / "l.jsonl")
        assert row["status"] == "started"
        assert row["gates"] == {} and row["controls"] == {}


@pytest.mark.parametrize("how", ["completed", "failed"])
def test_the_terminal_row_names_the_started_row(tmp_path, how) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    try:
        with _recorder(ledger):
            if how == "failed":
                raise RuntimeError("boom")
    except RuntimeError:
        pass
    started, terminal = _rows(tmp_path / "l.jsonl")
    assert (started["status"], terminal["status"]) == ("started", how)
    assert _metric(terminal, LIFECYCLE_STARTED_ROW) == started["row_id"]
    ledger.verify_chain()


def test_a_signalled_run_writes_one_terminal_row_that_names_the_started_row(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    try:
        with _recorder(ledger):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)
    started, terminal = _rows(tmp_path / "l.jsonl")
    assert terminal["status"] == "killed"
    assert _metric(terminal, LIFECYCLE_STARTED_ROW) == started["row_id"]


def test_a_recorder_that_cannot_write_its_started_row_does_not_start(tmp_path) -> None:
    blocked = tmp_path / "l.jsonl"
    blocked.mkdir()  # a directory where the ledger file must go: the append fails
    ran = []
    with pytest.raises(OSError), _recorder(Ledger(blocked)):
        ran.append(True)
    assert ran == []


def test_the_cap_defaults_to_the_cost_estimates_cap(tmp_path) -> None:
    from qd_train.run_control import CostEstimate, WallClockCap

    cost = CostEstimate.for_device(cap=WallClockCap(cap_s=1234.0), device="cpu")
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger, cost=cost):
        pass
    (started, _terminal) = _rows(tmp_path / "l.jsonl")
    assert _metric(started, "lifecycle.cap_s") == 1234.0


# ---------------------------------------------------------------------------
# Task 1: reconciliation
# ---------------------------------------------------------------------------


def _orphan(tmp_path: Path) -> tuple[Ledger, int]:
    path = tmp_path / "l.jsonl"
    pid = _sigkilled_run(path)
    return Ledger(path), pid


def test_reconcile_writes_one_killed_row_for_a_dead_pid(tmp_path) -> None:
    ledger, _pid = _orphan(tmp_path)
    written = ledger_module.reconcile_orphan_started_rows(ledger, datetime.now(UTC))
    assert len(written) == 1
    started, killed = _rows(tmp_path / "l.jsonl")
    assert killed["status"] == "killed"
    assert killed["quick"] is True
    assert ledger_module.ORPHAN_REASON in killed["notes"]
    assert _metric(killed, LIFECYCLE_STARTED_ROW) == started["row_id"]
    assert killed["protocol_hash"] == started["protocol_hash"]
    for gate in ledger_module.REQUIRED_GATES:
        assert killed["gates"][gate]["state"] == "not_run"
        assert killed["gates"][gate]["reason"] == ledger_module.ORPHAN_REASON
    ledger.verify_chain()  # append-only, chain preserved


def test_reconcile_is_idempotent(tmp_path) -> None:
    ledger, _pid = _orphan(tmp_path)
    first = ledger_module.reconcile_orphan_started_rows(ledger, datetime.now(UTC))
    second = ledger_module.reconcile_orphan_started_rows(ledger, datetime.now(UTC))
    assert (len(first), len(second)) == (1, 0)
    assert len(_rows(tmp_path / "l.jsonl")) == 2


def test_reconcile_leaves_a_live_process_alone(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger):  # this very process: alive
        written = ledger_module.reconcile_orphan_started_rows(
            ledger, datetime.now(UTC) + timedelta(days=30))
    assert written == []
    assert [r["status"] for r in _rows(tmp_path / "l.jsonl")] == ["started", "completed"]


def test_reconcile_never_writes_under_a_live_pid_however_overdue(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    recorder = _recorder(ledger, run_cap_s=60.0)
    recorder.__enter__()
    try:
        orphans = ledger_module.find_orphan_started_rows(
            ledger.rows(include_started=True), datetime.now(UTC) + timedelta(days=1), ledger_module.process_liveness)
        assert [o.verdict for o in orphans] == ["in_flight"]
        assert "recycled pid" in orphans[0].why
    finally:
        recorder.__exit__(None, None, None)


def test_a_pid_on_another_host_is_unverifiable_inside_its_cap_and_gone_after_it(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    recorder = _recorder(ledger, run_cap_s=100.0)
    recorder.__enter__()
    started = ledger.rows(include_started=True)[0]
    # Re-home the started row to another host: only the host metric differs.
    foreign = ledger_module.find_orphan_started_rows
    now = datetime.now(UTC)

    def elsewhere(pid: int, host: str) -> bool | None:
        return None  # what process_liveness answers for a pid on another host

    within = foreign([started], now + timedelta(seconds=50), elsewhere)
    after = foreign([started], now + timedelta(seconds=100 + ledger_module.ORPHAN_GRACE_S + 5),
                    elsewhere)
    recorder.__exit__(None, None, None)
    assert [o.verdict for o in within] == ["unverifiable"]
    assert [o.verdict for o in after] == ["gone"]


def test_a_started_row_with_no_cap_is_judged_on_its_pid_alone(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    recorder = _recorder(ledger)  # cost=None, no run_cap_s: no cap stated
    recorder.__enter__()
    started = ledger.rows(include_started=True)[0]
    far = datetime.now(UTC) + timedelta(days=365)
    verdicts = [o.verdict for o in ledger_module.find_orphan_started_rows(
        [started], far, lambda pid, host: None)]
    recorder.__exit__(None, None, None)
    assert verdicts == ["unverifiable"]


def test_a_started_row_a_terminal_row_names_is_not_an_orphan(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger):
        pass
    assert ledger_module.find_orphan_started_rows(
        ledger.rows(include_started=True), datetime.now(UTC) + timedelta(days=30), lambda pid, host: False) == []


# ---------------------------------------------------------------------------
# Task 1: visible at `make gates` (ledger-verify) and to promotion
# ---------------------------------------------------------------------------


def test_verify_fails_on_an_orphan_until_it_is_reconciled(tmp_path, capsys) -> None:
    ledger, _pid = _orphan(tmp_path)
    assert ledger_module.main(["verify", "--ledger", str(ledger.path)]) == 1
    assert "ORPHANED RUNS" in capsys.readouterr().err
    assert ledger_module.main(["reconcile", "--ledger", str(ledger.path)]) == 0
    assert ledger_module.main(["verify", "--ledger", str(ledger.path)]) == 0


def test_verify_is_clean_on_a_ledger_of_finished_runs(tmp_path) -> None:
    ledger = Ledger(tmp_path / "l.jsonl")
    with _recorder(ledger):
        pass
    assert ledger_module.main(["verify", "--ledger", str(ledger.path)]) == 0


def test_promotion_ignores_a_started_row_that_ended_and_refuses_one_that_did_not(
    tmp_path,
) -> None:
    ended = Ledger(tmp_path / "ended.jsonl")
    with _recorder(ended):
        pass
    family = _protocol().hash_without_seed()
    reasons_ended = " ".join(ended.promotion_verdict(family).reasons)
    assert "status is 'started'" not in reasons_ended

    ledger, _pid = _orphan(tmp_path)
    reasons_orphan = " ".join(ledger.promotion_verdict(family).reasons)
    assert "status is 'started'" in reasons_orphan
    time.sleep(0)  # no clock dependence: the orphan refuses before and after reconcile
    ledger_module.reconcile_orphan_started_rows(ledger, datetime.now(UTC))
    assert "status is 'started'" not in " ".join(ledger.promotion_verdict(family).reasons)
    assert not ledger.promotion_verdict(family).promoted
