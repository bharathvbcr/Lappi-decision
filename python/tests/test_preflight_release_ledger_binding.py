"""`make preflight-release` reaches the `ledger_binding` check
(GAP-W7A-LEDGER-BINDING-CHECK-NOT-WIRED-INTO-PREFLIGHT-RELEASE-2026-10-18).

The check is a line of `qd-release-check check` (crates/qd-preflight/tests/release_check_binding.rs
pins its verdicts), so the wiring is that the target's release-check step runs the `check`
subcommand over the release and the replay results. This pins that, and that an unset RELEASE_DIR
is NotRun rather than a pass, so no path through the target skips the line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-n", "-f", str(REPO / "Makefile"), "preflight-release", *args],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def test_the_release_check_step_runs_check_over_the_release_and_results() -> None:
    proc = _make(
        "RELEASE_DIR=/rel", "REPLAY_RESULTS=/results.jsonl", "MIN_ANSWERED=3",
        "GATES_VIA_MAKE_CMD=true",
    )
    assert proc.returncode == 0, proc.stderr
    expected = '--bin qd-release-check -- check --release "/rel" --results "/results.jsonl"'
    assert expected in proc.stdout
    assert "--min-answered \"3\"" in proc.stdout


def test_an_unset_release_dir_is_notrun_not_a_skipped_check() -> None:
    proc = _make("GATES_VIA_MAKE_CMD=true", "MIN_ANSWERED=3", "REPLAY_RESULTS=/r")
    assert "RELEASE_DIR is not set" in proc.stdout, proc.stdout
