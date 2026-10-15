"""The sibling-pin gate: ``make pin`` and its place in ``make gates``.

Lappi builds against the ``ojas`` checkout beside it by path, so what it builds against is
whatever commit that checkout is on. ``upstream.lock.json`` names the commit and ``make pin``
(``qd-pin``, ``crates/qd-preflight/src/pin.rs``) compares. Asserted here, against what make
itself runs:

1. ``make pin`` and ``make gates`` both run ``qd-pin``, with the lock and the overridable
   ``OJAS_DIR``.
2. A matching sibling prints ``pin PASS``; a sibling at another commit fails and names both
   commits; an absent or empty sibling is ``NotRun`` and never a pass. GNU make collapses every
   failed recipe to exit 2 (see the Makefile header), so the three-way distinction is read from
   the printed line, and ``qd-pin``'s own exit codes are pinned by the Rust unit tests.
3. The committed lock is well formed.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAKEFILE = REPO / "Makefile"
LOCK = REPO / "upstream.lock.json"
TIMEOUT_S = 600
LOCKED = "132b46ea63e87868f468c016e02fe99acf2bb164"
OTHER = "0123456789abcdef0123456789abcdef01234567"


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-f", str(MAKEFILE), "--no-print-directory", *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(REPO),
    )


def _dry_run(target: str) -> str:
    proc = _make("-n", target)
    assert proc.returncode == 0, (
        f"`make -n {target}` exited {proc.returncode}; its recipe could not be read.\n"
        f"stderr: {proc.stderr}"
    )
    return proc.stdout


def _sibling(tmp_path: Path, sha: str) -> Path:
    sibling = tmp_path / "sibling"
    (sibling / ".git").mkdir(parents=True)
    (sibling / ".git" / "HEAD").write_text(sha + "\n")
    return sibling


def test_the_committed_lock_names_a_full_commit():
    lock = json.loads(LOCK.read_text())
    assert lock["schema_version"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", lock["ojas"]["revision"]), lock["ojas"]
    assert lock["ojas"]["revision"] == LOCKED


def test_the_pin_target_runs_qd_pin_against_the_lock_and_the_sibling():
    recipe = _dry_run("pin")
    assert "qd-pin" in recipe, f"`make pin` does not run qd-pin:\n{recipe}"
    assert "upstream.lock.json" in recipe, f"`make pin` does not read the lock:\n{recipe}"
    assert "ojas" in recipe, f"`make pin` does not name the sibling:\n{recipe}"


def test_make_gates_runs_the_pin_gate():
    recipe = _dry_run("gates")
    assert "qd-pin" in recipe, f"`make gates` never runs the pin gate:\n{recipe}"


def test_a_sibling_at_the_locked_commit_passes(tmp_path):
    proc = _make("pin", f"OJAS_DIR={_sibling(tmp_path, LOCKED)}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pin PASS" in proc.stdout, proc.stdout


def test_a_sibling_at_another_commit_fails_and_names_both(tmp_path):
    proc = _make("pin", f"OJAS_DIR={_sibling(tmp_path, OTHER)}")
    assert proc.returncode != 0, "a stale sibling passed the pin gate"
    assert "pin FAILED" in proc.stdout, proc.stdout
    assert LOCKED in proc.stdout and OTHER in proc.stdout, proc.stdout
    assert "NotRun" not in proc.stdout, "a ran-and-failed pin was reported as NotRun"


def test_an_unresolvable_sibling_is_notrun_never_a_pass(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for sibling in (empty, tmp_path / "does-not-exist"):
        proc = _make("pin", f"OJAS_DIR={sibling}")
        assert proc.returncode != 0, f"`make pin` with no sibling exited 0: {proc.stdout!r}"
        assert "pin NotRun" in proc.stdout, proc.stdout
        assert "pin PASS" not in proc.stdout, proc.stdout


def test_the_pin_gate_reports_notrun_without_cargo():
    proc = _make("pin", "PATH=/nonexistent/bin")
    assert proc.returncode != 0, "`make pin` without cargo exited 0: a gate that never ran"
    assert "NotRun" in proc.stdout, f"`make pin` without cargo must say NotRun: {proc.stdout!r}"
