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
# Read from the lock so a bump does not edit this file. It is not a tautology: the committed lock
# is only ever compared with a sibling this test builds, and
# `test_a_lock_that_disagrees_with_the_sibling_fails` pins that a mismatch is caught.
LOCKED = json.loads(LOCK.read_text())["ojas"]["revision"]
OTHER = "0123456789abcdef0123456789abcdef01234567"
if OTHER == LOCKED:  # pragma: no cover - the fixture sha must differ from the lock
    OTHER = "fedcba9876543210fedcba9876543210fedcba98"


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


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=TIMEOUT_S, check=True,
    )
    return proc.stdout.strip()


def _real_pin(tmp_path: Path, *, report: bool = True) -> tuple[Path, str]:
    """A real one-commit sibling, a lock naming its HEAD and an AUDIT dir; returns the make args.

    Returns ``(sibling, args)``: the args are `OJAS_DIR=`, `PIN_LOCK=` and `PIN_AUDIT=` overrides.
    """
    sibling = tmp_path / "ojas"
    sibling.mkdir()
    _git(sibling, "init", "-q")
    (sibling / "lib.rs").write_text("fn a() {}\n")
    _git(sibling, "add", "lib.rs")
    _git(sibling, "commit", "-q", "-m", "one")
    sha = _git(sibling, "rev-parse", "HEAD")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "ojas": {"url": "u", "revision": sha}}))
    audit = tmp_path / "AUDIT"
    audit.mkdir()
    if report:
        report_dir = audit / f"ojas-pin-{sha[:7]}-2026-10-17"
        report_dir.mkdir()
        (report_dir / "delta-report.md").write_text("delta\n")
    return sibling, f"OJAS_DIR={sibling}\0PIN_LOCK={lock}\0PIN_AUDIT={audit}"


def _make_pin(args: str) -> subprocess.CompletedProcess[str]:
    return _make("pin", *args.split("\0"))


def test_a_clean_pinned_sibling_with_a_delta_report_passes(tmp_path):
    _, args = _real_pin(tmp_path)
    proc = _make_pin(args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pin PASS" in proc.stdout, proc.stdout


def test_a_sibling_with_a_modified_tracked_file_fails_and_names_it(tmp_path):
    sibling, args = _real_pin(tmp_path)
    (sibling / "untracked.txt").write_text("not in HEAD\n")
    assert _make_pin(args).returncode == 0, "an untracked file must not fail the pin"
    (sibling / "lib.rs").write_text("fn changed() {}\n")
    proc = _make_pin(args)
    assert proc.returncode != 0, "a dirty sibling passed the pin gate"
    assert "pin FAILED" in proc.stdout and "lib.rs" in proc.stdout, proc.stdout
    assert "untracked.txt" not in proc.stdout, proc.stdout
    assert "pin PASS" not in proc.stdout, proc.stdout


def test_a_pin_without_a_delta_report_fails(tmp_path):
    _, args = _real_pin(tmp_path, report=False)
    proc = _make_pin(args)
    assert proc.returncode != 0, "a pin with no delta report passed the gate"
    assert "pin FAILED" in proc.stdout and "delta-report.md" in proc.stdout, proc.stdout
    assert "pin PASS" not in proc.stdout, proc.stdout


def test_a_lock_that_disagrees_with_the_sibling_fails(tmp_path):
    # The guard against LOCKED being read from the lock: a mismatch must still be caught.
    sibling, _ = _real_pin(tmp_path)
    other_lock = tmp_path / "other.json"
    lock = {"schema_version": 1, "ojas": {"url": "u", "revision": OTHER}}
    other_lock.write_text(json.dumps(lock))
    audit = tmp_path / "AUDIT"
    proc = _make("pin", f"OJAS_DIR={sibling}", f"PIN_LOCK={other_lock}", f"PIN_AUDIT={audit}")
    assert proc.returncode != 0 and "pin FAILED" in proc.stdout, proc.stdout


def test_the_committed_pin_has_a_delta_report_or_a_cited_allowance():
    reports = list((REPO / "AUDIT").glob(f"ojas-pin-{LOCKED[:7]}-*/delta-report.md"))
    allowance = (REPO / "crates/qd-preflight/src/pin.rs").read_text()
    assert reports or LOCKED in allowance, f"upstream.lock.json pins {LOCKED} with no delta report"


def test_a_sibling_git_cannot_read_is_notrun_never_a_pass(tmp_path):
    # HEAD matches the lock but `git status` cannot run: not a pass.
    sibling = _sibling(tmp_path, LOCKED)
    proc = _make("pin", f"OJAS_DIR={sibling}")
    assert proc.returncode != 0, proc.stdout
    assert "pin NotRun" in proc.stdout and "pin PASS" not in proc.stdout, proc.stdout


def test_the_pin_target_runs_qd_pin_against_the_lock_and_the_sibling():
    recipe = _dry_run("pin")
    assert "qd-pin" in recipe, f"`make pin` does not run qd-pin:\n{recipe}"
    assert "upstream.lock.json" in recipe, f"`make pin` does not read the lock:\n{recipe}"
    assert "ojas" in recipe, f"`make pin` does not name the sibling:\n{recipe}"


def test_make_gates_runs_the_pin_gate():
    recipe = _dry_run("gates")
    assert "qd-pin" in recipe, f"`make gates` never runs the pin gate:\n{recipe}"


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
