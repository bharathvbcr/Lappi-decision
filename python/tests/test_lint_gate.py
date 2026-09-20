"""The lint gate, and the two ways it had silently stopped being a gate.

This repo's recurring defect is a gate that exists in configuration and has never
executed. It has now been found three times:

1. the ledger went its entire history without a single row;
2. ``ruff>=0.6`` sat declared in ``[project.optional-dependencies].dev`` and had never
   been installed into ``.venv``, so ``ruff check`` had never run against this tree;
3. ruff's own isort was resolving ``qd_data``/``qd_train``/``qd_wire``/``qd_label`` as
   third-party, because nothing told it the packages live under ``python/``. I001 was
   selected the whole time and was checking the wrong thing: 3 findings before
   ``[tool.ruff].src`` was set, 29 after.

(3) is the interesting one. A rule that is enabled but mis-resolved is worse than one
that is switched off, because the report says "clean" either way.

The gate lives here, in the pytest suite, rather than only in the Makefile, for a
reason that is specific to this repo: ``docs/ledger-schema.md`` records a suite by
parsing its counts, and there are exactly two parsers, cargo and pytest. ``ruff check``
prints no test summary, so ``run_suite`` would read a *clean* ruff run as ``not_run``
(exit 0, no counts, indistinguishable from having checked nothing) and a dirty one as
``ran/failed``. Asserting lint from inside pytest means the lint gate is counted, and
lands in the ledger row, without teaching the ledger a third parser.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
VENV_RUFF = REPO / ".venv" / "bin" / "ruff"
MAKEFILE = REPO / "Makefile"

# uv.lock already resolves `ruff>=0.6` to this. The gate is the locked version in the
# project venv -- not whatever standalone ruff happens to be on PATH, which on this
# host is a different release (0.15.20) that nothing in the project pins.
LOCKED_RUFF = "0.16.8"

TIMEOUT_S = 300


def _ruff(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(VENV_RUFF), *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(cwd) if cwd else None,
    )


requires_ruff = pytest.mark.skipif(
    not VENV_RUFF.exists(),
    reason="ruff is not installed in .venv; test_ruff_is_installed_not_merely_declared "
    "is the test that reports that, and it does not skip",
)


# --------------------------------------------------------------------------
# 1 -- the declared dependency is actually present
# --------------------------------------------------------------------------


def test_ruff_is_installed_not_merely_declared():
    """`ruff>=0.6` is declared in the dev extra. Declared is not installed.

    This test fails against the tree as it stood before this lane: the dependency was
    in pyproject.toml and `.venv/bin/python -m ruff` answered "No module named ruff".
    A linter nobody can run is a lint gate that has never once executed.
    """
    assert VENV_RUFF.exists(), (
        f"{VENV_RUFF} is missing. 'ruff>=0.6' is declared in pyproject's dev extra and "
        f"uv.lock resolves it to {LOCKED_RUFF}, but it was never installed. "
        f"Install the gate: uv pip install --python {REPO}/.venv/bin/python "
        f"'ruff=={LOCKED_RUFF}'"
    )
    proc = _ruff("--version")
    assert proc.returncode == 0, f"ruff is present but will not run: {proc.stderr}"


@requires_ruff
def test_the_installed_ruff_is_the_version_the_lockfile_resolved():
    """Pin the gate to the lockfile, not to whatever is on PATH.

    Two ruffs exist on this host and they are different releases. A gate whose version
    depends on PATH order is a gate whose findings are not reproducible.
    """
    proc = _ruff("--version")
    assert proc.stdout.split()[1] == LOCKED_RUFF, (
        f"venv ruff is {proc.stdout.strip()!r}, but uv.lock resolves ruff to "
        f"{LOCKED_RUFF}. The gate and the lockfile must agree."
    )


# --------------------------------------------------------------------------
# 2 -- the gate itself
# --------------------------------------------------------------------------


@requires_ruff
def test_the_repository_has_no_ruff_findings():
    """The gate. Fails against the pre-fix tree with 113 findings.

    Scoped to the whole repo rather than to `python/`: `stack/` and `tools/` are Python
    too, and a gate scoped so that it cannot fail is the defect this file is about.
    ruff honours .gitignore, so `.venv/` and the sibling agent worktrees under
    `.claude/worktrees/` are not scanned.
    """
    proc = _ruff("check", "--config", str(PYPROJECT), str(REPO))
    assert proc.returncode == 0, (
        "ruff reports findings:\n"
        + proc.stdout[-8000:]
        + "\nRun `make lint` to reproduce, or `ruff check --fix` for the mechanical ones."
    )


# --------------------------------------------------------------------------
# 3 -- the two ways the gate can be present and still not be a gate
# --------------------------------------------------------------------------


@requires_ruff
def test_ruff_resolves_this_projects_packages_as_first_party(tmp_path: Path):
    """`[tool.ruff].src` must name `python/`, or isort mis-groups every local import.

    Behavioural, not a config read: a file importing a third-party package and a local
    one must be told to separate them. Without `src = ["python"]` ruff believes
    `qd_train` is third-party, merges the two into one block, and I001 reports nothing
    wrong -- which is how 26 mis-grouped blocks passed a selected rule.
    """
    probe = tmp_path / "probe.py"
    probe.write_text("import numpy as np\nimport pytest\nfrom qd_train.tristate import Ran\n")

    proc = _ruff("check", "--config", str(PYPROJECT), "--select", "I001", str(probe))

    assert proc.returncode != 0, (
        "ruff accepted a third-party import and a first-party qd_train import in one "
        "unseparated block. That means it is resolving qd_train as third-party, so "
        "[tool.ruff].src no longer names the directory the packages live in."
    )


@requires_ruff
def test_narrowing_the_rule_set_makes_ruff100_report_live_directives_as_dead(tmp_path: Path):
    """Why the lint gate must never pass `--select`. This one drew blood.

    RUF100 ("unused noqa") is evaluated against the *enabled* rules, so a narrowed
    `--select` that omits E402 makes ruff declare every `# noqa: E402` unused --
    including directives that are load-bearing. Running
    `ruff check --select F401,RUF100 --fix` during this lane stripped 12 live
    directives and took the tree from 0 E402 findings to 27.

    The tree is the evidence for the general rule; this test pins it on a fixture so it
    keeps holding after the tree changes.
    """
    probe = tmp_path / "probe.py"
    # `import os` after a statement is E402; the directive suppresses it and is live.
    probe.write_text("VALUE = 1\nimport os  # noqa: E402\n\nprint(os, VALUE)\n")

    with_e402 = _ruff("check", "--isolated", "--select", "E402,RUF100", str(probe))
    without_e402 = _ruff("check", "--isolated", "--select", "RUF100", str(probe))

    assert with_e402.returncode == 0, (
        "with E402 enabled the directive is used, so RUF100 must stay silent; got:\n"
        + with_e402.stdout
    )
    assert "RUF100" in without_e402.stdout, (
        "expected the narrowed rule set to misreport the live directive as unused -- "
        "the premise this gate's no---select rule rests on. ruff may have changed "
        "behaviour; re-check the Makefile's lint target if so.\n" + without_e402.stdout
    )


def test_the_makefile_lint_target_does_not_narrow_the_rule_set():
    """The Makefile must invoke ruff with the configured rule set, whole."""
    assert MAKEFILE.exists(), f"no Makefile at {MAKEFILE}"
    lines = [
        ln for ln in MAKEFILE.read_text().splitlines()
        if "check" in ln and "RUFF" in ln and not ln.lstrip().startswith("#")
    ]
    assert lines, (
        "no `$(RUFF) check` invocation found in the Makefile -- the lint gate is not "
        "wired, or was renamed and this test no longer looks at it"
    )
    for ln in lines:
        assert "--select" not in ln, (
            f"the Makefile narrows ruff's rule set, which makes RUF100 misreport live "
            f"directives as dead: {ln.strip()!r}"
        )


# --------------------------------------------------------------------------
# 4 -- the asymmetry the whole repo rests on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "override"),
    [
        ("lint", "VENV=/nonexistent/venv"),
        ("pytest", "PY=/nonexistent/python"),
        ("ledger-verify", "LEDGER=/nonexistent/runs.jsonl"),
    ],
)
def test_a_gate_that_cannot_run_never_exits_zero(target: str, override: str):
    """Non-zero and a printed reason, never a silent zero.

    This is the invariant the ledger's tri-state, `stack/verify_*.py`'s `--require-*`
    flags and the Makefile all encode: a check that could not run must not leave the
    same trace as a check that ran and passed. Exit 0 here would make every gate pass
    on any machine missing the tool -- which is every machine that has not been set up,
    and was this one for ruff.

    The recipe exits 3, but GNU make collapses every failed recipe to its own status 2
    (measured: `make: *** [lint] Error 3` with `$?` == 2), so the assertion is on
    "non-zero and said NotRun" rather than on the literal 3. `make gates` recovers the
    real code by running the gate commands itself; that path is asserted separately by
    test_the_aggregate_reports_notrun_rather_than_passing.

    Each override is chosen to short-circuit before the gate's real work: `make pytest`
    without one would re-enter this very suite.
    """
    proc = subprocess.run(
        ["make", "-f", str(MAKEFILE), "--no-print-directory", target, override],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(REPO),
    )
    assert proc.returncode != 0, (
        f"`make {target} {override}` exited 0 although the gate could not run. "
        f"That is the failure this repo exists to prevent.\nstdout: {proc.stdout}"
    )
    assert "NotRun" in proc.stdout, (
        f"a gate that could not run must say so on stdout; got: {proc.stdout!r}"
    )


def test_the_aggregate_reports_notrun_rather_than_passing():
    """`make gates` must distinguish NotRun from PASS in its own RESULT line.

    Driven with every tool pointed at nothing, so all four gates short-circuit: this
    runs no suite and cannot re-enter pytest. It asserts the aggregation rule from
    docs/ledger-schema.md -- an aggregate is never more confident than its least
    informed input.
    """
    proc = subprocess.run(
        [
            "make", "-f", str(MAKEFILE), "--no-print-directory", "gates",
            "VENV=/nonexistent/venv", "PY=/nonexistent/python",
            "LEDGER=/nonexistent/runs.jsonl", "PATH=/nonexistent/bin",
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(REPO),
    )
    assert "RESULT: NotRun" in proc.stdout, (
        "with every tool absent the aggregate must report NotRun, not PASS and not "
        f"FAILED.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert proc.returncode != 0, "an aggregate that could not run must not exit 0"


def test_every_makefile_gate_declares_all_three_outcomes():
    """The contract is documented in the file that implements it, or it drifts."""
    text = MAKEFILE.read_text()
    for token in ("exit 0", "exit 1", "exit 3", "NotRun"):
        assert token in text, f"the Makefile never mentions {token!r}"


if __name__ == "__main__":  # pragma: no cover - convenience only
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
