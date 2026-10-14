"""The rustfmt gate (ft-f137c48e): every workspace member, never ``--all``.

The workspace went unformatted for its whole history (1,571 rustfmt hunks on 2026-10-07,
2,041 on 2026-10-13: every lane added more) because no gate checked it. It was reformatted
once, member by member, and ``make fmt`` now keeps it that way.

Two properties are asserted, each against what make itself would run:

1. The recipe checks exactly the members ``Cargo.toml`` declares. A hand-kept member list
   in the Makefile would drift the first time a crate is added, and the new crate would be
   the one unformatted crate with a green gate. ``--all`` is refused: it follows path
   dependencies into the ojas and tessl checkouts, which this repo does not own (rule 6).
2. ``make gates`` runs it, and when cargo is absent it reports NotRun, never PASS.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAKEFILE = REPO / "Makefile"
TIMEOUT_S = 300


def _members() -> list[str]:
    """Package names of the workspace members, read from each member's own manifest."""
    root = tomllib.loads((REPO / "Cargo.toml").read_text())
    names = []
    for member in root["workspace"]["members"]:
        manifest = tomllib.loads((REPO / member / "Cargo.toml").read_text())
        names.append(manifest["package"]["name"])
    assert names, "Cargo.toml declares no workspace members, so nothing below was checked"
    return names


def _dry_run(target: str) -> str:
    proc = subprocess.run(
        ["make", "-f", str(MAKEFILE), "--no-print-directory", "-n", target],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, (
        f"`make -n {target}` exited {proc.returncode}; its recipe could not be read.\n"
        f"stderr: {proc.stderr}"
    )
    return proc.stdout


def test_the_fmt_target_checks_every_workspace_member_and_never_all():
    recipe = _dry_run("fmt")
    assert "cargo fmt" in recipe and "--check" in recipe, (
        f"`make fmt` does not run `cargo fmt --check`:\n{recipe}"
    )
    assert "--all" not in recipe, (
        "`make fmt` passes --all, which follows path dependencies into ojas and tessl"
    )
    for name in _members():
        assert f"-p {name} " in recipe or recipe.rstrip().endswith(f"-p {name}") or (
            f"-p {name};" in recipe
        ), (
            f"workspace member {name} is not checked by `make fmt`:\n{recipe}"
        )


def test_make_gates_runs_the_fmt_gate():
    recipe = _dry_run("gates")
    assert "cargo fmt" in recipe, f"`make gates` never runs the fmt gate:\n{recipe}"


def test_the_fmt_gate_reports_notrun_without_cargo():
    proc = subprocess.run(
        ["make", "-f", str(MAKEFILE), "--no-print-directory", "fmt", "PATH=/nonexistent/bin"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=str(REPO),
    )
    assert proc.returncode != 0, "`make fmt` without cargo exited 0: a gate that never ran"
    assert "NotRun" in proc.stdout, f"`make fmt` without cargo must say NotRun: {proc.stdout!r}"
