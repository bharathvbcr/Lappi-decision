"""The root Cargo.toml's `exclude` keeps path dependencies under build/ out of this workspace.

The lead's integration worktree (build/v5-build-wt) reaches ojas and tessl through symlinks in
build/. Cargo caches every workspace root it loads, and when it resolves a path dependency's
inherited `workspace.package` keys it checks that cache for each ancestor directory before it walks
the filesystem. So once a dependency without a workspace of its own (tessl) has loaded this root,
ojas-core under build/ojas inherits `workspace.package` from Lappi instead of from ojas's root, and
fails on the `publish` key that Lappi does not define. `.claude/` was already excluded for the agent
worktrees; build/ needs the same entry. The fixture reproduces the shape with two throwaway
packages, under build/ because the trap needs this repository's root as an ancestor.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

FILES = {
    "ws/Cargo.toml": '[workspace]\nresolver = "3"\nmembers = ["a", "b"]\n',
    "ws/a/Cargo.toml": (
        '[package]\nname = "a"\nversion = "0.1.0"\nedition = "2024"\n\n'
        '[dependencies]\nplain = { path = "../../plain" }\n'
    ),
    "ws/b/Cargo.toml": (
        '[package]\nname = "b"\nversion = "0.1.0"\nedition = "2024"\n\n'
        '[dependencies]\ninh = { path = "../../other/inh" }\n'
    ),
    "plain/Cargo.toml": '[package]\nname = "plain"\nversion = "0.1.0"\nedition = "2024"\n',
    "other/Cargo.toml": (
        '[workspace]\nresolver = "3"\nmembers = ["inh"]\n\n[workspace.package]\npublish = false\n'
    ),
    "other/inh/Cargo.toml": (
        '[package]\nname = "inh"\nversion = "0.1.0"\nedition = "2024"\npublish.workspace = true\n'
    ),
}
LIBS = ("ws/a", "ws/b", "plain", "other/inh")


def test_a_path_dependency_under_build_inherits_from_its_own_workspace() -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is not on PATH; the workspace-root walk was not exercised")
    build = REPO / "build"
    build.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="cargo-exclude-test-", dir=build))
    try:
        for rel, text in FILES.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        for rel in LIBS:
            (root / rel / "src").mkdir(parents=True, exist_ok=True)
            (root / rel / "src" / "lib.rs").write_text("", encoding="utf-8")
        proc = subprocess.run(
            [
                cargo,
                "metadata",
                "--format-version",
                "1",
                "--no-deps",
                "--offline",
                "--manifest-path",
                str(root / "ws" / "Cargo.toml"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
    finally:
        shutil.rmtree(root)
