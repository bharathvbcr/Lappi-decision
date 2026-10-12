"""The torch-free ``make pytest`` collects: a test module that needs torch skips, never crashes.

``tools/real_ft_run.py`` raises ``SystemExit`` when torch is missing, and ``SystemExit`` out of
a test module's top level is not a collection error: it aborts the whole pytest session
(INTERNALERROR, exit 3) and nothing runs. ``test_needle_handoff.py`` imported it ungated, so
the project-venv suite ran zero tests from 2026-10-02 until it gained its ``importorskip``
(GAP-L-V5DATA-NEEDLE-HANDOFF-BREAKS-VENV-COLLECTION-2026-10-02).

This runs on any interpreter, including one that has torch: the subprocess puts a finder in
front of ``torch`` that refuses it, which is what a torch-free venv looks like to the modules.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "python" / "tests"

_DRIVER = """
import sys
import pytest


class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return None


sys.meta_path.insert(0, _NoTorch())
args = ["--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"]
sys.exit(pytest.main([*args, *sys.argv[1:]]))
"""


def _importers_of_real_ft_run() -> list[str]:
    """The test modules that import ``real_ft_run`` at their top level (a ``SystemExit`` there
    is the crash). Found by text, so a new importer is covered the day it is written."""
    out = []
    for path in sorted(TESTS.glob("test_*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(
            line.startswith(("import real_ft_run", "from real_ft_run import"))
            for line in lines
        ):
            out.append(str(path))
    return out


def test_every_module_that_imports_real_ft_run_collects_without_torch() -> None:
    files = _importers_of_real_ft_run()
    assert files, "found no importer of real_ft_run; the search above is stale"
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, *files],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    # Only the session-aborting crash is this test's business: an ordinary collection error
    # (exit 2, a missing optional package) is reported by the suite run itself.
    assert "INTERNALERROR" not in combined and proc.returncode != 3, combined[-2000:]
