"""``tools/span_refit_b.sh``: the tracked launcher for span-refit arm B.

Its predecessor (untracked build/w3-rescore/b.sh) was lost, so the chain it ran is pinned here in
``--print`` mode, which runs nothing: the step argv, that every heavy step goes through
``tools/mac_heavy.sh``, the log directory, and the rule-3 refusal of a held-out path before any
step is shown. Nothing here takes the Mac lock, loads a tower or reads the span-refit caches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "span_refit_b.sh"
MAIN = "/Users/bharath/Code/research/Lappi-decision"
HEAVY = f"{MAIN}/tools/mac_heavy.sh"
ML_PY = "/Users/bharath/.venvs/ml/bin/python"
CODE_REPO = "/Users/bharath/Code/research/Lappi-decision/build/span-pin-8a531e8"


def _run(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    own = ("SPAN_CODE_REPO", "SPAN_REFIT_B_DATE")
    merged = {k: v for k, v in os.environ.items() if k not in own}
    merged["SPAN_REFIT_B_PY"] = sys.executable
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=merged, check=False,
        timeout=120,
    )


def _steps(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("STEP ")]


def _step_line(label: str, log: str, tail: str, py: str = sys.executable) -> str:
    return (f"STEP {label} >> {MAIN}/build/span-refit-b-2026-10-14/{log}.log :: "
            f"bash {HEAVY} {label} env SPAN_CODE_REPO={CODE_REPO} {py} -u {tail}")


def test_print_mode_lists_the_three_pinned_steps_under_mac_heavy() -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO)
    assert res.returncode == 0, res.stderr
    assert _steps(res.stdout) == [
        _step_line("span-refit-b-extract-seed1-val", "extract-seed1-val",
                   f"{MAIN}/tools/span_refit_extract.py --tower seed1 --split val --device mps"),
        _step_line("span-refit-b-fit-c", "fit-c", f"{MAIN}/tools/span_refit_fit.py --arm c"),
        _step_line("span-refit-b-fit-e", "fit-e", f"{MAIN}/tools/span_refit_fit.py --arm e"),
    ]
    assert all(f"bash {HEAVY} " in s for s in _steps(res.stdout))


def test_print_mode_names_the_finished_avg_train_extraction_as_skipped_not_redone() -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO)
    assert res.returncode == 0, res.stderr
    assert "SKIP avg-train extraction: verified, not redone" in res.stdout
    assert "--split train" not in res.stdout
    assert "--tower avg" not in res.stdout


def test_print_mode_writes_nothing_and_logs_go_under_the_dated_build_directory() -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO, SPAN_REFIT_B_DATE="2031-01-02")
    assert res.returncode == 0, res.stderr
    assert f"{MAIN}/build/span-refit-b-2031-01-02/extract-seed1-val.log" in res.stdout
    assert not Path(f"{MAIN}/build/span-refit-b-2031-01-02").exists()


def test_unset_code_repo_is_refused_not_defaulted() -> None:
    res = _run("--print")
    assert res.returncode == 64
    assert "SPAN_CODE_REPO is not set" in res.stderr
    assert _steps(res.stdout) == []


def test_held_out_code_repo_is_refused_before_any_step_is_shown() -> None:
    res = _run("--print", SPAN_CODE_REPO="/Users/bharath/data/heldout/checkout")
    assert res.returncode == 65
    assert "held-out path refused" in res.stderr
    assert _steps(res.stdout) == []


def test_symlink_into_a_held_out_directory_is_refused_as_resolved(tmp_path: Path) -> None:
    target = tmp_path / "held-out"
    target.mkdir()
    link = tmp_path / "innocent"
    link.symlink_to(target)
    res = _run("--print", SPAN_CODE_REPO=str(link))
    assert res.returncode == 65
    assert "held-out path refused" in res.stderr
    assert _steps(res.stdout) == []


def test_malformed_date_and_stray_arguments_are_refused() -> None:
    assert _run("--print", SPAN_CODE_REPO=CODE_REPO, SPAN_REFIT_B_DATE="../x").returncode == 64
    assert _run("--run", SPAN_CODE_REPO=CODE_REPO).returncode == 64
    assert _run("--print", "extra", SPAN_CODE_REPO=CODE_REPO).returncode == 64


def test_default_python_is_the_ml_venv_the_ruling_and_handoff_use() -> None:
    assert f"PY=${{SPAN_REFIT_B_PY:-{ML_PY}}}" in SCRIPT.read_text(encoding="utf-8")
