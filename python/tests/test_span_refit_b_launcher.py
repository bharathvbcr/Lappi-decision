"""``tools/span_refit_b.sh``: the tracked launcher for span-refit arm B.

Its predecessor (untracked build/w3-rescore/b.sh) was lost, so the chain it ran is pinned here in
``--print`` mode, which runs nothing: the step argv, that every heavy step goes through
``tools/mac_heavy.sh``, the log directory, and the rule-3 refusal of a held-out path before any
step is shown. The ``--redo`` staging (avg-val, the gate, then everything else), the refusal of a
root under a repo ``build/`` and the ``SPAN_CODE_PIN`` check are pinned the same way. The run-mode
tests execute a fake python (a shell script) under the real ``mac_heavy.sh`` with a tmp lock; they
load no tower and read no span-refit cache.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "span_refit_b.sh"
MAIN = "/Users/bharath/Code/research/Lappi-decision"
HEAVY = f"{MAIN}/tools/mac_heavy.sh"
ML_PY = "/Users/bharath/.venvs/ml/bin/python"
CODE_REPO = "/Users/bharath/qd-campaign/worktrees/span-pin-8a531e8"
DEFAULT_ROOT = "/Users/bharath/qd-campaign/span-refit-2026-10-06"


def _run(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    own = ("SPAN_CODE_REPO", "SPAN_REFIT_B_DATE", "SPAN_REFIT_ROOT", "SPAN_CODE_PIN")
    merged = {k: v for k, v in os.environ.items() if k not in own}
    merged["SPAN_REFIT_B_PY"] = sys.executable
    merged.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=merged, check=False,
        timeout=120,
    )


def _steps(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("STEP ")]


def _step_line(label: str, log: str, tail: str, py: str = sys.executable,
               root: str = DEFAULT_ROOT, date: str = "2026-10-14") -> str:
    return (f"STEP {label} >> {root}/logs/{date}/{log}.log :: "
            f"bash {HEAVY} {label} env SPAN_CODE_REPO={CODE_REPO} {py} -u {tail}")


def _chain(root: str = DEFAULT_ROOT) -> dict[str, str]:
    """The full argv tail of each step, with its explicit paths under *root*."""
    ex, fit, score = (f"{MAIN}/tools/span_refit_{n}.py" for n in ("extract", "fit", "score"))
    return {
        "extract-avg-val": f"{ex} --tower avg --split val --device mps --out {root}/cache/avg-val",
        "gate": (f"{score} --gate-only --val-cache {root}/cache/avg-val "
                 f"--out {root}/results/gate.json"),
        "extract-avg-train": (f"{ex} --tower avg --split train --device mps "
                              f"--out {root}/cache/avg-train"),
        "extract-seed1-val": (f"{ex} --tower seed1 --split val --device mps "
                              f"--out {root}/cache/seed1-val"),
        "fit-c": f"{fit} --arm c --cache {root}/cache/avg-train --out {root}/heads/arm-c",
        "fit-e": f"{fit} --arm e --cache {root}/cache/avg-train --out {root}/heads/arm-e",
    }


def test_print_mode_lists_the_three_pinned_steps_under_mac_heavy() -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO)
    assert res.returncode == 0, res.stderr
    chain = _chain()
    assert _steps(res.stdout) == [
        _step_line("span-refit-b-extract-seed1-val", "extract-seed1-val",
                   chain["extract-seed1-val"]),
        _step_line("span-refit-b-fit-c", "fit-c", chain["fit-c"]),
        _step_line("span-refit-b-fit-e", "fit-e", chain["fit-e"]),
    ]
    assert all(f"bash {HEAVY} " in s for s in _steps(res.stdout))


def test_print_mode_names_the_finished_avg_train_extraction_as_skipped_not_redone() -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO)
    assert res.returncode == 0, res.stderr
    assert "SKIP avg-train extraction: verified, not redone" in res.stdout
    assert "--split train" not in res.stdout
    assert "--tower avg" not in res.stdout


def test_print_mode_writes_nothing_and_logs_go_under_the_dated_root(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO, SPAN_REFIT_B_DATE="2031-01-02",
               SPAN_REFIT_ROOT=str(root))
    assert res.returncode == 0, res.stderr
    assert f"{root}/logs/2031-01-02/extract-seed1-val.log" in res.stdout
    assert f"--out {root}/cache/seed1-val" in res.stdout
    assert not root.exists()
    assert not Path(f"{MAIN}/build/span-refit-b-2031-01-02").exists()


def test_redo_print_runs_the_gate_before_avg_train_and_lists_no_skip(tmp_path: Path) -> None:
    root = str(tmp_path / "campaign")
    res = _run("--print", "--redo", SPAN_CODE_REPO=CODE_REPO, SPAN_REFIT_ROOT=root)
    assert res.returncode == 0, res.stderr
    chain = _chain(root)
    want = [
        ("span-refit-b-extract-avg-val", "extract-avg-val"),
        ("span-refit-b-gate", "gate"),
        ("span-refit-b-extract-avg-train", "extract-avg-train"),
        ("span-refit-b-extract-seed1-val", "extract-seed1-val"),
        ("span-refit-b-fit-c", "fit-c"),
        ("span-refit-b-fit-e", "fit-e"),
    ]
    assert _steps(res.stdout) == [
        _step_line(label, log, chain[log], root=root) for label, log in want
    ]
    assert "SKIP" not in res.stdout
    assert not Path(root).exists()


def test_redo_flag_order_is_free_and_a_repeat_is_refused() -> None:
    assert _run("--redo", "--print", SPAN_CODE_REPO=CODE_REPO).returncode == 0
    assert _run("--redo", "--redo", "--print", SPAN_CODE_REPO=CODE_REPO).returncode == 64


@pytest.mark.parametrize("sub", ["build/span-cache", "build"])
def test_a_root_under_the_repo_build_directory_is_refused_before_any_step(sub: str) -> None:
    for repo in (MAIN, str(REPO)):
        res = _run("--print", SPAN_CODE_REPO=CODE_REPO, SPAN_REFIT_ROOT=f"{repo}/{sub}")
        assert res.returncode == 64, (repo, res.stderr)
        assert "build" in res.stderr
        assert _steps(res.stdout) == []


def test_a_symlink_into_the_repo_build_directory_is_refused_as_resolved(tmp_path: Path) -> None:
    link = tmp_path / "innocent"
    link.symlink_to(Path(MAIN) / "build")
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO, SPAN_REFIT_ROOT=str(link / "x"))
    assert res.returncode == 64
    assert _steps(res.stdout) == []


def test_a_held_out_root_is_refused_by_the_rule_3_check() -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO,
               SPAN_REFIT_ROOT="/Users/bharath/data/heldout/campaign")
    assert res.returncode == 65
    assert "held-out path refused" in res.stderr
    assert _steps(res.stdout) == []


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


def test_default_root_is_the_rulings_campaign_directory() -> None:
    assert f"${{SPAN_REFIT_ROOT:-{DEFAULT_ROOT}}}" in SCRIPT.read_text(encoding="utf-8")


# --- run mode: the pin check and the gate exit, with a fake python under the real mac_heavy ---


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True, env=env,
    )
    return done.stdout.strip()


@pytest.fixture
def pin_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "pin"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "one")
    return repo, _git(repo, "rev-parse", "HEAD")


def _fake_python(tmp_path: Path, gate_rc: int) -> tuple[Path, Path]:
    """A python that logs each non-check call and fails the scorer with *gate_rc*."""
    calls = tmp_path / "calls.log"
    fake = tmp_path / "fakepy"
    fake.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = "-I" ]; then exec {sys.executable} "$@"; fi\n'
        f'echo "$*" >> {calls}\n'
        f'case "$*" in *span_refit_score.py*) exit {gate_rc} ;; esac\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake, calls


def _run_fake(tmp_path: Path, fake: Path, repo: Path, pin: str | None,
              *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "SPAN_REFIT_B_PY": str(fake), "SPAN_CODE_REPO": str(repo),
        "SPAN_REFIT_ROOT": str(tmp_path / "campaign"),
        "MAC_HEAVY_LOCK": str(tmp_path / "lock.d"), "MAC_HEAVY_MAX_LOAD": "999999",
        "MAC_HEAVY_MAX_PROCS": "999999", "MAC_HEAVY_MIN_FREE_GB": "0",
        "MAC_HEAVY_MAX_WAIT_S": "60",
    }
    if pin is not None:
        env["SPAN_CODE_PIN"] = pin
    merged = {k: v for k, v in os.environ.items() if k != "SPAN_CODE_PIN"}
    merged.update(env)
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          env=merged, check=False, timeout=120)


def test_run_mode_without_a_pin_exits_67_before_any_step(
    tmp_path: Path, pin_repo: tuple[Path, str]
) -> None:
    repo, _ = pin_repo
    fake, calls = _fake_python(tmp_path, 0)
    res = _run_fake(tmp_path, fake, repo, None, "--redo")
    assert res.returncode == 67, res.stderr
    assert "SPAN_CODE_PIN" in res.stderr
    assert not calls.exists()


def test_run_mode_exits_67_when_head_is_not_the_pin(
    tmp_path: Path, pin_repo: tuple[Path, str]
) -> None:
    repo, _ = pin_repo
    fake, calls = _fake_python(tmp_path, 0)
    res = _run_fake(tmp_path, fake, repo, "0" * 40, "--redo")
    assert res.returncode == 67, res.stderr
    assert "HEAD" in res.stderr
    assert not calls.exists()


def test_run_mode_exits_67_when_the_pinned_tree_is_dirty(
    tmp_path: Path, pin_repo: tuple[Path, str]
) -> None:
    repo, head = pin_repo
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
    fake, calls = _fake_python(tmp_path, 0)
    res = _run_fake(tmp_path, fake, repo, head, "--redo")
    assert res.returncode == 67, res.stderr
    assert "not clean" in res.stderr
    assert not calls.exists()


def test_print_mode_does_not_verify_the_pin(tmp_path: Path) -> None:
    res = _run("--print", SPAN_CODE_REPO=CODE_REPO, SPAN_CODE_PIN="0" * 40,
               SPAN_REFIT_ROOT=str(tmp_path / "campaign"))
    assert res.returncode == 0, res.stderr


def test_a_failed_gate_exits_68_and_no_avg_train_step_runs(
    tmp_path: Path, pin_repo: tuple[Path, str]
) -> None:
    repo, head = pin_repo
    fake, calls = _fake_python(tmp_path, 3)
    res = _run_fake(tmp_path, fake, repo, head, "--redo")
    assert res.returncode == 68, (res.returncode, res.stderr)
    seen = calls.read_text(encoding="utf-8").splitlines()
    assert len(seen) == 2
    assert "span_refit_extract.py --tower avg --split val" in seen[0]
    assert "span_refit_score.py --gate-only" in seen[1]
    assert "--split train" not in calls.read_text(encoding="utf-8")


def test_a_passed_gate_lets_avg_train_run_then_stops_at_the_cache_verification(
    tmp_path: Path, pin_repo: tuple[Path, str]
) -> None:
    repo, head = pin_repo
    fake, calls = _fake_python(tmp_path, 0)
    res = _run_fake(tmp_path, fake, repo, head, "--redo")
    # The fake writes no cache, so the post-extraction verification refuses (66): the order,
    # gate then avg-train then verify, is what is under test.
    assert res.returncode == 66, (res.returncode, res.stderr)
    seen = calls.read_text(encoding="utf-8").splitlines()
    assert [("score" if "score" in s else "train" if "--split train" in s else "val")
            for s in seen] == ["val", "score", "train"]
