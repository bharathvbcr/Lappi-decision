"""``tools/stress_runner.sh`` (card ft-b5110a5c0ed): the three runner defects of 2026-10-06.

The script is sourced with ``STRESS_RUNNER_SOURCE_ONLY=1`` for its functions, so no test runs a
cargo step. What is pinned:

- a step that passes its bound is ``timed_out``, never ``failed:101``, and a test that killed a
  child at its own bound ("ran past ... and was killed", exit 101) is ``timed_out`` as well;
- the pin check is made on the worktree, not on the shared checkout: a worktree at the pinned sha
  is unaffected by commits landing on the main checkout afterwards, and a worktree whose HEAD is
  not the pin is refused (exit 70) with HEAD named;
- the job refuses to run outside the heavy-job lock, and the launcher goes through ``mac_heavy.sh``;
- U4 is tri-state and ``timed_out`` / ``not_run`` never read ``passed``.

Light: bash and git on temp repositories only, no build.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "stress_runner.sh"


def _bash(
    script: str, *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    full_env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/"),
        "STRESS_RUNNER_SOURCE_ONLY": "1",
        **(env or {}),
    }
    return subprocess.run(
        ["bash", "-c", f"source {RUNNER}\n{script}"],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _status(out: Path) -> dict[str, str]:
    rows = [line.split("\t") for line in (out / "status.tsv").read_text().splitlines()]
    return {r[0]: r[1] for r in rows}


@pytest.fixture
def out(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    (d / "status.tsv").write_text("")
    return d


def test_a_step_past_its_bound_is_timed_out_and_its_process_group_is_killed(
    tmp_path: Path, out: Path
) -> None:
    marker = tmp_path / "still-running"
    res = _bash(
        f"OUT={out}\nbounded_step slow 2 bash -c 'sleep 1 & (sleep 30; touch {marker}) & wait'\n"
        "echo rc=$?",
        cwd=tmp_path,
    )
    assert "rc=124" in res.stdout, res.stdout + res.stderr
    assert _status(out)["slow"] == "timed_out"
    assert "failed" not in (out / "status.tsv").read_text()
    # The grandchild went with the group: it never got to its touch.
    import time

    time.sleep(1)
    assert not marker.exists()


def test_a_child_the_test_killed_at_its_own_bound_is_timed_out_not_101(
    tmp_path: Path, out: Path
) -> None:
    res = _bash(
        f"OUT={out}\n"
        "bounded_step own 60 bash -c 'echo \"x: ran past 1800s and was killed\"; exit 101'\n",
        cwd=tmp_path,
    )
    assert _status(out)["own"] == "timed_out", res.stdout + res.stderr


def test_a_failure_stays_a_failure_with_its_code_and_a_pass_a_pass(
    tmp_path: Path, out: Path
) -> None:
    _bash(
        f"OUT={out}\nbounded_step bad 60 bash -c 'echo nondeterministic; exit 101'\n"
        "bounded_step good 60 true\n",
        cwd=tmp_path,
    )
    status = _status(out)
    assert status["bad"] == "failed:101"
    assert status["good"] == "passed"


def _git(repo: Path, *args: str) -> str:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(repo),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "F",
        "GIT_AUTHOR_EMAIL": "f@example.com",
        "GIT_COMMITTER_NAME": "F",
        "GIT_COMMITTER_EMAIL": "f@example.com",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "main"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_the_pin_is_a_worktree_so_commits_landing_on_main_do_not_move_it(
    tmp_path: Path, out: Path
) -> None:
    repo, sha = _repo(tmp_path)
    wt = tmp_path / "wt"
    script = f"OUT={out}\nREPO={repo}\npin_worktree {sha} {wt}\necho rc=$?\n"
    res = _bash(script, cwd=tmp_path)
    assert "rc=0" in res.stdout, res.stdout + res.stderr
    # Four commits land on the main checkout while the "lock" is waited for (the 2026-10-06 defect).
    for i in range(4):
        (repo / "a.txt").write_text(f"landed {i}\n")
        _git(repo, "commit", "-q", "-am", f"landed {i}")
    assert _git(wt, "rev-parse", "HEAD") == sha
    assert (wt / "a.txt").read_text() == "one\n"
    res = _bash(f"OUT={out}\ncheck_head {sha} {wt}\necho rc=$?\n", cwd=tmp_path)
    assert "rc=0" in res.stdout
    # The shared checkout's HEAD is not the pin any more; checking it is exactly what failed.
    res = _bash(f"OUT={out}\ncheck_head {sha} {repo}\necho rc=$?\n", cwd=tmp_path)
    assert "rc=70" in res.stdout and "not the pinned" in res.stderr, res.stdout + res.stderr


def test_a_worktree_that_is_not_at_the_pin_or_is_dirty_is_refused(
    tmp_path: Path, out: Path
) -> None:
    repo, sha = _repo(tmp_path)
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-q", "-am", "two")
    wt = tmp_path / "wt"
    _bash(f"OUT={out}\nREPO={repo}\npin_worktree {sha} {wt}\n", cwd=tmp_path)
    _git(
        wt,
        "checkout",
        "-q",
        "--detach",
        "main" if _git(repo, "branch", "--show-current") == "main" else "master",
    )
    res = _bash(f"OUT={out}\ncheck_head {sha} {wt}\necho rc=$?\n", cwd=tmp_path)
    assert "rc=70" in res.stdout and "HEAD of" in res.stderr, res.stdout + res.stderr
    _git(wt, "checkout", "-q", "--detach", sha)
    (wt / "a.txt").write_text("dirty\n")
    res = _bash(f"OUT={out}\ncheck_head {sha} {wt}\necho rc=$?\n", cwd=tmp_path)
    assert "rc=70" in res.stdout and "tracked changes" in res.stderr, res.stdout + res.stderr


@pytest.mark.parametrize("bad", ["HEAD", "abc123", "a" * 39, "A" * 40, ""])
def test_a_pin_that_is_not_a_full_lowercase_sha_is_refused(
    tmp_path: Path, out: Path, bad: str
) -> None:
    repo, _ = _repo(tmp_path)
    res = _bash(
        f"OUT={out}\nREPO={repo}\npin_worktree '{bad}' {tmp_path / 'wt'}\necho rc=$?\n",
        cwd=tmp_path,
    )
    assert "rc=70" in res.stdout and "full 40-hex sha" in res.stderr, res.stdout + res.stderr
    assert not (tmp_path / "wt").exists()


def test_the_job_refuses_to_run_outside_the_lock(tmp_path: Path) -> None:
    res = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "job",
            "--sha",
            "a" * 40,
            "--profile",
            "release",
            "--out",
            str(tmp_path / "o"),
        ],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=tmp_path,
    )
    assert res.returncode == 64 and "MAC_HEAVY_LABEL is unset" in res.stderr, res.stderr


def test_the_launcher_goes_through_the_heavy_job_lock_and_a_profile_is_required() -> None:
    text = RUNNER.read_text()
    assert 'tools/mac_heavy.sh" "stress-$LABEL"' in text
    res = subprocess.run(
        ["bash", str(RUNNER), "launch", "--sha", "a" * 40],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert res.returncode == 64, "no --profile: the profile is a recorded input, never a default"


@pytest.mark.parametrize(
    ("states", "want"),
    [
        (("passed", "passed", "passed"), "passed"),
        (("passed", "timed_out", "passed"), "undetermined"),
        (("passed", "not_run:x", "passed"), "undetermined"),
        (("passed", "failed:101", "timed_out"), "failed"),
        (("passed", "passed", ""), "undetermined"),
    ],
)
def test_u4_is_tri_state_and_only_three_passes_read_passed(
    tmp_path: Path, out: Path, states: tuple[str, str, str], want: str
) -> None:
    lines = [
        f"{n}\t{s}\tt0\tt1"
        for n, s in zip(("s8-email", "s8-jarvis", "s8-tools"), states, strict=False)
        if s
    ]
    (out / "status.tsv").write_text("\n".join(lines) + ("\n" if lines else ""))
    res = _bash(f"OUT={out}\nu4_state\n", cwd=tmp_path)
    assert res.stdout.strip() == want, res.stdout + res.stderr


def test_the_runner_names_no_test_that_has_been_renamed() -> None:
    """The 2026-10-06 runner named `a_pending_dataset_licence_is_kept_and_named_never_admitted`,
    which the licence registration renamed; a `--exact` step on a missing name runs nothing."""
    text = RUNNER.read_text()
    assert "a_pending_dataset_licence_is_kept_and_named_never_admitted" not in text
    lic = (ROOT / "crates/qd-prep/src/convert_licence.rs").read_text()
    for name in textwrap.dedent(
        """\
        an_unstated_licence_is_refused_with_its_own_reason
        aliases_and_whitespace_are_python_s_and_no_more
        non_commercial_is_disqualifying_by_segment_not_substring
        the_human_registered_dataset_licences_are_admitted_and_named
        """
    ).split():
        assert f"fn {name}" in lic, name
        assert name in text, name


def test_sigterm_to_the_runner_stops_the_running_step_before_it_exits(
    tmp_path: Path, out: Path
) -> None:
    """verify-lane6 D14: the EXIT trap ran and the step survived in its own process group."""
    import signal
    import time

    unique = f"sleep {770 + os.getpid() % 100}"
    script = f"source {RUNNER}\nOUT={out}\nbounded_step slow 60 /bin/{unique}\n"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "STRESS_RUNNER_SOURCE_ONLY": "1"}
    proc = subprocess.Popen(["bash", "-c", script], env=env, cwd=tmp_path)
    time.sleep(2)
    assert subprocess.run(["pgrep", "-f", unique], capture_output=True, check=False).returncode == 0
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    time.sleep(1)
    assert (
        subprocess.run(["pgrep", "-f", unique], capture_output=True, check=False).returncode == 1
    ), "the step outlived the runner"
    assert _status(out)["slow"] == "interrupted:TERM"


def test_the_synth_runs_own_timeout_panic_is_timed_out_not_101(tmp_path: Path, out: Path) -> None:
    """verify-lane6 D15: stress_v6.rs s6 panics "the synth run passed 1800s" at its own bound."""
    _bash(
        f"OUT={out}\nbounded_step s6-kill 60 "
        "bash -c 'echo \"the synth run passed 1800s\"; exit 101'\n",
        cwd=tmp_path,
    )
    assert _status(out)["s6-kill"] == "timed_out"
    text = RUNNER.read_text()
    assert "bounded_step s6-kill" in text, "s6-kill goes through bounded_step"


def test_no_function_is_defined_twice_and_the_job_body_appears_once() -> None:
    """65802d5 spliced a second copy of the script body into the first: sourcing stops at the
    SOURCE_ONLY return, so the copy never ran in the tests, but a real job ran the worktree pin
    twice (the second `pin_worktree` refuses its own tree)."""
    import re
    from collections import Counter

    text = RUNNER.read_text()
    defs = Counter(re.findall(r"^([a-z_0-9]+)\(\) [{(]", text, flags=re.M))
    assert {name: n for name, n in defs.items() if n > 1} == {}
    assert len(re.findall(r"^: > \"\$OUT/status.tsv\"$", text, flags=re.M)) == 1
    assert len(re.findall(r"^trap cleanup EXIT$", text, flags=re.M)) == 1
