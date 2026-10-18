"""tools/hooks/guard_branch_switch.sh: the PreToolUse hook that refuses a branch switch over
uncommitted work (owner rule 2026-10-17; Fable ruling Q6,
AUDIT/harden-2026-10-17/fable-rulings-2026-10-18.md).

Exit 2 blocks the tool call; exit 0 lets it run. Every case runs the real script against a
real throwaway git repository, so the dirty test is git's own answer, not a mock.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "tools" / "hooks" / "guard_branch_switch.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _run(cwd: Path, command: str) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_name": "Bash", "cwd": str(cwd), "tool_input": {"command": command}})
    return subprocess.run(["bash", str(HOOK)], input=payload, text=True, capture_output=True,
                          timeout=30, check=False)


BLOCKED_WHEN_DIRTY = [
    "git checkout main",
    "git switch other",
    "git -c core.pager=cat checkout other",
    "git -C . switch other",
    "git -p checkout other",
    "git stash",
    "git stash push -- a.txt",
    "git add -A",
    "git add --all",
    "git branch -f main HEAD~1",
    "git status; git checkout other",
    "true && git switch -c new",
]


@pytest.mark.parametrize("command", BLOCKED_WHEN_DIRTY)
def test_a_tree_moving_command_over_uncommitted_work_is_refused(repo: Path, command: str) -> None:
    (repo / "a.txt").write_text("two\n")
    out = _run(repo, command)
    assert out.returncode == 2, out
    assert "commit your own changes" in out.stderr.lower()
    assert "a.txt" in out.stderr


@pytest.mark.parametrize("command", BLOCKED_WHEN_DIRTY)
def test_the_same_command_on_a_clean_tree_runs(repo: Path, command: str) -> None:
    assert _run(repo, command).returncode == 0


@pytest.mark.parametrize("command", [
    "git status",
    "git add a.txt",
    "git commit -m x -- a.txt",
    "git log --oneline main..other",
    "echo git checkout-index is not a switch",
    "rg 'git stash' docs",
])
def test_commands_that_do_not_move_the_tree_run_even_when_dirty(repo: Path, command: str) -> None:
    (repo / "a.txt").write_text("two\n")
    assert _run(repo, command).returncode == 0


def test_an_untracked_file_alone_does_not_block(repo: Path) -> None:
    (repo / "scratch.txt").write_text("x\n")
    assert _run(repo, "git checkout -b other").returncode == 0


def test_an_unreadable_payload_refuses_rather_than_passes() -> None:
    out = subprocess.run(["bash", str(HOOK)], input="not json", text=True, capture_output=True,
                         timeout=30, check=False)
    assert out.returncode == 2


def test_a_cwd_git_cannot_read_refuses_rather_than_passes(tmp_path: Path) -> None:
    out = _run(tmp_path / "nowhere", "git checkout main")
    assert out.returncode == 2
