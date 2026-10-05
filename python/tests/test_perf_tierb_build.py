"""tools/perf_tierb_outcome.sh --build / --link: the clone gets the source's git-ignored data links.

`--build` makes the candidate's code with `git clone --local` of the phase-3 checkout, and a
clone carries no git-ignored file. The checkout's data is git-ignored symlinks
(data/pool/commitpackft-pool-v2.jsonl and eight more), so the fused outcome run (tierb2) died on
its first read of the pool at 2026-10-05 00:41:19Z (HANDOFF/lead-pipeline-2026-10-03.md).
Each test builds a fake checkout and a fake perf root under tmp_path and points the script at
them (TIERB_SRC, TIERB_PERF), so none touches /home/ubuntu.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "perf_tierb_outcome.sh"


def _git(cwd: Path, *args: str, env: dict[str, str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=60
    ).stdout


def _env(tmp_path: Path) -> dict[str, str]:
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(empty),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """(src checkout, perf root, data store, env): src has two ignored links into the store."""
    env = _env(tmp_path)
    store = tmp_path / "store"
    (store / "corpus").mkdir(parents=True)
    (store / "pool-v2.jsonl").write_text("pool\n")
    (store / "corpus" / "examples.jsonl").write_text("ex\n")
    src = tmp_path / "qd-lane2"
    (src / "data" / "pool" / "corpus").mkdir(parents=True)
    (src / "tools").mkdir()
    (src / ".gitignore").write_text("data/pool/*.jsonl\ndata/pool/*/*.jsonl\n__pycache__/\n")
    (src / "data" / "pool" / "corpus" / "manifest.json").write_text("{}\n")
    (src / "tools" / "a.py").write_text("a = 1\n")
    (src / "tools" / "b.py").write_text("b = 1\n")
    (src / "tools" / "c.py").write_text("c = 1\n")
    _git(src, "init", "-q", "-b", "main", env=env)
    _git(src, "add", "-A", env=env)
    _git(src, "commit", "-qm", "base", env=env)
    (src / "data" / "pool" / "pool-v2.jsonl").symlink_to(store / "pool-v2.jsonl")
    (src / "data" / "pool" / "corpus" / "examples.jsonl").symlink_to(store / "corpus" / "examples.jsonl")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_text("")
    perf = tmp_path / "perf"
    perf.mkdir()
    for name, rel, new in (
        ("trainstep.patch", "tools/a.py", "a = 2\n"),
        ("mirror.patch", "tools/b.py", "b = 2\n"),
        ("nomask.patch", "tools/c.py", "c = 1\nmask = None\n"),
    ):
        path = src / rel
        old = path.read_text()
        path.write_text(new)
        (perf / name).write_text(_git(src, "diff", "--", rel, env=env))
        path.write_text(old)
    return src, perf, store, env


def _run(
    src: Path, perf: Path, env: dict[str, str], cand: str, action: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), cand, action],
        env={**env, "TIERB_SRC": str(src), "TIERB_PERF": str(perf)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_build_links_every_ignored_data_symlink_of_the_source(tmp_path: Path) -> None:
    src, perf, store, env = _fixture(tmp_path)
    proc = _run(src, perf, env, "fused", "--build")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    code = perf / "p3fused"
    pool = code / "data" / "pool" / "pool-v2.jsonl"
    examples = code / "data" / "pool" / "corpus" / "examples.jsonl"
    assert pool.is_symlink() and os.readlink(pool) == str(store / "pool-v2.jsonl")
    assert examples.is_symlink() and os.readlink(examples) == str(store / "corpus" / "examples.jsonl")
    assert pool.read_text() == "pool\n"
    assert not (code / "__pycache__").exists()
    assert _git(code, "status", "--porcelain", env=env) == ""
    assert _git(code, "rev-list", "--count", "HEAD", env=env).strip() == "3"
    assert "dirty=[]" in proc.stdout


def test_nomask_build_links_too(tmp_path: Path) -> None:
    src, perf, store, env = _fixture(tmp_path)
    proc = _run(src, perf, env, "nomask", "--build")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pool = perf / "p3nomask" / "data" / "pool" / "pool-v2.jsonl"
    assert pool.is_symlink() and os.readlink(pool) == str(store / "pool-v2.jsonl")


def test_link_repairs_an_existing_clone_without_rebuilding_it(tmp_path: Path) -> None:
    src, perf, store, env = _fixture(tmp_path)
    code = perf / "p3fused"
    _git(tmp_path, "clone", "-q", "--local", str(src), str(code), env=env)
    head = _git(code, "rev-parse", "HEAD", env=env)
    proc = _run(src, perf, env, "fused", "--link")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pool = code / "data" / "pool" / "pool-v2.jsonl"
    assert pool.is_symlink() and os.readlink(pool) == str(store / "pool-v2.jsonl")
    assert _git(code, "rev-parse", "HEAD", env=env) == head
    again = _run(src, perf, env, "fused", "--link")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "2 ignored symlinks" in again.stdout


def test_a_heldout_target_refuses_and_links_nothing(tmp_path: Path) -> None:
    src, perf, store, env = _fixture(tmp_path)
    (store / "heldout").mkdir()
    (store / "heldout" / "h.jsonl").write_text("secret\n")
    (src / "data" / "pool" / "h.jsonl").symlink_to(store / "heldout" / "h.jsonl")
    code = perf / "p3fused"
    _git(tmp_path, "clone", "-q", "--local", str(src), str(code), env=env)
    proc = _run(src, perf, env, "fused", "--link")
    assert proc.returncode == 3
    assert "heldout" in proc.stdout
    assert not (code / "data" / "pool" / "pool-v2.jsonl").exists()
    assert not (code / "data" / "pool" / "h.jsonl").exists()


def test_a_dangling_source_link_refuses(tmp_path: Path) -> None:
    src, perf, store, env = _fixture(tmp_path)
    (src / "data" / "pool" / "gone.jsonl").symlink_to(store / "gone.jsonl")
    proc = _run(src, perf, env, "fused", "--build")
    assert proc.returncode == 3
    assert "dangles" in proc.stdout


def test_an_ignored_regular_file_refuses(tmp_path: Path) -> None:
    src, perf, _store, env = _fixture(tmp_path)
    (src / "data" / "pool" / "real.jsonl").write_text("not a link\n")
    proc = _run(src, perf, env, "fused", "--build")
    assert proc.returncode == 3
    assert "not a symlink" in proc.stdout


def test_a_destination_that_differs_refuses(tmp_path: Path) -> None:
    src, perf, store, env = _fixture(tmp_path)
    code = perf / "p3fused"
    _git(tmp_path, "clone", "-q", "--local", str(src), str(code), env=env)
    (code / "data" / "pool" / "pool-v2.jsonl").symlink_to(store / "corpus" / "examples.jsonl")
    proc = _run(src, perf, env, "fused", "--link")
    assert proc.returncode == 3
    assert "is a link to" in proc.stdout
    assert not (code / "data" / "pool" / "corpus" / "examples.jsonl").exists()


def test_a_source_with_no_ignored_links_refuses(tmp_path: Path) -> None:
    src, perf, _store, env = _fixture(tmp_path)
    for link in (src / "data" / "pool" / "pool-v2.jsonl", src / "data" / "pool" / "corpus" / "examples.jsonl"):
        link.unlink()
    proc = _run(src, perf, env, "fused", "--build")
    assert proc.returncode == 3
    assert "no ignored data symlinks" in proc.stdout


def test_link_without_a_clone_refuses(tmp_path: Path) -> None:
    src, perf, _store, env = _fixture(tmp_path)
    proc = _run(src, perf, env, "fused", "--link")
    assert proc.returncode == 3
    assert "not a git checkout" in proc.stdout
