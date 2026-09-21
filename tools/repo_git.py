"""Reading this repository's own tracked sources at a pinned revision.

Two tools build a corpus out of this repository: ``real_tokenizer_pipeline.py`` for rung 3's
shard set and ``rung0_real_run.py`` for rung 0's mutation pool. Both need the same two
things -- run git and get text, run git and get bytes -- and until this module existed they
each had their own copy. ``devmap_clones`` reported them as an Exact pair, which is how this
module came to exist: the second copy was written in the same session that removed five
others.

## Why bytes and text are separate functions rather than one with a flag

``git show`` returns a blob. Whether that blob is text is a property of the blob, not of the
caller's intent, and a repository contains files that are not valid UTF-8. A single helper
returning ``str`` would have to decide what to do with those -- and every available answer
is wrong: ``errors="replace"`` fabricates content, ``errors="ignore"`` silently shortens a
file, and both produce a corpus row that claims to be a file it is not. So the caller that
wants text asks for text on a command whose output git guarantees is text (``ls-tree``,
``rev-list``, ``log``), and the caller that wants a file's contents asks for bytes and
decides for itself.

## Why ``rev`` is never defaulted to HEAD

Several lanes commit to this worktree while a run is in progress. Two runs against "HEAD"
hours apart read different corpora and report the same protocol, so the corpora are not
comparable and nothing in the ledger says so. Both callers take ``rev`` as a required
argument for that reason, and this module does not offer a default that would undo it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["git_bytes", "git_text", "tracked_paths"]


def git_text(repo: Path, *args: str) -> str:
    """Run git in ``repo`` and return stdout as text.

    For commands whose output git defines as text -- ``ls-tree``, ``rev-list``, ``log``,
    ``diff --name-only``. A file's contents go through :func:`git_bytes` instead.
    """
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=True, text=True
    ).stdout


def git_bytes(repo: Path, *args: str) -> bytes:
    """Run git in ``repo`` and return stdout as raw bytes.

    Undecoded on purpose: a tracked file may not be valid UTF-8, and this is the boundary
    where the caller decides what that means for its corpus rather than having a lossy
    decode chosen for it.
    """
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout


def tracked_paths(repo: Path, *, rev: str, suffixes: frozenset[str]) -> list[str]:
    """Every tracked path at ``rev`` whose suffix is in ``suffixes``, sorted.

    Sorted because both callers turn this into a corpus whose row order must be a function
    of the revision and nothing else; git's own ordering is stable but is not part of its
    contract, and a corpus that reorders between runs produces two different
    ``data_snapshot_hash`` values for one revision.
    """
    if not suffixes:
        raise ValueError(
            "tracked_paths was given no suffixes, which selects nothing. An empty corpus "
            "from an empty filter is indistinguishable from an empty repository."
        )
    return sorted(
        name
        for name in git_text(repo, "ls-tree", "-r", "--name-only", rev).splitlines()
        if Path(name).suffix in suffixes
    )
