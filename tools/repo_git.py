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

__all__ = ["git_bytes", "git_text", "require_full_sha", "resolve_rev", "tracked_paths"]

_HEX = frozenset("0123456789abcdef")


def require_full_sha(rev: str, *, flag: str = "--rev") -> str:
    """``rev`` itself, if it is a full 40-character lowercase commit sha; else a refusal.

    :func:`resolve_rev` makes a *header* safe: it stores the commit a name pointed at. It
    does not make a *command* reproducible. ``--rev HEAD`` in a campaign config, a ledger
    recipe or a handoff names a different corpus after every commit to this worktree, and
    the 2026-09-29 smoke measured what that costs: the rebuild ``3e577d0b`` was built at
    ``HEAD``, picked up that day's commits as 23-37k-token rows, and had to be superseded.
    So a run that writes a ledger row takes the sha, not a name for it -- an abbreviation
    is refused too, because an abbreviation can become ambiguous as history grows.

    Pure: no git call. The caller still resolves the sha with :func:`resolve_rev`, which is
    what proves the commit exists.
    """
    if len(rev) != 40 or not set(rev) <= _HEX:
        raise ValueError(
            f"{flag} {rev!r} is not a full 40-character commit sha. A run that writes a "
            "ledger row must name its corpus revision exactly: a branch, tag, HEAD or an "
            "abbreviation is a different commit (or an ambiguous one) later, so the same "
            "command would rebuild a different corpus. Pass the full sha, e.g. "
            "`git rev-parse HEAD`."
        )
    return rev


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


def resolve_rev(repo: Path, rev: str) -> str:
    """The concrete commit ``rev`` names, as a full 40-character sha.

    Callers record *which revision* a corpus was read at, and a symbolic name is not that.
    ``HEAD`` is a different commit tomorrow, and several lanes commit to this worktree while
    a run is in progress -- which is the same reason this module refuses to default ``rev``
    at all.

    The asymmetry that makes this a function rather than a caller's ``strip()``: storing an
    unresolved name in a shard header would be **worse than storing nothing**. An absent
    ``corpus_rev`` reads ``NotRun`` and says so; "HEAD" compares equal to "HEAD" and reads
    as *verified*, so two sets built a day apart from different corpora would both pass the
    check that exists to tell them apart.

    Peeled with ``^{commit}`` so an annotated tag resolves to the commit it points at rather
    than to the tag object, which is not something ``git show`` reads a file from. A bad
    revision raises ``CalledProcessError`` rather than returning the string back, which is
    what plain ``rev-parse`` does and what would put an unresolvable name in a header.
    """
    resolved = git_text(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
    if len(resolved) != 40 or not all(c in "0123456789abcdef" for c in resolved):
        raise ValueError(
            f"git rev-parse resolved {rev!r} to {resolved!r}, which is not a commit sha. "
            "Refusing rather than recording it: a header pinned to a string that names no "
            "commit cannot be checked against anything, and reads as a pin."
        )
    return resolved


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
