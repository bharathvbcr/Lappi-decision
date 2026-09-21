"""``tools/repo_git.py``: the one owner of "read this repository at a revision".

It exists because ``devmap_clones`` reported ``_git`` and ``_git_bytes`` as an Exact group
across ``tools/real_tokenizer_pipeline.py`` and ``tools/rung0_real_run.py`` -- the second
copy written in the same session that removed five other duplications. These tests pin the
two properties the corpora built on it depend on, and the single-owner invariant itself.

Torch-free, so ``make gates`` runs them: a corpus that reorders between runs produces two
``data_snapshot_hash`` values for one revision, and the check for that must not need a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from repo_git import git_bytes, git_text, resolve_rev, tracked_paths  # noqa: E402


def test_tracked_paths_is_sorted_and_filtered() -> None:
    """Sorted because both callers turn this into a corpus whose row order must be a
    function of the revision and nothing else."""
    names = tracked_paths(REPO, rev="HEAD", suffixes=frozenset({".py"}))
    assert names, "this repository tracks Python files at HEAD"
    assert names == sorted(names)
    assert all(n.endswith(".py") for n in names)


def test_an_empty_suffix_set_is_refused_rather_than_selecting_nothing() -> None:
    """An empty corpus from an empty filter is indistinguishable from an empty repository,
    and only one of those is a bug in the caller."""
    with pytest.raises(ValueError, match="no suffixes"):
        tracked_paths(REPO, rev="HEAD", suffixes=frozenset())


def test_two_suffixes_return_the_union_not_the_first() -> None:
    py = set(tracked_paths(REPO, rev="HEAD", suffixes=frozenset({".py"})))
    rs = set(tracked_paths(REPO, rev="HEAD", suffixes=frozenset({".rs"})))
    both = set(tracked_paths(REPO, rev="HEAD", suffixes=frozenset({".py", ".rs"})))
    assert py and rs
    assert both == py | rs


def test_a_missing_revision_raises_rather_than_returning_an_empty_corpus() -> None:
    """Fail closed. A bad rev that returned [] would build a zero-row corpus and the only
    symptom would be a run that trained on nothing while reporting a protocol."""
    with pytest.raises(subprocess.CalledProcessError):
        tracked_paths(REPO, rev="no-such-revision-0000", suffixes=frozenset({".py"}))


def test_git_bytes_does_not_decode() -> None:
    """The boundary where the caller decides what a non-UTF-8 file means for its corpus.
    A helper that decoded with errors= would fabricate or silently shorten file content."""
    raw = git_bytes(REPO, "show", "HEAD:CLAUDE.md")
    assert isinstance(raw, bytes)
    assert raw.decode("utf-8").startswith("# qwen-decision")


def test_git_text_returns_text() -> None:
    head = git_text(REPO, "rev-parse", "HEAD").strip()
    assert isinstance(head, str)
    assert len(head) == 40


def test_only_one_module_implements_the_git_subprocess_call() -> None:
    """The invariant the module exists for, checked against the source rather than trusted.

    Both tools may keep a thin ``_git`` that binds ``REPO`` -- CLAUDE.md allows an adapter
    that delegates -- but neither may call ``subprocess.run(["git", ...])`` itself again.
    Without this, the next tool that needs a git helper writes a third copy and nothing
    notices until a clone report is read.
    """
    offenders: list[str] = []
    for path in sorted((REPO / "tools").glob("*.py")):
        if path.name == "repo_git.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            squashed = line.replace(" ", "").replace("'", '"')
            if 'subprocess.run(["git"' in squashed:
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "these tools call git directly instead of going through tools/repo_git.py: "
        f"{offenders}. One owner per behaviour; a thin adapter that delegates is fine, a "
        "second implementation is not."
    )


# -- resolving a revision, because a name is not a revision ----------------------------
#
# Added with `ShardHeader.corpus_rev`. The header pins which revision a corpus was read at,
# and `--rev` defaults to "HEAD" in `real_tokenizer_pipeline.py` -- so without this, the
# pin would record a pointer. Two sets built a day apart would both say "HEAD" and compare
# equal, which is worse than an absent pin: absent reads NotRun, "HEAD" reads as verified.


def test_resolve_rev_returns_a_full_sha_for_a_symbolic_name() -> None:
    head = resolve_rev(REPO, "HEAD")
    assert len(head) == 40
    assert head == git_text(REPO, "rev-parse", "HEAD").strip()


def test_resolve_rev_is_idempotent_on_a_sha_it_already_returned() -> None:
    """The property the shard check relies on. `real_ft_run.py` resolves whatever ``--rev``
    it was given and compares the result to a header written from a resolved rev; if
    resolving an already-resolved sha moved it, a set would refuse against itself."""
    head = resolve_rev(REPO, "HEAD")
    assert resolve_rev(REPO, head) == head


def test_an_abbreviated_sha_resolves_to_the_full_one() -> None:
    head = resolve_rev(REPO, "HEAD")
    assert resolve_rev(REPO, head[:7]) == head


def test_a_missing_revision_raises_rather_than_echoing_the_name_back() -> None:
    """Plain ``git rev-parse no-such-thing`` prints the string and errors; without
    ``--verify`` a caller that ignored the exit status would put an unresolvable name in a
    shard header, where it would be compared for equality against another one."""
    with pytest.raises(subprocess.CalledProcessError):
        resolve_rev(REPO, "no-such-revision-0000")


def test_the_pipeline_pins_the_resolved_revision_and_not_the_argument() -> None:
    """The defect this function exists to make impossible, checked against the source.

    ``real_tokenizer_pipeline.run`` takes ``rev`` and resolves it into ``resolved``; every
    corpus read below that line uses ``resolved``. A ``write_shards(corpus_rev=rev)`` there
    would pin the string the caller typed -- "HEAD" by default -- beside rows that were read
    from a commit. One name, two quantities, and the header would claim the wrong one.
    """
    source = (REPO / "tools" / "real_tokenizer_pipeline.py").read_text(encoding="utf-8")
    assert "corpus_rev=resolved," in source
    assert "corpus_rev=rev," not in source, (
        "real_tokenizer_pipeline pins the unresolved --rev, which defaults to HEAD: two "
        "shard sets built from different corpora would both record 'HEAD' and compare equal"
    )


def test_real_ft_run_checks_the_shard_set_against_the_rev_it_reconstructs_labels_at() -> None:
    """Both halves come from one resolved string.

    The tool does not read labels out of the shard set -- it rebuilds them from the repo at
    ``--rev`` and pairs them with the set's sequences. If the reader were given the raw
    argument while the reconstruction used a resolved one (or the reverse), the check would
    compare a name against a sha and refuse a set that was in fact correct.
    """
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert "rev = resolve_rev(REPO, args.rev)" in source
    assert "expect_rev=rev)" in source or "expect_rev=rev," in source
    assert "rev=args.rev" not in source, (
        "a label reconstruction at the unresolved --rev, checked against a header written "
        "from a resolved one, would refuse a set that matches"
    )
