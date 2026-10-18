"""What code turned a corpus into rows, so a shard set can say it and be checked on it.

``ShardHeader`` pins ``data_snapshot_hash``, ``tokenizer_hash`` and ``remap_hash``: the
corpus, the tokenizer, the vocabulary renumbering. Between the corpus and the rows sits
this package, and nothing pinned it.

That gap is ``GAP-SHARD-SET-GOES-STALE-AGAINST-THE-CORPUS-CODE-THAT-REPRODUCES-ITS-LABELS``
and it is not hypothetical. Measured 2026-09-21: the shard set at ``/home/ubuntu/shardset``
(341 sequences, 86 span rows, ``data_snapshot_hash`` 85acf409…) is what ledger rows
4541d72e / 5ebebda7 / 6374e188 trained on. Against the working tree of the same day the
same reconstruction yields **321** rows. Every hash in that header still matched: the
corpus had not moved, the tokenizer had not moved, the remap had not moved. The code had.

The incident was caught because the drift happened to change the row *count*, and
``tools/real_ft_run.py`` compares the reconstructed order against ``supervision.npz``
entry for entry -- "321 labels against 341 sequences … Every label would be on the wrong
sequence". A change that alters which label attaches to which row **without** changing how
many rows there are produces no count mismatch and no refusal anywhere. That is the case
this module exists for.

## What is hashed

Two rules, named by ``FINGERPRINT_VERSION``.

**Version 1** (every header written before 2026-10-18; ``code_fingerprint(directory)``):
every ``*.py`` in ``qd_data``, by source bytes. It over-invalidated -- any edit, to a comment
or to a module no shard builder imports, made every pinned set unreadable, and all 89 sets
on the campaign host already drifted (GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT-FROM-QD-DATA-HEAD).
A guard that is always red ends with someone bypassing it.

**Version 2** (``shard_code_fingerprint()``): only the modules that determine shard content,
as the explicit, sorted list :data:`SHARD_CONTENT_MODULES` -- this file is its one owner.
"Determine shard content" is not a judgement call kept current by hand:
``tests/test_shard_code_fingerprint.py`` computes the import closure (AST, absolute and
relative, top level and in-function) of the shard-building entry points
``qd_train/shards.py`` and ``tools/real_ft_run.py`` restricted to ``qd_data``, and fails if a
module in it is missing from the list or the list names one outside it. The one named
exemption is this module, which describes the code and runs in no row pipeline.

*Source bytes, so comments and docstrings of listed modules count.* An AST or token digest
would make every fingerprint a function of the running Python's ``ast``/``tokenize``
output, so a Python upgrade would invalidate every shard set on disk. Comment stripping was
considered and declined for that reason: a false alarm costs a rebuild, a missed change costs
a training run on mislabelled data.

*A per-module map, not one digest*, so a refusal names ``render.py`` and ``mixture.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

__all__ = [
    "FINGERPRINT_SUFFIX",
    "FINGERPRINT_VERSION",
    "SHARD_CONTENT_MODULES",
    "code_fingerprint",
    "describe_drift",
    "drifted_modules",
    "shard_code_fingerprint",
]

#: Only ``*.py``. A ``.pyc`` is derived, and ``__pycache__`` is excluded by it.
FINGERPRINT_SUFFIX: Final[str] = ".py"


#: The fingerprint rule a new shard header records (``code_fingerprint_version``). A header
#: that records none was written under rule 1: every module.
FINGERPRINT_VERSION: Final[int] = 2

#: The ``qd_data`` modules that determine shard bytes, sorted. Derived, and checked, as the
#: import closure of the shard-building entry points (see the module docstring); add a module
#: here when ``test_every_module_the_shard_builders_import_is_in_the_list`` says to.
SHARD_CONTENT_MODULES: Final[tuple[str, ...]] = (
    "caller_records.py",
    "config.py",
    "decisions.py",
    "dedupe.py",
    "defect_class.py",
    "errors.py",
    "general.py",
    "licences.py",
    "loaders.py",
    "minhash.py",
    "mixture.py",
    "render.py",
    "rows.py",
    "schema.py",
    "sources.py",
    "split.py",
)


def _package_dir() -> Path:
    """Where this package's sources live.

    ``__file__`` rather than a walk from the repository root: the question is what code is
    *loaded*, not what code is checked out. A run importing ``qd_data`` from a different
    interpreter's site-packages must fingerprint that copy, because that is the copy that
    produced the rows.
    """
    here = Path(__file__).resolve()
    directory = here.parent
    if not directory.is_dir():  # pragma: no cover - only reachable from a zipimport
        raise RuntimeError(
            f"qd_data does not live in a real directory ({directory}); its source cannot "
            "be fingerprinted, and a shard set written now could not be checked against "
            "the code that wrote it."
        )
    return directory


def code_fingerprint(directory: Path | None = None) -> dict[str, str]:
    """``{module name: sha256 of its source bytes}`` for every module in a package.

    Sorted, so the mapping is a value and not an iteration order. Module names carry no
    directory part -- the packages this is used on have no subpackages, and a name like
    ``render.py`` is what a refusal should print.

    ``directory`` defaults to ``qd_data``, which is what every shard header records and
    what this function was written for; passing none is byte-for-byte what it always did.
    It is a parameter because the same question -- *which source produced this* -- is now
    asked about ``qd_train`` by ``tools/real_ft_run.py``. A ledger row's ``code_commit``
    answers it with ``"<sha>-dirty"``, which on the GH200 stood for 6894 insertions across
    72 paths: one bit for an unbounded amount of divergence.
    """
    directory = _package_dir() if directory is None else Path(directory)
    fingerprint: dict[str, str] = {}
    for path in sorted(directory.glob(f"*{FINGERPRINT_SUFFIX}")):
        if not path.is_file():
            continue
        fingerprint[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not fingerprint:  # pragma: no cover - the package always contains this file
        raise RuntimeError(
            f"no {FINGERPRINT_SUFFIX} sources found in {directory}; refusing to return an "
            "empty fingerprint, which would be indistinguishable from a shard set created "
            "before fingerprints existed -- and, for a caller naming its own directory, "
            "from a path that does not hold the sources it thinks it does."
        )
    return fingerprint


def shard_code_fingerprint(directory: Path | None = None) -> dict[str, str]:
    """Rule 2: ``{module: sha256}`` over :data:`SHARD_CONTENT_MODULES` only.

    A listed module that is not on disk is refused, never skipped: a missing module would
    otherwise fingerprint as a smaller, passing set.
    """
    directory = _package_dir() if directory is None else Path(directory)
    missing = [m for m in SHARD_CONTENT_MODULES if not (directory / m).is_file()]
    if missing:
        raise RuntimeError(
            f"shard-content module(s) {', '.join(missing)} not found in {directory}; "
            "refusing to fingerprint a partial set"
        )
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in SHARD_CONTENT_MODULES
    }


def drifted_modules(
    recorded: dict[str, str], current: dict[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """``(changed, added, removed)`` between a recorded fingerprint and a current one.

    All three are drift. ``added`` and ``removed`` are not cosmetic: a module that did not
    exist when a shard set was written is code that could not have shaped its rows, and one
    that has since gone is code whose absence certainly reshapes them.
    """
    changed = tuple(
        sorted(
            name
            for name, digest in recorded.items()
            if name in current and current[name] != digest
        )
    )
    added = tuple(sorted(set(current) - set(recorded)))
    removed = tuple(sorted(set(recorded) - set(current)))
    return changed, added, removed


def describe_drift(recorded: dict[str, str], current: dict[str, str]) -> str:
    """The sentence a refusal prints. Empty string when nothing drifted."""
    changed, added, removed = drifted_modules(recorded, current)
    if not (changed or added or removed):
        return ""
    parts = []
    if changed:
        parts.append(f"changed: {', '.join(changed)}")
    if added:
        parts.append(f"added since: {', '.join(added)}")
    if removed:
        parts.append(f"removed since: {', '.join(removed)}")
    return "; ".join(parts)
