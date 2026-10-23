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
on the campaign host already drifted
(GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT-FROM-QD-DATA-HEAD-2026-10-17).
A guard that is always red ends with someone bypassing it.

**Version 2** (``shard_code_fingerprint()``): only the modules that determine shard content,
as the explicit, sorted list :data:`SHARD_CONTENT_MODULES` -- this file is its one owner.
"Determine shard content" is not a judgement call kept current by hand:
``tests/test_shard_code_fingerprint.py`` computes the import closure (AST, absolute and
relative, top level and in-function) of the shard-building entry points
``qd_train/shards.py`` and ``tools/real_ft_run.py`` restricted to ``qd_data``, and fails if a
module in it is missing from the list or the list names one outside it. The one named
exemption is this module, which describes the code and runs in no row pipeline.

**Version 3** (2026-10-23; ``FINGERPRINT_VERSION``): the same rule with the two modules the
closure test found missing from version 2's list, ``manifest.py`` and ``val_requests.py``
(ft-2eb31e7d9863d70a25807b3425fa6057). Version 2's list is frozen as
:data:`SHARD_CONTENT_MODULES_V2`. A reader branches on the rule the header records, never on
how many modules it lists: a v2 header is verified over its own recorded modules and the
verdict carries ``not_covered`` (:func:`uncovered_modules`), the modules rule 3 fingerprints
that the header could not have. Verifying a v2 header over rule 3's list would refuse every
pinned v2 shard set ("added since"); silently verifying it over its own list would read an
uncovered set as covered. Neither is done.

Containment attestations (``qd_train.exclusions.containment_corpus``) have the same two rules
under their own number, ``CONTAINMENT_FINGERPRINT_VERSION``: rule 1 is every module, rule 2 is
:data:`CONTAINMENT_CONTENT_MODULES`, the closure of ``qd_train/exclusions.py``'s ``qd_data``
imports. An attestation that records no version is rule 1 and stays checked by it.

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
    "CONTAINMENT_CONTENT_MODULES",
    "CONTAINMENT_FINGERPRINT_VERSION",
    "FINGERPRINT_SUFFIX",
    "FINGERPRINT_VERSION",
    "KNOWN_FINGERPRINT_VERSIONS",
    "SHARD_CONTENT_MODULES",
    "SHARD_CONTENT_MODULES_BY_RULE",
    "SHARD_CONTENT_MODULES_V2",
    "code_fingerprint",
    "containment_code_fingerprint",
    "describe_drift",
    "drifted_modules",
    "shard_code_fingerprint",
    "uncovered_modules",
]

#: Only ``*.py``. A ``.pyc`` is derived, and ``__pycache__`` is excluded by it.
FINGERPRINT_SUFFIX: Final[str] = ".py"


#: The fingerprint rule a new shard header records (``code_fingerprint_version``). A header
#: that records none was written under rule 1: every module.
FINGERPRINT_VERSION: Final[int] = 3

#: Every rule a reader knows. A header is read by the rule it records, never by counting its
#: modules; a number not in here is refused.
KNOWN_FINGERPRINT_VERSIONS: Final[tuple[int, ...]] = (1, 2, FINGERPRINT_VERSION)

#: Rule 2's list, frozen as the headers written under it recorded it (16 modules). A v2 header
#: is verified over these and no others, and the modules rule 3 added are its ``not_covered``.
#: Never edited: it is what the pinned v2 shard sets were written under.
SHARD_CONTENT_MODULES_V2: Final[tuple[str, ...]] = (
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

#: The ``qd_data`` modules that determine shard bytes, sorted: rule 3's list, the current one.
#: Derived, and checked, as the import closure of the shard-building entry points (see the
#: module docstring); add a module here, under a new rule number, when
#: ``test_every_module_the_shard_builders_import_is_in_the_list`` says to.
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
    "manifest.py",
    "minhash.py",
    "mixture.py",
    "render.py",
    "rows.py",
    "schema.py",
    "sources.py",
    "split.py",
    "val_requests.py",
)

#: The module list each shard-content rule verifies a header over.
SHARD_CONTENT_MODULES_BY_RULE: Final[dict[int, tuple[str, ...]]] = {
    2: SHARD_CONTENT_MODULES_V2,
    3: SHARD_CONTENT_MODULES,
}

#: The fingerprint rule a new containment attestation's corpus records
#: (``qd_data_fingerprint_version``; ``qd_train.exclusions.containment_corpus``). An attestation
#: whose corpus records none was written under rule 1: every module, by ``code_fingerprint()``.
#: Its own number: it moves with :data:`CONTAINMENT_CONTENT_MODULES`, not with the shard rule.
CONTAINMENT_FINGERPRINT_VERSION: Final[int] = 2

#: The ``qd_data`` modules that decide which rows a containment scan can exclude, sorted: the
#: import closure of ``qd_train/exclusions.py``'s ``qd_data`` imports (config, rows, split and
#: what they import), less this module. Checked by
#: ``tests/test_containment_code_fingerprint.py`` the way the shard list is.
CONTAINMENT_CONTENT_MODULES: Final[tuple[str, ...]] = (
    "config.py",
    "dedupe.py",
    "errors.py",
    "licences.py",
    "minhash.py",
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


def _listed_fingerprint(
    modules: tuple[str, ...], directory: Path | None, what: str
) -> dict[str, str]:
    """``{module: sha256}`` over a named module list. A listed module that is not on disk is
    refused, never skipped: a missing module would otherwise fingerprint as a smaller,
    passing set."""
    directory = _package_dir() if directory is None else Path(directory)
    missing = [m for m in modules if not (directory / m).is_file()]
    if missing:
        raise RuntimeError(
            f"{what} module(s) {', '.join(missing)} not found in {directory}; "
            "refusing to fingerprint a partial set"
        )
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in modules}


def shard_code_fingerprint(
    directory: Path | None = None, rule: int = FINGERPRINT_VERSION
) -> dict[str, str]:
    """``{module: sha256}`` over the modules rule ``rule`` lists (rule 2 or 3; 3 by default).

    A rule with no list here (1 hashes every module, by :func:`code_fingerprint`; anything
    else is unknown) is refused, never mapped to the current list.
    """
    modules = SHARD_CONTENT_MODULES_BY_RULE.get(rule)
    if modules is None:
        raise ValueError(
            f"fingerprint rule {rule!r} has no shard-content list "
            f"(rules with one: {sorted(SHARD_CONTENT_MODULES_BY_RULE)})"
        )
    return _listed_fingerprint(modules, directory, "shard-content")


def uncovered_modules(
    recorded: dict[str, str], directory: Path | None = None
) -> tuple[str, ...]:
    """The current rule's modules a recorded fingerprint cannot vouch for, sorted.

    A set difference over what the header RECORDED, not a function of its version number: a
    module on the current rule-3 list that the header did not record, plus a listed module
    the header recorded whose digest is not the code loaded now (a drifted set read with
    ``allow_stale_code`` is not covered either). A truncated v2 header, a forged version and a
    drifted v3 header all name their modules; only a header that records every listed module
    at today's bytes names none.
    """
    current = shard_code_fingerprint(directory, FINGERPRINT_VERSION)
    missing = set(current) - set(recorded)
    changed = {m for m, digest in current.items() if m in recorded and recorded[m] != digest}
    return tuple(sorted(missing | changed))


def containment_code_fingerprint(directory: Path | None = None) -> dict[str, str]:
    """Containment rule 2: ``{module: sha256}`` over :data:`CONTAINMENT_CONTENT_MODULES`."""
    return _listed_fingerprint(CONTAINMENT_CONTENT_MODULES, directory, "containment-content")


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
