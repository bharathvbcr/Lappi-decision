"""Build a qd-mutate pool from local source trees.

This is what takes rung 0's data off the blocked path. ``crates/qd-mutate/src/pool.rs``:

    `hunks` is optional and its absence is **carried, not defaulted**. A record with no
    hunks is a whole file with no diff attached; every site in it is fair game, and every
    example produced from it is stamped `hunk_constrained: false`.

So the mutation corpus does **not** need commitpackft, and therefore does not need Hugging
Face terms. It needs source files. The honesty cost is real and already handled downstream:
examples built this way are an *unconstrained* sample, counted separately in qd-mutate's
manifest, and reporting them as hunk-constrained would be the "capped sample as complete
coverage" failure this repo names in its rules.

## Licence is required, not inferred

:func:`build_pool` takes a licence id and puts it through
:func:`qd_data.licences.admit_licence` before reading a single file. There is no default and
no "unknown" tier: a pool whose licence nobody stated is a pool whose model card cannot be
written. The refusal names the licence, which is what makes it actionable.

## What is counted

Everything skipped is counted by reason, and :class:`PoolBuildReport` carries ``n_selected``
alongside ``n_seen``. A build that hits ``max_files`` reports :class:`NotRun` rather than a
clean pass, because a capped walk is not a complete one and the two must not read alike.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from qd_train.tristate import NotRun, Ran, TriState

from .licences import DEFAULT_LICENCE_CONFIG, LicenceConfig, admit_licence

__all__ = [
    "MAX_SOURCE_BYTES",
    "POOL_EXTENSIONS",
    "SKIP_DIRS",
    "PoolBuildReport",
    "PoolRecord",
    "build_pool",
    "language_from_path",
    "write_pool",
]

#: Mirrors ``language_from_path`` in ``crates/qd-lang/src/lib.rs``. A file this map does
#: not name is skipped rather than guessed at: qd-mutate would refuse it anyway, and a
#: record it cannot parse inflates the "files seen, zero examples" column.
POOL_EXTENSIONS: Final[dict[str, str]] = {
    ".rs": "Rust",
    ".go": "Go",
    ".py": "Python",
    ".pyi": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".mts": "TypeScript",
    ".cts": "TypeScript",
    ".swift": "Swift",
}

#: ``pool.rs`` excludes this explicitly: a declaration file has no function bodies, so it
#: parses, yields nothing, and pads the seen-but-useless count.
DECLARATION_SUFFIX: Final[str] = ".d.ts"

#: ``parse::MAX_SOURCE_BYTES``. A file over this is refused by the parser, so selecting it
#: here would only move the failure later.
MAX_SOURCE_BYTES: Final[int] = 10 * 1024 * 1024

#: Directories that hold no first-party source. Vendored and generated trees are the classic
#: way a corpus fills with code nobody wrote and no licence covers.
SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "target",
        "build",
        "dist",
        "out",
        "vendor",
        "third_party",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".devmap",
        ".devcouncil",
        ".next",
        ".svelte-kit",
        "Pods",
        "DerivedData",
    }
)


def language_from_path(path: str) -> str | None:
    """The language a path names, or ``None``. Mirrors ``qd_lang::language_from_path``."""
    if path.endswith(DECLARATION_SUFFIX):
        return None
    suffix = Path(path).suffix
    return POOL_EXTENSIONS.get(suffix)


@dataclass(frozen=True, slots=True)
class PoolRecord:
    """One pool row, in the shape ``pool.rs`` deserialises.

    ``hunks`` is deliberately absent rather than empty. An empty list would mean "a diff
    touched nothing"; absence means "no diff attached", and ``pool.rs`` carries that
    distinction all the way to ``hunk_constrained`` on every example.
    """

    id: str
    repo: str
    path: str
    source: str

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "repo": self.repo, "path": self.path, "source": self.source}


@dataclass(frozen=True, slots=True)
class PoolBuildReport:
    records: tuple[PoolRecord, ...]
    n_selected: int
    n_seen: int
    skipped: Counter[str] = field(default_factory=Counter)
    status: TriState = field(default_factory=lambda: NotRun(reason="build_pool did not run"))
    licence_id: str = ""
    obligations: tuple[str, ...] = ()

    def coverage(self) -> str:
        """``selected/seen``. The pair, never the first number alone."""
        return f"{self.n_selected}/{self.n_seen}"

    def summary(self) -> str:
        lines = [
            f"pool: {self.coverage()} files selected, licence {self.licence_id or '<unset>'}",
            f"  status: {self.status}",
        ]
        lines += [f"  skipped {reason}: {n}" for reason, n in sorted(self.skipped.items())]
        return "\n".join(lines)


def _walk(root: Path) -> Iterator[Path]:
    """Yield files, pruning directories that hold no first-party source."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if entry.is_symlink():
                # Not followed: a symlink out of the tree would attach this repo's licence
                # to code from somewhere else.
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file():
                yield entry


def build_pool(
    root: Path,
    *,
    repo: str,
    licence: str,
    config: LicenceConfig = DEFAULT_LICENCE_CONFIG,
    max_files: int | None = None,
) -> PoolBuildReport:
    """Walk ``root`` and build pool records for every admissible source file.

    The licence is checked **before** the walk, so a refused pool costs nothing and the
    refusal is not buried under a partial result.
    """
    policy = admit_licence(licence, config=config, source=f"pool:{repo}")

    if not root.is_dir():
        raise NotADirectoryError(f"pool root {root} is not a directory")

    records: list[PoolRecord] = []
    skipped: Counter[str] = Counter()
    seen = 0
    capped = False

    for file in _walk(root):
        seen += 1
        rel = file.relative_to(root).as_posix()

        if language_from_path(rel) is None:
            skipped["not a supported language"] += 1
            continue
        try:
            size = file.stat().st_size
        except OSError:
            skipped["unreadable"] += 1
            continue
        if size > MAX_SOURCE_BYTES:
            skipped["over MAX_SOURCE_BYTES"] += 1
            continue
        try:
            source = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # qd-mutate takes a Rust `String`, so a non-UTF-8 file cannot be a record.
            skipped["not valid utf-8"] += 1
            continue
        if not source.strip():
            skipped["empty"] += 1
            continue

        records.append(PoolRecord(id=f"{repo}:{rel}", repo=repo, path=rel, source=source))

        if max_files is not None and len(records) >= max_files:
            capped = True
            break

    if capped:
        status: TriState = NotRun(
            reason=(
                f"the walk stopped at max_files={max_files} with {seen} files examined, so "
                "this is a capped sample and its coverage is not the tree's coverage"
            )
        )
    elif not records:
        status = NotRun(
            reason=(
                f"no source files survived selection out of {seen} examined; an empty pool "
                "produces an empty corpus, which is not a clean build"
            )
        )
    else:
        status = Ran(
            passed=True,
            value=len(records),
            n=len(records),
            n_total=seen,
            detail=f"licence {policy.licence_id}; hunks absent, so every example is unconstrained",
        )

    return PoolBuildReport(
        records=tuple(records),
        n_selected=len(records),
        n_seen=seen,
        skipped=skipped,
        status=status,
        licence_id=policy.licence_id,
        obligations=tuple(policy.obligations),
    )


def write_pool(report: PoolBuildReport, path: Path) -> int:
    """Write the records as JSONL. Returns the number written.

    Refuses to write a pool whose status is not ``Ran``: a capped or empty build that lands
    on disk as an ordinary file is indistinguishable from a complete one the moment anyone
    reads the file instead of the report.
    """
    if not isinstance(report.status, Ran):
        raise ValueError(
            f"refusing to write a pool whose build did not complete: {report.status}"
        )
    with path.open("w", encoding="utf-8") as fh:
        for record in report.records:
            fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    return len(report.records)
