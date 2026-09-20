"""The golden fixture corpus: ``fixtures/wire/``, read from the Python side.

The seam between the two lanes is a corpus of real envelopes, not a paragraph
either lane can read its own way. The XLANG-RS lane owns the Rust generator
(``crates/qd-runtime/src/fixtures.rs``, run by
``QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures``); this module
is the Python half that parses the result back and is the thing that breaks when
the Rust changes.

Reading is **manifest-driven**, not glob-driven. ``index.json`` names every file the
generator wrote, and its own docstring says why: *"so a consumer can enumerate the
corpus without globbing a directory and can assert it read all of it. A parser that
skipped a file silently is a parser that agrees with nothing."* So this loader reads
the manifest first, then asserts the directory and the manifest name the same set —
:attr:`CorpusLoad.missing` and :attr:`CorpusLoad.strays` carry the two directions of
disagreement, and the tests gate on both being empty. Globbing would make a corpus
that lost half its files look like a smaller corpus that passed.

Three rules make it a check rather than a formality:

1. **Absence is ``NotRun``, never a pass.** A missing directory, an empty one, or one
   with no manifest returns a :class:`~qd_train.tristate.NotRun` carrying the reason.
   ``NotRun`` has no ``passed`` attribute, so a report that renders it as green has to
   fabricate the field.
2. **A fixture that does not parse raises.** :class:`~qd_wire.errors.WireParseError`
   propagates naming the file and the field. Unknown field and missing field are both
   failures; there is no lenient read.
3. **Coverage is carried as a pair.** ``Ran(n=…, n_total=…)`` where ``n_total`` comes
   from the manifest's own ``count``, so a partial read cannot be reported as the whole.

The tri-state comes from :mod:`qd_train.tristate` rather than a second copy: that
module is this repo's canonical owner of "a check that could not run must never read
as a check that passed".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qd_train.tristate import NotRun, Ran, TriState
from qd_wire.errors import WireParseError, check_keys
from qd_wire.response import Response, parse_response

__all__ = [
    "CORPUS_DIR",
    "MANIFEST_ENTRY_OPTIONAL",
    "MANIFEST_ENTRY_REQUIRED",
    "MANIFEST_KEYS",
    "MANIFEST_NAME",
    "CorpusLoad",
    "FixtureEnvelope",
    "Manifest",
    "ManifestEntry",
    "fixture_dir",
    "load_corpus",
]

#: ``crates/qd-runtime/src/fixtures.rs::CORPUS_DIR``. One path, named on both sides.
CORPUS_DIR = ("fixtures", "wire")
#: ``crates/qd-runtime/src/fixtures.rs::MANIFEST``.
MANIFEST_NAME = "index.json"

#: ``fixtures.rs::Manifest``, which is ``#[serde(deny_unknown_fields)]``. Checked exactly.
MANIFEST_KEYS = frozenset({"schema_version", "count", "by_status", "entries"})
#: ``fixtures.rs::ManifestEntry``. ``kind`` and ``request`` are ``skip_serializing_if`` —
#: absent rather than null when the reply has none — so they are optional here and
#: everything else is not.
MANIFEST_ENTRY_REQUIRED = frozenset({"file", "status", "note"})
MANIFEST_ENTRY_OPTIONAL = frozenset({"kind", "request"})


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row of ``index.json``: which file, what it holds, and what it is for."""

    file: str
    status: str
    note: str
    kind: str | None
    request: Any | None


@dataclass(frozen=True, slots=True)
class Manifest:
    """``index.json``, parsed as strictly as an envelope."""

    schema_version: int
    count: int
    by_status: dict[str, int]
    entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class FixtureEnvelope:
    """One parsed envelope, with the manifest row that promised it."""

    source: Path
    entry: ManifestEntry
    raw: dict[str, object]
    response: Response

    @property
    def label(self) -> str:
        return self.source.name


@dataclass(frozen=True, slots=True)
class CorpusLoad:
    """The result of reading ``fixtures/wire/``: a tri-state and what was read."""

    directory: Path
    state: TriState
    manifest: Manifest | None
    envelopes: tuple[FixtureEnvelope, ...]
    #: Named by the manifest, absent from the directory.
    missing: tuple[str, ...]
    #: Present in the directory, named by nothing.
    strays: tuple[str, ...]

    @property
    def ran(self) -> bool:
        """True only when the corpus was present and parsed. Never inferred from a count."""
        return isinstance(self.state, Ran)

    def by_status(self, status: str) -> tuple[FixtureEnvelope, ...]:
        return tuple(e for e in self.envelopes if e.entry.status == status)


def fixture_dir(start: Path | None = None) -> Path:
    """``<repo>/fixtures/wire``, located by walking up from this file.

    Returned whether or not it exists: its absence is a fact :func:`load_corpus`
    reports as a reason, not an exception to swallow here.
    """
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "fixtures").is_dir() and (parent / "crates").is_dir():
            return parent.joinpath(*CORPUS_DIR)
    raise FileNotFoundError(
        f"no repository root with a fixtures/ and a crates/ directory above {here}"
    )


def _not_run(directory: Path, reason: str) -> CorpusLoad:
    return CorpusLoad(
        directory=directory,
        state=NotRun(reason=reason),
        manifest=None,
        envelopes=(),
        missing=(),
        strays=(),
    )


def load_corpus(directory: Path | None = None) -> CorpusLoad:
    """Read and parse the corpus. Absent or unenumerable -> ``NotRun``; bad fixture -> raise."""
    directory = directory or fixture_dir()

    if not directory.exists():
        return _not_run(
            directory,
            f"the golden wire corpus at {directory} does not exist. The XLANG-RS lane "
            "owns the Rust generator that writes it "
            "(`QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test wire_fixtures`); "
            "until it runs, the Rust -> Python direction of docs/schema-api.md is "
            "unasserted against real runtime output. This is not a pass.",
        )
    if not directory.is_dir():
        raise WireParseError(str(directory), "the wire corpus path exists but is not a directory")

    on_disk = sorted(
        p.name for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    manifest_path = directory / MANIFEST_NAME

    if not on_disk:
        return _not_run(directory, f"{directory} exists but is empty. Nothing was parsed.")
    if not manifest_path.is_file():
        return _not_run(
            directory,
            f"{directory} holds {len(on_disk)} file(s) but no {MANIFEST_NAME} manifest, "
            f"so a complete read cannot be proved: {on_disk}. Reading what happens to be "
            "there and calling it the corpus is how a corpus that lost half its files "
            "looks like a smaller one that passed.",
        )

    manifest = _parse_manifest(_load_json(manifest_path), path=MANIFEST_NAME)

    named = [e.file for e in manifest.entries]
    missing = tuple(sorted(n for n in named if n not in on_disk))
    strays = tuple(sorted(n for n in on_disk if n != MANIFEST_NAME and n not in named))

    envelopes: list[FixtureEnvelope] = []
    for entry in manifest.entries:
        path = directory / entry.file
        if not path.is_file():
            continue  # reported through `missing`; the tests gate on it
        raw = _load_json(path)
        if not isinstance(raw, dict):
            raise WireParseError(
                entry.file, f"a wire fixture is one response object, got {type(raw).__name__}"
            )
        envelopes.append(
            FixtureEnvelope(
                source=path,
                entry=entry,
                raw=raw,
                response=parse_response(raw, path=entry.file),
            )
        )

    if not envelopes:
        return _not_run(
            directory,
            f"{MANIFEST_NAME} names {len(named)} file(s) and none of them is on disk "
            f"({missing}). Nothing was parsed, so nothing passed.",
        )

    return CorpusLoad(
        directory=directory,
        state=Ran(
            passed=not missing and not strays,
            n=len(envelopes),
            n_total=manifest.count,
            detail=(
                f"parsed {len(envelopes)} of the {manifest.count} envelope(s) "
                f"{MANIFEST_NAME} names"
                + (f"; missing {list(missing)}" if missing else "")
                + (f"; strays {list(strays)}" if strays else "")
            ),
        ),
        manifest=manifest,
        envelopes=tuple(envelopes),
        missing=missing,
        strays=strays,
    )


def _parse_manifest(raw: object, *, path: str) -> Manifest:
    if not isinstance(raw, dict):
        raise WireParseError(path, f"expected a manifest object, got {type(raw).__name__}")
    check_keys(raw, required=MANIFEST_KEYS, path=path)

    by_status = raw["by_status"]
    if not isinstance(by_status, dict) or not all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        for k, v in by_status.items()
    ):
        raise WireParseError(f"{path}.by_status", "expected an object of status -> count")

    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list):
        raise WireParseError(f"{path}.entries", "expected an array")

    entries: list[ManifestEntry] = []
    for i, item in enumerate(raw_entries):
        where = f"{path}.entries[{i}]"
        if not isinstance(item, dict):
            raise WireParseError(where, f"expected an object, got {type(item).__name__}")
        unknown = sorted(set(item) - MANIFEST_ENTRY_REQUIRED - MANIFEST_ENTRY_OPTIONAL)
        if unknown:
            raise WireParseError(where, f"unknown field(s) {unknown} on a manifest entry")
        missing = sorted(MANIFEST_ENTRY_REQUIRED - set(item))
        if missing:
            raise WireParseError(where, f"missing required field(s) {missing}")
        entries.append(
            ManifestEntry(
                file=_as_str(item["file"], f"{where}.file"),
                status=_as_str(item["status"], f"{where}.status"),
                note=_as_str(item["note"], f"{where}.note"),
                kind=None if item.get("kind") is None else _as_str(item["kind"], f"{where}.kind"),
                request=item.get("request"),
            )
        )

    count = raw["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise WireParseError(f"{path}.count", f"expected a non-negative integer, got {count!r}")
    if count != len(entries):
        raise WireParseError(
            path,
            f"`count` says {count} but `entries` holds {len(entries)}; the manifest "
            "disagrees with itself, so neither number can be used as a coverage total",
        )

    version = raw["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise WireParseError(f"{path}.schema_version", f"expected an integer, got {version!r}")

    return Manifest(
        schema_version=version,
        count=count,
        by_status=dict(by_status),
        entries=tuple(entries),
    )


def _as_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise WireParseError(where, f"expected a string, got {type(value).__name__}")
    return value


def _load_json(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise WireParseError(path.name, f"not valid JSON: {exc}") from exc
