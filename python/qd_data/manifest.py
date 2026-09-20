"""Manifests, and the ``data_snapshot_hash`` the ledger protocol is built on.

``qd_train.ledger.Protocol`` makes ``data_snapshot_hash`` one of the five components
that decide whether two runs are comparable. So an irreproducible hash does not
merely inconvenience a report -- it makes every "three rows differing only in seed"
claim unverifiable, and the promotion rule a fiction. The hash is therefore a pure
function of things that are themselves pure functions of the corpus:

* the config fingerprint (``DataConfig.fingerprint()``),
* the admitted-source roster and the refusal report, so admitting a new dataset
  changes the hash even before a row of it is read,
* the per-row content hashes, **sorted**, with the split each row landed in.

Deliberately **not** in the hash: wall-clock time, host name, row ordering, row ids,
absolute paths, and the tri-state statuses. The first five would change the hash for
a corpus that did not change. The last is subtler and is the one worth arguing:
statuses are *recorded in the manifest* and are what a reader consults, but they are
outside the hash because a dedupe pass that hit its bound and one that did not can
produce the same surviving corpus, and if they do, the two runs really are
comparable. What must never happen is a ``NotRun`` status being *lost* -- so
:meth:`Manifest.write` refuses to write a manifest whose overall status is not
``Ran`` unless the caller passes ``allow_not_run=True``, and the status travels in
the file either way.

A held-out manifest is written to its own file and carries ``split: "heldout"``.
``qd_train.data_access`` refuses to open it. That refusal is the mechanism for
``CLAUDE.md`` rule 3; this module's job is only to make the refusal *possible* by
saying, in the file, what the file is.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Self

from qd_train.tristate import NotRun, TriState, aggregate, parse_tristate

from .config import SPLITS, DataConfig
from .dedupe import DedupeReport
from .mixture import MixtureResult
from .rows import DataRow, row_content_hash
from .schema import canonical_json
from .sources import admitted_sources, refusal_report
from .split import SplitReport

__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "Manifest",
    "ManifestEntry",
    "build_manifests",
    "data_snapshot_hash",
]

#: Bumped when the hashed body changes shape. It is *inside* the hashed body, so an
#: old manifest and a new one never collide even if their corpora are identical.
MANIFEST_FORMAT_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row's line in the manifest. Everything the model card needs per row."""

    row_id: str
    content_hash: str
    split: str
    source_id: str
    host: str
    family_id: str
    repo_key: str
    identity_key: str
    licence_id: str
    obligations: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "content_hash": self.content_hash,
            "split": self.split,
            "source_id": self.source_id,
            "host": self.host,
            "family_id": self.family_id,
            "repo_key": self.repo_key,
            "identity_key": self.identity_key,
            "licence_id": self.licence_id,
            "obligations": list(self.obligations),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return cls(
            row_id=str(raw["row_id"]),
            content_hash=str(raw["content_hash"]),
            split=str(raw["split"]),
            source_id=str(raw["source_id"]),
            host=str(raw["host"]),
            family_id=str(raw["family_id"]),
            repo_key=str(raw["repo_key"]),
            identity_key=str(raw["identity_key"]),
            licence_id=str(raw["licence_id"]),
            obligations=tuple(str(o) for o in raw.get("obligations", ())),
        )

    def hashed_body(self) -> dict[str, Any]:
        """What enters ``data_snapshot_hash``. Not ``row_id``: two pulls that number
        the same corpus differently must hash the same."""
        return {
            "content_hash": self.content_hash,
            "split": self.split,
            "source_id": self.source_id,
            "family_id": self.family_id,
            "licence_id": self.licence_id,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    """One split's manifest. ``heldout`` gets its own file, and says so in it."""

    split: str
    entries: tuple[ManifestEntry, ...]
    config_fingerprint: dict[str, Any]
    admitted_source_ids: tuple[str, ...]
    refused_sources: dict[str, tuple[str, ...]]
    held_out_families: tuple[str, ...]
    licence_histogram: dict[str, int]
    obligations: dict[str, tuple[str, ...]]
    host_histogram: dict[str, int]
    mixture_json: dict[str, Any]
    dedupe_json: dict[str, Any]
    split_json: dict[str, Any]
    status: TriState
    #: Free-form, excluded from the hash. Wall-clock and host go here or nowhere.
    provenance: dict[str, str]

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; known: {list(SPLITS)}")
        wrong = [e.row_id for e in self.entries if e.split != self.split]
        if wrong:
            raise ValueError(
                f"manifest for split {self.split!r} carries {len(wrong)} entries from another "
                f"split, first five: {wrong[:5]}. A manifest that misreports its own split is "
                "how a held-out shard gets opened by a trainer."
            )
        ids = [e.row_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError(f"manifest for split {self.split!r} has duplicate row ids")

    # -- the hash ---------------------------------------------------------

    def hashed_body(self) -> dict[str, Any]:
        """Everything the hash covers, in a canonical order."""
        return {
            "manifest_format_version": MANIFEST_FORMAT_VERSION,
            "split": self.split,
            "config_fingerprint": self.config_fingerprint,
            "admitted_source_ids": sorted(self.admitted_source_ids),
            "refused_sources": {
                k: sorted(v) for k, v in sorted(self.refused_sources.items())
            },
            "held_out_families": sorted(self.held_out_families),
            "entries": sorted(
                (e.hashed_body() for e in self.entries),
                key=lambda b: (b["content_hash"], b["family_id"], b["source_id"]),
            ),
        }

    def snapshot_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self.hashed_body()).encode("utf-8")
        ).hexdigest()

    # -- file form --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest_format_version": MANIFEST_FORMAT_VERSION,
            "split": self.split,
            "data_snapshot_hash": self.snapshot_hash(),
            "n_rows": len(self.entries),
            "config_fingerprint": self.config_fingerprint,
            "admitted_source_ids": sorted(self.admitted_source_ids),
            "refused_sources": {k: sorted(v) for k, v in sorted(self.refused_sources.items())},
            "held_out_families": sorted(self.held_out_families),
            "licence_histogram": dict(sorted(self.licence_histogram.items())),
            "obligations": {k: list(v) for k, v in sorted(self.obligations.items())},
            "host_histogram": dict(sorted(self.host_histogram.items())),
            "mixture": self.mixture_json,
            "dedupe": self.dedupe_json,
            "split_report": self.split_json,
            "status": self.status.to_json(),
            "provenance": dict(sorted(self.provenance.items())),
            "entries": [e.to_json() for e in self.entries],
        }

    def write(self, path: Path, *, allow_not_run: bool = False) -> str:
        """Write the manifest and return its ``data_snapshot_hash``.

        Refuses to write a manifest whose status is ``NotRun`` unless the caller says
        so explicitly. Writing one silently is how a partial dedupe becomes a
        snapshot that later reads as verified.
        """
        if isinstance(self.status, NotRun) and not allow_not_run:
            raise ValueError(
                f"refusing to write manifest {path}: its status is not_run -- "
                f"{self.status.reason}. Pass allow_not_run=True to record it anyway; "
                "the reason is written into the file either way."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return str(payload["data_snapshot_hash"])

    @classmethod
    def read(cls, path: Path) -> Self:
        """Load a manifest and re-derive its hash, refusing a tampered file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: manifest must be a JSON object")
        version = raw.get("manifest_format_version")
        if version != MANIFEST_FORMAT_VERSION:
            raise ValueError(
                f"{path}: manifest_format_version {version!r} != "
                f"{MANIFEST_FORMAT_VERSION}. Guessing forward-compatibly is how a field "
                "changes meaning silently."
            )
        manifest = cls(
            split=str(raw["split"]),
            entries=tuple(ManifestEntry.from_json(e) for e in raw.get("entries", ())),
            config_fingerprint=dict(raw.get("config_fingerprint", {})),
            admitted_source_ids=tuple(raw.get("admitted_source_ids", ())),
            refused_sources={
                k: tuple(v) for k, v in dict(raw.get("refused_sources", {})).items()
            },
            held_out_families=tuple(raw.get("held_out_families", ())),
            licence_histogram=dict(raw.get("licence_histogram", {})),
            obligations={k: tuple(v) for k, v in dict(raw.get("obligations", {})).items()},
            host_histogram=dict(raw.get("host_histogram", {})),
            mixture_json=dict(raw.get("mixture", {})),
            dedupe_json=dict(raw.get("dedupe", {})),
            split_json=dict(raw.get("split_report", {})),
            status=parse_tristate(raw.get("status"), field=f"{path}:status"),
            provenance=dict(raw.get("provenance", {})),
        )
        recorded = raw.get("data_snapshot_hash")
        derived = manifest.snapshot_hash()
        if recorded != derived:
            raise ValueError(
                f"{path}: recorded data_snapshot_hash {recorded!r} does not match the hash "
                f"re-derived from the file's contents ({derived!r}). The file has been "
                "edited, or was written by a different manifest format."
            )
        return manifest


def _entry(row: DataRow, split_name: str) -> ManifestEntry:
    return ManifestEntry(
        row_id=row.row_id,
        content_hash=row_content_hash(row),
        split=split_name,
        source_id=row.source_id,
        host=row.host,
        family_id=row.family_id,
        repo_key=row.repo_key,
        identity_key=row.identity_key,
        licence_id=row.licence_id,
        obligations=row.obligations,
    )


def build_manifests(
    *,
    config: DataConfig,
    mixture: MixtureResult,
    dedupe_report: DedupeReport,
    split_report: SplitReport,
    provenance: dict[str, str] | None = None,
) -> dict[str, Manifest]:
    """One manifest per split, sharing one status aggregate.

    Every split's manifest carries the *whole* pipeline's reports, not just its own
    slice: a reader holding only the training manifest must still be able to see
    that dedupe hit its bound, because that is exactly the fact a training run needs
    and is exactly the one a per-split summary would drop.
    """
    prov = dict(provenance or {})
    prov.setdefault("built_at_utc", datetime.now(UTC).isoformat(timespec="seconds"))

    status = aggregate(
        {
            "mixture": mixture.status,
            "dedupe": dedupe_report.status,
            "split": split_report.status,
        },
        name="data_snapshot",
    )

    mixture_json = mixture.to_json()
    dedupe_json = dedupe_report.to_json()
    split_json = split_report.to_json()
    admitted = tuple(sorted(s.source_id for s in admitted_sources(config.licence)))
    refused = {k: tuple(v) for k, v in refusal_report(config.licence).items()}

    out: dict[str, Manifest] = {}
    for split_name in SPLITS:
        rows = split_report.rows_by_split.get(split_name, ())
        out[split_name] = Manifest(
            split=split_name,
            entries=tuple(sorted((_entry(r, split_name) for r in rows), key=lambda e: e.row_id)),
            config_fingerprint=dict(config.fingerprint()),
            admitted_source_ids=admitted,
            refused_sources=refused,
            held_out_families=tuple(sorted(config.held_out_families)),
            licence_histogram=_histogram(rows, "licence_id"),
            obligations={r.licence_id: r.obligations for r in rows},
            host_histogram=_histogram(rows, "host"),
            mixture_json=mixture_json,
            dedupe_json=dedupe_json,
            split_json=split_json,
            status=status,
            provenance=prov,
        )
    return out


def _histogram(rows: tuple[DataRow, ...], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, attr))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def data_snapshot_hash(manifests: dict[str, Manifest]) -> str:
    """The one hash the ledger's ``Protocol`` carries.

    Over **every** split, including held-out. A hash that covered only the training
    shard would be identical for two runs whose held-out sets differ, and the whole
    point of the protocol hash is that two rows are comparable only if they were
    measured against the same data.
    """
    if not manifests:
        raise ValueError(
            "data_snapshot_hash over zero manifests: there is no snapshot to identify, "
            "and returning the hash of an empty object would give every empty run the "
            "same protocol"
        )
    missing = [s for s in SPLITS if s not in manifests]
    if missing:
        raise ValueError(
            f"data_snapshot_hash needs every split; missing {missing}. A hash over a "
            "subset would collide across corpora that differ only in the missing split."
        )
    body = {name: manifests[name].hashed_body() for name in sorted(manifests)}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
