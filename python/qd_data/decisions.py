"""The general-decision pool: ``qd-prep decisions``' rows as DataRows.

``qd-prep decisions`` (``crates/qd-prep/src/decisions.rs``) reads the decision
sources, applies the label rule, the per-row licence, the structural refusals, the
decontamination against the external targets, the group-keyed split and the caps, and
writes ``DIR/{examples.jsonl, manifest.json, containment/}``. Every decision about a row is
made there (lead 2026-10-03: "Rust emits finished rows"); this module checks that the pool
is the one its manifest names and constructs each row through the one funnel
(``mixture._request`` / ``mixture._row``), so the invisible-format refusal, the licence
policy and ``DataRow``'s own checks still have their single owners.

The row's own question is the first paragraph of its context and the request's question is
the family's description -- the ``rewrite_squad`` / ``rewrite_mmlu`` convention. The pool's
split is pinned (``PINNED_SPLIT_KEY``) and is part of the repo key, as MMLU's is, so a group
(one state, one prompt) never straddles train and val.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .config import DataConfig
from .general import _checked_options
from .licences import LicenceConfig, admit_licence
from .loaders import MalformedRowRefusal, TypedDecisionRow, parse_typed_decision
from .mixture import RowRefused, _request, _row
from .rows import DataRow, GoldAnswer
from .schema import ChoiceSlot
from .sources import DECISION_FAMILIES, DECISION_POOL_OPT_INS, PINNED_SPLIT_KEY, source_by_id

__all__ = [
    "DECISION_POOL_SCHEMA",
    "MAX_POOL_ROWS",
    "DecisionPool",
    "DecisionPoolError",
    "load_decision_pool",
    "pool_data_config",
    "rewrite_typed_decision",
]

#: ``manifest.json``'s ``schema``, as ``qd_prep::decisions::MANIFEST_SCHEMA`` writes it.
DECISION_POOL_SCHEMA: Final[str] = "qd-decisions/v1"
#: A pool is read whole or refused: a capped pool would be a sample presented as the pool.
MAX_POOL_ROWS: Final[int] = 400_000
#: A context of at most 131,072 bytes plus 16 options of at most 512, with JSON escaping.
MAX_POOL_ROW_BYTES: Final[int] = 1 << 20

_FAMILY_SOURCE: Final[dict[str, str]] = {f: s for f, s, _ in DECISION_FAMILIES}


class DecisionPoolError(ValueError):
    """The pool directory is not the pool its manifest describes."""


@dataclass(frozen=True, slots=True)
class DecisionPool:
    #: source_id -> its rows, in file order.
    raw: dict[str, tuple[TypedDecisionRow, ...]]
    examples_sha256: str
    manifest: dict[str, object]


def _read_lines(path: Path) -> tuple[list[dict[str, object]], str]:
    """Every JSON object in ``path`` and the sha256 of every byte read, in one pass. Bounded:
    a line over :data:`MAX_POOL_ROW_BYTES` or a row past :data:`MAX_POOL_ROWS` refuses, read
    no further. The digest is of the bytes parsed, so the caller's check of it against the
    manifest is a check of these rows, not of a second read of the file."""
    h = hashlib.sha256()
    out: list[dict[str, object]] = []
    with path.open("rb") as fh:
        for lineno, line in enumerate(iter(lambda: fh.readline(MAX_POOL_ROW_BYTES + 1), b""), 1):
            if len(line) > MAX_POOL_ROW_BYTES:
                raise DecisionPoolError(f"{path}:{lineno}: a line over {MAX_POOL_ROW_BYTES} bytes")
            h.update(line)
            if not line.strip():
                continue
            if len(out) >= MAX_POOL_ROWS:
                raise DecisionPoolError(
                    f"{path} holds more than {MAX_POOL_ROWS} rows; a pool is read whole"
                )
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise MalformedRowRefusal(
                    expected="one JSON object per line",
                    actual=line[:64].decode("utf-8", "replace"),
                    detail=f"{path}:{lineno}: {exc}",
                ) from exc
            if not isinstance(obj, dict):
                raise MalformedRowRefusal(
                    expected="a JSON object", actual=type(obj).__name__, detail=f"{path}:{lineno}",
                )
            out.append(obj)
    return out, h.hexdigest()


def load_decision_pool(pool_dir: Path) -> DecisionPool:
    """The pool at ``pool_dir``, checked against its manifest. Refuses rather than caps."""
    manifest_path = pool_dir / "manifest.json"
    examples_path = pool_dir / "examples.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DecisionPoolError(f"{manifest_path}: not a JSON object")
    if manifest.get("schema") != DECISION_POOL_SCHEMA or manifest.get("mode") != "build":
        raise DecisionPoolError(
            f"{manifest_path}: schema {manifest.get('schema')!r} mode {manifest.get('mode')!r};"
            f" a {DECISION_POOL_SCHEMA!r} build pool is required (a survey has no rows)"
        )
    recorded = manifest.get("examples_sha256")
    lines, actual = _read_lines(examples_path)
    if actual != recorded:
        raise DecisionPoolError(
            f"{examples_path} hashes to {actual} but {manifest_path} records {recorded}: the "
            "file changed after it was written, so which rows it holds cannot be stated"
        )
    if len(lines) != manifest.get("examples"):
        raise DecisionPoolError(
            f"{examples_path}: {len(lines)} rows, the manifest records "
            f"{manifest.get('examples')!r}"
        )
    raw: dict[str, list[TypedDecisionRow]] = {}
    for i, line in enumerate(lines, start=1):
        row = parse_typed_decision(line, where=f"{examples_path}:{i}")
        owner = _FAMILY_SOURCE.get(row.family_id)
        if owner != row.source_id:
            raise MalformedRowRefusal(
                expected=f"a registered decision family of {row.source_id}",
                actual=row.family_id, detail=f"{examples_path}:{i}",
            )
        raw.setdefault(row.source_id, []).append(row)
    return DecisionPool(
        raw={s: tuple(rows) for s, rows in raw.items()},
        examples_sha256=actual,
        manifest=manifest,
    )


def pool_data_config(config: DataConfig | None = None) -> DataConfig:
    """``config`` (default ``DataConfig()``) with the pool's opt-in sources admitted, each with
    the human call recorded in :data:`~qd_data.sources.DECISION_POOL_OPT_INS`. The one place
    a build that reads a pool gets its admission from: the pipeline's build and every
    ``ft_splits`` rebuild of it must admit the same sources, or the rebuild refuses ARC's rows
    at load (``build_mixture``: an opt-in source not admitted). Apply it only when a pool is
    read; a build without one keeps its config, so its manifests do not move."""
    base = DataConfig() if config is None else config
    opted = {**DECISION_POOL_OPT_INS, **base.licence.admitted_sources_by_human}
    return dataclasses.replace(
        base,
        licence=LicenceConfig(
            admitted_by_human=dict(base.licence.admitted_by_human),
            admitted_sources_by_human=opted,
        ),
    )


def rewrite_typed_decision(
    raw: TypedDecisionRow, *, family_id: str, index: int, config: DataConfig
) -> DataRow:
    """One pool row. ``build_mixture`` routes a row only to its own family."""
    if raw.family_id != family_id:
        raise RowRefused(
            reason_code="decision_row_of_other_family", expected=raw.family_id,
            actual=family_id, detail=raw.example_id,
        )
    source = source_by_id(raw.source_id)
    where = f"{raw.source_id} {raw.example_id}"
    if not source.per_row_licence_field and raw.licence != source.declared_licence:
        raise RowRefused(
            reason_code="licence_differs_from_source", expected=source.declared_licence,
            actual=raw.licence, detail=f"{where}: the source carries no per-row licence",
        )
    admit_licence(raw.licence, config=config.licence, source=where)
    options = _checked_options(raw.options, where=where)
    pinned = source.pinned_split_of(raw.split)
    repo_key = f"{family_id}-{pinned}:{raw.group_key}"
    digest = hashlib.blake2b(
        "\x1f".join((raw.context, *options)).encode("utf-8"), digest_size=8
    ).hexdigest()
    row_id = f"decision:{raw.example_id}"
    gold = (
        GoldAnswer(slot_name=raw.slot_name, value=None, is_noul=True)
        if raw.gold_noul
        else GoldAnswer(slot_name=raw.slot_name, value=raw.gold_option)
    )
    return _row(
        row_id=row_id, source_id=source.source_id, family_id=family_id,
        repo_key=repo_key, identity_key=f"{repo_key}::{digest}", licence_id=raw.licence,
        request=_request(
            family_id=family_id, context=raw.context,
            slots=(ChoiceSlot(name=raw.slot_name, options=options),), example_id=row_id,
        ),
        gold=(gold,),
        dedupe_text="\n".join((raw.context, *options)),
        metadata={
            "stratum": raw.stratum,
            "label_basis": raw.label_basis,
            PINNED_SPLIT_KEY: str(pinned),
        },
    )
