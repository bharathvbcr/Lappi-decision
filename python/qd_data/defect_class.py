"""``code.defect_class``: qd-mutate's labelled examples, joined to their licence.

``crates/qd-mutate`` writes one JSONL row per example with the label *by construction*:
the class of the edit it applied (``stub``, ``logic``, ``cosmetic``) or ``clean`` for the
real commit, and for every non-clean row the line span the edit touched. This module turns
that corpus into :class:`DefectRow` s that :func:`qd_data.mixture.rewrite_defect_class`
renders into the one prompt format. It owns three things and nothing else.

**The licence join.** A qd-mutate row carries no licence, and neither does the pool it was
generated from (``commitpackft-pool-v2.jsonl`` rows are ``hunks, id, path, prior_source,
repo, source``). The licence lives in the ``bigcode/commitpackft`` download, whose rows the
pool builder keyed as ``f"{commit}:{new_file}"`` (``tools/build_commitpackft_pool.py``). So
the join is two hops: ``example.pool_id`` must be a pool id, and that pool id must be a
download key. Either hop failing refuses the **whole load**, not the row: a corpus whose rows
do not resolve against the pool it names was generated from some other pool, and every row
of it is then of unknown provenance, not just the ones that happened to miss. A download key
that appears twice with two licences is refused the same way -- the licence would be a pick.
Whether the resolved licence is *admitted* is a per-row question, asked by the rewriter
through ``qd_data.licences.admit_licence`` so a refusal is counted like every other source's.

Every file is checked against the sha256 its manifest records before a row is read.

**The span, rebased into the text the model reads.** The context is the unified diff, not
the post-image: ``qd_train.mutate_adapter`` records that the post-image holds no evidence a
change happened for 54% of this corpus (control on the same rows: 55.8% post-image, 93.8%
diff). qd-mutate's span is over ``after``; the span slot points at lines of the rendered
context. :func:`diff_line_span` maps both ends through
``mutate_adapter.diff_offset_of_after_line`` -- the one implementation of that walk -- and a
span the diff does not represent is carried as a refusal code, never clamped.

**The second-pass permutation.** ``permutation_consistency`` asks whether the choice head
gives the same *answer* when the options are presented in another order. The training
render already shuffles options per example (``qd_data.render.shuffle_options``); the second
pass must be a **derangement** of that order (``docs/schema-api.md``: no option may keep
its position), which is Sattolo's algorithm, not Fisher-Yates. :func:`second_pass_permutation`
draws one and :func:`with_permuted_options` applies it to a request, so rendering the result
at ``seed=None`` gives the second-pass prompt. The gold is by value, so it needs no remap.

The imports of ``qd_train.mutate_adapter`` and ``qd_train.data_access`` are local to the
functions that use them: ``data_access`` imports ``qd_data.manifest``, which imports
``qd_data.mixture``, which imports this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from .config import DataConfig
from .render import second_pass_permutation
from .schema import ChoiceSlot, Request

__all__ = [
    "CHOICE_SLOT",
    "CONTEXT_HEADER_LINES",
    "DEFECT_CLASSES",
    "DEFECT_FAMILY_ID",
    "DEFECT_SOURCE_ID",
    "SPAN_SLOT",
    "DefectCorpusError",
    "DefectLoad",
    "DefectRow",
    "diff_line_span",
    "load_defect_rows",
    "second_pass_permutation",
    "with_permuted_options",
]

DEFECT_FAMILY_ID: Final[str] = "code.defect_class"
DEFECT_SOURCE_ID: Final[str] = "qd-mutate/commitpackft"

#: ``MUTATION_CLASSES`` in ``qd_train.mutate_adapter``, which ``test_mutate_adapter.py``
#: pins against ``crates/qd-mutate/src/ops.rs``. Restated rather than imported for the
#: import-cycle reason in the module docstring; ``test_defect_class.py`` asserts the two
#: are equal, so they cannot drift apart silently. Measured over the 50,177-row
#: ``commitpackft-corpus-v2``: stub 23,074, logic 11,310, clean 8,449, cosmetic 7,344 --
#: exactly these four, so the option set is the corpus's own label set, not a choice.
DEFECT_CLASSES: Final[tuple[str, ...]] = ("stub", "logic", "cosmetic", "clean")

CHOICE_SLOT: Final[str] = "defect_class"
SPAN_SLOT: Final[str] = "defect_span"

#: ``file: <path>`` and a blank line precede the diff in the rendered context, so diff
#: line ``k`` is context line ``k + CONTEXT_HEADER_LINES``. One constant, used by the
#: rewriter to build the header and to offset the span, so the two cannot disagree.
CONTEXT_HEADER_LINES: Final[int] = 2

_DOWNLOAD_LANGUAGES_KEY: Final[str] = "languages"


class DefectCorpusError(ValueError):
    """The corpus, its pool or the licence source cannot be trusted as a whole."""


@dataclass(frozen=True, slots=True)
class DefectRow:
    """One qd-mutate example, with its licence joined and its span rebased.

    ``diff_span`` is 1-based and inclusive over the lines of ``diff``. It is ``None`` for
    ``clean`` (nothing was found, so the span slot abstains) and for a row whose span could
    not be rebased, in which case ``span_refusal`` names why and the rewriter refuses the row.
    """

    example_id: str
    pool_id: str
    repo: str
    path: str
    symbol: str
    arity: int
    language: str
    mutation_class: str
    operator: str
    diff: str
    diff_span: tuple[int, int] | None
    span_refusal: str | None
    licence: str

    def __post_init__(self) -> None:
        if self.mutation_class not in DEFECT_CLASSES:
            raise DefectCorpusError(
                f"{self.example_id}: class {self.mutation_class!r} is not one of {DEFECT_CLASSES}"
            )
        if self.diff_span is not None and self.span_refusal is not None:
            raise DefectCorpusError(
                f"{self.example_id}: a rebased span and a span refusal are two answers"
            )
        if self.mutation_class == "clean" and (
            self.diff_span is not None or self.span_refusal is not None
        ):
            raise DefectCorpusError(
                f"{self.example_id}: a clean example points at nothing, but carries a span"
            )
        if self.mutation_class != "clean" and self.diff_span is None and self.span_refusal is None:
            raise DefectCorpusError(
                f"{self.example_id}: class {self.mutation_class!r} has neither a span nor a "
                "reason it has none; an unlocated mutation must not become a silent noul"
            )


@dataclass(frozen=True, slots=True)
class DefectLoad:
    rows: tuple[DefectRow, ...]
    #: Rows in the corpus file, before any cap.
    n_corpus: int
    #: Whether ``max_rows`` bound. A capped load is a sample, and the mixture must be told.
    capped: bool
    #: ``class -> count`` over the rows returned, and span-rebase refusals by code.
    by_class: dict[str, int]
    span_refusals: dict[str, int]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_sha(path: Path, expected: str, *, recorded_in: Path) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise DefectCorpusError(
            f"{path} hashes to {actual} but {recorded_in} records {expected}: the file "
            "changed after it was recorded, so which rows it holds cannot be stated"
        )


def diff_line_span(diff: bytes, *, start_line: int, end_line: int) -> tuple[int, int]:
    """Map an ``after``-file span to 1-based, inclusive line numbers of ``diff``.

    Both ends are resolved separately through ``diff_offset_of_after_line``; neither is
    inferred from the other. Raises ``SpanOutsideDiff`` / ``MalformedExample`` /
    ``EmptyDiffContext`` from ``qd_train.mutate_adapter`` unchanged.
    """
    from qd_train.mutate_adapter import MalformedExample, diff_offset_of_after_line

    start = diff[: diff_offset_of_after_line(diff, start_line)].count(b"\n") + 1
    end = diff[: diff_offset_of_after_line(diff, end_line)].count(b"\n") + 1
    if end < start:
        raise MalformedExample(
            f"after lines {start_line}..{end_line} map to diff lines {start}..{end}; a "
            "hunk cannot represent a later line before an earlier one"
        )
    return start, end


def _load_licences(download_root: Path) -> dict[str, str]:
    """``f"{commit}:{new_file}" -> license`` over every file the download manifest pins."""
    manifest_path = download_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != "bigcode/commitpackft":
        raise DefectCorpusError(
            f"{manifest_path} names source {manifest.get('source_id')!r}, not "
            "bigcode/commitpackft; its rows are not the licence source this join needs"
        )
    out: dict[str, str] = {}
    for lang, meta in sorted(manifest[_DOWNLOAD_LANGUAGES_KEY].items()):
        path = download_root / f"{lang}.jsonl"
        _check_sha(path, str(meta["sha256"]), recorded_in=manifest_path)
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = f"{obj['commit']}:{obj['new_file']}"
                licence = obj.get("license")
                if not isinstance(licence, str) or not licence.strip():
                    raise DefectCorpusError(f"{path}:{lineno}: {key} carries no license")
                seen = out.get(key)
                if seen is not None and seen != licence:
                    raise DefectCorpusError(
                        f"{path}:{lineno}: {key} appears with licences {seen!r} and "
                        f"{licence!r}; which one governs the row cannot be decided"
                    )
                out[key] = licence
    return out


def _load_pool_ids(pool_path: Path) -> dict[str, str]:
    """``pool id -> repo``. The repo is kept to cross-check the example's own claim."""
    out: dict[str, str] = {}
    with pool_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj["id"])
            if pid in out:
                raise DefectCorpusError(f"{pool_path}:{lineno}: pool id {pid!r} appears twice")
            out[pid] = str(obj["repo"])
    return out


def _parse_one(
    obj: dict[str, Any], *, licence: str
) -> DefectRow:
    from qd_train.mutate_adapter import (
        EmptyDiffContext,
        MalformedExample,
        PhantomFinalLine,
        SpanOutsideDiff,
        parse_example,
    )

    ex = parse_example(obj)  # class/span consistency is refused there, not re-derived here
    if ex.diff is None or not ex.diff.strip():
        # Measured absent on corpus-v2; refused rather than encoded, because on the v1
        # corpus an empty diff meant `clean` and nothing else (AUDIT/after-vs-diff-leak.log).
        raise DefectCorpusError(
            f"{ex.example_id}: empty diff. On this corpus family an empty diff has been a "
            "perfect predictor of `clean`; the corpus must be regenerated, not read"
        )
    diff_span: tuple[int, int] | None = None
    span_refusal: str | None = None
    if ex.span is not None:
        try:
            diff_span = diff_line_span(
                ex.diff, start_line=ex.span.start_line, end_line=ex.span.end_line
            )
        except SpanOutsideDiff:
            span_refusal = "defect_span_outside_diff"
        except EmptyDiffContext:
            span_refusal = "defect_span_empty_diff"
        except (MalformedExample, PhantomFinalLine):
            span_refusal = "defect_span_malformed_diff"
    return DefectRow(
        example_id=ex.example_id,
        pool_id=str(obj["pool_id"]),
        repo=ex.repo,
        path=ex.path,
        symbol=ex.symbol,
        arity=ex.arity,
        language=ex.language,
        mutation_class=ex.mutation_class,
        operator=str(obj.get("operator") or ex.mutation_class),
        diff=ex.diff.decode("utf-8"),
        diff_span=diff_span,
        span_refusal=span_refusal,
        licence=licence,
    )


def load_defect_rows(
    corpus_dir: Path,
    *,
    download_root: Path,
    config: DataConfig,
    repo_root: Path,
    max_rows: int | None = None,
) -> DefectLoad:
    """Read, verify and join a qd-mutate corpus. Fails closed on any unresolved join.

    ``corpus_dir`` holds ``examples.jsonl`` and the ``manifest.json`` qd-mutate wrote; the
    pool is the one that manifest names (``pool.path``, re-anchored under
    ``repo_root/data/pool`` by file name, because the manifest was written in a checkout
    at a different absolute path) and is checked against the sha256 it records.

    ``max_rows`` caps the rows returned. A cap is a sample ordered by sha256 of the example
    id, not a prefix -- the file is grouped by language, so a prefix would be one language.

    Every path read goes through ``assert_path_not_held_out`` first (rule 3).
    """
    from qd_train.data_access import assert_path_not_held_out
    from qd_train.mutate_adapter import MalformedExample

    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    corpus_dir = Path(corpus_dir)
    examples_path = corpus_dir / "examples.jsonl"
    manifest_path = corpus_dir / "manifest.json"
    for p in (examples_path, manifest_path, download_root):
        assert_path_not_held_out(Path(p), config=config, repo_root=Path(repo_root))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("examples_sha256", "pool"):
        if key not in manifest:
            raise DefectCorpusError(
                f"{manifest_path} has no {key!r}; a corpus that does not pin its own bytes "
                "and its pool cannot be joined to a licence"
            )
    pool_meta = manifest["pool"]
    pool_path = Path(repo_root) / "data" / "pool" / Path(str(pool_meta["path"])).name
    assert_path_not_held_out(pool_path, config=config, repo_root=Path(repo_root))
    _check_sha(examples_path, str(manifest["examples_sha256"]), recorded_in=manifest_path)
    _check_sha(pool_path, str(pool_meta["sha256"]), recorded_in=manifest_path)

    pool = _load_pool_ids(pool_path)
    if len(pool) != int(pool_meta["records"]):
        raise DefectCorpusError(
            f"{pool_path} holds {len(pool)} pool ids, {manifest_path} records "
            f"{pool_meta['records']}"
        )
    licences = _load_licences(Path(download_root))

    raw: list[tuple[str, dict[str, Any]]] = []
    unresolved_pool: list[str] = []
    unresolved_licence: list[str] = []
    repo_mismatch: list[str] = []
    with examples_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > config.max_row_bytes:
                raise DefectCorpusError(
                    f"{examples_path}:{lineno}: {len(line)} characters, over the "
                    f"{config.max_row_bytes}-byte row bound"
                )
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DefectCorpusError(f"{examples_path}:{lineno}: not JSON: {e}") from e
            pid = obj.get("pool_id")
            if not isinstance(pid, str) or pid not in pool:
                unresolved_pool.append(f"{lineno}:{pid!r}")
                continue
            if pool[pid] != obj.get("repo"):
                repo_mismatch.append(f"{lineno}:{pid}")
                continue
            if pid not in licences:
                unresolved_licence.append(f"{lineno}:{pid}")
                continue
            raw.append((str(obj.get("id")), obj))
    for what, bad in (
        ("pool_id that is not a pool id", unresolved_pool),
        ("pool id with no row in the commitpackft download, so no licence", unresolved_licence),
        ("repo that disagrees with its pool row", repo_mismatch),
    ):
        if bad:
            raise DefectCorpusError(
                f"{len(bad)} row(s) of {examples_path} carry a {what}; first five: "
                f"{bad[:5]}. Refusing the whole corpus: rows that do not resolve against "
                "the pool the manifest names were generated from some other pool"
            )

    n_corpus = len(raw)
    if n_corpus != int(manifest.get("totals", {}).get("examples", n_corpus)):
        raise DefectCorpusError(
            f"{examples_path} holds {n_corpus} rows, {manifest_path} records "
            f"{manifest['totals']['examples']}"
        )
    capped = max_rows is not None and n_corpus > max_rows
    if capped:
        raw.sort(key=lambda kv: hashlib.sha256(kv[0].encode("utf-8")).hexdigest())
        raw = raw[:max_rows]

    rows: list[DefectRow] = []
    for example_id, obj in raw:
        try:
            rows.append(_parse_one(obj, licence=licences[str(obj["pool_id"])]))
        except MalformedExample as e:
            raise DefectCorpusError(f"{examples_path}: {example_id}: {e}") from e
    ids = Counter(r.example_id for r in rows)
    dupes = sorted(i for i, c in ids.items() if c > 1)
    if dupes:
        raise DefectCorpusError(f"duplicate example ids {dupes[:5]} ({len(dupes)} total)")
    return DefectLoad(
        rows=tuple(rows),
        n_corpus=n_corpus,
        capped=capped,
        by_class=dict(sorted(Counter(r.mutation_class for r in rows).items())),
        span_refusals=dict(
            sorted(Counter(r.span_refusal for r in rows if r.span_refusal).items())
        ),
    )


# -- the second-pass permutation ---------------------------------------------------
#
# ``second_pass_permutation`` lives in ``qd_data.render``, which owns option order; it is
# imported above and re-exported here for the callers that found it in this module.


def with_permuted_options(
    request: Request, *, slot_name: str, permutation: tuple[int, ...]
) -> Request:
    """``request`` with one choice slot's options reordered as ``options[permutation[k]]``.

    Refuses anything that is not a permutation of that slot's option indices -- the runtime
    once applied ``[5, 6, 7]`` to three options (``HANDOFF/stress-2026-09-19.md``).
    """
    slots = list(request.slots)
    for k, slot in enumerate(slots):
        if slot.name != slot_name:
            continue
        if not isinstance(slot, ChoiceSlot):
            raise ValueError(f"slot {slot_name!r} is not a choice slot; it has no option order")
        if sorted(permutation) != list(range(len(slot.options))):
            raise ValueError(
                f"{permutation} is not a permutation of the {len(slot.options)} options of "
                f"slot {slot_name!r}"
            )
        slots[k] = ChoiceSlot(
            name=slot.name, options=tuple(slot.options[i] for i in permutation)
        )
        return replace(request, slots=tuple(slots))
    raise ValueError(f"request has no slot named {slot_name!r}")
