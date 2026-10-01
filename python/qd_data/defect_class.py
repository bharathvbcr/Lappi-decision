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

**The noul corpus.** ``crates/qd-mutate``'s ``qd-noul-rows`` writes a second, separate corpus
of this family whose class is :data:`NOUL_CLASS`: contexts that are not this model's kind --
SQuAD prose, scrambled pool files, code in languages the pool never held -- for which the
right answer to "what kind of change is this diff" is to abstain. The OOD gate measured that
CLINC's out-of-scope rows did not teach that on code-shaped inputs (2026-09-30 probe:
prose 31/60, scrambled 1/60, unseen language 0/60). :func:`load_noul_rows` is the only door
a ``noul`` row comes in by: a qd-mutate example with that class is refused by
``parse_example``, and a :class:`DefectRow` carries ``noul`` only together with the
``noul_source`` that says which of the three it is. Its rule-3 check is per row: every row's
split unit is re-derived with the canonical ``qd_data.split`` functions at the run's own seed,
and one row outside ``train`` -- a held-out SQuAD title, a val or held-out repo -- refuses the
whole corpus. :func:`noul_allowlist` runs the same functions for the generator, which never
computes a split itself; its allowlist reads every SQuAD title in order to exclude the
held-out ones, exactly as the pipeline's own SQuAD routing reads them.

The imports of ``qd_train.mutate_adapter`` and ``qd_train.data_access`` are local to the
functions that use them: ``data_access`` imports ``qd_data.manifest``, which imports
``qd_data.mixture``, which imports this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from .config import DataConfig
from .errors import NOUL, LicenceRefused
from .licences import admit_licence, normalise_licence
from .render import INVISIBLE_FORMAT_RANGES, second_pass_permutation
from .schema import ChoiceSlot, Request
from .sources import source_by_id
from .split import SQUAD_TITLE_FAMILIES, assign_repo, squad_title_family, squad_title_repo_key

__all__ = [
    "CHOICE_SLOT",
    "CONTEXT_HEADER_LINES",
    "DEFECT_CLASSES",
    "DEFECT_FAMILY_ID",
    "DEFECT_SOURCE_ID",
    "NOUL_ALLOWLIST_SCHEMA",
    "NOUL_CLASS",
    "NOUL_PROSE",
    "NOUL_SCRAMBLED",
    "NOUL_SOURCES",
    "NOUL_TEMPLATE_LICENCE",
    "NOUL_TEMPLATE_UNIT_PREFIX",
    "NOUL_UNSEEN_LANGUAGE",
    "SPAN_SLOT",
    "DefectCorpusError",
    "DefectLoad",
    "DefectRow",
    "NoulLoad",
    "diff_line_span",
    "load_defect_rows",
    "load_noul_rows",
    "noul_allowlist",
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

#: The class of a noul-corpus row. Not one of :data:`DEFECT_CLASSES`, which are the choice
#: slot's OPTIONS: ``noul`` is the reserved abstention every option set already ends in
#: (``qd_data.render``), and a ``ChoiceSlot`` refuses it as an option. A noul row's choice
#: gold is that abstention.
NOUL_CLASS: Final[str] = NOUL

#: The three sources of ``qd-noul-rows`` (``crates/qd-mutate/src/noul_rows``), one per kind of
#: "not this model's kind": English prose, code scrambled past reading, and code in a language
#: the defect corpus never held.
NOUL_PROSE: Final[str] = "prose"
NOUL_SCRAMBLED: Final[str] = "scrambled"
NOUL_UNSEEN_LANGUAGE: Final[str] = "unseen-language"
NOUL_SOURCES: Final[tuple[str, ...]] = (NOUL_PROSE, NOUL_SCRAMBLED, NOUL_UNSEEN_LANGUAGE)

#: A template row's split unit is ``noul-template/<language>/<template>``: every
#: instantiation of one template is one unit, because they are near-duplicates of each other
#: and a per-row unit would have dedupe chain them across "repos" and drop all but one.
NOUL_TEMPLATE_UNIT_PREFIX: Final[str] = "noul-template/"

#: The templates are authored in this repository, so they carry its licence: ``LICENSE`` is
#: Apache-2.0 and so is the workspace ``license`` in ``Cargo.toml``. ``apache-2.0`` is on the
#: permissive allowlist (``qd_data.licences``); the loader refuses a template row claiming any
#: other id.
NOUL_TEMPLATE_LICENCE: Final[str] = "apache-2.0"
NOUL_TEMPLATE_LICENCE_BASIS: Final[str] = (
    "authored in this repository (crates/qd-mutate/src/noul_rows/templates.rs) and released "
    "under its LICENSE, Apache-2.0 -- the workspace licence in Cargo.toml"
)

#: The SQuAD source whose train-split paragraphs feed the prose rows, and whose registered
#: licence they carry.
_SQUAD_SOURCE_ID: Final[str] = "rajpurkar/squad_v2"
#: The family a SQuAD title must belong to for its paragraphs to feed a noul row: the trained
#: one. ``qa.answerability`` is a held-out family (rule 3) and its titles never feed a row.
_SQUAD_TRAINED_FAMILY: Final[str] = SQUAD_TITLE_FAMILIES[1]

NOUL_ALLOWLIST_SCHEMA: Final[str] = "qd-noul-allowlist/v1"


class DefectCorpusError(ValueError):
    """The corpus, its pool or the licence source cannot be trusted as a whole."""


@dataclass(frozen=True, slots=True)
class DefectRow:
    """One qd-mutate example, with its licence joined and its span rebased.

    ``diff_span`` is 1-based and inclusive over the lines of ``diff``. It is ``None`` for
    ``clean`` (nothing was found, so the span slot abstains) and for a row whose span could
    not be rebased, in which case ``span_refusal`` names why and the rewriter refuses the row.

    ``noul_source`` is set exactly when the class is :data:`NOUL_CLASS`, and names which of
    :data:`NOUL_SOURCES` the row came from; only :func:`load_noul_rows` builds such a row. A
    noul row, like ``clean``, points at nothing, so it carries no span.
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
    noul_source: str | None = None

    def __post_init__(self) -> None:
        if self.mutation_class == NOUL_CLASS:
            if self.noul_source not in NOUL_SOURCES:
                raise DefectCorpusError(
                    f"{self.example_id}: class {NOUL_CLASS!r} with noul_source "
                    f"{self.noul_source!r}; a noul row names which of {NOUL_SOURCES} it is "
                    "from, and only the noul corpus loader builds one"
                )
        elif self.mutation_class not in DEFECT_CLASSES:
            raise DefectCorpusError(
                f"{self.example_id}: class {self.mutation_class!r} is not one of {DEFECT_CLASSES}"
            )
        elif self.noul_source is not None:
            raise DefectCorpusError(
                f"{self.example_id}: class {self.mutation_class!r} carries noul_source "
                f"{self.noul_source!r}; only a {NOUL_CLASS!r} row has one"
            )
        if self.diff_span is not None and self.span_refusal is not None:
            raise DefectCorpusError(
                f"{self.example_id}: a rebased span and a span refusal are two answers"
            )
        if self.mutation_class in ("clean", NOUL_CLASS) and (
            self.diff_span is not None or self.span_refusal is not None
        ):
            raise DefectCorpusError(
                f"{self.example_id}: a {self.mutation_class} example points at nothing, but "
                "carries a span"
            )
        if (
            self.mutation_class not in ("clean", NOUL_CLASS)
            and self.diff_span is None
            and self.span_refusal is None
        ):
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
    #: Noul-corpus rows appended after the main corpus's (``noul_dir``), by source. Never
    #: capped: ``max_rows`` samples the main corpus only, and ``capped`` describes that sample.
    n_noul: int = 0
    noul_by_source: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NoulLoad:
    rows: tuple[DefectRow, ...]
    #: ``noul_source -> count``, in :data:`NOUL_SOURCES` order.
    by_source: dict[str, int]
    #: The corpus's ``examples_sha256``, as checked.
    examples_sha256: str


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
    noul_dir: Path | None = None,
) -> DefectLoad:
    """Read, verify and join a qd-mutate corpus. Fails closed on any unresolved join.

    ``corpus_dir`` holds ``examples.jsonl`` and the ``manifest.json`` qd-mutate wrote; the
    pool is the one that manifest names (``pool.path``, re-anchored under
    ``repo_root/data/pool`` by file name, because the manifest was written in a checkout
    at a different absolute path) and is checked against the sha256 it records.

    ``max_rows`` caps the rows returned. A cap is a sample ordered by sha256 of the example
    id, not a prefix -- the file is grouped by language, so a prefix would be one language.

    ``noul_dir`` names a ``qd-noul-rows`` corpus whose rows are appended, uncapped, after the
    main corpus's, through :func:`load_noul_rows`'s checks. Appended here rather than by each
    caller, so the pipeline and the FT rebuild cannot put the two in different orders.

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
    noul: NoulLoad | None = None
    if noul_dir is not None:
        noul = _load_noul(
            Path(noul_dir), config=config, repo_root=Path(repo_root), licences=licences,
            pools={str(pool_meta["sha256"]): pool},
        )
        rows.extend(noul.rows)
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
        n_noul=0 if noul is None else len(noul.rows),
        noul_by_source={} if noul is None else dict(noul.by_source),
    )


# -- the noul corpus ---------------------------------------------------------------


def _split_of(repo_key: str, config: DataConfig) -> str:
    return assign_repo(
        repo_key, seed=config.seed, train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )


def _noul_violation(
    obj: Mapping[str, Any],
    *,
    pool: Mapping[str, str],
    licences: Mapping[str, str],
    config: DataConfig,
) -> str | None:
    """Why ``obj`` may not be a training row, or ``None``. A reason code, so the refusal can
    count them; every check that touches rule 3 re-derives the split from the canonical
    functions rather than trusting what the generator was told."""
    for key in ("id", "repo", "path", "language", "diff", "licence"):
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"missing_{key}"
    if obj.get("class") != NOUL_CLASS:
        return "class_is_not_noul"
    source = obj.get("noul_source")
    if source not in NOUL_SOURCES:
        return "unknown_noul_source"
    if "\n" in obj["path"]:
        return "path_contains_newline"
    if not obj["diff"].startswith("@@ -"):
        return "diff_is_not_a_hunk"
    repo = str(obj["repo"])
    split = _split_of(repo, config)
    if split != "train":
        # Rule 3, and val too: a noul row is training data, so its unit must be a train unit.
        return f"split_unit_in_{split}"
    licence = normalise_licence(str(obj["licence"]))
    own = {NOUL_PROSE: "squad_title", NOUL_SCRAMBLED: "pool_id", NOUL_UNSEEN_LANGUAGE: "template"}
    for other_source, other_key in own.items():
        if other_source != source and obj.get(other_key) is not None:
            return f"carries_{other_key}_of_another_source"
    if source == NOUL_PROSE:
        title = obj.get("squad_title")
        if not isinstance(title, str) or not title.strip():
            return "prose_without_title"
        if repo != squad_title_repo_key(title):
            return "repo_is_not_the_title_unit"
        if squad_title_family(title, seed=config.seed) != _SQUAD_TRAINED_FAMILY:
            return "title_of_the_held_out_family"
        if licence != normalise_licence(source_by_id(_SQUAD_SOURCE_ID).declared_licence):
            return "licence_is_not_squads"
        return None
    if source == NOUL_SCRAMBLED:
        pid = obj.get("pool_id")
        if not isinstance(pid, str) or pid not in pool:
            return "pool_id_not_in_pool"
        if pool[pid] != repo:
            return "repo_disagrees_with_pool"
        if pid not in licences:
            return "pool_id_without_download_licence"
        if licence != normalise_licence(licences[pid]):
            return "licence_disagrees_with_download"
        return None
    template = obj.get("template")
    if not isinstance(template, str) or not template.strip():
        return "template_row_without_template"
    if not repo.startswith(NOUL_TEMPLATE_UNIT_PREFIX):
        return "repo_is_not_a_template_unit"
    if licence != NOUL_TEMPLATE_LICENCE:
        return "licence_is_not_the_template_licence"
    return None


def _load_noul(
    noul_dir: Path,
    *,
    config: DataConfig,
    repo_root: Path,
    licences: Mapping[str, str],
    pools: dict[str, dict[str, str]],
) -> NoulLoad:
    from qd_train.data_access import assert_path_not_held_out

    examples_path = noul_dir / "examples.jsonl"
    manifest_path = noul_dir / "manifest.json"
    for p in (examples_path, manifest_path):
        assert_path_not_held_out(p, config=config, repo_root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("examples_sha256", "pool", "split", "totals"):
        if key not in manifest:
            raise DefectCorpusError(f"{manifest_path} has no {key!r}")
    split = manifest["split"]
    ours = {
        "seed": config.seed, "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
    }
    if {k: split.get(k) for k in ours} != ours:
        raise DefectCorpusError(
            f"{manifest_path} was generated under split {split}, and this run splits under "
            f"{ours}: a unit that was train there may be val or held out here, so its rows "
            "cannot be admitted as training data. Regenerate the corpus at this run's split."
        )
    pool_meta = manifest["pool"]
    pool_path = repo_root / "data" / "pool" / Path(str(pool_meta["path"])).name
    assert_path_not_held_out(pool_path, config=config, repo_root=repo_root)
    _check_sha(examples_path, str(manifest["examples_sha256"]), recorded_in=manifest_path)
    pool_sha = str(pool_meta["sha256"])
    pool = pools.get(pool_sha)
    if pool is None:
        _check_sha(pool_path, pool_sha, recorded_in=manifest_path)
        pool = _load_pool_ids(pool_path)
    if len(pool) != int(pool_meta["records"]):
        raise DefectCorpusError(
            f"{pool_path} holds {len(pool)} pool ids, {manifest_path} records "
            f"{pool_meta['records']}"
        )

    rows: list[DefectRow] = []
    bad: Counter[str] = Counter()
    first_bad: list[str] = []
    with examples_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > config.max_row_bytes:
                raise DefectCorpusError(
                    f"{examples_path}:{lineno}: over the {config.max_row_bytes}-byte row bound"
                )
            if len(rows) + sum(bad.values()) >= config.max_rows_per_source:
                raise DefectCorpusError(
                    f"{examples_path} holds more than {config.max_rows_per_source} rows"
                )
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DefectCorpusError(f"{examples_path}:{lineno}: not JSON: {e}") from e
            reason = _noul_violation(obj, pool=pool, licences=licences, config=config)
            if reason is not None:
                bad[reason] += 1
                if len(first_bad) < 5:
                    first_bad.append(f"{lineno}:{obj.get('id')!r}:{reason}")
                continue
            source = str(obj["noul_source"])
            unit = str(obj.get("pool_id") or obj["repo"])
            rows.append(DefectRow(
                example_id=str(obj["id"]), pool_id=unit, repo=str(obj["repo"]),
                path=str(obj["path"]), symbol=str(obj["id"]), arity=0,
                language=str(obj["language"]), mutation_class=NOUL_CLASS,
                operator=f"{NOUL_CLASS}.{source}", diff=str(obj["diff"]), diff_span=None,
                span_refusal=None, licence=normalise_licence(str(obj["licence"])),
                noul_source=source,
            ))
    if bad:
        raise DefectCorpusError(
            f"{sum(bad.values())} row(s) of {examples_path} cannot be training rows: "
            f"{dict(sorted(bad.items()))}; first five {first_bad}. Refusing the whole corpus: "
            "a row outside the train split is held-out or val content (rule 3), and a row "
            "whose licence does not resolve is of unknown provenance"
        )
    if len(rows) != int(manifest["totals"].get("examples", -1)):
        raise DefectCorpusError(
            f"{examples_path} holds {len(rows)} rows, {manifest_path} records "
            f"{manifest['totals'].get('examples')}"
        )
    by_source = Counter(r.noul_source for r in rows)
    return NoulLoad(
        rows=tuple(rows),
        by_source={s: by_source[s] for s in NOUL_SOURCES if by_source[s]},
        examples_sha256=str(manifest["examples_sha256"]),
    )


def load_noul_rows(
    noul_dir: Path, *, download_root: Path, config: DataConfig, repo_root: Path
) -> NoulLoad:
    """Read and verify a ``qd-noul-rows`` corpus on its own. Fails closed on any row.

    Every row's split unit is re-derived with ``qd_data.split``'s canonical functions at
    ``config``'s seed: a prose row's title must belong to the trained SQuAD family and hash to
    ``train``, a scrambled row's repo must hash to ``train`` and agree with its pool row, a
    template row's unit must hash to ``train``. One row that fails refuses the whole corpus,
    and so does a manifest generated under another split. Licences are checked per row: a
    prose row carries SQuAD's registered licence, a scrambled row the one the commitpackft
    download gives its pool id (the main corpus's join, from the same ``download_root``), a
    template row :data:`NOUL_TEMPLATE_LICENCE`. Every path read goes through
    ``assert_path_not_held_out`` first (rule 3).
    """
    from qd_train.data_access import assert_path_not_held_out

    assert_path_not_held_out(Path(download_root), config=config, repo_root=Path(repo_root))
    return _load_noul(
        Path(noul_dir), config=config, repo_root=Path(repo_root),
        licences=_load_licences(Path(download_root)), pools={},
    )


def noul_allowlist(
    *,
    squad: Path,
    pool: Path,
    download_root: Path,
    template_units: Mapping[str, Any],
    config: DataConfig,
    repo_root: Path,
) -> dict[str, Any]:
    """The units ``qd-noul-rows generate`` may read, decided by the canonical split.

    * SQuAD: every title of ``squad`` whose article belongs to the trained family
      (``squad_title_family``) and whose unit (``squad_title_repo_key``) hashes to ``train``.
      Every title is read in order to exclude the rest -- the held-out ``qa.answerability``
      titles and the val and held-out ``qa.answer_span`` ones -- as the pipeline's own SQuAD
      routing reads them; they are counted by reason and nothing else of theirs is kept.
    * Pool: every pool id whose repo hashes to ``train`` and whose download licence is
      admitted (``admit_licence``), with that licence.
    * Templates: every unit of ``template_units`` (``qd-noul-rows units``) that hashes to
      ``train``.

    The input sha256s are recorded so the generator can refuse other bytes, and
    ``qd_data.render.INVISIBLE_FORMAT_RANGES`` is carried so the generator can skip a
    candidate the mixture would refuse, without restating the table.
    """
    from qd_train.data_access import assert_path_not_held_out

    for p in (squad, pool, download_root):
        assert_path_not_held_out(Path(p), config=config, repo_root=Path(repo_root))

    titles: set[str] = set()
    with Path(squad).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if lineno > config.max_rows_per_source:
                raise ValueError(f"{squad} holds more than {config.max_rows_per_source} rows")
            if len(line.encode("utf-8")) > config.max_row_bytes:
                raise ValueError(f"{squad}:{lineno}: over the {config.max_row_bytes}-byte bound")
            title = json.loads(line).get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"{squad}:{lineno}: no title, so it has no split unit")
            titles.add(title)
    allowed_titles: dict[str, str] = {}
    title_excluded: Counter[str] = Counter()
    for title in sorted(titles):
        family = squad_title_family(title, seed=config.seed)
        if family != _SQUAD_TRAINED_FAMILY:
            title_excluded[f"family:{family}"] += 1
            continue
        unit = squad_title_repo_key(title)
        split = _split_of(unit, config)
        if split != "train":
            title_excluded[f"split:{split}"] += 1
            continue
        allowed_titles[title] = unit

    pool_ids = _load_pool_ids(Path(pool))
    licences = _load_licences(Path(download_root))
    files: dict[str, str] = {}
    pool_excluded: Counter[str] = Counter()
    for pid, repo in sorted(pool_ids.items()):
        split = _split_of(repo, config)
        if split != "train":
            pool_excluded[f"split:{split}"] += 1
            continue
        licence = licences.get(pid)
        if licence is None:
            pool_excluded["no_download_licence"] += 1
            continue
        try:
            admit_licence(licence, config=config.licence, source=f"noul allowlist {pid}")
        except LicenceRefused:
            pool_excluded[f"licence:{normalise_licence(licence)}"] += 1
            continue
        files[pid] = normalise_licence(licence)

    units = template_units.get("units")
    catalogue = template_units.get("catalogue_sha256")
    if not isinstance(units, list) or not units or not isinstance(catalogue, str):
        raise ValueError("template_units must carry 'units' and 'catalogue_sha256'")
    allowed_units: list[str] = []
    unit_excluded: Counter[str] = Counter()
    for unit in sorted(str(u) for u in units):
        if not unit.startswith(NOUL_TEMPLATE_UNIT_PREFIX):
            raise ValueError(f"template unit {unit!r} lacks {NOUL_TEMPLATE_UNIT_PREFIX!r}")
        split = _split_of(unit, config)
        if split != "train":
            unit_excluded[f"split:{split}"] += 1
            continue
        allowed_units.append(unit)

    return {
        "schema": NOUL_ALLOWLIST_SCHEMA,
        "split": {
            "seed": config.seed, "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
        },
        "invisible_format_ranges": [[lo, hi] for lo, hi in INVISIBLE_FORMAT_RANGES],
        "squad": {
            "file": Path(squad).name, "sha256": _sha256_file(Path(squad)),
            "licence": normalise_licence(source_by_id(_SQUAD_SOURCE_ID).declared_licence),
            "titles": allowed_titles, "excluded": dict(sorted(title_excluded.items())),
        },
        "pool": {
            "file": Path(pool).name, "sha256": _sha256_file(Path(pool)),
            "records": len(pool_ids), "files": files,
            "excluded": dict(sorted(pool_excluded.items())),
        },
        "templates": {
            "catalogue_sha256": catalogue, "licence": NOUL_TEMPLATE_LICENCE,
            "licence_basis": NOUL_TEMPLATE_LICENCE_BASIS, "units": allowed_units,
            "excluded": dict(sorted(unit_excluded.items())),
        },
    }


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
