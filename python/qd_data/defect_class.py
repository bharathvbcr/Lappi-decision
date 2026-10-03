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

**The composed corpus.** ``qd-mutate compose`` builds PR-shaped rows out of a corpus's own
examples: 3 or more files under ``diff --git`` headers, one of them (or none, for ``clean``) a
mutated example. Its manifest names that corpus as ``base_corpus``, and :func:`load_defect_rows`
reads the base first and appends the composed rows -- so one ``--defect-class`` names both and
every tool that already takes that flag reads them in one order. A composed row carries its
span already in the composed diff's coordinates (``diff_span``), because the walk above refuses
``diff --git`` lines; the loader checks that span against the row's own file blocks instead.
Its rule-3 check is per constituent: every file's repo is re-split at the run's config, and a
row whose files are not all in its own split, or are in ``heldout``, refuses the whole corpus.

The imports of ``qd_train.mutate_adapter`` and ``qd_train.data_access`` are local to the
functions that use them: ``data_access`` imports ``qd_data.manifest``, which imports
``qd_data.mixture``, which imports this module.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    "COMPOSED_FILE_HEADER_LINES",
    "COMPOSED_SCHEMA",
    "CONTEXT_HEADER_LINES",
    "CONTRAST_TWIN_FAMILIES",
    "DEFECT_CLASSES",
    "DEFECT_FAMILY_ID",
    "DEFECT_SOURCE_ID",
    "G6_FORM",
    "G6_LANGUAGES",
    "G6_SCHEMA",
    "NOUL_ALLOWLIST_SCHEMA",
    "NOUL_CLASS",
    "NOUL_COMPOSITE_SCHEMA",
    "NOUL_FORM_KEY",
    "NOUL_PROSE",
    "NOUL_ROUTES",
    "NOUL_ROUTE_CONTRAST",
    "NOUL_ROUTE_G6",
    "NOUL_ROUTE_KEY",
    "NOUL_ROUTE_OWN_PROSE",
    "NOUL_SCRAMBLED",
    "NOUL_SOURCES",
    "NOUL_TEMPLATE_LICENCE",
    "NOUL_TEMPLATE_UNIT_PREFIX",
    "NOUL_UNSEEN_LANGUAGE",
    "NOUL_V5_ALLOWLIST_SCHEMA",
    "OOD_PROSE_PATH",
    "OWN_PROSE_MIN_ASCII_RATIO",
    "OWN_PROSE_MIN_PARAGRAPH_WORDS",
    "OWN_PROSE_QUESTION_MARKUP",
    "OWN_PROSE_QUESTION_WORDS",
    "OWN_PROSE_SCHEMA",
    "PROSE_FORMS",
    "PROSE_FORM_PARAGRAPH",
    "PROSE_FORM_QUESTION",
    "SPAN_SLOT",
    "ContrastSpec",
    "DefectCorpusError",
    "DefectLoad",
    "DefectRow",
    "NoulLoad",
    "OwnProseStrike",
    "composite_digest",
    "diff_line_span",
    "load_defect_rows",
    "load_noul_rows",
    "noul_allowlist",
    "noul_contrast_spec",
    "noul_v5_allowlist",
    "own_prose_ascii_ratio",
    "own_prose_words",
    "prose_defect_row",
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

#: ``MANIFEST_SCHEMA`` and ``FILE_HEADER_LINES`` in ``crates/qd-mutate/src/compose.rs``: a
#: composed corpus's manifest schema, and the ``diff --git`` / ``---`` / ``+++`` lines that open
#: every file block of a composed row.
COMPOSED_SCHEMA: Final[str] = "qd-compose/v1"
COMPOSED_FILE_HEADER_LINES: Final[int] = 3

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

#: ``DataRow.metadata`` keys of a v5 noul row built by one of the routes below: which route,
#: and which form within it. Absent on every row v4 built, so those rows' content is unchanged.
NOUL_ROUTE_KEY: Final[str] = "noul_route"
NOUL_FORM_KEY: Final[str] = "noul_form"

#: v5's new noul routes (campaign/v5-preregistered.DRAFT.json ``data.sources[3,4,6]``).
#: ``own-prose``: the human's own repositories' Markdown, route (i). ``contrast``: a surviving
#: MMLU/CSQA train twin's question, route (ii), derived by ``qd_train.contrast`` after the split.
#: ``commitpackft-g6``: real bigcode/commitpackft commits in eight languages no suite holds (G6).
#: Every routed row is pinned to train (``PINNED_SPLIT_KEY``): none may reach val or held-out.
NOUL_ROUTE_OWN_PROSE: Final[str] = "own-prose"
NOUL_ROUTE_CONTRAST: Final[str] = "contrast"
NOUL_ROUTE_G6: Final[str] = "commitpackft-g6"
NOUL_ROUTES: Final[tuple[str, ...]] = (NOUL_ROUTE_OWN_PROSE, NOUL_ROUTE_CONTRAST, NOUL_ROUTE_G6)

#: The forms a prose route's row takes: one question sentence, or one paragraph.
PROSE_FORM_QUESTION: Final[str] = "question"
PROSE_FORM_PARAGRAPH: Final[str] = "paragraph"
PROSE_FORMS: Final[tuple[str, ...]] = (PROSE_FORM_QUESTION, PROSE_FORM_PARAGRAPH)
#: A G6 row's form: one real commit's diff.
G6_FORM: Final[str] = "commit"

#: The OOD suite's prose path (``qd_train.ood.build_ood_suite``). No training row may carry it:
#: a path that predicts the gate's prose cases is a shortcut the gate would grade.
OOD_PROSE_PATH: Final[str] = "notes.txt"

#: A composite noul corpus's manifest schema: the parts it is the union of, each pinned by
#: sha256, and the contrast rows a build derives after its split (``qd_train.contrast``).
NOUL_COMPOSITE_SCHEMA: Final[str] = "qd-noul-composite/v1"
#: ``qd-noul-rows own-prose``'s manifest schema.
OWN_PROSE_SCHEMA: Final[str] = "qd-own-prose/v1"
#: ``qd-noul-rows g6``'s manifest schema.
G6_SCHEMA: Final[str] = "qd-noul-g6/v1"

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
    :data:`NOUL_SOURCES` the row came from; only :func:`load_noul_rows` and
    :func:`prose_defect_row` build such a row. A noul row, like ``clean``, points at nothing,
    so it carries no span.

    ``noul_route`` names which v5 route (:data:`NOUL_ROUTES`) built a noul row, ``noul_form``
    its form within the route, and ``identity_key`` overrides the rewriter's
    ``repo::path::symbol/arity`` identity for a route whose identity is fixed elsewhere (a
    contrast row is ``contrast:<twin identity>``). All three are ``None`` on every row v4
    read, so those rows render and hash exactly as they did.
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
    noul_route: str | None = None
    noul_form: str | None = None
    identity_key: str | None = None

    def __post_init__(self) -> None:
        if self.noul_route is None:
            if self.noul_form is not None or self.identity_key is not None:
                raise DefectCorpusError(
                    f"{self.example_id}: a noul_form or identity_key without a noul_route; "
                    "only a routed noul row carries them"
                )
        elif self.mutation_class != NOUL_CLASS or self.noul_route not in NOUL_ROUTES:
            raise DefectCorpusError(
                f"{self.example_id}: noul_route {self.noul_route!r} on class "
                f"{self.mutation_class!r}; a route is one of {NOUL_ROUTES} and only a "
                f"{NOUL_CLASS!r} row has one"
            )
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
    #: Composed rows (``qd-mutate compose``) appended after their base corpus's rows. Never
    #: capped: ``max_rows`` samples the base corpus. Counted in ``n_corpus`` too.
    n_composed: int = 0
    #: v5's routed noul rows by route (:data:`NOUL_ROUTES`), a subset of ``n_noul``.
    noul_by_route: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NoulLoad:
    rows: tuple[DefectRow, ...]
    #: ``noul_source -> count``, in :data:`NOUL_SOURCES` order.
    by_source: dict[str, int]
    #: The corpus's ``examples_sha256``, as checked. A composite corpus's is its parts' digest.
    examples_sha256: str
    #: ``noul_route -> count`` over the routed rows (v5); empty for a corpus with none.
    by_route: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """What a composite noul corpus asks a build to derive after its split
    (``qd_train.contrast``): ``per_family[f]`` contrast rows from family ``f``'s surviving
    train twins, drawn in the keyed blake2b order of their identity keys under ``seed``."""

    per_family: dict[str, int]
    seed: int


#: The general families whose questions a contrast row may carry: MMLU and CSQA, the OOD
#: suite's own prose families (``qd_train.ood``'s val-split questions come from these two).
CONTRAST_TWIN_FAMILIES: Final[tuple[str, ...]] = (
    "knowledge.multiple_choice", "commonsense.multiple_choice",
)


def prose_defect_row(
    text: str,
    *,
    route: str,
    form: str,
    example_id: str,
    repo: str,
    path: str,
    licence: str,
    identity_key: str | None = None,
) -> DefectRow:
    """THE one "prose text -> ``code.defect_class`` gold-noul row" builder (Fable's v5 review
    section 2.4): the own-prose route (:func:`_load_own_prose`) and the contrast route
    (``qd_train.contrast``) both call it, and every row it returns is rendered by the one
    ``qd_data.mixture.rewrite_defect_class``. Its context is the text itself under
    ``file: <path>`` -- not a hunk: v5's prose routes present prose as prose, the form the OOD
    suite's prose cases take (``campaign/v5-preregistered.DRAFT.json`` ``data.sources[3]``,
    "Context = the text").

    ``repo`` is the split unit and ``pool_id``; the route pins the row to train
    (``rewrite_defect_class`` sets ``PINNED_SPLIT_KEY``), so no prose route can reach val or
    held-out whatever its unit hashes to. Refuses the OOD suite's prose path, a multi-line
    path, an empty text, and a route or form it does not know.
    """
    if route not in (NOUL_ROUTE_OWN_PROSE, NOUL_ROUTE_CONTRAST):
        raise DefectCorpusError(f"{example_id}: {route!r} is not a prose route")
    if form not in PROSE_FORMS:
        raise DefectCorpusError(f"{example_id}: prose form {form!r} is not one of {PROSE_FORMS}")
    if not text.strip():
        raise DefectCorpusError(f"{example_id}: an empty text is not prose")
    if not path or "\n" in path or "\r" in path:
        raise DefectCorpusError(f"{example_id}: path {path!r} is not one line")
    if Path(path).name == OOD_PROSE_PATH:
        raise DefectCorpusError(
            f"{example_id}: path {path!r} is the OOD suite's prose path; a training row that "
            "carries it teaches the gate's own cue"
        )
    return DefectRow(
        example_id=example_id, pool_id=repo, repo=repo, path=path, symbol=example_id, arity=0,
        language=NOUL_PROSE, mutation_class=NOUL_CLASS, operator=f"{NOUL_CLASS}.{NOUL_PROSE}",
        diff=text, diff_span=None, span_refusal=None, licence=normalise_licence(licence),
        noul_source=NOUL_PROSE, noul_route=route, noul_form=form, identity_key=identity_key,
    )


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


def _composed_diff_span(
    obj: Mapping[str, Any], *, diff: str, clean: bool
) -> tuple[int, int] | None:
    """A composed row's ``diff_span``, checked against the row's own file blocks.

    The generator carried the needle's span into the composed diff and re-verified it on the
    rendered text; this is the loader's independent check, from the row alone. Each
    constituent names its block (``first_line`` .. ``last_line``); the blocks must tile the
    diff and open with their three file headers, and the span must sit in the needle's block
    body, start and end on a ``+`` or context line, and cross no hunk header. Any failure is a
    broken corpus, not a row to count, so it raises.
    """
    row_id = obj.get("id")
    raw_span = obj.get("diff_span")
    if clean:
        if raw_span is not None or obj.get("needle_index") is not None:
            raise DefectCorpusError(f"{row_id}: a clean composed row carries a span or a needle")
        return None
    if not isinstance(raw_span, Mapping):
        raise DefectCorpusError(f"{row_id}: a mutated composed row has no diff_span")
    start, end = raw_span.get("start_line"), raw_span.get("end_line")
    if not (isinstance(start, int) and isinstance(end, int)) or not 1 <= start <= end:
        raise DefectCorpusError(f"{row_id}: diff_span {dict(raw_span)} is not a 1-based range")
    lines = diff.split("\n")
    if not diff.endswith("\n") or end > len(lines) - 1:
        raise DefectCorpusError(f"{row_id}: diff_span ends at {end}, past the diff's lines")
    constituents = obj.get("constituents")
    if not isinstance(constituents, list) or not constituents:
        raise DefectCorpusError(f"{row_id}: a composed row names no constituents")
    expected_first = 1
    needle: Mapping[str, Any] | None = None
    for c in constituents:
        first, last, path = c.get("first_line"), c.get("last_line"), c.get("path")
        if not (isinstance(first, int) and isinstance(last, int) and isinstance(path, str)):
            raise DefectCorpusError(f"{row_id}: a constituent without first_line/last_line/path")
        if first != expected_first or last < first + COMPOSED_FILE_HEADER_LINES:
            raise DefectCorpusError(f"{row_id}: constituent blocks do not tile the diff")
        if lines[first - 1 : first + 2] != [
            f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}"
        ]:
            raise DefectCorpusError(f"{row_id}: the block of {path!r} lacks its file headers")
        if c.get("role") == "needle":
            if needle is not None:
                raise DefectCorpusError(f"{row_id}: two needles in one row")
            needle = c
        expected_first = last + 1
    if expected_first != len(lines):
        raise DefectCorpusError(f"{row_id}: constituent blocks do not cover the diff")
    if needle is None:
        raise DefectCorpusError(f"{row_id}: a mutated composed row has no needle")
    body_first = int(needle["first_line"]) + COMPOSED_FILE_HEADER_LINES
    if start < body_first or end > int(needle["last_line"]):
        raise DefectCorpusError(f"{row_id}: diff_span {start}..{end} leaves the needle's block")
    if any(lines[n - 1][:1] not in ("+", " ") for n in (start, end)):
        raise DefectCorpusError(f"{row_id}: a diff_span endpoint is not a '+' or context line")
    if any(lines[n - 1].startswith("@") for n in range(start, end + 1)):
        raise DefectCorpusError(f"{row_id}: diff_span crosses a hunk header")
    return start, end


def _composed_violation(
    obj: Mapping[str, Any],
    *,
    pool: Mapping[str, str],
    licences: Mapping[str, str],
    config: DataConfig,
) -> str | None:
    """Why a composed row may not be read, or ``None``: rule 3 and the licence join, per file.

    The row's split is re-derived from its repo at the run's config, and every constituent's
    repo must land in that same split, which may not be ``heldout`` -- whatever the generator's
    split map said. Every constituent must resolve through the pool to a download licence that
    is admitted, because the row carries all of their code.
    """
    if obj.get("composed") is not True:
        return "not_composed"
    split = _split_of(str(obj.get("repo")), config)
    if split == "heldout":
        return "row_in_heldout"
    if obj.get("split") != split:
        return f"row_split_{obj.get('split')}_is_{split}_here"
    constituents = obj.get("constituents")
    if not isinstance(constituents, list) or len(constituents) < 2:
        return "fewer_than_two_constituents"
    for c in constituents:
        pid, repo = c.get("pool_id"), c.get("repo")
        if not isinstance(pid, str) or pid not in pool:
            return "constituent_not_in_pool"
        if pool[pid] != repo:
            return "constituent_repo_disagrees_with_pool"
        if pid not in licences:
            return "constituent_without_download_licence"
        try:
            admit_licence(licences[pid], config=config.licence, source=f"composed {pid}")
        except LicenceRefused:
            return "constituent_licence_not_admitted"
        if _split_of(str(repo), config) != split:
            return "constituents_span_two_splits"
    return None


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
    if obj.get("composed") is True:
        diff_span = _composed_diff_span(obj, diff=ex.diff.decode("utf-8"), clean=ex.span is None)
    elif ex.span is not None:
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

    A ``qd-mutate compose`` corpus (manifest schema :data:`COMPOSED_SCHEMA`) names its
    ``base_corpus``, re-anchored under ``repo_root/data/pool`` like the pool: the base is read
    first, through this function (``max_rows`` samples it), and the composed rows follow,
    uncapped, then the noul rows. Every composed row passes :func:`_composed_violation` or the
    whole corpus is refused.

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
    base: DefectLoad | None = None
    if manifest.get("schema") == COMPOSED_SCHEMA or "base_corpus" in manifest:
        base = _load_composed_base(
            manifest, manifest_path=manifest_path, download_root=Path(download_root),
            config=config, repo_root=Path(repo_root), max_rows=max_rows,
        )
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
    composed_flags = sorted({obj.get("composed") is True for _, obj in raw})
    if composed_flags != ([True] if base is not None else [False] if raw else []):
        raise DefectCorpusError(
            f"{examples_path}: a composed corpus holds only composed rows and a plain corpus "
            f"none (found composed={composed_flags}, base_corpus named: {base is not None})"
        )
    if base is not None:
        bad: Counter[str] = Counter()
        first_bad: list[str] = []
        for example_id, obj in raw:
            reason = _composed_violation(obj, pool=pool, licences=licences, config=config)
            if reason is not None:
                bad[reason] += 1
                if len(first_bad) < 5:
                    first_bad.append(f"{example_id}:{reason}")
        if bad:
            raise DefectCorpusError(
                f"{sum(bad.values())} composed row(s) of {examples_path} cannot be read: "
                f"{dict(sorted(bad.items()))}; first five {first_bad}. Refusing the whole "
                "corpus: a row whose files cross a split boundary or reach a held-out repo "
                "carries that content into training (rule 3)"
            )
    capped = max_rows is not None and n_corpus > max_rows and base is None
    if capped:
        raw.sort(key=lambda kv: hashlib.sha256(kv[0].encode("utf-8")).hexdigest())
        raw = raw[:max_rows]

    rows: list[DefectRow] = []
    for example_id, obj in raw:
        try:
            rows.append(_parse_one(obj, licence=licences[str(obj["pool_id"])]))
        except MalformedExample as e:
            raise DefectCorpusError(f"{examples_path}: {example_id}: {e}") from e
    n_composed = 0
    if base is not None:
        n_composed = len(rows)
        rows = list(base.rows) + rows
        n_corpus += base.n_corpus
        capped = base.capped
    noul: NoulLoad | None = None
    if noul_dir is not None:
        noul = _load_noul_corpus(
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
        n_composed=n_composed,
        noul_by_route={} if noul is None else dict(noul.by_route),
    )


def _load_composed_base(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    download_root: Path,
    config: DataConfig,
    repo_root: Path,
    max_rows: int | None,
) -> DefectLoad:
    """The corpus a composed corpus was built from, read and checked before its rows are.

    The base must be the bytes the composer read (``examples_sha256``), over the same pool,
    split under this run's config, and must not itself be composed.
    """
    if manifest.get("schema") != COMPOSED_SCHEMA:
        raise DefectCorpusError(
            f"{manifest_path} names a base_corpus but its schema is {manifest.get('schema')!r}, "
            f"not {COMPOSED_SCHEMA!r}"
        )
    meta = manifest.get("base_corpus")
    if not isinstance(meta, Mapping) or not isinstance(meta.get("name"), str):
        raise DefectCorpusError(f"{manifest_path} has no base_corpus name")
    params = manifest.get("split_params")
    ours = {
        "seed": config.seed, "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
    }
    if not isinstance(params, Mapping) or {k: params.get(k) for k in ours} != ours:
        raise DefectCorpusError(
            f"{manifest_path} was composed under split {params}, and this run splits under "
            f"{ours}: a repo that was train there may be val or held out here. Recompose the "
            "corpus at this run's split."
        )
    base_dir = repo_root / "data" / "pool" / Path(str(meta["name"])).name
    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    if "base_corpus" in base_manifest or base_manifest.get("schema") == COMPOSED_SCHEMA:
        raise DefectCorpusError(f"{base_dir} is itself composed; a base must be a plain corpus")
    if base_manifest.get("examples_sha256") != meta.get("examples_sha256"):
        raise DefectCorpusError(
            f"{base_dir} records examples {base_manifest.get('examples_sha256')}, and "
            f"{manifest_path} was composed from {meta.get('examples_sha256')}"
        )
    pool = manifest.get("pool")
    if not isinstance(pool, Mapping) or base_manifest.get("pool", {}).get("sha256") != pool.get(
        "sha256"
    ):
        raise DefectCorpusError(f"{manifest_path} and its base {base_dir} name different pools")
    return load_defect_rows(
        base_dir, download_root=download_root, config=config, repo_root=repo_root,
        max_rows=max_rows,
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


# -- v5: the composite noul corpus and its two new parts --------------------------------

#: own-prose's content rules (Fable's v5 review section 2.9 (3); the DRAFT's ``data.sources[3]``):
#: a paragraph has at least this many words, a question 6-60, and either at least this share of
#: ASCII characters; at most this many rows come from one repository.
OWN_PROSE_MIN_PARAGRAPH_WORDS: Final[int] = 25
OWN_PROSE_QUESTION_WORDS: Final[tuple[int, int]] = (6, 60)
#: A question unit starts with an ASCII capital, holds none of these (inline code, emphasis cut
#: by the sentence split, a table cell, a link, a brace: the walker's ``QUESTION_MARKUP``), and
#: closes every double quote and parenthesis it opens.
OWN_PROSE_QUESTION_MARKUP: Final[tuple[str, ...]] = ("`", "*", "|", "](", "http", "{", "}")
OWN_PROSE_MIN_ASCII_RATIO: Final[float] = 0.9
OWN_PROSE_PER_REPO_CAP: Final[int] = 150
#: What a word is, for every count above: the inventory's definition
#: (``AUDIT/v5-plan-2026-10-02/own_repo_inventory.py``), which the walker
#: (``crates/qd-mutate/src/noul_rows/own_prose.rs``) implements byte for byte.
OWN_PROSE_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z'\-]+")
#: The split unit of an own-prose row: one per repository.
OWN_PROSE_REPO_PREFIX: Final[str] = "own-prose:"

#: G6's languages: bigcode/commitpackft's ``data/<lang>`` for eight languages that no suite and
#: no other training source holds (``campaign/v5-preregistered.DRAFT.json`` ``data.sources[6]``;
#: ``python/tests/test_defect_noul_v5.py`` asserts the disjointness against each suite's own
#: constants).
G6_LANGUAGES: Final[tuple[str, ...]] = (
    "clojure",
    "erlang",
    "fortran",
    "julia",
    "ocaml",
    "perl",
    "r",
    "tcl",
)


def own_prose_words(text: str) -> int:
    return len(OWN_PROSE_WORD.findall(text))


def own_prose_ascii_ratio(text: str) -> float:
    return sum(ch.isascii() for ch in text) / len(text) if text else 0.0


def _json_lines(path: Path, *, config: DataConfig, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > config.max_row_bytes:
                raise DefectCorpusError(
                    f"{path}:{lineno}: over the {config.max_row_bytes}-byte bound"
                )
            if len(out) >= limit:
                raise DefectCorpusError(f"{path} holds more than {limit} rows")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DefectCorpusError(f"{path}:{lineno}: not JSON: {e}") from e
            if not isinstance(obj, dict):
                raise DefectCorpusError(f"{path}:{lineno}: not a JSON object")
            out.append(obj)
    return out


@dataclass(frozen=True, slots=True)
class OwnProseStrike:
    """The human's strike as the own-prose manifest records it: whole repos, and repo-relative
    directories (``prefix`` ends in ``/``). The walker never reads a struck file; the loader
    refuses any file or unit one would cover."""

    basis: str
    repos: frozenset[str]
    paths: tuple[tuple[str, str], ...]

    def covers(self, repo: str, path: str) -> bool:
        return repo in self.repos or any(
            repo == r and path.startswith(prefix) for r, prefix in self.paths
        )


def _own_prose_strike(
    manifest: Mapping[str, Any], *, manifest_path: Path, admitted: set[str]
) -> OwnProseStrike | None:
    """The manifest's ``strike``, checked, or ``None`` when it has none."""
    raw = manifest.get("strike")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not {"basis", "repos", "paths"} <= set(raw):
        raise DefectCorpusError(f"{manifest_path}: strike must carry basis, repos and paths")
    basis = raw["basis"]
    repos = raw["repos"]
    paths = raw["paths"]
    if not isinstance(basis, str) or not basis.strip():
        raise DefectCorpusError(f"{manifest_path}: a strike without its basis records no decision")
    if not isinstance(repos, list) or not all(isinstance(r, str) for r in repos):
        raise DefectCorpusError(f"{manifest_path}: strike.repos must be a list of repo names")
    if not isinstance(paths, list) or not all(
        isinstance(p, Mapping)
        and isinstance(p.get("repo"), str)
        and isinstance(p.get("prefix"), str)
        for p in paths
    ):
        raise DefectCorpusError(f"{manifest_path}: strike.paths must be {{repo, prefix}} objects")
    pairs = tuple((str(p["repo"]), str(p["prefix"])) for p in paths)
    for repo, prefix in pairs:
        parts = prefix.split("/")
        if (
            not prefix.endswith("/")
            or prefix.startswith("/")
            or "//" in prefix
            or ".." in parts
            or "." in parts
        ):
            raise DefectCorpusError(
                f"{manifest_path}: strike path {repo}:{prefix} is not a repo-relative directory"
            )
    unknown = sorted({*repos, *(r for r, _ in pairs)} - admitted)
    if unknown or not (repos or pairs):
        raise DefectCorpusError(
            f"{manifest_path}: strike names no rule or names repos that were not admitted: "
            f"{unknown}"
        )
    return OwnProseStrike(basis=basis, repos=frozenset(repos), paths=pairs)


def _own_prose_violation(
    obj: Mapping[str, Any],
    *,
    files: set[tuple[str, str, str]],
    repos: set[str],
    strike: OwnProseStrike | None = None,
) -> str | None:
    """Why ``obj`` may not be an own-prose row, or ``None``: the walker's rules, re-checked."""
    for key in ("id", "repo", "path", "file_sha256", "form", "text"):
        if not isinstance(obj.get(key), str) or not str(obj[key]).strip():
            return f"missing_{key}"
    text, form, path = str(obj["text"]), str(obj["form"]), str(obj["path"])
    if strike is not None and strike.covers(str(obj["repo"]), path):
        return "struck_by_the_human"
    if obj["repo"] not in repos:
        return "repo_not_admitted"
    if (obj["repo"], path, obj["file_sha256"]) not in files:
        return "file_not_in_files_jsonl"
    if "\n" in path or "\r" in path or Path(path).name == OOD_PROSE_PATH:
        return "path_refused"
    if "\n" in text or "\r" in text:
        return "text_is_not_one_line"
    words = own_prose_words(text)
    if form == PROSE_FORM_QUESTION:
        lo, hi = OWN_PROSE_QUESTION_WORDS
        if not (lo <= words <= hi and text.rstrip().endswith("?")):
            return "question_out_of_band"
        if not ("A" <= text[0] <= "Z"):
            return "question_not_a_sentence_start"
        if any(mark in text for mark in OWN_PROSE_QUESTION_MARKUP):
            return "question_has_markup"
        if text.count('"') % 2 or text.count("(") != text.count(")"):
            return "question_unbalanced"
    elif form == PROSE_FORM_PARAGRAPH:
        if words < OWN_PROSE_MIN_PARAGRAPH_WORDS:
            return "paragraph_too_short"
    else:
        return "unknown_form"
    if own_prose_ascii_ratio(text) < OWN_PROSE_MIN_ASCII_RATIO:
        return "ascii_ratio_below_min"
    return None


def _load_own_prose(part_dir: Path, *, config: DataConfig, repo_root: Path) -> NoulLoad:
    """``qd-noul-rows own-prose``'s output as rows, through :func:`prose_defect_row`. Fails closed:
    a manifest that does not pin its two files, ranges that are not ``render``'s, a licence or a
    source that is not admitted, and any one row that breaks a rule refuse the whole part."""
    from qd_train.data_access import assert_path_not_held_out

    from .sources import OWN_REPOS_SOURCE_ID

    manifest_path = part_dir / "manifest.json"
    files_path = part_dir / "files.jsonl"
    units_path = part_dir / "units.jsonl"
    for p in (manifest_path, files_path, units_path):
        assert_path_not_held_out(p, config=config, repo_root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != OWN_PROSE_SCHEMA:
        raise DefectCorpusError(
            f"{manifest_path}: schema {manifest.get('schema')!r}, not {OWN_PROSE_SCHEMA!r}"
        )
    if manifest.get("source_id") != OWN_REPOS_SOURCE_ID:
        raise DefectCorpusError(
            f"{manifest_path}: source {manifest.get('source_id')!r}, not {OWN_REPOS_SOURCE_ID!r}"
        )
    refusals = source_by_id(OWN_REPOS_SOURCE_ID).admission_refusals(config.licence)
    if refusals:
        raise DefectCorpusError(
            f"{OWN_REPOS_SOURCE_ID} is not admitted by this run: {list(refusals)}"
        )
    licence = str(manifest.get("licence", ""))
    if licence != source_by_id(OWN_REPOS_SOURCE_ID).declared_licence:
        raise DefectCorpusError(f"{manifest_path}: licence {licence!r} is not the source's")
    admit_licence(licence, config=config.licence, source=OWN_REPOS_SOURCE_ID)
    ranges = [tuple(r) for r in manifest.get("invisible_format_ranges", [])]
    if ranges != [tuple(r) for r in INVISIBLE_FORMAT_RANGES]:
        raise DefectCorpusError(
            f"{manifest_path} was walked under invisible-format ranges {ranges}, not "
            "qd_data.render.INVISIBLE_FORMAT_RANGES: re-walk it with the current table"
        )
    _check_sha(files_path, str(manifest["files"]["sha256"]), recorded_in=manifest_path)
    _check_sha(units_path, str(manifest["units"]["sha256"]), recorded_in=manifest_path)
    files = _json_lines(files_path, config=config, limit=config.max_rows_per_source)
    units = _json_lines(units_path, config=config, limit=config.max_rows_per_source)
    if len(files) != int(manifest["files"]["count"]) or len(units) != int(
        manifest["units"]["count"]
    ):
        raise DefectCorpusError(f"{part_dir}: file or unit counts disagree with {manifest_path}")
    repos = {str(r["repo"]) for r in manifest.get("admitted_repos", [])}
    strike = _own_prose_strike(manifest, manifest_path=manifest_path, admitted=repos)
    file_keys = {(str(f.get("repo")), str(f.get("path")), str(f.get("sha256"))) for f in files}
    if strike is not None:
        struck_files = sorted(f"{r}/{p}" for r, p, _ in file_keys if strike.covers(r, p))
        if struck_files:
            raise DefectCorpusError(
                f"{files_path} lists {len(struck_files)} file(s) the manifest's strike covers, "
                f"first five {struck_files[:5]}: a struck file is never read"
            )
    bad: Counter[str] = Counter()
    first_bad: list[str] = []
    per_repo: Counter[str] = Counter()
    rows: list[DefectRow] = []
    for obj in units:
        reason = _own_prose_violation(obj, files=file_keys, repos=repos, strike=strike)
        if reason is None:
            per_repo[str(obj["repo"])] += 1
            if per_repo[str(obj["repo"])] > OWN_PROSE_PER_REPO_CAP:
                reason = "over_per_repo_cap"
        if reason is not None:
            bad[reason] += 1
            if len(first_bad) < 5:
                first_bad.append(f"{obj.get('id')!r}:{reason}")
            continue
        rows.append(
            prose_defect_row(
                str(obj["text"]),
                route=NOUL_ROUTE_OWN_PROSE,
                form=str(obj["form"]),
                example_id=f"{NOUL_ROUTE_OWN_PROSE}:{obj['id']}",
                repo=f"{OWN_PROSE_REPO_PREFIX}{obj['repo']}",
                path=str(obj["path"]),
                licence=licence,
            )
        )
    if bad:
        raise DefectCorpusError(
            f"{sum(bad.values())} unit(s) of {units_path} cannot be own-prose rows: "
            f"{dict(sorted(bad.items()))}; first five {first_bad}. Refusing the whole part"
        )
    return NoulLoad(
        rows=tuple(rows),
        by_source={NOUL_PROSE: len(rows)} if rows else {},
        examples_sha256=str(manifest["units"]["sha256"]),
        by_route={NOUL_ROUTE_OWN_PROSE: len(rows)} if rows else {},
    )


def _g6_violation(
    obj: Mapping[str, Any], *, config: DataConfig, manifest: Mapping[str, Any]
) -> str | None:
    """Why ``obj`` may not be a G6 row, or ``None``. The split and the licence are re-derived
    with the canonical functions, never read off the generator."""
    for key in ("id", "repo", "path", "language", "diff", "licence", "commit"):
        if not isinstance(obj.get(key), str) or not str(obj[key]).strip():
            return f"missing_{key}"
    if obj.get("class") != NOUL_CLASS or obj.get("noul_source") != NOUL_UNSEEN_LANGUAGE:
        return "not_an_unseen_language_noul_row"
    if obj.get("noul_route") != NOUL_ROUTE_G6 or obj.get("noul_form") != G6_FORM:
        return "not_a_g6_row"
    if obj["language"] not in G6_LANGUAGES:
        return "language_not_g6"
    repo, path, diff = str(obj["repo"]), str(obj["path"]), str(obj["diff"])
    if "," in repo or repo != repo.strip():
        return "repo_is_not_a_primary_repo"
    if "\n" in path or Path(path).name == OOD_PROSE_PATH:
        return "path_refused"
    if not diff.startswith("@@ -"):
        return "diff_is_not_a_hunk"
    band = manifest["diff_band"]
    hunks = sum(1 for line in diff.split("\n") if line.startswith("@@ -"))
    if not (1 <= hunks <= int(band["max_hunks"])):
        return "hunks_out_of_band"
    if not (int(band["min_chars"]) <= len(diff) <= int(band["max_chars"])):
        return "chars_out_of_band"
    split = _split_of(repo, config)
    if split != "train":
        return f"split_unit_in_{split}"
    try:
        admit_licence(str(obj["licence"]), config=config.licence, source=f"G6 {obj['id']}")
    except LicenceRefused:
        return "licence_not_admitted"
    return None


def _load_g6(part_dir: Path, *, config: DataConfig, repo_root: Path) -> NoulLoad:
    """``qd-noul-rows g6``'s output as rows. Fails closed on any row, on a manifest generated at
    another split, and on a language over the cap the manifest pins."""
    from qd_train.data_access import assert_path_not_held_out

    manifest_path = part_dir / "manifest.json"
    examples_path = part_dir / "examples.jsonl"
    for p in (manifest_path, examples_path):
        assert_path_not_held_out(p, config=config, repo_root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != G6_SCHEMA:
        raise DefectCorpusError(
            f"{manifest_path}: schema {manifest.get('schema')!r}, not {G6_SCHEMA!r}"
        )
    ours = {
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
    }
    if {k: manifest.get("split", {}).get(k) for k in ours} != ours:
        raise DefectCorpusError(
            f"{manifest_path} was generated under split {manifest.get('split')}, not {ours}"
        )
    _check_sha(examples_path, str(manifest["examples_sha256"]), recorded_in=manifest_path)
    cap = int(manifest["per_language_cap"])
    bad: Counter[str] = Counter()
    first_bad: list[str] = []
    by_language: Counter[str] = Counter()
    rows: list[DefectRow] = []
    for obj in _json_lines(examples_path, config=config, limit=config.max_rows_per_source):
        reason = _g6_violation(obj, config=config, manifest=manifest)
        if reason is None:
            by_language[str(obj["language"])] += 1
            if by_language[str(obj["language"])] > cap:
                reason = "over_per_language_cap"
        if reason is not None:
            bad[reason] += 1
            if len(first_bad) < 5:
                first_bad.append(f"{obj.get('id')!r}:{reason}")
            continue
        rows.append(
            DefectRow(
                example_id=str(obj["id"]),
                pool_id=str(obj["repo"]),
                repo=str(obj["repo"]),
                path=str(obj["path"]),
                symbol=str(obj["id"]),
                arity=0,
                language=str(obj["language"]),
                mutation_class=NOUL_CLASS,
                operator=f"{NOUL_CLASS}.{NOUL_UNSEEN_LANGUAGE}",
                diff=str(obj["diff"]),
                diff_span=None,
                span_refusal=None,
                licence=normalise_licence(str(obj["licence"])),
                noul_source=NOUL_UNSEEN_LANGUAGE,
                noul_route=NOUL_ROUTE_G6,
                noul_form=G6_FORM,
            )
        )
    if bad:
        raise DefectCorpusError(
            f"{sum(bad.values())} row(s) of {examples_path} cannot be G6 rows: "
            f"{dict(sorted(bad.items()))}; first five {first_bad}. Refusing the whole part"
        )
    if len(rows) != int(manifest["totals"]["examples"]):
        raise DefectCorpusError(
            f"{examples_path} holds {len(rows)} rows, {manifest_path} records "
            f"{manifest['totals']['examples']}"
        )
    return NoulLoad(
        rows=tuple(rows),
        by_source={NOUL_UNSEEN_LANGUAGE: len(rows)} if rows else {},
        examples_sha256=str(manifest["examples_sha256"]),
        by_route={NOUL_ROUTE_G6: len(rows)} if rows else {},
    )


def composite_digest(parts: list[Mapping[str, Any]]) -> str:
    """A composite noul corpus's ``examples_sha256``: sha256 over each part's name, kind and pins,
    one line per part, in part order -- the bytes every part's own manifest already pins."""
    lines = [
        json.dumps(
            {k: p[k] for k in ("name", "kind", "manifest_sha256", "data_sha256")}, sort_keys=True
        )
        for p in parts
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


_PART_KINDS: Final[tuple[str, ...]] = ("noul-rows", "own-prose", "g6")


def _load_composite(
    noul_dir: Path,
    manifest: Mapping[str, Any],
    *,
    config: DataConfig,
    repo_root: Path,
    licences: Mapping[str, str],
    pools: dict[str, dict[str, str]],
) -> NoulLoad:
    """A :data:`NOUL_COMPOSITE_SCHEMA` corpus: each part, a sibling directory of ``noul_dir``,
    is checked against the manifest's and data file's sha256 the composite pins, loaded by its
    own loader, and appended in part order."""
    manifest_path = noul_dir / "manifest.json"
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise DefectCorpusError(f"{manifest_path} names no parts")
    if manifest.get("examples_sha256") != composite_digest(parts):
        raise DefectCorpusError(f"{manifest_path}: examples_sha256 is not its parts' digest")
    rows: list[DefectRow] = []
    by_source: Counter[str] = Counter()
    by_route: Counter[str] = Counter()
    for part in parts:
        kind, name = part.get("kind"), str(part.get("name", ""))
        if kind not in _PART_KINDS or not name or "/" in name:
            raise DefectCorpusError(f"{manifest_path}: part {part!r} is not one of {_PART_KINDS}")
        part_dir = noul_dir.parent / name
        _check_sha(
            part_dir / "manifest.json", str(part["manifest_sha256"]), recorded_in=manifest_path
        )
        data_name = "units.jsonl" if kind == "own-prose" else "examples.jsonl"
        _check_sha(part_dir / data_name, str(part["data_sha256"]), recorded_in=manifest_path)
        if kind == "own-prose":
            load = _load_own_prose(part_dir, config=config, repo_root=repo_root)
        elif kind == "g6":
            load = _load_g6(part_dir, config=config, repo_root=repo_root)
        else:
            load = _load_noul(
                part_dir, config=config, repo_root=repo_root, licences=licences, pools=pools
            )
        if len(load.rows) != int(part["rows"]):
            raise DefectCorpusError(
                f"part {name}: {len(load.rows)} rows, {manifest_path} records {part['rows']}"
            )
        rows.extend(load.rows)
        by_source.update(load.by_source)
        by_route.update(load.by_route)
    if len(rows) != int(manifest["totals"]["examples"]):
        raise DefectCorpusError(
            f"{noul_dir}: {len(rows)} rows, {manifest_path} records "
            f"{manifest['totals']['examples']}"
        )
    return NoulLoad(
        rows=tuple(rows),
        by_source={s: by_source[s] for s in NOUL_SOURCES if by_source[s]},
        examples_sha256=str(manifest["examples_sha256"]),
        by_route={r: by_route[r] for r in NOUL_ROUTES if by_route[r]},
    )


def _load_noul_corpus(
    noul_dir: Path,
    *,
    config: DataConfig,
    repo_root: Path,
    licences: Mapping[str, str],
    pools: dict[str, dict[str, str]],
) -> NoulLoad:
    """A ``qd-noul-rows`` corpus, or a composite of them (v5's defect-noul-v3c)."""
    from qd_train.data_access import assert_path_not_held_out

    manifest_path = noul_dir / "manifest.json"
    assert_path_not_held_out(manifest_path, config=config, repo_root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") == NOUL_COMPOSITE_SCHEMA:
        return _load_composite(
            noul_dir,
            manifest,
            config=config,
            repo_root=repo_root,
            licences=licences,
            pools=pools,
        )
    return _load_noul(noul_dir, config=config, repo_root=repo_root, licences=licences, pools=pools)


def noul_contrast_spec(noul_dir: Path) -> ContrastSpec | None:
    """The contrast rows a noul corpus asks a build to derive, or ``None`` when it asks for none
    (every corpus before v5's composite). Refuses a malformed request rather than reading it as
    none."""
    manifest = json.loads((Path(noul_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != NOUL_COMPOSITE_SCHEMA or "contrast" not in manifest:
        return None
    raw = manifest["contrast"]
    per_family = raw.get("per_family") if isinstance(raw, dict) else None
    seed = raw.get("seed") if isinstance(raw, dict) else None
    if (
        not isinstance(per_family, dict)
        or not per_family
        or set(per_family) - set(CONTRAST_TWIN_FAMILIES)
        or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in per_family.values())
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise DefectCorpusError(
            f"{noul_dir}/manifest.json: contrast {raw!r} is not {{per_family: {{one of "
            f"{CONTRAST_TWIN_FAMILIES}: positive int}}, seed: non-negative int}}"
        )
    return ContrastSpec(per_family=dict(per_family), seed=seed)


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
    return _load_noul_corpus(
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


#: ``qd-noul-rows own-prose`` and ``qd-noul-rows g6`` read this allowlist.
NOUL_V5_ALLOWLIST_SCHEMA: Final[str] = "qd-noul-v5-allowlist/v1"


def noul_v5_allowlist(
    *,
    g6_root: Path,
    g6_pins: Mapping[str, str],
    config: DataConfig,
    repo_root: Path,
) -> dict[str, Any]:
    """What v5's two Rust generators may read, decided by the canonical functions.

    * ``invisible_format_ranges``: ``qd_data.render``'s table, which both generators skip on
      and the loaders re-check against.
    * G6: each of :data:`G6_LANGUAGES`' ``<g6_root>/<lang>/data.jsonl`` is read only after its
      sha256 equals ``g6_pins[lang]`` (the hashes the download record pins,
      ``AUDIT/v5-plan-2026-10-02/g6-commitpackft-download.md``). A row is admitted when its
      licence passes ``admit_licence`` under ``config.licence`` -- the per-row filter every
      commitpackft row of the defect corpus passes -- and its primary repo
      (``CommitPackFtRow.primary_repo``: the first of ``repos``) hashes to ``train`` under
      ``assign_repo``. Admitted rows are listed by line; every row is counted before and
      after each filter, by language and licence.
    """
    from qd_train.data_access import assert_path_not_held_out

    if set(g6_pins) != set(G6_LANGUAGES):
        raise ValueError(f"pins name {sorted(g6_pins)}, not the G6 languages {G6_LANGUAGES}")
    files: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[list[Any]]] = {}
    counts: dict[str, dict[str, Any]] = {}
    for lang in G6_LANGUAGES:
        path = Path(g6_root) / lang / "data.jsonl"
        assert_path_not_held_out(path, config=config, repo_root=Path(repo_root))
        sha = _sha256_file(path)
        if sha != g6_pins[lang]:
            raise DefectCorpusError(
                f"{path} hashes to {sha}; the download record pins {g6_pins[lang]}: these are "
                "not the bytes the human approved"
            )
        before: Counter[str] = Counter()
        excluded: Counter[str] = Counter()
        admitted: list[list[Any]] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > config.max_row_bytes:
                    excluded["over_max_row_bytes"] += 1
                    continue
                obj = json.loads(line)
                licence = normalise_licence(str(obj.get("license", "")))
                before[licence] += 1
                try:
                    admit_licence(licence, config=config.licence, source=f"G6 {lang}:{lineno}")
                except LicenceRefused:
                    excluded[f"licence:{licence}"] += 1
                    continue
                repos = str(obj.get("repos", ""))
                primary = repos.split(",")[0].strip() or repos.strip()
                if not primary:
                    excluded["no_repo"] += 1
                    continue
                split_name = _split_of(primary, config)
                if split_name != "train":
                    excluded[f"split:{split_name}"] += 1
                    continue
                admitted.append([lineno, primary, licence])
        files[lang] = {"file": f"{lang}/data.jsonl", "sha256": sha, "rows": sum(before.values())}
        rows[lang] = admitted
        n_licence = sum(n for lic, n in before.items() if f"licence:{lic}" not in excluded)
        counts[lang] = {
            "rows": sum(before.values()),
            "by_licence_before": dict(sorted(before.items())),
            "after_licence": n_licence,
            "after_split": len(admitted),
            "excluded": dict(sorted(excluded.items())),
            "repos_after_split": len({r[1] for r in admitted}),
        }
    return {
        "schema": NOUL_V5_ALLOWLIST_SCHEMA,
        "split": {
            "seed": config.seed, "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
        },
        "invisible_format_ranges": [[lo, hi] for lo, hi in INVISIBLE_FORMAT_RANGES],
        "g6": {"root": str(g6_root), "files": files, "rows": rows, "counts": counts},
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
