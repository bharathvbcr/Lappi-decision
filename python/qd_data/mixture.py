"""The open-task-mixture rewriter: every surviving source into the one prompt format.

Three rules shape this module.

**One format.** Every family here produces a :class:`~qd_data.schema.Request` and
nothing else. There is no per-dataset prompt template, because
``docs/schema-api.md`` opens with the reason: *"the training format and the serving
format are the same object. If they drift, the model is served a prompt shape it
never saw, and nothing in the eval catches it."* A second template for one dataset
would be exactly that drift, introduced on purpose.

**Licence is checked twice, at two different granularities.** At load a dataset
whose licence is not on the permissive allowlist is refused with a message naming
the licence. Then, for a source that carries a per-row ``license``, every row is
checked again -- ``bigcode/commitpackft`` is ``mit`` as a dataset while three of its
thirteen declared per-row values are copyleft or absent
(``qd_data.licences`` has the finding). A dataset-level check alone would admit
AGPL code into an Apache-2.0 model's pool.

**A skipped row is counted, never dropped quietly.** Every rewriter raises
:class:`RowRefused` with a stable ``reason_code`` instead of returning ``None``, and
:func:`build_mixture` histograms the codes into the manifest. A source whose rows
are 90% refused has a thinner mixture than its headline row count, and that has to
be visible.

The free ``noul`` supervision, which is why these two datasets are load-bearing:
CLINC150's out-of-scope class becomes an abstention on ``intent.classification``,
and SQuAD 2.0's unanswerable questions become an abstention on ``qa.answer_span``.
``docs/schema-api.md``: *"``noul`` is a first-class member of every option set, not a
dump class."* These rows are where that first-class status is actually taught.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from qd_train.tristate import NotRun, Ran, TriState

from .config import DataConfig
from .errors import LicenceRefused, QdRefusal
from .licences import admit_licence, classify
from .loaders import ClincRow, CommitPackFtRow, RawRow, SourceUnavailableRefusal, SquadRow
from .render import DeterministicRng, first_invisible_format_char
from .rows import DataRow, GoldAnswer
from .schema import MAX_CHOICE_OPTIONS, ChoiceSlot, Request, ScoreSlot, SpanSlot
from .sources import source_by_id, task_family_by_id

__all__ = [
    "CHANGE_SCOPE_BIN_EDGES",
    "LANGUAGE_OPTIONS",
    "N_INTENT_OPTIONS",
    "MixtureResult",
    "RowRefused",
    "build_mixture",
    "rewrite_clinc",
    "rewrite_commitpackft",
    "rewrite_squad",
]


class RowRefused(QdRefusal):
    """This row cannot become an example. Counted by ``reason_code``, never silent."""

    check = "row_usable"
    #: No runtime counterpart: usability is a corpus question, not a wire one.
    rust_kinds = ()

    def __init__(self, *, reason_code: str, expected: object, actual: object,
                 detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(expected=expected, actual=actual, detail=f"[{reason_code}] {detail}")


#: Ordinal bins for ``code.change_scope``, in changed lines. Fixed edges rather than
#: quantiles of the loaded sample: quantile edges would make the label depend on
#: which rows happened to be pulled, so the same commit would carry a different
#: ordinal in two runs and ``data_snapshot_hash`` would move for an unchanged corpus.
CHANGE_SCOPE_BIN_EDGES: Final[tuple[int, ...]] = (1, 5, 20, 100)

#: The closed language set for ``code.language_id``. Closed, and at most 16, because
#: the generic route decodes one token over a 16-row slice -- an open label set
#: cannot be asked as a ``choice`` question at all. A row in a language outside the
#: set is refused with ``lang_not_in_option_set`` and counted.
LANGUAGE_OPTIONS: Final[tuple[str, ...]] = (
    "C", "C++", "C#", "Go", "Java", "JavaScript", "TypeScript", "Python",
    "Ruby", "Rust", "PHP", "Shell", "Swift", "Kotlin", "Markdown", "YAML",
)

#: ``intent.classification`` asks over a sampled 16-way option set: CLINC150 has 150
#: intents and the letter slice holds 16.
N_INTENT_OPTIONS: Final[int] = MAX_CHOICE_OPTIONS

_YES_NO: Final[tuple[str, str]] = ("yes", "no")


def _bin_of(value: int, edges: tuple[int, ...]) -> int:
    """1-based ordinal bin. ``len(edges) + 1`` bins."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i + 1
    return len(edges) + 1


def _changed_lines(old: str, new: str) -> int:
    """Lines present in exactly one side, counted on ``\\n`` only.

    Not ``str.splitlines()``: that also splits on U+2028, U+0085 and friends, so two
    counts of "how many lines" would disagree on a file containing one. The renderer
    escapes those characters for the same reason -- see ``qd_data.render``.
    """
    old_lines = Counter(old.split("\n"))
    new_lines = Counter(new.split("\n"))
    removed = old_lines - new_lines
    added = new_lines - old_lines
    return sum(removed.values()) + sum(added.values())


def _line_span(text: str, start_offset: int, length: int) -> tuple[int, int]:
    """1-based inclusive line range covering ``[start, start+length)``.

    Counted on ``\\n`` only, consistently with :func:`_changed_lines`. An off-by-one
    here is ``docs/hardening.md`` section 1's exact failure: it teaches the pointer
    head to point one line off, systematically, and is invisible in class accuracy.
    """
    if start_offset < 0 or length < 0 or start_offset + length > len(text):
        raise RowRefused(
            reason_code="answer_span_out_of_bounds",
            expected=f"a span inside a {len(text)}-character passage",
            actual=f"[{start_offset}, {start_offset + length})",
        )
    start_line = text.count("\n", 0, start_offset) + 1
    end_line = text.count("\n", 0, max(start_offset, start_offset + length - 1)) + 1
    return start_line, end_line


def _refuse_invisible_format(text: str, *, field: str) -> None:
    """Refuse text that renders differently from how it tokenises.

    The Trojan Source class (``GAP-DATA-RENDER-BIDI-UNESCAPED``). A bidi override or
    a zero-width character makes a commit message, an option or a diff *display* as
    one thing to a reviewer while being another to the tokenizer. Every structural
    check in this lane -- option count, marker count, the letter map, the line count
    -- passes on such a row, which is exactly why it needs its own refusal: the
    corpus review gate is the control it defeats, and that gate is a human reading
    rendered text.

    Refused here rather than escaped in :mod:`qd_data.render` because the escape
    alphabet is a two-language contract with ``crates/qd-runtime/src/render.rs``;
    see the note beside :data:`~qd_data.render.INVISIBLE_FORMAT_CHARS`.
    """
    ch = first_invisible_format_char(text)
    if ch is None:
        return
    raise RowRefused(
        reason_code="invisible_format_characters",
        expected=f"a {field} with no Unicode format (Cf) characters",
        actual=f"U+{ord(ch):04X} {unicodedata.name(ch, 'unnamed codepoint')} in {field}",
        detail=(
            "this character contributes no glyph and changes how its neighbours are "
            "displayed, so the row would read to a reviewer as something other than "
            "what the model is trained on"
        ),
    )


def _request(
    *,
    family_id: str,
    context: str,
    slot_name: str,
    slot: ChoiceSlot | ScoreSlot | SpanSlot,
    example_id: str,
) -> Request:
    family = task_family_by_id(family_id)
    if slot.name != slot_name:
        raise ValueError(f"slot name mismatch: {slot.name!r} vs {slot_name!r}")
    # The single funnel every family's request passes through, so the invisible-
    # character check has one owner rather than one copy per rewriter.
    _refuse_invisible_format(context, field="context")
    if isinstance(slot, ChoiceSlot):
        for option in slot.options:
            _refuse_invisible_format(option, field=f"option {option!r}")
    return Request(
        task=family_id,
        context=context.encode("utf-8"),
        question=family.description,
        slots=(slot,),
        route="generic",
        example_id=example_id,
    )


def _row(
    *,
    row_id: str,
    source_id: str,
    family_id: str,
    repo_key: str,
    identity_key: str,
    licence_id: str,
    request: Request,
    gold: GoldAnswer,
    dedupe_text: str,
    metadata: dict[str, str] | None = None,
) -> DataRow:
    # The split keys are not rendered into a prompt, but they *are* written into the
    # manifest and read by a human auditing a snapshot -- and `canonical_json` uses
    # `ensure_ascii=False`, so a format character in a repo name reaches that file
    # raw. Same class, same refusal, second and last funnel.
    _refuse_invisible_format(repo_key, field="repo_key")
    _refuse_invisible_format(identity_key, field="identity_key")
    policy = classify(licence_id)
    source = source_by_id(source_id)
    return DataRow(
        row_id=row_id,
        source_id=source_id,
        host=source.host,
        family_id=family_id,
        repo_key=repo_key,
        identity_key=identity_key,
        licence_id=policy.licence_id,
        obligations=policy.obligations,
        request=request,
        gold=(gold,),
        dedupe_text=dedupe_text,
        metadata=metadata or {},
    )


# -- bigcode/commitpackft ----------------------------------------------------


def rewrite_commitpackft(
    raw: CommitPackFtRow,
    *,
    family_id: str,
    index: int,
    config: DataConfig,
    decoy_message: str | None = None,
) -> DataRow:
    """One commitpackft row into one of its three families.

    ``code.commit_intent`` needs a negative, and the only honest one available from
    this corpus is another row's commit message: ``decoy_message``. A rewriter that
    invented a negative (say, by perturbing the real message) would teach the model
    to detect the perturbation rather than to read the diff.
    """
    admit_licence(raw.licence, config=config.licence, source=f"bigcode/commitpackft row {index}")

    repo = raw.primary_repo
    if not repo.strip():
        raise RowRefused(
            reason_code="missing_repo",
            expected="a repos field naming at least one repository",
            actual=raw.repos,
            detail="a row with no split unit cannot be split at the repo level",
        )
    identity = f"{repo}::{raw.new_file}"
    # No enumeration index in the id: a pull at a different offset would renumber
    # every row, and although `row_id` is excluded from the content hash, an id that
    # moves with the read order makes two manifests over one corpus needlessly
    # un-diffable. (commit, path) is unique upstream.
    row_id = (
        f"cpft:{family_id}:{raw.commit}:"
        f"{hashlib.blake2b(identity.encode('utf-8'), digest_size=6).hexdigest()}"
    )
    dedupe_text = raw.new_contents
    if not dedupe_text.strip():
        raise RowRefused(
            reason_code="empty_new_contents",
            expected="non-empty new_contents", actual=len(raw.new_contents),
        )

    if family_id == "code.commit_intent":
        if decoy_message is None:
            raise RowRefused(
                reason_code="no_decoy_available",
                expected="a decoy commit message from another row",
                actual=None,
                detail="a yes/no intent question with no negative is a constant label",
            )
        # Keyed on the row's own upstream identity, never on its position in the
        # read: a label that flips when the corpus is iterated backwards is not a
        # label, and it moves `data_snapshot_hash` for an unchanged corpus.
        use_decoy = (
            DeterministicRng(
                "qd_data.mixture.intent.v1", config.seed, raw.commit, identity
            ).below(2)
            == 1
        )
        claimed = decoy_message if use_decoy else raw.message
        if use_decoy and claimed.strip() == raw.message.strip():
            raise RowRefused(
                reason_code="decoy_equals_true_message",
                expected="a decoy different from the true message", actual=claimed[:64],
                detail="an identical decoy would be labelled 'no' while being correct",
            )
        context = (
            f"commit message:\n{claimed}\n\n"
            f"file: {raw.new_file}\n\n"
            f"--- before ---\n{raw.old_contents}\n"
            f"--- after ---\n{raw.new_contents}\n"
        )
        return _row(
            row_id=row_id, source_id="bigcode/commitpackft", family_id=family_id,
            repo_key=repo, identity_key=identity, licence_id=raw.licence,
            request=_request(
                family_id=family_id, context=context, slot_name="implements_claim",
                slot=ChoiceSlot(name="implements_claim", options=_YES_NO),
                example_id=row_id,
            ),
            gold=GoldAnswer(slot_name="implements_claim", value="no" if use_decoy else "yes"),
            dedupe_text=dedupe_text,
            metadata={"decoy": str(use_decoy)},
        )

    if family_id == "code.language_id":
        if raw.lang not in LANGUAGE_OPTIONS:
            raise RowRefused(
                reason_code="lang_not_in_option_set",
                expected=f"one of {list(LANGUAGE_OPTIONS)}", actual=raw.lang,
                detail="the generic route decodes over 16 letters; the label set is closed",
            )
        context = f"file: {raw.new_file}\n\n{raw.new_contents}\n"
        return _row(
            row_id=row_id, source_id="bigcode/commitpackft", family_id=family_id,
            repo_key=repo, identity_key=identity, licence_id=raw.licence,
            request=_request(
                family_id=family_id, context=context, slot_name="language",
                slot=ChoiceSlot(name="language", options=LANGUAGE_OPTIONS),
                example_id=row_id,
            ),
            gold=GoldAnswer(slot_name="language", value=raw.lang),
            dedupe_text=dedupe_text,
        )

    if family_id == "code.change_scope":
        n = _changed_lines(raw.old_contents, raw.new_contents)
        if n == 0:
            raise RowRefused(
                reason_code="no_changed_lines",
                expected="a diff that changes at least one line", actual=0,
                detail="a no-op change has no scope to score",
            )
        bins = len(CHANGE_SCOPE_BIN_EDGES) + 1
        context = (
            f"file: {raw.new_file}\n\n"
            f"--- before ---\n{raw.old_contents}\n"
            f"--- after ---\n{raw.new_contents}\n"
        )
        return _row(
            row_id=row_id, source_id="bigcode/commitpackft", family_id=family_id,
            repo_key=repo, identity_key=identity, licence_id=raw.licence,
            request=_request(
                family_id=family_id, context=context, slot_name="scope",
                slot=ScoreSlot(name="scope", bins=bins), example_id=row_id,
            ),
            gold=GoldAnswer(
                slot_name="scope", value=_bin_of(n, CHANGE_SCOPE_BIN_EDGES)
            ),
            dedupe_text=dedupe_text,
            metadata={"changed_lines": str(n)},
        )

    raise RowRefused(
        reason_code="unknown_family_for_source",
        expected="one of code.commit_intent, code.language_id, code.change_scope",
        actual=family_id, detail="bigcode/commitpackft",
    )


# -- clinc/clinc_oos ---------------------------------------------------------


def rewrite_clinc(
    raw: ClincRow,
    *,
    family_id: str,
    index: int,
    config: DataConfig,
    intent_vocabulary: Sequence[str],
) -> DataRow:
    """One CLINC150 utterance. Out-of-scope becomes an abstention, not a class."""
    source = source_by_id("clinc/clinc_oos")
    admit_licence(source.declared_licence, config=config.licence, source=source.source_id)

    utterance = raw.utterance.strip()
    if not utterance:
        raise RowRefused(
            reason_code="empty_utterance", expected="a non-empty utterance", actual="",
        )
    # CLINC has no repository. The split unit is the intent label, so every
    # utterance of one intent moves together: a row-level split over 150 intents with
    # ~100 near-paraphrases each would put paraphrases on both sides.
    repo_key = f"clinc-intent:{raw.intent}"
    # Identity is a digest of the utterance, not the enumeration index. CLINC rows
    # carry no id, and keying identity on position means a pull at a different offset
    # renames every row -- which moves `data_snapshot_hash` for a corpus that did not
    # change. The index survives only in `row_id`, which the content hash excludes.
    digest = hashlib.blake2b(utterance.encode("utf-8"), digest_size=8).hexdigest()
    identity = f"{repo_key}::{digest}"
    row_id = f"clinc:{family_id}:{digest}:{index}"

    if family_id == "intent.classification":
        vocab = [i for i in intent_vocabulary if i != raw.intent]
        if len(vocab) < N_INTENT_OPTIONS - 1:
            raise RowRefused(
                reason_code="intent_vocabulary_too_small",
                expected=f"at least {N_INTENT_OPTIONS - 1} distractor intents",
                actual=len(vocab),
                detail=(
                    "a question with fewer options than the letter slice is a "
                    "different question"
                ),
            )
        rng = DeterministicRng("qd_data.mixture.clinc.v1", config.seed, row_id)
        perm = rng.permutation(len(vocab))
        if raw.is_oos:
            # No gold class: the option set is 16 real intents and the answer is noul.
            chosen = sorted(vocab[i] for i in perm[:N_INTENT_OPTIONS])
            gold = GoldAnswer(slot_name="intent", value=None, is_noul=True)
        else:
            chosen = sorted(
                [raw.intent, *(vocab[i] for i in perm[: N_INTENT_OPTIONS - 1])]
            )
            gold = GoldAnswer(slot_name="intent", value=raw.intent)
        return _row(
            row_id=row_id, source_id=source.source_id, family_id=family_id,
            repo_key=repo_key, identity_key=identity, licence_id=source.declared_licence,
            request=_request(
                family_id=family_id, context=utterance, slot_name="intent",
                slot=ChoiceSlot(name="intent", options=tuple(chosen)), example_id=row_id,
            ),
            gold=gold, dedupe_text=utterance,
            metadata={"oos": str(raw.is_oos)},
        )

    if family_id == "intent.in_scope":
        return _row(
            row_id=row_id, source_id=source.source_id, family_id=family_id,
            repo_key=repo_key, identity_key=identity, licence_id=source.declared_licence,
            request=_request(
                family_id=family_id, context=utterance, slot_name="in_scope",
                slot=ChoiceSlot(name="in_scope", options=_YES_NO), example_id=row_id,
            ),
            gold=GoldAnswer(slot_name="in_scope", value="no" if raw.is_oos else "yes"),
            dedupe_text=utterance,
            metadata={"oos": str(raw.is_oos)},
        )

    raise RowRefused(
        reason_code="unknown_family_for_source",
        expected="one of intent.classification, intent.in_scope", actual=family_id,
        detail="clinc/clinc_oos",
    )


# -- rajpurkar/squad_v2 ------------------------------------------------------


def rewrite_squad(
    raw: SquadRow, *, family_id: str, index: int, config: DataConfig
) -> DataRow:
    """One SQuAD 2.0 question. Unanswerable becomes an abstention on the span slot."""
    source = source_by_id("rajpurkar/squad_v2")
    admit_licence(source.declared_licence, config=config.licence, source=source.source_id)

    passage = raw.context
    if not passage.strip():
        raise RowRefused(reason_code="empty_passage", expected="a non-empty passage", actual="")
    row_id = f"squad:{family_id}:{raw.qid}"
    # The split unit is the article title: SQuAD draws many questions from one
    # article, so a row-level split would put questions about the same paragraph on
    # both sides of the boundary.
    repo_key = f"squad-title:{raw.title}"
    identity = f"{repo_key}::{raw.qid}"
    context = f"{raw.question}\n\n{passage}"
    # The question is part of the compared text: several SQuAD questions share one
    # passage, and comparing passages alone would make them one content unit, so one
    # question would stand in for all of them.
    dedupe_text = f"{raw.question}\n{passage}"

    if family_id == "qa.answerability":
        return _row(
            row_id=row_id, source_id=source.source_id, family_id=family_id,
            repo_key=repo_key, identity_key=identity, licence_id=source.declared_licence,
            request=_request(
                family_id=family_id, context=context, slot_name="answerable",
                slot=ChoiceSlot(name="answerable", options=_YES_NO), example_id=row_id,
            ),
            gold=GoldAnswer(
                slot_name="answerable", value="no" if raw.is_impossible else "yes"
            ),
            dedupe_text=dedupe_text,
            metadata={"impossible": str(raw.is_impossible)},
        )

    if family_id == "qa.answer_span":
        if raw.is_impossible:
            gold = GoldAnswer(slot_name="evidence", value=None, is_noul=True)
        else:
            # The span is over the rendered context, which is the question, a blank
            # line, then the passage. The offset is derived from that construction
            # rather than by searching for the passage: a passage that also occurred
            # inside the question would make a search land on the wrong occurrence.
            prefix_lines = raw.question.count("\n") + 2
            start, end = _line_span(passage, raw.answer_starts[0], len(raw.answers[0]))
            gold = GoldAnswer(
                slot_name="evidence", value=(start + prefix_lines, end + prefix_lines)
            )
        return _row(
            row_id=row_id, source_id=source.source_id, family_id=family_id,
            repo_key=repo_key, identity_key=identity, licence_id=source.declared_licence,
            request=_request(
                family_id=family_id, context=context, slot_name="evidence",
                slot=SpanSlot(name="evidence"), example_id=row_id,
            ),
            gold=gold, dedupe_text=dedupe_text,
            metadata={"impossible": str(raw.is_impossible)},
        )

    raise RowRefused(
        reason_code="unknown_family_for_source",
        expected="one of qa.answerability, qa.answer_span", actual=family_id,
        detail="rajpurkar/squad_v2",
    )


# -- the builder -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureResult:
    rows: tuple[DataRow, ...]
    #: ``source_id -> {reason_code: count}``. A source whose rows were mostly refused
    #: has a thinner mixture than its headline count, and this is where that shows.
    refusals: dict[str, dict[str, int]]
    #: ``source_id -> rows read from the source before rewriting``.
    n_input: dict[str, int]
    #: ``licence_id -> row count``, and the obligations each carries.
    licence_histogram: dict[str, int]
    obligations: dict[str, tuple[str, ...]]
    #: ``host -> row count``. Recorded for the model card.
    host_histogram: dict[str, int]
    #: Sources refused outright at load, with the reason naming the licence.
    refused_sources: dict[str, tuple[str, ...]]
    #: ``family_id -> TriState``, one entry per **requested** family.
    #:
    #: ``refusals`` above is keyed by source, and ``n_input`` counts the rows read
    #: from a source once -- but :func:`build_mixture` iterates those rows once per
    #: family of that source, so a commitpackft pull makes ``3 * n_input`` attempts.
    #: Dividing a reason count by ``n_input`` therefore answers a question nobody
    #: asked, and ``GAP-DATA-COMMITPACKFT-LANG-SET-UNVERIFIED`` proposed exactly that
    #: division as its workaround. This carries the honest pair: examples built and
    #: rows attempted, per family, with the refusal codes that account for the gap.
    #:
    #: A family every one of whose attempts was refused is ``NotRun``, never
    #: ``Ran(passed=False)``: it has no examples, so nothing about it was built and
    #: nothing about it can be checked -- and the training door
    #: (``qd_train.data_access.open_training_data``) branches on ``NotRun``.
    family_coverage: dict[str, TriState]
    status: TriState

    def to_json(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.rows),
            "n_input": dict(sorted(self.n_input.items())),
            "refusals": {k: dict(sorted(v.items())) for k, v in sorted(self.refusals.items())},
            "licence_histogram": dict(sorted(self.licence_histogram.items())),
            "obligations": {k: list(v) for k, v in sorted(self.obligations.items())},
            "host_histogram": dict(sorted(self.host_histogram.items())),
            "refused_sources": {k: list(v) for k, v in sorted(self.refused_sources.items())},
            "family_coverage": {
                k: v.to_json() for k, v in sorted(self.family_coverage.items())
            },
            "status": self.status.to_json(),
        }


def build_mixture(
    raw_by_source: Mapping[str, Sequence[RawRow]],
    *,
    config: DataConfig,
    families: Sequence[str] | None = None,
    capped_sources: Sequence[str] = (),
) -> MixtureResult:
    """Rewrite every admitted source into the one prompt format.

    ``families`` defaults to every registered family of every admitted source,
    **including the held-out ones** -- the holdout is applied by the splitter, not by
    omitting the rows here, because a family that is never built cannot be checked
    for abstention.

    ``capped_sources`` names sources whose read hit a bound. Any entry makes the
    result's status ``NotRun``: a capped sample must never be reported as complete
    coverage.
    """
    from .sources import TASK_FAMILIES  # local: avoids a cycle at import time

    wanted = (
        tuple(families)
        if families is not None
        else tuple(
            f.family_id
            for f in TASK_FAMILIES.values()
            if f.source_id in raw_by_source
        )
    )
    for fam in wanted:
        task_family_by_id(fam)  # raises on an unknown family id

    rows: list[DataRow] = []
    refusals: dict[str, dict[str, int]] = {}
    n_input: dict[str, int] = {}
    refused_sources: dict[str, tuple[str, ...]] = {}
    family_coverage: dict[str, TriState] = {}

    for source_id in sorted(raw_by_source):
        source = source_by_id(source_id)
        reasons = source.admission_refusals(config.licence)
        if reasons:
            refused_sources[source_id] = reasons
            policy = source.licence_policy
            if any("licence" in r for r in reasons):
                raise LicenceRefused(
                    expected="a licence on the permissive allowlist",
                    actual=policy.licence_id,
                    detail=(
                        f"{source_id}: refused at load. licence {policy.licence_id!r} -- "
                        + "; ".join(reasons)
                    ),
                )
            raise SourceUnavailableRefusal(
                expected="a source reachable unattended",
                actual=source.reachability.value,
                detail=f"{source_id}: refused at load -- " + "; ".join(reasons),
            )

        raws = list(raw_by_source[source_id])
        n_input[source_id] = len(raws)
        counts: Counter[str] = Counter()
        source_families = [f for f in wanted if task_family_by_id(f).source_id == source_id]

        intent_vocab = sorted(
            {r.intent for r in raws if isinstance(r, ClincRow) and not r.is_oos}
        )
        messages = [r.message for r in raws if isinstance(r, CommitPackFtRow)]

        for family_id in source_families:
            n_before = len(rows)
            family_counts: Counter[str] = Counter()
            for i, raw in enumerate(raws):
                try:
                    rows.append(
                        _dispatch(
                            raw,
                            family_id=family_id,
                            index=i,
                            config=config,
                            intent_vocabulary=intent_vocab,
                            messages=messages,
                        )
                    )
                except RowRefused as exc:
                    family_counts[exc.reason_code] += 1
                except LicenceRefused as exc:
                    family_counts[f"licence:{exc.actual}"] += 1
            counts.update(family_counts)
            # A family belongs to exactly one source, so this never collides.
            family_coverage[family_id] = _family_coverage(
                family_id=family_id,
                source_id=source_id,
                n_built=len(rows) - n_before,
                n_attempted=len(raws),
                reasons=dict(family_counts),
            )
        refusals[source_id] = dict(counts)

    licence_hist: Counter[str] = Counter()
    host_hist: Counter[str] = Counter()
    obligations: dict[str, tuple[str, ...]] = {}
    for row in rows:
        licence_hist[row.licence_id] += 1
        host_hist[row.host] += 1
        obligations[row.licence_id] = row.obligations

    capped = tuple(sorted(set(capped_sources)))
    uncovered = sorted(f for f, s in family_coverage.items() if isinstance(s, NotRun))
    if capped:
        status: TriState = NotRun(
            reason=(
                f"the read of {list(capped)} hit its row bound, so this mixture is a capped "
                "sample of those sources, not their full contents"
            )
        )
    elif not rows:
        status = NotRun(
            reason="no rows survived rewriting, so no mixture was built and nothing was verified"
        )
    elif uncovered:
        # The defect GAP-DATA-COMMITPACKFT-LANG-SET-UNVERIFIED was a proxy for. A
        # family every one of whose rows was refused produces no examples, yet the
        # mixture used to report Ran(passed=True) and the thinning showed up as
        # nothing worse than a slightly different mixture. For a held-out family
        # that is worse than a thin mixture: the abstention gate is then measured on
        # nothing while every downstream check passes vacuously, because a family
        # with no rows anywhere is trivially absent from every training shard.
        status = NotRun(
            reason=(
                f"{len(uncovered)} requested famil{'y' if len(uncovered) == 1 else 'ies'} "
                f"produced no examples at all: "
                + "; ".join(
                    f"{f} -> {family_coverage[f].reason}"  # type: ignore[union-attr]
                    for f in uncovered
                )
            )
        )
    else:
        total_refused = sum(sum(c.values()) for c in refusals.values())
        status = Ran(
            passed=True,
            value=len(rows),
            n=len(rows),
            n_total=len(rows) + total_refused,
            detail=(
                f"{len(rows)} examples built, {total_refused} rows refused across "
                f"{len(refusals)} sources; per-family coverage: "
                + ", ".join(
                    f"{f}={s.coverage_str()}"  # type: ignore[union-attr]
                    for f, s in sorted(family_coverage.items())
                )
            ),
        )

    return MixtureResult(
        rows=tuple(rows),
        refusals=refusals,
        n_input=n_input,
        licence_histogram=dict(licence_hist),
        obligations=obligations,
        host_histogram=dict(host_hist),
        refused_sources=refused_sources,
        family_coverage=family_coverage,
        status=status,
    )


def _family_coverage(
    *,
    family_id: str,
    source_id: str,
    n_built: int,
    n_attempted: int,
    reasons: dict[str, int],
) -> TriState:
    """One family's built/attempted pair, or ``NotRun`` when it built nothing."""
    top = ", ".join(
        f"{code}={n}" for code, n in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    if n_attempted == 0:
        return NotRun(
            reason=(
                f"{family_id}: no rows were read from {source_id}, so the family was "
                "never built and nothing about it was verified"
            )
        )
    if n_built == 0:
        return NotRun(
            reason=(
                f"{family_id}: all {n_attempted} rows read from {source_id} were refused "
                f"({top}), so the family has no examples -- nothing about it was built "
                "and nothing about it can be checked"
            )
        )
    return Ran(
        passed=True,
        value=n_built,
        n=n_built,
        n_total=n_attempted,
        detail=(
            f"{family_id}: {n_built} of {n_attempted} {source_id} rows became examples"
            + (f"; refused: {top}" if reasons else "")
        ),
    )


def _dispatch(
    raw: RawRow,
    *,
    family_id: str,
    index: int,
    config: DataConfig,
    intent_vocabulary: Sequence[str],
    messages: Sequence[str],
) -> DataRow:
    if isinstance(raw, CommitPackFtRow):
        decoy: str | None = None
        if messages:
            # A deterministic decoy from another row, never this row's own message.
            # Drawn from the *sorted set* and keyed on this row's own commit, so the
            # decoy -- and therefore the rendered context, and therefore the content
            # hash -- does not depend on the order the rows arrived in.
            others = sorted({m for m in messages if m.strip() != raw.message.strip()})
            if others:
                pick = DeterministicRng(
                    "qd_data.mixture.decoy.v1", config.seed, raw.commit
                ).below(len(others))
                decoy = others[pick]
        return rewrite_commitpackft(
            raw, family_id=family_id, index=index, config=config, decoy_message=decoy
        )
    if isinstance(raw, ClincRow):
        return rewrite_clinc(
            raw, family_id=family_id, index=index, config=config,
            intent_vocabulary=intent_vocabulary,
        )
    if isinstance(raw, SquadRow):
        return rewrite_squad(raw, family_id=family_id, index=index, config=config)
    raise RowRefused(
        reason_code="unknown_raw_row_type",
        expected="CommitPackFtRow | ClincRow | SquadRow", actual=type(raw).__name__,
    )
