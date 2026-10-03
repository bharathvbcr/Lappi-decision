"""The open-task-mixture rewriter: every surviving source into the one prompt format.

Four rules shape this module.

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

**The corpus must not contradict itself.** Every rule above is about one row. A
corpus can be built entirely of individually valid rows and still be unlearnable,
because two of them ask the model the same question and demand different answers.
``GAP-DATA-NOTHING-REFUSES-TWO-ROWS-THAT-CONTRADICT``: a SQuAD pair differing only
in ``is_impossible`` used to render byte-identical prompts, one labelled with a line
span and one with ``noul``. A causal model conditions on the prompt and nothing
else, so it must split its mass -- the FT lane measured the span channel converging
on 0.693147, which is ln 2, the entropy of a fair coin, with accuracy capped at 39
of 78 permanently. Dedupe *saw* the pair (``dedupe_text`` was byte-identical) and
kept it correctly, because leakage and contradiction are different questions and
dedupe asks only the first. :func:`check_prompt_consistency` asks the second, over
the rendered prompt, and :func:`build_mixture` removes every row of a prompt that
carries two golds (:func:`drop_contradictory_prompts`), counted as
``contradictory_prompt`` -- no winner is chosen. It used to refuse the whole corpus
instead; on 2026-09-29 the approved SQuAD v2 and CLINC caches were found to carry
33 such groups among ~190k rows, upstream duplicates rather than a generator
defect, and refusing 315k rows over them is the wrong remedy. ``qd_train.shards``
still refuses any contradiction that reaches the write boundary.

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
from typing import TYPE_CHECKING, Any, Final

from qd_train.tristate import NotRun, Ran, TriState

from .config import DataConfig
from .defect_class import (
    CHOICE_SLOT,
    CONTEXT_HEADER_LINES,
    DEFECT_CLASSES,
    DEFECT_FAMILY_ID,
    DEFECT_SOURCE_ID,
    NOUL_CLASS,
    NOUL_FORM_KEY,
    NOUL_ROUTE_KEY,
    SPAN_SLOT,
    DefectRow,
)
from .errors import LicenceRefused, QdRefusal
from .licences import admit_licence, classify
from .loaders import (
    ClincRow,
    CommitPackFtRow,
    CsqaRow,
    MmluRow,
    RawRow,
    SourceUnavailableRefusal,
    SquadRow,
    TypedDecisionRow,
)
from .render import (
    DEFAULT_CAPS,
    DeterministicRng,
    RenderCaps,
    first_invisible_format_char,
    render,
)
from .rows import DataRow, GoldAnswer
from .schema import (
    MAX_CHOICE_OPTIONS,
    ChoiceSlot,
    Request,
    ScoreSlot,
    SpanSlot,
    canonical_json,
)
from .sources import PINNED_SPLIT_KEY, source_by_id, task_family_by_id
from .split import SQUAD_TITLE_FAMILIES, squad_title_family, squad_title_repo_key

if TYPE_CHECKING:
    # qd_data.general imports this module's `_request`/`_row` funnel, so its runtime
    # import stays local to the functions that use it.
    from .general import ClincDomainMap

__all__ = [
    "CHANGE_SCOPE_BIN_EDGES",
    "CONTRADICTORY_PROMPT",
    "DEFAULT_MAX_CONSISTENCY_ROWS",
    "LANGUAGE_OPTIONS",
    "MAX_NAMED_CONTRADICTIONS",
    "MAX_NAMED_GOLDS",
    "MAX_NAMED_ROW_IDS",
    "N_INTENT_OPTIONS",
    "ClincKeys",
    "ConsistencyDrop",
    "MixtureResult",
    "PromptContradiction",
    "RowRefused",
    "abstention_supply",
    "build_mixture",
    "check_prompt_consistency",
    "clinc_keys",
    "drop_contradictory_prompts",
    "rewrite_clinc",
    "rewrite_commitpackft",
    "rewrite_defect_class",
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
    slots: tuple[ChoiceSlot | ScoreSlot | SpanSlot, ...],
    example_id: str,
) -> Request:
    """A tuple of slots, so a family asking two questions of one context -- the class and
    the lines, for ``code.defect_class`` -- is one request through this one funnel rather
    than a second builder beside it. That every gold names one of these slots is checked
    by ``DataRow`` itself."""
    family = task_family_by_id(family_id)
    # The single funnel every family's request passes through, so the invisible-
    # character check has one owner rather than one copy per rewriter.
    _refuse_invisible_format(context, field="context")
    for slot in slots:
        if isinstance(slot, ChoiceSlot):
            for option in slot.options:
                _refuse_invisible_format(option, field=f"option {option!r}")
    return Request(
        task=family_id,
        context=context.encode("utf-8"),
        question=family.description,
        slots=tuple(slots),
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
    gold: tuple[GoldAnswer, ...],
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
        gold=tuple(gold),
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
                family_id=family_id, context=context,
                slots=(ChoiceSlot(name="implements_claim", options=_YES_NO),),
                example_id=row_id,
            ),
            gold=(
                GoldAnswer(slot_name="implements_claim", value="no" if use_decoy else "yes"),
            ),
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
                family_id=family_id, context=context,
                slots=(ChoiceSlot(name="language", options=LANGUAGE_OPTIONS),),
                example_id=row_id,
            ),
            gold=(GoldAnswer(slot_name="language", value=raw.lang),),
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
                family_id=family_id, context=context,
                slots=(ScoreSlot(name="scope", bins=bins),), example_id=row_id,
            ),
            gold=(
                GoldAnswer(slot_name="scope", value=_bin_of(n, CHANGE_SCOPE_BIN_EDGES)),
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


@dataclass(frozen=True, slots=True)
class ClincKeys:
    """Where one CLINC utterance splits, what it is, and the digest both are built from."""

    repo_key: str
    identity_key: str
    digest: str


def clinc_keys(raw: ClincRow, utterance: str) -> ClincKeys:
    """The split unit and identity of one CLINC utterance, for every CLINC family.

    The one owner of both keys: :func:`rewrite_clinc` and
    ``qd_data.general.rewrite_clinc_two_stage`` call this, so one utterance lands in one
    repo split whichever CLINC family asks about it.

    **In-scope rows split by intent.** CLINC has no repository, so the split unit is the
    intent: every utterance of one intent moves together, because a row-level split over
    150 intents with ~100 near-paraphrases each would put paraphrases on both sides.

    **Out-of-scope rows split by utterance** (v5; human-decisions.md item 2,
    GAP-CLINC-OOS-ONE-REPO-KEY-ONE-SPLIT-2026-10-02). Keyed by intent, all 1,350 oos
    utterances shared ``clinc-intent:oos``, which the hash put in train, so no val or
    held-out row of any CLINC family was ever an abstention. oos has no intent whose
    paraphrases could straddle a boundary; its utterances are unrelated requests, so each
    is its own unit: ``clinc-oos:<blake2b-8 of the stripped utterance>``. MinHash dedupe
    still runs across them before the split.

    **Identity does not move**: ``clinc-intent:<intent>::<digest>`` for every row, oos
    included, exactly as v4 spelled it. Identity is a digest of the utterance, not the
    enumeration index -- CLINC rows carry no id, and keying identity on position means a
    pull at a different offset renames every row, which moves ``data_snapshot_hash`` for a
    corpus that did not change. Keeping it stable across the re-key means a v4 identity
    names the same utterance in a v5 build.
    """
    if raw.is_oos != (raw.intent == "oos"):
        raise RowRefused(
            reason_code="oos_flag_disagrees_with_intent",
            expected="is_oos exactly when the intent is 'oos'",
            actual=f"intent={raw.intent!r} is_oos={raw.is_oos}",
            detail="the split unit follows the flag; a row whose two answers disagree has none",
        )
    digest = hashlib.blake2b(utterance.encode("utf-8"), digest_size=8).hexdigest()
    intent_key = f"clinc-intent:{raw.intent}"
    repo_key = f"clinc-oos:{digest}" if raw.is_oos else intent_key
    return ClincKeys(repo_key=repo_key, identity_key=f"{intent_key}::{digest}", digest=digest)


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
    # Split unit and identity: see `clinc_keys`. The index survives only in `row_id`,
    # which the content hash excludes.
    keys = clinc_keys(raw, utterance)
    repo_key, identity, digest = keys.repo_key, keys.identity_key, keys.digest
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
                family_id=family_id, context=utterance,
                slots=(ChoiceSlot(name="intent", options=tuple(chosen)),), example_id=row_id,
            ),
            gold=(gold,), dedupe_text=utterance,
            metadata={"oos": str(raw.is_oos)},
        )

    if family_id == "intent.in_scope":
        return _row(
            row_id=row_id, source_id=source.source_id, family_id=family_id,
            repo_key=repo_key, identity_key=identity, licence_id=source.declared_licence,
            request=_request(
                family_id=family_id, context=utterance,
                slots=(ChoiceSlot(name="in_scope", options=_YES_NO),), example_id=row_id,
            ),
            gold=(GoldAnswer(slot_name="in_scope", value="no" if raw.is_oos else "yes"),),
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
    if family_id in SQUAD_TITLE_FAMILIES:
        if not raw.title.strip():
            raise RowRefused(
                reason_code="empty_title", expected="a SQuAD article title", actual=raw.title,
                detail="the title is the partition unit between the two SQuAD families",
            )
        # User decision 2026-09-29 (GAP-DATA-SQUAD-SPAN-NOUL-LEAKS-HELD-OUT-ANSWERABILITY,
        # option a): each SQuAD article feeds exactly one of the two families, so the held-out
        # answerability rows' contexts never appear in span training with a noul/span gold.
        owner = squad_title_family(raw.title, seed=config.seed)
        if owner != family_id:
            raise RowRefused(
                reason_code="squad_title_reserved_for_other_family",
                expected=owner, actual=family_id,
                detail=f"title {raw.title!r} belongs to {owner}",
            )
    row_id = f"squad:{family_id}:{raw.qid}"
    # The split unit is the article title: SQuAD draws many questions from one
    # article, so a row-level split would put questions about the same paragraph on
    # both sides of the boundary.
    repo_key = squad_title_repo_key(raw.title)
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
                family_id=family_id, context=context,
                slots=(ChoiceSlot(name="answerable", options=_YES_NO),), example_id=row_id,
            ),
            gold=(
                GoldAnswer(slot_name="answerable", value="no" if raw.is_impossible else "yes"),
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
                family_id=family_id, context=context,
                slots=(SpanSlot(name="evidence"),), example_id=row_id,
            ),
            gold=(gold,), dedupe_text=dedupe_text,
            metadata={"impossible": str(raw.is_impossible)},
        )

    raise RowRefused(
        reason_code="unknown_family_for_source",
        expected="one of qa.answerability, qa.answer_span", actual=family_id,
        detail="rajpurkar/squad_v2",
    )


# -- qd-mutate/commitpackft ----------------------------------------------------


def rewrite_defect_class(
    raw: DefectRow, *, family_id: str, index: int, config: DataConfig
) -> DataRow:
    """One qd-mutate example: which class of change, and which lines it touched.

    Two slots over one context, the unified diff. ``clean`` is a real class on the choice
    slot and a genuine abstention on the span slot -- nothing was found, so the span points
    at nothing -- which gives the span channel a ``noul`` that is taught, not dumped.

    A noul-corpus row (``NOUL_CLASS``, ``qd_data.defect_class.load_noul_rows``) abstains on
    both: its context is not this model's kind, so neither "which class" nor "which lines"
    has an answer. The options stay the four classes; the gold is the abstention every
    option set already ends in.

    ``index`` is unused beyond the signature every rewriter shares: the row id is the
    corpus's own example id, so a read at a different offset renames nothing.
    """
    del index
    if family_id != DEFECT_FAMILY_ID:
        raise RowRefused(
            reason_code="unknown_family_for_source",
            expected=DEFECT_FAMILY_ID, actual=family_id, detail=DEFECT_SOURCE_ID,
        )
    # The per-row licence check. The join in `qd_data.defect_class` resolved which
    # licence governs the row; whether that licence is admitted is asked here, where a
    # refusal is counted like every other source's.
    admit_licence(raw.licence, config=config.licence, source=f"{DEFECT_SOURCE_ID} {raw.example_id}")
    if not raw.repo.strip():
        raise RowRefused(
            reason_code="missing_repo", expected="a repository", actual=raw.repo,
            detail="a row with no split unit cannot be split at the repo level",
        )
    if raw.span_refusal is not None:
        raise RowRefused(
            reason_code=raw.span_refusal,
            expected="a span the diff represents", actual=raw.example_id,
            detail=(
                f"class {raw.mutation_class!r}: the mutation's span could not be rebased "
                "into the diff, and a span pointed at a nearby line trains the pointer on "
                "a fabricated target"
            ),
        )
    header = f"file: {raw.path}\n\n"
    if header.count("\n") != CONTEXT_HEADER_LINES:
        raise RowRefused(
            reason_code="path_contains_newline",
            expected=f"a one-line path ({CONTEXT_HEADER_LINES} header lines)",
            actual=header.count("\n"),
            detail="every span label would shift by the extra lines",
        )
    # The final newline is dropped so the context does not end on an empty line that is
    # a span candidate pointing at nothing; every earlier line keeps its number.
    body = raw.diff[:-1] if raw.diff.endswith("\n") else raw.diff
    context = header + body

    if raw.diff_span is None:
        span_gold = GoldAnswer(slot_name=SPAN_SLOT, value=None, is_noul=True)
    else:
        start, end = raw.diff_span
        span_gold = GoldAnswer(
            slot_name=SPAN_SLOT,
            value=(start + CONTEXT_HEADER_LINES, end + CONTEXT_HEADER_LINES),
        )
    choice_gold = (
        GoldAnswer(slot_name=CHOICE_SLOT, value=None, is_noul=True)
        if raw.mutation_class == NOUL_CLASS
        else GoldAnswer(slot_name=CHOICE_SLOT, value=raw.mutation_class)
    )
    row_id = f"qdm:{family_id}:{raw.example_id}"
    return _row(
        row_id=row_id, source_id=DEFECT_SOURCE_ID, family_id=family_id,
        repo_key=raw.repo,
        # `pair_key`'s identity (repo, path, symbol, arity): the same function mutated
        # twice is one identity, and the identity check keeps it on one side. A routed noul
        # row whose identity is fixed by its route (a contrast row is `contrast:<twin>`)
        # carries it instead.
        identity_key=(
            raw.identity_key
            if raw.identity_key is not None
            else f"{raw.repo}::{raw.path}::{raw.symbol}/{raw.arity}"
        ),
        licence_id=raw.licence,
        request=_request(
            family_id=family_id, context=context,
            slots=(
                ChoiceSlot(name=CHOICE_SLOT, options=DEFECT_CLASSES),
                SpanSlot(name=SPAN_SLOT),
            ),
            example_id=row_id,
        ),
        gold=(choice_gold, span_gold),
        # The diff, because it is the text the model reads: a leak is a diff seen in
        # training reappearing across a repo boundary. The whole file is not compared --
        # two vendored copies mutated at different sites share no diff lines, and neither
        # diff shows the model the other's content.
        dedupe_text=raw.diff,
        metadata={
            "language": raw.language,
            "operator": raw.operator,
            "pool_id": raw.pool_id,
            # Which of the noul corpus's sources a row is from: the licence and the
            # provenance of a SQuAD- or template-derived row are not commitpackft's, and
            # this source id is the family's single one.
            **({"noul_source": raw.noul_source} if raw.noul_source is not None else {}),
            # v5's routed noul rows (own-prose, contrast, G6): which route and form, and the
            # train pin that keeps every one of them out of val and held-out whatever its
            # unit hashes to (rule 2: a hashed noul unit would move the defect_class val and
            # held-out populations). Absent on every row v4 built.
            **(
                {
                    NOUL_ROUTE_KEY: raw.noul_route,
                    **({NOUL_FORM_KEY: raw.noul_form} if raw.noul_form is not None else {}),
                    PINNED_SPLIT_KEY: "train",
                }
                if raw.noul_route is not None
                else {}
            ),
        },
    )


# -- corpus-level self-consistency -------------------------------------------
#
# ``GAP-DATA-NOTHING-REFUSES-TWO-ROWS-THAT-CONTRADICT``. Three design calls, each
# made here rather than left implicit, because each one has a wrong answer that
# would look right.
#
# **Where.** Here, in the mixture, and nowhere else. Dedupe is the tempting seam --
# it already groups rows and already sees these pairs -- but its key is
# ``dedupe_text``, which is *deliberately* not the prompt (``DataRow.dedupe_text``:
# "for a code row it is the *changed code*, not the rendered prompt"), and its
# question is leakage. Teaching it a second question with a different key would be
# one helper answering two, with a flag meaning "actually do the other thing". The
# manifest writer is the other candidate and is downstream of the split, which is
# too late in the wrong way: a contradictory pair split across train and val is not
# unlearnable, it is a guaranteed eval error, and both are defects of the corpus
# rather than of a split. ``build_mixture`` is where a row becomes an example and
# where every other "this cannot be an example" verdict already lives.
#
# **What "the same prompt" means.** The prompt the model conditions on, as
# :func:`qd_data.render.render` produces it, at ``seed=None``. Not the source text:
# two rows can differ in source and render identically once escaped. Not the wire
# form: ``Request.to_wire`` carries ``metadata`` and ``expect``, which no token of
# the prompt depends on, so two rows with identical prompts can have different wire
# forms and hashing that would miss them. ``seed=None`` -- the serving order, no
# per-example shuffle -- because the alternative is a verdict that depends on which
# training seed the shard writer happened to use, admitting at one seed a corpus it
# refuses at another. The known limit is stated in
# ``GAP-DATA-CONSISTENCY-KEY-IGNORES-OPTION-ORDER``: two rows whose option *sets*
# agree but whose canonical orders differ are two prompts here, and are not grouped.
#
# **Why the legitimate case survives.** An unanswerable row is *supposed* to sit
# beside an answerable one -- that is what the two task-holdout families are for,
# and a check that refused the shape would be worse than no check. The shape is not
# what is removed: byte-identical prompts are. SQuAD 2.0 *usually* gives its
# unanswerable question different words, so the honest pair renders two prompts and
# never meets in a group. Not always: the approved cache (2026-09-29) holds 29
# identical-question pairs over one context, one answerable and one not, and CLINC
# files 4 identical utterances under two intents. Those groups are dropped whole.
#
# **Remove, not refuse.** The group goes and is counted; the corpus stays trainable.
# Neither gold can be preferred without a judgement about the upstream data that
# this pass cannot make, and refusing the corpus made one upstream duplicate veto
# every other row.

#: Bounded fan-out for the consistency pass. Measured on this repository's own
#: corpus (672 rows, ~26 KB of rendered text per row): 1.88 ms/row, so this bound is
#: roughly eight minutes of rendering. A corpus over it is ``NotRun`` with the bound
#: in the reason -- a pass over a subsample would be this very defect one level up.
DEFAULT_MAX_CONSISTENCY_ROWS: Final[int] = 250_000

#: How many contradictory groups the report names, how many row ids per group, and
#: how many distinct golds per group. The *counts* are always complete -- the
#: tri-state's ``value`` and each group's ``n_rows``/``n_golds`` -- and only the
#: worked examples are capped, so a reader can always tell which of the two they are
#: holding. Capped rather than complete because this payload is written into every
#: manifest: a pathological corpus of ten thousand identical prompts with ten
#: thousand different golds would otherwise put megabytes of examples in a file whose
#: job is to be read.
MAX_NAMED_CONTRADICTIONS: Final[int] = 20
MAX_NAMED_ROW_IDS: Final[int] = 8
MAX_NAMED_GOLDS: Final[int] = 4


@dataclass(frozen=True, slots=True)
class PromptContradiction:
    """Rows whose rendered prompts agree and whose gold answers do not."""

    #: Digest of the rendered prompt, so two reports over one corpus name the same
    #: group. The prompt itself is not carried: it is up to 800 KB.
    prompt_digest: str
    #: Row ids in the group, sorted, capped at :data:`MAX_NAMED_ROW_IDS`.
    row_ids: tuple[str, ...]
    #: Distinct gold answers behind that one prompt, canonical-JSON, sorted, capped
    #: at :data:`MAX_NAMED_GOLDS`. Always at least two -- a group with one is not a
    #: contradiction, it is a duplicate, and that is dedupe's question.
    golds: tuple[str, ...]
    #: The whole group, which may exceed ``len(row_ids)``.
    n_rows: int
    #: Distinct golds in the whole group, which may exceed ``len(golds)``.
    n_golds: int
    family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_golds < 2:
            raise ValueError(
                f"PromptContradiction {self.prompt_digest}: {self.n_golds} distinct gold "
                "answers. A group whose golds all agree is a duplicate, which is "
                "dedupe's question, not this one."
            )
        if len(self.golds) > self.n_golds or len(self.row_ids) > self.n_rows:
            raise ValueError(
                f"PromptContradiction {self.prompt_digest}: named more than it counted "
                f"({len(self.row_ids)}/{self.n_rows} rows, {len(self.golds)}/{self.n_golds} "
                "golds). The capped list is a sample of the count, never larger than it."
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_digest": self.prompt_digest,
            "row_ids": list(self.row_ids),
            "n_rows": self.n_rows,
            "golds": list(self.golds),
            "n_golds": self.n_golds,
            "family_ids": list(self.family_ids),
        }


def _gold_key(row: DataRow) -> str:
    """A row's supervision, canonically. Order-independent across slots."""
    return canonical_json(sorted((g.to_json() for g in row.gold), key=lambda g: g["slot_name"]))


@dataclass(frozen=True, slots=True)
class _PromptGroups:
    """Every rendered row, grouped by prompt digest and then by gold."""

    #: digest -> gold key -> row ids. Only the digest is kept, never the prompt: a
    #: rendered prompt runs to ``caps.max_rendered_bytes``, and holding one per row is
    #: an unbounded allocation wearing a dict.
    by_digest: dict[str, dict[str, list[str]]]
    families: dict[str, set[str]]
    n_total: int
    n_unrenderable: int

    @property
    def n_grouped(self) -> int:
        return self.n_total - self.n_unrenderable

    @property
    def conflicting(self) -> list[str]:
        return sorted(d for d, by_gold in self.by_digest.items() if len(by_gold) > 1)

    def rows_in(self, digests: Sequence[str]) -> set[str]:
        return {i for d in digests for group in self.by_digest[d].values() for i in group}


def _group_prompts(rows: Sequence[DataRow], *, caps: RenderCaps) -> _PromptGroups:
    """Render every row at ``seed=None`` and group by prompt, then by gold.

    A row whose context is over :class:`~qd_data.render.RenderCaps` cannot be rendered
    and therefore cannot be grouped -- ``qd_train.shards`` refuses exactly those rows
    too (it catches ``QdRefusal`` around its own ``render`` call), so such a row
    reaches no shard and can contradict nothing that does. It is counted, never
    dropped quietly: ``(n_grouped, n_total)`` is what keeps this from reading as full
    coverage.
    """
    by_digest: dict[str, dict[str, list[str]]] = {}
    families: dict[str, set[str]] = {}
    n_unrenderable = 0
    for row in rows:
        try:
            rendered = render(row.request, caps=caps, seed=None)
        except QdRefusal:
            n_unrenderable += 1
            continue
        hasher = hashlib.blake2b(digest_size=16)
        for part in (rendered.prefix, *(s.suffix for s in rendered.slots)):
            blob = part.encode("utf-8")
            # Length-prefixed framing rather than a separator character. A separator
            # is only unambiguous while nothing can emit it, which today is true --
            # U+0000 is in `render.HEX_ESCAPED` and every untrusted field reaches the
            # prompt through `escape_inline`/`escape_block` -- and that is a property
            # of another module's alphabet, not of this one. Framing does not depend
            # on it, and it also avoids materialising the join, which is up to
            # `caps.max_rendered_bytes` per row.
            hasher.update(len(blob).to_bytes(8, "big"))
            hasher.update(blob)
        digest = hasher.hexdigest()
        by_digest.setdefault(digest, {}).setdefault(_gold_key(row), []).append(row.row_id)
        families.setdefault(digest, set()).add(row.family_id)
    return _PromptGroups(
        by_digest=by_digest, families=families, n_total=len(rows),
        n_unrenderable=n_unrenderable,
    )


def _name_contradictions(groups: _PromptGroups) -> tuple[PromptContradiction, ...]:
    """The conflicting groups as a reader sees them, capped at the named maxima."""
    named: list[PromptContradiction] = []
    for digest in groups.conflicting[:MAX_NAMED_CONTRADICTIONS]:
        by_gold = groups.by_digest[digest]
        # One id per distinct gold first, then fill. Taking the lowest ids outright can
        # name eight rows that all carry the *same* answer, which shows a reader one
        # side of a disagreement and calls it the pair.
        witnesses = [sorted(group)[0] for _, group in sorted(by_gold.items())]
        chosen = set(witnesses)
        rest = sorted(i for group in by_gold.values() for i in group if i not in chosen)
        ids = (witnesses + rest)[:MAX_NAMED_ROW_IDS]
        named.append(
            PromptContradiction(
                prompt_digest=digest,
                row_ids=tuple(sorted(ids)),
                golds=tuple(sorted(by_gold)[:MAX_NAMED_GOLDS]),
                n_rows=sum(len(group) for group in by_gold.values()),
                n_golds=len(by_gold),
                family_ids=tuple(sorted(groups.families[digest])),
            )
        )
    return tuple(named)


def _bound_refusal(n_total: int, max_rows: int) -> NotRun:
    return NotRun(
        reason=(
            f"the corpus has {n_total} rows and the consistency pass is bounded at "
            f"{max_rows}; the prompts were not grouped, so this corpus is "
            "unchecked for contradictory supervision, not clean of it"
        )
    )


def _unrenderable_note(groups: _PromptGroups) -> str:
    if not groups.n_unrenderable:
        return ""
    return (
        f". {groups.n_unrenderable} row(s) could not be rendered and were not grouped; "
        "those rows reach no shard either"
    )


def check_prompt_consistency(
    rows: Sequence[DataRow],
    *,
    max_rows: int = DEFAULT_MAX_CONSISTENCY_ROWS,
    caps: RenderCaps = DEFAULT_CAPS,
) -> tuple[TriState, tuple[PromptContradiction, ...]]:
    """Group by rendered prompt; a group whose golds disagree is a contradiction.

    Read-only: reports, never removes. :func:`drop_contradictory_prompts` is the
    remediation :func:`build_mixture` applies, over the same grouping.

    Returns the verdict and the named groups. The verdict carries ``n``/``n_total``
    as *rows grouped* over *rows examined* (see :func:`_group_prompts`).

    ``NotRun`` above ``max_rows``: a consistency claim over a subsample is the defect
    this function exists to catch, one level up.
    """
    if len(rows) > max_rows:
        return _bound_refusal(len(rows), max_rows), ()
    groups = _group_prompts(rows, caps=caps)
    conflicting = groups.conflicting
    if not conflicting:
        return (
            Ran(
                passed=True,
                value=0,
                n=groups.n_grouped,
                n_total=groups.n_total,
                detail=(
                    f"{groups.n_grouped} of {groups.n_total} rows grouped into "
                    f"{len(groups.by_digest)} distinct rendered prompts; no prompt carries "
                    "two different gold answers" + _unrenderable_note(groups)
                ),
            ),
            (),
        )
    named = _name_contradictions(groups)
    first = named[0]
    n_rows_in_conflict = len(groups.rows_in(conflicting))
    return (
        Ran(
            passed=False,
            value=len(conflicting),
            n=groups.n_grouped,
            n_total=groups.n_total,
            detail=(
                f"{len(conflicting)} rendered prompt(s) carry more than one gold answer, "
                f"over {n_rows_in_conflict} rows in families "
                f"{sorted({f for d in conflicting for f in groups.families[d]})}. A causal "
                "model conditions on the prompt and nothing else, so it must split its "
                "mass between them and can never answer better than chance on the group. "
                f"First: prompt {first.prompt_digest} over rows {list(first.row_ids)} with "
                f"golds {list(first.golds)}."
                + (
                    f" {len(conflicting) - len(named)} further group(s) are counted here "
                    "but not named."
                    if len(conflicting) > len(named)
                    else ""
                )
            ),
        ),
        named,
    )


#: The refusal code a row dropped by :func:`drop_contradictory_prompts` is counted
#: under, in :attr:`MixtureResult.refusals` and in its family's coverage.
CONTRADICTORY_PROMPT: Final[str] = "contradictory_prompt"


@dataclass(frozen=True, slots=True)
class ConsistencyDrop:
    """What :func:`drop_contradictory_prompts` kept, removed and can say about both."""

    rows: tuple[DataRow, ...]
    #: Row ids removed, every row of every conflicting group -- never capped.
    dropped: frozenset[str]
    #: The verdict on the rows **kept**, with the drop in its detail and value.
    verdict: TriState
    #: The dropped groups, named and capped as in :func:`check_prompt_consistency`.
    named: tuple[PromptContradiction, ...]


def drop_contradictory_prompts(
    rows: Sequence[DataRow],
    *,
    max_rows: int = DEFAULT_MAX_CONSISTENCY_ROWS,
    caps: RenderCaps = DEFAULT_CAPS,
) -> ConsistencyDrop:
    """Remove every row of every rendered prompt that carries two golds.

    The whole group goes, never a winner: which of two golds behind one prompt is
    right is a question about the upstream data this pass cannot answer, and keeping
    either one trains a label the source itself contradicts. Measured on the approved
    caches (2026-09-29): SQuAD v2 asks the identical question over the identical
    context once answerable and once not, and CLINC files the identical utterance
    under two intents.

    Above ``max_rows`` nothing is grouped and nothing is dropped: the verdict is
    ``NotRun`` and the rows come back untouched, so the mixture is ``NotRun`` too.
    """
    if len(rows) > max_rows:
        return ConsistencyDrop(
            rows=tuple(rows), dropped=frozenset(),
            verdict=_bound_refusal(len(rows), max_rows), named=(),
        )
    groups = _group_prompts(rows, caps=caps)
    conflicting = groups.conflicting
    dropped = frozenset(groups.rows_in(conflicting))
    kept = tuple(r for r in rows if r.row_id not in dropped)
    named = _name_contradictions(groups)
    drop_note = (
        f"; {len(conflicting)} rendered prompt(s) carried more than one gold answer "
        f"(families {sorted({f for d in conflicting for f in groups.families[d]})}), and "
        f"all {len(dropped)} of their rows were dropped as {CONTRADICTORY_PROMPT!r} with "
        "no winner chosen"
        + (
            f" ({len(conflicting) - len(named)} of those groups counted, not named)"
            if len(conflicting) > len(named)
            else ""
        )
        if conflicting
        else ""
    )
    return ConsistencyDrop(
        rows=kept,
        dropped=dropped,
        verdict=Ran(
            passed=True,
            value=len(conflicting),
            n=groups.n_grouped,
            n_total=groups.n_total,
            detail=(
                f"{groups.n_grouped} of {groups.n_total} rows grouped into "
                f"{len(groups.by_digest)} distinct rendered prompts; no prompt in the "
                "emitted rows carries two different gold answers"
                + drop_note
                + _unrenderable_note(groups)
            ),
        ),
        named=named,
    )


# -- abstention supply -------------------------------------------------------


#: The decode channel a slot answers over. ``choice`` and ``score`` both decode one
#: letter from the option block, which is why the abstention question is asked per
#: channel rather than per slot type: a corpus can teach ``noul`` on the span
#: channel and never once on the letter channel, and did.
_CHANNEL_BY_SLOT: Final[dict[type, str]] = {
    ChoiceSlot: "choice",
    ScoreSlot: "score",
    SpanSlot: "span",
}

#: Families whose rewriter can produce ``is_noul=True``, and **on which channel**.
#: Derived by reading every branch of the rewriters above, not by running them:
#: CLINC's out-of-scope class, SQuAD's unanswerable questions, qd-mutate's ``clean``
#: examples, whose span points at nothing, and the noul corpus's rows
#: (``qd_data.defect_class.load_noul_rows``), which abstain on both slots. Keyed by
#: channel since ``code.defect_class`` became the first family with two slots: its
#: ``clean`` rows abstain on the span and never on the class, and only its noul-corpus rows
#: abstain on the class -- so a defect-class build without that corpus teaches the letter
#: channel nothing, which :func:`abstention_supply` says rather than calling it a defect.
ABSTAINING_FAMILIES: Final[dict[str, frozenset[str]]] = {
    "intent.classification": frozenset({"choice"}),
    "qa.answer_span": frozenset({"span"}),
    DEFECT_FAMILY_ID: frozenset({"span", "choice"}),
}

#: Families that abstain on a channel only through rows a separate corpus supplies, keyed
#: by channel, with the corpus named. Zero abstentions there is a composition choice -- the
#: corpus was not loaded -- not a rewriter that dropped them.
_ABSTAINS_ONLY_VIA_CORPUS: Final[dict[tuple[str, str], str]] = {
    (DEFECT_FAMILY_ID, "choice"): "the noul corpus (qd_data.defect_class.load_noul_rows)",
}


def abstention_supply(rows: Sequence[DataRow]) -> dict[str, TriState]:
    """Per decode channel: how many rows teach ``noul``, out of how many rows.

    ``docs/schema-api.md``: *"``noul`` is a first-class member of every option set,
    not a dump class."* A channel with zero abstaining rows trains the model
    *against* the abstain row at every step, on the channel where abstention is the
    product's whole point -- and it makes abstention decoding unmeasurable, because
    there is nothing to decode with.

    ``Ran(passed=False)`` on such a channel, **reported and not folded into the
    mixture's status.** Whether a corpus with no letter-channel abstention may be
    trained on is a kill criterion, and ``CLAUDE.md`` rule 2 puts adopting one out of
    an agent's reach: the measurement is this lane's, the threshold is not. The
    contradiction check above is different and does fail closed, because supervision
    that a causal model provably cannot fit is not a threshold question.

    A channel with no rows gets no entry: absent from the corpus and measured-empty
    are different facts, and only the second is a finding.
    """
    n_rows: Counter[str] = Counter()
    n_noul: Counter[str] = Counter()
    capable: dict[str, set[str]] = {}
    present: dict[str, set[str]] = {}
    for row in rows:
        for slot in row.request.slots:
            channel = _CHANNEL_BY_SLOT.get(type(slot))
            if channel is None:  # pragma: no cover - Request refuses other slot types
                continue
            n_rows[channel] += 1
            present.setdefault(channel, set()).add(row.family_id)
            if channel in ABSTAINING_FAMILIES.get(row.family_id, frozenset()):
                capable.setdefault(channel, set()).add(row.family_id)
            if any(g.is_noul and g.slot_name == slot.name for g in row.gold):
                n_noul[channel] += 1

    out: dict[str, TriState] = {}
    for channel in sorted(n_rows):
        total = n_rows[channel]
        taught = n_noul[channel]
        able = sorted(capable.get(channel, set()))
        if taught:
            out[channel] = Ran(
                passed=True, value=taught, n=taught, n_total=total,
                detail=(
                    f"{channel}: {taught} of {total} rows carry a noul gold, from "
                    f"{able}"
                ),
            )
            continue
        via_corpus = {
            f: _ABSTAINS_ONLY_VIA_CORPUS[(f, channel)]
            for f in able if (f, channel) in _ABSTAINS_ONLY_VIA_CORPUS
        }
        if any(f not in via_corpus for f in able):
            why = (
                "A family that can abstain produced none, which is a construction defect "
                "rather than a composition choice."
            )
        elif via_corpus:
            why = (
                "Every family here that can abstain on this channel does so only through "
                "a separately loaded corpus ("
                + "; ".join(f"{f}: {c}" for f, c in sorted(via_corpus.items()))
                + ") that this build did not load, so this is the corpus's composition "
                "rather than a rewriter dropping them."
            )
        else:
            why = (
                "No family on this channel can produce one, so this is the corpus's "
                "composition rather than a rewriter dropping them."
            )
        out[channel] = Ran(
            passed=False, value=0, n=0, n_total=total,
            detail=(
                f"{channel}: 0 of {total} rows carry a noul gold, so every step trains "
                "this channel against the abstain row. Families present: "
                f"{sorted(present.get(channel, set()))}; of those, able to abstain at "
                f"all: {able or 'none'}. {why}"
            ),
        )
    return out


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
    #: Did any two rows ask one question and demand two answers? See
    #: :func:`drop_contradictory_prompts`: the verdict on the rows **emitted**, whose
    #: ``value`` counts the contradictory prompts whose rows were dropped. ``NotRun``
    #: (over the bound) is folded into :attr:`status`, so an unchecked corpus is
    #: refused by ``qd_train.data_access.open_training_data`` like any other.
    prompt_consistency: TriState
    #: The dropped groups, capped at :data:`MAX_NAMED_CONTRADICTIONS`. The full count
    #: lives in the tri-state, and every dropped row in :attr:`refusals` under
    #: :data:`CONTRADICTORY_PROMPT`.
    contradictions: tuple[PromptContradiction, ...]
    #: ``channel -> TriState``: how much ``noul`` supervision each decode channel
    #: carries. **Reported, not folded into** :attr:`status` -- see
    #: :func:`abstention_supply` for why that boundary sits where it does.
    abstention: dict[str, TriState]
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
            "prompt_consistency": self.prompt_consistency.to_json(),
            "contradictions": [c.to_json() for c in self.contradictions],
            "abstention": {k: v.to_json() for k, v in sorted(self.abstention.items())},
            "status": self.status.to_json(),
        }


def build_mixture(
    raw_by_source: Mapping[str, Sequence[RawRow | DefectRow]],
    *,
    config: DataConfig,
    families: Sequence[str] | None = None,
    capped_sources: Sequence[str] = (),
    max_consistency_rows: int = DEFAULT_MAX_CONSISTENCY_ROWS,
    clinc_domain_map: ClincDomainMap | None = None,
) -> MixtureResult:
    """Rewrite every admitted source into the one prompt format.

    ``families`` defaults to every registered family of every admitted source,
    **including the held-out ones** -- the holdout is applied by the splitter, not by
    omitting the rows here, because a family that is never built cannot be checked
    for abstention.

    ``capped_sources`` names sources whose read hit a bound. Any entry makes the
    result's status ``NotRun``: a capped sample must never be reported as complete
    coverage.

    ``max_consistency_rows`` bounds the self-consistency pass
    (:func:`drop_contradictory_prompts`), which renders every row once. Over the bound
    the pass is ``NotRun``, nothing is dropped, and the mixture is ``NotRun``.
    """
    from . import general  # local: general imports this module
    from .sources import TASK_FAMILIES  # local: avoids a cycle at import time

    general_two_stage = {f.family_id for f in general.CLINC_TWO_STAGE_FAMILIES}

    wanted = (
        tuple(families)
        if families is not None
        else tuple(
            f.family_id
            for f in TASK_FAMILIES.values()
            if f.source_id in raw_by_source
            # The two-stage CLINC families need a domain map. Without one they are not
            # requested by default; naming them in `families` still fails closed.
            and not (clinc_domain_map is None and f.family_id in general_two_stage)
        )
    )
    for fam in wanted:
        task_family_by_id(fam)  # raises on an unknown family id

    rows: list[DataRow] = []
    refusals: dict[str, dict[str, int]] = {}
    n_input: dict[str, int] = {}
    refused_sources: dict[str, tuple[str, ...]] = {}
    family_coverage: dict[str, TriState] = {}
    #: family -> (source, built, attempted, refusal counts, routed), before the drop.
    attempts: dict[str, tuple[str, int, int, Counter[str], int]] = {}

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
            # SQuAD rows whose article belongs to the other SQuAD family are ROUTED, not
            # refused: nothing is wrong with them, they are that family's rows. Counting them
            # as refusals would read as an 80% data-quality loss and inflate every
            # built/attempted denominator. `rewrite_squad` still refuses them (fail closed)
            # for a caller that reaches it directly.
            routed = 0
            for i, raw in enumerate(raws):
                if (
                    family_id in SQUAD_TITLE_FAMILIES
                    and isinstance(raw, SquadRow)
                    and raw.title.strip()
                    and squad_title_family(raw.title, seed=config.seed) != family_id
                ):
                    routed += 1
                    continue
                # A decision-pool row names its own family (one source feeds six Open-Jev
                # families), so it is that family's row, routed, not five refusals.
                if isinstance(raw, TypedDecisionRow) and raw.family_id != family_id:
                    routed += 1
                    continue
                try:
                    rows.append(
                        _dispatch(
                            raw,
                            family_id=family_id,
                            index=i,
                            config=config,
                            intent_vocabulary=intent_vocab,
                            messages=messages,
                            clinc_domain_map=clinc_domain_map,
                        )
                    )
                except RowRefused as exc:
                    family_counts[exc.reason_code] += 1
                except LicenceRefused as exc:
                    family_counts[f"licence:{exc.actual}"] += 1
            counts.update(family_counts)
            # A family belongs to exactly one source, so this never collides. Coverage
            # is computed after the consistency drop below, which can take rows from it.
            attempts[family_id] = (source_id, len(rows) - n_before, len(raws) - routed,
                                   family_counts, routed)
        refusals[source_id] = dict(counts)

    # Every row of a prompt that carries two golds goes, counted, before anything is
    # histogrammed: a dropped row is in no split, no shard and no coverage number.
    drop = drop_contradictory_prompts(rows, max_rows=max_consistency_rows)
    dropped_by_family: Counter[str] = Counter(
        r.family_id for r in rows if r.row_id in drop.dropped
    )
    dropped_by_source: Counter[str] = Counter(
        r.source_id for r in rows if r.row_id in drop.dropped
    )
    rows = list(drop.rows)
    consistency, contradictions = drop.verdict, drop.named
    for source_id, n in dropped_by_source.items():
        refusals[source_id][CONTRADICTORY_PROMPT] = n
    for family_id, (source_id, n_built, n_attempted, family_counts, routed) in (
        attempts.items()
    ):
        if dropped_by_family[family_id]:
            family_counts[CONTRADICTORY_PROMPT] = dropped_by_family[family_id]
        family_coverage[family_id] = _family_coverage(
            family_id=family_id,
            source_id=source_id,
            n_built=n_built - dropped_by_family[family_id],
            n_attempted=n_attempted,
            reasons=dict(family_counts),
            routed=routed,
        )

    licence_hist: Counter[str] = Counter()
    host_hist: Counter[str] = Counter()
    obligations: dict[str, tuple[str, ...]] = {}
    for row in rows:
        licence_hist[row.licence_id] += 1
        host_hist[row.host] += 1
        obligations[row.licence_id] = row.obligations

    abstention = abstention_supply(rows)

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
            reason=(
                "no rows survived rewriting"
                + (
                    f" and the consistency drop ({len(drop.dropped)} row(s) of "
                    f"{consistency.value} contradictory prompt(s), {CONTRADICTORY_PROMPT!r})"
                    if drop.dropped
                    else ""
                )
                + ", so no mixture was built and nothing was verified"
            )
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
    elif isinstance(consistency, NotRun):
        # The pass is bounded and the bound bound. An unchecked corpus is not a clean
        # one, and `open_training_data` refuses a NotRun snapshot by default.
        status = NotRun(reason=consistency.reason)
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
                + f"; self-consistency: {consistency.coverage_str()} rows grouped, "
                + "no contradictory prompt in the emitted rows"
                + (
                    f" ({len(drop.dropped)} row(s) of {consistency.value} contradictory "
                    f"prompt(s) dropped as {CONTRADICTORY_PROMPT!r})"
                    if drop.dropped
                    else ""
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
        prompt_consistency=consistency,
        contradictions=contradictions,
        abstention=abstention,
        status=status,
    )


def _family_coverage(
    *,
    family_id: str,
    source_id: str,
    n_built: int,
    n_attempted: int,
    reasons: dict[str, int],
    routed: int = 0,
) -> TriState:
    """One family's built/attempted pair, or ``NotRun`` when it built nothing.

    ``routed`` rows belonged to another family by design (the SQuAD title partition) and
    are in neither number; the detail says how many, so the source's size stays visible.
    """
    routed_note = (
        f"; {routed} {source_id} row(s) routed to another family by design" if routed else ""
    )
    top = ", ".join(
        f"{code}={n}" for code, n in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    if n_attempted == 0:
        return NotRun(
            reason=(
                f"{family_id}: no rows were read from {source_id}, so the family was "
                "never built and nothing about it was verified" + routed_note
            )
        )
    if n_built == 0:
        return NotRun(
            reason=(
                f"{family_id}: all {n_attempted} rows read from {source_id} were refused "
                f"({top}), so the family has no examples -- nothing about it was built "
                "and nothing about it can be checked" + routed_note
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
            + routed_note
        ),
    )


def _dispatch(
    raw: RawRow | DefectRow,
    *,
    family_id: str,
    index: int,
    config: DataConfig,
    intent_vocabulary: Sequence[str],
    messages: Sequence[str],
    clinc_domain_map: ClincDomainMap | None = None,
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
        from . import general  # local: general imports this module

        if family_id in (general.CLINC_DOMAIN_FAMILY, general.CLINC_WITHIN_DOMAIN_FAMILY):
            if clinc_domain_map is None:
                raise RowRefused(
                    reason_code="no_clinc_domain_map",
                    expected="a ClincDomainMap passed to build_mixture", actual=None,
                    detail="the two-stage intent families cannot be asked without one",
                )
            return general.rewrite_clinc_two_stage(
                raw, family_id=family_id, index=index, config=config,
                domain_map=clinc_domain_map,
            )
        return rewrite_clinc(
            raw, family_id=family_id, index=index, config=config,
            intent_vocabulary=intent_vocabulary,
        )
    if isinstance(raw, SquadRow):
        return rewrite_squad(raw, family_id=family_id, index=index, config=config)
    if isinstance(raw, DefectRow):
        return rewrite_defect_class(raw, family_id=family_id, index=index, config=config)
    if isinstance(raw, (MmluRow, CsqaRow)):
        # Local import: qd_data.general imports this module's `_request`/`_row` funnel.
        from . import general

        if isinstance(raw, MmluRow):
            return general.rewrite_mmlu(raw, family_id=family_id, index=index, config=config)
        return general.rewrite_csqa(raw, family_id=family_id, index=index, config=config)
    if isinstance(raw, TypedDecisionRow):
        # Local import: qd_data.decisions imports this module's `_request`/`_row` funnel.
        from . import decisions

        return decisions.rewrite_typed_decision(
            raw, family_id=family_id, index=index, config=config
        )
    raise RowRefused(
        reason_code="unknown_raw_row_type",
        expected=(
            "CommitPackFtRow | ClincRow | SquadRow | DefectRow | MmluRow | CsqaRow | "
            "TypedDecisionRow"
        ),
        actual=type(raw).__name__,
    )
