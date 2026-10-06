"""General decision families: licence-clean public sets in the one ``Request`` format.

Plan 2026-09-28, "What gets built on the Mac first" item 5: Lappi should answer
any app's closed question, not only code questions. This module renders three
more question shapes into the same :class:`~qd_data.schema.Request` every other
family uses, through the same funnel (``qd_data.mixture._request`` and ``_row``)
so the invisible-character refusal, the licence classification and the row
shape each keep one owner.

``knowledge.multiple_choice`` (``cais/mmlu``)
    Question in the context, the four choices as a ``choice`` slot. Split unit is
    the subject, so a held-out subject is content the model never saw.
``commonsense.multiple_choice`` (``tau/commonsense_qa``)
    Same shape, five options. Split unit is ``question_concept``: CSQA writes
    several questions around one ConceptNet concept, and a row-level split would
    put near-paraphrases on both sides.
``intent.domain`` / ``intent.within_domain`` (``clinc/clinc_oos``)
    The answer to "CLINC has 150 intents and the slot holds 16". See
    :class:`ClincDomainMap`. Registered in ``qd_data.sources.TASK_FAMILIES``;
    ``build_mixture`` builds them only when given a ``clinc_domain_map``.

**No invented abstentions.** MMLU and CSQA carry no natural ``noul``. Making one
by deleting the gold option would teach "none of these" on option sets whose
remaining distractors are sometimes defensible answers -- CSQA's distractors are
written to be plausible -- so the gold would be wrong on exactly the rows that
matter. These two families add answer supervision only; the free ``noul`` signal
stays with CLINC's out-of-scope class. The rendered option set still ends in the
reserved ``noul`` row (``qd_data.render``, ``RESERVED_NOUL_ROWS`` on the Rust
side), so the model is always *offered* abstention.

Every refusal is a :class:`~qd_data.mixture.RowRefused` with a reason code, raised
*before* a slot is constructed. ``ChoiceSlot`` raises its own refusals for an
empty, duplicate or reserved option, and those are not ``RowRefused`` -- one bad
upstream row would otherwise abort the whole build instead of being counted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from .config import DataConfig
from .licences import admit_licence
from .loaders import ClincRow, CsqaRow, MmluRow
from .mixture import RowRefused, _request, _row, clinc_keys, refuse_benchmark_target_row
from .render import DEFAULT_CAPS, DeterministicRng
from .rows import DataRow, GoldAnswer
from .schema import MAX_CHOICE_OPTIONS, NOUL, ChoiceSlot
from .sources import PINNED_SPLIT_KEY, TaskFamily, source_by_id, task_family_by_id

__all__ = [
    "CLINC_DOMAINS_COMMIT",
    "CLINC_DOMAINS_SHA256",
    "CLINC_DOMAIN_FAMILY",
    "CLINC_TWO_STAGE_FAMILIES",
    "CLINC_WITHIN_DOMAIN_FAMILY",
    "CSQA_FAMILY",
    "DEFAULT_REPLAY_FRACTION",
    "MAX_DOMAIN_MAP_BYTES",
    "MMLU_FAMILY",
    "PINNED_SPLIT_KEY",
    "REPLAY_FAMILIES",
    "REPLAY_ONLY",
    "REPLAY_ROLE_KEY",
    "ClincDomainMap",
    "ReplayPartition",
    "load_clinc_domains",
    "partition_replay",
    "rewrite_clinc_two_stage",
    "rewrite_csqa",
    "rewrite_mmlu",
]

MMLU_FAMILY: Final[str] = "knowledge.multiple_choice"
CSQA_FAMILY: Final[str] = "commonsense.multiple_choice"
CLINC_DOMAIN_FAMILY: Final[str] = "intent.domain"
CLINC_WITHIN_DOMAIN_FAMILY: Final[str] = "intent.within_domain"

#: The two-stage CLINC families, read from the one registry that owns them
#: (``qd_data.sources.TASK_FAMILIES``) rather than restated here.
CLINC_TWO_STAGE_FAMILIES: Final[tuple[TaskFamily, ...]] = (
    task_family_by_id(CLINC_DOMAIN_FAMILY),
    task_family_by_id(CLINC_WITHIN_DOMAIN_FAMILY),
)

#: The upstream ``domains.json`` is 3,818 bytes (GitHub contents API, clinc/oos-eval
#: ``data/domains.json``, blob 60a74358e52060e60b6128ae4efe679e9d6ba69c). A file
#: sixteen times that is not the file.
MAX_DOMAIN_MAP_BYTES: Final[int] = 65_536

#: The domain map as fetched 2026-09-29 (user-approved): clinc/oos-eval at this commit,
#: whose LICENSE file is CC BY 3.0 Unported -- the licence the HF card declares.
#: Git blob id and sha256 both verified against the downloaded bytes.
CLINC_DOMAINS_COMMIT: Final[str] = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_DOMAINS_SHA256: Final[str] = (
    "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a"
)


#: The families replay may draw from (user-approved 2026-09-29: the general sets,
#: self-distilled), and the fraction of their training rows that is replay-only.
REPLAY_FAMILIES: Final[frozenset[str]] = frozenset({MMLU_FAMILY, CSQA_FAMILY})
DEFAULT_REPLAY_FRACTION: Final[float] = 0.15
#: ``DataRow.metadata`` key on a replay-only row. Its value is :data:`REPLAY_ONLY`.
REPLAY_ROLE_KEY: Final[str] = "replay_role"
REPLAY_ONLY: Final[str] = "replay_only"


def _digest(*parts: str) -> str:
    material = "\x1f".join(parts).encode("utf-8")
    return hashlib.blake2b(material, digest_size=8).hexdigest()


def _checked_options(options: Sequence[str], *, where: str) -> tuple[str, ...]:
    """Every check ``ChoiceSlot`` and the renderer would make, as ``RowRefused``.

    Also one check neither makes: two options equal after trimming and casefolding.
    ``ChoiceSlot`` compares them exactly, but ``"Paris"`` and ``"paris "`` are one
    answer under two letters to anyone reading the prompt, so the gold letter would
    be a coin flip between them.
    """
    opts = tuple(options)
    if len(opts) < 2:
        raise RowRefused(
            reason_code="too_few_options", expected="at least 2 options",
            actual=len(opts), detail=where,
        )
    if len(opts) > MAX_CHOICE_OPTIONS:
        raise RowRefused(
            reason_code="too_many_options",
            expected=f"at most {MAX_CHOICE_OPTIONS} options", actual=len(opts),
            detail=f"{where}: dropping an option would change the question",
        )
    seen: dict[str, str] = {}
    for opt in opts:
        norm = " ".join(opt.split()).casefold()
        if not norm:
            raise RowRefused(
                reason_code="empty_option", expected="non-empty options", actual=opt,
                detail=where,
            )
        if norm == NOUL:
            raise RowRefused(
                reason_code="option_named_noul",
                expected=f"no option literally named {NOUL!r}", actual=opt, detail=where,
            )
        if norm in seen:
            raise RowRefused(
                reason_code="duplicate_options",
                expected="options distinct after trimming and casefolding",
                actual=f"{seen[norm]!r} / {opt!r}", detail=where,
            )
        seen[norm] = opt
        size = len(opt.encode("utf-8"))
        if size > DEFAULT_CAPS.max_option_bytes:
            raise RowRefused(
                reason_code="option_too_long",
                expected=f"at most {DEFAULT_CAPS.max_option_bytes} bytes per option",
                actual=size, detail=f"{where}: refused rather than truncated",
            )
    return opts


def _require_key(value: str, *, reason_code: str, what: str) -> str:
    key = value.strip()
    if not key:
        raise RowRefused(
            reason_code=reason_code, expected=f"a non-empty {what}", actual=value,
            detail="a row with no split unit cannot be split without leaking",
        )
    return key


# -- cais/mmlu ---------------------------------------------------------------


def rewrite_mmlu(raw: MmluRow, *, family_id: str, index: int, config: DataConfig) -> DataRow:
    """One MMLU question. The answer is the option text, never its letter."""
    source = source_by_id("cais/mmlu")
    admit_licence(source.declared_licence, config=config.licence, source=source.source_id)
    if family_id != MMLU_FAMILY:
        raise RowRefused(
            reason_code="unknown_family_for_source", expected=MMLU_FAMILY,
            actual=family_id, detail=source.source_id,
        )
    refuse_benchmark_target_row(source.source_id, raw.upstream_split, config=config)
    subject = _require_key(raw.subject, reason_code="missing_subject", what="subject")
    question = raw.question.strip()
    if not question:
        raise RowRefused(reason_code="empty_question", expected="a question", actual="")
    options = _checked_options(raw.choices, where=f"mmlu {subject}")
    gold_text = options[raw.answer_index]

    # The split is pinned by upstream split (lead decision 2026-09-29): test and dev
    # train, validation is the family's val. The pinned split is part of the split
    # unit, so a train subject and a val subject are never one unit -- the
    # repo-disjointness check then holds by construction, and a question duplicated
    # across upstream splits is two units that the cross-split near-duplicate check
    # sees and refuses.
    pinned = source.pinned_split_of(raw.upstream_split)
    repo_key = f"mmlu-{pinned}:{subject}"
    # MMLU rows carry no id. The digest covers everything the model is shown, so a
    # re-read at another offset names the same row; the index keeps two exact
    # duplicates distinct as rows, leaving their merger to dedupe, whose job it is.
    digest = _digest(subject, question, *options)
    row_id = f"mmlu:{family_id}:{raw.upstream_split}:{digest}:{index}"
    return _row(
        row_id=row_id, source_id=source.source_id, family_id=family_id,
        repo_key=repo_key, identity_key=f"{repo_key}::{digest}",
        licence_id=source.declared_licence,
        request=_request(
            family_id=family_id, context=question,
            slots=(ChoiceSlot(name="answer", options=options),), example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="answer", value=gold_text),),
        dedupe_text="\n".join((question, *options)),
        metadata={
            "subject": subject,
            "upstream_split": raw.upstream_split,
            PINNED_SPLIT_KEY: str(pinned),
        },
    )


# -- tau/commonsense_qa ------------------------------------------------------


def rewrite_csqa(raw: CsqaRow, *, family_id: str, index: int, config: DataConfig) -> DataRow:
    """One CommonsenseQA question. The unlabelled test split is refused, counted."""
    source = source_by_id("tau/commonsense_qa")
    admit_licence(source.declared_licence, config=config.licence, source=source.source_id)
    if family_id != CSQA_FAMILY:
        raise RowRefused(
            reason_code="unknown_family_for_source", expected=CSQA_FAMILY,
            actual=family_id, detail=source.source_id,
        )
    if not raw.answer_key:
        raise RowRefused(
            reason_code="unlabelled", expected="an answerKey", actual="",
            detail=f"csqa {raw.qid!r}: the hub's test split carries no labels",
        )
    concept = _require_key(raw.concept, reason_code="missing_concept", what="question_concept")
    question = raw.question.strip()
    if not question:
        raise RowRefused(reason_code="empty_question", expected="a question", actual="")
    options = _checked_options(raw.texts, where=f"csqa {raw.qid!r}")
    gold_text = options[raw.labels.index(raw.answer_key)]

    # Pinned like MMLU (lead decision 2026-09-29): train trains, validation is val, and
    # the pinned split is part of the split unit so a concept never spans the boundary.
    pinned = source.pinned_split_of(raw.upstream_split)
    repo_key = f"csqa-{pinned}:{concept}"
    row_id = f"csqa:{family_id}:{raw.qid}"
    return _row(
        row_id=row_id, source_id=source.source_id, family_id=family_id,
        repo_key=repo_key, identity_key=f"{repo_key}::{raw.qid}",
        licence_id=source.declared_licence,
        request=_request(
            family_id=family_id, context=question,
            slots=(ChoiceSlot(name="answer", options=options),), example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="answer", value=gold_text),),
        dedupe_text="\n".join((question, *options)),
        metadata={
            "concept": concept,
            "upstream_split": raw.upstream_split,
            PINNED_SPLIT_KEY: str(pinned),
        },
    )


# -- clinc/clinc_oos, two-stage ----------------------------------------------


@dataclass(frozen=True, slots=True)
class ClincDomainMap:
    """Domain -> intents, validated so that both stages fit the 16-option slot.

    **The design.** CLINC150 is 150 intents in 10 domains of 15 (Larson et al.
    2019). A single ``choice`` over 150 cannot be asked: the slot holds 16. The
    existing ``intent.classification`` family answers that by sampling 15 random
    distractors, which makes the question easy in the wrong way -- a random
    distractor is almost never confusable with the gold, so the family mostly
    teaches topic detection. Two stages keep every option real and confusable:

    1. ``intent.domain``: the 10 domains, gold is the utterance's domain, and an
       out-of-scope utterance's gold is ``noul``.
    2. ``intent.within_domain``: the <= 16 intents of one domain. In-scope rows
       ask over their own domain. An out-of-scope row asks over a domain chosen
       deterministically from its own text, with gold ``noul``: abstention inside
       a plausible option set, which is the hard case stage 1 alone never trains.

    At serving time an app with more than 16 labels groups them the same way and
    asks twice from one prefilled context, which is the prefill-once /
    answer-many-slots shape the runtime is built for. The grouping is data, so it
    is validated rather than trusted: disjoint domains, 2..16 intents each, 2..16
    domains, and nothing named ``noul`` or the out-of-scope label.

    The mapping is **not hard-coded** here. It is read from upstream
    ``data/domains.json`` in ``clinc/oos-eval`` (fetched 2026-09-29, pinned by
    sha256); see :func:`load_clinc_domains`.
    """

    domains: Mapping[str, tuple[str, ...]]
    oos_label: str = "oos"

    def __post_init__(self) -> None:
        n = len(self.domains)
        if not 2 <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"stage 1 asks over the domains, so there must be 2..{MAX_CHOICE_OPTIONS}; "
                f"got {n}"
            )
        owner: dict[str, str] = {}
        for domain, intents in self.domains.items():
            for name in (domain, *intents):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"domain map contains an empty name under {domain!r}")
                if name.strip().casefold() in (NOUL, self.oos_label.casefold()):
                    raise ValueError(
                        f"{name!r} under {domain!r} is reserved: {NOUL!r} is in every option "
                        f"set and {self.oos_label!r} is the abstention, not a class"
                    )
            if not 2 <= len(intents) <= MAX_CHOICE_OPTIONS:
                raise ValueError(
                    f"domain {domain!r} has {len(intents)} intents; stage 2 asks over one "
                    f"domain, so it needs 2..{MAX_CHOICE_OPTIONS}"
                )
            for intent in intents:
                if intent in owner:
                    raise ValueError(
                        f"intent {intent!r} is in both {owner[intent]!r} and {domain!r}: "
                        "stage 1 would then have two correct answers"
                    )
                owner[intent] = domain

    def domain_of(self, intent: str) -> str | None:
        for domain, intents in self.domains.items():
            if intent in intents:
                return domain
        return None

    def unmapped(self, vocabulary: Sequence[str]) -> tuple[str, ...]:
        """In-scope intents of ``vocabulary`` the map does not place. Empty is good."""
        return tuple(
            sorted(
                i for i in set(vocabulary)
                if i != self.oos_label and self.domain_of(i) is None
            )
        )


def load_clinc_domains(
    path: Path,
    *,
    oos_label: str = "oos",
    expect_sha256: str | None = CLINC_DOMAINS_SHA256,
) -> ClincDomainMap:
    """Read a domain map shaped ``{"domain": ["intent", ...], ...}``.

    The shape was verified 2026-09-29 against the upstream file at
    :data:`CLINC_DOMAINS_COMMIT`: 10 domains of 15 intents, covering all 150 in-scope
    ``ClassLabel`` names of ``clinc/clinc_oos`` ``plus``. By default the bytes must
    also hash to :data:`CLINC_DOMAINS_SHA256`, so a different file -- another commit,
    a hand edit -- is refused rather than silently regrouping the intents. Pass
    ``expect_sha256=None`` only for a map that is deliberately not the upstream one.
    """
    size = path.stat().st_size
    if size > MAX_DOMAIN_MAP_BYTES:
        raise ValueError(
            f"{path}: {size} bytes exceeds {MAX_DOMAIN_MAP_BYTES}; refused rather than read"
        )
    body = path.read_bytes()
    if expect_sha256 is not None:
        got = hashlib.sha256(body).hexdigest()
        if got != expect_sha256:
            raise ValueError(
                f"{path}: sha256 {got} is not the pinned domain map {expect_sha256}"
            )
    obj = json.loads(body.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(obj).__name__}")
    domains: dict[str, tuple[str, ...]] = {}
    for domain, intents in obj.items():
        if not isinstance(intents, list) or not all(isinstance(i, str) for i in intents):
            raise ValueError(
                f"{path}: domain {domain!r} maps to {type(intents).__name__}, expected a "
                "list of intent names"
            )
        domains[domain] = tuple(intents)
    return ClincDomainMap(domains=domains, oos_label=oos_label)


def rewrite_clinc_two_stage(
    raw: ClincRow,
    *,
    family_id: str,
    index: int,
    config: DataConfig,
    domain_map: ClincDomainMap,
) -> DataRow:
    """One CLINC utterance as a stage-1 or a stage-2 question.

    ``repo_key`` and ``identity_key`` come from ``qd_data.mixture.clinc_keys``, the
    function ``qd_data.mixture.rewrite_clinc`` uses, so one utterance lands in one repo
    split whichever CLINC family asks about it.
    """
    source = source_by_id("clinc/clinc_oos")
    admit_licence(source.declared_licence, config=config.licence, source=source.source_id)
    refuse_benchmark_target_row(source.source_id, raw.upstream_split, config=config)
    utterance = raw.utterance.strip()
    if not utterance:
        raise RowRefused(
            reason_code="empty_utterance", expected="a non-empty utterance", actual="",
        )
    keys = clinc_keys(raw, utterance)
    repo_key, identity, digest = keys.repo_key, keys.identity_key, keys.digest
    row_id = f"clinc:{family_id}:{digest}:{index}"

    domain: str | None = None
    if not raw.is_oos:
        domain = domain_map.domain_of(raw.intent)
        if domain is None:
            raise RowRefused(
                reason_code="intent_not_in_domain_map",
                expected="an intent the domain map places", actual=raw.intent,
                detail="an unplaced intent has no stage-1 answer",
            )

    if family_id == CLINC_DOMAIN_FAMILY:
        options = tuple(sorted(domain_map.domains))
        gold = (
            GoldAnswer(slot_name="domain", value=None, is_noul=True)
            if domain is None
            else GoldAnswer(slot_name="domain", value=domain)
        )
        slot_name = "domain"
    elif family_id == CLINC_WITHIN_DOMAIN_FAMILY:
        if domain is None:
            names = sorted(domain_map.domains)
            asked = names[
                DeterministicRng(
                    "qd_data.general.clinc_oos_domain.v1", config.seed, digest
                ).below(len(names))
            ]
            gold = GoldAnswer(slot_name="intent", value=None, is_noul=True)
        else:
            asked = domain
            gold = GoldAnswer(slot_name="intent", value=raw.intent)
        options = tuple(sorted(domain_map.domains[asked]))
        slot_name = "intent"
    else:
        raise RowRefused(
            reason_code="unknown_family_for_source",
            expected=f"one of {CLINC_DOMAIN_FAMILY}, {CLINC_WITHIN_DOMAIN_FAMILY}",
            actual=family_id, detail=source.source_id,
        )

    options = _checked_options(options, where=f"clinc {family_id}")
    return _row(
        row_id=row_id, source_id=source.source_id, family_id=family_id,
        repo_key=repo_key, identity_key=identity, licence_id=source.declared_licence,
        request=_request(
            family_id=family_id, context=utterance,
            slots=(ChoiceSlot(name=slot_name, options=options),), example_id=row_id,
        ),
        gold=(gold,), dedupe_text=utterance,
        metadata={"oos": str(raw.is_oos)},
    )


# -- replay slice ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayPartition:
    """The general families' training rows, split into two roles that never overlap.

    ``gold_rows`` are trained with cross-entropy to their gold answer. ``replay_rows``
    are **replay-only**: ``qd_train.replay.PriorKLReplay`` trains them toward the
    base model's own cached answers (``KL(base || model)``) and never reads their
    gold. A row in both would be pulled toward its gold and toward the base's answer
    at once, and the replay term would stop meaning "stay near the prior". The
    invariant is checked at construction, so a partition that breaks it cannot exist.

    ``replay_rows`` go to the shard writer as their own shard set (``--replay-shards``),
    then through ``tools/replay_decontam.py`` for the attestation ``real_ft_run``
    requires. Each carries ``metadata[REPLAY_ROLE_KEY] == REPLAY_ONLY``.

    ``excluded_rows`` are training rows -- whether drawn for replay or for gold -- that a
    decontamination run against the val and held-out sets named contaminated. They are
    neither replayed nor gold-trained: a row near an evaluated row is not safe on
    either side, and moving it to the other role would trade one leak for another.
    Counted, never dropped silently -- :meth:`counts` carries them.
    """

    gold_rows: tuple[DataRow, ...]
    replay_rows: tuple[DataRow, ...]
    fraction: float
    excluded_rows: tuple[DataRow, ...] = ()

    def __post_init__(self) -> None:
        sides = (
            ("gold", self.gold_rows), ("replay", self.replay_rows),
            ("excluded", self.excluded_rows),
        )
        for i, (a, rows_a) in enumerate(sides):
            for b, rows_b in sides[i + 1:]:
                both = sorted({r.row_id for r in rows_a} & {r.row_id for r in rows_b})
                if both:
                    raise ValueError(
                        f"{len(both)} row(s) are both {a} and {b} (both gold-trained and "
                        f"replay-only is the case this exists for), first: {both[:3]}"
                    )
                shared = sorted(
                    {r.identity_key for r in rows_a} & {r.identity_key for r in rows_b}
                )
                if shared:
                    raise ValueError(
                        f"{len(shared)} identity key(s) have rows on both the {a} and the "
                        f"{b} side, first: {shared[:3]}"
                    )
        if any(r.metadata.get(REPLAY_ROLE_KEY) != REPLAY_ONLY for r in self.replay_rows):
            raise ValueError("a replay row is not marked replay-only")
        if any(REPLAY_ROLE_KEY in r.metadata for r in self.gold_rows):
            raise ValueError("a gold row carries a replay role")

    def counts(self) -> dict[str, dict[str, int]]:
        """``{family: {"gold": n, "replay": n, "excluded": n}}``, complete."""
        out: dict[str, dict[str, int]] = {}
        for role, rows in (
            ("gold", self.gold_rows), ("replay", self.replay_rows),
            ("excluded", self.excluded_rows),
        ):
            for r in rows:
                out.setdefault(r.family_id, {"gold": 0, "replay": 0, "excluded": 0})[role] += 1
        return dict(sorted(out.items()))


def partition_replay(
    train_rows: Sequence[DataRow],
    *,
    seed: int,
    fraction: float = DEFAULT_REPLAY_FRACTION,
    contaminated_identity_keys: frozenset[str] = frozenset(),
) -> ReplayPartition:
    """Mark a deterministic ``fraction`` of the general families' training rows replay-only.

    ``train_rows`` are rows already assigned to ``train`` -- a replay row drawn from
    ``val`` or ``heldout`` would put an evaluated row into training. Every row must
    belong to :data:`REPLAY_FAMILIES`, and an MMLU row must be pinned to ``train``.

    The draw is a keyed hash of ``(seed, identity_key)``, never a shuffle-and-slice:
    adding rows moves no existing row between roles, and rows sharing an identity
    key (exact duplicates) share a role. So the realised fraction is close to, not
    exactly, ``fraction``; :meth:`ReplayPartition.counts` reports what it was.

    ``contaminated_identity_keys`` is the rebuild half of refuse-on-hit, and **one rule
    covers both roles** (lead decision under the user's 2026-09-29 delegation, closing
    ``GAP-DATA-GENERAL-CE-ROWS-OVERLAP-FAMILY-VAL``): any row, replay-drawn or gold,
    that ``qd_train.replay.decontaminate`` finds at >= 0.5 8-gram containment against
    a val or held-out row moves to ``excluded_rows``. A contaminated gold row is not
    moved to replay; a contaminated replay row is not moved to gold. Nothing else
    moves, so both sides must be decontaminated again before use. A key that names
    no row here is refused -- it means the hits came from a different corpus.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"replay fraction must be in (0, 1), got {fraction}")
    gold: list[DataRow] = []
    replay: list[DataRow] = []
    excluded: list[DataRow] = []
    for row in train_rows:
        if row.family_id not in REPLAY_FAMILIES:
            raise ValueError(
                f"{row.row_id}: family {row.family_id!r} is not a replay family "
                f"({sorted(REPLAY_FAMILIES)})"
            )
        pinned = row.metadata.get(PINNED_SPLIT_KEY)
        if pinned is not None and pinned != "train":
            raise ValueError(
                f"{row.row_id}: pinned to {pinned!r}; replay is drawn from training rows only"
            )
        if REPLAY_ROLE_KEY in row.metadata:
            raise ValueError(f"{row.row_id}: already carries a replay role")
        rng = DeterministicRng("qd_data.general.replay.v1", seed, row.identity_key)
        u = rng.below(1 << 30) / float(1 << 30)
        if row.identity_key in contaminated_identity_keys:
            excluded.append(row)
        elif u >= fraction:
            gold.append(row)
        else:
            replay.append(
                replace(row, metadata={**row.metadata, REPLAY_ROLE_KEY: REPLAY_ONLY})
            )
    unmatched = contaminated_identity_keys - {r.identity_key for r in excluded}
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} contaminated identity key(s) name no training row here, "
            f"first: {sorted(unmatched)[:3]}; the hits came from a different corpus"
        )
    return ReplayPartition(
        gold_rows=tuple(gold), replay_rows=tuple(replay), fraction=fraction,
        excluded_rows=tuple(excluded),
    )
