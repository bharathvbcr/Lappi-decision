"""The dataset registry, and the task families the mixture is built from.

Every row here is a finding from ``AUDIT/external-surfaces.md`` recorded as data,
so the admission decision is computed rather than remembered. Two independent
reasons a source can fail, and they are kept separate on purpose:

* **Licence** -- resolved by :mod:`qd_data.licences`, default-deny.
* **Reachability** -- gated, script-only, or not on the host at all. A source can
  be perfectly licensed and still impossible to load.

Collapsing the two would make ``facebook/anli`` (non-commercial, loads fine) look
like ``PolyAI/banking77`` (fine licence, will not load under ``datasets>=3.0``),
and those need different responses: one is closed, the other is a human action.

**Nothing in this module blocks on ``nuprl/AgentPack``.** It is registered with
its gate recorded so it slots in when a human accepts the terms, and it is absent
from the admitted set until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from .licences import (
    DEFAULT_LICENCE_CONFIG,
    LicenceConfig,
    LicencePolicy,
    LicenceTier,
    classify,
)

__all__ = [
    "SOURCES",
    "TASK_FAMILIES",
    "Reachability",
    "SlotKind",
    "Source",
    "TaskFamily",
    "admitted_sources",
    "admitted_task_families",
    "refusal_report",
    "source_by_id",
    "task_family_by_id",
]


class Reachability(Enum):
    """Can this source actually be loaded, today, unattended?"""

    LOADABLE = "loadable"
    GATED = "gated"
    SCRIPT_ONLY = "script_only"
    NOT_ON_HOST = "not_on_host"


class SlotKind(Enum):
    CHOICE = "choice"
    SCORE = "score"
    SPAN = "span"


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    host: str
    declared_licence: str
    reachability: Reachability
    #: True when the rows carry their own licence, so filtering can be per row.
    per_row_licence_field: bool
    evidence: str

    @property
    def licence_policy(self) -> LicencePolicy:
        return classify(self.declared_licence)

    def admission_refusals(self, config: LicenceConfig) -> tuple[str, ...]:
        """Every reason this source is not admitted. Empty tuple means admitted."""
        out: list[str] = []
        policy = self.licence_policy
        if policy.tier is LicenceTier.DISQUALIFYING:
            out.append(
                f"licence {policy.licence_id!r} is disqualifying (non-commercial); "
                f"{policy.note}"
            )
        elif policy.tier is LicenceTier.NEEDS_HUMAN_CALL and not config.admits(policy.licence_id):
            out.append(
                f"licence {policy.licence_id!r} needs a human call and is not in "
                f"admitted_by_human; {policy.note or 'no policy note'}"
            )
        if self.reachability is Reachability.GATED:
            out.append(
                "gated: a human must accept the dataset's terms before any unattended "
                "fetch succeeds"
            )
        elif self.reachability is Reachability.SCRIPT_ONLY:
            out.append(
                "script-only: no data files on the host, and loading scripts fail under "
                "datasets>=3.0"
            )
        elif self.reachability is Reachability.NOT_ON_HOST:
            out.append("not present on the declared host")
        return tuple(out)


SOURCES: Final[dict[str, Source]] = {
    s.source_id: s
    for s in (
        Source(
            source_id="bigcode/commitpackft",
            host="huggingface",
            declared_licence="mit",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=True,
            evidence=(
                "702,062 rows / 1,545.02 MB, license: mit, not gated. Documents a per-row "
                "`license` field over 13 values plus a `repos` field. NOTE: three of those "
                "13 values (agpl-3.0, lgpl-2.1, unknown) are not permissive despite the "
                "card's prose, so rows are filtered individually -- see qd_data.licences"
            ),
        ),
        Source(
            source_id="nuprl/AgentPack",
            host="huggingface",
            declared_licence="apache-2.0",
            reachability=Reachability.GATED,
            per_row_licence_field=False,
            evidence=(
                'gated: "auto"; authenticated fetch returns 403 "not in the authorized '
                'list". 62.91 GB compressed. The per-row source-repo licence field the '
                "filter design wanted is UNVERIFIED and the authors' paper describes no "
                "licence-based filtering, so per_row_licence_field is recorded False. "
                "Registered so it slots in when a human accepts the terms; nothing "
                "depends on it"
            ),
        ),
        Source(
            source_id="clinc/clinc_oos",
            host="huggingface",
            declared_licence="cc-by-3.0",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=False,
            evidence=(
                "not gated, 3 configs; `plus` config = 15,250 / 3,100 / 5,500. Its "
                "out-of-scope class is a natural, free `noul` supervision signal"
            ),
        ),
        Source(
            source_id="rajpurkar/squad_v2",
            host="huggingface",
            declared_licence="cc-by-sa-4.0",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=False,
            evidence=(
                "not gated, 130,319 / 11,873, 128.4 MB. ShareAlike: the obligation is "
                "carried per row to the manifest so the model card can state it. Its "
                "unanswerable questions are a second free `noul` signal"
            ),
        ),
        # -- registered so the refusal is explicit and computed, not folklore --
        Source(
            source_id="facebook/anli",
            host="huggingface",
            declared_licence="cc-by-nc-4.0",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=False,
            evidence="169,265 rows, not gated, but NON-COMMERCIAL. Disqualifying",
        ),
        Source(
            source_id="mteb/stsbenchmark-sts",
            host="huggingface",
            declared_licence="unknown",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=False,
            evidence="8,628 rows, not gated, licence literally the string 'unknown'",
        ),
        Source(
            source_id="nyu-mll/glue",
            host="huggingface",
            declared_licence="other",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=False,
            evidence=(
                "licence 'other'; the card defers to each upstream task's own terms, so "
                "there is no single licence to check"
            ),
        ),
        Source(
            source_id="google/code_x_glue_cc_defect_detection",
            host="huggingface",
            declared_licence="c-uda",
            reachability=Reachability.LOADABLE,
            per_row_licence_field=False,
            evidence="Devign rehost, 27,318 rows, C-UDA: computational use only",
        ),
        Source(
            source_id="PolyAI/banking77",
            host="huggingface",
            declared_licence="cc-by-4.0",
            reachability=Reachability.SCRIPT_ONLY,
            per_row_licence_field=False,
            evidence=(
                "licence is fine; the repo contains banking77.py and no data files, and "
                'dataset-server returns HTTP 500 "Dataset scripts are no longer supported"'
            ),
        ),
        Source(
            source_id="AmazonScience/massive",
            host="huggingface",
            declared_licence="cc-by-4.0",
            reachability=Reachability.SCRIPT_ONLY,
            per_row_licence_field=False,
            evidence="same shape as banking77: massive.py, no data files, dataset-server 500",
        ),
        Source(
            source_id="microsoft/CodeReviewer",
            host="huggingface",
            declared_licence="unknown",
            reachability=Reachability.NOT_ON_HOST,
            per_row_licence_field=False,
            evidence=(
                "404 with valid auth, control-tested. The data is on Zenodo, DOI "
                "10.5281/zenodo.6900648, cc-by-4.0, 4.82 GB -- a different host and a "
                "different acquisition step, so it is not an HF source at all"
            ),
        ),
    )
}


@dataclass(frozen=True, slots=True)
class TaskFamily:
    """A question *shape*, not a dataset.

    The holdout is over families rather than datasets because the gate is that the
    model abstains on a question it has never been asked -- ``docs/schema-api.md``:
    *the held-out task families must go to ``noul``*. Two families can share a
    source and still be different questions.
    """

    family_id: str
    source_id: str
    slot_kind: SlotKind
    description: str

    @property
    def source(self) -> Source:
        return SOURCES[self.source_id]


TASK_FAMILIES: Final[dict[str, TaskFamily]] = {
    f.family_id: f
    for f in (
        TaskFamily(
            family_id="code.commit_intent",
            source_id="bigcode/commitpackft",
            slot_kind=SlotKind.CHOICE,
            description="Does the diff implement what the commit message claims?",
        ),
        TaskFamily(
            family_id="code.language_id",
            source_id="bigcode/commitpackft",
            slot_kind=SlotKind.CHOICE,
            description="Which language is the changed file written in?",
        ),
        TaskFamily(
            family_id="code.change_scope",
            source_id="bigcode/commitpackft",
            slot_kind=SlotKind.SCORE,
            description="How large is the change, on an ordinal scale?",
        ),
        TaskFamily(
            family_id="intent.classification",
            source_id="clinc/clinc_oos",
            slot_kind=SlotKind.CHOICE,
            description="Which intent does the utterance express?",
        ),
        TaskFamily(
            family_id="intent.in_scope",
            source_id="clinc/clinc_oos",
            slot_kind=SlotKind.CHOICE,
            description="Is the utterance in scope for the assistant at all?",
        ),
        TaskFamily(
            family_id="qa.answerability",
            source_id="rajpurkar/squad_v2",
            slot_kind=SlotKind.CHOICE,
            description="Does the passage contain an answer to the question?",
        ),
        TaskFamily(
            family_id="qa.answer_span",
            source_id="rajpurkar/squad_v2",
            slot_kind=SlotKind.SPAN,
            description="Which lines of the passage contain the answer?",
        ),
    )
}


def source_by_id(source_id: str) -> Source:
    try:
        return SOURCES[source_id]
    except KeyError as exc:
        raise KeyError(
            f"unregistered source {source_id!r}; a source must be registered with its "
            f"licence and reachability before it can be loaded. Known: {sorted(SOURCES)}"
        ) from exc


def task_family_by_id(family_id: str) -> TaskFamily:
    try:
        return TASK_FAMILIES[family_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown task family {family_id!r}; known: {sorted(TASK_FAMILIES)}"
        ) from exc


def admitted_sources(config: LicenceConfig = DEFAULT_LICENCE_CONFIG) -> tuple[Source, ...]:
    return tuple(s for s in SOURCES.values() if not s.admission_refusals(config))


def admitted_task_families(
    config: LicenceConfig = DEFAULT_LICENCE_CONFIG,
) -> tuple[TaskFamily, ...]:
    ok = {s.source_id for s in admitted_sources(config)}
    return tuple(f for f in TASK_FAMILIES.values() if f.source_id in ok)


def refusal_report(config: LicenceConfig = DEFAULT_LICENCE_CONFIG) -> dict[str, tuple[str, ...]]:
    """Every refused source and every reason, for the handoff and the model card."""
    return {
        s.source_id: reasons
        for s in SOURCES.values()
        if (reasons := s.admission_refusals(config))
    }
