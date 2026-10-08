"""Licence policy. Default-deny, with the obligations recorded rather than lost.

The finding that shapes this module, and it is not the one the plan expected:

    ``bigcode/commitpackft``'s card says *"Each sample comes from a code
    repository with a permissive license"*, and then enumerates its ``license``
    field's thirteen values as::

        ['mit', 'artistic-2.0', 'isc', 'cc0-1.0', 'epl-1.0', 'mpl-2.0',
         'unlicense', 'unknown', 'apache-2.0', 'bsd-3-clause', 'agpl-3.0',
         'lgpl-2.1', 'bsd-2-clause']

    Three of those thirteen are not permissive: ``agpl-3.0`` is strong copyleft,
    ``lgpl-2.1`` is weak copyleft, and ``unknown`` is not a licence at all. Two
    more (``epl-1.0``, ``mpl-2.0``) are file-level copyleft.

So "the 13-value permissive set" is a description the source's own data contradicts.
A filter that trusts the prose and admits all thirteen puts AGPL-licensed and
unlicensed code into the training pool of an Apache-2.0 model. That is why the
allowlist here is a **deliberate subset**, and why filtering is **per row** rather
than per dataset -- the dataset is ``mit``, but its rows are not.

Three tiers, and the default is deny:

``ALLOW``
    Permissive or attribution-only. Admitted automatically.
``NEEDS_HUMAN_CALL``
    Copyleft, non-standard, or absent. Refused by default; a human admits one by
    naming it in :class:`LicenceConfig.admitted_by_human` **with a justification**,
    which is one line and is recorded in the manifest.
``DISQUALIFYING``
    Non-commercial. Cannot be enabled by configuration at all -- ``facebook/anli``
    is ``cc-by-nc-4.0``, and ``docs/plan-corrections.md`` calls that
    *"disqualifying, not an edge case"*. A tier a config can override is not a
    tier that stops anything.

Obligations (attribution, share-alike) are carried through to the manifest rather
than being collapsed into a yes/no, because ``rajpurkar/squad_v2`` is
``cc-by-sa-4.0`` and a model card that does not say so is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from .errors import LicenceRefused

__all__ = [
    "COMMITPACKFT_DECLARED_VALUES",
    "LICENCE_POLICY",
    "LicenceConfig",
    "LicencePolicy",
    "LicenceTier",
    "admit_licence",
    "classify",
    "is_non_commercial",
    "normalise_licence",
]


class LicenceTier(Enum):
    ALLOW = "allow"
    NEEDS_HUMAN_CALL = "needs_human_call"
    DISQUALIFYING = "disqualifying"


@dataclass(frozen=True, slots=True)
class LicencePolicy:
    licence_id: str
    tier: LicenceTier
    #: Carried to the manifest and the model card. Never collapsed away.
    obligations: tuple[str, ...] = ()
    note: str = ""


def _p(lic: str, tier: LicenceTier, *obligations: str, note: str = "") -> LicencePolicy:
    return LicencePolicy(licence_id=lic, tier=tier, obligations=tuple(obligations), note=note)


_A = LicenceTier.ALLOW
_H = LicenceTier.NEEDS_HUMAN_CALL
_D = LicenceTier.DISQUALIFYING

#: The thirteen values ``bigcode/commitpackft`` documents for its per-row
#: ``license`` field, verbatim and in the card's order. Recorded as data so a test
#: can assert that every one of them is classified rather than falling through to
#: an unrecognised-string path.
COMMITPACKFT_DECLARED_VALUES: Final[tuple[str, ...]] = (
    "mit", "artistic-2.0", "isc", "cc0-1.0", "epl-1.0", "mpl-2.0", "unlicense",
    "unknown", "apache-2.0", "bsd-3-clause", "agpl-3.0", "lgpl-2.1", "bsd-2-clause",
)

LICENCE_POLICY: Final[dict[str, LicencePolicy]] = {
    p.licence_id: p
    for p in (
        # -- permissive code licences ------------------------------------
        _p("mit", _A, "attribution"),
        _p("apache-2.0", _A, "attribution", note="includes an explicit patent grant"),
        _p("bsd-2-clause", _A, "attribution"),
        _p("bsd-3-clause", _A, "attribution"),
        _p("isc", _A, "attribution"),
        _p("cc0-1.0", _A, note="public-domain dedication; no obligation"),
        _p("unlicense", _A, note="public-domain dedication; no obligation"),
        # -- permissive content licences (the open task mixture) ---------
        _p("cc-by-3.0", _A, "attribution", note="clinc/clinc_oos"),
        _p("cc-by-4.0", _A, "attribution"),
        _p("cc-by-sa-3.0", _A, "attribution", "share-alike"),
        _p(
            "cc-by-sa-4.0", _A, "attribution", "share-alike",
            note=(
                "rajpurkar/squad_v2. ShareAlike propagates to redistributed derivatives; "
                "the obligation is carried per row so the model card can state it"
            ),
        ),
        # -- the repository owner's own text --------------------------------
        _p(
            "owner-granted", _A,
            note=(
                "the human's own repositories, approved for training, eval and commit data "
                "on 2026-09-28 (docs/train-plan-2026-09-28.md, Human decisions). Carried only "
                "by rows of qd_data.sources.OWN_REPOS_SOURCE_ID that survive v5's own-prose "
                "provenance rules (Fable's v5 review section 2.9); it covers text the owner "
                "authored, not text hosted in their repositories"
            ),
        ),
        # -- the v6 sources, ruled by the human on 2026-10-08 ----------------
        _p(
            "oanc", _A, "attribution",
            note=(
                "nyu-mll/multi_nli: the Open American National Corpus terms of its non-fiction "
                "genres. Genre 'fiction' carries other terms and is dropped at conversion. "
                "Allowed by the human on 2026-10-08"
            ),
        ),
        _p(
            "odc-by-1.0", _A, "attribution",
            note=(
                "allenai/scirepeval: Open Data Commons Attribution 1.0 over the aggregate; the "
                "attribution is carried to the model card. Allowed by the human on 2026-10-08"
            ),
        ),
        _p(
            "synthetic-by-rule", _A,
            note=(
                "rows written by rule by qd-prep synth (lappi/synth-*), from no person's data: "
                "the human's 'Synthetic only' ruling (2026-10-06) for the caller families, and "
                "the human's allowance of this licence id on 2026-10-08"
            ),
        ),
        # -- copyleft / non-standard / absent: default-deny --------------
        _p(
            "agpl-3.0", _H, "share-alike", "network-use-disclosure",
            note=(
                "strong copyleft with a network clause. Present in commitpackft's own "
                "declared value set despite the card's 'permissive' prose. Admitting it "
                "into an Apache-2.0 model's pool is a legal judgement, not a filter default"
            ),
        ),
        _p("lgpl-2.1", _H, "share-alike", note="weak copyleft; human call"),
        _p("epl-1.0", _H, "share-alike", note="file-level copyleft; human call"),
        _p("mpl-2.0", _H, "share-alike", note="file-level copyleft; human call"),
        _p(
            "artistic-2.0", _H, "attribution",
            note="OSI-approved but carries conditions on modified distribution; human call",
        ),
        _p(
            "unknown", _H,
            note=(
                "the source declares no licence. Absence of a licence is absence of "
                "permission, not permission. commitpackft emits this value per row"
            ),
        ),
        _p(
            "other", _H,
            note=(
                "nyu-mll/glue. The card defers to each upstream task's own terms, so "
                "there is no single licence to check. Needs a human call per task"
            ),
        ),
        _p(
            "c-uda", _H,
            note=(
                "Computational Use of Data Agreement (Devign rehost). Computational use "
                "only; not a standard OSS licence"
            ),
        ),
        # -- non-commercial: never admissible ----------------------------
        _p(
            "cc-by-nc-4.0", _D, "attribution", "non-commercial",
            note=(
                "facebook/anli, lmsys/toxic-chat, Tobi-Bueck/customer-support-tickets. "
                "Non-commercial is disqualifying, not an edge case"
            ),
        ),
        _p("cc-by-nc-sa-4.0", _D, "attribution", "non-commercial", "share-alike"),
        _p("cc-by-nc-nd-4.0", _D, "attribution", "non-commercial", "no-derivatives"),
        _p(
            "cc-by-nc-3.0", _D, "attribution", "non-commercial",
            note="allenai/sciq, which RSI's own replay set includes. Disqualifying",
        ),
    )
}

#: Spellings seen in the wild that denote a licence already in the table. Kept
#: small and explicit: a fuzzy normaliser that guesses would eventually map an
#: unrecognised licence onto a permissive one, which is the failure this whole
#: module exists to prevent.
_ALIASES: Final[dict[str, str]] = {
    "apache 2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache2.0": "apache-2.0",
    "bsd-2": "bsd-2-clause",
    "bsd-3": "bsd-3-clause",
    "cc0": "cc0-1.0",
    "cc-0": "cc0-1.0",
    "mit license": "mit",
    "the unlicense": "unlicense",
    "public domain": "cc0-1.0",
}


def normalise_licence(raw: str) -> str:
    """Casefold and trim. Deliberately does no fuzzy matching."""
    if not isinstance(raw, str):
        raise TypeError(f"licence must be str, got {type(raw).__name__}")
    key = " ".join(raw.strip().casefold().split())
    return _ALIASES.get(key, key)


def is_non_commercial(licence_id: str) -> bool:
    """True when the id carries a non-commercial segment.

    Segment-exact rather than a substring test: ``"nc" in "unlicense"`` is true and
    would be a catastrophic false positive.
    """
    return "nc" in normalise_licence(licence_id).split("-")


def classify(raw: str) -> LicencePolicy:
    """Classify a licence string. An unrecognised string is never admitted.

    An id nobody has classified is exactly the case where guessing is most
    tempting and most wrong, so it lands in ``NEEDS_HUMAN_CALL`` -- unless it looks
    non-commercial, in which case it is disqualifying outright.
    """
    key = normalise_licence(raw)
    known = LICENCE_POLICY.get(key)
    if known is not None:
        return known
    if is_non_commercial(key):
        return LicencePolicy(
            licence_id=key, tier=LicenceTier.DISQUALIFYING,
            obligations=("non-commercial",),
            note="unrecognised id carrying a non-commercial segment",
        )
    return LicencePolicy(
        licence_id=key, tier=LicenceTier.NEEDS_HUMAN_CALL,
        note="licence id is not in the policy table; classify it before admitting it",
    )


@dataclass(frozen=True, slots=True)
class LicenceConfig:
    """Which licences this run admits.

    ``admitted_by_human`` maps a licence id to the justification for admitting it.
    A justification is required: an override with no recorded reason is
    indistinguishable from an accident, and this is the setting most likely to be
    flipped under deadline pressure.
    """

    admitted_by_human: dict[str, str] = field(default_factory=dict)
    #: Source ids a human has opted in to, each with its justification. For a source
    #: whose licence tier is ``ALLOW`` but which is still off by default -- the case
    #: is ``allenai/ai2_arc``: ``cc-by-sa-4.0`` is admitted for ``rajpurkar/squad_v2``,
    #: so the tier cannot be the switch, and a second share-alike source in an
    #: Apache-2.0 model's pool is a decision, not a filter default. Keyed by source,
    #: never by licence, so opting one in cannot silently admit another.
    admitted_sources_by_human: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for source_id, why in self.admitted_sources_by_human.items():
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(f"source opt-in names no source: {source_id!r}")
            if not isinstance(why, str) or not why.strip():
                raise ValueError(
                    f"source opt-in {source_id!r} carries no justification. An opt-in "
                    "without a recorded reason cannot be distinguished from an accident."
                )
        for lic, why in self.admitted_by_human.items():
            key = normalise_licence(lic)
            if not isinstance(why, str) or not why.strip():
                raise ValueError(
                    f"licence override {lic!r} carries no justification. An override "
                    "without a recorded reason cannot be distinguished from an accident."
                )
            policy = classify(key)
            if policy.tier is LicenceTier.DISQUALIFYING:
                raise LicenceRefused(
                    expected="a licence that is not non-commercial", actual=key,
                    detail=(
                        f"{key!r} is disqualifying and cannot be admitted by configuration. "
                        f"{policy.note}"
                    ),
                )

    def admits(self, licence_id: str) -> bool:
        return normalise_licence(licence_id) in {
            normalise_licence(k) for k in self.admitted_by_human
        }

    def admits_source(self, source_id: str) -> bool:
        """Exact match on the source id. No normalisation: ids are case-sensitive
        upstream, and a fuzzy match is how opting in one source admits another."""
        return source_id in self.admitted_sources_by_human


DEFAULT_LICENCE_CONFIG: Final[LicenceConfig] = LicenceConfig()


def admit_licence(
    raw: str,
    *,
    config: LicenceConfig = DEFAULT_LICENCE_CONFIG,
    source: str = "<unnamed>",
) -> LicencePolicy:
    """Return the policy for an admitted licence, or raise :class:`LicenceRefused`.

    The refusal message always names the licence, as the lane brief requires.
    """
    policy = classify(raw)
    if policy.tier is LicenceTier.ALLOW:
        return policy
    if policy.tier is LicenceTier.DISQUALIFYING:
        raise LicenceRefused(
            expected="a licence on the permissive allowlist", actual=policy.licence_id,
            detail=(
                f"{source}: licence {policy.licence_id!r} is disqualifying and cannot be "
                f"admitted by configuration. {policy.note}"
            ),
        )
    if config.admits(policy.licence_id):
        return policy
    raise LicenceRefused(
        expected="a licence on the permissive allowlist", actual=policy.licence_id,
        detail=(
            f"{source}: licence {policy.licence_id!r} needs a human call and is not in "
            f"admitted_by_human. {policy.note or 'no policy note recorded'}"
        ),
    )
