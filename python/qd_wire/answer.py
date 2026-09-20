"""The answer half of the wire: ``AnswerEnvelope``, ``SlotAnswer`` and the two untagged unions.

``docs/schema-api.md`` shows the answer as one example object and then says, in
*"What this document still does not specify"*, that it pins no ``status``
discriminant, no field-presence rules, and no statement of whether ``value`` is
absent iff ``noul``. This module is those rules made executable from the Python
side, read out of ``crates/qd-runtime/src/schema.rs`` and pinned in
:mod:`qd_wire.contract`.

Two untagged unions need care, because ``#[serde(untagged)]`` discriminates on the
JSON *shape* and a reader that guesses differently from serde is a second
implementation of the format:

``SlotValue``
    a JSON string is a ``choice``, an integer is a ``score``, an object is a
    ``span``. Unambiguous in both directions.

``ConformalSet``
    an array of strings is ``Choices``, an array of integers is ``Scores`` — and
    ``[]`` matches **both**. serde takes the first variant that fits, which is
    ``Choices``; :data:`EMPTY_CONFORMAL_SET_READS_AS` records that this reading is
    inherited from variant order in the Rust enum rather than chosen here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from qd_wire.contract import STRUCT_FIELDS
from qd_wire.errors import WireParseError, check_keys
from qd_wire.values import as_bool, as_finite_float, as_str, as_uint

__all__ = [
    "EMPTY_CONFORMAL_SET_READS_AS",
    "AnswerEnvelope",
    "ChoiceSet",
    "ChoiceValue",
    "ConformalSet",
    "ScoreSet",
    "ScoreValue",
    "SlotAnswer",
    "SlotValue",
    "SpanValue",
    "parse_answer_envelope",
    "parse_conformal_set",
    "parse_slot_answer",
    "parse_slot_value",
]

#: How ``"conformal_set": []`` is read. Not a choice made here: ``ConformalSet`` is
#: ``#[serde(untagged)]`` with ``Choices(Vec<String>)`` declared first, and serde
#: takes the first variant that deserializes. An empty array fits both, so the
#: declaration order in ``schema.rs`` decides, and this constant names that fact so
#: a reader does not have to infer it. Reordering the Rust variants would silently
#: change what an empty set means on this side. The value is a :class:`ChoiceSet`
#: ``kind``, so it is comparable against a parsed set without a second vocabulary.
EMPTY_CONFORMAL_SET_READS_AS: Final[str] = "choice"

_SPAN_KEYS: Final[frozenset[str]] = frozenset(STRUCT_FIELDS["SpanValue"])
_SLOT_ANSWER_KEYS: Final[frozenset[str]] = frozenset(STRUCT_FIELDS["SlotAnswer"])
_ANSWER_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(STRUCT_FIELDS["AnswerEnvelope"])


@dataclass(frozen=True, slots=True)
class ChoiceValue:
    """``SlotValue::Choice`` — the option text, not a letter."""

    option: str

    kind: Literal["choice"] = "choice"


@dataclass(frozen=True, slots=True)
class ScoreValue:
    """``SlotValue::Score`` — an ordinal bin."""

    bin: int

    kind: Literal["score"] = "score"


@dataclass(frozen=True, slots=True)
class SpanValue:
    """``SlotValue::Span`` — ``{start_line, end_line}``, 1-based and inclusive.

    One of the repo's three spellings of a line span. ``qd_data.rows.GoldAnswer``
    spells a training-label span as the two-element array ``[41, 47]``, and nothing
    converts between the two. ``GAP-XLANG-SPAN-THREE-SPELLINGS``: this class is the
    runtime-answer spelling only, and deliberately does not accept the array form.

    ``start_line <= end_line`` is **not** enforced here, and the reason is worth
    being exact about. The runtime never *emits* an inverted span:
    ``crates/qd-runtime/src/answer.rs:420`` abstains when ``end.top < start.top``,
    on the stated ground that a backwards span "is not a low-confidence span, it is
    not a span". But that check lives in the **producer**. ``SpanValue`` derives its
    ``Deserialize`` with no ordering check, so the Rust *reader* accepts an inverted
    span, and the type itself carries no such invariant.

    So the ordering is a property of one code path, not a term of the format. This
    parser therefore reports it rather than refusing on it — refusing would make
    Python stricter than the Rust reader on a rule the contract never states, which
    is a second opinion about the format and exactly what the golden corpus exists
    to remove. ``GAP-XLANG-SPAN-BOUNDS-UNPINNED``.
    """

    start_line: int
    end_line: int

    kind: Literal["span"] = "span"

    @property
    def is_ordered(self) -> bool:
        """Whether the span runs forwards. Reported, never assumed."""
        return self.start_line <= self.end_line


SlotValue = ChoiceValue | ScoreValue | SpanValue


@dataclass(frozen=True, slots=True)
class ChoiceSet:
    """``ConformalSet::Choices``."""

    options: tuple[str, ...]

    kind: Literal["choice"] = "choice"


@dataclass(frozen=True, slots=True)
class ScoreSet:
    """``ConformalSet::Scores``."""

    bins: tuple[int, ...]

    kind: Literal["score"] = "score"


ConformalSet = ChoiceSet | ScoreSet


@dataclass(frozen=True, slots=True)
class SlotAnswer:
    """One slot's answer: ``{value, conformal_set, score, noul, degraded}``.

    The biconditional ``value is None`` iff ``noul`` is enforced in
    :func:`parse_slot_answer` and again here, so an instance built in Python cannot
    hold the contradiction either. ``crates/qd-runtime/src/schema.rs::
    SlotAnswer::contradiction`` is the same check on the other side.
    """

    value: SlotValue | None
    conformal_set: ConformalSet | None
    score: float
    noul: bool
    degraded: bool

    def __post_init__(self) -> None:
        why = self.contradiction()
        if why is not None:
            raise WireParseError("SlotAnswer", why)

    def contradiction(self) -> str | None:
        """Mirror of ``SlotAnswer::contradiction`` in ``schema.rs``."""
        if self.value is not None and self.noul:
            return (
                "carries a `value` and `noul`: true; `noul` means the model declined to "
                "answer, so there is nothing for `value` to hold"
            )
        if self.value is None and not self.noul:
            return (
                "carries no `value` and `noul`: false; an answer with neither a value nor "
                "an abstention says nothing, and a caller reading `value` would see the "
                "same bytes as an abstention without the flag that names it"
            )
        return None

    @property
    def value_kind(self) -> str | None:
        return None if self.value is None else self.value.kind

    @property
    def conformal_set_kind(self) -> str | None:
        return None if self.conformal_set is None else self.conformal_set.kind


@dataclass(frozen=True, slots=True)
class AnswerEnvelope:
    """``{"status": "ok", schema_version, backend, degraded, slots}``.

    Five keys reach a caller, not the two in the ``docs/schema-api.md`` example.
    ``backend`` is the one a caller written from that example would have missed, and
    it is the field that says an answer came from ``reference-deterministic-v1``
    rather than from a model.
    """

    schema_version: int
    backend: str
    degraded: bool
    slots: dict[str, SlotAnswer]

    status: Literal["ok"] = "ok"

    @property
    def all_slots_abstained(self) -> bool:
        return bool(self.slots) and all(s.noul for s in self.slots.values())


def parse_slot_value(raw: object, *, path: str) -> SlotValue:
    """Discriminate ``SlotValue`` the way ``#[serde(untagged)]`` does: by JSON shape."""
    if isinstance(raw, bool):
        raise WireParseError(
            path,
            "a JSON boolean is not a `SlotValue`. The three variants are a string "
            "(choice), an integer (score) and an object (span); Python's bool is an "
            "int subclass, so it is refused here rather than read as score bin 1.",
        )
    if isinstance(raw, str):
        return ChoiceValue(option=raw)
    if isinstance(raw, int):
        return ScoreValue(bin=as_uint(raw, path=path, rust_type="u32"))
    if isinstance(raw, dict):
        check_keys(raw, required=_SPAN_KEYS, path=path)
        return SpanValue(
            start_line=as_uint(raw["start_line"], path=f"{path}.start_line"),
            end_line=as_uint(raw["end_line"], path=f"{path}.end_line"),
        )
    raise WireParseError(
        path,
        "not a `SlotValue`: expected a string (choice), an integer (score) or an "
        f"object (span), got {type(raw).__name__}",
    )


def parse_conformal_set(raw: object, *, path: str) -> ConformalSet:
    """Discriminate ``ConformalSet`` by element type; ``[]`` follows serde's variant order."""
    if not isinstance(raw, list):
        raise WireParseError(
            path, f"expected a `ConformalSet` as a JSON array, got {type(raw).__name__}"
        )
    if not raw:
        return ChoiceSet(options=())
    if all(isinstance(v, str) for v in raw):
        return ChoiceSet(options=tuple(as_str(v, path=f"{path}[{i}]") for i, v in enumerate(raw)))
    if all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
        return ScoreSet(
            bins=tuple(as_uint(v, path=f"{path}[{i}]", rust_type="u32") for i, v in enumerate(raw))
        )
    raise WireParseError(
        path,
        "a `ConformalSet` is all strings (`Choices`) or all unsigned integers "
        f"(`Scores`); this array mixes {sorted({type(v).__name__ for v in raw})}. "
        "serde's untagged read would reject it too, so it is refused rather than "
        "coerced to whichever variant happens to fit more of the elements.",
    )


def parse_slot_answer(raw: object, *, path: str) -> SlotAnswer:
    if not isinstance(raw, dict):
        raise WireParseError(path, f"expected a `SlotAnswer` object, got {type(raw).__name__}")
    check_keys(raw, required=_SLOT_ANSWER_KEYS, path=path)
    value = raw["value"]
    conformal = raw["conformal_set"]
    return SlotAnswer(
        value=None if value is None else parse_slot_value(value, path=f"{path}.value"),
        conformal_set=(
            None
            if conformal is None
            else parse_conformal_set(conformal, path=f"{path}.conformal_set")
        ),
        score=as_finite_float(raw["score"], path=f"{path}.score"),
        noul=as_bool(raw["noul"], path=f"{path}.noul"),
        degraded=as_bool(raw["degraded"], path=f"{path}.degraded"),
    )


def parse_answer_envelope(raw: dict[str, object], *, path: str = "$") -> AnswerEnvelope:
    """Parse the body of a ``"status": "ok"`` response. ``status`` is consumed by the caller."""
    check_keys(raw, required=_ANSWER_ENVELOPE_KEYS, path=path)
    slots = raw["slots"]
    if not isinstance(slots, dict):
        raise WireParseError(f"{path}.slots", f"expected an object, got {type(slots).__name__}")
    return AnswerEnvelope(
        schema_version=as_uint(
            raw["schema_version"], path=f"{path}.schema_version", rust_type="u32"
        ),
        backend=as_str(raw["backend"], path=f"{path}.backend"),
        degraded=as_bool(raw["degraded"], path=f"{path}.degraded"),
        slots={
            name: parse_slot_answer(body, path=f"{path}.slots[{name!r}]")
            for name, body in slots.items()
        },
    )
