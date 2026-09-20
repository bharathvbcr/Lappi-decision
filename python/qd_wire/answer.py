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

from qd_wire.contract import MAX_SLOT_NAME_BYTES, STRUCT_FIELDS
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
    "check_slot_name",
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

    ``1 <= start_line <= end_line`` is a **term of the format**, enforced here and in
    ``crates/qd-runtime/src/schema.rs::SpanValue``'s ``TryFrom<SpanValueWire>``.

    It was not, and this class previously said so: it parsed an inverted span and
    exposed ``is_ordered`` instead of refusing, because the Rust *reader* accepted one
    too and making Python stricter than Rust on a rule nothing stated would have been
    a second opinion about the format. The rule was real but lived in exactly one
    place — the **producer**, which abstains when the end pointer decodes below the
    start one — so every reader was trusting a check it could not see.

    Both sides now carry the invariant, so ``is_ordered`` is gone rather than left
    behind always returning ``True``. ``GAP-XLANG-SPAN-BOUNDS-UNPINNED``.
    """

    start_line: int
    end_line: int

    kind: Literal["span"] = "span"


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
        start = as_uint(raw["start_line"], path=f"{path}.start_line")
        end = as_uint(raw["end_line"], path=f"{path}.end_line")
        # The same two bounds `SpanValue::try_from` applies in `schema.rs`, with the same reasons.
        # Enforced here rather than reported, because they are now a term of the format on both
        # sides. `GAP-XLANG-SPAN-BOUNDS-UNPINNED`.
        if start == 0:
            raise WireParseError(
                path,
                "span starts at line 0, but a span is 1-based and inclusive, so line 0 "
                "does not exist",
            )
        if start > end:
            raise WireParseError(
                path,
                f"span runs backwards: start_line {start} is after end_line {end}. A "
                "backwards span is not a low-confidence span, it is not a span; it is "
                "refused rather than silently reordered into a plausible-looking answer",
            )
        return SpanValue(start_line=start, end_line=end)
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
            check_slot_name(name, path=f"{path}.slots"): parse_slot_answer(
                body, path=f"{path}.slots[{name!r}]"
            )
            for name, body in slots.items()
        },
    )


def check_slot_name(name: object, *, path: str) -> str:
    """Hold an answer's slot key to the same rule the request side holds ``slots[i].name`` to.

    Returns the name so it can be used inline where the key is built.

    The Rust reader enforces this in ``schema.rs::deserialize_slot_map`` and the request side in
    ``wire::parse_slot``, all three from the one number
    :data:`qd_wire.contract.MAX_SLOT_NAME_BYTES`, which is generated from
    ``crates/qd-runtime/src/schema.rs`` rather than typed here. Enforcing on one reader and not the
    other would recreate ``GAP-RT-SLOT-NAME-UNCAPPED`` in mirror image: a cap only one side keeps
    is not a cap.

    The cap is on **UTF-8 bytes**, not characters, because that is what the Rust ``str::len`` and
    ``MAX_PAYLOAD_BYTES`` are denominated in. ``len(name)`` would accept a 256-character name that
    Rust refuses at 512 bytes.
    """
    if not isinstance(name, str):
        raise WireParseError(path, f"a slot key must be a string, got {type(name).__name__}")
    if not name.strip():
        raise WireParseError(
            path,
            "the answer's `slots` map has an empty key; a slot answer that cannot be keyed "
            "cannot be matched to the slot that was asked",
        )
    size = len(name.encode("utf-8"))
    if size > MAX_SLOT_NAME_BYTES:
        raise WireParseError(
            path,
            f"a key of the answer's `slots` map is {size} bytes, over the slot-name cap of "
            f"{MAX_SLOT_NAME_BYTES}; no request carrying such a name can be accepted, so no "
            "answer may carry one back",
        )
    return name
