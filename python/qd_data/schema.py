"""The typed request: the one object shared by training and serving.

``docs/schema-api.md`` is the contract. The rule that shapes this module:

    The training format and the serving format are the same object. If they
    drift, the model is served a prompt shape it never saw, and nothing in the
    eval catches it.

So there is exactly one :class:`Request` type. The training pipeline builds one,
the serving path parses one off the wire, and :mod:`qd_data.render` is the single
renderer both go through. ``tests/test_render.py`` asserts the two paths produce
byte-identical prompts.

Structural validity is enforced in ``__post_init__``: an invalid request cannot be
constructed at all, so no downstream code has to remember to check. Byte caps are
checked at render time, where the caps are known.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Self

from .errors import (
    NOUL,
    BinsOutOfRangeRefusal,
    ContextLengthMismatchRefusal,
    ContextLenMissingRefusal,
    ContextNotBase64Refusal,
    ContextNotBytesRefusal,
    DuplicateOptionRefusal,
    DuplicateSlotNameRefusal,
    EmptyOptionRefusal,
    EmptySlotsRefusal,
    EmptyTaskRefusal,
    MalformedRequestRefusal,
    ReservedOptionNameRefusal,
    TooManyOptionsRefusal,
    UnknownSchemaVersionRefusal,
    UnknownSlotTypeRefusal,
)

__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_BINS",
    "MAX_SLOTS",
    "MIN_SCORE_BINS",
    "NOUL_LETTER",
    "OPTION_LETTERS",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "WIRE_REQUEST_KEYS",
    "ChoiceSlot",
    "HashExpectation",
    "Request",
    "Route",
    "ScoreSlot",
    "Slot",
    "SpanSlot",
    "canonical_json",
    "decode_context",
    "encode_context",
    "slot_set_digest",
]

SCHEMA_VERSION: Final[int] = 1
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({1})

#: The 16 letters the generic route decodes over -- the option letters themselves,
#: which is not the width of the slice they decode through. ``docs/schema-api.md``
#: gives the slice as **17** rows: 16 option letters plus one reserved ``noul`` row.
#: The phrase "the 16 letter-token slice of ``lm_head``" appears there only inside
#: the list of three plan constraints that section exists to reject, so this comment
#: used to cite the document for the reading the document rules out.
OPTION_LETTERS: Final[str] = "ABCDEFGHIJKLMNOP"

#: ``noul`` is "a reserved letter present in **every** option set". It is given a
#: letter *outside* the contiguous A-P block so that the option rows stay exactly 16
#: even when a request uses all 16 named options.
#:
#: GAP-SCHEMA-NOUL-LETTER-BUDGET is SETTLED, the way this module already read it:
#: 16 named option rows plus one reserved ``noul`` row, so a maximal choice slot is
#: 17 rows. The runtime lane answered on the doc comment of ``NOUL_LETTER`` in
#: ``crates/qd-runtime/src/render.rs``; ``docs/schema-api.md`` carries the resolution
#: under "The letter budget"; and ``crates/qd-runtime/tests/wire_context_crosslang.rs
#: ::the_option_and_bin_bounds_are_the_same_number_on_both_sides`` pins the constant
#: below to ``schema.rs::MAX_OPTIONS`` by importing this module at test time.
#:
#: This note used to prescribe the losing branch -- "set ``MAX_CHOICE_OPTIONS = 15``
#: here, one line, and every test follows" -- and both halves were wrong. The
#: direction is settled against it, and the cost is not one line: measured
#: 2026-09-19, 15 here stops pytest during collection with 12 errors (the
#: ``MAX_SCORE_BINS <= MAX_CHOICE_OPTIONS`` assert below is the first to go) and
#: fails 8 cross-language tests. Nothing follows the change; the repository refuses
#: it. A maintainer who needs to move this number moves ``schema.rs::MAX_OPTIONS``
#: first, because that constant is the owner and these are its mirrors.
NOUL_LETTER: Final[str] = "Z"

MAX_CHOICE_OPTIONS: Final[int] = 16
MIN_SCORE_BINS: Final[int] = 2
MAX_SCORE_BINS: Final[int] = 16

#: Bounded fan-out: a request with an unbounded number of slots is an unbounded
#: number of decode passes over one prefill.
MAX_SLOTS: Final[int] = 32

#: Every top-level key a request may carry, and the whole of it.
#:
#: The mirror of ``REQUEST_KEYS`` in ``crates/qd-runtime/src/wire.rs``, which
#: ``known_keys`` enforces there. Until 2026-09-19 this lane had no such list and
#: :meth:`Request.from_wire` read through ``dict.get``, so an unknown field was a hard
#: refusal in serving and a silently defaulted field in training -- the same typo
#: diagnosed by one lane and not the other. ``GAP-XLANG-UNKNOWN-FIELD-LENIENCY``.
#:
#: It is **not** verified by having been typed carefully:
#: ``python/tests/test_qd_data_wire_agreement.py`` parses the Rust declaration out of
#: ``wire.rs`` on every run and fails naming the field that drifted. Two of this
#: session's defects were a transcribed constant, which is why the comparison is a test
#: rather than a comment. The extractor raises rather than returning a short list, so a
#: parse failure cannot read as agreement.
#:
#: ``context`` is deliberately absent: it is the retired wire form, and both lanes name
#: it before this check so a caller still sending it is told what replaced it.
WIRE_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "task",
        "context_b64",
        "context_len",
        "question",
        "slots",
        "route",
        "expect",
        # Accepted and ignored by the runtime; emitted by this lane to steer the
        # *training* option shuffle only.
        "example_id",
        "metadata",
    }
)

Route = Literal["generic", "registered"]
_ROUTES: Final[frozenset[str]] = frozenset({"generic", "registered"})

assert len(OPTION_LETTERS) == MAX_CHOICE_OPTIONS, "letter budget and option cap must agree"
assert NOUL_LETTER not in OPTION_LETTERS, "the noul letter must not collide with an option letter"
assert MAX_SCORE_BINS <= MAX_CHOICE_OPTIONS, "score bins decode over the same letter slice"


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# -- the one wire encoding of `context` --------------------------------------
#
# ``docs/schema-api.md``: ``context_b64`` (base64, ASCII) plus a required
# ``context_len`` (the decoded byte count); they must agree or it is a typed
# refusal. These two functions are the single owner of that encoding on this side,
# and ``crates/qd-runtime/src/context.rs`` is the single owner on the other.
# ``crates/qd-runtime/tests/wire_context_crosslang.rs`` drives this encoder into
# that decoder rather than assuming the two agree.


def encode_context(data: bytes) -> str:
    """Canonical standard base64 of the exact bytes. Nothing normalizes them."""
    return base64.b64encode(data).decode("ascii")


def decode_context(blob: object, declared_len: object) -> bytes:
    """Decode the one accepted wire form, or raise the typed refusal that names why.

    ``base64.b64decode(validate=True)`` checks the alphabet but **not** the bits a
    partial quantum leaves unused, so it accepts both ``Zg==`` and ``Zh==`` for
    ``b"f"``. Two distinct strings naming one context is the kind of "undefined"
    this contract exists to remove, and ``qd-runtime``'s decoder rejects the
    non-canonical spelling -- so this one re-encodes and compares, and the two
    lanes are strict in the same place. The re-encode is also what catches padding
    a permissive decoder would have inferred.
    """
    if not isinstance(blob, str):
        raise ContextNotBytesRefusal(
            expected="a base64 str in 'context_b64'",
            actual="absent" if blob is None else type(blob).__name__,
            detail=(
                "context crosses the boundary as bytes and a length; a str context would "
                "invite a Unicode-normalizing round trip that moves token boundaries"
            ),
        )
    try:
        data = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContextNotBase64Refusal(
            expected="canonical standard base64", actual=blob[:32], detail=str(exc),
        ) from exc
    canonical = base64.b64encode(data).decode("ascii")
    if canonical != blob:
        raise ContextNotBase64Refusal(
            expected=canonical[:32],
            actual=blob[:32],
            detail=(
                "non-canonical base64: it decodes, but re-encoding the bytes does not "
                "reproduce it, so two strings would name one context"
            ),
        )
    if declared_len is None:
        raise ContextLenMissingRefusal(
            expected="a 'context_len' alongside 'context_b64'", actual=None,
            detail=(
                "there is no default: a payload whose length nobody declared is a payload "
                "whose truncation nobody would notice"
            ),
        )
    if isinstance(declared_len, bool) or not isinstance(declared_len, int):
        raise ContextLenMissingRefusal(
            expected="an int 'context_len'", actual=type(declared_len).__name__,
            detail="a length that is not an integer cannot be compared with a byte count",
        )
    if declared_len < 0:
        # Not a *mismatch*: a negative count is not a length at all, and reporting it
        # as a mismatch would put a number that cannot be a byte count on the
        # "expected" side. ``qd-runtime`` reads ``context_len`` as an unsigned
        # integer and lands in the same refusal, which is what keeps the identifier
        # a caller branches on the same in both lanes.
        raise ContextLenMissingRefusal(
            expected="a non-negative int 'context_len'", actual=declared_len,
            detail="a negative byte count is not a length",
        )
    if declared_len != len(data):
        raise ContextLengthMismatchRefusal(
            expected=declared_len, actual=len(data),
            detail=(
                "declared context_len does not match the decoded byte count; a payload "
                "truncated on a quantum boundary still decodes cleanly, just to fewer bytes"
            ),
        )
    return data


@dataclass(frozen=True, slots=True)
class Slot:
    """Base slot. Never constructed directly; see the three concrete types."""

    name: str

    def _check_name(self) -> None:
        if not self.name or not self.name.strip():
            raise UnknownSlotTypeRefusal(
                expected="a non-empty slot name", actual=self.name,
                detail="an unnamed slot cannot be keyed in the answer map",
            )

    @property
    def type_name(self) -> str:
        raise NotImplementedError

    def to_wire(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ChoiceSlot(Slot):
    """One of k <= 16 named options, or ``noul``."""

    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._check_name()
        n = len(self.options)
        if n == 0:
            raise TooManyOptionsRefusal(
                expected=f"1..{MAX_CHOICE_OPTIONS} options", actual=0,
                detail=f"choice slot {self.name!r} has no options",
            )
        if n > MAX_CHOICE_OPTIONS:
            raise TooManyOptionsRefusal(
                expected=f"at most {MAX_CHOICE_OPTIONS} options", actual=n,
                detail=(
                    f"choice slot {self.name!r}: the {MAX_CHOICE_OPTIONS}-row lm_head slice "
                    f"cannot express option {n}; dropping one would change the question"
                ),
            )
        for opt in self.options:
            if not isinstance(opt, str):
                raise UnknownSlotTypeRefusal(
                    expected="str option", actual=type(opt).__name__,
                    detail=f"choice slot {self.name!r}",
                )
            if not opt.strip():
                raise EmptyOptionRefusal(
                    expected="a non-empty option", actual=opt,
                    detail=f"choice slot {self.name!r}: an empty option is an unlabelled letter",
                )
            if opt.strip().casefold() == NOUL:
                raise ReservedOptionNameRefusal(
                    expected=f"no option literally named {NOUL!r}", actual=opt,
                    detail=(
                        f"choice slot {self.name!r}: {NOUL!r} is present in every option set "
                        "already, so listing it names one answer with two letters"
                    ),
                )
        seen: dict[str, int] = {}
        for i, opt in enumerate(self.options):
            if opt in seen:
                raise DuplicateOptionRefusal(
                    expected="pairwise distinct options", actual=opt,
                    detail=(
                        f"choice slot {self.name!r}: option {seen[opt]} and option {i} are "
                        "identical, so two letters name one answer"
                    ),
                )
            seen[opt] = i

    @property
    def type_name(self) -> str:
        return "choice"

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "type": "choice", "options": list(self.options)}


@dataclass(frozen=True, slots=True)
class ScoreSlot(Slot):
    """An ordinal bin, or ``noul``. Bins are ordered and are never shuffled."""

    bins: int = 0

    def __post_init__(self) -> None:
        self._check_name()
        if isinstance(self.bins, bool) or not isinstance(self.bins, int):
            raise BinsOutOfRangeRefusal(
                expected="int bins", actual=type(self.bins).__name__,
                detail=f"score slot {self.name!r}",
            )
        if self.bins < MIN_SCORE_BINS or self.bins > MAX_SCORE_BINS:
            raise BinsOutOfRangeRefusal(
                expected=f"{MIN_SCORE_BINS} <= bins <= {MAX_SCORE_BINS}", actual=self.bins,
                detail=(
                    f"score slot {self.name!r}: a {self.bins}-bin ordinal is not a question "
                    f"below {MIN_SCORE_BINS}, and is not decodable above {MAX_SCORE_BINS}"
                ),
            )

    @property
    def type_name(self) -> str:
        return "score"

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "type": "score", "bins": self.bins}


@dataclass(frozen=True, slots=True)
class SpanSlot(Slot):
    """``{start_line, end_line}`` in the context, or ``noul``."""

    def __post_init__(self) -> None:
        self._check_name()

    @property
    def type_name(self) -> str:
        return "span"

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "type": "span"}


def _slot_from_wire(raw: object, index: int) -> Slot:
    if not isinstance(raw, dict):
        raise UnknownSlotTypeRefusal(
            expected="a slot object", actual=type(raw).__name__, detail=f"slots[{index}]",
        )
    name = raw.get("name")
    if not isinstance(name, str):
        raise UnknownSlotTypeRefusal(
            expected="str name", actual=type(name).__name__, detail=f"slots[{index}]",
        )
    kind = raw.get("type")
    if kind == "choice":
        opts = raw.get("options")
        if not isinstance(opts, (list, tuple)):
            raise UnknownSlotTypeRefusal(
                expected="options list on a choice slot", actual=type(opts).__name__,
                detail=f"slots[{index}] ({name!r})",
            )
        return ChoiceSlot(name=name, options=tuple(opts))
    if kind == "score":
        bins = raw.get("bins")
        if bins is None:
            raise BinsOutOfRangeRefusal(
                expected="a bins field on a score slot", actual=None,
                detail=f"slots[{index}] ({name!r})",
            )
        return ScoreSlot(name=name, bins=bins)
    if kind == "span":
        return SpanSlot(name=name)
    raise UnknownSlotTypeRefusal(
        expected="one of 'choice', 'score', 'span'", actual=kind,
        detail=f"slots[{index}] ({name!r}): there is no default slot type",
    )


@dataclass(frozen=True, slots=True)
class HashExpectation:
    """What build the caller believes it is talking to. All five pins optional.

    The Python producer for the ``expect`` block in ``docs/schema-api.md``. Mirrors
    ``crates/qd-runtime/src/schema.rs::HashExpectation`` field for field; the runtime
    compares each present pin against the loaded backend's identity and refuses with
    ``hash_mismatch`` rather than answering from a build the caller did not mean.

    Until 2026-09-19 this lane could not pin anything: ``Request.to_wire()`` emitted no
    ``expect`` at all, so an eval run that silently used a different tokenizer was
    caught by nothing on this side. ``GAP-XLANG-NO-PY-HASH-EXPECTATION``.

    Three details are load-bearing, and each is asserted rather than described in
    ``python/tests/test_qd_data_wire_agreement.py``:

    * **The names are the Rust struct's.** They are compared against the *generated*
      ``qd_wire.contract`` table, itself re-derived from ``schema.rs`` on every run. The
      second pin is ``weight_hash``, not ``weights_hash`` -- that single missing letter
      is ``GAP-XLANG-EXPECT-FIELD-NAME-DOC-DRIFT``, and the struct is
      ``deny_unknown_fields``, so the misspelling earned an unknown-field refusal rather
      than a hash mismatch.
    * **Empty emits nothing.** An expectation with no pins is absent from the wire
      object, not five nulls: ``HashExpectation::default()`` is what the runtime
      substitutes for an absent ``expect``, so emitting nulls would be a second spelling
      of one state -- the defect shape this repo has shipped three times.
    * **A pin is a string or it is refused.** Rust types the five as
      ``Option<String>``; a number there is a serde type error, so coercing one here
      would make the two lanes accept different payloads.

    ``label_set_hash`` is a property of the *loaded build*. It is **not**
    :func:`slot_set_digest`, which is a property of the request's slot list; a caller
    that pinned the latter would earn ``hash_mismatch`` on every request.
    ``GAP-XLANG-LABEL-SET-HASH-TWO-MEANINGS``.
    """

    tokenizer_hash: str | None = None
    weight_hash: str | None = None
    head_hash: str | None = None
    label_set_hash: str | None = None
    calibration_hash: str | None = None

    def __post_init__(self) -> None:
        for name in _HASH_PINS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise MalformedRequestRefusal(
                    expected=f"`expect.{name}` as a string or absent",
                    actual=type(value).__name__,
                    detail=(
                        "the runtime types the five pins as Option<String>; a non-string "
                        "is a deserialization error there, so it is refused here rather "
                        "than coerced into one"
                    ),
                )

    def is_empty(self) -> bool:
        """True when nothing is pinned, which is what an absent ``expect`` means."""
        return all(getattr(self, name) is None for name in _HASH_PINS)

    def to_wire(self) -> dict[str, Any]:
        """Only the pins that are set. An unset pin is absent, never ``null``."""
        return {
            name: value
            for name in _HASH_PINS
            if (value := getattr(self, name)) is not None
        }

    @classmethod
    def from_wire(cls, raw: object) -> Self:
        """Parse an ``expect`` block, refusing an unknown pin the way the struct does."""
        if not isinstance(raw, dict):
            raise MalformedRequestRefusal(
                expected="`expect` as an object", actual=type(raw).__name__,
                detail="the runtime deserializes `expect` into HashExpectation",
            )
        unknown = sorted(set(raw) - set(_HASH_PINS))
        if unknown:
            raise MalformedRequestRefusal(
                expected=f"pins drawn from {sorted(_HASH_PINS)}", actual=unknown,
                detail=(
                    "HashExpectation is deny_unknown_fields in crates/qd-runtime/src/"
                    "schema.rs; an ignored pin is a pin the caller believes is checked"
                ),
            )
        return cls(**{name: raw[name] for name in _HASH_PINS if name in raw})


#: The five pin names, in the order ``docs/schema-api.md`` lists them. Derived from the
#: dataclass so the tuple and the fields cannot disagree.
_HASH_PINS: Final[tuple[str, ...]] = tuple(
    f.name for f in dataclasses.fields(HashExpectation)
)


@dataclass(frozen=True, slots=True)
class Request:
    """A typed request. Structurally valid by construction.

    ``context`` is **bytes**, never ``str``. ``docs/schema-api.md``: a ``String``
    round-trip normalizes Unicode and would silently change token boundaries and
    therefore line spans. Nothing in this module normalizes; ``test_render.py``
    asserts that NFC and NFD spellings of the same text render to different bytes.
    """

    task: str
    context: bytes
    question: str
    slots: tuple[Slot, ...]
    route: Route = "generic"
    schema_version: int = SCHEMA_VERSION
    #: Optional caller-supplied identity, used to seed per-example option shuffling.
    example_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    #: What build the caller believes it is talking to. Empty by default, which is what
    #: an absent ``expect`` means on the wire and what the runtime substitutes.
    expect: HashExpectation = field(default_factory=HashExpectation)

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnknownSchemaVersionRefusal(
                expected=f"one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}", actual=self.schema_version,
                detail="forward-compat guessing is how a field changes meaning silently",
            )
        if not self.task.strip():
            raise EmptyTaskRefusal(
                expected="a task with at least one non-whitespace character",
                actual=self.task,
                detail=(
                    "the task is the registered route's head-lookup key and one rendered "
                    "line of every prompt, so a blank one names nothing. The runtime "
                    "already refuses it as `empty_task`; this lane did not, so a training "
                    "row could carry a task no runtime would serve"
                ),
            )
        if not isinstance(self.context, (bytes, bytearray)):
            raise UnknownSlotTypeRefusal(
                expected="context as bytes", actual=type(self.context).__name__,
                detail=(
                    "context crosses the FFI boundary as bytes and a length; a str would "
                    "invite a Unicode-normalizing round trip that moves token boundaries"
                ),
            )
        if self.route not in _ROUTES:
            raise UnknownSlotTypeRefusal(
                expected=f"one of {sorted(_ROUTES)}", actual=self.route, detail="route",
            )
        if not self.slots:
            raise EmptySlotsRefusal(
                expected="at least one slot", actual=0,
                detail="a request with no slots has no answer map",
            )
        if len(self.slots) > MAX_SLOTS:
            raise EmptySlotsRefusal(
                expected=f"at most {MAX_SLOTS} slots", actual=len(self.slots),
                detail="each slot is a decode pass; the fan-out is bounded",
            )
        seen: set[str] = set()
        for s in self.slots:
            if not isinstance(s, (ChoiceSlot, ScoreSlot, SpanSlot)):
                raise UnknownSlotTypeRefusal(
                    expected="ChoiceSlot | ScoreSlot | SpanSlot", actual=type(s).__name__,
                )
            if s.name in seen:
                raise DuplicateSlotNameRefusal(
                    expected="pairwise distinct slot names", actual=s.name,
                    detail="two slots with one name make the answer map ambiguous",
                )
            seen.add(s.name)

    # -- wire form -------------------------------------------------------

    def to_wire(self) -> dict[str, Any]:
        """The JSON object a caller sends.

        ``docs/schema-api.md``, *"The wire encoding of ``context`` -- exactly one
        form"*: ``context_b64`` (base64, ASCII) of the exact bytes, plus a required
        ``context_len`` (the decoded byte count). ``context_len`` is emitted even
        though base64 makes it derivable, because a disagreement between the two is
        the signature of a truncated payload that still decodes cleanly -- the one
        failure a length field is for.

        ``expect`` appears only when something is pinned. Absent means "pinned
        nothing", which is exactly ``HashExpectation::default()`` on the other side;
        emitting an empty object or five nulls would be a second spelling of it.
        """
        wire: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task": self.task,
            "context_b64": encode_context(bytes(self.context)),
            "context_len": len(self.context),
            "question": self.question,
            "slots": [s.to_wire() for s in self.slots],
            "route": self.route,
            "example_id": self.example_id,
            "metadata": dict(self.metadata),
        }
        if not self.expect.is_empty():
            wire["expect"] = self.expect.to_wire()
        return wire

    @classmethod
    def from_wire(cls, raw: object) -> Self:
        """Parse a wire request, refusing every shape that could change meaning silently.

        ``context_b64`` and ``context_len`` are both required, and they must agree.
        **There is no second accepted form.** This module used to accept a
        ``context`` ``str`` as an alternative; ``qd-runtime`` meanwhile read
        ``context`` as a JSON array of byte values, so no caller could satisfy both
        lanes and the system did not work end to end even though both suites passed
        (``GAP-RT-WIRE-CONTEXT-ENCODING``). A payload carrying ``context`` in any
        spelling is now a typed refusal naming the field that replaced it, because a
        lenient read is exactly how two encodings coexist and "which one wins"
        becomes undefined.

        **An unknown top-level field is refused, not ignored.** The same posture as
        ``wire.rs::known_keys``, and in the same order: ``context`` is named first so a
        caller still sending the retired form is told what replaced it rather than that
        a field it has always sent is unknown. Reading through ``dict.get`` meant a
        caller who wrote ``contextlen`` was told ``context_len is required`` with the
        field visibly present in its own payload -- by the runtime only, while this lane
        defaulted it and trained on the result. ``GAP-XLANG-UNKNOWN-FIELD-LENIENCY``.
        """
        if not isinstance(raw, dict):
            raise UnknownSlotTypeRefusal(
                expected="a request object", actual=type(raw).__name__,
            )
        version = raw.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnknownSchemaVersionRefusal(
                expected=f"one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}", actual=version,
                detail="forward-compat guessing is how a field changes meaning silently",
            )
        if "context" in raw:
            raise ContextNotBytesRefusal(
                expected="'context_b64' (base64) and 'context_len'",
                actual="a 'context' field",
                detail=(
                    "the retired wire form. The context crosses as context_b64 plus "
                    "context_len and there is no second accepted form; reading 'context' "
                    "leniently is what let the two lanes diverge"
                ),
            )
        unknown = sorted(set(raw) - WIRE_REQUEST_KEYS)
        if unknown:
            raise MalformedRequestRefusal(
                expected=f"fields drawn from {sorted(WIRE_REQUEST_KEYS)}", actual=unknown,
                detail=(
                    f"unknown field {unknown[0]!r}; this build reads "
                    f"{sorted(WIRE_REQUEST_KEYS)}. A misspelling read leniently is a "
                    "field the caller believes was carried"
                ),
            )
        ctx = decode_context(raw.get("context_b64"), raw.get("context_len"))

        slots_raw = raw.get("slots")
        if not isinstance(slots_raw, (list, tuple)):
            raise EmptySlotsRefusal(
                expected="a slots list", actual=type(slots_raw).__name__,
            )
        slots = tuple(_slot_from_wire(s, i) for i, s in enumerate(slots_raw))

        task = raw.get("task", "")
        question = raw.get("question", "")
        route = raw.get("route", "generic")
        if not isinstance(task, str) or not isinstance(question, str) or not isinstance(route, str):
            raise UnknownSlotTypeRefusal(
                expected="str task, question and route",
                actual=f"{type(task).__name__}, {type(question).__name__}, {type(route).__name__}",
            )
        meta_raw = raw.get("metadata", {})
        meta = {str(k): str(v) for k, v in meta_raw.items()} if isinstance(meta_raw, dict) else {}
        return cls(
            task=task,
            context=ctx,
            question=question,
            slots=slots,
            route=route,  # type: ignore[arg-type]
            schema_version=version,
            example_id=str(raw.get("example_id", "")),
            metadata=meta,
            expect=HashExpectation.from_wire(raw["expect"]) if "expect" in raw
            else HashExpectation(),
        )


def slot_set_digest(slots: tuple[Slot, ...] | list[Slot]) -> str:
    """Digest of a request's **slot list** — the identity of the task's label set.

    .. warning::

       This is **not** the value that goes in ``expect.label_set_hash`` on the wire, and
       until 2026-09-19 it was called ``label_set_hash``, which said it was.

       ``HashExpectation.label_set_hash`` is compared by the runtime against the **loaded
       backend's** ``identity.label_set_hash`` (``qd-runtime/src/runtime.rs::check_hashes``)
       — a property of the build, fixed for a given set of weights. This function is a
       property of the *request*, and changes whenever the slots do. A caller that computed
       one and sent it as the other would earn ``hash_mismatch`` on **every** request,
       because the two can never be equal except by collision.

       Same defect shape as ``GAP-RT-WIRE-CONTEXT-ENCODING``: one name, two quantities, both
       lanes' suites green. Tracked as ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`` and asserted
       rather than described, by
       ``wire_context_crosslang.rs::python_has_no_hash_expectation_and_its_label_set_hash_is_a_different_quantity``.

    What it is for: binding a registered head to the task it was fitted against. A head is
    hash-bound to a backbone *and* to the label set it decodes, and this is the latter.

    Options are hashed **in the order given**: a permuted option list is a different prompt,
    so it is a different label set, and pretending otherwise would let a shuffled build pass
    a check it should fail.
    """
    body = [s.to_wire() for s in slots]
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
