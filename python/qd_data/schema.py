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
    ReservedOptionNameRefusal,
    TooManyOptionsRefusal,
    UnknownSchemaVersionRefusal,
    UnknownSlotTypeRefusal,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "OPTION_LETTERS",
    "NOUL_LETTER",
    "MAX_CHOICE_OPTIONS",
    "MIN_SCORE_BINS",
    "MAX_SCORE_BINS",
    "MAX_SLOTS",
    "Slot",
    "ChoiceSlot",
    "ScoreSlot",
    "SpanSlot",
    "Request",
    "Route",
    "slot_set_digest",
    "canonical_json",
    "encode_context",
    "decode_context",
]

SCHEMA_VERSION: Final[int] = 1
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({1})

#: The 16 letters the generic route decodes over. ``docs/schema-api.md``: one
#: token over "the 16 letter-token slice of ``lm_head``".
OPTION_LETTERS: Final[str] = "ABCDEFGHIJKLMNOP"

#: ``noul`` is "a reserved letter present in **every** option set". It is given a
#: letter *outside* the contiguous A-P block so that the 16-row option slice stays
#: exactly 16 rows even when a request uses all 16 named options.
#:
#: NOTE (spec ambiguity, recorded as GAP-SCHEMA-NOUL-LETTER-BUDGET): the contract
#: says both "options.len() > 16 refuses" and "noul is a reserved letter in every
#: option set". Those two cannot both hold inside a 16-row slice -- 16 named
#: options plus noul needs 17 rows. This module reads it as 16 option rows plus one
#: reserved noul row. If the runtime lane instead slices 16 rows *total*, the fix is
#: to set ``MAX_CHOICE_OPTIONS = 15`` here, one line, and every test follows.
NOUL_LETTER: Final[str] = "Z"

MAX_CHOICE_OPTIONS: Final[int] = 16
MIN_SCORE_BINS: Final[int] = 2
MAX_SCORE_BINS: Final[int] = 16

#: Bounded fan-out: a request with an unbounded number of slots is an unbounded
#: number of decode passes over one prefill.
MAX_SLOTS: Final[int] = 32

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

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnknownSchemaVersionRefusal(
                expected=f"one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}", actual=self.schema_version,
                detail="forward-compat guessing is how a field changes meaning silently",
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
        """
        return {
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
