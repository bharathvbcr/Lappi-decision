"""Typed refusals. A refusal is never ``noul``, and ``noul`` is never a refusal.

``docs/schema-api.md``:

    A refusal is a typed error carrying which check failed and both values
    compared. It is never an empty answer, and never ``noul`` -- ``noul`` means
    *the model abstained*, a refusal means *the request was not answerable as
    posed*. Collapsing the two would let a hash mismatch read as model humility.

The separation here is structural rather than a matter of care:

* A refusal is an **exception type**. It cannot be returned as a slot value,
  because raising and returning are different control flow.
* ``QdRefusal.to_json()`` emits a payload whose top-level key is ``refusal``.
  An answer payload's top-level key is ``slots``. The two objects share no keys,
  so no consumer can read one as the other. :func:`is_noul_payload` and
  :func:`is_refusal_payload` are mutually exclusive by construction, and a test
  asserts it over every refusal type in this module.
* ``QdRefusal`` deliberately has no ``value``, no ``noul`` and no ``score``
  attribute. A reporter that renders a refusal as an abstention has to fabricate
  the field, which raises.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "NOUL",
    "BinsOutOfRangeRefusal",
    "ContextLenMissingRefusal",
    "ContextLengthMismatchRefusal",
    "ContextNotBase64Refusal",
    "ContextNotBytesRefusal",
    "ContextTooLargeRefusal",
    "DuplicateOptionRefusal",
    "DuplicateSlotNameRefusal",
    "EmptyContextRefusal",
    "EmptyOptionRefusal",
    "EmptySlotsRefusal",
    "HashMismatchRefusal",
    "HeldOutViolation",
    "LicenceRefused",
    "MalformedRequestRefusal",
    "OptionTooLongRefusal",
    "QdRefusal",
    "RenderedPromptTooLargeRefusal",
    "ReservedOptionNameRefusal",
    "TooManyOptionsRefusal",
    "UnknownSchemaVersionRefusal",
    "UnknownSlotTypeRefusal",
    "is_noul_payload",
    "is_refusal_payload",
]

#: The abstain token. It is a *value* a model may produce, never an error.
NOUL: Final[str] = "noul"


class QdRefusal(Exception):
    """The request was not answerable as posed.

    Carries the name of the check that failed and both compared values, so a
    caller can act on the specific mismatch rather than on a string.
    """

    #: Stable identifier for the check, used in tests and in the ledger.
    check: str = "unspecified"

    #: The ``Refusal::kind()`` identifiers in ``crates/qd-runtime/src/refusal.rs`` that
    #: this class stands for, closest first — the runtime's name for the same condition.
    #:
    #: ``GAP-XLANG-REFUSAL-VOCABULARIES`` is the divergence this makes checkable. It was
    #: recorded as six pairs of differently-spelled names, in a flat table repeated in
    #: three files, which does not say *why* they differ and so cannot say whether a
    #: rename would fix it. Declared per class, it does: four of the six Python classes
    #: stand for **several** runtime kinds, so renaming ``options_len`` to
    #: ``too_many_options`` would put a failure's name on a class that also fires for
    #: ``too_few_options``. Only ``schema_version`` and ``slot_names_unique`` are
    #: one-to-one and renameable as they stand.
    #:
    #: An empty tuple means "the runtime has no such check", which is true of the
    #: data-lane refusals (licensing, held-out paths, upstream row shape). It is not a
    #: default anyone may fall into: ``test_qd_data_wire_agreement.py`` fails when a
    #: subclass inherits this rather than declaring it, and fails again when the set of
    #: classes declaring ``()`` is not exactly the recorded one. That is the executable
    #: form of the gap's *"do not add a refusal to one lane without checking whether the
    #: other already names the same condition differently"*.
    #:
    #: Every name here is checked against the **generated** ``qd_wire.contract``
    #: refusal-kind table, itself re-derived from ``refusal.rs`` on every run, so a kind
    #: renamed on the Rust side fails a Python test that points at the stale class.
    rust_kinds: tuple[str, ...] = ()

    def __init__(self, *, expected: object, actual: object, detail: str = "") -> None:
        self.expected = expected
        self.actual = actual
        self.detail = detail
        msg = f"REFUSED [{self.check}]: expected {expected!r}, got {actual!r}"
        if detail:
            msg = f"{msg} -- {detail}"
        super().__init__(msg)

    def to_json(self) -> dict[str, Any]:
        """A payload that cannot be mistaken for an answer.

        The top-level key is ``refusal``; an answer's is ``slots``. There is no
        ``noul`` key anywhere in this object, at any depth.
        """
        return {
            "refusal": {
                "check": self.check,
                "expected": _jsonable(self.expected),
                "actual": _jsonable(self.actual),
                "detail": self.detail,
            }
        }


def _jsonable(v: object) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)


# -- the refusals named in docs/schema-api.md and docs/hardening.md section 3 --


class MalformedRequestRefusal(QdRefusal):
    """The envelope itself is not a request this build reads.

    An unknown top-level field, an ``expect`` block that is not an object, or a pin
    that is not a string. Named and spelled exactly as ``Refusal::MalformedRequest``
    in ``crates/qd-runtime/src/refusal.rs``, because a refusal added to one lane under
    a new name is how ``GAP-XLANG-REFUSAL-VOCABULARIES`` widens.

    Unknown fields are refused rather than ignored for the reason
    ``wire.rs::known_keys`` gives: a caller who misspells ``context_len`` would
    otherwise be told ``context_len is required`` with the field visibly present in
    its own payload. Until 2026-09-19 this lane read straight past them, so the same
    typo was a hard refusal in serving and a silently defaulted field in training --
    which is the drift this contract exists to prevent.
    ``GAP-XLANG-UNKNOWN-FIELD-LENIENCY``.
    """

    check = "malformed_request"
    rust_kinds = ("malformed_request",)


class UnknownSchemaVersionRefusal(QdRefusal):
    """Forward-compat guessing is how a field changes meaning silently."""

    check = "schema_version"
    rust_kinds = ("unknown_schema_version",)


class EmptySlotsRefusal(QdRefusal):
    """An answer map with no slots is not an answer.

    Also raised for a slot list over ``MAX_SLOTS`` and for a ``slots`` field that is
    not a list at all, which the runtime names ``too_many_slots`` and
    ``malformed_request``. One class, three runtime kinds.
    """

    check = "slots_non_empty"
    rust_kinds = ("empty_slots", "too_many_slots", "malformed_request")


class DuplicateSlotNameRefusal(QdRefusal):
    """Two slots with one name make the answer map ambiguous."""

    check = "slot_names_unique"
    rust_kinds = ("duplicate_slot_name",)


class UnknownSlotTypeRefusal(QdRefusal):
    """A slot type outside the four in the contract.

    The broadest class here: it also carries an unnamed slot, an unknown route, a
    choice slot with no ``options`` field, a non-object request and a non-string
    ``task``/``question``/``route``. The runtime names those five conditions
    separately, so the single ``slot_type_known`` identifier a caller sees from this
    lane covers five of its kinds.
    """

    check = "slot_type_known"
    rust_kinds = (
        "unknown_slot_type",
        "unknown_route",
        "empty_slot_name",
        "slot_field_missing",
        "malformed_request",
    )


class TooManyOptionsRefusal(QdRefusal):
    """The lm_head letter slice cannot express another option; dropping one
    changes the question.

    Raised for **no** options as well as too many; the runtime splits those into
    ``too_few_options`` and ``too_many_options``.
    """

    check = "options_len"
    rust_kinds = ("too_many_options", "too_few_options")


class DuplicateOptionRefusal(QdRefusal):
    """Two identical options make the question malformed: two letters, one answer."""

    check = "options_unique"
    rust_kinds = ("duplicate_option",)


class EmptyOptionRefusal(QdRefusal):
    """An empty or whitespace-only option is an unlabelled letter."""

    check = "option_non_empty"
    rust_kinds = ("empty_option",)


class ReservedOptionNameRefusal(QdRefusal):
    """``noul`` is in every option set already; listing it is a duplicate."""

    check = "option_not_reserved"
    rust_kinds = ("reserved_option_name",)


class OptionTooLongRefusal(QdRefusal):
    """Refusal, not truncation -- a truncated option is a different option."""

    check = "option_len"
    rust_kinds = ("option_text_over_cap",)


class BinsOutOfRangeRefusal(QdRefusal):
    """A 1-bin ordinal is not a question; over the letter cap it is not decodable.

    Also raised for a score slot with no ``bins`` field and for ``bins`` of the wrong
    type, which the runtime names ``slot_field_missing`` and ``malformed_request``.
    """

    check = "bins_range"
    rust_kinds = ("bins_out_of_range", "slot_field_missing", "malformed_request")


class ContextTooLargeRefusal(QdRefusal):
    """Truncating moves the answer out of the window without saying so, and line
    spans would then point at the wrong lines.

    **The seventh divergence, and the one the pinned table never listed.** This class
    is also raised for context bytes that are not UTF-8, for an over-cap question and
    for an over-cap task -- four runtime kinds under one identifier. It is absent from
    ``KNOWN_VOCABULARY_DIFFERENCES`` on both sides because the cross-language negative
    set sends no over-cap payload, so the pin that was meant to make the divergence
    visible could not observe this one at all. ``GAP-XLANG-REFUSAL-VOCABULARIES``.
    """

    check = "context_bytes"
    rust_kinds = (
        "context_over_cap",
        "context_not_utf8",
        "question_over_cap",
        "task_over_cap",
    )


class ContextNotBytesRefusal(QdRefusal):
    """``context_b64`` is absent, or is not a JSON string.

    ``docs/schema-api.md``, *"The wire encoding of ``context`` -- exactly one
    form"*: ``context_b64`` plus a required ``context_len``, and no second form.
    An integer array in that position is this refusal, not a lenient read.

    ``check`` matches ``Refusal::ContextNotBytes::kind()`` in
    ``crates/qd-runtime/src/refusal.rs``; the two lanes name the same check with
    the same string so a caller can branch on one identifier.
    """

    check = "context_not_bytes"
    rust_kinds = ("context_not_bytes",)


class ContextNotBase64Refusal(QdRefusal):
    """``context_b64`` is not canonical standard base64.

    Canonical is the operative word: padding is required, the alphabet is the
    standard one, and the bits a partial quantum does not use must be zero.
    ``base64.b64decode(validate=True)`` checks the alphabet but not those slack
    bits, so this module re-encodes and compares -- otherwise ``Zg==`` and ``Zh==``
    would both name ``b"f"`` and two payloads would be one context.
    """

    check = "context_not_base64"
    rust_kinds = ("context_not_base64",)


class ContextLenMissingRefusal(QdRefusal):
    """``context_len`` is absent.

    There is no default. A payload whose length nobody declared is a payload whose
    truncation nobody would notice, which is the one failure the field is for.
    """

    check = "context_len_missing"
    rust_kinds = ("context_len_missing",)


class ContextLengthMismatchRefusal(QdRefusal):
    """``context_len`` disagrees with the decoded byte count.

    The signature of a payload truncated on a quantum boundary: it still decodes
    cleanly, just to fewer bytes. Both numbers are carried.
    """

    check = "context_length_mismatch"
    rust_kinds = ("context_length_mismatch",)


class EmptyContextRefusal(QdRefusal):
    """All-whitespace or empty context: refuse, never a confident letter."""

    check = "context_non_empty"
    rust_kinds = ("context_empty",)


class EmptyTaskRefusal(QdRefusal):
    """A request with no task names nothing.

    The task is the registered route's head-lookup key and one rendered line of
    every prompt, so a blank one is not a request with a small field -- it is a
    request that cannot be resolved.

    ``check`` is deliberately the runtime's own identifier rather than a positive
    restatement like the ``context_non_empty`` above it. The contract asks that both
    lanes "name the checks identically", and a one-to-one pair that already agrees
    should not be added to ``KNOWN_VOCABULARY_DIFFERENCES``.
    """

    check = "empty_task"
    rust_kinds = ("empty_task",)


class RenderedPromptTooLargeRefusal(QdRefusal):
    """Every payload is bounded. Escaping can grow a context; the bound is on the
    bytes the model actually sees, not only on the bytes that arrived."""

    check = "rendered_bytes"
    rust_kinds = ("rendered_prompt_over_cap",)


class HashMismatchRefusal(QdRefusal):
    """A swapped tokenizer / label set / head maps wrong ids silently. Answering
    would be confidently wrong."""

    check = "hash_match"
    rust_kinds = ("hash_mismatch",)


class LicenceRefused(QdRefusal):
    """A row or dataset whose licence is not on the permissive allowlist.

    A refusal rather than a filter-out at the dataset level: silently dropping a
    whole dataset because of its licence looks identical to the dataset failing
    to load, and the two need different responses.
    """

    check = "licence_allowlist"
    #: No runtime counterpart: qd-runtime never sees a licence.
    rust_kinds = ()


class HeldOutViolation(QdRefusal):
    """A training process reached for held-out data.

    ``CLAUDE.md`` rule 3: held-out data and the two task-holdout families are
    never read by a training process; removing the check is a refused change.
    """

    check = "held_out_path"
    #: No runtime counterpart: qd-runtime never reads a dataset path.
    rust_kinds = ()


def is_refusal_payload(payload: object) -> bool:
    """True for a refusal payload, and never for an answer payload."""
    return isinstance(payload, dict) and "refusal" in payload


def is_noul_payload(payload: object) -> bool:
    """True when an *answer* payload carries an abstention on any slot.

    Deliberately returns False for a refusal payload: a refusal is not an
    abstention, and this is the function a reporter would reach for to conflate
    them.
    """
    if not isinstance(payload, dict) or "refusal" in payload:
        return False
    slots = payload.get("slots")
    if not isinstance(slots, dict):
        return False
    return any(isinstance(s, dict) and s.get("noul") is True for s in slots.values())
