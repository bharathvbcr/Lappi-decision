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
    "QdRefusal",
    "UnknownSchemaVersionRefusal",
    "EmptySlotsRefusal",
    "DuplicateSlotNameRefusal",
    "TooManyOptionsRefusal",
    "DuplicateOptionRefusal",
    "EmptyOptionRefusal",
    "ReservedOptionNameRefusal",
    "OptionTooLongRefusal",
    "BinsOutOfRangeRefusal",
    "ContextTooLargeRefusal",
    "ContextNotBytesRefusal",
    "ContextNotBase64Refusal",
    "ContextLenMissingRefusal",
    "ContextLengthMismatchRefusal",
    "EmptyContextRefusal",
    "RenderedPromptTooLargeRefusal",
    "HashMismatchRefusal",
    "UnknownSlotTypeRefusal",
    "LicenceRefused",
    "HeldOutViolation",
    "is_refusal_payload",
    "is_noul_payload",
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


class UnknownSchemaVersionRefusal(QdRefusal):
    """Forward-compat guessing is how a field changes meaning silently."""

    check = "schema_version"


class EmptySlotsRefusal(QdRefusal):
    """An answer map with no slots is not an answer."""

    check = "slots_non_empty"


class DuplicateSlotNameRefusal(QdRefusal):
    """Two slots with one name make the answer map ambiguous."""

    check = "slot_names_unique"


class UnknownSlotTypeRefusal(QdRefusal):
    """A slot type outside the four in the contract."""

    check = "slot_type_known"


class TooManyOptionsRefusal(QdRefusal):
    """The lm_head letter slice cannot express another option; dropping one
    changes the question."""

    check = "options_len"


class DuplicateOptionRefusal(QdRefusal):
    """Two identical options make the question malformed: two letters, one answer."""

    check = "options_unique"


class EmptyOptionRefusal(QdRefusal):
    """An empty or whitespace-only option is an unlabelled letter."""

    check = "option_non_empty"


class ReservedOptionNameRefusal(QdRefusal):
    """``noul`` is in every option set already; listing it is a duplicate."""

    check = "option_not_reserved"


class OptionTooLongRefusal(QdRefusal):
    """Refusal, not truncation -- a truncated option is a different option."""

    check = "option_len"


class BinsOutOfRangeRefusal(QdRefusal):
    """A 1-bin ordinal is not a question; over the letter cap it is not decodable."""

    check = "bins_range"


class ContextTooLargeRefusal(QdRefusal):
    """Truncating moves the answer out of the window without saying so, and line
    spans would then point at the wrong lines."""

    check = "context_bytes"


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


class ContextNotBase64Refusal(QdRefusal):
    """``context_b64`` is not canonical standard base64.

    Canonical is the operative word: padding is required, the alphabet is the
    standard one, and the bits a partial quantum does not use must be zero.
    ``base64.b64decode(validate=True)`` checks the alphabet but not those slack
    bits, so this module re-encodes and compares -- otherwise ``Zg==`` and ``Zh==``
    would both name ``b"f"`` and two payloads would be one context.
    """

    check = "context_not_base64"


class ContextLenMissingRefusal(QdRefusal):
    """``context_len`` is absent.

    There is no default. A payload whose length nobody declared is a payload whose
    truncation nobody would notice, which is the one failure the field is for.
    """

    check = "context_len_missing"


class ContextLengthMismatchRefusal(QdRefusal):
    """``context_len`` disagrees with the decoded byte count.

    The signature of a payload truncated on a quantum boundary: it still decodes
    cleanly, just to fewer bytes. Both numbers are carried.
    """

    check = "context_length_mismatch"


class EmptyContextRefusal(QdRefusal):
    """All-whitespace or empty context: refuse, never a confident letter."""

    check = "context_non_empty"


class RenderedPromptTooLargeRefusal(QdRefusal):
    """Every payload is bounded. Escaping can grow a context; the bound is on the
    bytes the model actually sees, not only on the bytes that arrived."""

    check = "rendered_bytes"


class HashMismatchRefusal(QdRefusal):
    """A swapped tokenizer / label set / head maps wrong ids silently. Answering
    would be confidently wrong."""

    check = "hash_match"


class LicenceRefused(QdRefusal):
    """A row or dataset whose licence is not on the permissive allowlist.

    A refusal rather than a filter-out at the dataset level: silently dropping a
    whole dataset because of its licence looks identical to the dataset failing
    to load, and the two need different responses.
    """

    check = "licence_allowlist"


class HeldOutViolation(QdRefusal):
    """A training process reached for held-out data.

    ``CLAUDE.md`` rule 3: held-out data and the two task-holdout families are
    never read by a training process; removing the check is a refused change.
    """

    check = "held_out_path"


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
