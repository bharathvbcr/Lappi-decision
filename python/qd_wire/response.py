"""The three-variant response, and the one function that reads it.

``crates/qd-runtime/src/schema.rs::Response`` is ``#[serde(tag = "status")]``, so
the discriminant sits beside the envelope's own fields rather than wrapping them.
``docs/schema-api.md`` shows an answer with no ``status`` at all; a Python caller
written from that example would read the discriminant as an unknown key. This
module is the discriminant made executable.

:func:`caller_reading` is the load-bearing one. It is the Python form of
``CallerReading`` in ``schema.rs``, which exists so *"a refusal must not be
confusable with ``noul``"* is a total function rather than a paragraph: a refusal
and a backend error have no path to :data:`CallerReading.MODEL_ABSTAINED`, and
``python/tests/test_wire_refusal_is_not_noul.py`` asserts that over every refusal
kind and every backend-error kind this build declares.
"""

from __future__ import annotations

import enum
import json
from typing import Final

from qd_wire.answer import AnswerEnvelope, parse_answer_envelope
from qd_wire.contract import STATUS_TAGS
from qd_wire.errors import WireParseError
from qd_wire.refusal import (
    ErrorEnvelope,
    RefusalEnvelope,
    parse_error_envelope,
    parse_refusal_envelope,
)
from qd_wire.values import as_str

__all__ = ["CallerReading", "Response", "caller_reading", "parse_response", "parse_response_line"]

Response = AnswerEnvelope | RefusalEnvelope | ErrorEnvelope

_PARSERS: Final = {
    "ok": parse_answer_envelope,
    "refused": parse_refusal_envelope,
    "error": parse_error_envelope,
}


class CallerReading(enum.Enum):
    """How a caller may read a response. Mirrors ``schema.rs::CallerReading``."""

    MODEL_ANSWERED = "model_answered"
    MODEL_ABSTAINED = "model_abstained"
    REQUEST_REFUSED = "request_refused"
    BACKEND_FAILED = "backend_failed"


def parse_response(raw: object, *, path: str = "$") -> Response:
    """Parse one response object, dispatching on the ``status`` discriminant."""
    if not isinstance(raw, dict):
        raise WireParseError(path, f"expected a response object, got {type(raw).__name__}")
    if "status" not in raw:
        raise WireParseError(
            path,
            "no `status` discriminant. `Response` is internally tagged on `status` "
            "and the three variants are "
            f"{sorted(STATUS_TAGS)}. The example answer in docs/schema-api.md omits "
            "it, which is why this is a named refusal and not a KeyError.",
        )
    status = as_str(raw["status"], path=f"{path}.status")
    parse = _PARSERS.get(status)
    if parse is None:
        raise WireParseError(
            f"{path}.status",
            f"unknown status {status!r}; this build emits {sorted(STATUS_TAGS)}. "
            "Guessing which of the three it resembles is how a refusal comes to be "
            "read as an answer.",
        )
    body = {k: v for k, v in raw.items() if k != "status"}
    return parse(body, path=path)


def parse_response_line(line: str | bytes, *, path: str = "$") -> Response:
    """Parse one line of the JSON-lines reply protocol ``qd serve`` / ``qd oneshot`` speak."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WireParseError(path, f"not valid JSON: {exc}") from exc
    return parse_response(raw, path=path)


def caller_reading(response: Response) -> CallerReading:
    """The total function that keeps a refusal from reading as an abstention.

    An ``ok`` envelope with **no slots** is not reachable from the runtime — `slots`
    empty is ``Refusal::EmptySlots`` before a backend is ever asked — so it has no
    honest reading here and is refused rather than defaulted. Defaulting it either
    way would invent one: ``all(s.noul for s in [])`` is ``True`` in Python, which
    would report a slotless envelope as the model abstaining.
    """
    if isinstance(response, RefusalEnvelope):
        return CallerReading.REQUEST_REFUSED
    if isinstance(response, ErrorEnvelope):
        return CallerReading.BACKEND_FAILED
    if not response.slots:
        raise WireParseError(
            "$.slots",
            "an `ok` envelope with no slots has no caller reading. `slots` empty is "
            "refused as `empty_slots` before a backend is asked, so this envelope did "
            "not come from this runtime; reading it as an abstention would be "
            "`all([]) is True` deciding the contract.",
        )
    if all(slot.noul for slot in response.slots.values()):
        return CallerReading.MODEL_ABSTAINED
    return CallerReading.MODEL_ANSWERED
