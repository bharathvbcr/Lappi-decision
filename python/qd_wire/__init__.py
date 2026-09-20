"""The Rust -> Python half of ``docs/schema-api.md``, parsed strictly.

This package exists because of ``GAP-XLANG-NO-PY-ANSWER-PARSER``: the request half
of the wire contract has two implementations that a cross-language test drives
against each other, and the **answer** half had one. A contract with a single
implementation cannot disagree with itself, which is how
``GAP-RT-WIRE-CONTEXT-ENCODING`` and ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``
each survived two green suites.

What is here is a second implementation, and it is deliberately the **strict** one:

* every object's key set is checked exactly — an unknown field is a refusal, a
  missing field is a refusal, and there is no lenient mode;
* the ``kind`` tables for all 36 refusals and 12 backend errors are *generated*
  from ``crates/qd-runtime/src/{schema,refusal}.rs`` into :mod:`qd_wire.contract`,
  and ``python/tests/test_wire_contract_matches_rust.py`` re-derives them on every
  run, so a rename on the Rust side fails a Python test that names the field;
* ``fixtures/wire/`` is the golden corpus the two lanes share. When it is absent,
  :func:`qd_wire.corpus.load_corpus` reports ``NotRun`` with a reason — never a
  pass.

Two names are *not* here on purpose. ``HashExpectation`` lives on the request side
and ``python/qd_data`` has no producer for it (``GAP-XLANG-NO-PY-HASH-EXPECTATION``),
and ``qd_data.schema.label_set_hash`` is a different quantity from
``expect.label_set_hash`` (``GAP-XLANG-LABEL-SET-HASH-TWO-MEANINGS``). Neither is
re-spelled here; both are asserted as absences in
``python/tests/test_wire_gap_pins.py``.
"""

from __future__ import annotations

from qd_wire.answer import (
    AnswerEnvelope,
    ChoiceSet,
    ChoiceValue,
    ConformalSet,
    ScoreSet,
    ScoreValue,
    SlotAnswer,
    SlotValue,
    SpanValue,
    parse_answer_envelope,
)
from qd_wire.corpus import CorpusLoad, FixtureEnvelope, fixture_dir, load_corpus
from qd_wire.errors import WireParseError
from qd_wire.refusal import (
    BackendError,
    ErrorEnvelope,
    Refusal,
    RefusalEnvelope,
    parse_error_envelope,
    parse_refusal_envelope,
)
from qd_wire.response import (
    CallerReading,
    Response,
    caller_reading,
    parse_response,
    parse_response_line,
)

__all__ = [
    "AnswerEnvelope",
    "BackendError",
    "CallerReading",
    "ChoiceSet",
    "ChoiceValue",
    "ConformalSet",
    "CorpusLoad",
    "ErrorEnvelope",
    "FixtureEnvelope",
    "Refusal",
    "RefusalEnvelope",
    "Response",
    "ScoreSet",
    "ScoreValue",
    "SlotAnswer",
    "SlotValue",
    "SpanValue",
    "WireParseError",
    "caller_reading",
    "fixture_dir",
    "load_corpus",
    "parse_answer_envelope",
    "parse_error_envelope",
    "parse_refusal_envelope",
    "parse_response",
    "parse_response_line",
]
