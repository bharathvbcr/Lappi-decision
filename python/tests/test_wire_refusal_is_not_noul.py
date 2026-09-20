"""A refusal is never an abstention, asserted over every kind this build declares.

``docs/schema-api.md``: *"A refusal is a typed error carrying which check failed and
both values compared. It is never an empty answer, and never ``noul``."*
``crates/qd-runtime/tests/refusal_is_not_noul.rs`` asserts that on the Rust side.
Until now nothing asserted it on the Python side, because Python had no parser for
either envelope — ``GAP-XLANG-NO-PY-ANSWER-PARSER``.

The coverage here is total rather than sampled: the payloads are built from
:mod:`qd_wire.contract`'s generated tables, so *every* refusal kind and *every*
backend-error kind the Rust declares is exercised, and a kind added on that side
appears here on the next regeneration without anyone writing a case for it.
"""

from __future__ import annotations

import json

import pytest

from qd_wire.contract import BACKEND_ERROR_KINDS, HASH_KINDS, REFUSAL_KINDS
from qd_wire.errors import WireParseError
from qd_wire.refusal import (
    CONTEXT_CHECKS_NAMED_IDENTICALLY,
    KNOWN_VOCABULARY_DIFFERENCES,
    parse_refusal_envelope,
)
from qd_wire.response import CallerReading, caller_reading, parse_response

_SAMPLE: dict[str, object] = {
    "String": "x",
    "usize": 1,
    "u32": 2,
    "u64": 3,
    "bool": True,
    "Vec<String>": ["a"],
    "Vec<u32>": [1],
    "HashKind": HASH_KINDS[0],
}


def payload(kind: str, spec: dict[str, str]) -> dict[str, object]:
    body: dict[str, object] = {"kind": kind}
    for name, rust_type in spec.items():
        assert rust_type in _SAMPLE, (
            f"{kind}.{name} has Rust type `{rust_type}`, which this test has no sample "
            "value for. The runtime grew a field shape Python has never exercised; add "
            "the sample rather than dropping the field from coverage."
        )
        body[name] = _SAMPLE[rust_type]
    return body


def refusal_wire(kind: str, spec: dict[str, str]) -> dict[str, object]:
    return {
        "status": "refused",
        "schema_version": 1,
        "refusal": payload(kind, spec),
        "message": f"refused: {kind}",
    }


def error_wire(kind: str, spec: dict[str, str]) -> dict[str, object]:
    return {
        "status": "error",
        "schema_version": 1,
        "error": payload(kind, spec),
        "message": f"failed: {kind}",
    }


@pytest.mark.parametrize("kind", sorted(REFUSAL_KINDS))
def test_every_refusal_kind_parses_and_reads_as_refused_never_as_abstained(kind):
    parsed = parse_response(refusal_wire(kind, REFUSAL_KINDS[kind]))
    assert parsed.status == "refused"
    assert parsed.refusal.kind == kind
    assert caller_reading(parsed) is CallerReading.REQUEST_REFUSED
    assert caller_reading(parsed) is not CallerReading.MODEL_ABSTAINED


@pytest.mark.parametrize("kind", sorted(BACKEND_ERROR_KINDS))
def test_every_backend_error_kind_parses_and_reads_as_failed_never_as_abstained(kind):
    parsed = parse_response(error_wire(kind, BACKEND_ERROR_KINDS[kind]))
    assert parsed.status == "error"
    assert parsed.error.kind == kind
    assert caller_reading(parsed) is CallerReading.BACKEND_FAILED
    assert caller_reading(parsed) is not CallerReading.MODEL_ABSTAINED


#: Keys that belong to an answer. A refusal carrying any of them at any depth could
#: be read as one. ``schema_version`` is the single key the two envelopes share.
ANSWER_ONLY_KEYS = frozenset({"slots", "value", "conformal_set", "noul", "score", "backend"})


def _keys_at_every_depth(node: object) -> set[str]:
    if isinstance(node, dict):
        out = set(node)
        for value in node.values():
            out |= _keys_at_every_depth(value)
        return out
    if isinstance(node, list):
        out: set[str] = set()
        for item in node:
            out |= _keys_at_every_depth(item)
        return out
    return set()


@pytest.mark.parametrize("kind", sorted(REFUSAL_KINDS))
def test_no_refusal_carries_an_answer_key_anywhere_in_its_bytes(kind):
    """Structural, not a matter of care: the two objects share no key but ``schema_version``.

    Checked over the key set at every depth rather than over the serialized text: a
    refusal *named* ``empty_slots`` has the substring ``slots`` in a value, which is
    not the same fact and would make this a test about spelling.
    """
    wire = refusal_wire(kind, REFUSAL_KINDS[kind])
    carried = _keys_at_every_depth(wire) & ANSWER_ONLY_KEYS
    assert not carried, f"a `{kind}` refusal carries the answer key(s) {sorted(carried)}"
    # And the text still round-trips, so the check above is over the real payload.
    assert json.loads(json.dumps(wire)) == wire


@pytest.mark.parametrize("kind", sorted(REFUSAL_KINDS))
def test_a_refusal_object_has_no_value_score_or_noul_attribute(kind):
    """A reporter that renders a refusal as an abstention has to fabricate the field."""
    parsed = parse_response(refusal_wire(kind, REFUSAL_KINDS[kind]))
    for attr in ("value", "score", "noul", "conformal_set", "slots"):
        assert not hasattr(parsed.refusal, attr)
        assert not hasattr(parsed, attr)


def test_a_refusal_envelope_carrying_slots_is_refused():
    body = refusal_wire("empty_slots", REFUSAL_KINDS["empty_slots"])
    body["slots"] = {}
    with pytest.raises(WireParseError, match="unknown field"):
        parse_response(body)


def test_an_unknown_refusal_kind_is_refused_rather_than_defaulted():
    body = refusal_wire("empty_slots", {})
    body["refusal"] = {"kind": "vibes_mismatch"}
    with pytest.raises(WireParseError, match="unknown `Refusal` kind"):
        parse_response(body)


def test_a_refusal_missing_a_field_its_kind_declares_is_refused():
    """Every refusal "names the check and carries both values compared"."""
    spec = REFUSAL_KINDS["context_length_mismatch"]
    body = refusal_wire("context_length_mismatch", spec)
    del body["refusal"]["declared"]
    with pytest.raises(WireParseError, match="missing required field"):
        parse_response(body)


def test_a_refusal_with_an_extra_field_its_kind_does_not_declare_is_refused():
    spec = REFUSAL_KINDS["empty_slots"]
    body = refusal_wire("empty_slots", spec)
    body["refusal"]["hint"] = "add a slot"
    with pytest.raises(WireParseError, match="unknown field"):
        parse_response(body)


def test_an_untagged_refusal_payload_is_refused():
    body = refusal_wire("empty_slots", {})
    body["refusal"] = {}
    with pytest.raises(WireParseError, match="internally tagged on `kind`"):
        parse_response(body)


def test_hash_mismatch_carries_both_compared_values_and_a_known_hash_kind():
    spec = REFUSAL_KINDS["hash_mismatch"]
    assert set(spec) == {"which", "expected", "actual"}, spec
    body = refusal_wire("hash_mismatch", spec)
    body["refusal"].update({"which": "tokenizer", "expected": "aaa", "actual": "bbb"})
    parsed = parse_refusal_envelope({k: v for k, v in body.items() if k != "status"})
    assert parsed.refusal["which"] == "tokenizer"
    assert parsed.refusal["expected"] == "aaa"
    assert parsed.refusal["actual"] == "bbb"


def test_an_unknown_hash_kind_is_refused():
    body = refusal_wire("hash_mismatch", REFUSAL_KINDS["hash_mismatch"])
    body["refusal"]["which"] = "vibes"
    with pytest.raises(WireParseError, match="unknown HashKind"):
        parse_response(body)


def test_a_refusal_field_of_the_wrong_json_type_is_refused():
    body = refusal_wire("context_length_mismatch", REFUSAL_KINDS["context_length_mismatch"])
    body["refusal"]["declared"] = "999"
    with pytest.raises(WireParseError, match="JSON integer"):
        parse_response(body)


# -- the two vocabularies, pinned from the Python side too -----------------------------------


def test_the_four_context_checks_are_named_identically_in_both_lanes():
    """``docs/schema-api.md``, *"The four refusals, named identically in both lanes"*."""
    from qd_data import errors as py_errors

    python_checks = {
        cls.check
        for cls in vars(py_errors).values()
        if isinstance(cls, type)
        and issubclass(cls, py_errors.QdRefusal)
        and cls is not py_errors.QdRefusal
    }
    for name in CONTEXT_CHECKS_NAMED_IDENTICALLY:
        assert name in REFUSAL_KINDS, f"{name} is no longer a Rust refusal kind"
        assert name in python_checks, (
            f"{name} is one of the four identifiers docs/schema-api.md says both lanes "
            "spell the same; qd_data.errors no longer does"
        )


def test_the_six_known_vocabulary_differences_have_not_widened():
    """``GAP-XLANG-REFUSAL-VOCABULARIES``, mirrored from the Rust test onto this side.

    The Rust suite already pins these six pairs. Pinning them here too means the
    divergence is visible from whichever lane a reader is standing in, and a rename
    on either side fails a test in both.
    """
    from qd_data import errors as py_errors

    python_checks = {
        cls.check
        for cls in vars(py_errors).values()
        if isinstance(cls, type)
        and issubclass(cls, py_errors.QdRefusal)
        and cls is not py_errors.QdRefusal
    }
    for python_check, rust_kind in KNOWN_VOCABULARY_DIFFERENCES:
        assert python_check in python_checks, (
            f"qd_data.errors no longer declares `{python_check}`; the pinned pair "
            f"({python_check}, {rust_kind}) is stale"
        )
        assert rust_kind in REFUSAL_KINDS, (
            f"crates/qd-runtime no longer declares `{rust_kind}`; the pinned pair is stale"
        )
        assert python_check != rust_kind, (
            f"`{python_check}` and `{rust_kind}` now agree — the lanes unified this "
            "identifier, so remove the pair from KNOWN_VOCABULARY_DIFFERENCES here and "
            "in crates/qd-runtime/tests/wire_context_crosslang.rs"
        )
        assert python_check not in REFUSAL_KINDS, (
            f"`{python_check}` became a Rust refusal kind as well as a Python check name; "
            "one identifier now means two things across the lanes"
        )
