"""The primitive readers, tested where the envelope parsers cannot reach them.

:func:`qd_wire.answer.parse_slot_value` has its own shape guard, so a test that feeds
it ``True`` proves nothing about :func:`qd_wire.values.as_uint` — the guard fires
first and the reader is never called. That is a real gap and a mutation run found it:
disabling ``as_uint``'s bool check left every envelope test green, because every
integer that reaches ``as_uint`` comes from a **refusal payload** field, and the
envelope tests did not exercise one of the wrong type for each Rust integer type.

So the readers are tested directly here, once per Rust type the generated contract
table actually names.
"""

from __future__ import annotations

import math

import pytest

from qd_wire.contract import BACKEND_ERROR_KINDS, HASH_KINDS, REFUSAL_KINDS
from qd_wire.errors import WireParseError
from qd_wire.values import as_bool, as_finite_float, as_str, as_uint, check_rust_typed

#: Every Rust type the generated tables name. Derived, not listed, so a type the
#: XLANG-RS lane introduces shows up here as an unhandled case rather than going
#: unexercised.
RUST_TYPES_IN_USE = sorted(
    {ty for spec in REFUSAL_KINDS.values() for ty in spec.values()}
    | {ty for spec in BACKEND_ERROR_KINDS.values() for ty in spec.values()}
)
INTEGER_TYPES = [t for t in RUST_TYPES_IN_USE if t in {"usize", "u32", "u64"}]


def test_the_contract_only_uses_rust_types_this_module_can_validate():
    """A field shape with no reader must stop Python, not pass through it."""
    handled = {
        "usize", "u32", "u64", "String", "bool", "f64", "Vec<String>", "Vec<u32>", "HashKind",
    }
    unhandled = sorted(set(RUST_TYPES_IN_USE) - handled)
    assert not unhandled, (
        f"crates/qd-runtime declares refusal/error fields typed {unhandled}, which "
        "qd_wire.values cannot validate. Add a reader rather than letting the field "
        "through unchecked."
    )
    assert INTEGER_TYPES, "the contract names no integer field types, which cannot be right"


# -- as_uint ----------------------------------------------------------------------------------


def test_as_uint_refuses_a_json_boolean():
    """``True`` is an ``int`` in Python. A naive reader turns ``true`` into the bin ``1``."""
    with pytest.raises(WireParseError, match="JSON boolean"):
        as_uint(True, path="$.bins")
    with pytest.raises(WireParseError, match="JSON boolean"):
        as_uint(False, path="$.bins")


def test_as_uint_refuses_a_float_because_serde_json_does():
    """``serde_json`` refuses ``3.0`` for a ``u32``; Python must not be the lenient side."""
    with pytest.raises(WireParseError, match="JSON integer"):
        as_uint(3.0, path="$.bins")
    with pytest.raises(WireParseError, match="JSON integer"):
        as_uint(0.0, path="$.bins")


def test_as_uint_refuses_a_negative_integer():
    with pytest.raises(WireParseError, match="unsigned"):
        as_uint(-1, path="$.offset")


def test_as_uint_accepts_zero_and_large_integers():
    assert as_uint(0, path="$") == 0
    assert as_uint(2**63, path="$", rust_type="u64") == 2**63


@pytest.mark.parametrize("rust_type", INTEGER_TYPES)
def test_check_rust_typed_refuses_a_boolean_for_every_integer_type(rust_type):
    with pytest.raises(WireParseError, match="JSON boolean"):
        check_rust_typed(True, rust_type, path=f"$.{rust_type}")


@pytest.mark.parametrize("rust_type", INTEGER_TYPES)
def test_check_rust_typed_refuses_a_float_for_every_integer_type(rust_type):
    with pytest.raises(WireParseError, match="JSON integer"):
        check_rust_typed(1.5, rust_type, path=f"$.{rust_type}")


# -- the other readers ------------------------------------------------------------------------


def test_as_str_refuses_everything_that_is_not_a_string():
    assert as_str("x", path="$") == "x"
    for bad in (1, 1.0, True, None, [], {}):
        with pytest.raises(WireParseError, match="JSON string"):
            as_str(bad, path="$")


def test_as_bool_refuses_the_integers_that_python_would_accept_as_truthy():
    assert as_bool(True, path="$") is True
    for bad in (1, 0, "true", None):
        with pytest.raises(WireParseError, match="JSON boolean"):
            as_bool(bad, path="$")


def test_as_finite_float_refuses_null_nan_and_infinity():
    assert as_finite_float(0.5, path="$") == 0.5
    assert as_finite_float(1, path="$") == 1.0
    with pytest.raises(WireParseError, match="null"):
        as_finite_float(None, path="$")
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(WireParseError, match="finite"):
            as_finite_float(bad, path="$")
    with pytest.raises(WireParseError, match="boolean"):
        as_finite_float(True, path="$")


def test_vec_readers_refuse_a_wrong_element_type_rather_than_dropping_it():
    assert check_rust_typed(["a", "b"], "Vec<String>", path="$") == ("a", "b")
    assert check_rust_typed([1, 2], "Vec<u32>", path="$") == (1, 2)
    with pytest.raises(WireParseError, match=r"\[1\]"):
        check_rust_typed(["a", 2], "Vec<String>", path="$")
    with pytest.raises(WireParseError, match="JSON array"):
        check_rust_typed("a", "Vec<String>", path="$")


def test_a_hash_kind_outside_the_builds_enum_is_refused():
    assert check_rust_typed(HASH_KINDS[0], "HashKind", path="$") == HASH_KINDS[0]
    with pytest.raises(WireParseError, match="unknown HashKind"):
        check_rust_typed("vibes", "HashKind", path="$")


def test_an_unknown_rust_type_is_refused_rather_than_passed_through():
    """The loud failure that keeps a new field shape from going unvalidated."""
    with pytest.raises(WireParseError, match="no reader for the Rust type"):
        check_rust_typed({"secs": 1}, "Duration", path="$.timeout")
