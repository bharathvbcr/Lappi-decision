"""The seam: the pinned Python contract must equal the Rust source, field for field.

``python/qd_wire/contract.py`` is generated from ``crates/qd-runtime/src`` and never
hand-typed. These tests re-derive it and assert equality, so the day the XLANG-RS
lane renames ``backend`` or adds a refusal, a **Python** test fails naming it —
rather than Python silently parsing an envelope shape that no longer exists.

That is the whole lesson of ``GAP-RT-WIRE-CONTEXT-ENCODING`` and
``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``: a contract two lanes *read* drifts; a
contract two lanes *execute* cannot.

When the Rust tree is absent (an installed wheel, a partial checkout), the check is
recorded as ``NotRun`` with a reason and the test says so in its skip message. It
is never reported as a pass.
"""

from __future__ import annotations

import pytest
from qd_train.tristate import NotRun, Ran
from qd_wire import contract
from qd_wire.rust_source import (
    RustSourceMissing,
    derive_contract,
    extract_enum_variants,
    rust_src_dir,
    snake_case,
)


def _derived_or_not_run():
    """Re-derive the contract, or a ``NotRun`` naming why it could not be read."""
    try:
        src = rust_src_dir()
    except RustSourceMissing as exc:
        return NotRun(reason=str(exc))
    return derive_contract(src)


@pytest.fixture(scope="module")
def derived():
    result = _derived_or_not_run()
    if isinstance(result, NotRun):
        pytest.skip(f"NOT RUN — {result.reason}")
    return result


def test_the_rust_tree_is_reachable_or_the_check_is_recorded_as_not_run():
    """The tri-state itself: absence must not be indistinguishable from agreement."""
    result = _derived_or_not_run()
    if isinstance(result, NotRun):
        assert not hasattr(result, "passed"), (
            "NotRun grew a `passed` attribute; an unread contract would then render "
            "identically to one that matched"
        )
        pytest.skip(f"NOT RUN — {result.reason}")
    state = Ran(passed=True, detail="crates/qd-runtime/src was read and parsed")
    assert state.passed


def test_struct_fields_match_the_rust_definitions(derived):
    assert derived["structs"] == contract.STRUCT_FIELDS, (
        "python/qd_wire/contract.py::STRUCT_FIELDS no longer matches "
        "crates/qd-runtime/src/schema.rs. Regenerate with "
        "`PYTHONPATH=python python -m qd_wire.rust_source` and update the parser for "
        "whatever moved — this is the field-level drift the golden corpus exists to catch."
    )


def test_every_refusal_kind_and_its_fields_match(derived):
    assert derived["refusal_kinds"] == contract.REFUSAL_KINDS, (
        "the refusal vocabulary changed in crates/qd-runtime/src/refusal.rs. A caller "
        "branches on `refusal.kind`, so an added, removed or retyped variant is a "
        "wire change, not an internal one."
    )


def test_every_backend_error_kind_and_its_fields_match(derived):
    assert derived["backend_error_kinds"] == contract.BACKEND_ERROR_KINDS


def test_the_status_discriminant_and_its_three_variants_match(derived):
    assert derived["status_tags"] == contract.STATUS_TAGS


def test_the_scalar_tables_match(derived):
    assert tuple(derived["hash_kinds"]) == contract.HASH_KINDS
    assert tuple(derived["supported_schema_versions"]) == contract.SUPPORTED_SCHEMA_VERSIONS
    assert tuple(derived["slot_kinds"]) == contract.SLOT_KINDS
    assert tuple(derived["routes"]) == contract.ROUTES


def test_the_extractor_refuses_to_answer_with_nothing():
    """An extractor that returns ``{}`` on a parse failure would make every test above pass.

    Three distinct failure modes, because they take three different exits. A mutation
    run caught this: the "declaration not found" case alone left the *empty body* guard
    — the one that matters, since a body that parses to nothing is what a silent
    regex drift produces — completely unexercised.
    """
    from qd_wire.rust_source import RustParseError, extract_struct_fields

    with pytest.raises(RustParseError, match="declaration not found"):
        extract_enum_variants("// nothing here\n", "Refusal")

    with pytest.raises(RustParseError, match="parsed zero variants"):
        extract_enum_variants("pub enum Refusal {\n    // all gone\n}\n", "Refusal")

    with pytest.raises(RustParseError, match="parsed zero fields"):
        extract_struct_fields("pub struct AnswerEnvelope {\n}\n", "AnswerEnvelope")


def test_the_extractor_found_a_plausible_number_of_kinds(derived):
    """A parse that silently matched one variant would satisfy every equality above.

    ``docs/schema-api.md`` says the answer side leaves "refusal identifiers for the
    other 32 conditions" unspecified beyond its table of four, so a table of 36 is
    the number the document implies. The floor is asserted rather than the exact
    count, which would make every new refusal a two-line edit here.
    """
    assert len(derived["refusal_kinds"]) >= 30, derived["refusal_kinds"]
    assert len(derived["backend_error_kinds"]) >= 10, derived["backend_error_kinds"]


def test_serde_snake_case_is_reproduced_exactly():
    """The tag on the wire is serde's rename, not Python's idea of snake_case.

    ``ContextNotBase64`` is the case that separates the two: an implementation that
    splits before digit runs yields ``context_not_base_64`` and every refusal of
    that kind would be read as unknown.
    """
    assert snake_case("ContextNotBase64") == "context_not_base64"
    assert snake_case("ContextNotUtf8") == "context_not_utf8"
    assert snake_case("HashMismatch") == "hash_mismatch"
    assert snake_case("EmptySlots") == "empty_slots"
    assert snake_case("RegisteredRouteSpanUnsupported") == "registered_route_span_unsupported"


def test_kind_and_the_serde_tag_are_cross_checked_inside_the_rust(derived):
    """``Refusal::kind()`` and ``#[serde(rename_all)]`` are two declarations.

    :func:`derive_contract` refuses to build a table when they disagree, so reaching
    this assertion at all is the evidence. The table's keys are then known to be the
    strings that actually appear on the wire.
    """
    for kind in derived["refusal_kinds"]:
        assert kind == kind.lower()
        assert " " not in kind
