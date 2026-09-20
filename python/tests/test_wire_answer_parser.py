"""The answer envelope, parsed strictly.

``docs/schema-api.md`` shows the answer as one example object and then lists, under
*"What this document still does not specify"*, exactly what these tests pin:

* no ``status`` discriminant, though the runtime emits a three-variant one;
* no field-presence rules — in particular whether ``value`` is absent iff ``noul``;
* whether unknown fields are refused or ignored.

Every payload below is built from :mod:`qd_wire.contract`'s generated field tables
rather than typed out, so a field the XLANG-RS lane renames breaks the *fixtures*
these tests use as well as the parser, instead of leaving a stale literal that
keeps passing.
"""

from __future__ import annotations

import pytest

from qd_wire.answer import (
    ChoiceSet,
    ChoiceValue,
    ScoreSet,
    ScoreValue,
    SlotAnswer,
    SpanValue,
    parse_answer_envelope,
    parse_conformal_set,
    parse_slot_answer,
    parse_slot_value,
)
from qd_wire.contract import MAX_SLOT_NAME_BYTES, STRUCT_FIELDS
from qd_wire.errors import WireParseError
from qd_wire.response import CallerReading, caller_reading, parse_response, parse_response_line


def slot(**over):
    """A valid ``SlotAnswer`` payload, with the full key set from the contract table."""
    body = {
        "value": "stub",
        "conformal_set": ["stub", "logic"],
        "score": 0.71,
        "noul": False,
        "degraded": False,
    }
    assert set(body) == set(STRUCT_FIELDS["SlotAnswer"]), (
        "this fixture no longer carries the SlotAnswer key set the Rust declares; "
        "regenerate qd_wire.contract and fix the fixture"
    )
    body.update(over)
    return body


def envelope(**over):
    body = {
        "status": "ok",
        "schema_version": 1,
        "backend": "reference-deterministic-v1",
        "degraded": False,
        "slots": {"verdict": slot()},
    }
    assert set(body) - {"status"} == set(STRUCT_FIELDS["AnswerEnvelope"])
    body.update(over)
    return body


def body(**over):
    """An envelope body as :func:`parse_answer_envelope` wants it — without ``status``.

    ``status`` belongs to the ``Response`` discriminant and is consumed by the caller, so the
    envelope parser refuses it as an unknown field.
    """
    out = envelope(**over)
    del out["status"]
    return out


# -- the three slot kinds, discriminated the way `#[serde(untagged)]` does ------------------


def test_a_string_is_a_choice_an_integer_is_a_score_an_object_is_a_span():
    assert parse_slot_value("stub", path="$") == ChoiceValue(option="stub")
    assert parse_slot_value(3, path="$") == ScoreValue(bin=3)
    assert parse_slot_value({"start_line": 41, "end_line": 47}, path="$") == SpanValue(
        start_line=41, end_line=47
    )


def test_a_json_boolean_is_not_a_score_bin():
    """Python's ``bool`` is an ``int`` subclass, so ``True`` would read as bin 1."""
    with pytest.raises(WireParseError, match="boolean"):
        parse_slot_value(True, path="$.value")


def test_a_float_is_not_a_score_bin():
    """``serde_json`` refuses ``3.0`` for a ``u32``; accepting it would make Python lenient."""
    with pytest.raises(WireParseError, match="integer"):
        parse_slot_value(3.0, path="$.value")


def test_a_span_is_an_object_and_never_the_two_element_array_gold_answers_use():
    """``GAP-XLANG-SPAN-THREE-SPELLINGS``: ``qd_data.rows.GoldAnswer`` spells a span ``[41, 47]``.

    That is a training label, not a runtime answer, and nothing converts between the
    two. Feeding one to the other must fail here rather than at the point a caller
    reads ``start_line`` off an integer.
    """
    with pytest.raises(WireParseError):
        parse_slot_value([41, 47], path="$.value")


def test_a_span_refuses_an_unknown_or_missing_line_field():
    with pytest.raises(WireParseError, match="unknown field"):
        parse_slot_value({"start_line": 1, "end_line": 2, "start_col": 0}, path="$.value")
    with pytest.raises(WireParseError, match="missing required field"):
        parse_slot_value({"start_line": 1}, path="$.value")


def test_a_backwards_span_is_refused_by_both_readers():
    """``GAP-XLANG-SPAN-BOUNDS-UNPINNED``, settled: ordering is a term of the format.

    This test asserted the opposite. It read::

        inverted = parse_slot_value({"start_line": 47, "end_line": 41}, path="$")
        assert inverted.is_ordered is False

    because the rule lived only in the *producer* — ``crates/qd-runtime/src/answer.rs``
    abstains when the end pointer decodes below the start one — while both readers
    accepted an inverted span and ``SpanValue`` carried no such invariant. Refusing on
    one side alone would have been a second opinion about the format, so the previous
    lane correctly reported it instead. The invariant now belongs to the type on both
    sides, and ``is_ordered`` is gone rather than left always returning ``True``.
    """
    with pytest.raises(WireParseError, match="runs backwards"):
        parse_slot_value({"start_line": 47, "end_line": 41}, path="$")

    assert not hasattr(SpanValue(start_line=1, end_line=2), "is_ordered"), (
        "is_ordered existed only to report an unenforced rule; with the rule enforced it "
        "would always be True, and a predicate that cannot be False invites callers to "
        "keep checking a thing that can no longer happen"
    )


def test_a_span_starting_at_line_zero_is_refused_because_spans_are_one_based():
    """The other half of the same invariant, and the one the doc comment always claimed.

    ``SpanValue`` has said "1-based and inclusive" since it was written, and the producer
    cannot emit 0 — it decodes a row index and adds one. Line 0 was nevertheless accepted
    by both readers, which is the same defect as the ordering one: a rule stated in prose
    and enforced nowhere.
    """
    with pytest.raises(WireParseError, match="1-based"):
        parse_slot_value({"start_line": 0, "end_line": 3}, path="$")


def test_a_single_line_span_and_the_smallest_legal_span_are_accepted():
    """The boundary from the other direction: ``start == end`` is legal, and line 1 is legal.

    An implementation that used ``>=`` for the ordering check, or ``<= 1`` for the 1-based
    check, would refuse the two spans the corpus actually contains
    (``answer-span-only.json`` is ``{2, 2}``) and pass every test above.
    """
    assert parse_slot_value({"start_line": 2, "end_line": 2}, path="$") == SpanValue(
        start_line=2, end_line=2
    )
    assert parse_slot_value({"start_line": 1, "end_line": 1}, path="$") == SpanValue(
        start_line=1, end_line=1
    )


# -- the conformal set ----------------------------------------------------------------------


def test_a_conformal_set_is_all_strings_or_all_integers():
    assert parse_conformal_set(["a", "b"], path="$") == ChoiceSet(options=("a", "b"))
    assert parse_conformal_set([2, 3, 4], path="$") == ScoreSet(bins=(2, 3, 4))


def test_a_mixed_conformal_set_is_refused_rather_than_coerced():
    with pytest.raises(WireParseError, match="mixes"):
        parse_conformal_set(["a", 2], path="$")


def test_an_empty_conformal_set_follows_serdes_variant_order():
    """``[]`` fits both untagged variants; serde takes the first, which is ``Choices``.

    Pinned because it is inherited from declaration order in ``schema.rs`` rather
    than chosen here: reordering the Rust variants would change what an empty set
    means on this side, and this test is what would say so.
    """
    from qd_wire.answer import EMPTY_CONFORMAL_SET_READS_AS

    parsed = parse_conformal_set([], path="$")
    assert EMPTY_CONFORMAL_SET_READS_AS == "choice"
    assert parsed.kind == EMPTY_CONFORMAL_SET_READS_AS
    assert parsed == ChoiceSet(options=())


# -- the value / noul biconditional ----------------------------------------------------------


def test_a_slot_carrying_both_a_value_and_noul_is_refused():
    with pytest.raises(WireParseError, match="nothing for `value` to hold"):
        parse_slot_answer(slot(value="stub", noul=True), path="$")


def test_a_slot_carrying_neither_a_value_nor_an_abstention_is_refused():
    with pytest.raises(WireParseError, match="says nothing"):
        parse_slot_answer(slot(value=None, noul=False), path="$")


def test_an_abstention_parses_and_carries_no_value():
    parsed = parse_slot_answer(slot(value=None, conformal_set=None, noul=True), path="$")
    assert parsed.noul is True
    assert parsed.value is None
    assert parsed.value_kind is None


def test_the_biconditional_cannot_be_bypassed_by_constructing_the_dataclass():
    with pytest.raises(WireParseError):
        SlotAnswer(value=ChoiceValue("a"), conformal_set=None, score=0.5, noul=True, degraded=False)


# -- strictness in both directions -------------------------------------------------------------


def test_an_unknown_field_on_a_slot_answer_is_refused():
    """The point of this package. A field Python ignores is a contract that rots."""
    with pytest.raises(WireParseError, match="unknown field"):
        parse_slot_answer(slot(rationale="because"), path="$")


def test_a_missing_field_on_a_slot_answer_is_refused():
    body = slot()
    del body["degraded"]
    with pytest.raises(WireParseError, match="missing required field"):
        parse_slot_answer(body, path="$")


def test_an_unknown_field_on_the_envelope_is_refused():
    with pytest.raises(WireParseError, match="unknown field"):
        parse_response(envelope(latency_ms=12))


def test_a_null_score_is_refused_because_that_is_how_a_nan_reaches_the_wire():
    with pytest.raises(WireParseError, match="non-finite"):
        parse_slot_answer(slot(score=None), path="$")


def test_the_status_discriminant_is_required():
    body = envelope()
    del body["status"]
    with pytest.raises(WireParseError, match="no `status` discriminant"):
        parse_response(body)


def test_an_unknown_status_is_refused_rather_than_guessed():
    with pytest.raises(WireParseError, match="unknown status"):
        parse_response(envelope(status="partial"))


def test_the_backend_field_the_old_doc_example_omitted_is_required():
    """``docs/schema-api.md``'s answer example has two keys; the runtime emits five.

    ``backend`` is the one that says an answer came from
    ``reference-deterministic-v1`` rather than from a model, so reading an envelope
    without it must fail rather than default.
    """
    body = envelope()
    del body["backend"]
    with pytest.raises(WireParseError, match=r"missing required field.*backend"):
        parse_response(body)


# -- the whole envelope, and how a caller reads it ------------------------------------------


def test_a_full_three_slot_envelope_round_trips_field_for_field():
    parsed = parse_response(
        envelope(
            degraded=True,
            slots={
                "verdict": slot(value="stub", conformal_set=["stub", "logic"], degraded=True),
                "severity": slot(value=3, conformal_set=[2, 3, 4], score=0.55, degraded=True),
                "evidence": slot(
                    value={"start_line": 41, "end_line": 47},
                    conformal_set=None,
                    score=0.62,
                    degraded=True,
                ),
            },
        )
    )
    assert parsed.status == "ok"
    assert parsed.backend == "reference-deterministic-v1"
    assert parsed.degraded is True
    assert parsed.slots["verdict"].value == ChoiceValue("stub")
    assert parsed.slots["severity"].value == ScoreValue(3)
    assert parsed.slots["evidence"].value == SpanValue(41, 47)
    assert parsed.slots["severity"].conformal_set == ScoreSet((2, 3, 4))
    assert parsed.slots["evidence"].conformal_set is None
    assert caller_reading(parsed) is CallerReading.MODEL_ANSWERED


def test_every_slot_abstaining_reads_as_the_model_abstaining():
    parsed = parse_response(
        envelope(slots={"v": slot(value=None, conformal_set=None, noul=True)})
    )
    assert parsed.all_slots_abstained is True
    assert caller_reading(parsed) is CallerReading.MODEL_ABSTAINED


def test_one_answered_slot_among_abstentions_reads_as_answered():
    parsed = parse_response(
        envelope(
            slots={
                "a": slot(value=None, conformal_set=None, noul=True),
                "b": slot(value=2, conformal_set=[2]),
            }
        )
    )
    assert caller_reading(parsed) is CallerReading.MODEL_ANSWERED


def test_a_slotless_ok_envelope_has_no_reading_rather_than_a_default():
    """``all([])`` is ``True``, so a naive reading calls a slotless envelope an abstention.

    ``slots`` empty is ``Refusal::EmptySlots`` before a backend is asked, so such an
    envelope did not come from this runtime and gets no reading at all.
    """
    parsed = parse_response(envelope(slots={}))
    with pytest.raises(WireParseError, match="no caller reading"):
        caller_reading(parsed)


def test_a_reply_line_parses_from_bytes_and_from_str():
    import json

    line = json.dumps(envelope())
    assert parse_response_line(line).backend == "reference-deterministic-v1"
    assert parse_response_line(line.encode()).backend == "reference-deterministic-v1"


def test_a_malformed_line_names_itself_rather_than_raising_a_json_error():
    with pytest.raises(WireParseError, match="not valid JSON"):
        parse_response_line("{not json")


def test_parse_answer_envelope_is_reachable_without_the_discriminant():
    """The envelope parser is separable, for a caller that already dispatched."""
    body = envelope()
    del body["status"]
    assert parse_answer_envelope(body).schema_version == 1


# -- the slot-name cap, the same number on both sides ---------------------------------------
#
# `GAP-RT-SLOT-NAME-UNCAPPED`: a slot name was checked non-empty and never checked for length, in
# neither `wire::parse_slot` nor `render::slot_suffix`, and the answer map's keys were not checked
# at all here. The cap now exists on all four readers, from the single generated constant
# `qd_wire.contract.MAX_SLOT_NAME_BYTES`. These tests are the Python half; the Rust half is
# `wire_refusals.rs::a_slot_name_is_capped_in_bytes_and_the_boundary_is_exact`, and
# `test_wire_contract_matches_rust.py::test_the_slot_name_cap_is_one_number_on_both_sides` is what
# keeps the two halves talking about the same number.


def test_a_slot_key_at_the_cap_is_accepted_and_one_byte_over_is_refused():
    at_cap = "n" * MAX_SLOT_NAME_BYTES
    parsed = parse_answer_envelope(body(slots={at_cap: slot()}))
    assert at_cap in parsed.slots, "the cap is an inclusive maximum"

    with pytest.raises(WireParseError, match="over the slot-name cap"):
        parse_answer_envelope(body(slots={"n" * (MAX_SLOT_NAME_BYTES + 1): slot()}))


def test_the_slot_key_cap_counts_utf8_bytes_and_not_characters():
    """``len(name)`` would accept a 256-character name that Rust refuses at 512 bytes.

    ``é`` is two bytes in UTF-8, so 128 of them is exactly the cap and 129 is two over it while
    still being fewer characters than the cap. A character-counting implementation passes the
    first case and the over-cap case alike, which is why the second assertion is here.
    """
    at_cap = "é" * (MAX_SLOT_NAME_BYTES // 2)
    assert len(at_cap.encode("utf-8")) == MAX_SLOT_NAME_BYTES
    assert at_cap in parse_answer_envelope(body(slots={at_cap: slot()})).slots

    over = "é" * (MAX_SLOT_NAME_BYTES // 2 + 1)
    assert len(over) < MAX_SLOT_NAME_BYTES, "fewer characters than the cap, but more bytes"
    with pytest.raises(WireParseError, match="over the slot-name cap"):
        parse_answer_envelope(body(slots={over: slot()}))


def test_an_empty_slot_key_is_refused_rather_than_carried():
    for blank in ("", "   "):
        with pytest.raises(WireParseError, match="empty key"):
            parse_answer_envelope(body(slots={blank: slot()}))


def test_the_over_cap_message_states_both_numbers_and_does_not_echo_the_key():
    """A refusal whose size the input chooses is a small amplification, and an avoidable one."""
    key = "x" * 4096
    with pytest.raises(WireParseError) as caught:
        parse_answer_envelope(body(slots={key: slot()}))
    message = str(caught.value)
    assert "4096" in message and str(MAX_SLOT_NAME_BYTES) in message
    assert key not in message, "the oversized key must not be echoed back"
    assert len(message) < 512, f"the message must stay bounded, got {len(message)} bytes"
