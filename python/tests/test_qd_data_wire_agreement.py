"""Where ``qd_data`` and ``qd-runtime`` must say the same thing about a request.

``python/tests/test_wire_schema.py`` defends the *context* encoding. This file defends
the four remaining disagreements between the request **builder** in ``python/qd_data``
and the request **validator** in ``crates/qd-runtime/src/wire.rs``:

* ``GAP-XLANG-UNKNOWN-FIELD-LENIENCY`` — Rust refuses an unknown top-level field and
  Python read straight past it, so the same misspelling was a hard refusal in serving
  and a silently defaulted field in training.
* ``GAP-XLANG-NO-PY-HASH-EXPECTATION`` — the five hashes ``docs/schema-api.md`` lets a
  caller pin had no Python producer.
* ``GAP-XLANG-SPAN-THREE-SPELLINGS`` — the runtime spelling of a line span carries
  ``1 <= start_line <= end_line`` and the gold spelling carried nothing, with nothing
  converting between them.
* ``GAP-XLANG-REFUSAL-VOCABULARIES`` — the two lanes name the same condition with
  different identifiers.

**Nothing here retypes a Rust value.** The allowlist, the five field names and the
refusal-kind vocabulary are all read out of ``crates/qd-runtime/src`` at test time —
directly for ``REQUEST_KEYS``, and through :mod:`qd_wire.contract` (itself re-derived
from the Rust on every run by ``test_wire_contract_matches_rust.py``) for the rest. Two
of this session's defects were a transcribed constant; a third was a transcribed field
name (``weights_hash``).
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

from qd_data import errors as qd_errors
from qd_data.errors import QdRefusal
from qd_data.rows import GoldAnswer
from qd_data.schema import ChoiceSlot, Request
from qd_wire.contract import REFUSAL_KINDS, STRUCT_FIELDS
from qd_wire.rust_source import RustParseError, rust_src_dir

REPO = Path(__file__).resolve().parents[2]

# Imported inside the tests that need it, not at module scope. A collection error would
# fail every test in this file at once, and a test that fails because its *module* would
# not import is not evidence that the test itself bites.


def _request(**changes: object) -> Request:
    kwargs: dict[str, object] = {
        "task": "devcouncil.verdict",
        "context": b"fn add(a: i32) -> i32 {\n    a + 1\n}\n",
        "question": "Is the change safe?",
        "slots": (ChoiceSlot(name="verdict", options=("stub", "clean")),),
    }
    kwargs.update(changes)
    return Request(**kwargs)  # type: ignore[arg-type]


# ==========================================================================================
# The allowlist, read out of the Rust rather than retyped
# ==========================================================================================

#: ``const REQUEST_KEYS: &[&str] = &[ ... ];`` in ``crates/qd-runtime/src/wire.rs``.
_REQUEST_KEYS_DECL = re.compile(
    r"\bconst\s+REQUEST_KEYS\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\]\s*;", re.DOTALL
)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_STR_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def rust_request_keys() -> tuple[str, ...]:
    """The top-level keys ``wire.rs`` admits, parsed from the Rust source.

    Raises rather than returning a short list. An extractor that yields ``()`` on a parse
    failure would make the comparison below pass against nothing, which is the shape of
    the vacuous check this repo has already caught five times this session.
    """
    src = (rust_src_dir() / "wire.rs").read_text(encoding="utf-8")
    body = _REQUEST_KEYS_DECL.search(src)
    if body is None:
        raise RustParseError(
            "const REQUEST_KEYS: &[&str] not found in wire.rs. The allowlist "
            "python/qd_data/schema.py mirrors could not be read, so this comparison did "
            "not run and must not be reported as agreement."
        )
    keys = tuple(m.group(1) for m in _STR_LITERAL.finditer(_LINE_COMMENT.sub("", body.group(1))))
    if len(keys) < 5:
        raise RustParseError(f"REQUEST_KEYS parsed as {keys!r}, which is too short to be it")
    return keys


def test_the_allowlist_is_read_out_of_the_rust_and_is_not_a_second_copy() -> None:
    from qd_data.schema import WIRE_REQUEST_KEYS

    assert set(WIRE_REQUEST_KEYS) == set(rust_request_keys()), (
        "python/qd_data/schema.py::WIRE_REQUEST_KEYS and "
        "crates/qd-runtime/src/wire.rs::REQUEST_KEYS have drifted. One of the lanes now "
        "reads a field the other refuses, which is GAP-XLANG-UNKNOWN-FIELD-LENIENCY "
        "reopening from the other direction."
    )


def test_an_unknown_top_level_field_is_refused_rather_than_ignored() -> None:
    """The gap itself. ``wire.rs::known_keys`` refuses this payload as
    ``malformed_request``; ``from_wire`` used to read straight past it, so a caller who
    misspelled ``context_len`` was told by the runtime and not by the training lane."""
    wire = _request().to_wire()
    wire["totally_unknown_field"] = 1
    with pytest.raises(qd_errors.MalformedRequestRefusal) as caught:
        Request.from_wire(wire)
    assert caught.value.check == "malformed_request", (
        "the refusal identifier must be the one Rust uses for this condition; a new "
        "refusal spelled differently would widen GAP-XLANG-REFUSAL-VOCABULARIES"
    )
    assert "totally_unknown_field" in str(caught.value)


def test_a_misspelled_known_field_names_the_misspelling_not_the_absence() -> None:
    """Why the refusal is worth having at all: without it, ``contextlen`` is reported as
    ``context_len is required`` with the field visibly present in the caller's payload."""
    wire = _request().to_wire()
    wire["contextlen"] = wire.pop("context_len")
    with pytest.raises(qd_errors.MalformedRequestRefusal) as caught:
        Request.from_wire(wire)
    assert "contextlen" in str(caught.value)


def test_the_retired_context_field_is_still_named_before_the_unknown_field_check() -> None:
    """Order matters and Rust fixes it: ``wire.rs::validate`` tests ``context`` *before*
    ``known_keys`` so a caller still sending the retired form is told what replaced it
    rather than that a field it has always sent is unknown. ``context`` is not on the
    allowlist, so a Python check in the other order would answer ``malformed_request``."""
    wire = _request().to_wire()
    wire["context"] = [102, 110]
    with pytest.raises(qd_errors.ContextNotBytesRefusal):
        Request.from_wire(wire)


@pytest.mark.parametrize("key", ["expect", "example_id", "metadata", "route"])
def test_every_optional_key_the_allowlist_names_is_still_accepted(key: str) -> None:
    """A too-narrow allowlist refuses requests the runtime answers. Each of these is on
    ``REQUEST_KEYS`` and must survive the new loop."""
    wire = _request().to_wire()
    wire.setdefault(key, {} if key in {"expect", "metadata"} else "generic")
    Request.from_wire(wire)


def test_what_to_wire_emits_is_exactly_what_from_wire_admits() -> None:
    """The round-trip closure, which is the property that actually protects a caller: the
    builder cannot emit a key its own parser, or the runtime's, would refuse."""
    from qd_data.schema import WIRE_REQUEST_KEYS, HashExpectation

    emitted = set(
        _request(expect=HashExpectation(tokenizer_hash="a" * 64), metadata={"k": "v"}).to_wire()
    )
    assert emitted <= set(WIRE_REQUEST_KEYS), sorted(emitted - set(WIRE_REQUEST_KEYS))


# ==========================================================================================
# GAP-XLANG-NO-PY-HASH-EXPECTATION
# ==========================================================================================

FIVE_HASHES = {
    "tokenizer_hash": "a" * 64,
    "weight_hash": "b" * 64,
    "head_hash": "c" * 64,
    "label_set_hash": "d" * 64,
    "calibration_hash": "e" * 64,
}


def test_the_five_pins_are_the_rust_structs_fields_and_not_a_retyped_list() -> None:
    """``weights_hash`` for ``weight_hash`` cost a whole lane. The names come from the
    generated contract table, which ``test_wire_contract_matches_rust.py`` re-derives from
    ``crates/qd-runtime/src/schema.rs`` on every run."""
    from qd_data.schema import HashExpectation

    declared = {f.name for f in dataclasses.fields(HashExpectation)}
    assert declared == set(STRUCT_FIELDS["HashExpectation"])


def test_a_training_lane_request_can_now_pin_all_five_hashes() -> None:
    from qd_data.schema import HashExpectation

    wire = _request(expect=HashExpectation(**FIVE_HASHES)).to_wire()
    assert wire["expect"] == FIVE_HASHES


def test_the_expect_block_survives_the_round_trip_unchanged() -> None:
    from qd_data.schema import HashExpectation

    original = _request(expect=HashExpectation(**FIVE_HASHES))
    assert Request.from_wire(original.to_wire()).expect == original.expect


def test_an_empty_expectation_emits_no_expect_key_at_all() -> None:
    """``expect`` is optional and absent correctly means "pinned nothing".
    ``HashExpectation::default()`` is what the runtime substitutes, so emitting five
    nulls would be a second spelling of the same state."""
    from qd_data.schema import HashExpectation

    assert "expect" not in _request().to_wire()
    assert HashExpectation().is_empty()


def test_a_partially_filled_expectation_emits_only_the_pins_it_carries() -> None:
    from qd_data.schema import HashExpectation

    wire = _request(expect=HashExpectation(weight_hash="b" * 64)).to_wire()
    assert wire["expect"] == {"weight_hash": "b" * 64}


def test_an_unknown_hash_field_is_refused_rather_than_ignored() -> None:
    """``HashExpectation`` is ``deny_unknown_fields`` in Rust. A caller who wrote
    ``weights_hash`` must be told here too, or the two lanes disagree about the same
    typo that ``GAP-XLANG-EXPECT-FIELD-NAME-DOC-DRIFT`` was about."""
    wire = _request().to_wire()
    wire["expect"] = {"weights_hash": "b" * 64}
    with pytest.raises(qd_errors.MalformedRequestRefusal) as caught:
        Request.from_wire(wire)
    assert "weights_hash" in str(caught.value)


@pytest.mark.parametrize("bad", [17, ["a"], {"x": 1}, b"a" * 64])
def test_a_pin_that_is_not_a_string_is_refused(bad: object) -> None:
    """Rust types the five as ``Option<String>``; a non-string is a serde type error
    there, so it must not be coerced to one here."""
    from qd_data.schema import HashExpectation

    with pytest.raises(qd_errors.MalformedRequestRefusal):
        HashExpectation(tokenizer_hash=bad)  # type: ignore[arg-type]


def test_an_expect_block_that_is_not_an_object_is_refused() -> None:
    wire = _request().to_wire()
    wire["expect"] = "a" * 64
    with pytest.raises(qd_errors.MalformedRequestRefusal):
        Request.from_wire(wire)


def test_pinning_a_hash_does_not_change_the_rest_of_the_envelope() -> None:
    """The five pins are a caller's assertion about the *build*, not part of the prompt.
    A request that pins them must render the same bytes as one that does not, or the
    training and serving prompts drift on a field neither model ever sees."""
    from qd_data.render import render
    from qd_data.schema import HashExpectation

    plain = _request()
    pinned = _request(expect=HashExpectation(**FIVE_HASHES))
    assert render(pinned, seed=None).prompts() == render(plain, seed=None).prompts()


# ==========================================================================================
# GAP-XLANG-SPAN-THREE-SPELLINGS
# ==========================================================================================


def test_the_gold_spelling_is_still_a_two_element_array() -> None:
    """Not unified, deliberately: a training label and a runtime answer answer different
    questions. What is unified below is the *rule*, not the shape."""
    assert GoldAnswer(slot_name="evidence", value=(41, 47)).to_json()["value"] == [41, 47]


@pytest.mark.parametrize(
    ("label", "value", "says"),
    [
        ("backwards", (47, 41), "runs backwards"),
        ("zero start", (0, 5), "1-based"),
        ("zero both", (0, 0), "1-based"),
        ("negative", (-1, 5), "1-based"),
        ("one element", (41,), "exactly two values"),
        ("three elements", (41, 47, 50), "exactly two values"),
        ("not integers", ("41", "47"), "must be an int"),
        ("bool start", (True, 5), "must be an int"),
    ],
)
def test_a_gold_span_that_could_not_become_a_runtime_span_is_refused_at_construction(
    label: str, value: object, says: str
) -> None:
    """``GAP-XLANG-SPAN-BOUNDS-UNPINNED`` closed the runtime spelling on both sides and
    left the gold spelling unchecked, so a backwards gold span was constructible and blew
    up only at whatever later read it. It is the same rule; it is now in both places.

    The message is asserted, not only the exception type. Loosening the arity check from
    ``!= 2`` to ``< 2`` survived an earlier version of this test: a three-element span
    still raised, but from tuple unpacking, with Python's *"too many values to unpack"*
    rather than anything naming the span. A refusal that fires for the wrong reason
    reports the same result as one that fires for the right one.
    """
    with pytest.raises(ValueError, match=says):
        GoldAnswer(slot_name="evidence", value=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [(1, 1), (41, 47), (1, 10_000)])
def test_a_legal_gold_span_is_still_legal(value: tuple[int, int]) -> None:
    """The one-line span ``(1, 1)`` is legal and an off-by-one on the ordering check
    would refuse it while passing every "it refuses bad input" test."""
    assert GoldAnswer(slot_name="evidence", value=value).to_json()["value"] == list(value)


def test_the_converter_output_parses_as_the_runtime_spelling() -> None:
    """The named function ``GAP-XLANG-SPAN-THREE-SPELLINGS`` asks for, driven into the
    runtime parser rather than compared against a second copy of its field names."""
    from qd_data.rows import gold_span_to_wire
    from qd_wire.answer import SpanValue, parse_slot_value

    assert parse_slot_value(gold_span_to_wire((41, 47)), path="$.value") == SpanValue(
        start_line=41, end_line=47
    )


def test_the_conversion_round_trips_in_both_directions() -> None:
    from qd_data.rows import gold_span_to_wire, wire_span_to_gold

    for value in [(1, 1), (41, 47), (7, 900)]:
        assert wire_span_to_gold(gold_span_to_wire(value)) == value


@pytest.mark.parametrize("value", [(47, 41), (0, 5), (41,), (41, 47, 50), ("41", "47"), None])
def test_the_converter_refuses_exactly_what_the_runtime_parser_refuses(value: object) -> None:
    """Both halves asserted, so the converter cannot become a hole in the rule: a
    span-shaped value this refuses is one the runtime parser refuses too. Asserted by
    running both, not by comparing two copies of the bounds expression."""
    from qd_data.rows import gold_span_to_wire
    from qd_wire.answer import parse_slot_value
    from qd_wire.errors import WireParseError

    with pytest.raises(ValueError):
        gold_span_to_wire(value)  # type: ignore[arg-type]
    with pytest.raises((WireParseError, TypeError, ValueError)):
        parse_slot_value(value, path="$.value")


@pytest.mark.parametrize(("value", "kind"), [(41, "score"), ("clean", "choice")])
def test_a_value_of_another_slot_kind_is_refused_as_a_span_and_read_as_itself(
    value: object, kind: str
) -> None:
    """The one place the two do *not* agree, stated rather than left out of the list
    above. ``41`` is a legal ``SlotValue`` — a score bin — so the runtime parser reads
    it; it is not a span, so the converter refuses it. A converter that coerced it
    would turn a score answer into the line range ``41..41``."""
    from qd_data.rows import gold_span_to_wire
    from qd_wire.answer import parse_slot_value

    with pytest.raises(ValueError):
        gold_span_to_wire(value)  # type: ignore[arg-type]
    assert parse_slot_value(value, path="$.value").kind == kind


def test_the_reverse_converter_refuses_a_span_object_the_runtime_would_refuse() -> None:
    from qd_data.rows import wire_span_to_gold

    for bad in [
        {"start_line": 47, "end_line": 41},
        {"start_line": 0, "end_line": 5},
        {"start_line": 1},
        {"start_line": 1, "end_line": 2, "confidence": 0.5},
        [41, 47],
    ]:
        with pytest.raises(ValueError):
            wire_span_to_gold(bad)  # type: ignore[arg-type]


def test_the_converter_field_names_come_from_the_generated_contract() -> None:
    from qd_data.rows import gold_span_to_wire

    assert set(gold_span_to_wire((1, 2))) == set(STRUCT_FIELDS["SpanValue"])


def test_no_second_span_type_was_introduced_in_qd_data() -> None:
    """The gap's action is "converts explicitly, in one named function, and does not add
    a third spelling". A ``SpanValue`` class here would be exactly that third spelling."""
    for module in _qd_data_modules():
        assert not hasattr(module, "SpanValue"), (
            f"{module.__name__} grew a SpanValue. The runtime spelling has one owner per "
            "language; this lane converts to it and does not re-declare it."
        )


# ==========================================================================================
# GAP-XLANG-REFUSAL-VOCABULARIES
# ==========================================================================================


def _qd_data_modules() -> list[object]:
    """Every module in the package, imported. A walk over ``__subclasses__`` only sees
    classes whose module has been imported, so a scan that imports a hand-picked list is
    a scan that reports "no new refusal" for a module it never loaded."""
    import qd_data

    mods = []
    for info in pkgutil.iter_modules(qd_data.__path__):
        mods.append(importlib.import_module(f"qd_data.{info.name}"))
    assert len(mods) >= 10, f"only {len(mods)} qd_data modules were imported; the walk broke"
    return mods


def _refusal_classes() -> list[type[QdRefusal]]:
    _qd_data_modules()
    out: list[type[QdRefusal]] = []
    stack = [QdRefusal]
    while stack:
        cls = stack.pop()
        subs = cls.__subclasses__()
        stack.extend(subs)
        out.extend(subs)
    assert len(out) >= 20, f"only {len(out)} QdRefusal subclasses were found; the walk broke"
    return out


#: Classes that answer a question the runtime never asks: licensing, held-out paths,
#: upstream row shape, source reachability, row usability. They map to no ``Refusal``
#: kind because the runtime has no such check, and pinning the set is what stops a new
#: refusal defaulting into it unnoticed.
DATA_LANE_ONLY = {
    "HeldOutViolation",
    "LicenceRefused",
    "MalformedRowRefusal",
    "RowRefused",
    "SourceUnavailableRefusal",
}

#: ``Refusal::kind()`` identifiers no ``qd_data`` refusal stands for, because this lane
#: builds requests and does not serve them: head binding, calibration, the control
#: socket, the payload line cap, the slot-name cap, and the one check the builder makes
#: structurally impossible rather than refusing.
#:
#: ``empty_task`` was in this set until 2026-09-19, under the claim that it was one of
#: *two* structurally-impossible checks. It was not impossible, it was unchecked:
#: ``Request(task="")`` built and rendered here while the runtime refuses it. The entry
#: was an unexecuted claim, and the set is exactly where an unexecuted claim does the
#: most damage -- it is the list that says "we looked and there is nothing to find".
#: ``EmptyTaskRefusal`` now stands for the kind and
#: ``test_an_empty_task_is_refused_here_the_way_the_runtime_refuses_it`` is what looked.
NO_QD_DATA_COUNTERPART = {
    "ambiguous_envelope",
    "calibration_entry_missing",
    "payload_over_cap",
    "registered_head_missing",
    "registered_head_slot_missing",
    "registered_route_span_unsupported",
    "slot_field_not_allowed",
    "slot_name_over_cap",
    "unknown_op",
}


def test_every_refusal_declares_which_runtime_kinds_it_stands_for() -> None:
    """The gap's action: "do not add a refusal to one lane without checking whether the
    other already names the same condition differently". Declared per class and checked
    here, so the check happens at the moment the refusal is written."""
    undeclared = sorted(
        cls.__name__ for cls in _refusal_classes() if "rust_kinds" not in cls.__dict__
    )
    assert not undeclared, (
        "these refusals inherit an empty `rust_kinds` rather than declaring one. An "
        f"inherited default is indistinguishable from 'checked, and there is none': {undeclared}"
    )


def test_no_declared_kind_is_one_the_runtime_does_not_have() -> None:
    """Compared against the generated table, so a kind renamed in ``refusal.rs`` fails
    here naming the class that still points at the old spelling."""
    unknown = {
        cls.__name__: sorted(set(cls.rust_kinds) - set(REFUSAL_KINDS))
        for cls in _refusal_classes()
    }
    unknown = {k: v for k, v in unknown.items() if v}
    assert not unknown, (
        f"kinds crates/qd-runtime/src/refusal.rs does not declare: {unknown}"
    )


def test_the_classes_with_no_runtime_counterpart_are_exactly_the_data_lane_ones() -> None:
    empty = {cls.__name__ for cls in _refusal_classes() if not cls.rust_kinds}
    assert empty == DATA_LANE_ONLY


def test_the_four_context_checks_are_named_identically_in_both_lanes() -> None:
    """Already true before this file; asserted from the ``rust_kinds`` side too so the
    mapping cannot claim an identity the ``check`` string contradicts."""
    from qd_wire.refusal import CONTEXT_CHECKS_NAMED_IDENTICALLY

    identical = {
        cls.check for cls in _refusal_classes() if cls.rust_kinds == (cls.check,)
    }
    assert set(CONTEXT_CHECKS_NAMED_IDENTICALLY) <= identical


def test_every_runtime_kind_is_either_mapped_or_recorded_as_unmapped() -> None:
    """The half a per-class mapping cannot give you: a kind added to ``refusal.rs`` that
    nothing on this side stands for. It lands in neither set and fails here."""
    mapped = {k for cls in _refusal_classes() for k in cls.rust_kinds}
    assert mapped | NO_QD_DATA_COUNTERPART == set(REFUSAL_KINDS), {
        "unaccounted": sorted(set(REFUSAL_KINDS) - mapped - NO_QD_DATA_COUNTERPART),
        "claimed but gone": sorted(NO_QD_DATA_COUNTERPART - set(REFUSAL_KINDS)),
    }
    assert not (mapped & NO_QD_DATA_COUNTERPART)


def test_the_six_pinned_pairs_are_all_explained_by_the_mapping() -> None:
    """``KNOWN_VOCABULARY_DIFFERENCES`` is a flat list of six pairs in three files and
    says nothing about *why* they differ. Here each pair is resolved against the class
    that produces it, which is what makes the next question answerable."""
    from qd_wire.refusal import KNOWN_VOCABULARY_DIFFERENCES

    by_check = {cls.check: cls for cls in _refusal_classes()}
    for check, rust_kind in KNOWN_VOCABULARY_DIFFERENCES:
        cls = by_check.get(check)
        assert cls is not None, f"no qd_data refusal carries check {check!r} any more"
        assert rust_kind in cls.rust_kinds, (
            f"{cls.__name__} does not list {rust_kind!r} among the kinds it stands for, "
            f"but KNOWN_VOCABULARY_DIFFERENCES pairs it with {check!r}"
        )


def test_which_of_the_pinned_pairs_are_renames_and_which_are_conflations() -> None:
    """The finding the flat table hides. Four of the six Python classes stand for
    *several* runtime kinds, so renaming them to the runtime identifier would attach a
    failure name to a class that also fires for other failures — ``options_len`` is both
    ``too_many_options`` and ``too_few_options``. Only the two one-to-one pairs are
    renameable without first splitting the Python class.
    """
    from qd_wire.refusal import KNOWN_VOCABULARY_DIFFERENCES

    by_check = {cls.check: cls for cls in _refusal_classes()}
    renameable = {c for c, _ in KNOWN_VOCABULARY_DIFFERENCES if len(by_check[c].rust_kinds) == 1}
    assert renameable == {"schema_version", "slot_names_unique"}


def test_a_seventh_divergence_the_flat_table_never_listed() -> None:
    """``context_bytes`` is one Python class over four runtime kinds — the context cap,
    the context encoding, the question cap and the task cap. It is absent from
    ``KNOWN_VOCABULARY_DIFFERENCES`` because the cross-language negative set never sends
    an over-cap payload, so the pin that was supposed to make the divergence visible
    could not see it. ``GAP-XLANG-REFUSAL-VOCABULARIES``.
    """
    from qd_wire.refusal import KNOWN_VOCABULARY_DIFFERENCES

    assert qd_errors.ContextTooLargeRefusal.rust_kinds == (
        "context_over_cap",
        "context_not_utf8",
        "question_over_cap",
        "task_over_cap",
    )
    assert "context_bytes" not in {c for c, _ in KNOWN_VOCABULARY_DIFFERENCES}


def test_an_empty_task_is_refused_here_the_way_the_runtime_refuses_it() -> None:
    """``empty_task`` sat in ``NO_QD_DATA_COUNTERPART`` under the claim that it is one of
    "the two checks the builder makes **structurally impossible** rather than refusing".

    It was not structurally impossible. It was unchecked: ``Request(task="")`` built and
    rendered on this lane while ``wire::validate`` refuses it as ``empty_task`` on the
    other, so the training lane could emit a row carrying a task the runtime will not
    serve. The entry was a claim that nothing executed -- the same shape as the
    ``non_whitespace_len`` comment that produced ``GAP-XLANG-REFUSAL-VOCABULARIES``'s
    sibling on the Rust side.

    Blankness is ``str.strip()``, which is also what the runtime now asks via
    ``qd_runtime::is_blank``; ``crates/qd-runtime/tests/wire_context_crosslang.rs``
    compares the two notions codepoint for codepoint.
    """
    for task in ["", " ", "\t", "\xa0", "\x1c", " \x1c\t"]:
        with pytest.raises(qd_errors.EmptyTaskRefusal):
            _request(task=task)

    # The control: a task with content still builds, including one that merely contains
    # a separator. The rule is about blankness, not about the character.
    assert _request(task="devcouncil.verdict").task == "devcouncil.verdict"
    assert _request(task="\x1cverdict").task == "\x1cverdict"


def test_empty_task_is_no_longer_claimed_to_have_no_counterpart_here() -> None:
    """The record and the code have to agree about which kinds this lane cannot raise."""
    assert "empty_task" not in NO_QD_DATA_COUNTERPART, (
        "qd_data now refuses an empty task, so listing the kind as having no counterpart "
        "here is a claim that has stopped being true"
    )
    assert qd_errors.EmptyTaskRefusal.rust_kinds == ("empty_task",)
