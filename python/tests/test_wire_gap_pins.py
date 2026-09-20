"""The six XLANG gaps, pinned from the Python side so none can go stale unnoticed.

Each test here is a recorded gap made executable. Several assert an *absence* rather
than a behaviour, which is the pattern
``crates/qd-runtime/tests/wire_context_crosslang.rs`` established: an absence that is
asserted fails the day it stops being true, and demands the missing half. An absence
that is merely written down does not.

The six, and where each stands:

* ``GAP-XLANG-NO-PY-ANSWER-PARSER`` — **closed**, by
  :func:`test_python_now_parses_all_four_answer_side_types`
* ``GAP-XLANG-RS-GUARD-BLIND-TO-NEW-PACKAGE`` — **closed**, by
  :func:`test_the_rust_side_probe_searches_the_package_that_holds_the_parser`
* ``GAP-XLANG-LABEL-SET-HASH-TWO-MEANINGS`` — open, narrowed by the rename to
  ``slot_set_digest``; see
  :func:`test_qd_data_no_longer_spells_a_request_digest_label_set_hash`
* ``GAP-XLANG-NO-PY-HASH-EXPECTATION`` — open; see
  :func:`test_a_training_lane_request_still_pins_no_hashes`
* ``GAP-XLANG-EXPECT-FIELD-NAME-DOC-DRIFT`` — closed by the XLANG-FINISH lane; see
  :func:`test_the_documented_expect_block_deserializes_against_the_struct`
* ``GAP-XLANG-SPAN-THREE-SPELLINGS`` — open; see
  :func:`test_a_gold_answer_still_spells_a_span_as_a_two_element_array`

The two closed ones keep their tests rather than losing them: each now asserts the
*property* that replaced the absence, so it fails if the fix is undone. The
``RS-GUARD`` one in particular was a blind spot this lane found in the other lane's
guard and could not fix directly (``crates/`` is not this lane's); XLANG-RS landed
the fix, and the pin was retired only after the Rust guard was run with this package
hidden and observed to fail for the designed reason.

Two tests here are not gap pins but guards on this file's own honesty:
:func:`test_every_gap_id_this_lane_cites_exists_in_the_ledger` resolves every
``GAP-`` id cited in this lane's code against ``gaps.jsonl``, and
:func:`test_no_docstring_names_a_function_that_does_not_exist` resolves every
``:func:``/``:class:`` cross-reference in this lane's own docstrings. Both exist
because this lane shipped exactly those two defects — a citation with no record, and
this docstring naming a function after it had been renamed.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CROSSLANG_FIXTURES = REPO / "crates" / "qd-runtime" / "tests" / "crosslang_fixtures.py"
GAPS = REPO / "gaps.jsonl"

#: The four answer-side names ``GAP-XLANG-NO-PY-ANSWER-PARSER`` recorded as absent
#: from ``python/``. They are the reason this package exists.
ANSWER_SIDE_TYPES = ("SlotAnswer", "AnswerEnvelope", "RefusalEnvelope", "ErrorEnvelope")

#: Where a ``GAP-`` id may be cited from and still be checked by the test below.
CITING_PATHS = ("python/qd_wire", "python/tests")

#: A ``GAP-`` token, including a trailing glob so ``GAP-XLANG-*`` is recognised as the
#: family pattern it is and discarded whole, rather than being clipped to a shorter
#: id that no record carries.
_GAP_TOKEN = r"GAP-[A-Z0-9*]+(?:-[A-Z0-9*]+)*"


# ==========================================================================================
# every GAP- id this lane cites must exist in the ledger
# ==========================================================================================


def _gap_ids() -> set[str]:
    """Every id in ``gaps.jsonl``, refusing a malformed line rather than skipping it.

    The file is append-only and several lanes write it concurrently, so a line that
    does not parse is a real event — a torn or interleaved append — and silently
    dropping it would shrink the set this test checks against and turn a dangling
    citation into a pass.
    """
    ids: set[str] = set()
    for lineno, line in enumerate(GAPS.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"gaps.jsonl:{lineno} is not valid JSON ({exc}). The file is append-only "
                "and written concurrently; a torn line means an append was not atomic, "
                "and it must be repaired rather than parsed around."
            ) from exc
        assert "id" in record, f"gaps.jsonl:{lineno} has no `id`"
        ids.add(record["id"])
    assert ids, "gaps.jsonl parsed to zero records, which cannot be right"
    return ids


def test_every_gap_id_this_lane_cites_exists_in_the_ledger():
    """A docstring citing a ``GAP-`` id that was never appended is a dangling reference.

    This lane shipped three such citations before they were appended, and it took a
    human reading the diff to notice. That is the case; this is the class. Every
    ``GAP-`` id mentioned anywhere under ``python/qd_wire`` or ``python/tests`` is
    resolved against ``gaps.jsonl`` here, so the next one fails a test instead.
    """
    known = _gap_ids()
    cited: dict[str, set[str]] = {}
    for root in CITING_PATHS:
        for path in sorted((REPO / root).rglob("*.py")):
            for match in re.finditer(_GAP_TOKEN, path.read_text("utf-8")):
                token = match.group(0)
                # A trailing `*` makes the token a glob over a family in prose, not a
                # citation of one record. The `*` is matched as part of the token and
                # dropped here rather than excluded by a negative lookahead: a lookahead
                # makes the pattern backtrack and match a truncated prefix of the id,
                # which then dangles for a different and more confusing reason.
                if "*" in token:
                    continue
                cited.setdefault(token, set()).add(str(path.relative_to(REPO)))

    assert cited, (
        f"no GAP- id is cited anywhere under {CITING_PATHS}, so this test is checking "
        "an empty set. Either the citations were removed or the scan stopped working."
    )
    dangling = {gid: sorted(where) for gid, where in cited.items() if gid not in known}
    assert not dangling, (
        "these GAP- ids are cited in code but are not in gaps.jsonl: "
        + json.dumps(dangling, indent=2, sort_keys=True)
        + "\nAppend the record (printf '%s\\n' '<json>' >> gaps.jsonl, one atomic write) "
        "or fix the citation. A ledger id that does not resolve is worse than no "
        "citation: it reads as evidence that something was recorded."
    )


def test_no_docstring_names_a_function_that_does_not_exist():
    """A ``:func:`` cross-reference to a renamed function is prose asserting a fiction.

    This module's own docstring did it: it described
    ``test_the_rust_side_guard_does_not_yet_see_this_package`` as a live defect for a
    while after that function had been renamed and the defect closed. Small, and
    exactly the family this lane spent the day on — a document and the code
    disagreeing, with nothing that notices.

    So every ``:func:`` and ``:class:`` reference in this lane's own docstrings is
    resolved here. Dotted references are resolved through their module; bare ones
    against the module that wrote them, then against the package.
    """
    ref = re.compile(r":(?:func|class|meth):`~?([A-Za-z_][A-Za-z0-9_.]*)`")
    unresolved: dict[str, list[str]] = {}
    scanned = 0

    for root in CITING_PATHS:
        for path in sorted((REPO / root).rglob("*.py")):
            text = path.read_text("utf-8")
            names = {m.group(1) for m in ref.finditer(text)}
            if not names:
                continue
            scanned += 1
            module = _import_for(path)
            for name in sorted(names):
                if not _resolves(name, module):
                    unresolved.setdefault(str(path.relative_to(REPO)), []).append(name)

    assert scanned, (
        "no docstring in this lane carries a :func:/:class: reference, so this test is "
        "checking an empty set. Either the references were removed or the scan broke."
    )
    assert not unresolved, (
        "these docstring cross-references do not resolve:\n"
        + json.dumps(unresolved, indent=2, sort_keys=True)
        + "\nRename the reference or restore what it names. A docstring pointing at a "
        "function that is not there reads as evidence that something exists."
    )


def _import_for(path: Path):
    """The module a file defines, imported, or ``None`` when it is not importable."""
    rel = path.relative_to(REPO / "python")
    dotted = ".".join(rel.with_suffix("").parts)
    try:
        return importlib.import_module(dotted)
    except ImportError:
        return None


def _resolves(name: str, module) -> bool:
    """Whether a Sphinx-style reference names something that exists."""
    if "." not in name:
        if module is not None and hasattr(module, name):
            return True
        try:
            return hasattr(importlib.import_module("qd_wire"), name)
        except ImportError:
            return False
    # Dotted: walk it as a module path first, then as attributes from the deepest
    # importable prefix. `qd_wire.answer.SlotAnswer` and `qd_train.tristate.NotRun`
    # both land here.
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


# ==========================================================================================
# GAP-XLANG-NO-PY-ANSWER-PARSER — the absence is over, and both guards now see it
# ==========================================================================================


def test_python_now_parses_all_four_answer_side_types():
    """``GAP-XLANG-NO-PY-ANSWER-PARSER`` closed: there is a second implementation.

    The gap's whole impact line was "half the wire contract is unverified across the
    lanes, and will stay unverified until a Python caller parses answers". This is
    that caller.
    """
    qd_wire = importlib.import_module("qd_wire")
    for name in ANSWER_SIDE_TYPES:
        assert hasattr(qd_wire, name), f"qd_wire lost {name}"
        assert inspect.isclass(getattr(qd_wire, name))


def test_the_rust_side_probe_searches_the_package_that_holds_the_parser():
    """``GAP-XLANG-RS-GUARD-BLIND-TO-NEW-PACKAGE``, opened and closed the same day.

    ``wire_context_crosslang.rs::the_answer_side_has_no_python_parser_and_that_is_
    recorded_not_invented`` asserts the four names are absent from Python, and is
    meant to fail the day a parser lands. For a few minutes it did not, because the
    probe feeding it (``crosslang_fixtures.py::inventory``) searched a hardcoded
    4-tuple of module names that did not include ``qd_wire`` — a check that ran,
    looked in the wrong place, and came back green, which is the same shape as the
    divergences it exists to catch. XLANG-RS has since extended the tuple.

    What is asserted now is the property, not the history: the probe's module list
    must contain the package that actually defines the four types. A guard whose
    search path does not cover the code it is guarding is worse than no guard, so
    this fails if the tuple is ever narrowed again — including by a rename of this
    package that the probe is not told about.
    """
    assert CROSSLANG_FIXTURES.is_file(), (
        f"{CROSSLANG_FIXTURES} is gone; the Rust-side probe this test checks no longer "
        "exists and must be re-read rather than assumed equivalent"
    )
    source = CROSSLANG_FIXTURES.read_text(encoding="utf-8")
    match = re.search(r"^\s*searched\s*=\s*\(([^)]*)\)", source, re.MULTILINE | re.DOTALL)
    assert match is not None, (
        f"could not find the `searched = (...)` tuple in {CROSSLANG_FIXTURES}. The probe "
        "was restructured; re-read it rather than assuming it still covers this package."
    )
    searched = tuple(m.group(1) for m in re.finditer(r'"([^"]+)"', match.group(1)))
    assert searched, "the probe's module list parsed as empty"

    # Where each answer-side type actually lives, resolved rather than assumed.
    defining = {
        name: inspect.getmodule(getattr(importlib.import_module("qd_wire"), name)).__name__
        for name in ANSWER_SIDE_TYPES
    }
    covered = set(searched)
    uncovered = {
        name: module
        for name, module in defining.items()
        if module not in covered and module.split(".")[0] not in covered
    }
    assert not uncovered, (
        f"the Rust-side probe searches {list(searched)}, which does not reach "
        f"{uncovered}. `the_answer_side_has_no_python_parser_...` would report those "
        "types absent while they exist, so the guard is green about something untrue. "
        "Add the module to `searched` in crosslang_fixtures.py."
    )


# ==========================================================================================
# GAP-XLANG-LABEL-SET-HASH-TWO-MEANINGS — one name, two quantities
# ==========================================================================================


def test_qd_data_no_longer_spells_a_request_digest_label_set_hash():
    """The rename landed: ``label_set_hash`` -> ``slot_set_digest``.

    ``expect.label_set_hash`` is a property of the **build**; the Python function is
    a property of the **request**. While they shared a name, a caller doing the
    obvious thing earned ``hash_mismatch`` on every request.
    """
    schema = importlib.import_module("qd_data.schema")
    assert not hasattr(schema, "label_set_hash"), (
        "qd_data.schema.label_set_hash is back. It hashes this request's slot list, "
        "which is not what the runtime compares expect.label_set_hash against."
    )
    assert hasattr(schema, "slot_set_digest")


def test_the_slot_set_digest_moves_with_the_request_and_the_build_hash_does_not():
    """The two quantities, measured rather than described.

    A build hash is fixed for a set of weights. This one changes when the slots do,
    which is the proof they are not the same number and can never be compared.
    """
    from qd_data.schema import ChoiceSlot, ScoreSlot, slot_set_digest

    one = slot_set_digest((ChoiceSlot(name="v", options=("a", "b")),))
    two = slot_set_digest((ChoiceSlot(name="v", options=("a", "b", "c")),))
    three = slot_set_digest((ScoreSlot(name="v", bins=5),))
    assert len({one, two, three}) == 3, (
        "three different slot lists produced fewer than three digests; this function "
        "is supposed to be a property of the request"
    )
    assert slot_set_digest((ChoiceSlot(name="v", options=("a", "b")),)) == one


def test_qd_wire_does_not_re_spell_the_build_hash_under_the_request_name():
    """This package parses answers; it deliberately declares no ``label_set_hash``.

    Adding one here would be a third place the name lives, which is how the gap got
    its title.
    """
    import qd_wire

    assert not hasattr(qd_wire, "label_set_hash")
    assert not hasattr(qd_wire, "HashExpectation")


# ==========================================================================================
# GAP-XLANG-NO-PY-HASH-EXPECTATION — the five pins have no Python producer
# ==========================================================================================


def test_a_training_lane_request_still_pins_no_hashes():
    """Absent ``expect`` correctly means "pinned nothing"; it also means nothing is pinned.

    Asserted rather than described, so the day the training lane grows the type, this
    test fails and the cross-language coverage must be extended in the same change.
    """
    from qd_data.schema import ChoiceSlot, Request

    wire = Request(
        task="t", context=b"x", question="q", slots=(ChoiceSlot(name="v", options=("a", "b")),)
    ).to_wire()
    assert "expect" not in wire, (
        "Request.to_wire() now emits `expect`. The five hash pins have a Python "
        "producer, so GAP-XLANG-NO-PY-HASH-EXPECTATION is closed and the five fields "
        "need asserting against crates/qd-runtime/src/schema.rs::HashExpectation."
    )


def test_the_documented_expect_block_deserializes_against_the_struct():
    """``GAP-XLANG-EXPECT-FIELD-NAME-DOC-DRIFT``, closed — and closed structurally.

    The document spelled the second pin ``weights_hash``; the struct declares
    ``weight_hash`` and is ``deny_unknown_fields``, so a caller who copied the
    documented example earned a refusal naming an unknown field and went looking for a
    typo in their own code. The other four names matched, which is why reading past it
    was easy.

    The previous pin asserted the drift was *still there*, so that fixing it would fail
    the test and tell whoever ran it to delete the pin. That is the right shape for a
    gap nobody owns yet, and the wrong shape once it is fixed: it goes stale on the day
    it succeeds, and it only ever guarded the one word.

    This replaces it with the check the gap record asked for — the field names are read
    out of the document's own ``expect`` example and compared against the **generated**
    contract table, which is itself re-derived from ``schema.rs`` on every run by
    ``test_wire_contract_matches_rust.py``. So the document, the Python table and the
    Rust struct are one set of names, and *any* future divergence in any of the five
    fails here naming the offending word rather than only the one that happened first.
    """
    import re

    from qd_wire.contract import STRUCT_FIELDS

    doc = (REPO / "docs" / "schema-api.md").read_text(encoding="utf-8")
    match = re.search(r'^"expect":\s*\{(.*?)^\}', doc, re.DOTALL | re.MULTILINE)
    assert match is not None, (
        "the `expect` example block is no longer in docs/schema-api.md in the shape this "
        "test reads. It is the thing a first caller copies; if it moved, re-point this "
        "test at it rather than deleting the check."
    )
    documented = set(re.findall(r'"([a-z_]+)"\s*:', match.group(1)))
    assert documented, "parsed no field names out of the documented `expect` block"

    declared = set(STRUCT_FIELDS["HashExpectation"])
    assert documented == declared, (
        f"docs/schema-api.md's `expect` example names {sorted(documented)} but "
        f"crates/qd-runtime/src/schema.rs::HashExpectation declares {sorted(declared)}. "
        "HashExpectation is #[serde(deny_unknown_fields)], so a caller copying the "
        "documented example is refused with `unknown field`, not a hash mismatch — "
        "GAP-XLANG-EXPECT-FIELD-NAME-DOC-DRIFT."
    )


# ==========================================================================================
# GAP-XLANG-SPAN-THREE-SPELLINGS — a gold span and an answer span are different shapes
# ==========================================================================================


def test_a_gold_answer_still_spells_a_span_as_a_two_element_array():
    from qd_data.rows import GoldAnswer

    gold = GoldAnswer(slot_name="evidence", value=(41, 47))
    assert gold.to_json()["value"] == [41, 47], (
        "GoldAnswer's span stopped being an array. If it is now "
        "{start_line, end_line}, the two spellings converged and "
        "GAP-XLANG-SPAN-THREE-SPELLINGS needs re-reading."
    )


def test_a_gold_span_cannot_be_fed_to_the_answer_parser_by_accident():
    """The two shapes are incompatible, and the failure is loud at the boundary.

    ``deny_unknown_fields`` on the Rust ``SpanValue`` makes an array-vs-object
    mismatch a hard deserialization failure there; this is the same wall on this
    side, so an eval that maps gold onto answers hits it at the mapping rather than
    at the point something reads ``start_line`` off an integer.
    """
    from qd_wire.answer import parse_slot_value
    from qd_wire.errors import WireParseError

    with pytest.raises(WireParseError):
        parse_slot_value([41, 47], path="$.value")


def test_no_converter_between_the_two_span_spellings_has_been_invented_here():
    """The gap's action is "whoever writes the eval converts explicitly, in one named
    function, and does not add a third spelling". This package is not that place, and
    asserting so keeps a well-meaning helper from appearing without the eval it is for.
    """
    import qd_wire

    for name in dir(qd_wire):
        assert "gold" not in name.lower(), (
            f"qd_wire.{name} looks like a gold-answer bridge. The conversion belongs in "
            "the eval that needs it, named once, not in the wire parser."
        )
