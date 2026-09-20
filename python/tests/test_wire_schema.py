"""The wire encoding of ``context``: exactly one form, and it is checked.

``docs/schema-api.md``, *"The wire encoding of ``context`` -- exactly one form"*:

    ``context_b64`` (base64, ASCII) plus a required ``context_len`` (the decoded
    byte count). They must agree, or it is a typed refusal. There is no second
    accepted form.

This file is the Python half. It is a separate module from ``test_render.py``
because it defends a different claim: not *"the two renderers agree"* but *"the
encoder and the decoder on the other side of the wire agree"*. The other half, which
actually runs both implementations against each other, is
``crates/qd-runtime/tests/wire_context_crosslang.rs`` -- a test in this file alone
could not have caught ``GAP-RT-WIRE-CONTEXT-ENCODING``, because this lane was
internally consistent the whole time it was incompatible with the runtime.
"""

from __future__ import annotations

import base64

import pytest
from qd_data.errors import (
    ContextLengthMismatchRefusal,
    ContextLenMissingRefusal,
    ContextNotBase64Refusal,
    ContextNotBytesRefusal,
)
from qd_data.schema import ChoiceSlot, Request, SpanSlot, decode_context, encode_context

#: Contexts chosen for what they break, not for coverage of "some bytes":
#:
#: * a lone 0xFF is not valid UTF-8, so anything that reached for ``.decode()`` fails;
#: * the empty context has an empty blob and a zero length, which is the case a
#:   presence check confuses with "absent";
#: * NUL truncates any C string that later carries the bytes;
#: * the exotic separators are the ones ``splitlines()`` honours and ``split("\n")``
#:   does not, and they are multi-byte in UTF-8, so a length in characters and a
#:   length in bytes disagree on them;
#: * lengths 1, 2 and 3 straddle the base64 quantum, where padding appears.
CONTEXTS: tuple[tuple[str, bytes], ...] = (
    ("empty", b""),
    ("one byte", b"a"),
    ("two bytes", b"ab"),
    ("three bytes", b"abc"),
    ("ascii source", b"fn add(a: i32) -> i32 {\n    todo!()\n}\n"),
    ("non-utf8", b"a\xffb"),
    ("lone continuation byte", b"\x80\x80\x80"),
    ("nul", b"before\x00after"),
    ("exotic separators", "a\u000bb\u001cc\u0085d\u2028e\u2029f".encode()),
    ("crlf", b"one\r\ntwo\r\n"),
    ("every byte value", bytes(range(256))),
)

CONTEXT_IDS = [name for name, _ in CONTEXTS]


def _request(context: bytes) -> Request:
    return Request(
        task="devcouncil.verdict",
        context=context,
        question="Does this diff implement what the commit message claims?",
        slots=(
            ChoiceSlot(name="verdict", options=("stub", "logic", "cosmetic", "clean")),
            SpanSlot(name="evidence"),
        ),
    )


def _wire(context: bytes) -> dict[str, object]:
    return _request(context).to_wire()


# -- 1. what the encoder emits ------------------------------------------------


@pytest.mark.parametrize(("name", "context"), CONTEXTS, ids=CONTEXT_IDS)
def test_the_wire_form_is_context_b64_and_context_len(name: str, context: bytes) -> None:
    wire = _wire(context)
    assert "context" not in wire, "the retired field must not be emitted alongside"
    assert wire["context_b64"] == base64.b64encode(context).decode("ascii")
    assert wire["context_len"] == len(context)
    # ASCII, so it cannot carry the Unicode normalisation the rule is about.
    assert isinstance(wire["context_b64"], str)
    assert wire["context_b64"].isascii()


@pytest.mark.parametrize(("name", "context"), CONTEXTS, ids=CONTEXT_IDS)
def test_the_wire_form_round_trips_the_exact_bytes(name: str, context: bytes) -> None:
    parsed = Request.from_wire(_wire(context))
    assert parsed.context == context
    assert isinstance(parsed.context, bytes)


def test_context_len_is_bytes_and_not_characters() -> None:
    """The distinction the field exists to make. U+2028 is one character and three
    bytes; a ``context_len`` counted in characters would disagree with the decoder
    on every context containing one."""
    context = "\u2028".encode()
    assert len(context) == 3
    wire = _wire(context)
    assert wire["context_len"] == 3


# -- 2. there is no second accepted form --------------------------------------


def test_the_retired_context_field_is_refused_rather_than_read() -> None:
    """``GAP-RT-WIRE-CONTEXT-ENCODING``. This module used to accept a ``context``
    str as an alternative to ``context_b64``, while ``qd-runtime`` read ``context``
    as an array of byte values. Both readings are now refused, because a lenient
    read is how two encodings coexist and "which one wins" becomes undefined."""
    for retired in ("fn add() {}\n", [102, 110], None, 17):
        wire = _wire(b"fn add() {}\n")
        del wire["context_b64"]
        wire["context"] = retired
        with pytest.raises(ContextNotBytesRefusal) as caught:
            Request.from_wire(wire)
        assert "context_b64" in str(caught.value)

    # And even alongside a valid `context_b64`, which is the ambiguous case.
    wire = _wire(b"fn add() {}\n")
    wire["context"] = "fn add() {}\n"
    with pytest.raises(ContextNotBytesRefusal):
        Request.from_wire(wire)


@pytest.mark.parametrize(
    "blob", [None, 17, 1.5, [], {}, b"Zm9v", True], ids=lambda v: type(v).__name__
)
def test_a_context_b64_that_is_not_a_string_is_refused(blob: object) -> None:
    wire = _wire(b"foo")
    wire["context_b64"] = blob
    with pytest.raises(ContextNotBytesRefusal):
        Request.from_wire(wire)


def test_an_absent_context_b64_is_refused() -> None:
    wire = _wire(b"foo")
    del wire["context_b64"]
    with pytest.raises(ContextNotBytesRefusal) as caught:
        Request.from_wire(wire)
    assert caught.value.actual == "absent"


# -- 3. base64 is canonical, not merely decodable ------------------------------


@pytest.mark.parametrize(
    ("label", "blob"),
    [
        ("outside the alphabet", "Zm9v*g=="),
        ("whitespace inside", "Zm9 v"),
        ("a newline inside", "Zm9\nv"),
        ("url-safe alphabet", "Zm9-"),
        ("missing padding", "Zm9vYg"),
        ("over-padded", "Zg==="),
        ("padding in the middle", "Zg==Zg=="),
    ],
)
def test_base64_that_does_not_decode_is_refused(label: str, blob: str) -> None:
    wire = _wire(b"foo")
    wire["context_b64"] = blob
    wire["context_len"] = 3
    with pytest.raises(ContextNotBase64Refusal):
        Request.from_wire(wire)


def test_base64_that_decodes_but_is_not_canonical_is_refused() -> None:
    """``base64.b64decode(validate=True)`` checks the alphabet, not the bits a
    partial quantum leaves unused: it returns ``b"f"`` for both ``Zg==`` and
    ``Zh==``. ``qd-runtime``'s decoder rejects the second, so accepting it here
    would mean a payload this lane reads and that lane refuses -- the same class of
    divergence as ``GAP-RT-WIRE-CONTEXT-ENCODING``, one layer down."""
    assert base64.b64decode("Zh==", validate=True) == b"f"
    assert decode_context("Zg==", 1) == b"f"
    wire = _wire(b"f")
    wire["context_b64"] = "Zh=="
    with pytest.raises(ContextNotBase64Refusal) as caught:
        Request.from_wire(wire)
    assert "non-canonical" in caught.value.detail


@pytest.mark.parametrize(("name", "context"), CONTEXTS, ids=CONTEXT_IDS)
def test_the_encoder_only_ever_emits_what_the_decoder_calls_canonical(
    name: str, context: bytes
) -> None:
    blob = encode_context(context)
    assert decode_context(blob, len(context)) == context


# -- 4. the length is checked, and that is why it is carried -------------------


def test_an_absent_context_len_is_refused_and_does_not_default_to_zero() -> None:
    wire = _wire(b"foo")
    del wire["context_len"]
    with pytest.raises(ContextLenMissingRefusal):
        Request.from_wire(wire)


@pytest.mark.parametrize("declared", ["3", 3.0, None, True, [3]], ids=lambda v: repr(v))
def test_a_context_len_that_is_not_an_int_is_refused(declared: object) -> None:
    wire = _wire(b"foo")
    wire["context_len"] = declared
    with pytest.raises((ContextLenMissingRefusal, ContextLengthMismatchRefusal)):
        Request.from_wire(wire)


def test_a_disagreeing_context_len_is_refused_and_names_both_numbers() -> None:
    wire = _wire(b"foo")
    wire["context_len"] = 999
    with pytest.raises(ContextLengthMismatchRefusal) as caught:
        Request.from_wire(wire)
    assert caught.value.expected == 999
    assert caught.value.actual == 3
    assert "999" in str(caught.value) and "3" in str(caught.value)


def test_a_truncated_payload_that_still_decodes_is_caught_by_the_length() -> None:
    """Why ``context_len`` is kept although base64 makes it derivable.

    Chopping whole quanta off the blob leaves valid base64 that decodes to a shorter
    context. Nothing but the declared length notices -- the request is otherwise
    well-formed, and the model would be asked about a document missing its tail.
    """
    context = b"fn add(a: i32) -> i32 {\n    todo!()\n}\n"
    wire = _wire(context)
    full = wire["context_b64"]
    assert isinstance(full, str)
    truncated = full[: len(full) - 4]
    # Still valid base64 on its own, and it decodes to fewer bytes.
    short = base64.b64decode(truncated, validate=True)
    assert 0 < len(short) < len(context)
    assert context.startswith(short), "a prefix of the real context, which is the danger"
    wire["context_b64"] = truncated
    with pytest.raises(ContextLengthMismatchRefusal) as caught:
        Request.from_wire(wire)
    assert caught.value.expected == len(context)
    assert caught.value.actual == len(short)


# -- 5. the two lanes name the same checks -------------------------------------


def test_the_refusal_check_ids_match_the_runtime_refusal_kinds() -> None:
    """A caller that branches on the identifier gets the same string from both
    lanes. The Rust side of this pairing is asserted in
    ``crates/qd-runtime/tests/wire_context_crosslang.rs``; here it is pinned so a
    rename on this side fails a test rather than silently halving the contract."""
    assert ContextNotBytesRefusal.check == "context_not_bytes"
    assert ContextNotBase64Refusal.check == "context_not_base64"
    assert ContextLenMissingRefusal.check == "context_len_missing"
    assert ContextLengthMismatchRefusal.check == "context_length_mismatch"
