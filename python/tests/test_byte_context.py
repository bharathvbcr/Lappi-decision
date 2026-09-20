"""Byte-level context encoding. Torch-free, so this runs in the repo venv.

The line-start tests are exhaustive rather than sampled, because exactness is the whole
reason rung 0 is byte-level: in byte space a line start *is* an offset, so the answer is
either right for every input or the design does not hold.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from qd_train.byte_context import (
    BYTE_VOCAB_SIZE,
    ID_NOUL,
    ID_OPTION,
    ID_PAD,
    ID_QUESTION,
    N_BYTE_VALUES,
    ContextTooLarge,
    EncodedContext,
    SpanOutsideWindow,
    encode_context,
    line_of_offset,
    line_starts,
)

# ---------------------------------------------------------------------------
# line_starts: the exact answer to the question BPE could not answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b"", ()),
        (b"a", (0,)),
        (b"abc", (0,)),
        (b"a\n", (0,)),  # no line after a trailing newline: there is no line there
        (b"a\nb", (0, 2)),
        (b"a\n\nb", (0, 2, 3)),  # the empty line at 2 is a line
        (b"\n", (0,)),  # one empty line
        (b"\n\n", (0, 1)),
        (b"\n\nx", (0, 1, 2)),
        (b"a\r\nb", (0, 3)),  # CRLF is one break, and the line starts after the \n
        (b"a\rb", (0,)),  # a lone \r is not a terminator -- documented, not an oversight
        (b"a\nb\nc\n", (0, 2, 4)),
    ],
)
def test_line_starts_is_exact(raw: bytes, expected: tuple[int, ...]):
    assert line_starts(raw) == expected


@given(st.binary(max_size=400))
def test_every_line_start_follows_a_newline_or_is_zero(raw: bytes):
    """The defining property, checked over arbitrary bytes including all the awkward ones."""
    starts = line_starts(raw)
    if not raw:
        assert starts == ()
        return
    assert starts[0] == 0
    assert list(starts) == sorted(set(starts)), "starts must be strictly ascending"
    for s in starts:
        assert 0 <= s < len(raw)
        if s != 0:
            assert raw[s - 1] == 0x0A, f"line start {s} does not follow a newline"
    # Completeness: every newline that is not the final byte opens a line.
    expected = 1 + sum(1 for i, b in enumerate(raw) if b == 0x0A and i != len(raw) - 1)
    assert len(starts) == expected


@given(st.binary(min_size=1, max_size=400))
def test_every_offset_belongs_to_exactly_one_line(raw: bytes):
    """Total coverage of the byte range. Also why `check_span` cannot detect a wrong offset."""
    starts = line_starts(raw)
    for offset in range(len(raw)):
        i = line_of_offset(starts, offset)
        assert starts[i] <= offset
        if i + 1 < len(starts):
            assert offset < starts[i + 1]


def test_line_of_offset_refuses_rather_than_guessing():
    with pytest.raises(ValueError, match="no lines"):
        line_of_offset((), 0)
    with pytest.raises(ValueError, match="non-negative"):
        line_of_offset((0,), -1)


# ---------------------------------------------------------------------------
# The id space
# ---------------------------------------------------------------------------


def test_control_ids_cannot_collide_with_a_byte_value():
    """A control id inside the byte range would let context data act as a separator.

    A context containing that byte would silently restructure the request -- the model would
    read a question boundary where the source file merely had a control character.
    """
    for control in (ID_PAD, ID_QUESTION, ID_OPTION, ID_NOUL):
        assert control >= N_BYTE_VALUES
    assert len({ID_PAD, ID_QUESTION, ID_OPTION, ID_NOUL}) == 4
    assert max(ID_PAD, ID_QUESTION, ID_OPTION, ID_NOUL) < BYTE_VOCAB_SIZE


# ---------------------------------------------------------------------------
# encode_context
# ---------------------------------------------------------------------------


def test_a_context_inside_the_window_is_untouched_and_says_so():
    raw = b"def f():\n    return 1\n"
    enc = encode_context(raw, max_bytes=1024)
    assert enc.ids == tuple(raw)
    assert enc.n_bytes_kept == enc.n_bytes_total == len(raw)
    assert not enc.truncated
    assert enc.coverage() == f"{len(raw)}/{len(raw)}"
    assert enc.n_lines == 2


def test_ids_are_the_bytes_so_a_line_start_needs_no_second_mapping():
    """The identity the whole design rests on: `ids[i] == raw[i]` for every kept byte."""
    raw = b"alpha\nbeta\ngamma"
    enc = encode_context(raw, max_bytes=1024)
    for i, byte in enumerate(raw):
        assert enc.ids[i] == byte
    for start in enc.starts:
        assert enc.ids[start] == raw[start]


def test_truncation_keeps_the_tail_and_reports_both_counts():
    raw = b"\n".join(f"line {i}".encode() for i in range(200))
    enc = encode_context(raw, max_bytes=64)
    assert enc.truncated
    assert enc.n_bytes_total == len(raw)
    assert enc.n_bytes_kept <= 64
    assert enc.coverage() == f"{enc.n_bytes_kept}/{len(raw)}"
    assert bytes(enc.ids) == raw[len(raw) - enc.n_bytes_kept :], "the tail, not the head"
    assert b"line 199" in bytes(enc.ids)


def test_truncation_starts_on_a_line_boundary_not_mid_line():
    """The kept region begins where a line begins, so the leading line is not a fragment.

    Stated as an offset property rather than "it starts with line c": the raw window here
    opens mid-way through the 'a' line, and advancing to the first newline correctly lands on
    the *whole* 'b' line. Asserting a particular leading line would have been asserting the
    arithmetic, not the property.
    """
    raw = b"aaaaaaaaaa\nbbbbbbbbbb\ncccccccccc\n"
    enc = encode_context(raw, max_bytes=25)
    kept = bytes(enc.ids)
    assert enc.truncated
    assert kept == b"bbbbbbbbbb\ncccccccccc\n"

    offset = len(raw) - enc.n_bytes_kept
    assert offset == 0 or raw[offset - 1] == 0x0A, (
        f"kept region starts at {offset}, which is mid-line: {kept[:12]!r}"
    )


@given(st.binary(min_size=1, max_size=600), st.integers(min_value=1, max_value=120))
def test_a_truncated_context_either_aligns_to_a_line_or_had_no_newline_to_align_to(
    raw: bytes, max_bytes: int
):
    """The alignment property over arbitrary bytes, with its documented exception."""
    enc = encode_context(raw, max_bytes=max_bytes)
    offset = len(raw) - enc.n_bytes_kept
    if offset == 0:
        return
    window = raw[len(raw) - max_bytes :]
    if window.find(b"\n") in range(len(window) - 1):
        assert raw[offset - 1] == 0x0A, f"offset {offset} is mid-line"
    else:
        assert enc.n_bytes_kept == len(window), "no newline to align to: keep the whole window"


def test_a_window_with_no_newline_keeps_the_whole_window():
    raw = b"x" * 500
    enc = encode_context(raw, max_bytes=100)
    assert enc.n_bytes_kept == 100
    assert enc.n_lines == 1


def test_refusing_truncation_is_available_and_loud():
    raw = b"y" * 500
    with pytest.raises(ContextTooLarge, match="truncation was refused"):
        encode_context(raw, max_bytes=100, allow_truncation=False)


def test_an_empty_context_has_no_lines_rather_than_a_phantom_one():
    enc = encode_context(b"", max_bytes=16)
    assert enc.ids == ()
    assert enc.starts == ()
    assert enc.n_lines == 0
    assert not enc.truncated


def test_max_bytes_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        encode_context(b"abc", max_bytes=0)


# ---------------------------------------------------------------------------
# check_span
# ---------------------------------------------------------------------------


def test_a_gold_offset_resolves_to_its_line():
    raw = b"zero\none\ntwo\nthree"
    enc = encode_context(raw, max_bytes=1024)
    assert enc.starts == (0, 5, 9, 13)
    assert enc.check_span(0) == 0
    assert enc.check_span(4) == 0  # still inside line 0, on its newline
    assert enc.check_span(5) == 1
    assert enc.check_span(13) == 3
    assert enc.check_span(len(raw) - 1) == 3


def test_a_gold_offset_outside_the_window_is_refused_not_clamped():
    """Clamping would point the span head at whatever truncation happened to leave behind."""
    raw = b"\n".join(f"line {i}".encode() for i in range(100))
    enc = encode_context(raw, max_bytes=40)
    with pytest.raises(SpanOutsideWindow, match="clamping"):
        enc.check_span(enc.n_bytes_kept)
    with pytest.raises(SpanOutsideWindow):
        enc.check_span(-1)


def test_check_span_does_not_pretend_to_validate_correctness():
    """Documented limit, asserted so nobody later reads it as a correctness check.

    Every in-window offset belongs to some line, so a wrong-but-in-range gold resolves
    quietly. What byte space removes is the multi-hop mapping that could produce one, not the
    need for an independent check.
    """
    raw = b"aaa\nbbb\nccc"
    enc = encode_context(raw, max_bytes=1024)
    intended, wrong = 4, 8  # line 1 and line 2, both perfectly valid offsets
    assert enc.check_span(intended) == 1
    assert enc.check_span(wrong) == 2  # no complaint, and there cannot be one


# ---------------------------------------------------------------------------
# EncodedContext refuses to hold an inconsistent state
# ---------------------------------------------------------------------------


def test_a_control_id_in_the_context_payload_is_refused():
    with pytest.raises(ValueError, match="control ids do not belong here"):
        EncodedContext(ids=(ID_QUESTION,), starts=(0,), n_bytes_kept=1, n_bytes_total=1)


def test_keeping_more_than_the_total_is_refused():
    with pytest.raises(ValueError, match="not possible"):
        EncodedContext(ids=(97, 98), starts=(0,), n_bytes_kept=2, n_bytes_total=1)


def test_an_ids_length_that_disagrees_with_the_count_is_refused():
    with pytest.raises(ValueError, match="disagrees with n_bytes_kept"):
        EncodedContext(ids=(97, 98), starts=(0,), n_bytes_kept=5, n_bytes_total=9)


def test_a_line_start_outside_the_window_is_refused():
    with pytest.raises(ValueError, match="outside the kept window"):
        EncodedContext(ids=(97, 98), starts=(0, 7), n_bytes_kept=2, n_bytes_total=2)
