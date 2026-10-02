"""``shards._token_indices_for_chars`` against the per-position scan it replaced.

The scan, ``_token_index_for_char``, walked the offsets once per line start: O(lines x
tokens) interpreted steps, 18 of the 37 profiled seconds of ``tools/real_ft_run.py``'s needle
suite preparation on 2026-10-01 (300 cases of ~500 lines over ~8.5K tokens) and ~12 s of
every shard build. It is kept here, verbatim, as the reference oracle: the vectorised
projection must answer exactly what it answered -- the lowest token index whose span
contains the character -- on offsets of every shape a tokenizer produces (byte pieces
sharing one span, ``(0, 0)`` special tokens, unsorted spans) and must refuse the same
position with the same message when no span contains it.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qd_train import shards
from qd_train.artifacts import SLOT_SPAN
from qd_train.shards import SequenceSpec, UnencodableGold, _token_indices_for_chars


def reference(offsets: Sequence[tuple[int, int]], char_pos: int, *, where: str) -> int:
    """The scan ``_token_indices_for_chars`` replaced, byte for byte (shards.py at 73bcf12)."""
    for i, (start, end) in enumerate(offsets):
        if start <= char_pos < end:
            return i
    raise UnencodableGold(
        f"{where}: character {char_pos} lies in no token's offset span. The tokenizer's "
        "offsets and its ids describe different strings, or the offsets omit this region. "
        "Refused rather than mapped to a neighbouring token."
    )


def _reference_all(offsets, positions, *, where):
    return tuple(reference(offsets, c, where=where) for c in positions)


def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except UnencodableGold as exc:
        return ("refused", str(exc))


SPANS = st.lists(
    st.tuples(st.integers(0, 60), st.integers(0, 8)).map(lambda p: (p[0], p[0] + p[1])),
    max_size=40,
)


@settings(max_examples=600, deadline=None)
@given(offsets=SPANS, positions=st.lists(st.integers(-3, 75), max_size=30))
def test_every_offset_shape_answers_as_the_scan_did(offsets, positions):
    """Unsorted, overlapping, empty and duplicated spans; positions inside, between and
    outside them; the first refusal in position order, with the scan's own message."""
    assert _outcome(_token_indices_for_chars, offsets, positions, where="w") == _outcome(
        _reference_all, offsets, positions, where="w"
    )


def test_byte_pieces_sharing_one_span_resolve_to_the_first():
    """Qwen3.5 gives every byte of a multi-byte character the character's whole span."""
    offsets = [(0, 1), (1, 2), (2, 3), (2, 3), (2, 3), (3, 5), (19, 21), (20, 21), (20, 21)]
    for positions in ([0, 1, 2, 3, 4], [2], [20], [19, 20]):
        assert _token_indices_for_chars(offsets, positions, where="w") == _reference_all(
            offsets, positions, where="w"
        )
    assert _token_indices_for_chars(offsets, [2, 20], where="w") == (2, 6)


def test_special_tokens_with_empty_spans_are_never_chosen():
    offsets = [(0, 0), (0, 3), (0, 0), (3, 6)]
    assert _token_indices_for_chars(offsets, [0, 2, 3, 5], where="w") == (1, 1, 3, 3)


@pytest.mark.parametrize(
    ("offsets", "positions", "missing"),
    [
        ([(0, 2), (4, 6)], [0, 3, 5], 3),
        ([(0, 2), (4, 6)], [7, 3], 7),
        ([], [0], 0),
        ([(0, 0)], [0, 1], 0),
    ],
)
def test_a_position_no_span_contains_is_refused_by_name(offsets, positions, missing):
    with pytest.raises(UnencodableGold, match=f"w: character {missing} lies in no token"):
        _token_indices_for_chars(offsets, positions, where="w")


def test_no_positions_ask_nothing():
    assert _token_indices_for_chars([], [], where="w") == ()
    assert _token_indices_for_chars([(0, 1)], [], where="w") == ()


def test_the_block_bound_splits_without_changing_an_answer(monkeypatch):
    offsets = [(i, i + 1) for i in range(500)] + [(0, 500)]
    positions = list(range(499, -1, -7))
    want = _reference_all(offsets, positions, where="w")
    monkeypatch.setattr(shards, "TOKEN_INDEX_BLOCK_CELLS", 501 * 3)
    assert _token_indices_for_chars(offsets, positions, where="w") == want
    monkeypatch.setattr(shards, "TOKEN_INDEX_BLOCK_CELLS", 1)
    assert _token_indices_for_chars(offsets, positions, where="w") == want


def test_a_span_slot_projects_its_lines_in_one_call_not_one_per_line(monkeypatch):
    """The projection is whole-slot: the candidate set in one call and the gold pair in one,
    however many lines the context has. The per-line scan made one call per line start."""
    calls: list[int] = []
    real = shards._token_indices_for_chars

    def counting(offsets, positions, *, where):
        calls.append(len(tuple(positions)))
        return real(offsets, positions, where=where)

    monkeypatch.setattr(shards, "_token_indices_for_chars", counting)
    lines = [f"line {i}" for i in range(200)]
    text = "\n".join(lines) + "\n"
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    offsets = [(i, i + 1) for i in range(len(text))]
    spec = SequenceSpec(
        slot_name="defect_span", text=text, slot_kind=SLOT_SPAN,
        span_char_starts=(starts[3], starts[5]), line_char_starts=tuple(starts),
    )
    span, candidates = shards._span_token_positions(
        spec, list(range(len(text))), token_offsets=lambda _t: offsets, where="w"
    )
    assert calls == [200, 2]
    assert candidates == tuple(starts) and span == (starts[3], starts[5])
