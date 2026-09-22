"""How many rows the train remap can encode when it was not built from them.

The remap keeps every token the train rows use and nothing else; ``RemapTable.encode``
raises on any other id and never substitutes an ``UNK``. So a val or held-out row with one
unseen token is not scored badly under that remap -- it cannot be written or scored at all.
``tools/real_tokenizer_pipeline.remap_coverage`` counts those rows, and what these pin is the
counting: per ROW (a row is several sequences, one per slot, and one unencodable slot sinks
the row), over the rows that tokenized, with the rows that never did stated beside them.

Torch-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_train.artifacts import RemapTable  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

SOURCE_VOCAB = 32


def _remap(kept: list[int]) -> RemapTable:
    old_to_new = np.full(SOURCE_VOCAB, -1, dtype=np.int32)
    for new, old in enumerate(kept):
        old_to_new[old] = new
    return RemapTable(
        old_to_new=old_to_new,
        new_to_old=np.asarray(kept, dtype=np.int32),
        tokenizer_hash="tokhash-remap-coverage",
        special_ids=(),
    )


def _census(rows_in: int, sequences: list[tuple[str, list[int]]]) -> pipeline.Census:
    cen = pipeline.Census()
    cen.rows_in = rows_in
    for row_id, ids in sequences:
        cen.id_rows.append(row_id)
        cen.ids.append(np.asarray(ids, dtype=np.int64))
    return cen


def test_rows_using_only_kept_ids_are_all_encodable() -> None:
    cen = _census(2, [("a", [1, 2]), ("a", [3]), ("b", [2, 2])])
    got = pipeline.remap_coverage(cen, _remap([1, 2, 3]), split_name="val")
    assert isinstance(got, Ran)
    assert (got.passed, got.value, got.n, got.n_total) == (True, 2, 2, 2)


def test_one_unseen_token_in_one_slot_sinks_the_whole_row() -> None:
    """Row "a" has two sequences and only its second holds the dropped id 9. It is one row
    that cannot be encoded, not one of three sequences."""
    cen = _census(2, [("a", [1, 2]), ("a", [9]), ("b", [2])])
    got = pipeline.remap_coverage(cen, _remap([1, 2]), split_name="heldout")
    assert isinstance(got, Ran)
    assert (got.passed, got.value, got.n_total) == (False, 1, 2)
    assert "1 hold at least one dropped id" in got.detail


def test_the_dropped_tokens_are_counted_both_ways() -> None:
    cen = _census(1, [("a", [7, 7, 8, 1])])
    got = pipeline.remap_coverage(cen, _remap([1]), split_name="val")
    assert isinstance(got, Ran)
    assert "3 of 4 tokens, 2 distinct ids" in got.detail


def test_rows_that_never_tokenized_are_stated_not_counted() -> None:
    """Five rows came in and two tokenized: the coverage is of two, and the three the writer
    would have refused before the remap is consulted are named, not charged to it."""
    cen = _census(5, [("a", [1]), ("b", [1])])
    got = pipeline.remap_coverage(cen, _remap([1]), split_name="val")
    assert isinstance(got, Ran)
    assert (got.value, got.n_total) == (2, 2)
    assert "5 row(s) in, 3 never tokenized" in got.detail


def test_a_split_where_nothing_tokenized_is_not_run() -> None:
    got = pipeline.remap_coverage(_census(4, []), _remap([1]), split_name="heldout")
    assert isinstance(got, NotRun)
    assert "4 in" in got.reason


# -- what a byte fallback would cost on the same rows -----------------------------------

#: A toy vocabulary: ids 0-3 are the "byte" tokens; 7 spells in 3 bytes, 8 in 5.
_BYTES = frozenset({0, 1, 2, 3})
_LENGTHS = {7: 3, 8: 5}


def _byte_lengths(ids) -> dict[int, int]:
    return {int(i): _LENGTHS[int(i)] for i in ids}


def test_each_dropped_token_costs_its_bytes_minus_the_one_it_replaces() -> None:
    """Row a: 7 twice (2 extra each) and 8 once (4 extra): 3 of 6 tokens re-spelled, 8 added."""
    cen = _census(1, [("a", [7, 7, 8, 1, 2, 3])])
    got = pipeline.byte_fallback_cost(
        cen, _remap([0, 1, 2, 3]), byte_ids=_BYTES, byte_lengths=_byte_lengths,
        split_name="val",
    )
    assert isinstance(got, Ran)
    assert (got.value, got.n, got.n_total) == (8, 3, 6)
    assert "0 of the 256 byte tokens are not in this remap" in got.detail


def test_byte_tokens_the_remap_dropped_are_counted_as_rows_to_add() -> None:
    cen = _census(1, [("a", [1, 7])])
    got = pipeline.byte_fallback_cost(
        cen, _remap([1]), byte_ids=_BYTES, byte_lengths=_byte_lengths, split_name="heldout",
    )
    assert isinstance(got, Ran)
    assert "3 of the 256 byte tokens are not in this remap" in got.detail


def test_a_split_the_remap_fully_covers_costs_nothing() -> None:
    cen = _census(1, [("a", [1, 2])])
    got = pipeline.byte_fallback_cost(
        cen, _remap([0, 1, 2, 3]), byte_ids=_BYTES, byte_lengths=_byte_lengths,
        split_name="val",
    )
    assert isinstance(got, Ran)
    assert (got.value, got.n, got.n_total) == (0, 0, 2)


def test_a_fallback_over_nothing_is_not_run() -> None:
    got = pipeline.byte_fallback_cost(
        _census(3, []), _remap([1]), byte_ids=_BYTES, byte_lengths=_byte_lengths,
        split_name="val",
    )
    assert isinstance(got, NotRun)


def test_the_real_tokenizer_spells_every_text_in_exactly_its_bytes() -> None:
    """The two lookups the real run prices the fallback with, against the live tokenizer:
    256 byte tokens of one byte each, and a tokenization whose per-token byte counts add up
    to the text's UTF-8 length -- non-ASCII included, where one character is several."""
    pytest.importorskip("transformers")
    if not pipeline.MODEL_REF.exists():
        pytest.skip(f"{pipeline.MODEL} is not in this host's HF cache")
    tok = pipeline.RealTokenizer.load()
    byte_ids = tok.byte_token_ids()
    assert len(byte_ids) == 256
    assert set(tok.byte_lengths(byte_ids).values()) == {1}
    text = "def remap_coverage(cen):\n    return naïve_café('→', 0x1F600)  # ü"
    ids = tok.tokenize(text)
    lengths = tok.byte_lengths(ids)
    assert sum(lengths[i] for i in ids) == len(text.encode("utf-8"))
