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
