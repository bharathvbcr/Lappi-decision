"""The native MinHash block covers split's composed file-level near-duplicate check.

``split`` signs each file block of a composed row (``_composed_file_near_duplicates``, M9). The
native block (``real_tokenizer_pipeline.native_minhash``) answers shingling and signing from a
table built before dedupe and split run, and refuses any text outside it. It tabled only each
row's whole ``dedupe_text``, so a v6 build over composed rows died in ``split`` with "a N-character
text was not among the ... the block shingled" (W2-d, 2026-10-18, the LoRA-2B rebuild's
containment scan). This pins that the block now tables the file blocks too, and that the
reports it produces equal the reference's.

Fails on the pre-fix code: ``native_minhash`` raises ``SystemExit`` inside ``split``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_composed_file_near_duplicates import CONFIG, _corpus

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.split import split  # noqa: E402


@pytest.mark.parametrize("fork", [False, True])
def test_native_block_matches_the_reference_over_composed_rows(
    qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch, fork: bool,
) -> None:
    rows = _corpus(fork=fork)
    monkeypatch.delenv(pipeline.PREP_BIN_ENV, raising=False)
    want = split(dedupe(list(rows), config=CONFIG), config=CONFIG)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(qd_prep_bin))
    with pipeline.native_minhash(rows, config=CONFIG):
        got = split(dedupe(list(rows), config=CONFIG), config=CONFIG)
    assert got.composed_file_near_duplicate_disjoint == want.composed_file_near_duplicate_disjoint
    assert got.assignments == want.assignments
    assert got.rows_by_split == want.rows_by_split
