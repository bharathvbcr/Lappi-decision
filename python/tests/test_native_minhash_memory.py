"""The native MinHash block holds one ``bytes`` object per distinct shingle, not one per use.

Measured on a subsample of the LoRA-2B v6 rebuild (W2-d, 2026-10-18, stdlib tracemalloc around
``real_ft_run.ft_split_report``): at the end of the block's table build 5.96 of 7.5 live GiB were
the shingle frozensets (48.3M ``bytes`` objects for 82k distinct sets, ``qd_data/minhash.py``
``shingle``) and 0.4 GiB the signature tuples. Word 5-gram shingles recur across near-duplicate
texts, so ``native_minhash`` shares the object of equal shingles. Same values (sets compare and
hash by content), so every output is unchanged; the other tests of this block pin that.

This pins the memory: a fixed fixture of 60 texts that share 98% of their shingles must hold under
``BOUND_MIB`` of live Python memory once the table is built. It fails on the pre-change code
(23.2 MiB on this fixture; 8.9 MiB after), and the bound leaves room for interpreter drift.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import pytest
from test_composed_file_near_duplicates import CONFIG, _repo_in, _row

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

from qd_data.dedupe import dedupe  # noqa: E402

BOUND_MIB = 16
N_TEXTS = 60
WORDS = 4000


def _texts() -> list[str]:
    base = [f"w{i:05d}" for i in range(WORDS)]
    out = []
    for t in range(N_TEXTS):
        words = list(base)
        for j in range(40):
            words[(t * 37 + j * 91) % WORDS] = f"x{t:03d}y{j:02d}"
        out.append(" ".join(words))
    return out


def test_table_build_shares_equal_shingles(
    qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_in("train", "org/t")
    rows = [_row(f"r{i}", repo=repo, text=t) for i, t in enumerate(_texts())]
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(qd_prep_bin))
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        with pipeline.native_minhash(rows, config=CONFIG):
            held = tracemalloc.get_traced_memory()[0] - before
            dedupe(list(rows), config=CONFIG)
    finally:
        tracemalloc.stop()
    assert held < BOUND_MIB << 20, f"{held / (1 << 20):.1f} MiB held after the table build"
