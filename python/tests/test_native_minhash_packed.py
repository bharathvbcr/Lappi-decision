"""The native MinHash block holds hashes and a signature matrix, never every set's shingles.

GAP-W2D-V6-SCAN-FOOTPRINT-OVER-MAC-HEAVY-CAP-2026-10-18: the v6 containment scan's rebuild was
killed over mac_heavy's footprint cap (28,672 MiB, memguard's line minus 4 GiB) in four of five
runs, because ``native_minhash`` kept every distinct set's shingles as a frozenset of ``bytes``
(about 160 bytes a shingle) and a 7.3 KB tuple of Python ints per signature for the whole block.
It now packs each set to the sorted 64-bit hashes of its shingles (8 bytes each), signs at most
``batch_sets`` new distinct sets per ``qd-prep minhash`` call, and hands ``dedupe`` and ``split``
rows of one signature matrix. These tests pin that the result is the reference's whatever the batch
size, that the bytes held are bounded by the batch, and the refusals.

``test_the_block_holds_hashes_not_shingles`` fails on the pre-change code (27.8 MiB held on its
fixture against a 6 MiB bound; 2.4 MiB after).
"""

from __future__ import annotations

import math
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_composed_file_near_duplicates import CONFIG, _repo_in, _row

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

import qd_data.dedupe as dedupe_module  # noqa: E402
import qd_data.split as split_module  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.minhash import MinHasher, exact_jaccard, shingle  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.split import split  # noqa: E402

BOUND_MIB = 6


def _words(tag: str, n: int) -> list[str]:
    return [f"{tag}{i:04d}" for i in range(n)]


def _fixture() -> list[DataRow]:
    """Rows that exercise every way two texts can share a signature: an exact duplicate text,
    texts that differ only in whitespace (equal shingle sets, different strings), a near
    duplicate across repos on opposite sides of the split, NUL and non-ASCII payloads, and a text
    of fewer tokens than the shingle size (one shingle: the whole token tuple)."""
    train, val = _repo_in("train", "org/t"), _repo_in("val", "org/v")
    base = " ".join(_words("alpha", 300))
    near = " ".join(_words("alpha", 280) + _words("beta", 20))
    texts: list[tuple[str, str]] = [
        (train, base),
        (train, base),
        (train, base.replace(" ", "  ", 7)),
        (val, near),
        (train, " ".join(_words("gamma", 250))),
        (val, " ".join(_words("delta", 250))),
        (train, "café 中文 " + " ".join(_words("uni", 120)) + " \x00 nul"),
        (val, "tiny two"),
        (train, " ".join(_words("eps", 90))),
    ]
    return [_row(f"r{i:02d}", repo=repo, text=text) for i, (repo, text) in enumerate(texts)]


def _unique(rows: list[DataRow]) -> list[DataRow]:
    """``dedupe`` refuses two rows that share a row id; ``_row`` keys identity by row id too."""
    assert len({r.row_id for r in rows}) == len(rows)
    return rows


def _decisions(rows: list[DataRow]) -> tuple[Any, ...]:
    report = dedupe(list(rows), config=CONFIG)
    done = split(report, config=CONFIG)
    return (
        tuple(r.row_id for r in report.kept), report.clusters, report.n_candidate_pairs,
        done.assignments, {k: tuple(r.row_id for r in v) for k, v in done.rows_by_split.items()},
    )


@pytest.fixture
def prep(qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(qd_prep_bin))
    return qd_prep_bin


@pytest.mark.parametrize("batch_sets", [1, 2, 3, 5, pipeline.NATIVE_MINHASH_BATCH_SETS])
def test_dedupe_and_split_decide_as_the_reference_at_every_batch_size(
    prep: Path, monkeypatch: pytest.MonkeyPatch, batch_sets: int,
) -> None:
    rows = _unique(_fixture())
    monkeypatch.delenv(pipeline.PREP_BIN_ENV)
    want = _decisions(rows)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep))
    with pipeline.native_minhash(rows, config=CONFIG, batch_sets=batch_sets):
        got = _decisions(rows)
    assert got == want
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher
    assert want[1], "the fixture holds a duplicate cluster, so the comparison compares something"


def test_signing_calls_hold_at_most_the_batch_and_cover_every_distinct_set_once(
    prep: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _unique(_fixture())
    sizes: list[int] = []
    real = pipeline._prep_signature_matrix

    def spy(binary: Path, sets: list[frozenset[bytes]], reference: Any) -> np.ndarray:
        sizes.append(len(sets))
        return real(binary, sets, reference)

    monkeypatch.setattr(pipeline, "_prep_signature_matrix", spy)
    texts = {t for r in rows for t in (r.dedupe_text,)}
    distinct = len({frozenset(shingle(t, k=CONFIG.shingle_size).shingles) for t in texts})
    with pipeline.native_minhash(rows, config=CONFIG, batch_sets=2):
        _decisions(rows)
    assert sum(sizes) == distinct and max(sizes) <= 2
    assert len(sizes) == math.ceil(distinct / 2)


def test_a_signature_row_is_the_references_tuple(prep: Path) -> None:
    rows = _unique(_fixture())
    reference = MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)
    with pipeline.native_minhash(rows, config=CONFIG, batch_sets=2):
        hasher = dedupe_module.MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)
        for row in rows:
            packed = dedupe_module.shingle(row.dedupe_text, k=CONFIG.shingle_size).shingles
            plain = shingle(row.dedupe_text, k=CONFIG.shingle_size).shingles
            want = reference.signature(plain)
            for got in (hasher.signature(packed), hasher.signature(plain)):
                assert got == want and want == got
                assert len(got) == len(want) and tuple(got) == want
                assert got[3:11] == want[3:11] and got[-1] == want[-1]
                assert hash(got) == hash(want)
        _decisions(rows)


def test_packed_exact_jaccard_is_the_references_float(prep: Path) -> None:
    rows = _unique(_fixture())
    plain = [shingle(r.dedupe_text, k=CONFIG.shingle_size).shingles for r in rows]
    with pipeline.native_minhash(rows, config=CONFIG, batch_sets=4):
        packed = [dedupe_module.shingle(r.dedupe_text, k=CONFIG.shingle_size).shingles
                  for r in rows]
        for i, (a, pa) in enumerate(zip(plain, packed, strict=True)):
            for b, pb in zip(plain[i:], packed[i:], strict=True):
                assert dedupe_module.exact_jaccard(pa, pb) == exact_jaccard(a, b)
                assert split_module.exact_jaccard(pb, pa) == exact_jaccard(b, a)
        with pytest.raises(SystemExit, match="one packed shingle set and one that is not"):
            dedupe_module.exact_jaccard(packed[0], plain[0])
        with pytest.raises(TypeError, match="hashes of its shingles"):
            iter(packed[0])
        _decisions(rows)
    assert dedupe_module.exact_jaccard is exact_jaccard


def test_the_block_holds_hashes_not_shingles(prep: Path) -> None:
    """60 texts of 4,000 distinct words: ~240k shingles that share nothing, so nothing to
    intern. ``frozenset[bytes]`` held 27.8 MiB here (116 bytes a shingle); hashes held 2.4 MiB."""
    repo = _repo_in("train", "org/t")
    rows = [
        _row(f"r{i}", repo=repo, text=" ".join(f"t{i:03d}w{j:05d}" for j in range(4000)))
        for i in range(60)
    ]
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        with pipeline.native_minhash(rows, config=CONFIG, batch_sets=8):
            held = tracemalloc.get_traced_memory()[0] - before
            dedupe(list(rows), config=CONFIG)
    finally:
        tracemalloc.stop()
    assert held < BOUND_MIB << 20, f"{held / (1 << 20):.1f} MiB held after the table build"


@pytest.mark.parametrize("bad", [0, -1, pipeline.MAX_NATIVE_MINHASH_BATCH_SETS + 1, True, 2.5])
def test_a_batch_size_outside_its_bounds_is_refused(prep: Path, bad: Any) -> None:
    with (
        pytest.raises(SystemExit, match="batch_sets"),
        pipeline.native_minhash(_fixture(), config=CONFIG, batch_sets=bad),
    ):
        raise AssertionError("the block must not open")
    assert dedupe_module.MinHasher is MinHasher


def test_no_rows_signs_nothing_and_a_run_that_reads_nothing_is_not_refused(prep: Path) -> None:
    with pipeline.native_minhash([], config=CONFIG, batch_sets=3):
        assert dedupe_module.MinHasher is not MinHasher
    assert dedupe_module.MinHasher is MinHasher


def test_one_row_and_a_row_at_the_length_limit(prep: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qd_data.rows import MAX_DEDUPE_TEXT_BYTES

    repo = _repo_in("train", "org/t")
    one = [_row("only", repo=repo, text="a single row of exactly eight plain words here ok")]
    big = " ".join(f"w{j}" for j in range(60_000))[:MAX_DEDUPE_TEXT_BYTES]
    assert len(big.encode("utf-8")) == MAX_DEDUPE_TEXT_BYTES
    wide = [_row("wide", repo=repo, text=big)]
    for rows in (one, wide):
        monkeypatch.delenv(pipeline.PREP_BIN_ENV)
        want = _decisions(rows)
        monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep))
        with pipeline.native_minhash(rows, config=CONFIG, batch_sets=1):
            assert _decisions(rows) == want


def test_a_hash_collision_inside_one_set_is_refused_not_counted_as_one_shingle(
    prep: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "hash", lambda _b: 7, raising=False)
    with (
        pytest.raises(SystemExit, match="same 64-bit hash"),
        pipeline.native_minhash(_fixture(), config=CONFIG, batch_sets=2),
    ):
        raise AssertionError("the block must not open")
    assert dedupe_module.MinHasher is MinHasher


def test_a_signing_call_that_fails_part_way_leaves_nothing_installed(
    prep: Path, tmp_path: Path,
) -> None:
    """The second ``qd-prep minhash`` call exits 9: the block refuses, no partial table is
    installed, and ``dedupe`` and ``split`` read the reference's names again."""
    count = tmp_path / "count"
    wrapper = tmp_path / "qd-prep"
    wrapper.write_text(
        "#!/bin/bash\n"
        f"n=$(cat {count} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {count}\n"
        'if [ "$n" -ge 2 ]; then echo boom >&2; exit 9; fi\n'
        f'exec {prep} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    old = os.environ[pipeline.PREP_BIN_ENV]
    os.environ[pipeline.PREP_BIN_ENV] = str(wrapper)
    try:
        with (
            pytest.raises(SystemExit, match="exited 9"),
            pipeline.native_minhash(_fixture(), config=CONFIG, batch_sets=2),
        ):
            raise AssertionError("the block must not open")
    finally:
        os.environ[pipeline.PREP_BIN_ENV] = old
    assert count.read_text(encoding="utf-8").strip() == "2"
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher
    assert dedupe_module.shingle is shingle
    assert dedupe_module.exact_jaccard is exact_jaccard
