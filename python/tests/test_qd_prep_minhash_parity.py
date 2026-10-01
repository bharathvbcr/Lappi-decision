"""``crates/qd-prep`` signs MinHash shingle sets exactly as ``qd_data.minhash`` does.

``MinHasher.signature`` was 346 of 462 profiled seconds of the ``--score-checkpoint`` prelude
(2026-09-30); ``tools/real_tokenizer_pipeline.py::native_minhash`` hands it to the Rust
binary for ``ft_splits``. The Python stays the reference: these tests compare the binary's
signatures with it bit for bit -- on this repository's own source files, on the real
defect corpus when it is on disk, and on adversarial texts -- and pin every refusal of the
adapter: no binary, a relative path, an unrunnable file, a non-zero exit, a reply of the
wrong shape, a reply with the wrong numbers, a set the table does not hold, and a block in
which nothing asked for a signature.

The binary is built here with ``cargo build --release -p qd-prep``; without cargo the parity
tests are SKIPPED with that reason, never passed.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

import pytest
from data_fixtures import small_corpus, vendored_pair

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

import qd_data.dedupe as dedupe_module  # noqa: E402
import qd_data.minhash as minhash_module  # noqa: E402
import qd_data.split as split_module  # noqa: E402
from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.minhash import (  # noqa: E402
    DEFAULT_MAX_DOC_BYTES,
    MinHasher,
    ShingleResult,
    shingle,
)
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.split import split  # noqa: E402

CONFIG = DataConfig()
DEFECT_EXAMPLES = REPO / "data" / "pool" / "commitpackft-corpus-v2" / "examples.jsonl"


@pytest.fixture
def prep_bin(qd_prep_bin: Path) -> Path:
    return qd_prep_bin


def test_the_fixture_and_the_adapter_name_one_variable() -> None:
    assert pipeline.PREP_BIN_ENV == "QD_PREP_BIN", "python/tests/conftest.py names it too"


def _reference() -> MinHasher:
    return MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)


def _sets(texts: list[str]) -> list[frozenset[bytes]]:
    out: dict[frozenset[bytes], None] = {}
    for text in texts:
        s = shingle(text, k=CONFIG.shingle_size).shingles
        if s:
            out.setdefault(s, None)
    return list(out)


def _assert_parity(binary: Path, texts: list[str]) -> int:
    sets = _sets(texts)
    reference = _reference()
    native = pipeline._prep_signatures(binary, sets, reference)
    assert len(native) == len(sets)
    for s, sig in zip(sets, native, strict=True):
        assert sig == reference.signature(s), f"{len(s)}-shingle set differs"
        assert all(isinstance(v, int) for v in sig), "candidate_pairs calls int.to_bytes"
    return len(sets)


ADVERSARIAL = [
    "x",  # one token: fewer than k, so one shingle, the whole token tuple
    "a b c d",  # four tokens, still fewer than k=5
    "a b c d e",  # exactly k
    "  leading\tand\ttrailing   whitespace  \n",
    "crlf line one\r\ncrlf line two\r\n\r\nthird\r\n",
    # The reference shingles (Python str.isspace() semantics: NBSP, U+2028, U+3000 and the
    # \x1c-\x1f separators split tokens); the binary only ever sees the shingle bytes.
    "non\u00a0breaking\u2028line\u3000ideographic\x1cfile\x1dgroup\x1erecord\x1funit end",
    "\u00fcn\u00efc\u00f8d\u00e9 \u03c4\u03b1\u03c5 \u7edf\u4e00 emoji \U0001f9ea\U0001f9ec "
    "combining e\u0301 rtl \u05e9\u05dc\u05d5\u05dd x y",
    "nul\x00inside a token and \x7f delete and \ufeff bom tokens here",
    " ".join(f"tok{i}" for i in range(10_000)),  # ~10k shingles in one document
    # Over DEFAULT_MAX_DOC_BYTES with a 4-byte character straddling the cut, so the
    # reference's errors="ignore" decode drops a partial sequence.
    "a " * (DEFAULT_MAX_DOC_BYTES // 2 - 1) + "\U0001f9ea tail tokens past the bound",
    "dup dup dup dup dup dup dup dup dup",  # one distinct shingle repeated
    "dup dup dup dup dup dup dup dup dup",  # the same text twice: one set, signed once
    "\n".join(f"    value_{i} = compute(step={i})" for i in range(400)),
]


def test_adversarial_texts_are_signed_bit_for_bit_like_the_reference(prep_bin: Path) -> None:
    assert _assert_parity(prep_bin, ADVERSARIAL) == len(ADVERSARIAL) - 1  # the duplicate
    assert shingle("a " * (DEFAULT_MAX_DOC_BYTES // 2 - 1) + "\U0001f9ea tail", k=5).truncated


def test_this_repositorys_own_sources_are_signed_like_the_reference(prep_bin: Path) -> None:
    """Real code that is always present: every Python and Rust source file in the repo."""
    paths = sorted(
        p for p in [*REPO.glob("python/**/*.py"), *REPO.glob("tools/*.py"),
                    *REPO.glob("crates/*/src/**/*.rs")]
        if "target" not in p.parts
    )
    texts = [p.read_text(encoding="utf-8", errors="strict") for p in paths]
    assert len(texts) > 100
    assert _assert_parity(prep_bin, texts) > 100


@pytest.mark.skipif(not DEFECT_EXAMPLES.is_file(), reason=f"{DEFECT_EXAMPLES} is not on disk")
def test_real_defect_corpus_rows_are_signed_like_the_reference(prep_bin: Path) -> None:
    """The phase-3 corpus itself: every string field of its first 400 examples."""
    texts: list[str] = []
    with DEFECT_EXAMPLES.open(encoding="utf-8") as fh:
        for _, line in zip(range(400), fh, strict=False):
            texts.extend(v for v in json.loads(line).values() if isinstance(v, str) and v)
    assert _assert_parity(prep_bin, texts) > 400


#: A/B rounds of the benchmark below; each round runs the reference, then the binary.
BENCH_ROUNDS = 3
#: Examples whose ``before``/``after``/``diff`` texts the benchmark signs (~3 texts each).
BENCH_EXAMPLES = 3000


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the reference-vs-qd-prep benchmark runs only with QD_PREP_BENCH=1 (~1 minute)",
)
@pytest.mark.skipif(not DEFECT_EXAMPLES.is_file(), reason=f"{DEFECT_EXAMPLES} is not on disk")
def test_benchmark_reference_against_qd_prep_interleaved_min_of_n(prep_bin: Path) -> None:
    """The committed A/B behind the speedup claim: the same real shingle sets, signed by the
    reference and by the binary in alternating rounds, min of :data:`BENCH_ROUNDS` each.
    The binary's time includes writing the request, the process and reading the reply.

        QD_PREP_BENCH=1 pytest -s python/tests/test_qd_prep_minhash_parity.py -k benchmark
    """
    texts: list[str] = []
    with DEFECT_EXAMPLES.open(encoding="utf-8") as fh:
        for _, line in zip(range(BENCH_EXAMPLES), fh, strict=False):
            example = json.loads(line)
            texts.extend(str(example[k]) for k in ("before", "after", "diff") if example.get(k))
    sets = _sets(texts)
    reference = _reference()
    a: list[float] = []
    b: list[float] = []
    for _ in range(BENCH_ROUNDS):
        started = time.perf_counter()
        want = [reference.signature(s) for s in sets]
        a.append(time.perf_counter() - started)
        started = time.perf_counter()
        got = pipeline._prep_signatures(prep_bin, sets, reference)
        b.append(time.perf_counter() - started)
        assert got == want
    print(json.dumps({
        "benchmark": "minhash_signature", "sets": len(sets),
        "shingles": sum(len(s) for s in sets), "num_perm": reference.num_perm,
        "rounds": BENCH_ROUNDS, "reference_s": [round(x, 3) for x in a],
        "qd_prep_s": [round(x, 3) for x in b], "reference_min_s": round(min(a), 3),
        "qd_prep_min_s": round(min(b), 3), "speedup_min_over_min": round(min(a) / min(b), 1),
    }))
    assert min(b) < min(a)


def test_an_empty_set_is_refused_by_the_binary_as_by_the_reference(prep_bin: Path) -> None:
    with pytest.raises(ValueError, match="empty shingle set"):
        _reference().signature(frozenset())
    with pytest.raises(SystemExit, match="no shingles"):
        pipeline._prep_signatures(prep_bin, [frozenset()], _reference())


def _corpus_rows():
    corpus = small_corpus(18)
    corpus["bigcode/commitpackft"].extend(vendored_pair())
    return build_mixture(corpus, config=CONFIG).rows


def _reports(rows):
    report = dedupe(list(rows), config=CONFIG)
    return report, split(report, config=CONFIG)


def test_dedupe_and_split_decide_identically_and_sign_only_the_canaries_in_python(
    prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The port is live: inside the block the reference signs only the canaries, and
    every dedupe and split decision -- kept rows, clusters, split assignments, the
    near-duplicate check -- is what the reference makes on its own."""
    rows = _corpus_rows()
    monkeypatch.delenv(pipeline.PREP_BIN_ENV, raising=False)
    want_report, want_split = _reports(rows)
    calls = 0
    original = MinHasher.signature

    def counting(self, shingles):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(self, shingles)

    monkeypatch.setattr(minhash_module.MinHasher, "signature", counting)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep_bin))
    with pipeline.native_minhash(rows, config=CONFIG):
        assert dedupe_module.MinHasher is not MinHasher, "installed while the block runs"
        got_report, got_split = _reports(rows)
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher
    n_sets = len(_sets([r.dedupe_text for r in rows]))
    assert calls == min(n_sets, 2 * pipeline.NATIVE_MINHASH_CANARIES), "only the canaries"
    assert want_report.n_dropped_rows > 0, "the corpus must exercise a duplicate"
    assert got_report == want_report
    assert got_split.assignments == want_split.assignments
    assert got_split.rows_by_split == want_split.rows_by_split
    assert got_split.near_duplicate_disjoint == want_split.near_duplicate_disjoint


def test_ft_splits_dedupes_and_splits_inside_the_native_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch", reason="real_ft_run imports torch at module scope")
    import contextlib
    from types import SimpleNamespace

    import real_ft_run as rft

    import qd_data.mixture as mixture_module

    rows = _corpus_rows()
    state = {"inside": False, "rows": -1}
    seen: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def spy(block_rows, *, config):  # type: ignore[no-untyped-def]
        state["rows"] = len(list(block_rows))
        state["inside"] = True
        try:
            yield
        finally:
            state["inside"] = False

    def fake_dedupe(rows, *, config):  # type: ignore[no-untyped-def]
        seen.append(("dedupe", state["inside"]))
        return "report"

    class Stop(Exception):
        pass

    def fake_split(report, *, config):  # type: ignore[no-untyped-def]
        seen.append(("split", state["inside"]))
        raise Stop

    monkeypatch.setattr(pipeline, "native_minhash", spy)
    monkeypatch.setattr(pipeline, "base_sources", lambda **kw: SimpleNamespace(raw={}))
    monkeypatch.setattr(
        mixture_module, "build_mixture",
        lambda raw, **kw: SimpleNamespace(rows=rows, prompt_consistency=None),
    )
    monkeypatch.setattr(dedupe_module, "dedupe", fake_dedupe)
    monkeypatch.setattr(split_module, "split", fake_split)
    with pytest.raises(Stop):
        rft.ft_splits(commitpackft=None, max_pairs=1, rev="r", config=CONFIG,
                      repo_history=False)
    assert state["rows"] == len(rows), "every row of the mixture is signed up front"
    assert seen == [("dedupe", True), ("split", True)]


# -- the adapter refuses, loudly, everything that is not a working binary ---------------


def test_without_the_variable_nothing_is_signed_and_the_run_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python MinHasher is the oracle, never a fallback."""
    monkeypatch.delenv(pipeline.PREP_BIN_ENV, raising=False)
    with (
        pytest.raises(SystemExit, match=r"QD_PREP_BIN is unset.*cargo build --release -p qd-prep"),
        pipeline.native_minhash(_corpus_rows(), config=CONFIG),
    ):
        raise AssertionError("the block ran without a binary")
    assert dedupe_module.MinHasher is MinHasher


@pytest.mark.parametrize("named", ["relative/qd-prep", "/no/such/qd-prep"])
def test_a_binary_that_is_not_an_absolute_file_is_refused(
    named: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, named)
    with (
        pytest.raises(SystemExit, match="cargo build --release -p qd-prep"),
        pipeline.native_minhash(_corpus_rows(), config=CONFIG),
    ):
        pass
    assert dedupe_module.MinHasher is MinHasher


def _script(tmp_path: Path, name: str, body: str) -> Path:
    """A stand-in binary; argv is ``minhash --input REQ --output REPLY``."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _reply(n_docs: int, values: list[int], k: int = CONFIG.num_perm) -> bytes:
    return b"QDPMHOK1" + struct.pack("<IQ", k, n_docs) + struct.pack(f"<{len(values)}Q", *values)


def test_an_unrunnable_file_a_failure_and_a_wrong_reply_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _corpus_rows()
    n = len(_sets([r.dedupe_text for r in rows]))
    k = CONFIG.num_perm
    not_a_program = tmp_path / "not-a-program"
    not_a_program.write_bytes(b"\x7fELF\x02\x01\x01garbage")
    not_a_program.chmod(0o755)
    canned = tmp_path / "canned.bin"
    cases = [
        (not_a_program, "could not be run"),
        (_script(tmp_path, "fails", "echo boom >&2; exit 3"), "exited 3: boom"),
        (_script(tmp_path, "short", f'cp "{canned}" "$5"'), "not this request's"),
    ]
    for binary, match in cases:
        canned.write_bytes(_reply(n, [1] * (k * n - 1)))  # one value short
        monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(binary))
        with (
            pytest.raises(SystemExit, match=match),
            pipeline.native_minhash(rows, config=CONFIG),
        ):
            pass
        assert dedupe_module.MinHasher is MinHasher
    # The right shape and the wrong numbers: the canaries catch it.
    canned.write_bytes(_reply(n, [1] * (k * n)))
    wrong = _script(tmp_path, "wrong", f'cp "{canned}" "$5"')
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(wrong))
    with (
        pytest.raises(SystemExit, match=r"differently from qd_data\.minhash\.MinHasher"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        pass


def test_a_set_outside_the_table_and_an_unread_table_are_refused(
    prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _corpus_rows()
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep_bin))
    with (
        pytest.raises(SystemExit, match="refusing to sign it in Python"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        dedupe_module.MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed).signature(
            frozenset({b"never seen"})
        )
    with (
        pytest.raises(SystemExit, match="native table was signed at"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        split_module.MinHasher(num_perm=64, seed=CONFIG.seed)
    with (
        pytest.raises(SystemExit, match="installed and never read"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        pass
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher


def test_each_distinct_text_is_shingled_once_and_every_decision_is_unchanged(
    prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dedupe`` shingles every content unit and ``split`` every kept row; with the block's
    own pass that was three reference passes over the corpus (104 s of the J1-shaped
    ``ft_splits`` rebuild, GAP-PERF-PRELUDE-J1-REMAINDER-IS-SHINGLE-BANDING-AND-RENDER).
    Inside the block both read the block's shingles, so the reference ``shingle`` runs once per
    distinct text, and the decisions are still the ones the reference makes on its own."""
    rows = _corpus_rows()
    monkeypatch.delenv(pipeline.PREP_BIN_ENV, raising=False)
    want_report, want_split = _reports(rows)
    calls = 0

    def counting(
        text: str, *, k: int, max_doc_bytes: int = DEFAULT_MAX_DOC_BYTES
    ) -> ShingleResult:
        nonlocal calls
        calls += 1
        return shingle(text, k=k, max_doc_bytes=max_doc_bytes)

    for module in (minhash_module, dedupe_module, split_module):
        monkeypatch.setattr(module, "shingle", counting)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep_bin))
    with pipeline.native_minhash(rows, config=CONFIG):
        assert dedupe_module.shingle is not counting, "installed while the block runs"
        got_report, got_split = _reports(rows)
    assert dedupe_module.shingle is counting and split_module.shingle is counting, "restored"
    assert calls == len({r.dedupe_text for r in rows}), "once per distinct text"
    assert want_report.n_dropped_rows > 0, "the corpus must exercise a duplicate"
    assert got_report == want_report
    assert got_split.assignments == want_split.assignments
    assert got_split.rows_by_split == want_split.rows_by_split
    assert got_split.near_duplicate_disjoint == want_split.near_duplicate_disjoint


def test_a_text_outside_the_shingle_table_another_k_and_an_unread_table_are_refused(
    prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _corpus_rows()
    signed = shingle(rows[0].dedupe_text, k=CONFIG.shingle_size).shingles
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep_bin))
    with (
        pytest.raises(SystemExit, match="refusing to shingle it in Python"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        dedupe_module.shingle("a text no row carries", k=CONFIG.shingle_size)
    with (
        pytest.raises(SystemExit, match="the shingle table was built at k="),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        split_module.shingle(rows[0].dedupe_text, k=CONFIG.shingle_size + 1)
    with (
        pytest.raises(SystemExit, match="shingle table was installed and never read"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        dedupe_module.MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed).signature(signed)
    assert dedupe_module.shingle is shingle and split_module.shingle is shingle


def test_a_module_that_shingles_through_another_function_is_refused_before_signing(
    prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring ``qd_data.minhash.shingle`` on exit would overwrite whatever the module held,
    and a table keyed by the reference's results says nothing about another function's."""

    def other(text: str, *, k: int, max_doc_bytes: int = DEFAULT_MAX_DOC_BYTES) -> ShingleResult:
        return shingle(text, k=k, max_doc_bytes=max_doc_bytes)

    monkeypatch.setattr(split_module, "shingle", other)
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep_bin))
    with (
        pytest.raises(SystemExit, match=r"qd_data\.split\.MinHasher or \.shingle is not"),
        pipeline.native_minhash(_corpus_rows(), config=CONFIG),
    ):
        raise AssertionError("the block ran over a module it cannot serve")
    assert split_module.shingle is other and dedupe_module.shingle is shingle
    assert dedupe_module.MinHasher is MinHasher


#: A/B rounds of the shingle-table benchmark; each round runs both arms once.
SHINGLE_BENCH_ROUNDS = 5


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the shingle-table benchmark runs only with QD_PREP_BENCH=1 (~1 minute)",
)
@pytest.mark.skipif(not DEFECT_EXAMPLES.is_file(), reason=f"{DEFECT_EXAMPLES} is not on disk")
def test_benchmark_shingle_table_interleaved_min_of_n(
    prep_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed A/B behind the shingle table: ``dedupe`` + ``split`` over the whole real
    defect-class corpus (phase 3's mixture) inside the native block, in alternating rounds, min
    of :data:`SHINGLE_BENCH_ROUNDS`.
    Arm ``reshingle`` runs the reference ``shingle`` again at every lookup, as ``dedupe`` and
    ``split`` did before the table (slightly cheaper than before: the block's own pass is one
    per distinct text in both arms); arm ``table`` reads the table alone. Both arms include the
    block's pass, its qd-prep signing and canaries, and every round's decisions are equal.

        QD_PREP_BENCH=1 pytest -s python/tests/test_qd_prep_minhash_parity.py -k shingle_table
    """
    from qd_data.dedupe import DedupeReport
    from qd_data.defect_class import DEFECT_SOURCE_ID, load_defect_rows
    from qd_data.split import SplitReport

    load = load_defect_rows(
        DEFECT_EXAMPLES.parent, download_root=REPO / "data" / "pool" / "commitpackft",
        config=CONFIG, repo_root=REPO,
    )
    rows = build_mixture({DEFECT_SOURCE_ID: list(load.rows)}, config=CONFIG).rows
    monkeypatch.setenv(pipeline.PREP_BIN_ENV, str(prep_bin))

    def stage(*, reshingle: bool) -> tuple[DedupeReport, SplitReport]:
        with pipeline.native_minhash(rows, config=CONFIG):
            if reshingle:
                table = dedupe_module.shingle

                def again(
                    text: str, *, k: int, max_doc_bytes: int = DEFAULT_MAX_DOC_BYTES
                ) -> ShingleResult:
                    shingle(text, k=k, max_doc_bytes=max_doc_bytes)
                    return table(text, k=k, max_doc_bytes=max_doc_bytes)

                dedupe_module.shingle = again  # type: ignore[attr-defined]
                split_module.shingle = again  # type: ignore[attr-defined]
            report = dedupe(list(rows), config=CONFIG)
            return report, split(report, config=CONFIG)

    a: list[float] = []
    b: list[float] = []
    for _ in range(SHINGLE_BENCH_ROUNDS):
        started = time.perf_counter()
        want_report, want_split = stage(reshingle=True)
        a.append(time.perf_counter() - started)
        started = time.perf_counter()
        got_report, got_split = stage(reshingle=False)
        b.append(time.perf_counter() - started)
        assert got_report == want_report
        assert got_split.assignments == want_split.assignments
        assert got_split.near_duplicate_disjoint == want_split.near_duplicate_disjoint
    assert dedupe_module.shingle is shingle and split_module.shingle is shingle
    print(json.dumps({
        "benchmark": "dedupe_split_shingle_table", "rows": len(rows),
        "distinct_texts": len({r.dedupe_text for r in rows}), "rounds": SHINGLE_BENCH_ROUNDS,
        "reshingle_s": [round(x, 3) for x in a], "table_s": [round(x, 3) for x in b],
        "reshingle_min_s": round(min(a), 3), "table_min_s": round(min(b), 3),
        "speedup_min_over_min": round(min(a) / min(b), 2),
    }))
    assert min(b) < min(a)
