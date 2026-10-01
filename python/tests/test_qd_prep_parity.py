"""``qd-prep`` (Rust) against the reference it replaces on the pipeline's shipped path.

``tools/real_tokenizer_pipeline.py`` signs MinHash through ``crates/qd-prep``.
``qd_data.minhash.MinHasher.signature`` is the reference oracle these tests import -- it is
not edited (``qd_data`` sources are hashed into every shard header's ``code_fingerprint``)
and it is never a runtime fallback. Three layers are pinned here:

* the binary's signatures equal the reference's, on real corpus rows and on adversarial sets,
  whatever the thread count;
* ``dedupe_and_split`` -- ``dedupe`` then ``split`` with ``MinHasher`` answering from the
  binary's table -- produces reports equal to the reference's, on corpora built to straddle
  the 0.8 Jaccard threshold within and across repos;
* every way the native path can fail refuses: no binary, a binary speaking another protocol,
  a binary whose signatures disagree, a set it never signed, and a qd_data that stops looking
  up ``MinHasher`` where the replacement puts it.

``test_benchmark_native_against_the_reference`` is the committed A/B for the speedup claim.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from data_fixtures import small_corpus

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

import qd_data.dedupe as dedupe_module  # noqa: E402
import qd_data.split as split_module  # noqa: E402
from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.minhash import MinHasher, exact_jaccard, shingle  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.rows import DataRow, GoldAnswer  # noqa: E402
from qd_data.schema import ChoiceSlot, Request  # noqa: E402
from qd_data.split import split  # noqa: E402

CARGO = shutil.which("cargo")
DEFECT_EXAMPLES = REPO / "data" / "pool" / "commitpackft-corpus-v2" / "examples.jsonl"
CONFIG = DataConfig()


@pytest.fixture(scope="module")
def binary() -> Path:
    if CARGO is None:
        pytest.skip("cargo is not on PATH: qd-prep was not built")
    subprocess.run(
        [CARGO, "build", "--quiet", "--manifest-path", str(REPO / "Cargo.toml"),
         "-p", "qd-prep", "--bin", "qd-prep"],
        check=True, timeout=900,
    )
    built = REPO / "target" / "debug" / "qd-prep"
    assert built.is_file(), built
    return built


@pytest.fixture(scope="module")
def prep(binary: Path) -> pipeline.QdPrep:
    return pipeline.resolve_qd_prep(binary)


def _reference(sets: list[frozenset[bytes]], *, num_perm: int, seed: int) -> list[tuple[int, ...]]:
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    return [hasher.signature(s) for s in sets]


def _real_texts(limit: int) -> list[str]:
    if not DEFECT_EXAMPLES.is_file():
        pytest.skip(f"{DEFECT_EXAMPLES} is not on this host; only its manifest is tracked")
    texts: list[str] = []
    with DEFECT_EXAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            texts.append(json.loads(line)["diff"])
            if len(texts) >= limit:
                break
    return texts


# -- signatures ------------------------------------------------------------------------


def test_signatures_match_the_reference_on_real_rows(prep: pipeline.QdPrep) -> None:
    sets = [shingle(t, k=CONFIG.shingle_size).shingles for t in _real_texts(300)]
    got = pipeline.native_signatures(
        sets, num_perm=CONFIG.num_perm, seed=CONFIG.seed, qd_prep=prep
    )
    assert got == _reference(sets, num_perm=CONFIG.num_perm, seed=CONFIG.seed)
    assert all(type(v) is int for v in got[0]), "numpy scalars would break int.to_bytes"


ADVERSARIAL_TEXTS = (
    "x",  # one token: fewer than k, so one shingle of the whole text
    "a b c d",  # k-1 tokens
    "a b c d e",  # exactly k
    "line one\r\nline two\r\n\r\nline three\rfour",  # CRLF and a bare CR
    "tab\tsep\x0bvt\x0cff\x1cfs\x1dgs\x1ers\x1fus\x85nel\xa0nbsp",  # every ASCII/Latin-1 isspace
    "".join(("ogham", chr(0x1680), "em", chr(0x2003), "thin", chr(0x2009), "ls", chr(0x2028),
               "ps", chr(0x2029), "nnb", chr(0x202F), "mm", chr(0x205F), "ideo", chr(0x3000),
               "end")),  # the Unicode isspace characters
    "".join(("zwsp", chr(0x200B), "is", chr(0x200B), "not space mongolian", chr(0x180E),
               "vs")),  # look like spaces, are not
    "".join(("caf", chr(0xE9), " ", chr(0x4E2D), chr(0x6587), " ", chr(0x1F600), " emoji ",
               chr(0x301), "combining ", chr(0xD7FF), " edge")),
    "nul\x00byte inside a token and \x00 alone",
    "repeat " * 50,  # one distinct shingle, many times
    "x" * 70_000,  # one enormous token
    ("w" * 9 + " ") * 30_000,  # 300,000 bytes: over the 262,144-byte shingling bound
    (chr(0xE9) * 5 + " ") * 50_000,  # truncation lands inside a 2-byte character
)


def test_signatures_match_the_reference_on_adversarial_sets(prep: pipeline.QdPrep) -> None:
    sets = [shingle(t, k=CONFIG.shingle_size).shingles for t in ADVERSARIAL_TEXTS]
    sets += [frozenset({b""}), frozenset({b"\xff\xfe\x00"}), frozenset({b"a"}), frozenset({b"a"})]
    assert all(sets)
    for num_perm, seed in ((CONFIG.num_perm, CONFIG.seed), (1, 0), (7, -3), (64, 10**30)):
        got = pipeline.native_signatures(sets, num_perm=num_perm, seed=seed, qd_prep=prep)
        assert got == _reference(sets, num_perm=num_perm, seed=seed), (num_perm, seed)


def test_the_thread_count_does_not_change_a_signature(prep: pipeline.QdPrep) -> None:
    sets = [shingle(t, k=CONFIG.shingle_size).shingles for t in _real_texts(200)]
    kw = {"num_perm": CONFIG.num_perm, "seed": CONFIG.seed, "qd_prep": prep}
    one = pipeline.native_signatures(sets, threads=1, **kw)
    for threads in (2, 7, 64):
        assert pipeline.native_signatures(sets, threads=threads, **kw) == one, threads


def test_an_empty_set_is_refused_by_the_binary(prep: pipeline.QdPrep) -> None:
    with pytest.raises(SystemExit, match="cannot sign an empty shingle set"):
        pipeline.native_signatures([frozenset({b"a"}), frozenset()], num_perm=8, seed=1,
                                   qd_prep=prep)


# -- dedupe and split ------------------------------------------------------------------


def _row(row_id: str, *, repo: str, text: str, path: str = "a.py") -> DataRow:
    return DataRow(
        row_id=row_id,
        source_id="bigcode/commitpackft",
        host="huggingface",
        family_id="code.commit_intent",
        repo_key=repo,
        identity_key=f"{repo}::{path}",
        licence_id="mit",
        obligations=("attribution",),
        request=Request(
            task="code.commit_intent",
            context=text.encode("utf-8"),
            question="q?",
            slots=(ChoiceSlot(name="implements_claim", options=("yes", "no")),),
            example_id=row_id,
        ),
        gold=(GoldAnswer(slot_name="implements_claim", value="yes"),),
        dedupe_text=text,
    )


def _tokens(stem: str, n: int) -> list[str]:
    return [f"{stem}{i}" for i in range(n)]


def _threshold_corpus() -> list[DataRow]:
    """Pairs at, just under and well over J = 0.8, within one repo and across two.

    49 distinct tokens make 45 five-token shingles; replacing one token mid-text changes the
    5 shingles that contain it, so J = 40/50 = 0.8 exactly -- a duplicate, since dedupe keeps
    J >= 0.8. Replacing two adjacent tokens changes 6: J = 39/51, just under. With 100 tokens
    and one replaced, 91/101.
    """
    rows: list[DataRow] = []
    for g in range(12):
        at = _tokens(f"a{g}x", 49)
        under = _tokens(f"u{g}x", 49)
        over = _tokens(f"o{g}x", 100)
        for name, base, changed in (
            ("at", at, [*at[:20], "CHANGED", *at[21:]]),
            ("under", under, [*under[:20], "CHANGED", "TWICE", *under[22:]]),
            ("over", over, [*over[:50], "CHANGED", *over[51:]]),
        ):
            cross = g % 2 == 0
            rows.append(_row(f"{name}{g}-a", repo=f"org/r{g}", text=" ".join(base)))
            rows.append(_row(
                f"{name}{g}-b", repo=f"org/r{g}" + ("-fork" if cross else ""),
                text=" ".join(changed), path="b.py",
            ))
    # Exact duplicates: one content unit spanning two rows, and the same text in a third repo.
    rows.append(_row("dup-1", repo="org/dup", text="same text " * 20))
    rows.append(_row("dup-2", repo="org/dup", text="same text " * 20, path="a.py"))
    rows.append(_row("dup-3", repo="org/elsewhere", text="same text " * 20))
    return rows


def _check_the_corpus_straddles_the_threshold(rows: list[DataRow]) -> None:
    by_id = {r.row_id: r for r in rows}
    for name, lo, hi in (("at", 0.8, 0.8), ("under", 0.7, 0.799), ("over", 0.85, 0.95)):
        a = shingle(by_id[f"{name}0-a"].dedupe_text, k=CONFIG.shingle_size).shingles
        b = shingle(by_id[f"{name}0-b"].dedupe_text, k=CONFIG.shingle_size).shingles
        assert lo <= exact_jaccard(a, b) <= hi, (name, exact_jaccard(a, b))


def _assert_reports_equal(rows: list[DataRow], prep: pipeline.QdPrep) -> None:
    want_dedupe = dedupe(rows, config=CONFIG)
    want_split = split(want_dedupe, config=CONFIG)
    got_dedupe, got_split, note = pipeline.dedupe_and_split(
        rows, config=CONFIG, qd_prep=prep.path
    )
    assert "qd-prep" in note and prep.sha256[:16] in note
    assert got_dedupe.to_json() == want_dedupe.to_json()
    assert [r.row_id for r in got_dedupe.kept] == [r.row_id for r in want_dedupe.kept]
    assert got_dedupe.dropped_row_ids == want_dedupe.dropped_row_ids
    assert got_split.assignments == want_split.assignments
    assert {k: [r.row_id for r in v] for k, v in got_split.rows_by_split.items()} == {
        k: [r.row_id for r in v] for k, v in want_split.rows_by_split.items()
    }
    for name in ("repo_disjoint", "identity_disjoint", "near_duplicate_disjoint",
                 "held_out_families_absent_from_training", "dedupe_status",
                 "content_disjoint_families", "status"):
        assert getattr(got_split, name).to_json() == getattr(want_split, name).to_json(), name
    # The replacement was undone: the reference is what qd_data sees again.
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher


def test_dedupe_and_split_equal_the_reference_at_the_threshold(prep: pipeline.QdPrep) -> None:
    rows = _threshold_corpus()
    _check_the_corpus_straddles_the_threshold(rows)
    report = dedupe(rows, config=CONFIG)
    assert report.n_cross_repo_pairs > 0 and report.n_within_repo_pairs > 0, report.to_json()
    _assert_reports_equal(rows, prep)


def test_dedupe_and_split_equal_the_reference_on_a_mixture(prep: pipeline.QdPrep) -> None:
    rows = list(build_mixture(small_corpus(24), config=CONFIG).rows)
    _assert_reports_equal(rows, prep)


def test_dedupe_and_split_equal_the_reference_on_real_defect_rows(
    prep: pipeline.QdPrep,
) -> None:
    from qd_data.defect_class import load_defect_rows
    from qd_data.mixture import rewrite_defect_class

    _real_texts(1)  # skips where the corpus is absent
    load = load_defect_rows(
        DEFECT_EXAMPLES.parent, download_root=REPO / "data" / "pool" / "commitpackft",
        config=CONFIG, repo_root=REPO, max_rows=400,
    )
    rows = [
        rewrite_defect_class(r, family_id="code.defect_class", index=i, config=CONFIG)
        for i, r in enumerate(load.rows)
    ]
    _assert_reports_equal(rows, prep)


def test_no_rows_equal_the_reference(prep: pipeline.QdPrep) -> None:
    _assert_reports_equal([], prep)


# -- refusals --------------------------------------------------------------------------


def test_the_shipped_path_refuses_without_a_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(pipeline.QD_PREP_ENV, raising=False)
    with pytest.raises(SystemExit, match="pass --qd-prep or set QD_PREP_BIN"):
        pipeline.dedupe_and_split(_threshold_corpus(), config=CONFIG, qd_prep=None)


def test_the_environment_names_the_binary(
    monkeypatch: pytest.MonkeyPatch, binary: Path
) -> None:
    monkeypatch.setenv(pipeline.QD_PREP_ENV, str(binary))
    assert pipeline.resolve_qd_prep(None).path == binary


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-qd-prep"
    path.write_text(f"#!{sys.executable}\nimport sys\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_a_binary_that_is_not_executable_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "qd-prep"
    plain.write_text("not a program", encoding="utf-8")
    with pytest.raises(SystemExit, match="not an executable file"):
        pipeline.resolve_qd_prep(plain)
    with pytest.raises(SystemExit, match="not an executable file"):
        pipeline.resolve_qd_prep(tmp_path / "absent")


def test_a_binary_speaking_another_protocol_is_refused(tmp_path: Path) -> None:
    other = _script(
        tmp_path, 'print(\'{"name": "qd-prep", "version": "9", "minhash_protocol": 2}\')'
    )
    with pytest.raises(SystemExit, match="speaks minhash protocol 1"):
        pipeline.resolve_qd_prep(other)


def test_signatures_that_disagree_with_the_reference_are_refused(tmp_path: Path) -> None:
    """A binary that answers the protocol but returns wrong values: the canary refuses."""
    liar = _script(tmp_path, "\n".join((
        "import json, struct",
        "if sys.argv[1] == 'version':",
        "    print(json.dumps({'name': 'qd-prep', 'version': 'x', 'minhash_protocol': 1}))",
        "    raise SystemExit(0)",
        "args = dict(zip(sys.argv[2::2], sys.argv[3::2]))",
        "req = open(args['--input'], 'rb').read()",
        "num_perm, seed_len = struct.unpack_from('<II', req, 8)",
        "(n,) = struct.unpack_from('<Q', req, 16 + seed_len)",
        "out = b'QDMHOUT1' + struct.pack('<IIQ', num_perm, 0, n) + bytes(8 * num_perm * n)",
        "open(args['--output'], 'wb').write(out)",
    )))
    with pytest.raises(SystemExit, match=r"disagrees with qd_data\.minhash\.MinHasher"):
        pipeline.dedupe_and_split(_threshold_corpus(), config=CONFIG, qd_prep=liar)


def test_a_set_the_binary_never_signed_is_refused_not_signed_in_python() -> None:
    with pipeline._signer_replaced({}, num_perm=8, seed=1):
        hasher = dedupe_module.MinHasher(num_perm=8, seed=1)
        with pytest.raises(RuntimeError, match="refusing to sign it in Python"):
            hasher.signature(frozenset({b"x"}))
        with pytest.raises(ValueError, match="cannot sign an empty shingle set"):
            hasher.signature(frozenset())
        with pytest.raises(RuntimeError, match="qd-prep signed for"):
            split_module.MinHasher(num_perm=8, seed=2)
    assert dedupe_module.MinHasher is MinHasher and split_module.MinHasher is MinHasher


def test_signing_that_bypasses_the_replacement_is_refused(
    monkeypatch: pytest.MonkeyPatch, prep: pipeline.QdPrep
) -> None:
    """If qd_data stopped looking ``MinHasher`` up where the replacement puts it, Python
    would sign silently. The call counts are where that shows, and they refuse."""

    def bypassing_dedupe(rows: list[DataRow], *, config: DataConfig) -> object:
        replaced = dedupe_module.MinHasher
        dedupe_module.MinHasher = MinHasher
        try:
            return dedupe(rows, config=config)
        finally:
            dedupe_module.MinHasher = replaced

    monkeypatch.setattr(pipeline, "dedupe", bypassing_dedupe)
    with pytest.raises(SystemExit, match="bypassed the replacement"):
        pipeline.dedupe_and_split(_threshold_corpus(), config=CONFIG, qd_prep=prep.path)


# -- the committed benchmark ------------------------------------------------------------


def test_benchmark_native_against_the_reference(prep: pipeline.QdPrep) -> None:
    """Interleaved A/B, min of N, on real rows: reference ``MinHasher`` vs ``qd-prep``.

    Printed (``pytest -s``) rather than gated on a ratio, because the host is shared; the
    one assertion is that the native path is faster at all, and that both agree. The binary
    built here is the debug profile; ``cargo bench -p qd-prep`` times the release one.
    """
    sets = [shingle(t, k=CONFIG.shingle_size).shingles for t in _real_texts(1000)]
    kw = {"num_perm": CONFIG.num_perm, "seed": CONFIG.seed}
    ref_times: list[float] = []
    nat_times: list[float] = []
    for _ in range(3):
        t = time.perf_counter()
        want = _reference(sets, **kw)
        ref_times.append(time.perf_counter() - t)
        t = time.perf_counter()
        got = pipeline.native_signatures(sets, qd_prep=prep, **kw)
        nat_times.append(time.perf_counter() - t)
        assert got == want
    ref, nat = min(ref_times), min(nat_times)
    print(
        f"\nMinHash over {len(sets)} real shingle sets, min of 3 interleaved: reference "
        f"{ref:.3f} s, qd-prep {nat:.3f} s (incl. subprocess + file I/O), {ref / nat:.1f}x"
    )
    assert nat < ref
