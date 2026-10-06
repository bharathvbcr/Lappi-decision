"""``qd-prep lsh`` finds exactly the pairs ``qd_data.minhash.candidate_pairs`` finds.

``candidate_pairs`` was 24.6 of the 78 s phase-4 ``ft_splits`` rebuild on the Mac (2026-10-01,
one unprofiled run at load ~40: 10.0 s in ``dedupe``, 14.6 s in ``split``'s cross-check), and a
``tools/real_ft_run.py`` prelude with ``--ood`` rebuilds twice.
``real_tokenizer_pipeline.native_minhash`` now hands it to the Rust binary; the Python stays the
reference. These tests compare the two --
the pair set, the truncation flag, and under a bound the reference's own order-dependent
prefix -- on adversarial keys (unicode, lone surrogates, collisions, zero bands or rows), on
real signatures, and through ``dedupe`` and ``split`` themselves; and they pin every refusal
of the adapter.

The binary is built by ``conftest.qd_prep_bin``; without cargo these are SKIPPED, never passed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from data_fixtures import code_body, commitpackft_row, small_corpus, vendored_pair
from hypothesis import given, settings
from hypothesis import strategies as st

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
    BandConfig,
    MinHasher,
    candidate_pairs,
    choose_bands,
    shingle,
)
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.split import split  # noqa: E402

CONFIG = DataConfig()
BANDS = choose_bands(num_perm=CONFIG.num_perm, threshold=CONFIG.dedupe_threshold)
DEFECT_EXAMPLES = REPO / "data" / "pool" / "commitpackft-corpus-v2" / "examples.jsonl"
#: Characters whose UTF-8 bytes stress the ordering: ASCII, two-, three- and four-byte forms,
#: the private-use and U+FFFF edges either side of the surrogates, and lone surrogates (which
#: only ``surrogatepass`` can encode; ``str`` order is their code point, U+D800-DFFF).
KEY_CHARS = ["a", "b", "z", "\x00", "é", "߿", "ࠀ", "\ud800", "\udbff", "\udc00",
             "\udfff", "", "￿", "\U00010000", "\U0001f9ea"]


def _band(binary: Path, sigs: dict[str, tuple[int, ...]], config: BandConfig,
          max_pairs: int) -> pipeline.NativePairs:
    width = config.bands * config.rows
    banded = np.array([sig[:width] for sig in sigs.values()], dtype=np.uint64)
    return pipeline._prep_candidate_pairs(
        binary, list(sigs), banded.reshape(len(sigs), width), bands=config.bands,
        rows=config.rows, max_pairs=max_pairs,
        min_agreement_permille=config.min_agreement_permille,
    )


@settings(max_examples=300, deadline=None)
@given(
    data=st.data(),
    keys=st.lists(st.text(alphabet=st.sampled_from(KEY_CHARS), max_size=4), max_size=14,
                  unique=True),
    bands=st.integers(min_value=0, max_value=4),
    rows=st.integers(min_value=0, max_value=3),
    permille=st.sampled_from([0, 1, 333, 500, 650, 999, 1000]),
)
def test_pairs_and_truncation_match_the_reference_on_adversarial_keys(
    qd_prep_bin: Path, data: st.DataObject, keys: list[str], bands: int, rows: int,
    permille: int,
) -> None:
    """A three-value pool forces bucket collisions; every bound from 0 to past the pair count
    is reachable, so the truncated prefix -- which depends on the order the reference adds
    pairs in -- is compared as often as the full set. The agreement prefilter is drawn too
    (0 is v5's QDPLSIN1 request), at per-mille values on and between whole agreement counts."""
    values = st.sampled_from([0, 1, 2**64 - 1])
    sigs = {k: tuple(data.draw(st.lists(values, min_size=bands * rows,
                                        max_size=bands * rows))) for k in keys}
    config = BandConfig(bands=bands, rows=rows, threshold=0.8, min_agreement_permille=permille)
    max_pairs = data.draw(st.integers(min_value=0, max_value=len(keys) ** 2 // 2 + 2))
    got = _band(qd_prep_bin, sigs, config, max_pairs)
    assert (got.pairs, got.truncated) == candidate_pairs(sigs, config=config, max_pairs=max_pairs)
    assert got.pairs == frozenset(got.ordered)


def _real_signatures(binary: Path) -> dict[str, tuple[int, ...]]:
    """Signatures of real code: the repository's own sources, then the phase-3 corpus when it
    is on disk, with a lightly edited copy of every tenth text so near-duplicates exist."""
    paths = sorted(p for p in [*REPO.glob("python/**/*.py"), *REPO.glob("tools/*.py"),
                               *REPO.glob("crates/*/src/**/*.rs")] if "target" not in p.parts)
    texts = {f"src:{p.relative_to(REPO)}": p.read_text(encoding="utf-8") for p in paths}
    if DEFECT_EXAMPLES.is_file():
        with DEFECT_EXAMPLES.open(encoding="utf-8") as fh:
            for i, line in zip(range(1500), fh, strict=False):
                example = json.loads(line)
                for field in ("before", "after"):
                    if example.get(field):
                        texts[f"defect:{i:05d}:{field}:é"] = str(example[field])
    for key, text in list(texts.items())[::10]:
        texts[key + ":edited"] = text + "\n# edited"
    shingled = {k: shingle(t, k=CONFIG.shingle_size).shingles for k, t in texts.items()}
    shingled = {k: s for k, s in shingled.items() if s}
    sets = list(dict.fromkeys(shingled.values()))
    reference = MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)
    signed = dict(zip(sets, pipeline._prep_signatures(binary, sets, reference), strict=True))
    return {k: signed[s] for k, s in shingled.items()}


@pytest.mark.parametrize("permille", [0, 650])
def test_real_signatures_band_like_the_reference_at_every_bound(
    qd_prep_bin: Path, permille: int
) -> None:
    bands = dataclasses.replace(BANDS, min_agreement_permille=permille)
    sigs = _real_signatures(qd_prep_bin)
    full, truncated = candidate_pairs(sigs, config=bands, max_pairs=10**9)
    assert not truncated
    assert len(full) > 20, "near-duplicates must exist or the comparison is vacuous"
    if permille:
        unfiltered, _ = candidate_pairs(sigs, config=BANDS, max_pairs=10**9)
        assert full <= unfiltered, "the prefilter only ever removes candidates"
    for bound in (0, 1, 7, len(full) // 2, len(full) - 1, len(full), len(full) + 1):
        got = _band(qd_prep_bin, sigs, bands, bound)
        assert (got.pairs, got.truncated) == candidate_pairs(sigs, config=bands, max_pairs=bound)


def test_v6_dedupe_and_split_decide_identically_inside_the_native_block(qd_prep: Path) -> None:
    """The prefilter reaches qd-prep lsh through ``native_minhash`` (a QDPLSIN2 request), and
    every v6 dedupe and split decision is the reference's."""
    config = DataConfig().with_v6_dedupe_rules()
    rows = _corpus_rows()
    want = dedupe(list(rows), config=config)
    want_split = split(want, config=config)
    with pipeline.native_minhash(rows, config=config):
        got = dedupe(list(rows), config=config)
        got_split = split(got, config=config)
    assert want.n_dropped_rows > 0 and got == want
    assert got_split.assignments == want_split.assignments
    assert got_split.near_duplicate_disjoint == want_split.near_duplicate_disjoint


def _corpus_rows():
    """The minhash parity corpus plus six near-copies of one file in six repositories --
    distinct texts, so distinct content units that only the LSH search can pair."""
    corpus = small_corpus(18)
    corpus["bigcode/commitpackft"].extend(vendored_pair())
    near = code_body("near", lines=40)
    corpus["bigcode/commitpackft"].extend(
        commitpackft_row(100 + j, repo=f"org/near{j}", body=f"{near}\n    tweak_{j} = {j}")
        for j in range(6)
    )
    return build_mixture(corpus, config=CONFIG).rows


def _reports(rows, **kwargs):
    report = dedupe(list(rows), config=CONFIG, **kwargs)
    return report, split(report, config=CONFIG)


def test_dedupe_and_split_decide_identically_and_band_only_the_canaries_in_python(
    qd_prep: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The port is live: inside the block the reference bands only canary-sized subsets
    (before the port it banded every content unit and every row), and every dedupe and split
    decision is the one the reference makes on its own."""
    rows = _corpus_rows()
    want_report, want_split = _reports(rows)
    sizes: list[int] = []

    def counting(signatures, *, config, max_pairs):  # type: ignore[no-untyped-def]
        sizes.append(len(signatures))
        return candidate_pairs(signatures, config=config, max_pairs=max_pairs)

    for module in (minhash_module, dedupe_module, split_module):
        monkeypatch.setattr(module, "candidate_pairs", counting)
    with pipeline.native_minhash(rows, config=CONFIG):
        assert dedupe_module.candidate_pairs is not counting, "installed while the block runs"
        got_report, got_split = _reports(rows)
    assert dedupe_module.candidate_pairs is counting and split_module.candidate_pairs is counting
    assert len(rows) > 6 * pipeline.NATIVE_LSH_CANARIES, "the corpus must outnumber the canaries"
    assert len(sizes) == 2, "one canary check for dedupe, one for split"
    assert max(sizes) <= 6 * pipeline.NATIVE_LSH_CANARIES, f"the reference banded {sizes} keys"
    assert want_report.n_dropped_rows > 0, "the corpus must exercise a duplicate"
    assert want_report.n_candidate_pairs > 0, "the corpus must exercise the LSH search"
    assert got_report == want_report
    assert got_split.assignments == want_split.assignments
    assert got_split.rows_by_split == want_split.rows_by_split
    assert got_split.near_duplicate_disjoint == want_split.near_duplicate_disjoint


def test_a_truncated_search_is_the_reference_truncated_search(qd_prep: Path) -> None:
    """At a bound of one pair dedupe reports NotRun; the native block reports the same
    NotRun, after re-running the whole reference search to confirm the prefix."""
    rows = _corpus_rows()
    with pipeline.native_minhash(rows, config=CONFIG):
        got = dedupe(list(rows), config=CONFIG, max_candidate_pairs=1)
    assert got == dedupe(list(rows), config=CONFIG, max_candidate_pairs=1)
    assert "hit its bound of 1" in str(got.status)


def _table_signatures(rows) -> dict[str, tuple[int, ...]]:
    """Signatures as the block hands them out, for calling the swapped name directly."""
    hasher = dedupe_module.MinHasher(num_perm=CONFIG.num_perm, seed=CONFIG.seed)
    return {
        r.row_id: hasher.signature(dedupe_module.shingle(r.dedupe_text, k=CONFIG.shingle_size)
                                   .shingles)
        for r in rows
    }


def test_the_reference_refusals_and_a_huge_bound_behave_as_the_reference(
    qd_prep: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two calls the reference answers are counted apart from the native ones."""
    rows = _corpus_rows()
    with pipeline.native_minhash(rows, config=CONFIG):
        sigs = _table_signatures(rows)
        with pytest.raises(ValueError, match="max_pairs must be >= 0"):
            dedupe_module.candidate_pairs(sigs, config=BANDS, max_pairs=-1)
        wide = BandConfig(bands=BANDS.bands + 1, rows=BANDS.rows, threshold=0.8)
        with pytest.raises(ValueError, match="band configuration needs"):
            split_module.candidate_pairs(sigs, config=wide, max_pairs=10**6)
        got = dedupe_module.candidate_pairs(sigs, config=BANDS, max_pairs=10**18)
    assert got == candidate_pairs(sigs, config=BANDS, max_pairs=10**18)
    assert got[0], "the corpus must have candidate pairs"
    said = capsys.readouterr().err
    assert "lsh: 1 candidate-pair searches banded" in said
    assert "2 handed to the reference" in said


def test_a_signature_the_block_did_not_sign_is_refused(qd_prep: Path) -> None:
    rows = _corpus_rows()
    alien = {"x": tuple(range(CONFIG.num_perm)), "y": tuple(range(CONFIG.num_perm))}
    with (
        pytest.raises(SystemExit, match=r"2 of 2 signatures .* were not signed by"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        dedupe_module.candidate_pairs(alien, config=BANDS, max_pairs=10)
    with (
        pytest.raises(SystemExit, match="1 of 2 signatures"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        signed = _table_signatures(rows[:1])
        # Equal to a signed tuple but not one: only what the block signed is banded.
        signed["copy"] = tuple(list(next(iter(signed.values()))))
        split_module.candidate_pairs(signed, config=BANDS, max_pairs=10)
    assert dedupe_module.candidate_pairs is candidate_pairs


def test_a_module_that_bands_through_another_function_is_refused_before_signing(
    qd_prep: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other(signatures, *, config, max_pairs):  # type: ignore[no-untyped-def]
        return candidate_pairs(signatures, config=config, max_pairs=max_pairs)

    monkeypatch.setattr(split_module, "candidate_pairs", other)
    with (
        pytest.raises(SystemExit, match=r"qd_data\.split\.candidate_pairs is not"),
        pipeline.native_minhash(_corpus_rows(), config=CONFIG),
    ):
        raise AssertionError("the block ran over a module it cannot serve")
    assert split_module.candidate_pairs is other
    assert dedupe_module.candidate_pairs is candidate_pairs


def test_a_block_that_signs_and_never_bands_is_refused(qd_prep: Path) -> None:
    rows = _corpus_rows()
    with (
        pytest.raises(SystemExit, match="never banded by qd-prep lsh"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        _table_signatures(rows)
    assert dedupe_module.candidate_pairs is candidate_pairs


@pytest.mark.parametrize("drift", ["drops a pair", "adds a pair", "flips the flag"])
def test_a_binary_whose_pairs_drift_is_refused_by_the_canaries(
    qd_prep: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    """The canaries are a sample: :data:`NATIVE_LSH_CANARIES` keys from each end of the
    request and of the reply. Here the request is the six near-copies and one other row, so
    every key is a canary and any drift among them must be caught."""
    real = pipeline._prep_candidate_pairs

    def drifted(*args, **kwargs):  # type: ignore[no-untyped-def]
        got = real(*args, **kwargs)
        if drift == "drops a pair":
            ordered = got.ordered[1:]
        elif drift == "adds a pair":
            ordered = (*got.ordered, tuple(sorted((other.row_id, near[0].row_id))))
        else:
            return pipeline.NativePairs(got.ordered, got.pairs, not got.truncated)
        return pipeline.NativePairs(ordered, frozenset(ordered), got.truncated)

    rows = _corpus_rows()
    near = [r for r in rows if "tweak_" in r.dedupe_text]
    near = [r for r in near if r.family_id == near[0].family_id]  # one row per near-copy
    other = next(r for r in rows if "tweak_" not in r.dedupe_text)
    assert len(near) + 1 <= 2 * pipeline.NATIVE_LSH_CANARIES
    monkeypatch.setattr(pipeline, "_prep_candidate_pairs", drifted)
    with (
        pytest.raises(SystemExit, match="its pairs are not the reference's"),
        pipeline.native_minhash(rows, config=CONFIG),
    ):
        sigs = _table_signatures([*near, other])
        want, _ = candidate_pairs(sigs, config=BANDS, max_pairs=100)
        assert len(want) == 15, "the six near-copies pair with each other and nothing else"
        dedupe_module.candidate_pairs(sigs, config=BANDS, max_pairs=100)
    assert dedupe_module.candidate_pairs is candidate_pairs


def _lsh_reply(flag: int, pairs: list[tuple[int, int]], n_pairs: int | None = None) -> bytes:
    body = b"".join(struct.pack("<II", lo, hi) for lo, hi in pairs)
    count = len(pairs) if n_pairs is None else n_pairs
    return b"QDPLSOK1" + struct.pack("<BQ", flag, count) + body


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (b"QDPMHOK1" + bytes(9), "not a b'QDPLSOK1' file"),
        (_lsh_reply(0, [(0, 1)], n_pairs=2), "not a qd-prep lsh reply"),
        (_lsh_reply(2, []), "not a qd-prep lsh reply"),
        (_lsh_reply(1, [(0, 1)]), "only there"),  # truncated below the bound
        (_lsh_reply(0, [(0, 1), (0, 2), (1, 2)]), "only there"),  # past the bound, not flagged
        (_lsh_reply(0, [(0, 3)]), "past the 3 keys"),
        (_lsh_reply(0, [(1, 0)]), r"not \(lo, hi\) in key order"),
        (_lsh_reply(0, [(0, 1), (0, 1)]), "of which 1 are distinct"),
    ],
)
def test_a_reply_of_the_wrong_shape_is_refused(
    qd_prep: Path, monkeypatch: pytest.MonkeyPatch, reply: bytes, match: str
) -> None:
    monkeypatch.setattr(pipeline, "_run_prep", lambda *a, **k: reply)
    with pytest.raises(SystemExit, match=match):
        pipeline._prep_candidate_pairs(
            qd_prep, ["a", "b", "c"], np.zeros((3, 2), dtype=np.uint64), bands=2, rows=1,
            max_pairs=2,
        )


def test_a_failing_binary_and_a_misshapen_block_are_refused(tmp_path: Path) -> None:
    fails = tmp_path / "fails"
    fails.write_text("#!/bin/sh\necho boom >&2; exit 3\n", encoding="utf-8")
    fails.chmod(0o755)
    with pytest.raises(SystemExit, match="lsh exited 3: boom"):
        pipeline._prep_candidate_pairs(
            fails, ["a"], np.zeros((1, 2), dtype=np.uint64), bands=2, rows=1, max_pairs=0
        )
    with pytest.raises(SystemExit, match="signature block"):
        pipeline._prep_candidate_pairs(
            fails, ["a"], np.zeros((1, 3), dtype=np.uint64), bands=2, rows=1, max_pairs=0
        )


#: A/B rounds of the LSH benchmark; each round runs both arms once, alternating which is first.
LSH_BENCH_ROUNDS = 5


@pytest.mark.skipif(
    os.environ.get("QD_PREP_BENCH") != "1",
    reason="the reference-vs-qd-prep LSH benchmark runs only with QD_PREP_BENCH=1 (~15 s)",
)
@pytest.mark.skipif(not DEFECT_EXAMPLES.is_file(), reason=f"{DEFECT_EXAMPLES} is not on disk")
def test_benchmark_lsh_reference_against_qd_prep_interleaved_min_of_n(qd_prep: Path) -> None:
    """The committed A/B behind the LSH port: ``split``'s cross-check search over every row of
    the real phase-3 defect-class mixture, banded by the reference and by the swapped name
    inside the native block, in alternating rounds, min of :data:`LSH_BENCH_ROUNDS`. The native
    arm's time is everything the swap does: the matrix gather, the request, the process, the
    reply checks, the frozenset and the canaries. Every round's answers are equal.

        QD_PREP_BENCH=1 pytest -s python/tests/test_qd_prep_lsh_parity.py -k benchmark
    """
    from qd_data.defect_class import DEFECT_SOURCE_ID, load_defect_rows

    load = load_defect_rows(
        DEFECT_EXAMPLES.parent, download_root=REPO / "data" / "pool" / "commitpackft",
        config=CONFIG, repo_root=REPO,
    )
    rows = build_mixture({DEFECT_SOURCE_ID: list(load.rows)}, config=CONFIG).rows
    bound = 5_000_000
    times: dict[str, list[float]] = {"reference": [], "qd_prep": []}
    answers: dict[str, tuple[frozenset[tuple[str, str]], bool]] = {}
    with pipeline.native_minhash(rows, config=CONFIG):
        sigs = _table_signatures(rows)
        for r in range(LSH_BENCH_ROUNDS):
            for arm in ("reference", "qd_prep") if r % 2 == 0 else ("qd_prep", "reference"):
                search = candidate_pairs if arm == "reference" else split_module.candidate_pairs
                started = time.perf_counter()
                answers[arm] = search(sigs, config=BANDS, max_pairs=bound)
                times[arm].append(time.perf_counter() - started)
            assert answers["qd_prep"] == answers["reference"]
            assert answers["reference"][0] and not answers["reference"][1], "a real search"
    print(json.dumps({
        "benchmark": "lsh_candidate_pairs", "keys": len(sigs), "bands": BANDS.bands,
        "rows": BANDS.rows, "pairs": len(answers["reference"][0]), "rounds": LSH_BENCH_ROUNDS,
        "reference_s": [round(x, 3) for x in times["reference"]],
        "qd_prep_s": [round(x, 3) for x in times["qd_prep"]],
        "reference_min_s": round(min(times["reference"]), 3),
        "qd_prep_min_s": round(min(times["qd_prep"]), 3),
        "speedup_min_over_min": round(min(times["reference"]) / min(times["qd_prep"]), 1),
    }))
    assert min(times["qd_prep"]) < min(times["reference"])
