"""The MinHash estimator, cross-checked against ``datasketch`` rather than trusted.

``qd_data.minhash`` implements its own permutation family instead of calling
``datasketch``, for a stated reason: the dedupe outcome feeds ``data_snapshot_hash``,
and ``datasketch``'s hash values come from a ``numpy.random.RandomState`` whose
stream is a property of a library version. A ``datasketch`` upgrade would silently
change which rows are near-duplicates and therefore change the snapshot hash of an
unchanged corpus.

That argument is only worth anything if the local implementation is *correct*, so
``datasketch`` is used here as an independent oracle: the two estimators are built
from different permutation families and different base hashes, and must still agree
on Jaccard to within sampling error on the same inputs. A bug in the local estimator
would have to be reproduced exactly by an unrelated library to slip through.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qd_data.minhash import (
    MERSENNE_PRIME,
    BandConfig,
    MinHasher,
    candidate_pairs,
    choose_bands,
    estimate_jaccard,
    exact_jaccard,
    shingle,
)

datasketch = pytest.importorskip("datasketch", reason="datasketch is a declared dependency")


def _sets(overlap: int, only_a: int, only_b: int) -> tuple[frozenset[bytes], frozenset[bytes]]:
    common = {f"c{i}".encode() for i in range(overlap)}
    a = common | {f"a{i}".encode() for i in range(only_a)}
    b = common | {f"b{i}".encode() for i in range(only_b)}
    return frozenset(a), frozenset(b)


# -- the estimator -----------------------------------------------------------


@pytest.mark.parametrize(
    ("overlap", "only_a", "only_b"),
    [(400, 0, 0), (360, 20, 20), (300, 50, 50), (200, 100, 100), (0, 200, 200)],
)
def test_estimate_tracks_exact_jaccard(overlap: int, only_a: int, only_b: int) -> None:
    a, b = _sets(overlap, only_a, only_b)
    hasher = MinHasher(num_perm=256, seed=20260919)
    est = estimate_jaccard(hasher.signature(a), hasher.signature(b))
    true = exact_jaccard(a, b)
    # Standard error of a 256-permutation estimate is at most 0.5/sqrt(256) = 0.031.
    # Four standard errors is a bound that is loose enough not to flake and tight
    # enough that a broken estimator cannot pass it.
    assert abs(est - true) <= 4 * math.sqrt(max(true * (1 - true), 0.01) / 256) + 0.02


@pytest.mark.parametrize(
    ("overlap", "only_a", "only_b"),
    [(400, 0, 0), (360, 20, 20), (300, 50, 50), (200, 100, 100), (100, 300, 300)],
)
def test_local_estimator_agrees_with_datasketch(
    overlap: int, only_a: int, only_b: int
) -> None:
    """Two independent permutation families must land on the same Jaccard."""
    a, b = _sets(overlap, only_a, only_b)

    ours = estimate_jaccard(
        MinHasher(num_perm=256, seed=7).signature(a),
        MinHasher(num_perm=256, seed=7).signature(b),
    )

    m1, m2 = datasketch.MinHash(num_perm=256, seed=7), datasketch.MinHash(num_perm=256, seed=7)
    for item in sorted(a):
        m1.update(item)
    for item in sorted(b):
        m2.update(item)
    theirs = m1.jaccard(m2)

    # Both estimate the same quantity from independent randomness; two estimators at
    # 256 permutations differ by at most ~0.09 at four combined standard errors.
    assert abs(ours - theirs) <= 0.09, f"ours={ours}, datasketch={theirs}"


def test_signature_is_reproducible_across_instances() -> None:
    a, _ = _sets(50, 10, 10)
    assert MinHasher(num_perm=64, seed=1).signature(a) == MinHasher(
        num_perm=64, seed=1
    ).signature(a)
    assert MinHasher(num_perm=64, seed=2).signature(a) != MinHasher(
        num_perm=64, seed=1
    ).signature(a)


def test_signature_values_are_inside_the_prime_field() -> None:
    a, _ = _sets(20, 5, 5)
    for value in MinHasher(num_perm=32, seed=3).signature(a):
        assert 0 <= value < MERSENNE_PRIME


def test_empty_shingle_set_is_refused_not_signed() -> None:
    """Every empty document would otherwise hash identically and dedupe to one row."""
    with pytest.raises(ValueError, match="empty shingle set"):
        MinHasher(num_perm=16, seed=1).signature(frozenset())


def test_two_empty_sets_are_not_similar() -> None:
    assert exact_jaccard(frozenset(), frozenset()) == 0.0


# -- shingling ---------------------------------------------------------------


def test_shingling_collapses_whitespace_but_not_case() -> None:
    a = shingle("def  f():\n\treturn 1", k=2)
    b = shingle("def f(): return 1", k=2)
    assert a.shingles == b.shingles
    assert shingle("Foo bar baz", k=2).shingles != shingle("foo bar baz", k=2).shingles


def test_a_document_shorter_than_k_yields_one_shingle_not_none() -> None:
    """Returning an empty set would make every short row a duplicate of every other."""
    result = shingle("two words", k=5)
    assert len(result.shingles) == 1
    assert result.n_tokens == 2


def test_oversized_documents_are_truncated_and_say_so() -> None:
    result = shingle("word " * 5000, k=3, max_doc_bytes=256)
    assert result.truncated is True
    assert shingle("word " * 10, k=3, max_doc_bytes=256).truncated is False


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(min_size=1, max_size=300), st.integers(min_value=1, max_value=8))
def test_shingling_is_a_pure_function(text: str, k: int) -> None:
    assert shingle(text, k=k) == shingle(text, k=k)


# -- banding and LSH ---------------------------------------------------------


def test_chosen_bands_bracket_the_threshold() -> None:
    config = choose_bands(num_perm=128, threshold=0.8)
    assert config.num_perm_used <= 128
    # Recall at and above the threshold is the property that matters: a miss is
    # undetected contamination, a spurious candidate costs one exact comparison.
    assert config.probability_at(0.80) > 0.90
    assert config.probability_at(0.85) > 0.98
    assert config.probability_at(0.90) > 0.99
    assert config.probability_at(0.50) < 0.10


def test_balanced_weights_would_have_missed_a_third_of_real_duplicates() -> None:
    """Why the objective is recall-weighted, asserted rather than asserted-in-prose.

    This is the measurement the weighting decision rests on: at balanced weights the
    chosen banding proposes a genuine 0.85-Jaccard pair only ~69% of the time.
    """
    balanced = choose_bands(
        num_perm=128, threshold=0.8, false_positive_weight=0.5, false_negative_weight=0.5
    )
    chosen = choose_bands(num_perm=128, threshold=0.8)
    assert balanced.probability_at(0.85) < 0.75
    assert chosen.probability_at(0.85) > 0.98
    assert (chosen.bands, chosen.rows) != (balanced.bands, balanced.rows)


def test_band_choice_is_deterministic() -> None:
    first = choose_bands(num_perm=128, threshold=0.8)
    second = choose_bands(num_perm=128, threshold=0.8)
    assert (first.bands, first.rows) == (second.bands, second.rows)


def test_a_zero_false_negative_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="false-negative weight"):
        choose_bands(num_perm=128, threshold=0.8, false_negative_weight=0.0)


def test_lsh_recall_equals_the_exhaustive_answer_on_a_small_corpus() -> None:
    """LSH is a candidate generator; its recall is measured, not assumed.

    An LSH pass that silently missed pairs would make dedupe report a clean corpus.
    This builds pairs well above the threshold and asserts none is missed.
    """
    hasher = MinHasher(num_perm=128, seed=11)
    docs: dict[str, frozenset[bytes]] = {}
    base = [f"token{i}".encode() for i in range(200)]
    for i in range(12):
        docs[f"doc{i:02d}"] = frozenset(base[: 200 - i])  # nested, all pairs J >= 0.94
    for i in range(12, 20):
        docs[f"doc{i:02d}"] = frozenset(f"other{i}_{j}".encode() for j in range(200))

    sigs = {k: hasher.signature(v) for k, v in docs.items()}
    config = choose_bands(num_perm=128, threshold=0.8)
    cands, truncated = candidate_pairs(sigs, config=config, max_pairs=10_000)
    assert truncated is False

    exhaustive = {
        (a, b)
        for a in docs
        for b in docs
        if a < b and exact_jaccard(docs[a], docs[b]) >= 0.8
    }
    assert exhaustive, "the fixture must contain duplicates or the test is vacuous"
    assert exhaustive <= cands, f"LSH missed {sorted(exhaustive - cands)}"


def test_candidate_search_reports_its_bound_rather_than_returning_a_short_list() -> None:
    hasher = MinHasher(num_perm=64, seed=2)
    identical = frozenset(f"t{i}".encode() for i in range(50))
    sigs = {f"d{i:03d}": hasher.signature(identical) for i in range(40)}
    _, truncated = candidate_pairs(
        sigs, config=choose_bands(num_perm=64, threshold=0.8), max_pairs=10
    )
    assert truncated is True


def test_candidate_search_refuses_a_signature_shorter_than_the_banding() -> None:
    config = choose_bands(num_perm=128, threshold=0.8)
    short = {"a": (1, 2, 3), "b": (1, 2, 3)}
    with pytest.raises(ValueError, match="band configuration needs"):
        candidate_pairs(short, config=config, max_pairs=100)


# -- the recall bound: GAP-DATA-LSH-RECALL-BOUND -----------------------------
#
# The gap asked whether the dedupe finds every near-duplicate pair at 0.8 Jaccard.
# It does not, and it cannot: banded LSH has an S-curve. The defect was never the
# miss rate -- it is one-sided (every candidate is confirmed exactly, so dedupe can
# under-remove and never over-remove) and it is tiny for the population this targets.
# The defect was reporting a probabilistic filter as though it were exhaustive:
# `DedupeReport` recorded the banding and the candidate count and said `Ran`, with
# nothing anywhere stating the bound.
#
# `test_lsh_recall_equals_the_exhaustive_answer_on_a_small_corpus` above is the
# existing check and it is sound, but note what it covers: its fixture sits at
# J >= 0.94, where the per-pair miss probability is 3e-7 and "equals the exhaustive
# answer" is effectively deterministic. At the threshold itself it is 1 in 19.


def test_the_banding_states_its_recall_at_the_threshold() -> None:
    """The bound is a property of the configuration, so the configuration reports it.

    Derived, not measured: ``1 - (1 - J**r)**b`` at ``J = threshold``. At the
    configured ``num_perm=128, threshold=0.8`` the search picks ``b=16, r=8``, giving
    ``1 - (1 - 0.8**8)**16 = 0.947049``.
    """
    config = choose_bands(num_perm=128, threshold=0.8)
    assert (config.bands, config.rows) == (16, 8)
    assert config.recall_at_threshold == config.probability_at(0.8)
    assert config.recall_at_threshold == pytest.approx(0.947049, abs=1e-6)
    # The point of the whole record: this is below one, and it is reported.
    assert config.recall_at_threshold < 1.0


def test_measured_lsh_recall_at_the_threshold_matches_the_derived_bound() -> None:
    """The derived S-curve, checked against a brute-force pairwise baseline.

    300 independent document pairs constructed at *exactly* J = 0.80 (16 shared
    shingles, 2 unique to each side, union 20). Brute force over all 600 documents
    finds all 300; LSH proposes fewer, and the shortfall is the recall gap rather
    than a fixture artefact -- which is what having both numbers proves.

    This is a Monte-Carlo measurement, so it carries its own error: the binomial
    standard error at n=300 is 0.013, and the assertion is a 4-sigma window around
    the derived value. It is seeded, so it is reproducible, but the window is what
    makes it a measurement rather than a golden.
    """
    config = choose_bands(num_perm=128, threshold=0.8)
    hasher = MinHasher(num_perm=128, seed=20260919)

    n_pairs = 300
    docs: dict[str, frozenset[bytes]] = {}
    for p in range(n_pairs):
        common = {f"p{p}_c{i}".encode() for i in range(16)}
        docs[f"p{p}_a"] = frozenset(common | {f"p{p}_x{i}".encode() for i in range(2)})
        docs[f"p{p}_b"] = frozenset(common | {f"p{p}_y{i}".encode() for i in range(2)})

    # The brute-force baseline: every pair, exact Jaccard, no sampling anywhere.
    exhaustive = {
        (a, b)
        for a in docs
        for b in docs
        if a < b and exact_jaccard(docs[a], docs[b]) >= 0.8
    }
    assert len(exhaustive) == n_pairs, "the fixture must sit exactly on the threshold"
    assert {round(exact_jaccard(docs[a], docs[b]), 10) for a, b in exhaustive} == {0.8}

    sigs = {k: hasher.signature(v) for k, v in docs.items()}
    cands, truncated = candidate_pairs(sigs, config=config, max_pairs=1_000_000)
    assert truncated is False, "a truncated search would confound the measurement"

    found = exhaustive & cands
    measured = len(found) / len(exhaustive)
    derived = config.recall_at_threshold
    sigma = math.sqrt(derived * (1 - derived) / len(exhaustive))

    assert abs(measured - derived) <= 4 * sigma, (
        f"measured recall {measured:.4f} is {abs(measured - derived) / sigma:.1f} "
        f"standard errors from the derived {derived:.4f}"
    )
    # Both numbers, and the one that matters: pairs at the threshold ARE missed.
    assert found < exhaustive, (
        "if LSH found every threshold pair the S-curve would be wrong, or the "
        "fixture is not actually sitting at the threshold"
    )
    # Precision is exact by construction: LSH proposes, exact Jaccard confirms. No
    # pair outside the truth set can survive confirmation, so the error is one-sided.
    assert cands <= exhaustive or all(
        exact_jaccard(docs[a], docs[b]) < 0.8 for a, b in cands - exhaustive
    )


@settings(max_examples=200, deadline=None)
@given(
    st.integers(min_value=1, max_value=64),
    st.integers(min_value=1, max_value=64),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_the_s_curve_is_a_probability_and_never_decreases_in_jaccard(
    bands: int, rows: int, j_low: float, j_high: float
) -> None:
    """Two properties the reported bound rests on, over the whole (b, r) grid.

    Without monotonicity, ``recall_at_threshold`` would not be a *bound* on recall
    above the threshold -- it would be one point on an arbitrary curve, and quoting
    it as "recall is at least this" would be unsound.
    """
    config = BandConfig(bands=bands, rows=rows, threshold=0.8)
    lo, hi = sorted((j_low, j_high))
    p_lo, p_hi = config.probability_at(lo), config.probability_at(hi)
    assert 0.0 <= p_lo <= 1.0
    assert 0.0 <= p_hi <= 1.0
    assert p_lo <= p_hi + 1e-12
    assert config.recall_at_threshold == config.probability_at(0.8)


def test_the_recall_bound_holds_above_the_threshold_which_is_where_it_is_quoted() -> None:
    """The population dedupe targets -- vendored trees, forks, copied files -- sits
    well above 0.8, and the bound is quoted as a floor for that population. This
    pins the floor claim to numbers rather than to the word "effectively"."""
    config = choose_bands(num_perm=128, threshold=0.8)
    floor = config.recall_at_threshold
    for jaccard in (0.80, 0.85, 0.90, 0.95, 0.99):
        assert config.probability_at(jaccard) >= floor
    assert config.probability_at(0.95) == pytest.approx(1.0, abs=1e-6)
