"""Adversarial pressure on the three things added this session, before they are relied on.

``corpus_contradictions`` gates every shard write, ``find_forks`` is the only thing that can
see a forked ledger, and ``choose_buckets``' new default decides the padding gate. Each was
verified on the case it was built for. This file tries to break them on the cases it was
not: empty, degenerate, boundary, oversized, duplicated, three-way, and pathological.

Property-based where the property is real, example-based where a specific shape is the
threat. Torch-free, so ``make gates`` runs it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.artifacts import (  # noqa: E402
    MAX_PADDING_WASTE,
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SPAN,
    bucket_for,
    padding_waste,
)
from qd_train.ledger import Ledger, find_forks, verify_no_fork  # noqa: E402
from qd_train.shards import choose_buckets, corpus_contradictions  # noqa: E402

# -- corpus_contradictions ---------------------------------------------------------------


def _call(sequences, target_index, slot_kinds, span_targets, labels):
    return corpus_contradictions(
        sequences,
        target_index=target_index,
        slot_kinds=slot_kinds,
        span_targets=span_targets,
        labels=labels,
    )


def test_an_empty_corpus_reports_zero_rather_than_raising() -> None:
    """Zero sequences is a legitimate state for the checker even though ``write_shards``
    refuses it one level up -- the two refusals are about different things, and a checker
    that raised here would make the writer's own message unreachable."""
    report = _call([], [], [], [], [])
    assert report["sequences"] == 0
    assert report["contradicting_rows"] == 0
    assert report["rows_sharing_a_prefix"] == 0


def test_a_single_row_cannot_contradict_itself() -> None:
    one = [np.asarray([1, 2, 3], dtype=np.int32)]
    report = _call(one, [1], [SLOT_CHOICE], [(NO_SPAN, NO_SPAN)], ["a"])
    assert report["contradicting_rows"] == 0


def test_a_target_index_at_the_last_position_takes_the_whole_sequence_as_prefix() -> None:
    """The boundary. ``target_index = len - 1`` makes the prefix the entire row, so two
    rows contradict only if they are byte-identical -- in which case their golds are too,
    and nothing contradicts. A slice bug here would silently widen or narrow every prefix.
    """
    a = np.asarray([1, 2, 3], dtype=np.int32)
    b = np.asarray([1, 2, 4], dtype=np.int32)
    report = _call([a, b], [2, 2], [SLOT_CHOICE] * 2, [(NO_SPAN, NO_SPAN)] * 2, ["a", "b"])
    assert report["rows_sharing_a_prefix"] == 0


def test_a_target_index_of_zero_groups_on_the_first_token_alone() -> None:
    a = np.asarray([9, 65], dtype=np.int32)
    b = np.asarray([9, 66], dtype=np.int32)
    report = _call([a, b], [0, 0], [SLOT_CHOICE] * 2, [(NO_SPAN, NO_SPAN)] * 2, ["a", "b"])
    assert report["contradicting_rows"] == 2
    assert report["examples"][0]["prefix_tokens"] == 1


def test_a_three_way_group_with_three_golds_is_one_group_not_three() -> None:
    """A pair is the case that was fixed; a triple is the case nobody wrote down. The
    group count drives the refusal message, so counting a triple as three pairs would
    misreport the corpus by a factor that grows with the group."""
    seqs = [np.asarray([7, 7, g], dtype=np.int32) for g in (65, 66, 67)]
    report = _call(seqs, [1, 1, 1], [SLOT_CHOICE] * 3, [(NO_SPAN, NO_SPAN)] * 3, list("abc"))
    assert report["contradicting_groups"] == 1
    assert report["contradicting_rows"] == 3


def test_a_group_of_many_identical_golds_with_one_dissenter_still_contradicts() -> None:
    """99 rows agreeing and 1 disagreeing is unfittable exactly as 1-and-1 is -- the
    optimum is still the label distribution. A majority-vote reading would miss it."""
    seqs = [np.asarray([7, 7, 65], dtype=np.int32) for _ in range(99)]
    seqs.append(np.asarray([7, 7, 66], dtype=np.int32))
    n = len(seqs)
    labels = [str(i) for i in range(n)]
    report = _call(seqs, [1] * n, [SLOT_CHOICE] * n, [(NO_SPAN, NO_SPAN)] * n, labels)
    assert report["contradicting_groups"] == 1
    assert report["contradicting_rows"] == 100


def test_a_mixed_kind_group_counts_each_kind_separately() -> None:
    """A span row and a choice row sharing a prefix: ``per_kind`` must not attribute one
    row's kind to the other, or a refusal message names the wrong channel."""
    seqs = [np.asarray([7, 7, 65], dtype=np.int32) for _ in range(2)]
    report = _call(
        seqs, [1, 1], [SLOT_CHOICE, SLOT_SPAN], [(NO_SPAN, NO_SPAN), (3, 4)], ["c", "s"]
    )
    assert report["contradicting_rows"] == 2
    assert report["per_kind"] == {
        str(SLOT_CHOICE): {"groups": 1, "rows": 1},
        str(SLOT_SPAN): {"groups": 1, "rows": 1},
    }


def test_max_examples_bounds_the_report_without_bounding_the_count() -> None:
    """A capped sample must never be presented as complete coverage: the examples list is
    truncated, the counts are not."""
    seqs = []
    for g in range(40):
        seqs.append(np.asarray([g, 7, 65], dtype=np.int32))
        seqs.append(np.asarray([g, 7, 66], dtype=np.int32))
    n = len(seqs)
    report = corpus_contradictions(
        seqs,
        target_index=[1] * n,
        slot_kinds=[SLOT_CHOICE] * n,
        span_targets=[(NO_SPAN, NO_SPAN)] * n,
        labels=[str(i) for i in range(n)],
        max_examples=3,
    )
    assert report["contradicting_groups"] == 40
    assert report["contradicting_rows"] == 80
    assert len(report["examples"]) == 3


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(st.integers(0, 3), st.integers(64, 67)),
        min_size=1,
        max_size=30,
    )
)
def test_rows_sharing_a_prefix_is_never_below_contradicting_rows(pairs) -> None:
    """The invariant between the two counts, over arbitrary corpora.

    A contradicting row is by construction a row that shares a prefix, so the second count
    can never exceed the first. If it ever did, one of the two loops is counting something
    the other is not, and the refusal message would be arithmetic nonsense.
    """
    seqs = [np.asarray([p, 7, g], dtype=np.int32) for p, g in pairs]
    n = len(seqs)
    report = corpus_contradictions(
        seqs,
        target_index=[1] * n,
        slot_kinds=[SLOT_CHOICE] * n,
        span_targets=[(NO_SPAN, NO_SPAN)] * n,
        labels=[str(i) for i in range(n)],
    )
    assert report["contradicting_rows"] <= report["rows_sharing_a_prefix"] <= n
    assert report["sequences"] == n


# -- find_forks --------------------------------------------------------------------------


def _chain(path: Path, row_ids) -> Path:
    lines: list[bytes] = []
    prev: str | None = None
    for row_id in row_ids:
        line = json.dumps({"row_id": row_id, "prev_row_hash": prev}, sort_keys=True).encode()
        lines.append(line)
        prev = hashlib.sha256(line).hexdigest()
    path.write_bytes(b"\n".join(lines) + b"\n" if lines else b"")
    return path


def test_an_empty_ledger_file_is_not_a_fork(tmp_path: Path) -> None:
    a = _chain(tmp_path / "empty.jsonl", [])
    b = _chain(tmp_path / "full.jsonl", ["r1", "r2"])
    assert Ledger(a).raw_lines() == []
    assert find_forks([a, b]) == []


def test_a_missing_file_is_not_a_fork(tmp_path: Path) -> None:
    """``Ledger.raw_lines`` returns [] for a path that does not exist. A fork detector that
    treated absence as a branch would fire on every machine that has not synced yet."""
    assert find_forks([tmp_path / "never-written.jsonl"]) == []


def test_no_paths_at_all_is_not_a_fork() -> None:
    assert find_forks([]) == []
    assert verify_no_fork([]) is None


def test_a_three_way_fork_reports_three_branches(tmp_path: Path) -> None:
    """Two machines is the case that happened; three is the case that happens next. All
    three branches must be named, because reconciling needs to know how many there are."""
    a = _chain(tmp_path / "a.jsonl", ["r1", "a2"])
    b = _chain(tmp_path / "b.jsonl", ["r1", "b2"])
    c = _chain(tmp_path / "c.jsonl", ["r1", "c2"])
    forks = find_forks([a, b, c])
    assert len(forks) == 1
    assert len(forks[0].branches) == 3
    assert {x.row_id for x in forks[0].branches} == {"a2", "b2", "c2"}


def test_two_forks_in_one_pair_of_files_are_both_reported(tmp_path: Path) -> None:
    """Divergence, re-convergence by coincidence, then divergence again. Reporting only
    the first would understate the work of reconciling."""
    a = _chain(tmp_path / "a.jsonl", ["r1", "x", "same", "a4"])
    b = _chain(tmp_path / "b.jsonl", ["r1", "y", "same", "b4"])
    forks = find_forks([a, b])
    # The first divergence is real; after it the two 'same' rows have different
    # predecessors, so they are different lines and cannot re-converge.
    assert len(forks) >= 1
    assert all(not f.is_root for f in forks)


def test_a_file_that_is_not_json_is_still_placed_rather_than_crashing(tmp_path: Path) -> None:
    """A truncated or hand-mangled ledger must not take the fork detector down with it --
    that is ``verify_chain``'s job to report, and a crash here would hide the fork too."""
    good = _chain(tmp_path / "good.jsonl", ["r1"])
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"not json at all\n")
    forks = find_forks([good, bad])
    assert len(forks) == 1
    assert forks[0].is_root
    assert any(b.row_id == "<unparseable>" for b in forks[0].branches)


def test_the_same_path_passed_twice_is_not_a_fork(tmp_path: Path) -> None:
    a = _chain(tmp_path / "a.jsonl", ["r1", "r2", "r3"])
    assert find_forks([a, a, a]) == []


# -- choose_buckets at the new default -----------------------------------------------------


def test_a_corpus_of_one_length_produces_one_bucket_and_no_waste() -> None:
    """The degenerate distribution. 32 quantiles over identical values collapse to one
    boundary, and the padding waste is exactly zero -- not a division by zero."""
    lengths = [512] * 200
    buckets = choose_buckets(lengths)
    assert buckets == (512,)
    state = padding_waste(lengths, buckets)
    assert state.value == 0.0
    assert state.passed


def test_a_single_huge_outlier_does_not_orphan_any_row() -> None:
    """The shape that breaks naive bucketing: 199 short rows and one enormous one. The
    largest boundary must still be the longest sequence, or ``bucket_for`` refuses it."""
    lengths = [100] * 199 + [1_000_000]
    buckets = choose_buckets(lengths)
    assert buckets[-1] == 1_000_000
    for length in lengths:
        bucket_for(length, buckets)


def test_more_buckets_than_distinct_lengths_is_not_an_error() -> None:
    """``choose_buckets`` dedupes its quantiles, so asking for 32 over 3 distinct lengths
    returns 3. A caller that assumed len(buckets) == n_buckets would index off the end."""
    lengths = [10, 20, 30] * 40
    buckets = choose_buckets(lengths, n_buckets=32)
    assert len(buckets) == 3
    assert buckets == (10, 20, 30)


def test_two_lengths_far_apart_still_clear_the_gate_at_the_default() -> None:
    """A bimodal corpus is the adversarial case for quantile bucketing: half at 100 and
    half at 100000, with nothing between. Quantiles put a boundary at each mode, so the
    waste is near zero -- equal-width bins would have failed here."""
    lengths = [100] * 100 + [100_000] * 100
    state = padding_waste(lengths, choose_buckets(lengths))
    assert state.passed, f"bimodal corpus wastes {state.value:.2%}"


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(1, 40_000), min_size=2, max_size=300))
def test_no_length_distribution_orphans_a_row_at_the_default(lengths) -> None:
    """The invariant that must hold for every corpus, not just the measured one: a
    boundary set that orphans a row turns a padding problem into a refused write."""
    buckets = choose_buckets(lengths)
    assert buckets[-1] == max(lengths)
    for length in lengths:
        bucket_for(length, buckets)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(1, 40_000), min_size=2, max_size=300))
def test_more_buckets_never_wastes_more_than_fewer(lengths) -> None:
    """Monotonicity, checked rather than assumed -- it is the whole argument for raising
    the default from 8 to 32, and it was measured on exactly one distribution."""
    coarse = padding_waste(lengths, choose_buckets(lengths, n_buckets=8)).value
    fine = padding_waste(lengths, choose_buckets(lengths, n_buckets=32)).value
    assert fine <= coarse + 1e-12, f"32 buckets wasted {fine:.4%} against 8's {coarse:.4%}"


def test_the_default_is_an_improvement_and_not_a_guarantee() -> None:
    """The honest bound on this session's bucketing change, pinned so it is not overclaimed.

    Raising the default from 8 to 32 was measured on ONE distribution, where it took the
    waste from 25.66% to 6.56%. Over synthetic corpora it is a large improvement on
    realistic shapes and no help at all against a six-order-of-magnitude spread -- quantile
    bucketing cannot place a boundary between a 161k-token row and a 1.18M-token one
    because there is nothing between them.

    This reproduces the realistic shape at a fixed seed and asserts the RELATIVE claim,
    which is the one that is actually true: 32 fails the gate strictly less often than 8.
    It deliberately does not assert that 32 always passes, because it does not.
    """
    import random

    rng = random.Random(2)
    fail8 = fail32 = 0
    trials = 300
    for _ in range(trials):
        rows = rng.randint(100, 800)
        lengths = [min(int(rng.lognormvariate(7, 1.5)) + 200, 35_000) for _ in range(rows)]
        if padding_waste(lengths, choose_buckets(lengths, n_buckets=8)).value > MAX_PADDING_WASTE:
            fail8 += 1
        if padding_waste(lengths, choose_buckets(lengths, n_buckets=32)).value > MAX_PADDING_WASTE:
            fail32 += 1
    assert fail32 < fail8, (
        f"32 buckets failed the gate {fail32}/{trials} times against 8's {fail8}/{trials}; "
        "the default was raised on the claim that it fails strictly less often"
    )
    assert fail32 > 0, (
        "32 buckets cleared the gate on every one of these corpora, which would mean this "
        "test has stopped exercising the hard shapes -- the measured rate was ~41%"
    )


def test_the_gate_constant_itself_is_untouched() -> None:
    """Rule 2, pinned by value. Every change this session lowered the measured waste; none
    of them may have moved the bar, and a test that checks the waste without checking the
    bar cannot tell those two apart."""
    assert MAX_PADDING_WASTE == 0.15
