"""S2 tests. The torch-free half runs in the repo venv; the torch half skips without torch.

Nothing here re-implements the artifact contract: `RemapTable` and its invariants are tested
in `test_artifacts.py`, and these tests exercise the *policy* — which tokens `build_remap`
keeps, what it refuses, how the weights are sliced, and what the parity gate actually
measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from qd_train.artifacts import DROPPED, RemapTable, ShardContractViolation, TokenNotInRemap
from qd_train.remap import (
    LOGIT_EXACTNESS_TOL,
    LOSS_PARITY_TOL,
    CorpusCounts,
    RemapCoverageError,
    build_remap,
    count_corpus_tokens,
    remap_parity_report,
    verify_remap_parity,
)
from qd_train.tristate import NotRun

V_OLD = 64
TOK_HASH = "t" * 64


def _counts(pairs: dict[int, int], *, size: int = V_OLD) -> np.ndarray:
    dense = np.zeros(size, dtype=np.int64)
    for k, v in pairs.items():
        dense[k] = v
    return dense


# --- counting ---------------------------------------------------------------------------


def test_counting_reports_distinct_and_total():
    c = count_corpus_tokens(
        [np.array([1, 1, 2]), np.array([2, 3])],
        source_vocab_size=V_OLD,
        max_sequences=10,
    )
    assert c.n_sequences == 2
    assert c.n_tokens == 5
    assert c.n_distinct == 3
    assert c.counts[1] == 2 and c.counts[2] == 2 and c.counts[3] == 1
    assert c.source_vocab_size == V_OLD


def test_counting_an_empty_corpus_is_zero_not_an_error():
    c = count_corpus_tokens([], source_vocab_size=V_OLD, max_sequences=10)
    assert c.n_sequences == 0 and c.n_tokens == 0 and c.n_distinct == 0


def test_counting_skips_nothing_for_an_empty_sequence():
    c = count_corpus_tokens(
        [np.array([], dtype=np.int64), np.array([5])],
        source_vocab_size=V_OLD,
        max_sequences=10,
    )
    assert c.n_sequences == 2 and c.n_tokens == 1 and c.n_distinct == 1


def test_counting_refuses_to_truncate_rather_than_under_report():
    """A partial count under-reports the used set; the resulting remap drops what it never saw."""
    with pytest.raises(RemapCoverageError, match="partial count"):
        count_corpus_tokens(
            [np.array([1]), np.array([2]), np.array([3])],
            source_vocab_size=V_OLD,
            max_sequences=2,
        )


def test_counting_refuses_an_id_outside_the_source_vocabulary():
    with pytest.raises(RemapCoverageError, match="different tokenizer"):
        count_corpus_tokens([np.array([V_OLD])], source_vocab_size=V_OLD, max_sequences=10)
    with pytest.raises(RemapCoverageError, match="different tokenizer"):
        count_corpus_tokens([np.array([-1])], source_vocab_size=V_OLD, max_sequences=10)


def test_counting_refuses_non_integer_ids():
    with pytest.raises(RemapCoverageError, match="must be integers"):
        count_corpus_tokens([np.array([1.5])], source_vocab_size=V_OLD, max_sequences=10)


def test_counting_bounds_must_be_positive():
    with pytest.raises(RemapCoverageError, match="max_sequences must be positive"):
        count_corpus_tokens([], source_vocab_size=V_OLD, max_sequences=0)
    with pytest.raises(RemapCoverageError, match="source_vocab_size must be positive"):
        count_corpus_tokens([], source_vocab_size=0, max_sequences=1)


def test_corpus_counts_refuses_a_wrong_dtype():
    with pytest.raises(RemapCoverageError, match="int64"):
        CorpusCounts(counts=np.zeros(4, dtype=np.int32), n_sequences=0, n_tokens=0)


# --- the coverage policy ------------------------------------------------------------------


def test_every_used_token_survives_including_the_rarest():
    """The whole policy in one assertion: count 1 is kept exactly as hard as count 10^6."""
    counts = _counts({3: 1_000_000, 7: 1, 40: 1})
    r = build_remap(
        counts=counts,
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0, 1),
        target_vocab_size=None,
    )
    assert list(r.new_to_old) == [0, 1, 3, 7, 40]
    # And the corpus round-trips, which is the property the policy exists to guarantee.
    assert list(r.encode(np.array([3, 7, 40]))) == [2, 3, 4]


def test_unused_tokens_are_dropped():
    r = build_remap(
        counts=_counts({5: 2}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0,),
        target_vocab_size=None,
    )
    assert r.vocab_size == 2
    assert r.old_to_new[6] == DROPPED
    with pytest.raises(TokenNotInRemap):
        r.encode(np.array([6]))


def test_kept_ids_are_renumbered_in_ascending_old_id_order():
    """Not frequency order: remap_hash is over new_to_old, so ties would make it count-dependent."""
    ascending = build_remap(
        counts=_counts({9: 1, 2: 500}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(),
        target_vocab_size=None,
    )
    assert list(ascending.new_to_old) == [2, 9]

    # Same used set, wildly different counts -> byte-identical table and identical hash.
    swapped = build_remap(
        counts=_counts({9: 500, 2: 1}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(),
        target_vocab_size=None,
    )
    assert list(swapped.new_to_old) == [2, 9]
    assert swapped.remap_hash() == ascending.remap_hash()


def test_specials_are_kept_even_when_the_corpus_never_uses_them():
    r = build_remap(
        counts=_counts({30: 5}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0, 1, 2),
        target_vocab_size=None,
    )
    assert list(r.new_to_old) == [0, 1, 2, 30]


def test_duplicate_specials_are_deduped_not_counted_twice():
    r = build_remap(
        counts=_counts({30: 5}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(1, 1, 0, 1),
        target_vocab_size=None,
    )
    assert list(r.new_to_old) == [0, 1, 30]
    assert r.special_ids == (0, 1)


def test_a_special_that_is_also_used_is_counted_once():
    r = build_remap(
        counts=_counts({0: 9, 5: 1}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0,),
        target_vocab_size=None,
    )
    assert list(r.new_to_old) == [0, 5]


# --- the refusal ---------------------------------------------------------------------------


def test_a_corpus_larger_than_the_budget_is_refused_not_truncated():
    counts = _counts(dict.fromkeys(range(10), 1))
    with pytest.raises(RemapCoverageError) as exc:
        build_remap(
            counts=counts,
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(0,),
            target_vocab_size=4,
        )
    msg = str(exc.value)
    assert "10 distinct" in msg and "exceeds the ceiling of 4" in msg and "by 6" in msg
    assert "refusal, not a truncation" in msg


def test_specials_count_against_the_budget():
    """The refusal must name the specials too, or the arithmetic in the message is wrong."""
    with pytest.raises(RemapCoverageError, match=r"2 further special id\(s\)"):
        build_remap(
            counts=_counts({10: 1, 11: 1}),
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(0, 1),
            target_vocab_size=3,
        )


def test_the_budget_is_a_ceiling_not_a_target():
    r = build_remap(
        counts=_counts({10: 1, 11: 1}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(),
        target_vocab_size=50,
    )
    assert r.vocab_size == 2, "filling to the ceiling would cost rows in every training step"


def test_exactly_at_the_budget_is_allowed():
    r = build_remap(
        counts=_counts({10: 1, 11: 1}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0,),
        target_vocab_size=3,
    )
    assert r.vocab_size == 3


def test_a_special_outside_the_vocabulary_is_refused():
    with pytest.raises(RemapCoverageError, match="tokenizer mismatch"):
        build_remap(
            counts=_counts({1: 1}),
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(V_OLD,),
            target_vocab_size=None,
        )


def test_degenerate_sizes_are_refused():
    with pytest.raises(RemapCoverageError, match="source_vocab_size must be positive"):
        build_remap(
            counts={},
            source_vocab_size=0,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )
    with pytest.raises(RemapCoverageError, match="target_vocab_size must be positive"):
        build_remap(
            counts=_counts({1: 1}),
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=0,
        )


def test_a_remap_over_nothing_is_refused_by_the_contract():
    """Left to `RemapTable`, deliberately: the contract owns 'a remap that keeps no tokens'."""
    with pytest.raises(ShardContractViolation, match="keeps no tokens"):
        build_remap(
            counts=_counts({}),
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )


def test_the_contract_still_refuses_a_missing_tokenizer_hash():
    with pytest.raises(ShardContractViolation, match="tokenizer_hash"):
        build_remap(
            counts=_counts({1: 1}),
            source_vocab_size=V_OLD,
            tokenizer_hash="",
            special_ids=(),
            target_vocab_size=None,
        )


# --- count input shapes --------------------------------------------------------------------


def test_counts_accept_a_mapping_a_dense_array_and_a_corpus_counts():
    kwargs = dict(
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0,),
        target_vocab_size=None,
    )
    from_map = build_remap(counts={4: 1, 9: 3}, **kwargs)
    from_dense = build_remap(counts=_counts({4: 1, 9: 3}), **kwargs)
    from_counts = build_remap(
        counts=count_corpus_tokens(
            [np.array([4, 9, 9, 9])], source_vocab_size=V_OLD, max_sequences=2
        ),
        **kwargs,
    )
    assert from_map.remap_hash() == from_dense.remap_hash() == from_counts.remap_hash()


def test_a_mapping_key_outside_the_vocabulary_is_refused():
    with pytest.raises(RemapCoverageError, match="outside the source vocabulary"):
        build_remap(
            counts={V_OLD + 1: 1},
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )


def test_a_dense_array_of_the_wrong_length_is_refused():
    with pytest.raises(RemapCoverageError, match="different vocabularies"):
        build_remap(
            counts=np.zeros(V_OLD + 1, dtype=np.int64),
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )


def test_corpus_counts_over_a_different_vocabulary_are_refused():
    other = count_corpus_tokens([np.array([1])], source_vocab_size=V_OLD + 8, max_sequences=2)
    # Raw: the `|` is a deliberate alternation over the two wordings build_remap
    # can refuse with, not a literal pipe in the message.
    with pytest.raises(RemapCoverageError, match=r"different vocab|were taken over"):
        build_remap(
            counts=other,
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )


def test_negative_counts_are_refused():
    dense = _counts({1: 1})
    dense[2] = -1
    with pytest.raises(RemapCoverageError, match="negative"):
        build_remap(
            counts=dense,
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )


def test_a_nonsense_counts_type_is_refused():
    with pytest.raises(RemapCoverageError, match="must be a CorpusCounts"):
        build_remap(
            counts="lots",  # type: ignore[arg-type]
            source_vocab_size=V_OLD,
            tokenizer_hash=TOK_HASH,
            special_ids=(),
            target_vocab_size=None,
        )


# --- the cross-lane property ----------------------------------------------------------------


def test_a_remap_built_over_a_corpus_covers_that_corpus():
    """The S2 <-> S4 handshake: `assert_remap_covers` must not be able to fail on our own corpus."""
    from qd_train.artifacts import assert_remap_covers

    rng = np.random.default_rng(20260919)
    seqs = [rng.integers(0, V_OLD, size=int(n)) for n in rng.integers(2, 30, size=40)]
    counts = count_corpus_tokens(seqs, source_vocab_size=V_OLD, max_sequences=100)
    r = build_remap(
        counts=counts,
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0, 1),
        target_vocab_size=None,
    )
    for s in seqs:
        assert_remap_covers(s, r)


def test_a_remap_built_over_one_corpus_refuses_another():
    counts = count_corpus_tokens(
        [np.array([2, 3, 4])], source_vocab_size=V_OLD, max_sequences=10
    )
    r = build_remap(
        counts=counts,
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0,),
        target_vocab_size=None,
    )
    with pytest.raises(TokenNotInRemap, match="disagree"):
        r.encode(np.array([2, 3, 63]))


# --- the gate, with no torch and no weights ---------------------------------------------------


def _tiny_remap() -> RemapTable:
    return build_remap(
        counts=_counts({2: 1, 3: 1}),
        source_vocab_size=V_OLD,
        tokenizer_hash=TOK_HASH,
        special_ids=(0, 1),
        target_vocab_size=None,
    )


def test_parity_without_weights_is_not_run_never_a_pass():
    r = _tiny_remap()
    out = verify_remap_parity(
        model_before=None, model_after=None, remap=r, sequences=[np.array([0, 2, 3])]
    )
    assert isinstance(out, NotRun)
    assert not hasattr(out, "passed")
    assert "model_before and model_after" in out.reason


def test_parity_with_only_one_model_is_not_run():
    r = _tiny_remap()
    out = verify_remap_parity(
        model_before=object(),  # type: ignore[arg-type]
        model_after=None,
        remap=r,
        sequences=[],
    )
    assert isinstance(out, NotRun) and "model_after" in out.reason


def test_a_not_run_parity_report_serializes_without_a_passed_field():
    r = _tiny_remap()
    report = remap_parity_report(
        model_before=None, model_after=None, remap=r, sequences=[np.array([0, 2])]
    )
    blob = report.to_json()
    assert blob["gate"]["state"] == "not_run"
    assert "passed" not in blob["gate"]
    assert blob["max_abs_loss_delta"] is None
    assert blob["memory_before"] is None and blob["memory_after"] is None
    assert report.full_vocab_loss_delta is None


def test_the_gate_thresholds_are_module_constants_not_parameters():
    """Rule 2: an agent may report that a gate failed, not move the threshold."""
    import inspect

    from qd_train import remap as remap_mod

    assert LOSS_PARITY_TOL == 1e-5
    for fn in (remap_mod.verify_remap_parity, remap_mod.remap_parity_report):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"tol", "tolerance", "atol", "rtol", "loss_parity_tol"}
    assert LOGIT_EXACTNESS_TOL == 1e-5
