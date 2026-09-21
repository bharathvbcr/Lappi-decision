"""The S2/S4/trainer shared format.

This module is the executable contract three lanes build against, so its own tests are the
thing standing between "each lane is individually correct" and "the lanes can actually talk".
Both of this repo's cross-lane failures so far passed every test each lane wrote for itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_data.config import DataConfig
from qd_data.errors import HeldOutViolation
from qd_data.fingerprint import code_fingerprint
from qd_train.artifacts import (
    _MIN_ROW_TOKENS,
    _SLOT_KINDS,
    DROPPED,
    MAX_PADDING_WASTE,
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_LM,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
    RemapTable,
    ShardContractViolation,
    ShardHeader,
    TokenNotInRemap,
    assert_remap_covers,
    assert_shard_trainable,
    bucket_for,
    line_start_indices,
    padding_waste,
)
from qd_train.byte_context import line_starts as byte_line_starts
from qd_train.tristate import NotRun, Ran

V_OLD = 64


def _remap(keep: list[int], *, specials: tuple[int, ...] = (0, 1)) -> RemapTable:
    old_to_new = np.full(V_OLD, DROPPED, dtype=np.int32)
    for new_id, old_id in enumerate(keep):
        old_to_new[old_id] = new_id
    return RemapTable(
        old_to_new=old_to_new,
        new_to_old=np.array(keep, dtype=np.int32),
        tokenizer_hash="t" * 64,
        special_ids=specials,
    )


def _header(**over) -> ShardHeader:
    base = dict(
        split="train",
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        remap_hash="r" * 64,
        vocab_size=80_000,
        n_sequences=10,
        total_tokens=1_000,
        max_seq_len=512,
        buckets=(128, 512, 2048),
    )
    base.update(over)
    return ShardHeader(**base)  # type: ignore[arg-type]


# --- RemapTable ----------------------------------------------------------------------------


def test_a_well_formed_remap_reports_both_vocabulary_sizes():
    r = _remap([0, 1, 5, 9])
    assert r.source_vocab_size == V_OLD
    assert r.vocab_size == 4


def test_tables_that_are_not_inverse_are_refused():
    """A remap whose tables disagree relabels tokens, and the loss curve looks fine."""
    old_to_new = np.full(V_OLD, DROPPED, dtype=np.int32)
    old_to_new[0], old_to_new[1] = 0, 1
    with pytest.raises(ShardContractViolation, match="not inverse"):
        RemapTable(
            old_to_new=old_to_new,
            new_to_old=np.array([1, 0], dtype=np.int32),  # swapped
            tokenizer_hash="t" * 64,
            special_ids=(0, 1),
        )


def test_a_remap_that_drops_a_special_token_is_refused():
    with pytest.raises(ShardContractViolation, match="special token"):
        _remap([2, 3, 4], specials=(0, 1))


def test_table_sizes_that_disagree_are_refused():
    old_to_new = np.full(V_OLD, DROPPED, dtype=np.int32)
    old_to_new[0], old_to_new[1] = 0, 1
    with pytest.raises(ShardContractViolation, match="different vocabularies"):
        RemapTable(
            old_to_new=old_to_new,
            new_to_old=np.array([0, 1, 5], dtype=np.int32),
            tokenizer_hash="t" * 64,
            special_ids=(0, 1),
        )


def test_a_remap_without_a_tokenizer_hash_is_refused():
    old_to_new = np.full(V_OLD, DROPPED, dtype=np.int32)
    old_to_new[0], old_to_new[1] = 0, 1
    with pytest.raises(ShardContractViolation, match="tokenizer_hash"):
        RemapTable(
            old_to_new=old_to_new,
            new_to_old=np.array([0, 1], dtype=np.int32),
            tokenizer_hash="",
            special_ids=(0, 1),
        )


def test_encode_renumbers_kept_tokens():
    r = _remap([0, 1, 5, 9])
    assert list(r.encode(np.array([9, 5, 0, 1]))) == [3, 2, 0, 1]


def test_encode_raises_on_a_dropped_token_and_names_it():
    """The alternative is an UNK fallback, which trains the model on 'something was here'."""
    r = _remap([0, 1, 5, 9])
    with pytest.raises(TokenNotInRemap) as exc:
        r.encode(np.array([0, 1, 7]))
    msg = str(exc.value)
    assert "old id 7" in msg and "position 2" in msg
    assert "rebuild the remap" in msg


def test_encode_raises_outside_the_source_vocabulary():
    r = _remap([0, 1, 5, 9])
    with pytest.raises(TokenNotInRemap, match="outside the source vocabulary"):
        r.encode(np.array([0, V_OLD]))


def test_encode_of_nothing_is_empty_not_an_error():
    assert _remap([0, 1]).encode(np.array([], dtype=np.int64)).size == 0


def test_the_remap_hash_tracks_the_vocabulary():
    assert _remap([0, 1, 5]).remap_hash() != _remap([0, 1, 9]).remap_hash()
    assert _remap([0, 1, 5]).remap_hash() == _remap([0, 1, 5]).remap_hash()


def test_a_remap_round_trips_through_disk(tmp_path: Path):
    r = _remap([0, 1, 5, 9])
    written = r.write(tmp_path / "remap")
    back = RemapTable.read(tmp_path / "remap")
    assert back.remap_hash() == written == r.remap_hash()
    assert list(back.new_to_old) == [0, 1, 5, 9]


def test_a_tampered_remap_sidecar_is_refused(tmp_path: Path):
    """The hash on disk is recomputed, never trusted."""
    _remap([0, 1, 5, 9]).write(tmp_path / "remap")
    side = tmp_path / "remap.json"
    meta = json.loads(side.read_text())
    meta["remap_hash"] = "0" * 64
    side.write_text(json.dumps(meta))
    with pytest.raises(ShardContractViolation, match="modified after it was written"):
        RemapTable.read(tmp_path / "remap")


# --- ShardHeader ---------------------------------------------------------------------------


def test_a_packed_shard_set_is_refused():
    """SAFETY-2. A GDN layer bleeds recurrent state across a packed boundary, silently."""
    with pytest.raises(ShardContractViolation, match="SAFETY-2"):
        _header(packed=True)


def test_every_provenance_hash_is_required():
    for field in ("data_snapshot_hash", "tokenizer_hash", "remap_hash"):
        with pytest.raises(ShardContractViolation, match=field):
            _header(**{field: ""})


def test_a_bucket_smaller_than_the_longest_sequence_is_refused():
    with pytest.raises(ShardContractViolation, match="below max_seq_len"):
        _header(buckets=(128, 256), max_seq_len=512)


def test_unsorted_or_duplicated_buckets_are_refused():
    for bad in [(512, 128), (128, 128, 512)]:
        with pytest.raises(ShardContractViolation, match="strictly increasing"):
            _header(buckets=bad)


def test_uint16_token_storage_is_refused():
    """248,320 does not fit in uint16, and the overflow is silent."""
    with pytest.raises(ShardContractViolation, match="uint16 cannot hold"):
        _header(dtype="uint16")


def test_a_header_round_trips_and_a_tampered_one_is_refused():
    h = _header()
    raw = h.to_json()
    assert ShardHeader.from_json(raw).shard_hash() == h.shard_hash()
    raw["n_sequences"] = 11
    with pytest.raises(ShardContractViolation, match="modified after"):
        ShardHeader.from_json(raw)


# --- rule 3 at the new boundary --------------------------------------------------------------


def test_a_heldout_shard_set_is_refused_by_the_trainer_boundary(tmp_path: Path):
    """`open_training_data` guards manifests; the trainer reads shards. Same check, new door."""
    with pytest.raises(ShardContractViolation, match="Rule 3"):
        assert_shard_trainable(
            _header(split="heldout"), config=DataConfig(), path=tmp_path, repo_root=tmp_path
        )


def test_a_trainable_shard_set_reports_its_checks(tmp_path: Path):
    """A set as `write_shards` produces one -- which since 2026-09-21 means with a
    `code_fingerprint`. A header without one cannot answer whether `qd_data` has moved
    since it was written, and `shard_code_current` reports that as `NotRun` rather than as
    a pass; `test_fingerprint.py` asserts that case, so this one keeps its original claim
    that a good set passes everything.
    """
    checks = assert_shard_trainable(
        _header(split="train", code_fingerprint=code_fingerprint()),
        config=DataConfig(),
        path=tmp_path,
        repo_root=tmp_path,
    )
    assert all(isinstance(c, Ran) and c.passed for c in checks.values())
    assert set(checks) == {
        # The path check is listed because it is *recorded*, not merely performed. It was
        # missing entirely until 2026-09-20: this function asked what the header says about
        # itself and never where the shard set is, so a set at `data/heldout/shards-train`
        # declaring `split="train"` was admitted here while `assert_path_not_held_out`
        # refused the same path.
        "shard_path_not_held_out",
        "shard_split_trainable",
        "shard_provenance_pinned",
        "shard_not_packed",
        "held_out_families_configured",
        # `data_snapshot_hash` pins the corpus and `remap_hash` the vocabulary; between
        # them sits the code that turns one into rows, and nothing pinned it until a shard
        # set with three matching hashes was found to reproduce 321 rows where it stored
        # 341. See GAP-SHARD-SET-GOES-STALE-AGAINST-THE-CORPUS-CODE-THAT-REPRODUCES-ITS-LABELS.
        "shard_code_current",
    }


def test_a_shard_set_inside_a_held_out_root_is_refused_however_its_header_is_labelled(
    tmp_path: Path,
):
    """The location is the one question a header cannot lie about.

    This function used to ask only what the shard set says about *itself*. Measured against
    the pre-fix code: for a set at ``data/heldout/shards-train`` whose header honestly
    declares ``split="train"``, ``qd_train.data_access.assert_path_not_held_out`` REFUSED
    the path while ``assert_shard_trainable`` ADMITTED it and returned ``passed=True`` for
    every check it reported. Two doors, two different questions, and the trainer goes
    through this one.

    ``ShardReader`` had accepted a ``repo_root`` since it was written and passed it
    nowhere -- the parameter the check needs existed, and the check did not. That is the
    tell: a taken-and-discarded argument is usually a check someone meant to wire.

    Nothing here is malformed. The header is honest and the split is trainable; the shard
    set is simply somewhere rule 3 forbids a training process from reading.
    """
    held_out = tmp_path / "data" / "heldout" / "shards-train"
    held_out.mkdir(parents=True)
    with pytest.raises(HeldOutViolation) as exc:
        assert_shard_trainable(
            _header(split="train"),
            config=DataConfig(),
            path=held_out,
            repo_root=tmp_path,
        )
    assert "rule 3" in str(exc.value).lower()


def test_the_shard_door_and_the_manifest_door_agree_on_the_same_path(tmp_path: Path):
    """One question, one answer, whichever door asks it.

    The defect above was not that either check was wrong on its own -- it was that they
    disagreed, and which one you met depended on whether you arrived via a manifest or via
    a shard directory. This pins agreement rather than either verdict, so a future change
    that relaxes one of them fails here even if its own tests still pass.
    """
    from qd_train.data_access import assert_path_not_held_out

    config = DataConfig()
    for relative, expect_refused in (
        (Path("data") / "heldout" / "shards-train", True),
        (Path("data") / "shards" / "train", False),
    ):
        path = tmp_path / relative
        path.mkdir(parents=True)

        manifest_door_refused = False
        try:
            assert_path_not_held_out(path, config=config, repo_root=tmp_path)
        except HeldOutViolation:
            manifest_door_refused = True

        shard_door_refused = False
        try:
            assert_shard_trainable(
                _header(split="train"), config=config, path=path, repo_root=tmp_path
            )
        except HeldOutViolation:
            shard_door_refused = True

        assert manifest_door_refused == expect_refused, relative
        assert shard_door_refused == manifest_door_refused, (
            f"{relative}: the manifest door "
            f"{'refused' if manifest_door_refused else 'admitted'} this path and the shard "
            f"door {'refused' if shard_door_refused else 'admitted'} it. Rule 3 cannot "
            "depend on which door a training process happens to arrive through."
        )

# --- bucketing and padding waste -------------------------------------------------------------


def test_bucket_for_picks_the_smallest_fitting_bucket():
    b = (128, 512, 2048)
    assert bucket_for(1, b) == 0
    assert bucket_for(128, b) == 0, "a length equal to the boundary fits that bucket"
    assert bucket_for(129, b) == 1
    assert bucket_for(2048, b) == 2


def test_a_length_past_the_largest_bucket_is_refused_not_truncated():
    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        bucket_for(2049, (128, 512, 2048))


def test_a_nonpositive_length_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        bucket_for(0, (128,))


def test_padding_waste_over_no_sequences_is_not_run_not_zero():
    """0/0 waste reported as 0% would let an empty shard set clear the gate."""
    result = padding_waste([], (128,))
    assert isinstance(result, NotRun)
    assert "passed" not in result.to_json()


def test_padding_waste_passes_when_lengths_sit_near_their_buckets():
    result = padding_waste([128, 512, 500], (128, 512))
    assert isinstance(result, Ran) and result.passed
    assert result.value <= MAX_PADDING_WASTE


def test_padding_waste_fails_loudly_on_a_badly_bucketed_mixture():
    result = padding_waste([1, 1, 1, 1], (2048,))
    assert isinstance(result, Ran) and not result.passed
    assert result.value > MAX_PADDING_WASTE


def test_the_waste_gate_is_the_same_bucketing_the_sampler_uses():
    """A boundary length must land in the same bucket for the writer and the metric."""
    buckets = (128, 512)
    assert bucket_for(128, buckets) == 0
    # 128 in a 128-bucket is zero waste; if the metric used a different rule it would not be.
    result = padding_waste([128], buckets)
    assert isinstance(result, Ran) and result.value == 0.0


# --- the one line rule ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The five vectors from the Rust suite that owns the definition --
        # answering_procedure.rs::the_pointer_head_ranges_over_the_contexts_line_starts
        # _plus_the_abstain_row -- stated as offsets rather than as a count.
        ("a\n", (0,)),
        ("a\nb\n", (0, 2)),
        ("a\nb\nc", (0, 2, 4)),
        ("a\nb\nc\n", (0, 2, 4)),
        ("no newline at all", (0,)),
        # And the sixth, which that table does not carry and which is where the two Python
        # implementations diverged: context.rs:167 Context::line_count is 0 for an empty
        # context, asserted by context.rs::an_empty_context_is_a_legal_value_not_an_error.
        ("", ()),
        # CR is content, never a second terminator: docs/hardening.md section 1.
        ("a\rb", (0,)),
        ("a\r\nb", (0, 3)),
        # A blank line is a line.
        ("a\n\nb", (0, 2, 3)),
        ("\n\n", (0, 1)),
    ],
)
def test_line_start_indices_states_the_rust_line_rule(text: str, expected: tuple[int, ...]):
    """Offsets, not just a count: a count cannot tell a right grid from a shifted one."""
    assert line_start_indices(text) == expected
    # Same rule, same grid, whichever unit is counted -- `\n` is one char and one byte.
    assert line_start_indices(text.encode("utf-8")) == expected


def test_the_byte_path_line_rule_is_this_one():
    """The pin the module docstring promises, against the *other* implementation.

    ``qd_train.byte_context.line_starts`` is rung 0's, over bytes, and it cannot be
    deleted in favour of this one: its offsets *are* its model's positions, while S4 needs
    character offsets because that is what a tokenizer's offset mapping speaks. So the two
    stay separate and are held together here, which is the standing test
    ``GAP-S4-LINE-STARTS-SECOND-IMPLEMENTATION`` asked for.

    Over bytes the agreement is exact, offset for offset -- including the empty input,
    which is the vector they actually disagreed on while both suites were green.
    """
    for text in [
        "",
        "a",
        "a\n",
        "a\nb",
        "a\nb\n",
        "a\n\nb",
        "a\r\nb",
        "a\rb",
        "\n",
        "\n\n",
        "no newline at all",
        "a\nb\nc",
        "a\nb\nc\n",
        "é\nb",
        "\U0001f600\n\U0001f600\n",
    ]:
        raw = text.encode("utf-8")
        assert line_start_indices(raw) == byte_line_starts(raw), text


def test_a_line_rule_asked_of_something_that_is_not_text_is_refused():
    """A list of ints would silently count nothing and report one line."""
    with pytest.raises(TypeError, match="str or bytes"):
        line_start_indices([0x0A, 0x61])  # type: ignore[arg-type]


# --- the S2 <-> S4 cross-lane check ----------------------------------------------------------


def test_assert_remap_covers_catches_a_corpus_the_remap_was_not_built_over():
    """Each lane individually correct, the system still broken. The shape that bit us twice."""
    r = _remap([0, 1, 5, 9])
    assert_remap_covers(np.array([0, 5, 9]), r)  # fine
    with pytest.raises(TokenNotInRemap, match="disagree"):
        assert_remap_covers(np.array([0, 5, 42]), r)


# --- Batch: the seam between S4's reader and the trainer ---------------------------------------


def _batch(**over) -> Batch:
    base = dict(
        tokens=np.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=np.int32),
        lengths=np.array([3, 2], dtype=np.int32),
        bucket=0,
        index=0,
    )
    base.update(over)
    return Batch(**base)  # type: ignore[arg-type]


def test_a_well_formed_batch_reports_its_padding():
    b = _batch()
    assert b.padding_fraction == pytest.approx(3 / 8)


def test_a_row_claiming_more_tokens_than_it_has_is_refused():
    with pytest.raises(ShardContractViolation, match="exceeds the padded width"):
        _batch(lengths=np.array([9, 2], dtype=np.int32))


def test_a_zero_length_row_is_refused():
    with pytest.raises(ShardContractViolation, match="at least one real token"):
        _batch(lengths=np.array([3, 0], dtype=np.int32))


def test_a_one_token_row_is_refused_because_no_objective_supervises_it():
    """GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS, settled here rather than in the loop.

    `cpt_supervision` refused a one-token row -- it has no next-token pair, so it
    contributes zero gradient -- while this constructor accepted it. A constructor looser
    than its only consumer is a contract with two readings, which is the thing this module
    exists to prevent. Both objectives agree the row is unsupervisable: CPT needs
    `p < lengths-1` and FT needs `target_index` in `[0, lengths-1)`, an empty range at
    `lengths == 1`.
    """
    with pytest.raises(ShardContractViolation) as exc:
        _batch(lengths=np.array([3, 1], dtype=np.int32))
    message = str(exc.value)
    assert "row(s) [1]" in message, "the refusal must name the offending row"
    assert "at least 2" in message
    # The zero-length case keeps its own message: a length of 0 is a malformed length,
    # a length of 1 is a well-formed row that no objective can supervise.
    assert "at least one real token" not in message


def test_the_one_token_floor_agrees_with_what_the_writer_already_refuses():
    """`Batch` and S4's writer state one decision, so they are pinned to each other.

    `qd_train.shards._tokenize_checked` has refused `ids.size < 2` since S4 landed, for the
    same reason and with the same number. Two independent floors that could drift is the
    shape of `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`; this fails if either moves.
    """
    source = (Path(__file__).resolve().parents[1] / "qd_train" / "shards.py").read_text()
    assert "if ids.size < 2:" in source, (
        "qd_train.shards no longer refuses one-token sequences at write time, so "
        "artifacts._MIN_ROW_TOKENS is now the only floor and the writer can no longer "
        "name the source row"
    )
    assert _MIN_ROW_TOKENS == 2


def test_lengths_must_cover_every_row():
    with pytest.raises(ShardContractViolation, match="one entry per row"):
        _batch(lengths=np.array([3], dtype=np.int32))


def test_tokens_must_be_post_remap_int32():
    """int64 here usually means the source ids were handed over unremapped."""
    with pytest.raises(ShardContractViolation, match="post-remap"):
        _batch(tokens=np.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=np.int64))


def test_an_empty_batch_is_refused():
    with pytest.raises(ShardContractViolation, match="not a batch"):
        _batch(
            tokens=np.zeros((0, 4), dtype=np.int32),
            lengths=np.array([], dtype=np.int32),
        )


# --- supervision: the channel a span answer needs ----------------------------------------------
#
# The first version of Batch carried only tokens+lengths, so the only gold available to a span
# slot was the noul letter — the span head could only ever have learned to abstain, silently.
# `GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`.


def _ft(**over) -> Batch:
    base = dict(
        tokens=np.array([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=np.int32),
        lengths=np.array([4, 3], dtype=np.int32),
        bucket=0,
        index=0,
        slot_kind=np.array([SLOT_CHOICE, SLOT_CHOICE], dtype=np.uint8),
        target_index=np.array([2, 1], dtype=np.int32),
        span_target=None,
    )
    base.update(over)
    return Batch(**base)  # type: ignore[arg-type]


def test_a_cpt_batch_carries_no_supervision_and_that_is_valid():
    b = _batch()
    assert b.slot_kind is None and b.target_index is None and b.span_target is None


def test_slot_kind_and_target_index_are_all_or_nothing():
    with pytest.raises(ShardContractViolation, match="all-or-nothing"):
        _ft(target_index=None)


def test_span_target_without_slot_kind_is_refused():
    with pytest.raises(ShardContractViolation, match="nothing says which rows are spans"):
        _batch(span_target=np.array([[0, 1], [NO_SPAN, NO_SPAN]], dtype=np.int32))


def test_an_unknown_slot_kind_is_refused():
    with pytest.raises(ShardContractViolation, match="unknown slot kind"):
        _ft(slot_kind=np.array([SLOT_CHOICE, 9], dtype=np.uint8))


def test_an_lm_row_cannot_sit_in_the_slot_channel():
    """GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED, settled in the contract rather than guessed.

    `SLOT_LM` is the CPT kind and the CPT case is all three supervision fields `None`, so a
    batch carrying `slot_kind` is an FT batch and an LM row inside one has no gold letter
    for its `target_index` to name. The two available readings -- an LM row deliberately
    mixed into fine-tuning, versus a mislabelled choice/score row -- train different
    objectives and **both leave the loss curve looking fine**, so neither is guessed.
    """
    with pytest.raises(ShardContractViolation) as exc:
        _ft(slot_kind=np.array([SLOT_LM, SLOT_CHOICE], dtype=np.uint8))
    message = str(exc.value)
    assert "row(s) [0]" in message, "the refusal must name the offending row"
    assert "SLOT_LM" in message
    # Not the generic message: SLOT_LM is a *known* kind used wrongly, and a reader who
    # sees "unknown slot kind [0]" will go looking for a typo instead of for the writer.
    assert "unknown slot kind" not in message


def test_slot_lm_stays_a_known_kind_so_its_refusal_can_explain_itself():
    """The named refusal above only reaches the reader if `SLOT_LM` is in `_SLOT_KINDS`.

    Dropping it from the accepted set would refuse the same batches with the generic
    "unknown slot kind(s) [0]" message, which says nothing about the two readings. This
    pins the arrangement rather than the outcome, because the outcome is the same either
    way and the difference is entirely in what the writer is told.
    """
    assert SLOT_LM in _SLOT_KINDS
    assert {SLOT_LM, SLOT_CHOICE, SLOT_SCORE, SLOT_SPAN} == _SLOT_KINDS


def test_supervising_the_last_position_is_refused():
    """The supervised position's NEXT token is the gold, so the last one supervises nothing."""
    with pytest.raises(ShardContractViolation, match="supervises nothing"):
        _ft(target_index=np.array([3, 1], dtype=np.int32))


def test_a_span_row_without_a_span_target_is_refused():
    """This is the bug the channel exists for; the message has to name it."""
    with pytest.raises(ShardContractViolation) as exc:
        _ft(slot_kind=np.array([SLOT_SPAN, SLOT_CHOICE], dtype=np.uint8))
    assert "learns to abstain always" in str(exc.value)


def test_a_non_span_row_carrying_a_span_position_is_refused():
    with pytest.raises(ShardContractViolation, match="non-span row carries a span position"):
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_CHOICE], dtype=np.uint8),
            span_target=np.array([[0, 2], [0, 1]], dtype=np.int32),
        )


def test_a_span_that_starts_after_it_ends_is_refused():
    with pytest.raises(ShardContractViolation, match="start is after end"):
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_CHOICE], dtype=np.uint8),
            span_target=np.array([[3, 1], [NO_SPAN, NO_SPAN]], dtype=np.int32),
        )


def test_a_span_pointing_past_the_sequence_is_refused():
    with pytest.raises(ShardContractViolation, match="points past the sequence"):
        _ft(
            slot_kind=np.array([SLOT_CHOICE, SLOT_SPAN], dtype=np.uint8),
            span_target=np.array([[NO_SPAN, NO_SPAN], [0, 5]], dtype=np.int32),
        )


#: Candidates for the two-row fixture: row 0 has line starts at 1 and 3.
_LS = np.array([[False, True, False, True], [True, False, False, False]])


def test_a_well_formed_span_batch_is_accepted():
    b = _ft(
        slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
        span_target=np.array([[1, 3], [NO_SPAN, NO_SPAN]], dtype=np.int32),
        line_starts=_LS,
    )
    assert b.span_target is not None
    assert list(b.span_target[0]) == [1, 3]


# --- the pointer head's candidate set, and abstention ------------------------------------------


def test_a_span_row_without_the_candidate_set_is_refused():
    """Scoring every position while serving over line starts is a train/serve mismatch."""
    with pytest.raises(ShardContractViolation) as exc:
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
            span_target=np.array([[1, 3], [NO_SPAN, NO_SPAN]], dtype=np.int32),
        )
    assert "train/serve mismatch" in str(exc.value)


def test_a_gold_that_is_not_a_line_start_is_refused():
    """The invariant that separates a correct line->token mapping from a plausible one.

    Position 2 is in range and would look fine; it simply is not a candidate the pointer head
    can select. A mapping off by one line lands exactly here.
    """
    with pytest.raises(ShardContractViolation) as exc:
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
            span_target=np.array([[2, 3], [NO_SPAN, NO_SPAN]], dtype=np.int32),
            line_starts=_LS,
        )
    assert "not a line-start token" in str(exc.value)


def test_a_candidate_in_the_padding_is_refused():
    ls = _LS.copy()
    ls[1, 3] = True  # row 1 holds 3 real tokens, so position 3 is padding
    with pytest.raises(ShardContractViolation, match="which is padding"):
        _ft(
            slot_kind=np.array([SLOT_SCORE, SLOT_SPAN], dtype=np.uint8),
            span_target=np.array([[NO_SPAN, NO_SPAN], [0, 0]], dtype=np.int32),
            line_starts=ls,
        )


def test_a_span_row_with_no_candidates_at_all_is_refused():
    with pytest.raises(ShardContractViolation, match="marks no candidate"):
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
            span_target=np.array([[1, 3], [NO_SPAN, NO_SPAN]], dtype=np.int32),
            line_starts=np.zeros((2, 4), dtype=bool),
        )


def test_candidates_without_any_span_row_are_refused():
    with pytest.raises(ShardContractViolation, match="only meaningful for a pointer head"):
        _ft(line_starts=_LS)


def test_line_starts_must_be_a_bool_mask_shaped_like_tokens():
    with pytest.raises(ShardContractViolation, match="bool mask"):
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
            span_target=np.array([[1, 3], [NO_SPAN, NO_SPAN]], dtype=np.int32),
            line_starts=_LS.astype(np.int32),
        )


def test_an_abstaining_span_is_encodable():
    """The runtime can say "no evidence" (answer.rs:64). Until SPAN_ABSTAIN, training could not.

    `qa.answerability` is a held-out family, so the span slot is the only place abstention
    would be learned on this corpus — without this the model could never be taught it.
    """
    b = _ft(
        slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
        span_target=np.array([[SPAN_ABSTAIN, SPAN_ABSTAIN], [NO_SPAN, NO_SPAN]], dtype=np.int32),
        line_starts=_LS,
    )
    assert b.span_target is not None
    assert list(b.span_target[0]) == [SPAN_ABSTAIN, SPAN_ABSTAIN]


def test_abstaining_is_not_confusable_with_not_being_a_span():
    """-1 means 'not a span row'; -2 means 'this span abstains'. Collapsing them loses one."""
    assert NO_SPAN != SPAN_ABSTAIN


def test_a_half_abstaining_span_is_refused():
    with pytest.raises(ShardContractViolation, match="both positions or neither"):
        _ft(
            slot_kind=np.array([SLOT_SPAN, SLOT_SCORE], dtype=np.uint8),
            span_target=np.array([[SPAN_ABSTAIN, 3], [NO_SPAN, NO_SPAN]], dtype=np.int32),
            line_starts=_LS,
        )


def test_an_abstaining_span_needs_no_gold_on_a_line_start():
    """Abstention is the head's extra row, not a position, so the candidate check must skip it."""
    b = _ft(
        slot_kind=np.array([SLOT_SPAN, SLOT_SPAN], dtype=np.uint8),
        span_target=np.array([[SPAN_ABSTAIN, SPAN_ABSTAIN], [0, 0]], dtype=np.int32),
        line_starts=np.array([[True, False, False, False], [True, False, False, False]]),
    )
    assert b.span_target is not None


def test_span_positions_without_any_span_row_are_refused():
    with pytest.raises(ShardContractViolation, match="no row is SLOT_SPAN"):
        _ft(span_target=np.array([[0, 1], [NO_SPAN, NO_SPAN]], dtype=np.int32))
