"""Collation planning for rung 0. Torch-free, so it runs in the repo venv."""

from __future__ import annotations

import numpy as np
import pytest
from data_fixtures import mutate_row as row

from qd_train.artifacts import SPAN_ABSTAIN
from qd_train.byte_batch import _validate, padding_waste, plan_batch, span_supervision
from qd_train.byte_context import ID_PAD
from qd_train.mutate_adapter import CLEAN, parse_example, to_decision
from qd_train.schema_mirror import RESERVED_NOUL_ROWS

PAD = ID_PAD
WIDE = dict(pad_id=PAD, max_context_bytes=4096, max_option_bytes=96)
#: The default row mutates line 2, so a one-line `after` needs its span moved with it.
LINE1 = {"start_line": 1, "end_line": 1}


def decision(**over):
    return to_decision(parse_example(row(**over)), max_context_bytes=4096)


def clean(**over):
    return decision(id=over.pop("id", "c"), **{"class": CLEAN}, span=None, **over)


# ---------------------------------------------------------------------------
# The two abstain columns
# ---------------------------------------------------------------------------


def test_the_choice_and_span_abstain_columns_are_different_numbers():
    """They coincide only by accident, and reusing one for the other is the classic defect.

    Four options means the choice abstention sits at column 4; a three-line context means the
    span abstention sits at row 3.
    """
    d = decision(after="a\nb\nc")
    plan = plan_batch([d], **WIDE)

    assert plan.n_live_options[0] == 4
    assert plan.choice_abstain_column(0) == 4
    assert len(plan.line_starts[0]) == 3
    assert plan.span_abstain_row(0) == 3
    assert plan.choice_abstain_column(0) != plan.span_abstain_row(0)


def test_each_slot_reserves_exactly_one_row_for_its_abstention():
    d = decision(after="a\nb\nc")
    plan = plan_batch([d], **WIDE)
    assert plan.choice_rows(0) == plan.n_live_options[0] + RESERVED_NOUL_ROWS
    assert plan.span_rows(0) == len(plan.line_starts[0]) + RESERVED_NOUL_ROWS


def test_a_clean_example_targets_the_span_abstain_row_and_says_so():
    plan = plan_batch([clean(after="a\nb\nc")], **WIDE)
    assert plan.span_is_noul[0]
    assert plan.span_target[0] == plan.span_abstain_row(0) == 3


def test_a_mutated_example_targets_a_line_and_is_not_noul():
    plan = plan_batch([decision(after="a\nb\nc", span={"start_line": 2, "end_line": 2})], **WIDE)
    assert not plan.span_is_noul[0]
    assert plan.span_target[0] == 1
    assert plan.span_target[0] != plan.span_abstain_row(0)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_contexts_pad_to_the_batch_maximum_not_the_model_ceiling():
    short = decision(id="s", after="a\nb")
    long = decision(id="l", after="x\n" * 40 + "y")
    plan = plan_batch([short, long], **WIDE)

    width = plan.context_width
    assert width == long.context.n_bytes_kept
    assert all(len(r) == width for r in plan.context_ids)
    assert all(len(r) == width for r in plan.context_mask)
    assert width < 4096, "padding to the model ceiling would waste most of the compute"


def test_padding_is_masked_off_and_uses_the_pad_id():
    plan = plan_batch(
        [decision(id="s", after="ab", span=LINE1), decision(id="l", after="abcdefgh", span=LINE1)],
        **WIDE,
    )
    row0_ids, row0_mask = plan.context_ids[0], plan.context_mask[0]
    assert sum(row0_mask) == 2
    for i, live in enumerate(row0_mask):
        if not live:
            assert row0_ids[i] == PAD


def test_a_line_start_never_lands_on_padding():
    """A candidate outside the live context can still win a softmax."""
    plan = plan_batch(
        [decision(id="s", after="a\nb"), decision(id="l", after="x\ny\nz\nw\nv")], **WIDE
    )
    for i in range(plan.batch_size):
        for offset in plan.line_starts[i]:
            assert plan.context_mask[i][offset]


def test_dead_option_columns_are_fully_masked_rather_than_dropped():
    two = to_decision(
        parse_example(row(id="a")), max_context_bytes=4096, options=("logic", "clean")
    )
    four = decision(id="b")
    plan = plan_batch([two, four], **WIDE)

    assert plan.n_live_options == (2, 4)
    assert len(plan.option_ids[0]) == len(plan.option_ids[1]) == 4
    # The two dead columns on row 0 carry no live bytes at all.
    for col in (2, 3):
        assert not any(plan.option_mask[0][col])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="empty batch"):
        plan_batch([], **WIDE)


def test_padding_waste_on_an_empty_batch_raises_rather_than_returning_zero():
    """0.0 would read as 'no waste', which is `all([]) is True` one level down."""
    plan = plan_batch([decision()], **WIDE)
    empty = type(plan)(
        example_ids=(), context_ids=(), context_mask=(), option_ids=(), option_mask=(),
        n_live_options=(), choice_target=(), line_starts=(), span_target=(),
        span_end_target=(), span_is_noul=(),
    )
    with pytest.raises(ValueError, match="no padding waste"):
        padding_waste(empty)


def test_padding_waste_is_the_fraction_of_dead_context_positions():
    plan = plan_batch(
        [decision(id="s", after="ab", span=LINE1), decision(id="l", after="abcd", span=LINE1)],
        **WIDE,
    )
    # width 4, live 2 + 4 = 6 of 8.
    assert padding_waste(plan) == pytest.approx(0.25)


def test_a_context_wider_than_the_model_window_is_refused():
    d = decision(after="x" * 300, span=LINE1)
    with pytest.raises(ValueError, match="exceeds the model's"):
        plan_batch([d], pad_id=PAD, max_context_bytes=128, max_option_bytes=96)


def test_an_option_wider_than_the_budget_is_refused():
    d = decision()
    with pytest.raises(ValueError, match=r"exceeds the .* budget"):
        plan_batch([d], pad_id=PAD, max_context_bytes=4096, max_option_bytes=3)


def _broken(plan, **over):
    return type(plan)(**{**{f: getattr(plan, f) for f in plan.__slots__}, **over})


def test_a_span_target_disagreeing_with_span_is_noul_is_refused():
    """The two carry one fact; a plan where they disagree is malformed by construction."""
    plan = plan_batch([clean(after="a\nb\nc")], **WIDE)
    # Both pointers move to a real line while span_is_noul still says True.
    broken = _broken(plan, span_target=(0,), span_end_target=(0,))

    with pytest.raises(ValueError, match="one fact, and they disagree"):
        _validate(broken)


def test_half_an_abstention_is_refused():
    """A span answer is one decision: both pointers abstain or neither does.

    Letting the start abstain while the end points would render as a span with no
    beginning, which `answer.rs` has no way to express.
    """
    plan = plan_batch([clean(after="a\nb\nc")], **WIDE)
    with pytest.raises(ValueError, match="one pointer abstains"):
        _validate(_broken(plan, span_target=(0,)))  # end still on the abstain row


def test_an_end_before_its_start_is_refused():
    d = decision(after="a\nb\nc", span={"start_line": 3, "end_line": 3})
    plan = plan_batch([d], **WIDE)
    with pytest.raises(ValueError, match="end before its start"):
        _validate(_broken(plan, span_end_target=(0,)))


def test_the_end_pointer_carries_the_span_s_last_line_not_its_first():
    """Training the end pointer on the start line is a fabricated target."""
    plan = plan_batch(
        [decision(after="a\nb\nc\nd", span={"start_line": 2, "end_line": 4})], **WIDE
    )
    assert (plan.span_target[0], plan.span_end_target[0]) == (1, 3)


# ---------------------------------------------------------------------------
# The bridge to the pointer head's contract shape
# ---------------------------------------------------------------------------


def test_span_supervision_converts_ordinals_to_offsets():
    """BatchPlan counts lines; SpanSupervision counts bytes. That is the whole conversion."""
    plan = plan_batch(
        [decision(after="aa\nbb\ncc", span={"start_line": 2, "end_line": 3})], **WIDE
    )
    sup = span_supervision(plan)

    assert plan.span_target[0] == 1 and plan.span_end_target[0] == 2  # ordinals
    assert (int(sup.start[0]), int(sup.end[0])) == (3, 6)  # byte offsets
    assert plan.line_starts[0] == (0, 3, 6)


def test_span_supervision_marks_an_abstention_rather_than_pointing():
    sup = span_supervision(plan_batch([clean(after="a\nb\nc")], **WIDE))
    assert bool(sup.abstaining[0])
    assert int(sup.start[0]) == int(sup.end[0]) == SPAN_ABSTAIN
    assert SPAN_ABSTAIN < 0, "an abstaining row must not carry a readable position"


def test_span_supervision_line_start_mask_agrees_with_the_plan():
    plan = plan_batch([decision(id="a", after="a\nb"), decision(id="b", after="x\ny\nz")], **WIDE)
    sup = span_supervision(plan)
    for i in range(plan.batch_size):
        assert set(np.flatnonzero(sup.line_starts[i])) == set(plan.line_starts[i])


def test_the_query_index_is_a_live_position_on_every_row():
    """A query gathered from padding would read a state the encoder never supervised."""
    plan = plan_batch(
        [decision(id="a", after="a\nb"), decision(id="b", after="x\ny\nz\nw")], **WIDE
    )
    sup = span_supervision(plan)
    for i in range(plan.batch_size):
        assert plan.context_mask[i][int(sup.query_index[i])]


def test_span_supervision_on_an_empty_batch_raises():
    plan = plan_batch([decision()], **WIDE)
    with pytest.raises(ValueError, match="absent and empty differ"):
        span_supervision(_broken(plan, example_ids=(), context_ids=(), context_mask=()))


def test_the_head_plan_round_trips_the_ordinals_back():
    """plan_span_batch is the one owner of the abstain row, so the bridge must survive it."""
    heads = pytest.importorskip("qd_train.heads")
    plan = plan_batch(
        [
            decision(id="m", after="aa\nbb\ncc", span={"start_line": 2, "end_line": 3}),
            clean(id="c", after="x\ny"),
        ],
        **WIDE,
    )
    span_plan = heads.plan_span_batch(span_supervision(plan))

    assert [int(v) for v in span_plan.gold_start] == [plan.span_target[0], plan.span_abstain_row(1)]
    assert [int(v) for v in span_plan.gold_end] == [
        plan.span_end_target[0],
        plan.span_abstain_row(1),
    ]


# ---------------------------------------------------------------------------
# Ragged batches
# ---------------------------------------------------------------------------


def test_rows_stay_aligned_across_a_ragged_batch():
    ds = [
        decision(id="m1", after="a\nb\nc", span={"start_line": 2, "end_line": 2}),
        clean(id="c1", after="x\ny"),
        decision(id="m2", after="p\nq\nr\ns", span={"start_line": 4, "end_line": 4}),
    ]
    plan = plan_batch(ds, **WIDE)

    assert plan.example_ids == ("m1", "c1", "m2")
    assert [len(s) for s in plan.line_starts] == [3, 2, 4]
    assert plan.span_target == (1, 2, 3)
    assert plan.span_is_noul == (False, True, False)
    assert [plan.span_abstain_row(i) for i in range(3)] == [3, 2, 4]
