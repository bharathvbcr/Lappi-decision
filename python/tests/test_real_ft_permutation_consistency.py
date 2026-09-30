"""``permutation_consistency`` on the FT eval: the gate the phase-3 GO rows never ran.

``docs/schema-api.md``: "permutation consistency (>= 95%) is a training gate and not only a
runtime check". Rung 0 had it; the FT eval did not, so every FT row carried the gate as
absent. These pin the three pieces that decide what the number means: the second pass
draws the plan's DERANGEMENT (not the trainer's shuffle, whose fixed points let a
position-biased model agree with itself), the answer is mapped back through it the right
way round, and rows that could not be asked leave through the denominator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.render import second_pass_permutation  # noqa: E402
from qd_train.artifacts import NO_SPAN, SLOT_CHOICE, SLOT_SPAN, SPAN_ABSTAIN, Batch  # noqa: E402
from qd_train.eval_harness import PERMUTATION_CONSISTENCY_FLOOR  # noqa: E402
from qd_train.trainer import ChoicePermutation, permute_choice_row  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

A, B, C, D, Z, DOT, NL, ANS = 1, 2, 3, 4, 9, 5, 6, 7
LETTERS = ("A", "B", "C", "D")
SPEC = ChoicePermutation(
    seed=0, letter_ids={"A": A, "B": B, "C": C, "D": D}, noul_id=Z,
    line_end_ids=frozenset({NL}),
)


def _choice_row(gold: int) -> np.ndarray:
    toks = [30, 31, NL]
    for letter, value in zip((A, B, C, D), (40, 41, 42, 43), strict=True):
        toks += [letter, DOT, value, NL]
    toks += [Z, DOT, 44, NL, ANS, gold]
    return np.asarray(toks, dtype=np.int32)


def _label(row_id: str, kind: int) -> rft.Label:
    return rft.Label(
        row_id=row_id, family_id="code.defect_class",
        slot_name="defect_class" if kind == SLOT_CHOICE else "defect_span",
        slot_kind=kind, gold_letter="A" if kind == SLOT_CHOICE else rft.NOUL_LETTER,
        letters=(*LETTERS, rft.NOUL_LETTER) if kind == SLOT_CHOICE else (rft.NOUL_LETTER,),
    )


def _batch(index: int, rows: list[np.ndarray], kinds: list[int]) -> Batch:
    width = max(len(r) for r in rows)
    tokens = np.zeros((len(rows), width), dtype=np.int32)
    for i, r in enumerate(rows):
        tokens[i, : len(r)] = r
    lengths = np.asarray([len(r) for r in rows])
    is_span = np.asarray(kinds) == SLOT_SPAN
    # A span row abstains, over line starts at 0 and 3; a choice row carries neither --
    # the shape a mixed defect_class val batch has.
    span_target = np.repeat(np.where(is_span, SPAN_ABSTAIN, NO_SPAN)[:, None], 2, axis=1)
    line_starts = np.zeros(tokens.shape, dtype=np.bool_)
    line_starts[is_span, 0] = True
    line_starts[is_span, 3] = True
    return Batch(
        tokens=tokens, lengths=lengths, bucket=width, index=index,
        slot_kind=np.asarray(kinds), target_index=lengths - 2,
        span_target=span_target if is_span.any() else None,
        line_starts=line_starts if is_span.any() else None,
    )


def _val() -> rft.ValSet:
    choice_rows = [_choice_row(A), _choice_row(C)]
    plan = [
        _batch(0, [*choice_rows, _choice_row(B)], [SLOT_CHOICE, SLOT_CHOICE, SLOT_SPAN]),
        _batch(1, [_choice_row(B)], [SLOT_SPAN]),
    ]
    labels_for = {
        0: [_label("r0", SLOT_CHOICE), _label("r1", SLOT_CHOICE), _label("s0", SLOT_SPAN)],
        1: [_label("s1", SLOT_SPAN)],
    }
    return rft.ValSet(
        reader=None,  # type: ignore[arg-type]  # second_pass_batches never reads it
        labels=[x for rows in labels_for.values() for x in rows],
        plan=plan, labels_for=labels_for, letter_id={},
    )


def test_the_second_pass_is_the_plans_derangement_of_each_row_and_nothing_else():
    val = _val()
    batches, labels_for, perms = rft.second_pass_batches(val, SPEC, seed=11)

    assert len(batches) == 1, "a batch holding no choice row has nothing to derange"
    assert labels_for == {0: val.labels_for[0]}
    for r, row_id in enumerate(("r0", "r1")):
        want = second_pass_permutation(4, seed=11, example_id=row_id, slot_name="defect_class")
        assert perms[(row_id, "defect_class")] == want
        assert all(want[j] != j for j in range(4)), "not a derangement"
        original = val.plan[0].tokens[r]
        assert batches[0].tokens[r].tolist() == permute_choice_row(
            original, answer_at=int(val.plan[0].lengths[r]) - 1,
            letter_ids=(A, B, C, D), noul_id=Z, line_end_ids=SPEC.line_end_ids, perm=want,
        ).tolist()
    assert batches[0].tokens[2].tolist() == val.plan[0].tokens[2].tolist(), (
        "the span row was moved"
    )
    assert set(perms) == {("r0", "defect_class"), ("r1", "defect_class")}


def _verdict(row_id: str, top: int, *, kind: str = "choice") -> dict[str, object]:
    return {"kind": kind, "row_id": row_id, "slot_name": "defect_class", "top": top,
            "noul_row": 4}


PERM = (1, 2, 3, 0)  # position j shows the option first shown at PERM[j]


def _agreement(first_tops, second_tops, *, extra_first=()):
    ids = [f"r{i}" for i in range(len(first_tops))]
    scored = {"verdicts": [_verdict(i, t) for i, t in zip(ids, first_tops, strict=True)]
              + [_verdict(i, 0) for i in extra_first]}
    second = {"verdicts": [_verdict(i, t) for i, t in zip(ids, second_tops, strict=True)]}
    perms = {(i, "defect_class"): PERM for i in ids}
    return rft.permutation_agreement(scored, second, perms)


def test_a_position_biased_model_scores_zero_and_a_content_reader_scores_one():
    """Mapping back the wrong way round swaps these two results, and both look plausible."""
    biased = _agreement([0, 0, 0, 0], [0, 0, 0, 0])
    assert isinstance(biased, Ran) and biased.value == 0.0 and not biased.passed
    content = _agreement([0, 1, 2, 3], [PERM.index(t) for t in (0, 1, 2, 3)])
    assert isinstance(content, Ran) and content.value == 1.0 and content.passed
    assert (content.n, content.n_total) == (4, 4)


def test_abstaining_twice_agrees_and_abstaining_once_does_not():
    got = _agreement([4, 4, 0], [4, PERM.index(1), PERM.index(0)])
    assert isinstance(got, Ran)
    assert (got.n, got.n_total) == (2, 3), got.detail


def test_rows_never_asked_leave_through_the_denominator_and_are_said_to():
    got = _agreement([0], [PERM.index(0)], extra_first=["single"])
    assert isinstance(got, Ran) and (got.n, got.n_total) == (1, 1)
    assert "1 row(s) had fewer than two live options" in got.detail


def test_a_row_decoded_once_is_refused_rather_than_dropped():
    scored = {"verdicts": [_verdict("r0", 0), _verdict("r1", 0)]}
    second = {"verdicts": [_verdict("r0", 3)]}
    perms = {("r0", "defect_class"): PERM, ("r1", "defect_class"): PERM}
    with pytest.raises(SystemExit, match="not the second"):
        rft.permutation_agreement(scored, second, perms)


def test_without_a_tokenizer_the_gate_is_not_run_and_says_why():
    second_pass = rft.prepare_second_pass(_val(), reader=None, tokenizer_json=None, seed=0)  # type: ignore[arg-type]
    got = rft.score_permutation_consistency(None, _val(), second_pass, {"verdicts": []})  # type: ignore[arg-type]
    assert isinstance(got, NotRun) and "tokenizer.json" in got.reason


def test_the_ft_eval_and_rung0_gate_on_one_floor():
    import rung0_real_run

    assert rung0_real_run.PERMUTATION_CONSISTENCY_FLOOR is PERMUTATION_CONSISTENCY_FLOOR
    assert PERMUTATION_CONSISTENCY_FLOOR == 0.95


def test_the_score_row_carries_the_gate():
    import inspect

    src = inspect.getsource(rft._record_score)
    assert 'recorder.gate("permutation_consistency", permutation)' in src
