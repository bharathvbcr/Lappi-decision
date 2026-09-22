"""``tools/real_ft_run.py --score-val``: a fine-tuned model scored on rows it never saw.

``GAP-RUNG3-NOTHING-SCORES-A-TRAINED-MODEL-ON-ROWS-IT-DID-NOT-TRAIN-ON``: every number the
FT runner wrote was about its own training rows. The scorer reuses ``_decode`` -- the one
place that reads a trained model the way ``crates/qd-runtime/src/answer.rs`` does -- so what
is new, and pinned here, is what surrounds it: the baseline each kind is judged against, the
letter ids a val row is decoded with, and the refusals that keep a val number from being
computed under the wrong remap or with nothing to score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.artifacts import SLOT_CHOICE, SLOT_SCORE, SLOT_SPAN  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402


def _label(kind: int, gold: str, n: int) -> rft.Label:
    return rft.Label(
        row_id=f"r{n}", family_id="f", slot_name="s", slot_kind=kind, gold_letter=gold,
        letters=("A", "B", "C"),
    )


def _scored(**by_kind: dict[str, int]) -> dict[str, object]:
    return {"by_kind": by_kind, "verdicts": []}


def test_a_letter_kind_is_judged_against_the_val_sets_most_common_gold() -> None:
    """6 choice rows, gold B four times: the constant "B" scores 4 of 6. A model at 5 of 6
    is above it; the same model at 4 of 6 is not -- a tie is not a win."""
    labels = [_label(SLOT_CHOICE, g, i) for i, g in enumerate("BBBBAC")]
    above = rft.score_states(_scored(choice={"n": 6, "correct": 5, "abstaining": 0}), labels)
    tie = rft.score_states(_scored(choice={"n": 6, "correct": 4, "abstaining": 0}), labels)
    got = above["val_top1.choice"]
    assert isinstance(got, Ran)
    assert (got.passed, got.n, got.n_total) == (True, 5, 6)
    assert "answering 'B' on every row" in got.detail and "4 of 6" in got.detail
    assert tie["val_top1.choice"].passed is False


def test_a_span_is_judged_against_abstaining_everywhere() -> None:
    labels = [_label(SLOT_SPAN, "noul", i) for i in range(4)]
    got = rft.score_states(_scored(span={"n": 4, "correct": 2, "abstaining": 3}), labels)[
        "val_top1.span"
    ]
    assert isinstance(got, Ran)
    assert got.passed is False, "2 of 4 is below the 3 of 4 that always abstaining scores"
    assert "abstaining on every row scores 3 of 4" in got.detail


def test_a_kind_the_val_set_does_not_hold_is_not_run_rather_than_zero() -> None:
    labels = [_label(SLOT_SCORE, "A", 0)]
    states = rft.score_states(_scored(score={"n": 1, "correct": 1, "abstaining": 0}), labels)
    assert isinstance(states["val_top1.choice"], NotRun)
    assert isinstance(states["val_top1.span"], NotRun)
    assert isinstance(states["val_top1.score"], Ran)


def test_letter_ids_from_both_sets_merge_when_they_agree() -> None:
    assert rft.merge_letter_ids({"A": 5, "noul": 9}, {"A": 5, "H": 12}) == {
        "A": 5, "noul": 9, "H": 12,
    }


def test_a_letter_that_reads_as_two_ids_is_refused() -> None:
    with pytest.raises(SystemExit, match="is token 5 in the train set and 6"):
        rft.merge_letter_ids({"A": 5}, {"A": 6})


def test_two_letters_on_one_id_are_refused() -> None:
    with pytest.raises(SystemExit, match="two letters share a token id"):
        rft.merge_letter_ids({"A": 5}, {"B": 5})


def test_a_missing_val_set_is_refused_with_the_command_that_builds_one(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--val-shards"):
        rft.open_val_set(
            tmp_path, config=DataConfig(), rev="HEAD", rows=[], train=None,  # type: ignore[arg-type]
            letter_id={},
        )


def test_score_val_without_the_epoch_arm_is_refused_before_anything_loads(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--epoch was not passed"):
        rft.main(["--out", str(tmp_path), "--score-val"])
