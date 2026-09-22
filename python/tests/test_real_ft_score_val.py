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


# --- ece and degenerate_head, from the distributions _decode now keeps --------------------


def _letter(kind: str, logits: list[float], gold: int, n: int) -> dict[str, object]:
    return {
        "kind": kind, "row_id": f"r{n}", "rows": len(logits), "row_logits": logits,
        "gold_row": gold,
    }


def test_letter_rows_group_by_the_runtime_tables_shape_and_softmax_over_their_own_rows() -> None:
    verdicts = [
        _letter("choice", [2.0, 0.0, 0.0], 0, 0),
        _letter("choice", [0.0, 1.0, 0.0, 0.0, 0.0], 1, 1),
        _letter("score", [0.0] * 6, 3, 2),
        {"kind": "span", "row_id": "s", "rows": 40},
    ]
    groups = rft.letter_distributions(verdicts)
    assert sorted(groups) == ["choice.k2", "choice.k4", "score.k5"], "noul is not an option"
    probs, gold = groups["choice.k2"]
    assert probs.shape == (1, 3) and gold.tolist() == [0]
    assert probs.sum() == pytest.approx(1.0) and int(probs.argmax()) == 0
    assert groups["score.k5"][0][0] == pytest.approx([1 / 6] * 6)


def test_a_letter_verdict_without_its_distribution_is_refused() -> None:
    with pytest.raises(TypeError, match="was not produced by _decode"):
        rft.letter_distributions([{"kind": "choice", "row_id": "r", "rows": 3, "top": 0}])


def test_the_ece_gate_does_not_pass_on_the_per_k_half_of_its_breakdown() -> None:
    """120 rows answered right at probability ~1: ECE 0, and the per-k metric passes. The gate
    still does not, because nothing measured the per-language half -- and says why."""
    verdicts = [_letter("choice", [40.0, 0.0, 0.0], 0, i) for i in range(60)] + [
        _letter("choice", [0.0, 40.0, 0.0], 1, i) for i in range(60, 120)
    ]
    metrics, ece, _ = rft.calibration_states({"verdicts": verdicts})
    per_k = metrics["ece.choice.k2"]
    assert isinstance(per_k, Ran) and per_k.passed and per_k.n == 120
    assert isinstance(ece, NotRun), "a per-k pass is not the plan's per-k-and-per-language gate"
    assert "no language" in ece.reason


def test_a_group_under_the_sample_floor_is_not_run_rather_than_a_small_ece() -> None:
    metrics, _, _ = rft.calibration_states(
        {"verdicts": [_letter("score", [1.0, 0.0, 0.0, 0.0], 0, i) for i in range(30)]}
    )
    assert isinstance(metrics["ece.score.k3"], NotRun)


def test_degenerate_head_fires_on_a_head_that_answers_one_row_everywhere() -> None:
    constant = [_letter("choice", [1.0, 0.0, 0.0], i % 2, i) for i in range(40)]
    _, _, degenerate = rft.calibration_states({"verdicts": constant})
    assert isinstance(degenerate, Ran) and degenerate.passed is False
    varied = [_letter("choice", [0.6, 0.0, 0.0] if i % 2 else [0.0, 0.6, 0.0], 1 - i % 2, i)
              for i in range(40)]
    _, _, healthy = rft.calibration_states({"verdicts": varied})
    assert isinstance(healthy, Ran) and healthy.passed is True


def test_with_no_letter_rows_neither_the_gate_nor_the_control_claims_to_have_run() -> None:
    metrics, ece, degenerate = rft.calibration_states(
        {"verdicts": [{"kind": "span", "row_id": "s", "rows": 40}]}
    )
    assert metrics == {}
    assert isinstance(ece, NotRun) and isinstance(degenerate, NotRun)


def test_decode_keeps_the_distribution_its_verdict_was_an_argmax_of() -> None:
    """``_decode`` needs a trained step to run; the end-to-end check is the stand-in run in
    the commit message. This pins that the field the two functions above read is written."""
    import inspect

    assert '"row_logits": logits[' in inspect.getsource(rft._decode)
