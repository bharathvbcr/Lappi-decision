"""``tools/perf_parity.py --p2``: Fable's Tier-B P2 screen for training without the padding mask.

The rule (campaign/f-j7prime-preregistered.json ``no_mask.p2_rule``): per step t and channel,
D(t) <= S(t) on >= 90% of steps, both channels, both shapes, where S(t) is the largest
|B_i(t) - B_j(t)| over baseline pairs and D(t) the largest |M_i(t) - B_j(t)| over
(candidate, baseline) pairs. A P2 fail cancels the outcome run; a pass admits nothing. These
tests pin the statistic, and that a channel nothing exercised reads NOT RUN rather than pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import perf_parity as pp  # noqa: E402

DIGEST = "c" * 64


def _arm(tag: str, letter: list[float], span: list[float], *, consumed: str = DIGEST) -> dict:
    return {
        "tag": tag, "steps": len(letter), "consumed_digest": consumed,
        "letter": [float(x).hex() for x in letter], "span": [float(x).hex() for x in span],
    }


def _bases(n: int = 10) -> list[dict]:
    # Three baseline repeats whose spread grows from 0 at step 0 (a deterministic forward)
    # to 0.03 by the end, the way default-kernel atomics compound.
    return [
        _arm(f"B{k}", [2.0 - 0.1 * t + 0.001 * k * t for t in range(n)],
             [1.0 - 0.05 * t + 0.002 * k * t for t in range(n)])
        for k in range(3)
    ]


def test_a_candidate_inside_the_baseline_spread_passes_both_channels():
    bases = _bases()
    # Each candidate sits on a baseline from step 1 on; step 0 differs (S(0) == 0).
    cands = [
        _arm(f"M{k}", [2.0 + 1e-6 if t == 0 else 2.0 - 0.1 * t + 0.001 * k * t for t in range(10)],
             [1.0 + 1e-6 if t == 0 else 1.0 - 0.05 * t + 0.002 * k * t for t in range(10)])
        for k in range(3)
    ]
    out = pp.p2_screen(bases, cands)
    assert out["verdict"] == "pass"
    for ch in ("letter", "span"):
        c = out["channels"][ch]
        assert c["status"] == "ran" and c["live_steps"] == 10
        assert c["steps_d_le_s"] == 9, "step 0 has S == 0 and D > 0: a failing step"
        assert c["frac_d_le_s"] == pytest.approx(0.9)
        assert c["first_step_d_over_max_s"] is None


def test_a_candidate_outside_the_spread_fails_and_says_where():
    bases = _bases()
    cands = [_arm(f"M{k}", [2.0 - 0.1 * t + 0.05 * t for t in range(10)],
                  [1.0 - 0.05 * t + 0.002 * k * t for t in range(10)]) for k in range(3)]
    out = pp.p2_screen(bases, cands)
    assert out["verdict"] == "fail"
    letter = out["channels"]["letter"]
    assert letter["status"] == "ran" and letter["frac_d_le_s"] < 0.9
    assert letter["first_step_d_over_max_s"] is not None
    assert out["channels"]["span"]["frac_d_le_s"] >= 0.9


def test_a_channel_no_arm_ever_exercised_is_not_run_and_the_shape_does_not_pass():
    bases = [_arm(f"B{k}", [2.0 - 0.1 * t + 0.001 * k * t for t in range(5)], [0.0] * 5)
             for k in range(3)]
    cands = [_arm(f"M{k}", [2.0 - 0.1 * t + 0.001 * k * t for t in range(5)], [0.0] * 5)
             for k in range(3)]
    out = pp.p2_screen(bases, cands)
    assert out["channels"]["span"]["status"] == "not_run"
    assert out["channels"]["letter"]["status"] == "ran"
    assert out["verdict"] == "not_run"


def test_only_live_steps_count_toward_a_channel():
    """A span channel is 0.0 on every arm at a letter-only batch; those steps are not
    evidence either way, so they do not pad the fraction."""
    span = [0.0, 0.9, 0.0, 0.8]
    bases = [_arm(f"B{k}", [2.0, 1.9, 1.8, 1.7], [x + 0.01 * k if x else 0.0 for x in span])
             for k in range(3)]
    cands = [_arm(f"M{k}", [2.0, 1.9, 1.8, 1.7], [x + 0.5 if x else 0.0 for x in span])
             for k in range(3)]
    out = pp.p2_screen(bases, cands)
    assert out["channels"]["span"]["live_steps"] == 2
    assert out["channels"]["span"]["steps_d_le_s"] == 0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"consumed_digest": "d" * 64}, "different batches"),
        ({"letter": [(1.0).hex()] * 9, "steps": 9}, "steps"),
    ],
)
def test_arms_over_different_batches_or_lengths_are_refused(change, match):
    bases = _bases()
    cands = [{**_bases()[0], "tag": "M0", **change}, {**_bases()[1], "tag": "M1"},
             {**_bases()[2], "tag": "M2"}]
    with pytest.raises(SystemExit, match=match):
        pp.p2_screen(bases, cands)


def _verdict(shape: str, verdict: str) -> dict:
    return {"base": [f"{shape}-mask-1", f"{shape}-mask-2", f"{shape}-mask-3"],
            "cand": [f"{shape}-none-1"], "verdict": verdict}


@pytest.mark.parametrize(
    ("rows", "ok"),
    [
        ([_verdict("A", "pass"), _verdict("B", "pass")], True),
        ([_verdict("A", "pass")], False),
        ([_verdict("A", "pass"), _verdict("B", "not_run")], False),
        ([_verdict("A", "fail"), _verdict("B", "pass")], False),
        # A rerun of a shape supersedes its earlier verdict, in both directions.
        ([_verdict("A", "fail"), _verdict("B", "pass"), _verdict("A", "pass")], True),
        ([_verdict("A", "pass"), _verdict("B", "pass"), _verdict("B", "fail")], False),
        ([], False),
    ],
)
def test_the_outcome_run_gate_needs_a_p2_pass_at_both_shapes(rows, ok):
    """Fable: a P2 fail cancels the outcome run. A shape with no verdict, or not_run, cancels
    it too -- an unexamined screen is not a passed one."""
    passed, detail = pp.p2_gate(rows)
    assert passed is ok
    assert ("A" in detail and "B" in detail) or not ok


def test_the_gate_cli_refuses_a_missing_verdict_file(tmp_path):
    assert pp.main(["--p2-gate", str(tmp_path / "nope.jsonl")]) == 5


@pytest.mark.parametrize(("n_base", "n_cand"), [(2, 3), (3, 0)])
def test_fewer_than_three_baselines_or_no_candidate_is_refused(n_base, n_cand):
    """Fable: the spread is measured on >= 3 baseline repeats."""
    bases = _bases()[:n_base]
    cands = [{**b, "tag": f"M{i}"} for i, b in enumerate(_bases()[:n_cand])]
    with pytest.raises(SystemExit, match=r"baseline|candidate"):
        pp.p2_screen(bases, cands)
