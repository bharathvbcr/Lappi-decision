"""``tools/perf_parity.py --p2``: Fable's AMENDED P2 screen for training without the padding mask.

The rule is campaign/f-j7prime-preregistered.json ``no_mask.p2_rule`` (2026-10-01). It replaces
the D(t) <= S(t)-on->=90%-of-steps rule, which no exchangeable null can satisfy: per step
P(D <= S) = C(3,2)/C(6,2) = 1/5 (``p2_rule_retired``).

Per channel and per step set T in {first live step, all live steps}:
- bbar = mean baseline;
- L = mean over T of bbar;
- dev_i = mean over T of (M_i - bbar) / L;
- noise_j = mean over T of (B_j - mean of the other baselines) / L;
- with tau = 0.02 and kappa = 3: inconclusive if max|noise_j| > tau/kappa; else fail if
  max|dev_i| > tau; else pass.

A channel is the worst over T, report-only. A shape is decided from its four applications,
(letter, span) x (first, all), as amended again by Fable after c18d245's calibration
(``no_mask.p2_amendment_2``):
- it is fail only when every application is conclusive (pass or fail) and at least one fails;
- otherwise it is not_run if any application is not_run, else inconclusive if any is
  inconclusive, else pass.
A held shape (inconclusive or not_run) is not cancelled on one application's fail while another
says the noise is unresolvable. The row always names its failing applications in
``fail_applications``, in (letter, span) then (first, all) order, empty when none.

These tests pin the statistic, the aggregation, the refusals and what the verdict row records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import perf_parity as pp  # noqa: E402

DIGEST = "c" * 64
N = 10


def _arm(tag: str, letter: list[float], span: list[float], *, mask: str,
         consumed: str = DIGEST, termination: str = "steps_exhausted") -> dict:
    return {
        "tag": tag, "steps": len(letter), "consumed_digest": consumed,
        "termination": termination, "train_attention_mask": mask,
        "train_path": {"train_attention_mask": mask, "tag": tag},
        "letter": [float(x).hex() for x in letter], "span": [float(x).hex() for x in span],
    }


def _letter(t: int) -> float:
    return 2.0 - 0.1 * t


def _span(t: int) -> float:
    return 1.0 - 0.05 * t


def _bases(*, jitter: float = 1e-4, n: int = N) -> list[dict]:
    """Three baselines, bit-identical at step 0 (a deterministic forward), then apart by a
    relative `jitter` that alternates in sign so their leave-one-out noise stays ~jitter."""
    out = []
    for k, s in enumerate((-1.0, 0.0, 1.0)):
        out.append(_arm(
            f"A-mask-{k + 1}",
            [_letter(t) * (1 + (s * jitter if t else 0.0)) for t in range(n)],
            [_span(t) * (1 + (s * jitter if t else 0.0)) for t in range(n)],
            mask="padding",
        ))
    return out


def _cands(scale_letter=lambda t: 1.0, scale_span=lambda t: 1.0, *, k: int = 3,
           n: int = N) -> list[dict]:
    return [
        _arm(f"A-none-{i + 1}", [_letter(t) * scale_letter(t) for t in range(n)],
             [_span(t) * scale_span(t) for t in range(n)], mask="none")
        for i in range(k)
    ]


def test_a_candidate_on_the_baseline_passes_and_the_row_records_the_rule():
    out = pp.p2_screen(_bases(), _cands())
    assert out["verdict"] == "pass"
    assert (out["tau"], out["kappa"]) == (0.02, 3)
    for ch in ("letter", "span"):
        c = out["channels"][ch]
        assert c["verdict"] == "pass" and c["live_steps"] == N
        assert set(c["sets"]) == {"first", "all"}
        for t_set in c["sets"].values():
            assert set(t_set) >= {"L", "dev_i", "noise_j", "dev", "noise", "verdict", "steps"}
            assert len(t_set["dev_i"]) == 3 and len(t_set["noise_j"]) == 3
            assert t_set["L"] > 0
        # Report-only: the retired per-step profile still rides on the row.
        prof = c["profile"]
        assert set(prof) >= {"median_d", "max_d", "median_s", "max_s", "frac_d_le_s",
                             "first_step_d_over_max_s", "final_base", "final_cand"}
    assert set(out["train_paths"]) == {"A-mask-1", "A-mask-2", "A-mask-3",
                                       "A-none-1", "A-none-2", "A-none-3"}


def test_the_first_live_step_set_is_the_first_step_and_all_is_every_live_step():
    out = pp.p2_screen(_bases(), _cands())
    sets = out["channels"]["letter"]["sets"]
    assert sets["first"]["steps"] == 1 and sets["all"]["steps"] == N
    assert sets["first"]["L"] == pytest.approx(_letter(0))
    assert sets["all"]["L"] == pytest.approx(sum(_letter(t) for t in range(N)) / N)


def test_a_three_percent_shift_on_every_step_fails():
    out = pp.p2_screen(_bases(), _cands(lambda t: 1.03))
    assert out["channels"]["letter"]["verdict"] == "fail"
    assert out["channels"]["letter"]["sets"]["all"]["dev"] == pytest.approx(0.03, rel=1e-6)
    assert out["channels"]["span"]["verdict"] == "pass"
    assert out["verdict"] == "fail"


def test_a_one_percent_shift_passes():
    out = pp.p2_screen(_bases(), _cands(lambda t: 1.01, lambda t: 1.01))
    assert out["verdict"] == "pass"


def test_a_step_one_only_shift_fails_through_the_first_step_set():
    """A kernel change that moves only the first forward: invisible in the all-steps mean,
    caught by the first-live-step set; the channel is the worst of the two."""
    out = pp.p2_screen(_bases(), _cands(lambda t: 1.03 if t == 0 else 1.0))
    letter = out["channels"]["letter"]
    assert letter["sets"]["first"]["verdict"] == "fail"
    assert letter["sets"]["all"]["verdict"] == "pass"
    assert letter["verdict"] == "fail" and out["verdict"] == "fail"


def test_one_shifted_candidate_among_two_null_ones_fails():
    cands = _cands()
    cands[1] = _cands(lambda t: 1.04, lambda t: 1.04, k=1)[0] | {"tag": "A-none-2"}
    out = pp.p2_screen(_bases(), cands)
    assert out["verdict"] == "fail"
    assert out["channels"]["letter"]["sets"]["all"]["dev_i"][1] == pytest.approx(0.04, rel=1e-6)


def test_noisy_baselines_are_inconclusive_even_when_the_candidate_is_far_off():
    """noise > tau/kappa is checked first: a screen that cannot resolve tau says so rather
    than failing (or passing) the candidate on noise. The first-step applications (baselines
    bit-identical there) fail, but the all-steps ones cannot resolve tau, so the shape is held
    for the human and names the fails it carries."""
    out = pp.p2_screen(_bases(jitter=0.02), _cands(lambda t: 1.10, lambda t: 1.10))
    for ch in ("letter", "span"):
        assert out["channels"][ch]["sets"]["all"]["verdict"] == "inconclusive"
    assert out["verdict"] == "inconclusive"
    assert out["fail_applications"] == [["letter", "first"], ["span", "first"]]


def test_noisy_baselines_with_a_null_candidate_are_inconclusive_not_fail():
    out = pp.p2_screen(_bases(jitter=0.02), _cands())
    assert out["channels"]["letter"]["verdict"] == "inconclusive"
    assert out["verdict"] == "inconclusive"


def test_a_channel_no_arm_exercised_is_not_run_and_the_shape_with_it():
    bases = [b | {"span": [(0.0).hex()] * N} for b in _bases()]
    cands = [c | {"span": [(0.0).hex()] * N} for c in _cands()]
    out = pp.p2_screen(bases, cands)
    assert out["channels"]["span"]["verdict"] == "not_run"
    assert out["channels"]["letter"]["verdict"] == "pass"
    assert out["verdict"] == "not_run"


def test_a_held_shape_names_its_fails_and_not_run_outranks_inconclusive():
    """A letter fail beside a span channel no arm exercised is not a fail: two of the four
    applications never ran, so the shape is not_run, and it names the fails it carries."""
    zero = (0.0).hex()
    bases = [b | {"span": [zero] * N} for b in _bases()]
    failing = [c | {"span": [zero] * N} for c in _cands(lambda t: 1.05)]
    out = pp.p2_screen(bases, failing)
    assert out["verdict"] == "not_run"
    assert out["fail_applications"] == [["letter", "first"], ["letter", "all"]]
    noisy = [b | {"span": [zero] * N} for b in _bases(jitter=0.02)]
    null = [c | {"span": [zero] * N} for c in _cands()]
    out = pp.p2_screen(noisy, null)
    assert out["channels"]["letter"]["verdict"] == "inconclusive"
    assert out["verdict"] == "not_run"


def _letter_noisy_span_quiet() -> list[dict]:
    """Baselines whose letter channel is apart by 2% after step 0 (so letter/all cannot
    resolve tau) while the span channel stays within 1e-4 (every span application resolves)."""
    quiet = _bases()
    return [b | {"span": q["span"]} for b, q in zip(_bases(jitter=0.02), quiet, strict=True)]


@pytest.mark.parametrize(
    ("bases", "cands", "verdict", "fails"),
    [
        # All four applications conclusive and one fails: the shape fails and names it.
        pytest.param(_bases, lambda: _cands(lambda t: 1.03 if t == 0 else 1.0),
                     "fail", [["letter", "first"]], id="all-conclusive-one-fail"),
        # One fail (span/first) beside one inconclusive (letter/all): held, naming the fail.
        pytest.param(_letter_noisy_span_quiet,
                     lambda: _cands(scale_span=lambda t: 1.03 if t == 0 else 1.0),
                     "inconclusive", [["span", "first"]], id="one-fail-one-inconclusive"),
        # A pass still carries the list, empty.
        pytest.param(_bases, _cands, "pass", [], id="pass"),
    ],
)
def test_a_shape_fails_only_when_all_four_applications_are_conclusive(bases, cands, verdict,
                                                                      fails):
    out = pp.p2_screen(bases(), cands())
    apps = {(ch, t): out["channels"][ch]["sets"][t]["verdict"]
            for ch in ("letter", "span") for t in ("first", "all")}
    assert [list(k) for k, v in apps.items() if v == "fail"] == fails
    assert out["verdict"] == verdict
    assert out["fail_applications"] == fails


@pytest.mark.parametrize("apps", [["pass"] * 3, ["pass"] * 5, ["pass", "pass", "pass", "ok"]])
def test_the_shape_aggregation_refuses_anything_but_four_known_verdicts(apps):
    """Fail closed: a missing application or an unknown verdict never reads as pass."""
    with pytest.raises(ValueError, match="takes 4 verdicts"):
        pp._p2_shape(apps)


def test_a_set_whose_mean_baseline_is_not_positive_is_not_run():
    neg = [b | {"letter": [(-1.0).hex(), *b["letter"][1:]]} for b in _bases()]
    cands = [c | {"letter": [(-1.0).hex(), *c["letter"][1:]]} for c in _cands()]
    letter = pp.p2_screen(neg, cands)["channels"]["letter"]
    assert letter["sets"]["first"]["verdict"] == "not_run"
    assert letter["verdict"] == "not_run"


def test_only_live_steps_enter_a_channel():
    """A span channel is 0.0 on every arm at a letter-only batch: not evidence either way."""
    zero = (0.0).hex()
    bases = [b | {"span": [zero, *b["span"][1:]]} for b in _bases()]
    cands = [c | {"span": [zero, *c["span"][1:]]} for c in _cands()]
    span = pp.p2_screen(bases, cands)["channels"]["span"]
    assert span["live_steps"] == N - 1
    assert span["sets"]["first"]["L"] == pytest.approx(_span(1))


@pytest.mark.parametrize(
    ("which", "change", "match"),
    [
        ("cand", {"consumed_digest": "d" * 64}, "different batches"),
        ("cand", {"letter": [(1.0).hex()] * (N - 1), "steps": N - 1}, "steps"),
        ("cand", {"termination": "wall_clock_cap"}, "steps_exhausted"),
        ("base", {"termination": "wall_clock_cap"}, "steps_exhausted"),
        ("base", {"train_attention_mask": "none"}, "baseline"),
        ("cand", {"train_attention_mask": "padding"}, "candidate"),
        ("cand", {"train_attention_mask": None}, "candidate"),
    ],
)
def test_arms_the_rule_cannot_compare_are_refused(which, change, match):
    bases, cands = _bases(), _cands()
    if which == "base":
        bases[0] = bases[0] | change
    else:
        cands[0] = cands[0] | change
    with pytest.raises(SystemExit, match=match):
        pp.p2_screen(bases, cands)


@pytest.mark.parametrize(("n_base", "n_cand"), [(2, 3), (3, 0)])
def test_fewer_than_three_baselines_or_no_candidate_is_refused(n_base, n_cand):
    with pytest.raises(SystemExit, match=r"baseline|candidate"):
        pp.p2_screen(_bases()[:n_base], _cands()[:n_cand])


def test_tau_and_kappa_are_module_constants_and_there_is_no_override():
    """Fable: no override flag, env var or --tau. A changed tau is a recorded amendment and a
    commit to the constant, then --p2 recomputed from the saved rows."""
    assert (pp.P2_TAU, pp.P2_KAPPA) == (0.02, 3)
    assert not hasattr(pp, "P2_MIN_FRAC")
    with pytest.raises(SystemExit):
        pp.main(["--p2", "x.jsonl", "--base", "a", "--cand", "b", "--tau", "0.05"])


def test_the_cli_exit_code_reads_pass_fail_and_everything_else(tmp_path):
    rows = tmp_path / "arms.jsonl"
    rows.write_text("\n".join(__import__("json").dumps(r) for r in [*_bases(), *_cands()]) + "\n")
    argv = ["--p2", str(rows), "--base", "A-mask-1", "A-mask-2", "A-mask-3",
            "--cand", "A-none-1", "A-none-2", "A-none-3", "--p2-out", str(tmp_path / "v.jsonl")]
    assert pp.main(argv) == 0
    assert (tmp_path / "v.jsonl").read_text().count("\n") == 1


# --- the gate the outcome run is held behind (unchanged by the amendment) --------------------


def _verdict(shape: str, verdict: str) -> dict:
    return {"base": [f"{shape}-mask-1", f"{shape}-mask-2", f"{shape}-mask-3"],
            "cand": [f"{shape}-none-1"], "verdict": verdict}


@pytest.mark.parametrize(
    ("rows", "ok"),
    [
        ([_verdict("A", "pass"), _verdict("B", "pass")], True),
        ([_verdict("A", "pass")], False),
        ([_verdict("A", "pass"), _verdict("B", "not_run")], False),
        ([_verdict("A", "pass"), _verdict("B", "inconclusive")], False),
        ([_verdict("A", "fail"), _verdict("B", "pass")], False),
        # A rerun of a shape supersedes its earlier verdict, in both directions.
        ([_verdict("A", "fail"), _verdict("B", "pass"), _verdict("A", "pass")], True),
        ([_verdict("A", "pass"), _verdict("B", "pass"), _verdict("B", "fail")], False),
        ([], False),
    ],
)
def test_the_outcome_run_gate_needs_a_p2_pass_at_both_shapes(rows, ok):
    """Fable: fail cancels the outcome run; not_run or missing cancels it until a rerun;
    inconclusive holds it for the human. Only pass at both shapes opens it."""
    passed, detail = pp.p2_gate(rows)
    assert passed is ok
    assert ("A" in detail and "B" in detail) or not ok


def test_the_gate_cli_refuses_a_missing_verdict_file(tmp_path):
    assert pp.main(["--p2-gate", str(tmp_path / "nope.jsonl")]) == 5
