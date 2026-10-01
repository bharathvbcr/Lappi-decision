"""Null calibration of perf_parity.p2_screen: six EXCHANGEABLE arms (same path, iid noise),
three labelled baseline and three candidate. What fraction of steps does D(t) <= S(t) hold on,
and how often does the pre-registered rule (>= 0.9 per channel) pass a candidate that is, by
construction, the baseline itself?"""
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(
    "/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-a59bae74f0f03844b/tools"
)))
import perf_parity as pp


def arm(tag, rng, n=100):
    letter = [2.0 * 0.98**t + rng.gauss(0.0, 1e-3 * (1 + t / 10)) for t in range(n)]
    span = [1.0 * 0.97**t + 0.1 + rng.gauss(0.0, 1e-3 * (1 + t / 10)) for t in range(n)]
    return {"tag": tag, "steps": n, "consumed_digest": "c" * 64,
            "letter": [x.hex() for x in letter], "span": [x.hex() for x in span]}


def run(n_cand, trials=200):
    fr, passes = [], 0
    for k in range(trials):
        rng = random.Random(k)
        arms = [arm(f"X{i}", rng) for i in range(3 + n_cand)]
        out = pp.p2_screen(arms[:3], arms[3:])
        fr.append(out["channels"]["letter"]["frac_d_le_s"])
        passes += out["verdict"] == "pass"
    return statistics.mean(fr), min(fr), max(fr), passes / trials


for n_cand in (3, 1):
    mean, lo, hi, p = run(n_cand)
    print(f"null, 3 baselines vs {n_cand} candidate(s): frac D<=S mean {mean:.3f} "
          f"[{lo:.2f}, {hi:.2f}] over 200 trials; rule passes {p:.1%} of null candidates")
