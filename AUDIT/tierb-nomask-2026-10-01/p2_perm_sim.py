"""A calibrated alternative for Fable to accept or reject: the same D(t) <= S(t) fraction, but
judged against its own role-permutation distribution. Over the 20 ways to call 3 of the 6 arms
'baseline', the true labelling fails only if its fraction is the strict minimum (p <= 1/20).
Null pass rate, and pass rate when the candidates carry a systematic offset."""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(
    "/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-a59bae74f0f03844b/tools"
)))
import perf_parity as pp


def arm(tag, rng, offset=0.0, n=100):
    letter = [2.0 * 0.98**t + offset * (1 + t / 10) * 1e-3
              + rng.gauss(0.0, 1e-3 * (1 + t / 10)) for t in range(n)]
    span = [1.0 * 0.97**t + 0.1 + offset * (1 + t / 10) * 1e-3
            + rng.gauss(0.0, 1e-3 * (1 + t / 10)) for t in range(n)]
    return {"tag": tag, "steps": n, "consumed_digest": "c" * 64,
            "letter": [x.hex() for x in letter], "span": [x.hex() for x in span]}


def frac(bases, cands):
    out = pp.p2_screen(bases, cands, min_frac=0.0)
    return min(out["channels"][c]["frac_d_le_s"] for c in ("letter", "span"))


def perm_pass(arms):
    true = frac(arms[:3], arms[3:])
    others = []
    for idx in itertools.combinations(range(6), 3):
        if idx == (0, 1, 2):
            continue
        b = [arms[i] for i in idx]
        c = [arms[i] for i in range(6) if i not in idx]
        others.append(frac(b, c))
    return true >= min(others)


for offset in (0.0, 1.0, 2.0, 4.0):
    trials = 100
    ok = 0
    for k in range(trials):
        rng = random.Random(1000 + k)
        arms = [arm(f"B{i}", rng) for i in range(3)] + [arm(f"M{i}", rng, offset) for i in range(3)]
        ok += perm_pass(arms)
    print(f"candidate offset {offset:.0f} sigma: permutation rule passes {ok}/{trials}")
