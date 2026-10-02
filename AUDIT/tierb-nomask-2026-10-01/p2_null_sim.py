"""P2 calibration, v3: every target in Fable's amended rule, against the implemented function.

campaign/f-j7prime-preregistered.json ``no_mask.p2_rule`` (amended 2026-10-01, and again in
``no_mask.p2_amendment_2`` after v2's calibration at c18d245) lists the targets this simulation
must meet before any GPU P2 session. Each one is checked here by calling
``tools/perf_parity.p2_screen`` itself on synthetic arm rows, never a re-implementation of it.
A target that is missed is reported with its numbers and the script exits non-zero. Nothing
is tuned.

The configuration universe -- n in {100, 50}, 3v3 and 3v1, with and without bit-identical
baseline forwards -- applies to every target unless the target names its own.

History:
- v1 (7ba9df0, the retired D(t) <= S(t)-on->=90% rule) found 0 of 200 null candidates passing.
  The cause is exact and distribution-free: per step, P(D <= S) = C(3,2)/C(6,2) = 1/5 for six
  exchangeable arms. v1 put it at "about a quarter"; Fable sharpened it.
- v2 (c18d245) met 75 of 78 checks. The misses and their mechanisms are recorded in
  ``p2_miss_diag.py``/``.out``.
- v3 (this file) follows amendment 2:
  - the shape fails only when all four applications are conclusive;
  - noise x5 is added to the null grid;
  - the held shapes that carried a failing application are counted;
  - delta 2% (= tau) is report-only; delta 1% (pass) and 3% (fail) are added.

Seeds are fixed per configuration (``SEED``). Every configuration v2 ran keeps v2's seed, so its
trials are the same draws judged under the amended aggregation. The configurations v3 adds take
seeds after v2's last (76).

The noise model is v1's:
- letter(t) = 2.0 * 0.98**t + N(0, sigma_t);
- span(t) = 0.97**t + 0.1 + N(0, sigma_t);
- sigma_t = 1e-3 * (1 + t/10) * multiplier.

"Bit-identical baseline forwards" means step 0 carries no noise in any arm, as a default-kernel
forward at the same weights does. Candidates are labelled none and baselines padding, and every
arm ended steps_exhausted, as the rule's refusals require.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import perf_parity as pp  # noqa: E402

TRIALS = 1000
FAILED: list[str] = []
HELD = ("inconclusive", "not_run")

#: The first seed of each target's run of configurations, in loop order. v2's seeds are kept
#: (1..76); v3's additions follow.
SEED = {
    "null": 1, "x3": 9, "x10": 17, "x20": 25, "x100": 33,
    "d0.005": 41, "d0.02": 49, "d0.04": 57, "one1.04": 65, "step1.03": 69, "step1.01": 73,
    "x5": 77, "d0.01": 85, "d0.03": 93,
}


def series(rng: random.Random, n: int, mult: float, bitid: bool, scale: float = 1.0,
           step1_scale: float = 1.0) -> tuple[list[float], list[float]]:
    letter, span = [], []
    for t in range(n):
        sigma = 1e-3 * (1 + t / 10) * mult
        e1 = 0.0 if (bitid and t == 0) else rng.gauss(0.0, sigma)
        e2 = 0.0 if (bitid and t == 0) else rng.gauss(0.0, sigma)
        f = scale * (step1_scale if t == 0 else 1.0)
        letter.append((2.0 * 0.98**t + e1) * f)
        span.append((0.97**t + 0.1 + e2) * f)
    return letter, span


def row(tag: str, mask: str, letter: list[float], span: list[float]) -> dict:
    return {"tag": tag, "steps": len(letter), "consumed_digest": "c" * 64,
            "termination": "steps_exhausted", "train_attention_mask": mask,
            "train_path": {"sim": True}, "letter": [x.hex() for x in letter],
            "span": [x.hex() for x in span]}


def trial(rng, *, n, k, bitid, mult=1.0, cand_scale=(), step1_scale=1.0) -> dict:
    """One screen: 3 null baselines; k candidates, candidate i scaled by cand_scale[i]
    (default 1, the null) and its step 0 by step1_scale."""
    bases = [row(f"B{j}", "padding", *series(rng, n, mult, bitid)) for j in range(3)]
    scales = list(cand_scale) + [1.0] * (k - len(cand_scale))
    cands = [row(f"M{i}", "none", *series(rng, n, mult, bitid, scales[i], step1_scale))
             for i in range(k)]
    return pp.p2_screen(bases, cands)


def split(seed: int, **kw) -> tuple[dict[str, float], int]:
    """The verdict split over TRIALS screens, and how many held shapes (inconclusive or
    not_run) carried a failing application."""
    rng = random.Random(seed)
    counts = {"pass": 0, "fail": 0, "inconclusive": 0, "not_run": 0}
    held_with_fail = 0
    for _ in range(TRIALS):
        out = trial(rng, **kw)
        counts[out["verdict"]] += 1
        held_with_fail += out["verdict"] in HELD and bool(out["fail_applications"])
    return {v: c / TRIALS for v, c in counts.items()}, held_with_fail


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'MET ' if ok else 'MISS'}] {name}: {detail}", flush=True)
    if not ok:
        FAILED.append(f"{name}: {detail}")


def report(name: str, detail: str) -> None:
    print(f"  [INFO] {name}: {detail}", flush=True)


def fmt(s: dict[str, float]) -> str:
    return " ".join(f"{k} {v:.1%}" for k, v in s.items())


CONFIGS = [(n, k, bitid) for n in (100, 50) for k in (3, 1) for bitid in (False, True)]


def label(n, k, bitid) -> str:
    return f"n={n} 3v{k} {'bit-identical step 0' if bitid else 'noisy step 0'}"


def main() -> int:
    print(f"tau={pp.P2_TAU} kappa={pp.P2_KAPPA}; {TRIALS} trials per configuration\n")
    print("1. exchangeable null at the v1 noise model: pass >= 99%, fail 0")
    for i, (n, k, bitid) in enumerate(CONFIGS):
        s, _ = split(SEED["null"] + i, n=n, k=k, bitid=bitid)
        check(label(n, k, bitid), s["pass"] >= 0.99 and s["fail"] == 0.0, fmt(s))

    print("\n2. the same null at noise x3, x5, x10, x20, x100: fail <= 12% (split reported, with "
          "the held shapes that carried a failing application)")
    for mult in (3, 5, 10, 20, 100):
        for i, (n, k, bitid) in enumerate(CONFIGS):
            s, held = split(SEED[f"x{mult}"] + i, n=n, k=k, bitid=bitid, mult=mult)
            check(f"x{mult} {label(n, k, bitid)}", s["fail"] <= 0.12,
                  f"{fmt(s)}; held with a failing application {held}/{TRIALS}")

    print("\n3. all candidates x(1+delta): 0.5% pass >= 99%, 1% pass >= 99%, 2% (= tau) "
          "report-only, 3% fail >= 99%, 4% fail >= 99%")
    for delta, want, thr in ((0.005, "pass", 0.99), (0.01, "pass", 0.99), (0.02, None, None),
                             (0.03, "fail", 0.99), (0.04, "fail", 0.99)):
        for i, (n, k, bitid) in enumerate(CONFIGS):
            s, held = split(SEED[f"d{delta}"] + i, n=n, k=k, bitid=bitid,
                            cand_scale=[1 + delta] * k)
            name = f"delta {delta:.1%} {label(n, k, bitid)}"
            if want is None:
                report(name, f"{fmt(s)}; held with a failing application {held}/{TRIALS} "
                       "(no target: delta = tau is the boundary by construction)")
            else:
                check(name, s[want] >= thr, fmt(s))

    print("\n4. one candidate x1.04 with two null (3v3 by construction): fail >= 99%")
    i = 0
    for n in (100, 50):
        for bitid in (False, True):
            s, _ = split(SEED["one1.04"] + i, n=n, k=3, bitid=bitid, cand_scale=[1.04])
            check(f"{label(n, 3, bitid)}", s["fail"] >= 0.99, fmt(s))
            i += 1

    print("\n5. step-1-only shift, bit-identical baseline forwards: x1.03 fail >= 99%, "
          "x1.01 pass >= 99%")
    for factor, want in ((1.03, "fail"), (1.01, "pass")):
        i = 0
        for n in (100, 50):
            for k in (3, 1):
                s, _ = split(SEED[f"step{factor}"] + i, n=n, k=k, bitid=True,
                             step1_scale=factor)
                check(f"x{factor} {label(n, k, True)}", s[want] >= 0.99, fmt(s))
                i += 1

    print("\n6. Gaussian single application, worst case over sigma/tau of P(fail | null) <= 4.5%")
    print("   (one step, so the first-step and all-steps sets are the same single application,"
          " read as (letter, first); letter = 1 + sigma*z in every arm, span constant)")
    for k in (3, 1):
        worst = (0.0, 0.0)
        grid = [round(0.05 * i, 2) for i in range(1, 61)]
        trials = 20_000
        for r in grid:
            rng = random.Random(10_000 + int(r * 100) + 1000 * k)
            sigma = r * pp.P2_TAU
            fails = 0
            for _ in range(trials):
                bases = [row(f"B{j}", "padding", [1.0 + rng.gauss(0.0, sigma)], [1.0])
                         for j in range(3)]
                cands = [row(f"M{i}", "none", [1.0 + rng.gauss(0.0, sigma)], [1.0])
                         for i in range(k)]
                out = pp.p2_screen(bases, cands)
                fails += out["channels"]["letter"]["sets"]["first"]["verdict"] == "fail"
            p = fails / trials
            if p > worst[1]:
                worst = (r, p)
        check(f"3v{k} worst P(fail|null)", worst[1] <= 0.045,
              f"{worst[1]:.2%} at sigma/tau = {worst[0]} ({trials} trials per grid point, "
              f"grid 0.05..3.00 step 0.05)")

    print()
    if FAILED:
        print(f"{len(FAILED)} target(s) MISSED -- not tuned, reported:")
        for f in FAILED:
            print(f"  {f}")
        return 1
    print("every calibration target met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
