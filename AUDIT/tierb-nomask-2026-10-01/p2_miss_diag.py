"""Why three p2_null_sim.py v2 targets were missed: which (channel, step set) applications decide.

Diagnostic only; it changes nothing in the rule or the calibration. It replays the three missed
configurations with p2_null_sim.py's own seeds (so its counts reproduce that run's) and records,
per trial, the verdict of each of the four applications (letter/first, letter/all, span/first,
span/all) and the shape verdict that p2_screen aggregated from them.

Seeds follow p2_null_sim.main(): section 1 uses 1..8; section 2 uses 9..16 (x3), 17..24 (x10),
25..32 (x20) and 33..40 (x100); section 3 uses 41..48 (delta 0.5%) and 49..56 (delta 2%),
in CONFIGS order.
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

import p2_null_sim as sim  # noqa: E402
import perf_parity as pp  # noqa: E402

APPS = (("letter", "first"), ("letter", "all"), ("span", "first"), ("span", "all"))


def replay(seed: int, label: str, *, n: int, k: int, bitid: bool, mult: float = 1.0,
           cand_scale: tuple[float, ...] = ()) -> None:
    rng = random.Random(seed)
    combos: collections.Counter = collections.Counter()
    fails_per_app: collections.Counter = collections.Counter()
    shape: collections.Counter = collections.Counter()
    scales = list(cand_scale) + [1.0] * (k - len(cand_scale))
    for _ in range(sim.TRIALS):
        bases = [sim.row(f"B{j}", "padding", *sim.series(rng, n, mult, bitid)) for j in range(3)]
        cands = [sim.row(f"M{i}", "none", *sim.series(rng, n, mult, bitid, scales[i]))
                 for i in range(k)]
        out = pp.p2_screen(bases, cands)
        apps = tuple(out["channels"][ch]["sets"][s]["verdict"] for ch, s in APPS)
        combos[(apps, out["verdict"])] += 1
        shape[out["verdict"]] += 1
        for (ch, s), v in zip(APPS, apps, strict=True):
            if v == "fail":
                fails_per_app[f"{ch}/{s}"] += 1
    print(label)
    print(f"  shape verdicts: {dict(shape)}")
    print("  trials in which each application failed: "
          + ", ".join(f"{ch}/{s} {fails_per_app[f'{ch}/{s}']}" for ch, s in APPS))
    print("  application verdicts (letter/first, letter/all, span/first, span/all) -> shape:")
    for (apps, v), c in sorted(combos.items(), key=lambda kv: -kv[1]):
        print(f"    {c:4d}  {' '.join(f'{a:12s}' for a in apps)} -> {v}")
    print(flush=True)


def main() -> int:
    replay(25, "x20 n=100 3v3 noisy step 0 (seed 25)", n=100, k=3, bitid=False, mult=20)
    replay(51, "delta 2% n=100 3v1 noisy step 0 (seed 51)", n=100, k=1, bitid=False,
           cand_scale=(1.02,))
    replay(55, "delta 2% n=50 3v1 noisy step 0 (seed 55)", n=50, k=1, bitid=False,
           cand_scale=(1.02,))
    print("float check, a 2% shift at a noiseless step 0 (the bit-identical delta 2% rows):")
    for name, level in (("letter", 2.0), ("span", 0.97**0 + 0.1)):
        dev = (level * 1.02 - level) / level
        print(f"  {name}: ({level!r} * 1.02 - {level!r}) / {level!r} = {dev!r} > tau: "
              f"{dev > pp.P2_TAU}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
