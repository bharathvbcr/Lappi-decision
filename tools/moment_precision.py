"""Does keeping AdamW's second moment in bf16 actually degrade a run, or is that folklore?

``qd_train.memory`` asserts it:

    **This is the cheap option and the numerically poor one.** bf16 has 8 bits of mantissa,
    and `exp_avg_sq` accumulates over a whole run; keeping it in bf16 is how a long run
    degrades in a way no single step shows.

That is an assertion, and this repository's rule is that an assertion is not a measurement.
This script measures it, and it makes a falsifiable prediction first so that the measurement
can disagree.

**The prediction.** AdamW's second moment is an exponential moving average::

    v <- beta2 * v + (1 - beta2) * g**2        equivalently   v += (1 - beta2) * (g**2 - v)

With the torch default ``beta2 = 0.999`` the increment is one part in a thousand of the gap.
bfloat16 carries 8 mantissa bits (1 implicit + 7 stored), so its relative spacing is
``2**-8 = 3.906e-3``. A relative increment of ``1e-3`` is *below half* that spacing, so
round-to-nearest returns the original value unchanged: **the second moment stops moving**
once ``g**2`` is within about 0.4% of ``v``, no matter how many steps follow.

The first moment is not at risk: ``beta1 = 0.9`` gives an increment of one part in ten,
which is 25x bf16's spacing.

If the prediction holds, the failure is specific and nameable rather than general "bf16 is
imprecise": it is a **stalled second moment**, it is invisible in any single step, and it is
worst exactly where a run is converging -- when gradients stabilise and ``g**2`` approaches
``v``, which is the regime a long fine-tune spends most of its time in.

Run::

    PYTHONDONTWRITEBYTECODE=1 /Users/bharath/.venvs/ml/bin/python tools/moment_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - the repo venv has no torch by design
    raise SystemExit(
        "torch is not importable from this interpreter. Run this with\n"
        "    /Users/bharath/.venvs/ml/bin/python tools/moment_precision.py\n"
        "Refusing rather than reporting a vacuous pass."
    ) from exc

#: torch.optim.AdamW's defaults, which is what every optimizer in this repository builds.
BETA1 = 0.9
BETA2 = 0.999

#: Mantissa spacing of each dtype, as a relative epsilon.
EPS = {
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
    torch.float32: 2.0**-24,
}


def representable_increment(dtype: torch.dtype) -> float:
    """The smallest relative change this dtype can record at all."""
    return EPS[dtype] / 2.0  # round-to-nearest resolves half a spacing


def ema_stalls(dtype: torch.dtype, beta: float) -> bool:
    """Does a (1-beta) relative increment survive rounding in ``dtype``?"""
    return (1.0 - beta) < representable_increment(dtype)


def run_ema(dtype: torch.dtype, *, beta: float, steps: int, g_sq: float, v0: float) -> dict:
    """Drive the real EMA recurrence in ``dtype`` and report where it ends up.

    Uses torch ops on real tensors of the dtype rather than simulating the rounding, so the
    answer is the hardware's, not a model of it.
    """
    v = torch.tensor([v0], dtype=dtype)
    g2 = torch.tensor([g_sq], dtype=dtype)
    b = torch.tensor([beta], dtype=dtype)
    one_minus = torch.tensor([1.0 - beta], dtype=dtype)

    first_move_at = None
    frozen_since = 0
    for i in range(steps):
        prev = v.clone()
        v = b * v + one_minus * g2
        if not torch.equal(v, prev):
            if first_move_at is None:
                first_move_at = i
            frozen_since = 0
        else:
            frozen_since += 1

    exact = v0
    for _ in range(steps):
        exact = beta * exact + (1.0 - beta) * g_sq

    got = float(v.item())
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "start": v0,
        "target": g_sq,
        "exact_after_steps": exact,
        "reached": got,
        "absolute_error": abs(got - exact),
        "relative_error": abs(got - exact) / abs(exact) if exact else float("nan"),
        "moved_at_all": first_move_at is not None,
        "frozen_for_last_n_steps": frozen_since,
    }


def main() -> int:
    print("AdamW moment precision: is the bf16 second moment actually lossy?")
    print("=" * 78)
    print()
    print("PREDICTION, stated before measuring so the measurement can refute it:")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        eps = EPS[dtype]
        inc = representable_increment(dtype)
        b2 = ema_stalls(dtype, BETA2)
        b1 = ema_stalls(dtype, BETA1)
        name = str(dtype).replace("torch.", "")
        exponent = round(torch.log2(torch.tensor(eps)).item())
        print(f"  {name:9s} relative spacing 2^{exponent:<3d} = {eps:.3e}   "
              f"smallest recordable relative change {inc:.3e}")
        print(f"            beta2=0.999 -> increment 1.0e-03: "
              f"{'STALLS' if b2 else 'survives':8s}   "
              f"beta1=0.9 -> increment 1.0e-01: {'STALLS' if b1 else 'survives'}")
    print()

    # The regime a converging run lives in: v already near g**2. The EMA should close the
    # remaining gap and, in bf16, should not.
    print("MEASURED: the EMA v <- beta2*v + (1-beta2)*g^2, driven on real tensors.")
    print()
    cases = [
        ("converging (v within 0.4% of g^2)", 1.0, 1.004),
        ("converged  (v within 0.1% of g^2)", 1.0, 1.001),
        ("far from target (g^2 = 2x v)     ", 1.0, 2.0),
    ]
    steps = 5000
    for label, v0, g_sq in cases:
        print(f"  {label}   {steps} steps")
        for dtype in (torch.bfloat16, torch.float32):
            r = run_ema(dtype, beta=BETA2, steps=steps, g_sq=g_sq, v0=v0)
            verdict = "NEVER MOVED" if not r["moved_at_all"] else (
                f"frozen for last {r['frozen_for_last_n_steps']}" if r["frozen_for_last_n_steps"]
                else "still moving"
            )
            print(f"    {r['dtype']:9s} start {r['start']:.6f} -> reached {r['reached']:.6f}  "
                  f"(exact {r['exact_after_steps']:.6f}, rel err {r['relative_error']:.3e})  "
                  f"{verdict}")
        print()

    # What the stall does to the step size AdamW actually takes. The update is
    # m_hat / (sqrt(v_hat) + eps); a v that is wrong by a factor f scales the step by
    # 1/sqrt(f).
    print("WHAT IT COSTS: AdamW's step is m_hat / (sqrt(v_hat) + eps), so an exp_avg_sq")
    print("wrong by a factor f mis-scales every subsequent step by 1/sqrt(f).")
    r_bf = run_ema(torch.bfloat16, beta=BETA2, steps=steps, g_sq=2.0, v0=1.0)
    f = r_bf["reached"] / r_bf["exact_after_steps"]
    print(f"  after {steps} steps at g^2 = 2x v: bf16 v is {f:.4f}x the exact v,")
    print(f"  so the step is mis-scaled by {1 / f**0.5:.4f}x")
    print()

    # Where a run actually starts: exp_avg_sq is zero-initialised, so the early increments
    # are enormous in RELATIVE terms and represent fine. The freeze is not immediate -- it
    # is what the EMA converges INTO, which is why no single early step shows it.
    print("WHERE IT FREEZES from AdamW's real zero initialisation (g^2 = 1.0):")
    for dtype in (torch.bfloat16, torch.float32):
        v = torch.zeros(1, dtype=dtype)
        g2 = torch.tensor([1.0], dtype=dtype)
        b = torch.tensor([BETA2], dtype=dtype)
        om = torch.tensor([1.0 - BETA2], dtype=dtype)
        froze_at = None
        for i in range(20000):
            prev = v.clone()
            v = b * v + om * g2
            if torch.equal(v, prev):
                if froze_at is None:
                    froze_at = i
            else:
                froze_at = None
        name = str(dtype).replace("torch.", "")
        where = f"froze at step {froze_at}" if froze_at is not None else "never froze"
        print(f"  {name:9s} after 20000 steps v = {float(v.item()):.6f} (target 1.0), {where}")
    print()

    # The consequence that matters for a real run: gradient scale CHANGES over training --
    # warmup, decay, a new data regime. A frozen second moment cannot follow it.
    print("CAN A FROZEN SECOND MOMENT RECOVER when the gradient scale changes?")
    print("  v is first driven to its bf16 resting value at g^2 = 1.0, then g^2 jumps:")
    for jump in (2.0, 10.0, 100.0):
        v = torch.zeros(1, dtype=torch.bfloat16)
        b = torch.tensor([BETA2], dtype=torch.bfloat16)
        om = torch.tensor([1.0 - BETA2], dtype=torch.bfloat16)
        g2 = torch.tensor([1.0], dtype=torch.bfloat16)
        for _ in range(20000):
            v = b * v + om * g2
        settled = float(v.item())
        g2 = torch.tensor([jump], dtype=torch.bfloat16)
        for _ in range(20000):
            v = b * v + om * g2
        after = float(v.item())
        mis = (after / jump) ** -0.5
        print(f"    g^2: 1.0 -> {jump:6.1f}   v: {settled:.4f} -> {after:.4f} "
              f"(should reach {jump:.1f})   step mis-scaled {mis:.2f}x")
    print()

    # Against torch's own AdamW, not this script's recurrence -- so the claim is about the
    # optimizer this repository instantiates, not about a model of it.
    print("AGAINST torch.optim.AdamW ITSELF, on a bf16 parameter:")
    for dtype in (torch.bfloat16, torch.float32):
        p = torch.nn.Parameter(torch.ones(1024, dtype=dtype))
        opt = torch.optim.AdamW([p], lr=0.0, betas=(BETA1, BETA2))  # lr=0: isolate the state
        g = torch.full_like(p, 1.0)
        for _ in range(20000):
            p.grad = g.clone()
            opt.step()
        v_settled = float(opt.state[p]["exp_avg_sq"][0].item())
        # Now a 100x gradient-magnitude jump (g -> 10 means g^2 -> 100).
        g = torch.full_like(p, 10.0)
        for _ in range(20000):
            p.grad = g.clone()
            opt.step()
        v_after = float(opt.state[p]["exp_avg_sq"][0].item())
        name = str(dtype).replace("torch.", "")
        print(f"  {name:9s} exp_avg_sq settled {v_settled:9.4f} (target 1.0), "
              f"after g^2 -> 100: {v_after:9.4f} (target 100.0)")
    print()

    stalls = ema_stalls(torch.bfloat16, BETA2)
    moved = run_ema(torch.bfloat16, beta=BETA2, steps=steps, g_sq=1.001, v0=1.0)["moved_at_all"]
    print("=" * 78)
    if stalls and not moved:
        print("VERDICT: the prediction holds. bf16 exp_avg_sq stalls at beta2=0.999 in the")
        print("converged regime -- the increment is below the dtype's rounding threshold, so")
        print("the second moment stops updating while the run continues. memory.py's claim is")
        print("measured, not folklore, and the fp32-master recipe is worth implementing.")
        return 0
    print("VERDICT: the prediction FAILED. bf16 exp_avg_sq moved where it was predicted to")
    print("stall. memory.py's warning does not describe this dtype pair, and the case for the")
    print("fp32-master recipe has to be made on some other evidence.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
