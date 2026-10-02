"""Box smoke for the rebuilt overlay: the amended aggregation on three synthetic screens."""

import sys

sys.path.insert(0, "/home/ubuntu/perf/overlay-nomask/tools")
import perf_parity as pp

N = 10


def arm(tag, mask, scale_l=lambda t: 1.0, scale_s=lambda t: 1.0, jit_l=0.0, jit_s=0.0, s=0.0):
    letter = [(2.0 - 0.1 * t) * scale_l(t) * (1 + (s * jit_l if t else 0.0)) for t in range(N)]
    span = [(1.0 - 0.05 * t) * scale_s(t) * (1 + (s * jit_s if t else 0.0)) for t in range(N)]
    return {"tag": tag, "steps": N, "consumed_digest": "c" * 64, "termination": "steps_exhausted",
            "train_attention_mask": mask, "train_path": {}, "letter": [x.hex() for x in letter],
            "span": [x.hex() for x in span]}


def bases(jit_l=1e-4, jit_s=1e-4):
    return [arm(f"A-mask-{k}", "padding", jit_l=jit_l, jit_s=jit_s, s=s)
            for k, s in enumerate((-1.0, 0.0, 1.0), 1)]


def cands(**kw):
    return [arm(f"A-none-{i}", "none", **kw) for i in (1, 2, 3)]


cases = [
    ("all-conclusive-one-fail", bases(), cands(scale_l=lambda t: 1.03 if t == 0 else 1.0),
     "fail", [["letter", "first"]]),
    ("one-fail-one-inconclusive", bases(jit_l=0.02),
     cands(scale_s=lambda t: 1.03 if t == 0 else 1.0), "inconclusive", [["span", "first"]]),
    ("pass", bases(), cands(), "pass", []),
]
bad = 0
for name, b, c, verdict, fails in cases:
    out = pp.p2_screen(b, c)
    ok = out["verdict"] == verdict and out["fail_applications"] == fails
    bad += not ok
    print(f"{name}: verdict {out['verdict']} fail_applications {out['fail_applications']} "
          f"-> {'ok' if ok else 'WRONG'}")
print(f"python {sys.version.split()[0]}; {len(cases) - bad}/{len(cases)} ok")
raise SystemExit(1 if bad else 0)
