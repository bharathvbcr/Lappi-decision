#!/usr/bin/env python3
"""Export .npy parity fixtures for the PUBLISHED gated delta rule.

Reference operator: ``nanolab.mixers.gdn_chunked(..., rule="published")`` from
/Users/bharath/Code/research/MLSystemsLab/nanolab/mixers.py (line 684).

WHY ``published`` IS SPELLED IN EVERY FILENAME
----------------------------------------------
``gdn_chunked``'s signature default is ``rule="repo"`` (mixers.py:684) and the
``GatedDeltaNet`` docstring (mixers.py:771) states that *every committed GDN run
used* ``repo``, which is NOT the operator of arXiv:2412.06464. The two rules
differ only in which state the delta correction reads:

    repo:       e_t = v_t -         S_{t-1} k_t
    published:  e_t = v_t - alpha_t S_{t-1} k_t     (eq. 8)

A fixture that does not name its rule is therefore ambiguous between two
operators that produce different numbers on the same inputs. This script asserts
that they differ (``_assert_rules_differ``) so the naming is not decorative.

WHAT IS EXPORTED, PER CASE
--------------------------
    inputs   q, k, v        float32 [B, H, L, D]
             alpha, beta    float32 [B, H, L]        (RAW - see CLAMPS below)
    goldens  y_chunked      float32 [B, H, L, D]     gdn_chunked(rule="published")
             y_seq_f64      float64 [B, H, L, D]     independent fp64 sequential

``y_seq_f64`` is the golden a Rust f64 sequential reference should match tightly
(it is computed in float64 from the same clamped inputs). ``y_chunked`` is the
actual fp32 kernel output and carries fp32 chunk-parallel rounding; match it
loosely. Each case also emits a ``*_meta.json`` naming B/H/L/D, the chunk size,
the rule, and the observed max |y_chunked - y_seq_f64|.

CLAMPS ARE PART OF THE OPERATOR, NOT A DETAIL
---------------------------------------------
``gdn_chunked`` clamps its own inputs before anything else (mixers.py:712-713):

    alpha = alpha.float().clamp(min=1e-4, max=1.0)
    beta  = beta.float().clamp(min=0.0,  max=1.0)

alpha is clamped because the kernel takes ``log(alpha)``; the comment at
mixers.py:709-711 records the failure it prevents (a real GH200 run whose loss
stayed finite to step 50 and whose gnorm blew at 55). The RAW alpha/beta are
exported, so a reimplementation that omits the clamp will diverge - the
``_tinyalpha`` case exists to make that divergence unmissable (alpha = 1e-8,
five orders of magnitude below the floor).

RUN
---
    PYTHONPATH=/Users/bharath/Code/research/MLSystemsLab \
        /Users/bharath/.venvs/ml/bin/python \
        /Users/bharath/Code/research/qwen-decision/tools/gen_gdn_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

NANOLAB = pathlib.Path("/Users/bharath/Code/research/MLSystemsLab")
OUT = pathlib.Path("/Users/bharath/Code/research/tessl/tests/fixtures/gdn")

if str(NANOLAB) not in sys.path:
    sys.path.insert(0, str(NANOLAB))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nanolab.mixers import gdn_chunked  # noqa: E402

# Mirror of mixers.py:712-713. Named so the doc can cite one place.
ALPHA_MIN, ALPHA_MAX = 1e-4, 1.0
BETA_MIN, BETA_MAX = 0.0, 1.0


def _clamp(alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly mixers.py:712-713."""
    return (alpha.clamp(min=ALPHA_MIN, max=ALPHA_MAX),
            beta.clamp(min=BETA_MIN, max=BETA_MAX))


def seq_published_kernel_orientation(q, k, v, alpha, beta):
    """fp64 sequential published rule, in ``gdn_chunked``'s state orientation.

    S has shape [B, H, D_v, D_k] - value index first, key index second - which is
    the orientation the kernel builds (mixers.py:758 accumulates
    ``dS[d,e] = sum_c u[c,d] k[c,e]``, i.e. u in the value slot, k in the key
    slot) and the orientation ``S @ k -> value`` reads naturally.

        pred_t = S_{t-1} k_t                    [B,H,D_v]
        e_t    = v_t - alpha_t * pred_t         PUBLISHED: decayed read
        u_t    = beta_t * e_t
        S_t    = alpha_t * S_{t-1} + u_t k_t^T
        y_t    = S_t q_t

    Note y_t reads the state AFTER the write at t (mixers.py:835 appends after
    the ``S =`` assignment); the write is visible to its own query.
    """
    q, k, v = (t.double() for t in (q, k, v))
    alpha, beta = (t.double() for t in (alpha, beta))
    alpha, beta = _clamp(alpha, beta)
    B, H, L, D = k.shape
    S = torch.zeros(B, H, D, D, dtype=torch.float64)
    ys = []
    for t in range(L):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]          # [B,H,D]
        at = alpha[:, :, t][..., None]                            # [B,H,1]
        bt = beta[:, :, t][..., None]
        pred = torch.einsum("bhde,bhe->bhd", S, kt)               # S k_t
        u = bt * (vt - at * pred)                                 # PUBLISHED
        S = at[..., None] * S + torch.einsum("bhd,bhe->bhde", u, kt)
        ys.append(torch.einsum("bhde,bhe->bhd", S, qt))           # S q_t
    return torch.stack(ys, dim=2)


def seq_published_transposed_orientation(q, k, v, alpha, beta):
    """fp64 sequential published rule, transcribed from ``GatedDeltaNet._sequential``
    (mixers.py:816-837), which carries S in the OPPOSITE orientation
    [B,H,D_k,D_v] and is documented there as "the transpose of the kernel's
    state; outputs are identical".

    Existing only as a cross-check: if this and the function above disagree, one
    of the two transcriptions is wrong and no fixture should be written.
    """
    q, k, v = (t.double() for t in (q, k, v))
    alpha, beta = (t.double() for t in (alpha, beta))
    alpha, beta = _clamp(alpha, beta)
    B, H, L, D = k.shape
    S = torch.zeros(B, H, D, D, dtype=torch.float64)
    ys = []
    for t in range(L):
        kt, vt = k[:, :, t], v[:, :, t]
        at = alpha[:, :, t][..., None, None]
        bt = beta[:, :, t][..., None]
        pred = torch.einsum("bhpn,bhp->bhn", S, kt)
        pred = alpha[:, :, t][..., None] * pred            # PUBLISHED: decayed read
        delta = (vt - pred) * bt
        S = at * S + torch.einsum("bhp,bhn->bhpn", kt, delta)
        ys.append(torch.einsum("bhpn,bhp->bhn", S, q[:, :, t]))
    return torch.stack(ys, dim=2)


def _rand(shape, gen, scale=1.0):
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale)


def make_inputs(B, H, L, D, seed, *, l2_normalize_k=True, alpha_mode="sigmoid",
                h_kv=None):
    """Draw one case's inputs at a fixed seed.

    ``l2_normalize_k`` mirrors ``GatedDeltaNet._project`` (mixers.py:801), which
    L2-normalizes keys along the head dim before the scan. ``gdn_chunked`` itself
    does NOT normalize - it is the caller's job - so the fixtures ship keys that
    are already unit-norm, matching how the module drives the kernel.

    ``h_kv``, when set, draws only ``h_kv`` key heads and widens them to ``H`` by
    ``repeat_interleave`` (the repo's own K/V-widening convention, mixers.py:316
    and :458). See the GQA note in the audit document - ``gdn_chunked`` has no
    key-head concept of its own.
    """
    gen = torch.Generator().manual_seed(seed)
    q = _rand((B, H, L, D), gen)
    v = _rand((B, H, L, D), gen)
    k_src_heads = h_kv if h_kv is not None else H
    k_narrow = _rand((B, k_src_heads, L, D), gen)
    if l2_normalize_k:
        k_narrow = torch.nn.functional.normalize(k_narrow, dim=-1)
    if h_kv is not None:
        assert H % h_kv == 0, "H must be a multiple of h_kv"
        k = k_narrow.repeat_interleave(H // h_kv, dim=1)   # INTERLEAVE, not tile
    else:
        k = k_narrow

    if alpha_mode == "sigmoid":
        # matches GatedDeltaNet._project (mixers.py:803): sigmoid of a gate, then
        # the same clamp the kernel re-applies.
        alpha = torch.sigmoid(_rand((B, H, L), gen))
    elif alpha_mode == "tiny":
        # far below ALPHA_MIN: only a reimplementation that clamps survives this.
        alpha = torch.full((B, H, L), 1e-8, dtype=torch.float32)
    else:
        raise ValueError(alpha_mode)
    beta = torch.sigmoid(_rand((B, H, L), gen))
    return q, k, v, alpha, beta, k_narrow


def _assert_rules_differ(q, k, v, alpha, beta, chunk, name):
    """A fixture whose two rules coincide pins nothing. Refuse to ship one.

    EXCEPT at L == 1, where they provably cannot differ: the only correction is
    at t = 0, where S_{-1} = 0, so ``alpha_0 * S_{-1} k_0 == S_{-1} k_0 == 0``
    and the two rules are the same function. L=1 is kept because it exercises the
    maximal tail pad (``pad = (-1) % 32 = 31``), not because it discriminates the
    rule - and the exact-zero assertion below pins that invariant rather than
    letting a coincidence pass as a check.
    """
    L = k.shape[2]
    y_pub = gdn_chunked(q, k, v, alpha, beta, chunk=chunk, rule="published")
    y_repo = gdn_chunked(q, k, v, alpha, beta, chunk=chunk, rule="repo")
    gap = (y_pub - y_repo).abs().max().item()
    scale = y_pub.abs().max().item()
    if L == 1:
        if gap != 0.0:
            raise SystemExit(
                f"[{name}] L=1 must be rule-invariant (S_{{-1}} = 0) but the two "
                f"rules differ by {gap:.3e}")
        return gap
    if not (gap > 1e-6 * max(scale, 1.0)):
        raise SystemExit(
            f"[{name}] published and repo rules agree to {gap:.3e} (scale "
            f"{scale:.3e}); this fixture would not distinguish them")
    return gap


CASES = [
    # name,            B, H,  L,    D,  chunk, seed, h_kv, alpha_mode
    ("L1",             2, 3,     1,  8,  32, 1001, None, "sigmoid"),
    ("L63",            2, 3,    63,  8,  32, 1002, None, "sigmoid"),
    ("L64",            2, 3,    64,  8,  32, 1003, None, "sigmoid"),
    ("L65",            2, 3,    65,  8,  32, 1004, None, "sigmoid"),
    ("L127",           2, 3,   127,  8,  32, 1005, None, "sigmoid"),
    ("L129",           2, 3,   129,  8,  32, 1006, None, "sigmoid"),
    ("L1000",          1, 2,  1000,  8,  32, 1007, None, "sigmoid"),
    ("L8191",          1, 2,  8191,  8,  32, 1008, None, "sigmoid"),
    # chunk sweep: the chunk size must not change the answer
    ("L127c64",        2, 3,   127,  8,  64, 1005, None, "sigmoid"),
    ("L127c16",        2, 3,   127,  8,  16, 1005, None, "sigmoid"),
    # clamp probe: alpha five orders of magnitude below the 1e-4 floor
    ("L65_tinyalpha",  2, 3,    65,  8,  32, 1009, None, "tiny"),
    # FEWER KEY HEADS THAN VALUE HEADS, widened by repeat_interleave
    ("L64_gqa_kv2",    2, 6,    64,  8,  32, 1010,    2, "sigmoid"),
    ("L129_gqa_kv3",   2, 6,   129,  8,  32, 1011,    3, "sigmoid"),
]


def main() -> int:
    torch.use_deterministic_algorithms(False)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- transcription cross-check before any file is written -----------------
    gen_chk = torch.Generator().manual_seed(7)
    qc = _rand((2, 2, 37, 8), gen_chk)
    kc = torch.nn.functional.normalize(_rand((2, 2, 37, 8), gen_chk), dim=-1)
    vc = _rand((2, 2, 37, 8), gen_chk)
    ac = torch.sigmoid(_rand((2, 2, 37), gen_chk))
    bc = torch.sigmoid(_rand((2, 2, 37), gen_chk))
    d_orient = (seq_published_kernel_orientation(qc, kc, vc, ac, bc)
                - seq_published_transposed_orientation(qc, kc, vc, ac, bc)
                ).abs().max().item()
    if d_orient > 1e-10:
        raise SystemExit(
            f"the two fp64 transcriptions disagree by {d_orient:.3e}; one of them "
            "misreads mixers.py - refusing to write fixtures")
    print(f"[selfcheck] fp64 transcriptions agree to {d_orient:.3e} "
          "(kernel vs _sequential state orientation)")

    manifest = {
        "rule": "published",
        "source": "nanolab.mixers.gdn_chunked (research/MLSystemsLab/nanolab/mixers.py:684)",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "alpha_clamp": [ALPHA_MIN, ALPHA_MAX],
        "beta_clamp": [BETA_MIN, BETA_MAX],
        "orientation_selfcheck_max_abs": d_orient,
        "cases": [],
    }

    for name, B, H, L, D, chunk, seed, h_kv, alpha_mode in CASES:
        q, k, v, alpha, beta, k_narrow = make_inputs(
            B, H, L, D, seed, alpha_mode=alpha_mode, h_kv=h_kv)

        rule_gap = _assert_rules_differ(q, k, v, alpha, beta, chunk, name)

        y_chunked = gdn_chunked(q, k, v, alpha, beta, chunk=chunk, rule="published")
        y_seq = seq_published_kernel_orientation(q, k, v, alpha, beta)

        assert y_chunked.shape == (B, H, L, D), (name, y_chunked.shape)
        assert torch.isfinite(y_chunked).all(), f"{name}: non-finite chunked output"
        assert torch.isfinite(y_seq).all(), f"{name}: non-finite fp64 output"

        parity = (y_chunked.double() - y_seq).abs().max().item()
        scale = y_seq.abs().max().item()

        stem = f"gdn_published_{name}"
        arrays = {
            "q": q.numpy(),
            "k": k.numpy(),
            "v": v.numpy(),
            "alpha": alpha.numpy(),
            "beta": beta.numpy(),
            "y_chunked": y_chunked.detach().numpy(),
            "y_seq_f64": y_seq.numpy(),
        }
        if h_kv is not None:
            # the pre-widening keys, so a reimplementation can verify its own
            # repeat_interleave rather than trusting ours
            arrays["k_narrow"] = k_narrow.numpy()

        written = []
        for field, arr in arrays.items():
            path = OUT / f"{stem}_{field}.npy"
            np.save(path, np.ascontiguousarray(arr))
            written.append(path.name)

        meta = {
            "name": name,
            "rule": "published",
            "B": B, "H": H, "L": L, "D": D,
            "chunk": chunk,
            "seed": seed,
            "h_kv": h_kv,
            "kv_expansion": "repeat_interleave" if h_kv is not None else None,
            "alpha_mode": alpha_mode,
            "k_l2_normalized": True,
            "alpha_clamp": [ALPHA_MIN, ALPHA_MAX],
            "beta_clamp": [BETA_MIN, BETA_MAX],
            "max_abs_chunked_vs_f64seq": parity,
            "max_abs_output": scale,
            "max_abs_published_vs_repo": rule_gap,
            "files": written,
        }
        (OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        manifest["cases"].append(meta)
        print(f"[{name:15s}] B{B} H{H} L{L} D{D} c{chunk}"
              f"{f' kv{h_kv}' if h_kv else ''}  "
              f"parity={parity:.3e}  scale={scale:.3e}  pub-vs-repo={rule_gap:.3e}")

    (OUT / "gdn_published_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    # checksum every file so a silently-regenerated fixture is detectable
    sums = []
    for p in sorted(OUT.glob("gdn_published_*")):
        if p.name == "gdn_published_SHA256SUMS":
            continue
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT / "gdn_published_SHA256SUMS").write_text("\n".join(sums) + "\n")

    n_npy = len(list(OUT.glob("gdn_published_*.npy")))
    print(f"\nwrote {n_npy} .npy + {len(CASES)} meta + manifest + sha256sums to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
