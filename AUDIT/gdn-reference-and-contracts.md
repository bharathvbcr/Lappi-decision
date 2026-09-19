# GDN Golden Reference & Product Contracts

**Lane:** reference-algorithm & product-contract
**Date:** 2026-09-19
**Scope:** specification + fixture generation only. No production code was written.

Navigation used: DevMap (`repo_path=/Users/bharath/Code`, `repository.root` confirmed
`/Users/bharath/Code`, generation 25, 180,660 nodes / 602,286 edges, `is_fresh: true`,
`pending_count: 0`, `degraded_reason: null`) and GitPulse Insights per-repo. Every
`devmap_search` result carried a `walk_incomplete` notice; see **GAPS**.

Claims are labelled **[V]** verified (I ran it or read the line), **[I]** inferred,
**[U]** unverified.

---

## 0. Executive summary

| Question | Answer |
|---|---|
| Published recurrence | `e_t = v_t − α_t·S_{t−1}k_t`; `S_t = α_t S_{t−1} + β_t e_t k_tᵀ`; `y_t = S_t q_t` **[V]** |
| Fixtures generated? | Yes — **93 `.npy`** + 13 `_meta.json` + manifest + SHA256SUMS = **108 files**, all with `published` in the name **[V]** |
| Plan K7 α formula | **REFUTED.** α is a **sigmoid**, not `exp(−exp(A_log)·softplus(·))` **[V]** |
| α ≥ 1e-4 NaN guard | **PRESENT**, in two places, not absent **[V]** |
| K-head expansion | Repo convention is **`repeat_interleave` (INTERLEAVE)** — but GDN itself has **no** K-head concept **[V]** |
| "dcverify owns PASS" | **REFUTED.** There is no `PASS` token anywhere; the **Go host** derives `passed` **[V]** |
| DevType "no query text" | **Confirmed for the schema, qualified at the key** — some command IDs *are* query text, guarded only at the call site **[V]** |

---

# JOB 1 — THE GDN GOLDEN REFERENCE

## 1. Provenance and the `published` naming rule

| Artifact | Absolute path | Lines |
|---|---|---|
| `gdn_chunked` | `/Users/bharath/Code/research/MLSystemsLab/nanolab/mixers.py` | 684–760 |
| `GatedDeltaNet` | same | 763–837 |
| `GatedDeltaNet._project` | same | 793–807 |
| `GatedDeltaNet._sequential` | same | 816–837 |
| `gdn_chunked_matches_sequential` | `.../nanolab/tests.py` | 157–176 |
| `gdn_rules_differ_as_documented` | `.../nanolab/tests.py` | 180–201 |
| `gdn_chunked_survives_odd_lengths_and_tiny_gates` | `.../nanolab/tests.py` | 1958–1970 |
| `gdn_rule` config default | `.../nanolab/config.py` | 148, 282–283 |

The signature default is `repo`, not `published` **[V]**:

```python
# mixers.py:684
def gdn_chunked(q, k, v, alpha, beta, chunk=32, rule="repo"):
```

```python
# config.py:148
    gdn_rule: str = "repo"       # repo | published
```

And the class docstring states the record plainly **[V]**:

> ```
> mixers.py:771-773
>     Every committed GDN run used `repo`; it is NOT the published operator (the
>     own-key transition of a stored association is alpha-beta rather than
>     alpha(1-beta)). Verified against `_sequential` for both rules in nanolab.tests.
> ```

This is why plan rule 9 holds: an unlabelled GDN fixture is ambiguous between two
operators that give different numbers on the same inputs. Every fixture file emitted
by this lane spells `published`. **[V]**

---

## 2. THE SEQUENTIAL RECURRENCE (normative)

Transcribed from `_sequential` (mixers.py:816–837), then re-derived in the kernel's
own state orientation and cross-checked numerically (§6.1).

### 2.1 State orientation

There are **two orientations in the repo** and they are transposes of each other.
`_sequential` says so itself (mixers.py:817–818): *"``S`` here is the transpose of the
kernel's state; outputs are identical."* **[V]**

- **Kernel orientation** (what `gdn_chunked` builds, and what a Rust reference should
  use): `S ∈ ℝ^{D_v × D_k}`, **value index first**. `S·k → value`.
  Evidence: mixers.py:758 accumulates `dS[d,e] = Σ_c u[c,d]·k[c,e]` — `u` (a value) in
  slot `d`, `k` in slot `e`. **[V]**
- **`_sequential` orientation**: `S ∈ ℝ^{D_k × D_v}`, key index first
  (mixers.py:830 `einsum("bhpn,bhp->bhn", S, kt)`). **[V]**

**Use the kernel orientation.** Everything below is in it.

### 2.2 The recurrence

Inputs per `(b, h)`: `q_t, k_t, v_t ∈ ℝ^D` for `t = 0 … L−1`; scalars `α_t, β_t`.
Here `D_k = D_v = D` (§4.3).

**Step 0 — clamp (part of the operator, not a detail):**

```
α_t ← min(max(α_t, 1e-4), 1.0)
β_t ← min(max(β_t, 0.0),  1.0)
```

**Step 1 — initial state:** `S_{−1} = 0 ∈ ℝ^{D×D}`.

**Step 2 — for t = 0 … L−1, in this exact order:**

```
pred_t = S_{t−1} · k_t                      ∈ ℝ^D        # contract S's 2nd index with k
e_t    = v_t − α_t · pred_t                 ∈ ℝ^D        # PUBLISHED: decayed read
u_t    = β_t · e_t                          ∈ ℝ^D        # the "write vector"
S_t    = α_t · S_{t−1} + u_t k_tᵀ           ∈ ℝ^{D×D}    # outer product, u in the value slot
y_t    = S_t · q_t                          ∈ ℝ^D        # reads the POST-write state
```

Elementwise, with no ambiguity:

```
pred_t[d] = Σ_e S_{t−1}[d,e] · k_t[e]
S_t[d,e]  = α_t · S_{t−1}[d,e] + u_t[d] · k_t[e]
y_t[d]    = Σ_e S_t[d,e] · q_t[e]
```

**The `repo` rule is the same with `α_t` deleted from the `e_t` line only.** It is the
sole difference in the sequential form. Source (mixers.py:830–835) **[V]**:

```python
pred = torch.einsum("bhpn,bhp->bhn", S, kt)
if self.rule == "published":
    pred = alpha[:, t].unsqueeze(-1) * pred      # read the DECAYED state
delta = (vt - pred) * bt
S = at * S + torch.einsum("bhp,bhn->bhpn", kt, delta)
ys.append(torch.einsum("bhpn,bhp->bhn", S, q[:, t]))
```

### 2.3 Three semantics that are easy to get wrong

1. **`y_t` reads the state AFTER the write at `t`.** `ys.append` follows the `S =`
   assignment (mixers.py:834→835). A write is visible to its own query. In the chunked
   form the same fact appears as `Wy = (...).tril(0)` — **inclusive** diagonal
   (mixers.py:742). **[V]**
2. **`α_t` multiplies `S_{t−1}` in the state update under BOTH rules.** The rule
   switch changes only the *correction read*. **[V]**
3. **`e_t` is the residual, `u_t = β_t e_t` is the write.** `β` scales the correction,
   not the decay. **[V]**

### 2.4 Analytic anchor (hand-checkable Rust unit test)

`tests.py:180–201` pins the rules with closed-form numbers. `D=4`;
`q = k = v = e_0` (unit first coordinate) at both of `L=2` positions **[V]**:

| α | β | `repo` y | `published` y |
|---|---|---|---|
| 0.5 | 0.5 | `[0.5, 0.5]` | `[0.5, 0.625]` |
| 1.0 | 0.5 | identical to published | identical to repo |
| 0.5 | 0.0 | identical to published | identical to repo |
| 0.3 | 0.9 | `[0.9, 0.9(α−β)+0.9] = [0.9, 0.36]` | `[0.9, 0.9·α(1−β)+0.9] = [0.9, 0.927]` |

I re-derived all four rows from §2.2 by hand and they match. **[V]** The last row is the
"own-key transition" the class docstring cites: under `repo` it is `α−β = −0.6`
(a **sign flip** on the stored association), under `published` it is `α(1−β) = +0.03`.

**Degenerate cases where the rules coincide** (all provable from §2.2, all confirmed by
`tests.py:196-197` and by fixture `L1`):
- `α ≡ 1` (no decay → `α·pred = pred`);
- `β ≡ 0` (no write → state stays 0);
- **`L = 1`** — at `t=0`, `S_{−1}=0` so `α_0·S_{−1}k_0 = S_{−1}k_0 = 0`. **A single-token
  fixture cannot discriminate the rules.** The generator asserts this gap is *exactly*
  zero rather than letting the coincidence read as a check. **[V]**

---

## 3. THE CHUNK-PARALLEL (WY / UT-TRANSFORM) FORM

Per chunk `m` of size `C`, local index `t = 0 … C−1`, carry-in state `S_in`.
Line numbers are `mixers.py`. All **[V]** (read from source).

### 3.1 Cumulative decay — computed in LOG space

```python
# 724
cum      = torch.cumsum(torch.log(alpha.view(B, H, N, C)), dim=-1)   # A_t = exp(cum_t)
# 725
cum_prev = cum - torch.log(alpha.view(B, H, N, C))                   # A_{t-1}
```

So, **within a chunk** (the cumulative product resets at every chunk boundary; the
cross-chunk carry is handled separately in §3.5):

```
cum[t]      = Σ_{i≤t} log α_i        A_t     = exp(cum[t])     = Π_{i≤t} α_i
cum_prev[t] = cum[t] − log α_t       A_{t−1} = exp(cum_prev[t]) = Π_{i<t} α_i
```

`A_{−1} = 1` by the empty product. Log space is the point: the direct product
underflows, and the ratio `A_t/A_j` is formed as `exp(cum[t] − cum[j])`.

### 3.2 The rule switch — ONE assignment, TWO consumers

This is the precise answer to "where does `published` differ from `repo`":

```python
# 730-734
if rule not in ("repo", "published"):
    raise ValueError(f"gdn rule must be repo|published, got {rule!r}")
# The cumulative decay the correction term sees: A_{t-1} under the repo
# rule, A_t = a_t A_{t-1} under the published one.
cum_corr = cum if rule == "published" else cum_prev
```

**Line 734 is the only rule-dependent statement in the entire function.** **[V]**
It feeds exactly two downstream consumers, which is what the docstring's "in two
places" means:

| # | Consumer | Line | Effect |
|---|---|---|---|
| 1 | `logM` → `decM` → `M` (the triangular system) | **735** | `M[t,j] = β_t·(A_corr_t/A_j)·(k_t·k_j)` |
| 2 | `A_corr` → `R` (the cross-chunk carry term) | **744 → 753** | `R_t = β_t v_t − β_t·A_corr_t·(S_in k_t)` |

`A_corr_t = A_t` under `published`, `A_{t−1}` under `repo`. The docstring's claim at
mixers.py:703–705 ("replaces A_{t-1} with A_t in both the triangular system and the
carry term ... and changes nothing else") is **CONFIRMED against the code**. **[V]**
Nothing else in lines 707–760 branches on `rule`.

### 3.3 The M matrix and the `(I+M)⁻¹R` solve

```python
# 735-739
logM = (cum_corr[..., :, None] - cum[..., None, :]).tril(-1)
decM = torch.exp(logM.clamp(max=0.0))
M    = (bc[..., :, None] * decM * (kc @ kc.transpose(-1, -2))).tril(-1)
eye  = torch.eye(C, ...).expand(B, H, N, C, C)
Tinv = torch.linalg.solve_triangular(eye + M, eye, upper=False, unitriangular=True)
```

```
M[t,j] = β_t · exp(cum_corr[t] − cum[j]) · (k_t · k_j)     for j <  t
M[t,j] = 0                                                  for j >= t
```

`.tril(-1)` is **strictly** lower — the diagonal is **excluded**. `(I+M)` is therefore
**unit** lower-triangular, hence always invertible, and `unitriangular=True` tells the
solver to *assume* a unit diagonal rather than read it. A Rust implementation solves
`(I+M)U = R` by **forward substitution**:

```
for t = 0 … C−1:
    U[t] = R[t] − Σ_{j<t} M[t,j] · U[j]          # U[t], R[t] ∈ ℝ^D (row vectors)
```

No division is needed (unit diagonal). This is the WY / UT transform: `U` holds the
write vectors `u_t` that the sequential loop would have produced one at a time.

### 3.4 Output: intra-chunk term + cross-chunk state read

```python
# 740-743
logY = (cum[..., :, None] - cum[..., None, :]).tril(0)
decY = torch.exp(logY.clamp(max=0.0))                 # A_t/A_j
Wy   = (decY * (qc @ kc.transpose(-1, -2))).tril(0)   # output weights
A_t  = torch.exp(cum)[..., None]                      # (B,H,N,C,1)
# 752-756
sk = einsum("bhde,bhce->bhcd", S, km)                 # S_in k_t
R  = bc[...,None] * vm - (bc * A_corr)[...,None] * sk
u  = einsum("bhtj,bhjd->bhtd", Tinv, R)               # u = (I+M)^-1 R
sq = einsum("bhde,bhce->bhcd", S, qm)                 # S_in q_t
y  = A_t * sq + einsum("bhtj,bhjd->bhtd", Wy, u)
```

```
Wy[t,j] = exp(cum[t] − cum[j]) · (q_t · k_j)     for j <= t      # INCLUSIVE diagonal
R[t]    = β_t·v_t − β_t·A_corr_t·(S_in k_t)
y_t     = A_t·(S_in q_t)  +  Σ_{j<=t} Wy[t,j]·u_j
          └─ cross-chunk ─┘     └──── intra-chunk ────┘
```

`.tril(0)` on `Wy` includes `j = t`: **`u_t` contributes to `y_t`**, matching §2.3(1).

### 3.5 The chunk-carry state update

```python
# 745-746, 758-759
dec_jC = torch.exp((cum[..., -1:] - cum).clamp(max=0.0))[..., None]  # A_C/A_j
A_C    = torch.exp(cum[..., -1])[..., None, None]
dS = einsum("bhcd,bhce->bhde", u * dec_jC, km)   # Σ (A_C/A_j) u_j k_jᵀ
S  = A_C * S + dS
```

```
A_C          = exp(cum[C−1]) = Π_{i<C} α_i                 (whole-chunk decay)
dS[d,e]      = Σ_{j<C} (A_C / A_j) · u_j[d] · k_j[e]
S_out        = A_C · S_in + dS
```

`S` is initialised once to zeros at mixers.py:748 and carried across all `N` chunks by
the loop at 750. **Only this carry is sequential** — `O(L/C)` steps.

Final slice: `return torch.cat(ys, dim=2)[:, :, :L]` (mixers.py:760) drops the tail pad.

### 3.6 Padding (`L` not a multiple of `C`)

```python
# 716-720
pad = (-L) % C
if pad:
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))   # zeros
    alpha = F.pad(alpha, (0, pad), value=1.0)     # identity: keep state
    beta  = F.pad(beta,  (0, pad), value=0.0)     # no write
```

Padding is **provably inert**: `β=0` ⇒ `u=0` ⇒ no write and `R=0`; `α=1` ⇒ `A_t`
constant through the pad ⇒ `A_C` equals `A` at the last real position. **[V]**
A Rust *sequential* reference needs no padding at all. Worst case is `L=1, C=32`
⇒ `pad = 31`, which fixture `L1` exercises.

### 3.7 The `.clamp(max=0.0)` calls are forward-inert

Lines 736, 741, 745 clamp log-ratios to `≤ 0`. In the triangular region actually used
(`t ≥ j`) with `α ≤ 1`, `cum` is non-increasing, so `cum[t] − cum[j] ≤ 0` already and
the clamp **cannot fire**. Its purpose is stated at mixers.py:727–729: the *masked-out*
upper triangle overflowed to `inf` and NaN'd **autograd**, "even though `.tril()` hid
those entries in the forward". **[V]**

**Consequence for Rust:** a forward-only `f64` sequential reference does **not** need
these clamps. Empirical support: my `f64` reference (§6) has no such clamps and matches
the kernel to ~1e-6 relative on all 13 cases. **[V]**

---

## 4. NORMALIZATION AND GATE PRODUCTION

### 4.1 Where q and k are L2-normalized — and where they are not

`gdn_chunked` itself normalizes **nothing**. The caller does. In
`GatedDeltaNet._project` (mixers.py:799–801) **[V]**:

```python
q = self.q_norm(q.view(B, T, H, P))       # RMSNorm(head_dim), learned weight
k = self.k_norm(k.view(B, T, H, P))       # RMSNorm(head_dim), learned weight
k = F.normalize(k, dim=-1)                # L2-normalize keys
v = v.view(B, T, H, P)                    # v: NO norm at all
```

**Precisely:**
- **`k`**: RMSNorm **then** L2-normalize (`‖k_t‖₂ = 1`). Both, in that order.
- **`q`**: RMSNorm **only**. **`q` is NOT L2-normalized.** **[V]**
- **`v`**: neither.

`RMSNorm` is `F.rms_norm(x, (x.shape[-1],), self.weight, eps=1e-6)` (mixers.py:50–56),
i.e. `x / sqrt(mean(x²) + eps) * weight`, **no mean subtraction**. **[V]**

A separate output RMSNorm is applied after the scan: `out_proj(out_norm(y))`
(mixers.py:814), outside the operator this spec covers.

### 4.2 Gate production — **THE PLAN'S K7 IS WRONG**

> **Plan K7 claim:** `alpha = exp(-exp(A_log)*softplus(a+dt_bias))`, `beta = sigmoid(b)`.

**Actual code (mixers.py:803–804) [V]:**

```python
alpha = torch.sigmoid(a_gate + self.decay_bias).clamp(1e-4, 1.0)
beta  = torch.sigmoid(b_gate + self.update_bias).clamp(0.0, 1.0)
```

with `a_gate, b_gate` sliced from a single projection (mixers.py:781, 797–798):

```python
self.in_proj = nn.Linear(d, 3 * d + 2 * self.n_head, bias=False)
q, k, v, a_gate, b_gate = torch.split(
    self.in_proj(x), [D, D, D, self.n_head, self.n_head], dim=-1)
```

and `self.decay_bias`, `self.update_bias` = `nn.Parameter(torch.zeros(n_head))`
(mixers.py:784–785), i.e. **per-head, zero-initialised**.

**So, itemised:**

| K7 element | Verdict | Reality |
|---|---|---|
| `alpha = exp(-exp(A_log)*softplus(...))` | **REFUTED** | `alpha = sigmoid(a_gate + decay_bias)` |
| symbol `A_log` | **REFUTED** — not a GDN parameter | belongs to Mamba2 |
| symbol `dt_bias` | **REFUTED** — not a GDN parameter | belongs to Mamba2; GDN's is `decay_bias` |
| `beta = sigmoid(b)` | **CONFIRMED in form** | but `b = b_gate + update_bias`, and it is clamped to `[0,1]` |
| clamps | **OMITTED by the plan** | both gates are clamped; α's floor is load-bearing |

**Where K7's formula actually came from.** It is the **Mamba2 / SSD** mixer in the same
file, not GDN (mixers.py:621–642) **[V]**:

```python
# 621  self.A_log  = nn.Parameter(torch.log(torch.arange(1, self.n_head + 1).float()))
# 623  self.dt_bias = nn.Parameter(torch.zeros(self.n_head))
# 638  dt = F.softplus(dt + self.dt_bias).float()    # (B,T,H)
# 639  A  = -torch.exp(self.A_log.float())            # (H,)
# 642  log_dA = dt * A                               # (B,T,H)    <= 0
```

`exp(log_dA) = exp(−exp(A_log)·softplus(dt + dt_bias))` — **exactly** K7's formula, for a
**different mixer**. I confirmed by `rg` that `A_log`, `dt_bias` and `softplus` appear
**nowhere** in the GDN code path; the only hits in `nanolab/` are mixers.py:507, 512–513
(minGRU) and 621–642 (Mamba2). **[V]**

This matters beyond pedantry: `sigmoid` is bounded in `(0,1)` and **symmetric about
0.5 at zero bias**, whereas `exp(−exp(A_log)·softplus(·))` is a near-1 decay whose
per-head `A_log` initialisation (`log(1..H)`) deliberately spreads timescales across
heads. A CPT recipe tuned for one gate parameterisation is not tuned for the other.

### 4.3 Head and dimension layout

- `n_head = max(1, d_model // 64)`; `head_dim = d_model // n_head` (mixers.py:779–780).
- `in_proj` emits `3*d` for q/k/v ⇒ **q, k and v all have `n_head` heads and the same
  `head_dim`**. `D_k = D_v`. **[V]**
- Tensor layout into `gdn_chunked` is `[B, H, L, D]` for q/k/v and `[B, H, L]` for
  α/β (mixers.py:806–807), fp32, inside `torch.autocast(..., enabled=False)`
  (mixers.py:707) — **the scan is fp32 even under bf16 autocast**. **[V]**

---

## 5. NUMERICAL GUARDS — PRESENT, NOT ABSENT

The plan describes "clamping alpha >= 1e-4 as a NaN guard nanolab needed on a real
GH200 run" and asks whether it exists. **It exists, in two places.** **[V]**

**Guard 1 — inside the kernel (mixers.py:709–713):**

```python
# log(alpha) in the WY kernel: sigmoid gates can be 0 after a large
# update, then backward of log is 1/alpha = inf and the run NaNs
# (loss stayed finite through step 50 on GH200; gnorm blew at 55).
alpha = alpha.float().clamp(min=1e-4, max=1.0)
beta  = beta.float().clamp(min=0.0, max=1.0)
```

The comment corroborates the plan's GH200 story **exactly**, including the step numbers.

**Guard 2 — at the caller (mixers.py:803–804):** the same clamps again, in `_project`.
The kernel does not trust its caller; this is defence in depth, and it means a caller
that drives `gdn_chunked` directly still gets clamped.

**Guard 3 — the `.clamp(max=0.0)` triples** (736, 741, 745), forward-inert, backward-
load-bearing. See §3.7.

**Guard 4 — input validation** (mixers.py:730–731): an unknown `rule` raises rather than
silently selecting a default. Fails closed and loud.

**Is the α clamp load-bearing, or cosmetic?** Measured, not assumed. Fixture
`L65_tinyalpha` feeds `α = 1e-8` (five orders below the floor). Against the shipped
golden:

```
clamped   fp64 reference vs golden:  0.000e+00
UNclamped fp64 reference vs golden:  3.990e-04
```

against a chunked-vs-f64 parity noise floor of `5.803e-07` for that case — **~690× the
noise floor**. A Rust reference that omits the clamp fails this fixture at any tolerance
tighter than `1e-4`. **[V]**

> The divergence is moderate rather than catastrophic because at `α ≈ 0` the state
> decays to the last write under either value, so the two answers are close in
> absolute terms. It is nonetheless unambiguously outside tolerance. Set the
> fixture tolerance at `1e-5` and the clamp is pinned.

There is a matching regression test in the repo: `gdn_chunked_survives_odd_lengths_and_tiny_gates`
(tests.py:1958–1970) drives `alpha = 1e-8` at `L ∈ {1,17,33,64}` and asserts finiteness. **[V]**

---

## 6. THE FIXTURES

**Generator:** `/Users/bharath/Code/research/qwen-decision/tools/gen_gdn_fixtures.py`
**Fixtures:** `/Users/bharath/Code/research/tessl/tests/fixtures/gdn/`
**Environment:** `/Users/bharath/.venvs/ml/bin/python` — Python 3.14.7, torch 2.12.1,
numpy 2.5.0, CPU. **[V]** (`tests/fixtures/` did not exist; it was created.)

Run:

```
PYTHONPATH=/Users/bharath/Code/research/MLSystemsLab \
  /Users/bharath/.venvs/ml/bin/python \
  /Users/bharath/Code/research/qwen-decision/tools/gen_gdn_fixtures.py
```

### 6.1 Self-checks the generator refuses to write without

1. **Two independent fp64 transcriptions** — one in the kernel orientation, one
   transcribed from `_sequential`'s opposite orientation — must agree.
   **Result: `6.661e-16`.** This is the evidence that §2.2 is a correct reading of
   the code and not a paraphrase. **[V]**
2. **`published` must differ from `repo`** on every case, or the fixture pins nothing.
   Enforced with an exception for `L=1`, where the gap is asserted to be *exactly*
   `0.0` because the rules are provably identical there (§2.4). **[V]**
3. Shape, finiteness, and dtype assertions on every array.

### 6.2 Files: 108 total

```
93  .npy       (13 cases x 7 arrays, + k_narrow for the 2 GQA cases)
13  _meta.json
 1  gdn_published_MANIFEST.json
 1  gdn_published_SHA256SUMS
```

**Every file name contains `published`** — verified by enumeration, not by intent:
`files WITHOUT 'published' in name: none`. **[V]**

Per case `gdn_published_<NAME>_<field>.npy`:

| field | dtype | shape | meaning |
|---|---|---|---|
| `q`, `k`, `v` | f32 | `[B,H,L,D]` | inputs; `k` is unit-norm per §4.1 |
| `alpha`, `beta` | f32 | `[B,H,L]` | **RAW, unclamped** — the reference must clamp |
| `y_chunked` | f32 | `[B,H,L,D]` | `gdn_chunked(..., rule="published")` |
| `y_seq_f64` | **f64** | `[B,H,L,D]` | independent fp64 sequential golden |
| `k_narrow` | f32 | `[B,H_kv,L,D]` | GQA cases only: pre-widening keys |

**Match a Rust `f64` sequential reference against `y_seq_f64`** (tight, ≤1e-12).
`y_chunked` carries fp32 chunk-parallel rounding; match it at ~1e-5.

### 6.3 The 13 cases — measured, not asserted

| case | B | H | L | D | chunk | h_kv | max&#124;chunked−f64&#124; | max&#124;y&#124; | max&#124;pub−repo&#124; |
|---|---|---|---|---|---|---|---|---|---|
| `L1` | 2 | 3 | 1 | 8 | 32 | – | 7.936e-08 | 1.690 | **0.000e+00** (provable) |
| `L63` | 2 | 3 | 63 | 8 | 32 | – | 1.744e-06 | 6.836 | 1.879 |
| `L64` | 2 | 3 | 64 | 8 | 32 | – | 2.360e-06 | 5.957 | 0.863 |
| `L65` | 2 | 3 | 65 | 8 | 32 | – | 3.009e-06 | 3.185 | 0.891 |
| `L127` | 2 | 3 | 127 | 8 | 32 | – | 1.772e-06 | 3.961 | 1.519 |
| `L129` | 2 | 3 | 129 | 8 | 32 | – | 2.450e-06 | 5.110 | 0.959 |
| `L1000` | 1 | 2 | 1000 | 8 | 32 | – | 5.726e-06 | 6.292 | 1.279 |
| `L8191` | 1 | 2 | 8191 | 8 | 32 | – | 4.787e-06 | 7.257 | 1.915 |
| `L127c64` | 2 | 3 | 127 | 8 | **64** | – | 5.836e-06 | 3.961 | 1.519 |
| `L127c16` | 2 | 3 | 127 | 8 | **16** | – | 1.313e-06 | 3.961 | 1.519 |
| `L65_tinyalpha` | 2 | 3 | 65 | 8 | 32 | – | 5.803e-07 | 4.775 | 1.470 |
| `L64_gqa_kv2` | 2 | **6** | 64 | 8 | 32 | **2** | 2.384e-06 | 4.777 | 1.072 |
| `L129_gqa_kv3` | 2 | **6** | 129 | 8 | 32 | **3** | 3.180e-06 | 5.684 | 0.881 |

All eight required lengths `{1, 63, 64, 65, 127, 129, 1000, 8191}` are covered. Relative
parity is ~`1e-6` throughout, including at `L=8191` — **no error growth with sequence
length**, which is the interesting result for a scan.

**Chunk invariance, verified independently:** `L127`, `L127c16`, `L127c64` share seed
1005, so their inputs are **bytewise identical** (`np.array_equal` = True) and their
`y_seq_f64` goldens are **bytewise identical**. The three `y_chunked` differ by
`1.788e-06` (c32 vs c16) and `6.080e-06` (c32 vs c64) — pure fp32 reassociation.
**Chunk size is not part of the operator's semantics.** **[V]**

### 6.4 GQA — INTERLEAVE, and an honest caveat

> **The plan warns HF repeats K by INTERLEAVE not tile. What does the reference do?**

**`gdn_chunked` does neither, because it has no key-head concept at all.** Its `q`, `k`
and `v` are all `[B,H,L,D]` with the *same* `H`, and `GatedDeltaNet` projects `3*d`
(mixers.py:781), so `n_kv_head` never enters the GDN path. `rg` over `nanolab/` confirms
`n_kv_head` appears only in the attention-family mixers, never in `GatedDeltaNet`. **[V]**

**What the repo does where it *does* widen K/V** — attention, mixers.py:314–317, and
again at 456–459 **[V]**:

```python
if self.n_kv_head != self.n_head:
    rep = self.n_head // self.n_kv_head
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
```

`repeat_interleave` is **INTERLEAVE** (`[k0,k0,k0,k1,k1,k1]`), not tile
(`[k0,k1,k0,k1,k0,k1]`). So the plan's warning matches the repo's convention **for
attention**, and the GDN reference is simply silent.

**What I encoded, and why it is labelled:** the GQA fixtures draw `h_kv` key heads and
widen to `H` with `repeat_interleave`, following the repo's own convention. This is
**[I] inferred**, not `[V]` — it is *my* choice, not something `gdn_chunked` does. To
make it checkable rather than trusted, `k_narrow.npy` ships the pre-widening keys, and I
verified the widening against both candidates:

```
L64_gqa_kv2:  H=6 h_kv=2 rep=3 | interleave match=True | tile match=False
L129_gqa_kv3: H=6 h_kv=3 rep=2 | interleave match=True | tile match=False
```

**α and β are kept at the full `H` heads**, not `h_kv` — there is one recurrent state per
value head and the gates act on that state. This is **[I]**; the reference does not
answer it. See **GAP-GDN-GQA**.

---

## 7. `nanolab/schedules.py` — WSD API

`/Users/bharath/Code/research/MLSystemsLab/nanolab/schedules.py`, 166 lines. **[V]**

**Entry point:** `make_schedule(cfg) -> callable(step) -> lr` (line 20). Dispatches on
`cfg.schedule ∈ {constant, cosine, wsd, plateau}`; anything else raises (line 31).

**`_Base.__init__` (47–51)** — the knob that makes WSD "extend-or-decay":

```python
self.peak   = _peak_lr(cfg)
self.warmup = cfg.warmup_steps
self.total  = cfg.lr_max_steps if cfg.lr_max_steps > 0 else cfg.max_steps
```

`_peak_lr` (34–41) returns `cfg.matrix_lr` for the Muon-family optimizers
(`muon_ns5_adamw`, `muon_ns3_adamw`, `muon_polar_adamw`, `normuon_adamw`, `muown_adamw`,
`mona_adamw`, `mimuon_adamw`) and `cfg.lr` otherwise — the schedule shapes the *matrix*
LR when Muon owns the matrices.

**`WSDSchedule.__call__` (80–89)** — the exact function:

```python
if step < self.warmup:
    return self.peak * self._warmup_mult(step)        # min(1, (step+1)/warmup)
decay_steps = int(self.cfg.wsd_decay_frac * self.total)
decay_start = self.total - decay_steps
if step < decay_start:
    return self.peak                                  # flat
frac = (step - decay_start) / max(1, decay_steps)
return self.peak * (floor + (1 - floor) * (1 - frac)) # LINEAR to floor*peak
```

Defaults (`config.py`): `warmup_steps=256` (177), `lr_floor_frac=0.1` (178),
`wsd_decay_frac=0.2` (179), `lr_max_steps=0` (201). **[V]**

**"Extend-or-decay" is `lr_max_steps`, and it is the only mechanism** (config.py:201:
*"if >0, schedule decays over this many steps instead of max_steps"*). Raising
`lr_max_steps` pushes `decay_start` out and **extends the flat phase**; leaving it `0`
ties the decay to `max_steps`. There is **no** state, no checkpoint, no resume hook —
`WSDSchedule` is a pure function of `step` (`reactive = False`, line 45). Only
`PlateauSchedule` is stateful (`reactive = True`, 96; `observe(val_loss)`, 109). A CPT
that "extends" is therefore just a config change, not a schedule-object migration. **[V]**

`apply_lr(optimizers, lr, cfg)` at line 120 applies the multiplier across param groups.

---

## 8. `nanolab/mqar.py` + `mqar_suite.py` — the needle-hunk pattern

The generator is **`nanolab/mqar.py`** (173 lines); `mqar_suite.py` (567 lines) is the
sweep driver around it. **[V]**

### 8.1 `mqar.py` API

```python
IGNORE = -1                                        # matches model.py's ignore_index (:46)
def vocab_for(n_keys, n_values) -> int:            # 1 + n_keys + n_values      (:49)
class MQARBatcher:                                                              # (:54)
    def __init__(cfg, device, split="train", *, n_pairs=None, n_queries=None,
                 n_keys=None, n_values=None)                                    # (:64)
    def batch(block_size=None, frontier=1.0) -> (x, y)                          # (:101)
    def iterator()                                                              # (:141)
    def __len__() -> batch_size * seq_len                                       # (:92)
@torch.no_grad()
def recall_accuracy(model, batcher, ctx, iters=20) -> float                     # (:147)
```

**Sequence shape:** `k v k v … | k v k v …` — `n_pairs` pairs then `n_queries` queries.
`seq_len = 2*(n_pairs + n_queries)` and `block_size` **must** equal `seq_len − 1`
(x is `seq[:-1]`, y is `seq[1:]`).

**Token ids:** `0` unused; keys `1 … n_keys`; values `1+n_keys … 1+n_keys+n_values`
(`_key`/`_value`, lines 95–99).

**Generation (`batch`, 112–135):** per row, `randperm(n_keys)[:n_pairs]` gives **distinct**
keys; `randint(n_values)` gives an independently drawn value per pair;
`randperm(n_pairs)[:n_queries]` picks which pairs are queried, **in a random order,
without replacement**. Determinism comes from one CPU `torch.Generator` seeded
`cfg.seed + (0 if train else 1)` (lines 89–90) — a **split-offset seed**, so train and
val never coincide.

**Masking:** `y` is `IGNORE` everywhere except the answer positions
`ans_at = arange(2*n_pairs, seq_len-1, 2)` (line 134). **The training loss IS recall
loss** with no model or loop change.

### 8.2 The validity invariants (the part worth copying)

`MQARBatcher.__init__` refuses malformed configurations rather than degrading, with
messages that state the *reason* (lines 71–87) **[V]**:

- `n_queries > n_pairs` → *"a query with no stored pair has no right answer"*
- `n_pairs > n_keys` → *"keys must be unique within a sequence or a query has two right answers"*
- `block_size != seq_len - 1` → *"Set it from the task."*
- `batch(block_size=…)` mismatch → *"MQAR sequence length is fixed by the task"*
- `frontier != 1.0` → *"MQAR is generated, not sorted; frontier is meaningless"*

This is the pattern a needle-hunk generator should copy: **well-posedness is enforced at
construction**, so a query always has exactly one right answer.

`recall_accuracy` (147–173) scores exact match at masked positions, and **raises** rather
than returning a wrong number when `cfg.fused_ce` hid the logits (line 162) or when no
query position was scored (line 171).

### 8.3 A probe-design warning carried in the docstring

`mqar.py:25–40` records that with `tie_embeddings=True` (the repo default) attention caps
at **0.555** on this task and untied reaches **0.990** under an otherwise identical
config, measured 2026-08-27 at P=4/Q=4/K=16/V=16, d=256, 4 layers. **[V]** If the
needle-hunk generator copies this pattern it must also copy `tie_embeddings=False`, or
the reference arm sits at a ceiling *"that has nothing to do with recall"* and the metric
cannot separate arms.

`mqar_suite.py` adds `mqar_lr_for_batch(lr, batch, rule="sqrt")` (:104), `e8_config(...)`
(:137, defaults `n_pairs=4, n_queries=4, n_keys=16`), `run_one` (:167), sharding/worker
spawn (:230/:242), Wilson intervals (:300), and `board`/`report` (:337/:355).

---

# JOB 2 — PRODUCT CONTRACTS

Research for this section was delegated. **[V]** here means *I* read the line;
**[Vd]** means a delegate read it and quoted it, and I did not re-open the file. The four
load-bearing claims (§9.1 `StatusFromGaps` and its single assignment, the absence of a
`PASS` literal, §9.4 `allow`, §10.1 the call-site guard) I re-verified myself — those are
**[V]**. Nothing was compiled and no test was run in either repo (**GAP-JOB2-NOTRUN**).

## 9. DevCouncil — the verification / admission contract

### 9.1 There is no `PASS`. There is a derived `passed bool`.

`rg -uu '\bPASS\b'` over all `.go` and `.rs` under `backend/go_orchestrator` and `rust/`
returns **zero matches** (re-run by me with `--no-ignore --hidden`; rg exit 1). **[V]**
The verdict is two JSON fields, and it has exactly **one producer**:

```go
// backend/go_orchestrator/devcouncil/verify/orchestrate.go:139-155
func StatusFromGaps(gaps []Gap, gateMode string) (status string, passed bool) {
	if verificationSkipped(gateMode) {
		return "skipped", false
	}
	mode := strings.TrimSpace(strings.ToLower(gateMode))
	advisory := mode == "advisory" || mode == "warn"
	for _, g := range gaps {
		if !g.Blocking {
			continue
		}
		if advisory && gating.MayDemote(g.GapType, true) {
			continue
		}
		return "blocked", false
	}
	return "verified", true
}
```

`status ∈ {"skipped", "blocked", "verified"}`. It reaches the wire as
`MCPResult.Passed bool \`json:"passed"\`` / `Status string \`json:"status"\``
(types.go:58–59). I confirmed by `rg` that within `devcouncil/verify` the field
`Passed:` is assigned at **exactly one** non-test site, `orchestrate.go:188` — the other
hits are a structurally unrelated `check` type in `internal/mapcli/commands.go`. **[V]**

**`passed` is a pure function of the gap list and the gate mode.** No component "emits"
it. A component can only contribute `Gap` values.

### 9.2 Rule 7: right outcome, wrong mechanism, and the second clause is false

> Plan rule 7: product wiring *"never emits PASS, `allow`, or discharges a requirement"*
> and *"dcverify still owns PASS"*.

| Clause | Verdict |
|---|---|
| product wiring never emits PASS / `allow` | **Accurate — and for a stronger reason than the plan gives.** Not policy: `Passed` is *derived*, so writing to it is architecturally unavailable outside `ToMCP`. **[V]** |
| **"dcverify still owns PASS"** | **REFUTED.** dcverify has **no pass concept at all**. It emits `Finding`, `CoverageReport`, `SubstanceReport`. The **Go host** owns the verdict. **[Vd]** |

dcverify's exit `0` means *"the report is valid"*, not *"the diff is clean"*
(`rust/dc-verify/src/bin/dcverify.rs:8-10`) **[Vd]**:

> *"a diff it cannot parse is exit 2 with an error, never an empty result: an empty
> finding list means 'these gates ran and found nothing', and a caller cannot be allowed
> to read 'could not run' as that."*

### 9.3 What an admission-only integration looks like

A caller contributes `Gap` values (`verify/types.go:16-33`: `Severity`, `GapType`,
`Blocking`, `Evidence`, `RecommendedFix`, `File`, `Line`) and lets `StatusFromGaps` do
the arithmetic. Three postures, increasing in strength **[Vd]**:

1. **Advisory / finding-only** — `Gap{Blocking: false}`. Lands in
   `MCPResult.AdvisoryActions`; never touches `passed`. Shipped examples:
   `gapsFromCoverage` (rigor.go:288), `gapFromSubstance` (rigor.go:175).
2. **Blocking, demotable** — `Gap{Blocking: true}` with a `GapType` not in
   `gating.HardSafetyGapTypes`. Demoted to advisory under `warn`/`advisory` by
   `gating.MayDemote` (`devcouncil/gating/policy.go:33-41`).
3. **Hard safety** — `GapType ∈ HardSafetyGapTypes` (policy.go:9-23: `security_risk`,
   `stub_detected`, `orphan_diff`, `rigor_check_unavailable`, …). Never demotable.

On the Rust side the emission type is `rust/dc-verify/src/rigor.rs:71-83` — `Finding`,
carrying **two orthogonal axes**: `Severity::{Blocking, Advisory}` (rigor.rs:19-24) and
`Strength::{Proven, Observed, Derived}` (rigor.rs:40-54). `Strength` is precisely the
"never launder inferred into verified" axis; its doc says so (rigor.rs:31-35) **[Vd]**.

**A product caller cannot inject a `Finding`** — dcverify is a subprocess reading a diff
on stdin, and its gate namespace is closed: `dc/dcverify/dcverify.go:68-71`, *"a gate this
client does not know is refused in validate rather than silently carried to a caller that
has no gap type for it."* **Product wiring integrates at the `Gap` seam, not the
`Finding` seam.** **[Vd]**

### 9.4 `allow` is downstream of `passed`

```go
// devcouncil/stopgate/stopgate.go:117-120
	allow := mcp.Passed && gateMode != "off"
	if gateMode == "off" {
		allow = true
	}
```
**[V]** — read by me. `Allow bool \`json:"allow"\`` is stopgate.go:18.

### 9.5 The substring gate, and what it cannot see

There are **two**, both in `rust/dc-verify/src/rigor.rs` **[Vd]**:

- **`detect_stubs`** (rigor.rs:114-170) — `.contains()` over one lowercased added line,
  against `EMPTY_BODIES` (rigor.rs:102-108: `todo!()`, `unimplemented!()`,
  `panic!("todo")`, `raise notimplementederror`, `pass  # todo`) and `STUB_MARKERS`
  (rigor.rs:88-99). Empty bodies are `Blocking`; bare markers are `Advisory` and only
  inside a comment or string (`is_comment_or_string`, rigor.rs:178-187).
- **`scan_secrets`** (rigor.rs:451-499) — prefix + length floor + character class over
  `SECRET_PATTERNS`.

**Why a silent stub slips past.** The escaping class is *a syntactically complete
implementation, with no marker text, that returns a plausible fabricated value*:
`func isValid() bool { return true }`; a handler that logs and returns `nil`; a
`verified` field set by assignment rather than by a check. There is **no substring to
match**. Both gates read added lines as text, one line at a time; neither parses, neither
knows what a function is supposed to do. **A substring gate can only catch a defect that
names itself.**

**Is the limitation acknowledged? Unevenly — and this is the gap worth carrying into the
runtime design [Vd]:**

- *False positives* — acknowledged in detail (rigor.rs:125-133): *"this is a substring
  test over one line of text: the same bytes inside a string literal, a doc example or a
  macro that quotes its input match identically."* This is why a blocking stub finding is
  still only `Derived`.
- *Secret-gate false negatives* — acknowledged **explicitly**, with the governing
  principle (rigor.rs:303-317): *"a gate is only safe to rely on while what it cannot see
  is known."* Two evasions are enumerated.
- *Stub-gate false negatives* — **not acknowledged.** The nearest text is
  `substance.rs:1-18`, which names a **different** case (a relocation/boilerplate diff
  with *"no stub marker, no credential, and … no coverage gap either"*). The substance
  gate answers that one, and only **non-blockingly** (`gapFromSubstance`, rigor.go:175-179:
  *"Non-blocking, and that is not timidity"*). A markerless fake implementation that is
  genuinely new hand-written text scores as **substantive** and passes every gate.
- **No negative-control test.** `rust/dc-verify/tests/rigor.rs` has 22 tests; the
  stub-detection block (`an_unimplemented_body_blocks` :109,
  `a_todo_comment_is_advisory_not_blocking` :122,
  `identifiers_containing_marker_words_are_not_flagged` :133) is entirely
  true-positive / false-positive. Nothing plants a markerless fake. **GAP-DC-SILENTSTUB**.

### 9.6 "Ran and passed" vs "could not run" — already the organising principle

This is the strongest part of the existing contract and the runtime design should inherit
it rather than reinvent it.

**Go side** — `devcouncil/verify/rigor.go:63-77`, with the invariant stated **[Vd]**:

```go
// applied and skippedReason are a pair: exactly one of them is populated.
// Anything else is the state this file exists to prevent, where a report claims
// gates it did not run or stays silent about gates it skipped.
type rigorOutcome struct {
	gaps    []Gap
	applied []string
	skippedReason string
	coverageMeasured      bool
	coverageSkippedReason string
}
```

Built through a constructor so a reasonless skip is unrepresentable (rigor.go:79-91:
*"A skip with an empty reason is indistinguishable in every report from a clean pass."*).
Coverage is tracked separately because *"the findings gates can run perfectly well on a
change nobody measured."*

The middle exit is the important one: a verifier that was **configured but failed** is a
**blocking gap**, not a skip — `GapType: "rigor_check_unavailable"` (rigor.go:119-149),
*"a run where the secret scanner silently did not happen must not come back looking like
a run where it happened and found nothing."* That gap type is **un-demotable**
(policy.go:18-22: *"a failing test is evidence, while this is the absence of evidence
about whether a credential is in the change."*). All four facts reach the wire
(`types.go:63-70`: `coverage_measured`, `coverage_skipped_reason`, `rigor_applied`,
`rigor_skipped_reason`), and `stopgate.go:44-47` notes a skipped decision leaves MCP at
its zero value *"so a host cannot treat 'could not run' as a pass."*

**Rust side — equivalent, but structural rather than a field [Vd]:** exit `2` for
"could not run" (a missing coverage file is fatal, not silently unmeasured —
`bin/dcverify.rs:210-213`); `CoverageReport`'s `unmeasured` and `skipped_by_type` buckets
(rigor.rs:601-622), where `is_clean()` counts `unmeasured` as *not* clean; and `Strength`
as the same rule one step further in.

A recorded past failure of exactly this kind (`bin/dcverify.rs:180-190`): a non-UTF-8 byte
made `read_to_string` refuse, which propagated as a degradation, and *"a degradation
produces no blocking finding: the secret scanner, the stub detector and the coverage gate
were all skipped for the whole change while the report still said passed."*

**`RigorClient` trust boundary** (rigor.go:50-61): resolves `dcverify` via
`proc.LookPathOutside`, **refusing any candidate inside the repository under analysis** —
*"repository contents are input to verification"* — and returns `nil` rather than a
guessed path. **[V]** (read directly via DevMap source span.)

---

## 10. DevType — the usage store's privacy contract

### 10.1 The plan's claim is CONFIRMED for the schema, and QUALIFIED at the key

**Schema — no query text field exists.** `Sync/CommandUsageStatsStore.swift:10-39`:

```swift
public struct Stat: Codable, Equatable {
    public var usageCount: Int
    public var lastUsedAt: Date?
}
private struct Document: Codable {
    static let currentSchemaVersion = 1
    var schemaVersion: Int
    var stats: [String: Stat]
}
```

On-disk (`command-usage-stats.json`, written with `.sortedKeys` at :224-237):

```json
{"schemaVersion":1,"stats":{"<commandID>":{"lastUsedAt":<date>,"usageCount":<int>}}}
```

`usageCount` + `lastUsedAt` + `schemaVersion` is genuinely all. **[V]**

**THE QUALIFICATION — and this is the finding that matters.** The schema has no query
*field*, but **the map key is a command ID, and some command IDs are built from query
text** (`Sync/CommandPaletteCatalog.swift`, confirmed by me at those lines) **[V]**:

```
:1228   id: "routed.\(routed.query)"          <- the raw palette query
:1263   id: "math.\(expression)"              <- the typed expression
:1301   id: "date.relative.\(days).\(format)"
:1321   id: "date.offset.\(trigger)"
```

**The store does not defend against this.** `recordUsage` accepts *any* non-empty trimmed
string (CommandUsageStatsStore.swift:90-92) **[V]**. The guard is at the **call sites**:

```swift
// DevTypeAppCore/AppDelegate.swift:1383-1387
// Ephemeral rows are built from the query text, so their ids are unique per keystroke
// and resolve to nothing in the catalogue. Recording them grew the stats file forever.
if !command.isEphemeral {
    CommandUsageStatsStore.shared.recordUsage(for: command.id)
}
```

I enumerated **all three** non-test call sites myself **[V]**: AppDelegate.swift:942 and
:1386 (both guarded by `if !command.isEphemeral`, verified at :941-943 and :1385-1387) and
MacroPalettePanel.swift:574, which passes `MacroPaletteRanking.usageID(descriptor)` =
`"macro.\(descriptor.id)"` (MacroPaletteRanking.swift:32) — catalogue-derived, so it
correctly needs no guard.

**So the invariant holds today, but it is enforced by convention at the caller, not by the
store's type.** The note in the comment reveals the guard's motivation was **file growth**,
not privacy — the privacy property is a side effect. The delegate reports the pinning test
(`Tests/.../PaletteToolRoutingTests.swift:154-167`) asserts the *ephemeral flag*, not the
call-site guard **[Vd]**. A fourth call site added without the guard would write query text
to disk and no test would fail. **GAP-DT-EPHEMERAL-CONVENTION.**

Free-text AI instructions are safe by a different route: `ai.custom.oneshot`
(CommandPaletteCatalog.swift:1281) uses a **constant** id; the user's text lives only in
`action:` / `preview:`, which `recordUsage` never sees. **[Vd]**

### 10.2 Siblings persist no query text either

- **`UsageStatsStore`** (`Models/UsageStatsStore.swift`) — `Stat { usageCount, lastUsedAt, buckets }`,
  `UsageBucket { startedAt, usageCount, lastUsedAt }`, `Document { schemaVersion: 2, stats }`,
  keyed by `UUID.uuidString` (:156), file `usage-stats.json`, bucket count bounded at 800.
  Counters and timestamps only. **[Vd]**
- **`DebouncedSidecarWriter`** (`Models/DebouncedSidecarWriter.swift`) is content-blind by
  construction (:6-9: *"the writer never learns the shape of the document, only how to take
  the bytes that are due"*); `SidecarPayloadSource` is two methods returning opaque `Data?`. **[Vd]**

### 10.3 Query text exists in memory only — so a query log is a NEW file

The only retention of raw query strings is `Sync/SnippetSearch.swift:234-245`: a
`private static` FIFO ring capped at `maxQueryCacheEntries = 128`, never handed to a
`SidecarPayloadSource`, never encoded, never written. `PaletteToolRouter.Routed.query`
is a transient value type. **[Vd]**

`rg` over `Sources/` for `recentQuer|queryHistory|searchHistory|lastQuery|recentSearches`
returns **zero hits** (delegate re-ran this with `-uu`). There is **no recent-queries list,
no search history, no query log, and no existing telemetry opt-in** — the only `telemetry`
symbol is `InjectTelemetryLog`, which records refusal kinds and bundle IDs, not text. **[Vd]**

**Therefore: an opt-in local-only query log is a NEW sidecar, not an extension.** No
existing store has a field that could host it, and both existing stores are `[ID: counters]`
maps whose key space is explicitly guarded against query-derived strings.

### 10.4 What such a log would touch

| Concern | File:line | Note |
|---|---|---|
| Sidecar path | `Models/DebouncedSidecarWriter.swift:76-82` (`resolveFileURL`) | Already generic over `fileName`. Chain: caller override → `SnippetStore.storeDirEnvKey` (`Models/SnippetStore.swift:408`, `DEVTYPE_STORE_DIR`) → `defaultLocalSupportDirectory` (:552) → `SupportDirectory.devType` (`Models/SupportDirectory.swift:34-40`). Comment at :72-75 pins device-local: *"Never the synced library directory."* |
| schemaVersion / migration | `Sync/CommandUsageStatsStore.swift:21, 29-33, 213-220` | **CommandUsageStatsStore has no migration** — it tolerates a missing key and returns `[:]` on any decode failure, but never rewrites on a version bump. The precedent that *does* migrate is `Models/UsageStatsStore.swift:484` (`needsRewrite`) against `currentSchemaVersion = 2`. **Copy the UsageStatsStore shape, not the Command one.** |
| Flush / terminate | `Models/DebouncedSidecarWriter.swift:87-97, 116-122, 127-152` | A new store gets the `willTerminate` observer free from the shared writer. **Asymmetry:** `AppDelegate.applicationWillTerminate` (:2443-2461) explicitly flushes only the *snippet* sidecar (:2451); the command sidecar relies solely on its own observer. |
| Settings toggle | `DevTypeAppCore/PreferencesWindowController.swift:731-738` (`PreferencesTab.advanced`) | Closest precedent is `Voice/VoicePreferences.swift:20, 273, 277-285, 466` — an opt-in, default-off, local-only diagnostic recording with a purge path, whose `object(forKey:) != nil` guard keeps "unset" distinguishable from "off". A localization key `prefs.advanced.telemetry` already exists (`Localization/LocalizationManager.swift:1194`). |

All of §10.4 is **[Vd]**.

---

## GAPS

| id | gap |
|---|---|
| **GAP-DC-SILENTSTUB** | No comment or test in `dc-verify` names the markerless-fake-implementation false negative. `substance.rs:1-18` is the nearest acknowledgment and covers a *different* case (relocation/boilerplate). The 22 tests in `rust/dc-verify/tests/rigor.rs` contain no negative control planting a syntactically complete fake. Recorded as **unacknowledged**; a contrary statement was searched for and not found, which is weaker than proving absence. |
| **GAP-DT-EPHEMERAL-CONVENTION** | DevType's "no query text on disk" property is enforced by an `if !command.isEphemeral` guard repeated at each call site (AppDelegate.swift:941-943, :1385-1387), **not** by `CommandUsageStatsStore`'s type — `recordUsage` accepts any non-empty string (:90-92). All three current non-test call sites are correct, but a fourth added without the guard would persist query text and **no test would fail**; the pinning test asserts the `isEphemeral` flag, not the guard. |
| **GAP-JOB2-NOTRUN** | Nothing was compiled and no test was run in DevCouncil or DevType. Every Job 2 claim is a source read. Test *names* and *assertions* are quoted as written; **whether they pass today is unverified**. |
| **GAP-JOB2-DELEGATED** | Items marked **[Vd]** were read and quoted by a delegate, not by me. I independently re-verified the four load-bearing ones (`StatusFromGaps` + its single `Passed:` assignment, the absent `PASS` literal under `-uu`, `allow` at stopgate.go:117-120, and all three DevType `recordUsage` call sites with their guards). The remainder are single-sourced. |
| **GAP-GDN-GQA** | `gdn_chunked` has **no key-head concept**, so the reference cannot answer interleave-vs-tile for GDN. The fixtures encode `repeat_interleave` by analogy with the repo's attention path (mixers.py:316, :458). **[I], not [V].** Whether α/β should follow key heads or value heads under GQA is likewise unanswered by the reference; I kept them at full `H`. |
| **GAP-GDN-FP64-PUBLISHED** | The `<1e-9 in fp64` provenance claimed at mixers.py:682–683 traces to `/Users/bharath/Code/research/MLSystemsLab/Rust_MLKit/reference/verification/verify_gdn_wy.py`, which implements the **repo** rule only (no `rule` parameter; line 17 `e_t = v_t - v_hat`, undecayed). **The published rule has no fp64 provenance in the repo** — its only in-repo check is `gdn_chunked_matches_sequential` in fp32 at tol `1e-4`. The fp64 goldens shipped here are the first. |
| **GAP-DEVMAP-WALKINCOMPLETE** | Every `devmap_search` response carried `walk_incomplete`: 20 files failed to parse, 34 pattern-recovered with no call extraction, 7 in languages with no call extractor. Dead-code/unwired findings from DevMap are a **lower bound**. Nothing in §1–§8 depends on a negative DevMap result; the negatives (`A_log`/`dt_bias` absent from the GDN path, `n_kv_head` absent from `GatedDeltaNet`) were established by reading the code and by `rg`, and are labelled accordingly. |
| **GAP-RG-TRACKED-ONLY** | The `rg` sweeps in §4.2 and §6.4 ran with the user's default config (`--hidden`, `--glob=!.git/`), i.e. over **tracked source**. They were not re-run with `-uu`. For these two questions the scope is the `nanolab/` package, which is fully tracked, so the risk is low — but the sweeps are not exhaustive over generated or ignored trees. |
| **GAP-MLSYSLAB-CODEINTEL-STALE** | `gitpulse_insights` on MLSystemsLab reports its **repo-local** DevMap store `is_fresh: false` with `freshness_reason` "analyzer freshness unverified ... built without the parsing frontend". All facets returned `ok: true`. Navigation used the **root** store at `/Users/bharath/Code/.devcouncil/codeintel/devmap.sqlite`, which is fresh (gen 25). The two stores are distinct; no finding here rests on the stale one. |
| **GAP-INDEX-MOVED-MIDSESSION** | The root DevMap index was `is_fresh: true`, `degraded_reason: null` at generation **25** when this lane opened it. It later reported `is_fresh: false` / *"source tree differs from the indexed generation"* and advanced 25 → 26 → 27. **This lane caused it** by writing `research/qwen-decision/tools/` and the fixture tree. The delta is confined to files this lane created; no symbol cited in this document lives in them. Line numbers in §1–§10 were re-verified against source by direct read, so they are code-confirmed rather than graph-confirmed. |

---

## CORRECTIONS TO THE PLAN

1. **K7's α formula is wrong.** The plan says
   `alpha = exp(-exp(A_log)*softplus(a+dt_bias))`. GDN actually uses
   `alpha = sigmoid(a_gate + decay_bias).clamp(1e-4, 1.0)` (mixers.py:803). The plan's
   formula is the **Mamba2/SSD** decay (mixers.py:621–642), a different mixer in the same
   file. `A_log` and `dt_bias` are **not GDN parameters**; GDN's per-head biases are
   `decay_bias` and `update_bias` (mixers.py:784–785).

2. **K7's β formula is right in form but incomplete.** `beta = sigmoid(b)` holds, but
   `b = b_gate + update_bias` and the result is `.clamp(0.0, 1.0)` (mixers.py:804).

3. **The α ≥ 1e-4 guard is PRESENT, not absent.** The plan asks whether it exists. It
   does, **twice** — mixers.py:712 (inside the kernel) and mixers.py:803 (at the caller).
   The comment at mixers.py:709–711 corroborates the plan's GH200 anecdote including the
   step numbers ("finite through step 50 ... gnorm blew at 55"). I measured it to be
   load-bearing, not cosmetic (§5).

4. **"It swaps A_{t−1} for A_t in two places" — correct, but it is ONE assignment.**
   Line 734 (`cum_corr = cum if rule == "published" else cum_prev`) is the only
   rule-dependent statement in the function. It feeds two consumers: `logM` at line 735
   and `A_corr` at 744 (used at 753). Anyone patching "two places" in a port will have
   introduced a second source of truth.

5. **`gdn_chunked` has no key-head concept, so it cannot confirm interleave-vs-tile.**
   The plan's framing presumes the reference expands K. It does not — q, k and v all
   carry `n_head` heads. The repo's `repeat_interleave` convention lives in the
   **attention** mixers (mixers.py:316, :458). The fixtures encode INTERLEAVE by analogy
   and ship `k_narrow` so the choice is checkable rather than trusted. See **GAP-GDN-GQA**.

6. **`rule="published"` alone does not make a fixture meaningful at L=1.** At `L=1` the
   two rules are provably the *same function* (`S_{−1}=0`), and the same holds whenever
   `α ≡ 1` or `β ≡ 0` (tests.py:196–197). A single-token "published" fixture is
   rule-agnostic; the generator asserts the gap is exactly zero there instead of letting
   the coincidence pass as verification.

7. **The repo's fp64 provenance covers the `repo` rule only.** mixers.py:682–683 cites
   `verify_gdn_wy.py` for "<1e-9 in fp64", but that file has no `rule` parameter and
   implements the undecayed read. The published rule's only in-repo verification is fp32
   at `1e-4`. See **GAP-GDN-FP64-PUBLISHED**.

8. **Plan rule 7's DevCouncil framing names only half the system.** It cites
   `rust/dc-verify/src/rigor.rs`. There is a **Go host half** at
   `backend/go_orchestrator/devcouncil/verify/` + `devcouncil/stopgate/` that resolves and
   invokes the binary, and it — not the Rust side — owns both the verdict and the
   applied/skipped bookkeeping.

9. **"dcverify still owns PASS" is REFUTED.** dcverify has **no pass concept**; it emits
   `Finding` / `CoverageReport` / `SubstanceReport` and exits `0` to mean *"this report is
   valid"*, never *"this diff is clean"* (bin/dcverify.rs:8-10). The verdict is owned by
   `verify.StatusFromGaps` (orchestrate.go:139-155) in the **Go host**, and admission by
   `stopgate.go:117-120`.

10. **There is no `PASS` token at all.** `rg -uu '\bPASS\b'` over every `.go` and `.rs` in
    both trees returns zero matches. The verdict is `status ∈ {"skipped","blocked","verified"}`
    plus `passed bool`. A spec or test that greps for `PASS` is testing a string that does
    not exist.

11. **"Product wiring never emits PASS" is true for a stronger reason than the plan gives.**
    It is not a policy anyone could violate: `passed` is a *pure function* of the gap list
    and gate mode, assigned at exactly one site (`orchestrate.go:188`). The correct way to
    state the constraint is *"product wiring contributes `Gap` values and never computes a
    verdict"* — and the integration seam is `Gap`, not `Finding`, because dcverify is a
    subprocess with a closed gate namespace (dcverify.go:68-71).

12. **The substring gate's blind spot is real, and is the one limitation the codebase does
    NOT write down.** dc-verify is unusually rigorous about naming what its gates cannot
    see — but that discipline is applied to the *secret* scanner (rigor.rs:303-317) and to
    stub *false positives* (rigor.rs:125-133), not to stub false negatives. A markerless
    fake implementation passes `detect_stubs`, passes `scan_secrets`, and scores as
    substantive in `substance.rs`. See **GAP-DC-SILENTSTUB**.

13. **DevType's "no query text" claim is CONFIRMED for the schema but QUALIFIED at the
    key.** No persisted *field* holds query text. But command IDs for ephemeral rows are
    *built from* the query (`"routed.\(routed.query)"`, `"math.\(expression)"`,
    CommandPaletteCatalog.swift:1228/1263/1301/1321), and `recordUsage` accepts any string
    — the property is held by a call-site convention, not by the store's type. The guard's
    own comment says it was added to stop the stats file *growing forever*, so privacy here
    is a side effect of a size fix. The plan's conclusion (training pairs must be
    templated) is unaffected and correct; its *reason* should be restated. See
    **GAP-DT-EPHEMERAL-CONVENTION**.

14. **An opt-in query log would be a new sidecar, not an extension.** No existing DevType
    store has a field that could host query text, and there is no recent-queries list,
    search history, or telemetry opt-in anywhere in the app to hang it off. If one is
    built, copy `UsageStatsStore`'s migration shape (`needsRewrite`, schemaVersion 2), not
    `CommandUsageStatsStore`'s — the latter has **no migration path at all**.
