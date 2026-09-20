# Corrections to the Build Plan

Every row is a plan assumption checked against a primary source on 2026-09-19. Evidence is in
`AUDIT/external-surfaces.md`. **Where the plan and the evidence disagree, the evidence wins and the
plan text is stale.** Nothing here is a judgement call about the plan's strategy; these are facts it
got wrong.

Severity: **BLOCKING** = a lane cannot proceed. **DESIGN** = a design decision changes.
**SAFETY** = a gate that would have passed silently while broken. **RELIEF** = the plan was
pessimistic; work is removed.

---

## SAFETY-1 — The fla fallback gate is fail-open. Replace it.

**Plan:** *"A job refuses to start if the warning is present in a 30-second dry run"*, greping the
startup log. The plan calls this rule 2 of the three that keep the budget safe.

**Evidence:** the warning fires on **first forward, not at import**, and is **suppressed entirely
under `torch.compile`** by a `not is_torchdynamo_compiling()` guard. A 30-second dry run of a compiled
job produces no warning **whether or not the fast path is active**. The grep then passes, and the run
proceeds on the fp32 chunk loop at full cost.

This is the worst failure shape in the plan: a check that cannot fail, guarding the item the plan
names as its second-largest budget risk.

**Also:** the penalty is **74% slower** (1.22 vs 0.70 s/step), not the plan's "43% longer" — 43% is
the *saving*, read from the wrong direction. The transformers source puts the `chunk_gated_delta_rule`
gap at "more than an order of magnitude on an H100."

**Change:** delete the log grep. Replace with a **positive runtime assertion** that the fast path
object is the one in use — assert on the resolved implementation, not on the absence of a string.
Record it in the ledger as `env.fla_present`, a tri-state: confirmed-present, confirmed-fallback, or
not-checked. An unchecked environment must not read as a clean one.

## SAFETY-2 — Packing is supported now, and the real hazard is worse than the plan's

**Plan:** *"Sequence packing is not supported for the hybrid (issue #46093), so batches are
length-bucketed; expect 10-20% padding waste and budget for it."*

**Evidence:** issue #46093 closed `completed` 2026-05-26; PR #45034 merged 2026-05-13 plumbs
`cu_seq_lens_q` / `seq_idx` through GDN. Packing works.

**But the live hazard is sharper than the one the plan describes.** Passing **`position_ids` alone**
— the TRL / veRL convention, and what `DataCollatorWithFlattening` does by default — runs **without
error** while sample boundaries **silently leak across packed sequences**. Maintainers have confirmed
no `position_ids`-only path is coming and that failing fast is impractical.

**Change:** enable packing and reclaim the padding waste, **and** add an explicit assertion that
`cu_seq_lens_q` / `seq_idx` are present on every packed batch. A packed batch without them trains on
cross-sample attention and reports a perfectly normal loss curve.

## RELIEF-1 — Model step 1 is already done, and `transformers` main is not required

**Plan:** step 1 is a "text-only export" dropping the vision tower and MTP head, gated by logit parity
on 100 prompts. The plan also carries a whole risk row for *"`transformers` main-branch churn for
Qwen3.5 — the model needs main; main moves daily."*

**Evidence:** `Qwen3_5ForCausalLM` already declares
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`. Loading it drops both with no
surgery. There is **no MTP module in transformers at all** — the only two `mtp` references in the file
are ignore-lists, so there is nothing to drop. And Qwen3.5 has shipped in **tagged releases since
v5.2.0 (2026-02-16)**; stable is **v5.17.0**.

**Change:** **pin a release, not a main SHA.** The "main-branch churn" risk row is deleted, not
mitigated. Step 1 shrinks to a load-and-verify: assert the visual and mtp tensors are absent from the
loaded model and that the parameter count matches text-only. Keep the logit-parity check — it is cheap
and it is what proves the assertion — but the surgery is gone.

**Sizing the plan got wrong:** the checkpoint is **4.55 GB, ~2.27B params**, carrying **297
`model.visual.*` and 15 `mtp.*` tensors out of 632**. Qwen3.5-2B-Base **is a multimodal checkpoint**
(`pipeline_tag: image-text-to-text`, full `vision_config`) — the plan never says so. The 509M
embedding is ~22% of the *shipped* checkpoint, not of a 2.0B text-only model.

## DESIGN-1 — GDN key and value heads are equal. The interleave test moves.

**Plan:** *"Qwen3.5 has fewer linear key heads than value heads and HF repeats K by interleave, not
tile. One test with distinct K heads catches the wrong order."* Listed as both a parity-suite item and
a risk row.

**Evidence:** `linear_num_key_heads = 16` and `linear_num_value_heads = 16`. **There is no K/V
asymmetry in the GDN blocks.** The *full-attention* layers are asymmetric — 8 heads / 2 KV heads —
which is the likely source of the transposition.

**Change:** the interleave-vs-tile parity test does not apply to K1/K2 and must not be written as if
it does. The GQA repeat concern is real **for the attention layers**, where `flash_attn_swa_h256`
already handles it — so the test belongs there, against 8/2, if it is not already covered. Confirmed
as claimed: Apache-2.0, ungated, 262144 context, 24 layers = 18 linear + 6 full, vocab 248,320, state
16x128x128, partial rotary 64 of 256, q_norm/k_norm with **zero** `v_norm`, depthwise causal conv
kernel 4 over a single fused `in_proj_qkv` split `[key_dim, key_dim, value_dim]`.

## DESIGN-2 — MLX shipped the Metal GDN kernels four days ago

**Plan:** the kernel work is *"publishable on its own"* because *"nothing in the open currently
ships"* a gated delta rule on Metal.

**Evidence:** **MLX PR #4020 merged 2026-09-15.** The claim as written is falsifiable in one API call.

**Change:** the narrowing that survives is real but much narrower: the merged work is **forward-only,
no backward**, and the latest MLX release (v0.32.2, 2026-08-25) **predates the merge**, so nothing
*released* ships it yet. A claim about a *trainable* or *released* Metal GDN, or about the f64 parity
suite itself, is still defensible. The unqualified claim is not. This does not change what gets built
— K1-K7 are still the only way this Mac runs the hybrid — it changes what may be said about it.

## DESIGN-3 — Apple Silicon has no fast path, and installing `fla` on the Mac is a trap

**Evidence:** `is_flash_linear_attention_available()` gates on cuda/xpu/mlu — **no MPS**.
`is_causal_conv1d_available()` is CUDA-only. `causal-conv1d` 1.7.0 is sdist-only, CUDA/nvcc.

**The trap:** the modeling file dispatches through a bare `importlib.import_module("fla")`, **not**
the CUDA-gated helper. So installing `fla` on this Mac makes the import succeed and sends execution
into Triton, which has no Metal backend.

**Change:** **do not install `fla` in the Mac environment.** Treat the Mac depth probe as
**correctness-only** — its timings cannot be extrapolated to the H100 at all. (Minor: fla does ship a
pure-python wheel, so the plan's "Triton, JIT, no wheel needed" is wrong in detail, right in
substance.)

## BLOCKING-1 — AgentPack is gated. A human must accept its terms.

**Plan:** AgentPack is the unlabeled pool and the CPT corpus, 1.8M edits, *"filtered to permissive
repos"*, with the licence-filter design resting on a per-row source-repo licence field the plan itself
flags as unconfirmed.

**Evidence:** `"gated": "auto"`. An authenticated fetch returns **403, "not in the authorized list"**.
A human must accept terms on the dataset page before even the *schema* is readable — so Phase 1 cannot
confirm the column list unattended. Size is **62.91 GB compressed** (not 371 GB); row count 1.8M per
the paper, against 1.3M in a stale README.

**The per-row licence field almost certainly does not exist.** It could not be confirmed (every route
401/403/404, no public repo — genuinely UNVERIFIED), but the authors' paper contains `licen` exactly
twice in 48,665 characters, neither as a data field, and describes no licence-based repo filtering.

**Change:** **`bigcode/commitpackft` becomes the primary licence-filtered pool.** It has exactly the
field the design needs — a documented per-row `license` from a 13-value permissive set, plus `repos` —
at 702k rows, MIT, **ungated**, 40x smaller. The plan demotes it to a "human-authored contrast set";
that is backwards given what is actually reachable. AgentPack stays the target distribution for the
*teacher* and *held-out* work once the gate is accepted, because it is genuinely what DevCouncil
verifies — but it cannot be the unattended CPT corpus.

## BLOCKING-2 — Four of the open-mixture datasets fail a licence filter; three will not load

**Plan:** *"Licences are confirmed before each one enters the pool; the list below is the shape, not a
final roster."* Confirmed now:

| Dataset | Finding | Disposition |
| --- | --- | --- |
| `facebook/anli` | **`cc-by-nc-4.0` — non-commercial** | **Out.** Disqualifying, not an edge case |
| Devign (rehost) | `c-uda` | Out of an automated permissive filter; needs a human call |
| `mteb/stsbenchmark-sts` | licence literally **`unknown`** | Out until resolved |
| `nyu-mll/glue` (MNLI, STS-B) | **`other`**, card defers to upstream terms | Needs a human call |
| `PolyAI/banking77` | **script-only**, fails under `datasets>=3.0` | Loadable only via a pinned older `datasets` or manual parquet |
| `AmazonScience/massive` | **script-only**, same | Same |
| `microsoft/CodeReviewer` | **does not exist on HuggingFace** (404 with valid auth, control-tested) | Zenodo DOI `10.5281/zenodo.6900648`, CC-BY-4.0, 4.82 GB |

**Change:** the open task mixture is rebuilt from the subset that survives. CLINC150 (with its
natural out-of-scope class, which the plan correctly identifies as a free `noul`) and SQuAD 2.0 remain
the load-bearing ones. **The two held-out task families must be chosen from what actually survives** —
holding out a family that was never going to load is not a holdout.

## STACK-1 — "One lockfile, one image" does not resolve. The teacher needs its own.

**Plan (S1):** *"One lockfile (uv) and a container image: `transformers` at one main-branch SHA with
`Qwen3_5`, `flash-linear-attention`, `causal-conv1d`, torch 2.11 cu128, vLLM for the teacher."*

**Evidence (four real build failures on this host, not analysis):** a single lockfile holding both the
trainer and vLLM pinned `torch==2.11.0+cu128` on paper while pulling **both CUDA wheel families** —
`nvidia-cuda-runtime==13.4.92` beside `nvidia-cuda-runtime-cu12==12.8.90` — and installed
`torch 2.14.0+cu130`. `causal-conv1d`'s source build then stopped the image:

```
RuntimeError: The detected CUDA version (12.8) mismatches the version
that was used to compile PyTorch (13.0)
```

vLLM drags in `torchaudio`, `torchvision` and `torchcodec`, each with its own torch constraint, and
`--index-strategy unsafe-best-match` — itself only needed *because* of vLLM — lets them mix across
indexes.

**Measured by splitting them:**

| | Combined | Training only | Teacher only |
| --- | --- | --- | --- |
| Packages | 206 | **72** | 196 |
| torch | claims `2.11.0+cu128`, installs `2.14.0+cu130` | `2.11.0+cu128` | `2.13.0` |
| CUDA 13 wheels | 12 | **0** | 12 |
| `unsafe-best-match` | required | **not needed** | **not needed** |

The two environments want **different CUDA majors**, so they want different base images. That is a
property of the dependency graph, not a preference.

**Change:** two lockfiles and two images — `train.lock` (CUDA 12, `-devel` base because causal-conv1d
compiles) and `teacher.lock` (CUDA 13, prebuilt wheels only). This costs nothing the plan values: the
teacher runs once for ~40 minutes at the head of the block and the trainer for ~30 hours after it, so
they are never resident together — the plan's own session ordering already separates them in time.
It also **removes** the dependency-confusion caveat that `unsafe-best-match` carried, rather than
accepting it.

**Follow-on: torch is pinned to 2.10, not the plan's 2.11.** `causal-conv1d` ships prebuilt wheels per
`(cuda x torch x cxx11abi)`, and its `setup.py` downloads the matching one rather than compiling —
when one exists. The v1.7.0 release's cu12 wheels **stop at `torch2.10`**, so at 2.11 it falls back to
a from-source `nvcc` build. Pinning one minor version back turns the single most fragile step in the
image into a wheel download, which is the better engineering choice independent of any host: it takes
a from-source CUDA compile off the critical path of a $1,085 block, along with the `-devel` base, the
six `-gencode` targets and the OOM exposure.

Five further S1 findings, each a real failure rather than a predicted one:

1. `uv pip install --system` dies on Ubuntu 24.04 (PEP 668, externally-managed). The image installs
   into a venv at `/opt/venv`.
2. `uv pip compile` **strips the `--extra-index-url`** from its output unless `--emit-index-url` is
   passed, so the lockfile stopped describing its own resolution and the build could not find
   `torch+cu128` at all.
3. A build launched as `podman build ... | tail -40` reported **exit 0** — `tail`'s status, not the
   build's. A failed build looked like a passing one. This is the same shape as SAFETY-1: a success
   signal not attached to the thing being checked.
4. `causal-conv1d`'s CUDA compile fans out one job per core and the OOM killer took `cc1plus` on a
   1.89 GiB builder. The message names the *compiler*, so it reads as a toolchain problem rather than
   a resource one. `MAX_JOBS=4` now lives in the Containerfile: peak memory is `MAX_JOBS`-shaped, not
   total-RAM-shaped, so raising the builder's RAM alone only moves the core count at which it dies.
5. **qemu cannot run `nvcc`** — `uncaught target signal 11 (Segmentation fault)`. Ordinary
   compilation emulates fine at ~1.4x, measured on `gcc`; generalising that to "cross-arch works
   here" was an inference, and it is false for the one compiler this image needs. This is what makes
   finding 4's fix insufficient on its own and the torch 2.10 pin necessary.

## DESIGN-4 — The 40-hour cap costs more than the whole budget estimate

**Plan:** block cost $1,085 at ~34 h, wall-clock cap 40 h, program total "about $1,250".

**Evidence:** the plan's **rates are correct** (8xH100 SXM $31.92/hr; the Lambda page lists
*per-GPU* prices, $3.99, which is the misreading to avoid). But the cap is not costed:
**40 h x $31.92 = $1,277 for the block alone**, before session 0 ($10) and storage ($150). At the cap
the program is **~$1,437**, not $1,250.

**Change:** this is a real risk, not an error — the plan is honest that $1,250 is the estimate and
$2,000 the ceiling, and $1,437 is inside it. But the *cap* should be stated as what it costs, so
"we hit the cap" is not also a silent 15% budget overrun. Note also: GH200 and H100 PCIe are
**1x-only**; A100 80GB is **8x-only**. And `lambdalabs.com/pricing` now 404s — the site is `lambda.ai`.

---

## KERNEL-1 — The plan's K7 formula is a different mixer. It would have shipped a wrong kernel.

**Plan:** K7 `gdn_gate_prologue` computes
`α = exp(−exp(A_log)·softplus(a + dt_bias))`, `β = σ(b)`.

**Evidence:** that is the **Mamba2/SSD** decay, from a *different mixer in the same file*
(`nanolab/mixers.py:621-642`). `A_log` and `dt_bias` are **not GDN parameters**. The actual GDN gate
is:

```
alpha = sigmoid(a_gate + decay_bias).clamp(1e-4, 1.0)     # mixers.py:803
```

This is the most dangerous single error found: K7 is described as a "small elementwise pass, later
folded into K1/K2", so it reads as the safest kernel in the set. Written from the plan it would have
been confidently wrong, and every kernel that folds it in would inherit the error.

**Change:** K7 implements the sigmoid form. The parity fixtures (now generated) are the guard.

## KERNEL-2 — The α clamp is present, not absent, and it is load-bearing

**Plan:** *"Clamp α ≥ 1e-4, the NaN guard nanolab needed on a real GH200 run"* — phrased as something
the kernel lane adds.

**Evidence:** the guard is **already in the reference, in two places** (`mixers.py:712` and `:803`),
with a comment corroborating the GH200 incident down to step numbers. It is not optional: measured,
the unclamped path diverges by **3.99e-04** against a **5.8e-07** noise floor.

**Change:** the clamp is part of the *reference semantics* the kernel must reproduce, not a defensive
addition the kernel may apply differently. A port that clamps at a different point in the expression
is not bit-comparable.

## KERNEL-3 — "Two places" is one assignment

**Plan:** the published rule *"replaces A_{t−1} with A_t in both"* of two places.

**Evidence:** line 734 is the **only** rule-dependent statement; it feeds `logM` (735) and `A_corr`
(744→753). Patching "two places" in a port creates a **second source of truth** that can drift.

**Change:** the Rust f64 reference has exactly one rule-dependent assignment.

## KERNEL-4 — GDN has no key-head concept (second, independent confirmation)

The reference lane reached the same conclusion as the config lane by a different route: the GDN path
has no key-head repeat at all, so the reference **cannot** confirm interleave-vs-tile for GDN.
`repeat_interleave` is the repo's *attention* convention (`mixers.py:316`, `:458`).

Two independent confirmations. **DESIGN-1 stands: the interleave test does not belong in the GDN suite.**

## KERNEL-5 — `rule="published"` is not meaningful at L=1

At L=1 the two rules are **provably the same function** (`S_{−1} = 0`), so a fixture at L=1 named
`published` proves nothing about the rule. The generator now asserts the rule gap is **exactly zero**
there, rather than letting a coincidence read as a passing check.

## KERNEL-6 — The repo's fp64 provenance covers `repo` only

nanolab's `verify_gdn_wy.py` has **no `rule` parameter**; its "<1e-9 fp64" provenance describes the
`repo` rule. The published rule's only in-repo check was fp32 at 1e-4.

**The fixtures generated in this program are the first fp64 goldens for the published rule.** That is
a stronger claim than the plan makes, and it is the one worth writing down.

## PRODUCT-1 — "dcverify still owns PASS" is refuted; the rule holds for a better reason

**Plan:** rule 7, *"dcverify still owns PASS"*.

**Evidence:** there is **no `PASS` token anywhere** in DevCouncil (`rg -uu`, zero hits). `passed` is
derived by `verify.StatusFromGaps` in a **Go host the plan never mentions**; dcverify itself has no
pass concept and exits 0 to mean "this report is valid".

**Change:** rule 7 survives, and is *stronger* than written: `passed` is a pure function assigned at
one site, so emitting it is **architecturally unavailable**, not merely forbidden. The integration
seam for admission is **`Gap`, not `Finding`** — wire to that.

## PRODUCT-2 — DevType command IDs can contain query text

**Plan:** *"The usage store keeps command IDs and counts, no query text"*, which is why training pairs
must be templated.

**Evidence:** confirmed for the *schema*, but qualified at the *key*: some command IDs **are** query
text (`"routed.\(query)"`, `"math.\(expression)"`). `recordUsage` accepts any string; the guard lives
at the call sites, and its comment shows it was added to stop file growth — **privacy is a side
effect, not a design property.**

**Change:** the plan's conclusion (template the pairs) stands; its stated reason does not. Recorded as
`GAP-DT-EPHEMERAL-CONVENTION`: a fourth unguarded call site would persist query text and **no test
would fail**. If an opt-in local-only query log is added later, that guard needs to become a property
of the store, not a convention at three call sites.

---

## Confirmed as claimed — no action

- **Teacher:** `Qwen/Qwen3.8-27B` exists, dense, Apache-2.0, ungated, and shares the student's 248,320
  vocab. `mlx-community/Qwen3.8-27B-4bit` is **complete on this Mac** (15 GB, 3 shards, no
  `.incomplete`). The rubric iteration and the kappa >= 0.6 check need no download and no rental.
- *"Qwen3.8 shipped nothing dense under 27B"* — **VERIFIED**. The official lineup is six repos; the
  only sub-27B entries are MoE.

## One correction to an earlier finding in this session

`google/gemma-4-31B` exists and is **Apache-2.0 and ungated** (not Gemma Terms). But the locally
cached `gemma-4-31B-it-assistant-bf16` is **926 MB against ~62 GB expected — a partial cache, not a
usable model**, and no 4-bit MLX Gemma-4-31B is on this machine. The alternative teacher is *not*
available locally; only the Qwen one is. `AUDIT/local-environment.md` has been corrected.
