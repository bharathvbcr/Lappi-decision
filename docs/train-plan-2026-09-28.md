# Lappi as the developer workhorse — plan, 2026-09-28

Asked on 2026-09-28, in four parts: *train the 2B fast* (after reading seven external decision
models); *borrow code and replay data*; *make inference fast on the Mac through tessl and gusset,
called from Go*; *make Lappi a jevgrep-style workhorse for DevCouncil and other apps, including
GitPulse commit messages.*

Labels: **[V]** read or run this session, **[I]** inferred, **[U]** unverified. External numbers are
README / card claims, not reproduced here. Internal numbers cite a ledger row or a file.

## The shape of it: three lanes, one dependency

| Lane | Produces | Depends on |
| --- | --- | --- |
| **A — Train** | the only thing that makes a model | nothing |
| **B — Serve on Mac** (Go → gusset → Rust → tessl Metal) | a fast backend | A's checkpoint to validate against (base weights suffice until then) |
| **C — Workhorse task families** (DevCouncil relevance, GitPulse commit decisions) | data + evals | A's pipeline to train on |

A runs first because it is the only lane that produces a model. B and C are parallel preparation, not
a queue: B can reach whole-model parity on the **base** weights before A finishes, and C's data
builders are ordinary `qd_data` families.

---

## Where things stand (verified 2026-09-28)

| Fact | Evidence |
| --- | --- |
| Base weights cached | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base` [V] |
| One real 2B FT: ~2k commitpackft rows, lr 1e-5, `--optimizer master`, 3 seeds | `ledger/gh200-ft-commitpackft-2026-09-22.jsonl` [V] |
| choice (`code.commit_intent`, yes/no) 46/90 on every seed = constant-'A' | rows `144ee025` `6172cb03` `c33dc423` [V] |
| score (`code.change_scope`) 38 / 54 / 55 of 90 vs 37 constant — **the pipeline learns** | same rows [V] |
| "Trails its control in 17/18 arms" is the 256×2 from-scratch model, not the 2B | `AUDIT/operator-holdout-model.md` [V] |
| 50,177-example mutation corpus on disk; the 2B has never seen it (`qd_data.mixture` builds only commitpackft/SQuAD families) | `data/pool/commitpackft-corpus-v2/manifest.json`, `python/qd_data/mixture.py` [V] |
| Model: 24 layers = 18 GDN (16 heads, dk=dv=128, conv k=4) + 6 full attention (h256, 8q/2kv, output gate, partial RoPE 0.25, θ=1e7) | `config.json` [V] |
| Held-out 300 hand-labelled diffs: **not on disk** | `data/` holds only `pool/` [V] |

## Borrowed code (user-approved 2026-09-28)

Cloned to `~/Code/research/external/`: **RSI-Jev** `8f34a4f` (MIT), **decider** `23579f7`
(Apache-2.0). **Both trainers are CUDA-hard-coded** (RSI `release_train.py:42`, decider
`train.py:88`), and decider's Metal op is inference-only (`mps_ops.py:76`). So: **port pieces into
`qd_train`, don't run their trainers.** [V, via code read]

Port list (small tensor code; keep MIT/Apache attribution in the file header):

| Piece | From | Into | Note |
| --- | --- | --- | --- |
| Layer-wise LR (layers 0–7 at 0.1×) | RSI `fit.py:116-153` | `optim.py` | `MasterWeightAdamW` takes one flat lr, and four drivers overwrite `group["lr"]` each step (`backbone.py:903`, `byte_train.py:355`, `real_ft_run.py:707`, `ft_toy_run.py:338`). Needs per-group `lr_scale` at all four, or ×0.1 silently becomes ×1. Test asserts layer-0's effective lr |
| Replay toward the base's own answers | RSI `prior_kl` `train.py:61-93` (base→model direction) | `qd_train` loss | decider describes this only in a model card; RSI has the code |
| Option permutation + un-permute; label smoothing over own options; checkpoint averaging | RSI `encode.py:72-80,201-221`, `fit.py:431-437`, `train.py:189-200` | `trainer.py` | pairs with the permutation-consistency control the handoff already wants |
| Numeric-option first-token fallback | RSI `encode.py:53-69` | render | Qwen collides on ` 0`..` 3` |
| 8-gram replay decontamination, refuse-on-hit | RSI `build_specialist_replay_corpus.py:44-86,158-172` | `qd_data` | |

Not ported: RSI scorer head / RL, decider `train.py`. Calibration: `calibration_fit.py` already
does per-kind NLL temperature — nothing to port.

**Replay data** must pass `qd_data.licences` like everything else. Clean: MMLU (MIT),
CommonsenseQA (MIT), clinc (cc-by-3.0, already registered). Share-alike: ARC. **Refused:** SciQ
(cc-by-nc-3.0 — in RSI's own replay set), OpenBookQA (unknown), toxic-chat, customer-support-tickets
(nc). Self-distillation replay (base model's own answers, cached once) needs no licence at all and is
the default.

---

## Lane A — Train

Rules 2, 5, 8 stand. Fast means cheapest decisive experiment first, not fewer seeds.

**A1. Zero-shot baseline (rung 1).** Base weights, 17-row letter slice, on every val set. $0 on the
Mac. Every FT number is read against this.

**A2. Diagnose the stuck choice slot on the 2k set.** 3 seeds × ~9 min on one GPU, ~$0.70.
Arms: `--optimizer bf16` with **beta2 = 0.95** (decider's value; `optim.py` measured bf16 second
moments settling 50% low at 0.999), layer-wise LR, lr 3e-5. Dump the val **prediction distribution**.
This matters more now: `commit_intent` (message + change → yes/no) is the closest existing task to a
jevgrep or GitPulse question, and it is the one that collapsed. 90 val rows cannot separate arms —
build A3's val first or accept that A2 only detects gross failure.

**A3. Mutation corpus into the mixture — the critical path.** ~1–1.5 d. A `code.defect_class` family
(4-way class choice + span) rendering qd-mutate `Example`s into `Request`s. Repo-disjoint split.
Rebuild shards once, together with the `language` metadata and permutation hook the handoff asks for.
New families are trainable; `DEFAULT_HELD_OUT_FAMILIES` (`code.language_id`, `qa.answerability`) is
unchanged.

**A4. Linear control for FT rows + operator-holdout arm.** ~1 d, parallel with A3.
`paired_margin_vs_linear` is not built for FT; without it a run has an accuracy and no verdict.

**A5. The run.** One GPU, 3 seeds, bf16 or master (A2's winner), layer-wise LR, ~15% replay
(self-distilled), 1 epoch, then per-kind temperature. Est. $10–25; under $20/job → no rule-4 yes.
The GH200 box's status is unverified (marked safe-to-delete 2026-09-23).

**Train on a rented GPU, not the Mac — measured.** torch 2.12.1 MPS on the M5 Pro (64 GB),
transformers' pure-torch GDN fallback, batch 1, 5 warmup + 10 timed steps. Session scratchpad
`ftbench/results_torch.jsonl`, **not a ledger row** (a benchmark, stopped part-way on battery; the
MLX rows were never taken):

| mode | seq | s/step | tok/s | peak GB |
| --- | --- | --- | --- | --- |
| forward only | 1024 | 1.19 | 860 | 9.6 |
| forward only | 2048 | 2.33 | 881 | 13.9 |
| full FT, AdamW | 1024 | 6.41 | 160 | 32.4 |
| full FT, AdamW | 2048 | 26.7 | 77 | 48.2 |
| top-4 layers trainable | 1024 | 2.05 | 501 | 13.2 |
| top-4 layers trainable | 2048 | 5.15 | 398 | 22.0 |
| full FT, batch 4 | 1024 | — | — | 77.0 — exceeded RAM, no timed step |

Full FT at 77–160 tok/s means ~2–4 days per 30M training tokens on the Mac, on mains power, with
nothing else using the GPU. A single GH200 runs the same epoch in hours for ~$3/h. **Mac role:
zero-shot baseline (A1), teacher labelling for C2 (MLX 27B), and inference — not the A5 run.**
The forward-only 860–880 tok/s is also the **baseline lane B must beat**: the tessl ceiling
arithmetic is 4–5k tok/s, i.e. ~5× over torch MPS.

**A6. Add lane C families** once A5 shows the 2B beats its control on the mutation task.

---

## Lane B — Mac inference: Go → gusset → qd-runtime → tessl

User decision 2026-09-28: **Metal work starts now**, reversing the plan's "no kernel before the
8×H100 gate" ordering (`README.md`, `crates/qd-runtime/src/backend.rs:3-4`). Recorded here as that
decision.

**The load it has to carry** (jevgrep's pattern [V via summarised fetch]: ≤128 boolean items or
~38 KB per request, 32 concurrent): **throughput-bound batched prefill**, not single-slot latency.
Arithmetic [I]: ≈3.0 GFLOP/token; tessl's measured bf16 TensorOps peak 26.6 TFLOP/s
(`tessl/README.md:146`) caps prefill at ~8.9k tok/s, realistically 4–5k → ~2 s per full 38 KB
request. **The biggest win is not a kernel — it is prefix reuse:** cache snapshots by
`prompt_digest` (`backend.rs:108`) so many questions over one file context skip prefill. Lappi's
prefill-once / answer-many-slots design is exactly this.

**Why the Mac is slow, and the kernel order that follows from it:** `AUDIT/mac-speed-2026-09-28.md`.
In short: torch runs at ~12% of the chip's measured matmul rate because transformers' GDN fallback is
~13–16k tiny fp32 dispatches per forward (a 63-step Python solve loop plus a per-chunk loop), and the
benchmark also paid for a full-vocab lm_head that decision inference never needs. Kernel priority:
fused chunked GDN (K1) → fused prologue (K7) → conv1d+SiLU → gated RMSNorm → readonly decode (K2) →
attention epilogue → 17-row lm_head → projection fusion. A 15-minute profile (audit §5) confirms the
ranking before K1 is written.

**Phase 0 — fix the reference before any kernel (1–2 d).** Verified this session:
Qwen3.5 gates are `beta = sigmoid(b)`, `g = −exp(A_log)·softplus(a + dt_bias)`, `α = exp(g)`
(`modeling_qwen3_5.py:516-518`). Canonical tessl `tests/gdn_gates.rs` pins a **sigmoid** α taken
from nanolab's toy GDN, per `AUDIT/gdn-reference-and-contracts.md` §4.2 and `docs/plan-corrections.md`
KERNEL-1 — which checked the wrong model. The α∈[1e-4,1] clamp has no counterpart in transformers.
Repin; add Qwen-shaped published fixtures (H=16, D=128, log-space g, l2-normed q/k). **GAP-K7-GOLDEN-PINS-NANOLAB-SIGMOID-NOT-QWEN35-GATE.**

**Phase 1 — interim fast path (2–3 d).** `mlx_lm` 0.31.3 is already installed in
`~/.venvs/ml` with `qwen3_5` support [V via design agent]; put it behind `DecisionBackend` as a
sidecar. Gate: letter-logit parity vs transformers on real weights. Pinning it as a Lappi
dependency needs a yes.

**Phase 2 — new crate `crates/qd-metal`** implementing `DecisionBackend`, kernels in canonical tessl
as new files (avoids the `PROMOTED.len()==44` assert). Op inventory:

| Exists in tessl | Adapt | Missing |
| --- | --- | --- |
| bf16 TensorOps GEMM, RMSNorm (store 1+w at load), `mlp_silu`, `flash_attn_rows` h256, `flash_attn_decode` h256 | Qwen partial RoPE (first 64 dims, pair p↔p+32, no V-norm, scale 1/16 — a new variant, not a mask), 17-row lm_head gather | bf16 embed, GDN gated RMSNorm, causal conv1d k=4, gate prologue (K7), GDN prefill (K1), readonly GDN decode (K2), attention output gate, safetensors loader (header + mmap — no new dep) |

Gate: per-layer hidden-state parity vs torch fp32 on real weights. **~1.5 wk.**

**Phase 3 — K1 as a recurrent port** (template: `mlx_lm/models/gated_delta.py`, MIT) + K2 readonly
(state broadcast at stride 0, no store). Gates: published fixtures; prefill(T)+decode(1) ==
prefill(T+1); snapshot hash unchanged after readonly. **~1 wk.**

**Phase 4 — throughput.** Cascade shared-prefix attention for multi-slot decode, varlen packing, a
single GPU-owning scheduler thread with a **bounded** queue that refuses when full. gusset pool ≈32–40
thin handlers (`handle.go:53-66`); 38 KB inputs use the `NewBuffer` path (`MAX_INLINE_INPUT=4096`).
DevCouncil's `rust/gusset-engine` + `backend/go_orchestrator/gussetfn` is the existing adopter
pattern. **~1.5 wk.**

**Phase 5 — chunked K1** (decider `mps_ops.py` WY structure as reference) — an optimisation: a
sequential K1 is estimated at 10–200 ms per 10k tokens vs ~2 s of GEMM [I, to be measured].

**Isolation caveat:** gusset's docs say Metal driver faults are SIGKILL-class and not survivable
in-process. Keep `qd serve` (Unix socket) as the crash-isolated deployment of the same backend.
**GitPulse is Rust/Tauri** — it links `qd-runtime` directly; gusset is the Go path (DevCouncil).

**Quantisation later:** Q4/Q8/W8A8 paths change the 17-row logits → new calibration hash → treated as
a new model. Ship bf16 first.

---

## Lane C — Workhorse task families

The interface does not change: jevgrep's 11 questions are all boolean [V via summarised fetch of
`requests.ts`], which a `choice` slot over yes/no plus `noul` covers. What changes is **data**.

**C1. DevCouncil code relevance (jevgrep-shaped).**

| Family | Question | Labels by construction from |
| --- | --- | --- |
| `code.relevance.file` | does file F help answer query Q | commit history: message = Q, files touched = positives, same-repo untouched files = hard negatives |
| `code.relevance.range` | does lines a–b of F implement/test the behaviour | touched hunks = positive spans (the `span` slot neither Jev nor jevgrep has) |
| `code.file_role` | implementation / caller / test / fixture / helper | DevMap graph edges (callers, tests-of) + paths |

commitpackft is **one file per commit**, so it has no same-repo negatives. The source must be full
repositories: the user's own repos (licence-clean — which ones is a human decision) and, if its
licence clears, SWE-bench train (issue → patched files). Held-out eval: repo-disjoint queries.
**Whether DevCouncil / DevType log real queries is unverified** — if not, that is a named gap, not a
set to assume.

**Product lever (not training):** jevgrep crawls breadth-first with previews because it has no code
graph. DevCouncil has DevMap — graph-generated candidates mean far fewer model calls per query.

**C2. GitPulse commit messages.** Lappi does not generate text. GitPulse already classifies type,
scope and subject from the patch and asks a model only to phrase the subject
(`GitPulse/src-tauri/src/ai/commit_brief.rs:1-16`) [V]. Lappi takes the **ambiguous classifications**:

| Slot | Kind |
| --- | --- |
| type (11 `KNOWN_TYPES`) | choice — fits ≤16 |
| scope (candidate dirs from the patch) | choice |
| breaking change | choice yes/no |
| headline change | span |
| does this draft message match the patch | choice yes/no — this **is** `code.commit_intent` |

Measured this session: only **687 of 69,893** raw commitpackft messages carry a conventional prefix,
227 with a scope [V]. Too few. Labels come from a **local teacher** — `Qwen3.8-27B` MLX weights are
already cached on this Mac (`docs/teacher-plan.md` exists) — plus the user's own conventional-commit
history. Faster: GitPulse keeps its deterministic path, calls Lappi only when ambiguous, and only
phrases with a generator when it must.

---

## Human decisions

Decided by the user 2026-09-28:

- **All of the user's own repositories and data may be used** for training, eval and commit data
  (C1, C2). Recorded in the source registry as the user's own, licence `owner-granted`; held-out
  splits are still repo-disjoint and still refused to training by `qd-train`'s path check (rule 3).
- **`mlx_lm` approved** as the interim backend (B1), **`tokenizers` crate approved** for
  `qd-metal` (B2).
- **Metal work starts now** (lane B), reversing the kernel-after-gate ordering.

Decided 2026-09-29 (user approved the downloads and SQuAD option (a), and delegated the rest:
"choose the best"):

- **Downloads approved:** `cais/mmlu` (test/validation/dev only; `auxiliary_train` stays refused —
  its own card says it is ARC/OBQA/RACE), `tau/commonsense_qa`, `clinc/clinc_oos` `plus`,
  `rajpurkar/squad_v2`, at the revisions pinned in `sources.py`. CLINC `domains.json` only if the
  `clinc/oos-eval` README states a compatible licence (GitHub reports NOASSERTION).
- **SQuAD v2 — option (a):** `qa.answerability` (held out) draws only from a held-out set of SQuAD
  article titles; `qa.answer_span` from the rest. This re-partitions a held-out set; it is permitted
  because the user chose it (rule 2), and ledger rows citing `qa.answerability` from before are
  non-comparable. Closes `GAP-DATA-SQUAD-SPAN-NOUL-LEAKS-HELD-OUT-ANSWERABILITY`.
- **MMLU:** trained on `test` + `dev`, `validation` is the family's val; **MMLU is not a reportable
  benchmark for Lappi.**
- **ARC: off.** MMLU + CSQA fill the general multiple-choice role; ARC adds a snapshot-hash change
  and nothing the go/no-go needs. (Not a licence call — SQuAD is cc-by-sa too.)
- **Replay source:** 15% of MMLU/CSQA rows, replay-only (KL-to-base, no gold CE), disjoint from those
  families' training rows, decontaminated against every val/held-out set. Replay from our own
  training shards is refused: 222/234 candidates overlap held-out `code.language_id` rows built
  from the same commits (`GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN`).
- **bf16 + beta2 0.95 is still refused** past 64 steps (settles 1.95% low vs a 1% bar), so the A2
  arm uses `--optimizer master`; every long phase names its optimizer (the driver refuses otherwise).
- **Provider: Lambda 1×GH200** (the existing image, sync scripts and instance name). The hourly rate
  is read from the price page at launch; nothing hardcodes it.
- **Sync: the Mac pulls.** The box writes `phase-<n>.done`; the Mac pulls, verifies, writes
  `pulled-<n>.ok` back. Terminate — including the cap guard — waits for the final `pulled.ok`
  within a bounded grace window, and never destroys the only copy of checkpoints and ledger.

Still open:

1. Whether DevCouncil / DevType / GitPulse **query logs exist** — permission is given; existence is
   unverified.
2. The 300 hand-labelled held-out diffs: needed for the ship gate; do not exist.

## The GH200 campaign — one GPU, 2–3 days (user decision 2026-09-28)

The user will rent **one GH200** for up to **2–3 days** of training. That is ~$110–220 at the
~$1.49/h implied by the 2026-09-22 FT rows (`c1db813d`: $0.2242 for a ~542 s epoch) — confirm the
provider's hourly rate before launch. Above rule 4's $20 single-GPU line, so **every job longer than
~10 h carries a wall-clock cap, auto-terminate and a cost estimate**, and the campaign as a whole is
approved by this decision. One GPU means `GAP-FINAL-TRAIN-HAS-NO-MULTI-GPU-PATH` does not block.

**Throughput is the first measurement, not an assumption.** Our trainer's measured rate is 3,364
sequences / ~542 s on the 2k set (6.2 seq/s); tokens/s was not recorded on those rows. External
claims (decider: 455M tokens in 5.3 h on a GH200 ≈ 24k tok/s) are a ceiling to compare against. Hour
0 of the box measures tokens/s on our stack, and the schedule below is re-cut from it.

### What gets built on the Mac first (plugged in; CPU/light work, no GPU needed)

Every hour of GH200 is spent training, not debugging. Before renting:

1. **A3** `code.defect_class` family + shards (and the `pool_id` licence join).
2. **A4** linear control for FT rows + operator-holdout arm.
3. **Ported pieces:** layer-wise LR (per-group `lr_scale` at all four drivers), `prior_kl` replay,
   option permutation, checkpoint averaging, replay decontamination.
4. **C1 / C2 builders** over the user's own repositories (approved): relevance from git history,
   file roles from DevMap, GitPulse commit slots.
5. **General decision families** from licence-clean public sets (MMLU, CommonsenseQA, CLINC, SQuAD
   v2 — all already registered or MIT) so Lappi is a workhorse for *any* app, not only code.
6. **Campaign driver:** one script that runs the phases below in order, checkpoints every ≤2 h,
   resumes from the last checkpoint (`HANDOFF/resume-2026-09-20.md`), writes every ledger row,
   rsyncs ledger + checkpoints to the Mac after each phase, and auto-terminates the instance at the
   cap. Smoke-tested end to end on the Mac with a 2-layer stand-in.
7. **Box image:** the existing GH200 stack (torch 2.10 + cu128, fla 0.5.2 — `AUDIT/operator-holdout-model.md`
   provenance) — fla gives CUDA GDN kernels, so the box does not use the slow pure-torch fallback.

### The schedule (~60 h, re-cut after hour-0 throughput)

| Phase | Hours | What | Gate / output |
| --- | --- | --- | --- |
| 0 | 1 | smoke, throughput, zero-shot baseline (A1) on every val set | tok/s row; rung-1 numbers |
| 1 | 2 | A2 diagnosis: 3 arms × 3 seeds on the 2k set | why choice collapsed |
| 2 | 8 | **Teacher labelling on the GPU** (Qwen3.8-27B, bf16 ≈ 54 GB, fits 96 GB HBM): letter-logit soft labels for C2 commit slots, ambiguous C1 file roles, and soft targets for distillation (RSI's `soft_ce`). Prefill-only scoring — no generation, no vLLM | labelled sets rsynced home. Needs: 27B weights in HF format on the box, transformers support for its arch **[U]** |
| 3 | 4–6 | **A5 go/no-go:** mutation family only, 3 seeds | paired margin vs linear control, operator-holdout margins. **If the 2B does not beat its control here, stop and report — do not spend the rest** |
| 4 | 24–30 | **The workhorse run:** full mixture (mutation + C1 + C2 + commitpackft families + general decision sets + ~15% self-distilled replay), 1 epoch, 3 seeds, longer context (8–16k with gradient checkpointing, since jevgrep-shaped requests are ~10k tokens) | three checkpoints, full eval rows |
| 5 | 6–8 | ablations, 1 seed each: no replay; no layer-wise LR; hard vs soft labels | which recipe pieces earn their place |
| 6 | 2 | per-kind temperature fit, all gates, checkpoint average, export bf16 safetensors | release candidate + ledger |
| buffer | 6–8 | preemption / reruns | |

Rule 8 holds throughout: phases 0, 1 and 5 are `quick`; only phases 3 and 4 (3 seeds, full
schedule, no subsample) can promote anything, and promotion itself is still the human's call (rule 2).
Held-out families and the repo-disjoint held-out splits never go on the box's training path (rule 3).

### Mac pre-rental results (2026-09-29) and the schedule changes they force

Measured by the GPU lane; `HANDOFF/mac-gpu-lane-2026-09-29.md` has every row.

- **The training path works end to end on the real 2B** (MPS, frozen `c65d7da`): shard build → FT →
  checkpoint → resume → val scoring → ledger. The **resume is bit-exact**: `7a93bb47` restarted at
  step 80 and matched the uninterrupted `f0f0fa95` on loss-log digest, final loss and every verdict.
- **bf16 optimizer state is refused past 383 steps at beta2 0.999**, and at beta2 0.95 past 64
  steps. Every long phase runs `--optimizer master`. On the Mac, master on the 2k set peaked at
  80.8 GB and was killed, so the full epoch arm cannot run here (`GAP-MAC-EPOCH-ARM-CANNOT-RUN-ON-THE-2K-SET`).
  That is a Mac limit, not a GH200 one.
- **Phase 2 (teacher labelling) moves to the Mac.** Qwen3.8-27B 4-bit through MLX scores 0.67
  items/s (374 tok/s, 16 GB peak; row `c6d2170e`): **~8.2 h per 20k items, overnight, $0**, against 8
  GH200 hours. The labels are from a **4-bit teacher**; how far they are from bf16 labels is
  unmeasured, and they are recorded as such. The 27B tokenizer reproduces the 2B's ids 100/100, and
  every option letter is one token. The campaign drops to **~52 h**.
- **Rung-1 zero-shot on Mac (eval `9004c495`, choice 43/90, score 9/90) is not the baseline of
  record.** It was scored through torch bf16 on MPS, which drifts ~0.8 nats on letter logits
  (`GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT`). Phase 0 on the box re-measures rung 1 in bf16 on CUDA,
  or it is re-scored on the Mac in fp32 or MLX.
- **GDN is ~64% of a torch forward** (`c352cee1`), confirming the tessl kernel order. MLX prefill
  reaches 7.6k tok/s on the 2B.

### Found by the committed-tree smoke (2026-09-29, `25000da`) — fixed before renting

- **The trainer could not run a campaign.** `tools/real_ft_run.py` hardcoded a 30-min wall-clock cap
  (`WALL_CLOCK_CAP_S`), so phase 4's 24–30 h epoch would have stopped at 30 min. It also recorded
  every row `quick=True`, so under rule 8 no phase could ever promote.
- **The pipeline's default corpus is this repo's own git history at `--rev HEAD`**, not the plan's
  pool. At HEAD it includes today's commits: rows up to 37k tokens that MPS cannot backprop, a
  corpus that moves with every commit, and a capped (NotRun) data snapshot. **Decision: `--rev`
  is always a pinned full sha** (the 09-22 GH200 rows used `0632f693`). **Phase 3 trains on
  `code.defect_class` only**, as the plan already says, with no repo-history rows.
- Proven on the real 2B at `25000da`:
  - The fp32 optimizer, lower-layer LR and option permutation run: memorise arm `246249ed`, loss fell
    on both channels.
  - The final full rebuild `3e577d0b`: 295,496 train sequences, padding waste 4.1%; 3,568 replay-only
    sequences; the contradiction drops are CLINC 8 rows, SQuAD 58. It is superseded, because it was
    built at `--rev HEAD`.
- Not yet proven: the epoch → `--score-val` → `--verdicts-out` → `ft_linear_control` path on the real
  tower. The smoke's epoch arm was refused on the 23–32k-token repo-history buckets.
  **Proven at `c97c39f`** (`ledger/mac-phase3-smoke-2026-09-29.jsonl`):
  - ft `cfb2eb01`: 193 steps, `steps_exhausted`.
  - eval `75ffa23e`: 22 verdicts.
  - `paired_margin_vs_linear` `00b57d10` ran; the baseline was ahead on 11 val rows.
  - All are quick rows on 300 training rows, so this is not a decision signal.

### Vocabulary: full tokenizer vocabulary (user decision 2026-09-29)

- **The corpus-built remap could not encode what it was not counted over.** It kept 56,170 of
  248,077 ids and refused 608 of 2,210 held-out rows (row `d849d700`). It would also refuse any
  served input with a dropped id, and serving is the product.
- **Decision (c):** keep every tokenizer id.
  - `real_tokenizer_pipeline.py --vocab full` is now the default, built with
    `qd_train.remap.full_vocab_remap` (the identity table).
  - `--vocab corpus` keeps the old remap.
  - The recipe carries `vocab: full`, so full and trimmed sets never share a `recipe_hash`.
  - The Rust side (`qd-metal`) never had a remap; it reads `vocab_size` from the model config.
- **Cost, measured:**
  - Training: +480M parameters. Live tensors on MPS (bf16, AdamW, one [4, 965] batch) peak at
    16.06 GiB, against 12.45 GiB trimmed: +3.6 GiB.
  - Under the fp32-master recipe the arithmetic adds about 5.7 GB more. The GH200 fit is
    arithmetic, not yet a measurement.
  - Serving on the Mac: about 1 GB in bf16.
  - The 7.16 GiB saving in `HANDOFF/memory-2026-09-20.md` no longer applies.
- **The first full-vocabulary smoke swapped, and it was not the tensors.**
  - Its footprint reached 61 GiB, against 32.5 GiB for the trimmed smoke. It was killed after
    58 minutes and wrote no rows.
  - The MPS caching allocator grew to 30 GiB after three steps while live tensors held at
    16 GiB. torch's default watermarks let it grow into swap on a 64 GiB Mac.
  - `real_ft_run.py` now caps the allocator at the recommended working set, so it reclaims
    its cache below that and fails fast above it
    (GAP-MPS-ALLOCATOR-CACHE-SWAPS-INSTEAD-OF-REFUSING).
  - CUDA frees its cache before raising out-of-memory, so this is Mac-only.
- **Open:** the trained embedding has 248,077 rows, and the checkpoint config says 248,320. The
  export must re-pad 243 rows that no token can reach
  (GAP-EXPORT-FULL-VOCAB-CHECKPOINT-IS-243-ROWS-SHORT-OF-THE-CONFIG).
- **Every campaign shard set is rebuilt under `--vocab full`**, and the phase-3 smoke is re-run on
  it before renting.

### What would change the schedule

- **Throughput below ~8k tok/s** → phase 4 drops to the mutation + C1 + C2 families only, general
  sets as replay.
- **Phase 3 fails its control** → the campaign stops at ~15 h; the remaining budget is not spent.
- **Teacher arch unsupported** by the box's transformers → teacher labelling moves to the Mac (MLX,
  slower, on power) and phase 2 is skipped on the box.

## Execution state (2026-09-28, paused on battery)

| Lane | State | First action when resumed |
| --- | --- | --- |
| B0 tessl K7 golden | running in a separate cloud session (tessl repo) | review its branch |
| A3 mutation family | stopped at the read stage, no edits kept. Found: `examples.jsonl` rows carry no licence field; join to the pool on `pool_id` for it | build the family |
| B1 `mlx_lm` backend | stopped at the read stage, no edits kept. DevMap had no store for the worktree | build the sidecar + parity gate |
| Mac benchmark | stopped; torch rows above, MLX rows not taken | take the MLX rows inside B1 |
| C1/C2 data | not started; waits for A3 (same files in `qd_data`) | enumerate the user's repos |

Nothing compute-bound runs on the Mac while it is on battery. Lane A's GPU work is remote anyway.

## Timeline (effort, one agent lane each; A is the gate)

| Week | A — train | B — Mac serving | C — workhorse families |
| --- | --- | --- | --- |
| 1 | A3, A4, ported pieces, campaign driver + Mac smoke | Phase 0 (K7 golden), Phase 1 MLX sidecar | C1/C2 builders, general decision sets |
| 2 | **GH200 campaign, 2–3 days** (phases 0–6) | Phase 2 against **base** weights | evals on campaign checkpoints |
| 3 | read results, calibration, release candidate | Phase 3; swap in the trained checkpoint | DevCouncil / GitPulse integration via MLX sidecar |
| 4–5 | | Phases 4–5, gusset wiring into DevCouncil | move integrations onto `qd-metal` |

A go/no-go on the 2B vs its control: ~15 h into the campaign. A trained workhorse release candidate:
end of the campaign, ~$110–220. A Mac Metal backend at parity: ~3 weeks. Throughput-tuned: ~5 weeks.
