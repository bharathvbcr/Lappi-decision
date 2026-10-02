# Fable on improving the model after F seeds 0-1, and on Mac inference (2026-10-02)

Asked by the lead at ~05:35 UTC 2026-10-02 on the human's request, verbatim: "Ask fable to further enhance and improve the model and optimizations for inference on mac". Advisor: claude-fable-5-1, through a read-only lane. Recorded verbatim from the ruling heading on (source sha256 263cf792efb0fadf8da3bfba5cfb514cda5dc69df4d186c021172649cc899fbc, as the lane reported it). The lead follows it under the human's standing instruction to follow Fable's recommendations, except the items Fable names as the human's, which the lead puts to the human. Nothing in it changes a gate.

# Fable ruling: improve the model after F seeds 0-1, and optimize Mac inference (2026-10-02)

Read-only lane: no repo file written, no git state changed, nothing launched, killed or benchmarked on the Mac or the box. Rules 2, 3, 4, 8 untouched. Builds on `AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md` and `human-decisions.md`; nothing there is repeated unless new evidence changes it.

**Navigation.** DevMap gen 3037, `degraded_reason` "stored extraction payload is obsolete", `is_fresh:false`; GitPulse `REPOSITORY_TRUST_REQUIRED` on every facet; no ListAgents. Everything below is Read/rg on committed files, `git diff` of the in-flight edits, and read-only ssh (06:0x UTC: GPU 0%, 13.6 GiB used, seed 2 not yet training; 9 ledger rows). Labels: **[V]** read or computed here, **[I]** inferred, **[U]** unverified. Nothing was run on a GPU.

## Q1(a) Diagnosis: two seeds, same accuracy, opposite abstention

**What the verdict files say [V, `/home/ubuntu/p4-v4/suite-verdicts-s{0,1}.jsonl`, pass-1 `row_logits_1` log-softmaxed].**

| OOD prose, 60 cases | seed 0 (f4feac15) | seed 1 (aeca8d69) |
|---|---|---|
| Z top-1 / abstained | 50 / 54 | 0 / 0 |
| Z runner-up | 7 | 0 |
| non-Z winners: median top-vs-Z gap | **1.06 nats** (min 0.12) | **9.06 nats** (min 6.56, max 10.8) |
| non-Z winners: median top-vs-second margin | 1.00 nats | 4.25 nats |
| median p(noul) | 0.992 | 1.1e-4 |
| which class wins (rows 0-3) | 2/3/5 of 10 | 12/20/12/16, value-consistent across both passes |

Unseen-language is milder on seed 1 (0 Z top-1, but Z runner-up on 13/60, median gap 7.3 nats, min 1.38, winners 22/14/13/11), so a margin threshold could catch some of those cases where it can touch none of prose. Scrambled is fine on both (58, 56; gap ≥8.7 nats to Z).

**Reading.** Seed 1 is not "slightly less abstaining": it classifies prose and C/Java hunks *as code hunks*, confidently, consistently under derangement, spread over all four defect classes. Seed 0's ten non-abstentions are near-ties. So **[I]** the feature "this context is not the kind the task was trained on" was learned by seed 0 and not by seed 1; both learned the classifier equally (val_top1 identical). The prose noul supply is 2,010 SQuAD rows (1,005 paragraph + 1,005 question, `HANDOFF/noul-v3b-2026-10-01.md`), 1.1% of 183,252 letter rows, against 23,819 MMLU/CSQA rows that teach "exam prose → answer a letter" under another task marker; the OOD suite is MMLU/CSQA *val* questions under `code.defect_class` (`ood.py:10-12`). The behaviour that decides the suite is whether the 2,010 rows generalise from Wikipedia prose to exam prose, and one pass leaves that unpinned.

**This is not new to F [V].** J4 (v3, three seeds, no layer flag): prose 4/11/4 and unseen-language 40/20/56 (`AUDIT/j7-avg-ood-diag-2026-10-01/noul-rank-fable-i-diag.json`, rows 741c7ebe/6fcba23d/d3c19098); their average 0/0/0 (1eb74618). J6(b) (v3 + flag): 44 (f4958492). F (v4 + flag): 54, 0. Tabulated: with the flag 2 of 3 seeds ≥44, without it 0 of 3 >11, on two data versions. Not significant, but it is the evidence, and it says the OOD quantities are **bimodal per seed**, on v3 and v4 alike. Any one-seed arm reads as a Bernoulli draw.

**Candidates, weighed.**
- *Final batches under the globally shuffled plan* (`shards.rs:1899-1902`, one width bucket per batch): **demoted [I]**. Late drift at the cosine floor (1e-6, 1e-7 for layers 0-7; Adam-normalised steps ≤ lr each) can flip 1-nat near-ties, not produce 9-nat confidence. No trajectory diagnostic exists: `box_q_f.sh` passes `--checkpoint-every 100000`, so seeds 0-2 have only their final checkpoint [V].
- *Late loss spikes* [V, both logs parsed]: seed 0 at steps 2568/3265/3411/6139 (30.9/22.5/54.2/26.0), seed 1 at 7007 (85.4). Clipped at `max_grad_norm 1.0` (`backbone.py:927,1193`), so a spike moves weights little; it is a symptom (a gold 20-85 nats below the winner) of the overconfidence that makes the Z boundary brittle. Late fit is heavy on both: 48% (seed 1, steps 6000-6999) and 52% (seed 0, 7000-7999) of logged batches under 0.01.
- *noul supply at 4.9%*: the number that matters is prose at 1.1% (question form 0.55%) against a 23,819-row counter-signal; **the cause, with the next item.**
- *Abstention as an unpinned boundary*: yes, and the files show it is pinned on seed 0 and absent on seed 1, not "between".
- *Cosine floor lr/10* (`schedule.rs:94-96`, hard-coded in `real_ft`): a recipe nit, not the cause; cheap to change in v5 (below).
- *Seed 1's extra in-distribution abstentions* [V arithmetic]: both-noul rows = abstentions − derangement disagreements = 923 − 774 = **149** (seed 0: 98), all on CLINC classification/domain/within_domain (32/99/18 vs 34/52/12); MMLU/CSQA/defect/in_scope are still disagreements only. The CLINC oos re-key the human already chose is the relevant fix there.
- *Needle 8K* (40/61 vs 61/61): the suite sits past the trained width (7,936 vs 8,473 median tokens); seed 1 extrapolates, seed 0 does not. Expected to vanish once the suite is 8,192 real tokens and v5 composes to 9-10k (both decided). Seeds 3-4 run automatically (spread 0.344 > 0.30).

## Q1(b) Remedies, ranked against cost and the queue

| Remedy | Verdict | Why | Enters where |
|---|---|---|---|
| **More, and closer, non-code noul sources (G6)** | **first** | the failure is generalisation from Wikipedia prose to exam prose; only data pins it. Two routes, the human's: (i) *contrastive pairs* — MMLU/CSQA **train** questions rendered under `code.defect_class` with gold Z beside their own letter-gold rows (rule 3 safe; the v5 exclusion list removes E_val hits from both uses) — the sharpest "task marker × content" signal, but it makes the suite's prose category near-in-distribution by form, which changes what the suite measures; (ii) a *third prose source* not in the suite (approved own-repo prose: READMEs, commit bodies, issue text) keeps the suite's meaning. Either way ≥3,000 prose-noul rows, question-form favoured | v5 build; **human** (G6, and suite meaning under (i)) |
| **Checkpoint cadence + trajectory readout** | second, $0.5/seed | `--checkpoint-every 1000` (10 × 8 GB on the box, disk [U]) and the 180-case OOD suite scored per checkpoint (`--score-checkpoint … --ood`, ~1-2 min each) answers "set early or flipped late" for the first time, and gives a last-k average as a report-only row | v5 recipe; no pre-registration (report-only) |
| **noul loss weight** (`--noul-weight w` on Z-gold letter rows, w≈3-5) | third; new flag, new pre-registration | the only recipe lever on the 23:1 ratio; risks the in-distribution cap (seed 1 is already 8.4%) — F3's ≤2% defect-class bound is the guard | a v5 arm, **3 seeds** (bimodal quantity; one seed proves nothing): ≈$34, **human** funds |
| J6(g) permutation | queued, stands | targets the general-family permutation gates, not prose | — |
| J6(f) (no layer flag) | fires automatically | read it as one Bernoulli draw into the table above: prose ≥30 says v4's data suffices without the flag; ≤9 is consistent with both hypotheses | — |
| Averaging (J7' avg / avg-np) | queued, stands; **expect loss of prose abstention** | J4's average went 47/34/67 → 0 with Z runner-up by 2.6-4.8 nats; a mean of a 9-nat-confident seed 1 and a 1-nat seed 0 lands on seed 1's side | — |
| ens3 | queued, stands | mean log-softmax is dominated by the confidently wrong member; the pre-registered reading (f-j7prime `reading_ens3`) already covers it; serving cost is G8 | human (G8) |
| EMA / SWA over final steps | low | weights barely move at the floor; subsumed by the last-k average above | report-only |
| Cosine floor → 0 (`--min-lr`) | cheap, enters v5 | `schedule.rs` hard-codes lr/10; needs a flag + oracle parity | v5 recipe |
| Calibrated-margin half / explicit abstention head | **cannot rescue this failure** | seed 1's winners carry 4.25-nat margins (p_top ≈ 0.98); a threshold that catches them abstains on most of val. The margin half still belongs in serving for seed 0's near-ties | — |
| Distillation from an ensemble | not for v5 | bakes in whichever side the ensemble fell on | later, human |
| Second epoch on noul + long rows only | **no** | a short final phase on a subset is the recipe for forgetting and for blowing the in-distribution cap | — |

## Q1(c) A larger base

`Qwen3.5-4B-Base` exists [V, `AUDIT/external-surfaces.md:23`, the HF sibling list]; its config is not in the repo and I did not fetch it, so parameters, layer mix and kernel limits (head_dim 256, GDN key dim 128 — `config.rs:134-226` refuses others) are **[U]**. Cost ≈2× F's per seed ≈ $21 [I]; memory: F's 2B peaks 61.6 of 97.9 GiB with fp32 masters + AdamW (≈16 B/param of state), so a 4B at `--batch-tokens 35403` does not fit one GH200 without halving batch tokens or an 8-bit optimizer [I]. Mac: ≈8 GB bf16 weights, ≈2× prefill time (≈2.6 s per 8K decision on today's kernels) [I]. **Ruling: not now.** The MMLU/CSQA ceiling is a population question (G1), the product population already meets its gates on the 2B, and a base change buys nothing for the failure in Q1(a). The human's, and I recommend against until v5 reads.

## Q1(d) v5 recipe, ranked, with cost

Per F seed [V]: train $10.19 (973cd4e3) + eval $0.37 (f4feac15) + needle control $0.12 (a4f244f0) = **$10.68**.

1. Data (human-decided): CLINC oos re-key; compose 9-10k with `--max-seq-len 10240`; `qd-prep containment` exclusions + attestation v2; the 8,192-real-token needle suite. **Plus Q1(b) item 1** (human's route choice).
2. Recipe, no new words needed: F's flags; `--checkpoint-every 1000` + per-checkpoint OOD scoring (report-only); `--min-lr 0`; `--option-permutation-seed` iff J6(g) wins; `--lower-layers-*` kept unless J6(f) and seeds 3-4 argue otherwise (2-of-3 vs 0-of-3 leans keep).
3. Tier-B: `--train-attention-mask none` + fused AdamW iff P2 and the outcome run pass: −20-40% per seed [I].
4. Seeds 0-2, the 8K spread rule for 3-4, J5' ×3 on v5's exact path.
5. First v5 arm, pre-registered now: noul-weight ×3 seeds.

Cost per seed: $11-12 at 9-10k composition on F's step time, $8-9 with no-mask, +$0.6 eval/controls, +$0.5 trajectory scoring. Three seeds + J5' ≈ **$70** (as told); seeds 3-4 +$24; the noul-weight arm +$34. Launch is the human's yes.

## Q2(a) What exists and what is measured

**Nothing qd-metal has a ledger row [V]: `rg qdm-parity ledger/` is empty; `GAP-QDM-GPU-GATES-NOT-RUN` is open (owner: human, a permission rule).** Every Mac GPU number below is either MLX/torch (rows) or tessl's random-weight layer bench, by report, measured before today's Condvar change; a post-Condvar baseline must be taken before any A/B.

| Quantity | Value | Source |
|---|---|---|
| tessl 2B forward, bf16 weights, f32 acts, 307 launches | 1k 168 ms (6.1k tok/s), 2k 344 ms, 8k 1,318 ms (**6,213 tok/s** after TensorOps attention; 4.1k before) | `AUDIT/mac-speed-2026-09-28.md` §7, by report |
| MLX prefill | 7,606 / 7,528 / 5,285 tok/s at 1k/2k/8k, peak 5.75 GB | row 45595699 |
| Parity | MLX fp32 vs torch fp32 max Δ 0.012, 50/50 argmax (8bcc3cec); torch-MPS bf16 is the outlier (0.8); **qd-metal vs torch: NOT RUN**; CPU: tokenizer ids 24/24 + 20/20, snapshot header tests pass | `HANDOFF/mac-gpu-lane-2026-09-29.md`, `qd-metal-2026-09-29.md` |
| Weights | 3,763,692,048 B bf16 (`model.safetensors`); load time unmeasured | `j7-export:125` |
| Latency per decision, memory per request | **never measured**; `qd-metal-bench` times prefill + 17 rows only | `bench.rs:9-12` |
| `qd serve` | **wires no model backend** (`main.rs:47-49`); only `qd-metal-parity` and `qd-metal-bench` touch the model | [V] |
| span slot | not served: no pointer head in qd-metal (`backend.rs:589-595` refuses PointerStart) | [V] |

**In-flight uncommitted edits (another session's; in flight, not landed) [V by diff].** (i) `qd-metal/tokenizer.rs`: `encode_untrusted` splits added non-special tokens such as `<tool_call>` char by char so untrusted context cannot emit their ids. (ii) `bin/parity.rs`: an all-NaN GPU output used to pass every absolute gate (`fold(0, f64::max)` drops NaN); `gate_over` now fails on non-finite. (iii) `qd-runtime/service.rs`: the cold-start build runs outside the lifecycle lock (status/ping answer during a model load); a second poison fails closed until `reset()`; `serve.rs` releases a connection slot on panic. (iv) `qd-preflight/tristate.rs`: a failure is never overridden by a later NotRun. (v) `Cargo.lock` regenerated. None changes model numerics; (iii) is what makes a multi-second Metal cold start livable. Defect in (i): it is one-sided — training rows were tokenized by Python's `encode`, which emits the `<tool_call>` id, and the 24/24 tokenizer parity used `encode`. The neutralisation belongs in the renderer's escape rules on both sides (pinned by `render_contract`), and `qd-metal-parity --tokenizer-only` needs an adversarial prompt.

## Q2(b) Where a decision's time goes

**The second pass does not double prefill [V].** The prefix is begin/version/task/route/question/context (`render.rs:524-546`); every slot's options live in its *suffix* (`render.rs:645-765`: `<|qd_slot|>…<|qd_options_begin|>\nA. …\nZ. noul\n<|qd_options_end|>\n<|qd_answer|>`), and the deranged pass re-renders the suffix only (`render.rs:569-580`, `answer.rs:305-317`). Both passes continue from one snapshot: GDN/conv at batch stride 0, prefix K/V shared (`model.rs:34-39`). The doubling exists only in the Python eval (`real_ft_run.py:4230` re-runs whole val batches) and would exist in an MLX/torch sidecar without a prefix cache.

**Where the time is [I; sizes from `config.json`: 6 attention layers, kv_heads 2 × head_dim 256 f32 = 24 KB/position; 18 GDN layers × 16×128×128 f32 = 18.9 MB; conv 1.3 MB].**

| Phase | 500-token choice, k=4 | 8.5K choice + span |
|---|---|---|
| host embedding gather (`model.rs:462-478`, maps the 1 GB table, converts T×2048) | 1-2 ms | 15-30 ms |
| prefill GPU | ≈80-90 ms | ≈1.32 s (by report) |
| `PrefixState::digest()` after prefill: host SHA-256 of every state buffer (`backend.rs:487-490`, `model.rs:132-143`) | 32 MB ≈ 15-25 ms | 224 MB ≈ 100-150 ms |
| 2 continuations (≈50 tokens, chunked GDN + `attn_prefix_rows`, ≈300 launches each) | ≈10-20 ms each | ≈30-60 ms each |
| digest after each decode (`backend.rs:661-665`) | 2 × 15-25 ms | 2 × 100-150 ms |
| **total / host-hash share** | **≈170-230 ms / 25-35%** | **≈1.8-2.0 s / 20-25%** |
| span slot | — | not served |

SHA-256 throughput assumed 1.5-2.5 GB/s (`sha2` 0.10.9 with `cpufeatures`) [U]. Every digest also maps shared buffers, which waits for all encoded GPU work: three serial host syncs per single-slot request (one after the prefill, two per choice slot) besides the score readbacks. After today's Condvar change each sync costs the wait itself, not an extra ~1.25 ms.

**DevCouncil's shape** (`train-plan:118-125`: ≤128 boolean items, 32 concurrent) is 32 slots × 2 decodes = 64 continuations per prefix: the decode phase dominates there (whether multi-slot requests match the trained row shape is [U]). Two questions over one diff do **not** share a prefill, because the question precedes the context; moving it after is a format change (both renderers, `render_contract` goldens, a v5 rebuild) — the human's.

## Q2(c) Optimizations, ranked

Every item is proven by one committed benchmark: `qd-metal-bench --decision T=512,2048,8192 k=4` (prefill + 2 decodes + digests + readback, the real request shape), flags for each item, **interleaved A/B, min-of-7 and median**, dispatch and sync-wait counts from `tessl::infer_trace`, written as a `quick` row to `ledger/mac-qd-metal-<date>.jsonl`. Gate suite = `qd-metal-parity` (thresholds fixed in `parity.rs`, unchanged) + `tests/gpu.rs`.

| # | Item | Expected | Gate risk / oracle | Owner |
|---|---|---|---|---|
| 0 | Run the GPU gates and the post-Condvar baseline | the first qd-metal numbers | parity vs the torch fp32 dumps (`dump_torch_reference.py`) | human permission + lane |
| 1 | **Device-side state digest**: a deterministic reduction kernel over the state buffers (fixed tree order, 32-byte output) replacing host SHA-256; host path kept behind a flag for A/B | −25-35% at 500 tokens, −20-25% at 8K; removes 3 syncs/slot | none to logits; oracle: the three `gpu.rs` readonly tests + a CPU test that any one-bit flip in any buffer changes the digest | tessl kernel + qd-metal |
| 2 | **`qwen35::embed_rows`** (exists, `tessl/src/qwen35.rs:2814`) replaces the host gather | −1-2%, one fewer host map per run | exact (bf16→f32 is exact): parity must be bit-identical | qd-metal |
| 3 | **Batch the two passes and all slots** as one `Model::run` of B rows (stride-0 prefix by design; `*_varlen` kernels now exist for unequal suffixes). Cross-*request* batching shares no state and, at T≥500, compute-bound GEMMs: launch amortisation only — deferred behind this | halves decode launches and digests; dominant for 32-slot requests | logits equal to the sequential path to 1e-5 (extend `gpu.rs`); touches a contract pin: `tests/answering_procedure.rs` counts 2 decodes per choice and `docs/schema-api.md`'s table — add `decode_slots` with a default two-call impl and count *queries*, not calls | qd-runtime trait + qd-metal |
| 4 | **Span serving**: gather the prefill's residual at line starts + the continuation's answer position, `qwen35::rms_norm` with w_offset 1 (the head reads `last_hidden_state`, post-norm [I], `backbone.py:1080`), two [H,H] GEMVs + dots + abstain vectors from `span_head.safetensors`; qd-train's Rust span head (L-head, 9 parity cases) is the port | makes the product's third differentiator servable at ≈0 extra GPU time | oracle: `heads.py` on dumped hidden states, bit-level; the abstain row last (`answer.rs:69-71`) | qd-metal + qd-runtime release |
| 5 | **Wire `qd serve --metal`**: a `qd-metal-serve` bin (qd-runtime cannot depend on qd-metal: tessl links Metal) using `Release::open` → `MetalBackend::start` → `Runtime::from_release` → `Service::with_factory` → `Server::bind` | the product path exists at all | CPU-testable with the reference backend; `--idle-timeout-ms` already exists (10-min default is the plan's; a longer value for a 3.8 GB resident model is a launcher flag, flagged to the human) | qd-metal |
| 6 | tessl GDN chunk forward on TensorOps, then SwiGLU + cast fusion (the agreed order; attention is done) | −8-15% at 8K [by report] | numeric: full parity re-pass; fixtures `rule="published"` (rule 9) | tessl |
| 7 | Aligned serving blob for mmap: safetensors forbids gaps, so either order page-multiple tensors first or write a separate *packed* blob (tessl's `pack_linear_weights_bf16` layout, which `prepare_layer` builds on the host today) hashed into `release_manifest.json`; qd-metal wraps it with `bytesNoCopy` | cold load from seconds (unmeasured) to ~0.1 s + page-in | `weight_hash` unchanged by construction; a round-trip test in qd-export | qd-export + qd-metal |
| 8 | Prompt order (context before question) | N× fewer prefills for N questions per diff | format change: **human** | — |
| 9 | Weight quantization | int8 weight-only: memory 3.76→1.9 GB and load, ≈0 speed (GEMMs at T≥500 are compute-bound on TensorOps); int4 moves every letter logit, forces a calibration refit and will likely fail the 0.10-nat parity bar [I]; tessl's `quant-prep` path does not dispatch (`lib.rs:132`) | **last**; not on a 64 GB Mac | — |
| 10 | MLX backend | tessl beats MLX at 8K (6,213 vs 5,285), 80% at 1-2k; MLX needs Python or a new Rust binding (human) and has no readonly snapshot or 17-row head; keep MLX as the Mac fp32 oracle | stay on tessl | — |

Scoring only the 17 letter rows is already done (`model.rs:803-854`, `score_answer_rows`); the eval harness's full-vocab path is the GH200's business.

## Q2(d) Frozen, and what starts now

Frozen while F and its arms are read: `qd-lane8` at a502670 and every box waiter; F's eval path (Python/torch — qd-metal and tessl changes cannot touch F's numbers); `parity.rs` thresholds; tessl kernel numerics that `qd-train-metal`'s rung (b)-(d) ladders pin (coordinate any tessl kernel change with the ojas sessions; `bench_qwen35_layers` and the GDN fixtures stay green at every commit).

Starts now on the Mac, CPU only, `CARGO_BUILD_JOBS=2`, after L-prep/L-j6g finish compiling, no model load: items 1 (kernel + host tests; MSL compiles at build), 2, 3 (trait + reference-backend tests), 4 (Rust head port + oracle fixtures), 5 (bin + CPU tests), the `--decision` bench harness, and the J6(h)/v5 pre-registrations. GPU runs wait for a free Mac and the permission rule.

## One ranked to-do list

1. **Lead, now, $0:** pre-register the noul-weight arm and `--checkpoint-every 1000` + per-checkpoint OOD scoring + `--min-lr` into the v5 plan; add the `--decision` bench harness and the tokenizer adversarial prompt.
2. **Human:** G6 route — contrastive MMLU/CSQA-train pairs (changes the suite's meaning) or a third prose source; fund the 3-seed noul-weight arm (≈$34).
3. **Mac, CPU:** `qd-metal-serve` wiring (item 5) — the product path does not exist today.
4. **Mac, CPU:** device-side digest (item 1) behind a flag.
5. **Mac, CPU:** `embed_rows` (item 2) and batched `decode_slots` (item 3) with the contract-pin change.
6. **Human:** the `cargo test/run -p qd-metal` permission rule; then item 0: GPU gates + post-Condvar baseline → first `mac-qd-metal` rows.
7. **Mac:** span serving (item 4).
8. Read J6(f), J6(g), seeds 3-4, J7' avg/ens3 as pre-registered — expect avg to lose prose; do not fund further one-seed OOD arms.
9. v5 build and launch on the human's yes (≈$70; +$24 if the spread rule fires).
10. tessl GDN/SwiGLU fusion (item 6), then the packed mmap blob (item 7); quantization and MLX stay off the list.

**Plainly the human's:** G6 route and the suite-meaning consequence; noul-weight arm funding; the 4B base (recommend no); prompt order; the qd-metal GPU permission; the idle-timeout default; ens3 serving cost (G8); everything already listed as open in `human-decisions.md`.

## GAP ids to append (the lead appends)

- GAP-FABLE-INFER-RULING-NAVIGATION-2026-10-02 (DevMap gen 3037 obsolete payload; GitPulse untrusted on every facet; no ListAgents; ruling used Read/rg/ssh)
- GAP-OOD-PROSE-ABSTENTION-BIMODAL-ACROSS-SEEDS-2026-10-02 (J4 4/11/4, J6(b) 44, F 54/0; seed 1 classifies prose at ≥6.5 nats; one-seed arms cannot read it)
- GAP-F-NO-INTERMEDIATE-CHECKPOINTS-FOR-TRAJECTORY-2026-10-02 (`--checkpoint-every 100000`)
- GAP-F-SEED1-BOTH-NOUL-ROWS-ON-CLINC-FAMILIES-2026-10-02 (149 vs 98; domain 99 vs 52; closes with the oos re-key or not — measure in v5)
- GAP-SCHEDULE-MIN-LR-HARDCODED-LR-OVER-10-2026-10-02
- GAP-QDM-NO-LEDGER-ROW-ALL-MAC-INFERENCE-NUMBERS-BY-REPORT-2026-10-02
- GAP-QDM-STATE-DIGEST-HOST-SHA256-PER-DECODE-2026-10-02
- GAP-QDM-TOKENIZER-UNTRUSTED-ENCODE-ONE-SIDED-2026-10-02
- GAP-QDM-BATCHED-DECODE-TOUCHES-ANSWERING-PROCEDURE-PIN-2026-10-02
- GAP-RT-QUESTION-BEFORE-CONTEXT-BLOCKS-CROSS-QUESTION-PREFIX-CACHE-2026-10-02
- GAP-QD-SERVE-WIRES-NO-METAL-BACKEND-2026-10-02
- GAP-QDM-SPAN-HEAD-NEEDS-POST-NORM-HIDDEN-AND-QUERY-ROW-2026-10-02 (extends GAP-J7-EXPORT-SPAN-HEAD-UNSERVED)
