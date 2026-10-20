# Lappi v2 plan (2026-10-20, revised after the second audit)

Status: **plan, owner-ruled scope; nothing here is measured.** Every number is a ruling, a
citation, or a proposal marked as one. No ledger row backs any v2 claim yet, and v1's LoRA-2B
has no ledger row either (`ledger/runs.jsonl`: zero LoRA-2B rows as of this date [V, advisor]).
Labels: **[V]** read in a file or primary source, **[I]** inferred, **[U]** unverified.
DevMap was stale on Lappi-decision, tessl and ojas and GitPulse's git facets were deferred, so
nothing here is graph-confirmed (`GAP-V2-PLAN-SURVEY-DEVMAP-STALE-2026-10-20`).

Revision note: the first draft was audited the same day by three read-only auditors (citations,
Gemma 4 feasibility, rules and consistency) and reviewed twice by a Fable advisor. Wrong facts
were corrected (§1, §6), four owner rulings changed (§2.9-2.12), and gates were added (§6, §10).
A third gap audit (plan-to-card coverage, open `gaps.jsonl` records, Fable) added rulings
§2.13-2.15, the evaluation and calibration rules in §6, serving operations in §7, and the
blocking gaps in §10.

This plan **supersedes** for v2 (not for the running v1 LoRA-2B campaign):
- `CLAUDE.md` and `docs/lappi.md` "What Lappi is" (one Qwen3.5-2B model; "does not generate text").
- `docs/caller-contract.md` §5 "Synthetic only", for Vani only (§2.1).
- The 2026-10-18 ruling to port d1 natively to tessl (`docs/third-party-decider-runtime-2026-10-18.md`).

It does **not** touch: `campaign/lora-2b-preregistered.json` (frozen), the ten binding rules,
rule 7 admission-only, rule 3 held-out doors.

## 1. Goal and today's models

Cut the owner's model count. Vani's live config today [V, Vani config + README]:

| Task | Model today |
| --- | --- |
| `career.jd_triage` (4 slots), `jd_reasons`, `form_field` | OpenRouter `liquid/d1-ultra` |
| fallback / escalation (escalation off in live config) | `claude-haiku-5-5` |
| `career.kb_match` | `claude-haiku-5-5` (`kbmatch.model` unset → `llm.triage_model`; `Vani/README.md:345-350`) |
| application drafts | `claude-opus-5-5` |
| similarity, near-dup | EmbeddingGemma 2 on tessl |

Jarvis calls `gemini-3.8-flash` for discovery (`Jarvis/internal/app/discovery.go:26`) [V].

Target: **Lappi-D + Lappi-G + EmbeddingGemma**, all local. The first paid model retired is
d1-ultra (§10). Opus only writes drafts, so retiring it is last and not load-bearing.

## 2. Owner rulings, 2026-10-20

1. **Real Vani data trains, at lower weight.** Teacher readings on real public postings may
   become training rows: **≤25% of each Vani family's train rows, loss weight 0.5, admitted
   only where two teachers read the same posting and agree** (ruling 9 replaced the original
   margin ≥ 0.3). Effective share of loss at the cap is about 14% (0.125 / 0.875) [I].
   Owner hand labels are held out. Personal data never enters a row (§6 gates).
2. **Lappi replaces d1** as Vani's decision model. The d1 eval still runs as the baseline, with
   d1-ultra added as a comparator by an amendment committed before any result is read (§10 P0).
3. **Base family target: Gemma 4** (E4B, E2B bake-off), chosen over Qwen3.5. Ruling 10 decides
   when it is adopted.
4. **Two generative tracks:** Lappi-G (generation adapter on the shared base) and an NCM model
   from scratch. Ruling 11 decides when NCM runs.
5. **Claims:** rkb `Claims.verify.json` + DevPrism own truth; DevMap supplies code evidence;
   Lappi judges `supported | contradicted | insufficient | noul` and never confirms.
6. **Vani-trained adapters are private.** Never published.
7. **Licences for private adapters:** CC BY 4.0, CC BY-SA 3.0, Mind2Web's research-only wording;
   further classes proposed per source.
8. **Best architecture even if hard; Mac now, cloud later.** Experiments run on the Mac; scale
   runs on cloud GPUs later. Rule 4 binds every 8xH100 job (wall-clock cap, auto-terminate,
   cost estimate, human yes).
9. **Real-row filter is two-teacher agreement.** Margin ≥ 0.3 cannot be computed: Vani's ledger
   keeps only the chosen label's probability (`Vani/internal/ledger/annotations.go:27`;
   `devtools/decide/jev/jev.go:454`), and Opus/Haiku readings carry none [V, auditor]. Agreement
   also drops rows where one teacher is confidently wrong, which a margin cannot.
10. **Sequencing: retire d1-ultra on the existing Qwen3.5-2B path first.** Gemma stays the
    target and is chosen by a frozen, kernel-free bake-off run in parallel (§4). Gemma kernels
    and the Gemma training path are built only after it wins.
11. **NCM is parked until Lappi-G exists.**
12. **Real Vani rows never leave the Mac.** Cloud seeds train on synthetic and public rows only.
    A real-row run promotes only as a full-recipe ≥3-seed Mac run (rule 8).
13. **v2's full recipe is a Vani adapter: one epoch over the v2 Vani pool** (synthetic plus
    agreed real rows) on the frozen base, not v5's 525k-sequence mixture. The v5 full recipe is
    12,176 steps × 35,403 tokens = 431M tokens per seed
    (`campaign/lora-2b-preregistered.json:123,127`) [V]; at the measured LoRA rates of
    261.6-335.9 tok/s that is 357-458 GPU-hours **per seed**, about 1,070-1,370 for three [I,
    arithmetic]. The prereg's line 202 figure ("about 500-600 GPU-hours" for 3 seeds) is wrong
    by about 2×. The v2 pre-registration states tokens and projected hours per seed.
14. **Evaluation window: Lappi shadows every posting** (`shadow_every: 1`) while owner labels
    are collected, and labels go to shadow-asked postings. Each slot needs **≥100 paired
    shadow-asked owner labels**, and is promoted only when `Promotion.Verdict()` says
    "promotable" in **two disjoint label windows** (multiple-comparison control across slots).
    `Verdict()` stays as written (`Vani/internal/eval/report.go:122-141`: ≥30 paired, ≥10
    discordant, one-sided 95% lower bound ≥ 0.4) [V]; the pre-registration adds the n and the
    replication condition.
15. **`career.form_field` and `career.jd_reasons` leave the first scope.** Form fields are not
    postings, so `Verdict()` cannot judge them; `jd_reasons` today is a multi-select
    explanation, so single-choice agreement is undefined. Each returns with its own
    pre-registered gate.

On ruling 7: `python/qd_data/licences.py:108-109` already places `cc-by-4.0` and `cc-by-sa-3.0`
in the global `ALLOW` tier [V], so they would flow into any pool. Scoping to private adapters
needs the manifest field in §6. Mind2Web's wording is not an SPDX id and needs a per-source
`admitted_by_human` entry with a justification [V]. Non-commercial is `DISQUALIFYING` by design.

## 3. Model lineup

| Model | What it is | Output | Jobs |
| --- | --- | --- | --- |
| **Lappi-D** | Qwen3.5-2B LoRA first (ruling 10); Gemma 4 E2B/E4B after the bake-off | `choice`, `span`, `noul`; later `score` and a vector `line_label` | every decision in §5 |
| **Lappi-G** | generation LoRA on the shared base, schema-constrained decode | short structured text | bullet rewrites, short form answers |
| **Lappi-NCM** | NCM (arXiv 2610.14162) from scratch | coarse-to-fine text | parked (ruling 11) |
| EmbeddingGemma 2 | on tessl (`tessl/docs/embedgemma2.md`) [V] | vectors | retrieval, near-dup |

Admission-only holds for all (rule 7). Lappi-G output is a draft for the user, never submitted.

**Base vs IT checkpoint.** Lappi-D reads a Base checkpoint. Lappi-G needs instruction-following,
which a Base LoRA learns slower than an IT one. One base file is the goal (one resident model
under memguard's 32 GiB line); the bake-off includes a Base-vs-IT arm to decide once.

**JD cleaning** is mostly per-line labelling (responsibilities, requirement_must,
requirement_nice, pay, benefits, company, legal, noise), extractive and from one shared
prefill. It needs a vector `line_label` slot, which does not fit today's wire (§5), so it is
deferred until JD sectioning is shown to move ranking or drafting quality.

## 4. Base model: Gemma 4 target, bake-off first

**Facts [V, primary sources fetched by the auditor].** Apache 2.0
(`ai.google.dev/gemma/docs/gemma_4_license`; a linked prohibited-use policy is unread [U]).
From the raw `config.json`:

| | E4B | E2B |
| --- | --- | --- |
| layers | 42 | 35 |
| hidden / MLP | 2560 / 10240 | 1536 / 6144 |
| heads / KV heads | 8 / 2 | 8 / 1 |
| head_dim / global | 256 / 512 | 256 / 512 |
| sliding : full | 5:1 (7 full layers) | 4:1 (7 full layers) |
| KV-shared layers | 18 | 20 |

Shared by both: sliding window 512, `final_logit_softcapping` 30.0, tied 262,144 vocabulary,
PLE (~5.6 GiB in bf16), 131,072 max positions.

**What exists [V].** `ojas/gemma-metal` targets E4B inference ("Prove on E4B" is its goal,
`README.md:3`). Real-weight parity against transformers or MLX logits does not exist
(`README.md:50`, `docs/dev.md:134`). It is IT-only, has no prefix snapshot or multi-read, and
E4B token streams are non-deterministic in its shipping mode (`DECISIONS.md` D13). Decode is
about 15.9-23 tok/s against mlx-lm's ~76. It is a port target, not a reference.

**Decision-model risks specific to Gemma 4 [I].**
- *Soft-capping.* `30·tanh(x/30)` keeps the argmax but compresses margins. Calibration, the
  abstention threshold and permutation-consistency figures are refit, never inherited.
- *Long-range evidence.* Format 2 puts the question after the context, so at the answer token
  the 512-token local layers see only the question and options. Posting evidence reaches the
  answer only through 7 full-attention layers (4 distinct global KV sets after sharing). The 8K
  needle gate, which v1 seeds already split on (0.000-0.750, `docs/lappi.md`), is the decisive
  bake-off gate.
- *Slot reads.* Qwen3.5's snapshot is mostly a recurrent state (18 of 24 layers GDN). Gemma's is
  a KV cache, so each slot read attends over the full prefix in global layers. Per-slot cost is
  measured, not assumed.
- *Training memory.* E4B bf16 LoRA at 8K fits under 32 GiB only with per-layer checkpointing,
  slice-only loss and a head-dim-512 flash backward that does not exist (estimate 21-24 GB).
  **E4B LoRA at 8K is not runnable on the Mac today**; E2B LoRA comes first on the Mac.
- *Latency [I, advisor].* v1 prefill is 2.3 s at 8,185 tokens on the M5 Pro (README). E4B has
  about 2.2× the matmul-bearing weights (PLE is a lookup), so about 5 s before any Gemma kernel
  exists. That fits Vani's 10 s batch deadline but not Jarvis's 300 ms.

**Bake-off design (pre-registered before any logit is read).** Rung 1: frozen Base checkpoints,
option logits through transformers on MPS. No kernels, no training. Same held-out rows for every
arm. Arms:
1. Qwen3.5-2B-Base, the v1 base.
2. Gemma 4 E2B Base. Against arm 1 this is matched in size, so it isolates architecture.
3. Gemma 4 E4B Base. Against arm 2 this isolates size.
4. Base vs IT for one family. This decides Lappi-G's checkpoint.

Report per arm:
- accuracy and calibration (ECE), with the softcap applied;
- prefill ms at 2K, 4K and 8K, and per-slot read cost;
- the 8K needle with evidence more than 512 tokens before the slot;
- OOD abstention;
- permutation consistency over derangements;
- peak footprint;
- Q4 vs bf16 decision flips.

The matched-latency margin E4B must clear is written into the pre-registration. Only the winner
gets a LoRA arm, and only then kernels.

**Falsifiers for "Gemma 4 is the better long-term base".** Any of these refutes it:
- E2B does not beat Qwen3.5-2B at matched size.
- The needle gate fails beyond 512 tokens.
- E4B does not beat E2B.
- Q4 serving shifts decisions or calibration.
- 9K prefill is too slow for a caller's deadline.

**Cost of adopting Gemma, after it wins [I].**
- tessl kernels: PLE, KV-share, sliding/global attention, head dims 256/512, softcap, and a
  parity fixture per kernel naming its reference (the spirit of rule 9).
- A Gemma step provider in ojas, plus a CUDA path for cloud seeds. `ojas-cuda` is a probe whose
  Qwen3.5 step refuses compute (`ojas-cuda/src/lib.rs:17-20`) [V].
- Rebuilt tokenizer, remap, shards, spancheck, 17-row slice, calibration, span head, wire
  fixtures and render byte-identity. `tools/real_tokenizer_pipeline.py:167` hard-codes
  Qwen3.5-2B-Base [V].

Raw-text pools and text-level dedupe/containment carry over [V, v1 data survey].

## 5. Task families

The first scope is **choice-only**. `decide.Slot` (`{Name, Options}`) and the wire stay as they
are, so **no cross-app schema change sits on the d1-retirement path**.

| Family | Caller | Slot | Data | Scope |
| --- | --- | --- | --- | --- |
| `career.jd_role_fit`, `_sponsorship`, `_work_mode`, `_seniority` | Vani | choice | synth (`synth_career.rs`) + real agreed rows | **first**; generator written; stated zero in `v6-allocation.json:70-73` [V] |
| `career.kb_match` | Vani | choice, 4 ordered options + `noul` | synth + real agreed rows | **first** (CORAL `score` waits until a score slot is trained elsewhere) |
| `career.jd_reasons` | Vani | choice per bucket: fit / gap / check, 15 reasons each, within the 16 cap | synth + real agreed rows | later (ruling 15): needs a top-reason-per-bucket gate; catalog is 45 reasons (`state/reasons.json`) [V] |
| `career.form_field` | Vani | choice, 6 kinds (`fieldtriage.go:30-57`) | synth (reuse `synth_jarvis.rs` field tables) | later (ruling 15): needs a labelled field set and its own gate |
| `career.req_kb_support` | Vani | choice: met / partial / unmet / noul | synth | later |
| `career.ats_keyword` | Vani | choice: present / variant / absent | synth over DevPrism `resume_ats_check` output | later |
| `career.jd_section` | Vani | vector `line_label` | synth; SkillSpan test as held-out | deferred (needs `schema_version: 2`) |
| `claims.evidence_support` | rkb/DevPrism | choice: supported / contradicted / insufficient / noul | synth with an executable oracle + VitaminC/FEVER (private only) | later |
| `jarvis.target`, `field_fill`, `step_risk` | Jarvis | choice | synth (`synth_jarvis.rs`) + Mind2Web train (private only) | later; v6 capped pools exist [V], no caller wired |
| `browser.tier_route` | Vani/Jarvis | choice | synth | later; plan item only (`Vani/docs/PLAN.md:78`) |

`form_field` maps a field to a *kind*; `jarvis.field_fill` maps it to a *profile key*. Different
questions, kept separate.

**Why `line_label` waits [V].**
- `MAX_SLOTS = 32` (`python/qd_data/schema.py:111`), and a maximal request is already refused by
  `payload_over_cap` (`docs/schema-api.md:571`).
- A response has five fixed slot keys and `deny_unknown_fields`.
- A 6-8K-token posting has far more than 32 lines.
- So it needs a new vector slot type and `schema_version: 2`, landed as one cross-app change
  under the global rule, not as 32-line chunking.

**Cross-app procedure for any later schema bump (global rule).** Land everything in one piece of
work, in this order:
1. Lappi runtime and wire fixtures.
2. The `~/Code/devtools/decide` module.
3. Vani's `go.mod` replace and pins, plus the fixture sha in `jdtriage_test.go:136`.
4. Jarvis `upstream.lock.json`.
5. The DevPrism caller.

The bump is done only when every installed consumer reports the new number. New Vani families in
the first scope still change Vani's pinned fixture (`fixtures/caller/career-jd-triage-fixture.json`;
`synth_career.rs:97-110`), so they also land in both repos together.

## 6. Data plan

**Sources.**
- **Synthetic by rule.** This carries the load. Extend `crates/qd-prep/src/synth_career.rs` (one
  table and one `generate` arm per single-slot family). `claims.evidence_support` needs an
  executable oracle so its labels are exact.
- **Real, agreed.**
  - What exists: Vani's ledger has 2,548 `triaged` events over 2,318 unique postings (d1-ultra
    2,213, jev 283, llm 52) [V, auditor]. Usable volume is roughly 2.2k postings before the
    agreement filter and the 25% cap, so distillation is a supplement, not the volume source.
  - The agreement filter needs a second teacher on the same posting. Haiku is the cheap second
    reader. The re-query cost is an estimate owed before the pool build.
  - Each row carries `source: vani-agreed`, the teacher ids and both readings.
- **Public (private adapters only).**
  - TechWolf Synthetic-ESCO skill sentences (CC BY 4.0).
  - VitaminC and FEVER (CC BY-SA 3.0), for `claims.evidence_support`.
  - Mind2Web train (CC BY 4.0 plus research-only wording) for `jarvis.target`.
  - Re-read each card before its pool build.

**Held out, never trained (rule 3 doors):**
- owner labels;
- SkillSpan test;
- TechWolf skill-extraction eval sets;
- Mind2Web test splits;
- ScreenSpot;
- ST-WebAgentBench;
- SciFact.

**Excluded:** WebLINX, CommitBench, Fliar and unlistedcareers postings (non-commercial), and the
keeper-security form set (proprietary).

**Gates before any real-row pool build.** Each ships with a test that fails against the code
before it.
1. **Content-level leakage door.** Vani exports owner-labelled posting keys plus normalised-text
   MinHash signatures to a held-out path. `qd-prep` (which has MinHash/LSH) refuses any train
   row within the v5 decontamination threshold of them. Test: a reposted near-duplicate of a
   held-out posting is refused. `crates/qd-train/src/held_out.rs` stays path-based; this door
   sits beside it [V: `held_out.rs` has no content door for postings].
2. **Redactor.** Removes emails, phone numbers and signature-block names, and refuses a row that
   still carries one.
3. **KB digest.** `kb_match` and `req_kb_support` rows embed the owner's KB digest. These rows
   are real-and-private: Mac only (ruling 12), never in a public or cloud pool.
4. **Visibility scoping.** Pool manifests gain `adapter_visibility: public | private`.
   `preflight-release` refuses a public export whose manifest contains BY-SA, research-only or
   `vani-agreed` rows.
5. **Mac-only path check.** A cloud-run manifest containing `vani-agreed` rows is refused.

**Loss weight is new machinery.**
- `rg` finds no `loss_weight` in any batch or shard header [V, auditor].
- It must be carried pool → shard → trainer.
- Tests: a 0.5 row contributes half the gradient of a 1.0 row; the span loss and D17 are
  unaffected; calibration fits use unweighted held-out rows.
- Also report held-out agreement-vs-teacher as a function of the real-row fraction, so the
  weight's effect is measured, not assumed.

**Owner labels gate everything.**
- Today: 0 in Vani's live state. The only `labels.jsonl` is a 3-line scratch file under
  `Vani/build/d1-ultra-check/` [V].
- Target: about 30 for the bake-off held-out, then at least 100 shadow-asked postings per
  window, two windows (ruling 14). One labelled posting covers all five first-scope slots, so
  that is about 230 postings in total. Labelled through Vani's stratified audit (RANKING_PLAN
  Phase 1); teacher disagreements among d1-ultra, Haiku and Lappi first.
- A 30-posting held-out split resolves only about ±15 pp. The pre-registration says so, and
  verdicts are precision-at-coverage per slot with intervals.

**Calibration honesty.** A conformal table fitted on teacher-labelled rows is coverage against
the teacher, and is labelled so in every report. Owner labels go to `Promotion.Verdict()`.

**Calibration is shared between slots.** Entries are keyed by `(SlotKind, rows)`
(`crates/qd-runtime/src/calibration.rs:58-111`) [V, advisor], so `role_fit` and `seniority`
share `choice:6`, and `sponsorship` and `work_mode` share `choice:5`. A release that changes any
entry a promoted slot reads re-runs that slot's verdict before it serves; a preflight test
reports the change per affected family.

**Measuring the evaluation.**
- The bake-off's 30 held-out owner labels resolve about ±15 pp, too coarse to separate E2B from
  Qwen3.5-2B. Arm comparisons use paired McNemar tests on the ~2.3k teacher-labelled postings;
  owner labels measure precision-at-coverage.
- Synthetic-to-real shift: a linear discriminator (qd-prep's linear controls) over synthetic vs
  real rows, a MinHash nearest-distance histogram, and owner-label accuracy for a synthetic-only
  arm vs a synthetic + real arm.

**After the teachers retire.** Haiku stays as a 1-in-N audit reader. Abstention rate,
permutation consistency and OOD score are tracked per slot. A 30-posting owner-label refresh
each quarter triggers a retrain when precision-at-coverage falls below the pre-registered bar.
The Haiku second-teacher pass is about 2.2k postings × ~8K tokens ≈ 18M input tokens [I];
price it from the current rate sheet before the pool build.

## 7. Speed and efficiency

- **Lappi-D is prefill-bound.** A decision is a one-token read per slot from a snapshot
  (`docs/schema-api.md:376-405`) [V]. The measured v1 number is 2.3 s at 8,185 tokens
  (README) [V]. Each caller gets a stated latency budget per base before adoption (§4).
- **Shared prefix across postings.** Put the stable prefix (target summary, KB digest) first so
  it is computed once per run. This is a render-order change and a retrain, decided before
  shards are built.
- **One daemon, priority classes.** Jarvis interactive (≤300 ms) runs ahead of Vani batch
  (≤10 s). One resident base, adapters per request, a stated memory budget under 32 GiB.
- **Serving operations.**
  - `qd serve` is a launchd agent (`crates/qd-runtime/src/main.rs:4`) [V, advisor]. It needs an
    install script for its plist, a health check, and a model path and sha named in the
    handoff.
  - Shadow windows and training windows on the Mac are disjoint. Under critical memory
    pressure memguard kills the largest process ≥ 2 GiB, which would be the training run,
    mid-seed, with no log.
- **Release identity reaches the caller.** Replies carry `identity.weight_hash` and
  `calibration_hash` (`docs/schema-api.md:203`). Vani records both per triage row, pins the
  promoted release in `decide.lappi`, and refuses a serve whose identity differs. Vani today
  records only the backend name (`internal/ledger/annotations.go:15-45`) and has one
  `decide.lappi.primary` switch, so per-slot primary routing needs a config field.
- **Quantisation.** Mixed precision only after a measured flip rate against bf16 on held-out.
  Per tessl, Q4/Q8 kernels exist but are not wired into the Qwen3.5 forward [V, tessl survey].
- **Lappi-G is decode-bound.** Speculative decoding is the first lever. gemma-metal's MTP
  acceptance rate is unmeasured [V], so it is not yet evidence. NCM resumes here after Lappi-G
  exists (ruling 11).

## 8. NCM (parked)

**Facts [V].** arXiv 2610.14162 (2026-10-09), Kumbong et al. (Hazy Research), code MIT at
github.com/hazyresearch/ncm. It reports ~10-31× faster generation than the strongest diffusion
LM baseline at matched scale.

**Unverified [U].** The 86.1% vs 58.4% Sudoku figure, and parameter counts (a secondary source
says ~100M). NCM trains from scratch and does not convert a pretrained AR model.

**Resumes when Lappi-G exists.** It is scored on Lappi-G's gates, with a pre-registered kill
criterion: match Lappi-G's held-out quality at lower wall-clock, or stop. Two things need asking
first, as new dependencies:
- adopting `hazyresearch/ncm` and FineWeb-Edu;
- running PyTorch beyond a reference oracle, since language policy allows Python only as
  oracle, binding or throwaway analysis.

A win is ported to ojas in Rust and the Python is deleted.

## 9. Integrations

- **Vani.**
  - RANKING_PLAN step 6 ("Lappi v2 swap when it is served", `Vani/docs/RANKING_PLAN.md:168`)
    is the switch. Shadow 1-in-12 (`config.go:281`), per-slot promotion via
    `Promotion.Verdict()` (`eval/report.go:122`) [V].
  - Wire DevPrism's `resume_ats_check`, which exists and is not called by Vani [V].
  - The extension asks Lappi for `jarvis.field_fill` (profile key only) through a
    native-messaging host. `fill.ts` still never submits.
- **Jarvis.**
  - A new Go caller over `~/Code/devtools/decide/lappi`, after the first scope.
  - Discovery stays on Gemini until a vision path or an AX-tree-only discovery is proven.
  - Approval gates are unchanged.
- **Claims.**
  - rkb's `CONFIRMED` verdicts stay the truth, and `claims.evidence_support` only annotates.
  - Lappi-G may only emit claim tokens that resolve to a `CONFIRMED` record.

## 10. Phases

P0 and the first scope never wait on Gemma.

| Phase | Work | Entry | Exit |
| --- | --- | --- | --- |
| **P0** | v1 open items (below); owner labels (~150); d1 eval amendment adding d1-ultra, committed before any result is read; rulings in the contract docs | now | v1 LoRA-2B ledger rows exist; labels split; amendment committed |
| **P0b, parallel** | rung-1 frozen bake-off (§4), pre-registered | owner labels exist (≥30 held-out) | pre-registered verdict on the base |
| **P1** | first-scope families (§5) on Qwen3.5-2B: generators, the §6 gates, loss_weight, an R2 cap ruling for the new families, a new pre-registration | P0 | pools built and attested |
| **P2** | train on the Mac; real-row arms are full recipe, ≥3 seeds, to promote (ruling 12) | P1 | ledger rows; pre-registered verdict |
| **P3** | Vani shadow 1-in-12 → per-slot `Promotion.Verdict()` → retire d1-ultra per passing slot | P2 | d1-ultra off for every promoted slot |
| **P4** | if Gemma won P0b: tessl Gemma kernels with parity, ojas Gemma step (Mac + CUDA), E2B LoRA on Mac, E4B LoRA on cloud (synthetic and public rows only) | P0b verdict | parity green; bake-off LoRA verdict |
| **P5** | later scope (§5), `schema_version: 2` for `line_label`, Jarvis caller, Lappi-G + claims gate, then NCM | P3/P4 | per family |

**v1 open items** (named here because "B1-B5" is undefined in this repo and collides with
`docs/train-plan-2026-09-28.md:239`'s B1; status per the auditor [V]):
- **LoRA checkpoint converter:** not started. `BUILD_REPORT.md:20,113` says it is unblocked by
  ojas adapter layout v1.
- **D3 quantisation gate tool:** absent from `tools/`.
- **ojas pin 4123dfb verification:** not run (`GAP-OJAS-PIN-4123DFB-SOURCE-DIFF-AND-PIN-GATE-PENDING-2026-10-19`).
- **`test_containment_exclusions.py`:** 4 failures recorded in
  `AUDIT/harden-2026-10-17/w2d-v6-rebuild-2026-10-20/mega-summary.txt:25-33`. Rerun today
  without `QD_PREP_BIN` and outside `mac_heavy.sh`, it gives 4 passed, 29 skipped, which is
  **not run**, not green.
- **Open gaps that would block the first v2 run** (third audit, from `gaps.jsonl`; effective
  open set about 522 ids):
  - pin gate never run: `GAP-OJAS-PIN-4123DFB-SOURCE-DIFF-AND-PIN-GATE-PENDING-2026-10-19`
    and `GAP-W4B-PIN-GATE-BASELINE-NOT-RUN-IN-AN-EXTRACTED-TREE`. Owner-run.
  - heavy-job lock and load gate starving runs:
    `GAP-MAC-HEAVY-LOCK-AND-LOAD-GATE-BIND-LAPPI-ONLY` and
    `GAP-MAC-HEAVY-HOLDER-WAITS-ON-MACHINE-WIDE-CARGO-SLOT-GATE`.
  - containment scan killed near 29 GiB in 4 of 5 runs:
    `GAP-W2D-V6-SCAN-FOOTPRINT-OVER-MAC-HEAVY-CAP`.
  - shard headers: 0 of 89 verify against HEAD (`GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT`,
    `GAP-W4C-V1-HEADERS-STILL-REFUSED-UNDER-V1-RULE`).
  - owner rulings: `GAP-M12-TASK-HOLDOUT-CONTENT-OVERLAP-IS-EVERY-COMMITPACKFT-TRAINING-ROW`
    (rule 3 exposure, 5,148 rows) and `GAP-W3B-V5-DECISION-POOL-FAILS-THE-V6-LENGTH-BAND`
    (P = 0.572 against 0.45-0.55).
  - LoRA run rows: `GAP-LORA-FT-ROW-HAS-NO-PRODUCER...` and
    `GAP-L2-LORA-TRAINER-DISPATCH-AND-RUN-WIRING`.
  - span promotion: `GAP-FT-ROWS-CARRY-NO-SPAN-NULL` and `GAP-W4A-SPAN-REFIT-ROWS-CARRY-NO-SPAN-NULL`.
  - red tests: `GAP-W11A-THREE-TORCH-REDS-ON-MAIN-2026-10-20` and
    `GAP-W6C-QDPREP-CARGO-TEST-FAILS-3-OF-287-IN-BUILD-ROW`.
  - tools absent: `GAP-LORA2B-QUANT-GATE-TOOL-MISSING` (D3) and
    `GAP-W9B-NO-TRACKED-VAL-ROWS-FILE-WRITER`.
- **Open gaps that would block the first v2 run** (third audit, from `gaps.jsonl`; effective
  open set about 522 ids):
  - pin gate never run: `GAP-OJAS-PIN-4123DFB-SOURCE-DIFF-AND-PIN-GATE-PENDING-2026-10-19`
    and `GAP-W4B-PIN-GATE-BASELINE-NOT-RUN-IN-AN-EXTRACTED-TREE`. Owner-run.
  - heavy-job lock and load gate starving runs:
    `GAP-MAC-HEAVY-LOCK-AND-LOAD-GATE-BIND-LAPPI-ONLY` and
    `GAP-MAC-HEAVY-HOLDER-WAITS-ON-MACHINE-WIDE-CARGO-SLOT-GATE`.
  - containment scan killed near 29 GiB in 4 of 5 runs:
    `GAP-W2D-V6-SCAN-FOOTPRINT-OVER-MAC-HEAVY-CAP`.
  - shard headers: 0 of 89 verify against HEAD (`GAP-LDATA-ALL-89-SHARD-HEADERS-DRIFT`,
    `GAP-W4C-V1-HEADERS-STILL-REFUSED-UNDER-V1-RULE`).
  - owner rulings: `GAP-M12-TASK-HOLDOUT-CONTENT-OVERLAP-IS-EVERY-COMMITPACKFT-TRAINING-ROW`
    (rule 3 exposure, 5,148 rows) and `GAP-W3B-V5-DECISION-POOL-FAILS-THE-V6-LENGTH-BAND`
    (P = 0.572 against 0.45-0.55).
  - LoRA run rows: `GAP-LORA-FT-ROW-HAS-NO-PRODUCER...` and
    `GAP-L2-LORA-TRAINER-DISPATCH-AND-RUN-WIRING`.
  - span promotion: `GAP-FT-ROWS-CARRY-NO-SPAN-NULL` and `GAP-W4A-SPAN-REFIT-ROWS-CARRY-NO-SPAN-NULL`.
  - red tests: `GAP-W11A-THREE-TORCH-REDS-ON-MAIN-2026-10-20` and
    `GAP-W6C-QDPREP-CARGO-TEST-FAILS-3-OF-287-IN-BUILD-ROW`.
  - tools absent: `GAP-LORA2B-QUANT-GATE-TOOL-MISSING` (D3) and
    `GAP-W9B-NO-TRACKED-VAL-ROWS-FILE-WRITER`.

v1's own finding matters to v2: training added little out-of-distribution over its base on
JevArena (README) [V, advisor]. The v2 changes aimed at that are real-distribution rows
(ruling 1), disagreement-first owner labels (§6), and the bake-off's OOD abstention gate.

## 11. Open items

- `GAP-V2-PLAN-SURVEY-DEVMAP-STALE-2026-10-20`: surveys could not use the graph.
- Mac schedule. Experiments are sequenced, not overlapped, under memguard. A full-recipe real-row
  run costs on the order of 500-600 GPU-hours at 2B, so the P2 schedule is owed before P1 ends.
- A cloud cost estimate under rule 4 before any cloud seed.
- Second-teacher (Haiku) re-query cost for the agreement filter.
- R2 caps for every new family (only the four `career.jd_*` have stated zeros).
- Gemma 4's prohibited-use policy is unread, and Base repo gating is unconfirmed.
- Retraining cadence for drifting postings. Teacher labels end once d1-ultra and Haiku are
  retired, so later refreshes rely on owner labels and synthetic rows.
- `caller-contract.md` §4/§6 and `schema-api.md:66-67` ("score slots planned for v6") are
  updated when the first-scope families land, not before.
