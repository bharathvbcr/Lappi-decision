# Fable's v6 data-design ruling (2026-10-08)

**What this is.** Rulings R1-R9 from Fable, asked by the v6prep lane on 2026-10-08 after the human
asked to "prepare all the data and training scripts needed for next model train". Fable worked
read-only; the lead records the rulings here as Fable returned them, lightly condensed. File:line
citations are Fable's, read on 2026-10-08. [I] means inferred and [U] means unmeasured. "rg" means
grep or git-log evidence, not graph-confirmed.

**The human's rulings the same day** (the lead's session, AskUserQuestion):
- Order: P1, then v6.
- Licences allowed: `synthetic-by-rule`, `oanc` and `odc-by-1.0`.
- The promotion re-spec is approved (`paired_margin_population`, `shuffled_label_seeds = 3`).
- Arm B mining runs on the Mac under the lock.
- The caller families are not deferred.

## R1. Allocation seam: one `qd-prep allocate` writing one applied pool

- **The subcommand.** A new Rust subcommand:
  `qd-prep allocate --config data/decisions/v6-allocation.json --pool NAME=DIR ... --out DIR`
  - It reads N pools: the v5 pool plus every v6 candidate pool.
  - It applies one cap table.
  - It writes **one** `qd-decisions/v1` build pool with `allocation.state = "applied"` (`crates/qd-prep/src/pool.rs:253-271`).
  - The loader, the pipeline and the trainer stay single-pool: `python/qd_data/decisions.py:143-152`, `tools/real_tokenizer_pipeline.py:3418-3426` and `tools/real_ft_run.py:4522-4526`.
- **Why not multi-pool loading, or a raised bound.**
  - The sizes fit: the v5 pool's 214,760 rows, plus about 69,900 train and at most 10,000 val from v6, come to about 293,000. That is under `MAX_POOL_ROWS` (400,000).
  - Raising a read-whole bound, or teaching three readers to merge pools, is more code and more places for a family to enter uncapped.
- **Owner: a new `crates/qd-prep/src/allocate.rs`.**
  - It is registered beside the other subcommands in `main.rs`.
  - It reuses `decisions::select` (`decisions.rs:1527-1640`) for the cap logic and `pool::examples` (`pool.rs:274`) for the bytes.
  - It does **not** move `decisions::run`'s scan block onto `pool.rs`, the gap's option 1. v6 does not need that refactor, and it touches v5's byte-identical path.
- **Config shape.**
  - `"pools": {"pool-v5-decisions-v4": "as_built", ...}` passes v5's already-selected rows through untouched, with the same split and the same bytes.
  - Every v6 pool gets `train_caps` keyed by family. Within a family's cap, allocate draws per stratum by largest remainder, seeded.
  - Val follows the v5 rule: `val_cap_per_family` 1,000 and a floor of 100.
  - `select`'s refusals carry over. A family present in a pool but absent from the table is refused, and so is a non-zero cap that names no candidates.
- **Dedupe.** Cross-pool exact dedupe on `identity_key` happens inside allocate, keeping the val copy. Near-duplicate dedupe stays the pipeline's job, under R2's `--v6-data-rules`.
- **The defect-share bound (≥ 55.8% of tokens) is enforced in the pipeline, not in allocate.**
  - The bound is from `AUDIT/finalize-2026-10-03/fable-weighting-ruling.md:34-36`.
  - `code.defect_class` comes through `--defect-class`, not through the pool, so only the pipeline sees every family's tokens.
  - The shard ledger row gets `train_tokens_by_family` and `defect_token_share`, with state `ran` and `passed` iff the share is ≥ 0.558. A failed check exits non-zero and writes no shard set.
  - Allocate's `text_bytes_by_family` is a report-only pre-estimate.
- **Tests:**
  - allocate refuses a config that misses one of a pool's families;
  - two pools sharing one identity key yield one row, the val one;
  - the output loads through `load_decision_pool`.

## R2. The v6 cap table

Val is capped at 1,000 per family with a floor of 100. `val_cap_total` rises from 6,000 to 10,000 because v6 adds 10 families. The held-out sets are untouched.

| Family | Train cap | Why |
|---|---|---|
| the v5 pool (15 families) | as built | unchanged bytes |
| `knowledge.multiple_choice` | **0** | see below |
| `mnli.nli` | 5,000 | the NLI redundancy cap `openjev.nli` and `vitaminc.nli` already use |
| `scirepeval.search_rel` | 10,000 | a new skill (pointwise rerank, the ScholarLM caller); short rows |
| `csn.func_match` | 15,000; **cc-by-sa-4.0 rows refused** (16 of 145,153) | jevgrep-shaped relevance; a share-alike obligation is not worth 16 rows |
| `when2call.tool_select` | 19,914 (all) | a caller use case |
| `toolace.tool_select` | 4,740 (all) | the same |
| `apps.tool_select` | 3,288 (all) | the apps' own catalogs |
| `email.category` | 5,343 (all, injection stratum included) | |
| `jarvis.target` / `field_fill` / `step_risk` | 2,400 / 3,200 / 1,000 (all) | |

The new train rows total about 69,900, roughly 40M tokens [I]. The defect share stays at about 60-64% [I], and R1's check measures it.

- **MMLU at zero.**
  - Today `cais/mmlu` pins dev and test to train (`python/qd_data/sources.py:72`; `general.py:197-205`).
  - A single pipeline and trainer flag, `--v6-data-rules`, applies both `DataConfig.with_v6_benchmark_targets()` and `with_v6_dedupe_rules()` (`python/qd_data/config.py:228-231, 259-264`). It is wired at `real_tokenizer_pipeline.py:3286` and `real_ft_run.py:4526`.
  - The pipeline asserts that the built train count for `knowledge.multiple_choice` is 0, or it refuses.
  - The 1,200 `knowledge.multiple_choice` contrast rows move to `commonsense.multiple_choice`, which holds 2,000 in total.
  - Both rules are in the fingerprint. That is correct for v6 and leaves v5's pins alone.
  - GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND is closed for v6 by this wiring; v5 stays lexical.
- **convert-injections is not in v6.**
  - The email pool already carries label-flip injections from its own templates (`injection_rate 0.1`).
  - The external corpus is not a decisions pool.
  - Nothing has measured its effect on the 13-label task.
  - It is parked, with `swe-rebench-filter`.

## R3. Needle long-input data

- **The diagnosis changes the lever.**
  - All 39 v5 needle misses are Swift or TypeScript, and none is an abstention.
  - Every wrong pick is 1-2 hunks before the needle (`HANDOFF/needle-span-mac-2026-10-06.md:40-80`).
  - Length and depth do not separate hits from misses.
  - So depth weighting in `compose.rs` is **not** built.
- **The training corpus explains it [I].**
  - Composed-v2's needles by language: python 17,237, go 1,608, typescript 1,081, rust 972, swift 0.
  - corpus-v3 has 0 Swift files.
  - The needle suite covers five languages, Swift among them (`python/qd_train/needle.py:323`).
  - Swift is neither in the OOD unseen list nor in G6, so Swift defect rows are admissible.
- **Ruling: two compose runs.**
  - Both use v2's `Options` except `train_rows` and `seed`: min 850, max 9,100 and hard limit 9,400 tokens, 3-48 files (`compose.rs:1022-1048`).
  - Both reuse v2's split map (sha `f5621909…`; `compose.rs:216-230`).
  1. `commitpackft-composed-ts-v1`: corpus-v3 restricted to TypeScript mutants (2,161 examples); 4,000 rows; seed 20261008.
  2. `own-swift-composed-v1`: 4,000 rows from Swift mutants of the admitted own train repos.
     - Those repos hold DevType 566, MarkDev 234, ChronosFlow 205, M5Blade 158, Curio 149 and Meridian 115 `.swift` files (rg).
     - It needs a small Rust emitter in `crates/qd-prep`. The emitter writes own-repo files in the pool record shape that `qd-mutate generate` reads. `qd-mutate generate --language swift --seed 20261008` then mutates them.
     - GitPulse is held out, and the strike list applies.
- **First, the free check.** Render one Swift and one TS miss through `needle.py`'s builder, and compare the hunk-boundary tokens with a Rust case. If a tokenisation defect appears, it becomes a gap and is fixed in the renderer first. The compose runs still happen either way.
- **It rides on Arm B's seeds,** as one arm. It adds about 8,000 long rows, roughly +10% step time [I].

## R4. Arm B miner

- **Scorer: v5 seed 1's checkpoint.**
  - It is the only seed checkpoint on the Mac (sha256 `84edcc91…`, `campaign/span-refit-mac-preregistered.json`).
  - Not the average: `average_may_promote` is false, and the average abstains least, so it would mine the fewest rows.
- **Candidates: fresh rows, labelled by construction, never duplicates.**
  - R3's 8,000 composed rows.
  - Plus 12,000 single-file mutants from `qd-mutate generate --seed 20261008` over the existing corpus pool.
  - Exact `identity_key` dedupe against v5's train, val and held-out sets.
  - The scan is capped at 20,000.
  - Duplicating v5 train rows is refused.
- **Throughput [U].**
  - The first step is a 200-row probe under `tools/mac_heavy.sh`.
  - The rows scanned are `min(20,000, floor(rows_per_s × 36 h))`, a pre-registered formula, run in lock chunks of at most 6 h.
- **Selection: exactly `choice_rule_abstentions`** (`tools/real_ft_run.py:5731-5775`), at fp32. A candidate is kept iff it abstains and its constructed gold is not `noul`.
- **Label:** the row's constructed class, untouched.
- **Cap and floor:**
  - At most 5,000 mined rows go into train.
  - Fewer than 300 reads `no_data`: v6 runs without them, and Arm B's reading is `not_run`.
- **Leak checks:**
  - the containment scan against val, held-out and targets;
  - the v2 split map;
  - the pipeline's v6 split-priority dedupe.
- **Language: Python, extending the existing scoring path.**
  - `_decode`, `prepare_second_pass` and `--verdicts-out`, run through `--score-checkpoint` over a candidate shard set.
  - qd-metal is refused: its batched decode is KNOWN RED on the GPU digest pin (9c563ed) and serves bf16.
- **Pre-registered before the first mined row** (`campaign/v6-arm-b-mining-preregistered.json`):
  - the scorer's sha256;
  - the candidate corpora's sha256 and seeds;
  - the rule's citation;
  - the throughput formula and the wall cap;
  - the cap and the floor;
  - the exclusion rule;
  - the `no_data` branch;
  - R8's reading rule.

## R5. Caller families

- **`gitpulse.commit_type` can be built without a human decision.**
  - "Synthetic only" governs caller records. Own-repo commit history is the human's own data, admitted on 2026-09-28.
  - Measured today over the 23 admitted train repos (rg): 4,837 commits, 2,667 with a conventional prefix, and 1,715 of those (64%) carrying a `Co-Authored-By` trailer.
  - GitPulse (805 commits, 579 prefixed) is held out and becomes the held-out eval.
  - **Builder:** a Rust qd-prep subcommand reading through the git CLI (`gitcli.rs`).
    - The context is the commit's unified diff, as bytes (contract §4). Rows over the 131,072-byte context bound are counted and dropped.
    - The options are exactly `KNOWN_TYPES`, in order (`GitPulse/src-tauri/src/ai/commit_brief.rs:40-49`).
    - Each repo is capped at 400, because DevCouncil and scholarlm are 67% of the supply.
    - Val is 5%, within the train repos.
  - Amendment 2 is satisfied by reporting R1/R2 split by agent-authored and human-authored commits.
  - The ambiguous-subset replay is deferred past v6.
  - About 1,500 train rows are expected [I].
- **`devtype.route`: option (a).**
  - It is built as a `synth_tools` variant with family id `devtype.route`, the exact question and options of contract §4. `none` is the generator's unable and direct classes.
  - The phrasings use dates in `PaletteToolRouter.swift:122-158`'s accepted forms, with synthetic macro and snippet names.
  - 1,500 train and 150 val rows, with at least 4 templates per class and 1 held out (the synth.rs rules).
  - Option (b) has no DevType shape. Option (c) would make real queries into training data, which is forbidden.
- **Deadline rule.** Either family enters v6 iff its applied rows exist before the v6 allocation is pinned; the allocation is not reopened for it. Every other draft family is out of v6.

## R6. When2Call gold and CSN licences

- **When2Call: the rule-derived gold is accepted** (`convert_tools.rs:108-132, 461-469`).
  - `label_basis` stays on every row.
  - `DIRECT` stays in the option list, and the generator's direct share rises to at least 10% of its rows.
  - The family's val accuracy is reported by `label_basis`.
  - The linear control runs on it.
- **CSN: GitHub's current licence detection is accepted as the stand-in.**
  - The caveat is carried in the pool manifest and the release notes.
  - The 16 cc-by-sa-4.0 rows are refused.
  - Residual: a repo relicensed from permissive to restrictive since 2019 is misclassified.
  - **This is Fable's ruling, with a veto point for the human.**

## R7. Span head for v6 seeds: one shared initialisation, then averaged

- **One file.** Every v6 seed starts from the **same** span head file, generated once by `tools/qd_train_oracle_span_head_init.py --seed 0`.
- **The refusal is relaxed only for a shared head.** The seed-specific refusal at `tools/real_ft_run.py:9786-9792` gives way only when:
  - the manifest carries `construction.shared_across_seeds: true`, written by the generator under a new `--shared` flag and absent otherwise, so existing fixtures stay byte-identical; and
  - the recipe records `span_head_init.shared = true`.
- **Test:** a 3-seed run with the shared manifest proceeds; the pre-fix code refuses it.
- **Two rows, both fixed before scoring.**
  - Primary: the averaged tower plus the averaged head.
  - Fallback: the averaged tower plus seed 0's head.

## R8. v6 recipe, seeds and controls

- **The recipe is v5's exactly:**
  - `--optimizer master --lr 1e-5 --epoch --batch-tokens 35403 --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers 6 --min-lr 0 --batch-order seed`;
  - noul unweighted;
  - plus the shared `--span-head-init` (R7) and `--v6-data-rules` (R2).
  - `master` whatever P1 reads; P1 changes nothing for 2B.
- **Seeds 0, 1 and 2**, with:
  - J5′ ×3;
  - the needle control at `1024,2048,4096`;
  - one averaged row (R7), a release candidate only.
- **Reading rule:**
  - **Arm B:** `ood_abstain.in_distribution` over `code.defect_class`, Wilson upper ≤ 5% on at least 2 of 3 seeds, with the OOD suite half ≥ v5's same seed on every seed.
  - **Needle:** worst depth bucket ≥ 0.95 on every seed, with Swift and TS reported per language.
  - **Guards:** val choice top-1 ≥ 0.8278 and span ≥ 0.9066 on every seed.
  - The new families are report-only.
- **Box.**
  - On one GH200: 3 × ~7.7 h + 3 J5′ × ~6 h ≈ 41 h ≈ **$105** [I].
  - On 2×H100: about $175-235.
  - It is a cost question for the human (rule 4).

## R9. Cuts and order

- **Cut from v6:**
  - Arm A;
  - every other caller family;
  - convert-injections and swe-rebench-filter;
  - the `decisions::run` → `pool.rs` refactor;
  - kahan on 2B;
  - the ambiguous-subset replay;
  - the commit-history teacher families.
- **Order, Mac first, no box:**
  1. Commit the uncommitted qd-prep work, and the S2 fix.
  2. R1's allocate, R2's wiring and the defect-share check.
  3. The Swift/TS tokeniser check, then R3's compose runs.
  4. R4's pre-registration, the probe and the scan.
  5. Allocate, the containment scan, the one build, A7.
  6. R7's code.
  7. `campaign/v6-preregistered.json`, then the box.
- **P1** can run in parallel; nothing in v6 waits on it.
