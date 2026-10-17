# Gap triage 2026-10-14: the ojas/tessl/cuda remainder (read-only lane)

Scope: the 43 ids the previous lane did not reach (`HANDOFF/wave3-l4-gaps-remainder-2026-10-13.md`, "Not done"). Current
state of an id is its LAST line in `gaps.jsonl`. `GAP-WORKSPACE-IS-NOT-RUSTFMT-CLEAN-2026-10-06` skipped (closed today).
Nothing was appended, edited, built or run: every row below is a read (Read/rg/jq/`git show`/`git merge-base` on Lappi only).
Labels: [V] verified by reading the cited line or command output, [I] inferred, [U] unverified.

Heads read from ref files (no git against them): ojas `bc498c4375c2e4753d784e6e9fc267a8d18b038a` (`.git/refs/heads/main`);
tessl `29720274752f91815334a6151e723407c82ca280` (`.git/packed-refs:28`, `refs/heads/main`). Working-tree dirtiness of
ojas/tessl is not known (no git status), so every ojas/tessl citation is "tree as read", not "at HEAD".

## Mapping

My family regex resolved to **48 open ids** (the handoff said 43; the extra 5 are ids an earlier pass touched or the regex
widened: `GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01` (reopened by the last lane),
`GAP-TESSL-K3-IDENTITY-NOT-SPECIFIED`, `GAP-TESSL-GDN-FIXTURE-STORES-UNCLAMPED-ALPHA` (both re-triaged 2026-09-19),
`GAP-SCORE-CHECKPOINT-SINGLE-SEED-ROW-DOES-NOT-INHERIT-QUICK-2026-10-01` (the handoff's "SCORE-CHECKPOINT-QUICK"),
and `GAP-FABLE-CUDA-CUBLAS-GEMMEX-TABLE-UNFETCHED-2026-10-01`). **All 43 named families mapped; I resolved 48 ids.** The 11 ids
of those families already in a closed status (e.g. `GAP-OJAS-RUNGD-SPAN-HEAD-INIT-UNOWNED-2026-10-01`,
`GAP-QDM-GPU-GATES-NOT-RUN`) were not re-examined; the closed-status set is `python/qd_train/gaps.py:62`.

## Tool notes

- `devmap_status` on Lappi: generation 3889, `source_freshness: true` but `is_fresh: false`, `analyzer_freshness: false`
  ("stored extraction payload is obsolete; rebuild with the current analyzer"). This disagrees with the coordinator's
  "is_fresh=true"; I did not rely on the graph for any verdict.
- `gitpulse_insights`: facets `agents`, `changes`, `codeintel`, `collisions`, `ledger`, `worktrees` all `ok: true`,
  `truncated: false`. Two live Claude sessions in this worktree (pid 12401 busy, 47940 idle); shared dirty files are
  `BRANCH_REPORT.md` and `BUILD_REPORT.md`, none of which I touched.
- Everything else was rg/Read/jq (see the fallback list in the final message).

## Table

| id | prior status | verdict | evidence | next action |
|---|---|---|---|---|
| GAP-CUDAEMU-1002-DEVMAP-GITPULSE-UNAVAILABLE | open | CLOSEABLE | [V] `git show --stat 83a39da`: 7 files all under `tools/cuda_emu/`, +5098/-0, so no existing symbol touched; the "tools missing" was one cloud session's | append proposed record |
| GAP-CUDAEMU-1002-LIBM-IS-NOT-LIBDEVICE | open | STILL-OPEN | [V] limit still stated and enforced: `tools/cuda_emu/tests/cuda_emu/mod.rs:35,443-470,1173-1176`; selftests `bitwise_comparison_is_refused_*` at `cuda_emu_selftest.rs:1189,1199`. A documented limit, not a defect | keep open until a device run compares libdevice vs libm |
| GAP-CUDAEMU-1002-MAC-TOOLCHAIN-PATHS-UNVERIFIED | open | NEEDS-RUN | [V] only ledger row for the emulator is `ledger/cloud-cuda-emu-2026-10-02.jsonl` (1 line, Linux 81a27f66); no Mac row | `cargo test --test cuda_emu_selftest` in `tools/cuda_emu/` on the Mac (clang++/TSan build; I did not run it: heavy lock) |
| GAP-CUDAEMU-1002-MEMORY-ORDERING-NOT-MODELED | open | STILL-OPEN | [V] `mod.rs:34` (order invariance only), selftests `cuda_emu_selftest.rs:364,371` exist; no change | none; limit |
| GAP-CUDAEMU-1002-NVRTC-FRONT-END-NOT-MODELED | open | STILL-OPEN | [V] `cuda_emu.h:1274` (`__CUDA_ARCH__` from `CUDA_EMU_ARCH`), `mod.rs:403` default 900, `mod.rs:1592-1609` refuses inline asm; subset unchanged | none; limit |
| GAP-CUDAEMU-1002-OJAS-KERNELS-NOT-EXECUTED | open | STILL-OPEN | [V] `rg -uu 'cuda_emu\|cuda-emu'` over ojas (excluding target, .git) returns nothing: the emulator was never copied in. [V] the crate the record names, `ojas-qwen35-cuda`, no longer exists; it is `ojas/ojas-cuda` (dir listing) so the record's paths are stale. Real-device coverage since: ojas-cuda README.md:40-43 claims GH200 ran every device-test binary 2026-10-05 [U: logs not in tree] | owner decides: retire the emulator plan in favour of the device run, or port `tests/cuda_emu` into `ojas-cuda/tests` |
| GAP-CUDAEMU-1002-PTX-SEMANTICS-NOT-AUDITED | open | NEEDS-RUN | [V] `shfl_source_lane` still checked only against a second formula from the same memory (`cuda_emu_selftest.rs:585`); PTX ISA not consulted | one GH200 shuffle-table print vs the emulator (single GPU, under $20: no human yes needed per rule 4) |
| GAP-CUDAEMU-1002-WARP-DIVERGENCE-NOT-MODELED | open | STILL-OPEN | [V] `rg '__activemask\|__match_(any\|all)_sync' cuda_emu.h` empty; lanes are OS threads | none; limit |
| GAP-FABLE-CUDA-CUBLAS-GEMMEX-TABLE-UNFETCHED-2026-10-01 | open | SUPERSEDED | by GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01 (same question). The "moot while no cuBLAS" premise is false now: [V] ojas-cuda `README.md:53,89-91` and `src/gemm.rs:15` use cublasGemmEx bf16->f32 | append proposed record |
| GAP-K7-GOLDEN-PINS-NANOLAB-SIGMOID-NOT-QWEN35-GATE | open | STILL-OPEN | [V] tessl `tests/gdn_gates.rs:1-40` still pins the nanolab sigmoid form and `alpha_mamba2_refuted`; no softplus golden there. Narrowed: the Qwen form is golden-tested in ojas, `ojas-cuda/src/gates_published.rs:12-20,64-67` (`g=-exp(A_log)*softplus(a+dt_bias)`, no clamp) and `tests/fixtures/gen_goldens.py:176-199` | owner: retire as covered-in-ojas, or repin tessl's golden (the tessl K7 Metal kernel does not exist, so nothing is judged by it today [I]) |
| GAP-KAHAN-RECIPE-HAS-NO-RUST-COUNTERPART-2026-10-06 | open | STILL-OPEN | [V] premise partly stale: tessl now has `UpdateRule::{F32Master,Bf16Kahan,Bf16Stochastic}` and Block8 moments (`tessl/docs/qwen35.md:1162-1200`; symbols in `src/qwen35_adamw.rs`, tests `tests/qwen35_adamw.rs`). [V] but `rg 'Bf16Kahan\|UpdateRule'` over `crates/` and `ojas/ojas-qwen35` finds none: qd-train and ojas-qwen35 do not select it | update the record to "wired in tessl, not in qd-train/ojas-qwen35"; parity vs the Python oracle not seen [U] |
| GAP-KAHAN-RECIPE-NOT-MEASURED-ON-A-GPU-2026-10-06 | open | NEEDS-RUN | [V] `rg -il kahan ledger` finds no ledger row; only HANDOFF text | P1 in `HANDOFF/next-training-plan-2026-10-06.md` (master vs kahan, 600 steps, one seed, then `gh200_footprint.py --optimizer master/kahan`) |
| GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01 | open | NEEDS-OWNER | ojas README.md:40-42 says GH200 `rung0` passed 174/174 on 2026-10-05 and :208 says rung 0 includes the GemmEx bf16->f32 probe with `cublas_status` recorded; [U] the report is not in this tree ("merged crate has not yet run on a device") | fetch the box rung0 JSON, cite its `cublas_status`/check row, then close |
| GAP-L-CUDA-ORACLE-GOLDEN-TRANSCRIPTIONS-2026-10-01 | open | STILL-OPEN | [V] gate golden still a transcription with no module cross-check: `ojas-cuda/tests/fixtures/gen_goldens.py:193` (`g = -xal.exp() * F.softplus(...)`); recurrence and gated RMSNorm do cross-check (`:130,:148,:251-256`). Path moved from `ojas-qwen35-cuda/` to `ojas-cuda/` | none new; cross-check the gate against transformers if wanted |
| GAP-L-CUDA-SMALL-NO-LOCAL-CUDA-COMPILER-2026-10-01 | open | NEEDS-OWNER | [I] limit is a property of the Mac; README.md:206 says rung 0 compiles all modules through NVRTC and :40-42 says it passed 2026-10-05 [U: log absent]. Merged crate not re-run | pull the box log, cite the NVRTC-compile check, then close |
| GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED-2026-10-12 | open | STILL-OPEN | [V] `rg -uu -i lora` over ojas finds only task files and `ojas-qwen35/tests/gpu_memory_rows.rs`: no adapter-checkpoint writer exists. Reader contract at `tools/ckpt_average.py:126,146-156` | wait for the ojas LoRA writer |
| GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT | open | NEEDS-RUN | [V] `python/qd_train/memory.py:1255-1268` `SAFETY_FRACTION = 0.35` unchanged (:100); `tools/gh200_rows.py` exists; contract is already "upper bound" per the record | sweep with `tools/gh200_rows.py` on a device before any long run |
| GAP-OJAS-ADVICE-INVENTORY-FROM-READS-2026-10-01 | open | NEEDS-OWNER | [I] provenance caveat on an audit table (graph gens 2972/2106, now stale: Lappi is 3889 and `is_fresh:false`); nothing re-derivable | owner: accept as a recorded limitation and close, or re-derive the table from the graph |
| GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01 | open | NEEDS-RUN | [V] tessl `docs/qwen35.md:869-876` gives 30.7 GB peak for a recompute step but the table is not an 8K step with AdamW moments; `bench/results/qwen35_train_step_bf16_m5pro.txt` is T=2048. [I] bf16-storage training now exists (`qwen35.md:1162-1200`), so the estimate's basis has moved | `bench_qwen35_train --bf16 --step=8192` under `/usr/bin/time -l` (Mac heavy lock; I did not run it) |
| GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01 | open | NEEDS-RUN | [V] tessl now documents it leaves `rope_parameters.mrope_*` unread (`src/qwen35_model.rs:174`); ojas refuses non-collapsing sections (`ojas-qwen35/src/config.rs:22-30`). Falsifier still ignored: `ojas-qwen35/tests/gpu_parity.rs:736` `#[ignore] gpu_real_2b_single_step_matches_the_transformers_reference` | `cargo test -p ojas-qwen35 --test gpu_parity -- --ignored gpu_real_2b_single_step_matches_the_transformers_reference` (about 40 GB unified, Mac heavy lock) |
| GAP-OJAS-RUNGD-8K-POWERMODE-UNRECORDED-2026-10-01 | open | NEEDS-OWNER | [I] a retrospective `powermode` cannot be recovered from `pmset -g log`; `rg -il powermode` over ojas bench results, HANDOFF, ledger finds none that recorded it for that window (the tessl 09-30 bench file records `powermode 1` for its own run only) | owner: close as unrecoverable (state the consequence) or re-run the 8K bench with powermode recorded |
| GAP-OJAS-RUNGD-FP32-ARM-MEMORY-UNMEASURED-2026-10-01 | open | NEEDS-OWNER | [V] `HANDOFF/lead-pipeline-2026-10-03.md:408-409`: T-fp32 exit 0, ft row 1170ff52, 2026-10-05 13:13Z, so the arm completed on the box [I: it fit]; [V] `rg -l 1170ff52 ledger` finds nothing: the row and its peak memory are not in this tree | pull the box ledger row, record the peak, then close |
| GAP-OJAS-RUNGD-TRAINER-RATE-INFERRED-2026-10-01 | open | NEEDS-RUN | [U] no measured s/step for the tessl trainer at that shape located; `bench/results/2026-10-08-gpu-step-ab` has no README | measure with tessl `bench_qwen35_train` (Mac heavy lock) |
| GAP-OJAS-RUNGD-WORKING-SET-UNREAD-2026-10-01 | open | CLOSEABLE | [V] `AUDIT/ojas-training-2026-10-01/ojas-qwen35-gpu-diag-none.log:21`: recommended working set 51539607552 B, wired budget 46385646796 B (0.9x). Note 46.39 GB is just under the "47 GB" criterion at `HANDOFF/ojas-training-2026-10-01.md:200` | append proposed record; owner reads the 0.6 GB shortfall |
| GAP-QD-METAL-PINS-MOVED-WITH-TESSL-0B1124F-2026-10-12 | open | NEEDS-OWNER | [V] `crates/qd-metal/tests/gpu.rs:355` still `PIN_TESSL_HEAD = cf65d9d5...` and `PINNED_V1` (:374-) old digests; the record says the human owns the re-pin plus a `qd-metal-parity` run | human re-pin + parity run |
| GAP-QDM-8K-PREFILL-3797-VS-REPORTED-6213-TOKS-2026-10-03 | open | NEEDS-RUN | [V] `ojas/bench/results/2026-10-03-lappi-inference-rerun/research-notes.md:70,127`: hypothesis that 6,213 came from `attn_prefill` and `--attn-rows` gives 4,189 [I], the discriminating bench not run; prefill since reverted to `flash_attn_rows` (the pins record) | `bench_qwen35_layers 8192` with and without `--attn-rows`, one window (Mac heavy lock) |
| GAP-QDM-BACKEND-CANNOT-CARRY-AB-FLAGS-2026-10-02 | open | CLOSEABLE | [V] `rg -uu 'EmbedPath\|DigestPath' crates`: only the frozen context fixture text; `decision.rs:85-91`; commits ae77718, 9299dea on HEAD | append proposed record |
| GAP-QDM-BATCHED-DECODE-TOUCHES-ANSWERING-PROCEDURE-PIN-2026-10-02 | open | CLOSEABLE | [V] `crates/qd-runtime/src/backend.rs:238-254`, `tests/answering_procedure.rs:565-600`, `docs/schema-api.md:464-476`, 9c563ed on HEAD. [U] test not run by me | append proposed record |
| GAP-QDM-BENCH-PREFILL-MODE-NO-SNAPSHOT-FLAG-2026-10-03 | open | STILL-OPEN | [V] `crates/qd-metal/src/bin/bench.rs:62` `resolve_snapshot(None)` in prefill mode, vs `:129` `args.snapshot` in `--decision`; prefill mode also reads `include_str!("../model.rs")` at :64 | add `--snapshot` to prefill mode; qd-metal owner |
| GAP-QDM-DECISION-BENCH-PROMPT-FROM-OWN-SOURCE-2026-10-03 | open | CLOSEABLE | [V] `decision.rs:81` `include_str!("../fixtures/decision-context-v1.txt")`, pinned by `decision.rs:1429`; 9fd6891 on HEAD | append proposed record |
| GAP-QDM-DIGEST-SHA2-SOFT-BACKEND-NO-ASM-FEATURE-2026-10-03 | open | CLOSEABLE | [V] `Cargo.toml:36` sha2 0.11, `Cargo.lock` 0.11.0, sha2-0.11.0 `sha256.rs:33-72` aarch64_sha2; bb4df87; ledger row 94736e9b t8192 digest median 56.31 ms (quick=true) | append proposed record |
| GAP-QDM-NO-TRAINED-RELEASE-ON-MAC-2026-10-02 | open | CLOSEABLE | [V] 4 `release_manifest.json` on this Mac incl. `/Users/bharath/qd-campaign/lappi-bench-2026-10-03/p4-v4-avg-masters-v1/`; rows 94736e9b etc. weight hash a68f19bc vs base 92f6bd1c. Residual: serve-on-release not verified [U] | append proposed record |
| GAP-QDM-ONE-SLOT-PER-DECODE-CALL | open | STILL-OPEN | [V] narrowed: `decode_slots` exists, but the Metal backend batches only two read-only letter queries of equal row count (`crates/qd-metal/src/backend.rs:344-370`, `batchable`); N slots / varlen not supported | none until varlen GDN/conv/attention exists in tessl |
| GAP-QDM-PARITY-COVERS-T-LE-1397-2026-10-02 | open | STILL-OPEN | [V] only parity ledger rows are the two of 2026-10-02 (`ledger/mac-qd-metal-2026-10-02.jsonl`, 0112dade, c9452efc); no T>=4K fixture found in `crates/qd-metal/tests/fixtures` (fixtures are passed by `--fixtures <dir>`, `bin/parity.rs:5`) | generate a >=4K fixture, rerun parity (GPU) |
| GAP-QDM-PARITY-LEDGER-ROW-NOT-SCHEMA-2026-10-02 | open | STILL-OPEN | [V] `crates/qd-metal/src/bin/parity.rs:428` still writes `"kind": "qd-metal-parity"` via `append_line` (:446), not `ledger::write_row` (`ledger.rs:64`) | change parity.rs as the record says |
| GAP-QDM-PARITY-ROW-MARKED-QUICK-2026-10-02 | open | NEEDS-OWNER | [V] `crates/qd-metal/src/parity_row.rs:340` still `quick` with the same reason | owner decides quick vs not (rule 8 reading) |
| GAP-QDM-POOLED-FEATURES-UNDEFINED | open | NEEDS-OWNER | [V] `backend.rs` `pooled_features` still returns `BackendError::Unavailable` | human decision on the pooling |
| GAP-QDM-REFIT-HEAD-HAS-NO-EXPORT-MODE-2026-10-12 | open | STILL-OPEN | [V] `crates/qd-export/src/export.rs:215-218,333` accepts only `seed`/`averaged`; `campaign/span-refit-mac-preregistered.json` exists | wait for the pre-registration's reading rule |
| GAP-QDM-RUNTIME-OPENED-WITH-TIMESTAMP-HEAP-2026-10-02 | open | CLOSEABLE | [V] rows 94736e9b/91832401 (`new`) vs a2ed6bc8/552e6073 (`new_inference`), same commit 75c683b: t8192 medians 2367.4/2354.0 vs 2399.8/2347.7 ms; within noise. quick=true | append proposed record |
| GAP-QDM-SERVE-COLD-START-UNDER-LIFECYCLE-LOCK-2026-10-02 | open | CLOSEABLE | [V] `crates/qd-runtime/src/service.rs:276-301`, test `tests/lifecycle.rs:305`, 1056fc4 on HEAD. [U] test not run | append proposed record |
| GAP-QDM-STATE-DIGEST-HOST-SHA256-PER-DECODE-2026-10-02 | open | CLOSEABLE | [V] row 94736e9b digest_ms 17.56 ms (T=2048) / 56.31 ms (T=8192) vs 478.6 / 1513.5 ms at ebb17358; sha2 0.11 backend. Residual: device digest design not built, not justified at 2.4% | append proposed record |
| GAP-QDM-TESSL-PREDATES-EXE-COUNTS-NON-BUILD-FILES-2026-10-02 | open | STILL-OPEN | [V] `crates/qd-metal/src/ledger.rs:218-238,358` `latest_change_s` still takes every dirty file's mtime except build output | filter to library inputs as the record says |
| GAP-QDM-TOKENIZER-UNTRUSTED-ENCODE-ONE-SIDED-2026-10-02 | open | STILL-OPEN | [V] `tokenizer.rs:79` `encode_untrusted` is used by `tests/gpu_d17.rs:39,147`, `gpu_span.rs:88`, `backend.rs:617`, with a unit test at `tokenizer.rs:226`; the training-side tokenization still uses plain encode semantics [I: no `tool_call` handling in `python/`, `rg` only matches tests]; no `--tokenizer-only` adversarial prompt found [U] | renderer escape on both sides + adversarial parity prompt |
| GAP-SCORE-CHECKPOINT-SINGLE-SEED-ROW-DOES-NOT-INHERIT-QUICK-2026-10-01 | open | STILL-OPEN | [V] `python/qd_train/ledger.py:1287-1344` `promotion_verdict` judges candidate rows' own `quick` and needs 3 seeds; no join to the source ft row; `promotion_verdict_avg` (:1346) does check ft inputs non-quick (:1564-1572) | add a regression test that a quick ft row re-scored does not promote |
| GAP-SIDECAR-BIG-ENDIAN-AND-CUDA-ARE-NOT-RUN | open | NEEDS-RUN | [V] `python/qd_train/run_control.py:1326,1667,1754` still refuse non-little-endian; no big-endian host exists here. [U] whether a box-written sidecar has been read on the Mac with `Checkpoint.read` (`real_ft_run.py:12043`) | copy one box checkpoint pair here and run `Checkpoint.read` |
| GAP-TESSL-GDN-FIXTURE-STORES-UNCLAMPED-ALPHA | open | STILL-OPEN | [V] fixture `tests/fixtures/gdn/gdn_published_L65_tinyalpha_alpha.npy` mtime Sep 19 22:59, 1688 bytes; `tests/gdn_fixtures.rs:383,434` guard it. [U] I could not re-load the .npy (no numpy; harness blocks reading outside the repo), so the 1e-8 values are not re-measured today | rule 2: regenerating a published fixture is human-only |
| GAP-TESSL-INTEGRATION-AUDIT-LINE-CITES-DRIFTED-2026-10-02 | open | STILL-OPEN | [V] `AUDIT/tessl-integration.md:1-3` header still names no tessl revision; last touched by 337876f, earlier than the record | name the audited revision when next touched |
| GAP-TESSL-K3-IDENTITY-NOT-SPECIFIED | open | STILL-OPEN | [V] `tessl/docs/verification.md:238,325-345` still says open. Naming conflict found: tessl calls K3 the state update (`gdn_state.rs`, :238) while ojas-cuda K3 is the gates (`ojas-cuda/src/gates_published.rs:1-2`); `rg '\bK3\b'` finds no K3 in tessl `src/` or `kernels/` | owner reconciles the K3 label across tessl and ojas |

## Counts

Total resolved: **48** (all 43 named-family ids mapped, plus 5 widened/reopened).

| verdict | count |
|---|---|
| STILL-OPEN | 20 |
| CLOSEABLE | 10 |
| SUPERSEDED | 1 |
| NEEDS-RUN | 9 |
| NEEDS-OWNER | 8 |
| **sum** | **48** |

Proposed records for the 11 CLOSEABLE/SUPERSEDED ids: `AUDIT/gaps-triage-2026-10-14/proposed-records.jsonl`.

## Caveats

- Several verdicts rest on code reads plus ancestry checks (`git merge-base --is-ancestor <c> HEAD` on Lappi); I ran no
  Lappi tests, so "test exists" is [V] and "test passes" is [U] wherever stated.
- The GH200 logs behind the 2026-10-05 device run are not in this tree; no CUDA gap is closed on the README's word alone.
- Four ledger rows used for the digest/runtime closures (94736e9b, a2ed6bc8, 552e6073, 91832401) are `quick=true`; they are
  cited as measurements of latency on one process, never as promotable results.
