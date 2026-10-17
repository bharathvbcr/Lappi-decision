# D: Mac runtime audit, Qwen3.5 LoRA via ojas Metal (main 5b67cdd, 2026-10-16/17)
Labels: [V] verified (command/file read this session), [I] inferred, [U] unverified.
Tool gaps: `devmap status --repo-path` is not a valid flag in CLI 0.2.4 (status ran from cwd, capabilities listed, I did not check walk_incomplete beyond that); I used rg/sysctl/file reads for this audit, not the graph, and GitPulse Insights was not callable (not loaded). Nothing below is "graph-confirmed". Empty rg results were re-run with `-uu` (RIPGREP_CONFIG_PATH empty) where absence is claimed.

## 1. Hardware [V]
- hw.memsize 68719476736 (64 GiB), hw.model Mac17,8; MacBook Pro, Apple M5 Pro, 18 cores (6 Super + 12 Performance); macOS 27.0.1 (26A434). Metal recommended working set 51,539,607,552 B (51.5 GB) per bench env.txt (not re-queried: no GPU run allowed).
- Disk: /System/Volumes/Data 926 GiB, 153 GiB free (83% used). Repos and ~/.cache/huggingface are on that one volume. mac_heavy refuses below 15 GB free: fine.
- Power: AC, battery 85% not charging; `pmset -g therm`: no thermal/performance warning recorded.

## 2. Guards [V]
- memguard: ~/Library/LaunchAgents/com.bharath.memguard.plist (KeepAlive, pid 89858 loaded), flags `-cap-gib 32 -floor-gib 2 -critical-samples 3 -interval 2s -grace 10s -settle 5s -heartbeat 1h`. Source ~/Code/devtools/memguard (README, policy.go, proc_darwin.go:22).
- Exact rule: every 2 s sample ri_phys_footprint of every process owned by the user; (a) over-cap: any unprotected process with footprint > 32 GiB (34.36 GB) gets SIGTERM, SIGKILL after 10 s; (b) critical-pressure: kern.memorystatus_vm_pressure_level critical for 3 consecutive samples kills the largest unprotected process of >= 2 GiB. It is per process, not per tree. README admits "whether ri_phys_footprint counts GPU-wired memory has not been verified"; the 2026-10-11 provider rows (killed at 32.98 GB of device bytes, no Metal alloc failure) are consistent with Metal bytes counting [I]. Log dir ~/Library/Logs/memguard/ does not exist now [V] (README of 2026-10-11 says it was deleted), so kills leave no record: infer from SIGTERM exit.
- mac_heavy.sh (Lappi-decision/tools/mac_heavy.sh): one mkdir lock at Lappi-decision/build/mac-heavy.lock.d (wait up to 4 h, then exit 75); refuses <15 GB free (74); waits while 1-min load > 12 or procs > 2500 (76); CARGO_BUILD_JOBS=2; memory cap on summed phys_footprint of the job tree (python mac_footprint.py, proc_pid_rusage), default 53248 MiB (52 GiB), SIGTERM then SIGKILL, exit 70; 3 reader failures stop the job (71). So it DOES see Metal memory (footprint, not RSS; script comment: 3 GiB MPS alloc moved RSS +0.04 GiB, footprint +3.06 GiB). My premise "does not see Metal" is false for this version. Effective ceiling: memguard's 32 GiB per-process (binds first), not mac_heavy's 52 GiB.
- Important gap: the footprint a LoRA step reports includes device bytes plus host heap (weights loaded via host, safetensors reads). Measured rows count device bytes only (peak_allocated_bytes); host footprint was not recorded in the 2B rows [V: no footprint column in 2B rows.jsonl]. Only the 124M row has footprint (2026-10-10-bf16-memory, footprint ~ peak + 0.1 GB).

## 3. Measured memory rows [V] (all device bytes, tessl peak_allocated_bytes, micro-batch 1, every layer checkpointed, Qwen3.5-2B instruct weights; text_config identical to 2B-Base)
Source: bench/results/2026-10-11-bf16-frozen-base, 2026-10-14-bf16-tied-table, 2026-10-15-qwen35-2b-tape-gate. All three are on main (the 2b-tape-gate commit 3cb7828 is contained in main; `git branch --contains` lists main). All are FULL-GRADIENT Tape steps (every param incl. frozen base gets an f32 gradient), no optimizer. None is a LoRA step.
| T | linears | table | resident GB | after fwd GB | step peak GB | wall |
|---|---|---|---|---|---|---|
| 128 (2B-Base, f32 all) | f32 | f32 | 7.55 | n/a | 17.60 | 853 ms (1714 first run) |
| 2048 | f32 | f32 | 7.55 | 11.06 | 19.14 | 8.7 s |
| 2048 | bf16 | f32 | 4.80 | 8.34 | 16.37 | 8.3-10.1 s |
| 2048 | bf16 | bf16 | 3.78 | 7.32 | 15.35 | 8.5 s |
| 9638 | f32 | f32 | 7.55 | 12.74 | 22.73 | 34.3 s |
| 9638 | bf16 | f32 | 4.80 | 9.95 | 18.97 | 34.5 / 40.6 s |
| 9638 | bf16 | bf16 | 3.78 | 9.00 | 18.02 | 41.4 s |
- No activation-checkpointing-off row exists for the 2B on the Tape (every row checkpointed) [V]. Only the 124M nanolab rows (2026-10-10-bf16-memory, T=1024, ckpt off) exist for the off case; not transferable.
- Provider (tessl Qwen35Step, full FT + AdamW): 30.13 GB resident; f32 T=2048 peak 33.89 GB; bf16 T=2048 and both T=9638 were SIGTERMed by memguard. That is the full-FT path; irrelevant to LoRA except as evidence of where the 32 GiB guard bites.
- LoRA: ojas-model/src/lora.rs and Metal tests exist (qwen35_lora_metal.rs, tiny fixtures; tasks/gp-ft-bf16-frozen-base.md:119 says criterion 7 open: gpu_memory_rows.rs has no LoRA arm and no 0.8B run). No real-2B LoRA step has been run or measured [V]. tasks/gp-ft-frozen-params-and-lora.md status: review; says D17 and span head do not exist yet [A, per tasks/gp-ft-bf16-frozen-base.md:119 "5 partial"].
- Still unmeasured: (a) real 2B LoRA step peak at T=9,638 (device and host footprint); (b) AdamW moments + LoRA grads on top; (c) D17 and span head wired in a step; (d) 0.8B any row; (e) host-side footprint (memguard's actual metric); (f) multi-sequence token-budget batch (35,403 tokens) memory; (g) peak over many steps (fragmentation/pool growth), because every row is a single step.
- Note the LoRA fused head: frozen-head gradient fix is claimed on gp branch (ft-2bbced18, tasks/gp-ft-frozen-params-and-lora.md:168-172), main state not re-verified here [U]; the 2026-10-14 README says the fused head still computes the table-sized gradient (2.03 GB f32 / 1.02 GB bf16) in measured rows.

## Memory budget table, 2B, T=9,638, micro-batch 1 (GB, device)
| component | bf16 frozen base | q8 frozen base | basis |
|---|---|---|---|
| frozen linears | 2.75 | ~1.46 (q8 g64: 1.0625 B/param incl bf16 scale+bias, vs 2) | bf16 measured [V]; q8 inferred [I] |
| tied embedding | 2.03 f32 / 1.02 bf16 | same | measured [V] |
| norms, conv, A_log, etc | ~0.002 | same | measured [V] |
| resident (bf16 table) | 3.78 | ~2.5 | measured / [I] |
| LoRA r=16 params+grad+AdamW m,v (15.67M params x 16 B) + D17 (35k) + span head | ~0.25 | ~0.25 | [I], computed from config.json dims (H=2048, I=6144, 8/2 heads x 256, GDN 16x128 key/value); span head size not counted |
| activations + checkpoints (after-fwd minus resident) | ~5.2 (9.00-3.78) | ~5.2 | measured [V] |
| backward transient | ~+9.0 in full-grad row, minus ~5.5 linear dW (+1.0-2.0 table dW if not frozen) | same, + dequant transient | [I] |
| step peak, full gradient (reference) | 18.02 (bf16 table) / 18.97 (f32 table) | not built | measured [V] |
| step peak LoRA estimate | ~12-13.5 (+0.25 optimizer) | ~11-12.5 | [I] (repo's own estimate 13-14 GB: tasks/gp-ft-bf16-frozen-base.md:121) |
| host footprint add (safetensors read, host f32 staging) | unmeasured | unmeasured | [U] |
Verdict: memory fits with large margin against both memguard (34.4 GB) and the 51.5 GB working set; q8 buys only ~1.3 GB. q8 is NOT available: gp-ft-quantized-frozen-base is backlog/gated (needs mlx quantization gate and a demonstrated memory need; the audit note says memory need is unmet), the pre-registration says "No 8-bit or 4-bit ... not built here" and lists it under not_decided_here (campaign/lora-2b-preregistered.json base.precision, not_decided_here), and rg -uu finds quantization only in gemma-metal (inference GEMV Q4/Q8), ojas-simd/arch.rs, ojas-cuda/k11_golden.rs, none in ojas-core/ojas-model/ojas-metal training paths [V]. The q8 arm is a weeks-scale build, not a config change, and is not pre-registered.

## 4. Throughput and runtime fit
Measured Tape step wall (full gradient, checkpointed, bf16 linears, one step each, includes first-step pipeline warmup and no optimizer): T=2048 8.3-10.1 s = 203-246 tok/s; T=9,638 34.5-41.4 s = 233-279 tok/s; T=128 0.85-1.7 s. Real LoRA step throughput: unmeasured.
What a step is [V, campaign/lora-2b-preregistered.json recipe.batch]: token_budget 35,403, grad_accum 1, "One step is one batch". Train shard 558,790 seqs / 386,246,788 tokens (mean ~691 tokens/seq), widest 9,638. So a step is a token-budgeted multi-row bucket batch (about 35k tokens), not one 9,638-token sequence. "micro-batch 1" in the memory rows means one sequence per forward; how a 35,403-token bucket is run at micro-batch 1 with grad_accum 1 is internally inconsistent in the prereg text [U]: if the 35k tokens are processed as many B=1 forward/backward passes inside one optimizer step, that is effectively grad_accum ~ many; if they are one padded batch, memory scales with batch tokens (activations ~ 35k tokens, not 9.6k) and no row exists for it. Needs a ruling.
Estimate [I]: 35,403 tok / (233-279 tok/s full-grad) = 127-152 s/step. LoRA skips frozen dW GEMMs (~25-30% of step FLOPs [I]) -> optimistic 90-120 s/step; short rows are slower per token (T=128 ~150 tok/s, T=2048 ~205-245). 
- 200 steps: 5.0-8.4 h (optimistic-to-measured); 400 steps: 10-17 h.
- Fits 3-6 h only if 200 steps AND the optimistic LoRA speed-up holds AND no contention; 400 steps does not fit. Safe plan inside 6 h: roughly 140-190 steps at 35k tokens [I].
- Alternative reading (one step = one <=9,638-token sequence): 28-41 s/step -> 200 steps 1.6-2.3 h, 400 steps 3.1-4.6 h: fits. That contradicts the registered token_budget, so it would be a recipe change.
- 0.8B smoke (60 steps): no Tape 0.8B row exists. Scaling by params (~0.8/2 [I]) gives ~40-50 s/step at 35k tokens -> ~40-50 min for 60 steps; prereg estimates ~2 h incl torch arm. Unmeasured.
- Prior evidence on the torch MPS side (Lappi ledger mac-smoke-frozen-2026-09-29.jsonl, real 2B-Base text tower b1485b2f, short rows <=1515 tokens): 352 optimizer steps in 1809.7 s (5.1 s/step) on MPS; a 21-seq sibling run was killed by SIGTERM at 246.8 s (mac-smoke-2026-09-29.jsonl, status killed, "received SIGTERM", 1.43 B trainable params, bf16 + grad checkpointing). Those are not 9.6k-token rows.

## 5. Contention right now [V]
- mac_heavy lock held: label w11-merge-final2, pid 37673 (alive), cwd ojas-merge-pass2, started 2026-10-17T04:52Z; two rustc compiles and `cargo build --release --all-targets` running. A LoRA job would queue (or wait on load).
- load average 12.14 / 15.95 / 17.16 (mac_heavy waits above 12). vm.swapusage: 30.8 GB of 31.7 GB used (almost full!), compressor holding ~8 GB; ~13.8 GB free pages. Top RSS: root fseventsd 13.5 GB (104% CPU, 2255 CPU-min), WebKit 1.7 and 1.4 GB, Chrome, GitPulse.app 0.66 GB at 220% CPU, multiple Claude Code sessions (~0.6-0.8 GB each, 5 seen). Nothing holds big GPU memory right now [I]: no ojas/tessl test binary in top-RSS list.
- Worktrees (git worktree list): ojas (main), ojas-clipnorm (gp/clip-sq-norm), ojas-merge-pass2 (detached 5b67cdd), ojas-w11-oracle, ojas-w11-readout-sync, ojas-w11-scratch. Sibling dirs also exist for tessl (tessl-w11-2bmem, tessl-w12-kcov, ...). Peer agent sessions: several Claude Code CLIs visible; GitPulse Insights not queried (tool not loaded; gap).
- Risk: the Mac has been under sustained load ~12-17 with swap full; a 3-6 h GPU run needs the machine quiet and the lock released (see risks).

## 6. Snapshots [V]
- ~/.cache/huggingface/hub: models--Qwen--Qwen3.5-2B-Base snapshot b1485b2fa6dfa1287294f269f5fb618e03d52d7c, model.safetensors-00001-of-00001 -> blob 928acbf1... (matches prereg sha), 8.5 GB dir; models--Qwen--Qwen3.5-0.8B-Base snapshot dc7cdfe2... blob c2b1e5a1... (matches prereg), 3.3 GB; Qwen3.5-2B (instruct) 15852e8c... 8.5 GB; 4B-Base config-only 44 MB. Both required base weights are present (the 2026-10-11 README "not on this machine" is stale).
- Test defaults: ojas-qwen35/tests/common/mod.rs:25 (default `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots`, "the one snapshot"), env QWEN35_2B_SNAPSHOT (gpu_tape_2b.rs:87, gpu_parity.rs:746, cpu_names.rs:174, gpu_per_entry.rs:139 which defaults to the instruct 2B dir), QWEN35_MEM_SNAPSHOT (gpu_memory_rows.rs:76). 0.8B has no default path in tests (set env). Note safetensors are stored as a single "model.safetensors-00001-of-00001.safetensors" file, 2B storage mostly bf16, 36 F32 tensors.
- Data: train shard path /Users/bharath/qd-campaign/phase4-v5r-2026-10-03/shards/train (prereg). Existence of the shard dir not checked [U] (`ls ~/qd-campaign` returned entries but I did not verify the v5r path).

## 7. Thermal/long-run evidence [V/I]
- No ledger row of a multi-hour Rust-Metal training run exists; Rust-Metal rows are single-step or scoring (mac-qd-metal-*). Longest Mac training evidence: torch MPS 1809.7 s (30 min) 2B-Base short-row run completed; a 247 s run killed by SIGTERM (memguard-era or user kill; reason says only "received SIGTERM") [V].
- memguard history: 2026-10-08 a Python process hit 75 GB (plist comment); mac_heavy comments cite kernel panics 2026-10-02 at load ~350 and again at load ~30 with one job; JetsamEvent showed test processes at 36-120 GiB resident [V in mac_heavy.sh header]. 2026-10-11 provider rows and 2026-10-15 provider gradient leg were SIGTERMed by memguard [V, READMEs].
- No thermal throttle evidence either way: pmset reports no thermal warnings now; MacBook (fan-cooled, not Mac Studio) sustained 3-6 h GPU load is untested here [U].
- Hazards for a long run: the 2026-10-12 incident where an in-place edit of mac_heavy.sh killed four re-score jobs is mitigated by the function wrapping; macOS auto-sleep/lid-close not examined [U].

## Runtime-fit verdict
- bf16 frozen base, 2B: memory FITS (est. 12-14 GB device, upper bound measured 18-19 GB at T=9,638, all under both guards). Time: 400 steps at 35,403 tokens does NOT fit in 6 h; 200 steps is marginal (5-8 h). Only a sequence-per-step reading fits.
- q8 arm: not runnable (not built, not pre-registered, gated); memory gain is ~1.3 GB, so no need. If the user wants "q8" it must be via the gated mlx measurement plus kernel work.
- 0.8B smoke: weights present; no measured row; likely feasible (< ~1 h for 60 steps [I]).

## Risks
1. No real LoRA memory/throughput row exists; criterion 7 open and prereg says the T=9,638 memory row must precede the run. Run it first (est. 1 h).
2. Step definition ambiguity (35,403-token bucket vs micro-batch 1 vs grad_accum 1) changes time by ~4x and memory unknown.
3. memguard is per-process at 32 GiB with no log; swap is 97% full and load is 12-17 now, so a heavy job would wait or thrash; mac_heavy lock currently held by w11-merge-final2.
4. Host footprint (not measured) adds to device bytes in memguard's metric.
5. Single-step walls include warmup; sustained-throughput and thermal behavior over hours unmeasured.
6. D17 + span head not yet implemented on the Tape per task notes; real run needs them.
