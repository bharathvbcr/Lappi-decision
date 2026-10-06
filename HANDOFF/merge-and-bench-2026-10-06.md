# Merge, suite and Mac benchmark lane, 2026-10-06

The human, 2026-10-06 (~05:45Z): "yes, run the suite on main; Commit and Merge all branches into
the main branch by resolving conflicts carefully and precisely, then delete all other branches.
Ask fable for decisions and work on them. Benchmark the latest Lappi model using PyTorch, Ojas, or
Tessl on my Mac for inference. Compare the original model against different tasks in the Jevarena
benchmark and evaluate how Lappi performs relative to other models in its class."

**Labels.** [V] is measured by a command this lane ran (log path cited). [I] is inferred. [U] is
not verified. Fable's rulings this lane: run a baseline before anything moves; gate main's
uncommitted work on its own tests and do not fix peer code; merge the three branches; keep the
UnsafeCell variant as a patch; delete branches with `-d` only; no downloads or paid calls without
the human's yes.

Scripts and logs: `build/merge-2026-10-06/` (git-ignored; `README.md` there) and
`/Users/bharath/qd-campaign/lappi-bench-2026-10-06/`.

## 1. The suite on main before the merge (39cfe6c), in a clean worktree

[V, `build/merge-2026-10-06/baseline-suite{,2,3}.log`, `baseline-lint.log`]
- `cargo test --workspace`: **1093 passed, 0 failed**, 22 ignored.
- qd-metal GPU tests (`--ignored`, this Mac's Metal): **6/6 passed**. The first run's one failure
  was the harness: `gpu_product_path_host_cost_vs_model` needs `QDM_PRODUCT_LEDGER` (`none` keeps
  it a test).
- Python, `python/tests` single process, ML env: **4276 passed, 68 skipped, 2 failed, 69 errors**.
  - The 2 failures are the known worktree-only tests: `test_gaps_writer` (checks the checkout's
    name) and `test_lint_gate::test_ruff_is_installed_not_merely_declared` (no `.venv/bin/ruff` in
    a worktree). Not fixed; installing ruff into a worktree venv was not asked.
  - The 69 errors were this lane's harness: the suite set `CARGO_TARGET_DIR`, and fixtures look for
    the binaries they build under `<worktree>/target`. The 7 files, rerun without it: 77 passed,
    20 skipped, 0 failed.
- uv's cache no longer held pytest, so the suite now runs README's own command without `--offline`
  (uv fetched the project's declared test tooling into its cache).

## 2. Main's 16 uncommitted paths: tested, held, not committed

The batched `decode_slots` work (Fable's ruling item 3 in `HANDOFF/metal-inference-2026-10-02.md`),
edited 2026-10-03 19:25 to 10-04 08:58 local. Owner unknown: the two sessions asked (f88b2d,
4b8cd0) both said these paths are not theirs.

Tested byte for byte in a worktree (`build/merge-2026-10-06/peer-hashes.txt`) [V, `peer-suite2.log`]:
- `cargo test --workspace`: 1107 passed, 0 failed. The new
  `gpu_two_suffixes_as_one_batch_match_batch_one` passes.
- **`gpu_prefix_state_digest_v1_is_pinned` fails**: all four v1 digests move, where main passes on
  the same tessl 52c091c. The cause is in the diff: `crates/qd-metal/src/model.rs` swaps prefill
  attention from `nn::flash_attn_rows` to `qwen35::attn_prefill_by_length`. No re-pin and no parity
  row came with it.
- Clippy adds 2 warnings. Its docs describe 10-04 ("v5 running", "no model has shipped") and say
  the batch "halves" dispatch overhead with no committed measurement.

Per Fable: not fixed, not committed, left untouched in main's checkout (still 16 paths after every
step below). GAP-MAIN-UNCOMMITTED-BATCHED-DECODE-FAILS-ITS-DIGEST-PIN-2026-10-06; GitPulse card
`lappi-held-batched-decode-digest-pin`.

A hazard found on the way: with one `CARGO_TARGET_DIR` shared by two worktrees, cargo judged the
baseline's release test binary fresh and ran it for the peer tree
(GAP-SHARED-CARGO-TARGET-DIR-RUNS-ANOTHER-WORKTREES-CODE-2026-10-06). The suite now uses one target
dir per worktree.

## 3. The merge (main 39cfe6c → 1feae5a, fast-forward)

Built in a detached worktree (`build/merge-int-wt`), tested, then fast-forwarded; main's 16 held
paths were unchanged across it (`ff_main.sh` checks).

| Commit | What |
|---|---|
| f23c890 | Merge `preview-export-train-manifest-bound` (a766fcf, the v0.1 preview's exporter). No conflicts. |
| 07ece42 | Merge `v6-linfit-pool` (be65fa5 and the team of threads). No conflicts. Its commits include all of `v6-linfit-pool-cell`'s. |
| e25c098 | Merge `v6-spancheck` (`qd-prep spancheck`, off by default). Two conflicts, below. |
| 1feae5a | `HANDOFF/v6-linfit-pool-unsafecell-variant-2026-10-04.patch`: `v6-linfit-pool-cell-wt`'s one uncommitted file, the UnsafeCell `team.rs` (the crate's first `unsafe`, the human's policy call), kept and not applied. `git apply --check` passes; never built as a whole fit. |

Conflicts, resolved by hand and identical to a dry run's resolution (`resolved-qd-prep-lib.rs`):
- `crates/qd-prep/src/lib.rs`: both sides added a module. Both doc bullets kept (`team`, then
  `spancheck`); `pub mod spancheck;` before `pub mod team;`.
- `gaps.jsonl`: main's side kept as it is (it reordered records since the merge base); the
  branch's one appended record added after it. `merge_one.sh` checks the branch's side is the base
  plus appends.

**Suite on the merged tree** [V, `int-suite.log`, `int-suite-py.log`, `int-parity*.log`]:
- `cargo test --workspace`: **1121 passed, 0 failed**, 22 ignored (+28 tests from the merges).
- qd-metal GPU: **6/6 passed**, the digest pins included.
- Python: **4409 passed, 74 skipped, 2 failed** (the same two worktree-only tests), 0 errors.
- qd-prep parity on the merged binary, including the real commitpackft corpus (main's data,
  symlinked into the worktree): `test_qd_prep_linear_parity` all passed, among them
  `test_real_defect_prompts_fit_bit_for_bit_and_within_tolerance_of_dense`;
  `test_qd_prep_spancheck_parity` all passed, among them the four end-to-end `run()` cases. Only the
  opt-in benchmarks (`QD_PREP_BENCH=1`) skipped.
- Clippy (`qd-runtime`, `qd-metal`): the same 2 warnings as before the merge.
- rustfmt is not clean on main and was not before: 1,558 diff hunks at 39cfe6c, 1,571 after (the
  13 new in `spancheck.rs`). No gate runs fmt. GAP-WORKSPACE-IS-NOT-RUSTFMT-CLEAN-2026-10-06.

The first Python pass on the merged tree failed 141 tests and errored 14: at ~06:44Z something
deleted `/Users/bharath/qd-campaign/target-merge-2026-10-06` wholesale, mid-suite, taking
`QD_PREP_BIN` with it. The second such deletion today; the deleter is not identified
(GAP-CARGO-TARGET-DIR-DELETED-MID-SUITE-2026-10-06; card updated). Rerun from a copy outside every
target dir, all of them passed. `deleter_watch.sh` then kept a sentinel in that dir for its
2-hour limit and saw no further deletion.

## 4. Branches deleted

Every local branch but main, with `git branch -d` after a check that main contains it:
`preview-export-train-manifest-bound`, `v5-build`, `v6-ctl-minrows`, `v6-ctlcache`,
`v6-linfit-pool`, `v6-linfit-pool-cell`, `v6-spancheck`, `v6-startup`. Nothing was pushed;
`origin/main` is unchanged. A branch checked out in a worktree was first detached there (same
commit, files untouched). This lane's own scratch worktrees (merge-base, merge-peer, merge-dry)
were removed.

**The 15 leftover worktrees, archived then removed** (the human's answer to this lane's question: "Copy
cited logs, then remove") [V, `build/merge-2026-10-06/worktree-archive.log`]. `worktree_archive.sh` checked
each HEAD is in main, then kept under `build/worktree-archive-2026-10-06/<worktree>/` (611 MB):
- every file not tracked at its HEAD (untracked, git-ignored, or a path a HANDOFF/AUDIT cites) that
  is not byte-identical to main's checkout, one copy per sha256 (APFS clones), each copy verified
  by hash, with `MANIFEST.tsv`;
- `uncommitted.patch` for the four with uncommitted edits (`v6-linfit-pool-cell-wt`, identical to
  the committed UnsafeCell patch; `j6g-prelude-build-a502670`; `lappi-bench-2026-10-03/h1-wt`;
  `v5-decisions-data-2026-10-03/wt`);
- `NOT-COPIED.tsv`: regenerable caches (`target/` but cited files, `__pycache__`, the two 1.6 GB
  split-cache `*.rows.pkl`) and symlinks; `CITED-MISSING.txt`: cited paths already gone.

Of note: main's checkout has no copy of the v5 corpus files `commitpackft-composed-v2/examples.jsonl`
(628 MB), `commitpackft-g6-v1`, `defect-noul-v3b` and `own-prose-v1/units.jsonl`; the only local
copies were in `v5-build-wt` and `v5-L-wt` (identical). They are now in
`build/worktree-archive-2026-10-06/build_v5-build-wt/files/data/pool/`. Then `git worktree remove`
on each and `git worktree prune`; the lane's own `merge-int-wt` and bench `main-wt` remain until it
closes.

## 5. Inference speed on this Mac (M5 Pro, Metal 4)

`speed_all.sh` on main 1feae5a, tessl 52c091c, 07:32-07:41Z; one hold, ABBA order, busy gate
(load ≤ 12; the 1-minute load peaked at 14.8 during the run). Base is Qwen3.5-2B-Base (weight hash
92f6bd1c…); Lappi is the v0.1 preview release (6f7b9ba7…). Tables: `lappi-bench-2026-10-06/summary.txt`.

- **Ojas: no arm.** ojas has no Qwen3.5 inference path: `ojas-infer` decodes nanolab GPTs only, and
  `ojas-qwen35` is a training-step provider (its `forward` is the training forward with the letter
  loss) over the same tessl kernels.
- **tessl prefill** (qd-metal-bench, min of 7, min over the two ABBA runs) [V, `logs/prefill-*.log`]:
  200 tokens 42.3 / 44.8 ms (base / Lappi), 409 tokens 79.1 / 79.6, 2,048 tokens 356.8 / 357.4,
  8,192 tokens 2,083 / 2,095.
- **PyTorch 2.12.1 on MPS** (bf16, sdpa, same last-position letter readout) [V,
  `torch-mps-results.json`]: 1.6× to 3.2× slower than tessl at the same lengths (e.g. 200 tokens
  ~89 ms, 8,192 tokens ~3.3 s).
- **Lappi costs what base costs:** Lappi ÷ base is 0.93-1.06 on tessl prefill, 0.93-1.02 on the
  full decision, 0.96-1.05 on torch. Same architecture, same kernels; only the weights differ.
- **A full decision on the product path** (prefill + the two read-only choice passes + state
  digests, `--decision`, every sample bit-identical): ~72 ms at a 124-token prefix, ~85 ms at 197,
  ~127 ms at 408, ~180 ms at 741, ~0.41-0.44 s at 2,042, ~2.3 s at 8,185. Rows 31fc6f1b, a15e495a,
  553debc1, 21b75996 (quick throughput rows) in
  `lappi-bench-2026-10-06/ledger/mac-qd-metal-bench-2026-10-06.jsonl`.
- Against 10-03b (v4 release, build 75c683b) the decision path reads 1.13-1.31× faster, but prompt
  format 2 moved the question into the passes (61 + 61 tokens now, shorter prefixes), the build
  differs, and the load differed. Not claimed as a speedup.

## 6. JevArena (JevJudge-Bench) quality

**Setup.** `external/jevarena` (chenmingtang830/jevarena d01667e), its own protocol: a three-way
pairwise judge (A / B / TIE) over JudgeBench, RM-Bench and RewardBench 2, every pair asked in both
answer orders. Data `test-g6`: `jevjudge prepare --groups-per-domain 6 --partition test --seed 42`,
849 pairs in 168 prompt groups, 28 domains; gold JudgeBench A 62 / B 59, RewardBench 2 A 392 /
TIE 66, RM-Bench A 270. Accuracy is the mean over both orders on pairs both orders finished;
intervals are the harness's prompt-group bootstrap (1,000 draws, seed 42). Local models only, $0
(the human: no paid calls, so no hosted judge). Tables: `lappi-bench-2026-10-06/jev-compare-all-test-g6.txt`
and `jev-compare-pilot.txt` (`compare_all.sh`; `compare.py` is throwaway analysis).

Judges [V, each run's `runs/<name>/` in `external/jevarena`]:
- **Lappi v0.1 preview**, prompt format 2, through `jevarena-lappi` (Rust, qd-metal on tessl): the
  letter logits A / B / C(=TIE), first-pass argmax, uncalibrated. **Lappi v4** (format 1) and the
  **base read the same way** (letter readout, formats 2 and 1) for reference.
- **Qwen3.5-2B-Base as a chat judge** and **Qwen3.5-2B** (the post-trained release, downloaded
  this lane with the human's yes) through mlx_lm 0.31.3, chat template with thinking off, stop
  `<|im_end|>`, 64-token cap.
- **Gemma 4 E4B** (`gemma4:e4b-it-q4_K_M`, Ollama 0.35.0; `ollama show`: 8.0B parameters, Q4_K_M)
  with `reasoning_effort: none`, 2,048-token cap.

| Judge (test-g6, 849 pairs) | domain macro | without Ties | all (micro) [95%] | JudgeBench | RB2 | RM-Bench | order consistency | failed calls | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| Lappi v0.1 | **0.534** | 0.528 | 0.610 [0.534, 0.662] | 0.512 | 0.689 | 0.519 | 0.716 | 0 | 158 |
| Qwen3.5-2B, chat | 0.522 | 0.514 | 0.646 [0.575, 0.694] | 0.466 | 0.712 | 0.613 | 0.523 | 2 | 304 |
| Qwen3.5-2B-Base, chat | 0.468 | 0.460 | 0.569 [0.499, 0.614] | 0.446 | 0.619 | 0.531 | 0.531 | 26 | 323 |
| Base, letter readout (f2) | 0.495 | 0.498 | 0.461 [0.431, 0.499] | 0.496 | 0.428 | 0.500 | 0.001 | 0 | 152 |
| Lappi v4 (f1) | 0.444 | 0.438 | 0.531 [0.444, 0.594] | 0.393 | 0.596 | 0.483 | 0.620 | 0 | 145 |
| Gemma 4 E4B (8.0B total), no reasoning | **0.632** | 0.621 | 0.812 [0.733, 0.863] | 0.522 | 0.892 | 0.800 | 0.801 | 7 | 318 |
| decider-2b v11 (2B, same base), CPU float32 | **0.612** | 0.605 | 0.719 [0.661, 0.767] | 0.554 | 0.763 | 0.719 | 0.749 | 0 | 5,679 |

Macro weights each of the 28 domains equally (point estimates, complete pairs); the `all` row is a
micro diagnostic dominated by RewardBench 2's Ties domain (368 of 849 pairs). Latency is the
harness's per-call wall clock across different transports (stdio vs loopback HTTP), not a kernel
measurement; section 5 has those.

What the numbers say:
- **Against its same-size peer, Lappi ties.** Qwen3.5-2B minus Lappi, paired on the 847 pairs both
  completed: **+0.035 [-0.039, +0.094]**; JudgeBench -0.055 [-0.120, +0.009], RewardBench 2 +0.023
  [-0.078, +0.095], RM-Bench +0.094 [-0.028, +0.219]. None excludes 0. Macro 0.534 vs 0.522.
  Lappi's real edges over it: order consistency 0.716 vs 0.523 (0.695 vs 0.510 with TIE-TIE pairs
  excluded, since a judge that always says TIE is perfectly consistent; Lappi's TIE rate is 0.121,
  Qwen's 0.080), no failed calls, and one forward pass per verdict instead of generated JSON.
- **The base comparison that counts is base as a chat judge:** Lappi is ahead by 0.043 [-0.011,
  +0.101] overall, significant only on JudgeBench (+0.074 [+0.005, +0.144]). The letter readout
  of the base is degenerate (it answers A on 99.6% of calls: 0.5 on every A/B pair and 0 on TIE
  golds), so Lappi's +0.149 [+0.051, +0.213] over it mostly says Lappi learned prompt format 2 and
  the base did not. It is not the headline. Over v4, Lappi v0.1 gains +0.078 [+0.009, +0.162],
  +0.120 on JudgeBench.
- **Outside RewardBench 2's Ties domain Lappi is at chance.** Macro without Ties 0.528, where
  always-A scores 0.5 on A/B pairs; JudgeBench 0.512 on balanced gold; RM-Bench 0.519 on all-A gold.
  Its one clear lift is Ties (0.719 vs base-letter 0.410).
- **RM-Bench code: 0.111 [0.019, 0.222] on its 54 pairs** (gold is always the correct program).
  - Of 108 calls Lappi answers TIE on 53, the rejected program on 43 and the correct one on 12.
  - The same pairs: Gemma 0.639, base-as-chat 0.361, Qwen3.5-2B 47 rejected vs 55 correct.
  - Exploratory: 6 prompt groups.
  - *Corrected after the first commit of this file:* this is not "a defect in the product's own
    domain". These pairs are out of Lappi's task: they route through its weakest family,
    pairwise.helpfulness, not code.defect_class. The defect probe in section 7 shows the
    defect-class skill does not carry over to them either.
  - GitPulse card `lappi-rmbench-code-prefers-rejected` (board item
    `ft-882ed8fcd29cfacd3e1cf14748a0e9de`), revision 3 carries the diagnosis and the probe.
- **A larger model is clearly better.** Gemma 4 E4B (8.0B total parameters, about four times
  Lappi's) minus Lappi: **+0.200 [+0.133, +0.257]**; RewardBench 2 +0.203, RM-Bench +0.281, but
  JudgeBench +0.000 [-0.075, +0.073]: every judge here is near chance on JudgeBench. Its 7 failed
  calls: 6 replies that were not the required JSON, 1 Ollama HTTP 500.

**Gemma 4 E4B with its default reasoning, on the 191-pair pilot.**
- **Setup:** the human's choice. The cap is 8,192 tokens, because PROTOCOL.md requires "enough total
  output/reasoning tokens to finish a verdict" and a 2,048 cap truncated 2 of 10 probe calls.
- **Run:** finished complete at 14:26:09Z: 382 calls, 1 failed (ValueError),
  implementation_sha256 `5ccaa922…`.
- **Table:** `jev-compare-pilot.txt`. Every test-g6 run is re-scored on the same 191 pairs, a subset
  of test-g6.

| Judge (pilot, 191 pairs) | domain macro | all [95%] | JudgeBench | RB2 | RM-Bench | order consistency | p50 s |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B, reasoning on | **0.708** | 0.816 [0.728, 0.882] | 0.632 | 0.855 | 0.867 | 0.853 | 14.0 |
| Gemma 4 E4B, no reasoning | 0.635 | 0.721 [0.599, 0.815] | 0.528 | 0.782 | 0.756 | 0.729 | 0.59 |
| Qwen3.5-2B, chat | 0.547 | 0.616 [0.512, 0.709] | 0.487 | 0.710 | 0.606 | 0.484 | 0.37 |
| Lappi v0.1 | 0.529 | 0.571 [0.466, 0.671] | 0.487 | 0.734 | 0.494 | 0.639 | 0.24 |

- **Reasoning on, minus Lappi**, paired: **+0.245 [+0.122, +0.347]**. JudgeBench +0.145
  [+0.012, +0.284], RM-Bench +0.372 [+0.183, +0.533].
- **No reasoning, minus Lappi**, on the same pairs: +0.144 [+0.002, +0.258].
- **Output tokens:** median 698, p90 2,505, max 5,470, so 0 calls reached 8,192. That is
  [inferred]: the record's `reasoning_tokens` is empty, and it assumes Ollama's `completion_tokens`
  counts the thinking tokens, which the latency fits.
- **Limits:** the pilot has no TIE golds (JudgeBench A 23 / B 16, RewardBench 2 A 62, RM-Bench A 90),
  so it says nothing about Ties. Reasoning-on is a distinct configuration, at about 58× Lappi's
  median latency.

**decider-2b (Mapika/decider-2b v11, rev 533964da, Apache-2.0; the human approved the download and
run).**
- **Its Mac-GPU path cannot be scored.**
  - The first smoke (probe5b, 14:26Z, Decider's own default MPS float16) failed 5 of 5 calls.
    decider_stdio.py could not encode the NaN answers as JSON.
  - Direct calls to its own `Decider.system_one` (`lappi-bench-2026-10-06/decider_debug.py`, logs
    `decider-debug{,2,3,4,5}.log`) answer NaN or a flat 1/3 on MPS in float16, bfloat16 and
    float32, with or without its MPS patch. With only its MoE patch off, they answer P=1.000.
  - That happens even on the 23-token warm-up. On CPU the same calls are finite, and bfloat16 and
    float32 agree (P(B) 0.777 vs 0.773).
  - The cause inside its MPS path is [unverified].
- **The CPU run.** The human: "run decider-2b on CPU for the full test-g6".
  - Config `decider-2b-v11-cpu`: float32, 6 torch threads, 600 s call timeout. The longest
    test-g6 state is 3,292 tokens; float32 took ~7 s per 950-token call, and bfloat16 was 3–5×
    slower on CPU.
  - decider_stdio.py now refuses to report ready unless the warm-up's easy question gets a finite
    A > 0.5.
- **Result** (row in the table above):
  - Smoke: 10/10 calls, median 7.7 s.
  - test-g6: complete at 17:44:34Z, 1,698 calls, 0 failed, implementation_sha256 `d39ba018…`.
  - **decider-2b minus Lappi, paired: +0.110 [+0.071, +0.157]** on all 849 pairs. JudgeBench +0.041
    [-0.017, +0.097], RewardBench 2 +0.074 [+0.038, +0.133], RM-Bench +0.200 [+0.106, +0.304].
  - Domain macro 0.612 vs Lappi 0.534. Order consistency 0.749 vs 0.716.
  - RM-Bench code (108 calls): correct 59, rejected 29, TIE 20; Lappi 12 / 43 / 53.
  - On the 191-pair pilot: 0.707 [0.590, 0.796], paired +0.136 [+0.048, +0.226].
- **Reading:** a decision model of the same base and size beats Lappi clearly on this benchmark.
  - Its model card lists "RewardBench" (no version given) and Arena preferences as held out of
    its training, and HelpSteer2 as trained (as it is for Lappi).
  - Whether RewardBench 2, RM-Bench or JudgeBench overlap its training is [unverified]. The card
    does not mention RM-Bench or JudgeBench.
  - Its latency here is CPU-only and not comparable; the card cites 4 ms per request with CUDA
    graphs.

Provenance and limits:
- The jevjudge harness carries a **local, uncommitted patch** in `external/jevarena` (never
  committed or pushed): a `lappi` stdio provider; plain HTTP to loopback only; `stop` in
  `extra_body`; and `JEVJUDGE_CONTINUE_ON_PARSE_ERROR=1`, which lets a local, zero-priced judge
  record a failed call (cost 0, wrong in attempted accuracy, absent from complete pairs) and go on,
  stopping after 5 failures in a row.
  - **Later fix:** a stdio judge whose reply timed out, broke or never came was asked again. Its late
    reply would have been read as the next request's answer.
    - Now the process is killed and the next call starts a fresh one.
    - The test `test_stdio_judge_that_timed_out_is_never_read_again` failed first against the old
      code ('slow' != 'fast').
    - No finished run timed out: the Lappi runs answer in milliseconds, and the decider smoke failed
      on error replies, not timeouts.
  - The harness's tests pass: 46 in `tests/`.
  - The patch changed during the lane; each run's `manifest.json` carries `implementation_sha256`:
    - the four Lappi/letter runs `c862e1d5…`;
    - base-as-chat `4728cb62…` (continued past parse errors only);
    - Qwen3.5-2B, Gemma no-reasoning and the reasoning pilot `5ccaa922…`;
    - the failed decider smoke `9f3fb0d9…`.
    - The decider CPU run gets a new hash, because the timeout fix changed `providers.py`.
- Gemma no-reasoning's first run stopped at 333 calls on the HTTP 500 (the harness then refused to
  resume a directory with a failed call); it is kept as `runs/chat-gemma4-e4b-ollama-nothink-test-g6-stopped-at-333`
  and the full rerun is the one reported.
- JevArena is not Lappi's task: Lappi was trained for code-change decisions and its letter readout
  is uncalibrated here. Nothing in this section is a gate, and no threshold was read against it.

## 7. Does the defect-class skill carry over to model-written bugs? (RM-Bench defect probe)

The human: turn each RM-Bench program into a new-file diff and ask Lappi its own defect question;
"run those". Lane dir `~/qd-campaign/lappi-bench-2026-10-06/defect-probe/`. $0, Mac only.

**What the trained task is** [V, rebuilt from the corpus]. code.defect_class asks whether a real
commit's diff carries a planted qd-mutate edit (stub, logic or cosmetic) or is clean. It is not a
code-correctness question. v5's corpus is commitpackft-corpus-v3: all 2,304 val rows rebuild
byte-identically (row_content_hash = the val manifest's content_hash). 21,893 of its 49,953 rows
are commits that add a whole file, rendered `@@ -1,1 +1,N @@` with every line `+`. The probe renders
programs exactly that way, checked equal on 918 whole-file val rows.

**Instrument.** `jevarena-lappi score`, sha256 `16aab7f5ff9058b21ed7e07c56e34302c3397e155f429eac2b6557fa60055e8f`.
- It runs on release-lappi-v0.1-preview: weight_hash `6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb`,
  tokenizer_hash `fe000e3e…`.
- Per request it runs the runtime's own `parse_line` → `admit_defect_context` → `render` → the
  `defect_class` slot, both passes. The judge path never ran admission; this one does.
- Its unit tests include admission (py/go/rs/ts admitted; cpp/java/js refused) and a test that the
  choice prompt does not depend on the span slot.
- Requests: `requests.jsonl`, sha256 `6bee669f…`, 909 lines.

**Readout (pre-registered in `build_requests.py`, then `analyze.py` → `report.txt`):**
- **C1, fidelity: trained val rows, must read ≥ 0.90.**
  - First registration: **0.887, FAIL.** The sample took ≤ 12 rows per class×language×shape cell,
    so it over-weights small, hard cells against ledger 6af73bef's population figure (0.950–0.958).
    `c1_reweight_posthoc.txt` (post-hoc) reweights it to 0.950.
  - Re-registration (`REREGISTRATION-c1full.md`, written before scoring): all 2,304 val rows,
    `requests-c1full.jsonl` sha256 `6d7658d0…` → `report-c1full.txt`.
    - First-pass top-1 **0.9562** (2,203/2,304), permuted-pass agreement 2,295/2,304 (= the ledger):
      **PASS**.
    - whole_file 0.945, edit 0.965; Rust is lowest at 0.862.
- **C2, positive control: the same file, real vs planted.** A whole-file val row with a planted
  defect, scored on its own trained context, against the pool's real file at the same path.
  - **0.906 [0.835, 0.965]** of 85 pairs give the real file the higher P(clean): SEPARATES.
  - stub 1.000 (36), logic 0.960 (25), cosmetic 0.708 [0.500, 0.875] (24, no separation).
- **A: RM-Bench code_filtered** (THU-KEG 73c52d7), style 2 (markdown; styles 0/1 flatten the code),
  the longest fenced block as `solution.<ext>`.
  - **0.536 [0.448, 0.624]** of 125 pairs give the chosen program the higher P(clean): **no
    separation**.
  - Go 0.583 (36), Python 0.519 (27), Rust 0.516 (62). Median P(clean): chosen 0.74, rejected 0.71.
  - 103 pairs (C++ 40, Java 35, JavaScript 28) are refused by admission
    (`context_language_not_in_pool`).
  - Only 2 of test-g6's 6 code prompts are admissible, too few to read.

**Answer:** no. Lappi's defect-class skill detects stub and logic mutations planted into real
commits. It does not tell RM-Bench's correct programs from their model-written broken twins in
Go, Python or Rust. The intervals are the analysis's bootstrap (2,000 draws, seed 20261006) over
pairs. Not a gate.

## 8. Other notes from this lane

- **GH200:** a fourth MLresearch hold. The human answered "Yes, hand it over"; the holder started
  holding gpu.lock at 14:03:12Z, and Lappi has no GH200 work.
  `build/gh200-mlr/{state_check.sh,archive_cycle.sh,launch_hold4.sh}`.
- **Needle misses:** v5 seeds 0 and 2 miss needles at 60–100% depth, the end of the context.
  GAP-V5-NEEDLE-END-OF-CONTEXT-MISSES-SEEDS-0-2-HAVE-NO-ID-2026-10-06.
  - The training-audit session's own diagnosis (its measurement, not re-derived here) is in
    `HANDOFF/needle-span-mac-2026-10-06.md` (commit ee1dfab): 39 misses, all 1–2 hunks early, Swift
    and TypeScript only.
  - Its span-candidate-state hypothesis is GAP-SPAN-CANDIDATE-STATE-HAS-NOT-READ-ITS-LINE-2026-10-06
    (code claims verified here).
- **HANDOFF/v6-plan-proposal-2026-10-06.md:** its needle line was corrected. Seeds 1, 3 and 4 clear
  0.95; seeds 0 and 2 do not.

## Open

- GAP-MAIN-UNCOMMITTED-BATCHED-DECODE-FAILS-ITS-DIGEST-PIN-2026-10-06 (the human / whoever owns it).
- GAP-CARGO-TARGET-DIR-DELETED-MID-SUITE-2026-10-06 (deleter unidentified).
- GAP-WORKSPACE-IS-NOT-RUSTFMT-CLEAN-2026-10-06 (a one-time reformat is the human's call).
- RM-Bench code (card `lappi-rmbench-code-prefers-rejected`): out of task.
  - Remaining criteria: measure the style prior on more than 6 groups before any public claim; any
    training change is a pre-registered campaign and the human's decision.
- decider-2b, same base and size, leads Lappi by +0.110 [+0.071, +0.157] on test-g6.
  - What it does differently (its training mix, its calibration-aware RL) is the comparison
    worth reading before v6's data plan.
  - Whether these benchmarks overlap its training is [unverified].
- The published JevArena page is https://claude.ai/artifact/Hdq9CDPFQrqN6DJLnkKNEd, version 3 (private).
  - It adds decider-2b, the Gemma reasoning pilot and the probe, and corrects the RM-Bench code
    framing.
  - Rebuild it with `python3 build/jevarena-artifact/build_page.py` after `compare_all.sh`.
- Worktrees:
  - `build/merge-int-wt` was removed at the lane's end (clean, at 1feae5a;
    `build/merge-2026-10-06/remove_merge_int_wt.sh`).
  - `~/qd-campaign/lappi-bench-2026-10-06/main-wt` (1feae5a, clean) is kept: the `jevarena-lappi`
    bench crate's path dependencies point into it. If it is removed, recreate it before building
    that crate: `git worktree add --detach ~/qd-campaign/lappi-bench-2026-10-06/main-wt 1feae5a`.
- GAP-16-OPTION-CALIBRATION-FAILS-AND-HAS-NO-ID-2026-10-06,
  GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06 and
  GAP-DEVMAP-INDEX-MALFORMED-2026-10-06 (appended this lane).

## First command for the next lane

    bash build/merge-2026-10-06/suite.sh <a clean worktree of main> "build cargo gpu py"

(under `tools/mac_heavy.sh`; one target dir per worktree; qd-prep is read from
`build/merge-2026-10-06/bin/<worktree>/`).
