# Delta report: ojas pin 5b67cdd -> 04277f8 (2026-10-17)

The owner said "merge all" on 2026-10-17, after being asked whether to merge ojas `gp/lora-step-rows`, re-bump
Lappi's ojas pin with a delta report, and write the amendment-1 step count. The previous bump is Lappi commit
6d7c807 (132b46e -> 5b67cdd, same day), whose shape this one follows.

Labels: **[V]** read or run this session, **[I]** inferred. Heads: ojas local main
`04277f8fc722f02dbbbb7ad8cbe93d540876ba7c` [V `git rev-parse`]; ojas `origin/main` is still
`bc498c4375c2e4753d784e6e9fc267a8d18b038a` [V `git rev-parse origin/main`, local ref, not re-fetched], so, as with
6d7c807, the lock's `url` does not serve this revision until ojas main is pushed.

## 1. The range

`git -C /Users/bharath/Code/research/ojas log --oneline 5b67cdd..04277f8`: **9 commits** (6 non-merge, 3 merges)
[V]. 5b67cdd is an ancestor of 04277f8 [V].

| Commit | What |
|---|---|
| 7ead208 | docs(tasks): sync fine-tune briefs to the board after Fable's 2026-10-17 ruling |
| b68971c | docs(tasks): drop the superseded heads.rs from decision-head's planned_files |
| 881888e | merge gp/ft-audit2-brief-sync |
| 06fc333 | ojas-wgpu, ojas-metal: failing-first tests for scratch stacking (ft-81e49280) |
| cc9f5ad | ojas-wgpu: mark the two failing-first scratch tests ignored until ft-81e49280's fix |
| 67dc393 | merge gp/w11-scratch (session ojas-1c) |
| 0f13b37 | bench: LoRA step rows (tape-lora arm) at 0.8B-Base and 2B-Base, T = 2,048 and 9,638 |
| 4da214b | merge gp/lora-step-rows |
| 04277f8 | site: regenerate bench plots and reference pages (ci:local `plots -check` failed on 4da214b; caught by session ojas-1c) |

## 2. What Lappi builds against

`git diff --name-only 5b67cdd..04277f8` outside `tasks/` and `bench/` [V]:

- `ojas-qwen35/tests/common/mod.rs`, `ojas-qwen35/tests/gpu_memory_rows.rs` (test-only: the tape-lora arm,
  `single_weights_file`, a `proc_pid_rusage` footprint reader)
- `ojas-metal/tests/scratch_stacking.rs`, `ojas-wgpu/tests/accumulate_grad.rs`, `ojas-wgpu/tests/dispatch_cost.rs`
  (test-only)
- `docs/bench-plots.md`, `site/` (generated docs)

`git diff --name-only 5b67cdd..04277f8 -- '*/src/*' '*/Cargo.toml' Cargo.toml Cargo.lock` is empty [V]: no library
source, manifest or lockfile changed. So no item Lappi imports from `ojas-core` or `ojas-qwen35` (the two path
dependencies of `crates/qd-train-metal`) changed, and Lappi's `Cargo.lock` is not expected to change [I; checked by
the verification below].

## 3. What the range adds that Lappi reads

`bench/results/2026-10-17-lora-step/rows.jsonl` is the measured LoRA tokens/s row that amendment 1's
`mac_steps_rule` names (ft-2bbced18 criterion 7). Line 4: 2B-Base, T = 9,638, checkpointing on, 261.607 tok/s,
step peak 13,095,419,904 B device, 14,314,148,328 B host footprint max, status ok [V]. Line 3: T = 2,048,
335.915 tok/s [V].

## 4. Verification

**Not yet run** when this report was committed (2026-10-17, session lappi-decision-34). `make pin`,
`python/tests/test_pin_gate.py` and `cargo test -p qd-train-metal` are queued under `tools/mac_heavy.sh`
(label `pin-bump-verify`) behind another session's heavy job; their output is added in a follow-up commit on this
branch. Until then the pin is unverified here.
