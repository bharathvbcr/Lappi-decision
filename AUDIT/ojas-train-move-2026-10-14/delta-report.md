# Delta report: moving qd-train's trainer toward `ojas-train` (2026-10-14)

Read-only research lane. Nothing was moved, bumped, built or committed; this file is the only write.
Labels: [V] read or run this session (cite given), [I] inferred, [U] unverified.
This report is for the owner to read BEFORE any dependency moves (Fable ruling item 6,
`AUDIT/board-sweep-2026-10-13/fable-owner-decisions-ruling.md:14`).

## 0. Heads and tool state

- Lappi `main` = `0418a25a8b3ba53de1cbd1d7746e02317824e8a9` [V `git log -1`]. The wave-3 commits `39654ea`
  and `bc1471e` (run()-level tests, per-entry lr_scale test) are on main [V `git branch -a --contains 39654ea` -> `main`].
- ojas `main` = `bc498c4375c2e4753d784e6e9fc267a8d18b038a` [V `/Users/bharath/Code/research/ojas/.git/refs/heads/main`];
  `gitpulse/frozen-params-lora` = `67379824d012127b9155ba2144f96cd9014291c3` [V ref file]. No `packed-refs` file. Other branches
  exist for ci-main-green, docs-pipeline, simd-band-soundness (refs not read).
- GitPulse Insights, Lappi: all facets `ok:true`. 2 live claude sessions in the shared main worktree (pids 12401 busy, 47940 idle),
  main dirty with 3 files (`gaps.jsonl`, `BRANCH_REPORT.md`, `BUILD_REPORT.md`). Two other worktrees
  (`wave4-l1-own-swift-clean-rows`, `wave1-l3-armb-corpora`) overlap only on `data/pool/*`, nothing under `crates/qd-train*` [V insights].
- GitPulse Insights, ojas: all facets `ok:true`, 5 worktrees, 0 dirty, 0 overlaps, 1 live claude session (pid 5279, busy) in ojas main [V].
- DevMap Lappi: gen 3889, but `is_fresh:false` (`analyzer_freshness:false`, "stored extraction payload is obsolete"; `source_freshness:true`).
  The coordinator said it is fresh; the status call this session disagrees, so I treated it as stale. `devmap_impact` on
  `crates/qd-train/src/lib.rs` returned 1 edge with `walk_incomplete` (parse failures), so it could not answer dependents; dependents below are rg, not graph-confirmed.
  ojas DevMap gen 6959 is also `is_fresh:false`; I did not query it. Fallbacks are listed in the final message.

## 1. Card ft-3163ce items 4 and 9, verbatim

The card is `ft-3163ce2315f5d63c03d05de888434469` (the short id `ft-3163ce` is not accepted by `gitpulse_get_task`; the full item id is) [V].
Acceptance criteria, in order (items 4 and 9 of 9):

- Item 4: "A second epoch runs: the loop refuses a fresh call on a model that has already stepped and resuming into another epoch (crates/qd-train/src/trainer.rs:497-502, 452-456) [A]"
- Item 9: "Each fix lands during the qd-train -> ojas-train move (ojas gp-ft-trainer-loop; Fable D8, user-approved 2026-10-09) with a test that fails first"

Card log, wave 2: "Item 4 (second epoch) and item 9 (the move) are deferred to the qd-train -> ojas-train move (D8, cross-repo pin)" and
wave 3: "Items 4, 9 remain the owner's (qd-train->ojas-train move)" [V card logs].

## 2. Inventory of Lappi's trainer

### 2.1 `crates/qd-train` (library `qd_train`, 14,161 lines across qd-train + qd-train-metal src, [V wc])

Manifest: `crates/qd-train/Cargo.toml:19-26`. Deps: `qd-export` (path), `serde`, `serde_json`, `sha2`, `thiserror` (workspace).
Header comment `Cargo.toml:10-13`: trainer placement "is the human's decision"; no tessl/qd-metal dependency [V].
`lib.rs:18-38` modules (line counts [V wc]):

| Module | Lines | Role | Moves? |
| --- | ---: | --- | --- |
| `step` | 383 | `StepProvider` trait (`step.rs:210`), `ParamSpec`, `AdamWHyper`, `StepError`, `check_per_entry`, `refuse_lr_scale` | moves |
| `trainer` | 1307 | `train` (`trainer.rs:387`), `Objective`, `ConsumedBatch`, `HostParams`, `TrainConfig`, `ResumeState`, `Hooks`, `EtaRule`, `parameters_digest` | moves, with seam work (2.3) |
| `schedule` | 243 | `LrSchedule`, `ScheduleError` | moves |
| `recipe` | 344 | `OptimizerRecipe`, `LowerLayers`, `lr_scales`, `weight_decays`, `optimizer_table`, `clip_coefficient` | moves |
| `adamw` | 245 | host AdamW (`adamw_entry_step`, `adamw_update`, `Moments`) | moves |
| `mock` | 486 | `ToyProvider` CPU provider that exercises the loop | moves (it is the loop's test double) [I] |
| `run_control` | 413 | wall-clock cap, loss log, digests | shared, see 2.3 |
| `pyjson` | 438 | the bytes Python hashes | shared, see 2.3 |
| `objective` | 327 | `LetterSpanObjective` (letter + span loss), `FtBatch`, `SpanHead` trait | stays |
| `span_head` | 864 | host span head | stays |
| `supervision` | 351 | letter/span supervision port | stays |
| `shards`, `held_out`, `files`, `npy`, `npz`, `np_random` | 2383/1235/118/477/418/236 | shard reader and the rule-3 door | stay (rule 3) |
| `ft_data` | 472 | puts `Batch` and `SpanHead` behind the objective traits | stays |
| `ledger`, `export`, `tristate` | 945/467/380 | `ft` row, scorer-format export | stay (also used by qd-metal) |

### 2.2 `crates/qd-train-metal` (adapter, macOS)

`src/{args,lib,main,provider,remap,run}.rs` (430/34/52/493/104/478 lines) [V wc]. It implements `qd_train::step::StepProvider`
for ojas-qwen35's `Qwen35Step` (`provider.rs:1-3`) and drives `train()` from `run.rs`. Manifest `qd-train-metal/Cargo.toml:30-38`:
`qd-train`, `qd-export` (path); macOS-only `ojas-qwen35 = { path = "../../../ojas/ojas-qwen35" }` and
`ojas-core = { path = "../../../ojas/ojas-core" }` [V]. It "belongs with" the move only as the consumer of `ojas-train`'s trait; the
adapter itself stays in Lappi (it holds Lappi's shard open, remap, ledger row, export) [I]. Tests: `args, gpu_rung_b, gpu_tiny, names_2b,
plan_groups, recipe_label, remap, run_door, rung_b_data, run_v64, tiny_published` [V ls].

### 2.3 Who depends on it (rg; DevMap could not answer, see section 0)

| Consumer | Call sites | Uses |
| --- | --- | --- |
| `qd-train-metal` | `src/run.rs:28-48`, `src/args.rs:34-38`, `src/provider.rs:58-59`, `src/remap.rs:18-20` | export, files, ft_data, held_out, ledger, objective, recipe, run_control, schedule, shards, step, trainer, tristate |
| `qd-metal` | `Cargo.toml:58`; `src/decision.rs:51-52`, `src/parity_row.rs:23-24`; `tests/ledger_rows.rs:26,164,173,297,389,398,414,466,482,497,551,557` | only `ledger` and `tristate` (the ledger row and its chained append) |
| qd-train's own tests | `tests/{adamw_oracle,adamw_decay_sensitive,schedule_oracle,trainer_loop,optimizer_table,...}.rs` | see 4.4 |

Not in the graph: `Makefile:135` lists `qd-train qd-train-metal` in `FMT_MEMBERS` [V].

Coupling that blocks a clean cut [V imports]: `trainer.rs:41` imports `crate::pyjson`, `:43` `crate::run_control`, `:45`
`crate::shards::ConsumedPrefix`; `run_control.rs:26` and `schedule.rs:21` import `pyjson`; `objective.rs:28` imports `trainer`.
So `trainer` reaches into the shard layer (`ConsumedPrefix`) and into `pyjson`, both of which other staying modules also use.

## 3. What exists in ojas today (ojas main `bc498c4`)

- No `ojas-train` crate. Workspace members: `ojas-core, -cpu, -autograd, -oracle, -data, -io, -infer, -metal, -device, -kernels, -wgpu,
  -cuda, -hip, -capi, -gusset-engine, -simd, -qwen35, -model` [V `ojas/Cargo.toml:3-20`]. `rg ojas-train|qd-train` over
  non-`target*`, non-`arch02` Cargo.toml files: only `ojas-qwen35` matches [V rg, tracked-source scope].
- `ojas-qwen35` (`Cargo.toml`, "Whole-step training provider over canonical tessl (Metal only). Not an ojas Backend"): deps `ojas-core`, `ojas-io`;
  macOS-only `tessl = ../../tessl`, `ojas-model`; empty off macOS [V `ojas-qwen35/Cargo.toml:9-30`]. `src/step.rs`: `Qwen35Step` (`:353`),
  `forward` (`:535`), `backward(pending, external)` (`:576`), `grad_sq_norm` (`:658`), `adamw_step(hyper, plan)` (`:679`), its own `AdamWHyper`
  (`:280`) and `clip_coefficient` (`:259`). `src/groups.rs`: `OptimizerPlan`, `GroupSpec`, `LrRule`, `WdRule` (`:65-100`), i.e. per-entry lr/decay tables.
  [V rg of pub items]
- A second, overlapping optimizer-grouping implementation therefore already exists in ojas (`groups.rs`) beside Lappi's `recipe.rs`
  (`optimizer_table`). Which one is canonical after the move is open question Q3 [I].
- Training loop code in ojas today: `ojas-cpu/src/train.rs` (`GradAccumulator`, `clip_grads`, `HybridParam`), `ojas-model` nanolab `Trainer<B>`
  (per the task brief, "AdamW hard-wired, trainer.rs:792 [A]"), `arch02/metal-native/src/{optim,checkpoint,engine}.rs` [V rg file list; contents not read]. None is a
  `StepProvider`-shaped loop with bank accumulation, wall-clock cap and resume.
- LoRA: `gp-ft-frozen-params-and-lora` says "There is no LoRA code in ojas, tessl or Lappi (rg -uu) [V]" (brief, 2026-10-09). The branch
  `gitpulse/frozen-params-lora` is at `67379824`; I did not read its tree (no git against ojas).
- Plan of record: `ojas/tasks/gp-ft-trainer-loop.md` (status `ready`, owner unassigned, priority 1). Its AC1 says the new crate "holds ... the training loop,
  LrSchedule, recipe parameter groups, host AdamW and the StepProvider trait ... Lappi's qd-train depends on ojas-train and keeps only the rule-3 held-out
  door, shard reader, supervision, objective, span head, ledger and export; the moved copies are deleted from Lappi in the same change". AC2: "Before the move
  starts, GitPulse shows no live lane holding Lappi crates/qd-train". AC3: `ParamSpec` gains `trainable` and a storage dtype, `accumulate` may take several sequences.
  Last AC: "The nanolab Trainer<B> ... is left alone" [V file lines 25-35]. Depends on `gp-ft-frozen-params-and-lora`, `gp-ft-decision-head-and-data-path`,
  `gp-oom-error-class`, `gp-device-memory-probes` [V line 63].
- History: Fable 10-01 advised trainer-in-Lappi first, ojas later (`AUDIT/ojas-training-2026-10-01/fable-advice.md` Q5 placement; ask 2 option B);
  the user chose it (`HANDOFF/ojas-training-2026-10-01.md:25`); Fable D8 (2026-10-09) reversed to "move now"
  (`AUDIT/lora-2b-2026-10-09/human-decisions.md:13`) [V].

## 4. Deltas the move introduces

### 4.1 Dependency direction and pins

- Today: Lappi -> ojas, by path, only on macOS, only in `qd-train-metal` (`Cargo.toml:37-38`) [V]. `qd-train` and `qd-metal`
  (which depends on `qd-train`) are ojas-free and build on the Linux box [V `qd-train/Cargo.toml:10-13` comment].
- After: `qd-train` (the portable crate that `qd-metal` also pulls in) would depend on `ojas-train`. That puts an ojas dependency on EVERY target,
  including Linux and the public GitHub repo's clean checkout, unless the dependency is target-gated or `ojas-train` is vendored/published [I].
  Lappi has no `.github` directory [V rg error], so there is no CI to break, but `make gates` and the Linux box would [I].
- Pin: no committed Lappi file names an ojas sha (`GAP-LAPPI-PINS-NO-OJAS-SHA-2026-10-13`, open: "No committed Lappi file names an ojas sha";
  the ojas commit is read at run time into row descriptions, `provider.rs:105-107`, `:158`) [V gaps.jsonl:1648]. Path deps resolve to whatever is checked out
  at `../../../ojas`, including the ojas worktree symlink convention (`qd-train-metal/Cargo.toml:30-36` comment). A move without a pin makes
  every Lappi ledger row's trainer behaviour depend on an unrecorded ojas state [I].
- Cargo changes needed [I]: add `ojas-train = { path = "../../../ojas/ojas-train" }` to `qd-train` (and a `[patch]`/lock note if both repos
  are in one build, because `links = "tessl"` forces one tessl source, `Cargo.toml:30-36`); in ojas add `"ojas-train"` to `members`
  (`ojas/Cargo.toml:3-20`). The pin is a sha file (for example `ojas.lock` or a test that compares `ojas HEAD` to it) because path deps cannot carry one.
  Per global rules, bumping a pin needs the owner's yes after this report.
- Reverse edge: ojas must not depend on Lappi. `provider.rs` doc says "so ojas never depends on Lappi" (`Cargo.toml:25-26` comment) [V]; the plan keeps that.

### 4.2 Toolchain and dependency mismatches [V Cargo.toml lines]

| | Lappi | ojas |
| --- | --- | --- |
| edition | 2024 (`Cargo.toml:15`) | 2021 (`ojas/Cargo.toml:24`) |
| rust-version | 1.98 | 1.97 |
| `serde`/`serde_json`/`sha2`/`thiserror` | workspace deps (`sha2 = 0.11`, `thiserror = 2`) | none in the ojas workspace members (only in `gemma-metal` and `arch02/*`, which are not members) |

- Moved code is edition-2024 and the trainer needs `serde_json`, `sha2`, `thiserror` (`qd-train/Cargo.toml:23-26`). Landing it in a 2021 / 1.97
  workspace means either a per-crate edition override or code changes, and three new third-party dependencies in ojas.
  CLAUDE.md rule: "No new dependency without asking" [V]. This is a decision for the owner (Q4).
- `ojas-train` must also stay macOS-free and tessl-free so Linux builds (`ojas-qwen35` is empty off macOS, `lib.rs` cfg) [V].

### 4.3 What must stay in Lappi (rule 3 and ownership)

- The held-out door and shard reader: `held_out.rs`, `shards.rs`, `files.rs`, `npy.rs`, `npz.rs`. CLAUDE.md rule 3: a path check in `qd-train`
  refuses held-out data and removing it "is a refused change". Moving it to ojas would put Lappi's rule-3 enforcement in a repo whose reviewers
  do not carry rule 3. The door test (`tests/door.rs`, `tests/caller_records_are_held_out.rs`) is the fail-first proof and must keep running in Lappi [V rule text; I for reviewer point].
- Supervision (`supervision.rs`), objective (`objective.rs`), span head (`span_head.rs`, `ft_data.rs`): Lappi-specific loss semantics (letter + span, `noul_row` layout).
- `ledger`, `export`, `tristate`: used by `qd-metal` (`decision.rs:51-52`, `parity_row.rs:23-24`) and by rule 8 (`quick` rows); the Python-wire formats are Lappi's.
- Parity oracles that pin Lappi's recipe to Python (`trainer-oracle.json`, `optimizer-table-2b.json`, `adamw-decay-sensitive`) are Lappi facts; see 4.4.

### 4.4 Tests

Likely move with the code [I, from imports at `tests/*.rs:12-34`]: `schedule_oracle.rs` (imports only `schedule`), `adamw_oracle.rs`
(`adamw`, `recipe`, `step`), `adamw_decay_sensitive.rs` (`adamw`, `recipe`, `step`), `trainer_loop.rs` (`mock`, `objective`, `trainer`, `run_control`, ...).
Not separable without a seam: `trainer_loop.rs` and `optimizer_table.rs` import `objective`, `span_head`, `ft_data`, `pyjson` (Lappi-owned),
so they either stay in Lappi against the moved crate or are split into generic-loop tests (move) and Lappi-objective tests (stay).
The plan brief says "the schedule oracle test moves too" (AC1) [V]. Fixtures that go with them: `trainer-oracle.json`, `adamw-decay-sensitive`, `tiny-published`
if they are generated by Lappi's `tools/qd_train_oracle_*.py` oracles, which stay Python-in-Lappi [I].
GPU tests in `qd-train-metal/tests/gpu_tiny.rs` and `run_v64` stay in Lappi but would need a pinned ojas build (they build `ojas-qwen35` by path) [V Cargo.toml:37-38].
Card item 3's kill -9 resume tests and the wave-3 `run()`-level tests are on main and exercise `train()`; after the move they run against `ojas-train`'s `train()` [I].

### 4.5 Seams to cut before a file can move [V imports; I conclusions]

1. `trainer.rs:45` `shards::ConsumedPrefix` -> the loop should depend on a trait (`ConsumedBatch` already exists, `trainer.rs:84`), not a Lappi type.
2. `pyjson` (`trainer.rs:41`, `run_control.rs:26`, `schedule.rs:21`, plus `ledger`/`shards`): either copy into ojas-train (drift against the Python-hash oracle) or
   move it and make Lappi depend on it for the ledger too. One canonical owner is required; copying is the failure mode the global rules warn about.
3. `run_control` (cap, loss log, digests) is used by both the loop and Lappi's `run.rs`/`args.rs` -> move with the loop and re-export, or split.
4. `step::ParamSpec`/`AdamWHyper` vs ojas-qwen35's own `AdamWHyper` and `groups.rs` (3 overlapping types): converge in the same change or the move doubles the surface.

## 5. Risks and ordered step plan

Risks:
- R1 [I] Linux/clean-checkout breakage: `qd-train` gains a path dependency on a repo that is not in the Lappi tree.
- R2 [I] Unpinned ojas: parity numbers in ledger rows cannot be reproduced without an ojas sha (the open gap).
- R3 [V] Live peers: Lappi main has 2 live sessions; ojas main has 1 busy session. `gp-ft-trainer-loop` AC2 requires no lane holding `crates/qd-train`; today
  no other worktree touches it, but the Lappi main worktree sessions are not excluded by GitPulse's collision facet (CLAUDE.md rule 3 on `ListAgents`) [V insights + CLAUDE.md]. I did not run `ListAgents`; run it before step 3.
- R4 [I] Silent semantics drift: bit-exact claims (resume byte-equality, schedule f64 bit-equality, decay-sensitive golden) must be re-run on the moved code.
- R5 [I] Edition/dep mismatch (4.2) forces either owner-approved new deps or a rewrite of moved code.
- R6 [U] Whether ojas is private. `publish = false` is set, the repo field points to GitHub (`ojas/Cargo.toml:25-26,37`); I did not verify visibility. If private, the public Lappi repo would depend on a private path.

Steps, each with its gate (a step is done only when the gate output is read):

1. Record the ojas pin (the open gap). Write a committed file naming the ojas sha Lappi's last green GPU suite built against (candidate: ojas `bc498c4`, but that is not
   known to be the one the last GPU suite used [U]). Gate: a Lappi test that fails when `OJAS_DIR` HEAD differs from the file, reported as NotRun if ojas is absent, never as pass.
   Closes `GAP-LAPPI-PINS-NO-OJAS-SHA-2026-10-13`.
2. Owner decisions Q1-Q6 below. Gate: written answers in `AUDIT/ojas-train-move-2026-10-14/`.
3. Quiesce: `ListAgents` in both repos, `gitpulse_collision_risk` on `crates/qd-train/**` and `ojas/ojas-train/**`; message peers. Gate: no live lane holds either path.
4. In Lappi first, cut seams 4.5(1)-(3) in place (no cross-repo change yet): `trainer` stops importing `shards`; one `pyjson` owner. Gate: all `qd-train` and `qd-train-metal` tests
   unchanged and green (`cargo test -p qd-train -p qd-train-metal`, GPU tests reported as not run if not run), plus clippy `-D warnings` and `make gates` fmt gate.
5. In ojas (a session rooted there): add `ojas-train` with moved modules, git history kept per AC1, edition/deps per Q4, no tessl and no macOS gate. Gate: its moved tests
   pass on Linux (`cargo test -p ojas-train`), including the schedule and AdamW oracles with their fixtures.
6. Lappi switches `qd-train` to depend on `ojas-train`, deletes the moved copies in the same change (replace, do not accumulate). Gate: step-4 test set green, `tests/door.rs` and
   `caller_records_are_held_out.rs` green, `rg` shows no leftover copies of `adamw`/`schedule`/`recipe`/`step`/`trainer` in Lappi, and the pin file updated by the owner-approved bump.
7. Card item 4 (second epoch) on the moved loop, with a test that fails first (the refusals at the cited `trainer.rs:497-502, 452-456` are the pre-fix behaviour [A on the card, not re-read here]).
   Gate: red against pre-fix code, green after.
8. GPU re-verification under `tools/mac_heavy.sh`: the `--ignored` GPU suites (`gpu_tiny`, kill -9 resume, `gpu_run_*`). Gate: the same results as the wave-3 rows, bytes equal for resume. Row cites its ledger id; otherwise "not run".
9. Close card items 4 and 9 and the pin gap; write `HANDOFF/ojas-train-move-<date>.md` (rule 10).

## 6. Questions for the owner

- Q1. Target: does `qd-train` depend on `ojas-train` on every target (breaking the Linux/clean-checkout build), or only on macOS behind a gate with the Linux box keeping a copy? The ruling and brief assume the first; this report found the Linux build is the cost.
- Q2. Pin mechanism: a committed sha file plus a checking test, a git dependency, or a vendored snapshot? (Path deps cannot carry a sha.) And which sha is the baseline?
- Q3. Canonical owner of per-entry optimizer tables: Lappi's `recipe::optimizer_table` (moves) or ojas-qwen35's `groups.rs::OptimizerPlan`, or converge them in the move?
- Q4. Three new third-party deps (`serde_json`, `sha2`, `thiserror`) and the edition 2024 / rust 1.98 vs 2021 / 1.97 mismatch in the ojas workspace: approve the deps, or write the crate dependency-free (more work, matches the hand-rolled style Lappi already uses)?
- Q5. `pyjson` and `run_control`: where does each live once, and does Lappi's ledger depend on ojas-train for it?
- Q6. Is ojas private? If so, does the public Lappi repo take a path/git dependency on it, and who owns the ojas-side lane (ojas has a busy session now; the plan's `owner` is "unassigned")?
