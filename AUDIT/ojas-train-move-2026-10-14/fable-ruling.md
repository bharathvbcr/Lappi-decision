# Fable ruling: the qd-train -> ojas-train move (2026-10-14)

The user said "Continue, choose best"; these are the owner decisions on `delta-report.md` Q1-Q6, the memory
criterion, and the order of work. Read-only lane: this file is the only write. Labels: [V] read or run this
session, [I] inferred, [U] unverified. Lappi HEAD is now `262eeab` [V `git log -1`], one past the report's
`0418a25`; ojas main is still `bc498c4`, main worktree clean [V ref file; `gitpulse_insights`].

Tooling: `gitpulse_insights` on both repos, every facet `ok:true` [V]: Lappi has two live claude sessions here
(12401 busy, 47940 idle), worktree overlaps only under `data/pool/*`; ojas has one busy session (5279) in main and
3 dirty files in `ojas-wt-lora` (`gitpulse/frozen-params-lora`). `devmap_status` on Lappi: gen 3889,
`is_fresh:false`, analyzer and source freshness both false [V], which is
`GAP-DEVMAP-MCP-AND-CLI-DISAGREE-ON-ANALYZER-FRESHNESS-2026-10-14` (the CLI rejects `--repo`; not run bare).
`devmap_search ConsumedPrefix` answered
(`crates/qd-train/src/shards.rs:2268`) with `walk_incomplete` [V]: partial confirmation only. Everything else is
rg/Read, not graph-confirmed. No `ListAgents` tool is exposed to this lane; the Lappi session runs it first.

## Q1. Target: every target, ungated

**Decision.** `qd-train` depends on `ojas-train` by path on every target; no macOS gate, no Linux copy.
**Reason.** `ojas-train` is portable by construction (no tessl, no Metal: AC1), so the only cost is that a Lappi
checkout needs `../../../ojas` beside it on Linux too. That cost exists already: `qd-metal` is a workspace member
with an ungated `tessl = { path = "../../../tessl" }` (`crates/qd-metal/Cargo.toml:60` [V]), so a lone clean
checkout does not build today either, and qd-train on Linux is only ever a cross-check
(`HANDOFF/L-trainer-2026-10-01.md:49` [V]). A gate with a Linux copy is two trainers, the defect every audit
named. Gate (step 6): `cargo check -p qd-train --target aarch64-unknown-linux-gnu` green with ojas checked out, and
`python/tests/test_cargo_workspace_exclude.py` green (same symlink path `ojas-qwen35` uses,
`crates/qd-train-metal/Cargo.toml:30-38` [V]).

## Q2. Pin: a committed sha file plus a tri-state gate; baseline `bc498c4`

**Decision.** `upstream.lock.json` at the Lappi root (Jarvis's file name; one owner for every sibling pin later)
names the ojas sha. Checked three ways: (1) a `pin` entry in `make gates`, implemented in `qd-preflight` (the
tri-state owner, `crates/qd-preflight/src/tristate.rs` [V]), resolving `.git/HEAD` -> ref / `packed-refs` /
`gitdir:` by plain file reads, exit 0 / 1 stale / `NOT_RUN`=3 when the sibling is absent (`Makefile:84,252-253`
already carry that code [V]); (2) a unit test of the comparison that fails first; (3) `qd-train-metal` refuses a
run when `git_commit(OJAS_DIR)` (`provider.rs:70,158` [V]) differs from the lock unless `--unpinned-ojas` is given,
and then the row is `quick` with reason `unpinned ojas` (rule 8: stricter, never looser). Not a git dependency:
`ojas-qwen35`/`ojas-core` are path deps and tessl has `links = "tessl"`, so a git-sourced `ojas-train` puts a second
`ojas-core` beside the path one [I]. Not a vendored snapshot: a copy drifts.
**What is pinned.** Lappi -> ojas commit, consumed by `qd-train-metal`'s path deps today and by `qd-train`'s
`ojas-train` after step 6. Baseline `bc498c4375c2e4753d784e6e9fc267a8d18b038a` (ojas main HEAD, clean [V]). No
committed Lappi ledger row names an ojas sha (`rg ojas ledger/`: three files, no commit [V]), so "the sha the last
green GPU suite used" is unknowable [U]; the step-8 rerun under the lock makes the baseline a verified one. Closes
`GAP-LAPPI-PINS-NO-OJAS-SHA-2026-10-13` when the gate prints `pin PASS`. The later bump is a human-approved commit.

## Q3. Per-entry optimizer tables: converge on `OptimizerPlan`, owned by ojas-train

**Decision.** ojas-qwen35's `groups.rs` (`Select`, `LrRule`, `WdRule`, `GroupSpec`, `OptimizerPlan`,
`ojas-qwen35/src/groups.rs:65-100` [V]) moves into `ojas-train` and is the one per-entry table. Lappi's `recipe.rs`
moves per AC1 rebuilt as a builder (`OptimizerRecipe`/`LowerLayers` -> `GroupSpec`); `optimizer_table` and
`EntryHyper` (`adamw.rs:35` [V]) fold into `OptimizerPlan`, which the host `adamw_entry_step` reads. One
`AdamWHyper` and one `clip_coefficient` in `ojas-train`; ojas-qwen35's own (`step.rs:259,280` [V]) and the
provider's translation (`provider.rs:54` [V]) are deleted in the same change.
**Reason.** Three overlapping types exist because the provider predates Lappi's trainer; the rule-based plan is the
model-generic one and already what the provider hands tessl. Lappi keeps the Python-pinned 2B oracle
(`tests/optimizer_table.rs`, `optimizer-table-2b.json`) as a Lappi test against the moved API, and the Amendment
2(ii) recording-provider test. Gate: that oracle, `adamw_decay_sensitive`, `adamw_oracle` green, goldens unchanged.

## Q4. Deps and edition: no new third-party crate; one question to the human

**Decision.** `ojas-train` sets `edition = "2024"` per crate and `rust-version = "1.97"` (ojas floor,
`ojas/Cargo.toml:24-27` [V]); any 1.98-only construct is rewritten [U whether one exists; gate: `cargo check` on
the floor]. Third-party: none approved. `thiserror`: derives in `step`, `trainer:58`, `schedule`,
`run_control:34`, `pyjson:29` [V] become hand-written impls, in Lappi, before the move. `serde_json`: `pyjson`,
`trainer`, `schedule`, `run_control` use `serde_json::Value` [V]; ojas-io already has a strict dependency-free
reader with `JsonValue`/`JsonNumber`/`JsonLimits` (`ojas-io/src/json.rs:1-25`, `lib.rs:20,34` [V]; ojas-io depends
only on ojas-core [V]), so `pyjson` is re-targeted onto it in the ojas step. `sha2`: ojas has no SHA-256
(`ojas-data/src/rng.rs:5-6` says so; no `fn sha256` in ojas-core/-data [V]) and `trainer:36`,
`run_control:22,261,302,329` need one [V]. The dependency-free ruling is a `sha256.rs` in `ojas-train` with FIPS
180-4 vectors, a parity test on the committed loss-log digest fixtures, and a committed benchmark, because
`parameters_digest` hashes the 7.5 GB table and scalar SHA-256 is several times slower than sha2's aarch64 path
[I]. The better long-term answer is `sha2 = "0.11"` in ojas: a new dependency, put to the human below. If
unanswered when the ojas lane starts, hand-rolled it is.

## Q5. `pyjson` and `run_control`: both move; Lappi imports them from ojas-train

**Decision.** `run_control` and `pyjson` move with the loop. The generic half of `ConsumedPrefix` (`fold`,
`hexdigest`, `n_folded`) moves into `run_control`, which already documents it (`run_control.rs:18` [V]);
`fold_batch` stays in Lappi `shards.rs`. That removes `trainer.rs:45`'s `shards` import. Lappi's `held_out`,
`shards`, `export`, `ledger`, `tristate` import `ojas_train::pyjson` (`held_out.rs:57`, `shards.rs:56`,
`export.rs:48`, `ledger.rs:45`, `tristate.rs:343` [V]).
**Reason.** One owner; the loss-log and consumed digests are the loop's own checkpoint format, and one digest
formula serves torch and Rust (`fable-rung-d-resize.md:55`). Lappi's ledger then depends on ojas-train for its
bytes: the same edge qd-train already takes. Oracle tests whose generator is Lappi Python (`pyjson_oracle`,
`schedule_oracle`, `adamw_*`) move with their committed fixtures; generators stay in `tools/`, each fixture
manifest names the generating Lappi commit, so regeneration is a Lappi action handed across.

## Q6. ojas is public; path dep; the ojas lane is the `gp-ft-trainer-loop` card

**Decision.** ojas is PUBLIC (`gh repo view bharathvbcr/ojas`: `isPrivate:false` [V]; Lappi likewise), so R6
closes: the public Lappi repo names a public sibling by path and sha. The ojas lane is whichever session, rooted
in ojas, takes `gp-ft-trainer-loop` (status `ready`, owner unassigned [V]) from the board; the brief is appended to
that card. Pid 5279 holds nothing uncommitted in main [V] but its work is unknown to me; the Lappi lead resolves
that with `ListAgents` and `SendMessage` before posting. The move (AC1-AC3) does not depend on the LoRA branch; the
Tape `StepProvider` (AC4 on) does and waits for `gitpulse/frozen-params-lora`.

## The memory criterion: 46.39 GB against ">= 47 GB"

`HANDOFF/ojas-training-2026-10-01.md:200`: working set 0.9 x recommended >= 47 GB, "if not, the host-AdamW
contingency is costed first, and the moments are never bf16". Measured: recommended 51,539,607,552 B, wired budget
46,385,646,796 B (`ojas-qwen35-gpu-diag-none.log:21` [V]); `GAP-OJAS-RUNGD-WORKING-SET-UNREAD-2026-10-01` is closed
on that read [V last line]. Rule 2: the number stands. What the miss means: (1) the full-FT 2B rung-(d) Mac window
as written does not start; its own text names the consequence and I apply it: the host-AdamW contingency is costed
before any such window, bf16 moments stay refused. (2) The "47" was an inferred sum (30.78 GB step + 16 GB moments,
`fable-rung-d-resize.md:20` [V]); replacing it with a measured peak-with-moments re-specifies a pre-registered
precondition and is the human's. (3) The costing must count that on unified memory the moments leave the Metal
working set but not the 64 GB: the provider already holds a 7.5 GB host f32 table (`provider.rs:97-100` [V]), so
the physical sum is ~30.8 + 7.5 + 16 GB plus the OS [I]. (4) The log shows 52.4 GB allocated past the wired budget
with `poisoned false` (`:23` [V]): the budget is a wiring ceiling, a fact for the costing, not a reason to proceed.
(5) The LoRA-2B track (D8/D3: frozen bf16 base, trainable LoRA + D17) has its own approved memory rows and is not
bound by this criterion; its pre-flight is the brief's "Metal-aware memory pre-flight" AC.
`GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01` stays open until a step with moments is measured.

## Lappi session does now (in order; a step is done when its gate output is read)

1. **Quiesce.** `ListAgents`; `SendMessage` to the other Lappi session claiming `crates/qd-train/**`,
   `crates/qd-train-metal/**`, `crates/qd-preflight/**`, `Makefile`, `upstream.lock.json`. Gate: the peer's claims
   show no overlap; `gitpulse_collision_risk` on those paths empty (today it is [V]). Append a `GAP-` record if
   `ListAgents` cannot answer.
2. **Pin (report step 1).** `upstream.lock.json` with `bc498c4`; the `qd-preflight` `pin` check; the `make gates`
   entry; the provider refusal / `quick` marking. Fail-first tests: the comparison, the refusal with a fake
   `OJAS_DIR`, the NotRun case. Gate: `make gates` prints `pin PASS` here and `NotRun` (exit 3) in a temp tree
   without ojas; append the closure of `GAP-LAPPI-PINS-NO-OJAS-SHA-2026-10-13` (O_APPEND + fsync, one line).
3. **Card item 4, second epoch, in Lappi before the move.** The refusals are `trainer.rs:480-483` (epoch
   mismatch) and `:529-531` (fresh run on a stepped provider) [V]; the card's `497-502, 452-456` are stale
   (`:497-502` is the step-count check, `:452-456` the lr_scale refusal [V]). Fail-first test: finish epoch 0, call
   for epoch 1 from the final checkpoint; refused today, afterwards it continues with the step count carried, the
   consumed digest chained across epochs, the loss log monotone. Gate: red then green; the kill -9 resume
   byte-equality tests unchanged. Done here so the files move with the fix in their history.
4. **Seams in place (report step 4).** (a) generic `ConsumedPrefix` into `run_control`, `fold_batch` stays;
   (b) hand-written error impls replace `thiserror` in the moving set; (c) the moving set
   `{step, trainer, schedule, recipe, adamw, mock, run_control, pyjson}` imports nothing from `shards`, `objective`,
   `span_head`, `ft_data`, `held_out`, `ledger`, `export`. Gate: `cargo test -p qd-train -p qd-train-metal` green
   with the `--ignored` GPU suites reported not run, clippy `-D warnings`, `make gates` PASS, and
   `rg 'crate::(shards|objective|span_head|ft_data|held_out|ledger|export)'` over the moving set empty.
   (`serde_json` -> `JsonValue` and SHA-256 are ojas-side edits, Q4.)
5. **Hand-off.** Commit, push main (the 10-13 push ruling stands), write `HANDOFF/ojas-train-move-2026-10-14.md`
   (rule 10) naming the start sha and the ledger rows of steps 2-4, post the brief to `gp-ft-trainer-loop`.
6. **Contingency costing (doc only).** `AUDIT/ojas-train-move-2026-10-14/host-adamw-contingency.md`: host-moment
   design, physical-memory sum, s/step estimate, what `adamw.rs` already covers. Gate: every number cites the diag
   or bench log; no run claimed.

## Handed to ojas (the brief; a session rooted in ojas, after step 5)

- Start from Lappi `<step-5 sha>`; `ListAgents` in ojas first; agree the GPU with pid 5279's owner.
- Phase A, no LoRA dependency: create `ojas-train` (`edition = "2024"`, `rust-version = "1.97"`, deps `ojas-core`,
  `ojas-io` only), add to `members` (`ojas/Cargo.toml:3-20`); move the eight modules and their tests with history
  (fetch Lappi locally, subtree/filter-repo; gate: `git log --follow ojas-train/src/trainer.rs` shows Lappi's
  commits); `pyjson` onto `ojas_io::JsonValue`; SHA-256 per Q4; `groups.rs` into ojas-train and the Q3 deletions
  in `ojas-qwen35`; `ParamSpec` gains `trainable` and storage dtype (AC3), provider behaviour unchanged. Gate:
  `cargo test -p ojas-train` green on a non-macOS target (podman linux/arm64 or `--target` check), schedule and
  AdamW oracles bit-identical to their fixtures, `ojas-qwen35` tests green, no tessl or `cfg(target_os)` in
  `ojas-train`.
- Phase B (Tape `StepProvider`, all-or-nothing Metal AdamW, checkpoint, memory pre-flight) waits for
  `gitpulse/frozen-params-lora`; not part of this hand-off.
- Report back the ojas sha, the test output and the Linux evidence; Lappi step 6 cannot start without them.

## Waits / still needs the human

- **`sha2 = "0.11"` in ojas** (Q4): yes or no. No other dependency is asked for.
- **The pin bump** `bc498c4` -> the ojas-train sha after phase A: delta reported with phase A's gate output, then
  the human's yes; only then Lappi step 6 (switch `qd-train`, delete the moved copies, door tests green, `rg` empty
  for leftovers), step 8's GPU rerun under `tools/mac_heavy.sh` (single GPU, $0: no rule-4 yes), step 9's closures.
- **The 47 GB precondition** for a full-FT 2B Mac window: cost-and-run the host-AdamW contingency (step 6 doc
  first) or re-specify it as a measured peak-with-moments; a lane may not do the second (rule 2).
