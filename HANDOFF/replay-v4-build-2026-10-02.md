# L-replay phase 2: the CLEAN v4 replay set for J6(a), built, attested and on the box (2026-10-02)

This is lane L-replay's second phase. It carries out option (a) of the phase-1 plan,
`HANDOFF/replay-v4-plan-2026-10-02.md` (on main): exclude the replay-drawn hits only.

**Outcome:** build 2's replay slice is CLEAN.

- 3,383 rows: v4's 3,568 replay-drawn MMLU/CSQA rows minus the 185 that overlap val.
- Gold train equal to a plain `--replay-shards` build's.
- val, held-out and remap unchanged from F's v4.
- On the box at `/home/ubuntu/phase4-v4-replay-2026-10-02`, accepted by `real_ft_run`'s own
  prelude at `a502670` on the Mac and on the box.

**What this phase did not do:**

- Ran no GPU work, and trained nothing.
- Wrote nothing on the box except:
  - the rsynced set;
  - `/home/ubuntu/scratch/l-replay-prelude/` (one log, two empty directories).
- Did not touch the j6a waiter, the pre-registration or the checker.
  - The lead filled the six amendments at `0f9c421` and the four waiter pins at `6168513`.
  - Both were compared here against the measured values: equal (verified, `git show main:`).

Labels: **verified** (run or read here, with the evidence named), **inferred** (follows from
code that was read but not run), **unverified**.

## The six amendment values (verified)

They are in the key order of `HANDOFF/j6a-rule-2026-10-02.md`'s `amendments` block (on main).
main's `campaign/j6a-preregistered.json` carries exactly these values.

| Key | Value | Source |
| --- | --- | --- |
| `data_snapshot_hash` | `895d393b632aa417250578b04e22a559985f5145a98ad59c81907a75a6f7097a` | build 2's `data/pool/train.json`, and its row's `protocol.data_snapshot_hash` |
| `shard_hash` | `be617e7e3261b6e96a9f5eeb160e125066234548643ae29558a9a12728f94ff1` | build 2's `shards/train/header.json` |
| `replay_shard_hash` | `290f69a48a44f15018dc8a52deb4655502182cfa9d14cf84b4f00cd4cddb7cea` | build 2's `shards/replay/header.json` |
| `replay_attestation_sha256` | `dc69ffce43b5af2a62943603d27052ea16011d191be3de876eec9d5e9b2b6c94` | `shasum -a 256` of decontam 2's attestation, the same on the box |
| `h` | `185` | `replay_rows_hit` in decontam 1's hit list; equals its distinct hit rows |
| `hit_list_sha256` | `8cfcbccbf0ab4dd7b02515b7d37f93f46b4db149a4036ecc60b319c29154e87f` | `shasum -a 256` of the hit list; equals build 2's `recipe.replay_exclude_sha256` |

## What was measured: ledger rows

`ledger/mac-phase4-v4-replay-2026-10-02.jsonl`:

- A byte copy of `/Users/bharath/qd-campaign/ledger-mac-phase4-v4-replay-2026-10-02.jsonl`,
  sha256 `2b531b94f2f2baa8e827934ae5962ae5db480979082fd7a44d91bb4b8b8f9264`.
- Two rows, both `completed` and both `quick`: one seed, `--no-repo-history`.

| Row | What | Wall clock | `recipe_hash` |
| --- | --- | --- | --- |
| `da3da869-b0b2-4084-a710-5cafaf9a269e` | build 1: plain `--replay-shards` | 5,774 s | `cbcf5c70…` |
| `a48c2987-9088-4243-be7c-26089a2104a6` | build 2: `--replay-exclude` decontam 1's hit list | 4,386 s | `4f44cbb8…` |

Both rows ran from `.claude/worktrees/l-replay-e571065`, detached at `e571065`, with `qd-prep`
from `/Users/bharath/qd-campaign/target-replay-a502670` (sha256 `d605432a…`).

- **Both rows read `code_commit` `e571065…-dirty`** (verified). What makes it dirty:
  - `Cargo.lock` only: four inserted lines, tessl's `block2`/`dispatch2`/`libc` deps.
  - Found by a file-by-file diff of the lane against this worktree at the same commit. The
    session's isolation guard refuses `git` aimed at the lane.
  - The other difference, the corpora symlinks under `data/pool/`, is gitignored.
- **What ran is pinned anyway** (verified): all 35 `code_that_ran` hashes match `e571065`'s
  committed bytes, aggregate `977d469e…`. The two rows' lists are identical. This is the
  `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN` mechanism doing its job.
- The decontam tool writes attestation files, not ledger rows.

The wall clocks were measured on a Mac whose load average was 78, with 0% idle, from other
sessions. They are not a timing of the pipeline.

## Build 1 and its STOP gates: 15 of 15 PASS (verified)

- Command: plan §6 "Build 1", with `--out /Users/bharath/qd-campaign/phase4-v4-replay-build1-2026-10-02`.
- Gate script: `AUDIT/replay-v4-build-2026-10-02/gates.py`.
- Evidence: `gates-build1.txt` and `build1.log`.

Results:

- val shard `ef06ab99…` and val manifest `69d45fd0…` equal F's.
- The held-out manifest hash equals v4's: `116a67c4…`. Only that field was read.
- train, val and replay `remap_hash` are all `e7d0890c…`.
- The replay manifest is `19a772846da4b759…`, 3,568 rows, the same draw as phase 4 and v3.
- `train.json` is v4's entries minus exactly the 3,568 drawn `row_id`s, compared as
  `(row_id, content_hash)`: 285,574 entries, diff 0.
- Every recipe key shared with F's `d96409bd` is equal. The only added key is `replay_shards`.

**A first attempt died of ENOSPC.** Another session's build filled the disk during
`write_shards`.

- Its log is `build1-enospc-probe.log`.
- Its partial `--out`, `phase4-v4-replay-probe-2026-10-02`, holds `data/` manifests only. No
  ledger row was written.
- Nothing reads that directory. It can be deleted.
- Build 1 was rerun to a new `--out`.

## Decontam 1: CONTAMINATED, exit 1, as expected (verified)

- Evidence: `decontam1.log` and `replay-attestation-build1-contaminated.json`.
- 3,568 of 3,568 replay rows checked, 0 too short, against val 18,428 rows and held-out 39,887
  rows. 0 unrenderable targets.
- Hits: val 185, held-out 0. That is the same count as phase 4's attestation (185). Whether
  they are the same rows was not checked: phase 4's attestation names only 50, on another
  shard set.
- The hit list, `replay-hits-build1.json`, written by `e571065`'s `--hits-out`:
  - 3,414 pairs;
  - 185 distinct replay rows (`h`);
  - 184 identity keys.
- **184 keys for 185 rows:** two replay rows share one key,
  `mmlu-train:high_school_psychology::adbe29a4f7032d97` (the same MMLU question twice, rows
  `…:4930` and `…:5210`). Both are hits.
  - Excluding by key therefore removes exactly the 185 hit rows.
  - Build 2's gates confirm this.

## Build 2 and its STOP gates: 18 of 18 PASS (verified)

- Command: plan §6 "Build 2", with `--replay-exclude
  /Users/bharath/qd-campaign/phase4-v4-replay-build1-2026-10-02/replay-hits.json`.
- Gates: the same script with `--probe <build 1> --hits <hit list>`.
- Evidence: `gates-build2.txt` and `build2.log`.

Results:

- **Partition:**
  - commonsense: gold 8,179, replay 1,428, excluded 12;
  - knowledge: gold 12,072, replay 1,955, excluded 173.
- **Replay rows:** 3,383 = 3,568 − 185, exactly build 1's replay rows minus the hit rows.
- **Train shard and manifest** equal build 1's: `be617e7e…` / `895d393b…`. So the exclusion
  moved no gold row.
- **Unchanged from v4:** val `ef06ab99…` / `69d45fd0…`, the held-out manifest, and remap
  `e7d0890c…`.
- **Recipe:** it adds only `replay_shards` and `replay_exclude_sha256` (= the hit list's sha256).

## Decontam 2: CLEAN, exit 0 (verified)

- Evidence: `decontam2.log` and `replay-attestation-build2-clean.json`.
- `replay 290f69a48a44f150: 3383 of 3383 rows checked (0 too short)`, against val 18,428 and
  held-out 39,887 rows, 0 unrenderable.
- `hits {val: 0, heldout: 0} -> CLEAN`.
- `box_q_j6a.sh`'s own attestation check, run here verbatim: exit 0.

## Prelude: Mac, then box (verified)

Both runs use `real_ft_run.main`'s own prelude, stopping at `Ledger(args.ledger)`, the first
statement after `_replay_plan` at `a502670`. Both print the line the waiter matches:

```
replay: 99 batches from <dir>/shards/replay (attestation dc69ffce43b5af2a), weight 1.0, every 6
PRELUDE OK in 3xx s ...
```

The waiter matches it with `REPLAY_DIR` as pinned at `6168513`.

| | Mac | Box |
| --- | --- | --- |
| Script | `AUDIT/replay-v4-build-2026-10-02/j6a_prelude.py` | `/home/ubuntu/post-f/j6a_prelude_box.py`, sha256 `415d09eb…` (= the waiter's pin) |
| Code | `.claude/worktrees/l-replay-a502670` at `a5026707` | `/home/ubuntu/qd-lane8` at `a5026707`, `git status` clean |
| Time | 321 s | 328 s, `nice -n 10`, `CUDA_VISIBLE_DEVICES=` |
| Log | `prelude-mac.log` | `prelude-box.log` |

The Mac wrapper also records the plan: 99 batches, 3,383 rows, shard `290f69a4…`.

**Argv: the waiter's `J6A_ARGV`, with three deliberate differences:**

1. **`--wall-clock-cap-s 3600` and no `--approved-by`.**
   - At a 32,400 s cap, a502670 refuses `--devices cuda` at argv time without a human yes, and
     says an agent cannot supply one.
   - The prelude trains nothing, so the cap is not read before it stops.
   - The waiter still runs its exact-argv prelude under `gpu.lock` before J6(a).
2. **Checkpoint, cache, verdicts and ledger paths are in scratch.**
   - Nothing was written there: the prelude stops before any of them is opened.
3. **On the box, CUDA was hidden.**
   - a502670 runs a stand-in feasibility probe per bucket on each `--devices` entry before
     `_replay_plan`.
   - With the queue's job holding the GPU, all 32 probes saw "No CUDA GPUs are available", and
     `nvidia-smi` showed no prelude process. The GPU was not touched.
   - On the Mac the same probes refuse with "Torch not compiled with CUDA enabled".
   - In both cases main continues: the probe is a report, not a gate.

## Shipped to the box (verified)

- `rsync -a --exclude /shards/train-no-decode --exclude /data/heldout …/phase4-v4-replay-2026-10-02/
  ubuntu@…:/home/ubuntu/phase4-v4-replay-2026-10-02/`.
  - 34 files, 1.53 GB, exit 0 (`rsync-box.log`).
- `shasum -a 256` on the Mac against `sha256sum` (niced) on the box, file by file: identical
  (`sha-mac.txt`, `sha-box.txt`).
- On the box, `data/heldout` and `shards/train-no-decode` are absent (rule 3, and the waiter's
  check).

## Code: commit `e571065` (on this branch, `l-replay-build`; not yet on main)

`replay: every decontam pair at or over the threshold, --hits-out, and --replay-exclude`. It is
`a502670` plus one additive commit: 6 files, +455 −11. Nothing in `python/qd_data` changed.

- **`qd_train.replay.decontaminate`** carries `DecontamReport.pairs`: every pair at or over the
  threshold.
  - `to_json` is unchanged, so the attestation is version 1 byte for byte.
  - A characterization test pins the pre-change body's sha256.
- **`tools/replay_decontam.py --hits-out FILE`** writes that list. Each pair carries
  `(row_id, slot)` and `identity_key`.
  - It refuses a replay manifest that is not the shard set's.
  - It refuses to overwrite, before doing any work.
- **`tools/real_tokenizer_pipeline.py --replay-exclude FILE`** requires `--replay-shards`.
  - It refuses a key the draw puts on the gold side.
  - It records `replay_exclude_sha256` only when given.

**Fail-first** (verified, `failfirst-pre-change.txt`):

- The 11 new tests at `a502670`: 10 failed, and the attestation characterization test passed,
  as designed.
- At `e571065`: all 11 pass (`after-e571065.txt`).

**End to end at `a502670` on fixtures**, earlier in this lane (verified; quick fixture rows in
scratch ledgers under `/Users/bharath/qd-campaign/replay-v4-2026-10-02/e2e/`, not evidence
about the 2B):

- **The fixture:** a plain `--replay-shards` build over slices of the real MMLU/CSQA caches,
  with `--hits-out`. CLEAN: 163 of 163 rows checked, 0 hits.
- **The tiny Qwen3.5 tower, CPU** (`e2e-tiny.log`):
  - `real_ft_run` accepted the attestation: `replay: 163 batches … (attestation
    e88674210a00bd0e), weight 1.0, every 6`;
  - it cached the base's letter logits on 163 rows;
  - it trained 826 steps with 137 replay micro-batches;
  - `replay_kl_first` was −1.8e-07 and `replay_kl_last` 0.0068;
  - it checkpointed and scored: ft row `285cb0e7`, score row `98fd098f`.
- **The stand-in tower:** ft row `8a768c50` (`e2e-standin2.log`).
- **Resuming that replay checkpoint without the replay flags is refused**
  (`e2e-tiny-resume-noreplay.log`):
  - `BackboneContractViolation`: the state carries `['replay']`, which this run does not have.
  - That is the fail-closed behaviour wanted.

## Findings

1. **The decontam hit list and the gold-side check agree on rows, not on pairs** (verified;
   `GAP-REPLAY-HITS-AND-GOLD-SIDE-PAIRS-DIFFER-2026-10-02`).
   - **Agree:** both find exactly the same 185 hit replay rows, and every one of the decontam's
     3,414 pairs is in the gold-side list.
   - **Differ:** the gold-side list has 10,167 pairs on those rows. All 6,753 extras are MMLU
     val targets.
   - `decontaminate` is exact (an inverted n-gram index), so the two compare different texts.
     - The decontam decodes the replay shards.
     - The gold-side renders with `qd_data.render`.
     - Inferred, not verified: the option order differs.
   - **The exclusion and CLEAN are per row, so they are unaffected.**
   - **What is open:** whether E_val counts the overlap that F and J6(a) actually train on.
2. **The gold-side measurement** (verified, `gold-side/summary.json`, `gold-side.log`;
   `gold_side_check.py`, 161 s, 17 GB max RSS):
   - 1,196 of v4's 23,819 MMLU/CSQA train rows reach ≥ 0.5 containment against v4 val, and 0
     reach it against held-out.
   - F gold-trains all 1,196.
   - 185 of them are replay-drawn: exactly decontam 1's hit rows, excluded from J6(a). The other
     1,011 are in J6(a)'s gold train.
   - E_val is 215 val keys, 183 MMLU and 32 CSQA. Pairs sha256 `142d152f…`. The 32 MB pair file
     stays at `/Users/bharath/qd-campaign/replay-v4-2026-10-02/gold-side/`.
   - This is the measurement behind the plan's decision point 6.
   - Its open record is lead-owned and is listed in the plan's "Open (gap ids)". It is not on
     this branch's ledger, so it is not cited by id here.
3. **The Mac was the constraint, not the work.**
   - The first build 1 hit ENOSPC from another session's build.
   - Free disk later fell from 61 GB to 33 GB in about half an hour from other sessions' Rust
     and Go build trees. It recovered to more than 200 GB.
   - Swap held at 46 of 48 GB throughout.
   - Each heavy step here ran alone and after a free-space check (at least 15 GB).

## Suite at `e571065`

**The run of record** is the project's own runner on the build lane:

- `make -f …/l-replay-e571065/Makefile torch-pytest`, with `QD_PREP_BIN` set to the `a502670`
  `qd-prep` and `HF_HUB_OFFLINE=1`.
- **5 failed, 3,235 passed, 48 skipped, in 720 s** (verified, `suite-e571065-torch-pytest.txt`).
- The lane's three modules (`test_replay.py`, `test_replay_decontam.py`,
  `test_pipeline_general.py`) were collected and passed.

The five failures, classified by running the same modules at `a502670` with the same launcher
(`failing5-at-a502670.txt`):

| Test | At `a502670` | Cause |
| --- | --- | --- |
| `test_gaps_ledger::test_every_gap_id_cited_in_a_tracked_file_exists_in_the_ledger` | passes | `e571065`'s `test_replay.py` cites `GAP-DECONTAM-ATTESTATION-NAMES-50-OF-N-HITS-2026-10-02`. That record was written on main and was not in this `a502670`-based branch's ledger. Fixed on this branch by the resolved record appended here. |
| `test_wire_gap_pins::test_every_gap_id_this_lane_cites_exists_in_the_ledger` | passes | The same citation, the same fix. |
| `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this` | fails | The test asserts the repo directory is named `qwen-decision` or `Lappi-decision`; every worktree fails it. |
| `test_lint_gate::test_ruff_is_installed_not_merely_declared` | fails | The worktree has no `.venv/bin/ruff`. |
| `test_qd_data_wire_agreement::test_every_runtime_kind_is_either_mapped_or_recorded_as_unmapped` | fails | `context_language_not_in_pool` and `context_not_unified_diff` are unaccounted. Pre-existing; `e571065` changes neither `qd_data` nor Rust. |

**On this branch, after the gap records:**

- `test_gaps_ledger`, `test_wire_gap_pins`, `test_replay` and `test_replay_decontam`: 64 pass.
- The only failure is `test_gaps_writer`'s directory-name check (`gaps-tests-branch.txt`).
- The three AUDIT scripts pass the lint gate's `ruff check` (ruff 0.16.8, the repo config).
  - Two long lines in each of `gates.py` and `gold_side_check.py` were wrapped for E501 after
    they ran. That is a formatting change only.

**Earlier runs that are not the run of record:**

- The first suite run at `e571065` died with the ENOSPC (`E` everywhere).
- A hand-launched `uv run … pytest` lacked `--with datasketch` and `-o addopts=`. It added a
  sixth failure, `test_declared_dependencies`, which the test itself attributes to the
  launcher.

**Not run:**

- `make gates` (lint, clippy and ledger record/verify): the lint gate needs the worktree
  `.venv`'s ruff. Lint was checked here on the new files only.
- Rust tests: no Rust changed.

## What changed (commits)

- `e571065`: the code commit described above (this branch).
- This phase's commit, on top of it:
  - this file;
  - `ledger/mac-phase4-v4-replay-2026-10-02.jsonl` (both rows);
  - `AUDIT/replay-v4-build-2026-10-02/`: logs, gates, preludes, rsync, sha lists, both
    attestations, the hit list, the gold-side summary and E_val, and the three verification
    scripts;
  - four `gaps.jsonl` records appended with `qd_train.gaps.append_gap`.

**Merging into main** (the plan's decision point 3: build on `a502670`, merge to main):

- `gaps.jsonl` conflicts by construction: both sides appended. Keep both sides' lines, with
  this branch's four after main's, so they are the current records.
- `tools/real_tokenizer_pipeline.py` also differs between `a502670` and main
  (`span_collapse_policy` in headers). That is unverified as a textual conflict.

## Open (gap ids)

- **Closed by this phase:**
  - `GAP-PHASE4-REPLAY-SLICE-OVERLAPS-VAL` (resolved);
  - `GAP-DECONTAM-ATTESTATION-NAMES-50-OF-N-HITS-2026-10-02` (resolved).
- **Open, this lane:**
  - `GAP-REPLAY-HITS-AND-GOLD-SIDE-PAIRS-DIFFER-2026-10-02` (new);
  - `GAP-L-REPLAY-NAVIGATION-2026-10-02` (updated for phase 2: `gitpulse_insights` refused every
    facet with `REPOSITORY_TRUST_REQUIRED`; DevMap was not consulted; ListAgents is not exposed).
- **Open, lead-owned, from phase 1:** see the plan's "Open (gap ids)" on main.
  - This phase measured the one-rule exclusion's gold side (finding 2).
  - It sidestepped the a502670-reader question by building on `a502670`. A set written at
    main's pipeline is still unread by `a502670`.
  - It did not run replay on the real tower: the preludes stop before any tower loads.
- **Context:**
  - `GAP-DATA-GENERAL-CE-ROWS-OVERLAP-FAMILY-VAL`;
  - `GAP-V4-TRAIN-HEADER-OMITS-SPAN-COLLAPSE-POLICY`;
  - `GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN`.

## First command for the next lane

The set, its attestation, the amendments and the pins are all in place. The j6a waiter is
queued on the box:

- `/home/ubuntu/queue/j6a.queued` exists, written 07:07:42 UTC (verified, read-only `ls` at
  07:35 UTC; main `619ff52` records the launch).
- `j6a.started` does not exist yet.
- It runs J6(a) when its queue conditions hold. What remains on the Mac is bringing this branch to main, from the main checkout:

```
git merge l-replay-build
```

After the merge, keep both sides of `gaps.jsonl`, this branch's lines last. Then run
`make torch-pytest` on main.
