# Lead lane, 2026-10-03: making the pipeline robust, and the 2× H100 box for v5

This lane follows the human's ~15:41Z message: "Download needed, ask Fable for the rest; make the
training pipeline robust and purpose-built." It also covers the human's GPU answer, given at
~16:26Z: "Go with 2× H100 (Lambda)". Fable's rulings are in
`AUDIT/finalize-2026-10-03/fable-pipeline-ruling.md`, and the human-facing state is in
`AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md`.

## ~12:00Z 2026-10-04: the post-queue session is built and dry-run (not launched)

The post-queue session Fable ruled at ~10:55Z is built. It is not launched. It runs once the v5
queue ends, about 16:00-17:00Z on Oct 5.

- **Location:** `build/post-queue/`, git-ignored local tooling.
- **Command:**
  `post_queue.sh print|run|check <main sha> [--cache-option-control] [-- <control cache flags>]`.

**What it runs:**
- **Source:** a fresh clone of main from `main.bundle` at the given sha, with the pinned clone's
  untracked `data/` copied in (1.1 GB, no `data/heldout`). It verifies that no tracked file changed.
- **Controls:** the 16 letter and option controls, v5 seeds 0-4 and arm seeds 0-2. Each runs only
  where `existing_controls.py` finds no row of that kind for the eval row, and each row carries a
  `--note` naming the refusal it replaces.
- **Control-side cache check, when flags are given:**
  - seed 2's letter control runs without the flags, its row on a scratch ledger via
    `--write-ledger`; this is the OFF arm;
  - seed 0's real letter control stores the cache entry (MISS);
  - seed 2's real letter control reads it (HIT);
  - `compare_control_rows.py` passes only if the two rows are equal and the HIT row's
    `split_cache` metric says hit.
- **Parity, on GPU 0:** HANDOFF/v6-startup's two-snapshot run on seed 2, done twice with
  `--split-cache`:
  - run A (MISS) is compared with v5's rows, which ran without the cache (OFF);
  - run B (HIT) is compared with run A.
  - Both comparisons use `compare_traj_rows.py`, extended in **6ce4ec8**: `--cross-commit` names
    code_commit and code_that_ran, `--old/--new-split-cache` requires each row's cache state, and
    `--old-plan` covers two plan runs.

**Scheduling:** at most one control writes to a ledger at a time. The v5 stream and the arm stream
run side by side, with the parity beside both.

**Caps:**

| Step | Cap |
| --- | --- |
| Each control | 2,400 s |
| Each parity run | 1,800 s, with `flock -w 900` on gpu0.lock |
| The session | no step starts after 10,800 s |

The ceiling is ~3.7 h, ~$31 at $8.38/h. The expected run is ~1.2 h, ~$10.

**Preconditions:** both `v5laneN.done` markers exist, every `*traj-s*.queued` has its `.done`, no
queue, ft or control process is running, both GPU locks are free, and no earlier session or clone
exists.

**Verified:**
- `test/post_queue_test.sh` passes 35 of 35 checks on the Mac. It runs a fake box home with doubles
  for the two tools. The comparators, the guard, git, rsync and the script itself are real. Among
  the checks:
  - every precondition refusal;
  - one row per seed and kind;
  - a pre-existing row is not written twice;
  - a HIT arm that missed is reported as a failure;
  - a drifted parity metric is reported.
- On the box, `check` finds every required input present. The arm's ledger and the verdicts of
  unscored seeds show as "not yet".
- On the box, `print` under bash 5.1.16 shows 17 controls and 2 parity runs, and creates nothing.
- **The guard's matching, on real rows:**
  - A control row's `recipe.eval_row_id` is the full 36-character id. On v5's box, verdicts-s2
    names `2af3c80d-…`, which is exactly the `row_id` of seed 2's `epoch-score-val` eval row.
  - The option prefixes are the tool's own `OPTION_ARM` and `OPTION_MARGIN`
    (`tools/ft_linear_control.py:135-136`).
  - `existing_controls.py` on F's ledger finds letter row c89b89a1 and option row 4e8ec654 for
    eval row f4feac15.

**How to read the parity:**
- **A against v5 crosses commits** (main against 142a67c). Fields this lane cannot predict, such
  as `protocol_hash`, or `recipe_hash` if the recipe embeds code facts, may differ beyond the two
  that are named. Read that compare field by field; a nonzero exit there is not by itself a cache
  failure.
- **B against A is at the same commit and must be exactly EQUAL.** That is the cache's own test.

**Not verified:**
- The real tools on the real rows, which only the session itself runs.
- The control cache flags, which wait on the v6-ctlcache lane's report.
- Whether the option control accepts those flags. They are passed to letter controls only unless
  `--cache-option-control` is given.

**To launch:**
1. The v6-ctlcache lane reports, and its merge is approved.
2. Bundle main at the sha.
3. Copy `main.bundle`, `existing_controls.py`, `compare_control_rows.py` and `plan-s2.json`, with
   the script, to `/home/ubuntu/post-queue/`.
4. Run `check`, then `nohup setsid bash /home/ubuntu/post-queue/post_queue.sh run <sha> -- <flags>`,
   alone in its ssh call, with the caps and cost stated.

## ~10:55Z 2026-10-04: v6-startup merged; the human's four approvals and Fable's rulings

**The human, ~10:30Z:** "Merge the fix", then "I approve all those decisions ask fable too".
The four decisions were:
1. merge `v6-startup`;
2. the ~$2 box run of the 2B trajectory parity;
3. the rerun of v5's CPU controls;
4. `--split-cache` in v6's launches.

**On main, in order:**
- **14f9329, the control fix.** `qd_train/exclusions.corpus_matches` compares `max_pairs` only
  where it bounds a row. Both attestation checks use it: `read_exclusions` and
  `real_ft_run.py`'s replay check.
  - Fail-first: in a detached worktree of main's HEAD, `test_exclusions_corpus_key.py` failed
    with the box's exact error.
  - After the fix: 176 passed, 3 opt-in skipped.
  - Gap: GAP-V5-CONTROL-MAX-PAIRS-KEY-MISMATCH-2026-10-04.
- **18a4e6d.** GAP-DEVMAP-LAPPI-DB-MALFORMED-2026-10-03, a record another session appended
  uncommitted, committed alone and byte for byte so the merge could update `gaps.jsonl`. Fable
  ruled it; `test_gaps_ledger.py` was green with that line first.
- **fbd43f5, merge of `v6-startup` (b1f20ed..7138ed4).** Made in a temporary detached worktree,
  with the `gaps.jsonl` end-of-file conflict resolved as a union. Main then took it by
  `--ff-only`. The other session's 9 uncommitted paths in main's tree were unchanged across it
  [V, status compared].
  - `real_ft_run.py` auto-merged and kept both `corpus_matches` and `--split-cache`.
  - `PLAN_MAX_KINDS` is still 8 (not among the four approvals).
  - The lane's change to the existing `test_v5_queue_scripts.py` was read before the merge: it
    still requires exactly one `ft_split_rows` binding and no `**kwargs`.

**Tests on the merged tree** [V, `build/merge-v6/suite.log`]:
- The full Python suite on the ML venv: all passed except three.
  - Two are worktree-only: `test_gaps_writer` needs the checkout's name, and `test_lint_gate`
    needs `.venv/bin/ruff`. Both were rerun in main after the fast-forward.
  - `test_declared_dependencies` failed because the launcher lacked the declared `datasketch`.
    Rerun with it: `test_minhash`, `test_export_oracle` and `test_declared_dependencies` pass,
    with 4 skips (`QD_EXPORT_BIN` unset).
- In main: `test_gaps_writer` and `test_gaps_ledger` pass.
- `test_lint_gate::test_the_repository_has_no_ruff_findings` fails on 147 findings, every one
  in a file the merge did not touch: 97 in `AUDIT/finalize-2026-10-03/apply_v5_freeze.py`, the
  rest in other `AUDIT/finalize-2026-10-03/` scripts, `tools/qd_train_oracle_*.py` and
  `python/tests/test_decisions_pool.py`. It predates this merge [I: by the file sets; not rerun
  at 18a4e6d].
- ruff is clean on every Python file the merge or the fix changed [V].

**Fable's rulings on the four approvals:**
- **Controls rerun: on the box, after the v5 queue. Not on the Mac, and not during the queue.**
  - The rows are hash-chained (`prev_row_hash`) in the box's ledger, and the candidate rule reads
    the control from that ledger.
  - It is one CPU-only job: the fixed tool from merged main in a fresh clone (never the pinned
    `qd-lane-v5`), over every v5 seed and the arm seeds. Roughly 10 calls × ~6 min ≈ 1 h ≈ $8.
    Its cap and cost are stated at launch (rule 4).
  - The existing candidate rule (Tier-B) finds the control row by `recipe.eval_row_id` and
    `recipe.tool`, not by `code_commit` (`qd_post_f_rules.rs:5285-5290`) [V]. No v5 candidate
    rule exists yet.
- **`--split-cache` wiring.** v6 has no launch scripts yet. This line binds: every v6
  `tools/real_ft_run.py` call carries `--split-cache <one 0700 dir per box on local disk>`, at
  most 3.4 GB.
  - Its first consumer is the post-queue controls job: run one seed's control with the cache
    OFF and then HIT, and compare the two rows minus ids and timestamps. That is the on-box
    check the lane could not run.
- **The 2B trajectory parity** runs in the same post-queue session, after the controls. Its
  command is `HANDOFF/v6-startup-2026-10-04.md` "Next": `timeout 1800`,
  `--wall-clock-cap-s 1800`, at most $2.10, one GPU, its own ledger.
- **`PLAN_MAX_KINDS` 8→16** stays out until the 2B parity holds.

**Pending, post-queue (~17:00Z Oct 5):** one session on the H100 box.
1. Controls with the cache OFF then HIT, then the rest.
2. Then the parity.

Launch needs its cap and cost stated, and it adds ~1.5 h of box time, ≈ $12. The projected box
total is then ~$387, inside the human's ~$400.

## ~07:15Z 2026-10-04: where each v5 seed's GPU-idle time goes (measured; v6 targets)

**Source.** Gawk-timestamped tails of the H100's lane, trajectory and train logs, in ignored
`build/v5-h100/startup-capture-round2c.log`. The earlier `startup-capture-round2.log` used a
per-line `date` fork that lagged ~3.5 min behind a burst; its timestamps are not used.

**A seed's job is four steps, each a separate `tools/real_ft_run.py` process with its own
startup:**
1. train + score: ~11 min startup [V, 00:19:23 → step 1 ~00:30:18];
2. needle control (`--score-checkpoint`): ~8 min startup [V, 06:48:57 → GPU ~06:57];
3. trajectory: 13 step snapshots, each one `--score-checkpoint` OOD-only call
   (`box_q_v5traj.sh:72-79`), all under one 3,600 s cap, holding the lane's GPU lock;
4. CPU controls, outside the lock.

**One trajectory call, `v5traj-s2` step 12176 [V, the capture]:**

| Phase | Time |
|---|---|
| Python before the MinHash block | ~3.6 min [I: the MinHash line prints at 07:06:10 after 89.2 s; LSH took 32.2 s] |
| the MinHash block (see the correction below), the same input on every call | ~2.0 min |
| val, OOD and shard load | ~12 s |
| model load | ~24 s |
| the OOD scoring itself | ~16 s on the GPU |

**Correction, ~07:50Z, from the v6-startup lane.** The lead read the "minhash … in 89.2 s" and
"lsh … in 32.2 s" lines as Rust time, and told the human twice that "122 s is already in Rust".
That was wrong. The lines print at the block's exit, and their times include Python shingling,
packing and canaries.

The lane's phase timers on the Mac, v5 data, unprofiled (`AUDIT/v6-startup-2026-10-04/timed_prelude.py`
on branch `v6-startup`), give a ~204 s split rebuild:

| Part | Time on the Mac |
|---|---|
| source loads | 12.7 s |
| `build_mixture` | 73.4 s, of which `drop_contradictory_prompts` 51.7 s; cProfile names `render._escape`, a per-character Python scanner, 5.3M calls |
| the native_minhash block | 120.3 s |
| … its entry | 57.2 s, mostly Python shingling; the `qd-prep` minhash subprocess is 11.8 s |
| … dedupe | 47.9 s of Python `exact_jaccard` etc.; LSH inside it 1.4 s |
| … split | 15.1 s; LSH inside it 1.6 s |

So the Rust is ~15 s of ~204 s on the Mac [V, the lane's numbers]. The box's split is not
measured. The Python share of startup is larger than the lead first reported.

So one snapshot is ~6.5 min of wall time for ~16 s of GPU. The 3,600 s cap fit 9 of the 13
snapshots on both seeds [V]:
- `v5traj-s2` at 08:00:04Z: "the 3600s cap left steps 9000 10000 11000 12000 NOT RUN";
- `v5traj-s3` at 08:00:41Z: the same four steps.

The scoring order was 12176 first, then 1000–8000. The last-3 average row is NOT RUN on both,
per GAP-V5TRAIN-TRAJECTORY-AVERAGE-NOT-SCORABLE-2026-10-02. The trajectory is pinned and
pre-registered, so this is a finding reported to the human, not acted on. It will recur on
every v5 seed. Seeds 0 and 1 were picked at 08:00:04Z and 08:00:41Z.

**v6 targets, in order of size:**
1. Trajectories through `--score-plan`: one startup for all snapshots. Corrected by the
   v6-startup lane, ~07:35Z. This section's first version said `_check_score_plan_flags` lists
   `--ood` as clashing; it doesn't. `real_ft_run.py:9254-9263` is the `--shuffled-label` clash
   list, and `_check_score_plan_flags` accepts `--ood`. A plan still cannot express the
   trajectory row today:
   - a plan's `ood` pass writes the diagnostic row (tag `<model>-ood-diagnostic`, metrics
     `ood_diagnostic.*`, gate `ood_abstain.diagnostic`, no `checkpoint_step`);
   - every plan needs `--score-val`.

   The lane is drafting a `trajectory` plan pass on branch `v6-startup`. Its parity on a toy
   corpus proves the plumbing only. The 2B parity needs v5's step snapshots, which exist only on
   the H100 box, so it is written as a box command and is NOT RUN. Spending box time on it is
   the human's call.
2. The Python split rebuild, measured above. The lead redirected the lane at ~07:50Z.
   - Item 3, a cache of qd-prep's dedupe result, has a ceiling of ≤15 s and was stopped.
   - The lane first answers whether the whole rebuild's result can be cached, content-addressed,
     for a ceiling of ~190 s per startup.
   - Failing that: port `_escape`, shingling and the `exact_jaccard` bucket through the
     native_minhash seam (ceiling ~135 s). It must have no pre-pass (spancheck's 0.81× lesson),
     and its no-go margin is named before the interleaved A/B.

## 06:51Z 2026-10-04: v5 seeds 2 and 3 scored; both hold R9's bar, not poor

Both exited 0: seed 2 at 06:48:56Z, seed 3 at 06:50:39Z. Ledger:
`/home/ubuntu/ledger/h100x2-v5-2026-10-03.jsonl` on the H100 box. Rows [V, `read_v5_rows.py`
and the rows' gate details]:
- seed 2: ft 8079c966, score 2af3c80d;
- seed 3: ft 1c8a0ab5, score e09b652a.

| | F s0 / s1 / s2 (f4feac15, aeca8d69, 8c3a774a) | v5 s2 | v5 s3 |
|---|---|---|---|
| val_top1.span (of 7,238) | 0.902 / 0.912 / 0.905 | 0.913 | 0.908 |
| val_top1.choice | 0.831 / 0.831 / 0.833 (of 10,985) | 0.839 (of 17,254) | 0.847 |
| 8K needle worst bucket (gate 0.95) | 0.656 / 1.000 / 0.770 | 0.705 | **0.967, passes** |
| OOD abstained of 180 (Wilson lower vs 0.9) | 73.3% / 31.1% / 37.8% | 84.4% (0.784) | 86.7% (0.809) |
| in-distribution abstain (upper vs cap 0.05) | 7.7% / 8.4% / 8.6% | 6.5% (0.069) | 5.6% (0.060) |
| permutation_consistency (floor 0.95) | 0.932 / 0.930 / 0.927 | 0.943 | 0.949 |

**R9's bar holds on both seeds,** so by the rule written at ~03:05Z the results are not poor.
The queue continues.

**Against F:** v5 is ahead on every gate quantity. It still fails ood_abstain,
permutation_consistency (seed 3 short by 0.001) and seed 2's needle gate. Rule 2 stands: these
are readings, not promotions. The choice populations differ (v5's val set adds the new sources),
so choice is not a like-for-like comparison.

**Each seed's job continues** with its needle control (cap 5,400 s) and trajectory before the
lanes pick seeds 0 and 1. A timestamped capture of that startup is being written to ignored
`build/v5-h100/startup-capture-round2.log`.

## 04:53Z 2026-10-04: J6(f) done; the holder has the GH200's GPU

**J6(f)** ended at 04:52:58Z ("item 6 all done", `/home/ubuntu/logs/q-j6f.log`). Its rows are
in `/home/ubuntu/ledger/gh200-j6-v4-ablations-2026-10-01.jsonl`:
- **ft row** cb3a0cda;
- **score row** cd2e4f82 [V, `read_v5_rows.py`]:
  - span 6537/7238 = 0.903;
  - 8K needle worst bucket 80–100% at 43/61 = 0.705 (gate fails against 0.95);
  - ood_abstain 0.483 (fails), permutation_consistency 0.911 (fails);
- **needle-control row** 5bb906f0: 1K 0.789, 2K 0.803, 4K 0.797.

That is within F's spread. Its reading belongs to the post-F rules that read it later in the
chain.

**The handover [V]:**
- **The holder.** `mlr.holding` is 04:52:58Z; it took the lock the moment J6(f) exited.
- **Item 9 (nomask)** started at 04:53:17Z and refused before the lock: "P2 verdicts {'A':
  'inconclusive', 'B': 'inconclusive'}: outcome run cancelled", exit 5. That is its own P2 gate,
  not the holder; `nomask.done` follows.
- **Item 7 (tierb2)** started at 04:53:47Z at f6a0928. It is blocked in
  `flock /home/ubuntu/queue/gpu.lock timeout 2400 … real_ft_run.py` (pid 1708295) behind the
  holder, as designed.
- **The MLresearch session** had three GPU processes by 04:54Z (13.7, 16.2 and 13.7 GiB), working
  under `/home/ubuntu/mlr`.

## ~03:05Z 2026-10-04: overnight rule, written before any v5 row exists

**The human, ~02:40Z:** "I would sleep by 06:45 Z; if the results are poor, post-train and
fine-tune it till morning, try to keep the GPU busy."

**Asked and answered, ~03:00Z:** the queue already keeps both H100s fine-tuning until ~17:00Z
Oct 5. So "post-train if poor" could only mean stopping the pre-registered queue (V5_STOP) and
running something else. The human chose **"Continue as pre-registered (Recommended)"**. The
alternative was v5's data with F's flags, quick-marked.

**What this means:**
- Whatever seeds 2 and 3 read, the queue runs on: seeds 0 and 1 at ~07:20Z, then the remaining
  rounds.
- The lead reads seeds 2 and 3 against R9's bar and reports beside F.

**R9's bar, used here as a report-only spending reading, not a gate (rule 2 untouched):**
- per seed, `val_top1.span` with 5n ≥ 4·n_total;
- and the 8K needle worst depth bucket with 2n ≥ n_total;
- "poor" = both seeds 2 and 3 fail it.

F scored, on the GH200's `gh200-p4-v4-2026-10-01.jsonl` [V, read with the reader below]:

| Row | span | 8K worst bucket |
|---|---|---|
| f4feac15 | 0.902 | 0.656 |
| aeca8d69 | 0.912 | 1.000 |
| 8c3a774a | 0.905 | 0.770 |

**Hard failures** (no row, NaN, crash) stay on the standing contingency: re-queue by the repo
rule; V5_STOP only if "decide room" is threatened.

**The reader:** `build/v5-h100/read_v5_rows.py`, stdlib-only throwaway analysis. It prints each
eval row's span count, 8K worst bucket, the bar and every gate's verdict. Run it as
`ssh <box> 'python3 - <ledger>' < build/v5-h100/read_v5_rows.py`; v5's ledger is
`/home/ubuntu/ledger/h100x2-v5-2026-10-03.jsonl` on 68.209.74.244. It was validated on F's rows
above. Rows carry no seed field, so map seeds through the lane logs' `ft row` / eval lines.

The lane's `v5-pause` is NOT RUN under `V5_R9=waive` (`v5_common.sh:1040`). That contradicts
R9's "v5-pause's reading is still written, report-only"; the reader stands in for it here.

## ~01:31Z 2026-10-04: the GH200 is reserved for MLresearch after J6(f)

**The human, ~01:25Z:** "spin a new session to work and hand over gh200 to work on MLresearch,
when gh200 is free".

**Why the post-F chain will stall after J6(f).** A holder, `/home/ubuntu/mlr-hold/mlr_hold.sh`,
was launched at 01:31:07Z: pid and pgid 621107, log `/home/ubuntu/logs/mlr-hold.log`.
- Source: `build/gh200-mlr/mlr_hold.sh`, sha256 63a267e7….
- It does `exec 9>/home/ubuntu/queue/gpu.lock; flock 9`, blocked behind J6(f) (`box_q_j6f.sh`
  holds fd 9 for its whole job).
- Every Lappi GPU stage blocks on the same lock with no timeout [V]:
  - `flock 9` in j6dv4, rung0, cudadev, rungd, fsucc, j5pp, j6a, j6g;
  - `flock gpu.lock timeout N python` in `perf_tierb_outcome.sh`, which serves both nomask and
    tierb2 (`perf_tierb_fused.sh` execs it).
- So the chain queues behind the holder: no waiter was stopped or edited. At launch no other
  process had `gpu.lock` open (`/proc/<pid>/fd` of every `box_q_*`) [V].

**Markers in `/home/ubuntu/queue`:**
- `mlr.queued`: the holder started.
- `mlr.holding`: it has the lock.
- `mlr.release`: the MLresearch session is done; the holder exits within 60 s.
- `mlr.draining`: the cap was reached.
- `mlr.released`: the holder exited, with the reason, written on every exit path.

**It also releases by itself:**
- after 90 min with no compute process on the GPU (`nvidia-smi --query-compute-apps`; an
  nvidia-smi failure counts as busy);
- after 30 h, when it drains for up to 2 h first.

**Tested** on the GH200 by `build/gh200-mlr/mlr_hold_test.sh`, against scratch queue dirs and a
fake nvidia-smi: 27 of 27 checks passed [V]. The cases:
- waits behind a held lock, holds, then releases on `mlr.release`;
- busy resets the idle count; idle releases;
- a failing nvidia-smi is never counted as idle;
- the cap drains, and the drain is bounded;
- refuses over an earlier holder's markers;
- TERM to its session while waiting exits without holding.

**To cancel it:** `kill -TERM -- -621107` (the whole session; TERM to the bash alone is deferred
until `flock` returns).

**The MLresearch session** is a chip offered to the human (task_a0c9c840, cwd MLSystemsLab):
experiments 1–10, ~160 jobs, ~17 GPU-h. It must never flock `gpu.lock`, must gate on
`mlr.holding`/`mlr.released`, and must touch `mlr.release` when done. When `mlr.released`
appears, Lappi's chain resumes with nomask (item 9), then tierb2, j6dv4 and the rest.

## State at ~00:50Z 2026-10-04: v5's measured rate; v6 spancheck is a negative result

**v5 rate [V]** (`/home/ubuntu/v5/train-s{2,3}.log`, 00:45:45Z):
- Both seeds run at 1.81–1.82 s/step and ~18,500 pos/s. Peak allocated is 61.3 GiB; nvidia-smi
  shows 72.8 and 73.5 GiB with both GPUs at 100%.
- 12,176 steps make ~6.1 h of training per seed. Seeds 2 and 3 should finish training ~06:40Z
  and be scored by ~07:20Z, under the 9 h cap.
- The six rounds still end ~16:00–17:00Z Oct 5, at ~$375 box spend [I: assumes the scoring
  phase fits the pre-registration's ~7.0 h per seed; read it when seed 2 ends].

**Per-run startup** is ~11 min, not the 456 s in the section below. That 456 s was seeds 0 and
1 failing at the needle step.
- Seed 2 started at 00:19:23Z (lane log) and reached step 1 at ~00:30:18Z [V].
- MinHash took 88.3 s and LSH 33.8 s, already in Rust (`qd-prep-v5`) [V].
- The feasibility probe took ~70 s (32 buckets × ~2.1 s, a 128×4 stand-in on the GPU) [V].
- The other ~7.5 min have no per-stage timestamps in the log [U]: corpus rebuild, val, needle,
  OOD, shard load and model load. Take a timestamped tail of the next startup (seed 0, ~07:20Z)
  before choosing a v6 port.
- Cheap v6 candidates, not built:
  - cache the dedupe result across seeds (same data, same pairs);
  - skip the stand-in feasibility probe once the memory probe has passed.
- v5 is pinned at L 142a67c, so none of this changes the running seeds.

**v6 spancheck lane: done, negative, not merged.**
- **Branch:** `v6-spancheck` in `build/v6-spancheck-wt`: a9ddf61, cbc0edc, 363bb08, 4e97e02,
  de50cb1, 19644dc. Its handoff is `HANDOFF/v6-spancheck-2026-10-03.md` on that branch.
- **What it does:** `qd-prep spancheck` matches `_span_token_positions` byte for byte, end to
  end on the Qwen3.5 tokenizer: every shard set and every measured value is equal.
- **Speed:** slower. On a fixed 600-slot slice it ran at 0.81× the Python's speed with Qwen and
  0.94× with the byte tokenizer, interleaved min-of-N.
- **Why:** the bucket it replaces was ~13 s of the 626 s profile in
  `AUDIT/perf-pipeline-shards-2026-09-30.md`. Encode (60.9 s) and decode check (35.2 s) dominate,
  and the port adds a pre-pass encode.
- **Disposition:** the branch stays as the reference implementation; `--native-spancheck` would
  ship off by default. The lead checked 4e97e02's "wrong expectation": it was in the lane's own
  new test (its fixture collapsed every row), not a moved reference value.
- **Open, the human's call:**
  - the decode check in Rust needs the `tokenizers` crate, a new dependency;
  - encode-once under `--memo-limit 0` is a Python-only change, proposed and not made.
- **The lane's four suite failures** were all against its base 8e6a009 or the worktree's name.
  Main's `test_gaps_ledger.py` passes 10/10 at c28bcef (run 00:46Z).

## State at ~00:30Z 2026-10-04: seeds 0 and 1 failed at startup and were re-queued

**What happened.** The probe passed on attempt 1 at 00:11:47Z:
- ft row `51c3e0f9-b63b-405e-9f42-92a4fd8d5b13`;
- peak 65.7 GiB, with the 12 GiB margin;
- 836 s, $0.97.

Seeds 0 and 1 then both exited 1 at 00:19Z, at the end of startup, with no ledger row and no
checkpoint. The cause is GAP-V5-H100-HF-CACHE-NO-REFS-MAIN-2026-10-04:
- the box's HF cache had the snapshot but no `refs/main`;
- the backbone loads by path, so the probe passed;
- the needle tokenizer loads by repo id, so the seeds failed.

The lanes moved on to seeds 2 and 3.

**Fixes:**
1. At 00:21Z: `refs/main` = b1485b2f… written on the box. An offline repo-id tokenizer load
   then passed.
2. At 00:24Z: seeds 0 and 1 re-queued under `v5.sched.lock`, by the repo rule "a run that did
   not write its row is rerun".
   - Their markers moved to `/home/ubuntu/queue/failed-startup-2026-10-04/` and their logs to
     `/home/ubuntu/v5/failed-startup-2026-10-04/`.
   - `v5.spend` keeps the $1.07.
   - `v5_next_job` printed `job v5-s0`.
   - Fable ruled to re-queue at once, so that a second failure could not make the next pick
     `decide room` on two empty seeds.
3. In ignored `build/`:
   - `phase_a_transfer.sh` writes and checks `refs/main`;
   - `box_prelaunch.sh` loads the tokenizer by repo id offline, and printed 248077.

**Now.** Seeds 2 and 3 passed the needle step at ~00:29Z. Their batch-order digests equal the
prelude record's: seed 2 876b9187…, seed 3 4f14ddb4….

**Order:** [s2|s3] → [s0|s1] → [s4|arm0] → [arm1|arm2] → [J5′0|J5′1] → [J5′2|idle]. That ends
~17:00Z Oct 5, with box spend ~$375–385.

**Per-run startup** is ~456 s on CPU with the GPU idle (Python corpus rebuild): a v6 port
candidate.

**DevMap** broke mid-session: GAP-DEVMAP-INDEX-MALFORMED-2026-10-04.

**GH200.** J6(f) is running and should finish ~04:30Z. The human chose to run their MLSystemsLab
experiments (~17 GPU-h) on the GH200 after J6(f), from a separate session. The handover is open:
either their session holds `/home/ubuntu/queue/gpu.lock` under flock, or the ten Lappi waiters
are stopped, which needs the human's explicit yes.

## State at ~23:58Z: v5 LAUNCHED (superseded where the section above disagrees)

**Launched at 23:56:52Z.** `build/v5-h100/launch_lanes.sh` started both lanes on the 2× H100 box:
`box_q_v5.sh 0` and `box_q_v5.sh 1`, each under nohup setsid in its own ssh call. It then wrote
`/home/ubuntu/queue/V5_LAUNCH_YES` with the human's words: "68.209.74.244 ;  2xh100 are back
optimize training pipeline and start training."

At 23:57:51Z both lane logs (`/home/ubuntu/logs/q-v5lane{0,1}.log`) printed:
- the launch yes;
- the pre-registration's cost figures: approved ~$356.0 / 84.96 GPU-h, projected ~$309.7 / 74.0
  GPU-h for the 11 runs, at $4.19 per GPU-h;
- the prelude digests, with "mismatches: None";
- the recipe: C1, C2a and C2b off, lower keep, lr f, `--checkpoint-skip-layers 6`.

Lane 0 is running the memory probe on GPU 0. The probe has `--probe-shapes 12`, a 1,800 s cap
and an estimate of $2.10, and writes to `/home/ubuntu/ledger/h100x2-v5-probe-2026-10-03.jsonl`.
Lane 1 waits for the probe's marker before its first pick, as designed. So GPU 1 is idle during
the probe.

**L = main = 142a67c.** All L checks pass (`build/v5-build/L_checks2.log`):
1. pytest, six files: pass.
2. cargo `post_f_rules_v5` and `post_f_rules_tierb`: 21 and 34 passed.
3. The lock cargo wrote is a669ae0e, restored afterwards; the tracked tree is clean.
4. The Mac prelude: pass in 435 s.
   - Record sha256 `83f531e6fbd4c3f55fc8d89ce0cc554b73211968426c4b73d1434967a00d98ed`.
   - span_check 177,240/177,240.
   - Padding waste 5.79%, against a gate of ≤15%.
5. The musl build at L:
   - `qd-prep` sha256 `900534f1c0cbcb83460eb52291f59b4939bcf0c25c804bc358ce62d2c5d22ef6`;
   - `qd-post-f-rules` sha256 `6b3771fc496d5000bd88888ddd1cafe611c25982829ed4b66651d7d1323e606a`.

**Pins and deploy.**
- `build/v5-h100/fill_pins.py` filled the pins (sha256 `f2ac5e9a…`): pre-registration
  `62749d4608d41fc3113635cabd38ca2ba0c21dcf578e707598d183bdd356af96`, the prelude record above,
  and the binaries above.
- `build/v5-h100/deploy.log`: every file was sha-checked on the box against the Mac.
  - the lane bundle `f157493b…`, cloned and checked out at L, tracked tree clean;
  - 23 shard files equal;
  - both binaries equal;
  - the queue files and the prelude record equal.
- `build/v5-h100/box_prelaunch.sh` printed PASS: both GPUs at 0 MiB, no processes, no markers,
  no v5 ledgers, `UNSET=[] CHECK=[]`.
- Fable's extra rule-3 check, a `find` for `pairs.tsv`, `*.request.bin` and `*heldout*` over the
  lane pool, the shards and `v5-stage`, printed nothing.

**What `build/` (ignored) changed, recorded here because no committed file does:**
- **deploy_lane.sh's guard** was `main == L`. A peer session ("Branch merge and cleanup", on its
  user's instruction) moved main to 2922990, an `-s ours` merge of perf-pipeline-rust with a tree
  identical to 142a67c. It then moved main back to 142a67c with a guarded `update-ref`. The guard
  is now "L is an ancestor of main", and the script prints both trees. The box checks L out by
  hash and asserts HEAD == L, so that is the real requirement.
- **`git bundle verify`** fails outside a repository on the box's git 2.34.1 ("need a repository
  to verify a bundle"). The first deploy attempt stopped there, with nothing created beyond the
  copied bundle. The script now compares the bundle's sha256 on both ends, clones, and verifies
  from inside the clone.

**Peers.** The same peer deleted 68 fully merged branch refs. It recreated perf-pipeline-rust at
4d9242c, and redoes its `-s ours` merge after this lane's all-clear.

**Experiment 3** (short rows): no-mask +8.6%, cuDNN with the mask kept +5.4%. See
`AUDIT/finalize-2026-10-03/h100-perf-experiments-2026-10-03.md`, run 3. The attention levers'
gain grows with row width.

**Measured on the Mac, not acted on:** the prelude spent 410 s in Python before `main reached
Ledger(args.ledger)`. That is a candidate for a Rust port.

**Open:**
- the probe's marker (`v5.probe-passed` or `v5.probe-failed`);
- seed 0's phases, the score phase first;
- the box spend, which `watch_h100.sh` counts as box time since 19:45Z (tell the human at $350);
- the v6 spancheck subagent, released after launch.

## State at ~21:25Z (superseded by the section above)

**Measured** (logs under `build/v5-build/`, gitignored):
- **The scoped containment scan** at v5-build cd2967e (`scan-scoped.log`):
  - CLEAN; 7,260 train keys excluded (= 104,905 unscoped − 97,645 the scope block leaves
    unexcluded);
  - attestation sha256 `70e2c2a9…`, exclusions `c38dd2ca…`;
  - the CLINC zero-checks are unchanged (val: 1 exact, 1 subsequence; held-out: 1 subsequence),
    as the human accepted them.
- **The trial merge of v5-build into main** (`merge_test.log`, branch `v5-main-merge` at
  2e33adf). Rust failures:
  - two trained-families targets, caused by the merge (main's fixtures have no prompt_format);
  - `post_f_rules_v5` (23) and `gate_report_exclude` (2), present on v5-build alone. The
    noul-weight checker took seeds.v5's 0-4 where the arm envelope is 0-2, so the arm's room
    reading would refuse at launch and the arm would never run. The gate-report golden predates
    7320707/c59ccda.

  Python: one failure, the worktree-name check (known). `python/qd_data` at the merge equals
  cd2967e, so the shard fingerprint holds.
- **The token-share method** (`build/v5-build/defect_token_share.py`) reproduces v4 exactly:
  306,926,895 tokens, 83.7072%.
- **Staged on the box and sha-verified:**
  - the corpora → `/home/ubuntu/v5-stage/pool`;
  - the decision pool at its Mac path;
  - exclusions and attestation → `/home/ubuntu/v5-exclusions-2026-10-03`. The queue's split
    check refuses any path containing "containment".

**Changed:**
- **main:**
  - 3629505: GAP-MAIN-COMMITTED-CARGO-LOCK-STALE-2026-10-03;
  - 126c4b4: the human's optimize answer, "Launch as pre-registered";
  - a2b4664: merge of `l-v5-freeze`, the freeze script.
- **v5-build:** cd2967e, the same-family scope.
- **Uncommitted, tested only after the build frees the lock:**
  - in `build/v5-merge-wt`:
    - the noul-weight seeds fix and the seeds34 retirement refusal, with tests;
    - the gate-report golden (diff checked: 14 added not_checked names, nothing else);
    - the prelude record's `needle_suite` block for item 5 (Fable ~21:20Z), with a test;
  - on main: `AUDIT/finalize-2026-10-03/apply_v5_freeze.py`, item 5's fill reads that block.

**Open, in order:** the build (`build/v5-build/build.log`, started 20:56Z), then:
1. the merge tests with the fixes;
2. v5-build: the ledger row and the Cargo.lock revert;
3. main: the merge, reusing 2e33adf's resolutions, then the cherry-picked fixes;
4. the freeze (`--scan` is the SCOPED scan; `--family-rates` and `--zero-checks` are
   `build/v5-build/scoped-*.json`) and the readers patch, with `post_f_rules_v5.rs` through
   `build/v5-build/readers_v5rules.py` (the lane's hunk at :493 no longer applies). That is L.
5. At L: tests, the prelude, `build/v5-h100/musl_build_at_L.sh`,
   `build/v5-h100/fill_pins.py`, `build/v5-h100/deploy_lane.sh`, the checks, the launch.

**First command for the next lane:** `tail -n 5 build/v5-build/build.log`.

## What was measured

No new ledger row: nothing trained, and no GPU job ran. The evidence:

- **The memory arithmetic for the H100.** `AUDIT/finalize-2026-10-03/v5_h100_budget.py` and
  `.out` reproduce F's recorded `device_budget` value (ft row 973cd4e3) to the byte:
  68,510,315,980 B.
  - The costliest batch is the narrowest bucket with the most rows (258×137, 63.81 GiB).
    v5's 3×10,240 prices at 60.01 GiB, and v5's costliest shapes at 64.46 GiB or less.
  - F's torch-allocated peak was 61.6 GiB (F seed 1, `train-s1.log`). The earlier "7.5 GiB
    headroom" compared the process-level 72.1 GiB, and was corrected at bf95f56.
- **Retained checkpoint size on the GH200.** Each is 25 GB (`du /home/ubuntu/ckpt/p4-v4`,
  read-only, ~16:21Z). The v4 shards are 2.7 GB and the base snapshot 4.3 GB.
- **The CLINC test identity keys** (Fable's ruling 5), in
  `AUDIT/finalize-2026-10-03/clinc-test-keys/`.
  - 5,500 unique keys.
  - v4's manifests hold them in train (4,958 per intent family, 4,957 for intent.domain), val
    (300) and held-out (240).
- **x86_64 static (musl) cross-builds** on the Mac, made without any download, from v5-build
  1ab477f:
  - qd-prep, sha `a719d37f…`;
  - qd-post-f-rules, sha `6b55c1a1…`.

  They are in `/Users/bharath/qd-campaign/target-x86_64-linux-musl-v5/` and have NOT been run.
  They need a box: a parity run and a timed split rebuild. Use glibc if musl is more than about
  2× slower.

## What changed

**Main:**
- c96290e, bf95f56: the AUDIT ruling and the report, then the memory correction.
- 1bff325: merge of L-trained-families. A release admits only the task families it was trained
  on; the lane's commits are ce24e56, 0f491bc and 73007f1.
- 18b9851: the amendment script, the CLINC keys, and the 11-run correction.

**v5-build:**
- 3a2fc85: the bench's v2 decision pool, reviewed; the characterization re-pinned.
- 1ab477f: the rebuild tools read the pool.
- 43a075c: `--probe-shapes MARGIN_GIB`, the pre-registered memory probe.
  - 24 new tests failed first on 1ab477f (`build/v5-h100/probe-failfirst.log`).
  - The full suite was 4,096 passed and 4 failed. Two failures were environmental. The other
    two were caused by the change and are fixed: the trainer is again handed `ProgressLine(`,
    and the test docstring's references are now literals.
  - The re-run after the fix: 85 passed, 3 skipped (opt-in).
- ce0acdf: the bench's v3, applied verbatim (patch d5a427e0…).
  - qd-prep: 74 + 2 tests pass; clippy -D warnings is clean.
  - Targeted: 123 passed.
  - Full suite: 4,099 passed; 2 failed, both environmental (the real-ledger guard and .venv
    ruff); 49 skipped (`build/v3-verify.log`).

**Main, later:**
- 4eca691: the 1bff325 merge verified, and its gap resolved.
  - 25 test binaries passed in the clean export.
  - tracked_source_is_text passed 12/12 once the export was made a repo of all 1,522 tracked
    files. In the bare export it had refused an empty `git ls-files`, as designed.
- ca9e9a3: report-to-human corrected: the box needs no rustup, and the disk floor is 600 GiB.
- cf99884: the dedupe-bound ruling and its probe evidence, `AUDIT/finalize-2026-10-03/dedupe-probe/`.
- c10612b: the residual measured on the native path (5.87M dedupe / 2.89M split candidates), Fable's
  ruling (A) POOL_MAX_CANDIDATE_PAIRS = 12.5M, and (E) deferred as
  GAP-DEDUPE-LSH-BAND-CANDIDATES-NOT-DUPLICATES-2026-10-03.
- 3d25ecc: the human's answers on the v5 launch, verbatim ("Yes to all, waive R9, approve ~$400"),
  `AUDIT/finalize-2026-10-03/human-answers-2026-10-03-v5-launch.md`.
- 1a6f5d0: `AUDIT/finalize-2026-10-03/apply_v5_launch_amendment.py` (amendment 2), for L-v5-2gpu
  to apply. Its dry run on v5-2gpu's DRAFT gives sha256 `6098345f…`; a rerun prints "already
  applied".

**v5-build, later:**
- 51272d5: the lead's half of v4.
  - Exact-content dedupe for the four structured Open-Jev families, which are scoped out of
    MinHash and reported not_run by ruling; a leak is the same content digest across the final
    split.
  - `DataConfig.max_candidate_pairs`, with `POOL_MAX_CANDIDATE_PAIRS = 12_500_000`.
  - 15 new tests, 7 of which failed first on ce0acdf (`build/failfirst-exact.log`).
  - The characterization re-pinned to `d65616af…`, accounted by compare.py.
  - Full suite: 4,113 passed; 2 failed, both environmental (the real-ledger guard and .venv
    ruff); 49 skipped.

## What is open

**Superseded at ~19:10Z.** The items below "The v5 decision pool is not admitted" are kept as
written. What stands now:
- **The dedupe design is settled and implemented** (51272d5). The second `qd-prep decisions`
  build waits on the bench's v4 patch: the Open-Jev val draw key on group_key, owned by the bench.
  The lead reviews the patch and applies it to v5-build. After that build, the bench runs the
  VitaminC candidate check (more than ~1M candidates means the lead hears before the v5 build).
- **The human's answers are in** (3d25ecc). None of the "still needed" items below is open,
  except the 300 diffs, which are unmet under rule 2.
- **The 2× H100 box is UP (superseding the next item):** the human launched it at ~19:46Z,
  `ubuntu@68.209.74.244` ("2xh100 are back optimize training pipeline and start training";
  `AUDIT/finalize-2026-10-03/human-answers-2026-10-03-h100-launch.md`). Python 3.12.15 was
  installed with the human's yes. The venv setup started ~19:54Z (`/home/ubuntu/setup/h100_setup.log`).
  The box watch is `~/qd-campaign/watch_h100.sh`. The 1× fallback and the V5_GPUS commit are dropped.
- **(Superseded) Lambda's 2× H100 was sold out at ~19:03Z** (the human: "2xh100 are out").
  - Fable's ruling: wait, and the human rechecks at ~21:30Z.
  - If it is still out, the fallback is 1× H100 80GB SXM5 at $4.29/h: ~74 GPU-h ≈ $317 for the
    runs, ~$330–345 for the box, ~76–78 h of wall time. It needs:
    - the human's own "launch 1× H100" words;
    - a lead amendment: hardware.box and lanes, ledger names h100x1-…, every price at $4.29,
      launch.approved re-derived with no idle-GPU terms;
    - the lane's V5_GPUS pin.
  - L-v5-2gpu's reading at 0e5f2ff: lane 0 alone has no dependency on lane 1. The constant
    `V5_GPUS=2` (v5_common.sh:100), the box-name check (:368) and the rate checks (:378-379)
    refuse a 1× box before anything runs.
- **L-v5-2gpu still owes:**
  - amendment 2, launch.approved read and enforced, and the whitelist class fix (with
    `--decisions-pool`); then its final sha, which the lead merges into v5-build with
    `build/v5_build_merge.sh`;
  - after that, as a separate commit, `V5_GPUS` as a fail-closed pin (UNSET, 1 or 2), and a
    refusal of LANE ≥ V5_GPUS.

The earlier list, as written:

- **The v5 decision pool is not admitted.** Its one Mac build (examples sha256 `e3a03f38…`)
  hit the 5M candidate-pair bound in both dedupe and near_duplicate_disjoint, so split_status is
  not_run.
  - **Fable's ruling** (`AUDIT/finalize-2026-10-03/dedupe-probe/RULING.md`): the bound stays.
    For the structured Open-Jev families (policy, evidence, routing, rubric), dedupe uses exact
    content and the near-duplicate check is not_run by ruling. A leak is the same content
    digest across splits.
  - **A second build is granted only if `residual.py` does not truncate.** Its output is
    `build/dedupe-probe/residual.{log,json}`.
  - **Before the v4 patch:**
    - the bench's answers on group_key, per-group val, and the 123 collisions;
    - the DRAFT amendment for the scoping, after the v5-2gpu merge;
    - the file split: the bench owns qd-prep decisions.rs and qd_data/{decisions,sources}.py;
      the lead owns qd_data/{dedupe,split,minhash,config}.py.
  - **The planned seam:**
    - `DataConfig.exact_content_families`, set by `pool_data_config`, in the fingerprint only
      when non-empty, so no-pool hashes do not move;
    - dedupe and split read it;
    - the scoped families are recorded as not_run by ruling, with both row counts.
- **The queue lane L-v5-2gpu** (a subagent in its own worktree, branch `v5-2gpu`). Spec:
  `build/v5-h100/queue-lane-spec.md` and its 17:45Z revision.
  - Its first commit runs `AUDIT/finalize-2026-10-03/apply_h100_amendment.py` on its own DRAFT.
  - The canonical DRAFT is v5-build's; main's copy is an older ancestor.
- **Gaps:**
  - GAP-DEVMAP-FTS5-CORRUPT-SEARCH-REFUSES-2026-10-03: devmap_search fails, so lookups fall
    back to rg;
  - GAP-RUNTIME-WITH-BACKEND-RUNS-NO-TRAINED-FAMILY-CHECK-2026-10-03;
  - the lane's two DevMap and ListAgents gaps.
- **The human's answers, still needed:**
  - the cost yes: about $310 at the per-GPU rate, $344–400 for the box;
  - R9: waive or keep;
  - C1, C2b and C3 off;
  - the J5′-before-arm interleave;
  - the setup downloads (70 wheels, plus causal_conv1d's build);
  - VitaminC;
  - the box-CPU fallback;
  - the 300 diffs.
- **Box setup files** (not committed; `build/` is ignored): `build/v5-h100/h100_setup.sh`
  (preflight with no download, then the pinned venv) and `requirements-x86.txt`. They move into
  the repo with the deploy.

## The first command for the next lane

```bash
ssh -i /Users/bharath/.ssh/bharath_m5_macbook_pro.pem -o BatchMode=yes ubuntu@68.209.74.244 'ls /home/ubuntu/queue; tail -n 5 /home/ubuntu/logs/q-v5lane0.log /home/ubuntu/logs/q-v5lane1.log'
```
