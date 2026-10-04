# Lead lane, 2026-10-03: making the pipeline robust, and the 2× H100 box for v5

This lane follows the human's ~15:41Z message: "Download needed, ask Fable for the rest; make the
training pipeline robust and purpose-built." It also covers the human's GPU answer, given at
~16:26Z: "Go with 2× H100 (Lambda)". Fable's rulings are in
`AUDIT/finalize-2026-10-03/fable-pipeline-ruling.md`, and the human-facing state is in
`AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md`.

## ~23:00Z 2026-10-04: the post-queue session is armed to launch when both v5 lanes end

The human, in reply to the post-queue session's yes/no: "ask fable for advice". Fable's ruling
(~22:45Z) was **yes, under that delegation**, on these terms:
- each control is capped at 14,400 s, and no step starts after 79,200 s;
- a box-side launcher starts the session when the lanes end (lane 0's J5′ s2, ~06:00Z), so the
  start does not depend on this Mac session being awake;
- the session uses the queue's `qd-prep-v5`. One binary per session: be65fa5's binary is not
  deployed for v5, since it is bit-exact but only 2.6% faster;
- GPU 1's idle window, from J5′ s1's end (~04:30Z) to the launch, is accepted;
- the merge of `v6-linfit-pool` (7f0e3d1) into main waits for the human;
- the human is told with a push that includes the cancel path. The ~7 h before the launch is
  their window to object.

**Done** [V]:
- `build/post-queue/main.bundle` holds main at 4ebb790, sha256 `e2a7dd10…`. It is on the box,
  checked by list-heads and sha256.
- `post_queue.sh check 4ebb790… --cpus <even>` reported 0 missing, with verdicts-s4 present.
- `build/post-queue/pq_launch_after_lanes.sh` is git-ignored tooling, sha256 `cb1c96c7…` at both
  ends. It runs in three stages:
  1. It polls `/home/ubuntu/queue/v5lane{0,1}.done` every 60 s, for at most 12 h. If
     `/home/ubuntu/post-queue/LAUNCH_CANCEL` exists at any poll, it exits 5 and launches
     nothing.
  2. It runs `check`. A non-zero exit makes it exit 6 with no launch.
  3. It runs `run 4ebb790… --ctl-cap-s 14400 --session-cap-s 79200 --cpus <even>
     --cache-option-control -- --split-cache …/split-cache-ctl --split-cache-shards
     …/v5-data-2026-10-03`, appending to `q-post-queue.log`. Exit 3 (a refused precondition,
     such as a lane script still exiting) is retried every 120 s, at most 15 times. Any other
     exit ends the launcher.
- `build/post-queue/test/pq_launch_test.sh`: 13 passed, 0 failed.
- The dry print at the launcher's flags reads: "caps: each control 14400 s at nice 10 on cpus
  0,2,…,50, each parity run 1800 s on GPU 0, no step starts after 79200 s; ceiling 1560 min =
  ~$217".
- **Launched at 22:57:30Z** (pid 1406115). Its log, `/home/ubuntu/logs/q-post-queue-launcher.log`,
  reads "launcher: waiting for … v5lane0.done and v5lane1.done".

**Cost** [I]: the session's expected ~15.5 h is ~$130, with a ceiling of ~$217 at $8.38/h. The
box stood at $225.23 at 22:37Z (`watch_h100.sh`). The lanes to ~06:00Z add ~$59 and the session
~$130, so the box ends near ~$415, with a ceiling near ~$500.

**Cancel** before ~06:00Z, with either of these:
- `touch /home/ubuntu/post-queue/LAUNCH_CANCEL`;
- `kill 1406115`.

**Queue at 22:59Z** [V]:
- J5′ s1 is on GPU 1 at step 1,380 of 12,176 (1.81 s/step, ETA 19,583 s, so ~04:25Z plus eval).
- v5traj-s4 is on lane 0. Step 3,000 finished at 22:55:07Z (row bfb6bd32). GPU 0 is at 0% by
  design, because each snapshot rebuilds the split on the CPU.
- J5′ s2 follows on GPU 0.

## ~22:25Z 2026-10-04: early attempt 4 (be65fa5's qd-prep) is bit-exact on v5's control and only 2.5% faster; v5 s4 and J5′ s0 scored

The human: "Ask fable for advice and work on it". Fable's rulings (~20:55Z):
- UnsafeCell parked.
- The new binary prepared with sha256, parity and a replay.
- One binary per session.
- The branch merged up to main.

**Done** [V]:
- `qd-prep` built from be65fa5 (`build/linfit-pool/musl_qdprep.sh`): sha256 `28f72e67…`, on the
  box at `/home/ubuntu/bin/qd-prep-linfit-be65fa5`. `qd-prep-v5` (`900534f1…`) is untouched.
- `post_queue.sh` gained `--qd-prep PATH` (default unchanged):
  - it refuses a path that is not an executable file before anything starts;
  - its say line names the binary and its sha256.
- `post_queue_test.sh`: 71 passed (8 new), 0 failed. The script is git-ignored tooling,
  sha256 `c67bcdb8…` at both ends.
- Rows record the engine: `recipe.control_engine_sha256` (`tools/ft_linear_control.py:1306`).
- `v6-linfit-pool` merged main at 7f0e3d1 (disjoint files; no push). The full Python suite there
  passed but for the two known worktree-only failures (`build/linfit-pool/suite-merge.log`).

**Early attempt 4** (`post_queue.sh early 97bb3e2 --attempt 4 --ctl-cap-s 14400 --nice 19 --cpus
<even> --qd-prep /home/ubuntu/bin/qd-prep-linfit-be65fa5`; 20:53:30-22:23:26Z; row 96c693fc,
guard exit 1) [V]:
- **Bit-exact on the real workload.** `build/post-queue/compare_engine_rows.py` compared
  8f895758 with 96c693fc over 4,179 leaf fields, 4,112 of them metrics. Equal in every metric,
  gate and recipe field but `recipe.control_engine_sha256`, `row_id`, `wall_clock_s` and
  `written_at`. The comparator was checked to flag a one-character change in a metric detail.
- **Speed:**

  | | attempt 3 | attempt 4 |
  |---|---|---|
  | n-gram arm (98 fits) | 4,749 s | 4,778 s (+0.6%) |
  | length arm (98 fits) | 226 s | 70 s |
  | total | 4,975 s | 4,848 s (−2.6%) |
  | the control's wall_clock_s | 5,006 s | 4,881 s |

  173 fits were faster, 9 slower. The slow ones are the large tasks that are probably
  many-class: decider.routing 1.57×, openjev.policy 1.37×, openjev.evidence 1.30×,
  intent.domain 1.19×, defect 1.06×. The class counts are unverified.
- The ~20:50Z projection (0.74–0.89×) was wrong. The ab11 table had 4- and 8-class shapes only,
  and box_ab part 1's 32-class openjev measured copy/old 1.20.

**v5 s4 scored** (a51c5bf4) [V, `read_v5_rows.py`]:
- span 0.907, choice 0.828;
- needle worst bucket 0.966, which passes;
- OOD abstained 86.1%, Wilson lower 0.803, which fails;
- permutation consistency 0.929, which fails.

R9's bar holds. Seed 4's choice accuracy and permutation consistency are the lowest of the five
seeds.

**J5′ s0** (the shuffled-label control on v5 s0's recipe, row ba375781): shuffled-label accuracy
0.2370 vs chance 0.2769, ceiling 0.3049 at z = 3, n = 2,304. **Passed.**

**Queue state:**
- J5′ s1 is training on GPU 1 (started 22:06:16Z; first step 22:17:36Z).
- v5 s4's needle control and trajectory run next on GPU 0, then J5′ s2.
- The running total is $185.36 / 44.24 GPU-h at 22:17:50Z.

## ~20:50Z 2026-10-04: the GPUs stay busy to ~05-06Z (J5′ s1, s2 queued); the linfit loss found and fixed (be65fa5)

**The queue after the current runs** [V for the rule and the recorded spend; the arithmetic
inferred from the pre-registration's estimates]:
- `box_q_v5.sh:4-17` gives the order v5 s0-4, the arm iff room, then J5′ s0-2. With no room the
  rounds are [s4 | J5′0], then [J5′1 | J5′2]. J5′ s1 and s2's inputs exist: v5 s0-2's ft rows
  and v5 s1/s2's eval rows.
- `v5_budget_ok` (`v5_common.sh`) admits a job while spent + in-flight + its estimate stay
  within the approved ~$356 / 84.96 GPU-h.
- `/home/ubuntu/queue/v5.spend` records $131.49 and 31.4 GPU-h. With the two running jobs
  (~$33 + ~$28) and two J5′ seeds (~$25 each), the total is ~$240.
- So both lanes take J5′ s1 and s2 when v5 s4 and J5′ s0 end (~22:10-23:15Z), and the GPUs stay
  busy to ~05:00-06:00Z.
- The post-queue session's `run` mode needs `v5lane{0,1}.done`, so it cannot start before then.
  It still needs the human's yes. Its clone path and started marker are absent [V].

**The linfit loss, found and fixed** (lane v6-linfit-pool; branch commits be65fa5 and 620dde4;
the chain in `HANDOFF/v6-linfit-pool-2026-10-04.md` on that branch) [V,
`/home/ubuntu/logs/q-linfit-ab{4..11}.log`]:
- Fable withdrew its ~17:57Z decision rule. box_ab3's "the loss is the team mechanism;
  UnsafeCell would not help" was wrong:
  - its `--iters 3` was dominated by the single-threaded transpose;
  - at 1 thread, phase B's gain masked phase A's loss.
- **The cause:** the hot loops indexed the W / diff snapshots through their `RwLock` read
  guards, reloading the Vec header for every nonzero. be65fa5 binds plain slices once per item,
  bit for bit: 85 Rust tests and 111 parity tests passed.
- **Deployment table** (box_ab11, 26 threads on the even CPUs, slices/old): tiny 0.06,
  openjev-k4 0.59, openjev-k8 0.72, arc-k4 0.85, defect-k4 1.09, defect-k8 1.03.
- **Projected on early attempt 3's 197 fits:** 4,974 s → 3,662–4,446 s (0.74–0.89×)
  [inferred: band ratios].
- **The residual** on defect at 26 threads (1.03–1.08) is open, and the `UnsafeCell` variant
  (the crate's first `unsafe`) is the human's choice.
- No rebuilt `qd-prep` reaches a v5 control without Fable's ruling and the human's yes.
- The box lane is closed: no further box runs in this lane.

## 19:36Z 2026-10-04: early attempt 3 complete; seed 2's OFF-arm control row 8f895758

[V, `/home/ubuntu/logs/q-post-queue-early-3.log`, the control's stamped log and the scratch row,
copied to `build/post-queue/early3-row.jsonl` (git-ignored)]:
- **Timing.** The split rebuild ran 18:04:41–18:12:57Z. The n-gram arm ran 18:12:57–19:32:31Z;
  the length arm ran 19:32:31–19:36:24Z. The row was written at 19:36:26Z and the session ended
  at 19:36:35Z. The control's `wall_clock_s` is 5,006 s, inside the 18,000 s cap.
- **Not run.** 98 of 332 tasks were fitted in each arm. The other 234, all `synth.general`, are
  `not_run` with their reasons. Examples:
  - `synth.general/access_route`: "the training split holds 0 row(s)";
  - `synth.general/account_age`: no label space every row shares.
- **The fix held.** The task after `synth.general/action_next`, attempt 2's crash, was skipped
  without error. The control log has no traceback.
- **The length arm** ran over all 98 fitted tasks for the first time at this scale, at about
  1–6 s a task (5 features).
- **The guard:** "1 completed letter control row(s) ['8f895758-…']", exit 1, as required.
  `ft_linear_control` exited 3: its pooled gate is `not_run`. The early check is complete.
- **Projection vs actual.** I had projected 7,800–12,700 s. The actual run was much shorter
  because 234 tasks are not fitted at all. The 98 fitted tasks ran ~0.65× attempt 2's fit
  times on 26 physical cores instead of 13 × 2.

**Row 8f895758 (scratch; not a gate's row; v5 seed 2, eval row 2af3c80d; main 97bb3e2):**
- `paired_margin_vs_linear.choice.code.defect_class` (the successor rule's `MARGIN_KEY`):
  **+0.3915 [+0.3707, +0.4128], ran, passed.**
- The pooled `gates.paired_margin_vs_linear` is **`not_run`**: "234 of 332 inputs did not run".
  The synth.general family margin is `not_run` too (234 of 267 inputs). Every other gate in the
  row is `not_run`: this tool measures only the margins.
- **The pooled gate's `not_run` is not new** [V]. F's three control rows carry it too:
  - `c89b89a1`, `c0e438e6` and `3f72112a` have `gates.paired_margin_vs_linear` `not_run`,
    because 1 of 7 tasks, `intent.domain/domain`, did not converge in 6,000 iterations.
  - Every uncached v5 control will carry it as well [inferred], because v5's split holds the
    synth.general tasks that cannot be fitted.
  - Rule 2 stands: nothing here moves a threshold.
- **Exit 3** is `ft_linear_control`'s "the pooled gate is not `Ran`" [V,
  `tools/ft_linear_control.py:1387`].
- **What reads the pooled gate** [V, `crates/qd-runtime/src/bin/qd_post_f_rules.rs`]:
  - `paired_ci` (:4703) refuses a gate that did not run (`row.ran("gates", …)?` at :4705).
  - Its callers are the TierB rules (:5064, the pinned phase-3 reference control; :5292, a
    candidate's control row). A TierB candidate whose control row has a `not_run` pooled gate is
    refused, closed and loud.
  - v5's successor rule reads the code.defect_class family margin (`MARGIN_KEY`, :1270), which
    ran and passed here.
  - Whether a v5 row is ever a TierB candidate is not established [unverified]; the TierB rule
    pins phase-3's `val_shard_hash`.
- **intent.in_scope: −0.0293 [−0.0452, −0.0140], ran, failed**, the whole interval below zero.
  The n-gram and the length control give the same margin there.
- **openjev.game: +0.1935 [−0.0323, +0.4194]** against the n-gram control: the CI includes
  zero (31 val rows). Against the length control it is +0.4194 [+0.1935, +0.6129].
- The other 18 of the 22 families pass against both controls, as code.defect_class does:
  - lowest: pairwise.helpfulness +0.0860;
  - highest: intent.classification +0.8295 against the n-gram control.
- `paired_margin_vs_linear.span` is `not_run` by design: the 7,238 pointer rows are not scored.

**The box's CPUs are free again.** That reopens the team-anomaly discriminators, which need
Fable's ruling (`HANDOFF/v6-linfit-pool-2026-10-04.md` on its branch).

## ~18:05Z 2026-10-04: the fix merged (97bb3e2); attempt 3 runs on 26 physical cores, cap 18,000 s

The human: "ok, merge it and launch attempt 3 after the suite".

**The merge** [V]:
- `v6-ctl-minrows` 97bb3e2 was fast-forwarded onto main (015585a → 97bb3e2) by
  `build/ctl-minrows/ff_main.sh`. Main's status was unchanged: the peer's 16 paths, none of them
  the commit's.
- The full Python suite on the branch's worktree passed except the two known worktree-only
  failures (`build/ctl-minrows/suite.log`):
  - `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this` checks the
    directory's name.
  - `test_lint_gate::test_ruff_is_installed_not_merely_declared` finds no `.venv` in a
    worktree.

  The fail-first run is in `build/ctl-minrows/failfirst.log`.
- `git diff --stat 142a67c 97bb3e2 -- crates/qd-prep` is empty, so `post_queue.sh:86`'s
  `qd-prep-v5` pin holds.
- The bundle's sha256 is `6e3843ba…5cc50`, with one head, 97bb3e2; the checksum was checked at
  both ends. The `print-early` and `check` dry-runs at 97bb3e2 found 0 required inputs missing.

**Attempt 3** [V, the launch and the clone]:
- Command: `post_queue.sh early 97bb3e2… --attempt 3 --ctl-cap-s 18000 --nice 19 --cpus
  0,2,4,…,50`.
- Started 18:04:39Z. Log: `/home/ubuntu/logs/q-post-queue-early-3.log`; root:
  `/home/ubuntu/post-queue/early-3/`.
- The clone is at 97bb3e2, and no tracked file changed.
- The session cap is 18,600 s, so the check ends by ~23:15Z at the latest.

Fable revised two of its earlier settings on box_ab2's evidence (~17:57Z):
- **CPUs: the 26 even CPUs, one per physical core, not 0-51.**
  - On the box, hyperthread siblings are the pairs (2i, 2i+1) [V,
    `/sys/.../thread_siblings_list`].
  - Attempt 3 runs the old binary, and for it 52 threads was never better than 26 [V,
    `q-linfit-ab2.log`]: defect 74.9 vs 76.4 ms/it, arc 6.9 vs 4.5, openjev 5.7 vs 3.2.
- **Cap: 18,000 s, not 10,800.** The projection comes from attempt 2's 58 timestamped fits [V,
  `build/post-queue/project_early.py`]:
  - The control has 332 tasks, not 300.
  - The 58 fitted took 5,925 s on 13 cores × 2 threads; 17 tasks of at least 60 s account for
    5,077 s of that.
  - Of the 274 left, 263 are `synth.general` at 2.5–7 s each, 10 are `typed.workflow` and 1 is
    `vitaminc.nli` (700 val rows).
  - The length arm then runs over every task. F's 7 length tasks converged in 70–1,527
    iterations (`c89b89a1`) [V].
  - Projected total: 7,800–12,700 s [inferred]. 10,800 s risked attempt 1's outcome.
- **The bound on the cap:** run mode refuses while any control is alive (`post_queue.sh:194`).
  - At 18:01Z the lanes' current runs (v5 s4, J5′ s0) had training ETAs of 22:05–22:15Z, with
    scoring after [V].
  - The queue's later jobs (J5′ s1, s2) are inferred, not verified.
  - So attempt 3 should end before the lanes do. If no further job is queued and scoring takes
    under an hour, the worst case overlaps the lanes' end by minutes.
- **The length arm** has run through the native engine only on F's 7 tasks [V: `c89b89a1`, and
  the Mac refit rows `d877d6bc`, `84510263` and `27a263d5` reproduce its margins]. It has never
  run over 332 tasks or on tiny ones.
  - The fix covers both arms (`fit_task`).
  - A late failure there is a first run of that arm, not a regression of the fix.

**The team's large-shape anomaly (lane v6-linfit-pool), narrowed** [V, `q-linfit-ab2.log`,
`q-linfit-ab3.log`; `build/linfit-pool/notes.md`]:
- box_ab2: nice 10 vs 19, and 26 vs 52 threads, made no difference. It is neither the nice
  level nor oversubscription.
- box_ab3, defect-k8: the copy variant against old was 0.95 at 1 thread and 1.15 at 4.
  - `with_team(1)` runs inline: it starts `threads - 1` workers (`team.rs:9,266`).
  - So the per-element code (the atomic cells) is not the cause; the team mechanism is.
    **(Wrong; corrected at ~20:50Z, the section above.)**
- Per Fable's decision rule, the `UnsafeCell` branch (`v6-linfit-pool-cell`) would not help and
  stays on hold. **(The rule was withdrawn at ~20:37Z; see the section above.)**
- The next diagnostics need the box's CPUs, which attempt 3 holds until ≤23:15Z.

**Next:**
- Watch attempt 3 with `build/post-queue/mon_early3.sh`, which counts each arm.
- Check that the lanes' step rate holds near 1.81 s/step.
- At the end, read the scratch row: `existing_controls.py` must exit 1.
- The post-queue session itself still needs the human's yes.

## ~17:45Z 2026-10-04: attempt 2 crashed on a 3-row task; the post-queue controls would too

The human: "yes, launch attempt 2 at 15:50Z"; later "Follow fable's advice and work on it",
"yes, run the box bench", and "yes, run the box A/B; don't ask me, ask fable for advice".

**Attempt 2 of the early check** (`post_queue.sh early bbed808 --attempt 2 --ctl-cap-s 10800
--nice 19 --cpus 0-25`, launched 15:50:11Z; CPUs 0-25 are 13 cores x 2 hyperthreads, half the
box) [V, `/home/ubuntu/post-queue/early-2/`, its timestamped logs]:
- The split rebuild ended 15:58:46Z. It fitted 58 tasks by 17:37:48Z. The first 14 (attempt 1's)
  took 2,511 s of fits at 26 threads against attempt 1's 1,428 s at 52.
- **Then it crashed on task 59, `synth.general/action_next`'s successor:**
  `linear_control_native.fit` raised `ValueError: need at least 4 examples to fit and validate,
  got 3`. The run recorder wrote row `2646e366` (`failed`). The guard counted 0 completed rows,
  and the early check said FAILED: the fix of ~14:55Z did its job.
- **The bug:** `fit_task` (`tools/ft_linear_control.py:439`) documents "Never raises for a weak
  fit" and returns `not_run` for a single-class task, but not for one with fewer than 4 training
  rows, which `LinearBaseline.fit` refuses. v5's 235 `synth.general` tasks include some that
  small, so **every uncached v5 control at bbed808 dies at this task.** The post-queue session
  is blocked until the fix is merged into main.
- **What the fix changes in the gates** [inferred, from the code's docs]: the task becomes
  `not_run` with its reason; the pooled `gates.paired_margin_vs_linear` is `not_run` whenever
  any task is, as its docstring already says. v5's successor rule reads the code.defect_class
  family margin (`MARGIN_KEY`, `crates/qd-runtime/src/bin/qd_post_f_rules.rs:1270`); the pooled
  gate is read by the TierB outcome rule's clause 2 (`paired_ci`, `:4696`), which judges a TierB
  control row. No threshold moves.
- **Fable's rulings:** fix in `fit_task` before the single-class check (the reference's refusal
  order), with a fail-first test, on its own branch, merged with fbd43f5's mechanics (local; the
  human delegated the call to Fable). Attempt 3 (scratch ledger, CPU only, capped, ~$0) runs
  under that delegation and is reported. Real-ledger writes still need the human's yes.
- The length-only control runs over every task after the n-gram one; no attempt has reached it.

**The control's per-iteration floor** (lane `v6-linfit-pool`, `HANDOFF/v6-linfit-pool-2026-10-04.md`
on that branch): on the box a thread start costs ~32 us a thread a phase, ~3.4 ms an iteration
at 52 threads. A team of threads per fit (bit for bit against the Python reference) cuts small
shapes 2-2.3x at 52 threads, but loses 10-39% on a large shape. Not deployable yet; diagnosis
continues.

## ~15:00Z 2026-10-04: v6-ctlcache merged (bbed808); v5 seeds 0 and 1 scored; the early control check hit its cap

The human, ~13:40Z: "yes to both, merge and run the check; ask fable for more advice". Later:
"ask fable what to do about the control cap".

**The merge.** `bbed808` merges `v6-ctlcache` (03d335d..7099cdb) into main, following fbd43f5's
mechanics (scripts in ignored `build/merge-ctlcache/`).
- There were no conflicts.
- Main was fast-forwarded from f62d7ca, and the 16 uncommitted paths in main's tree were
  unchanged across it.
- Suite on the merged tree [V, `build/merge-ctlcache/suite.log`]: 4,335 passed, 68 skipped,
  2 failed.
  - Both failures are worktree-only: `test_gaps_writer` (it needs the checkout's name) and
    `test_lint_gate::test_ruff_is_installed_not_merely_declared` (it needs `.venv/bin/ruff`).
  - Rerun in main with `test_gaps_ledger`: 32 passed [V, `build/merge-ctlcache/main-recheck.log`].

**v5 seeds 0 and 1 scored** [V, `read_v5_rows.py` and `build/v5-h100/read_row_detail.py`]:

| | v5 s0 (166f7ebd) | v5 s1 (e1bafd6f) | v5 s2 | v5 s3 |
|---|---|---|---|---|
| val_top1.span | 0.911 | 0.915 | 0.913 | 0.908 |
| val_top1.choice | 0.840 | 0.846 | 0.839 | 0.847 |
| 8K needle worst bucket (gate 0.95) | 0.918 fails | 1.000 passes | 0.705 fails | 0.967 passes |
| OOD abstained of 180 (Wilson lower vs 0.9) | 87.8% (0.822) | 90.0% (0.847) | 84.4% (0.784) | 86.7% (0.809) |
| in-distribution abstain (Wilson upper vs cap 0.05) | 6.0% (0.064) | 6.3% (0.067) | 6.5% (0.069) | 5.6% (0.060) |
| permutation_consistency (floor 0.95) | 0.946 | 0.945 | 0.943 | 0.949 |

R9's bar holds on all four seeds. Rule 2 stands: these are readings, and ood_abstain and
permutation_consistency fail on every seed. Both seeds' batch-order digests match the prelude,
and their needle controls exited 0. Both trajectories started at 14:45 and 14:47Z.

**The early check, attempt 1: killed by its cap.** It ran `post_queue.sh early bbed808…` from
14:10:15Z. That is seed 2's uncached letter control, in its own clone, writing to a scratch
ledger.
- **What happened** [V, the box's `/home/ubuntu/post-queue/early/`]:
  - `timeout` killed it at 14:50:16Z with exit 124, after 14 of the control's 300 tasks.
  - No `qd-prep` or control process survived.
  - Only the scratch ledger was written.
- **A false pass, found and fixed.** The tool's run recorder wrote a row even for the killed run:
  `5231c9a0`, status `killed`, only `code_that_ran` in its metrics, every gate `not_run`.
  - `existing_controls.py` counted any row naming the tool and eval row, so the early check said
    "complete".
  - The same guard would have skipped the post-queue rerun over a killed row in v5's ledger.
  - It now counts only `status == "completed"` rows, the successor rule's own criterion
    (`qd_post_f_rules.rs:615-617` and `:1803-1820`), and lists the others. `compare_control_rows.py`
    uses it, and the early check fails without a completed row.
  - `post_queue_test.sh` passes 63/63. The 4 killed-row cases failed first. The re-run guard on
    the box reads 5231c9a0 as "not counted (killed)", exit 0.
- **Why 2,400 s was wrong.** It came from F's letter control: c89b89a1 on the GH200's
  `gh200-p4-v4-2026-10-01.jsonl` covered 7 tasks and 10,985 rows in 434.5 s [V, read-only].
  v5's control covers 300 tasks and 17,254 rows; its decisions pool adds 36
  `procedural.decisions` and 235 `synth.general` tasks.
- **Fit time is set by iterations, not size.**
  - arc.science: 3,178 rows, 201 s, at ~6,100 iterations for l2 = 1e-4.
  - openjev.game: 971 rows, 72 s.
  - knowledge.multiple_choice: 13,124 rows, 14 s.
  - code.defect_class: 71,435 rows, 489 s.

  The 14 fits took 1,428 s of the run's ~2,400 s. The rest was the split rebuild plus Python
  between fits, and it is untimed per task because the lines carry no time.
- **Contention.** The two needle-control startups ran beside it at 14:31-14:41Z: MinHash took
  116.9 s against 89.2 s (+31%) and LSH 41.1 s against 32.2 s (+28%). Training's step rate held
  at 1.84 s/step (14:10-14:28Z).

**Fable's rulings (~14:48Z):**
1. **No new cap without a measurement.** Rerun the OFF arm as attempt 2 with:
   - a 10,800 s cap, nice 19, `taskset -c 0-25`;
   - every log line timestamped;
   - its own root (`$PQ/early-2/`, `qd-post-queue-early-2`), so attempt 1's evidence stays.

   Launch it at ~15:50Z, when both lanes are back in training. All 13 snapshots per seed are kept
   under `ckpt/v5/` [V], so a slowed trajectory costs later GPU time, not data. It needs the
   human's yes.
2. **Fable withdraws its ruling "controls after the queue, not during".** That ruling assumed
   ~6 min per control. At ~1 h per control, 16 controls are ~16 h of box time with both GPUs
   idle (~$130).
   - The ledger concern is answered: `Ledger.append` flocks (`ledger.py:1253-1274`), and
     `Ledger.rows` is an unlocked read (`:1186-1195`).
   - Contention is bounded by the core restriction.
   - Running the real controls during the queue's training windows needs the human's yes, asked
     with attempt 2's measurement.
3. **`CTL_CAP_S` and `SESSION_CAP_S` are operational caps,** not gates; rule 2 is untouched. They
   are set from the measurement. `post_queue.sh` now takes `--ctl-cap-s`, `--session-cap-s`,
   `--nice`, `--cpus` and (early only) `--attempt`, and states a computed ceiling.
4. **The solver is recorded, not changed.** A warm-started l2 path in `qd-prep linfit` (0.1 →
   1e-4, each fit starting from the last) is a v6 target. It needs a committed parity test against
   the current solver, and it is not for v5's rows, whose gate opponent it would change.

**Open:** GAP-DEVMAP-FTS5-CORRUPT-2026-10-04. DevMap's search failed this session on an fts5
corruption, so the no-lock and only-`--write-ledger` claims are rg- and read-confirmed.

**Next, once the human says yes** (dry-run on the box: `print-early`, `check --cpus 0-25`, 0
missing). Alone in its ssh call:
`nohup setsid bash /home/ubuntu/post-queue/post_queue.sh early bbed80896d676b18baffa64bcd69dd4ae2b2bea3 --attempt 2 --ctl-cap-s 10800 --nice 19 --cpus 0-25 > /home/ubuntu/logs/q-post-queue-early-2.log 2>&1 < /dev/null &`

## ~12:20Z 2026-10-04: v6-ctlcache reported; Fable's rulings; the merge waits on the human

The lane's work is on branch `v6-ctlcache`, six commits, the change being 5602617. Its report is
`HANDOFF/v6-ctlcache-2026-10-04.md`.
- **The change:** `tools/ft_linear_control.py` takes `--split-cache DIR` and
  `--split-cache-shards DIR`, both letter and option controls.
- **The test:** fails first at 0dd472c.
- **Measured on the Mac:** getting v5's split took 221 s with the cache empty (MISS) and 9.9 s
  from it (HIT). Peak memory was 26.8 GiB on the MISS and 3.77 GiB on a HIT.

I read the diff. Without the flags the tool behaves as before. It keys the cache through
`real_ft_run`'s own `ft_split_rows` and `split_rebuild_inputs`, and it checks the shard set before
rebuilding.

**Fable's rulings:**
1. **The option row's cache metric** is named `linear_option_control.split_cache`. Accepted: the
   bare name would mean relaxing a test. `compare_control_rows.py` now picks the name by row kind
   and pops the dotted key whole.
2. **The shard-set refusal.** The control now refuses unless the shard set's `train.json` names the
   eval row's `data_snapshot_hash`. Accepted: it fails closed in 0.8 s.
   - **Verified on the box, read-only** (`build/post-queue/probe_snapshot.py`): the shard set and
     the eval rows of seeds 2 (`2af3c80d`) and 3 (`e09b652a`) all carry `cb4b7d3a…a7b5`.
   - **Inferred, not verified:** seeds 0, 1 and 4 and the arm come from the same `--out`.
   - If a later seed differs, its cached control refuses loudly and the summary shows 0 rows for
     it.
3. **Guard files.** The post-queue session calls the tool directly and guards by ledger rows
   (`existing_controls.py`). It never goes through `controls_block`, so the stale
   `*.letter.started` markers do not stop it.
4. **The merge goes into main,** which the post-queue session clones. It follows fbd43f5's
   mechanics: a temp worktree, `--no-ff --no-commit`, `gaps.jsonl` union-resolved, the full suite
   with datasketch, then `--ff-only`. **Not merged yet:** the human's yes is pending.

**After the merge, the launch line:**
`run <merge-sha> --cache-option-control -- --split-cache /home/ubuntu/post-queue/split-cache-ctl --split-cache-shards /home/ubuntu/v5-data-2026-10-03`.
The test now passes both flags, and its fake control refuses one flag without the other, as the
tool does. 35 of 35 checks pass.

**Phase A's memory:** three rebuilds run side by side, so the lane's "run first calls alone"
warning needs an answer:
- OFF, ~27 GiB;
- seed 0's letter MISS, ~27 GiB plus its store;
- parity A's `real_ft_run` rebuild, of a similar order.

That is under ~100 GiB of the box's 442 GiB, after the queue has ended.

**One gap, not two:** the lane's "not end to end on real rows" and this session's "real tools on
real rows" are the same gap. Only a real control on a v5 eval row retires it: either the early
OFF-arm check (pending the human's yes) or the session itself.

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
ssh -i /Users/bharath/.ssh/bharath_m5_macbook_pro.pem -o BatchMode=yes ubuntu@68.209.74.244 'ls /home/ubuntu/queue; tail -n 5 /home/ubuntu/logs/q-v5lane0.log /home/ubuntu/logs/q-v5lane1.log /home/ubuntu/logs/q-post-queue-launcher.log'
```
