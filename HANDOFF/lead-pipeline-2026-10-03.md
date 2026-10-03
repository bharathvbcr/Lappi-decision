# Lead lane, 2026-10-03: making the pipeline robust, and the 2× H100 box for v5

This lane follows the human's ~15:41Z message: "Download needed, ask Fable for the rest; make the
training pipeline robust and purpose-built." It also covers the human's GPU answer, given at
~16:26Z: "Go with 2× H100 (Lambda)". Fable's rulings are in
`AUDIT/finalize-2026-10-03/fable-pipeline-ruling.md`, and the human-facing state is in
`AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md`.

## State at ~23:58Z: v5 LAUNCHED (supersedes everything below where they disagree)

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
