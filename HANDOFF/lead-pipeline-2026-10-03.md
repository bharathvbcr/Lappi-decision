# Lead lane, 2026-10-03: making the pipeline robust, and the 2× H100 box for v5

This lane follows the human's ~15:41Z message: "Download needed, ask Fable for the rest; make the
training pipeline robust and purpose-built." It also covers the human's GPU answer, given at
~16:26Z: "Go with 2× H100 (Lambda)". Fable's rulings are in
`AUDIT/finalize-2026-10-03/fable-pipeline-ruling.md`, and the human-facing state is in
`AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md`.

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

## What is open

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
tail -n 20 /Users/bharath/Code/research/Lappi-decision/build/dedupe-probe/residual.log
```
