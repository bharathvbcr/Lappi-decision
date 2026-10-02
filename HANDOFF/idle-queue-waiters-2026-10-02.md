# HANDOFF: L-waiters, the idle-GH200 queue's waiters (2026-10-02)

**Lane:** L-waiters. **Branch:** `worktree-agent-a2e3d82a191861570`, merged with main at 0077cd3.

**Sources:**
- Fable's idle-queue ruling: `AUDIT/idle-gpu-queue-2026-10-02/fable-idle-queue.md` (Q1, Q2, Q4, "Lane tasks").
- Fable's J6(a) +replay ruling: `AUDIT/idle-gpu-queue-2026-10-02/fable-j6a-replay.md`.
- Pre-registrations: `campaign/f-successor-preregistered.json` and `campaign/j6a-preregistered.json`.

**Status:**
- Every waiter is written, dry-run and staged on the box. **None has been launched.**
- The lead launches them. j6a needs four pins first (see "Before launch").
- Labels used below:
  - **[V]** verified by a run or a read cited here;
  - **[I]** inferred;
  - **[U]** unverified.

## The chain

```
rung0.done -> cudadev -> rungd -> fsucc -> j5pp -> j6a -> STOP
j6ctl (CPU, niced) waits on j6f.done + j6dv4.done.
fsucc also waits on j6ctl.done and on '=== F controls all done' in q-fcontrols.log.
```

Each waiter does the following. "Writes" lists what it writes on the box; ".done" is the marker its EXIT trap touches.

- **j6ctl**
  - Runs on CPU only. Each control runs as `nice -n 10`.
  - Uses qd-lane10 at 3460afc with `qd-prep-m1` (`9ccaf5aa`), in `box_f_controls.sh`'s form.
  - Runs a letter control, then an option control, for J6(d)-v4 and then for J6(f)-v4.
  - A guard marker stops a second letter control from starting.
  - Writes: rows to `gh200-j6-v4-ablations-2026-10-01.jsonl`.
  - .done: `j6ctl.done`.
- **cudadev**
  - Runs runga, then 11 device-test binaries, each `--ignored --test-threads=1 --nocapture`.
  - Holds gpu.lock. Every binary is sha-pinned and has its own timeout.
  - Total cap 1,500 s = $0.95 at $2.29/h.
  - A failed test is recorded and the next binary still runs.
  - Writes: `/home/ubuntu/ojas-cuda/cudadev-<UTC>/summary.tsv`.
  - .done: `cudadev.done`.
- **rungd**
  - Runs T-bf16 (cap 3,600 s), then T-fp32 (cap 10,800 s).
  - Runs from the new overlay `/home/ubuntu/perf/overlay-rungd` at main 01b6db2, which is verified clean.
  - Writes: `gh200-rungd-torch-2026-10-02.jsonl`.
  - .done: `rungd.done`.
- **fsucc**
  - Runs `qd-post-f-rules-succ successor` (be020297) and accepts exactly `fires:j6f`, `fires:j6dv4`, `quiet` or `refused`.
  - On a fire: three F′ seeds under `approved()`, then their needle and linear controls.
  - Writes: `$Q/fsucc.word` and the decision JSON in `idle-decisions-2026-10-02/`.
  - .done: `fsuccgpu.done` (the GPU part) and `fsucc.done`.
- **j5pp**
  - Runs only if fsucc fired and wrote three completed F′ ft rows (checked by `qd-post-f-rules ft-rows`). Otherwise it logs "j5pp SKIPPED".
  - When it runs: three shuffled-label seeds with F′'s recipe.
  - Writes: the F′ arm's ledger.
  - .done: `j5pp.done`.
- **j6a**
  - Runs only if fsucc's word is `quiet` and every pin checks. Otherwise it logs that it was deferred.
  - Runs the prelude first, then J6(a) at seed 0, then its needle control.
  - Writes: `gh200-j6a-v4-2026-10-02.jsonl`.
  - .done: `j6a.done`. **STOP**: nothing runs after it.

## What is staged on the box [V]

Every target path was absent before the copy (read-only ssh, 2026-10-02 ~03:3x UTC). The one pre-existing directory among the targets is `/home/ubuntu/ojas-cuda`, which rung 0 created; cudadev writes only to a new `cudadev-<UTC>` subdirectory of it.

The copies were made with `scp -p`. Afterwards `sha256sum` on the box printed exactly the sha256 values below.

### Scripts in `/home/ubuntu/post-f/`, from `campaign/post-f-queue/` at d8e45e4

| File | sha256 |
| --- | --- |
| `idle_common.sh` | `1a449f7aa3db3072f41f1b1a2a47f19b41780d58b51d0b36cbf794edf9004c6a` |
| `box_q_j6ctl.sh` | `6055800ce28573a5568a2aae24c8897951575633cbfcba6fd21ee41ef764a536` |
| `box_q_cudadev.sh` | `c061e824a323e2e6bf51ef13ce145facf9d0c337f1dbe30221dfd202a61284b9` |
| `box_q_rungd.sh` | `8c2488896b415403d69b373790778758e811e39342acef878f8d0792f587e077` |
| `box_q_fsucc.sh` | `9bbabced3c354ecafa11985033f1b3fd2914c235d46c2decdcd335f0aee757dd` |
| `box_q_j5pp.sh` | `0eb4a5f797d354c2db5776b9fb1bb9f5f8637e9d977ca06dfaec666991dd6409` |
| `box_q_j6a.sh` (pins UNSET) | `6d2caea7eec05629ee1639f2738de15a477b67788f4084767ae7b974fc040af9` |
| `box_mk_rungd_overlay.sh` | `e76798f786dacb0215fc5996ce5f2c8ae946a0f25a9262d63eca93aa7d2662ac` |
| `j6a_prelude_box.py` | `415d09eb799b50e9d756ec1527dcf193b01c93c510634b97ff6673848ec96e70` |

### Binaries in `/home/ubuntu/bin/`, all new names

| Box path | sha256 | Source |
| --- | --- | --- |
| `ojas-qwen35-cuda-runga` | `bb3276e7a42ae3d0e20171ed78e0df537f2581d8f144d22b8b6c054a0814c805` | one ojas cross-build (below) |
| `ojas-cuda-device-k0` | `b10542091a32f9bf293d1cefd36d754221115f5ed88386c1b2e3f970fa9820a6` | same build |
| `ojas-cuda-device-k1` | `9546b2df0da4334ab735673885b9812a9f02ca4021172b60fb4c1f6b9eaa93bd` | same build |
| `ojas-cuda-device-k8` | `75f05d3a42c7964fc0143dc5149a02d37b2af04aaca3f7ea271cefcdb8074ff1` | same build |
| `ojas-cuda-device-k11` | `bc65b3bca3c48f1814185fc74859e6c623cffceda5bf3f9736d9b8b82c72ad90` | same build |
| `ojas-cuda-device-k3-gates-published` | `fd4e1e2410b2b8ae11176aa0b3d3f75a469b48fb0c8a4387c7f17fe27819bfbb` | same build |
| `ojas-cuda-device-k4-conv1d` | `eb12cb197f5d259a4b3bf79396f247f431c222461afbdb8191a763bf70e4b666` | same build |
| `ojas-cuda-device-k6-qk-norm-rope` | `03a641f055494f8c68bb1f8eb320906fe892010e59fe3ac2375037f5498611fe` | same build |
| `ojas-cuda-device-k7-rmsnorm` | `c56c71146f7af183c3af1a3252e51818705d1a61e24d0eef7cdfc3440bd095e2` | same build |
| `ojas-cuda-device-k9-embed` | `363a21f68eefd664cf72bb8af6ce78be8a43b4fa99ff0ad4d9b0f4d1053f6d8b` | same build |
| `ojas-cuda-device-k10-ce-rows` | `54c8755464b913adb407c74c52ebfb6d421da0e39e844af34cba61950a5c707e` | same build |
| `ojas-cuda-device-gdn-published` | `97034105198442010a12661228442197a9d20a3eb7685480e239daebdb8f4f28` | same build |
| `qd-post-f-rules-succ` | `be02029779eca1510c75f24e22c23cf7d762f1d87bbbca08b238ff9aede5eb62` | L-room's `/Users/bharath/qd-campaign/target-aarch64-linux-succ-room/aarch64-unknown-linux-gnu/release/qd-post-f-rules`, copied as is (1,278,720 bytes) |

- `/home/ubuntu/bin/qd-post-f-rules` was not touched: it is still `cadbdc74eabdc26a2395bd31cb20eab35152c958025af46ba50ef30365111339` [V].
- **The ojas build:**
  - one `--locked --offline` aarch64 cross-build into the new target dir `/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-idle`;
  - sha256 of every source file recorded;
  - no git run in ojas.
  - It reproduced runga `bb3276e7` and L-cuda-M1's 11 device binaries byte for byte.
  - Evidence: `AUDIT/idle-gpu-queue-2026-10-02/waiters-ojas-build.log` [V].
- `device_gdn_published_mirror` was built but not staged. It is 8 host tests (no `#[ignore]`) that the GDN lane already ran on the Mac (`HANDOFF/ojas-l-cuda-gdn-2026-10-01.md:29,137`), not a device test [V].

### Bundle and overlay

- `/home/ubuntu/perf/rungd.bundle`
  - sha256 `bfa05d752a744b7d56cfddc0402fc0f81fb3f1e45384c6f1a67a578439804245`;
  - tip `01b6db2fbaba1d149bb585940e6436308acfa4a5 refs/heads/main`;
  - `git bundle verify` printed "is okay" on the box [V].
- `/home/ubuntu/perf/overlay-rungd` was built on the box, CPU only, with
  `nice -n 10 bash /home/ubuntu/post-f/box_mk_rungd_overlay.sh 01b6db2fbaba1d149bb585940e6436308acfa4a5`.
  - It printed `overlay /home/ubuntu/perf/overlay-rungd at 01b6db2…; 15 ignored data entries copied from /home/ubuntu/qd-lane8; dirty=[]` and then `overlay check OK`.
  - `.git/HEAD` = `01b6db2fbaba1d149bb585940e6436308acfa4a5` [V].
  - 01b6db2 contains 055a7ec. tools/ and python/ have no diff between 28000af and 01b6db2 (`git diff --stat`) [V].

### Box dependencies the waiters pin, unchanged [V]

| File | sha256 (prefix) |
| --- | --- |
| `post_f_common.sh` | `4867546e…`, equal to the repo's |
| `qd-prep-m1` | `9ccaf5aa…` |
| `perf_mkoverlay.sh` | `0b8860b9…` |
| `/home/ubuntu/box_q_f.sh` | `a1bab9b0…` |

Two more facts:
- `f.ckpt-skip` = 6.
- cudadev's `LD_LIBRARY_PATH` directories (`qd-venv/.../nvidia/{cublas,cuda_nvrtc}/lib`) exist and hold `libcublas.so.12` and `libnvrtc.so.12`.

## Before launch (lead)

1. **The successor binary smoke,** as the lead planned: `/home/ubuntu/bin/qd-post-f-rules-succ; echo $?` must print usage and `2`.
   - It ran on aarch64 in the container: usage exit 2; `seeds34` on J4 → `fires`, exit 0; `successor` on F's seed-0 snapshot → `refused`, exit 3 (`waiters-dryrun.log`, "be020297 smoke") [V].
   - It has **not** run on the box [U].
2. **j6a's four pins.** `box_q_j6a.sh` carries `J6A_DATA`, `REPLAY_DIR`, `ATTESTATION` and `ATTESTATION_SHA256` as the literal `UNSET`.
   - While any is UNSET, it logs `j6a deferred: pins UNSET: …` and touches `j6a.done`. Trains nothing.
   - Fill them by commit from L-replay's HANDOFF after build 2, decontam 2 (CLEAN), the rsync without held-out data, and the box prelude.
     - The plan names `/home/ubuntu/phase4-v4-replay-2026-10-02`, `…/shards/replay` and `…/replay-attestation.json`.
   - Then stage the filled copy and record its new sha256.
   - **Never edit the file while its waiter runs.** Bash reads a script as it executes it.
   - j6a waits only on `j5pp.done`, so it can be launched any time after the pins are filled, even after j5pp finished.
3. **The replay_log reading is a decision, and it is not blocking.**
   - `campaign/j6a-preregistered.json` (`report_only_requirement`) asks for `replay_log` for the first and last 20 replay steps, beside the concurrent CE.
   - An untouched a502670 run cannot print it [V, by read]:
     - `replay_log` lives in memory only (`python/qd_train/replay.py:478,499`);
     - the printed report carries `replayed`, `replay_kl_first` and `replay_kl_last` (`tools/real_ft_run.py:2457-2459`, printed at `:8825`);
     - `PriorKLReplay.state()` persists counts, not the log (`replay.py:517-528`).
   - So the waiter prints what the run does emit:
     - those keys;
     - the checkpoint's `loss_log` first and last 20 (the per-optimizer-step loss without the replay term) [I, from `replay.py:503-511`];
     - the replay counters.
   - It also logs one line saying the requirement is **NOT MET by this run**.
   - The options, which are the lead's or Fable's call, are in `GAP-J6A-REPLAY-LOG-NOT-EMITTED-AT-A502670-2026-10-02`:
     - accept what is printed;
     - a print-only observer around `main()`, whose ft row would still say `code_commit a502670`;
     - an amendment.
   - L-waiters did **not** wire an observer.
4. **The human: a confirmation, not a question** (Fable Q4).
   - About $80–95 in total if `fsucc` fires: three F′ seeds and three J5″ seeds, each capped at $20.61 under the existing words (`post_f_common.sh:69-70`).
   - About $19 if it is quiet.
5. **Marker check.** This must print nothing; it printed nothing at staging time [V]:
   `ls /home/ubuntu/queue | grep -E '^(j6ctl|cudadev|rungd|fsucc|fsuccgpu|j5pp|j6a)[.-]'`

## Launch commands, in launch order

All six can be launched together, because every one waits on its predecessor's marker.
- rung 0's waiter is already queued (`rung0.queued`).
- j6f and j6dv4 are already queued.
- j6a can go now (it records its deferral) or later, once its pins are filled.

```
nohup setsid bash /home/ubuntu/post-f/box_q_j6ctl.sh > /home/ubuntu/logs/q-j6ctl.log 2>&1 < /dev/null &
nohup setsid bash /home/ubuntu/post-f/box_q_cudadev.sh > /home/ubuntu/logs/q-cudadev.log 2>&1 < /dev/null &
nohup setsid bash /home/ubuntu/post-f/box_q_rungd.sh > /home/ubuntu/logs/q-rungd.log 2>&1 < /dev/null &
nohup setsid bash /home/ubuntu/post-f/box_q_fsucc.sh > /home/ubuntu/logs/q-fsucc.log 2>&1 < /dev/null &
nohup setsid bash /home/ubuntu/post-f/box_q_j5pp.sh > /home/ubuntu/logs/q-j5pp.log 2>&1 < /dev/null &
nohup setsid bash /home/ubuntu/post-f/box_q_j6a.sh > /home/ubuntu/logs/q-j6a.log 2>&1 < /dev/null &
```

## j6a, as rewritten for the J6(a) pre-registration (d8e45e4)

- **Run condition:**
  - fsucc's word is `quiet` and every pin checks. Otherwise it logs why and touches `.done`.
  - A word that is not quiet gives exit 0 and also logs any UNSET pins.
  - A pin failure gives exit 3. A prelude failure gives exit 5.
- **Pin checks:**
  - `J6A_DATA` is not F's own `--out`.
  - There is **no** `$J6A_DATA/data/heldout` (rule 3; the plan's rsync excludes it).
  - These files exist: the train header, `data/pool/train-replay.json`, the replay header and the attestation.
  - The attestation sha256 equals `ATTESTATION_SHA256`.
  - The attestation has `clean: true`, hits on both `val` and `heldout` that are all 0, and `replay_rows_checked == replay_rows_total >= 1` (the plan's decontam-2 criteria).
  - The prelude's sha256 equals its pin.
- **Fixed from the pre-registration:**
  - `--replay-weight 1.0 --replay-every 6`;
  - ledger `/home/ubuntu/ledger/gh200-j6a-v4-2026-10-02.jsonl`;
  - seed 0;
  - F's recipe (`F_RECIPE`, with `--lower-layers-n 8 --lower-layers-lr-scale 0.1` and `--checkpoint-skip-layers 6`, equal to the plan's argv);
  - `--replay-partition`;
  - cap 32,400 s under `approved()`.
  - It runs from qd-lane8 at a502670, untouched.
  - This pre-registration answers L-replay's `GAP-REPLAY-WEIGHT-HAS-NO-PINNED-VALUE-2026-10-02`, which is left for that lane and the lead to close.
- **The prelude:**
  - `j6a_prelude_box.py` takes the form of `/home/ubuntu/scratch/f_prelude_box.py` (sha256 `9f86f797…`).
  - It patches `real_ft_run.Ledger`. At a502670 the first statement after `_replay_plan` is `ledger = Ledger(args.ledger)` (`:8512`, after `:8498`). Every other `Ledger(` call (`:8350`, `:8369`, `:8403`) sits in a `--score-plan` or `--score-checkpoint` branch that returns earlier [V, by read].
  - `_replay_plan` does not create the replay cache. It only records `cache_path` (`:7240-7306`) [V, by read].
  - The waiter runs it on the **training argv itself**, niced, after flock, with cap 1,800 s.
  - It passes only if it exits 0, prints `PRELUDE OK`, and main printed `replay: N batches from <REPLAY_DIR> (attestation <sha16 of the pin>), weight 1.0, every 6`.
  - The needle-control argv is not preluded, because it needs the checkpoint training has not yet written. That control refuses at the same argv-time point as everything else.
- **Not wired:** `qd-post-f-rules j6a` (L-room). The lead reads it by hand after the row exists.

## What was measured [V]

| Check | Result | Evidence |
| --- | --- | --- |
| Container dry run | **172 PASS, 0 FAIL** | `AUDIT/idle-gpu-queue-2026-10-02/waiters-dryrun.log` |
| Argv checks | **22 ARGV OK, 5 negative controls REFUSED** | `AUDIT/idle-gpu-queue-2026-10-02/waiters-argv-check.log` |
| Static checks | all files clean | `AUDIT/idle-gpu-queue-2026-10-02/waiters-static.log` |

**The dry run.** It ran under podman `python:3.11-trixie` on arm64, with `--network none --pull never`.
- Real in the container: the aarch64 binaries, `flock`, `timeout` and `tee`.
- Faked: python (slow, and it records argv), `sleep` and `git`.
- Coverage:
  - every branch of every waiter;
  - `.done` on every exit path;
  - `kill -9` of a child in fsucc (F′ seed 1) and in j6a (its training);
  - the rungd cap kill;
  - order checks that fail when any wait is deleted.
- New j6a scenarios:
  - prelude fail, no replay line, and the wrong attestation in the replay line;
  - an unclean attestation, hits, and `checked < total`;
  - an attestation pin mismatch and a prelude pin mismatch;
  - `data/heldout` present, F's data dir, and one pin UNSET;
  - the child killed, and no word.

**The argv checks.** Each recorded `real_ft_run.py` argv goes through `main()` of the pinned commit's `git archive`, with `resolve_rev` replaced by a sentinel. torch 2.12.1 on the Mac.
- The two rung (d) arms pass at 01b6db2.
- These pass at a502670: F′ for both arms, J5″ for both arms, J6(a) training with weight 1.0, and J6(a)'s needle control.
- These are REFUSED as expected:
  - T-fp32 with `--train-attention-mask none`;
  - F′ without `--approved-by`;
  - J6(a) without `--replay-weight`;
  - the two rung (d) arms at a502670.

**The static checks.**
- `bash -n` passes on the eight shell files.
- shellcheck 0.11.0 at every severity, with `-x`, exits 0.
- `ast.parse` passes on the prelude.

## Deviations and readings (lead to confirm; `GAP-IDLE-WAITERS-DESIGN-READINGS-2026-10-02`)

1. fsucc's CPU linear controls run **after** F′'s GPU work. It releases gpu.lock and touches `fsuccgpu.done` first, so j5pp's GPU work does not wait on CPU controls.
2. fsucc's wait for F's controls line is bounded at 7,200 s after rungd and j6ctl finish. After that it writes `refused` (exit 3), never quiet.
3. cudadev's per-binary caps are 150 s, with 300 s for runga and 600 s for GDN, under the 1,500 s total. The device-test HANDOFFs say 900 s; the brief's ≤1,800 s and <$1 total binds.
4. cudadev sets `NVIDIA_TF32_OVERRIDE=0` for every binary, not only runga.
5. j6a refuses a data dir that holds `data/heldout`, and it requires `checked == total` from the attestation.
6. j6a's needle control is not preluded (see above).

## Not verified

- **[U] No waiter has run on the box.** Not run on the box:
  - the device tests and runga;
  - the rung (d) arms;
  - F′, J5″ and J6(a);
  - the successor binary.
- **[U] Version difference.** The box's torch is 2.10.0; the argv checks ran under 2.12.1.
- **[U] qd-prep use.** qd-prep-m1 use past `resolve_rev` was not exercised (the argv checks stop at the sentinel).
- **[U] Linear-control argv.** `ft_linear_control.py`'s argv was not driven through its own `main()`.
- **[U] j6a's prelude** has never run against real data. L-replay's Mac and box preludes are the first runs of that path.
- **[I] What `loss_log` holds.** The checkpoint `loss_log` the j6a report prints is the loss without the replay term; this is inferred from `replay.py:503-511` (the wrapper returns the inner value).
- **Citation mismatch.** `campaign/j6a-preregistered.json` cites "a502670 tools/real_ft_run.py:9191-9192" for the weight's missing default. At a502670 those lines are `:7818-7819`; `:9191-9192` is main's [V, `git show`]. Recorded in `GAP-J6A-REPLAY-LOG-NOT-EMITTED-AT-A502670-2026-10-02` for an amendment. L-waiters did not edit the pre-registration.
- **A lost note.** An earlier note on "J6(d)-v4 log noise" was planned before a context compaction and could not be reconstructed. `/home/ubuntu/j6d-v4/train.log` does not exist yet, and `q-j6dv4.log` is empty [V]. Nothing about it is claimed here.

## Gaps (appended with `qd_train.gaps.append_gap`)

- `GAP-IDLE-WAITERS-NAVIGATION-2026-10-02` (open)
- `GAP-IDLE-WAITERS-NOT-RUN-ON-BOX-2026-10-02` (open)
- `GAP-IDLE-WAITERS-DESIGN-READINGS-2026-10-02` (open)
- `GAP-J6A-REPLAY-LOG-NOT-EMITTED-AT-A502670-2026-10-02` (open)
- `GAP-IDLE-WAITERS-CUDADEV-CAP-ARITHMETIC-2026-10-02` (resolved)
- `GAP-IDLE-WAITERS-CUDA-BINARY-SOURCE-STATE-2026-10-02` (closed-no-defect)
- `GAP-J6-V4-ARMS-HAVE-NO-LINEAR-CONTROL-ROW-2026-10-02`: resolved-with-residual. box_q_j6ctl.sh is staged; the rows will not exist until it runs.

Cited but not touched:
- `GAP-REPLAY-WEIGHT-HAS-NO-PINNED-VALUE-2026-10-02` (L-replay's)
- `GAP-GAPS-WRITER-TEST-ASSUMES-CHECKOUT-DIR-NAME-2026-10-02` (L-room's)

## Commits on this branch

- aa8f88b: the waiters and the overlay builder.
- 3595b8c: overlay pinned to 01b6db2, plus fsucc's bounded wait.
- d8e45e4: j6a per the J6(a) pre-registration.
- This HANDOFF and the gap records.
- Merges of main: eeb9998, 5788bf8 and 65be9c4.

## First command for the next lane (the lead)

```
ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246 '/home/ubuntu/bin/qd-post-f-rules-succ; echo "exit $?"'
```

Then run the marker check, then the launch commands above.
