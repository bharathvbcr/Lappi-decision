# v6-linfit-pool, 2026-10-04: the FT linear control's per-iteration floor

Branch `v6-linfit-pool` (worktree `build/v6-linfit-pool-wt`), from main 8b01cd0. The human,
~15:25Z: "Follow fable's advice and work on it"; ~16:00Z: "yes, run the box bench". Fable's
rulings this lane: measure on the box before choosing a fix; bit-exact changes first (Fable's
ruling 4, a warm-started L2 path, changes the solution and stays a v6 target); no rebuilt
`qd-prep` reaches a v5 control without Fable's ruling and the human's yes.

## What was measured

**Attempt 1 of the early control check** (52 threads, nice 10; `/home/ubuntu/post-queue/early/`
on the H100 box) [V, its log]: ms per iteration = fit seconds / (grid + refit iterations):
openjev.game (971 rows, 1.73M nnz) 5.4; arc.science (3,178 rows, 3.78M nnz) 6.6;
commonsense (9,619 rows) 10; code.defect_class (71,435 rows, 369M nnz) 245.

**Attempt 2** (26 threads on CPUs 0-25 = 13 cores x 2 hyperthreads, nice 19; launched 15:50:11Z
with the human's yes; cap 10,800 s, ends by 18:50:13Z) [V, its timestamped log]: the same first
14 tasks fitted in 2,511 s against attempt 1's 1,428 s. arc.science 167.5 s (5.49 ms/it, faster
than at 52 threads); code.defect_class 816.9 s (attempt 1: 488.5 s); intent.domain 489.1 s;
openjev.evidence 524.5 s. The three large tasks are 73% of the 2,511 s.

**Box topology** [V, read-only]: Xeon Platinum 8480+, 1 socket, 26 cores x 2 threads, 1 NUMA
node; hyperthread siblings are adjacent CPUs (0,1), (2,3), ...

**`linfit_bench` on the box** (7af2810's example, x86 musl sha256 7987d536..., nice 19 on CPUs
26-51 beside the training lanes and attempt 2, load 20-24, 16:01-16:03Z, the human's yes)
[V, /home/ubuntu/logs/q-linfit-bench.log]:
- near-empty problem: 1 / 13 / 26 threads = 0.086 / 0.854 / 1.678 ms an iteration: ~32 us per
  thread per phase, two phases an iteration. At 52 threads ~3.4 ms: the whole unexplained floor.
- openjev shape: 1 / 13 / 26 = 15.56 / 3.03 / 3.66 ms; 13 threads one per core: 3.32.
- arc shape: 13 / 26 = 5.45 / 6.18 ms; 13 one per core: 5.55.
- Hyperthread pairing is not the cause; past 13 threads the start-up cost outweighs the compute.

Mac (18 CPUs, unloaded) measurements are in `build/linfit-pool/notes.md` (git-ignored).

## What changed (commits on v6-linfit-pool)

- 7af2810: `crates/qd-prep/examples/linfit_bench.rs` (std only): per-iteration cost against
  thread count, interleaved rounds, every call's result checked equal to the first.
- d775305 (red): `crates/qd-prep/src/team.rs`, a team of threads started once and reused for
  every phase (claims stamped with the phase, completion counted per item, relaxed-atomic
  `SharedF64s` cells, no `unsafe`, no dependency; five tests), and
  `a_fit_starts_its_threads_per_training_run_not_per_iteration`, failing as intended:
  "40 iterations started 1620 threads, 10 started 420".
- f405be7 (green): `train_once` runs both phases on one team per call. cargo test -p qd-prep
  --lib 85 passed; the Python reference bit for bit (test_qd_prep_linear_parity and the five
  control test files) 109 passed, 0 failed, skips with reasons (no torch in the venv; the
  commitpackft corpus not in the worktree; the opt-in benchmark).

## What is open

**f405be7 is not deployable: it regresses the large tasks.** Mac A/B, interleaved, min of 5,
old = 7af2810, new = f405be7 (build/linfit-pool/ab-mac.log):

| shape (threads) | old ms/it | new ms/it |
|---|---|---|
| near-empty (18) | 0.350 | 0.015 |
| openjev (1 / 8 / 18) | 14.37 / 2.80 / 1.81 | 9.48 / 2.13 / 1.28 |
| arc (4 / 18) | 10.68 / 3.51 | 7.65 / 2.64 |
| defect-small, k=8, 20 it a call (18) | 92.0 | 122.9 |

The defect figure includes the single-threaded transpose once per training run; Fable's estimate
from the two old-binary points (63.3 ms/it at 40 it a call, 92.0 at 20) puts the hot loop at
~35 ms/it old and ~65 new, ~1.9x. Not yet established why. The next experiments (Fable): defect
at --iters 100; a `row()` accessor so each hot loop pays one bounds check per row, not per
element; classes 4 vs 8 on both shapes. If the cause is the vectorization that atomic loads
forbid, the remedies are a policy choice for the human: an RwLock-guarded plain snapshot of `w`
and `diff` copied by the leader each iteration (safe; ~10% on small tasks), or `UnsafeCell`
under the phase discipline (no runtime cost; the crate's first `unsafe`). Default: the safe copy.

No box run of f405be7 has been proposed or made.

## First command for the next lane

    bash build/linfit-pool/test.sh check --lib

(then `bash tools/mac_heavy.sh linfit-ab bash build/linfit-pool/ab_all.sh 5` after editing the
shapes in ab_all.sh; build/linfit-pool/ is git-ignored local tooling in main's checkout.)
