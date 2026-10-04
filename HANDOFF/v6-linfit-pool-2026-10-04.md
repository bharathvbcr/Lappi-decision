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
- 121bdcc: this handoff's first version.
- 150a2e7 ("rows"): `SharedF64s::row` and `team::load`, so each hot loop paid one bounds check
  a row; `logits_row` takes a `w_row(at)` closure, so the fit's cells and a scoring pass's plain
  slices share one arithmetic.
- 7c7b410 ("copy", Fable's default remedy): phase A reads `W` and phase B reads `diff` from plain
  `RwLock<Vec<f64>>` snapshots the leader copies between phases; `w_next`, `m`, `v`, `gW`,
  `diff` and the loss terms stay `SharedF64s`; `row`/`load` removed. cargo test -p qd-prep --lib
  85 passed; the Python parity set 111 passed (the commitpackft pool data symlinked into the
  worktree, git-ignored).
- ebf45ca: this handoff's second version (its box_ab3 conclusion is corrected below).
- be65fa5 ("slices"): phase A and phase B bind a plain `&[f64]` from each snapshot guard once per
  item. Bit for bit; cargo test -p qd-prep --lib 85 passed; the parity set 111 passed, 4 skipped
  (no torch; the opt-in benchmark).

## The A/B results (interleaved, min over rounds; old = 7987d536, rows = 81cf0d0e, copy = a8312a5a)

Mac, 18 threads unless noted (`build/linfit-pool/ab-copy-mac.log`, min of 5) [V]:
defect-k8 old 54.3 / rows 75.2 / copy 56.2 ms/it; defect-k4 82.7 / 75.3 / 69.6;
openjev-k8 (1 / 18 threads) copy/old 0.97 / 0.96; openjev-k4 0.63 / 0.86. On the Mac the copy
removed the regression.

H100 box (`/home/ubuntu/logs/q-linfit-ab{,2,3}.log`; CPU only, nice 19 unless noted, under the
human's "yes, run the box A/B; don't ask me, ask fable", each further run on Fable's ruling) [V]:

| shape | 26 thr on CPUs 26-51 (lanes + attempt 2 running) | 52 thr, all CPUs | copy/old, 52 thr |
|---|---|---|---|
| tiny (64 rows) | old 1.645, copy 0.089 | old 2.498, copy 0.099 | 0.04 |
| openjev-k4 | 3.90 / 2.97 | 5.73 / 2.48 | 0.43 |
| openjev-k8 | 5.03 / 5.18 | 6.87 / 3.81 | 0.55 |
| openjev-k32 | 12.0 / 14.4 | 11.1 / 10.7 | 0.97 |
| arc-k4 | 5.84 / 5.53 | 7.02 / 3.85 | 0.55 |
| defect-k8 | 108.9 / 153.2 | 77.0 / 99.6 | **1.29** |
| defect-k4 | 76.9 / 93.7 | 60.8 / 66.9 | **1.10** |

box_ab2 (17:47-17:52Z, all 52 CPUs, the box otherwise idle but for the lanes): at nice 10,
defect-k8 copy/old is 1.17 at 26 threads and 1.27 at 52; nice 19 at 26 threads gives 1.17 again.
So the anomaly is **neither the nice level nor oversubscription**. The old binary is no faster at
52 threads than at 26 on defect (74.9 vs 76.4) and slower on the small shapes (arc 6.9 vs 4.5,
openjev 5.7 vs 3.2): which is why early attempt 3 runs on the 26 even CPUs.

box_ab3 (17:58-17:59Z, nice 10, defect-k8, --iters 3): 1 thread: old 1,384 / copy 1,320 ms/it
(0.95); 4 threads: 770 / 886 (1.15). **The conclusion first drawn from it was wrong.** It read:
the per-element code is not the cause; the loss is the team mechanism, so `UnsafeCell` would
not help. Two flaws:
- At `--iters 3` the single-threaded transpose (~1.6 s a training run, once per run) dominates
  ms/it.
- At 1 thread, phase B's gain masked phase A's loss (box_ab8 below).

The chain that found the cause (each run on the box, CPU only, under Fable's ~19:40Z ruling;
`/home/ubuntu/logs/q-linfit-ab{4..11}.log`; `build/linfit-pool/notes.md`) [V]:

| run | what | finding |
|---|---|---|
| ab4/ab5 | `with_team(threads+1)` ("plus1"), `SPIN = 0` ("spin0"), --iters 50 | 4 thr: copy 1.05, plus1 0.84, spin0 1.05; 26 thr: 1.18, 1.17, 1.19. Not the spin; plus1 only adds a thread |
| ab6 | 26 thr pinned to the even CPUs vs free | 1.17 vs 1.15: not wake placement on hyperthread siblings |
| ab7 | timed copy build | the leader's serial sections ~0.7 ms an iteration: not the snapshot copies |
| ab8 | timed old vs timed copy | phase A +10% (4 thr) and +9% (26 thr), phase B -6% (4) and +14% (26): the hot loops |
| ab9 | slices (a plain `&[f64]` bound from each snapshot guard once per item) | 4 thr 0.99, 26 thr 1.08 |
| ab10 | slices2 (plus a per-item view of every `SharedF64s`) | 0.98 / 1.08: no further gain; slices reproduced at 0.97 / 1.07 |
| ab11 | old vs slices, 26 thr on the even CPUs (attempt 3's configuration) | tiny 0.06, openjev-k4 0.59, openjev-k8 0.72, arc-k4 0.85, defect-k4 1.09, defect-k8 1.03 |

**The cause:** indexed through the `RwLock` read guard, the loops over the nonzeros reloaded the
snapshot `Vec`'s pointer and length for every nonzero. The loops' stores to the local `z` / `g`
may alias the lock's memory as far as the compiler can tell. be65fa5 binds plain slices once
per item.

## What is open

**be65fa5 ("slices") is the deployable candidate.** Projected on early attempt 3's 197 fits
(4,974 s on 26 cores) with ab11's ratios by size band: 3,662-4,446 s (0.74-0.89x) [inferred:
band ratios, not a replay].
- The three defect-like tasks (>= 50M nonzeros, 1,471 s) get 3-9% slower.
- The 43 start-up-bound tasks (588 s) and the length arm (226 s) get 3-15x faster.

No rebuilt `qd-prep` reaches a v5 control without Fable's ruling and the human's yes.

**The residual, open:** ~5.5 ms an iteration on defect-k8 at 26 threads (1.03-1.08).
- ~0.8 ms of it is the leader's serial snapshot copies and grad norm (ab7).
- Candidates for the rest, untested: phase B's Adam loop storing to atomic cells (no
  vectorization); memory bandwidth at 26 threads.
- The `UnsafeCell` variant (branch `v6-linfit-pool-cell`, worktree
  `build/v6-linfit-pool-cell-wt`, uncommitted, `team.rs` only) would remove both the copies and
  the atomics. It is the crate's first `unsafe`, a policy choice for the human. On hold until
  the human decides.

The f405be7 record below stands as the first measurement.

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

No box run of f405be7 has been proposed or made; the box runs above used 150a2e7 and 7c7b410.
No rebuilt `qd-prep` reaches a v5 control without Fable's ruling and the human's yes.

## First command for the next lane

    bash build/linfit-pool/test.sh check --lib

(the variants were built by `build/linfit-pool/diag_*.sh` in the detached worktree
`build/v6-linfit-pool-diag-wt` and run by `build/linfit-pool/box_ab*.sh`; build/linfit-pool/ is
git-ignored local tooling in main's checkout. Box runs need Fable's ruling and must not overlap a
control or the lanes' ends.)
