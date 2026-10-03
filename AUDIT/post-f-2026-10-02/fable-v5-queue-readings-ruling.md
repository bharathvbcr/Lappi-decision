# Fable's ruling: lint-gate pinned files, R9 refused, L-v5-queue readings, v5 data build staging

Consulted by the lead on 2026-10-02 at about 20:20Z through the advisor tool. The questions were
the four pending items in the lead's notes:

1. the lint gate's five pinned files;
2. R9 treating `refused`, an unknown word or no seed-0 row as `pause`;
3. L-v5-queue's readings 2-8 (HANDOFF/v5-queue-2026-10-02.md section 2);
4. whether to wait for Apple Diagnostics before the v5 data build or to stage it now.

The ruling is verbatim below. Only its opening status paragraph is omitted, which assessed the
lead's integration work.

---

Verdicts on the four:

**1. Lint gate.** Per-file-ignores for the 3 AUDIT records, with the ignore line naming the gap — they're evidence, reformatting them changes the evidence. The 2 oracle scripts: reformat and regenerate *only if* the regeneration shows every field except `script_sha256` byte-identical; if anything else moves, stop and record it. That's a small qd-train-owner lane, not on the critical path. Don't spend the lock on it before v5 staging.

**2. R9 refused → hold.** Confirm. Strictly more conservative than the DRAFT, consistent with rule 7 and the rules lane's own note. One requirement: the pause marker must carry *why* (`pause` / `refused` / `unknown:<word>` / `no-seed-0-row`) so the human can tell a reading from a tool failure without reading logs.

**3. Readings 2–8.** Confirm 2, 3, 5, 6, 8 as written. Reading 4 (GPU-step time only): confirm — the approval unit was GPU-h, so this matches it — but carry the box wall-clock beside it in the handoff so nobody mistakes the tracked total for the bill. Reading 7 (read race): your alternative of waiting for `traj-s2.done` has a cost you haven't priced — the trajectory waiter releases the GPU lock *before* its CPU controls, so gating the next reading on `.done` idles the GPU for the whole control phase (seed 3's had a 5400 s cap). The constraint that discriminates: the fix must not make the GPU wait on CPU work, and must still hold on a genuine refused. That points at a bounded retry in the waiter on the specific transient (`refused` + "malformed ledger line" on the last line), K attempts a few seconds apart, then hold. Read-only, idempotent, bounded. Worth doing: a human-gated hold overnight is exactly what the standing goal forbids.

**4. Data build.** The earlier ruling (Diagnostics before the 20 GB build) stands as the human's action; you can't run Diagnostics and shouldn't wait idle on it either. What you can't decide yet, because you haven't looked: **how long the v4 build took, its peak RSS, and whether the pipeline forks a pool.** Those three facts are in `HANDOFF/replay-v4-build-2026-10-02.md` and the pipeline source. They decide everything:
- Peak RSS > 32 GiB → the RSS cap kills the build; you need the number before launching under `mac_heavy.sh` at all. The 10-01 jetsam (python3.12 at 36 GiB, 3,638 processes) looks like a pool-based data build — if this pipeline uses `multiprocessing`, it's "fork-heavy" and outside "resume, gentler" without a `--workers 1` equivalent.
- Deadline = j6g.done ETA − (build + containment scan + rsync). Box has ~3 h of seed 4 left plus j6a and j6g, so likely 8+ h of slack. If the deadline is more than ~4 h out, stage everything light now and ask the human once (AskUserQuestion: timeline, Diagnostics ~5 min but needs shutdown, options run-Diagnostics-first / stage-now-under-cap / wait). This is genuinely theirs. If the deadline is inside ~4 h, the GPU-idle rule binds: launch under the lock with the cap, tell peers the lock is held for hours, and make sure the build is re-runnable first (your own audit listed that as undone).

**Not waiting on anything — do now:** the v5-queue handoff section 7. Every pin in `v5_common.sh` is UNSET; the rules binary needs its new box path and sha; waiters need deploying. None of that depends on the data build or the human. Read section 7, fill what's fillable, and record what blocks on the build.

Housekeeping: re-arm `bjxds7hi1` before ~20:37Z; the notes' "To Fable" CLINC item is stale (ruled at 0f1649d); add f7ae29d / adbaec1 / a94e9de to the 20:10 notes once the suite reports.

---

## What the lead found when it followed item 4 (read, not run)

- **Duration.** HANDOFF/replay-v4-build-2026-10-02.md records 5,774 s and 4,386 s for the two v4
  builds, on a Mac at load 78. That handoff says these are not a timing of the pipeline.
- **Forking.** The pipeline does not fork a pool. `tools/real_tokenizer_pipeline.py` has no
  `multiprocessing`, `ProcessPool` or `concurrent.futures`. Its only children are
  `subprocess.run` calls to qd-prep, one at a time.
- **Peak RSS.** The build's peak RSS is not recorded anywhere read. The DRAFT's
  `amendments_pending` asks for it ("the census: 9-10k supply and the build's peak RSS"), so the
  build is to be run under `/usr/bin/time -l`. The containment-scan exporter's peak was
  estimated at about 14 GB (HANDOFF/prep3-2026-10-02.md).
- **Deadline.** It is not near. Before v5 can launch, the box still has seed 4's training and
  scoring and then j6a and j6g. Several of v5's pins (C1, C2a, C2b, C3, R4 and the fsucc
  reading) are decided by post-F rows that do not exist yet.
