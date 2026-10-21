# Mac serving and training windows (2026-10-21)

Card ft-5dfe48e7, criterion 2. Plan: `docs/lappi-v2-plan-2026-10-20.md` section 7 ("Shadow windows
and training windows on the Mac are disjoint") and ruling 14.

## The rule

On the Mac, a **shadow window** and a **training window** never overlap.

| Window | Daemon (`qd-metal-serve`) | Heavy job | Who ends it |
| --- | --- | --- | --- |
| Shadow | up, bound to the pinned release, model resident | none marked training | the owner, before a training window |
| Training | **down**: no process listens on the daemon socket | one job under `tools/mac_heavy.sh` with `MAC_HEAVY_TRAINING=1` | the job, then the owner restarts the daemon |

The two are exclusive because the Mac has one memory budget and memguard decides who dies when it
is short. From CLAUDE.md (memguard facts, verified 2026-10-17): it samples `ri_phys_footprint` of
every process the user owns every 2 s; it SIGTERMs an unprotected single process over 32 GiB; and under
critical memory pressure for 3 consecutive samples it SIGTERMs the **largest** process of 2 GiB or
more, SIGKILL 10 s later, and leaves no log (`~/Library/Logs/memguard` does not exist on this Mac). With a
resident model beside a training run, the largest process is the training run, mid-seed.

## What is measured, and what is not

| Quantity | Value | Source |
| --- | --- | --- |
| LoRA-2B step, widest sequence (T = 9,638), checkpointing on: peak phys_footprint | 14,314,148,328 B = 13.33 GiB | `ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L4` (`footprint_max_bytes`; 3 timed steps after 1 warmup) |
| Same, T = 2,048 | 7,949,980,016 B = 7.40 GiB | `rows.jsonl#L3` |
| Same, checkpointing off, T = 9,638 | refused, `CapacityExceeded` at the 28 GiB budget | `rows.jsonl#L5` |
| Release weights `model.safetensors` (bf16) | 3,763,692,048 B = 3.51 GiB | release_manifest.json of the v0.1 preview (`files.model.safetensors.bytes`) |
| `mac_heavy.sh` footprint cap | 28,672 MiB while memguard's `-cap-gib` is 32 | `bash tools/mac_heavy.sh --print-limits` |
| **Daemon footprint with the model resident and a training run beside it** | **not measured** | no row exists |
| Memory pressure level during a LoRA step with the daemon up | **not measured** | no row exists |

Read together, 13.3 GiB plus about 3.5 GiB of weights is well under 32 GiB per process, so the
per-process threshold would not fire. That is arithmetic over two separate measurements, not a
measured budget for both at once: the failure this rule exists for is system memory pressure, which a
sum of per-process footprints does not bound, and no row measures it. The guard therefore refuses; it
does not rely on a budget. A future row that runs the LoRA step with the daemon resident, records
`kern.memorystatus_vm_pressure_level` through the run and the peak footprint of both, could relax
the rule; until it exists nothing here does.

## The guard

`tools/mac_heavy.sh` (the header comment names it "Serving guard"):

- **Opt-in marker.** Nothing in a job's label or command identifies a training run today (labels are
  free text; `tools/real_ft_run.py` is launched from many wrappers). A training launch sets
  `MAC_HEAVY_TRAINING=1`. Unset or `0` means unmarked; any other value exits 64. This is the
  explicit opt-in the card allows, and its limit is plain: **an unmarked training job is not
  guarded.** Wrappers that launch training (`tools/launch_*.sh`, `tools/span_refit_b.sh`, the
  campaign driver's queue) were not edited in this lane and must set the variable.
- **Detection.** By the daemon's socket, never by process name. `qd serve` and `qd-metal-serve` write
  no pid file (`crates/qd-runtime/src/main.rs`, `crates/qd-metal/src/bin/serve.rs`); the socket
  (default `$HOME/Library/Caches/qd/qd.sock`, override `MAC_HEAVY_QD_SOCKET`) is the one liveness
  surface. The probe is `python3 -I tools/qd_serve_launchd.py held --socket S`: it sends the
  protocol's `{"op":"status"}` and reads the reply.
- **Outcomes.**

  | Socket state | Probe exit | `mac_heavy.sh` exit |
  | --- | --- | --- |
  | no file | 0 | runs |
  | file, nothing listening (a crash leaves it) | 0 | runs |
  | daemon answers, warm or cold | 10 | **77, refused** |
  | not a socket, no answer, not a status reply | 11 | **78, refused** (a guard that could not run is not a pass) |

  A cold daemon is refused too: it loads the 3.5 GiB model on any request, so "cold" is a state that
  ends by itself.
- **When it checks.** Before the job takes a queue ticket, and again once it holds the lock (a daemon
  may have been started during a wait of hours). It does **not** watch during the run: a daemon
  started mid-seed is not stopped by this guard. That is the owner's window discipline, below.

## Procedure for the owner

Nothing in this lane touched launchd. The commands are the owner's to run.

1. End the shadow window: `launchctl bootout gui/$(id -u)/com.bharath.lappi.qd-serve` (label
   `com.bharath.lappi.qd-serve` is the script's default; use the label that was installed). Confirm
   with `python3 -I tools/qd_serve_launchd.py held --socket ~/Library/Caches/qd/qd.sock`; exit 0 means
   free.
2. Run the training job: `MAC_HEAVY_TRAINING=1 MAC_HEAVY_MAX_WAIT_S=28800 bash tools/mac_heavy.sh <label> ...`.
3. After it ends (any exit), restart the daemon with `tools/qd_serve_launchd.py install` (see
   `HANDOFF/w3b-serving-ops-2026-10-21.md` for the exact arguments) and require its health check to
   pass before the shadow window is considered open.

## Open assumptions

- During a training window the daemon is down, so a caller sees no socket. How Vani and Jarvis behave
  then (fail to d1-ultra, queue, or error) is the caller's contract and was not read in this lane
  (`Vani/` is a separate repository): **unverified**. Plan section 10 puts P2 (training) before P3
  (shadow), so the first training windows do not collide with a live shadow; retraining after
  P3 does, and its schedule must carry a shadow gap.
- The card's criterion 6 (one daemon, priority classes, a stated budget under 32 GiB) is not part of
  this lane. If that daemon later holds a base plus adapters, its footprint changes and the table above
  needs a new row.
