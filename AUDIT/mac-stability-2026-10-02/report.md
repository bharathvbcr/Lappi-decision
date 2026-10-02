# Mac stability audit, 2026-10-02

The Mac kernel-panicked twice on 2026-10-02, at 08:53 and 14:24 local. This audit is read-only:
the panic and jetsam reports, the unified log, system limits, launch agents, mounts and the process
table. Fable reviewed it (fable-mac-hardening-ruling.md, beside this file). Labels: [V] read or
measured, [I] inferred.

## What the panics are

- **Same panic both times.** Both reports say `initproc exited -- exit reason namespace 2 subcode
  0xa`: launchd (PID 1) ended by signal 10 (SIGBUS). When launchd dies, the kernel panics. [V]
  - `/Library/Logs/DiagnosticReports/panic-base+socd-2026-10-02-085349.000.panic`
  - `/Library/Logs/DiagnosticReports/panic-base+socd-2026-10-02-142456.000.panic`
- **Not the cause: memory at the moment of the panic.** The compressor and swap were OK both times
  (13% and 4% of the compressed-pages limit, with 42 and 4 swapfiles). [V]
- **Not the cause: disk.** 224 GiB free, SMART Verified, no external drive mounted. [V]
- **Not the cause: autofs.** It was the last kext started in both reports, but it loads about one
  minute after boot, hours before each panic. [V]
- **Not the cause: load.** Load was about 350 at 08:53 and about 30 at 14:24, with one job under
  the lock. [V]
- **No thermal signal.** `pmset -g log` records no thermal or performance warning. [V]
- **The OS changed just before.** macOS 27.0.1 was installed 2026-09-30 at 11:41
  (`system_profiler SPInstallHistoryDataType`). The 27.0 builds ran without a reboot from Sep 22
  until the update (`last reboot`). Both panics came within 2 days of 27.0.1. [V]
- **The cause is not determinable from user space.** The unified log of the previous boot ends at
  14:24:15, so the last ~40 s before the 14:24 panic were never written. [V] The 08:53 window is
  contaminated by the next boot, so it is not used.

## Two abnormal, fixable pathologies

### 1. Runaway memory: JetsamEvent reports this week [V]

| When | Largest process | Resident | Kills |
| --- | --- | --- | --- |
| 09-29 14:15 | Python | 71.5 GiB | daemons, low-swap |
| 10-01 17:02 | python3.12 (3,638 processes on the machine) | 36.1 GiB | daemons, low-swap |
| 10-02 10:55 | go.test | 119.9 GiB | daemons, low-swap |
| 10-02 11:02 | go.test + go.test | 84.4 + 29.7 GiB | daemons, low-swap |

- **Scale.** The Mac has 64 GiB of RAM.
- **Who launched them.** Both go.test events share one launching coalition (1571). The reports do
  not name it.
  - The ojas adaptive-resources session says the 10:55 event was very likely its own test: a case
    in `go/profile_test.go` generated a record with 4,294,967,295 entries (about 38 GB, more with
    append growth). The OS killed it after 491 s. It was fixed the same morning; the case now
    patches the count field of a 16-entry record, and the rerun took 17.8 s.
  - That session cannot say whether 11:02 was also its test.
  - This attribution is the peer's own report; it was not independently verified.
- **Relation to the panics.** Swap was heavy before panic 1 (42 swapfiles), but not before panic 2.

### 2. Process churn [V]

- **git spawns 31 times a second on an idle machine.** 3,682 distinct git pids in 2 minutes. The
  children caught were from GitPulse.app and the Claude desktop app's node helper. About 60
  worktrees of this repo alone are polled.
  - Pruning 54 merged worktrees (60 -> 6) did NOT lower the rate: 4,470 git pids in the next 2
    minutes, 37.2/s.
  - A 10 s back-to-back `ps` burst caught only GitPulse's children (`git -c ...`, parent
    `gitpulse --background`). Those processes live a few milliseconds.
  - So GitPulse.app is the dominant spawner, and its rate does not scale with this repo's worktree
    count. It likely polls other repositories too. [I]
- **The clean pre-panic window shows the same churn.** 14:21 to 14:24:38 has 85,317 log lines:
  - gitpulse 23,982 and git 9,978, with consecutive pids milliseconds apart;
  - syspolicyd exec-policy errors (`Unable to initialize qtn_proc`, MACH_SEND_INVALID_DEST), the
    sign of processes exiting before their policy check finished.
- **A launch agent crash-loops.** `homebrew.mxcl.postgresql@17` exits within 31 ms and launchd
  respawns it every 10 s, forever.
- **Many long-lived MCP servers.** 108 MCP-server processes run across 16 Claude Code sessions.
  Separately, non-Claude `agy` agents run vitest and golangci-lint.
- **Standing reservations.** podman-machine-default holds 12 GiB, and ollama is started.

Exec churn appears at both panics and right now. That is a correlate and a hypothesis [I], not a
proven cause.

## Hardening, ranked

### Yours: the only path to a root cause

1. **Run Apple Diagnostics once**, before the ~20 GB v5 data build. Shut down, then hold the power
   button. It is the only test that separates marginal RAM from an OS bug.
2. **File Feedback Assistant** with both `.panic` files and a `sudo sysdiagnose`. Install any
   27.0.x update as soon as one is offered.
3. **Stop standing load that gives nothing:**
   - `brew services stop postgresql@17` (the crash loop);
   - stop ollama if it isn't serving;
   - `podman machine stop` when not building;
   - close idle Code sessions;
   - stop or throttle the `agy` agents.
4. **GitPulse, your app:** lower its poll rate or move to fsevents/fsmonitor-triggered scans, and
   cap the number of worktrees it scans.
5. **Optional machine-wide guard.** Add a user LaunchAgent watchdog that sends SIGTERM to any user
   process above about 40 GB resident and logs it. macOS enforces no per-process memory cap; it
   would have caught every jetsam event above. Alternatively, run heavy suites inside the podman
   VM with a memory limit.

### Lappi-side (the lead)

- **Prune merged lane worktrees.** Done: 54 removed, 60 -> 6. It cut the Cargo.lock contention
  noise but not the git churn (31/s before, 37/s after; see pathology 2). GitPulse's own poll rate
  is the lever there.
- **Harden `tools/mac_heavy.sh`.**
  - Add an RSS watchdog on the job's process tree: kill the job and exit nonzero above a cap.
  - Refuse to start when load or the process count is too high.
  - Fail loud, with tests that fail before the change.
- **Mutation testing stays off this host.** L-v5-queue's round 2 is NOT RUN, for this reason.
- **Make the v5 Mac build re-runnable** and stage it while the human is at the keyboard, after
  Apple Diagnostics.
