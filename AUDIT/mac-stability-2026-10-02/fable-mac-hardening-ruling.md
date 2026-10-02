<!-- Fable's ruling on the Mac stability audit, 2026-10-02 ~19:55 UTC, given through the advisor (the human set "/advisor fable" and asked "How to harden my mac to prevent these crashes. Do deeper audit, ask fable if needed"). Recorded verbatim by the lead from line 5 to the end of the file; nothing below this header is edited. The lead's audit evidence is in AUDIT/mac-stability-2026-10-02/report.md. -->
<!-- The two read-only checks it asked for ran after it: macOS 27.0.1 was installed 2026-09-30 11:41 local (system_profiler SPInstallHistoryDataType), and `pmset -g log` records no thermal or performance warning level. -->
<!-- Related record: GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02. -->

**Verdict first:** your audit is good enough to rule things out but not to name the cause — and it won't get there from user space. Say that plainly. What it *does* establish is two abnormal, fixable pathologies on this machine, plus one open question (OS vs. hardware) that only two cheap checks can discriminate. Build the report around that, not around a cause.

**What the evidence supports [V]**
- Both panics: launchd SIGBUS; compressor/swap/disk fine; autofs irrelevant (loads at boot); no external drive; load ≈350 vs ≈30 — load is not the common factor.
- Pathology 1 — runaway memory: go.test 120 GB and 84+30 GB, Python 72 GB, python3.12 36 GB with 3,638 processes; 42 swapfiles at panic 1; daemons jetsammed for low-swap. Present before panic 1, absent at panic 2 (4 swapfiles).
- Pathology 2 — exec churn: git spawning 31/s on an idle machine (3,682 in 2 min); syspolicyd exec-policy errors (processes gone before the check); postgresql@17 crash-looping every 10 s; 108 MCP processes across 16 Code sessions; 60 worktrees. Present at both panics and right now.
- The 08:53 aggregation is contaminated by the next boot (audiomxd's 1.5M lines are boot-time XPC retries). Only the 14:21–14:24:38 window is clean. Don't cite the 08:53 numbers.

**Two checks before you write anything** (read-only, seconds):
1. `system_profiler SPInstallHistoryDataType | rg -B1 -A4 "macOS"` — when did 27.0.1 land? `last reboot` shows Sep 30 11:39 and then nothing until today's two panics. If the update was Sep 30, this is a 2-day-old-OS problem and the report leads with that.
2. `pmset -g log | rg -i "thermal|pressure" | tail -20` — thermal state near both panics. Two PID-1 SIGBUS in six hours with no software commonality fits marginal RAM under heat as well as it fits a kernel bug; Apple Diagnostics (user holds power at boot, ~10 min) is the only test that separates them.

**Hardening, ranked by evidence × cost**

*Human's items — the only path to a root cause:*
- Run Apple Diagnostics once. If it flags memory, nothing below matters.
- File Feedback Assistant with both `.panic` files and a `sudo sysdiagnose`. Install any 27.0.x update.
- Stop the standing load that gives you nothing: `brew services stop postgresql@17` (crash loop), stop ollama if not serving, `podman machine stop` when not building (12 GiB reserved), close idle Code sessions (each keeps its MCP servers alive), stop or throttle the `agy` agents.
- GitPulse is your own app: it is the likely bulk of the 31 git/s against 60 worktrees. Lower the poll rate or move to fsevents/fsmonitor-triggered scans, and cap scanned worktrees. Claude desktop's helper also polls; fewer sessions = fewer polls.

*Machine-wide guard the human can install (user LaunchAgent, 20 lines):* a watchdog that SIGTERMs any user process above a configurable RSS (say 40 GB) and logs it. It would have caught every jetsam event this week. macOS enforces no per-process memory cap otherwise; running heavy suites inside the podman VM with a memory limit is the alternative.

*Lappi-side, yours, now:*
- Prune merged lane worktrees via `mcp__ccd_host__clean_up_worktrees` — data, prep2, prep3, fmt, recency, rules, lint are all merged. Cuts git churn and the Cargo.lock "contended across 8 worktrees" noise.
- Add to `tools/mac_heavy.sh`: an RSS watchdog on the job's process group (kill and exit nonzero above, e.g., 32 GB) and a refusal to start when load > 12 or process count > some bound. Fail loud, with a test that fails pre-fix.
- Keep mutation testing off this host. L-v5-queue's round 2 is NOT RUN with that reason; don't resume the agent into it.
- Make the v5 Mac build re-runnable so a mid-build panic costs a rerun, not a corrupt output. Stage it when the human is at the keyboard.

**What to tell the human, and what not to claim**
- Root cause: unknown and not determinable from logs; the last 40 s before each panic were never written.
- Exec churn is a correlate at both panics and a hypothesis [I]; memory runaway is a documented pathology that preceded panic 1 only. Don't present either as the cause.
- The lock protected against something that didn't cause panic 2. Keep it as hygiene.

**Does this block?** The hardening doesn't block the goal — the box is unaffected and has days of queued work. But recommend Apple Diagnostics *before* the ~20 GB v5 build: if RAM is marginal, "resume, gentler" just crashes slower. Record this ruling verbatim as before, with full gap ids this time.
