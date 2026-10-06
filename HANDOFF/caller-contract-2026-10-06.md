# HANDOFF: caller contract — Lappi wired into DevCouncil, GitPulse and DevType (2026-10-06)

Session "Lappi integration and system hardening" [32beaa]. The human asked to wire Lappi into
their apps, collect clean data for future training, audit with DevMap, and harden and stress-test.

## What is true now

- **Every caller is wired and none is served.** v0.1 preview's `trained_families` hold none of the
  three callers' decisions (`code.defect_class` is trained but refused at serving:
  `calibration_entry_missing` on the span slot, `HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md`).
  So DevCouncil, GitPulse and DevType each send their pinned request, read the refusal, and keep
  their own path. App behaviour is unchanged by default (both settings off) and in practice (every
  request refused). Nothing remaps a caller onto a look-alike trained family: that would serve an
  untrained prompt.
- **Collection is off by default, local only, and not admitted.** The human said "Synthetic only"
  on 2026-10-06 (relayed by the data-clean session [764787]; not found in any file here, so it is
  recorded as relayed). Records are stored under `.../Lappi/heldout/caller-records/<app>/`, carry
  `admission: "not_admitted"` (the only legal value), and `qd-train`'s rule-3 door refuses the path.
  **This conflicts with the request to "collect clean data for training"; the human decides.**
- The contract is `docs/caller-contract.md`; the checker is `qd_runtime::caller_record` and the
  `qd-caller-records` binary.

## Measured (tests, not ledger rows: no model was run in this lane)

All runs under `tools/mac_heavy.sh`. Logs: this session's task output for
`lappi-caller-red-green-3` (17:56–17:58Z) and `build/gitpulse-lappi-*.log`.

| What | Red (unmodified HEAD a683c27) | Green (working tree) |
| --- | --- | --- |
| `qd-runtime` client (`tests/client_deadline.rs`) | drip **FAILED** (caller held ~6 s on a 500 ms timeout); early reply **FAILED** (`Broken pipe`) | 4/4 |
| `qd-runtime` agent (`tests/serve_stress.rs`) | drip **FAILED**; slow reply **FAILED** on the round-1 tree | 5/5, and 5 more parallel repeats of both suites all ok (18:13–18:16Z) |
| `qd-runtime` full suite (`cargo test --no-fail-fast`) | — | every binary ok, 0 failed |
| `qd-train` `tests/caller_records_are_held_out.rs` | — | 2/2 |
| `qd-metal-serve` build against the changed `serve` | — | ok |
| `qd-caller-records` on fixture / non-heldout / torn / duplicate / missing | — | exit 0 / 2 / 2 / 2 / 1, as wanted |
| DevCouncil `go test ./devcouncil/lappi/... ./devcouncil/verify/...` | — | ok (both); cross-check through `qd-caller-records` PASS |
| GitPulse `cargo test --lib -- lappi commit_brief tool_config ai::` | — | 158 passed; clippy `-D warnings` clean; bun tests 67+11; IPC contract 247/247; cross-check 16 lines, 0 invalid; after fixes 3-4: lappi 55/55, clippy clean (18:16Z) |
| DevType `swift test --filter 'Lappi\|PaletteToolRouting\|LocalizationParity\|AIPreferencesCoverage'` | — | 107 passed, 0 failed (57 Lappi); cross-check 12 lines, 0 invalid |

**Re-run after fix 4** (`lappi-caller-round3-apps`, under `mac_heavy.sh` 18:50:09–18:50:26Z, exit 0):
- **DevCouncil:** `gofmt -l` printed nothing and `go vet` exited 0. `go test ./devcouncil/lappi/... ./devcouncil/verify/...` passed both packages, including `TestAReplyWrittenAtAcceptIsReadEvenWhenTheRequestWriteFails` and `TestRecordsPassTheRustChecker`.
- **DevType:** the same filter ran 108 tests with 0 failures, including `testAReplyWrittenAtAcceptIsReadEvenWhenTheRequestWriteFails`. The `qd-caller-records` cross-check read 12 lines (6 decisions, 6 outcomes, one per reading) with 0 invalid and 0 duplicates.

**Not run:** GitPulse full `cargo test` (only the filtered lib tests); DevType full `swift test`
(`DevTypeAppTests` compiled and ran 0); DevCouncil full `go test ./...`; any run against the real
model (`qd-metal-serve` on the GPU): every result above is against fake agents or the reference
backend. The app clients have no red run: they are new code, so their deadline tests pin
behaviour rather than prove a fix.

## What changed (uncommitted, in four repos)

**Lappi-decision** (none of the 16 held batched-decode paths touched):
- `crates/qd-runtime/src/deadline.rs` (new): one deadline per exchange, shared by both ends.
- `crates/qd-runtime/src/oneshot.rs`: `ask_over_socket` bounds connect+write+reply by one deadline.
- `crates/qd-runtime/src/serve.rs`: `read_timeout` / `write_timeout` now bound a whole line.
- `crates/qd-runtime/src/caller_record.rs`, `src/bin/qd_caller_records.rs` (new), `lib.rs`,
  `Cargo.toml` (one `[[bin]]`).
- Tests: `crates/qd-runtime/tests/{client_deadline,serve_stress,caller_records}.rs`,
  `crates/qd-train/tests/caller_records_are_held_out.rs`; fixture `fixtures/caller/valid-records.jsonl`.
- `docs/caller-contract.md` (new).
- This lane's 9 `gaps.jsonl` records (+1 from the DevType lane) **are committed**: the training session committed the file whole in a60cd8c, by agreement, and it is on origin/main under 33da559. That commit does not commit or approve any file those records cite, and its message says so.

**DevCouncil**: `backend/go_orchestrator/devcouncil/lappi/` (client, request, diffsplit, record,
asker + tests), `verify/lappi.go` (+test), `verify/orchestrate.go` (+5), `verify/types.go` (+6).
**GitPulse**: `src-tauri/src/lappi/` (client, request, record, mod, tests), `ai/commit_brief.rs`
(`prefill_type`, the only path an answer changes a draft), `ai/mod.rs`, `commands/mod.rs`,
`lib.rs`, `tool_config.rs`, `src/lib/stores/lappiStore.ts`, `src/lib/components/LappiSettings.svelte`,
`SettingsModal.svelte`, `settingsCatalog.ts`, `scripts/check-ipc-contract.test.ts` (245→247).
`src-tauri/tests/native_watcher_faults.rs` was dirty before and is not this lane's.
**DevType**: `Sources/ExpanderEngine/Lappi/` (6 files), `PaletteToolRouter.swift`,
`CommandPaletteCatalog.swift`, `InlineSearchPanel.swift`, `PreferencesWindowController.swift`,
`LocalizationManager.swift` (6 keys × en/ko/ja), `DevTypeLog.swift`; 5 test files.

## Defects found and fixed (each with a test that fails on the pre-fix code)

1. **Per-read timeouts on both ends of the socket.** A peer moving one byte inside the timeout
   reset it forever: a "500 ms" caller was held ~6 s in the test (and unboundedly in principle), and
   one dripping client could hold an agent connection slot indefinitely. Fixed once in
   `deadline.rs`, used by client and agent.
2. **The agent's write deadline ran through the model's decode** (introduced by fix 1 in this
   lane, found by review): it was set when the request line arrived, so a reply that took longer
   than `write_timeout` (30 s shipped) to compute was dropped. Red on the working tree 18:12:57Z
   ("closed the connection without replying" at 1.0 s with a 1 s backend and 300 ms write
   timeout); fixed by starting it just before each write
   (`serve_stress::a_reply_slower_than_the_write_timeout_to_compute_is_still_delivered`).
3. **macOS `EINVAL` from `setsockopt` on a peer-closed socket** threw away a buffered reply
   (`io:InvalidInput` once in 479 burst replies). `deadline.rs::apply` tolerates it; GitPulse's
   client had copied the same pattern and is fixed the same way.
4. **A failed request write discarded the agent's reply.** `serve::refuse_connection` writes
   `overloaded` at accept and closes, so a large request met a closed socket and the client
   returned the write error (`Broken pipe`, red at HEAD round 0) instead of the reply. Fixed in
   all four clients (qd-runtime, GitPulse, DevType, DevCouncil), each with a 20-50 round test.
   The three app tests were written after their fix, so they have no red run of their own; the
   identical test is red against the reference client at HEAD.
5. Found while testing, not product code: a shared `CARGO_TARGET_DIR` between a worktree and main
   linked the HEAD `qd-runtime` into the green build (same relative package path → same artifact
   names; mtime said fresh). Anyone comparing two checkouts must give each its own target dir.
6. In the DevCouncil lane: its backend-name check would have rejected `ensemble/<n>[a,b]`, so
   every real ensemble answer would have read `reply_unparseable`. Fixed there, with a test.

Measured, not a defect: a macOS `AF_UNIX` connect on a full backlog fails `ECONNREFUSED` at once
(errno 61, 0.0 ms), so connect cannot hang (GAP-UNIX-CONNECT-HAS-NO-TIMEOUT-2026-10-06, closed).

## DevMap audit

- Lappi-decision gen 3460: Rust call resolution net 28.9%; every walk on the socket surface came
  back `walk_incomplete`. Callers of `ask_over_socket`, `read_one_line` and `handle_line` are
  DevMap lower bounds (GAP-DEVMAP-RUST-CALL-RESOLUTION-28PCT-2026-10-06). The app seams were
  located with DevMap first and rg after it came back truncated; the lanes say which.
- Dead-symbol candidates at confidence 0.9, **not deleted** (other lanes' code, one signal only):
  `qd-mutate/src/noul_rows/own_prose.rs::FileRecord`, `python/qd_train/replay.py::replay_texts_from_prompts`.
- Clones: the two `parse_header`s (`qd-export/src/safetensors.rs`, `qd-metal/src/safetensors.rs`,
  ~1,000 nodes each) are the largest structural clone; not touched here.

## Open (gap ids)

GAP-CALLER-REQUEST-SHAPES-PROPOSED-NOT-TRAINED-2026-10-06 (v6 must render the pinned strings) ·
GAP-CALLER-RECORD-SECRET-PREFIXES-ARE-A-SECOND-PATTERN-SET-2026-10-06 (needs a dependency yes) ·
GAP-CALLER-RECORDS-HELD-OUT-BY-PATH-ONLY-2026-10-06 · GAP-GITPULSE-LAPPI-CONTEXT-IS-LOSSY-UTF8-2026-10-06 ·
GAP-GITPULSE-AI-TESTS-READ-REAL-TOOLS-JSON-AND-COULD-RECORD-2026-10-06 ·
GAP-DEVTYPE-LAPPI-TOOL-CALL-WIRING-UNTESTED-2026-10-06 · GAP-CALLER-RECORD-DROP-COUNTS-NOT-SURFACED-2026-10-06 ·
GAP-DEVMAP-RUST-CALL-RESOLUTION-28PCT-2026-10-06 · GAP-DEVTYPE-LAPPI-LANE-NO-LISTAGENTS-2026-10-06.

**For the human:** (1) whether caller records may ever become training or eval data ("Synthetic
only" says no today); (2) whether to commit the four repos' changes (nothing is committed);
(3) a live check against the real model, which is a GPU job under the one-heavy-job rule.

## First command for the next lane

Run these in order, one heavy job at a time, from the repo root. The first job also builds
`target/debug/qd-caller-records`, which the app suites use to check what they write. The last
three re-run the app suites (green at 18:50Z, above).

```bash
bash tools/mac_heavy.sh lappi-caller-recheck cargo test --no-fail-fast -p qd-runtime --test client_deadline --test serve_stress --test caller_records --test two_modes_one_contract --test lifecycle
```

```bash
gofmt -l /Users/bharath/Code/devtools/DevCouncil/backend/go_orchestrator/devcouncil/lappi /Users/bharath/Code/devtools/DevCouncil/backend/go_orchestrator/devcouncil/verify
```

```bash
bash tools/mac_heavy.sh devcouncil-lappi-vet go -C /Users/bharath/Code/devtools/DevCouncil/backend/go_orchestrator vet ./devcouncil/lappi/ ./devcouncil/verify/
```

```bash
QD_CALLER_RECORDS_BIN=/Users/bharath/Code/research/Lappi-decision/target/debug/qd-caller-records bash tools/mac_heavy.sh devcouncil-lappi-test go -C /Users/bharath/Code/devtools/DevCouncil/backend/go_orchestrator test ./devcouncil/lappi/... ./devcouncil/verify/... -count=1
```

```bash
QD_CALLER_RECORDS_BIN=/Users/bharath/Code/research/Lappi-decision/target/debug/qd-caller-records bash tools/mac_heavy.sh devtype-lappi-test /Users/bharath/Code/apps/DevType/Scripts/test.sh --package-path /Users/bharath/Code/apps/DevType --filter 'Lappi|PaletteToolRouting|LocalizationParity|AIPreferencesCoverage'
```

Want: `gofmt` prints nothing. Every other command exits 0. The new tests
`TestAReplyWrittenAtAcceptIsReadEvenWhenTheRequestWriteFails` (Go) and
`testAReplyWrittenAtAcceptIsReadEvenWhenTheRequestWriteFails` (Swift) pass.

The scratch worktree `build/client-red-wt` (HEAD a683c27, for red runs) was removed at the end of
this lane with its own target dir.
