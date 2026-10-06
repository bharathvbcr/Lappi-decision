# The caller contract: how any app asks Lappi, and what it may record

`docs/schema-api.md` is the wire contract: what a request and a reply *are*. This document is the
**caller's** side of it: how an app connects, how long it waits, what it does with each kind of
reply, what it must never let Lappi decide, and what it may write down for later. It is written for
any app. DevCouncil (Go), DevType (Swift) and GitPulse (Rust) are its first three users, and each
implements it natively. No app links the runtime in-process: the model is ~2B parameters, and a
palette keystroke must never be what loads it.

Status, 2026-10-06: **every caller is wired, and none of them is served yet.** The v0.1 preview's
`trained_families` contain none of the callers' decisions (`code.defect_class` is trained but not
servable; see below). Every request below is refused today, and that refusal is what the wiring
shows until v6 trains these families. That is the intended state: wired, refusing, collecting.
It is not "integrated" in the sense of Lappi changing any app's behaviour.

## 1. Transport

- **Socket.** A Unix stream socket, JSON lines: one request line out, one reply line back.
  Default path `$HOME/Library/Caches/qd/qd.sock` (`qd serve` and `qd-metal-serve` both default
  to it). An app may let the user override it, and `LAPPI_SOCKET` in the environment overrides
  both. The socket is `0600` and per-user.
- **One deadline for the whole exchange.** Connect, write and the complete reply line share a
  single deadline. A per-read timeout is not one: an agent that drips bytes resets it forever
  (`crates/qd-runtime/tests/client_deadline.rs` pins this for the runtime's own client, and each
  app's client carries the same test). Interactive callers use ≤ 300 ms; batch callers ≤ 10 s.
  The first request after the agent starts loads the model (3.37 s measured on the preview), so
  an interactive caller's first ask is expected to come back `unavailable`.
- **Never an in-process fallback** for an app. `qd oneshot --socket` falls back to answering
  in-process because dcverify must run with nothing else up. An app must not: no socket means
  `unavailable`, and the app keeps its own path.
- **Bounds before sending.** A request line over `MAX_PAYLOAD_BYTES` (`crates/qd-runtime/src/wire.rs`)
  is not sent: the agent would refuse it and close the connection. Build `context_b64` from the
  context's **bytes**, never from a re-encoded string (`docs/schema-api.md`, the `String` rule).
- **Read one line, bounded** to the same cap. A connection closed before a newline is
  `unavailable` (`closed_without_reply`), not an empty answer.

## 2. Reading a reply: six outcomes, closed

| Outcome | When | What the app does |
| --- | --- | --- |
| `model_answered` | `status: ok`, at least one slot not `noul` | may use it **only as §3 allows** |
| `model_abstained` | `status: ok`, every slot `noul` | its own path |
| `request_refused` | `status: refused` | its own path; record `refusal` as the kind |
| `backend_failed` | `status: error` | its own path; record `error` as the kind |
| `unavailable` | no socket, connect refused, deadline passed, reply not one line, reply not parseable | its own path; record why (`socket_not_found`, `connect_refused`, `deadline`, `closed_without_reply`, `reply_over_cap`, `reply_unparseable`) |
| `not_asked` | the app chose not to send (asking disabled, over a bound, secure input) | its own path |

The first four are `Response::caller_reading` (`crates/qd-runtime/src/schema.rs`). An app matches
all six. A reply carrying a `status` the app does not know is `unavailable` / `reply_unparseable`,
never a guess. `degraded: true` on an answer (a non-model backend, such as the reference
backend) is treated as `model_abstained`: a reference backend's answer is not a model's.

## 3. What Lappi may decide: admission only

Rule 7 of `CLAUDE.md`, applied to callers. **Lappi's answer can only narrow, rank or annotate
among options the app already considers valid. It never grants, passes, allows, or discharges
anything.**

- **DevCouncil**: a Lappi answer may add an **advisory, non-blocking** gap
  (`GapType` prefixed `advisory_lappi_`) to a verify run. It never removes, rewrites or demotes
  another gap, never sets `Blocking`, never names a requirement or acceptance criterion, and its
  gap types are never in `MayDemote`'s path. Since `verified` means "no blocking gap", an advisory
  gap cannot create a pass; the rule above is what keeps it from removing a fail.
- **GitPulse**: Lappi is asked only when the deterministic classifier returned no type
  (`high_confidence: false`). An answer may pre-select a type in the draft the user edits; it is
  never committed without the user, and it never touches the hook contract, which emits no
  `allow` on any path (`GitPulse/src-tauri/src/hooks/mod.rs:27-28`).
- **DevType**: an answer may reorder or annotate the palette's routed row. It never inserts text
  on its own and never runs a macro the user did not pick.

## 4. Ask only trained shapes

The runtime refuses a task its release did not train (`task_not_trained`), and for
`code.defect_class` a context not in the trained shape. A caller must send **exactly** the trained
request: task, question and slot names are rendered into the prompt, so a renamed one serves a
prompt the model never saw. The shapes below are pinned here. For the two families no release
trains yet, the shape is **proposed for v6**: the v6 builder must render these exact strings, or
this document and the caller change together (`GAP-CALLER-REQUEST-SHAPES-PROPOSED-NOT-TRAINED-2026-10-06`).

| Decision point | `task` | `question` | slots | context | Served by v0.1? |
| --- | --- | --- | --- | --- | --- |
| DevCouncil per-file defect | `code.defect_class` | `What kind of change is this diff, and which lines does it touch?` | `defect_class` choice `[stub, logic, cosmetic, clean]`; `defect_span` span | `file: <path>`, a blank line, that file's unified-diff hunks only (no `diff --git` / `---` / `+++` preamble) | **No**: `calibration_entry_missing` on the span slot |
| GitPulse ambiguous commit type | `gitpulse.commit_type` | `Which conventional commit type describes this staged change?` | `commit_type` choice `[feat, fix, refactor, docs, test, chore, perf, build, ci, style, revert]` | the staged unified diff, as bytes | **No**: `task_not_trained` (proposed for v6) |
| DevType palette route | `devtype.route` | `Which DevType tool should answer this palette query?` | `route` choice `[resolveDate, runMacro, findSnippet, none]` | the query, as bytes | **No**: `task_not_trained` (proposed for v6) |

**More candidates than a slot holds: the caller pre-filters.** A `choice` slot holds at most 16
options (`docs/schema-api.md`, the letter budget), and the runtime refuses more rather than
dropping any. A caller choosing among a larger catalogue narrows it deterministically *before*
asking, and records the narrowing in `facts`. For example, a tool-selection caller with Manvi's 45
tools would go through a pre-filter such as DevCouncil's `devcouncil_search_tools` first.
(The data-clean lane reports that its tool-selection rows offer at most 13 tools plus three
actions; not verified here, because no committed file carries that layout yet.) Lappi then
chooses only among what the pre-filter kept, which is admission (§3) applied to the candidate
list itself.

Each request also carries `schema_version: 1`, `route: "generic"`, `context_len`, an
`example_id` of the form `<app>:<decision_point>:<record_id>` and `metadata: {}`.

## 5. Caller records: what an app may write down

The point of recording is to learn which decisions callers actually face, how often Lappi is
asked, and what the user really did next. That is the evidence v6's caller families and their
held-out sets need (`docs/train-plan-2026-09-28.md`, open item 1: "whether query logs exist").

### The constraint that governs everything below

The human, 2026-10-06: **"Synthetic only"**. No real personal data (emails, forms, queries) goes
into training or eval. So caller records are **not training data and not eval data**, and nothing
an app or a lane does can make them so. Admitting any of it needs a new human decision and a new
`record_version`. Until then:

- **Off by default** in every app, behind its own explicit setting. `LAPPI_COLLECT=0` in the
  environment forces it off regardless of the setting.
- **Local only.** Nothing is uploaded, synced or sent anywhere, including to Lappi: recording is
  independent of asking.
- **Held out by path.** The store is
  `$HOME/Library/Application Support/Lappi/heldout/caller-records/<app>/<YYYY-MM-DD>.jsonl`
  (`caller_record::STORE_RELATIVE_TO_HOME`). The `heldout` segment makes `qd-train`'s rule-3 door
  refuse it with no caller-specific code (`crates/qd-train/tests/caller_records_are_held_out.rs`).
  Records are never copied into `data/` or any repository: the door refuses by path, so a copy is
  exactly how they would leak.
- **`admission: "not_admitted"`** on every line; it is the only legal value under version 1.

### Format: `lappi.caller_record`, version 1

One JSON object per line. Two kinds share a `record_id` (32 lowercase hex, random): a `decision`
line written when the app decides, and at most one `outcome` line written when the app later sees
what the user actually did. `crates/qd-runtime/src/caller_record.rs` is the checker and
`fixtures/caller/valid-records.jsonl` holds valid examples of each.

```
decision: record, record_version, kind="decision", record_id, app, app_version,
          decision_point, created_at, admission, facts{}, app_choice{}, lappi{}, redaction{}
outcome:  record, record_version, kind="outcome", record_id, app, app_version,
          decision_point, created_at, admission, observed{}, redaction{}
lappi:    asked, task, reading (one of the six outcomes in §2), kind, backend, slots, latency_ms
redaction: policy (the app's redactor), fields_redacted
```

Unknown keys are refused at every level, and `lappi` must be consistent: `asked` is false exactly
when `reading` is `not_asked`; a refusal, error or unavailability names its `kind`; only an answer
carries `slots` and `backend`.

### What goes in `facts`: structure, not content

Record the **facts the decision was made from**, not the raw inputs, wherever the facts suffice:

- GitPulse: the classifier's tally (file counts by role and change kind, truncation, whether the
  repo is conventional). No patch text and no file contents. The outcome is the final message's
  parsed type, scope and breaking flag, not the message.
- DevCouncil: per file, the language, hunk and line counts, and the verify gaps' types. No diff
  text.
- DevType: the query's length and shape (`DiagnosticPrivacy.textShape`), never its text. No
  record at all while secure input is active or the focused element is a secure field. The
  outcome is which palette row the user picked (routed, another command, a snippet, cancel).

Every string still passes the app's own redactor before it is written, and `validate_line`
refuses a line carrying a credential-shaped string anywhere, as a backstop.

### Writing

- `O_APPEND | O_CREAT`, file mode `0600`, directories `0700`; one `write(2)` per complete line
  including its newline, then `fsync`. Never read-modify-write.
- A line over 64 KiB is not written. A day's file over 32 MiB, or a store over 256 MiB, stops
  recording for that app until the user clears it; the app counts what it dropped and says so in
  its diagnostics. A failed write never changes the app's decision and is logged once.
- `qd-caller-records <files>` checks a store: it reports counts, invalid lines with reasons,
  duplicate ids and unpaired outcomes, and exits 2 if anything is invalid. It admits nothing.

## 6. Where each app implements this

| App | Client | Decision point | Deadline | Records | Setting (default off) |
| --- | --- | --- | --- | --- | --- |
| DevCouncil | `backend/go_orchestrator/devcouncil/lappi/` (Go, stdlib `net`) | `verify.Run`, after `runRigorGates`; at most 16 files, 10 s in total | per file, inside the 10 s | one decision per file; no outcome | `DEVCOUNCIL_LAPPI_ASK=1`, `DEVCOUNCIL_LAPPI_COLLECT=1` |
| GitPulse | `src-tauri/src/lappi/` (Rust, std `UnixStream`) | `ai::generate_commit_message`, drafts with no type and `high_confidence: false`, in a conventional repo | 2 s (off the UI thread) | decision at draft, outcome at commit (and amend) | Settings → "Ask Lappi on ambiguous commit types", "Record Lappi caller data (local only)" |
| DevType | `Sources/ExpanderEngine/Lappi/` (Swift, POSIX socket) | palette routing, alongside the on-device router (macOS 26) | 300 ms | decision at route, outcome at pick or cancel; none under secure input or without Accessibility trust | `devtype.lappi.askEnabled`, `devtype.lappi.recordEnabled` |

Each client carries the drip test from §1 (one byte per 150 ms against a 500 ms deadline, back in
under 1.5 s), a test that maps all six outcomes, and a cross-check that writes records through
its real writer and runs `qd-caller-records` over them (`QD_CALLER_RECORDS_BIN`); without the
binary the cross-check reports itself **not run**, never passed. All three cross-checks ran
2026-10-06 and exited 0.
