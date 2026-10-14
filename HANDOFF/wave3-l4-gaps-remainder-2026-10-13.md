# HANDOFF wave 3, lane W3-4: gaps.jsonl remainder (ft-5c92df4c, revision 6) and ft-a4ef800 / ft-ab55b250 ledger reconciliation

Branch `wave3-l4-gaps-remainder`, cut from 174452f. Only `gaps.jsonl` changed in the tracked tree, plus
`tools/w3l4/` (the scripts that produced the appends). Labels: **[V]** run or read this session, **[I]**
inferred, **[U]** not verified. No `complete_task` was called; the orchestrator owns the cards.

## Both numbers (latest record per id; closed = resolved, closed, resolved-with-residual, closed-no-defect)

| | lines | ids | open |
|---|---|---|---|
| before (174452f) | 1,449 | 933 | 503 |
| after | 1,577 | 934 | 448 |

128 records appended, all through `qd_train.gaps.append_gap` (O_APPEND + fsync, one line each), no existing
line touched. 934 ids: one new id (the NAV canonical). Open: 503 + 1 (new canonical) - 48 duplicates
- 7 code-evidenced closures - 1 needle-verdict closure = 448 [V, `tools/w3l4/recount.py gaps.jsonl`].
The cap I set myself: at most 150 appended records and no more than one closure per record I had
read the text of. It was not reached; what limited the work is below under "Not done".

| appended | n |
|---|---|
| NAV canonical `GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03` (open) | 1 |
| duplicates closed with `duplicate_of` (item 1) | 48 |
| closures with ojas/tessl code evidence (item 2) | 7 |
| keeps and re-owns with evidence (item 2) | 22 |
| board-card annotations (item 3) | 13 |
| repair lines (see below) | 34 |
| ft-a4ef800 / ft-ab55b250 records (items 4, 5) | 3 |

## Item 1: the NAV canonical and 48 duplicates (15 ListAgents-only, 6 GitPulse-trust-only, 27 composite)

L4's list of "39" was not preserved anywhere in the tree (its commits touch `gaps.jsonl` only), so I
re-enumerated open records dated 2026-10-03 or earlier that state a tooling limit. 48 is therefore my
count, not a reproduction of 6 + 33. Method [V, `tools/w3l4/dups.py`]: each duplicate is assigned the
component classes it states (no DevMap store in the worktree, GitPulse REPOSITORY_TRUST_REQUIRED,
ListAgents unavailable, DevMap database malformed, real_ft_run.py parse failure, plus the NAV class),
and the script requires a regex hit for each class in that record's own text. A class with no hit is
dropped from `duplicate_of` and named in the record's `answer`. Each closure carries
`duplicate_text` (the matching sentence of the duplicate) and `canonical_text` (the matching sentence of
each canonical); both are asserted to be verbatim substrings. Seven records matched no NAV sentence and
so name only their component canonicals.

The NAV canonical L4 left uncreated: I interpreted it as the composite "three navigation tools, three
limits, findings rest on rg and Read" id that the multi-tool `*-NAVIGATION-*` records share. It stays
open, owner DevCouncil ft-620d41f2 / GitPulse ft-8e09f101 (cards from ft-5c92df4c). That interpretation is
mine [I]; if L4 meant something else, the 27 composite closures are the ones to revisit.

Left open on purpose (doubt means no merge): `GAP-L-REPLAY-NAVIGATION-2026-10-02` (carries a code_commit
dirty-build provenance fact), `GAP-L-PREP-NAVIGATION-2026-10-02` and `GAP-L-V5DATA-NAVIGATION-2026-10-02`
(worktree base was 456 and 509 commits behind the brief), `GAP-L-V5-QUEUE-NAVIGATION-2026-10-02` and
`...-AB-...` (shell functions not indexed), `GAP-L-LINT-NAVIGATION-2026-10-02` (DevMap has no
sha-pin query), `GAP-RESCORE-NAVIGATION-2026-10-02` (affected_tests misses subprocess-driven tests),
`GAP-FABLE-CLINC-RULING-NAVIGATION-2026-10-02` (unproven qd-prep parity of a Python oracle),
`GAP-IDLE-ADVICE-NAVIGATION-2026-10-02` (keyword-only: no tool failure, only "made no call"),
the two FABLE-INFER navigation records of 2026-10-03 (tessl GitPulse facet, different cause),
`GAP-BRANCH-CLEANUP-GITPULSE-COLLISIONS-UNSCANNED-2026-10-03`, `GAP-CUDAEMU-1002-DEVMAP-GITPULSE-UNAVAILABLE`
(cloud session where the tools do not exist), the ojas-not-indexed DevMap records (a different cause),
and the FTS5 / malformed DevMap records outside L4's canonical (`...FTS5-CORRUPT...`).

## Item 2: ojas / tessl / cuda records, re-checked read-only

Method [V]: rg and Read on the two trees, no DevMap index covers ojas (no `.devmap` at its root) and tessl's
is `is_fresh=false`, so none of it is graph-confirmed, and each record says so. The harness refuses git
against a foreign tree, so HEADs were read from `.git/refs/heads/main` (ojas **bc498c4**, 0 dirty files by
`gitpulse_active_changes`) and `.git/logs/HEAD` (tessl **2972027**, 4 dirty files: CHANGELOG.md,
docs/qwen35.md, src/qwen35_adamw.rs, tests/qwen35_train.rs, none cited as evidence). No build or test
was run in either tree.

The "81" list was also not preserved. My filter (id, lane, tool or owner mentions ojas, tessl, cuda or
rsijev) matches 92 open ids at 174452f; I touched 49 of them, and 43 are untouched, many of which are
Lappi-owned records that merely mention a GPU or a sibling crate (they are named at the end).

Closed (7): `GAP-FABLE-CUDA-CUDARC-TWO-PINS-UNVERIFIED-2026-10-01` (one pin: ojas-cuda/Cargo.toml:22-30,
Cargo.lock:127-128), `GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01` (resolved-with-residual:
ojas-qwen35/src/config.rs:22-30 and :545-585, tests/cpu_config.rs:293; the real-2B check
gpu_parity.rs:735 is an ignored GPU test, not run), `GAP-L-CUDA-ORACLE-TESSL-PATH-INCLUDE-2026-10-01`
(in-crate sha-pinned copy, ojas-cuda/tests/reference_gdn_published_vs_tessl.rs:35, fmt_boundary.rs:127),
`GAP-TESSL-GELU-DUP` (tessl kernels/act.h:24-30 is the one definition; tests/common/mod.rs:472-476),
`GAP-TESSL-HOST-LEASE-EXCLUSIVE-BLOCKS-PARALLEL-MAPPING-2026-10-03` (src/runtime.rs:854-860, :552, :728),
and both tessl stale-worktree records (`gitpulse_insights` on tessl today: collisions ok, 10 scanned, 0
failed, no wt_head/wt_norm under `.git/worktrees`).

Kept or re-owned with evidence (22): PROMOTED-44 (tests/promoted_kernels.rs:120-131 still asserts 44),
ABSENT-KEY-DEFAULTS (ojas config.rs:286-291 still says inferred), GDN-FIXTURE-RULE-UNNAMED (tessl fixture
names unchanged, re-owned to tessl), STAGING-HEADROOM (ojas state.rs:496,536,735 still stage; ojas
ft-c57c98e0), GPU-PARITY-NOT-RUN (10 `#[ignore]` items, not 7), CLIP-COEF (ojas step.rs:259 returns f32), the
14 CUDA device-run records (re-owned to ojas ft-71629756; **doc-only evidence**: ojas-cuda/README.md:39-43,
ojas/tasks/gp-cuda-backend-provider.md:79,106, and this repo's HANDOFF/lead-pipeline-2026-10-03.md:540-542
report a 2026-10-05 GH200 run of the pre-merge standalone binaries whose logs are on the box, not here;
the merged crate has not run on a device, so they stay open), `GAP-CUDA-EMU-STAGED-IN-LAPPI-BELONGS-IN-OJAS-2026-10-12`
(ojas still has no cuda_emu, `rg -uu`), and `GAP-TESSL-GDN-VALUE-HEADS-UNVERIFIED-FOR-4B-9B-2026-10-06`
(tessl's step repeats key heads over value heads, src/qwen35_train.rs:23-25,693-695; the refusal is
ojas-qwen35/src/config.rs:504-510; whether qd-train-metal over tessl trains a 4B is not shown).

Already closed before this lane (no action; the card's line numbers 721, 748, 762, 872, 873 are
`resolved` or `closed` by their latest record): the ojas Metal gradient-read fault, the cargo-fmt-edits-tessl
record, the rung-b untied fixture record, and the two K8 NaN records (closed 2026-10-07 by the L-CUDA consolidation, no duplicate).
The README's "recycle step" claim for 721 is therefore moot here.

## Item 3: board annotations (13, annotation only, nothing re-verified, no other board written)

ojas ft-71629756 (device group above), ojas ft-c57c98e0, tessl ft-62f3bba (stale worktrees, closed),
tessl ft-b7376d2e / ft-836fdba (value heads), tessl ft-7d20b218 (three rsijev-metal records),
DevCouncil ft-72b89073 (`GAP-DEVMAP-METAL-ENTRY-POINTS-ARE-NOT-SYMBOLS-2026-10-06`, `GAP-DEVMAP-METAL-EDGES`),
ft-32522da4 (`GAP-OJAS-TESSL-DEVMAP-BLIND-SPOTS-2026-10-06`, `GAP-OJAS-QWEN35-DEVMAP-WALK-INCOMPLETE-2026-10-01`),
ft-620d41f2 (`GAP-STRESS-SUBAGENT-NAVIGATED-WITHOUT-DEVMAP-MCP-2026-10-06`), ft-400418e8 (struct-literal,
already resolved), GitPulse ft-8e09f101 (`GAP-DEVTYPE-LAPPI-LANE-NO-LISTAGENTS-2026-10-06`) and ft-d10620bf
(the two GitPulse caller records). `GAP-GITPULSE-N8N-EMAIL-ENGINE-UNTRUSTED-2026-10-06`: no card can be filed
until n8n-email-engine is trusted; recorded as such.

**Repair [V].** My first pass appended re-check records carrying only question/status/answer, which hid
each open record's earlier finding/impact text in the latest line. `tools/w3l4/merge.py` appended 34
further lines that restore the pre-lane record and put this lane's check under `recheck_wave3_l4`. If you
re-run `evidence.py`, run `merge.py` after it. This doubled some ids' appended lines; it is the price of append-only.

## Item 4: ft-a4ef800 (evidence for the card log)

- Item 1 done: `GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06` had one record,
  `open`; closed `resolved`, `evidence_commit` 39ccdbc [V: `git cat-file -t` commit, ancestor of HEAD].
  `python/qd_train/needle.py:485-490` names `_POSITION[worst.label]` and asserts no cause;
  `python/tests/test_needle_verdict_position.py` has 5 test functions (:41,:49,:55,:62,:67) and **13 passed in
  0.19s** here; no source under python/ or tools/ still contains "losing early context" except that test's
  docstring (:3).
- Items 2 and 3 done: `HANDOFF/needle-end-of-context-2026-10-12.md:33-38` (per-seed rows 166f7ebd, e1bafd6f,
  2af3c80d, e09b652a, a51c5bf4, 6af73bef with worst bucket and miss counts), `:68-69` (owner of
  `GAP-V5-NEEDLE-END-OF-CONTEXT-MISSES-SEEDS-0-2-HAVE-NO-ID-2026-10-06` = the arm-(e) lane), `:78-80`
  (`choice:16` has 15 options; the 16-real-option entry is `choice:17`).
- Item 5 done: `HANDOFF/ood-feature-space-feasibility-2026-10-12.md:3` (owner v6 planning), and the file exists.
- Item 4 blocked: appended on `GAP-SPAN-CANDIDATE-STATE-HAS-NOT-READ-ITS-LINE-2026-10-06` (still open):
  `HANDOFF/span-rescore-2026-10-12.md:3` and `:19` (seed-1 val extraction 4,811 of 7,238 rows; killed by
  exit 143 three times plus once detached; no arm row, no (e)-vs-(c) reading). Needs the human's Mac-risk yes.
- The five ledger rows named in the Fable ruling live in `ledger/` and the box, not re-queried here [U].

## Item 5: ft-ab55b250

Appended on the exact id `GAP-DEVMAP-RUST-CALL-RESOLUTION-28PCT-2026-10-06` (status stays
resolved-with-residual): `board_card` names DevCouncil ft-5ec7b111af07d336b89f2f31044a5767. ft-5c92df4c lists
a different DevCouncil card (ft-32522da4) for Rust call resolution; I recorded both and did not decide
whether they are the same work [U].

## Verify [V]

- `python/tests/test_gaps_ledger.py`, `test_gaps_writer.py`, `test_wire_gap_pins.py`: **44 passed, 1 skipped**
  (torch). (L4 had 42/1 on its base; this base has more tests. The first run failed one test because scratch
  output files under `tools/` quoted ids; deleted, then green.)
- `make lint` reports NotRun here (no ruff in this worktree's `.venv`); `ruff check .` with the main
  checkout's ruff: **All checks passed** over the whole tree, and `ruff format` applied to `tools/w3l4`.
  The four large scripts carry `# ruff: noqa: E501` because they hold long evidence strings.

## Not done

- 43 of the 92 keyword-matched ojas/tessl/cuda ids were not examined: the Lappi-owned ones that only mention
  a GPU (SIDECAR-BIG-ENDIAN, MEMORY-CAPACITY, SCORE-CHECKPOINT-QUICK, K7-GOLDEN-PINS, QDM-*, KAHAN, LORA-CKPT,
  QD-METAL-PINS, WORKSPACE-NOT-RUSTFMT-CLEAN) and the ojas/tessl ones I did not reach (the OJAS-RUNGD-* and
  OJAS-ADVICE-* estimates, the cuBLAS and ORACLE-GOLDEN records, GDN-FIXTURE-STORES-UNCLAMPED-ALPHA, K3-IDENTITY,
  TESSL-INTEGRATION-AUDIT-LINE-CITES, the eight CUDAEMU-1002-* records, SMALL-NO-LOCAL-CUDA-COMPILER). They stay open
  with their old text; none was closed or called checked.
- The GH200 logs behind the 2026-10-05 device run are not in this tree, so no CUDA device record could be closed.
- No `git` against ojas or tessl (harness), so HEADs are from ref files.
- Not pushed, not merged; no card completed.

## Rework after the Opus review of 6a5bc4f (supersedes the sections above where they conflict)

The scripts under `tools/w3l4/` were one-shot and are removed from the branch (`git rm -r`); the commit
`6a5bc4f` still holds them. 60 further lines were appended (append-only, each carrying every earlier field
forward). Counts: **before the rework 1,577 lines / 934 ids / 448 open; after 1,637 lines / 936 ids / 452 open**
(+2 new canonicals, +2 reopened). Total appended by this lane: 128 + 60 = 188.

- **Struct-literal record restored.** `GAP-DEVMAP-DEAD-SYMBOLS-MISSES-STRUCT-LITERAL-USE-2026-10-06` again carries its
  real answer (DevCouncil ec243008, FileRecord kept at own_prose.rs:952, replay_texts_from_prompts deleted in
  a861d08) and updated/owner/tool, plus the ft-400418e8 annotation.
- **Reopened, with reasons in the records.** `GAP-DEVMAP-UNIFIED-CALLERS-UNRESOLVED-R1-2026-09-30` (it is a
  call-resolution gap, related to `GAP-DEVMAP-RUST-CALL-RESOLUTION-28PCT-2026-10-06`, not a navigation duplicate) and
  `GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01` (its falsifier, ojas-qwen35/tests/gpu_parity.rs:735, is an ignored
  GPU test that has not run; the ojas-side refusal evidence is kept in the record, so the closed count of item 2 is 6, not 7).
- **Hidden limits: I chose the two-canonical route, applied to the whole class.** New canonicals
  `GAP-GITPULSE-COLLISIONS-FACET-PARTIAL-IN-LANE-SESSIONS-2026-10-03` and
  `GAP-DEVMAP-WALK-INCOMPLETE-OR-TRUNCATED-IN-LANE-NAVIGATION-2026-10-03`. Rather than only the five named records, every
  one of the 47 remaining duplicates was re-scanned in its own text and re-pointed at the new canonicals where it states the
  limit (6 links to the collisions canonical and 12 to the walk canonical, 13 records in all, including the five: L-RECENCY, L-V5-RULES, FABLE-G-AVG-NEEDLE-CONTROL, L-ROOM-J6A, FABLE-J6A-RULING).
  Nothing was dropped.
- **NOT-USED-NEEDLE-OOD-LANE and OOD-ABSTAIN-DECISION-NAV-UNGRAPHED** now say the cause is DevMap tools not loaded, not a missing store.
  **OJAS-ADVICE-NO-LISTAGENTS and OJAS-ADVICE-GITPULSE-UNTRUSTED**: re-checked; the tool limit was real and the closure stays for that, but
  the lead's ListAgents answered the concurrency question afterwards (c1688c busy, 00e12d idle), recorded in both.
- **Quote correction.** The earlier `duplicate_text` quotes were matched against the record's fields joined with spaces. All 47
  were re-derived field by field (`{"field", "quote"}` for the duplicate and for each canonical); no class failed to match.
  The correction is stated on `GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03`. The ListAgents canonical now has an owner.
- **Field restoration [V].** Every duplicate and evidence closure line now carries the base record's fields (owner, tool, impact, finding,
  triage fields and so on); the original `answer` of a closed record is kept as `answer_before_closure`. Check run after the appends:
  for every one of the **93 ids this lane touched that had an earlier record**, the latest record's keys are a superset of the
  last pre-lane record's keys: **0 violations** (before the rework the same check found 56).
- **evidence_commit wording** on the six evidence closures now reads, e.g., "ojas bc498c4 (the working tree as read at that HEAD ... by
  read-only rg/Read; no build or test was run)", with `evidence_basis: read-only; nothing run`.

## First command for the next lane

`rg -c '"id"' gaps.jsonl` is not enough; count open ids by latest record (the rule in `qd_train.gaps.current_records` plus
`CLOSED_STATUSES`) and expect 1,637 lines, 936 ids, 452 open on this branch, then the three gap tests.
Merge order per the ruling: W3-4 last, resolving `gaps.jsonl` as the union of both sides' lines.
