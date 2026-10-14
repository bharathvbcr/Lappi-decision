# Fable ruling: wave 3 of the 2026-10-13 GitPulse backlog run

Ruled 2026-10-13 against `main` 05b8237 and the unmerged integration branch
`wave-integration-2026-10-13b` 174452f. Read-only session: nothing in the tree was changed except
this file. Labels: **[V]** read or run this session, with the file:line or command; **[I]** inferred;
**[U]** not verified.

## 0. Basis and limits

- **Board** [V]: `gitpulse_list_tasks` returned 37 cards. No card is `in_progress`. Owner and
  needs-human cards (ft-bd80c52e, ft-b4f737de, ft-be802e07, ft-1b5f5f56, ft-12da8e67, ft-a37989f8,
  ft-afb31d47, ft-f137c48e, ft-ac6677c7) are not touched by any lane below.
- **GitPulse Insights** [V]: 9 worktrees scanned, collisions facet `ok`, 0 overlapping files,
  0 shared-worktree files; agents facet `ok`, one live Claude session (pid 12401, `main` checkout,
  busy: the orchestrator). The L3 worktree `agent-aee488bfe3833f10e` reports 12 dirty files;
  no wave-3 lane may use any existing agent worktree — each lane cuts its own.
- **DevMap** [V]: gen 3887, `is_fresh=false`, `analyzer_freshness=false`
  ("stored extraction payload is obsolete"). `devmap_impact python/qd_data/defect_class.py`
  depth 2: 1,139 edges, `truncated`, `walk_incomplete`. `devmap_impact crates/qd-train-metal/src/run.rs`:
  20 edges (callers `lib.rs`, `main.rs::main`), `walk_incomplete`. The index was built from `main`,
  so L3's `_own_repos_licences` is not in the graph at all. Every blast radius below is a lower
  bound and reads as "unknown", not "safe". Lane W3-1 records this as a `GAP-` (section 3).
- **Branch state** [V] (`git log --oneline -1` per branch): `main` 05b8237;
  `wave-integration-2026-10-13b` 174452f = main + L2 (ed9dfa8, af83ae0) + L4 (fe577f4, 02d7a08)
  + L3 (5ba07a7, bd618af) + W2-A (8507c44). `git diff --stat main..174452f`: 17 files,
  +3,136 / −73; the only Python source it changes is `python/qd_data/defect_class.py` (+204) with
  `python/tests/test_defect_class_own_repos.py` (+352, new) and `tools/arm_b_mine.py` (+338).
- **Hard limits honoured**: no box or cluster job; no heavy MPS job (ft-0eb828f1's Metal-memory
  fix is in review, Mac risk not accepted); no new dependency; no migration; no public API, CLI or
  MCP contract change; nothing listed under a card's "Owner decisions".
- **Harness**: `git -C`, heredocs and `python3 <scratchpad script>` are blocked by the MANVI gate;
  lanes run git from the repository root, write scripts inside the lane's own tree, and use literal
  paths.

## 1. Candidates checked against HEAD, not the cards

| Candidate | What the card says | What HEAD says [V] | Ruling |
|---|---|---|---|
| ft-a4ef800 item 1 (needle verdict string) | bug open | Fixed in 39ccdbc: `python/qd_train/needle.py:485-490` names the bucket's position (`_POSITION[worst.label]`); the "losing early context" string is gone; `python/tests/test_needle_verdict_position.py` pins it | Done; card stale |
| ft-a4ef800 items 2, 3 | needle per-case split; 16-option miss re-read | Both written up in `HANDOFF/needle-end-of-context-2026-10-12.md` with ledger row ids (166f7ebd, e1bafd6f, 2af3c80d, e09b652a, a51c5bf4, 6af73bef) and the `choice:16` = 15-option naming finding; the per-case script and verdict files live in `~/qd-campaign` and `build/` (the `build/needle-span-mac-2026-10-06/` dir is gone from this Mac: `build/` holds only `mac-heavy.lock.d`) | Done; nothing CPU-feasible is left that is not already read |
| ft-a4ef800 item 4 (span arm e) | run or record blocked | Needs the MPS feature extraction; `HANDOFF/span-rescore-2026-10-12.md`: B partly run, seed-1 extraction killed four times, no row | Blocked (heavy MPS); record only |
| ft-a4ef800 item 5 (OOD feature-space) | feasibility note | `HANDOFF/ood-feature-space-feasibility-2026-10-12.md` exists; gap owner "v6 planning" | Done |
| ft-a4ef800 gap records | — | GAP-16-OPTION-CALIBRATION-FAILS-AND-HAS-NO-ID-2026-10-06, GAP-SPAN-CANDIDATE-STATE-HAS-NOT-READ-ITS-LINE-2026-10-06, GAP-OOD-ABSTAIN-HAS-NO-FEATURE-SPACE-SCORE-2026-10-06 are `open` with owners; GAP-V5-NEEDLE-END-OF-CONTEXT-MISSES-SEEDS-0-2-HAVE-NO-ID-2026-10-06 updated 2026-10-12 with owner "arm-(e) lane"; GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06 last record is from the merge-and-bench lane and I could not read its status field in the cut [U] | Ledger reconciliation folds into W3-4 |
| ft-a59b211c (own-repos loader) | inbox, P3 | `_own_repos_licences` (:533), `_own_pool_records` (:666), `_own_files_list` (:684), the download clash check (:625-636) exist only on the L3 branch; `main`'s `defect_class.py` has the `OWN_REPOS_SOURCE_ID` admission at :1260-1279 and none of the loader | Lane W3-1, base 174452f |
| ft-5c92df4c remainder | 39 older NAV/ListAgents duplicates, 81 ojas/tessl/cuda listed-only, criterion 4 | L4's triage is on 174452f (02d7a08). `qd_train.gaps.append_gap` (`python/qd_train/gaps.py:156`) with `REQUIRED_KEYS = {id, question, status}` (:42). `~/Code/research/ojas` and `~/Code/research/tessl` are on disk for read-only evidence | Lane W3-4, base 174452f, sole gaps.jsonl owner |
| ft-3163ce (d) + CLI `--resume` | not covered: no 64-vocab shard set | gpu_tiny.rs:362-366 (8507c44) says so; `tools/qd_train_oracle_shards.py:79,107-112` builds `shards-tiny` at `SOURCE_VOCAB = 512` with a remap keeping 256; tessl's `tests/fixtures/qwen35_train/config.json:36` is `vocab_size: 64`; `run()` (`run.rs:152`) requires the identity remap over the snapshot vocab, checks `--resume`'s recipe at :189-203 and appends one `quick` row at :382; the h64 head init `crates/qd-train/tests/fixtures/span-head-init-seed0-h64.safetensors` exists; `fixture_pins.rs:46-47` pins `shards-tiny` and `span-head` by `SHA256SUMS` | CPU-cheap to build, in scope: Lane W3-2, base 174452f |
| ft-2ed2fd55 torch tests, gpu_d17 | not run | `python/tests/test_lora_reference.py:1` "tiny synthetic tower, CPU only", `importorskip` torch + transformers (:24-25); `~/.venvs/ml` has torch 2.12.1, transformers 5.19.0 with `models/qwen3_5`; `test_ckpt_average_lora_merged.py` likewise. `crates/qd-metal/tests/gpu_d17.rs`: two `#[ignore = "GPU"]` tests on the committed tiny fixture (`tests/fixtures/span/tiny`) and one `#[ignore = "GPU + model snapshot"]` (:107-108) that loads the real 2B via `resolve_snapshot` | Lane W3-3 runs the CPU torch tests and the two tiny-fixture Metal tests; the real-2B test stays NOT RUN |
| ft-a73b8f81 / `QD_PYTHON` | in review | The three Rust Python-oracle tests (`qd-prep/tests/convert_licence_parity.rs:30-38`, `qd-metal/tests/ledger_rows.rs:33-41`, `qd-preflight/tests/schema_compat.rs:43-51`) each resolve `QD_PYTHON` → `<repo>/.venv/bin/python` → `python3` and **fail loudly by design** (the skip pattern was retired on purpose). The L2 symptom is a worktree with no `.venv` on a host whose `python3` is Xcode's 3.9.6 (`/usr/bin/python3`; `.venv` is 3.13.14); the W2-A/integration worktree worked around it with an untracked `.venv` symlink. `Makefile:132` `CARGO_TEST_RUN` exports no `QD_PYTHON` | Actionable but small and not top-4: deferred (section 5) |
| ft-0f5a10c9 (span serving) | inbox P1 | 862ab53 and 244e103 landed `crates/qd-metal/src/span.rs`, `tests/span_cpu.rs`, `tests/gpu_span.rs`; GAP-J7-EXPORT-SPAN-HEAD-UNSERVED is `resolved-with-residual`. Criteria 2-3 need a re-score (GPU/MPS) and the v5 checkpoint JSON bodies on the box | Owner-gated; deferred |

## 2. The four lanes, in order

No two lanes share a planned file. Lanes W3-1, W3-2 and W3-4 are cut from **174452f** because the
code or ledger state they extend exists only there; W3-3 is cut from **main 05b8237**. If the human
declines 174452f, W3-1 cannot be rebased onto `main` (its target code is absent), W3-2 can be
re-cut onto `main` + 8507c44, and W3-4 onto `main` + fe577f4 + 02d7a08 + both `-gapsfix` tips.

### W3-1 — ft-a59b211c2dff160e5deae98dd5708171: own-repos loader provenance hardening

- **Base**: 174452f. Branch `wave3-l1-own-repos-loader`. Merge-conflict risk against 174452f:
  none (no other lane touches these files); against `main`: not mergeable without L3.
- **Scope (acceptance to meet)**:
  1. Each own-repos pool row's `sha256(source)` is verified against the blob at the pinned head of
     its repo **or** the loader itself verifies the pool manifest's pinned sha, so the guard no
     longer depends on `tools/arm_b_mine.py`. The lane states which of the two it implements and
     why; if the pinned-head check needs repo checkouts that are not on this Mac, it implements the
     manifest-sha check and records the other as a residual with the reason.
  2. `path` refused when absolute, empty, or containing a `..` segment (also a NUL byte and a
     backslash-separated path), red-first.
  3. The download clash check compares like with like (content digest against download content)
     or is removed with the rationale in the record; a download manifest with empty `languages` is
     refused, red-first.
  4. The v5 commitpackft load stays byte-identical (pin 931c0f63…): the existing parity test is
     re-run and its digest quoted.
- **Excluded**: anything in ft-ac6677c7 (caller-record guard, owner-gated); any change that admits
  more than today (this lane only narrows); any pool rebuild or MPS probe.
- **Planned files**: `python/qd_data/defect_class.py`, `python/tests/test_defect_class_own_repos.py`,
  `tools/arm_b_mine.py`, `python/tests/test_arm_b_mine.py`; ≤3 `gaps.jsonl` appends (its own ids,
  including the DevMap gap in section 3).
- **Risk**: low; a data-admission guard, fail-closed. Security-impact statement required in the
  handoff: the change narrows admission and widens nothing.
- **Regression test**: the red-first tests in `test_defect_class_own_repos.py` (each must fail on
  174452f and pass on the lane tip, with both outputs kept); `make pytest` on the touched files;
  `make lint`; the 931c0f63… parity.

### W3-2 — ft-3163ce2315f5d63c03d05de888434469 (d) and the CLI `--resume` path: a 64-vocab shard fixture and a `run()`-level resume test

- **Base**: 174452f (needs 8507c44's `gpu_tiny.rs`). Branch `wave3-l2-run-resume-fixture`.
  Merge-conflict risk: `gpu_tiny.rs` only against W2-A itself, which is already in the base.
- **Ruling on the design question**: the **fixture**, not a `run()` reader seam. A seam changes
  `run()`'s surface for a test's convenience and bypasses the held-out door the test exists to
  exercise; the fixture goes through the real door, the real remap check and the real ledger
  writer.
- **Scope (acceptance to meet)**:
  1. `tools/qd_train_oracle_shards.py` gains a documented option (e.g. `--vocab 64`) that writes a
     separate fixture set `crates/qd-train/tests/fixtures/shards-tiny-v64/` whose remap is the
     identity over 64 ids; `shards-tiny` and its `SHA256SUMS` stay byte-identical (the lane diffs
     them and says so). The new set gets its own `SHA256SUMS` and an entry in
     `crates/qd-train/tests/fixture_pins.rs`.
  2. A new `#[ignore = "GPU"]` test in `crates/qd-train-metal/tests/gpu_tiny.rs` that drives
     `run()` (`TrainArgs`, `args.rs:61`) on tessl's `qwen35_train` snapshot with that fixture, the
     h64 head init and a temp ledger path: the completed run writes exactly one row (`quick`,
     `optimizer_recipe` not `master`), and a run killed before completion writes none.
  3. A second test that resumes through `run()` with `resume: Some(dir)` (and the CLI parser in
     `tests/args.rs` for `--resume`), asserting the recipe check at `run.rs:189-203` refuses a
     changed recipe and that a matching resume runs to `StepsExhausted` with the final checkpoint
     byte-equal to an uninterrupted run.
- **Excluded**: items 4 and 9 (the qd-train → ojas-train move, owner's cross-repo pin); item 5b
  (ruled by design); any change to `run.rs` semantics; the `write_checkpoint`-refuses-existing-
  `.partial` edge, which `run()` cannot reach (record it, do not change it).
- **Planned files**: `tools/qd_train_oracle_shards.py`, `crates/qd-train/tests/fixtures/shards-tiny-v64/**`
  (new), `crates/qd-train/tests/fixture_pins.rs`, `crates/qd-train-metal/tests/gpu_tiny.rs`,
  `crates/qd-train-metal/tests/args.rs`; ≤2 `gaps.jsonl` appends.
- **Metal footprint**: the tiny tower (2 layers, hidden 64, vocab 64), the same class as W2-A's
  tests that ran in ~11 s this run. This is not a heavy MPS job. The lane runs its GPU tests only
  after `ListAgents` shows no other Metal/MPS job live and after agreeing a window with W3-3 by
  `SendMessage`; the two lanes never open a Metal runtime at the same time.
- **Risk**: medium-low. The fixture generator runs under `~/.venvs/ml` (torch is needed for the
  oracle dump) — CPU, seconds. `make cargo-test` must stay green with the new tests ignored;
  `fixture_pins.rs::every_fixture_set_matches_its_pins` is the check that the pinned sets did not
  move.
- **Regression test**: these are characterisation tests (W2-A found no bug), so "red first" is
  satisfied by the fixture's absence: the lane shows the new tests fail to open the door before the
  fixture exists and pass after. Plus `cargo test -p qd-train --test fixture_pins`,
  `cargo test -p qd-train-metal` (ignored tests listed as ignored, not passed), the two new tests
  under `--release --ignored --test-threads=1`, and `make clippy`.

### W3-3 — ft-2ed2fd55e93dc1019b02deb7e1e2c754: run what was reported NOT RUN (verification-only)

- **Base**: main 05b8237. Branch `wave3-l3-lora-notrun`. Merge-conflict risk: none (adds one
  `AUDIT/` file; no source).
- **Scope (acceptance to meet)**: each item below is reported with its real counts and exit status;
  a check that could not run is reported as NOT RUN, never as green.
  1. `python/tests/test_lora_reference.py` and `python/tests/test_ckpt_average_lora_merged.py`
     through the Makefile's torch launcher pattern (`Makefile:165`,
     `uv run --no-project --python ~/.venvs/ml/bin/python --with pytest --with hypothesis --with datasketch python -m pytest <files> -o addopts=`).
     CPU only.
  2. `cargo test --release -p qd-metal --test gpu_d17 -- --ignored --test-threads=1 --skip gpu_snapshot_d17_zero_is_the_tied_embedding_on_the_real_2b`:
     the two tiny-fixture Metal tests. The real-2B snapshot test is **excluded** and reported NOT
     RUN with the reason (it loads the 2B on Metal; the Mac risk is not accepted).
  3. Recompute the sha256 of the two approved downloads named in
     `AUDIT/lora-2b-2026-10-09/downloads.md` and compare with the recorded snapshot hashes (disk
     only; a few minutes).
  4. Write `AUDIT/lora-2b-2026-10-13/notrun-closure.md` with the outputs, and log the result on the
     card. The card stays in review: (b′)/(d′) wait on ojas's Tape StepProvider.
- **Excluded**: any code change. If a test fails, the lane stops, records the failure verbatim and
  files or folds a card; it does not fix on this lane (a fix would need its own red-first test and
  could land on files W3-2 holds).
- **Planned files**: `AUDIT/lora-2b-2026-10-13/notrun-closure.md` (new); a `gaps.jsonl` append only
  if something could not run.
- **Risk**: lowest of the four. Metal: tiny fixture only, serialised with W3-2 (above).
- **Regression test**: not applicable (no change); the deliverable is the evidence file with the
  pytest summary line and cargo's `test result:` lines quoted.

### W3-4 — ft-5c92df4cc912f172600c150dbd6c891f remainder, plus the ledger reconciliation of ft-a4ef800 and ft-ab55b250

- **Base**: 174452f (holds L4's 02d7a08 and both `-gapsfix` tips; cutting from `main` would conflict
  with every wave-1/2 branch's appends). Branch `wave3-l4-gaps-remainder`.
- **Scope (acceptance to meet)**:
  1. The 6 ListAgents and 33 NAV duplicates dated ≤ 2026-10-03: create the NAV canonical id L4
     left uncreated, then close each duplicate with `duplicate_of` pointing at the right canonical
     (two-tool records name both), each as an appended record.
  2. The 81 ojas/tessl/cuda records that were only listed: re-check each **read-only** against
     `~/Code/research/ojas` and `~/Code/research/tessl` at their current HEADs (cite the commit and
     file:line, `rg` is allowed there because those trees have no DevMap index for this session —
     say so in each record) and close, keep, or re-own with evidence. A record whose evidence is
     only a doc claim (e.g. the README recycle-step claim at line 721) stays open with that said.
  3. Criterion 4 as written: annotate the other-repo records in **this** `gaps.jsonl` with the
     owning board card ids already listed on ft-5c92df4c; no write to another repo's board.
  4. ft-a4ef800 ledger reconciliation: if
     GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06's latest record is
     still open, close it citing 39ccdbc and `test_needle_verdict_position.py`; append a record
     on GAP-SPAN-CANDIDATE-STATE-HAS-NOT-READ-ITS-LINE-2026-10-06 that item 4
     (arm e) is blocked on the MPS extraction with `HANDOFF/span-rescore-2026-10-12.md` as the
     reason; then log on the card that items 1, 2, 3, 5 are done (citing the two 10-12 handoffs
     and 39ccdbc) and move it to review. No code.
  5. ft-ab55b250 criterion 1's missing append: a record on GAP-DEVMAP-RUST-CALL-RESOLUTION-28PCT-2026-10-06
     naming DevCouncil ft-5ec7b111af07d336b89f2f31044a5767 (the L4 lane declined to append because
     `main`'s checkout had the file dirty; this lane's own worktree does not).
  6. Carry both numbers: open ids by latest record before and after, and the count of records
     appended, with the cap the lane set itself.
- **Excluded**: any edit of an existing line; `main`'s two uncommitted lines (owner decision 2 /
  ft-be802e07); any other repo's board; anything needing a run.
- **Planned files**: `gaps.jsonl` only (through `qd_train.gaps.append_gap`, `O_APPEND` + `fsync`,
  one line per record).
- **Risk**: low in code, medium in judgment — the 10-13 adversarial pass found false merges in L4's
  consolidation, so every `duplicate_of` here names the matching text of both records, and
  keyword-only matches are labelled as such.
- **Regression test**: `python/tests/test_gaps_ledger.py`, `test_gaps_writer.py`,
  `test_wire_gap_pins.py` (L4 ran 42 passed, 1 skipped; the lane quotes its own counts) plus a
  recount script kept in the lane tree, not in `build/`.

## 3. The `gaps.jsonl` conflict rule for this wave

- **One owner**: W3-4 is the only lane whose planned files include `gaps.jsonl`. W3-1, W3-2 and
  W3-3 may append at most a handful of records for their own `GAP-` ids (the DevMap gap on L3's
  code is W3-1's to record: "DevMap gen 3887 indexes `main`; `_own_repos_licences` and its callers
  are absent from the graph; blast radius taken by direct read").
- **Resolution is the union**: every lane appends only; on integration the conflict is resolved by
  keeping both sides' lines, verified line by line against each lane tip (the procedure the wave-1
  integration used), and `test_gaps_ledger.py` is the gate after the merge.
- **Merge order**: W3-3, W3-1, W3-2, then W3-4 last, so the large append sits on top and the
  small ones never have to be re-resolved.
- **Never**: `git stash`, `git add -A`, or a read-modify-write of the file, in any worktree.

## 4. Base-commit and conflict matrix

| Lane | Base | Files vs other wave-3 lanes | vs 174452f | vs `main` alone |
|---|---|---|---|---|
| W3-1 | 174452f | disjoint | clean | not mergeable (needs L3) |
| W3-2 | 174452f | disjoint | clean | re-cut onto main + 8507c44 |
| W3-3 | 05b8237 | disjoint (AUDIT only) | clean | clean |
| W3-4 | 174452f | `gaps.jsonl` (sole owner; others append ≤ a few lines) | union | re-cut onto main + L4 + gapsfix tips |

## 5. Deferred, and why

| Item | Why not in wave 3 |
|---|---|
| ft-a4ef800 item 4 (span arm e) and the Swift/TS two-filler needle suite variant | Both need a model pass on MPS (the extraction OOM'd at 48 GiB on 10-06; `HANDOFF/span-rescore-2026-10-12.md` B killed four times). Building a report-only suite without scoring it has no value. Human's Mac-risk yes first. |
| ft-0f5a10c9 criteria 2-3 | Re-score with `--verdicts-out` (GPU/MPS) and the v5 checkpoint JSON bodies on the box. The loader half is already on main (862ab53, 244e103). Owner. |
| ft-2ed2fd55 real-2B `gpu_d17` test, (b′)/(d′) | 2B on Metal (Mac risk); ojas Tape StepProvider. |
| ft-3163ce items 4, 9 | qd-train → ojas-train move: cross-repo pin, owner's call. |
| `QD_PYTHON` class fix (ft-a73b8f81 residual) | Real but small: `Makefile:132` could export `QD_PYTHON ?= $(PY)` when `$(PY)` is executable, and the three identical `python_candidates`/`require_python_from` resolvers could converge on one shared test helper (a path crate or a `tests/common` include). Not top-4 by value; file it on ft-a73b8f81's card as the remaining actionable piece rather than a new card. |
| ft-505105 items 1, 4; ft-cbbc092b; ft-882ed8fc; ft-2d0a5e4b C3; ft-882ed8fc C2 | Box, training, or already ruled deferred. |
| Owner decisions 1-8 in `BUILD_REPORT.md` | The human's. |

## 6. What I could not verify

- The `status` of GAP-NEEDLE-VERDICT-STRING-NAMES-A-CAUSE-THE-BUCKET-CONTRADICTS-2026-10-06's latest
  record (the gate blocked my JSON reader; the plain-text cut did not show the field). W3-4 reads it.
- Whether the own-repos checkouts needed for W3-1's pinned-head blob check are on this Mac. W3-1
  decides between the two acceptance forms on that basis and says which.
- The torch-pytest and cargo-test gates on 174452f itself were inferred green by the build report,
  not re-run; W3-3's runs are on `main`, not on 174452f.
- `dev verify` remains NOT RUN for every card (no `.devcouncil/config.yaml`); nothing moves to
  `done` in wave 3 either.

## 7. First commands

- W3-1: `git worktree add .claude/worktrees/wave3-l1 -b wave3-l1-own-repos-loader 174452f`, then
  `.venv/bin/python -m pytest python/tests/test_defect_class_own_repos.py -o addopts= -q` (expect
  green before any edit; the new tests must then be red).
- W3-2: `git worktree add .claude/worktrees/wave3-l2 -b wave3-l2-run-resume-fixture 174452f`, then
  `cargo test -p qd-train --test fixture_pins` (expect green) and read
  `crates/qd-train-metal/tests/gpu_tiny.rs:355-372`.
- W3-3: `git worktree add .claude/worktrees/wave3-l3 -b wave3-l3-lora-notrun 05b8237`, then the
  torch launcher line in section 2 on the two test files.
- W3-4: `git worktree add .claude/worktrees/wave3-l4 -b wave3-l4-gaps-remainder 174452f`, then
  `.venv/bin/python -m pytest python/tests/test_gaps_ledger.py -o addopts= -q` (expect green) and a
  recount of open ids by latest record, kept as the "before" number.
