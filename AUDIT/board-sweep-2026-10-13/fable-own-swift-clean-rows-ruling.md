# Fable ruling: own-swift `#clean` rows carry an empty diff, so the Arm B Swift pin does not load

Date: 2026-10-13. Judge: Fable. Task: ft-b1dfb6da27342a82ccc48f11873d89ed (review, rev 3).
Code read at `174452f` (head of `wave-integration-2026-10-13b`, which contains L3's branch
`wave1-l3-armb-corpora` 5ba07a7 + bd618af). Data read from the L3 worktree
`.claude/worktrees/agent-aee488bfe3833f10e/data/pool/` (untracked; the only copy).

Tooling note. `devmap_status` reports generation 3887, `is_fresh: false`, `analyzer_freshness:
false` ("source tree differs from the indexed generation"), so no claim below is graph-confirmed;
every `file:line` is from `git show 174452f:<path>` and every count from reading the files on disk.
`ListAgents` is not in this session's tool set. The implementing lane appends the `GAP-` record for
both (I write only this file).

## Evidence

Verified by running the loader (`qd_data.defect_class.load_defect_rows`, ml venv, repo_root = the
L3 worktree, download_root = `data/pool/commitpackft`):

| corpus (data/pool/) | rows | clean | clean rows with empty diff | load_defect_rows |
| --- | ---: | ---: | ---: | --- |
| commitpackft-corpus-v3 (main worktree) | 49,953 | 8,449 | 0 | (not re-run; v5 loads it) |
| commitpackft-ts-v3 | 2,161 | 470 | 0 | loads as the base below |
| single-file-mutants-v1-prefixed | 10,467 | 2,290 | 0 | not re-run |
| commitpackft-composed-ts-v1-prefixed | 1,250 (all composed) | 211 | 0 | **OK, 3,411 rows** (2,161 base + 1,250) |
| own-swift-corpus-v1 | 11,141 | 247 | **247** | **refused**: `...CommandAssist.swift#clean: empty diff` |
| own-swift-composed-v1-prefixed | 1,058 (all composed) | 179 | 0 | **refused** with the same error, from its base |
| own-swift-composed-v1-planbase | 0 | 0 | 0 | OK, 0 rows |

Manifests (verified): every commitpackft-derived corpus records `diff.clean_without_prior: 0`;
`own-swift-corpus-v1` records `diff.clean_without_prior: 247`, `hunks_per_diff_by_class.clean: {"0": 247}`
and `examples_without_hunk_constraint: 11141`. `own-swift-composed-v1-prefixed` records
`record_mode: "single_version"` and `base_corpus: {name: own-swift-corpus-v1, examples_sha256: 08c8e2fb...}`.
The reviewer's finding is confirmed exactly, and it is pre-existing at 174452f.

The code (verified at 174452f):

- `crates/qd-mutate/src/generate.rs:337-361`: on a pool whose records have no `prior_source`, a
  `clean` example is emitted with `before == after` and `diff: String::new()` **on purpose** ("A pool
  walked off local sources has no prior version, and inventing one would be worse than admitting
  there is none: `refuse_leaky_diff_corpus` refuses such a corpus in diff mode rather than letting it
  train"). `generate.rs:1130-1147` pins that shape with a test. The counter `clean_without_prior`
  (`generate.rs:424`) is the generator announcing it.
- `crates/qd-prep/src/own_swift.rs:31` and `:389`: the own-swift pool record is `{id, repo, path,
  source}`; "hunks and prior_source absent (a pool walked off a pinned tree has no diff attached)".
- `crates/qd-mutate/src/compose.rs:938-951`: under `--single-version-records` the base's clean
  examples are never read ("its stored clean examples carry an empty diff ... and no composed row
  reads them"); they are counted as `single_version_clean_examples_unused` and the pristine rendering
  is `render_single_version` (`compose.rs:836-845`, whole file against the empty file). So the 247
  rows contribute nothing to the composed corpus.
- `crates/qd-mutate/src/compose.rs:923-932`: the same function already **refuses** an auxiliary
  corpus unless its manifest says `totals.clean == 0`, with a message naming `--clean-permille 0`.
  The precedent for the guard I order below is in the same function.
- `python/qd_data/defect_class.py:818-836` (`_parse_one`): refuses any row whose diff is empty,
  citing `AUDIT/after-vs-diff-leak.md` (`diff == ""` <=> `clean` on the v1 corpus; 353/353
  validation rows answered from context length). `:1057-1107` (`_load_composed_base`) loads the base
  through `load_defect_rows`, so a composed corpus inherits its base's refusal.
- `python/qd_train/mutate_adapter.py:145-170`: `is_void_diff` / `EmptyDiffContext` /
  `refuse_leaky_diff_corpus` are the training-side guards for the same leak.
- `tools/arm_b_mine.py:465-586` (`materialise`): writes a composed corpus over an empty
  `<corpus>-planbase` so "the loader reads that base's rows first. The plan holds none of them". The
  shard pipeline (`tools/real_tokenizer_pipeline.py:3440`) then loaded the *materialised* corpus,
  which is why ledger row e1ccb673 exists (914 rows written, 1,828 sequences) although the *pinned*
  corpus does not load. `plan` reads the pinned corpus with its own raw JSON reader (`_rows`,
  `arm_b_mine.py:141-151`), never the loader.

## Q1. How clean rows are represented elsewhere; emission bug or loader bug?

Neither. Every corpus that loads was generated from `commitpackft-pool-v2`, whose records carry
`prior_source`, so its `#clean` row is the agent's own commit rendered as a real unified diff
(`generate.rs:349-352`; verified 0 empty clean diffs in v3, ts-v3 and single-file; v3's manifest even
shows one `clean_diff_empty` refusal, which is the pre-image-equals-post-image drop at
`generate.rs:364-367`). A composed clean row is a multi-file pristine rendering (verified: 211 and
179 clean composed rows, none empty).

The own-swift pool is single-version: no record has a pre-image, so the generator's only honest
clean rendering is the empty one, and it says in the same breath that such a corpus is not readable
in diff mode. The loader's refusal is the deliberate guard for exactly that leak and is correct.
What went wrong is the **recipe**: the pre-registered build runs `qd-mutate generate --language
swift --seed 20261008` at the default `clean_permille: 200` (`generate.rs:121`) on a pool for which
a clean row cannot be rendered, producing 247 rows the loader refuses by design and `compose
--single-version-records` discards by design. The pin (`077aa842...`, amendment in
`campaign/v6-arm-b-mining-preregistered.json:42-62`) therefore names a corpus that the one reader
refuses, and the planbase mechanism hid it.

Ruled: emission is correct for the pool it was given; the loader is correct; the recipe omitted
`--clean-permille 0`, and nothing in the build chain refused the result.

## Q2. Canonical owner of the fix

**Not the loader.** Admitting empty-diff clean rows "in the base" is the leak of
`AUDIT/after-vs-diff-leak.md` reopened for one corpus family; it would also be a special case for a
state that should not exist. Refused. (A manifest-level early refusal in `load_defect_rows` --
`diff.clean_without_prior > 0` -> refuse before reading rows, naming `--clean-permille 0` -- is
permitted as hardening under ft-a59b211c; it changes no admission and is not required here.)

**Owner 1: qd-mutate `compose --single-version-records`** (`crates/qd-mutate/src/compose.rs`,
`prepare_with`). A single-version base whose manifest records `totals.clean > 0` (equivalently
`diff.clean_without_prior > 0`) is **refused**, not counted, with a message naming
`--clean-permille 0` -- the mirror of the aux-corpus check at `compose.rs:923-932`. Rationale: a
composed corpus names its base by sha256 and the loader reads the base first; a base the loader
refuses makes the composed corpus unloadable, so compose must not build on one. The
`single_version_clean_examples_unused` counter and the test at `crates/qd-mutate/tests/compose.rs:591-605`
encoded the wrong expectation ("counted, not hidden") and are replaced by the refusal; that is a
strengthening, and the reason is stated here.

**Owner 2: the recipe / pre-registration.** `own-swift-corpus-v1` is rebuilt as
`own-swift-corpus-v2` with `--clean-permille 0`, recomposed, and re-pinned by amendment. The
superseded pin stays in the record, as the two earlier supersessions did.

**Owner 3: `tools/arm_b_mine.py materialise`** proves each pinned composed corpus loads through
`load_defect_rows` *as pinned* (base + composed, in memory, before anything is written) and refuses
otherwise. The loader stays the one reader; `materialise` merely calls it once. This is what would
have caught the defect: the planbase is a correct device for excluding base rows from the shard, but
it must never be the only form of the corpus that loads.

Rule 2 is not touched: no gate, threshold, seed or held-out set moves. Rule 3 is served, not weakened:
the rebuilt corpora go through the loader's path check like every other.

## Q3. Does the fix change the built Swift shard (ledger e1ccb673)?

Yes, and it must be rebuilt. With `--clean-permille 0`, every own-swift record is mutated
(`generate.rs:555`: disposition is `bucket % 1000 < clean_permille`), so the 247 formerly-clean
records join the needle supply (1,016 files -> up to 1,263 [I]) and the composed sample at seed
20261008 changes; the examples bytes, the sha256, the plan's candidate set and the shard all change.
Row e1ccb673 is superseded by a new ledger row (append-only; never edited) and a `gaps.jsonl` record
saying so. The shard content behind e1ccb673 was not corrupt (its 1,058 composed rows carry
non-empty diffs; verified) -- it is superseded because its pin is not reproducible by the code at
HEAD once Owner 1 lands, not because its rows were wrong.

It is CPU-only and within the run's autonomy: `qd-mutate generate` and `compose` are deterministic
CPU builds (L3 built each twice, identical), `arm_b_mine.py plan`/`materialise` are CPU, and the
shard build is `tools/real_tokenizer_pipeline.py` tokenizer-only with `HF_HUB_OFFLINE=1` (the argv in
GAP-ARM-B-SHARDS-BUILT-PROBE-NOT-RUN-2026-10-13). No GPU, no cost, no scoring. The pre-registration's
own amendment rule ("before any scoring; no probe or mining result was read") applies: the probe has
not run (same gap record), so the pin may be superseded without a human yes. The human is told in the
handoff.

The TS and single-file shards (165ba699, 247678e1) stand **only if** the re-run plan's materialised
bytes for those two corpora are byte-identical to the previous materialisation; the lane compares
and reports both outcomes. If either differs, it is rebuilt and re-ledgered too.

## Q4. ft-b1dfb6da leaves review; fold this in

Yes. The task's CPU half rests on a pin that the canonical loader refuses and that HEAD will no
longer be able to rebuild; "mined rows with dedupe and leakage checks recorded" is not met for the
Swift corpus until the re-pin, re-plan and re-materialise land. Move it to `in_progress` (rev 4)
with a log line pointing at this ruling, and keep the work on this task: same pre-registration,
same driver, same ledger file, same three corpora. No new board card. The MPS probe stays pending
as before. The optional loader early-refusal belongs to ft-a59b211c, not here.

Side observation for the board sweep (not asked): L3 wrote no `HANDOFF/*.md` (rule 10); its
record is the task log plus five gap lines. The lane below writes one.

## Lane scope

- **Base commit:** `174452f` (`wave-integration-2026-10-13b`). Branch e.g.
  `wave4-l?-own-swift-clean-rows`. Data: copy `own-swift-pool-v1.jsonl`,
  `own-swift-pool-v1.manifest.json`, `own-swift-pool-v1.files.jsonl`,
  `own-swift-pool-v1.split-map.json` from the L3 worktree's `data/pool/` (it is `locked`; do not
  work inside it) after checking `sha256(own-swift-pool-v1.jsonl) == fd8cc926...` and the manifest
  sha `8842700d...` against the pre-registration; `data/pool/commitpackft` (download) is also needed.
- **Files:**
  - `crates/qd-mutate/src/compose.rs` (`prepare_with`): refuse a single-version base with
    `totals.clean > 0`; drop the `single_version_clean_examples_unused` counter and its manifest key.
  - `crates/qd-mutate/tests/compose.rs`: red-first test
    `a_single_version_base_with_clean_examples_is_refused_and_names_clean_permille_0` -- the
    existing `single_version_fixture` (generated at `--clean-permille 200`, line 561) must now fail
    compose with no output and a message containing `--clean-permille 0`; it passes on the pre-fix
    binary, so it is red first. The test at :591 moves to a fixture generated at
    `--clean-permille 0` and asserts the counter key is absent; the byte-identity test for the
    commit-pool path (:757) must still pass.
  - `tools/arm_b_mine.py` (`materialise`): before staging, call `load_defect_rows` on each pinned
    composed corpus (`src.parent`, `download_root` = the pre-registration's download, `config =
    DataConfig()`, `repo_root = repo`) and raise `Refusal` quoting the loader's error; record
    `loaded_rows` (base + composed) in `selected_by_plan`.
  - `python/tests/test_arm_b_mine.py`: red-first test
    `test_materialise_refuses_a_pinned_composed_corpus_whose_base_the_loader_refuses` -- a composed
    corpus (reuse the loadable composed fixture of `python/tests/test_defect_class.py:500`) whose
    base holds one `clean` row with `"diff": ""`; `materialise` must refuse before writing
    (`out_dir` absent, no `-planbase` directory). On pre-fix code it writes the planbase and
    returns, so it is red first. The existing
    `test_materialise_gives_a_composed_corpus_an_empty_plan_base` (:765) gets a loadable base
    instead of the `{"examples_sha256": "a"*64}` stub.
  - `campaign/v6-arm-b-mining-preregistered.json`: amendment superseding the own-swift pin
    (`077aa842...`, manifest `e7f35b65...`) with `own-swift-composed-v2-prefixed`, `id_prefix`
    `arm-b-swift-v2:`, the new `train_rows` N, both sha256s, built twice identical, and the reason
    ("base own-swift-corpus-v1 carries 247 clean rows with an empty diff, `clean_without_prior:
    247`, which `load_defect_rows` refuses; superseded, not rewritten: no scoring had happened").
    Note the recipe text at `:62` still says `--pool data/pool/own-swift-pool-v1/pool.jsonl`; the
    flat layout at `:204` is the one on disk -- fix the command in the amendment.
  - `ledger/arm-b-shards-2026-10-13.jsonl` (append only), `gaps.jsonl` (append only, `O_APPEND`),
    `HANDOFF/arm-b-swift-rebuild-2026-10-13.md`.
- **Build commands (recipe, in order):**
  1. `qd-mutate generate --pool data/pool/own-swift-pool-v1.jsonl --language swift --seed 20261008
     --clean-permille 0` -> `data/pool/own-swift-corpus-v2`; twice, identical.
  2. `qd-mutate compose --corpus data/pool/own-swift-corpus-v2 --pool data/pool/own-swift-pool-v1.jsonl
     --split-map data/pool/own-swift-pool-v1.split-map.json --seed 20261008 --train-rows N
     --val-rows 0 --min-tokens 4000 --max-tokens 9100 --hard-max-tokens 9400
     --max-share-permille 700 --single-version-records --id-prefix arm-b-swift-v2:`
     -> `data/pool/own-swift-composed-v2-prefixed`; N = the largest that composes by the same
     descending search L3 used (record the refusals); twice, identical.
  3. Amendment commit; `arm_b_mine.py plan` from HEAD; `materialise`; compare TS and single-file
     materialised bytes with the previous materialisation; `real_tokenizer_pipeline.py` with the
     argv of GAP-ARM-B-SHARDS-BUILT-PROBE-NOT-RUN-2026-10-13 over the Swift materialised dir
     (`--rev` = the amendment commit, not `-dirty`).
- **Acceptance (all must hold, each cited by command output in the handoff):**
  - `own-swift-corpus-v2/manifest.json`: `totals.clean == 0`, `diff.clean_without_prior == 0`;
    `load_defect_rows` returns `totals.examples` rows.
  - `own-swift-composed-v2-prefixed` loads **as pinned** through `load_defect_rows`
    (rows == base rows + N) and again as materialised (planbase).
  - `cargo test -p qd-mutate` green including the new refusal test; `cargo clippy -p qd-mutate`
    clean; `python -m pytest python/tests/test_arm_b_mine.py python/tests/test_defect_class.py
    python/tests/test_defect_class_own_repos.py` green including the new test; the two new tests
    shown red on 174452f.
  - New ledger row for the Swift shard; `gaps.jsonl` records e1ccb673 superseded, states whether
    165ba699 and 247678e1 stand (byte-identical inputs) or were rebuilt, and records the DevMap
    staleness and the absent `ListAgents` from this ruling.
  - ft-b1dfb6da moved to `in_progress` with a log line; still pending: the MPS probe, the R3
    shortfall, ft-a59b211c.
  - Nothing in `python/qd_data/defect_class.py`'s admission changes. No gate, seed or threshold moves.
