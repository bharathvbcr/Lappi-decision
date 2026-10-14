# Fable ruling: the own-repos loader admits text it never tied to the pool (ft-a59b211c, 8dabde3)

Date: 2026-10-13. Judge: Fable. Card: ft-a59b211c2dff160e5deae98dd5708171 (own-repos loader
provenance hardening), branch `wave3-l1-own-repos-loader`, head 8dabde3 (= 174452f + ab80131 +
8dabde3). Read-only session: nothing in any tree was changed except this file. Labels: **[V]**
read or run this session with the file:line or command; **[I]** inferred; **[U]** not verified
by me (the orchestrator's measurement, taken as reported).

## 0. Basis and limits

- **Code** [V]: every `file:line` below is from `git show 8dabde3:<path>` (and
  `git show 174452f:...`, `git diff 174452f..b15d753` for W4-1). DevMap is generation 3887,
  `is_fresh=false`, `analyzer_freshness=false` (`devmap_status`, this session), so nothing here is
  graph-confirmed; the DevMap gap on this code is already recorded
  (`GAP-DEVMAP-INDEX-LACKS-OWN-REPOS-LOADER-2026-10-13`, per the lane's handoff).
- **GitPulse Insights** [V]: `collisions.ok=true` but **`truncated=true`, `unscanned_worktrees=3`**
  (16 of 19 scanned), so the collision scan is incomplete, not clean. The one overlap it found
  between `wave3-l1-own-repos-loader` and `wave4-l1-own-swift-clean-rows` is `.venv` (the
  untracked symlink both worktrees carry), not a source file. One live Claude session (pid 12401,
  the `main` checkout, busy). `ListAgents` is not in this session's tool set; the implementing
  lane runs it and records both facts in one `GAP-` record (section 7).
- **W4-1 state** [V]: branch tip b15d753 = 174452f + 2b2435f (compose refuses a single-version
  base holding clean rows) + b15d753 (`materialise` loads each pinned composed corpus through
  `load_defect_rows` first). Its worktree holds `data/pool/own-swift-corpus-v2/examples.jsonl`
  (648,482,245 bytes, 21:10) and **no amendment yet**: its `campaign/v6-arm-b-mining-preregistered.json`
  is byte-identical to 174452f's (`git show 174452f:... | diff -`, exit 0). Its 6 dirty files
  were not read.
- **The measurements in the brief** (10,894/10,894 plain old-sides equal the pool source; `+`
  lines <= 45 per plain row; 5,689/5,689 filler blocks equal the pool source; 879 needle blocks,
  748 one edit region, 131 several, <= 170 lines changed) are **[U]**: not re-run here. The ruling
  below does not depend on them, which is part of why it is the ruling.
- **Spot checks run** [V]: a real own-swift composed row's constituents carry
  `"role":"filler","example_id":null,"example_source":null` for fillers and
  `"role":"needle","example_id":"own-swift:bharathvbcr/m5blade:...swift#10","example_source":"base"`
  for the needle; the composed manifest says `"record_mode": "single_version"`.

## 1. The defect, located

The loader's licence admission for an own-repos pool (`_own_repos_licences`,
`defect_class.py:582-708`) proves, correctly, that the **pool** is the human's: manifest pinned
(`:628`), pool and files list bound by sha (`:638`, `:647`), repos in the pinned own-repos
manifest (`:660`), one source only (`:684-700`). The text the model reads is not the pool. It is
`DefectRow.diff`, built from `ex.diff` in `_parse_one` (`:1009-1033`), and the only tie between an
example and the pool is `_own_example_content_violation` (`:842-867`): `before` (and `after` for
`clean`) must hash to the normalised pool source. It returns `None` for any row with
`composed: True` (`:855`) and never reads `diff`. So:

- (a) a plain row's `diff` can carry anything: the hunk's `-`/context lines, the `+` lines, all
  of it, provided `before` is the pool file;
- (b) a composed row has no `before`; `_composed_diff_span` (`:886-943`) checks tiling, the three
  header lines and span geometry only, and `_composed_violation` (`:946-984`) checks pool ids,
  repos, licences and splits only, so every block body is free text;
- (c) a noul scrambled row over the own pool is admitted by `_noul_violation` (`:1339-1348`) on
  pool id, repo and licence equality; its `diff` is free text.

In each case the row is emitted with `licence = "owner-granted"` from `dict.fromkeys(pool, licence)`
(`:708`). The corpus manifest's `examples_sha256` is self-declared and `_check_sha` (`:1097`) only
proves the file is the one the manifest describes. `tools/arm_b_mine.py` checks the
pre-registration's pin (`check_input`, `:99-117`) in `plan` and `materialise` only, and
`materialise` writes a new `examples.jsonl` whose `examples_sha256` it recomputes (`:527`). The
finding stands as described.

## 2. Ruling: option B, in the form that needs no new artifact

**The loader admits an own-repos corpus's rows only if the examples file's bytes are ones the Arm B
pre-registration pins, mirrored into a code constant beside `OWN_POOL_MANIFEST_SHA256S`; a
materialised subset is admitted only by proving every row is a line of that pinned file.**

Why B and not A, C or D:

1. **The loader's existing design is "the pre-registration is the provenance oracle; the loader
   refuses what it did not name."** That is what `OWN_POOL_MANIFEST_SHA256S` (`:165`) already does
   for the pool, with the comment "a self-declared manifest is not provenance" (`:631`). The
   examples file is the same kind of claim and gets the same treatment. One principle, one seam.
2. **A's caps are new admission numbers fitted to the one build they would admit.** 45 and 170
   are facts about own-swift-v1 at seed 20261008, not invariants of `generate.rs`: a mutated
   example's `+` lines are whatever the operator wrote into one function body
   (`generate.rs:679-740`), bounded only by the body's length, and the composed needle is the whole
   mutated file re-rendered (`compose.rs:1049-1056`). A cap chosen from the data at hand is the
   pattern rule 2 exists to stop an agent doing, and even with the caps A leaves the inserted text
   unproven: up to 45 lines x 10,894 rows of free text would still be admitted as owner-granted.
   Refused.
3. **D (tie the needle to its base example) is exact for the composed needle** — the needle's
   `example_id` names the base row and `example_source` says `base` [V] — but it only moves the
   hole to the base row's `+` lines, which still need A's cap. Refused for the same reason, as
   this lane's work. The cap-free parts of A and D (plain old-side equals the pool source at each
   `@@`; filler block equals `unified_multi("", source)`; needle block equals
   `unified_multi("", base_row.after)`) are **integrity** checks, not licence checks: under B they
   can only catch a generator bug in bytes the human already registered. Permitted as a later
   hardening on this card's log, not this lane.
4. **B does not need the per-row sha file the brief describes.** `materialise` copies kept rows
   byte for byte (`arm_b_mine.py:518-524`: `body += raw + b"\n"` on the raw split line) and records
   `selected_by_plan.source_examples_sha256` (`:534`), and the pinned source corpus sits under
   `data/pool/<name>/` on the machine that runs the loader (the shard build is
   `tools/real_tokenizer_pipeline.py:3440` on the Mac, where `data/pool` holds the pinned corpora
   [I — the lane confirms by running it]). So membership is proved against the pinned file
   itself; nothing new is committed except one constant and its mirror test.

Nothing in this ruling moves a gate, seed, threshold, held-out set or cap. It narrows admission
and widens nothing. Adding a pin is a code edit that mirrors an amendment the pre-registration's
own rule already governs ("before any scoring"); it is within the run's autonomy, and the human is
told in the handoff. **No threshold is introduced, so nothing here needs a human yes.**

## 3. Exact acceptance

One new constant and one new helper in `python/qd_data/defect_class.py`, the canonical owner.

- `OWN_CORPUS_PINS: Final[Mapping[str, str]]` — corpus directory name (the parent of the
  pre-registered `path`) -> `examples.jsonl` sha256, for every `own_repos_train` corpus the
  pre-registration pins **and its base corpus**. `superseded_pin` values are never in it.
- `_require_pinned_own_corpus(examples_path, manifest, *, repo_root, config) -> None`, called
  from exactly two places: `load_defect_rows` once `_own_repos_pool_manifest(pool_path)` returned
  non-`None` (after `_own_repos_licences`, before any row is read), and `_load_noul` under the
  same condition on its own `pool_path`. `_load_composite` routes scrambled parts through
  `_load_noul` (`:1890-1893`), so the composite is covered at the same seam. The check runs
  **after** `_check_sha(examples_path, manifest["examples_sha256"])` (`:1097`, `:1393`), so the
  sha it judges is the file's.

What it admits, by kind:

| Kind | Admitted iff | Refusal text names |
|---|---|---|
| **Plain** (a base such as own-swift-corpus-v2) | `manifest["examples_sha256"] in OWN_CORPUS_PINS.values()` | the sha, the constant, and "a self-declared corpus is not provenance; register it by amendment" |
| **Composed, as pinned** | the composed `examples.jsonl` passes the same test; its base is loaded by `_load_composed_base` -> `load_defect_rows` (`:1283-1286`) and passes its own | same |
| **Composed, materialised** (and any other materialised own corpus) | `manifest["selected_by_plan"]["source_examples_sha256"]` is a pinned value; `name` is the key it belongs to; `src = repo_root/"data"/"pool"/name/"examples.jsonl"` passes `assert_path_not_held_out` and `assert_no_caller_records`; `sha256(src) == pin`; **every** non-empty line of the materialised file is, byte for byte, a line of `src` (compare raw bytes split on `b"\n"`, the way `materialise` wrote them, not through `jsonl_lines` text decoding); the number of such lines equals `selected_by_plan["kept"]` and `totals["examples"]` | which of those failed; an absent, unreadable or mismatching source file refuses with "could not be proved against the pinned corpus", never passes |
| **Empty** (0 bytes: a `-planbase`, or a materialised corpus the plan kept nothing from) | admitted without a pin: zero rows is zero text, and the composed rows over a planbase are judged on their own file | — |
| **Noul over an own pool** (scrambled; any source whose `pool_id` resolves through an own pool) | the noul `examples.jsonl` sha is pinned. No noul corpus over an own pool is registered today, so **every** such corpus is refused until one is | same |
| **Noul over commitpackft, v5 composite, every commitpackft corpus** | untouched: `_own_repos_pool_manifest` returns `None` (`:566`) and the helper is not reached | — |

Kept as they are: `_own_example_content_violation` (still exact for `before`/`after`; a second,
cheap tie), every check in `_own_repos_licences`, `_composed_diff_span`, `_composed_violation`,
`_noul_violation`. No caps. No change to `tools/arm_b_mine.py`, `test_arm_b_mine.py`, `compose.rs`
or the campaign file (W4-1's files).

Mirror test (the template is `test_the_pinned_pool_manifest_is_the_one_the_preregistration_pins`,
`test_defect_class_own_repos.py:610-636`): read the pre-registration; for every corpus with
`own_repos_train`, take `{Path(path).parent.name: sha256}` and
`{Path(base_corpus.path).parent.name: base_corpus.sha256}`; ignore `superseded_pin`; assert the
dict equals `OWN_CORPUS_PINS` and is non-empty. It is the tripwire for every ordering mistake in
section 4.

## 4. Interaction and ordering with W4-1 (ft-b1dfb6da)

Who pins what:

- **W4-1 pins corpora in the pre-registration** (its file). Its amendment for own-swift v2 must
  carry, beside the composed `path`/`sha256`, a **structured** base pin — today the base's sha
  (08c8e2fb...) appears only inside the amendment prose, which no test can read:

  ```json
  "base_corpus": {
    "path": "data/pool/own-swift-corpus-v2/examples.jsonl",
    "sha256": "<sha256 of the v2 base examples.jsonl, built twice identical>",
    "rows": "<totals.examples>",
    "why": "read by load_defect_rows before the composed rows; pinned so the loader can mirror it"
  }
  ```

  This is an addition to W4-1's scope from my earlier ruling (`fable-own-swift-clean-rows-ruling.md`,
  "Files", campaign bullet); the orchestrator relays it to W4-1 by `SendMessage`. It is one field,
  before any scoring, under the pre-registration's own amendment rule.
- **This lane (ft-a59b211c) owns `OWN_CORPUS_PINS` and the mirror test.** W4-1 never edits
  `defect_class.py` (unchanged from my earlier ruling).

Order: **W4-1 merges first, then this lane.** Reasons: (i) the values this lane pins do not exist
until W4-1's amendment commit; (ii) W4-1's acceptance ("composed-v2 loads as pinned through
`load_defect_rows`") must be met against the loader at its own base, then **re-met** after this
lane lands, because this lane narrows admission and would refuse the v2 corpora without the pins.

Concretely for the Sonnet lane on `wave3-l1-own-repos-loader`:

1. Implement the helper, the constant (empty is not allowed: start it from 174452f's
   pre-registration with the only structured own pin it has, the composed v1 sha 077aa842...,
   and expect the mirror test red until step 3) and the tests of section 5 against synthetic
   fixtures. Commit. All synthetic tests green; the mirror test red is stated in the commit body,
   not hidden.
2. Wait for W4-1's amendment commit (the orchestrator says when). Do not merge before it.
3. Merge `wave4-l1-own-swift-clean-rows` into `wave3-l1-own-repos-loader` (planned files are
   disjoint: W4-1 touches `compose.rs`, `tests/compose.rs`, `arm_b_mine.py`,
   `test_arm_b_mine.py`, the campaign file, the ledger, `gaps.jsonl`; this lane touches
   `defect_class.py` and `test_defect_class_own_repos.py` plus one `gaps.jsonl` append — on
   `gaps.jsonl` keep both sides' lines). Fill `OWN_CORPUS_PINS` from the amended file: the v2
   composed sha and the v2 base sha. Mirror test green. Commit.
4. Re-run on real data, with `repo_root` = a checkout whose `data/pool` holds the v2 corpora
   (copy them from W4-1's worktree after checking their sha256 against the amendment; W4-1's
   worktree is `locked`, do not work inside it): `load_defect_rows(own-swift-composed-v2-prefixed)`
   as pinned (rows = base rows + N) and the materialised form from `arm_b_mine.py materialise`
   (rows = kept, over the planbase). Both row counts and both outcomes go in the handoff. Also
   `python -m pytest python/tests/test_arm_b_mine.py` (W4-1's materialise test must still pass).
5. Integration merges `wave3-l1-own-repos-loader` after `wave4-l1-own-swift-clean-rows`.

If W4-1 does not amend in this run, this lane stops at step 1 with the mirror test red and says so;
it does not pin a value that is not registered, and it does not merge. Every later corpus change
is: amendment, constant edit, mirror test — that friction is the design, not a cost to engineer
away.

Both lanes record, in their own `gaps.jsonl` appends, that `ListAgents` was unavailable to the
judge and that the GitPulse collisions facet was truncated (3 worktrees unscanned) at ruling time.

## 5. Red-first tests (`python/tests/test_defect_class_own_repos.py`)

Fixture changes first, because the existing suite must keep passing for the right reason:

- `_finish_own_world` gains `pin_corpus: bool = True`: after writing `examples.jsonl`, monkeypatch
  `defect_class.OWN_CORPUS_PINS` to `{corpus.name: sha256(file)}` with `raising=False` (so the
  helper also runs against 8dabde3, which has no such name — the same device as
  `_pin_pool_manifest`, `:210-217`).
- `_rewrite` (`:219-246`) re-pins every sha "as an attacker would". An attacker cannot edit code, so
  it gains `repin_corpus: bool`. Tests of checks that run **after** the pin (the files-list,
  content-tie, clash and path tests) pass `repin_corpus=True`: they ask what happens to registered
  bytes. The three attack tests below pass `repin_corpus=False`. Without this split the new check
  would mask every later check and the suite would go green for the wrong reason.

Each test below must **fail on 8dabde3** (run the new file in a detached worktree of 8dabde3, as
the lane did for ab80131, and keep the output) and **pass on the tip**:

| Test | Setup | 8dabde3 | tip |
|---|---|---|---|
| `test_a_plain_row_whose_diff_is_not_registered_is_refused` (repro a) | `_own_world`; `_rewrite(repin_corpus=False)` replaces one row's `diff` with a GPL-headed hunk whose `-`/context lines are still the pool file; `before` untouched | 6 rows admitted owner-granted | `DefectCorpusError` naming `OWN_CORPUS_PINS` |
| `test_a_composed_row_with_unregistered_block_bodies_is_refused` (repro b) | own base (pinned) + composed corpus over it built with `test_defect_class._composed_row`/`_block` shapes but own pool ids and `qd-compose/v1` manifest; filler bodies replaced by third-party text; composed file not pinned | admitted | refused |
| `test_a_noul_scrambled_row_over_an_own_pool_is_refused_unless_registered` (repro c) | `noul_dir` whose manifest's `pool` is the own pool (same sha), one `class: noul, noul_source: scrambled` row with `pool_id` in the pool, `licence` = `OWN_LICENCE`, free-text `diff`; not pinned | admitted (the lane confirms; if 8dabde3 refuses it for an unrelated reason, adjust the fixture until the pre-fix loader admits it — the repro was verified) | refused |
| `test_a_materialised_subset_of_a_pinned_corpus_is_admitted` | pinned composed corpus; a second dir holding 2 of its 3 raw lines byte for byte, manifest with recomputed `examples_sha256`, `totals.examples=2`, `selected_by_plan={source_examples_sha256: pin, kept: 2}`, `base_corpus` pointing at an empty planbase | refused? no: **admitted on both** — this is the characterisation that the materialise path survives; not red-first, say so | admitted, 2 rows |
| `test_a_materialised_row_that_is_not_a_line_of_the_pinned_corpus_is_refused` | as above, one kept line altered by one byte (and a variant: `kept` says 2 but 3 lines present) | admitted | refused |
| `test_a_materialised_corpus_whose_source_cannot_be_read_is_refused` | as above, source `examples.jsonl` deleted; and a variant where it exists but hashes to another value | admitted | refused, message says the proof could not run |
| `test_an_empty_own_corpus_is_admitted_without_a_pin` | planbase-shaped dir, 0 bytes | admitted | admitted (characterisation) |
| `test_the_pinned_own_corpora_are_the_ones_the_preregistration_pins` | section 3's mirror | fails (no constant) | passes once step 3 of section 4 is done |

Unchanged and re-run: the 42 existing tests of this file (`_rewrite` callers updated), the v5
parity digest `931c0f63b4b333e8aa575d7457f3b5be386da84019cd232e39df8a3f84935fa9` through
`test_the_real_v5_defect_corpus_loads_byte_identically` (run from the main checkout's data, as the
lane did), `python/tests/test_defect_class.py`, `test_defect_noul*.py`, `make pytest`, `make lint`.

## 6. Does ft-a59b211c stay this lane?

**Yes.** The finding is a HIGH against acceptance item 1 of this card ("the loader itself verifies
the pin"), the fix lives in the card's two planned files, and the branch already carries the two
commits it builds on. Card to `in_progress`, revision 3, with a log line pointing here; the
handoff's residual (a) ("the composed rows' own diff text is not tied to the pool") is closed by
this commit and the handoff is amended, not rewritten. Not W4-1: that lane is a corpus rebuild
and is barred from `defect_class.py`. No new card.

Security-impact statement for the handoff: narrows admission only; new refusals are an own-repos
corpus (plain, composed, materialised or noul) whose bytes the pre-registration does not pin, and a
materialised row that is not a line of the pinned file. No access, quota, crypto or secret handling
is touched. The one behavioural risk is a false refusal of a registered corpus, which the real-data
re-run in section 4 step 4 is there to catch.

## 7. What I could not verify

- The brief's counts (10,894/10,894; 5,689/5,689; 879 needles; 45 and 170) — not re-run; the
  ruling does not rest on them.
- That the shard pipeline's loader call runs on a machine whose `data/pool` holds the pinned
  corpora (inferred from `real_tokenizer_pipeline.py:3440` and the Mac-only Arm B recipe); the
  lane's step 4 is the check.
- W4-1's uncommitted work (6 dirty files, not read) and whether it will add the structured
  `base_corpus` pin without being told; hence the relay in section 4.
- `ListAgents` (absent from this session) and the 3 worktrees GitPulse did not scan.

## 8. First commands for the lane

- `git worktree list` (expect `wave3-l1-own-repos-loader` at 8dabde3 in
  `.claude/worktrees/agent-a0d9673d86a38aba9`; work there, not in a new tree), then
  `PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_defect_class_own_repos.py -o addopts= -q`
  (expect `42 passed, 1 skipped` before any edit).
- Read `python/qd_data/defect_class.py:552-570` (`_own_repos_pool_manifest`), `:1038-1230`
  (`load_defect_rows`), `:1360-1400` (`_load_noul`'s pool and sha checks) and
  `tools/arm_b_mine.py:518-585` (what a materialised corpus looks like) before writing the helper.
