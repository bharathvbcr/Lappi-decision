# Handoff: wave 3 lane W3-1, own-repos loader provenance hardening (2026-10-13)

Card ft-a59b211c2dff160e5deae98dd5708171 (revision 2). Branch `wave3-l1-own-repos-loader`, base
174452f. Ruling: `AUDIT/board-sweep-2026-10-13/fable-wave3-ruling.md` section 2 W3-1.

## What changed (all in `python/qd_data/defect_class.py`, the canonical owner)

1. **Manifest-sha form chosen** (acceptance 1). `_own_repos_licences` now hashes the pool manifest
   itself and requires it in `OWN_POOL_MANIFEST_SHA256S` (= `8842700d...`, the value the Arm B
   pre-registration pins for `own-swift-composed-v1`), then requires the pool file to match the
   manifest's `pool.sha256`. The guard no longer depends on `tools/arm_b_mine.py`; the driver's own
   check is left in place (belt and braces). The files list was already bound to the manifest and
   the pool to the files list by content, so the pin transitively binds every row's bytes.
   Why not the pinned-head blob check: it needs the repo checkouts at `code_root`, subprocess calls
   and every pinned head present; the checkouts do exist on this Mac (`~/Code/apps`, `devtools`,
   `research`, `web`, `scholarlm`) but object presence could not be verified from this harness
   (version-control commands against other repositories are blocked) and the box has none of it.
   Recorded as residual `GAP-OWN-REPOS-PINNED-HEAD-BLOB-CHECK-NOT-IMPLEMENTED-2026-10-13`.
2. **Path refusal** (acceptance 2). `_own_relative_path` is applied to every pool record and every
   files-list row: refused when not text, empty, absolute, NUL, backslash, or a `..` segment
   (segment, not substring: `v1..2/a.swift` is still admitted). Pool `source` must be text.
3. **Clash check** (acceptance 3). The old check compared pool ids (`own-swift:<repo>:<path>`)
   with download keys (`<commit>:<new_file>`): different key spaces, so it could never fire. It now
   compares sha256(pool source) with sha256(download `new_contents`) (empty content ignored). A
   download manifest with empty or missing `languages` is refused in the shared reader
   `_download_rows`, which also hardens the v5 `_load_licences` path (v5's real manifest has four
   languages, so v5 is unchanged).

## Rework after the Opus review of ab80131 (second commit)

1. **Example content tied to pool bytes** (HIGH). Repro was: every example's before/after set to
   third-party text, corpus manifest re-pinned, 6 rows admitted owner-granted. Now
   `_own_example_content_violation` runs in `load_defect_rows` for each example of an own-repos
   pool. Shape, from `crates/qd-mutate/src/generate.rs` (the `normalized_prior`/`prior` block,
   ~lines 278, 331-334, 354, 407-408, `before: prior.to_string()`, `after: source.clone()`) and
   `normalize.rs::normalize` (strip one BOM; CRLF and lone CR to LF): `before` is the normalised
   pre-image when the pool carries one, otherwise the normalised pool `source`; an own-repos pool
   has no `prior_source`, so `before` must equal the normalised `source` for every example, and
   for `clean` `after` must too. Compared by sha256 of the normalised text. Checked against the
   real L3 own-swift corpus (11,141 examples): none refused, so the check matches how the corpus
   was built.
2. **Clash digest normalised** (MEDIUM). `_clash_digest`: strip BOM, CRLF/CR to LF, collapse the
   whitespace within and around each line (tabs against spaces, indentation, trailing blanks), drop
   trailing blank lines; empty results are ignored. Applied to the pool source and to both
   `new_contents` and `old_contents` of every download row.
3. **TOCTOU** (LOW). `_own_repos_pool_manifest` reads the bytes once and returns
   `(parsed, sha256 of those bytes)`; `_own_repos_licences` takes the sha as an argument.
4. **Path aliasing** (LOW). A segment that is empty, `.`, `..` or only those once whitespace is
   stripped refuses (`a//b`, trailing `/`, `./a`, `a/./b`, `a/ ../b`, `..\t`, `a/ /b`).

Residuals added: (a) composed rows carry no before/after (their text is the constituents' diff),
so the content tie covers plain corpora and a composed corpus only through its base corpus, which
`load_defect_rows` reads through the same door; the composed rows' own diff text is not tied to the
pool. (b) Unicode lookalikes (U+FF0E and the like) are opaque names and are not refused; pinned by
a test. (c) The files list and pool file are still re-read after their sha check (same class as
item 3, not in the review's list). The own-swift `#clean` empty-diff failure (defect_class.py
`_parse_one`) is untouched; the content check runs before `_parse_one` and neither masks nor
depends on it (the real base corpus passes the new check and then stops at that failure).
Security impact: narrows admission only; widens nothing.

## Round 3: the trained text is tied to a registered corpus (step 1 of Fable's section 4)

Ruling: `AUDIT/board-sweep-2026-10-13/fable-own-repos-diff-tie-ruling.md`. Closes residual (a)
above for plain rows, composed block bodies and noul-over-own-pool rows once step 3 fills the pins.
`OWN_CORPUS_PINS` (corpus directory -> `examples.jsonl` sha256) and `_require_pinned_own_corpus`
in `defect_class.py`, called after `_own_repos_licences` in `load_defect_rows` and in `_load_noul`
(so the v5 composite's scrambled parts are covered at the same seam). Admitted: an empty file, a
pinned sha, or a materialised subset in which every non-empty line is, byte for byte, a line of the
pinned source under `data/pool/<name>/` (hashed first) and the line count equals `kept` and
`totals.examples`. No caps. The constant is seeded only with the composed v1 sha the 174452f
pre-registration pins (`077aa842...`).

State: step 1 only. `test_the_pinned_own_corpora_are_the_ones_the_preregistration_pins` is RED on
purpose: the pre-registration carries no structured `base_corpus {path, sha256}` pin for
`own-swift-composed-v1` (it appears only in prose), so there is nothing to mirror for the base. It
goes green at step 3, after W4-1's amendment, when `OWN_CORPUS_PINS` is filled from it. Not merged
with W4-1; steps 3-4 (merge, real-data v2 loads, `test_arm_b_mine.py`) are open.
Red against 8dabde3: 7 failed (the 3 repro tests, the 3 materialised-refusal tests, the mirror);
the 3 characterisation tests (composed as pinned, materialised subset admitted, empty admitted)
pass on both by design.
Security impact: narrows admission only; new refusals are an own-repos corpus (plain, composed,
materialised or noul) whose bytes the pre-registration does not pin, and a materialised row that is
not a line of the pinned file. Risk: a false refusal of a registered corpus, which step 4 checks.
Gap recorded: `GAP-OWN-REPOS-DIFF-TIE-JUDGE-TOOLING-2026-10-13` (ListAgents unavailable to the
judge; collisions facet truncated, 3 worktrees unscanned).

### Round 3, steps 3-4 (done)

- Merged `wave4-l1-own-swift-clean-rows` (4484091; amendment 1a946c1). `gaps.jsonl` conflict
  resolved as a union (ours 1452 lines, theirs 1452, result 1455; every line of both parents
  present, all parse). `OWN_CORPUS_PINS` filled from the amended pre-registration: composed v2
  `d5e9341e...e26` and its structured base `own-swift-corpus-v2` `02cbfdac...886`. The v1 pin is
  gone because the amendment superseded v1 in the registration (its `superseded_pin` is not a pin).
  Mirror test green.
- Real data (corpora copied read-only from W4-1's worktree into this tree's gitignored `data/pool`,
  sha256 of both `examples.jsonl` checked against the amendment): composed v2 as pinned loads
  **14,962** rows (13,647 base + 1,315 composed); the materialised form (`.build-b/mat`, kept
  1,315, dropped 0, byte-identical to the pinned file so admitted by sha) loads **1,315**; a real
  100-row subset of it (a genuine subset, exercising the line-membership proof) loads **100**.
- Bug found by step 4 and fixed: `materialise` names its output after the corpus
  (`own-swift-composed-v2`), but the pinned directory is `own-swift-composed-v2-prefixed`; the
  subset proof keyed the pin by the materialised directory's name and would have refused every
  genuine subset. It now finds the pinned source by `source_examples_sha256`. The fixture's
  materialised directory is renamed to cover it (red before the fix: the subset-admitted test).

### Round 3, final attack on 84078b5 (fixed)

HIGH: `_load_noul` decided "own pool" from the noul manifest's own `pool.path` (a sidecar manifest
beside that file), so a byte-identical decoy copy of the pool with no sidecar, named by the noul
manifest, hid the corpus from the pin check and was licensed owner-granted. Fix at the seam:
`load_defect_rows` passes `own_pool_path` (the admitted own pool) through `_load_noul_corpus` and
`_load_composite` to `_load_noul`; when set, the noul manifest must name that very pool and the
corpus must be pinned (the old sidecar test remains as an extra, narrowing-only trigger).
`test_a_noul_corpus_naming_a_decoy_copy_of_the_own_pool_is_refused[pinned False/True]`: red on
84078b5 (2 failed, DID NOT RAISE), green on the tip.
Residuals added: the subset proof holds the subset's lines in memory (about 2-3x the file's size);
the source is hashed and then re-read for the line walk (sha-then-reread, same class as residual c).
Security impact: narrows admission only.

## Evidence (first commit, ab80131)

- Red on 174452f (new test file copied into a detached worktree of 174452f, since removed):
  `24 failed, 18 passed, 1 skipped`: every `test_an_unsafe_pool_path_is_refused[*]` (18),
  `test_an_empty_languages_download_manifest_is_refused`,
  `test_a_commitpackft_load_over_an_empty_languages_download_manifest_is_refused`,
  `test_the_loader_itself_refuses_a_pool_manifest_the_registration_does_not_pin`,
  `test_a_pool_file_that_is_not_the_one_its_manifest_records_is_refused`,
  `test_the_pinned_pool_manifest_is_the_one_the_preregistration_pins`, and the clash test
  (`..._pool_id_in_the_commitpackft_download_is_refused`, now by content: the old id-keyed check
  was blind to it). The existing helper pins use `raising=False` so they run on the old loader.
  `test_a_download_row_keyed_like_a_pool_id_but_with_other_content_is_not_a_clash` and
  `test_a_dotted_but_plain_path_is_still_admitted` pass on both: they guard that nothing widens
  or over-narrows.
- Green on tip: `test_defect_class_own_repos.py` 42 passed, 1 skipped (the real-v5 test skips in a
  worktree: its data is git-ignored).
- v5 parity, real data, run from this tree's code with `repo_root` = the main checkout:
  `49953 931c0f63b4b333e8aa575d7457f3b5be386da84019cd232e39df8a3f84935fa9` (matches the pin).
- `make pytest`: `3668 passed, 489 skipped in 587.16s`. `make lint`: `All checks passed!`.
- The real L3 own-swift pool passes the new licence gate (pin, content clash against the real
  commitpackft download); loading `own-swift-composed-v1-prefixed` then stops at a later,
  pre-existing check (`empty diff` on an own-swift `#clean` example in its base corpus,
  `_parse_one`), unrelated to this lane and not touched. Open for the corpus owner (L3).

## Security-impact statement

This change narrows admission and widens nothing. New refusals only: an unpinned pool manifest, a
pool file differing from its manifest, an unsafe path, a pool file whose bytes are in the
commitpackft download, an empty-`languages` download manifest. No access, licence, quota, crypto
or secret handling is touched. The one behavioural risk is a false refusal; the real own-swift pool
and the v5 corpus both pass.

## Open

- `GAP-OWN-REPOS-PINNED-HEAD-BLOB-CHECK-NOT-IMPLEMENTED-2026-10-13` (residual above).
- `GAP-DEVMAP-INDEX-LACKS-OWN-REPOS-LOADER-2026-10-13` (DevMap gen 3887 indexes main; blast radius
  by direct read, not graph-confirmed).
- Re-registering a pool means adding its manifest sha to `OWN_POOL_MANIFEST_SHA256S`; the test
  `test_the_pinned_pool_manifest_is_the_one_the_preregistration_pins` fails if code and
  pre-registration disagree.
- `tools/arm_b_mine.py` and `python/tests/test_arm_b_mine.py` were not changed (no need).

## First command for the next lane

`PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_defect_class_own_repos.py -o addopts= -q`
