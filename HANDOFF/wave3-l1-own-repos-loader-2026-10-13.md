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
