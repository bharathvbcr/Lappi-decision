# R1 — multi-hunk diffs in qd-mutate, corpus v3 (2026-09-30)

Decision B of the phase-4 plan. Branch `worktree-agent-ae0eb2dcf4ba68960`, off `e3477eb`.
Addresses `GAP-NEEDLE-TRAINING-DIFFS-ARE-ALL-SINGLE-HUNK` (opened on main in `99dfc93`).

## Why

Every `commitpackft-corpus-v2` diff has exactly one `@@` header (50,177 / 50,177, re-measured
below), so a needle gate asking which of 14–158 hunks holds the defect had nothing to train on.
Cause: `diffspan::unified` trims a common prefix and suffix and prints everything between as one
hunk; and the mutated branch diffed post-image → mutated text (the injected edit alone) while
the clean branch diffed pre-image → post-image (the whole commit).

## What changed (commits)

| commit | what |
| --- | --- |
| `92550e5` | `diffspan::unified_multi` (in-crate linear-space Myers; split at > 2·context; refuses over 50,000 lines/side or edit distance > 8,192 — exact, clock-free). Default `DiffShape::MultiHunk`: `before` = normalized pre-image, `diff = unified_multi(prior, mutated_after)`; clean `unified_multi(prior, source)`. Span label unchanged (both derivations still post-image → after). Refusals `needle_not_in_diff`, `needle_split_across_hunks`, `diff_refused`. `manifest.diff`: renderer, hunk histogram overall and by class, without-prior counts. `--single-hunk` = v2 byte-for-byte. Fixture `python/tests/data/qd_mutate_multi_hunk/`, `tests/multi_hunk_fixture.rs`, `python/tests/test_multi_hunk_rebase.py`. |
| `d74c817`, `85776c0`, `aa0cc69`, `0255958` | The needle-visibility rule, corrected against real rows (below). Final: `MultiHunkDiff::shows_edit(base, source, after)`. |

**The visibility rule, and why it moved.** `92550e5` asked positionally "is there a change block
at the span". On the pool that (a) refused 153 `logic.drop_else` rows whose defect sits at the
edge of code the commit added (e.g. `brainhack/covariance/pearson.py`: the commit replaced
`corr = np.corrcoef(...)` with `if self.spatial: … else: corr = np.corrcoef(...)`, the edit drops
the `else` — the reader sees the old computation deleted and only the spatial branch added) and
(b) could not tell a commit's change beside the span from the edit. `d74c817` (an end-of-file
clamp) admitted exactly two rows and both were wrong — reverts of a commit-added `else`, sitting
beside the commit's own end-of-file edit. `85776c0` (removed text must show) over-refused 455.
Final rule, located from `source`/`after` the way derivation B locates the edit:

* an edit that **wrote lines** shows when a written line is an added line, or when a change
  block touching it removes a line the edit removed (`nsqhandler.go`: a body gutted to the
  `return nil` it already ended in — Myers aligns that line as context, and the reader sees the
  body deleted around it);
* an edit that **only removed lines** shows when a change block touches its deletion point.

A written line restoring the pre-image (python `stub.panic` writing the `pass` the commit had
implemented away; a `<=` widened back) shows nothing and is refused — sampled, all reverts.

## What was measured

No ledger row: a corpus build writes a sha-pinned manifest, not a run row (v2 had none either).
Every number is from the named manifest or from a throwaway analysis that loaded the corpus
through `qd_data.defect_class.load_defect_rows` (worktree `python/`, `repo_root` = main checkout,
`DataConfig()`).

* **v2 reproduction.** `qd-mutate generate --pool data/pool/commitpackft-pool-v2.jsonl --seed 0
  --limit 200000 --single-hunk` → examples sha256
  `9ad8f17fad5e71f0190eae4b148959c2740d82832ba895c68fd2cf85f21b9824`, equal to
  `commitpackft-corpus-v2/manifest.json` `examples_sha256` — with the pre-change binary (no flag)
  and with the binary at every commit above. The v2 command was never recorded; it is inferred
  from the manifest (seed 0, default `--clean-permille 200`, non-binding limit) and fixed by the
  digest match. Formatter availability matched v2's manifest (rustfmt, gofmt found; black
  unusable; prettier not found); it decides which cosmetic operators exist, so another machine
  may not reproduce.
* **v3** at `data/pool/commitpackft-corpus-v3/` (main checkout; git-ignored), binary at
  `0255958`, same command without `--single-hunk`: examples sha256
  `bb2288c30456afb45addef62d2dd0e0fb9852fafb30ff288c822009d4f60deb3`, manifest.json sha256
  `021602f8df9ac34a8585d5edb4b11f6d8a08537ed85d29a2c06a726189769f35`; pool
  `commitpackft-pool-v2.jsonl` sha256 `2cb31231…5d2`, 61,193 records (v2 and the pool unchanged).
  * rows 49,956: clean 8,449 · cosmetic 7,246 · logic 11,296 · stub 22,965 (v2: 8,449 · 7,344 ·
    11,310 · 23,074). 221 fewer mutated: `needle_not_in_diff` 125 (python `stub.panic` 96),
    `needle_split_across_hunks` 96 (all `cosmetic.rename_local`; python 66, go 17, ts 7, rust 6).
    `diff_refused` 0; `mutated_without_prior` 0 (every pool record carries a pre-image).
  * hunks per diff: 1: 41,329 · 2: 7,382 · 3: 1,128 · 4: 103 · 5: 14 (17.3% multi-hunk). By
    class, multi-hunk share: clean 21.7%, stub 16.7%, cosmetic 16.6%, logic 15.6%.
  * diff lines p50/p90/p99/max: 33 / 72 / 102 / 132 (v2: 10 / 31 / 78 / 162). By class p50:
    clean 19, cosmetic 38, logic 40, stub 33 (v2: clean 26, cosmetic 11, logic 9 — logic p90 was
    also 9 — stub 11).
  * Qwen3.5-2B-Base tokens, diff alone: 278 / 644 / 813 / 959; rendered training sequence (max
    over the row's two slots, `rewrite_defect_class` → `training_texts`): 424 / 789 / 959 /
    1,101. **0 rows over 1,625** (v2's rendered max is exactly 1,625 — consistent with it being
    the widest bucket; not traced to the bucketing code).
  * load: 0 span-rebase refusals, 0 rewrite refusals. `refuse_leaky_diff_corpus` in diff mode
    returns `{}`; 0 rows with CR in `before` or `diff`. A Python-side positional re-check over
    all 41,507 mutated rows: no span crosses a header; 44 spans have no added line and no
    adjacent removal but sit directly beside an added-only run (41 `drop_else`) — the
    edge-of-added-code case above; 0 have no adjacent change.

## What is open

* `GAP-DEVMAP-UNIFIED-CALLERS-UNRESOLVED-R1-2026-09-30`, `GAP-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS-R1-2026-09-30`.
* **Merge:** main moved to `99dfc93` (`10ce63f`, `99dfc93`: `gaps.jsonl` + a ledger file only).
  This branch conflicts with main in `gaps.jsonl` alone — both appended; resolve as the union.
  Files touched outside `diffspan/generate/manifest/main.rs`: `src/parse.rs` (three `Refusal`
  variants), `tests/{adversarial,coverage}.rs` (the new `Options` field), new
  `tests/multi_hunk_fixture.rs`, `python/tests/test_multi_hunk_rebase.py` + fixture data.
* **Shape asymmetry, reversed and smaller, not gone.** v2's logic diffs were 9 lines at p50 and
  p90; v3 clean diffs are shorter (p50 19 vs 33–40) and more often multi-hunk (21.7% vs
  15.6–16.7%), and a mutated diff is a commit plus one more edit. Whether length or hunk count
  predicts the class on v3 is unmeasured.
* **Hunk counts are far below the gate's 14–158.** v3 teaches choosing among ≤ 5 hunks.
* `tool_version` stays `0.1.0` (it is in every row; bumping it breaks the v2 proof); the shape is
  named by `manifest.diff.renderer`.
* `python/qd_train/mutate_adapter.py:182-186` says `diffspan::unified` emits every header; v3
  headers come from `unified_multi` in the same `@@ -a,b +c,d @@` form. Left unedited (training
  closure digest).

## First command for the next lane

Rebuild the shard set over v3, from the repo root at a clean commit that includes `0255958`:

    /Users/bharath/.venvs/ml/bin/python tools/real_tokenizer_pipeline.py \
      --out <out dir> --max-pairs 400 --rev HEAD --val-shards \
      --defect-class data/pool/commitpackft-corpus-v3 --defect-download data/pool/commitpackft \
      --memo-limit 0 \
      --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
      --replay-shards --ledger <ledger path>
