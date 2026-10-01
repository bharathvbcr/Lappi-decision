# R1 — multi-hunk diffs in qd-mutate, corpus v3 (2026-09-30)

Decision B of the phase-4 plan. Branch `worktree-agent-ae0eb2dcf4ba68960`, off `e3477eb`, main
(`d34a936`, with R2 and perf-prelude) merged in at `5d5c300`. Addresses
`GAP-NEEDLE-TRAINING-DIFFS-ARE-ALL-SINGLE-HUNK`.

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
| `d74c817` → `85776c0` → `aa0cc69` → `0255958` → `1fb3146` | The needle-visibility rule, corrected against real rows. **Generate only at `1fb3146` or later**: `d74c817` admits two poisoned rows, `85776c0` and `aa0cc69` over-refuse (455 and 14 real needles), `0255958` admits three rows shown only by a deleted blank line. |
| `5d5c300` | Merge of main `d34a936`; `gaps.jsonl` resolved as the union (the only conflict). |

**The visibility rule** (`MultiHunkDiff::shows_edit(base, source, after)`): the edit is located
from `source`/`after` the way derivation B locates it, then judged by content.

* An edit that **wrote lines** shows when a written line is an added line, or when a change
  block touching it removes a substantive line the edit removed. Example: `nsqhandler.go`, where
  the body is gutted to the `return nil` it already ended in; Myers aligns that line as context,
  and the reader sees the body deleted around it.
* An edit that **only removed lines** shows when a change block touching its deletion point
  carries a substantive line. Example: `pearson.py`, where the commit replaced
  `corr = np.corrcoef(...)` with `if self.spatial: … else: corr = …` and the edit drops the
  `else`; the reader sees the old computation deleted and only the spatial branch added.
* Blank lines and lines of only brackets, separators and whitespace never count as evidence.
  Three reverted `else`s were otherwise "shown" by the commit's deleted blank line.

The rule judges what a reader can see, not authorship. A commit's own substantive change
touching a pure deletion's point counts as the edit showing. That case is the 44 rows where the
span sits beside an added-only run of commit code (41 `drop_else`), and it is deliberate: on a
new file it is the only way a defect can appear. So `needle_not_in_diff` counts edits nothing
shows, not every edit the commit's changes surround. A written line that restores the pre-image
is refused, e.g. python `stub.panic` writing back the `pass` the commit had implemented away, or
a `<=` widened back. Samples of these refusals were all reverts.

## What was measured

There is no ledger row: a corpus build writes a sha-pinned manifest, not a run row, and v2 had
none either. Every number below comes from the named manifest or from a throwaway analysis.
That analysis loaded the corpus through `qd_data.defect_class.load_defect_rows` on the **merged**
code (`5d5c300`), with worktree `python/`, `repo_root` = main checkout, `DataConfig()`, and
`QD_PREP_BIN` set.

* **v2 reproduction.**
  * Command: `qd-mutate generate --pool data/pool/commitpackft-pool-v2.jsonl --seed 0 --limit 200000 --single-hunk`.
  * Result: examples sha256 `9ad8f17fad5e71f0190eae4b148959c2740d82832ba895c68fd2cf85f21b9824`, equal to `commitpackft-corpus-v2/manifest.json` `examples_sha256`.
  * It matches with the pre-change binary (no flag) and with the binary at every commit above, including the merge.
  * The v2 command was never recorded. It is inferred from the manifest (seed 0, default `--clean-permille 200`, non-binding limit) and fixed by the digest match.
  * Formatter availability matched v2's manifest: rustfmt and gofmt found, black unusable, prettier not found. It decides which cosmetic operators exist, so another machine may not reproduce.
* **v3** at `data/pool/commitpackft-corpus-v3/` (main checkout; git-ignored). Same command without `--single-hunk`, at `1fb3146`; the merged binary reproduces both files byte-for-byte.
  * examples sha256 `610bb1f0f1d86dc23f419001d8670e01418485644bea93e2b33631e5650c62e0`
  * manifest.json sha256 `5164251c7dec84c8dcffe0210cc06179b97c6605c1f2cd93c80b016e58736d12`
  * pool: `commitpackft-pool-v2.jsonl`, sha256 `2cb31231…5d2`, 61,193 records. v2 and the pool are unchanged.
  * **Rows: 49,953** — clean 8,449, cosmetic 7,246, logic 11,293, stub 22,965 (v2: 8,449, 7,344, 11,310, 23,074).
  * 224 fewer mutated rows than v2:
    * `needle_not_in_diff` 128 (python `stub.panic` 96);
    * `needle_split_across_hunks` 96 (all `cosmetic.rename_local`: python 66, go 17, ts 7, rust 6);
    * `diff_refused` 0; `mutated_without_prior` 0.
  * **Hunks per diff:** 1: 41,328 · 2: 7,380 · 3: 1,128 · 4: 103 · 5: 14, so 17.3% are multi-hunk. Multi-hunk share by class: clean 21.7%, stub 16.7%, cosmetic 16.6%, logic 15.6%.
  * **Diff lines** p50/p90/p99/max: 33 / 72 / 102 / 132 (v2: 10 / 31 / 78 / 162). p50 by class: clean 19, cosmetic 38, logic 40, stub 33 (v2: clean 26, cosmetic 11, logic 9 with p90 also 9, stub 11).
  * **Qwen3.5-2B-Base tokens** (p50/p90/p99/max):
    * diff alone: 278 / 644 / 813 / 959;
    * rendered training sequence (max over the row's two slots, `rewrite_defect_class` → `training_texts`): 424 / 789 / 959 / 1,101.
    * **0 rows over 1,625.** v2's rendered max is exactly 1,625, consistent with it being the widest bucket (not traced to the bucketing code).
  * **Load:**
    * 0 span-rebase refusals and 0 rewrite refusals.
    * With `noul_dir=data/pool/defect-noul-v1`: 52,455 rows (2,502 noul: prose, scrambled and unseen-language, 834 each), still 0 span refusals.
    * `refuse_leaky_diff_corpus` in diff mode returns `{}`; 0 rows have CR in `before` or `diff`.
    * Python re-check over 41,504 mutated rows: no span crosses a header, and none is shown only by trivial lines.

## What is open

* `GAP-DEVMAP-UNIFIED-CALLERS-UNRESOLVED-R1-2026-09-30`, `GAP-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS-R1-2026-09-30`.
* **Shape asymmetry: reversed and smaller, not gone.** v2's logic diffs were 9 lines at p50 and
  p90. On v3, clean diffs are shorter (p50 19 vs 33–40) and more often multi-hunk (21.7% vs
  15.6–16.7%), and a mutated diff is a commit plus one more edit. Nobody has measured whether
  length or hunk count predicts the class on v3.
* **Hunk counts are far below the gate's 14–158.** v3 teaches choosing among ≤ 5 hunks.
* `tool_version` stays `0.1.0`: it is in every row, and bumping it breaks the v2 proof. The shape
  is named by `manifest.diff.renderer` instead.
* `python/qd_train/mutate_adapter.py:182-186` says `diffspan::unified` emits every header; v3
  headers come from `unified_multi`, in the same `@@ -a,b +c,d @@` form. Left unedited because of
  the training closure digest.
* Files touched outside `diffspan/generate/manifest/main.rs`:
  * `src/parse.rs` (three `Refusal` variants);
  * `tests/{adversarial,coverage}.rs` (the new `Options` field);
  * new `tests/multi_hunk_fixture.rs`;
  * `python/tests/test_multi_hunk_rebase.py` and its fixture data.

## First command for the next lane

Build the v3 shard set from the repo root, at a clean commit that contains `5d5c300`:

    QD_PREP_BIN=<checkout>/target/release/qd-prep /Users/bharath/.venvs/ml/bin/python tools/real_tokenizer_pipeline.py \
      --out <out dir> --rev <pinned sha> --val-shards --replay-shards --vocab full \
      --defect-class data/pool/commitpackft-corpus-v3 --defect-download data/pool/commitpackft \
      --defect-noul data/pool/defect-noul-v1 \
      --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
      --general-max-rows 200000 --memo-limit 0 --no-repo-history --ledger <ledger path>
