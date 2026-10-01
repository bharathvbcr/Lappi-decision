# R1 — multi-hunk diffs in qd-mutate, corpus v3 (2026-09-30)

Decision B of the phase-4 plan. Branch `worktree-agent-ae0eb2dcf4ba68960`, off `e3477eb`.

## Why

Every `commitpackft-corpus-v2` diff has exactly one `@@` header (50,177 / 50,177, re-measured
below), so a needle gate asking which of 14–158 hunks holds the defect had nothing to train on.
Cause: `diffspan::unified` trims a common prefix and suffix and prints everything between as one
hunk; and the mutated branch diffed post-image → mutated text (the injected edit alone) while
the clean branch diffed pre-image → post-image (the whole commit).

## What changed (commits)

| commit | what |
| --- | --- |
| `92550e5` | `diffspan::unified_multi` (in-crate linear-space Myers; split at > 2·context; refuses over 50,000 lines/side or edit distance > 8,192 — exact, clock-free). Default `DiffShape::MultiHunk`: `before` = normalized pre-image, `diff = unified_multi(prior, mutated_after)`; clean `unified_multi(prior, source)`. Span label unchanged (both derivations still post-image → after). New refusals `needle_not_in_diff`, `needle_split_across_hunks`, `diff_refused`. `manifest.diff`: renderer, hunk histogram overall and by class, without-prior counts. `--single-hunk` = v2 byte-for-byte. Fixture `python/tests/data/qd_mutate_multi_hunk/`, `tests/multi_hunk_fixture.rs`, `python/tests/test_multi_hunk_rebase.py`. |

## What was measured

No ledger row: a corpus build writes a sha-pinned manifest, not a run row (v2 had none either).
Every number below is from the named manifest or from a throwaway analysis that loaded the
corpus through `qd_data.defect_class.load_defect_rows` (worktree `python/`, `repo_root` = main
checkout, `DataConfig()`).

* **v2 reproduction.** `qd-mutate generate --pool data/pool/commitpackft-pool-v2.jsonl --seed 0
  --limit 200000 --single-hunk` with the `92550e5` release binary → examples sha256
  `9ad8f17fad5e71f0190eae4b148959c2740d82832ba895c68fd2cf85f21b9824`, equal to
  `commitpackft-corpus-v2/manifest.json` `examples_sha256`. The pre-change binary gives the same
  digest with no flag. The v2 command was never recorded; it is inferred from the manifest
  (seed 0, default `--clean-permille 200`, non-binding limit) and fixed by the digest match.
  Formatter availability matched v2's manifest (rustfmt, gofmt found; black unusable; prettier
  not found) — it decides which cosmetic operators exist, so another machine may not reproduce.
* **v3** at `data/pool/commitpackft-corpus-v3/` (main checkout; git-ignored): examples sha256
  `d111379a53e81402c23568b18c9bc3543cd540bd76e2f5a5bffa2dd36d38bbc6`, manifest.json sha256
  `c27395bfc815b984684c4110c53ac358da2410b84265f0274ea86914e476d291`, pool
  `commitpackft-pool-v2.jsonl` sha256 `2cb31231…5d2`, 61,193 records. Same command without
  `--single-hunk`; regenerating it twice gave the same digest.
  * rows 49,790: clean 8,449 · cosmetic 7,245 · logic 11,139 · stub 22,957 (v2: 8,449 · 7,344 ·
    11,310 · 23,074). The 387 fewer mutated rows are the two new refusals:
    `needle_not_in_diff` 291 (python `logic.drop_else` 162, `stub.panic` 99 — the commit
    added the code the mutation removes), `needle_split_across_hunks` 96 (all
    `cosmetic.rename_local`: python 66, go 17, typescript 7, rust 6 — inferred, not inspected
    row by row: renames whose touched lines have more than six unchanged lines between them).
  * hunks per diff: 1: 41,217 · 2: 7,342 · 3: 1,115 · 4: 103 · 5: 13 (17.2% multi-hunk). By
    class, multi-hunk share: clean 21.7%, stub 16.7%, cosmetic 16.6%, logic 15.3%.
  * diff lines p50/p90/p99/max: 33 / 72 / 102 / 132 (v2: 10 / 31 / 78 / 162).
  * Qwen3.5-2B-Base tokens, diff alone: 278 / 644 / 813 / 959; rendered training sequence (max
    over the row's two slots, `rewrite_defect_class` → `training_texts`): 424 / 789 / 959 /
    1,101. **0 rows over 1,625** (v2 rendered max is exactly 1,625 — the widest bucket).
  * load: 0 span-rebase refusals, 0 rewrite refusals; a Python-side re-check found 0 of 41,341
    mutated rows whose diff-space span shows no change or crosses a header.

## What is open

* `GAP-DEVMAP-UNIFIED-CALLERS-UNRESOLVED-R1-2026-09-30` — DevMap resolved no callers of `unified`.
* `GAP-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS-R1-2026-09-30` — no collision scan ran.
* **Shape asymmetry, smaller but not gone.** Clean diffs are multi-hunk more often (21.7%) than
  mutated ones (15–17%), and a mutated diff is a commit plus one more edit. Nobody has measured
  whether hunk count or diff length predicts the class on v3.
* **Hunk counts are far below the gate's 14–158.** v3 teaches choosing among ≤ 5 hunks; the
  real commits in this pool are small files. Whether that transfers to the needle gate is the
  experiment, not a property of this corpus.
* `tool_version` stays `0.1.0` (it is in every row, and bumping it would break the v2 proof);
  the shape is named by `manifest.diff.renderer`.
* `python/qd_train/mutate_adapter.py:182-186` says `diffspan::unified` emits every header; v3
  headers come from `unified_multi` in the same `@@ -a,b +c,d @@` form. Left unedited (training
  closure digest).

## First command for the next lane

Rebuild the shard set over v3, from the repo root at a clean commit that includes `92550e5`:

    /Users/bharath/.venvs/ml/bin/python tools/real_tokenizer_pipeline.py \
      --out <out dir> --max-pairs 400 --rev HEAD --val-shards \
      --defect-class data/pool/commitpackft-corpus-v3 --defect-download data/pool/commitpackft \
      --memo-limit 0 \
      --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
      --replay-shards --ledger <ledger path>
