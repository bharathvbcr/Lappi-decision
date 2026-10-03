# What the v5 rebuild must read, written before it runs (2026-10-03, ~22:15Z)

The human ratified one rebuild (AUDIT/finalize-2026-10-03/human-answers-2026-10-03-a7.md):
- the train rows that knocked out A7's eval rows leave the corpus before dedupe, via
  `--drop-before-dedupe`;
- the rebuild is bounded to one.

Fable (~22:05Z): pre-register the rebuild's numbers so the bound can be checked. Any deviation
from what follows is "new keys": stop and report, never iterate.

## The drop list [V]

- `/Users/bharath/qd-campaign/v5-predrops-2026-10-03/pre-dedupe-drops.txt`.
- sha256 `a8ab2a11d3a42c7f7d9e063d8ffb79d689657fd62c28e1f174a6b7070356a9e7`.
- 4,063 identity keys: decider.commands 571, procedural.decisions 3,491, boolq.reading 1.

How it was derived (scripts in build/v5-build/, copied to AUDIT with this record's commit):
1. `knockers.py` re-ran the first build's stages 1–2 on its exact argv. Its parity with the
   build log was exact: 561,681 rows in, 555,469 kept, train/val/held-out
   495,208 / 22,315 / 37,946. It named the 55 dedupe components that removed the 98 eval units.
2. `derive_drops.py` listed every train-side member of those components (never an eval
   unit). The pipeline's `drop_before_dedupe` refuses a listed row the split would not put in
   train, so a wrong list cannot shrink an eval set.

## The dry run [V, build/v5-build/dryrun_drops.py, dryrun-drops.json]

The same argv, with dedupe and split run twice in memory:
- **As built:** the parity above, and A7 failing on exactly the first build's 98 keys
  (val decider.commands 66, procedural.decisions 24, vitaminc.nli 1; held-out
  qa.answer_span 7).
- **With the list:** 557,618 rows into dedupe and 555,500 kept. Train / val / held-out is
  495,154 / 22,393 / 37,953. A7 fails on 13:
  - val decider.commands 11; the first five: decider.commands-val:081de6eb8122c77e,
    0a03c8b33b7e74d7, 2e658c88c63c39fd, 5f96ed83884499f2, 869e12012dcfd33f;
  - val procedural.decisions 1: procedural.decisions-val:state_perturbation:train:5715;
  - val vitaminc.nli 1: vitaminc.nli-val:5eeb7d3fc9e77c0008d204ac.

  Held-out qa.answer_span is restored: all 7 Bermuda rows come back.
- **The 13 are deduplicated within val.** Every train-side member of their components is
  gone, yet they are still removed, so the unit kept in their place is another val row of
  the same pool family. This is put to the human separately as "reading C".

## The rebuild must read

1. Stage 1b: 4,063 rows dropped under 4,063 keys, sha256 a8ab2a11..., by family as above.
2. Stage 2: rows in 557,618; kept 555,500; split train 495,154, val 22,393, held-out 37,953.
3. A7, with the refused-row fix (pool val rows `rewrite_typed_decision` refuses are counted,
   not missing):
   - google/boolq reports 1 refused val row;
   - A7 refuses on val decider.commands 11, procedural.decisions 1 and vitaminc.nli 1;
   - every missing key is one of the first build's 98 (build/v5-build/a7-missing-where.json);
   - every other family passes, held-out qa.answer_span included.
4. code.defect_class token share is re-measured and must be at least 55.8% (the first build
   read 56.0607%). The rebuild removes no code.defect_class row.
5. The scan of the rebuild's corpus (its corpus names the drop list's sha256) is CLEAN, with
   the two zero-checks unchanged from the scoped scan: val 1 exact + 1 subsequence; held-out
   0 + 1, the caveat the human accepted.

Any other number in 1–3, a missing key outside the 98, a share under 55.8%, or a scan that is
not CLEAN stops the lane and is reported to the human before anything else runs.
