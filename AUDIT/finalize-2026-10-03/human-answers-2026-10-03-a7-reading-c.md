# The human's answer on A7's within-val duplicates (2026-10-03, ~22:27Z)

## What the dry run found [V, AUDIT/finalize-2026-10-03/predrops/dryrun-drops.json]

With the pre-dedupe drop list, the rebuild restores 85 of the first build's 98 eval rows:
- all 7 held-out qa.answer_span (Bermuda) rows, so every v4 family is restored;
- 55 of decider.commands' 66;
- 23 of procedural.decisions' 24.

13 val rows stay removed: decider.commands 11, procedural.decisions 1, vitaminc.nli 1.
- Every train-side member of their dedupe components is gone before dedupe, so the unit kept in
  each one's place is another val row of the same pool family.
- These are near-duplicates within the pool's own val split, across groups, and dedupe keeps
  one of each.
- No training row is involved, and no drop list can restore them: their twin is a val row,
  and removing a val row would shrink val.

The DRAFT's data.sources says of the new families: "Per-family val results are report-only
targets."

## Fable's ruling (~22:05Z)

These are deduplication within val, not knock-outs, and A7 as written cannot tell the two
apart. Fable recommends **reading C**:
- a pool val row that dedupe removed in favour of another val row of the same family counts
  as "deduplicated within val", named and counted;
- it is accepted only with evidence from the build: the rebuild's own dedupe report, written
  by `--dedupe-report-out`. It is never a waiver list;
- a row removed in favour of a train row still refuses.

That makes A7 sharper, not looser. Nothing in the rescan or the rebuild depends on the answer.

## The question, as asked (AskUserQuestion, ~22:26Z)

> The rebuild is running: rescan ~5 min, then ~46 min build. A dry run predicts it restores 85
> of the 98 eval rows, including all 7 held-out SQuAD rows, so every v4 family will match
> exactly. The other 13 are validation rows in the new pool families (decider.commands 11,
> procedural.decisions 1, VitaminC 1). Each has an exact or near-duplicate twin that is
> another validation row of the same family, and dedupe keeps only one; no training row is
> involved. The pre-registration says these new families' per-family val results are
> report-only targets. A7 as written still refuses on these 13. How should A7 read them?

The options:
1. **"Accept reading C (Recommended)"**: Fable's pick, as above. Launch at about 01:00Z, with
   a total idle cost of about $45 by then.
2. **"Keep A7 as written"**: the rebuild's A7 refuses on the 13, the lane stops, and the lead
   reports. Nothing launches until another fix, such as a change to qd_data's keep rule (a
   fingerprint change, another rescan and rebuild, about 1.5 h more).

## The answer, verbatim

**"Accept reading C (Recommended)"**

A7 gains `--dedupe-report`. For a pool family's missing val key, A7 finds the dedupe component
that removed its unit in the rebuild's own report. If the unit kept there (the cluster's kept
unit, or the searched unit of an exact-content cluster) is a row of the same family in v5's
val manifest, the key counts as `deduplicated_within_val`, and is named and counted. Anything
else is missing and refuses. Without the flag, A7 is unchanged. v4's families are unchanged:
every one must equal v4's, held-out included.

The rebuild's own A7 step runs the code at 8e6a009, which predates this reading, and is
expected to refuse on exactly the 13 (AUDIT/finalize-2026-10-03/v5-rebuild-prediction-2026-10-03.md).
Reading C's A7 is then run standalone on the same build, and both outputs are recorded.
