# The v5 containment scope: Fable's ruling and the human's ratifications, 2026-10-03

## What the full scan found

The full v5 containment scan (strip version 2) ran at v5-build ca48960, 20:14–20:19Z:
- `build/v5-build/v5_scan.sh`, using `build/v5-build/v5_flags.sh`'s corpus flags with the bench's
  v4 pool (examples sha256 f3942c14…);
- qd-prep fb9337ef;
- its log is `build/v5-build/scan.log`.

It was **CLEAN**: remaining hits 0 on val and on held-out. But it excluded **104,905** train
identity keys out of 573,286 train rows.

`build/v5-build/exclusion_crosstab.py` maps each excluded key to the val or held-out family its
enforced pairs hit. The decision-pool families account for ~97.6k of the keys. **Every pool
exclusion hits a val row of its own family:**

| Family | Excluded / train keys | What the pairs hit | An example pair's shared / target 8-grams |
|---|---|---|---|
| procedural.decisions | 42,818 / 91,963 | val, the same family | 79 / 153 |
| openjev.policy | 20,709 / 20,881 | val, the same family | 199 / 387 |
| openjev.evidence | 14,850 / 16,930 | val, the same family | 141 / 255 |
| openjev.routing | 5,902 / 6,299 | val, the same family | 131 / 219 |
| openjev.nli | 4,999 / 4,999 | val, the same family | 16 / 26 |
| decider.commands | 3,879 / 3,879 | val, the same family | 12 / 19 |
| openjev.rubric | 2,820 / 4,101 | val, the same family | 144 / 231 |
| decider.routing | 842 / 10,854 | val, the same family | 25 / 41 |
| openjev.game | 727 / 971 | val, the same family | 29 / 55 |
| typed.workflow, vitaminc.nli, boolq.reading, pairwise.helpfulness, arc.science, synth.general | 51, 30, 11, 7, 2, 2 | val of the same family, except 4 cross-family pairs (below) | |

What the pairs share is task templates and rule prose; neither row's text is the other's.
Applying them would leave openjev.policy with 172 train rows and decider.commands with 0.

The other exclusions are in v4's families:
- code.defect_class: 6,165 of 70,800, hitting held-out and val defect rows;
- knowledge.multiple_choice (MMLU): 1,076 of 14,172;
- CLINC: 3 per intent family, of ~21,008;
- qa.answer_span (SQuAD): 12;
- CSQA: 0.

## Fable's ruling (~20:35Z, verbatim headline)

> Ruling: scope same-family decision-pool pairs out of 8-gram containment; every other exclusion
> stands.

The reasons:
- **The pre-registration has settled this instrument question twice for these rows.**
  - The dedupe ruling: MinHash pairs distinct problems in the structured Open-Jev families, so
    they are deduplicated by exact content.
  - The open GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02 asks "does each
    family's constant question make short rows hit each other?". For the pool, this scan
    answers yes.
- **Within-pool train/val hygiene already has its own pre-registered instruments, and all of
  them ran and passed on this pool:**
  - the group-keyed draw;
  - near_duplicate_disjoint at Jaccard 0.8;
  - exact_content_disjoint;
  - the bench's checks: 0 of 15,681 Open-Jev scenes straddle the split, and 0 of 214,638
    context-plus-options digests are on both train and val.
- **What stands:** code.defect_class 6,165, MMLU 1,076, CLINC 3, SQuAD 12, and the four
  cross-family pool hits (boolq.reading→vitaminc.nli and qa.answer_span; pairwise.helpfulness→code.defect_class).

How the scope is applied:
- It goes inside the attestation, never into a filtered list.
- `qd-prep containment` reads the scope from the request's corpus object
  (`decisions_pool_same_family_not_enforced`, written by `real_tokenizer_pipeline.corpus_identity`
  from the pool manifest's families).
- A same-family pair of a scoped family is still written to pairs.tsv, but it neither excludes
  nor counts as a remaining hit.
- The attestation's rule text names the scope, and `same_family_not_enforced` counts what it
  left unenforced, per family.
- `qd_train.exclusions.read_exclusions` refuses an attestation whose scope is not exactly the
  build's corpus scope. That covers an old binary's attestation, an unscoped attestation for a
  scoped corpus, and another family list.

## The human's ratifications (AskUserQuestion, ~20:38Z, the options chosen verbatim)

- **Pool scope:** "Ratify Fable's scoping (Recommended)". The question gave the counts above,
  including that openjev.policy would fall from 20,881 to 172 train rows and decider.commands to 0.
- **CLINC strip version 2's full-scan numbers:** "Accept as a caveat (Recommended)".
  - CLINC lost 3 of 21,008 train keys per intent family.
  - key_ii_blind is 599 keys in val and 558 in held-out.
  - The two zero-checks are **not zero**:
    - val: 1 exact match, also a contiguous subsequence of a train utterance:
      `clinc-intent:reminder::48151f2275060577`, "what is on my to do list";
    - held-out: 0 exact matches, 1 contiguous subsequence: `clinc-intent:credit_limit::24fb7b19446cf753`,
      "credit limit".
  - Both are too short for the 8-gram key to see. They are recorded as a stated limitation of
    CLINC's val and held-out numbers.
  - This is the human's ratification that GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02
    waits on ("the human ratifies by the commit that renames the DRAFT"), given ahead of that
    commit with the full-scan numbers.

## Not changed

n = 8, the 0.5 threshold, the strip and its version, the enforced targets, and the val and
held-out populations (rule 2). Every pair is still computed and written. Only same-family pairs
of the pool's families are not enforced, and they are counted in the attestation.
