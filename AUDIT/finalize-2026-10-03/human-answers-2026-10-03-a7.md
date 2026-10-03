# The human's answer on A7's refusal of the v5 build (2026-10-03, ~21:52Z)

## What happened

The one v5 build (v5_build.sh at v5-build cd2967e, the scoped scan's 7,260 exclusions) finished at
21:42:04Z with exit 0 [V, build/v5-build/build.log; ledger row 849d3ae5-f78c-4127-a32a-a12c7af1f50a
in v5-build-wt's ledger/mac-v5-shards-2026-10-03.jsonl]:
- 558,844 train sequences, 386,259,227 tokens;
- every manifest row tokenized (487,461 of 487,461);
- padding 5.79%;
- read-back invariants all 0.

code.defect_class's token share is 56.0607%, against the bound of at least 55.8% (two-thirds of
v4's 83.7%) [V, build/v5-build/defect_token_share.py, 21:50Z].

A7 (tools/v5_a7_check.py --decisions-pool) then exited 1 on a traceback, not a verdict.
`pool_expectation` rebuilds each pool val row with `rewrite_typed_decision` and does not catch
`RowRefused`. One google/boolq val row carries U+200E in its context; `build_mixture` refuses and
counts such a row (`invisible_format_characters`), but A7 raised on it.

## What the lead found [V unless marked]

- **A7 with the refused row dropped from the expectation** [build/v5-build/a7_pool_refusals.py]
  still REFUSES: 41 of 45 families pass. For every pool family, the pool manifest's
  `val_by_family` equals built + refused. Four families fail:
  - val decider.commands: 66 of 199 missing;
  - val procedural.decisions: 24 of 701 missing;
  - val vitaminc.nli: 1 of 701 missing;
  - held-out qa.answer_span: 7 of v4's 6,563 missing, all `squad-title:Bermuda::...`.
- **All 98 missing keys are in no v5 manifest** [build/v5-build/a7_missing_where.py]. None moved
  into train, and none is in exclusions.txt, so rule 3 holds.
- **They are dedupe knock-outs, not refusals.** The v4 replay build refused the same SQuAD rows
  (356 `invisible_format_characters`, 58 `contradictory_prompt`).
  - `dedupe` keeps `members[0]` of each near-duplicate component, the lexically smallest
    `content_unit_key`. That key is `identity_key|digest` (python/qd_data/dedupe.py:135), so the
    rule is lexical and blind to the split.
  - "boolq..." sorts before "squad-title:...", and a pool `-train:` key sorts before `-val:`.
  - The pool holds google/boolq train rows whose passage is the Wikipedia Bermuda introduction.
    That is the inferred knocker of the 7 held-out SQuAD rows; build/v5-build/knockers.py
    names the exact clusters.
- **The DRAFT's remedy cannot restore a knock-out.** build_order[3] and the data.sources A7 rule
  say "an A7 missing key is a dedupe knock-out: exclude that row and rebuild".
  `--exclude-identity-keys` runs in `exclusions_then_contrast`, after `dedupe` and `split`
  (tools/real_tokenizer_pipeline.py:2333, 2659-2661). It never touches dedupe, so excluding the
  kept train twin and rebuilding drops the same eval rows again.

## Fable's ruling (~21:47Z)

This is a pre-registration defect, not a build defect. Under rule 2, Fable recommends and the
human ratifies. Recommendation **B**:
- rebuild once, with a new, small, named list of the knocking rows removed between
  `build_mixture` and `dedupe`;
- the change goes in tools/, not python/qd_data, so the shard fingerprint and the 583190d9
  qd-prep binary stay valid;
- the list's sha goes on the ledger row as exclusions.txt's does;
- order: new v5-build commit, rescan at it, rebuild, A7;
- bounded to one rebuild: if A7 refuses on new keys (second-order knock-outs), stop and report.

A7's `pool_expectation` is fixed to mirror `build_mixture`: it catches `RowRefused`, counts it,
and reconciles stated == built + refused, with a test that fails before the fix. It is **not**
widened to absorb dedupe knock-outs, and after the fix it must still refuse the first build on
the four families. Option A, accepting the first build and waiving A7 at freeze, was not
recommended: it shrinks a held-out set, which rule 2 names.

## The question, as asked (AskUserQuestion, ~21:51Z)

> The v5 build passed (exit 0, every row tokenized, code.defect_class share 56.06% against the
> 55.8% bound), but the pre-registered A7 check refuses it. Dedupe removed 98 eval rows that were
> near-duplicates of training rows: 7 v4 held-out SQuAD 'Bermuda' rows (a new BoolQ train row
> carries the same Wikipedia paragraph), plus 66 decider.commands, 24 procedural.decisions and 1
> VitaminC validation rows. None of them reached training. The pre-registration's fix ('exclude
> that row and rebuild') can't work, because exclusions apply after dedupe. Both H100s idle at
> $8.38/h while you decide. Which path?

The options:
1. **"Rebuild once (Recommended)"**: Fable's option B, about 1.5–2 h more idle (~$13–17). It
   includes A7 counting a pool row the build refuses as refused, not as missing.
2. **"Accept this build"**: A7 recorded REFUSED with the 98 keys and waived at freeze; launch in
   about an hour.

## The answer, verbatim

**"Rebuild once (Recommended)"**

The human ratified option B. This covers the pre-dedupe removal of the named knocking rows, the
one-rebuild bound, and A7's reading of pool rows the build refuses (stated == built + refused).
Gates and thresholds are unchanged: A7's per-family sets are what they were.
