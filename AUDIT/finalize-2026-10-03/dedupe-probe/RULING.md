# The v5 decision pool hit the dedupe pair bound: evidence and Fable's ruling, 2026-10-03

## What happened

The bench's one Mac build of the v5 decision pool ran from 17:15 to 17:37Z, with no panic. The
pool is `pool-v5-decisions/examples.jsonl`, sha256 `e3a03f38…`, with 211,531 rows.

The pipeline's own stages ran over it in `pool_check.py`, and its split status came back
**not_run**:
- `dedupe`'s candidate-pair search hit its bound of 5,000,000;
- so did `near_duplicate_disjoint`'s.

The partial dedupe found 1,017,487 cross-repo pairs before the bound and dropped 19,318 rows,
41.6% of them from openjev.policy. Source: the bench's
`~/qd-campaign/v5-decisions-data-2026-10-03/patches/POOL-RESULT-v3.md`.

## What the near-duplicates are (the lead's probe, this directory)

`probe.py` takes a seeded sample of 400 rows per family from the built pool. It computes exact
Jaccard for every pair, over the pipeline's own shingles: `qd_data.minhash.shingle`, k=5, on the
pool row's `dedupe_text`, which is the context plus the options. It ran on that text as built
(`probe-as-built.out`), then with JSON punctuation split into tokens (`probe-spaced.out`).

**[V] means measured on the 400-row samples. [I] means extrapolated to the whole family**, by
the sample's pair rate × n(n−1)/2.

| Family (rows) | Sample pairs ≥0.8 [V] | With different gold [V] | Same template [V] | Pairs ≥0.8 [I] | Pairs 0.5–0.8 [I] |
|---|---|---|---|---|---|
| openjev.policy (21,744) | 349 | 110 | 91 | ~1.03M | ~13.6M |
| openjev.evidence (17,629) | 598 | 292 | 105 | ~1.16M | ~7.25M |
| openjev.routing (6,647) | 275 | 221 | 275 | ~76k | ~4.63M |
| openjev.rubric (4,365) | 1,184 | 908 | 1,184 | ~141k | ~1.51M |
| decider.commands (5,335) | 47 | 0 | 0 | ~8k | ~5.81M |
| procedural.decisions (96,825) | 0 | 0 | 0 | 0 | ~8.16M |

**The structured Open-Jev families.**
- **The ≥0.8 pairs are distinct decision problems, not copies.** 31–77% of them have different
  gold answers, and even the same-gold pairs differ in their facts: a median of 113–401
  differing context characters.
- **The cause.** About 2 KB of rule and policy prose is shared, and each case's facts are packed
  into JSON with no whitespace. Whitespace 5-shingles see the prose and barely see the facts.
- **JSON-aware tokenizing does not fix it.** With JSON punctuation spaced out:
  - procedural.decisions goes from 0 to ~7M pairs at ≥0.8 [I];
  - the Open-Jev pairs that remain are minimal pairs, with a median of 7–101 differing
    characters and often a different gold. That is contrast data.
- **"Template" is not a sibling index for every family.** Here it means `group_key` with its
  last `/` component removed.
  - In routing and rubric, every ≥0.8 pair shares one template.
  - In policy and evidence, 74–82% of the ≥0.8 pairs cross templates, because the policy prose
    is shared across scenarios.

## Fable's ruling (~18:00Z)

1. **All three of the bench's options are refused, because each deletes real decision data.**
   - (1) raising the pair bound;
   - (2) collapsing near-duplicate groups in qd-prep before selection;
   - (3) capping the templated Open-Jev strata.

   **The 5,000,000 bound stays.**
2. **The direction:**
   - For the structured decision families, use exact-content dedupe, keyed on the digest of
     context and options **alone**: not `identity_key`, which includes `repo_key`.
   - The MinHash near-duplicate check is scoped out of those families. For them it is reported
     as `not_run` by this ruling, never as passed.
3. **The leak definition, for structured decision rows:**
   - a leak is the **same content** (context plus options, by digest) across splits;
   - different facts make a different problem, whatever prose they share;
   - cross-template pairs are therefore not leaks.
4. **A second pool build is granted, only if this measurement passes:** the pipeline's real
   `dedupe()` and `split()` over the pool, minus the four families, must not truncate.
   - The script is `residual.py`, run under `mac_heavy`.
   - If it truncates, the fix is not sufficient, and the number goes back to Fable instead of
     into a build.
5. **Before any patch:**
   - read what Open-Jev's `group_key` encodes, and how qd-prep picks val (per group or per row);
   - check whether the 123 exact prompt collisions that `prompt_consistency` saw (211,471 rows,
     211,348 distinct prompts) span train and val;
   - record the scoping as a pre-registration amendment, before any v5 result exists. The
     DRAFT names "MinHash dedupe (Jaccard >= 0.8, before the split)" as a pipeline step.
6. **File split, agreed by message before anyone edits:**
   - the bench owns `crates/qd-prep/src/decisions.rs` and `python/qd_data/{decisions,sources}.py`;
   - the lead owns the pipeline core: `python/qd_data/{dedupe,split,minhash}.py` and the
     pipeline tools.

## Evidence files

| File | sha256 |
|---|---|
| `probe.py` | `fd0b6e919bbb068f6c53f07a098fdab6a5ae57f99b22245e85be687d7be59fd7` |
| `probe-as-built.out` | `e58a3fb33421f4246002b1b0df2f1396f9ed0b6a9b3561e1b8f68b0d3004d46c` |
| `probe-spaced.out` | `4524ab2e89ccd72618acdd6db558996b27436481bd50e284b1c4b1da565c6230` |
| `residual.py` | `868c6fde410f48b4fdbcb00ece116209fe39f079311feefc7a2295a6354c9fed` |

`residual.py`'s result is appended below once it runs. It was queued behind the v3 verification
on the Mac's one heavy-job lock.
