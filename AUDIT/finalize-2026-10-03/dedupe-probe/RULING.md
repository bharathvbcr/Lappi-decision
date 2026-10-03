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

## The residual measurement: it truncated (~18:09Z)

`residual.py` ran the pool minus the four families, under `mac_heavy`, on the pure-Python
reference path: 928 s, peak RSS 7.27 GB. Output: `residual.json`.
- **Dedupe hit the bound:** `n_candidate_pairs` 5,000,001, `not_run`.
- **The split's re-derivation over the 155,689 kept rows completed.** It found 1 crossing pair:
  procedural `evidence_sufficiency` train:11275 vs train:2982, at J=0.808, val/train, same gold
  S1.
- **The content check over the whole pool:**
  - 211,531 rows, 211,408 distinct contents;
  - 123 colliding groups (openjev.policy 122, arc.science 1);
  - 0 span train and val;
  - 0 have two golds.

Fable's condition for a second build was not met, so the number came back instead.

**Where the candidates come from.** `candidates_by_family.py` / `.out`: per family, a reservoir
sample of ≤400 rows, exact Jaccard, and the pipeline's own LSH candidate probability. The banding
is b=16, r=8, so P(0.5)=0.061, P(0.6)=0.237, P(0.7)=0.613 and P(0.8)=0.947.

| Family | Expected candidates [I] | Sampled pairs ≥0.8 (different gold) [V] |
|---|---|---|
| procedural.decisions | ~4.49M | 1 (0) |
| decider.commands | ~2.27M | 14 (2) |
| openjev.nli | ~36k | 0 |
| openjev.game | ~125k | 0 |
| all other searched families | under 3k each | 0–1 |

`pairs_look.py` / `.out` shows what those ≥0.8 pairs are. They are **true near-duplicates**:
- decider.commands: the same command plus or minus a flag (`--no-pager`, `--volumes`), with the
  same gold;
- procedural: the crossing pair is one template with relabelled evidence items, same gold.

The overflow is band work in J 0.5–0.8, not a pathological corpus.

## Fable's second ruling (~18:20Z)

- **(D), a per-family search, is refused.** Procedural alone is within sampling noise of 5M,
  and cross-family pairs would go unsearched.
- **(E), filtering candidates on signature agreement before the bound, is deferred.** It is the
  right structural fix, but it changes what every dedupe run measures, and touches the Rust
  LSH, the Python reference, the parity tests and the recall figures. It is recorded as
  GAP-DEDUPE-LSH-BAND-CANDIDATES-NOT-DUPLICATES-2026-10-03.
- **(A) is granted: a measured, recorded bound for builds that read the pool.** The default is
  not moved. The earlier "the bound stays" protected real problems from deletion, and that
  reason does not apply here.
- **The measurement** (`residual_measure.py` / `.json` / `.log`): the native path at a 50M
  ceiling, under `mac_heavy`, 93 s, peak RSS 8.9 GB.
  - Dedupe: 5,874,260 candidates, completed, 5,460 rows dropped, 20,479 cross-repo pairs.
  - The split's re-derivation: 2,892,003 candidates, completed, near_duplicate_disjoint
    passed.
- **The bound:** `POOL_MAX_CANDIDATE_PAIRS` = 12,500,000. That is twice the pool's count plus
  v4's whole-corpus 550,147 (a proxy for v5's unmeasured non-pool rows), rounded up. It is
  carried by `DataConfig.max_candidate_pairs`, which `pool_data_config` sets in the bench's v4.
  Dedupe and split both read it, and both reports name a non-default bound. A build that still
  exceeds it reports not_run and refuses.
- **The draw key: yes.** Open-Jev's val draw is keyed on the scene (group_key) alone, not
  (family, group). That closes the 3 painting-geometry scenes that were train in one family and
  val in another. It only tightens the split, and the pool is not yet admitted.
- **Pre-registration:** the four-family scoping, the bound with its basis, and the draw key go
  into the v5 DRAFT as an amendment before any v5 result exists, after the v5-2gpu merge.

## Landed

- **v5-build 51272d5 (lead):** exact-content dedupe for marked rows, `exact_content_disjoint`,
  the config-carried bound, and 15 tests.
  - Fail-first: 7 fail on ce0acdf on their behaviour.
  - The characterization re-pin, acd6ea15 → d65616af, is fully accounted
    (`char-v4-compare.out`): 30 of 33 files are byte-identical, every manifest included, and 0
    binary files differ.
- **The bench's v4** (marker, `pool_data_config` bound, draw key) is cut on 51272d5.

| File | sha256 |
|---|---|
| `residual.json` | `317d2066121a88a7994392b10548ffd139750eeb382fbf1892884341ec9927df` |
| `candidates_by_family.py` | `079201102151d78db5ab83443f20e0ecc9b48a16ad9538be4bb48485fdee38f4` |
| `candidates_by_family.out` | `81809145a59151826b4efe9ef21a7bcd58bac563312b77d6bd661ee2a8d599a7` |
| `pairs_look.py` | `fa392a061cf9244fe88b0f89673692ed2f17af4c521fe96a45e5ae80287943c7` |
| `pairs_look.out` | `74f1e4a8c82dfe17e1039f9b19390238784a41b5d7576695403b0fc8a741a190` |
| `residual_measure.py` | `c431369238013d53d95d0b165e2661f3502c3ca2c19032f310d43ea596432516` |
| `residual-measure.json` | `b57131a45b071d77f9d48ab31de5b21bbd5ce7442e351a64b8234b9be45be38c` |
| `residual-measure.log` | `977e09e01b62e3c89a07b8bf48427580e10d63e972c1c3d2e132b83d355737e7` |
| `char-v4-compare.out` | `daf0184b279b39b0c10610dd695f310f92e6208faec1e522750db4f2873d1d17` |
