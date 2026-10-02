<!-- Fable's ruling on F's one batch order and v5's data order, 2026-10-02 ~17:58 UTC. Read-only lane (code-reader, model fable); asked by the lead after L-recency (HANDOFF/recency-2026-10-02.md, merged at 4e538c0). -->
<!-- Recorded by the lead from line 5 to the end of the file, verbatim but for ONE disclosed normalization: two gap ids Fable wrote without their date suffix (lines 19 and 36) have "-2026-10-02" appended, because each is an exact prefix of exactly one record and test_gaps_ledger refuses a whole citation that does not resolve. The text as Fable wrote it, lines 5 to the end, had sha256 7bcb0119c9fa7f4aa70490e8f85c1b4fb3a4587b5bf4c2c123544daf355a2bd6. Nothing else below this header is edited. The lead follows it (the human's standing instruction). -->
<!-- Applied by AUDIT/post-f-2026-10-02/apply_seed_order_amendment.py (the DRAFT) and routed to L-v5-train (the flag, the metrics, the three comment sites) and L-v5-rules (the pairing check). -->

# Fable ruling: F's one batch order and v5's data order (2026-10-02)

Read-only lane. No file edited, nothing launched. Labels: [V] read here, [I] inferred, [U] unverified. DevMap/GitPulse not called (read-only ruling; every cite is a direct read).

Premise corrected first: v5 cannot "keep F's order". F's order is `_plan` at seed 20260919 over F's shard set (`shards.py:2389-2392` [V]); v5's shard set differs, so even option (a) is a fresh single order, and `comparability.rule` already forbids reading v5 against F's envelope (`v5-preregistered.DRAFT.json:162` [V]). The live question is only what v5's own envelope should contain.

## 1. v5's data order: (b), paired, in this exact form

**Recommendation: (b).** Vary the order with the training seed; arm seed s and J5' seed s share v5 seed s's order.

Form:
- **Flag** `--batch-order seed` in `tools/real_ft_run.py`; absent = today's behaviour (plan at `DataConfig().seed`, `:10674` [V]). **Recipe key** `batch_order = "seed"`, written by `_recipe_pieces` only when given, like `min_lr` and `option_permutation_seed` (`:541-542`, `:568-569` [V]), so every F/v4 row hashes as before. The value must be the constant string, never the numeric seed: `one_configuration` refuses an envelope, average or ensemble whose ft rows do not share one `recipe_hash` (`qd_post_f_rules.rs:2869-2906` [V]), and the arm's identity is "v5's recipe plus exactly two keys" (DRAFT:119 [V]). A per-seed numeric key would break seeds34, J7'-style averaging and the arm checker at once.
- **Derivation**: plan seed = the training seed, passed only to `reader.batches(seed=...)` at `:10674`. Not through `config.seed`: that feeds `data_snapshot_hash` (`config.py:152` [V]) and every suite, alphabet and inventory call (`:4440, :5215, :5926, :10441` [V]). The planner null already computed plans at seeds 0, 1, 2 (`recency-f-2026-10-02.json` `seeds_0_1_2_counterfactual` [V]), so the derived orders are on record.
- **Pairing**: arm seed s and J5' seed s inherit the flag through "v5 recipe flags, conditionals as v5 ran them" (DRAFT:103 [V]); make it explicit (section 2).
- **Digest**: two ft-row metrics beside `corpus.plan_batches` (`:3205-3212` [V]): `corpus.plan_seed` (the number) and `corpus.plan_order_digest` (sha256 over the plan's `(bucket, rows)` tuples in order, computable on the Mac without tokens), plus `train.consumed_digest` copied from the final checkpoint (`:1950-1968` [V]). Metrics do not feed `recipe_hash`. This closes GAP-F-FT-ROWS-CARRY-NO-ORDER-SENSITIVE-PLAN-DIGEST-2026-10-02 forward; for F it is resolved by the box: four checkpoints carry `1f382091...`, which also shows the box's numpy permuted as the Mac's did [V, the lead's read].

Why (b): the arm "feeds the next re-plan only" (DRAFT:155 [V]); a hold-rate measured over three orders is a claim about the recipe, one over a single order is a claim about that order, and L-recency showed one order can sit low on late noul rows (v3b choice-slot p 0.011, uncorrected [V]). Pairing keeps the targets' seed-for-seed comparison as tight as (a). And it makes the recency diagnostic my post-F ruling asked for runnable on v5 at $0: three orders beside three hold/no-hold outcomes and the trajectory rows.

Code consequences L-v5-train must not ship half [V from `:10674-10703, :10974-10976`]: `plan_all`, `labels_by_batch_all`, `keep`, `plan_small` and C1's `epoch_alphabets` all derive from the one plan and move inside the seed loop, one plan materialised at a time; the permutation itself is seed-independent (`:3237` [V]) so C1 is unchanged; batch count, widths, `plan_rows` and padding are seed-invariant by construction (`shards.py:2386-2396` [V]) so the prelude's batches/width check stands, and it should print per-seed `plan_order_digest` and assert the invariants. Test that fails pre-fix: plans differ across seeds under the flag, identical and recipe-hash-unchanged without it.

## 2. Envelope and threshold consequences

**Recommendation: no threshold moves; four DRAFT text amendments.**
- **Seeds 3-4 spread (&gt; 0.30)**: unchanged. F tripped it at 0.344 under one order (post-F ruling:22 [V]), so order is not needed to fire it; under (b) it may fire more readily, exposure +$32 / +14 h, already inside the human's yes (DRAFT:23 [V]). Amend `seeds.seeds_3_4`: "seeds 3 and 4 run with `--batch-order seed`, plan seeds 3 and 4."
- **R9**: seed 0 only, absolute bars (DRAFT:188 [V]). Nothing changes.
- **v5nw.room / seed_holds**: read hold counts, not orders (DRAFT:125-126 [V]). Nothing changes. Amend `arm_noul_weight.confounds`: "seed for seed, including the batch order (plan seed s on both sides)"; amend `identity.ft_rows` to add "`corpus.plan_order_digest` equal to v5's of the same seed" as a pairing check for L-v5-rules.
- **Strict-envelope guards**: text unchanged, but under (b) v5's min/max widen with order variance, so "arm min &lt; v5 min" fires less readily. Named, accepted: the price of a generalisable result. [I]
- **J5'**: amend `j5prime.what`: "J5' seed s trains on v5 seed s's batch order (`--batch-order seed`)."

## 3. Readings already made on F

- **R4_lower_layers**: rule stands; the x/5 count is over one order. DRAFT amendment (not bound): append "F seeds 0-4 share one batch order (GAP-REAL-FT-RUN-BATCH-ORDER-IGNORES-THE-TRAINING-SEED-2026-10-02), so the count measures init and kernel variance at one order."
- **Seeds 3-4 rule that fired**: bound (`f-v4-preregistered.json:7` [V]); report-only note: 0.344 arose with no order variance at all.
- **Bimodality statement**: strengthened, not changed. "The recipe does not determine the outcome" becomes "the recipe and the data order do not determine the outcome": init plus kernel nondeterminism alone separate 54/60 from 0/60. One blind spot: "2 of 4 with flag, 0 of 3 without" is clean only if J4's three seeds also shared one order, [U] at their commits (gaps.jsonl:986 [V]). Report-only CPU check, not a blocker.
- **fsucc / F' envelope**: bound and launched; nothing changes. Report-only note: F's envelope excludes order variance and the arms share F's order, so it is a tighter (more sensitive, less general) test than its text implies; the room arithmetic stands as computed.
- **J6(f)'s pairing**: true of every F seed; the pairing J6(f) relies on is real. Report-only note in the handoff; no edit to a bound file.

## 4. The three code sites

- `backbone.py:1064` [V]: a comment; false as written. Code-free correction by L-v5-train, which already edits `_letter_loss` in that file: "determined the training stream (and, only where the epoch arm passes it to the planner, the batch order)".
- `trainer.py:692` [V]: a refusal message; the actual order guard is the consumed-digest check at `:777` [V]. Under (b) the sentence becomes true; amend it to name both the stream and the digest. Code-free, L-v5-train.
- J6(f)'s sentence (`f-v4-preregistered.json:27` [V]): bound; report-only note only.
- `artifacts.py:604-606` [V] is accurate about `_plan`'s arguments; add "the plan seed, which the epoch arm sets from `DataConfig().seed` unless `--batch-order seed`" when touched.

## 5. Anything else before launch

- **Cost**: unchanged, $137.2 / 60 GPU-h (DRAFT:13, :19 [V]); (b) is $0 GPU; three Mac plans at ~38 s, 1.57 GB each (handoff:173 [V]).
- Record the box digest read (four checkpoints, one digest) in GAP-F-FT-ROWS-… and GAP-REAL-FT-RUN-… as the confirming evidence; the first is resolved for F with the metric above as the forward fix.
- `amendments_pending`: add "per-seed `corpus.plan_order_digest` as the prelude printed them" and "`batch_order` key present on every v5, seeds 3-4, arm and J5' ft row".
- Nothing here touches the live waiters, F's rows, fsucc or the bound pre-registrations.

**Not verified**: that the v5-noulw checker reuses `one_configuration` (DRAFT:159 says it reuses the successor's arithmetic; the identity code does not exist yet [U]); J4/J6(b) order at their commits [U].
