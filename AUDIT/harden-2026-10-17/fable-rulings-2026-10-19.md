# Fable rulings, 2026-10-19

Asked by session lappi-decision-ce, under the owner's standing delegation, after lane W2-d's v6 shard build (ledger row eaac82e0-b65b-485b-81ec-5e09fed7eb43, branch w2d-v6-shards-2026-10-17) failed `qd-train-data` on two tiers. The body below is Fable's answer, verbatim. Earlier rounds are in fable-rulings-2026-10-18.md, -18b.md and -18c.md. The owner may veto.

## Q22. CLINC train rows equal to a test target: remove them at the builder; within the delegation

**(a) Yes, and within the delegation.** Removing train rows that equal a test target is decontamination: it moves data, touches no threshold, seed count or held-out item (rule 2 untouched; targets never move). It is also *forced*: the gate's `exact_short` tier (crates/qd-preflight/src/train_data.rs:21-23, 455-467) is stricter than the builder's rule, which is word-8-gram only and blind to targets under 8 words (containment.rs:1-20; the positive control shows `too_short: 2343` of 5,500 CLINC targets). A builder weaker than its gate fails forever; aligning the builder upward is not moving the gate. Owner not required — but record it as a Fable ruling in AUDIT, and the owner may veto.

**(b) Canonical owner (verified).** `qd_train.exclusions.apply_exclusions` is "the ONE place those keys leave the train split" (python/qd_train/exclusions.py:5-10), consuming `exclusions.txt` written by `qd-prep containment` (crates/qd-prep/src/containment.rs:18-19, 79) via tools/containment_scan.py, which under v6 already adds MMLU/CLINC targets (containment_scan.py:31-36; sources.py:71-76 `BENCHMARK_TARGET_SPLITS`). So: add an exact whole-normalised-line tier for short targets to the qd-prep scan, writing pairs and exclusion keys through the same seam, with the attestation naming the new rule so `read_exclusions` refuses an old list (exclusions.py:216-230 pattern). Move `normalised`/`exact_matches` from train_data.rs:285-288, 455-467 into qd-prep so gate and builder share one matcher. Not a loader-level CLINC filter — that would be a second exclusion seam.

**(c) Verify first.**
1. *Upstream duplicates*: digest = blake2b-8 of `utterance.strip()` (mixture.py:532, 551), un-normalised; `exact_short` lowercases and strips punctuation (train_data.rs:285-288). Hence 8 digest hits (2 utterances byte-identical) vs 44 normalised-equal. So ~42 are case/punctuation variants, not exact upstream duplicates. Run `parse_clinc` over the sha-checked general-record cache, intersect train/val vs `test` by digest and by normalised text; report both counts. Inferred, not yet run.
2. *Over-breadth*: all sampled hits are CLINC rows; the delta 1,275 stripped vs 4,833 unstripped must be explained row-by-row (stripped hits should be a subset). 1,275/525,418 = 0.24% — list the 44 utterances; a target like "yes" matching a template line is a false positive to report, not drop.
3. *Pre-registration*: it pins no per-family counts, only shard hash/sequences/tokens/max_width and `n` from width distribution (campaign/lora-2b-preregistered.json amendment 3, pending items 1-4). No CLINC cap in fable-v6-data-design-ruling.md (rg: no match).

**(d) Order.** Rule change + failing-then-passing test → rebuild on the merged tree → `qd-train-data` exit 0 → amendment 4 writes paths, hashes, re-derived `n`, names the exact-short rule → step 0. Amendment 3 already requires all of this before step 0 (lines 255-268); it must land before any result is read.

**Rebuild on merged tree (verified):** build was at `60e92ee-dirty`; d81e3ef (now merged, 44732cb) edited python/qd_data/sources.py, and `shard_code_fingerprint` hashes every `*.py` there (fingerprint.py:146-192).

## Lead's notes (lappi-decision-ce)

- The `qd-train-data` outputs and the W2-d ledger row cited above are on branch w2d-v6-shards-2026-10-17, not yet on main.
- 44732cb is W2-d's merge of main into its branch.
- In Q23 (a), the one gap id Fable cited without its date suffix is completed to its full form, as train_data.rs:22 cites it. No other word of Fable's text is changed.

## Q23. exact_short false positives: an owner question; only the 92 CLINC rows move under the delegation

(Asked after W2-d's verification, branch w2d-v6-shards-2026-10-17 at bc9eb0b: AUDIT/harden-2026-10-17/w2d-clinc-train-test-duplicates-2026-10-19.txt and w2d-exact-short-hits-2026-10-19.txt. Fable's answer, verbatim.)

**(a) Owner question — the tier is under-specified, and data movement alone cannot clear the committed gate.**

- The gate's example text is `context` + every option value appended as its own line (crates/qd-prep/src/pool.rs:105-116), and `exact_matches` probes every line (train_data.rs:455-467). The only test (train_data.rs:812-827) plants the target *inside a multi-line context* with options `["A","B"]`; no test says whether an option line or a code line counts. The doc (train_data.rs:21-23) motivates the tier by `GAP-CLINC-TEST-TARGETS-8GRAM-BLIND-2026-10-06`: short CLINC utterances. So: per-line matching within context is tested intent; option-line matching is untested implementation. Neither doc nor test says "utterance rows only" — scoping it is a gate change under rule 2.
- Decisive: the committed gate runs unstripped and reports **4,833**, not 1,275 (w2d-qd-train-data-v6-unstripped). The 1,275 is W2-d's stripped variant. Dropping 1,275 rows leaves the gate FAILED on 3,558 CLINC rows whose option list carries the intent label `goodbye`. The project's own containment rule already says template text is not contamination (exclusions.py:14-18 refuses a list made without strip v2; containment_strip.py (2b) strips every intent.* option value). The gate contradicts the rule it polices. Fixing that is the owner's.
- Cost of dropping all 1,275: the 1,106 `code.defect_class` rows (1,000 with a bare `false` line) are not contamination of CLINC and the drop correlates with boolean-literal mutants (ops.rs:710 `negate_condition`) — a label-biased deletion. No train floor exists for the family (not in `train_caps`; v6-allocation.json), val is untouched, and R1's ≥55.8% defect-token share (design ruling:37-40) moves by ~0.2% [inferred, not computed]. Whether `apply_exclusions` even reaches `--defect-class` rows is **unverified** (real_tokenizer_pipeline.py:3217 vs design ruling:39).

**Within the delegation now:** drop the 92 CLINC rows (23 utterances × 4 families, including the 8 digest hits) through the exclusions seam, as Q22 ruled. Not the 1,183.

**(b)** Drop both digest duplicates from train (they are inside the 92). Also record them as noisy test items ("what's your designation" 11/75, "where did you grow up" 77/99) in AUDIT and `gaps.jsonl` — report-only; the test set is never edited and no item is excluded from scoring (rule 3).

**(c) Owner question:** "Should `exact_short` read the strip-v2 text the builder's containment is held to, and does a short target equal to one line inside a longer non-utterance row count?"
1. Leave the gate; drop 4,833 rows (3,558 for an option label). Not recommended.
2. Strip v2 first; drop the 1,275.
3. Strip v2, and match the whole stripped slot text, keeping the 1,183 per-line hits as a *counted* residual in coverage (train_data.rs:25-26 already names embedded hits as residual); test 812-827 is rewritten by the ruling. **Recommended.**
