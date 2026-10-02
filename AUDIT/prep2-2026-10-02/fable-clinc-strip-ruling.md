# Fable's ruling on the CLINC containment strip (2026-10-02)

The lead asked at ~15:50 UTC 2026-10-02, after lane L-prep2's subsample re-run (HANDOFF/prep2-2026-10-02.md). Advisor: claude-fable-5-1, in a read-only lane. Recorded verbatim below (source sha256 ec2c1e6d00c8342cfeb50e389ded98eadb4cff2ea80e7b13c99dfbb3a288d53e). Fable's Python oracle is kept beside it as `clinc_oracle.py` (sha256 638eb349ea9b9be2f691d37ab8ca4e2f3b839821efc2477334a01c8148c90370); it is throwaway analysis and is never shipped. The lead applies the ruling under the human's standing instruction to follow Fable's recommendations. The human owns GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02 and ratifies by the commit that renames the DRAFT.

# Fable ruling: CLINC containment strip (2026-10-02)

Read-only. DevMap is degraded and GitPulse was not consulted; every cite is a direct read. [V] read or computed here, [I] inferred, [U] unverified. n = 8, threshold 0.5, val and held-out populations do not move. The lead commits this ruling as `AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md` in or before the DRAFT amendment commit, so the pre-registration's citation below resolves.

One correction of mine first: §3(a)'s caveat named `intent.classification`. The per-domain lists are `intent.within_domain`'s (`general.py:441-454`); classification samples 16 labels per row (`mixture.py:480-502`) [V]. The lane read the rule as written; the 63-66% is the rule's, not the lane's.

Rule 2, pre-empted: the amendment moves CLINC's measured exclusion after the number was seen, and that is the procedure §3(a) pre-registered — the subsample decides, the amendment follows, before the full scan. Containment is not a gate, and the strip removes template, not overlap. The expected result is ~0, not a convenient one: CLINC's split unit is the intent (`repo_key = clinc-intent:{intent}`, `mixture.py:468-471`, `general.py:418`) [V], so every val/held-out utterance belongs to an intent absent from train, and same-intent paraphrases — the only population that shares 8-word runs — never cross a split by construction.

## 1. The amendment: (b), which equals (c)

What a CLINC option value is — a label from a closed vocabulary (`domains.json`: 10 domains; `intent_names.json`: 151 names; yes/no) [V] — is a property of the family's schema, not a statistic of the sample. (a) is a sample statistic: it would strip `within_domain`'s 10 lists and MMLU's moral-scenarios set (51 rows at 5%, more on the full corpus), and what counts as template would move with the corpus. Rejected. (b) and (c) yield the same text by construction (question by constancy, options by (2b), the context block remains); stated as (b) so the strip stays one function with one more rule, fail-closed.

Replace the strip paragraph of `data.decontamination.rule` (from "Constant template text is stripped" through "before the full scan") with:

> Constant template text is stripped before n-gramming, for containment only (`qd_train.containment_strip`, STRIP_VERSION 2). Template is: (1) the question line — every qd_data family renders `family.description` there (`python/qd_data/mixture.py:238-264`), so it is constant within every (family, slot) and is stripped in every family; the per-row question of MMLU, CSQA and SQuAD is in the context block, which is never stripped; (2a) within each (family, slot), every option value that every rendered row of that family-slot carries (membership, canonical order, `seed=None`, over the union of sources and targets; a group of fewer than 2 rows strips nothing and is named in the attestation); (2b) in intent.classification, intent.domain, intent.in_scope and intent.within_domain, every option value, because their options are drawn from CLINC's closed label vocabulary (`mixture.py:480-525`, `general.py:433-454`) and are never written per row. What remains of an intent.* row is its context block, the utterance, identical across the four families; the function refuses if any (2b) row's stripped text differs from its context block byte for byte. No other family is in (2b): MMLU's and CSQA's option values are per-row content and are kept, shared sets included; a per-(family, option-set) statistic was considered and rejected as a property of the sample, not the schema. The strip is one function in python/qd_train, applied identically to sources and targets, in the exporter before the request is written and in the decontaminate oracle; qd-prep containment's n-gram core is unchanged and the parity test still compares complete pair lists. A slot text left with fewer than 8 words is too_short, as today: key (ii) cannot see it; attestation v2 counts it per family-slot and per set (`too_short_after_strip`) and, per set, the distinct identity keys all of whose slot texts are too short (`key_ii_blind`); such rows are covered by keys (i) and (iii) only — (iii) shingles CLINC's dedupe_text, the bare utterance, at k = 5 tokens and Jaccard >= 0.8 across intent splits — and are reported as unprotected by (ii), never as clean. Attestation v2 also records per family-slot the stripped strings' count and sha256, and for each (2b) family-slot the number of distinct option values seen. Keys (i) and (iii) are unaffected; val and held-out rows never move; n = 8 and 0.5 do not move. Measured before applied: the 1/2/5% subsample scans are re-run under version 2 and must meet the pass condition in AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md §4 before the full scan runs.

MMLU's shared option sets: untouched; `option_sets.largest_share_rows` (51 at 5%) stays in the record [V, slot-summary-p05.json].

## 2. Too-short rows: no new key

Under version 2 every intent.* slot text is the utterance, so at 5% 451/1089 train, 21/69 val, 34/64 held-out keys are invisible to key (ii) [V]. They are covered:

- (iii) hashes `dedupe_text`, the bare utterance for CLINC (`mixture.py:510,523`, `general.py:470`), as whitespace-normalised token 5-shingles, case kept, under 5 tokens yielding one shingle (`minhash.py:100-108`; `config.py:63-64`), Jaccard >= 0.8, across repo splits (`split.py:460-476`) — CLINC's repo being the intent, a short utterance crosses only if token-identical. It ran and passed on each subsample (`attestation.json` splitter_checks; 5%: "8713 candidate pairs, 0 crossing") [V].
- (i) covers the same-intent exact duplicate (`general.py:418-420`) [V].
- My oracle over the lane's three request.bin files (scratchpad `clinc_oracle.py`): utterance-only containment hits 0 train keys at 1/2/5%; of the key-(ii)-blind val/held-out utterances, 0 have a lower-cased `\w+` exact match in train and 0 are a contiguous word-subsequence of any train utterance [V, Python oracle — qd-prep parity not shown here; its too_short counts reproduce slot-summary-p05's 451/21/34].

A key specified after measuring its population at zero is a story, not protection. Required instead: the full scan repeats the two counts for the key-(ii)-blind val/held-out keys (an AUDIT script over the request.bin) and the handoff reports them; a nonzero is a `GAP-` for the human, not a rule change. Reporting: `export.template_strip.key_ii_blind` per set; the handoff names those keys "unprotected by key (ii), covered by (i) and (iii)".

## 3. The CLINC unit

Measure per slot text, report per identity key. The exclusion unit is already the key (`exclusions.py:189-198`); an exporter collapse is a new behaviour that changes no exclusion, and the parity oracle is keyed `row_id#slot`. Under (2b) the four texts per key are identical, so set-level `rows`/`too_short` count 4 per CLINC key and pairs inflate 16x — hence `key_ii_blind` counts keys, and the handoff reports CLINC as `family_rates.py`'s group (distinct keys) with distinct (source key, target key) pairs.

## 4. Verification before the full scan

Same `sub-p01/p02/p05` inputs (their `subsample.json`), rev 881ab304, same corpus flags, version-2 strip. Pass iff all hold at all three sizes:

1. CLINC (intent.*) excluded keys 0/246, 0/464, 0/1089; enforced intent.* pairs 0.
2. The non-intent subset of `exclusions.txt` byte-identical to version 1's: 8 / 16 / 41 keys. Recipe: the keys not starting with `clinc-intent:`, in file order, each followed by LF (the trailing LF is hashed); sha256 `2f172e00166496dc1ca66b568ffd303c1694d8e167e53c825b419057046aae31`, `6f4e9364f30da9468553651df787c1a8b1791a4ecf673d6a201a53a90b79c268`, `889e0ecf85f395af7b762980bc38da6e933f9a27e38ff757fe8c462c335d71ac` [V]. (Assumes cross-family hits stay 0, as in all six scans: `hit_by_own_rows == excluded` for every non-intent family [V, rates-subsample.json].)
3. Every intent.* slot: question stripped, every row's text equals its context; `too_short_after_strip` per set 102/4/2, 197/9/7, 451/21/34 (train/val/heldout) [V, oracle]; `key_ii_blind` the same per set.
4. CLEAN; all three splitter checks ran and passed.
5. Reported: the side-by-side table with a version-2 column per family (CLINC per key, MMLU, CSQA, SQuAD, defect_class), pairs, keys, too-short per set per key, distinct option values per (2b) slot [I on those counts].

Two tests ship with it, each failing against version 1: two same-domain `within_domain` rows sharing only the list (v1 pairs them, v2 does not); the hook refusing a version-1 attestation. Any deviation: stop, record a gap, no full scan. The full scan's CLINC rate, key-(ii)-blind counts and the two zero-checks go in the handoff. No fresh ratification step: the human ratifies by the rename commit, as §3(a) said; the gap is the human's, so the rename commit and the human-answers line cite the three subsample rows and the full-scan numbers.

## 5. Merge

No objection, with the window named: until the version-2 commit, a version-1 full-scan list would pass `read_exclusions` (`exclusions.py:132-137`); the gate is procedural and closes when `STRIP_VERSION` becomes 2 and `STRIP_RULE` names the amendment — bump both; the hook compares both exactly. Conditions: no full scan and no list applied to a build in the window; `gaps.jsonl` UU resolved as an append-only union, no line dropped, `test_gaps_ledger` re-run; do not commit the build-rewritten `Cargo.lock` or the other session's ` M` files (qd-metal, qd-runtime, qd-preflight, docs) — no `git add -A`; the three refit rows stay in their own ledger file, and the merge commit states no reader is pointed at it.

## GAP ids (the lead appends)

- GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02 — append: (2b) chosen, oracle prediction 0/0/0, version 2 pending; stays open, human.
- GAP-CONTAINMENT-STRIP-RULE-PREMISES-2026-10-02 — resolve: the four readings adopted into the amendment text, plus the caveat correction (within_domain, not classification).
- GAP-CONTAINMENT-KEY-II-BLIND-SHORT-UTTERANCES-2026-10-02 — new, open, implementing lane: 451/21/34 keys at 5% invisible to key (ii); covered by (i)/(iii); full-scan counts and the two zero-checks pending.
- GAP-CONTAINMENT-STRIP-V1-WINDOW-2026-10-02 — new, open, lead: version-1 lists pass the hook until version 2 lands; no full scan in the window.
- GAP-FABLE-CLINC-RULING-NAVIGATION-2026-10-02 — new: read directly, DevMap degraded, GitPulse not consulted; the oracle is Python, qd-prep parity not shown.
