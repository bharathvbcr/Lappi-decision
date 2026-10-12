# HANDOFF needle end-of-context split and the 16-option calibration miss, 2026-10-12

Lane: board-sweep lane 7 (ft-a4ef800cc7a items 1, 2, 3; item 4 is wave 3, item 5 is
`HANDOFF/ood-feature-space-feasibility-2026-10-12.md`). Nothing was trained, scored or loaded;
every number below was re-read from a ledger row or a verdict/report file this session.
**Nothing here is a gate.** Thresholds, seed counts and held-out sets are untouched (rule 2).

Labels: **[V]** read or computed this session from the named file; **[I]** inferred; **[U]**
unknown.

## 1. The verdict string (item 1)

`python/qd_train/needle.py` appended "A recurrent model losing early context looks exactly like
this" to every failing verdict. It now names the failing bucket's position, for example
`FAILS: depth 80-100% (end of the context) is below the 0.95 threshold.`, and asserts no cause.
Free text only; `passed`, `value` and `n` are pinned unchanged by
`python/tests/test_needle_verdict_position.py`. [V] No code reads the verdict half of `detail`
(`rg` over `python/ tools/ crates/`). Rows already written keep the old text (append-only ledger,
row hashes were taken over what was written); nothing was rewritten.

## 2. The v5 needle misses, per case (item 2)

Eval rows are in `/Users/bharath/qd-campaign/ledgers-h100-2026-10-05/ledger/h100x2-v5-2026-10-03.jsonl`
(the average's row in `h100x2-v5avg-2026-10-05.jsonl`); they are not under this repo's `ledger/`.
Verdict files: `/Users/bharath/qd-campaign/verdicts-2026-10-05/v5/suite-verdicts-s{0..4}.jsonl`
(sha256 prefixes `be77b06392d6b267`, `d6c93b3fe7650147`, `933e134c17fb0fcc`, `456845d55bb028e7`,
`b0ec043cb629fe7a`, matching `HANDOFF/needle-span-mac-2026-10-06.md`). Each holds the same 300
cases, 60 per language. Script: `build/needle-span-mac-2026-10-06/needle_diag.py` plus this lane's
`scratchpad/lanes/lane7_needle_split.py` (throwaway, never ships).

| Seed | Eval row | Worst bucket (ledger `value`) | Misses | Gate (0.95) |
|---|---|---|---|---|
| 0 | 166f7ebd | 80-100%: 0.918 (56/61) | 7 | fails |
| 1 | e1bafd6f | 0-20%: 1.000 | 0 | passes |
| 2 | 2af3c80d | 80-100%: 0.705 (43/61); 60-80% 0.864 | 26 | fails |
| 3 | e09b652a | 80-100%: 0.967 | 2 | passes |
| 4 | a51c5bf4 | 60-80%: 0.966 | 4 | passes |
| 5-seed average | 6af73bef | 0-20%: 1.000 (all buckets 1.000) | 0 | passes |

[V] Misses by language and depth bucket (misses/cases; every other cell is 0 on every seed, and
Rust, Go and Python have 0 misses at every depth on every seed):

| Seed | Swift 60-80% | Swift 80-100% | TypeScript 60-80% | TypeScript 80-100% |
|---|---|---|---|---|
| 0 | 2/12 | 2/13 | 0/12 | 3/12 |
| 2 | 5/12 | 8/13 | 3/12 | 10/12 |
| 3 | 0/12 | 0/13 | 0/12 | 2/12 |
| 4 | 0/12 | 0/13 | 2/12 | 2/12 |

Other facts [V]: 39 misses over seeds 0-4, 27 distinct cases (16 missed by one seed, 10 by two,
1 by three). No abstentions. Every wrong pick is 1 (34) or 2 (5) hunks **before** the needle.
Miss depth fractions run 0.632-1.0. Hunk counts per case are Rust 108-116, Go 120-125, Python
137-147, TypeScript 107-108, Swift 126; Rust (more hunks than TypeScript) has no misses, so the
hunk count is not the explanation.

**What the misses are:** a late-context, language-specific confusion between adjacent hunks. They
are not early-context loss (0 misses in 0-60% on any seed) and not abstentions.

**Cause: [U].** The candidate hypothesis is in `HANDOFF/needle-span-mac-2026-10-06.md` section 1b
[I]: the span pointer scores a line by the state at its first token, the needle's first two tokens
(`@@`, ` -`) are shared with every hunk, and TypeScript and Swift each have one filler template
(`needle.py` `_FILLER`), so consecutive prefixes differ only in drawn names. No ablation has been
run. The measurement that would discriminate is arm (e) of
`campaign/span-refit-mac-preregistered.json` (last-token candidates on frozen features; wave 3,
heavy, report-only) and, separately, a suite variant with two Swift/TypeScript filler templates
(a new report-only suite, not a change to the gate's population).

**Owner:** `GAP-V5-NEEDLE-END-OF-CONTEXT-MISSES-SEEDS-0-2-HAVE-NO-ID-2026-10-06` is now owned by
the v6 span/needle arm-(e) lane (wave 3 of the board sweep), with the per-case facts above.

## 3. The 16-option calibration miss, re-read from rows (item 3)

Source: `/Users/bharath/qd-campaign/preview-2026-10-05/calib-two-fold/report.json` (sha256 prefix
`9011f4561e19a438`), the two-fold `qd-calib-fit` report over eval row 6af73bef (17,254 choice
rows; `rows.jsonl` sha256 prefix `c15e367b7d008d2a`, 17,254 lines). ECE: 15 bins, threshold 0.05.

**A naming discrepancy first [V].** The release notes' "16-option questions" is the table entry
`choice:16`, which has `options: 15` and `rows: 16` (15 real options plus the NUL row). The entry
with 16 real options is `choice:17`, and it passes. Anything quoting "16-option" means the 15-option
`choice:16` entry. This lane did not edit the notes' numbers (ruling 2j.11: no number changes).

| Quantity (`choice:16`, n = 1,583) | Value | Check against the notes |
|---|---|---|
| ECE before calibration | 0.1726 | notes 0.173: matches |
| ECE after, two-fold | 0.0790 (fails 0.05) | notes 0.079: matches |
| Refused at the 0.80 target | 802/1583 = 50.7% (765 by margin, the rest by the NUL row) | notes 51%: matches |
| Retained precision after refusing | 617/818 = **0.754** | not in the notes; **below the 0.80 target** |
| Conformal coverage / mean set size | 0.891 / 1.83 | |
| Fold-a vs fold-b fitted `noul_margin` | 0.922 vs 0.241 | not in the notes |
| Fold-a vs fold-b fitted temperature | 1.94 vs 1.74 | |
| Gold-NUL rows | 72 | |

Compare `choice:17` (16 real options, n = 1,604): ECE 0.0229 before, 0.0131 after; retained
precision 0.947; refused 96 of 1,604.

Not re-derivable here: "about 98% refused at 0.95" (`v6-plan-proposal-2026-10-06.md:176`). The
report was fitted at target 0.80; no 0.95 fit exists on disk, so that figure is **[U]** until
`qd-calib-fit --target-precision 0.95` is run.

[I] The two folds disagree by 4x on the margin that decides abstention in this entry, which fits
the entry's NUL-row confusion (72 gold-NUL rows) more than a smooth miscalibration. Not tested.

**Carried as a named report-only v6 item: `V6-REPORT-16OPT-CALIB` (the `choice:16` entry).**
It is **not** added to v6's reading rule (R8 is pre-registered; a gate added after the fact is
what rule 2 forbids, Fable ruling 2j.5). A v6 evaluation reports these rows' numbers for
`choice:16` next to v5's above: ECE before/after, refused share at 0.80, retained precision, and
the fold-wise margin spread. Owner: v6 design (lead). Whether it ever becomes a gate is the
human's decision.

## Open

- Cause of the Swift/TypeScript late-context adjacent-hunk confusion: [U] (arm (e), wave 3).
- 0.95-target refusal share for `choice:16`: [U] (needs a fit).

First command for the next lane: `python3 build/needle-span-mac-2026-10-06/needle_diag.py`
(reproduces section 2) then arm (e) via `tools/mac_heavy.sh` from `build/span-pin-8a531e8`.
