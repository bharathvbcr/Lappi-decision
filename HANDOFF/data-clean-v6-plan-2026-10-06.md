# HANDOFF data-clean-v6 (plan), 2026-10-06

Lane: CPU-only data cleaning and new-data preparation for v6. GPU is unavailable.

- **Asked for by the human (2026-10-06):** "clean all the training data properly with scripts and
  fetch high-quality data ... it shouldn't repeat lappi 0.1 mistakes".
- **Designed by:** Fable, in a read-only ruling (this session). Decisions go to Fable; the human
  is asked only about cost and downloads (memory: decisions-to-fable-cost-to-human).
- **State:** a plan. Nothing has been built, fetched or rebuilt yet.
- **Labels:** [V] read by Fable or this session at the cited path; [I] inferred; [U] unknown.

## 1. v0.1 mistakes, binned

| Mistake | Bin | Evidence |
|---|---|---|
| 215 val rows also in the MMLU/CSQA train set | fixed in v5; keep the test | `/Users/bharath/qd-campaign/v5-containment-v2-2026-10-03/attestation.json`: `clean:true`, 2,334,089 pairs, 104,905 exclusions, remaining hits val 0 / heldout 0 [V] |
| Mutants of one function across splits | fixed | `identity_disjoint` passed [V]; `python/tests/test_dedupe_split.py::test_no_function_identity_appears_in_two_repo_splits` |
| Contrast rows knocking out val rows | fixed | `phase4-v5r-2026-10-03.a7-reading-c.json`, 45/45 [V] |
| Dedupe keep rule is lexical and split-blind (`python/qd_data/dedupe.py:444`) | **open; data; CPU** | v5 worked around it with a named drop list; `test_dedupe_keeps_the_lexicographically_smallest_unit_whatever_the_input_order` pins the bug [V] |
| The LSH bound counts pairs at J 0.5-0.8 that are not duplicates | open; data; CPU | GAP-DEDUPE-LSH-BAND-CANDIDATES-NOT-DUPLICATES-2026-10-03 |
| Composed rows: file-level near-duplicates across splits are not checked | open; data; CPU | GAP-COMPOSE-NEAR-DUP-FILES-ACROSS-SPLITS-UNCHECKED |
| MMLU test and CLINC test items are training data | open; data; CPU (re-pin) | GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03 |
| `pairwise.helpfulness` is a 2% share with a length prior | open; data; CPU (balance) plus a download | audit 2.6 [V] |
| Families with no train rows in the paired-margin pool | open: add train rows (dropping val rows is the human's re-spec, rule 2) | GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-TRAINING-ROWS-2026-10-05 |
| No natural-bug data; the transfer gate is unspecified | open; data; CPU (mine own repos, held-out only) | audit 4.7 |
| Admission, diff marker, line-start candidate state, single-template needle | not data: a serving or eval contract; reported only | audit 2.8-2.9; `HANDOFF/needle-span-mac-2026-10-06.md` |
| No caller traffic | not data: integration. Synthetic-by-construction rows are the CPU stand-in. | audit 2.7 |
| Decontamination is lexical (8-gram) only | open; CPU; report-only tier | audit 4.8 |

## 2. Inventory [V, Fable read the manifests]

- **`phase4-v5r-2026-10-03`**
  - Train: 558,790 sequences, 386M tokens.
  - MinHash splitter: 3.58M candidates, 0 cross-split confirmed at 0.8.
  - Containment v2: clean against val and held-out.
- **`v5-data-2026-10-02/.../commitpackft-composed-v2`**
  - 25,000 composed rows, train only.
  - Dedupe and containment were row-level only; the file-level check is the open gap above.
- **`v5-decisions-data-2026-10-03/pool-v5-decisions-v4`**
  - 214,760 rows from 9 sources, with a group split.
  - Containment targets: arc-test, boolq-val, vitaminc-test, jevbench-public and jevjudge
    (RM-Bench 11,605 + RewardBench2 13,323 + JudgeBench 620).
  - Those targets apply only to the decisions build, not to the code families.
- **Not done anywhere:**
  - a semantic decontamination tier;
  - MMLU-test and CLINC-test as containment targets;
  - RM-Bench code pairs checked against the code families;
  - file-level units for composed rows;
  - a source-function cluster id on mutant rows.

## 3. The cleaning change set

All of this lands as one `qd-prep` change set, followed by one heavy rebuild. The rebuild is
report-only, with no training.

| # | Check | Reuse or extension | Test that fails on the pre-fix code |
|---|---|---|---|
| 1 | Licence per row: default deny; non-commercial is disqualifying | `python/qd_data/licences.py` (oracle); `crates/qd-prep/src/decisions.rs` per-row pattern | a row licence `unknown` under dataset `mit` is refused |
| 2 | Exact-digest dedupe | `dedupe._exact_content_clusters`; `qd-prep sha256` | – |
| 3 | Near-duplicate dedupe with a **split-aware keep**: heldout > val > train, then lexical; within-repo pairs reported, not merged | `qd-prep minhash` and `lsh`, plus a new `qd-prep dedupe` subcommand (union-find with the keep rule) | flip `test_dedupe_keeps_the_lexicographically_smallest_unit...` to "keeps the eval member"; a fixture whose train key sorts first |
| 4 | LSH prefilter on signature agreement (estimated J ≥ 0.65) before the pair bound | `crates/qd-prep/src/lsh.rs` and the `qd_data.minhash` reference, changed together | a band collision at J=0.3 is no longer a candidate |
| 5 | `cluster_key` (source-function identity) on every mutant row; cap mutants per function per class; composed rows signed per file block | `tools/containment_scan.py` exporter; containment and lsh | a composed train row holding a val file's near-duplicate is excluded |
| 6 | Unpin MMLU test/dev and CLINC test from train (`python/qd_data/sources.py`) and pin them as targets; add the jevjudge targets to the code-family scan | `qd-prep decisions --target`; `qd-prep containment` | no train `identity_key` derives from a test file |
| 7 | Semantic tier, report-only: all-MiniLM-L6-v2 (already in the HF cache; sentence-transformers 5.6.0 is in the ML venv, so no new dependency), top-k cosine from val/held-out/targets to train | throwaway Python; heavy, so it takes the lock | – |
| 8 | Label sanity: per-family option-position prior; `linfit` per family as a format-leak detector | `qd-prep linfit`; `spancheck` | a synthetic family with the label in the option text → linfit ≈ 1.0, flagged |
| 9 | Pairwise length: stratify so P(gold is the longer answer) ∈ [0.45, 0.55] per split; record `len_ratio`; add length features to the linear control | `decisions.rs helpsteer()`; `crates/qd-prep/src/linwire.rs` | a fixture where the longer answer always wins → rebalanced or refused |
| 10 | Family balance: report any val stratum with 0 train rows | `qd-prep decision-caps`; `survey_train_availability` | a config with a val-only stratum → a report line |
| 11 | Re-run A7 | `tools/v5_a7_check.py` | – |

## 4. New data with no GPU

**Never train on:**
- BFCL
- RM-Bench
- RewardBench 2
- JudgeBench
- SWE-bench, any split
- MTEB/BEIR test sets
- When2Call test splits
- FormGym

The full refused list is in `AUDIT/data-licence-survey-2026-10-06.md`. Rule 7 applies to every
tool and computer-use slot: Lappi admits, it never authorises.

| App | Slot shape | Source | Held out |
|---|---|---|---|
| jevgrep | choice over candidate files; span in **file mode** (no `+` marker) | own-repo commits by construction (`campaign/v6-caller-families.DRAFT.json`, build_order 2); licence-filtered public code search | 20% repo-hash holdout |
| Jarvis | choice over enumerated accessibility elements; a risk slot (execute / ask / abstain) | synthetic trees and forms generated by rule from `Jarvis/profiles/*.json` and `scenarios/*.inputs.json` | the real `scenarios/*.contract.json` |
| Email sorter | choice over `n8n-email-engine/config/label_taxonomy.json` labels + junk + abstain, with label-flip injections | synthetic rows from the `mailFilters.xml` rules; injection sets (licence-cleared) | real mail is a human privacy decision |
| ScholarLM rerank | pointwise yes/no with an instruction field | SciRepEval train configs (ODC-BY); BM25 hard negatives in Rust (std only) | BEIR scientific test sets; real queries |
| Tool selection | choice over tool names + {ask, unable, answer}, with names maskable | the apps' MCP catalogs → rule-generated queries; irrelevance by removing the gold tool; When2Call train and ToolACE | MANVI `contracts/verdict.cases.json`; GitPulse's ambiguous subset |
| GitPulse | commit type, scope, breaking, commit intent | own-repo history | held-out repos |
| Defect families | a natural-bug **held-out** set only: 100-300 single-statement fix commits from held-out repos, dated after the base model's cutoff ([U]) | own repos | all of it |

## 5. Order while the lock is busy

- **Pure code now (no lock):**
  - §3 items 3, 4, 5, 6, 9 and 10 in `crates/qd-prep`, with tests;
  - the own-repo enumerator;
  - the download list.
- **Needs the lock:**
  - `cargo build` and `cargo test` (`-j 2`);
  - the single rebuild (more than 2 GB);
  - the MiniLM pass.
- **The span refit goes first when the lock frees,** but only if its pre-registration is
  committed by then. Otherwise `cargo test -p qd-prep` goes first.
- **First command** (light, no lock):
  `/Users/bharath/.venvs/ml/bin/python -m pytest python/tests/test_dedupe_split.py -q -k keep`.

## 6. For the human

- Approve one consolidated download list.
- Two privacy items, surfaced but not decided:
  - real mail, form or query data as training or held-out data;
  - a Rust embedding crate (a new dependency), if the semantic tier is ever to enforce rather
    than report.

## Open

All of sections 3-4. The first command is in §5.
