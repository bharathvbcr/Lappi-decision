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

## 7. Fetched (2026-10-06) [V]

The human approved all four bundles (AskUserQuestion this session). Scope was trimmed to what
the human was told:
- train splits only; no eval or test split was fetched;
- SciRepEval `search`: train shards 0-1 of 14 ("capped");
- llmail-inject: phase 2 labelled plus metadata ("small"). Phase 1 and the raw submissions
  (~2.1 GB) need a new yes.

**Run:**
- Script: `python3 /Users/bharath/qd-campaign/v6-data-2026-10-06/fetch_v6.py`. It follows v5's
  `fetch_decisions.py` format, stdlib only, with pinned revisions, a 1 GB per-file cap, sha256
  for every file and md5 checks for Zenodo. It fetches raw files only; conversion is a
  data-build step that runs under the lock.
- Record: `/Users/bharath/qd-campaign/v6-data-2026-10-06/fetch-record-v6-2026-10-06.json`
  (sha256 `7b45d05aad3103efbd774f503a0821ce27a1bf30e61491e6f782cc047af7dcdd`).
- Result: 32 files, 2,342.2 MB, exit 0.

**Contents:**
- When2Call train_sft/train_pref
- ToolACE
- SWE-rebench (filtered + test parquet)
- SciRepEval search train 0-1
- HelpSteer3 preference train/validation
- MNLI train
- llmail-inject phase 2 + metadata
- deepset/prompt-injections train
- Lakera gandalf train
- TSSB-3M `tssb_data_3M.zip`
- ManySStuBs4J `sstubsLarge`, README and licence text

**Second fetch (2026-10-06, ~16:45Z) [V].**
- **Approval:** the human said "download any open datasets". Fable scoped that to the survey's recommended permissive set, plus evals fetched as decontamination targets only.
- **Script:** the same `fetch_v6.py`. Every record entry now carries `use`, either `train` or `target-only`, and target-only files live under `targets/`.
- **Record:** sha256 `72be4cdf4307d792edcfd3b46a4cf11fbf19850910569b70517c09e24221a702`. 125 files, 9,000.1 MB in total: 50 train files (8,712.6 MB) and 75 target-only files (287.5 MB).
- **Checks:** every entry has `use`, and the `targets/` placement matches it.
- **Train files added:**
  - llmail-inject phase-1 labelled submissions and raw phase-1/2 submissions. The 1.67 GB raw phase-1 file is allowed past the 1 GB cap by a named per-file cap.
  - SciRepEval `search` train shards 2-13.
  - CodeSearchNet `python` and `go` train only. These are Lappi-admitted languages; Java, JS, PHP and Ruby are not fetched.
- **Target-only files added:**
  - When2Call test;
  - MNLI validation_matched/mismatched;
  - SciRepEval search evaluation;
  - CodeSearchNet go/python test+validation;
  - SWE-bench (dev/test/train), SWE-bench_Lite and Loc-Bench (no licence stated [U]);
  - BFCL v3 (all files);
  - scirepeval_test search/nfcorpus/trec_covid;
  - BeIR scidocs.
- **Not fetched:**
  - SPML: its README says GPT-4 generated it (provider-output flag).
  - WANLI, UltraFeedback, glaive and hermes (flagged).
  - Every non-commercial or unlicensed set.
  - CodeSearchNet's upstream `_licenses.pkl`: a pickle is code.

**Obligations carried downstream:**
- CodeSearchNet: no per-example licence. Conversion resolves each repository's licence and default-denies unknown ones.
- SciRepEval: ODC-BY attribution.
- MNLI: drop genre `fiction`.
- SWE-rebench: filter `license_name` to permissive licences; dedupe against SWE-bench, -Lite
  and LocBench.
- TSSB-3M and ManySStuBs4J: filter by each project's licence.
- ToolACE: its generator is undisclosed.

## 8. Lane status (2026-10-06, ~16:00Z)

### Lane 1: split-aware dedupe and LSH prefilter (§3 items 3-4)

Worktree `.claude/worktrees/agent-a2092c503b3782eea`. Uncommitted, not merged.

**Python [V].** This session re-ran the lane's tests against its worktree:
- `PYTHONPATH=<wt>/python QD_PREP_BIN=<wt>/target/release/qd-prep .venv/bin/python -m pytest -q test_dedupe_split.py test_dedupe_exact_content.py test_minhash.py test_qd_prep_dedupe_parity.py test_qd_prep_lsh_parity.py`
- Result: all passed, 1 skipped.
- The `qd-prep` binary that run used was built outside the lock, from the lane's code at 15:26Z.

**Rust [U]: not run.** The lane's locked script `<wt>/build/dataclean/locked-dataclean.sh` is queued on `tools/mac_heavy.sh` (label `dataclean-dedupe`). It runs `cargo test`, the release build, the parity pytest and a torch run of `test_pre_dedupe_drops`.

**Design: v6 rules are opt-in.**
- `DataConfig.with_v6_dedupe_rules()` turns on the split-priority keep rule and the J ≥ 0.65 prefilter.
- The defaults stay v5, so `test_decisions_pool`'s pinned v5 hashes do not move.
- The v6 rebuild must call it in `real_tokenizer_pipeline.run` and `real_ft_run.ft_splits`.

**Lock breach.** The pytest fixture built `cargo` outside the lock twice, at ~15:11Z and ~15:26Z. Recorded as GAP-PYTEST-QD-PREP-FIXTURE-BUILDS-OUTSIDE-THE-MAC-LOCK-2026-10-06. Lane 2 sets `QD_PREP_BIN` to an existing binary to avoid repeating it.

**Open, proposed by the lane** (not yet recorded as gaps):
- A v6 build refuses when an exact-content unit outranks the searched owner of its text. Needs a ruling.
- `qd-prep dedupe` has no caller yet; the rebuild lane wires it.
- The worktree is 75 commits behind main. Its `lib.rs`, `main.rs` and `wire.rs` need a 3-way merge with the additive edits below.

### Lane 2: benchmark re-pin, pairwise length balance, family-balance report (§3 items 6, 9, 10)

Worktree `agent-aa39b9b3c7106b39c`. The lane fast-forwarded it to main `2cee2a8` before editing, so it is NOT on the stale base. Uncommitted.

**Changes:**
- `decisions.rs`: +528/−12.
- `containment_scan.py`: +261.
- `qd_data` config, general, loaders, mixture and sources: small additive edits.
- New test file `test_benchmark_targets_v6.py`.

**v6 is opt-in; v5 stays byte-identical:**
- `DataConfig.with_v6_benchmark_targets()`: refuses MMLU test/dev and CLINC test rows, and counts each refusal.
- `containment_scan.py --target NAME=FILE` and `--v6-benchmark-targets`. Target sizes: mmlu-test 14,042, mmlu-dev 285, clinc-test 5,500.
- Jevjudge can now be a code-family target. It holds 25,548 rows and its sha256 matches the v5 pin.
- An optional `pairwise_length_balance` band in decisions configs, with `len_ratio` recorded per row and drop counts in the manifest.
- `val_without_train` reported in every decisions manifest.

**Tests [V]:**
- The lane ran 10/10 tests in its new file, and 335 passed, 9 skipped across the affected suites.
- This session re-ran `test_benchmark_targets_v6`, `test_decisions_pool` and `test_general_families` against the worktree: all passed.
- Rust tests and clippy: not run [U]. They are step 4 of the runner.

**Fable's rulings on lane 2's three questions:**
1. **MMLU stays a target, both dev and test.** `knowledge.multiple_choice` gets a *stated* zero train cap in the v6 cap config.
   - It leaves the paired-margin pool, and R1 is `not_run` for it, not a failure.
   - It is read for accuracy and abstention, as an untrained-family probe.
   - v6's Decision Index MMLU number will be honest, and may fall against v5's contaminated one.
2. **CLINC val and held-out shrink under v6: accepted.** This is decontamination defining v6's population before any v6 result, not a gate change.
   - v5's pins (including `v5_a7_check`) are never edited.
   - v6 counts are pinned in `campaign/` from the rebuild manifest before the first v6 ft row.
   - A val intent left with zero rows is reported, not patched.
3. **Length balance applies to val too**, refused below the existing `val_floor_per_family`. The lane is adding that floor and its break-it test.

**Integration prerequisites:**
- `tools/real_tokenizer_pipeline.py:1749` must pass `split_name=split_name` to `parse_clinc`. Until then v6 refuses every CLINC row (closed failure). This file is the dedupe lane's and main rewrote it, so the fix applies after the dedupe diff lands.
- `config.py` has one trivial conflict with the dedupe lane, in `fingerprint()`.
- `PINNED_REQUEST_SHA256` must be re-pinned at merge: it was pinned with the binary built outside the lock.
- `synth.rs` should call `val_strata_without_train` once this `decisions.rs` merges.

**Gaps recorded:**
- GAP-CLINC-TEST-TARGETS-8GRAM-BLIND-2026-10-06: 2,454 of 5,500 targets are under 8 words. The v6 fix holds by source-split refusal, not by containment.
- GAP-PIPELINE-PARSE-CLINC-NO-SPLIT-NAME-2026-10-06
- GAP-DEVMAP-ASK-EVIDENCE-MISSED-MMLU-PIN-2026-10-06

All three carry the date suffix 2026-10-06. Lane 2 also hit the ListAgents gap; it is covered by lane 4's record.

### Lane 3: natural-bug held-out miner

Worktree `agent-a44b2df31f3358a5f`. It was fast-forwarded to main `2cee2a8` before editing, so it is not stale. Uncommitted, +2,870 lines. Nothing has compiled or run [U]; the runner's last step runs it.

**Code:**
- `qd-prep own-repos`: the canonical enumerator, writing its manifest before any commit is read.
- `qd-prep natural-bugs`: the miner, with a required `--cutoff` and no default. Its output is refused unless the path has a held-out marker.
- New files: `gitcli.rs`, `own_repos.rs`, `natural_bugs.rs`, `tests/natural_bugs.rs`. There are 23 tests.
- `qd-lang` is added as a workspace path dependency (serde only).

**Preview of the enumeration** (throwaway Python over `.git/config` only, not the Rust output):
- 49 repos found, 27 admitted.
- 4 held out: `devtools/GitPulse`, `research/GenoThermalTargeting`, `research/StressProject`, `scholarlm/wisdev-arc`.
- Zero verdict changes against v5's rule.
- Flag: GitPulse is held out, so `gitpulse.commit_type` cannot train on GitPulse's own history.

**Cutoff [U]:** the Qwen3.5 card states no training cutoff; its release is February 2026 [V]. The run is a labelled yield probe at `--cutoff 2026-03-01`. That date is an upper bound, not the cutoff. Expect fewer than 100 fixes from 4 held-out repos, and report that rather than loosening the filter.

**Rules:**
- A fix-like commit is one whose message contains a bug/fix/fault-family word.
- A single-statement fix has exactly one line removed and one line added, in one file. Comment-only and blank-line changes are excluded.
- Every row has `mutation_class: null`: a natural bug has no class, so the span is the label.
- Rows carry a `pre_last_line_whitespace_only` flag.
- No file-mode span renderer exists, and none was built.

**Outputs:**
- `/Users/bharath/qd-campaign/v6-data-2026-10-06/own-repos/manifest-2026-10-06.json`
- `.../heldout/natural-bugs/probe-cutoff-2026-03-01/`

**Gaps recorded:**
- GAP-LANE3-MINER-RUNS-GIT-IN-OTHER-REPOS-WHERE-THE-HARNESS-BLOCKS-GIT-C-2026-10-06: read-only, and requested by the human, but a policy question surfaced to the human.
- GAP-OWN-REPO-ADMISSION-RULE-DUPLICATED-2026-10-06
- GAP-MAC-HEAVY-WAIT-EXCEEDS-BASH-BACKGROUND-CAP-2026-10-06

All three carry the date suffix 2026-10-06.

**Runner 2 result (2026-10-06 18:18Z) [V, `build/queue-2026-10-06/lane3-natural-bugs-2.log`]:**
- clippy clean; cargo test 117 lib + 3 + 4 integration, all ok.
- `own-repos`: 48 discovered, 27 admitted (4 held out, 23 train), 21 excluded. Manifest sha256 `9fbdd904…596854`.
- Yield probe at `--cutoff 2026-03-01`: 778 commits scanned, 306 fix-like after the cutoff, **0 single-statement fixes** (target 100-300: SHORT). Report `probe-cutoff-2026-03-01/report.json`, sha256 `3859b69c…725e`.

**Where the 306 go** (the miner's report; first failing check wins): `v5_own_prose_file` 146, `multi_file` 113, `multi_hunk` 20, `language_not_admitted` 12, `not_one_line_each` 10, `not_a_modification` 5. The v5 rule removes only Markdown, so no code fix was lost to it.

**The zero is real [V, two independent counts]:** lane 3's throwaway oracle (`git log --numstat`, same isolation) matches the miner's fix-like counts exactly (288 / 16 / 2 / 0) and finds 0 fix-like one-file +1/−1 commits after the cutoff in any file type, and 0 before it. 249 of the 306 carry an agent `Co-Authored-By` trailer.

**The 23 train repos, same oracle** (Fable asked for it; script `<lane 3 wt>/build/lane3/train_oracle.py` sha256 `059912c1…`, output sha256 `de53783f…`): 4,081 non-merge commits, 1,838 fix-like, **9** single-file +1/−1 fixes in an admitted language (4 before the cutoff, 5 after), an upper bound on what v1 would admit. 75.8% of fix-like commits carry an agent trailer (84.4% after the cutoff). 19 of the 23 repos have no commit before 2026-03-01. The v1 rule does not fit this corpus, so expanding the holdout would not have helped.

**Fable's ruling (2026-10-06):**
- The zero is a result. The own-repos natural-bug held-out set is **0 rows under v1: not run** wherever a gate expected it (rule 5).
- More held-out own repos: refused (changing the holdout after reading the probe; rule 2's shape). Statement-level v2: dead (about 2 rows).
- TSSB-3M becomes the natural-bug eval set, **held-out only**: moved under `heldout/` (addendum `fetch-record-v6-2026-10-06.addendum-tssb-heldout.json`; sha256 unchanged, `11ca0f8c…`), pre-registered in `campaign/natural-bug-tssb-heldout-preregistered.json` before any row value was read. Report-only; whether it binds v6 is a separate decision. The builder is not written. ManySStuBs4J stays reference-only (Java is not admitted).
- **v6 planning input, not acted on:** most fixes in the human's repos are large agent-authored multi-file commits. The defect families (single-statement mutations by construction) do not produce that shape. It sits beside decider-2b's +0.110 on test-g6 (`HANDOFF/merge-and-bench-2026-10-06.md` §6).

**Ledger:** the probe has no ledger row. The schema has no `run_kind` for a data probe (`build` is for suites; the run kinds need snapshot, tokenizer and backbone), and inventing one would be a fabricated entry. The numbers above cite the report and oracle outputs by sha256: GAP-LEDGER-HAS-NO-RUN-KIND-FOR-A-DATA-PROBE-2026-10-06.

**Read-only git outside Lappi:** the miner and both oracles ran read-only `git log`/`git show` with isolated config in the human's own repos. Fable reads the 2026-09-28 approval ("own repos/data usable for training+eval") as covering that.

### Lane 4: conversion of the downloaded sources (Fable's ruling)

Worktree `agent-a613aeaf4c545e7a4`. The code is written. Nothing has compiled or run [U].

- **One locked job is queued:** `bash tools/mac_heavy.sh convert-v6 bash <wt>/build/lane4/locked-convert.sh`.
- **It does, in order:** clippy, `cargo test -p qd-prep`, the release build, the JSONL views, then the pool builds. Logs go to `<wt>/build/lane4/{locked.log,status.tsv}`.

**Code:**
- Rust modules: `convert.rs`, `convert_licence.rs` (with a parity test against `qd_data.licences`), `convert_tools.rs`, `convert_text.rs`, `convert_code.rs`, `convert_inject.rs`.
- A `qd-prep convert` subcommand, and the config `data/convert/convert-v6-2026-10-06.json`.
- Beside the data: `view_v6.py` (pyarrow views, ML venv) and `csn_licences.py`.

**Families** (counts are the lane's previews; the locked run's manifests are the record):

| Family | Source | Rows (preview) | Licence |
|---|---|---|---|
| `when2call.tool_select` | When2Call train | 15,000 sft + 9,000 pref | cc-by-4.0; generator unnamed |
| `toolace.tool_select` | ToolACE | 11,300; multi-turn and no-tool-list rows refused and counted; irrelevance rows by construction for 15% (seeded) | apache-2.0; generator undisclosed |
| `mnli.nli/<genre>` | MNLI train | 77,348 fiction rows refused (licence); 315,354 candidates | `oanc` |
| `scirepeval.search_rel` | SciRepEval search | 35% of queries drawn at read; 1 positive and 1 negative per query | `odc-by-1.0`, attribution |
| `csn.func_match` | CodeSearchNet python/go | — | per repo: 16,052 repos looked up via `gh api graphql`, 13,138 allowlisted, 2,914 denied |

- The CSN licence cache is `csn-licences-2026-10-06.jsonl`, sha256 `a8d125c3…`.
- Prompt-injection sets become an injection-text corpus, `pool/convert-injections-v1/`, the future `--injections` input to `synth_email`. They are not a family of their own.

**Parked:**
- HelpSteer3: JSONL view only; it goes through lane 2's path.
- SWE-rebench: filtered view only. The preview keeps 8,945 of 27,878 rows; 11,491 need a licence call and 503 overlap SWE-bench, Lite or Loc-Bench.
- TSSB-3M: unzip only; its role needs a ruling under §4.7.
- ManySStuBs4J: reference-only (Java).
- BM25 hard negatives: proposed, not built.

**Licence ids pending registration in Python:** `odc-by-1.0`, `oanc`, and the SWE-rebench display names. Until they are registered, the loader refuses those rows, and that refusal is a closed failure.

**Gaps recorded (7):**
- GAP-DEVMAP-NO-STORE-IN-LANE4-WORKTREE-2026-10-06
- GAP-LISTAGENTS-UNAVAILABLE-TO-SUBAGENT-2026-10-06
- GAP-WHEN2CALL-TRAIN-NO-LABEL-FIELD-2026-10-06
- GAP-SCIREPEVAL-TEST-NFCORPUS-TRECCOVID-UNCHECKABLE-2026-10-06
- GAP-LICENCE-REGISTRATION-PENDING-ODCBY-OANC-SWEREBENCH-NAMES-2026-10-06
- GAP-CSN-LICENCE-DETECTED-TODAY-NOT-AT-COMMIT-2026-10-06
- GAP-CONTAINMENT-ONE-UNASSIGNED-CODEPOINT-REFUSES-WHOLE-REQUEST-2026-10-06

Each carries the date suffix 2026-10-06.

### The dedupe and lane 4 worktrees are on a stale base [V]

Lanes 2 and 3 fast-forwarded to main before editing, so they are not affected.

- **The finding:** agent worktrees branch from `origin/main` = `f62d7ca`. Local main is `2cee2a8`, 75 commits ahead and unpushed.
- **What those commits change:**
  - add `crates/qd-prep/src/{spancheck,team}.rs`;
  - change `linfit.rs`, `lib.rs`, `main.rs` and `wire.rs`;
  - change `tools/real_tokenizer_pipeline.py` by +954 lines.
- **Consequence:** a lane's green on its own base is not green on main.
- **Integration plan:** apply each lane's diff onto local main in a fresh worktree (`git diff | git apply -3`), then re-run clippy, `cargo test -p qd-prep` and the lane's pytest under the lock. The dedupe lane's merge is the riskiest, because it edits `real_tokenizer_pipeline.py`.
- **Recorded as:** GAP-AGENT-WORKTREES-BRANCH-FROM-STALE-ORIGIN-MAIN-2026-10-06. Pushing main is the human's call.

### One ordered lock runner (Fable's ruling)

**What it replaced:** six separate `mac_heavy.sh` waiters. They timed out at their limit, got requeued, and would have taken the lock in random order.

**Command:** `bash tools/mac_heavy.sh queue-2026-10-06 bash build/queue-2026-10-06/run-queue.sh`. It runs detached under `nohup`, with `MAC_HEAVY_MAX_WAIT_S=36000`, and logs to `build/queue-2026-10-06/runner.log`.

**Steps, in order:**
1. span-refit smoke;
2. `synth-v6`;
3. `dataclean-dedupe`;
4. lane 2: clippy, then `cargo test`;
5. lane 4: `convert-v6`;
6. lane 3: the natural-bug miner.

**How it runs:**
- Each step logs to `<step>.log` and appends to `status.tsv`.
- A failed step does not stop the steps after it.
- Lane 2's step runs in a subshell, so its `cd` and its exports do not leak into the later steps.
- Every lane script sets its own `CARGO_TARGET_DIR` and `--manifest-path`.

### Shared pool seam

- `crates/qd-prep/src/pool.rs` (new, in main): target reading, the containment scan, examples, and gold-position histograms.
- `synth` and `convert` both write through it. `decisions` keeps its own copy and is the only producer that applies caps: GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06.

### Synthetic generators: email sorter and Jarvis (§4)

Written by this session in main, in `crates/qd-prep`. Fable's ruling (this session) set the design.

**Code:**
- `src/synth.rs`: the shared assembly.
- `src/synth_email.rs`, `src/synth_jarvis.rs`: the generators.
- `qd-prep synth --config C [--target NAME=FILE] --out-dir DIR`: the CLI.

**Existing code changed:**
- `decisions.rs`: helpers made `pub(crate)`; `containment_request` now delegates to the new `containment_request_with`, with byte-identical output for `decisions`.
- `lib.rs`, `main.rs`: additive only.

**Configs:**
- `data/synth/email-v6-2026-10-06.json`
- `data/synth/jarvis-v6-2026-10-06.json`
- Jarvis's decontamination target, `/Users/bharath/qd-campaign/v6-data-2026-10-06/targets/jarvis-scenarios.jsonl` (sha256 `a562ba77…`): the 17 scenario, profile, policy and overlay files.

**Families:**
- `email.category`: the engine's 13 labels, with the engine's own instruction and criteria text, plus `noul`.
  - The `noul` rule: the email is clearly about the job search but nothing in it decides between job labels.
  - Label-flip injection rows (10% of rows) go in their own stratum; the gold stays the true label.
- `jarvis.target`: choice over a synthetic tree's enumerated elements.
  - The tree includes label drift and same-word decoys.
  - The gold is `noul` when the control is absent (12% of rows).
- `jarvis.field_fill`: a profile *key*, or "leave this field empty", or `noul`.
  - No value is ever written.
  - Secrets, payment fields and emergency-contact fields are left empty.
- `jarvis.step_risk`: what the policy requires before a step runs.
  - The options are classifications, never grants (rule 7).
  - The gold is `noul` when the target is off-screen.

**Guards against 0.1's mistakes, applied in `synth.rs`:**
- Template-disjoint val: at least 4 templates per class, 1 held out. A template on both sides is refused.
- Options are shuffled per row, and the gold-position histogram is recorded.
- Per-class context lengths are recorded.
- Exact dedupe keeps the val copy.
- Containment runs against the targets.
- A linear leak probe runs on each fixed-option family: char 3-5-grams, fitted on train templates, scored on val templates. A build is refused above the config's bound, 0.95.
- The planted-leak test shows the probe catches a label written into the context.

**Privacy (the human's "Synthetic only" ruling):**
- No mailbox, `mailFilters.xml` (a list of the human's correspondents), `classifier_vectors.json`, profile value or scenario input is read.
- Every address and URL is on a `.example` domain.
- Tests check: every address; that the name "bharath" never appears; that the scenario literals (`M-1001`, `M-1002`, `M-9999`, `125000`, `USD`) never appear.
- Deviation from §4's table: §4 said "synthetic rows from the `mailFilters.xml` rules". That file was not used, on privacy grounds.

**Not yet loadable [V by reading `python/qd_data/decisions.py:60,136` and `licences.py`]:**
- `load_decision_pool` refuses any family not registered in `qd_data.sources.DECISION_FAMILIES`.
- The licence id `synthetic-by-rule` is default-denied until `qd_data.licences` registers it, with the human's ruling as its note.
- Both registrations are in files lane 2 is editing, so they come after lane 2 merges. Until then, nothing can train on these pools by accident.

**Status.** Runner 1's synth step and the re-run that followed (clippy passed, 129 of 131 tests passed, including both leak-bound tests and the planted-leak probe) [V, `build/synth-v6-2026-10-06/locked.log`]. Two tests failed, and both are fixed but not yet re-run:
- the email injection test counted a label contained in a longer label's name (a test bug, not a generator bug);
- the tool-catalog privacy test caught `generated_by` holding a home path; the catalog now names `~/qd-campaign/...` and `~/Code`-relative sources.

The pools are built by runner 2's first step (`synth-2`). Pools land at `/Users/bharath/qd-campaign/v6-data-2026-10-06/pool/synth-{email,jarvis,tools}-v1/`.

### Synthetic generator: tool selection (§4)

**Code:** `crates/qd-prep/src/synth_tools.rs`, family `apps.tool_select`, row shape identical to lane 4's `when2call.tool_select` / `toolace.tool_select` (`convert_tools.rs`; its ASK/UNABLE/DIRECT/QUESTION strings are copied verbatim, to converge on one owner when the lanes merge).

**Inputs, pinned by sha256 in `data/synth/tools-v6-2026-10-06.json`** (paths relative to the config's directory, so the committed config names no home path):
- `data/synth/tool-catalog-v6-2026-10-06.json` (`3f642099…`): 10 apps, 186 tools under 178 names, every field extracted from source by `~/qd-campaign/v6-data-2026-10-06/extract_tool_catalog.py` (stdlib), 60 authored confusable groups.
- `data/synth/tool-phrasings-v6-2026-10-06.json` (`7667618b…`): 712 complete, 258 underspecified, 30 direct; authored, synthetic.
- The authoring agent's overlap measure: median 0.000, p90 0.20 (stemmed). The generator re-measures and refuses a median over 0.35.

**Labels by construction:** call (complete phrasing, tool offered with a same-group sibling), ask (underspecified, tool offered), unable (15%: the gold's whole confusable group removed), direct (general question). `noul` is a stated zero. 30% of rows mask tool names to `tool_k`.

**Two data facts the generator handles:**
- Four shared names (`devcouncil_renew_lease`, `_release_task`, `_verify_task`, `_get_gaps`) require `task_id` in devcouncil and take nothing in manvi. An underspecified phrasing makes an "ask" row only in a catalog where its missing parameter is required; elsewhere it is skipped and counted (`ask_phrasings_skipped_not_required_in_catalog`).
- `exclude_tools` in the config, each with its reason, refused if stale: `devcouncil_ask_question` (it *is* the ASK action, so "call" and "ask" would both be right) and `permission_prompt` (machine-only; no user asks for it).

**Decontamination target** `tool-heldout` (`5f8bf09b…`, 9,204 rows), built by `~/qd-campaign/v6-data-2026-10-06/build_tool_targets.py`: every BFCL v3 user turn (5,251), every When2Call test question (3,952), and DevType's `PaletteToolRoutingTests.swift` whole (the only real routing held-out set). BFCL ids repeat inside a file (`live_relevance_3-3-0` twice), so rows are keyed by line.

**Tests [U, queued in runner 2]:** shape, unable removes the group, call offers a sibling, ask/direct gold, masking, assembly, the overlap break-it pair, underspecified naming a required parameter, a one-member group refused, the per-catalog ask rule (fails against the pre-fix parser, which refused the shipped file), exclusions (stale refused), and the shipped config end to end.

## 9. Integration and hardening (2026-10-06, 18:50-20:00Z)

**Integration [V].** Branch `integrate-v6-data` (`22c38b6`) holds S (the synth lane), the four lane merges, F (the dedupe lane's fingerprint merge) and C′ (every pool producer states its `allocation`; the loader refuses an unallocated pool). It was built in object space; main's HEAD and working tree were not touched. Worktree: `build/integ-v6/wt`. The workspace's relative `../../../tessl` and `ojas` paths resolve beside `build/integ-v6/`, so `build/integ-v6/{tessl,ojas}` are symlinks to `~/Code/research/{tessl,ojas}`. Runner 4's first launch (18:50Z) died on that and is void; its logs are kept as `runner-4.void-tessl-path.*`.

**Runner 4 [V, `build/queue-2026-10-06/status-4.tsv`].** The tree was `22c38b6`, plus lane 4's held-out patch to `convert.rs`, plus `not_viewed` in the injections and swe-rebench manifests.

| step | exit | note |
|---|---|---|
| integ-clippy | 0 | qd-prep, `-D warnings` |
| integ-cargo-test | 0 | 197 lib tests, including lane 4's `an_entry_moved_to_held_out_is_loaded_unreachable_and_counted` |
| integ-release | 0 | |
| integ-torch-pytest | 1 | 4382 passed, 4 failed, 74 skipped, 69 errors. The 69 errors and 2 of the failures are the worktree environment, not code (no `<tree>/target/debug` fixture bins, the checkout named `wt`, no `.venv/bin/ruff`): **not run**, not passes. The other 2 failures are 18 gap ids that exist only in main's working-tree `gaps.jsonl`, all present there [V]; the merge carries it. GAP-INTEG-WORKTREE-PYTEST-NEEDS-MAIN-ENVIRONMENT-2026-10-06. |
| integ-red-clinc | 0 | the new CLINC split test fails on the pre-fix tree `49c08ce`, as it must |
| integ-pools | 1 | email refused, "0 scans: a decontamination against nothing is not a check". Jarvis and tools were **not run** (`set -e`, email first). The refusal left no `synth-email-v1` and no `.partial`, which is U2 observed. GAP-SYNTH-EMAIL-HAS-NO-HELD-OUT-TARGET-2026-10-06. |
| span chain | running | val extract finished all 7,238 rows at `--batch-tokens 4096` (522 s, max RSS 20.7 GiB); the gate exited 0. Results are read only as pre-registered. |

`write_pool` (the `not_viewed` list in the build-path manifest, with its test) was written after runner 4's cargo steps compiled the tree, so runner 5 re-checks it.

**Runner 5** (queued behind 4, `run-queue-5.sh`) runs on the same tree:
- qd-prep clippy, test and release;
- the jarvis and tools synth pools, one step each, gated only on the build;
- `view_v6.py`, which applies the TSSB addendum and never opens a held-out entry;
- the six conversions.

**Branch `stress-v6`** (worktree `build/integ-v6/stress`, off `22c38b6`; uncompiled until runner 6):
- `crates/qd-prep/src/heldout.rs`: the one owner of rule 3's path marker in qd-prep.
  - `check_held_out_path` moved here from `natural_bugs`.
  - `refuse_training_input` checks both the lexical and the canonical path; `--target` files are exempt by design.
  - Wired into `synth` (config, tools catalog and phrasings), `convert` (config, view record, licence cache, and `Views::train_rows`) and `decisions` (config, fetch record, every source, the decider files). GAP-QD-PREP-TRAINING-INPUTS-NOT-HELD-OUT-CHECKED-2026-10-06.
- `crates/qd-prep/src/files.rs`: `read_bounded` and `open_regular`.
  - The type is checked before open, since a FIFO would block, and again on the handle. The read stops one byte past the bound.
  - Replaces four size-then-read copies, which let a symlink to `/dev/zero` read without end, and four unbounded reads. `decisions::for_lines` opens through it. GAP-QD-PREP-SIZE-CHECK-THEN-UNBOUNDED-READ-2026-10-06.
- Synth held-out templates. Config key `heldout_templates_per_class` (absent = 0, which splits exactly as before; a test pins this). The held-out templates are drawn before val and dropped from the pool, counted as `refused.heldout_template`. Dedupe runs held-out, then val, then train, so a pool row duplicating a held-out row is the one dropped.
  - `qd-prep synth --emit-heldout --out-dir <marked dir>` writes `examples.jsonl` (decision rows, split `heldout`), `targets.jsonl` (`{"id","text"}`, the scanned text) and a `qd-synth-heldout/v1` manifest.
  - Email config: `heldout_templates_per_class: 1`.
- `convert::shuffled` converged onto `synth::Scope::shuffle`. Its own doc planned this at merge; a pinned permutation test shows no option moved.
- `qd_data.defect_class.sha256_file` is public and the one owner of the file digest. The exact clone in `tools/real_tokenizer_pipeline.py` is deleted, and `tools/containment_scan.py` imports the public helper. `ruff check` is clean.
- The pre-registered stress suite's tests, `crates/qd-prep/tests/stress_v6.rs` and `python/tests/test_stress_v6.py`, written by a subagent. They are compiled by runner 6 and run once, on main, after the merge.

**Runner 6** (`run-queue-6.sh`) runs on the stress tree:
- clippy, every test target built, and qd-prep's tests with `stress_v6` excluded;
- the debug fixture bins in `<tree>/target`;
- the Python suite without `test_stress_v6.py`;
- release, then `--emit-heldout` to `~/qd-campaign/v6-data-2026-10-06/heldout/synth-email-v1/`.

After it: pin `targets.jsonl`'s sha256 in the email config as `email-heldout`, and build the email pool with `--target email-heldout=…/targets.jsonl`. Read its exclusion rate; do not assume it.

**Second round on `stress-v6` (2026-10-06 ~20:00-20:40Z; uncompiled until runner 6).** Every item below came from the stress-suite subagent's reading or from Fable's review. Each one has a test.
- **The `wt` changes are in the tree:** lane 4's held-out patch, `write_pool`, and `not_viewed` in every convert manifest, applied as a patch. The stress tree is now the whole candidate for main.
- **`pool::examples` runs `structural_refusal` on every row and refuses a split other than train or val.** Before this, a gold out of range panicked (exit 101) and a held-out row would have been written as train.
- **`pool::decontaminate` refuses no target set, or an empty one, itself.** It no longer relies on its callers to.
- **One owner for "write whole or not at all".**
  - `files::write_new_dir`, `write_new_file`, `write_synced_new` and `partial_path` replace six copies: `decisions::write_out`, `convert::write_dir`, `containment::write_dir`, `own_repos::write_new`, main's `write_atomically`, and the writers in natural-bugs and emit-heldout.
  - A failure after the partial is created now removes it. A concurrent run's partial is never touched.
  - `PATH.partial` is appended, not `with_extension`, so `pool.v5` and `pool.v6` no longer share one partial. `containment` already knew this; the other five did not.
  - `decision-caps` wrote with a plain `fs::write`; it now writes atomically too.
- **`for_lines` returns `Lines`.**
  - It counts a torn last line, CRLF lines and blank lines in `LineFacts`. `decisions::record_input` writes the counts to the manifest's `inputs` as `line_facts/<key>`.
  - It refuses an escaped NUL with the line number.
  - `pool::read_targets` refuses a repeated id with both line numbers. `decisions`' own copy of `read_targets` (which accepted a target named twice) now delegates to it.
- **`synth::generate` refuses a config past `MAX_ROWS` before drafting the pool's rows.** It does this through a one-row-per-template probe.
- **`qd_data.decisions.load_decision_pool` refuses a pool under a held-out path marker,** as spelled and as resolved. The S10 tests were red before this and are green after, in a light pytest on the stress tree: 101 passed.
- **My `holding_no_template_out_splits_exactly_as_before` could not fail.** It compared heldout=0 with heldout=0. It is replaced by a test that recomputes the seeded order and pins held-out = order[0], val = order[1].
- **Pre-registration clarifications (S1, S4, S5, S11) are dated in `AUDIT/data-stress-suite-2026-10-06.md`, before the run.** The subagent's S11 test for `dedupe` and `own-repos` now asserts the ruled behaviour.
- **Gap:** GAP-STRESS-SUBAGENT-NAVIGATED-WITHOUT-DEVMAP-MCP-2026-10-06.
- **One subagent claim was wrong:** `tool-heldout.jsonl` does exist (2.5 MB, 13:01).

**Runners after this round.**
- Runner 5 was retired before it took the lock. Pools must come from the tree that is committed, and that is the stress tree. Its pool and conversion steps moved into runner 6.
- Runner 6 waited on a `stress6-ready` sentinel, which its launcher checks before asking for the lock. Its steps:
  - clippy, every test target built, and qd-prep's tests;
  - `stress6-red`: each fixed stress test is run alone on the pre-fix tree `22c38b6` (`build/integ-v6/red2`) and must fail there;
  - the fixture bins and the Python suite;
  - release, the email held-out emit, the jarvis and tools pools, `view_v6.py`, and the six conversions.
- Runner 4 finished at 20:14:43Z. The span train extract died at MPS out-of-memory at batch 781 of 1087 (ps RSS 24.2 GiB; MPS "other allocations" 39.84 of 48 GiB). That is a second observation for GAP-MAC-HEAVY-RSS-CAP-MAY-NOT-COUNT-METAL-2026-10-06.
- Runner 7 (`run-queue-7.sh`) resumes the span chain from the train extract at `--batch-tokens 2084`, the script's floor. It is queued after runner 6. The MPS watermark override is not used.

**Third round: runner 6 against the real views (2026-10-06 20:55-22:25Z).**

Runner 6 [V, `status-6.tsv`]:
- **Passed:**
  - clippy, the test build, and cargo test (230 passed);
  - the fixture bins and release;
  - the email held-out emit;
  - the jarvis pool (8,199 rows, 909 noul gold) and the tools pool (4,061);
  - views, `convert-tools` and `convert-mnli`.
- **Failed, each with a cause:**
  - **The red check (exit 2).** The suite built `synth::Config` from a struct literal naming a field that 22c38b6 lacks. The test now builds it through `Config::parse`, so one source compiles on both trees. Attempt 1's red evidence stands.
  - **Torch pytest: 152 failed, 25 errors.** The runner's fault: `QD_PREP_BIN` named the release binary one step before release built it. `run-queue-6.sh` keeps that order as evidence; 6b and 6c run pytest after release.
  - **Four conversions:**
    - scirepeval and injections: escaped NULs in source rows;
    - swe-rebench-filter: a SWE-bench **target** row of 117,859,127 bytes;
    - csn: a go train row of 9,945,948 bytes.

    The 8 MiB bound was already in 22c38b6. It had simply never met the real views (GAP-QD-PREP-LINE-BOUND-NEVER-EXERCISED-ON-REAL-VIEWS-2026-10-06).

Runner 6b [V]:
- **Clippy passed.**
- **The red check passed:** all 12 Rust items and Python S10 fail on 22c38b6.
- **Pytest: 4,464 passed, 4 failed, 61 skipped.** All four failures are the worktree's environment, not code (GAP-INTEG-WORKTREE-PYTEST-NEEDS-MAIN-ENVIRONMENT-2026-10-06):
  - two gap-ledger tests: the 18 cited ids are all in main's `gaps.jsonl`, but not in the stress tree's older copy;
  - the directory name: `stress` is not `Lappi-decision`;
  - `ruff` is not installed in the tree's `.venv`.
- **The email pool was refused:** "no row survived decontamination".

**S1 scoped by role (the advisor's ruling; the AUDIT clarification is dated ~21:30Z).** `decisions::for_lines` takes a `Role`:
- **`Source`** (rows that can become training examples): an escaped NUL, or a line past `MAX_LINE_BYTES`, drops **that row**. It is counted per cause, with its first 16 line numbers. The bytes are still hashed, and an oversized line is skipped without being buffered.
- **`Reference`** (targets, held-out id sets, the licence cache, decider files): never dropped. A NUL is accepted and counted. The bound is `MAX_REFERENCE_LINE_BYTES`, 256 MiB, and past it the file is refused.
- `line_facts/<key>` is now written for every input, a clean one too.
- `convert::read_view` is the source reader. `Views::target_rows` returns a `TargetView` that reads only as a reference, so a target cannot be handed to the dropping reader; the type system refuses it.
- **Tests:**
  - unit: both roles, the hash of dropped bytes, a line whose `\n` is the byte past the bound, a torn oversized last line, past `NOTED_LINES`;
  - S1 stress items rewritten: a target NUL is read and counted; a target past 8 MiB is read; a source NUL row and an oversized source row are dropped and counted. The counts are read through `{:?}`, so the tests build on 22c38b6.
- Gaps: GAP-LLMAIL-INJECT-NUL-PAYLOAD-ROWS-DROPPED-2026-10-06 (40 lines of an attack class, a known loss), and the DevMap worktree gap, updated. DevMap answered `for_lines`' callers from main's store; the stress tree's 13 others were found with rg, labelled as rg.

**The email held-out target text (my defect; the advisor's ruling).** `synth::held_out` wrote each target as `Candidate::text()`: question, context and every option.
- A v1 target was 2,167 bytes, of which about 300 were the email. The rest is identical in every row, so every candidate matched every target.
- **Fix:** a target's text is the row's `context`, the instance and never the shared scaffolding. The test runs `pool::decontaminate` with two controls: a candidate carrying a held-out email is excluded; the genuine candidates are not all excluded.
  - Red proof: the test's own run on the tree before the fix (`email-heldout-target-prefix.log`: "no row survived decontamination"). `held_out` does not exist at 22c38b6, so the red2 check cannot carry it.
  - After the fix: 218 of 218 lib tests passed (`email-heldout-target-postfix.log`).
- **v2 emitted:** `heldout/synth-email-v2`, 1,667 rows, targets `6cf99185…be1c`, pinned in the email config. Its `examples.jsonl` is v1's byte for byte (`8db76ea2…9733`), since only the target text changed. v1 moved to `heldout/superseded/synth-email-v1`.
- The exclusion rate is read against `campaign/synth-email-heldout-v2-exclusion-2026-10-06.preregistered.json`, written before the pool ran.
- GAP-SYNTH-HELDOUT-TARGET-TEXT-CARRIED-SHARED-SCAFFOLDING-2026-10-06.

**Not decontaminated yet: the jarvis and tools pools (GAP-SYNTH-POOLS-ZERO-EXCLUSION-NOT-POSITIVE-CONTROLLED-2026-10-06).** Both excluded 0. That is what a disjoint target set reports, and also what a scan that cannot hit reports. Checkable targets: 14 of 17 (jarvis) and 8,143 of 9,204 (tools). A positive control per real target set is not run.

**Runner 6c [V, `status-6.tsv`, 22:42:56-22:46:37Z; every step exit 0].**
- **Clippy** passed, warnings denied, on every target.
- **cargo test** passed.
- **The red check:** all 14 Rust items and Python S10 fail on 22c38b6. The three rewritten S1 items fail for the reasons they name:
  - the target NUL was read but not counted;
  - the target line past 8 MiB was refused;
  - the source file was refused whole for one oversized row.
- **Release** built. **The email pin check** passed: v2's examples equal v1's, and this binary's re-emit equals v2 on both files.
- **Every pool rebuilt.** Runner 6's pools are in `pool-runner6/`. Their manifests gained `line_facts` for every input, +61 lines and −0 across the four compared.
- **Clean pools reproduce byte for byte:** `convert-mnli`, `convert-tools`, `synth-jarvis` and `synth-tools`. That covers `examples.jsonl` and every `containment/` file. The policy change touched no clean path.
- **Rows dropped, counted in each manifest's `line_facts`:**
  - csn: 1 oversized row (go train line 249501);
  - llmail-inject: 39 NUL rows (36 in phase 1, 3 in phase 2);
  - scirepeval search train: 12 NUL rows over 7 shards.

  The SWE-rebench and SWE-bench views report `nul_rows=0`. An independent parse confirms it: their `\u0000` text matches (2, 4, 5 and 2 lines) are all a literal backslash in patch text, with 0 real NUL rows. swe-rebench-filter kept 8,945 of 27,878 rows, and the 117.9 MB target row was read.
- **The email pool:** 6,992 rows (475 noul gold). It excluded **0 of 6,992**; all 1,667 targets were checkable at n=8. That is the pre-registered band "0", read together with the scan's positive control (`synth::tests`, where the carried-email copy is excluded). The email held-out set stands as a test of unseen templates.
- **Not measured:** the RSS of the swe-rebench-filter step. The runner records wall time only, about 6 s.

**DevMap audit (data side) [V for the calls; second signals by rg, labelled].**
- `devmap_dead_symbols`: 71 rows, 2 at 0.9.
  - `FileRecord` is a false positive: it is constructed at `own_prose.rs:952` (GAP-DEVMAP-DEAD-SYMBOLS-MISSES-STRUCT-LITERAL-USE-2026-10-06).
  - `replay_texts_from_prompts` is on the training side; it is already named in `HANDOFF/caller-contract-2026-10-06.md:110` and is not deleted here.
  - The 0.4 rows are unresolved-namesake rows; none was deleted on that alone.
- `devmap_clones` at budget 100000: 216 of 216 groups, not truncated. At budget 8000 it showed 113 of 216; that was a capped sample, not the list.
  - Acted on: the exact `_sha256_file` clone, and `convert::shuffled` (a clone DevMap did not group).
  - Named, not touched: the other `sha256_file` copies in `tools/` and `campaign/` (other lanes' one-offs); the exact `hex` clone across `qd-prep/src/blake2b.rs` and qd-runtime/qd-train bins (other owners); the qd-mutate per-language query wrappers (alike in shape, different grammars).

## Open

- The stress-v6 merge into `integrate-v6-data` and then main (the advisor reviewed the plan at ~22:55Z).
- A positive control per real target set (jarvis-scenarios, tool-heldout), before those pools are called decontaminated.
- The stress suite's one run on main, reported tri-state.
- The v6 allocation over every pool (GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06, the rebuild lane).
- Sections 3-4 otherwise as before.
