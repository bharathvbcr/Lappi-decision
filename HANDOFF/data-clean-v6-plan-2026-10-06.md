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

## Open

All of sections 3-4. The first command is in §5.
