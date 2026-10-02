# HANDOFF prep-containment (lane L-prep), 2026-10-02

Fable's 2026-10-02 ruling (`AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md`), ranked
items 3 and 4, with GAP-LINEAR-CONTROL-RERUN-WOULD-DOUBLE-THE-SELECTED-ROW-2026-10-02 and
GAP-E-VAL-SOURCES-ARE-MMLU-CSQA-ONLY-2026-10-02. Part A: the linear-control refit budget. Part B:
`qd-prep containment`, the v5 decontamination scan, and the `--exclude-identity-keys` hook (Q2
"The rule, end to end", items 1-3). No gate, threshold, seed rule or verdict logic moved (rule 2).
`python/qd_data` untouched. No dependency added. CLINC oos re-keying not touched.

Lane: worktree `agent-ae3fa4a2291e83af6`, branch `worktree-agent-ae3fa4a2291e83af6`. The branch
was created at `c65d7da`; it was fast-forwarded to `dec48d8` and L-replay's `e571065` was merged
(`d554702`) before any change (GAP-L-PREP-NAVIGATION-2026-10-02).

Labels: **[V]** produced by a command this lane ran, or read at file:line; **[I]** inferred;
**[E]** an estimate, extrapolated from the runs named beside it; **[U]** not verified.

## What was measured

**On the box or a GPU: nothing. Ledger rows: none.** Nothing was written to
`ledger/gh200-p4-v4-2026-10-01.jsonl` or to `ledger/mac-linear-control-refit-2026-10-02.jsonl`
(see Part A for why). Every number below is a Mac CPU run; the scratch scripts and their outputs
are under this session's scratchpad (`parta/`, `partb/`), and each number is labelled.

### Part A: the intent.domain control never converged because Adam cycled, not because 6,000 ran out

- **The curve** [V, scratch `parta/proto.py`, a dense copy of `_train_once` instrumented with the
  grad norm, on 300 of the control's own train documents, l2=1e-4]: grad norm 7.49e-2 at
  iteration 1, 1.83e-3 at 100, 4.34e-4 at 500, **1.60e-4 at 1,000**, then 9.95e-3 at 1,500,
  1.34e-2 at 2,000, 1.13e-2 at 4,000, 7.46e-3 at 6,000; over iterations 1,001-6,000 it ranges
  1.04e-4 to 3.55e-2 while the loss stays within 1.77266-1.77663. On 2,000 documents the same
  shape (2.38e-3 to 3.41e-2 over 1,001-6,000). On the full task (21,142 train documents, F seed
  0's, rebuilt from v4's manifests and verified by row_id and content_hash) the pre-fix native
  refit's loss is 2.11435678 at 1,000 and 2.11533914 at 6,000 (higher), UNCONVERGED, grad norm
  1.746e-2 [V, scratch `parta/fit6000.json`, qd-prep at a502670, 633.7 s]. A limit cycle: the
  fit reaches the tolerance's neighbourhood by iteration 1,000 and the constant step throws it out
  again, so no budget at that step converges.
- **Pre-fix engines at the new budget** [V, scratch `parta/prefix_check.py` ->
  `parta/prefix_8000.json`, on the committed fixture]: Python (`baseline.py` at HEAD `d554702`,
  no schedule) UNCONVERGED at 8,000, grad norm 1.487e-2; qd-prep `a502670` UNCONVERGED at 8,000,
  grad norm 1.351e-2. Raising the budget alone fixes nothing.
- **The fix** (both engines, identical): the step is `lr` for iterations 1-6,000 -- every fit
  that converged within the old budget is unchanged bit for bit -- and halves every 500 after
  (`qd_train.baseline.step_size`, `LR_CONSTANT_ITERS`, `LR_HALVING_PERIOD`;
  `crates/qd-prep/src/linfit.rs` `step_size`). `0.5**n` is a power of two, so `lr * 0.5**n` is
  exact in both. The budget is 8,000 in `tools/ft_linear_control.py`, `tools/rung0_linear_control.py`
  and `tools/rung0_real_run.py` (still pinned equal by their tests); the tolerance did not move.
  The diagnostic tools `after_vs_diff.py` and `operator_holdout.py` keep their own 6,000.
- **After** [V, scratch `parta/fit_sched20000.json`, this branch's qd-prep, max_iter 20,000 so the
  cap did not bind]: the full task's grid l2=1e-4 converges in 6,089 iterations (val carve
  2,866/4,228), the refit in **6,093** (grad norm 9.52e-5); l2 1e-3/1e-2/1e-1 converge in
  294/167/210 as before (the schedule never reached them); 421.1 s. The converged control's own
  val top-1 is 554/1,500 (unconverged it was 618/1,500). **No ledger row**: this is the one task's
  control, not F seed 0's control run (see the first command below).
- **Mac vs box**: the box row c89b89a1 recorded grad norm 1.118e-2 where the Mac's pre-fix refit
  ended at 1.746e-2 -- the cycle amplifies platform libm differences [I]. Under the schedule both
  converge; whether the box's converged fit is bit-identical to the Mac's is [U].

### Part B: `qd-prep containment`

- **Parity with the oracle** [V]: `python/tests/test_qd_prep_containment_parity.py` builds the
  request from `crates/qd-prep/tests/fixtures/containment-parity.jsonl` (100 synthetic rows,
  Unicode torture vocabulary, containment exactly 0.5, rows too short on every side, a source
  hitting three target sets) and compares `pairs.tsv` with `decontaminate`'s complete pair list:
  **byte-identical** under Python 3.14.7/Unicode 16.0.0 (8 passed) and 3.13.14/Unicode 15.1.0
  (7 passed, the table check skipped with its reason). On real data, the requests of the 1%, 2%
  and 5% subsamples below: byte-identical in every round of the A/B (48, 135 and 762 pairs).
- **What it scans** (`tools/containment_scan.py`): sources = every train row of every family before
  the replay draw (`real_ft_run.ft_split_report`), one text per slot; enforced targets = `val` and
  the repo-disjoint `heldout` rows of trainable families; unenforced = one set per task-holdout
  family (`heldout-family:code.language_id`, `heldout-family:qa.answerability`), reported per
  (source family, target); report-only = `val` against every held-out set. Keys (i) and (iii) are
  the splitter's `identity_disjoint`, `near_duplicate_disjoint` and `repo_disjoint` tri-states,
  carried into the attestation; it is CLEAN only when all ran and passed, every enforced target
  and the source have an indexed row, and no enforced pair survives the exclusion.
- **`remaining_hits`** is re-counted from the pair list after excluding by identity key, so it is
  0 by construction today; it guards a future change of exclusion granularity. The checks that are
  not tautological: `apply_exclusions` refuses a key naming no train row, and
  `test_the_scan_and_the_hook_take_out_exactly_the_contaminated_train_rows` re-scans the
  decontaminated split and finds no enforced pair.
- **The hook** (`python/qd_train/exclusions.py`, one function, `apply_exclusions`): called once
  after `split` and before `split_off_replay` by `tools/real_tokenizer_pipeline.py
  --exclude-identity-keys` and by `tools/real_ft_run.py`'s `ft_splits` (so `ft_split_rows` and
  `tools/ft_linear_control.py`, which both rebuild through it, take the same flag). Refused: a list
  that is not byte-sorted unique LF lines, an attestation that is missing, not v2, not qd-prep's,
  not n=8/threshold=0.5, not CLEAN, for another corpus (the pipeline's corpus identity plus the
  `qd_data` fingerprint), or for another file; a key naming no train row; the flag together with
  `--replay-exclude`. The train and replay shard headers carry `exclusions_sha256` (in
  `shard_hash` only when set); the pipeline recipe, `real_ft_run`'s recipe pieces and the control
  row's recipe carry it only when used. `real_ft_run` refuses a rebuild whose flag disagrees with
  the train header, in both directions; `ft_linear_control` refuses one that disagrees with the
  eval row. `replay_decontam` needs no flag: its targets are val and held-out, which no list moves.
- **Byte-identity without the flag** [V]: `test_a_build_without_the_flag_writes_what_it_wrote_before`
  (pinned at `d554702` before any edit) passes on the final tree, as does the header pin.
- **Rule 2 with the flag** [V]: `test_a_build_with_the_list_drops_those_train_rows_and_not_one_val_byte`:
  a 60-pair build with a two-key list drops exactly those keys' 4 train rows; every val shard file
  is byte-identical to the build without the list; the val manifest differs only in its record of
  the train count (86 -> 82); the trainer's `ft_splits` rebuild with the list holds exactly the
  set's train rows.

### Benchmarks

`python/tests/test_qd_prep_containment_parity.py::test_benchmark_containment_against_the_oracle_interleaved_min_of_n`
(`QD_PREP_BENCH=1`; oracle then binary, alternating, min of 5, pair lists compared every round;
the binary's arm includes writing the request), Mac, 18 cores [V]:

| fixture tiled | rows | text | pairs | oracle min | qd-prep min | speedup |
|---|---|---|---|---|---|---|
| x1 | 100 | 11.9 KB | 41 | 0.0020 s | 0.0170 s | 0.1x |
| x20 | 2,000 | 330 KB | 763 | 0.0381 s | 0.0288 s | 1.3x |
| x100 | 10,000 | 1.74 MB | 3,803 | 0.1917 s | 0.0652 s | 2.9x |
| x500 | 50,000 | 9.57 MB | 19,003 | 1.109 s | 0.3265 s | 3.4x |

Real data, the subsample requests (scratch `partb/scale_ab.py`, min of 3, binary timed alone) [V]:
1% 1.369 s vs 0.275 s (5.0x), RSS 83 MB vs 31 MB; 2% 3.925 s vs 0.356 s (11.0x), 122 MB vs 62 MB;
5% 8.957 s vs 0.711 s (12.6x), 225 MB vs 142 MB.

### Full-corpus cost, from 1%, 2% and 5% subsamples (SUBSAMPLES, not the full corpus)

Every v4 input was sampled by content hash (scratch `partb/make_subsample.py`: the eleven general
caches, the 25,000 composed rows, the 5,004 noul rows; the composed corpus's base through the
pipeline's own `--defect-max-rows` 500/999/2,498), then `tools/containment_scan.py` ran end to end
under `/usr/bin/time -l` [V]:

| sample | split rows | slot texts | request | split | render | scan (tool) | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 1% | 3,385 | 4,180 | 10.65 MB | 4.5 s | 0.4 s | 0.3 s | 6.5 s | 776 MB |
| 2% | 6,945 | 8,540 | 21.09 MB | 5.7 s | 0.8 s | 0.4 s | 8.4 s | 880 MB |
| 5% | 17,105 | 21,115 | 53.30 MB | 9.2 s | 2.1 s | 0.7 s | 13.4 s | 1,309 MB |

v4's split holds 342,971 rows (the L-replay build log), so the samples are 0.99%, 2.02% and
4.99%. Linear in rows from these three [E]: request about 1.07 GB (about 423k slot texts); split
about 2 min, render about 45 s; qd-prep alone about 11 s and 2.8 GB (its RSS was 2.66x the request
at every size); the exporter's peak about **14 GB**. The pair count grew faster than linear (48,
135, 762) and so does the posting-list work, so the scan time is a floor [E]. Total about 3-4
minutes. The exporter drops its rows before the binary starts; whether the OS takes the memory
back is [U], so plan for both at once (about 17 GB). The oracle on the full corpus, by the same
fits: about 5 minutes and 4.5 GB [E].

### What the subsamples found (read before applying a full list)

At 5% the scan excluded 293 identity keys: CLINC 210 of 1,044 utterance keys (20.1%; 8.5% at 1%,
9.1% at 2%), MMLU 41 of 692, code.defect_class 42 of 3,744, CSQA 0, SQuAD answer_span 0 [V,
scratch `partb/family_rates.py`]. The CLINC hits are mostly the family's constant question: the
definition keeps the question in `prompt_content`, and every intent.* row carries the same 11-word
one, so a train utterance that starts like a short val utterance shares half its 8-grams
(GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02, with the example pairs). A CLINC
key is shared by all four intent families, so one hit drops the utterance from all four. The
rate grows with the sample; at full scale it is likely much higher [I]. The definition, n and
threshold were not changed (the brief and the ruling fix them): the human decides before the
full list is applied.

## What changed (commits on `worktree-agent-ae3fa4a2291e83af6`)

- `d554702` merge of main `dec48d8` and L-replay's `e571065` (before any edit of this lane's).
- `81f6c33` Part A: `step_size` in `qd_train/baseline.py` and `crates/qd-prep/src/linfit.rs`;
  budget 8,000 in `ft_linear_control`, `rung0_linear_control`, `rung0_real_run`;
  `ft_linear_control --note`; fixture `crates/qd-prep/tests/fixtures/linfit-intent-domain-300.jsonl`
  (300 of the control's own documents, CLINC150 CC BY 3.0, attribution in its header line);
  `python/tests/test_linfit_step_schedule.py`; the halving parity test.
- `42ec232` Part B: `crates/qd-prep/src/{containment,pyunicode,pyunicode_tables,sha256}.rs`,
  `qd-prep containment`; `tools/qd_prep_unicode_tables.py`; `tools/containment_scan.py`;
  `python/qd_train/exclusions.py`; the hooks in `real_tokenizer_pipeline.py`, `real_ft_run.py`
  (`ft_split_report` split out of `ft_splits`; `check_exclusion_source`; the recipe piece),
  `ft_linear_control.py`, `artifacts.py`, `shards.py`; fixture `containment-parity.jsonl`;
  `test_qd_prep_containment_parity.py` (with the bench), `test_containment_exclusions.py`, and
  three test fakes given the header's new field.
- This handoff and the GAP records (`gaps.jsonl`, appended through `qd_train.gaps.append_gap`).
- `Cargo.lock` was rewritten by the builds (+46 lines, ojas path crates); not committed.

## Tests, with exit codes

- `cargo test --release -p qd-prep -j 4`: 54 unit + 2 integration passed, exit 0. `cargo clippy
  --release -p qd-prep --all-targets -- -D warnings`: clean. `cargo fmt -p qd-prep --check`: clean.
  The release binary these numbers used: sha256 `9ad514e82ec91b83c7a9d8a19b95280cb3f6cb0312e2d455dc09d58b790d5d6f`
  (the fixture benchmark ran on the build before a `clippy`-driven rewrite of one loop in
  `sha256.rs` and `rustfmt`; the SHA-256 vectors pass on both).
- The affected Python suites **on the committed tree** (`6c94a38`; the code is `42ec232`'s),
  `/Users/bharath/.venvs/ml` (3.14.7, torch 2.12.1, Unicode 16.0.0) via `uv run --no-project
  --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis --with datasketch
  python -m pytest`, 50 files (every `test_real_ft_*`, the control, `rung0`, baseline, replay,
  shard, pipeline, minhash, LSH, dedupe and gap files, and this lane's), 1,109 collected:
  **1079 passed, 1 failed, 29 skipped, exit 1** (log `target/lprep-stash/head-pytest.log`, ignored;
  counts read from its progress lines and checked against `--collect-only`, because the repo's
  `addopts -q` plus `-q` suppressed the summary line). The one failure is
  `test_gaps_writer::test_the_real_ledger_is_not_touched_by_any_of_this`, which asserts the
  ledger's directory is named `Lappi-decision` or `qwen-decision` and so fails in any worktree
  (pre-existing; below). The skips are env-gated benchmarks, `test_defect_noul` data not on this
  host, and opt-ins. An earlier 42-file run on the pre-lint tree (1063 passed, 27 skipped, exit 0)
  is superseded by this one.
- `.venv` (3.13.14, Unicode 15.1.0): `test_qd_prep_containment_parity` 7 passed, 1 skipped (the
  table check, with its reason); `test_gaps_ledger` 10 passed, exit 0.
- Fail-first: Part A's tests ran against the old budget and failed (3 failed, 1 passed; the
  native and reference fits unconverged at 6,000, grad norm 1.378e-2); the pre-fix engines at
  8,000 stay unconverged (above). Part B's are new behaviour; the two characterization pins were
  written at `d554702` before any edit and pass unchanged.
- Repo guards (`.venv`): `test_gaps_writer`, `test_ledger_provenance`, `test_lint_gate`,
  `test_provenance_on_every_exit`, `test_tool_call_sites`, `test_wire_gap_pins`,
  `test_gaps_ledger`: 80 passed, 3 failed, 5 skipped, exit 1. The 3 are the ones
  HANDOFF/rungd-flags-2026-10-02.md records as pre-existing in any worktree: the ledger's
  directory name, no `.venv/bin/ruff` in a worktree, and `qd_train_oracle_trainer.py` unpinned in
  `test_tool_call_sites`. Ruff 0.16.8 from the main checkout's `.venv` over every changed Python
  file: clean.

## What is open

- GAP-LINEAR-CONTROL-RERUN-WOULD-DOUBLE-THE-SELECTED-ROW-2026-10-02 (resolved-with-residual):
  F seed 0's full control not re-run; the command is below.
- GAP-E-VAL-SOURCES-ARE-MMLU-CSQA-ONLY-2026-10-02 (open): the full v5 scan has not run.
- GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02 (open, human).
- GAP-CONTAINMENT-SOURCES-RENDERED-AT-SEED-NONE-2026-10-02 (open): sources are rendered at
  `seed=None` as the oracle and `replay_decontam` render, not at the shard seed.
- GAP-CONTAINMENT-UNICODE-TABLES-PINNED-16-0-2026-10-02 (resolved-with-residual).
- GAP-ONE-RULE-EXCLUSION-NOT-WIRED-INTO-ANY-BUILD-2026-10-02 (open): wired, no build used it yet.
- GAP-L-PREP-NAVIGATION-2026-10-02 (open): DevMap degraded, GitPulse untrusted, no ListAgents.
- Not done: CLINC oos re-keying (out of scope by the brief).

## The exact first commands

Both need this branch's qd-prep (`cargo build --release -p qd-prep -j 4` in a checkout with these
commits; on this Mac the build is `target/release/qd-prep`, sha256 `9ad514e8...`), the v4 inputs
under `data/pool` (in a worktree, link `commitpackft-composed-v1/examples.jsonl`,
`commitpackft-corpus-v3/examples.jsonl` and `defect-noul-v3b/examples.jsonl` as the l-replay
worktree does), and the memory gate first: both are full-corpus runs over 2 GB, so they wait for
the lead's go.

**The full v5 scan** (from the repository root; every corpus flag must equal the v5 pipeline
build's, or `--exclude-identity-keys` will refuse the list as another corpus's):

```
QD_PREP_BIN=$PWD/target/release/qd-prep HF_HUB_OFFLINE=1 /Users/bharath/.venvs/ml/bin/python -u \
  tools/containment_scan.py --out-dir /Users/bharath/qd-campaign/v5-containment-2026-10-02 \
  --rev 881ab304f15ea13529002391dda8520c2ea47af4 --no-repo-history \
  --defect-class data/pool/commitpackft-composed-v1 --defect-download data/pool/commitpackft \
  --defect-noul data/pool/defect-noul-v3b \
  --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
  --general-max-rows 200000
```

It exits 0 only when the attestation is CLEAN. The v5 build then takes
`--exclude-identity-keys /Users/bharath/qd-campaign/v5-containment-2026-10-02/exclusions.txt`
with the same corpus flags, and the trainer's run the same flag.

**F seed 0's linear control under the schedule** (report-only; never into F's ledger):

```
QD_PREP_BIN=$PWD/target/release/qd-prep HF_HUB_OFFLINE=1 /Users/bharath/.venvs/ml/bin/python -u \
  tools/ft_linear_control.py --ledger ledger/gh200-p4-v4-2026-10-01.jsonl \
  --eval-row f4feac15-db49-4159-bb9b-695866c855cc \
  --verdicts /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl \
  --write-ledger ledger/mac-linear-control-refit-2026-10-02.jsonl \
  --note "report-only re-run under the step schedule (L-prep, 2026-10-02); which row a gate reads is the human's call" \
  --no-repo-history --rev 881ab304f15ea13529002391dda8520c2ea47af4 \
  --defect-class data/pool/commitpackft-composed-v1 --defect-download data/pool/commitpackft \
  --defect-noul data/pool/defect-noul-v3b \
  --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
  --general-max-rows 200000
```

The verdicts file is the box's seed-0 copy already on this Mac (18,223 lines, eval row
f4feac15); the fetch record's sha256 is the one c89b89a1 records (a0841f0d...). The row's
`max_iter` is 8,000, so it can never hash as c89b89a1.
