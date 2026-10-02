# HANDOFF: L-rescore, the excluded re-score tool (2026-10-02)

This lane built and tested `qd-gate-report --exclude-rows` and the excluded row it writes, for
Fable's Q6 re-score (`AUDIT/idle-gpu-queue-2026-10-02/fable-j6a-replay.md`, Q6). F gold-trains
all 23,819 v4 MMLU/CSQA rows, because the 09-29 exclusion was never wired into a build. At least
50 of them, and likely 185, overlap F's own val. The re-score recomputes F's numbers with those
val rows (E_val) held out, and over E_val alone. It is report-only: no threshold, gate or
population moves (rule 2).

**F seed 0 is re-scored** (row `e663248b`, below). The run used L-replay's E_val at
`/Users/bharath/qd-campaign/replay-v4-2026-10-02/gold-side/e-val.txt`: sha256
`e933ea3010ba136f3a5cf6d007ac2642fddef43ef9d74c14ca4d9faebcecb45c`, 215 keys (183 MMLU, 32 CSQA,
all slot `answer`). Seeds 1 and 2 follow when their verdict files land on the box
(`GAP-RESCORE-EXCLUDED-NOT-YET-RUN-2026-10-02`).

## What was measured

Ledger `ledger/mac-rescore-excluded-2026-10-02.jsonl`, row
**`e663248b-0532-412f-addf-a48fa6cf86ba`** (chain verifies, 1 row):
- `quick`, report-only, tag `gate-report-excluded`;
- `code_commit` `026473290d5a18097095885432bc4d9a4364c1ec`, clean: this lane's HEAD, run from a
  clean detached worktree;
- binary sha256 `ec88665ff984b5bde930ce8e2b8a763a4e132071d815ca151d5806da31cd34f5` (release,
  built at that commit).

It re-reads F seed 0's eval row `f4feac15-db49-4159-bb9b-695866c855cc` from the files pulled
read-only from the box, each sha256-equal to the box's copy:
- `verdicts-s0.jsonl` `4452eb66…`;
- `suite-verdicts-s0.jsonl` `d93971cd…`;
- the box ledger `abb48449…`.

Checks:
- All 215 E_val keys name a verdict line (0 absent; seed 0 decoded 18223/18223), so the default
  refuse path ran, with no allow flag.
- The binary recomputed 65 of the eval row's own numbers equal, with none left uncompared.
- Every `full` value below is the eval row's own.

F seed 0, three views (count / n). All verified, read off row `e663248b`:

| Population | Metric | full | excluded (not in E_val) | only (E_val) |
| --- | --- | --- | --- | --- |
| knowledge.multiple_choice (MMLU) | top-1 | 905/1485 = 0.6094 | 798/1302 = 0.6129 | 107/183 = 0.5847 |
| | permutation agreement | 1168/1485 = 0.7865 | 1020/1302 = 0.7834 | 148/183 = 0.8087 |
| | in-distribution abstention | 317/1485 = 0.2135 | 282/1302 = 0.2166 | 35/183 = 0.1913 |
| | ECE choice.k4 | 0.0489 | 0.0525 | 0.0979 (183 rows) |
| | last option (k4) predicted sigma | -1.306 | -1.019 | -0.983 |
| commonsense.multiple_choice (CSQA) | top-1 | 914/1197 = 0.7636 | 887/1165 = 0.7614 | 27/32 = 0.8438 |
| | permutation agreement | 1053/1197 = 0.8797 | 1026/1165 = 0.8807 | 27/32 = 0.8438 |
| | in-distribution abstention | 144/1197 = 0.1203 | 139/1165 = 0.1193 | 5/32 = 0.1562 |
| | ECE choice.k5 | 0.0278 | 0.0278 | not_run (32 < 100) |
| | last option (k5) predicted sigma | +0.678 | +0.764 | -0.453 |
| pooled, every family | top-1 choice (`val_top1.choice`) | 9130/10985 = 0.8311 | 8996/10770 = 0.8353 | 134/215 = 0.6233 |
| | permutation agreement (the gate's population) | 10240/10985 = 0.9322 | 10065/10770 = 0.9345 | 175/215 = 0.8140 |
| | in-distribution abstention (the gate's half) | 843/10985 = 0.0767 | 803/10770 = 0.0746 | 40/215 = 0.1860 |
| | ECE choice.k4 (defect_class + MMLU) | 0.0270 (3789) | 0.0275 (3606) | 0.0979 (183) |

Reading. This is report-only; it moves nothing, and a quick row decides nothing (rule 8).

**Within a family, holding E_val out barely moves anything on seed 0, and not in one
direction.**
- MMLU top-1 rises 0.0035 and CSQA's falls 0.0022.
- MMLU permutation falls 0.0031 and CSQA's rises 0.0010.
- MMLU abstention rises 0.0031 and CSQA's falls 0.0010.

**The E_val rows themselves are not answered better than their family.** MMLU's E_val top-1 is
0.5847, against 0.6129 for the rest. Fable's direction ("contamination can only flatter", Q6)
was inferred and is not borne out for top-1 on this seed. Inferred, not tested: 183 rows give
the `only` share a binomial SE of about 0.036, so these within-family gaps are inside one SE.

**Pooled, holding E_val out RAISES the pooled gates' values.**
- Permutation goes 0.9322 to 0.9345, still under the 95% bar.
- In-distribution abstention goes 0.0767 to 0.0746, still over the 5% bar.

That is family composition, not memorisation (inferred): E_val sits in the two weakest families.
No failing gate passes either way.

## What changed

| Commit | What |
| --- | --- |
| `2561065` | The fail-first tests. `crates/qd-runtime/tests/gate_report_exclude.rs` has 12 tests over a deterministic SplitMix64 fixture. Its no-flag goldens, `crates/qd-runtime/tests/fixtures/gate_report_exclude/no_flag.{report.json,stdout.txt}`, were written by `qd-gate-report` at `74aea95` (debug, sha256 `ca6dc0aa3a1bcd0cb1603579f56149014d05f9ed7302e99f4fd74faec9686b10`). `python/tests/test_gate_report_row.py` gains 5 tests. The pre-change logs are in `AUDIT/rescore-excluded-2026-10-02/`. |
| `396e8fe` | `qd-gate-report --exclude-rows FILE [--allow-empty-exclude-rows] [--allow-absent-exclude-rows]`, and `tools/gate_report_row.py --exclude-rows`, which writes the `gate-report-excluded` row. |
| this commit | The gap records, this runbook and the post-change test logs. |

### The flag's semantics

- **FILE form (sent to the lead, accepted, relayed to L-replay).** One `row_id#slot_name` key
  per line, UTF-8, LF line ends. This is exactly the key `tools/replay_decontam.py:76`
  (`row_texts`) gives a val target. For MMLU/CSQA a key looks like
  `mmlu:knowledge.multiple_choice:<split>:<digest>:<i>#answer` or
  `csqa:commonsense.multiple_choice:<qid>#answer`, because the slot is `answer`
  (`python/qd_data/general.py:218,:265`).
- **Matching.** A key is matched byte for byte against each verdict line's
  `row_id + "#" + slot_name` and is never split on `#`.
- **Refused:**
  - a blank line;
  - whitespace around a key, a CR included;
  - a key with no `#`;
  - a duplicate key;
  - a FILE that names no row, unless `--allow-empty-exclude-rows` is given;
  - two verdict lines that join to one key.
- **Absent keys.** A key that names no line of a `--verdicts` file is refused for that file, and
  the refusal names it. Each eval row is reported separately.
  - `--allow-absent-exclude-rows` (the lead's change) records absent keys instead: per eval
    row, their count, the bytewise-sorted list, and the sha256 of that list, one key per line
    with LF line ends (what `LC_ALL=C sort` writes).
  - Even with that flag, a FILE none of whose keys match is refused.
  - The tool cannot tell an undecoded val row from a typo
    (`GAP-RESCORE-ABSENT-KEYS-UNDECIDABLE-FROM-VERDICTS-2026-10-02`). Step 6's pre-check against
    the val manifest makes that call.
- **Output.**
  - A top-level `exclude_rows` object: `path`, `sha256`, `keys` and the two flags.
  - Per eval row, an `excluded_rows` object with `keys_matched`, `absent{count,keys,sha256}`,
    `per_family{<affected family>}` and `pooled`, each holding `full`, `excluded` and `only`.
  - Each view carries `n`, top-1 per kind, `permutation_consistency`,
    `ood_abstain_in_distribution`, ECE and G3's slot diagnostics per slot shape (every shape,
    so CSQA's k5 too), and `last_option`. `last_option` is row `noul_row - 1`, with its
    predicted and mean-probability deviation in sigma. On `choice.k4` of
    `knowledge.multiple_choice` that is Fable's withdrawn "MMLU last-option bias".
  - The text report appends an `EXCLUDED ROWS` section per eval row.
- **One owner.** The views call the existing functions: `share`, `ece_of`/`ece_value`,
  `slot_stats` and `in_distribution`. A test asserts that a family's `full` view equals G1's and
  G3's own per-family values, and that the pooled `full` view equals G1's pooled values.
- **No flag, no change.** Without the flag, both reports are byte-identical to `74aea95`'s (the
  golden test).
- **The ledger row** (`tools/gate_report_row.py --exclude-rows`):
  - `run_kind` eval and `quick`, with `QUICK_REASON_EXCLUDED`.
  - Recipe tag `gate-report-excluded`. The recipe also carries `reported_eval_rows` (the eval
    row), `exclude_rows_sha256` and the two allow flags.
  - Metrics: `gate_report.cross_check`, `rescore.exclude_rows`, `rescore.absent` (with the key
    list), and `rescore.{pooled|family.<f>}.{full|excluded|only}.{n, accuracy.<kind>,
    permutation_consistency, ood_abstain.in_distribution, ece.<shape>,
    last_option.<shape>.{predicted,mean_probability}_sigma}`.
  - It is never tagged `epoch-score-val` and has no `ft_run_row_id`, the two things
    `qd-post-f-rules`' `eval_row_of` selects by (`crates/qd-runtime/src/bin/qd_post_f_rules.rs:872-878`).
  - Without the flag the row is the plain `gate-report` row, with the same recipe keys and so the
    same recipe hash.
- **Pooled views.** These go beyond "three views per affected family". The gates Fable says
  are flattered pool every family, and the ECE k4 shape pools defect_class with MMLU. A pooled
  ECE cannot be derived from per-family numbers.

### Tests (verified, logs in `AUDIT/rescore-excluded-2026-10-02/`)

| Suite | Before the change | After |
| --- | --- | --- |
| `cargo test -p qd-runtime --offline --test gate_report_exclude` | 2 passed (fixture shape; byte identity, the characterization), 10 failed on `unexpected argument '--exclude-rows'` (`failfirst-pre-change*.log`) | 12 passed |
| `cargo test -p qd-runtime --offline` (whole crate) | not separately logged | 23 suites, 311 passed, 0 failed (`qd-runtime-tests-post.log`) |
| `cargo clippy -p qd-runtime --all-targets --offline -- -D warnings` | n/a | clean (`clippy-post.log`) |
| `rustfmt --edition 2024 --check` on the two Rust files | n/a | clean |
| `python/tests/test_gate_report_row.py` | 6 passed, 4 failed on `unrecognized arguments: --exclude-rows` (run in a detached worktree at `74aea95` holding only the new test file, against the new binary; `failfirst-python-pre-change.log`) | 10 passed |
| `python/tests/test_gate_report_parity.py` (no flag) | n/a | 11 passed |
| `python/tests/test_gaps_ledger.py` | n/a | 10 passed |
| `python/tests/test_tool_call_sites.py` | the same 1 failure at `74aea95` (verified in a clean detached worktree) | 18 passed, 1 failed: `test_every_tool_that_hashes_a_recipe_is_found_by_this_check`, not this lane's (`GAP-TOOL-CALL-SITES-MISSES-QD-TRAIN-ORACLE-TRAINER-2026-10-02`; `python-post.log`) |

`devmap_affected_tests` (main checkout's index, generation 3029) named only the bin's unit tests
and `test_gate_report_row.py`. It missed `test_gate_report_parity.py` and the conftest
`gate_report_bin` fixture, which reach the binary by subprocess; rg found them
(`GAP-RESCORE-NAVIGATION-2026-10-02`).

## What is open

- `GAP-RESCORE-EXCLUDED-NOT-YET-RUN-2026-10-02`: the real run, below.
- `GAP-RESCORE-ONLY-VIEW-ECE-NEEDS-100-ROWS-2026-10-02`: ECE on E_val alone is `not_run` below
  100 rows of a shape. That is expected output, not a clean result.
- `GAP-RESCORE-ABSENT-KEYS-UNDECIDABLE-FROM-VERDICTS-2026-10-02`: resolved with residual; the
  residual is step 6's pre-check.
- `GAP-RESCORE-NAVIGATION-2026-10-02`: what DevMap, GitPulse and ListAgents could not answer.
- `GAP-TOOL-CALL-SITES-MISSES-QD-TRAIN-ORACLE-TRAINER-2026-10-02`: an unrelated failure on main,
  found in passing.

## Runbook: the real run

The rows are report-only, `quick` and on CPU at $0. Run them once E_val exists, for each F seed
whose eval row exists:
- seed 0 is `f4feac15-db49-4159-bb9b-695866c855cc`, from ft row `973cd4e3…`
  (`ledger/gh200-p4-v4-2026-10-01.jsonl`, verified);
- seeds 1 and 2 were due on the box at about 05 and 11 UTC.

Everything is written under `/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/`, outside
any checkout. The harness blocks `cd`, `git -C` and redirects into variable paths, so every
path below is literal. Replace `<MERGED>` with the main commit carrying `396e8fe`. Steps 3 to 7
are written for seed 0; repeat them with `s1` and `s2`.

### 0. Preconditions (STOP if any fails)

- This lane's commits are on main (`<MERGED>`).
- L-replay's E_val file exists, and its handoff pins its sha256.
- The seed's eval row (tag `epoch-score-val`) is in the box ledger.

### 1. Two worktrees at `<MERGED>`: one to build in, one kept clean to run from

**For seeds 1 and 2, skip steps 1 and 2.** Both worktrees already exist at `0264732` with the
binary built. Reusing them keeps all three seeds' rows on one `code_commit` and one
`bin_sha256`.

    git worktree add --detach /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/build-wt <MERGED>
    git worktree add --detach /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt <MERGED>
    ln -s /Users/bharath/Code/research/tessl /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/tessl
    ln -s /Users/bharath/Code/research/ojas /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/ojas

The two symlinks are needed because the workspace reaches tessl by a relative path,
`crates/qd-metal/Cargo.toml:38` `../../../tessl`, as `.claude/worktrees/` does with its own
`tessl` and `ojas` links. Without them the build fails to load the workspace (seen on the
seed-0 run).

Nothing is ever built or written inside `run-wt`. `code_commit` counts untracked files as dirty
(`python/qd_train/ledger.py:1112-1127`), and a cargo build re-resolves `Cargo.lock` (the tessl
path dependency).

### 2. Build the binary in `build-wt`

    cargo build --release --offline -p qd-runtime --bin qd-gate-report --manifest-path /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/build-wt/Cargo.toml --target-dir /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/target
    cargo test --offline -p qd-runtime --test gate_report_exclude --manifest-path /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/build-wt/Cargo.toml --target-dir /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/target
    shasum -a 256 /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/target/release/qd-gate-report

The test must print 12 passed. The row records the binary's sha256 as `recipe.bin_sha256`.

### 3. Pull F's files from the box, read-only

    mkdir -p /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box
    ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246 'ls -la /home/ubuntu/p4-v4/ && sha256sum /home/ubuntu/p4-v4/verdicts-s0.jsonl /home/ubuntu/p4-v4/suite-verdicts-s0.jsonl /home/ubuntu/ledger/gh200-p4-v4-2026-10-01.jsonl'
    scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246:/home/ubuntu/p4-v4/verdicts-s0.jsonl /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/
    scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246:/home/ubuntu/p4-v4/suite-verdicts-s0.jsonl /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/
    scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246:/home/ubuntu/ledger/gh200-p4-v4-2026-10-01.jsonl /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/
    shasum -a 256 /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/*

- **Hashes.** The local sha256s must equal the box's.
- **Paths.** The verdicts path is `$F_OUT/verdicts-s$s.jsonl` with
  `F_OUT=/home/ubuntu/p4-v4` (`campaign/post-f-queue/post_f_common.sh:33`, `box_q_j7p_cpu.sh:90`).
  The suite file name is inferred from `box_q_s34.sh:34`, because F's own `box_q_f.sh` is not
  in the repo. Use the name `ls` shows. With no suite file, drop `--suite-verdicts`: the
  cross-check then compares fewer numbers, and the row says how many.
- **Timing.** Pull the ledger after the seed's eval row is written. A torn last line is refused,
  never read.
- Pass only this box copy as `--eval-ledger`. The row must appear exactly once.

### 4. Find the seed's eval row and check its verdicts

    jq -c 'select(.run_kind=="eval" and .recipe.tag=="epoch-score-val") | {row_id, seed: .protocol.seed, ft: .metrics.ft_run_row_id.value, decoded: (.metrics.val_rows_decoded | {n, n_total})}' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/gh200-p4-v4-2026-10-01.jsonl
    head -1 /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl | jq -c '{eval_row_id, seed}'
    jq -r 'select(.kind != "span") | has("row_logits")' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl | sort | uniq -c
    jq -r '.gate' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/suite-verdicts-s0.jsonl | sort | uniq -c

The last command must print only `needle_hunk_recall` and `ood_abstain`. `qd-gate-report`
refuses any other suite gate (`read_suite`, "unknown suite gate"). If another value appears,
drop `--suite-verdicts` in steps 7 and 8 and record it in the lead's handoff.

Expected for seed 0:
- row `f4feac15-db49-4159-bb9b-695866c855cc`, ft `973cd4e3…`, `decoded` 18223/18223;
- the verdict file names that row and seed 0;
- every letter line has `row_logits`. F ran at `a502670`, which descends from `444bedc`, the
  commit that added them (verified by `git merge-base --is-ancestor`).

Any `false` is a STOP: the tool refuses such lines anyway.

### 5. Pin E_val and check its form

    cp <L-replay's E_val file> /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt
    shasum -a 256 /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt
    wc -l /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt
    rg -c '^(mmlu|csqa):' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt

The sha256 must equal L-replay's pin. The tool refuses a malformed file on its own.

### 6. Pre-check: are any E_val keys absent from this seed's verdicts?

    jq -r '"\(.row_id)#\(.slot_name)"' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl | LC_ALL=C sort -u > /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/keys-s0.txt
    LC_ALL=C sort -u /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt > /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.sorted.txt
    LC_ALL=C comm -23 /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.sorted.txt /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/keys-s0.txt > /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/absent-s0.txt
    wc -l /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/absent-s0.txt
    jq -r '.data_snapshot_hash' /Users/bharath/qd-campaign/phase4-v4-2026-10-01/data/pool/val.json
    jq -r '.entries[].row_id' /Users/bharath/qd-campaign/phase4-v4-2026-10-01/data/pool/val.json | LC_ALL=C sort -u > /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/val-row-ids.txt
    sed 's/#[^#]*$//' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/absent-s0.txt | LC_ALL=C sort -u | LC_ALL=C comm -23 - /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/val-row-ids.txt
    rg -v '#answer$' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/absent-s0.txt | rg '^(mmlu|csqa):'

The val manifest's `data_snapshot_hash` must be `69d45fd04fad01751b3f102f09041c0ba67176e44725dbeb977d550f4b0abbd4`
(read from that file; Fable Q4 names `69d45fd0…` as F's val). Then decide:
- **`absent-s0.txt` is empty:** run step 7 without `--allow-absent-exclude-rows`.
- **It is not empty, and all four hold:**
  - the last two commands print nothing, so every absent key is a v4 val row, with slot
    `answer` for MMLU/CSQA;
  - every absent key is undecoded;
  - the eval row's undecoded count (`decoded.n_total - decoded.n` from step 4) is not 0;
  - the absent count is at most that undecoded count.

  Then add `--allow-absent-exclude-rows`. Record the count and
  `shasum -a 256 …/absent-s0.txt` in the lead's handoff. That sha256 equals the report's
  `absent.sha256`, because both are the sorted keys, LF-terminated.
- **Anything else: STOP.** On seed 0, `decoded` is 18223/18223, so no val slot went undecoded,
  and any absent key is a key this val set does not decode: a typo, a held-out row or another
  build's row. Record a gap with the printed keys and return E_val to L-replay. Never edit E_val
  by hand and never pass the flag to get past it.

### 7. Write the row

    PYTHONDONTWRITEBYTECODE=1 /Users/bharath/.venvs/ml/bin/python /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt/tools/gate_report_row.py --bin /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/target/release/qd-gate-report --verdicts /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl --suite-verdicts /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/suite-verdicts-s0.jsonl --eval-ledger /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/gh200-p4-v4-2026-10-01.jsonl --ledger /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl --out-json /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/report-s0.json --exclude-rows /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt
    jq -c 'select(.recipe.tag == "gate-report-excluded") | {row_id, status, code_commit, quick, of: .recipe.reported_eval_rows, e_val: .recipe.exclude_rows_sha256, absent: .metrics["rescore.absent"].value}' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl

It prints `wrote row <id> to …`. Check the row:
- `code_commit` is `<MERGED>` with no `-dirty`;
- `quick` is true;
- `e_val` is step 5's sha256.

The decisions record defaults to `run-wt/docs/promotion-decisions.json`. A refusal writes a
`failed` row whose notes hold the binary's message. That row stays (append-only), and the
refusal is the finding: report it, do not retry around it.

### 8. Read it (report-only; every number cites the step-7 row id)

    /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/target/release/qd-gate-report --verdicts /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl --suite-verdicts /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/suite-verdicts-s0.jsonl --eval-ledger /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/gh200-p4-v4-2026-10-01.jsonl --decisions /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt/docs/promotion-decisions.json --exclude-rows /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt | sed -n '/EXCLUDED ROWS/,$p'

Carry full | excluded | only into the re-plan, each with its n.

Per family (`knowledge.multiple_choice`, `commonsense.multiple_choice`):
- top-1;
- permutation agreement;
- in-distribution abstention;
- ECE on `choice.k4` / `choice.k5`;
- the last option's predicted sigma.

Pooled:
- permutation agreement and in-distribution abstention, the populations of the two pooled gates
  in `docs/promotion-decisions.json`'s open questions;
- ECE on `choice.k4`, which pools defect_class with MMLU (n 3,789 on seed 0's full view);
- ECE on `choice.k5`.

Sanity check: seed 0's full view must reproduce the eval row's own numbers (all read from
`f4feac15` in `ledger/gh200-p4-v4-2026-10-01.jsonl`). The binary refuses a mismatch with the
eval row before it reports.
- knowledge permutation 1168/1485;
- commonsense permutation 1053/1197;
- knowledge in-distribution abstention 317/1485;
- pooled permutation 10240/10985.

ECE on `only` is `not_run` below 100 rows of a shape
(`GAP-RESCORE-ONLY-VIEW-ECE-NEEDS-100-ROWS-2026-10-02`). Report that as not measured, not as
calibrated. Under rule 8 a `quick` row decides nothing.

### 9. Land the ledger, then clean up

    PYTHONPATH=/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt/python /Users/bharath/.venvs/ml/bin/python -m qd_train.ledger verify --ledger /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl
    cp /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl /Users/bharath/Code/research/Lappi-decision/ledger/mac-rescore-excluded-2026-10-02.jsonl

Commit `ledger/mac-rescore-excluded-2026-10-02.jsonl` by explicit path.

For a later seed, append to the same outside ledger, `verify`, and copy again. The file only
grows: check that the committed copy is a byte prefix of the new one before replacing it, with
`cmp -n <committed size> <committed copy> <new copy>`.

When all seeds are in:

    git worktree remove --force /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/build-wt
    git worktree remove /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt

`--force` is needed for `build-wt` only, because its `Cargo.lock` re-resolved. `run-wt` must
come out without it. If it does not, something wrote inside it, and the rows' `code_commit` will
say `-dirty`.

## First command for the next lane

After this lane's commits are on main as `<MERGED>`, and once L-replay has pinned E_val:

    git worktree add --detach /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/build-wt <MERGED>
