# HANDOFF: L-rescore, the excluded re-score tool (2026-10-02)

This lane built and tested `qd-gate-report --exclude-rows` and the excluded row it writes, for
Fable's Q6 re-score (`AUDIT/idle-gpu-queue-2026-10-02/fable-j6a-replay.md`, Q6). F gold-trains
all 23,819 v4 MMLU/CSQA rows, because the 09-29 exclusion was never wired into a build. At least
50 of them, and likely 185, overlap F's own val. The re-score recomputes F's numbers with those
val rows (E_val) held out, and over E_val alone. It is report-only: no threshold, gate or
population moves (rule 2).

**F seeds 0, 1 and 2 are re-scored** (rows `e663248b`, `9e521e77` and `10382ac2`, below). All
three runs used L-replay's E_val at
`/Users/bharath/qd-campaign/replay-v4-2026-10-02/gold-side/e-val.txt`, which is also committed as
`AUDIT/replay-v4-build-2026-10-02/gold-side/e-val.txt`: sha256
`e933ea3010ba136f3a5cf6d007ac2642fddef43ef9d74c14ca4d9faebcecb45c`, 215 keys (183 MMLU, 32 CSQA,
all slot `answer`). The three-seed picture is under "Three seeds", below.

## What was measured

All three rows are in ledger `ledger/mac-rescore-excluded-2026-10-02.jsonl`, whose chain
verifies with 3 rows.

### Seed 0

Row **`e663248b-0532-412f-addf-a48fa6cf86ba`**:
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

### Seed 1

Row **`9e521e77-8e50-4fb8-ab37-92770a198f2c`**:
- `quick`, report-only, tag `gate-report-excluded`;
- chained to `e663248b`;
- the same `code_commit` (`0264732`, clean), binary sha256 (`ec88665f…`) and E_val sha256 as
  seed 0.

It re-reads F seed 1's eval row `aeca8d69-4733-4593-92de-2073021ed684` (ft row `95fa4854…`) from
files pulled read-only from the box at 05:28 UTC (local mtimes). Each is sha256-equal to the box's
copy:
- `verdicts-s1.jsonl` `04ea658a…` (8,269,462 bytes);
- `suite-verdicts-s1.jsonl` `94a76fe3…`;
- the box ledger `00bf2152…` (7 lines).

The box ledger only grew: its first 65,082 bytes hash to seed 0's pull, `abb48449…`. That pull is
kept as `box/gh200-p4-v4-2026-10-01.s0-pull.jsonl`.

Checks:
- All 18223 verdict lines name `aeca8d69` and seed 1, and all 10985 letter lines carry
  `row_logits`. The suite file holds only `needle_hunk_recall` (300) and `ood_abstain` (180).
- Step 6 printed val hash `69d45fd0…`, verdict keys 18223, E_val keys 215 and absent 0, so the
  default refuse path ran, with no allow flag.
- The binary recomputed 65 of the eval row's own numbers equal, with none left uncompared.
- The `full` values were also checked by hand against `aeca8d69`. Pooled permutation 10211/10985
  is the gate's value, which fails. Knowledge permutation is 1146/1485 and in-distribution
  abstention 339/1485; commonsense is 1070/1197 and 127/1197. `val_top1.choice` is 9124/10985.

F seed 1, three views (count / n). All verified, read off row `9e521e77` by a script, not by hand:

| Population | Metric | full | excluded (not in E_val) | only (E_val) |
| --- | --- | --- | --- | --- |
| knowledge.multiple_choice (MMLU) | top-1 | 925/1485 = 0.6229 | 824/1302 = 0.6329 | 101/183 = 0.5519 |
| | permutation agreement | 1146/1485 = 0.7717 | 1006/1302 = 0.7727 | 140/183 = 0.7650 |
| | in-distribution abstention | 339/1485 = 0.2283 | 296/1302 = 0.2273 | 43/183 = 0.2350 |
| | ECE choice.k4 | 0.0327 | 0.0326 | 0.0712 (183 rows) |
| | last option (k4) predicted sigma | -2.553 | -1.848 | -2.295 |
| commonsense.multiple_choice (CSQA) | top-1 | 921/1197 = 0.7694 | 895/1165 = 0.7682 | 26/32 = 0.8125 |
| | permutation agreement | 1070/1197 = 0.8939 | 1041/1165 = 0.8936 | 29/32 = 0.9062 |
| | in-distribution abstention | 127/1197 = 0.1061 | 124/1165 = 0.1064 | 3/32 = 0.0938 |
| | ECE choice.k5 | 0.0258 | 0.0279 | not_run (32 < 100) |
| | last option (k5) predicted sigma | +1.129 | +1.222 | -0.453 |
| pooled, every family | top-1 choice (`val_top1.choice`) | 9124/10985 = 0.8306 | 8997/10770 = 0.8354 | 127/215 = 0.5907 |
| | permutation agreement (the gate's population) | 10211/10985 = 0.9295 | 10042/10770 = 0.9324 | 169/215 = 0.7860 |
| | in-distribution abstention (the gate's half) | 923/10985 = 0.0840 | 877/10770 = 0.0814 | 46/215 = 0.2140 |
| | ECE choice.k4 (defect_class + MMLU) | 0.0174 (3789) | 0.0166 (3606) | 0.0712 (183) |

CSQA's `only` predicted sigma, -0.453, equals seed 0's. That is not a stale value. It comes from
counts: 5 of 32 predicted against 6 of 32 gold on both seeds. The mean-probability sigma, which
is continuous, differs (-0.448 on seed 0, -0.554 on seed 1).

Reading. This is report-only, and a quick row decides nothing (rule 8).

**Within a family, holding E_val out again moves little, except MMLU top-1.**
- MMLU top-1 rises 0.0100 and CSQA's falls 0.0012.
- MMLU permutation rises 0.0010 and CSQA's falls 0.0003.
- MMLU abstention falls 0.0010 and CSQA's rises 0.0003.

**The E_val rows are again not answered better than their family, now on both seeds.**
- MMLU's E_val top-1 is 0.5519, against 0.6329 for the rest. The difference is -0.0810, about 2.1
  SE of the difference; seed 0's was -0.0282, 0.7 SE.
- CSQA's E_val top-1 is 0.8125 against 0.7682 (+0.6 SE over 32 rows; seed 0 was +1.3 SE).
- These are normal-approximation z values (inferred, not tested), with no correction for the
  many comparisons in these tables.
- Fable's "contamination can only flatter" (Q6) is not borne out for MMLU top-1 on either seed.

**MMLU's ECE on E_val alone is 0.0712 over 183 rows (seed 0: 0.0979).** Both are above the ece
gate's 0.05 bar, which is stated, not applied. CSQA's is `not_run` again, at 32 rows.

**Pooled, holding E_val out RAISES the pooled gates' values again.**
- Permutation goes 0.9295 to 0.9324, still under the 95% bar.
- In-distribution abstention goes 0.0840 to 0.0814, still over the 5% bar.

No failing gate passes either way.

### Seed 2

Row **`10382ac2-e70b-450c-b042-5b53d24a5c31`**:
- `quick`, report-only, tag `gate-report-excluded`;
- chained to `9e521e77`;
- the same `code_commit` (`0264732`, clean), binary sha256 (`ec88665f…`) and E_val sha256 as
  seeds 0 and 1;
- written 2026-10-02 12:39:59 UTC.

It re-reads F seed 2's eval row `8c3a774a-87d7-4452-a33b-49cc6fbd911e` (ft row `32990e1a…`). The
box wrote the verdict files at 11:13 UTC, and they were pulled read-only at 12:35 UTC (local
mtimes). Each is sha256-equal to the box's copy:
- `verdicts-s2.jsonl` `dd4c6c8e…` (8,271,191 bytes);
- `suite-verdicts-s2.jsonl` `8adc5be5…` (206,995 bytes);
- the box ledger `14ec1d56…` (15 lines, 195,429 bytes).

The box ledger only grew: its first 96,619 bytes hash to seed 1's pull, `00bf2152…`. That pull is
kept as `box/gh200-p4-v4-2026-10-01.s1-pull.jsonl`.

The branch was fast-forwarded to main `9f6b1fa`, which includes `a59c447`, before the run.

Checks:
- All 18223 verdict lines name `8c3a774a` and seed 2, and all 10985 letter lines carry
  `row_logits`. The suite file holds only `needle_hunk_recall` (300) and `ood_abstain` (180).
- Step 6 printed val hash `69d45fd0…`, verdict keys 18223, E_val keys 215 and absent 0, so the
  default refuse path ran, with no allow flag.
- The binary recomputed 65 of the eval row's own numbers equal, with none left uncompared.
- The `full` values were also checked by hand against `8c3a774a`:
  - pooled permutation 10181/10985 is the gate's value, which fails;
  - knowledge permutation 1098/1485 and in-distribution abstention 387/1485;
  - commonsense permutation 1071/1197 and in-distribution abstention 126/1197;
  - pooled in-distribution abstention 945/10985;
  - `val_top1.choice` 9146/10985;
  - ECE k4 0.0166 over 3789 rows and k5 0.0339 over 1197.
- The `only` and `excluded` counts were recounted from the raw verdict lines with `jq`,
  independently of `qd-gate-report`, for all three seeds and both families (`correct` and
  `permutation_agreed`, split by E_val membership). Every count equals the rows'. So seed 2's MMLU
  `only` view, 85/183 correct and 86/183 agreed, is measured, not a scoring defect.
- Peak RSS of step 7 was 295 MB (`/usr/bin/time -l`). The Mac was loaded by another lane's ojas
  build, and the binary's own wall clock was 7.1 s, against 0.2-0.3 s on seeds 0 and 1.

F seed 2, three views (count / n). All verified, read off row `10382ac2` by a script, not by hand:

| Population | Metric | full | excluded (not in E_val) | only (E_val) |
| --- | --- | --- | --- | --- |
| knowledge.multiple_choice (MMLU) | top-1 | 894/1485 = 0.6020 | 809/1302 = 0.6214 | 85/183 = 0.4645 |
| | permutation agreement | 1098/1485 = 0.7394 | 1012/1302 = 0.7773 | 86/183 = 0.4699 |
| | in-distribution abstention | 387/1485 = 0.2606 | 290/1302 = 0.2227 | 97/183 = 0.5301 |
| | ECE choice.k4 | 0.0282 | 0.0287 | 0.0450 (183 rows) |
| | last option (k4) predicted sigma | -3.503 | -2.485 | -3.278 |
| commonsense.multiple_choice (CSQA) | top-1 | 927/1197 = 0.7744 | 900/1165 = 0.7725 | 27/32 = 0.8438 |
| | permutation agreement | 1071/1197 = 0.8947 | 1042/1165 = 0.8944 | 29/32 = 0.9062 |
| | in-distribution abstention | 126/1197 = 0.1053 | 123/1165 = 0.1056 | 3/32 = 0.0938 |
| | ECE choice.k5 | 0.0339 | 0.0341 | not_run (32 < 100) |
| | last option (k5) predicted sigma | +0.376 | +0.458 | -0.453 |
| pooled, every family | top-1 choice (`val_top1.choice`) | 9146/10985 = 0.8326 | 9034/10770 = 0.8388 | 112/215 = 0.5209 |
| | permutation agreement (the gate's population) | 10181/10985 = 0.9268 | 10066/10770 = 0.9346 | 115/215 = 0.5349 |
| | in-distribution abstention (the gate's half) | 945/10985 = 0.0860 | 845/10770 = 0.0785 | 100/215 = 0.4651 |
| | ECE choice.k4 (defect_class + MMLU) | 0.0166 (3789) | 0.0157 (3606) | 0.0450 (183) |

MMLU's `only` ECE ran over 183 rows, 12 of 15 bins with mass. CSQA's `only` predicted sigma is
-0.453 for the third time, from the same 5-of-32 against 6-of-32 counts. Its mean-probability
sigma, -0.514, differs again.

Reading. This is report-only, and a quick row decides nothing (rule 8).

**Seed 2's MMLU E_val rows break permutation.** They agree under permutation on 86 of 183 rows
(0.4699), against 1012/1302 (0.7773) for the rest of MMLU. On seeds 0 and 1 the same 183 rows
agreed on 148 and 140. Top-1 on them is 85/183 = 0.4645, against 0.6214.
- MMLU's `excluded` permutation is steady across seeds (0.7834, 0.7727, 0.7773). Its `full` value
  is not (0.7865, 0.7717, 0.7394). So seed 2's MMLU permutation drop on the eval row comes from
  these 183 rows (inferred from the three views).
- Why is not known. One hypothesis, inferred and not tested, is that F memorised a letter or
  position from the gold-trained train copies of these questions, and that it breaks when the
  permutation moves the answer. The probe is the first command below
  (`GAP-RESCORE-SEED2-EVAL-MMLU-PERMUTATION-COLLAPSE-2026-10-02`).

### Three seeds

Every value below is derived by a script from rows `e663248b`, `9e521e77` and `10382ac2`.

**Holding E_val out, excluded minus full:**

| Metric | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| MMLU top-1 | +0.0035 | +0.0100 | +0.0193 |
| MMLU permutation agreement | -0.0031 | +0.0009 | +0.0379 |
| MMLU in-distribution abstention | +0.0031 | -0.0009 | -0.0379 |
| MMLU ECE k4 | +0.0036 | -0.0001 | +0.0005 |
| CSQA top-1 | -0.0022 | -0.0012 | -0.0019 |
| CSQA permutation agreement | +0.0010 | -0.0003 | -0.0003 |
| CSQA in-distribution abstention | -0.0010 | +0.0003 | +0.0003 |
| CSQA ECE k5 | -0.0000 | +0.0020 | +0.0002 |
| pooled top-1 choice | +0.0041 | +0.0048 | +0.0062 |
| pooled permutation agreement (gate) | +0.0024 | +0.0029 | +0.0078 |
| pooled in-distribution abstention (gate) | -0.0022 | -0.0026 | -0.0076 |
| pooled ECE k4 | +0.0005 | -0.0008 | -0.0009 |

**E_val rows against the rest of their family, only minus excluded:** z is a normal
approximation (inferred, not tested), with no correction for the many comparisons here.

| Metric | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| MMLU top-1 | 107/183 vs 798/1302: -0.0282 (z -0.73) | 101/183 vs 824/1302: -0.0810 (z -2.07) | 85/183 vs 809/1302: -0.1569 (z -4.00) |
| MMLU permutation agreement | 148/183 vs 1020/1302: +0.0253 (z +0.81) | 140/183 vs 1006/1302: -0.0076 (z -0.23) | 86/183 vs 1012/1302: -0.3073 (z -7.95) |
| MMLU ECE k4 | 0.0979 vs 0.0525 | 0.0712 vs 0.0326 | 0.0450 vs 0.0287 |
| CSQA top-1 | 27/32 vs 887/1165: +0.0824 (z +1.26) | 26/32 vs 895/1165: +0.0443 (z +0.63) | 27/32 vs 900/1165: +0.0712 (z +1.09) |
| CSQA permutation agreement | 27/32 vs 1026/1165: -0.0369 (z -0.57) | 29/32 vs 1041/1165: +0.0127 (z +0.24) | 29/32 vs 1042/1165: +0.0118 (z +0.23) |
| CSQA ECE k5 | not_run (32) | not_run (32) | not_run (32) |

**In MMLU and CSQA, in-distribution abstention is exactly 1 minus permutation agreement.** In
every view of every seed, agreed + abstained = n: for example 86 + 97 = 183 and 1098 + 387 = 1485.
The two pooled gates' populations therefore share these families' rows as one measurement.

What moves the same way on all three seeds:
- **MMLU top-1 rises when E_val is held out:** +0.0035, +0.0100, +0.0193. That is because MMLU's
  E_val rows score below the rest on every seed (z -0.73, -2.07, -4.00). Fable's direction
  ("contamination can only flatter", Q6) is not borne out for MMLU on any seed. These rows are
  answered worse, not better.
- **MMLU's ECE on E_val alone is above the rest on every seed** (0.0979, 0.0712, 0.0450, against
  0.0525, 0.0326, 0.0287), over 183 rows each. Report-only; the 0.05 bar is stated, not applied.
- **The pooled gates move toward their bars on every seed, and neither reaches it.**
  - Permutation rises (+0.0024, +0.0029, +0.0078), at most to 0.9346, under 0.95.
  - In-distribution abstention falls (-0.0022, -0.0026, -0.0076), at least to 0.0746, over 0.05.
  - No failing gate passes on any seed.
- **CSQA top-1 falls by at most 0.0022 on every seed.** Its 32 E_val rows score above the rest on
  every seed (z +1.26, +0.63, +1.09), each within 1.3 SE.

What the pooled shift is made of. A counterfactual, derived by script from the rows: the 215 E_val
rows are given their own family's `excluded` rate, and the pooled full value is recomputed. That
counterfactual alone, i.e. composition, accounts for these permutation shifts:
- seed 0: +0.0027 of +0.0024;
- seed 1: +0.0028 of +0.0029;
- seed 2: +0.0027 of +0.0078.

On seeds 0 and 1 the pooled rise is composition: E_val sits in the two weakest families. On seed
2, the remaining +0.0051 comes from the E_val MMLU rows' permutation collapse above.

What does not move consistently: MMLU permutation and abstention (one direction on seed 2, the
other on seed 0), CSQA permutation, abstention and ECE, and pooled ECE k4.

## What changed

| Commit | What |
| --- | --- |
| `2561065` | The fail-first tests. `crates/qd-runtime/tests/gate_report_exclude.rs` has 12 tests over a deterministic SplitMix64 fixture. Its no-flag goldens, `crates/qd-runtime/tests/fixtures/gate_report_exclude/no_flag.{report.json,stdout.txt}`, were written by `qd-gate-report` at `74aea95` (debug, sha256 `ca6dc0aa3a1bcd0cb1603579f56149014d05f9ed7302e99f4fd74faec9686b10`). `python/tests/test_gate_report_row.py` gains 5 tests. The pre-change logs are in `AUDIT/rescore-excluded-2026-10-02/`. |
| `396e8fe` | `qd-gate-report --exclude-rows FILE [--allow-empty-exclude-rows] [--allow-absent-exclude-rows]`, and `tools/gate_report_row.py --exclude-rows`, which writes the `gate-report-excluded` row. |
| `ff9c3b8` | This handoff and runbook, five gap records, and the post-change test logs (`AUDIT/rescore-excluded-2026-10-02/{clippy-post,python-post,qd-runtime-tests-post}.log`). |
| `0264732` | Runbook step 4 checks the suite file's gates, because `qd-gate-report` refuses an unknown suite gate. This is the `code_commit` of row `e663248b`, and `build-wt` / `run-wt` stay at it for seeds 1 and 2. |
| `0e00e9b` | F seed 0's row `e663248b` in `ledger/mac-rescore-excluded-2026-10-02.jsonl`, plus this file's measured section. |
| `d68a5c3` | Runbook step 6 becomes one `jq` that writes no file. The harness refuses a redirect outside the project root, which the seed-0 run hit. |
| `ffc6e6d` | Two gap updates: `GAP-RESCORE-EXCLUDED-NOT-YET-RUN-2026-10-02` (seed 0 done) and `GAP-RESCORE-ONLY-VIEW-ECE-NEEDS-100-ROWS-2026-10-02` (resolved with residual, 183/32). Also this table, and step 6 no longer names a file it does not write. |
| `4b24a75` | F seed 1's row `9e521e77` in `ledger/mac-rescore-excluded-2026-10-02.jsonl`, the Seed 1 section above, and gap updates for seed 1. Three runbook fixes. Step 9's prefix check is now `head -c \| shasum`: macOS `cmp -n N` exits 1 with "EOF" when the shorter file is exactly N bytes, so it failed on a true prefix during the seed-1 run. Step 9 also copies into the lane's own checkout, not main's. Step 3 keeps the previous seed's ledger pull. |
| `f364a52` | The first command for the next lane became seed 2's read-only step 3. |
| (fast-forward) | The branch was fast-forwarded to main `9f6b1fa` before seed 2. It carries no change of this lane's. |
| this commit | F seed 2's row `10382ac2`, the Seed 2 and Three seeds sections, and gap updates: `GAP-RESCORE-EXCLUDED-NOT-YET-RUN-2026-10-02` resolved, `GAP-RESCORE-ONLY-VIEW-ECE-NEEDS-100-ROWS-2026-10-02` now verified on three seeds, and the new `GAP-RESCORE-SEED2-EVAL-MMLU-PERMUTATION-COLLAPSE-2026-10-02`. Step 9's worktree cleanup is done: `run-wt` came out without `--force`. |

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

- `GAP-RESCORE-SEED2-EVAL-MMLU-PERMUTATION-COLLAPSE-2026-10-02` (new, open): on seed 2, MMLU's
  E_val rows agree under permutation on 86/183, against 148 and 140 on seeds 0 and 1. Why is
  unknown. The first command below is the probe.
- `GAP-RESCORE-EXCLUDED-NOT-YET-RUN-2026-10-02`: resolved. All three seeds are re-scored (rows
  `e663248b`, `9e521e77`, `10382ac2`).
- `GAP-RESCORE-ONLY-VIEW-ECE-NEEDS-100-ROWS-2026-10-02`: resolved with residual. On all three
  seeds, MMLU's `only` view has 183 rows and its ECE ran (0.0979, 0.0712, 0.0450). CSQA's has 32
  rows and its ECE is `not_run`: not measured, which is not the same as calibrated.
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
- seed 1 is `aeca8d69-4733-4593-92de-2073021ed684`, from ft row `95fa4854…`; its files landed on
  the box at 05:25 UTC;
- seed 2 is `8c3a774a-87d7-4452-a33b-49cc6fbd911e`, from ft row `32990e1a…`; its files landed on
  the box at 11:13 UTC.

All three are re-scored, and the worktrees are removed (step 9). To re-run a seed, start again at
step 1. The binary is still at `…/target/release/qd-gate-report` (`ec88665f…`), so step 2 can be
skipped if its sha256 still matches.

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
- **The previous seed's ledger pull.** The ledger's name has no seed in it, so the `scp` above
  overwrites the previous seed's pull. Before pulling, keep that copy. Seed 1's run kept seed 0's
  as `.s0-pull.jsonl`; for seed 2, run
  `cp -p …/box/gh200-p4-v4-2026-10-01.jsonl …/box/gh200-p4-v4-2026-10-01.s1-pull.jsonl`.
  Afterwards, check that its bytes are a prefix of the new pull, by hash as in step 9.
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

This is one `jq`, and it writes no file. The harness refuses a shell redirect to any path
outside the project root, which the seed-0 run hit.

    jq -rn --rawfile e /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt --slurpfile val /Users/bharath/qd-campaign/phase4-v4-2026-10-01/data/pool/val.json '([inputs | "\(.row_id)#\(.slot_name)"] | map({key: ., value: true}) | from_entries) as $have | ($val[0].entries | map({key: .row_id, value: true}) | from_entries) as $valrows | [$e | split("\n")[] | select(length > 0)] as $keys | [$keys[] | select($have[.] | not)] as $absent | "val data_snapshot_hash \($val[0].data_snapshot_hash)", "verdict keys \($have | length)", "E_val keys \($keys | length)", "absent \($absent | length)", ($absent[] | "absent-key \(.)"), ($absent[] | select((sub("#[^#]*$"; "")) as $r | $valrows[$r] | not) | "NOT-A-VAL-ROW \(.)"), ($absent[] | select(test("^(mmlu|csqa):") and (test("#answer$") | not)) | "BAD-SLOT \(.)")' /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s0.jsonl

Seed 0 printed `val data_snapshot_hash 69d45fd04fad01751b3f102f09041c0ba67176e44725dbeb977d550f4b0abbd4`,
`verdict keys 18223`, `E_val keys 215` and `absent 0`. The hash must be that value (Fable Q4
names `69d45fd0…` as F's val). To get the sha256 of the absent keys, which matches the report's
`absent.sha256`, pipe the same command through
`rg '^absent-key ' | sed 's/^absent-key //' | LC_ALL=C sort | shasum -a 256`. Then decide:
- **`absent 0`:** run step 7 without `--allow-absent-exclude-rows`.
- **Absent keys, and all four hold:**
  - no `NOT-A-VAL-ROW` or `BAD-SLOT` line is printed, so every absent key is a v4 val row, with
    slot `answer` for MMLU/CSQA;
  - every absent key is undecoded;
  - the eval row's undecoded count (`decoded.n_total - decoded.n` from step 4) is not 0;
  - the absent count is at most that undecoded count.

  Then add `--allow-absent-exclude-rows`. Record the count, and the sha256 printed by the pipe
  above, in the lead's handoff. That sha256 equals the report's `absent.sha256`: both hash the
  keys sorted byte-wise, each LF-terminated (`crates/qd-runtime/src/bin/qd_gate_report.rs:1721-1722`).
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
(`GAP-RESCORE-ONLY-VIEW-ECE-NEEDS-100-ROWS-2026-10-02`). On seed 0, MMLU (183) ran and CSQA (32)
did not. Report a `not_run` as not measured, not as calibrated. Under rule 8 a `quick` row
decides nothing.

### 9. Land the ledger, then clean up

    PYTHONPATH=/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt/python /Users/bharath/.venvs/ml/bin/python -m qd_train.ledger verify --ledger /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl
    cp /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl <your checkout>/ledger/mac-rescore-excluded-2026-10-02.jsonl

`<your checkout>` is the checkout of the branch that will carry the row: for this lane, its
worktree `/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-ab8fcc502a43caacc`.
It is never main's checkout while another lane may be working there. Commit
`ledger/mac-rescore-excluded-2026-10-02.jsonl` by explicit path.

For a later seed, append to the same outside ledger, `verify`, and copy again. The file only
grows, so before replacing the committed copy, check that it is a byte prefix of the new one.
These two hashes must be equal:

    stat -f %z <committed copy>
    shasum -a 256 <committed copy>
    head -c <that size> /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/mac-rescore-excluded-2026-10-02.jsonl | shasum -a 256

Do not use `cmp -n <size>`. On this Mac it exits 1 with "EOF on <shorter file>" when the shorter
file is exactly `<size>` bytes and the other is longer: it fails on a true prefix. That
happened on the seed-1 run.

When all seeds are in:

    git worktree remove --force /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/build-wt
    git worktree remove /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/run-wt

`--force` is needed for `build-wt` only, because its `Cargo.lock` re-resolved. `run-wt` must
come out without it. If it does not, something wrote inside it, and the rows' `code_commit` will
say `-dirty`.

Done after seed 2: both commands exited 0, and `run-wt` came out without `--force`. Still under
`/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/`:
- `target/`, about 421 MB, holding the binary;
- `report-s{0,1,2}.json`;
- the box pulls, including `.s0-pull` and `.s1-pull`;
- `e_val.txt` and the outside ledger.

`target/` is regenerable; the rest is evidence.

## First command for the next lane

The re-score is complete. What is open is
`GAP-RESCORE-SEED2-EVAL-MMLU-PERMUTATION-COLLAPSE-2026-10-02`.

The first command asks whether the MMLU permutation disagreements are position-locked: the model
picks the same presented position under both orderings (`top_permuted == top`). It splits them by
E_val membership, on the verdict files already pulled. It is read-only and writes no file:

    for s in 0 1 2; do jq -rn --arg s "$s" --rawfile e /Users/bharath/qd-campaign/rescore-excluded-2026-10-02/e_val.txt '([$e | split("\n")[] | select(length > 0) | {key: ., value: true}] | from_entries) as $ev | [inputs | select(.family_id == "knowledge.multiple_choice" and .kind == "choice" and (.permutation_agreed | not)) | {inE: ($ev["\(.row_id)#\(.slot_name)"] // false), locked: (.top_permuted == .top), noul: (.top == .noul_row or .top_permuted == .noul_row)}] | group_by([.inE, .locked, .noul]) | map("seed \($s) \(if .[0].inE then "only" else "excluded" end) position-locked=\(.[0].locked) noul-involved=\(.[0].noul): \(length)") | .[]' "/Users/bharath/qd-campaign/rescore-excluded-2026-10-02/box/verdicts-s$s.jsonl"; done

What the fields mean, verified on seed 2's 10985 letter lines:
- `perm` maps a presented position to an original row;
- `top` is an original row, and `top_permuted` a presented position;
- outside abstention, `permutation_agreed` equals `perm[top_permuted] == top`;
- the 141 other lines abstain under both orderings, which counts as agreement.

None of the probe's counts is in a ledger row yet. Whoever runs it records them before citing any
of them (rule 5). The next step after that is which v4 train rows hit these 183 keys, and whether
their gold letter differs from the val permutation. E_val came from L-replay's
`AUDIT/replay-v4-build-2026-10-02/gold_side_check.py`. Its `gold-side/summary.json` records the
train-to-val pairs only as counts and a hash (`pairs_total`, `pairs_sha256`), so getting the pairs
means re-running that check (`HANDOFF/replay-v4-build-2026-10-02.md`). `replay-hits-build1.json`
is not the right file: it compares the replay set against val and heldout, not F's gold-train
rows.
