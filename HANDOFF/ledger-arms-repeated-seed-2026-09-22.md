# Handoff — ledger_arms repeated-seed lane, 2026-09-22

## One paragraph

`tools/ledger_arms.py` grouped rows into arms by `(recipe_hash, backbone_commit)` and
counted every row as a seed, so a re-run was pooled with the run it repeated. Six committed
ledgers measure a seed twice inside one arm. One of them fed the four-way table in
`HANDOFF/context-source-2026-09-22.md`, which reported the `after` arm as n=13 over 8 seeds.
The tool now refuses such an arm, naming the rows, and `--split-by-code` reads each code
version as its own arm. No conclusion anywhere changes: the pooled figures move by at most
0.11pp and every margin stays negative. The pooled read did claim about 1.7× the resolution
the arm has, which is the direction this repository keeps finding its errors in.

## What was measured

Every tracked ledger, grouped the way the tool groups, counting a seed that appears more
than once in one arm. "Measured" rows carry both `val_choice_top1_over_baseline` and
`train_choice_top1_over_train_majority`; only those enter a sample.

| ledger | repeated seeds, all rows | among measured rows | what the repeat is |
| --- | --- | --- | --- |
| `gh200-2026-09-20` | 43 | 0 | build, smoke and memorise rows |
| `gh200-2026-09-21` | 53 | 0 | the same, plus 41 early rung-0 rows with no train gate |
| `runs.jsonl` | 44 | 0 | build, smoke and memorise rows |
| `gh200-overnight-2026-09-21` | 18 | 0 | phase 1's 8× seed 0 (by design) and verdict rows |
| `gh200-context-source-2026-09-22` | 8 | **8** | the `after` arm re-run at later code |
| `gh200-fourway-2026-09-22` | 6 | **5** | the `after` arm re-run at later code, plus a killed row |
| `gh200-determinism-2026-09-21` | 4 | 0 | by design |
| `gh200-commitpackft-2026-09-22` | 3 | **3** | the width-128 arm re-run, plus a killed row |
| `gh200-det-2026-09-21` | 2 | 0 | by design |
| `gh200-seedcheck-2026-09-21` | 2 | 0 | by design |
| `mac-rung0-prereg-2026-09-21` | 2 | **2** | two runs over two data snapshots |
| `gh200-rung0-repro-2026-09-21` | 1 | **1** | reproducibility pair, 2.8pp apart |
| `gh200-rung0-reprodet-2026-09-21` | 1 | **1** | reproducibility pair, bit-identical |

**The rev does not separate the re-runs; the code digest does.** In all three re-run
ledgers both runs carry launch rev `0632f693d3` and differ only in `code_that_ran`:

| ledger | first run | re-run |
| --- | --- | --- |
| four-way `after` | rows 1–6, `065920c4f8635051` (seed 5 killed) | rows 7–14, `d10f23542617e9f5` |
| context-source `after` | rows 1–8, `31fc25217350b381` | rows 17–24, `065920c4f8635051` |
| commitpackft width 128 | rows 1–3, `7e74367192350f64` | rows 13–20, `a5da29c2d707b3dc` (killed row 12) |

Every other measured arm in every tracked ledger has exactly one code digest, one data
snapshot and one tokenizer.

### Numbers that pooled a repeated seed

Line numbers are as the file reads now; the correction notes sit directly below each one.

| where | reported | complete run alone | changes? |
| --- | --- | --- | --- |
| `HANDOFF/context-source-2026-09-22.md:84`, four-way `after` n | 13 | 8 (rows 7–14) | yes |
| same row, paired margin | −0.0764 | −0.0774 | yes, 0.001 |
| same row, range | −0.088 … −0.065 | −0.088 … −0.065 | no |
| same row, positive | 0/13 | 0/8 | denominator only |
| same row, model top-1 | 50.20% | 50.10% | yes, 0.10pp |
| same row, control | 57.8% | 57.8% | no, it is one fixed control |
| `:87`, `after` over the prior | +4.1% (spread 2.3) | +4.10pp, range 2.33 | no: already this run; pooled is +4.20 |
| `:90–92`, "pooled for those and nothing else" | — | — | contradicted: `:84`'s top-1 is pooled too |
| `:94`, "29 completed rows" | 29 | 37 margin rows, 32 distinct seeds | matches neither; zero wins holds |
| `:192`, span start | 1.50% on 13 rows | 1.49% on 8, every seed below 3.2% chance | yes, 0.01pp |
| `gaps.jsonl`, `GAP-RUNG0-SPAN-HEAD-COLLAPSES-WHEN-IT-MUST-ALSO-ABSTAIN` | 1.50% over 13 rows | 1.49% over 8 | yes; human-owned, not re-appended |
| `:340`, "~+25 points to the model" | 75.93 − 50.20 = 25.73 | 25.83 | not at the stated precision |
| commit `b72aa54` message, width-128 val | 50.18% (11 rows) | 50.29% (rows 13–20) | yes; a commit message, not amendable |

What the tool printed for the pooled four-way arm, and for its complete run
(`python tools/ledger_arms.py --split-by-code ledger/gh200-fourway-2026-09-22.jsonl`):

| read | n | mean | sd | honest floor, fixed reference / another arm |
| --- | --- | --- | --- | --- |
| pooled, before `dd59c68` | 13 | 50.20% | 0.60pp | 0.51 / 0.69pp |
| complete run, `d10f23542617e9f5` | 8 | 50.10% | 0.76pp | 0.87 / 1.14pp |
| first run, `065920c4f8635051` | 5 | 50.37% | 0.12pp | 0.19 / 0.23pp |

The two runs are not samples of one thing. At the same five seeds the first run's sd is
0.12pp against the re-run's 0.76pp, so pooling them hid a change in seed sensitivity as
well as double-counting. What changed between those two code digests was not established
here.

### Checked and not affected

* Every ledger this tool is run on in an AUDIT or HANDOFF file:
  `AUDIT/span-in-diff.md:23` and `HANDOFF/rung0-control-2026-09-22.md:81` (span-in-diff),
  `HANDOFF/coordinator-fork-2026-09-21.md:363` and `HANDOFF/coordinator-2026-09-21.md:2646`
  (curve-n24), and the e10, e30, sw005-nondet and learning-curve sweeps behind the floors
  and prior sds. All are clean, and the tool's output on each is byte-identical before and
  after the fix.
* `HANDOFF/context-source-2026-09-22.md:62`, the context-source margins at n=8: they are
  the re-run alone, because the first run's margins are `not_run`.
* `:218`, "~60.5%": the first run's first two seeds, not a pool.
* `AUDIT/capacity-and-learning-rate.md:11`, "3.9 points over 24 seeds": this matches the
  complete run (3.87), not the pooled 3.76.
* `HANDOFF/coordinator-2026-09-21.md:607`, the repro pair (50.3% and 53.1%): compared as a
  pair, which is what it is.
* `HANDOFF/coordinator-2026-09-21.md:1008–1016`, the prereg rows: each run's own
  `sweep_can_resolve`, never a pool.
* `AUDIT/ft-overnight-2026-09-21.json`: arms are grouped by label, and each phase-2 arm has
  12 distinct seeds. Phase 1's eight seed-0 repeats are reported as a fixed-seed residual,
  which is their purpose.
* The 2026-09-20 rung-0 AUDIT JSONs: per-configuration seed lists of 3 or 5 values, with no
  seed twice.

## What changed

| commit | what |
| --- | --- |
| `dd59c68` | `ledger_arms` refuses a seed measured twice in one arm, naming both rows, and takes `--split-by-code`. `RepeatedSeed`, `SeedClaims` and `code_of` have one owner there, and `operator_holdout_report` imports them; its output is byte-identical on its committed ledger. Two correction notes in `HANDOFF/context-source-2026-09-22.md`. |
| this commit | `gaps.jsonl`: the repeated-seed gap is resolved with a residual, plus one new residual gap. This file. |

Why refuse-plus-flag rather than the alternatives:

* **Dedupe** has to choose a run. Keep-first would have built the four-way arm from two code
  versions, and keep-last rewards re-running until the number suits.
* **Per rev** separates none of the six measured repeats: the re-runs share their rev, and
  the 09-21 ledgers carry no rev at all.
* **The split** keeps the repro pairs refused, because one seed twice at one code is a
  reproducibility check, not a population.
* **Repeats on unmeasured rows** are counted and said rather than refused, so the chain
  ledgers still read.

Verified: 22 new test items, 21 of which fail against the pre-fix code; the 22nd guards
against over-refusal. `make pytest` 1760 passed, 77 skipped. `make torch-pytest` 2160
passed, 14 skipped. `make lint` is clean. Over the 29 tracked ledgers, HEAD's tool and this
one print:

* 15 byte-identical;
* 8 that differ only by an added line saying their row count is not a seed count;
* 6 refused, of which the three re-runs read with `--split-by-code`.

## What is open

| gap | state |
| --- | --- |
| `GAP-LEDGER-ARMS-POOLS-A-REPEATED-SEED-AS-A-NEW-ONE` | resolved-with-residual, `dd59c68` |
| `GAP-LEDGER-ARMS-KEY-OMITS-THE-DATA-SNAPSHOT` | **open**: two data snapshots with disjoint seeds still pool silently; latent in every committed ledger |
| `GAP-RUNG0-SPAN-HEAD-COLLAPSES-WHEN-IT-MUST-ALSO-ABSTAIN` | human-owned; still reads "1.50% over 13 rows" (1.49% over 8). Left to its owner. |

## The exact first command for the next lane

The one committed ledger whose arm key is too coarse. Its refusal names both data snapshots:

```bash
/Users/bharath/.venvs/ml/bin/python /Users/bharath/Code/research/qwen-decision/tools/ledger_arms.py --split-by-code /Users/bharath/Code/research/qwen-decision/ledger/mac-rung0-prereg-2026-09-21.jsonl
```
