# Evidence: `tools/real_ft_run.py`'s CPU prelude, before and after (2026-10-01)

This is the record behind `HANDOFF/perf-ft-run-prep-2026-10-01.md`. The raw rounds are in
`AUDIT/perf-ft-run-prep-2026-10-01.json`, which the benchmark itself wrote.

- **Host.** Everything ran on the Mac CPU (18 cores, 64 GiB). Nothing ran on MPS and nothing ran
  on the GH200. The machine was shared with other agents' jobs. Load averages (1/5/15 min) were
  17.62 / 39.34 / 34.28 at the start (10:51) and 6.18 / 5.98 / 10.17 at the end (11:22). That
  is why the runs are interleaved and reported as the minimum of N.
- **Inputs.** These are the ones the J4 and J7g runs used:
  - shard set `/Users/bharath/qd-campaign/phase4-v3-2026-10-01`;
  - `--rev b381a03cb12e816c380915b1983f4e13cd1c843c --no-repo-history`;
  - `--defect-class data/pool/commitpackft-corpus-v3`, `--defect-download data/pool/commitpackft`
    and `--defect-noul data/pool/defect-noul-v1`;
  - `--general-record ~/.cache/qd-decision/general/fetch-record-2026-09-29.json
    --general-max-rows 200000 --replay-partition`;
  - the Qwen3.5-2B-Base snapshot as `--real-backbone`;
  - in every stage, `--score-val --needle --ood --ood-general-record <same record> --devices cpu`.
- **Arms.**
  - **before** is a checkout of `73bcf12`, this branch's base.
  - **after** is this branch.
  - Both arms sign MinHash with the same `QD_PREP_BIN`
    (`/Users/bharath/qd-campaign/target-prepperf/release/qd-prep`). The before arm only ever
    sends it `QDPMHIN1` requests, which this change leaves untouched.

## 1. The committed A/B (`python/tests/test_real_ft_prep_benchmark.py`)

Each run is a fresh process. It is timed from `main()` to the first step that is not CPU
preparation.

| stage | flags | stops at | stubbed (in both arms) |
| --- | --- | --- | --- |
| `score` | `--score-checkpoint AVG --seeds 0 1 2 --score-dtype fp32` | `_score_checkpoint` (the tower load) | the needle worker's launch; the after arm still writes the handoff, since that is new work it pays |
| `worker` | the `--needle` worker | `needle_predictions` (the decode) | `_checkpoint_step` (the same checkpoint load in both arms) |
| `train` | `--epoch --no-memorise --seeds 0 1 2 --batch-tokens 16384 --optimizer master` (J4) | `_train` | `_probe_one` (device probes, **not measured**) |

Run as:

```
QD_PREP_BENCH_BEFORE=<checkout of 73bcf12 with data/pool linked> \
QD_PREP_BIN=/Users/bharath/qd-campaign/target-prepperf/release/qd-prep \
QD_PREP_BENCH_RESULTS=AUDIT/perf-ft-run-prep-2026-10-01.json \
pytest python/tests/test_real_ft_prep_benchmark.py -o addopts= -q -s
```

Three rounds. Round 0 ran before then after, round 1 after then before, round 2 before then
after. All times are seconds.

| stage | before (3 runs) | after (3 runs) | before min | after min | saved |
| --- | --- | --- | --- | --- | --- |
| `score` | 152.10, 157.41, 154.29 | 77.58, 82.48, 70.54 | **152.10** | **70.54** | 81.56 (2.16x) |
| `worker` | 145.52, 138.04, 132.53 | 0.90, 0.77, 0.79 | **132.53** | **0.77** | 131.76 |
| `train` | 151.06, 129.87, 129.48 | 105.20, 95.71, 96.58 | **129.48** | **95.71** | 33.77 (1.35x) |

A `--needle` scoring job pays for the `score` and `worker` stages in sequence. On the minimums
that came to 284.63 s before and 71.31 s after.

**Digests are equal across arms, stage by stage** (`summary.*.same_digest` is true):

| stage | digest |
| --- | --- |
| `score` | `c44774b56e3e4fc8...` |
| `worker` | `08477aec3538dbaa...` |
| `train` | `49a27e49b787233f...` |

Each digest hashes what the stage hands on:

- the val plan, labels and letter ids;
- the second pass;
- the needle and OOD suites, and the eval widths;
- for `train` only, the epoch plan, the train labels and the inventory.

So the faster prelude scores and trains on exactly the inputs the slower one did. The
benchmark asserts this, and it passed (`1 passed in 1851.30s`).

## 2. Where the before time went (attribution, not separately A/B'd)

**cProfile** of the pre-change training prelude, the `train` stage at `73bcf12`: `main` took
310.0 s cumulative under the profiler. The cumulative times were:

- `ft_split_rows`: 164.1
- `render`: 127.5. Of that, `render._escape` took 103.0 over 9,054,971 calls.
- `_labels`: 87.5 over 482 calls
- `build_mixture`: 81.9. Of that, `drop_contradictory_prompts` took 54.3.
- `candidate_pairs`: 46.5 over 4 calls
- `prepare_needle`: 37.4. Of that, `_token_index_for_char` took 18.0 over 202,415 calls.
- `prepare_ood`: 11.8
- `_matching_tokenizer`: 7.9 over 2 calls

**Unprofiled phase times** come from one run of each before stage, at load around 5-10:

- `score` prelude, 142.67 s in total:
  - `ft_split_rows`: 69.18
  - the train-side `_labels`: about 30
  - `prepare_needle`: 27.71
  - `_batch_inventory`: 5.92
  - `prepare_ood`: 5.80
  - `open_val_set`: 1.80
  - `_contradictions`: 1.16
- `train` prelude, 133.57 s in total:
  - `ft_split_rows`: 64.79
  - `_labels`: 29.86
  - `prepare_needle`: 27.52
  - `prepare_ood`: 5.83

**Inside `ft_splits`**, from one unprofiled run at load around 40, the main rebuild took 78.45 s:

- `build_mixture`: 24.09, of which `drop_contradictory_prompts` took 14.41
- `dedupe`: 17.72, of which `candidate_pairs` took 10.04
- `split`: 18.21, of which `candidate_pairs` took 14.58
- `native_minhash` setup: about 14, of which `_prep_signatures` took 4.66
- `load_defect_rows`: 2.48
- `general_rows`: 1.16

The OOD rebuild took 7.49 s.

## 3. What each change removed

This section is inferred from section 2. Only the LSH port has its own A/B (section 4).

- **Needle worker (`worker` stage).** The worker no longer runs any prelude of its own. The
  parent writes the suite it built, the batch arrays and the letter ids into an `.npz`
  handoff, keyed by digest. The worker checks that handoff against the train and val shard
  hashes, the val plan's shapes and the seed, and re-derives both digests. It opens only the
  train reader and the val reader, which pairs val with the train remap.
  - The worker stays a fresh process, and the reason it was one is kept. MPS keeps a compiled
    graph per input shape, and `torch.mps.empty_cache` does not release it. See
    `run_needle_worker`'s docstring (`73bcf12:tools/real_ft_run.py:3960`).
  - The worker prints a line saying the rebuild was not repeated.
- **Score mode.** The train relabel, inventory, contradictions and epoch plan are skipped, and
  each skip prints `NOT RUN -- <reason>`. Two conditions must both hold:
  - a tokenizer.json is present;
  - every letter offered by the val and OOD non-span rows is a val gold.

  If either fails, the relabel runs as before.
- **Both modes.**
  - `_labels` renders each row once instead of twice.
  - `_token_index_for_char`'s per-position Python scan is replaced by
    `_token_indices_for_chars`, one vectorised comparison.
  - The tokenizer loads once per process instead of once per suite.
  - `candidate_pairs` runs in `qd-prep lsh`.

## 4. The LSH port alone (`python/tests/test_qd_prep_lsh_parity.py`, gated `QD_PREP_BENCH=1`)

The workload is `split`'s cross-check search over every row of the phase-3 defect-class mixture:

- 50,177 keys;
- 16 bands of 8 rows;
- 744 pairs.

It runs inside the native block. Arms alternate their order per round. The `qd_prep` arm
includes everything the swap does: the matrix gather, the request, the process, the reply
checks, the frozenset and the canaries. Load was about 6.2.

```
reference_s [1.372, 1.247, 1.197, 1.445, 1.229]  min 1.197
qd_prep_s   [0.123, 0.127, 0.129, 0.135, 0.148]  min 0.123   (9.7x)
```

The answers were equal in every round. The phase-4 rebuild's two searches are about five times
this many keys (24.6 s of reference time in section 2). That saving is inside the `score` and
`train` figures above, and is not measured on its own at that scale.

## 5. Not measured here

- The GH200 box.
- MPS or CUDA decode.
- The device probes.
- Any ledger row. The `score` stage stops before `_score_checkpoint`, so no row is written on
  the Mac.

See `GAP-PERF-PREP-BOX-TIMINGS-AND-CUDA-WORKER-NOT-RUN` and
`GAP-PERF-PREP-PROBES-AND-LEDGER-ROWS-NOT-MEASURED`.
