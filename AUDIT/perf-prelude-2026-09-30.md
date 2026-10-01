# Evidence: the `--score-checkpoint` prelude, before and after `crates/qd-prep` (2026-09-30)

This is the record behind `HANDOFF/perf-prelude-2026-09-30.md`.

- **Host and inputs.** Everything ran on the Mac CPU (18 cores, 64 GiB), not on MPS and not on
  the GH200. The inputs are the phase-3 set: `--out
  /Users/bharath/qd-campaign/box-final-2026-09-30/phase3 --no-repo-history --defect-class
  data/pool/commitpackft-corpus-v2 --rev be3073300cf4efb664f7065a32a44e4b9c12bd37 --real-backbone
  <Qwen3.5-2B-Base snapshot b1485b2f> --score-val --needle --ood --ood-general-record
  /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json`, plus `--devices cpu`.
- **Where the prelude stops.** It ends where `main()` launches the needle worker
  (`run_needle_worker`). A worker process would stop at `_checkpoint_step` instead, and its
  prelude is the same code.
- **Shared machine.** Another session was using the machine, so wall clock varies between runs.
  That is why the A/B is interleaved and reported as min-of-N.

## 1. Profile (stdlib cProfile, 63f5b68, before any change)

`main()` prelude: 461.90 s under the profiler. Import of `real_ft_run` took 0.79 s.

```
   ncalls  tottime  cumtime  function
        2    0.203  390.720  real_ft_run.py:2609(ft_splits)
   151812    6.404  346.112  qd_data/minhash.py:175(signature)
 19583748    8.229  329.410  qd_data/minhash.py:191(<genexpr>)
 19538531   91.769  321.195  {built-in method builtins.min}
        1    0.012  317.953  real_ft_run.py:2715(ft_split_rows)
        2    0.275  234.476  qd_data/split.py:267(split)
        2    0.612  233.842  qd_data/split.py:419(_cross_split_near_duplicates)
809071616  229.426  229.426  qd_data/minhash.py:192(<genexpr>)
        2    0.473  133.230  qd_data/dedupe.py:245(dedupe)
        1    0.006   74.809  real_ft_run.py:3761(prepare_ood)
        1    0.010   45.111  real_ft_run.py:3551(prepare_needle)
      480    0.007   36.469  real_ft_run.py:3474(encode_slot_batch)
   182047    0.762   32.167  qd_data/render.py:667(render)
      480    0.071   31.158  qd_train/shards.py:836(encode_slot)
      480    0.052   28.318  qd_train/shards.py:1028(_span_token_positions)
  2082189   17.362   26.134  qd_data/render.py:274(_escape)
      482    0.234   22.220  real_ft_run.py:490(_labels)
   202415   21.603   21.603  qd_train/shards.py:319(_token_index_for_char)
        2    0.078   18.572  qd_data/mixture.py:1240(build_mixture)
        4    3.277   11.899  qd_data/minhash.py:320(candidate_pairs)
  6169060    7.224   10.298  qd_data/minhash.py:171(base_hash)
        2    0.000    9.867  real_ft_run.py:3459(_matching_tokenizer)
      780    4.145    4.145  {method 'encode_batch' of 'tokenizers.Tokenizer' objects}
```

## 2. Unprofiled baseline (0f4a18b, per-phase wall clock, phases nest)

```
TIMING {"label": "before-0f4a18b", "import_s": 0.85, "prelude_s": 238.76, "phases": {"ft_splits": [186.64, 2], "dedupe": [88.13, 2], "split": [86.88, 2], "prepare_ood": [54.59, 1], "prepare_needle": [39.97, 1], "_labels": [8.63, 482], "_batch_inventory": [1.42, 1], "prepare_second_pass": [0.61, 2], "open_val_set": [0.53, 1], "_contradictions": [0.38, 1], "ShardReader": [0.24, 2], "_letter_ids": [0.11, 2], "check_defect_source": [0.09, 1], "corpus_facts": [0.09, 1], "pair_labels": [0.08, 2], "_inventory": [0.08, 2]}}
```

## 3. Interleaved A/B on the full prelude (perf-prelude-rust at 3c90201)

The two arms run the same code:

- **reference:** `real_tokenizer_pipeline.native_minhash` is replaced by a pass-through, so
  `qd_data`'s own `MinHasher` signs.
- **native:** `QD_PREP_BIN=<worktree>/target/release/qd-prep`.

The rounds ran in the order reference, native, reference, native, reference, native.

`output_digest` is a sha256 over every row id of both `ft_splits` results, in order. It also
covers the needle suite digest, the OOD record sha and case ids, and the val set size, plan
length and letter ids.

**Truncated lines.** The lines from rounds 1 and 2 were cut at 250 columns by this host's
`~/.ripgreprc` (`--max-columns=250`). They were piped through `rg`, so only `prelude_s` and
`output_digest` survive. Round 3 is complete.

```
TIMING {"arm": "reference", "import_s": 0.85, "prelude_s": 285.46, "output_digest": "edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469", "phases": {"ft_splits": [209.45, 2], "dedupe": [104.92, 2], "split": [91.69, 2], "prepare_needle": [... omitted end of long line]
minhash: 49840 distinct sets signed by .../target/release/qd-prep in 3.5 s (shingling and canaries included), 100018 lookups, 8 canaries equal to the reference
minhash: 16829 distinct sets signed by .../target/release/qd-prep in 0.6 s (shingling and canaries included), 51794 lookups, 8 canaries equal to the reference
TIMING {"arm": "native", "import_s": 0.94, "prelude_s": 102.12, "output_digest": "edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469", "phases": {"prepare_needle": [43.84, 1], "ft_splits": [35.75, 2], "_labels": [15.68, 482], "prepare_o [... omitted end of long line]
TIMING {"arm": "reference", "import_s": 1.35, "prelude_s": 322.37, "output_digest": "edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469", "phases": {"ft_splits": [267.82, 2], "split": [146.44, 2], "dedupe": [101.71, 2], "prepare_ood": [ [... omitted end of long line]
minhash: 49840 distinct sets signed by .../target/release/qd-prep in 1.6 s (shingling and canaries included), 100018 lookups, 8 canaries equal to the reference
minhash: 16829 distinct sets signed by .../target/release/qd-prep in 0.9 s (shingling and canaries included), 51794 lookups, 8 canaries equal to the reference
TIMING {"arm": "native", "import_s": 0.74, "prelude_s": 66.67, "output_digest": "edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469", "phases": {"prepare_needle": [34.28, 1], "ft_splits": [22.1, 2], "prepare_ood": [9.35, 1], "_labels":  [... omitted end of long line]
TIMING {"arm": "reference", "import_s": 0.71, "prelude_s": 175.68, "output_digest": "edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469", "phases": {"ft_splits": [133.11, 2], "dedupe": [63.18, 2], "split": [61.57, 2], "prepare_ood": [37.06, 1], "prepare_needle": [32.28, 1], "_labels": [6.81, 482], "_batch_inventory": [1.16, 1], "prepare_second_pass": [0.66, 2], "open_val_set": [0.42, 1], "_contradictions": [0.39, 1], "pair_labels": [0.1, 2], "_letter_ids": [0.09, 2], "corpus_facts": [0.08, 1], "_inventory": [0.06, 2]}}
minhash: 49840 distinct sets signed by .../target/release/qd-prep in 1.7 s (shingling and canaries included), 100018 lookups, 8 canaries equal to the reference
minhash: 16829 distinct sets signed by .../target/release/qd-prep in 2.6 s (shingling and canaries included), 51794 lookups, 8 canaries equal to the reference
TIMING {"arm": "native", "import_s": 0.81, "prelude_s": 70.75, "output_digest": "edcd6bacdb7d4bd32f85ca7fbbd46ca256f06799166c888652db73ffb8f95469", "phases": {"prepare_needle": [36.6, 1], "ft_splits": [23.67, 2], "prepare_ood": [9.77, 1], "_labels": [7.02, 482], "split": [4.97, 2], "dedupe": [4.03, 2], "_batch_inventory": [1.29, 1], "prepare_second_pass": [0.71, 2], "open_val_set": [0.41, 1], "_contradictions": [0.22, 1], "pair_labels": [0.1, 2], "_letter_ids": [0.09, 2], "corpus_facts": [0.08, 1], "_inventory": [0.06, 2]}}
```

**Results:**

| arm | runs (s) | min (s) |
| --- | --- | ---: |
| reference | 285.46, 322.37, 175.68 | 175.68 |
| native | 102.12, 66.67, 70.75 | 66.67 |

- The ratio of the minima is 2.6×.
- In round 3, `dedupe` + `split` took 124.75 s on the reference arm and 9.00 s on the native arm.
- The digest was identical in all six runs.

## 4. Committed kernel A/B (`test_benchmark_reference_against_qd_prep_interleaved_min_of_n`)

`QD_PREP_BENCH=1`, run at 3c90201. The input was 3,000 real examples (`before`/`after`/`diff`)
from the phase-3 defect corpus, which gives 8,279 distinct sets and 639,023 shingles.

The JSON line was displayed through `rg` and so cut at 250 columns. Its final field (the
speedup) is recomputed here from the two minima, which did survive:

```
{"benchmark": "minhash_signature", "sets": 8279, "shingles": 639023, "num_perm": 128, "rounds": 3, "reference_s": [20.907, 22.126, 14.137], "qd_prep_s": [0.401, 0.168, 0.114], "reference_min_s": 14.137, "qd_prep_min_s": 0.114, ...
```

The minima give 14.137 / 0.114 = 124×. The binary's time includes writing the request, starting
the process and reading the reply.

## 5. Full-corpus parity

This compares every distinct set the prelude signs, from both `ft_splits` mixtures. The
signatures come from `real_tokenizer_pipeline._prep_signatures` (the qd-prep binary) and from
`MinHasher.signature`.

```
PARITY {"rows": 85133, "distinct_sets": 66669, "shingles": 3038587, "mismatched": 0, "native_s": 0.93, "reference_s": 60.2}
```

## 6. The aarch64 box binary

The cross-build command was the coordinator's, and it built. Its `DT_NEEDED`, read from the
ELF dynamic section, is:

```
DT_NEEDED: ['libgcc_s.so.1', 'libc.so.6']
```

The binary was not executed on the box.

## 7. The J1-shaped epoch prelude (phase-4 full mixture, 294,988 sequences), native only

These are inputs shaped like J1, not J1's own argv:

- **Data.** The shard set is `/Users/bharath/qd-campaign/phase4-fullvocab-2026-09-30`, which
  has 294,988 train sequences. Its recipe is rev d43790d, `--general-record
  fetch-record-2026-09-29.json` (sha a0841f0d), `--general-max-rows 200000`,
  `--replay-shards` and `--defect-class`.
- **Run.** `real_ft_run.main` was given `--epoch --no-memorise --replay-partition --devices
  cpu`. It stopped at the first bucket probe (`_probe_one`), so the probes, the plan and the
  tower load are not in this time.
- **Arms.** Mac CPU, one run, native arm only. The reference arm was not run here.

```
minhash: 241886 distinct sets signed by .../target/release/qd-prep in 30.1 s (shingling and canaries included), 555870 lookups, 8 canaries equal to the reference
J1TIMING {"arm": "native", "prelude_to_first_probe_s": 502.08, "phases": {"ft_splits": [424.1, 1], "dedupe": [219.09, 1], "build_mixture": [84.0, 1], "split": [75.36, 1], "_labels": [57.29, 1], "_batch_inventory": [9.22, 1], "general_rows": [3.09, 1], "ShardReader": [2.75, 1], "check_defect_source": [2.39, 1], "corpus_facts": [2.29, 1], "_contradictions": [1.71, 1], "pair_labels": [0.93, 1], "_letter_ids": [0.42, 1], "_inventory": [0.41, 1], "vocab_letter_ids": [0.25, 1]}}
```

The `dedupe` timer wraps only `qd_data.dedupe.dedupe`. The qd-prep call happens before it,
inside `ft_splits`.

**Unexplained:** this run timed `dedupe` at 219 s and `split` at 75 s. The profiled run below
puts them at 87 s and 155 s. I have not explained the difference. Two candidates are
garbage-collection pauses landing in different phases and the shared machine's load. Their sum
is 294 s here and 242 s under the profiler.

Next, a cProfile of the same `ft_splits` rebuild with native signing. It covers what is left
once signing is native. The whole run took 452.4 s under the profiler.

```
   ncalls  tottime  cumtime  function
        1    0.766  462.126  real_ft_run.py:2620(ft_splits)
        1    1.032  165.080  qd_data/mixture.py:1240(build_mixture)
        1    3.871  155.302  qd_data/split.py:267(split)
        1    2.476  133.209  qd_data/split.py:419(_cross_split_near_duplicates)
        2   38.504  121.010  qd_data/minhash.py:320(candidate_pairs)
        1    0.030  107.025  qd_data/mixture.py:1014(drop_contradictory_prompts)
   869849   40.500  103.755  qd_data/minhash.py:94(shingle)
   314045    3.531   94.773  qd_data/render.py:667(render)
        1    2.524   86.869  qd_data/dedupe.py:245(dedupe)
  3062529   50.009   76.898  qd_data/render.py:274(_escape)
        2    1.594   42.940  real_tokenizer_pipeline.py:1457(native_minhash)
        1   12.804   14.947  qd_data/split.py:345(content_disjoint_families)
   510524    7.021    7.099  qd_data/minhash.py:196(exact_jaccard)
```

The reference arm, the probes and the box were not run. Using phase-3's measured cost per
signature (175 s / 151,812 ≈ 1.15 ms), the 555,870 signatures J1 asks for would take about 640 s
in Python on this Mac. Qd-prep took 30 s. Both of those figures are inferred, not measured.
