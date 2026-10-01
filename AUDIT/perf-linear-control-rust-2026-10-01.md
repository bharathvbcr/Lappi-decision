# Evidence: the FT linear control in Rust (2026-10-01)

These are the outputs behind every number in `HANDOFF/perf-linear-control-rust-2026-10-01.md`.
Where a section says "verbatim", every number and word is as printed. Long lines are wrapped
and indented, though, and the temp-file paths after `->`, the ledger path after `wrote row`
and the trailing `state='ran'` are dropped. Section 6 is condensed further, as it says. All of them come from the Mac: 18 cores,
numpy 2.5.0, CPython 3.14.7, on branch `perf-linear-control-rust`. The two ledger rows are in
`ledger/mac-linear-control-python-engine-2026-10-01.jsonl` (`bba89379`) and
`ledger/mac-linear-control-qd-prep-engine-2026-10-01.jsonl` (`43a366d0`).

## 1. numpy's float64 semantics, probed before the port was written

The probe compares numpy's results with pure-Python transliterations of each candidate rule.
`pw` is numpy's `pairwise_sum` (8 accumulators, blocks of 128, recursive split at
`n/2 - (n/2) % 8`). `first+pw` is the first element plus `pw` over the rest. `seq` is a
sequential sum from 0.0. The output, verbatim:

    exp == math.exp: True
    log == math.log: True
    exp 2D == math.exp: True
    flat np.sum: pairwise-from-zero 303/303, first+pairwise(rest) 121/303
    2D sum (65536, 10) True True False
    2D sum (65536, 2) True True True
    2D sum (1000, 7) True True False
    2D sum (5, 3) True True True
    axis=1 k=1..7: seq True pw True
    axis=1 k=8..19: seq False pw True
    axis=0 n in (5, 9, 200, 5000), k in (2, 4, 10): seq True (pw False for n >= 9)
    bincount sequential: True
    bincount unsorted sequential: True
    pow: True

I folded the per-k lines into ranges; every line inside a range printed the same thing. For
`axis=1`, `first+pw` was False from k=3 up. The pow check covered `0.9 ** it` and
`0.999 ** it` against `math.pow` for it in 1..6999. The exp and log checks used 200,000 values
each.

## 2. The fit against the dense oracle: the committed tests' printout

The source is `pytest -s -k dense` on `test_qd_prep_linear_parity.py`. Every fixture had the
same L2 and the same iterations in both arms, 0 prediction disagreements and 0 near-ties.

| fixture | iterations | Frobenius `‖ΔW‖/‖W‖` | max elementwise `|ΔW|/max|W|` |
| --- | ---: | ---: | ---: |
| conflicting-empty-short | 128 | 1.2496844755408687e-15 | 1.0393835615546233e-15 |
| four-rows-two-classes | 708 | 4.636629379493786e-16 | 6.460060437025227e-16 |
| ft-request-prompts | 875 | 3.878271679850073e-16 | 4.0238548750362426e-16 |
| keywords-k12-unconverged | 60 | 2.271904455679884e-16 | 3.5826328958867577e-16 |
| keywords-k3-converges | 173 | 2.0986405317183355e-16 | 1.1764378054972604e-16 |
| keywords-k9-pairwise-block | 400 | 2.739354414392816e-16 | 4.0411744636215276e-16 |
| length-features | 1355 | 4.504121183708948e-16 | 5.071305166895742e-16 |
| max-iter-zero | 0 | 0.0 | 0.0 |
| defect-corpus-400-rows | 777 | 9.582517636915695e-08 | 6.257533367201413e-07 |

The ft-request-prompts row was re-run alone after its task choice was made deterministic, so
it is the committed fixture's.

## 3. At a realistic size: native, sparse oracle and dense oracle

This is a single run of a script, not committed: it fits the three arms on 12,000
defect-corpus rows (10,800 training prompts, 1,200 held out). The output, verbatim:

    {"rows": 12000, "train_docs": 10800, "held_docs": 1200, "classes": 4, "native_s": 23.2,
     "python_dense_s": 467.8, "python_sparse_s": 1231.1,
     "native_vs_sparse_bit_identical": {"weights": true, "bias": true, "history": true,
       "iterations": [850, 850], "grad_norm_bits_equal": true, "held_logits": true,
       "predictions": true},
     "dense_vs_native": {"l2": [0.0001, 0.0001], "converged": [true, true],
       "iterations": [850, 850], "grad_norm": [9.979158572601117e-05, 9.977742492838338e-05],
       "max_weight_rel_diff": 0.001083022649059546,
       "frobenius_weight_rel_diff": 0.00015428824440311213,
       "max_held_logit_rel_diff": 5.872341009262769e-06, "prediction_disagreements": 0,
       "disagreement_top2_gaps": [], "rows_with_top2_gap_below_1e-9": 0,
       "min_top2_gap": 0.002324607116545706}}

## 4. The committed benchmark

The command:

    QD_PREP_BENCH=1 QD_PREP_BENCH_DEFECT_ROWS=12000 QD_PREP_BENCH_ROUNDS=2 \
      pytest -s python/tests/test_qd_prep_linear_parity.py -k benchmark

The result was `2 passed, 35 deselected in 881.16s`. Below, the two printed lines with each
fit round's check dict folded: the dicts were identical in every round, and they are given
once after the line.

    {"benchmark": "linear_control", "fixture": "ft-request-prompts:code.commit_intent/implements_claim",
     "docs": 30, "classes": 2, "chars_mean": 1578, "nnz_per_doc_first_500": 1456, "host_cpus": 18,
     "hashing": {"rounds": 3, "reference_s": [0.05, 0.057, 0.053], "qd_prep_s": [0.062, 0.009, 0.009],
                 "speedup_min_over_min": 5.4, "checked": ["byte-identical" x3]},
     "fit": {"rounds": 3, "reference_s": [2.497, 2.54, 2.524], "qd_prep_s": [0.765, 0.77, 0.772],
             "speedup_min_over_min": 3.3}}
      fit check (each round): l2 [1e-4, 1e-4], converged [true, true], iterations [757, 757],
      held_rows 3, disagreements 0, min_top2_gap 1.9999543603800956,
      max_held_logit_rel_diff 1.5119973451361337e-14,
      frobenius 4.051234088563643e-16, max elementwise 3.9364816585277164e-16

    {"benchmark": "linear_control", "fixture": "defect-corpus:12000-rows", "docs": 12000,
     "classes": 4, "chars_mean": 856, "nnz_per_doc_first_500": 1547, "host_cpus": 18,
     "hashing": {"rounds": 2, "reference_s": [11.304, 11.232], "qd_prep_s": [0.284, 0.301],
                 "speedup_min_over_min": 39.6, "checked": ["byte-identical" x2]},
     "fit": {"rounds": 2, "reference_s": [383.715, 412.924], "qd_prep_s": [23.099, 23.795],
             "speedup_min_over_min": 16.6}}
      fit check (each round): l2 [1e-4, 1e-4], converged [true, true], iterations [850, 850],
      held_rows 1200, disagreements 0, min_top2_gap 0.002324607116545706,
      max_held_logit_rel_diff 5.872341009262769e-06,
      frobenius 0.00015428824440311213, max elementwise 0.001083022649059546

## 5. The full `ft_linear_control` on eval row `784868b3`

The native run's binary summary lines, verbatim:

    task code.defect_class/defect_class (linear_control): qd-prep linfit: 45405 rows x 65536 cols,
      71593580 nonzeros, 2332 eval rows; features 0.21 s, fit 113.05 s on 18 thread(s); grid
      [l2=0.0001 7730/9081 converged in 842; l2=0.001 6048/9081 converged in 330; l2=0.01
      4186/9081 converged in 157; l2=0.1 4178/9081 converged in 201]; refit l2=0.0001 converged
      in 845 iterations (grad norm 9.945408e-5)
    task code.defect_class/defect_class (length_control): qd-prep linfit: 45405 rows x 5 cols,
      227025 nonzeros, 2332 eval rows; features 0.00 s, fit 1.92 s on 18 thread(s); grid
      [l2=0.0001 4686/9081 converged in 1177; l2=0.001 4586/9081 converged in 403; l2=0.01
      4178/9081 converged in 131; l2=0.1 4178/9081 converged in 135]; refit l2=0.0001 converged
      in 1175 iterations (grad norm 9.980034e-5)
    paired_margin_vs_linear: Ran(passed=True, value=0.15008576329331047, n=2332, n_total=2332,
      detail='paired margin +0.1501, 95% CI [+0.1359, +0.1651] over 10000 bootstrap resamples')
    wrote row 43a366d0-e15b-4d68-81c8-dc8b59993206
    133.69 real  1647.46 user  10.48 sys   (/usr/bin/time -l)

The Mac thread scaling quoted in the handoff comes from one scratch run per thread count, on
3,000 defect prompts (4,702,443 nonzeros, 2,354 iterations), while the Python full run was
also running:

    fit 62.80 s on 1 thread(s); fit 17.32 s on 6 thread(s); fit 9.39 s on 18 thread(s)

## 6. The native engine against the GH200's own eval rows

These ran on the Mac, against `ledger/gh200-seed0-weights-2026-09-30.jsonl` and
`box-final-2026-09-30/phase3/verdicts-s{0,1,2}-weights.jsonl`. They wrote scratch ledgers,
which are not committed; the handoff says why. The summaries are condensed from
`target/scratch/ledger_rows.py`'s output and the binary's `fit ... s` lines: every number is as
printed, but metric names are shortened and `value n n_total` is written `n/n_total`.

    8095435c eval 6d170b3c  wall 96.56779583299794
      length_control_top1 1163/2332; linear_control_convergence 9.945408220449172e-05
      "converged in 845 iterations at l2=0.0001"; linear_control_top1 1977/2332
      paired_margin_vs_linear.choice 0.15008576329331047 "+0.1501, 95% CI [+0.1359, +0.1651]"
      n-gram fit 93.93 s; length fit 1.76 s
    6f932163 eval 60f29b07  wall 90.65052925000055
      length_control_top1 1163/2332; linear_control_convergence 9.905635742422174e-05
      "converged in 846 iterations at l2=0.0001"; linear_control_top1 1977/2332
      paired_margin_vs_linear.choice 0.14965694682675815 "+0.1497, 95% CI [+0.1355, +0.1642]"
      n-gram fit 88.08 s; length fit 1.75 s
    79d27778 eval 5c19c0e8  wall 90.85180258299806
      length_control_top1 1163/2332; linear_control_convergence 9.950334844114837e-05
      "converged in 847 iterations at l2=0.0001"; linear_control_top1 1977/2332
      paired_margin_vs_linear.choice 0.14965694682675815 "+0.1497, 95% CI [+0.1355, +0.1642]"
      n-gram fit 88.24 s; length fit 1.82 s

The box's Python rows for the same eval rows, from `ledger/gh200-seed0-weights-2026-09-30.jsonl`:

    eeda5db4: 845 iterations, 9.944766763795073e-05, 1977/2332, 1163/2332, +0.15008576329331047
    2b08a357: 846 iterations, 9.905473550717985e-05, 1977/2332, 1163/2332, +0.14965694682675815
    635c19d9: 847 iterations, 9.950111892916253e-05, 1977/2332, 1163/2332, +0.14965694682675815

## 7. The Python run on eval row `784868b3`

The Python run's lines, verbatim:

    paired_margin_vs_linear: Ran(passed=True, value=0.15008576329331047, n=2332, n_total=2332,
      detail='paired margin +0.1501, 95% CI [+0.1359, +0.1651] over 10000 bootstrap resamples')
    wrote row bba89379-3dad-4cf9-8823-7426975d1eb5
    1940.95 real  3054.53 user  86.30 sys   (/usr/bin/time -l; peak RSS 26.7 GB)
