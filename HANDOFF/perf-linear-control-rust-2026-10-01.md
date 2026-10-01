# Perf lane: the FT linear control in Rust (2026-10-01)

Branch `perf-linear-control-rust`, off main at `9882bb7` (local, not pushed, not merged).
Everything ran on the Mac's CPU (18 cores: 6 performance, 12 efficiency; numpy 2.5.0). Nothing
ran on MPS or on the GH200, and nothing was sent to the box.

## Why

`tools/ft_linear_control.py` on one eval row of the full mixture (150,225 training docs, 11,013
val, about a dozen tasks) ran 2.3 h+ at ~26 cores on the GH200 and had not finished (the
lane's brief; not measured here). Its engine was `qd_train.baseline`:
`CharNGramHasher.transform` (an interpreted FNV-1a byte loop over every 3..5-gram) and
`LinearBaseline.fit` (four Adam fits on the L2 grid plus the refit, up to 6,000 iterations
each, on a dense BLAS operand where `n * 65,536 * 8` fits `--dense-budget-gb 24` and on a
numpy gather/bincount operand where it does not -- any task above ~49k training docs).

## What changed

Commits on the branch:

- `8f08c52` adds the engine, the adapter, the wiring and the tests.
- The commit that adds this file brings the rest:
  - the parity test's dense bound moves onto the Frobenius norm, the fixtures' task choice
    becomes deterministic, and the benchmark checks every round by (a)-(d);
  - the two comparison ledger files;
  - `AUDIT/perf-linear-control-rust-2026-10-01.md`, which holds every number below verbatim;
  - 5 `GAP-` records.

The changes themselves:

- `crates/qd-prep` gains two subcommands beside `minhash`, in the same binary (the box has no
  Rust toolchain, and one cross-built file is one thing to copy):
  - `qd-prep ngrams` -- `CharNGramHasher.transform` (`src/ngram.rs`).
  - `qd-prep linfit` -- `LinearBaseline.fit` plus the evaluation rows' logits
    (`src/linfit.rs`), over either n-gram documents (hashed in Rust) or CSR rows (the length
    control's five dense features, featurised by the reference in Python).
  - `src/pairwise.rs` is numpy's pairwise summation; `src/linwire.rs` the file formats;
    `main.rs` now runs all three subcommands through one read/atomic-write path.
  - No new dependency: `Cargo.lock` is unchanged.
- `tools/linear_control_native.py` (new) is the adapter: it replicates `LinearBaseline.fit`'s
  refusals and its two RNG draws, sends the request, and checks the reply.
- `tools/ft_linear_control.py` fits BOTH arms (n-gram gate and length control) through it.
  `QD_PREP_BIN` is resolved after the verdict checks and before the split rebuild; without it
  the tool refuses, and there is no Python fallback. The recipe gains `control_engine:
  "qd-prep linfit"` and `control_engine_sha256` (the binary's bytes) -- native MinHash records
  nothing, so this is the smallest honest marker: the fit is not byte-identical to the dense
  operand the box would otherwise have used.
- **Not changed:** `python/qd_train/baseline.py` and `python/qd_train/control_cache.py`.
  `control_cache._digest_source()` folds baseline.py's sha256 into every cache key, so editing
  it would orphan every cached control on the box. The key, its inputs and its semantics are
  unchanged, and cached entries stay valid. `--dense-budget-gb` and `--max-fit-minutes` are
  kept (campaign argv), but now they only price the refusal projection and cap the binary's
  run time.

### The RNG question

numpy's `default_rng` is PCG64 with numpy's own ziggurat normal sampler. I did not take the
"preferred" route of reproducing it in Rust. It would mean carrying numpy's ziggurat tables
and a second implementation of what the seed means, and it buys nothing: the draws take
milliseconds. So the adapter makes both draws exactly as the reference does. The permutation is
`default_rng(seed).permutation(n)` with `n_val = max(1, int(n * 0.2))`. `W0` is
`default_rng(seed).normal(0, 0.01, (d, k))`, and it is the same for every fit because
`_train_once` reseeds. Both go over the wire. The bit-identity test against
`LinearBaseline.fit` is what would catch drift between the adapter's copy of those two calls
and baseline.py.

## Parity, as defined and measured

The Rust arithmetic is the reference's SPARSE operand (`CSR.matmul` / `CSR.rmatmul`), operation
for operation. Products accumulate in bincount order (X@W per row in column order; X^T@D per
column in row order, over a transposed copy). The softmax row sums, `sum(W*W)`, `sum(gW*gW)`,
`sum(gb*gb)` and the loss sum use numpy's pairwise rule. `diff.sum(axis=0)` is sequential.
Every elementwise step keeps the reference's operand order, nothing is fused into a
multiply-add, and `exp`/`ln`/`powf` are libm's. Before writing it, I measured on this Mac that
numpy 2.5.0's float64 `exp`/`log` and Python's `**` equal `math.exp`/`math.log`/`math.pow`
(libm) on 200k values. I also measured that `np.sum` matches the pairwise-from-zero rule on 303
lengths up to 655,360, that `axis=1` matches it for row widths 1..19, that `axis=0` is
sequential, and that `bincount` is sequential (probe: `target/scratch/np_semantics.py`, not
committed).

Each output element is produced by one thread in a fixed order, so the result does not depend
on the thread count. It was **verified** identical for 1, 2, 3, 5 and 18 threads and on
repeated runs (`test_the_fit_does_not_depend_on_the_thread_count_or_the_run`). The Rust unit
tests also cover 1 to 1000 threads for hashing and 1 to 64 for the fit.

| check | result | how |
| --- | --- | --- |
| hashing: `indptr`/`indices`/`data`/`rows`, byte for byte | **identical** on 18 adversarial docs x 4 hasher configs (empty, < n, = n, CRLF, NUL/DEL, BOM/U+FFFD, combining, astral, RTL, NBSP/U+2028/U+3000, a 250k-char doc, U+0020..U+1FFF), 1/8 of the repo's sources, FT request prompts and contexts, 300 real defect-corpus prompts; a lone surrogate is refused by both | `test_*_hash_byte_for_byte` |
| fit vs the **sparse** oracle, bit for bit: W, b, loss history, iterations, converged, final grad norm, all four grid points (iterations, grad norm, val accuracy), selected L2, eval logits, predictions | **identical** on 8 fixtures (k = 2, 3, 9, 12; converged, unconverged, max_iter 0, conflicting labels with empty and short docs, 4 rows, the length features, FT prompts) and on 400 real defect prompts (777 iterations) | `test_the_fit_is_the_sparse_reference_bit_for_bit`, `test_real_defect_prompts_fit_bit_for_bit_and_within_tolerance_of_dense` |
| fit vs the **dense** oracle (what the box ran below 24 GB): (a) convergence and iterations within 1%, (b) selected L2, (c) predictions except top-2 logit gaps < 1e-9 rel, (d) Frobenius `‖ΔW‖/‖W‖` <= 1e-3 | (a)(b) identical on all 9; (c) 0 disagreements, 0 near-ties; (d) Frobenius 2e-16..1.2e-15 on the synthetic fixtures and 9.6e-8 on 400 defect prompts (largest elementwise difference 6.3e-7) | `test_the_fit_is_within_tolerance_of_the_dense_reference` |

**At a realistic size** (`target/scratch/scale_parity.py`, run once, not committed). I loaded
12,000 defect-corpus rows: 10,800 training prompts and 1,200 held out, k=4, 850 iterations.

- Native vs the **sparse** oracle: **bit-identical** on weights, bias, loss history,
  iterations (850/850), grad-norm bits, held-out logits and predictions.
- Native vs the **dense** oracle:
  - same L2, same convergence, 850/850 iterations;
  - 0 prediction disagreements on 1,200 held-out rows (smallest top-2 gap 2.3e-3);
  - held-out logits within 5.9e-6 relative;
  - Frobenius weight difference 1.54e-4, largest elementwise difference **1.08e-3**.

That last number is why (d) is stated on the Frobenius norm. It is the reference's OWN
dense-vs-sparse spread, because native equals sparse bit for bit. Adam's normalised step moves
a rarely-active feature's weight by about `lr` whatever its gradient's size, so rounding in
`X^T D` shows up there first. `test_the_fit_is_unchanged_by_densification` pins rtol 1e-9, but
only at 40 iterations on a toy set. At 850 real iterations, the reference's two operands agree
to 1e-4 overall and are identical in every decision measured here. Those three runs took 23.2 s
native, 467.8 s Python dense and 1,231.1 s Python sparse. They are single runs, not the
benchmark.

## Old vs new on a real eval row (the full `ft_linear_control`)

Eval row `784868b3` (Mac, mps, fp32, phase-3 defect_class: **45,405 train / 2,332 val** letter
docs, one task), `--defect-class commitpackft-corpus-v2 --no-repo-history --rev be30733`.
Both rows inherit the eval row's `quick` flag and promote nothing. Both carry `code_commit
9882bb7…-dirty`:

- the Python row's `code_that_ran` digest was taken at 9882bb7's code, before any edit;
- the native row ran the working tree that became `8f08c52`, with binary `79affddf…`.

| | Python engine, `bba89379` | qd-prep engine, `43a366d0` | GH200 Python, seed 0 (`eeda5db4`, `3b33c282`) |
| --- | --- | --- | --- |
| ledger file | `ledger/mac-linear-control-python-engine-2026-10-01.jsonl` | `ledger/mac-linear-control-qd-prep-engine-2026-10-01.jsonl` | `gh200-seed0-weights-2026-09-30.jsonl`, `gh200-campaign-phase0-3-2026-09-30.jsonl` |
| n-gram control: converged / iterations / L2 | yes / 845 / 1e-4 | yes / 845 / 1e-4 | yes / 845 / 1e-4 |
| final grad norm | 9.944292411500652e-05 | 9.945408220449172e-05 | 9.944766763795073e-05 |
| `linear_control_top1` | 1977/2332 | 1977/2332 | 1977/2332 |
| `length_control_top1` | 1163/2332 | 1163/2332 | 1163/2332 |
| `paired_margin_vs_linear` | +0.15008576329331047, CI [+0.1359, +0.1651] | identical | (its own eval row) +0.15008576329331047 |
| wall clock inside the recorder | 1926.0 s | 116.0 s (n-gram fit 113.1 s: 71,593,580 nonzeros, 2,375 iterations; length fit 1.9 s) | 700.8 s (box) |

The three grad norms differ in the 4th-5th digit. They come from three operands: Accelerate
dense, sparse (native), and the box's dense BLAS. Every decision-bearing number is identical.
Per-row identity of the 2,332 control verdicts is **inferred, not verified**: the per-task
counts, the margin and its bootstrap CI (same seed) are equal, but neither row stores per-row
verdicts and the Python run was not repeated with `--control-cache`. The two wall clocks are
N=1 each, and the machine was contended: the Python run overlapped my test runs, and the native
run overlapped the Python run. **They are not the speedup claim.** The committed A/B below is.

**The GH200's own phase-3 eval rows, seeds 0-2.** These are the rows whose Python (OpenBLAS
dense) controls the box wrote: `eeda5db4`, `2b08a357` and `635c19d9` in
`ledger/gh200-seed0-weights-2026-09-30.jsonl`. I scored them on the Mac with the native
engine, using the box's local verdicts (`box-final-2026-09-30/phase3/verdicts-s{0,1,2}-weights
.jsonl`) and the campaign's argv. Seeds 1 and 2 have their own permutations and W0, so these
are three independent comparisons:

| seed | GH200 Python: iterations / grad norm | native: iterations / grad norm | top1 n-gram, length (both) | margin, CI (both) |
| --- | --- | --- | --- | --- |
| 0 | 845 / 9.944766763795073e-05 | 845 / 9.945408220449172e-05 | 1977/2332, 1163/2332 | +0.15008576329331047 [+0.1359, +0.1651] |
| 1 | 846 / 9.905473550717985e-05 | 846 / 9.905635742422174e-05 | 1977/2332, 1163/2332 | +0.14965694682675815 [+0.1355, +0.1642] |
| 2 | 847 / 9.950111892916253e-05 | 847 / 9.950334844114837e-05 | 1977/2332, 1163/2332 | +0.14965694682675815 [+0.1355, +0.1642] |

Every row has L2 1e-4. The native rows' recorded wall clocks are 96.6, 90.7 and 90.9 s on the
Mac, with n-gram fits of 93.9, 88.1 and 88.2 s. The box's Python rows record 700.8, 691.3 and
707.7 s. Those are two different machines, so that is not a speedup claim.

The three native rows are **not committed**; their summaries are verbatim in the AUDIT file.
They would be non-quick supplements of non-quick GH200 eval rows, written from a Mac on a dirty
tree. Promotion reads an eval row and its supplements as one unit (`SUPPLEMENT_KEY`), so adding
them to `ledger/` is a decision for a human, not for this lane.

## Benchmark (committed, interleaved A/B, min of N)

This is `test_qd_prep_linear_parity.py::test_benchmark_*`, reference arm first in every
round:

    QD_PREP_BENCH=1 QD_PREP_BENCH_DEFECT_ROWS=12000 QD_PREP_BENCH_ROUNDS=2 \
      pytest -s python/tests/test_qd_prep_linear_parity.py -k benchmark

The Mac was otherwise idle (log `target/scratch/bench2.log`, not committed).

- **Hashing:** `CharNGramHasher.transform` against `qd-prep ngrams` over all the docs.
- **Fit:** `LinearBaseline.fit` against `qd-prep linfit`, the whole fit including
  featurisation, on the first 90% of the docs, scoring the last 10%. The reference runs at the
  box's `--dense-budget-gb 24`, so it is dense wherever that fits.

The binary's times include writing the request, the process start and reading the reply. Every
round's answers were checked outside the timers: hashing byte for byte, and the fit by
(a)-(d).

| fixture | docs | stage | reference runs (s) | qd-prep runs (s) | min / min |
| --- | ---: | --- | --- | --- | ---: |
| (i) FT request prompts, `code.commit_intent/implements_claim` (synthetic rows, real renderer) | 30 | hashing | 0.050, 0.057, 0.053 | 0.062, 0.009, 0.009 | 0.050 / 0.009 = **5.4x** |
| | | fit | 2.497, 2.540, 2.524 | 0.765, 0.770, 0.772 | 2.497 / 0.765 = **3.3x** |
| (ii) real defect-corpus prompts (12,000 rows, k=4, 856 chars mean, 1,547 nonzeros/doc) | 12,000 | hashing | 11.304, 11.232 | 0.284, 0.301 | 11.232 / 0.284 = **39.6x** |
| | | fit (10,800 train, 850 iterations) | 383.715, 412.924 | 23.099, 23.795 | 383.715 / 23.099 = **16.6x** |

The checks on (ii) were the same in both rounds: same L2, converged, 850/850 iterations, 0
prediction disagreements on 1,200 held-out rows, held-out logits within 5.9e-6, Frobenius
weight difference 1.5e-4. Fixture (i) is too small to measure anything but process overhead.

Three things the table does **not** say:

- The box's slow case is the SPARSE Python operand: above ~49k training docs per task, dense
  does not fit 24 GB. On (ii), Python sparse took 1,231.1 s against 23.2 s native, about 53x,
  but that is a single run, not in the committed A/B.
- The Mac has 6 performance and 12 efficiency cores. Thread scaling there was 62.8 s on 1
  thread, 17.3 s on 6 and 9.4 s on 18, for one 3,000-doc fit run while the Python full run
  was also running. I did not measure scaling on the GH200's 72 Neoverse cores.
- I did not measure on the box, so the GH200 speedup is **unverified**.

## Tests (all run on this Mac)

- `cargo test --release -p qd-prep`: **29 passed** (9 new in `ngram`, `pairwise`, `linfit`,
  `linwire`). `cargo clippy --release -p qd-prep --all-targets`: clean. `cargo fmt --check`:
  clean.
- pytest (the runner from the brief, `-k "not mps"`) over 13 files: **310 passed, 4
  skipped**. The files are `test_qd_prep_linear_parity.py` (new), `test_ft_linear_control.py`,
  `test_length_control.py`, `test_control_cache_key_agreement.py`, `test_margin_probe_row.py`,
  `test_tool_call_sites.py`, `test_qd_prep_minhash_parity.py` (the minhash path through the
  refactored `main.rs`), `test_baseline_dense_operand.py`, `test_eval_harness.py`,
  `test_ledger.py`, `test_wall_clock.py`, `test_ledger_provenance.py` (with the two new ledger
  files) and `test_gaps_ledger.py`. The 4 skips are the opt-in benchmarks, two here and two in
  minhash, skipped without `QD_PREP_BENCH=1`. With it set, both of this file's benchmarks
  passed (above).
- **Verified failing pre-fix:** I swapped in `tools/ft_linear_control.py` at `9882bb7` and ran
  the 6 new tool tests. All 6 failed: `KeyError: 'control_engine'`, no refusal without
  `QD_PREP_BIN` (twice), and `TypeError` on the `engine` keyword (three times). With the fixed
  file they pass.
- ruff 0.16.8: clean on every touched Python file.

## Cross-built binary for the box (not copied, not run there)

I built it with the brief's command, into a new target dir, so the binary the box uses was not
overwritten:

    /Users/bharath/qd-campaign/target-aarch64-linux-lc/aarch64-unknown-linux-gnu/release/qd-prep
    sha256 af9d3a7625347ae65f7fa25c44d92dfa3ba2dfe969c254ca76abfc721a7887c2
    ELF 64-bit LSB pie executable, ARM aarch64, dynamically linked, /lib/ld-linux-aarch64.so.1

The binary carries the new subcommands. Its bytes hold the request magics `QDPLFIN1`,
`QDPNGIN1` and `QDPMHIN1`, plus the `linfit` and `ngrams` help strings; I checked with `rg -a`,
since an aarch64 binary cannot be executed here.

GLIBC requirement: the highest symbol version is **GLIBC_2.34**. I read it with the rustup
toolchain's `llvm-objdump -T`; the full set is 2.17, 2.18, 2.28, 2.29, 2.30, 2.32, 2.33, 2.34.
The sysroot is the box's own glibc 2.39. I built it from the source at `8f08c52`: the binary
postdates the last edit to `crates/qd-prep/src`, and cargo found nothing to rebuild at commit
time. The Mac build used for every Mac number is
`target/release/qd-prep` = `79affddf…7a1`. That is the `control_engine_sha256` on row
`43a366d0`.

## Open (gap ids)

- `GAP-LINEAR-CONTROL-NATIVE-PARITY-UNMEASURED-ON-THE-BOX`: bit-identity with the sparse
  oracle is measured on the Mac only. On the box, numpy's float64 exp/log/`**` and its
  reduction order are not measured. The binary links glibc 2.39's libm.
- `GAP-LINEAR-CONTROL-PROJECTION-IS-THE-PYTHON-ENGINES`: `--max-fit-minutes` still refuses on
  the Python-calibrated projection. It over-refuses, which is the safe direction, and it is off
  by default.
- `GAP-LINEAR-CONTROL-CACHE-HIT-NAMES-NO-ENGINE`: the cache key does not name the engine, just
  as it already does not name the dense budget. This is a human decision.
- `GAP-LINEAR-CONTROL-OTHER-CALLERS-STILL-FIT-IN-PYTHON`: `fit_linear_control`,
  `rung0_linear_control`, `rung0_real_run`, `after_vs_diff` and `operator_holdout` still fit in
  Python. The qd-prep subprocess runner is near-duplicated with `_prep_signatures`.
- `GAP-PERF-LINEAR-CONTROL-NAVIGATION-2026-10-01`: there is no DevMap store in the worktree,
  so I used the main index. GitPulse returned `REPOSITORY_TRUST_REQUIRED` on every facet, and
  `ListAgents` was unavailable.
- Known cost I did not optimise: per iteration, the pairwise sums over `d*k` (`sum(W*W)` for
  the loss and `sum(gW*gW)` for the norm) run on one thread. I timed one sum of squares on the
  Mac as min of 20, in a scratch Rust test that I did not commit:
  - 0.060 ms at 262,144 values (k=4);
  - 0.149 ms at 655,360 values (k=10);
  - 1.910 ms at 9,830,400 values (k=150, a CLINC-sized label space).

  That is small next to the 47.6 ms per iteration of the 45k-doc fit. The pairwise tree splits
  at fixed points, so it can be evaluated in parallel without changing a bit if it ever
  matters.

## First command for the next lane

Copy the cross-built binary to the box. Point `QD_PREP_BIN` at it: it is a superset of the
binary the box runs, its `minhash` is unchanged, and `test_qd_prep_minhash_parity.py` passes
with it. Then measure parity there before re-running the stalled control:

    QD_PREP_BIN=<box path to the new qd-prep> /home/ubuntu/qd-venv/bin/python -m pytest \
      python/tests/test_qd_prep_linear_parity.py -o addopts= -q
