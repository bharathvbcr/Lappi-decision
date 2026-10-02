# HANDOFF — L-cuda-gdn: K2(i), tessl's GDN training scan on CUDA at the published rule (2026-10-01)

Lane L-cuda-gdn of "train Lappi with ojas". It ports tessl's token-sequential GDN training scan
(`tessl/kernels/gdn_train.metal`, host `tessl/src/gdn_train.rs`) to CUDA: forward, backward and
the backward's finish, at the **published** rule (decayed read). This is option (i) of
`cuda-backend-scoping.md` §3.2 and Fable's decision 6 (`fable-cuda-asks.md`).

**CPU-side work on a Mac with no NVIDIA GPU. No device code has run.** The 10 device tests are
NOT RUN. **No ledger row was written**, because nothing was measured on a GPU.

Labels: **[V]** I ran it or read it (cited). **[I]** inferred, with the reasoning. **[U]**
unverified.

## What was built

All files are in `/Users/bharath/Code/research/ojas/ojas-qwen35-cuda/`. No git was run in ojas,
so nothing there is committed. The source state is pinned by sha256 (raw list:
`AUDIT/ojas-training-2026-10-01/l-cuda-gdn-cross-build.txt`, "source hashes at LATE build time",
which supersedes the earlier "source hashes at build time").

| File | Lines | sha256 (prefix) | What |
| --- | ---: | --- | --- |
| `src/gdn_plan.rs` | 641 | `2d4d0977` | Host plan: variable lengths, offsets, checkpoint layout, buffer lengths, refusals, geometry |
| `src/gdn_kernels.rs` | 556 | `b09fb0d1` | CUDA-C module `k2_gdn_published_scan`: `qd_gdn_published_fwd`, `_bwd`, `_bwd_finish` |
| `src/gdn_host.rs` | 1002 | `ed89e4ff` | Bitwise host mirror of the three kernels; test cases; the pre-registered bounds |
| `src/gdn.rs` (`cuda`) | 360 | `a82799cf` | Device launches through the plan: `GdnPublishedLayout`, `GdnPublishedWorkspace`, forward, backward |
| `src/gdn_smoke.rs` (`cuda`) | 511 | `b01f2ecf` | `gdn_published_checks` (runga hook), device-run helpers, report-only timing |
| `tests/device_gdn_published.rs` (`cuda`) | 396 | `65bf6116` | 10 `#[ignore]` device tests |
| `tests/device_gdn_published_mirror.rs` | 169 | `727b88ef` | 8 host tests: the mirror against the float64 reference |
| `tests/device_gdn_published_common/mod.rs` | 404 | `86a4c675` | Shared test support: the embedded corpus, the f64 reference driver, the judges |

The shared module sits in a directory under `tests/`. That path is outside the literal
`tests/device_gdn*.rs` pattern of the brief, so here it is by name. cargo does not build it as a
target; the two GDN test files `mod` it.

**`src/lib.rs`: two Edits, both approved.**
- The first added the five lines at the shared spot.
- The second, which the lead approved (because `cargo fmt --check` flagged the order), moved only
  my own lines into rustfmt order. No other lane's line was touched; lib.rs was re-read before each
  Edit.

My lines are:

    pub mod gdn_host;        (non-cuda group, after `pub mod error;`)
    pub mod gdn_kernels;
    pub mod gdn_plan;
    #[cfg(feature = "cuda")]
    pub mod gdn;             (cuda group, after `pub mod buffer;`)
    #[cfg(feature = "cuda")]
    pub mod gdn_smoke;

Not edited by this lane: `Cargo.toml`, `Cargo.lock`, `src/kernels.rs`, `src/bin/rung0.rs`,
`runga`, README, `tests/reference/**`, `tests/fixtures/**`, and every other lane's file.

`Cargo.toml` and `Cargo.lock` **did** change during this lane, and the change was not mine [V, by
mtime and content]. At 20:12, L-cuda-M1's Cargo.toml switched from `ojas-core` to `ojas-io`, and
cargo regenerated the lock. It went from M0's pinned 1,107 bytes (`2f15711f…`) to 1,186 bytes (md5
`2c7ce4f0a22bfc10cfbec855c7c4d216`). My code depends on neither crate.

**Late fixes (after commit `5475de0`).**
- **`src/gdn.rs`:** the `SAFETY:` comments now count each launch's parameters correctly. The
  forward has 15 (11 device pointers and 4 `unsigned int`). The backward has 22 (17 device
  pointers, 4 `unsigned int` and one `unsigned long long`). The finish has 12.
- **`src/gdn_kernels.rs`:** the only library call left in the CUDA body, `min(T, t0 + 64)`, is now
  an explicit ternary.
- After the fixes, a scan of the body's call sites (comment lines excluded) finds only these [V]:
  - `__fadd_rn`, `__fsub_rn`, `__fmul_rn`, `__fdiv_rn`, `__fsqrt_rn`, `__shfl_xor_sync` and
    `__syncthreads`;
  - the module's own `qd_*` helpers, plus the crate's `qd_exp_nonpos`.
- The new source hashes are in the table above and in the cross-build file.

## Design, with the evidence

1. **The layout is one layout: the concatenation of the `B = 1` layouts.**
   - Tokens are packed `[N, H, D]`, with `tok_off[B+1]`.
   - Each sequence's checkpoints are a block `[H, NC_b, 128, Dv]` at `ck_off[b]`, with
     `NC_b = ceil(T_b / 64)`.
   - States are `[B, H, 128, Dv]`.
   - With equal lengths this is tessl's dense `[B,T,H,D]` / `[B,H,NC,128,Dv]` index for index
     (`published_plan_with_equal_lengths_is_tessls_dense_layout`) [V]. So the oracle's reference
     compares directly, and a variable-length batch compares against per-sequence `B = 1` runs.
2. **Structure: tessl's, term for term.**
   - One 128-thread block per (16-column value slice, sequence × head); thread `i` owns state row
     `i`.
   - A checkpoint before each 64th token's decay.
   - The backward recomputes each chunk from its checkpoint into a per-block scratch, then runs the
     reverse recurrence, writes per-slice partials, and a finish kernel sums them in slice order and
     applies the l2norm backward.
   - The two alternating `red` buffers and the barriers are where tessl has them
     (`gdn_train.metal:45-59,195-240`).
3. **Geometry never reads the SM count.**
   - Scan: grid `(Dv/16, B*H)`.
   - Finish: `min(rows, 65,535)` blocks, block-striding over rows.
   - `B*H ≤ 65,535` is refused in the plan (gridDim.y).
4. **Every float operation in the scan is an explicitly rounded intrinsic**: `__fadd_rn`,
   `__fsub_rn`, `__fmul_rn`, `__fdiv_rn`, `__fsqrt_rn`.
   - `rsqrt` is `1/sqrt`, both correctly rounded.
   - Column sums are a fixed tree: a `__shfl_xor_sync` butterfly at offsets 16,8,4,2,1, then
     `(w0+w1)+(w2+w3)`.
   - So the bits depend only on the inputs. `gdn_host` replays them in f32 **bit for bit**, which
     makes the whole algorithm testable on the Mac and gives a bitwise device oracle.
   - Bitwise parity with Metal is not a goal: `simd_sum` has no specified order (§5.1).
5. **`exp(g)` is the crate's one device exponential.** It is L-cuda-M1's `qd_exp_nonpos`
   (`src/k8_act.rs`, spliced via `crate::act_prelude!()`), mirrored by
   `k8_act::exp_nonpos_f32`.
   - I first wrote a local IEEE-only exp. The lead ruled one exp per crate, so I deleted mine and
     swapped (`the_published_decay_uses_the_crates_one_exp`).
   - M1 measured it at 0.918 ulp on `[-104, 0]`.
   - My pre-registered 2 ulp (CUDA `expf`'s documented max) is re-checked on it on the host:
     `the_published_decay_exp_is_within_its_pre_registered_ulps` [V].
   - **Precondition `g ≤ 0`.** A positive or NaN `g` gives NaN, and the NaN propagates to every
     output it reaches (`a_positive_log_decay_is_loud_in_the_published_mirror`) [V].
6. **Absent optional buffers are null pointers plus a flags word.** These are `s0`, `s_fin`,
   `d_fin` and `ds0`. Their `SAFETY:` comments say the kernel touches each only under its flag.
   - Outputs are `&mut` borrows of whole buffers, so an output cannot alias an input.
   - The backward refuses:
     - a workspace sized for another plan;
     - an `s0` / `ds0` that does not pair;
     - any length mismatch (tessl `gdn_train.rs:324-342`).

## Pre-registered bounds (`gdn_host::published_bounds`, written before any run)

| Bound | Value | Citation and derivation |
| --- | --- | --- |
| Every output vs the oracle's float64 reference (o, s_fin, ckpt, dq, dk, dv, dg, dbeta, ds0) | `≤ 1e-4 · max\|ref\|` per tensor | tessl's Metal-vs-f64 bound, `tessl/tests/gdn_train.rs:227-249` (`rel <= 1e-4`, `:240-241`), over the same edges `:365-371` and the 2B-heads case `:378-382`. Against the f32 accumulation width (u = 2^-24): the depth-7 column-sum tree is ≤7u ≈ 4.2e-7 of the magnitude sum, and the backward's partials add ≤(16+8)u ≈ 1.4e-6. The recurrence is a contraction, so per-token errors do not compound. The worst-case `γ_T = T·u` (4.9e-4 at T=8191) exceeds the bound, but the same operator's observed f32 error on tessl's corpus is 11u–24.7u (`tessl/tests/gdn_fixtures.rs:36-47`). That leaves >60× headroom while failing the repo rule by >100× |
| Corpus forward vs the corpus's own `y_seq_f64`, after undoing the seam's q normalization in f64 | `≤ 1e-4 · max\|golden\|` | The only operator difference is the seam re-normalizing an already unit `k` (factor 1 − 5e-7): a ~1e-6 coefficient perturbation of a contraction |
| Device vs host mirror, and run vs run | bit-identical | The kernels round every operation explicitly |
| Decay exp vs `f64::exp` | ≤ 2 ulp | CUDA `expf`'s documented maximum |

## Tests: run vs NOT RUN

Raw outputs are `AUDIT/ojas-training-2026-10-01/l-cuda-gdn-*.txt`.

| Suite | Ran on the Mac | Result | Evidence |
| --- | --- | --- | --- |
| GDN unit tests in `src/gdn_{plan,kernels,host}.rs` | yes | **17 passed** | `l-cuda-gdn-unit-tests-after-rule9-rename.txt` (18 lines incl. M1's `k8_act` exp test) |
| `tests/device_gdn_published_mirror.rs` (host mirror vs f64) | yes | **8 passed** | `l-cuda-gdn-mirror-first-run.txt` (the first run after the bounds were written), `l-cuda-gdn-host-tests.txt` |
| Whole crate, no feature | yes | lib 91, mirror 8, plus other lanes' targets; 0 failed | `l-cuda-gdn-host-tests.txt` |
| Whole crate, `--features cuda` | yes | lib 93, rung0 4, mirror 8, refusal 1, …; 0 failed | `l-cuda-gdn-cuda-feature-tests.txt` |
| **Late re-run** of the whole crate, after the late fixes and M1's ojas-io migration | yes | no feature: lib 104, mirror 8, plus other lanes' targets; `--features cuda`: lib 104, rung0 4, mirror 8, refusal 1, …; 0 failed in both. The lane's 17 unit tests and 8 mirror tests are all among them, and pass | `l-cuda-gdn-late-rerun.txt` |
| `tests/device_gdn_published.rs` | **NOT RUN** | **10 ignored** (need sm_90), in both runs | same |
| `gdn_smoke::gdn_published_checks`, the timing | **NOT RUN** | — | — |

**Totals for this lane:** 25 distinct tests run on the Mac and passing (17 + 8). **10 device tests
NOT RUN.**

**Measured on the host mirror** (the device's arithmetic, on the Mac) [V,
`l-cuda-gdn-mirror-first-run.txt`]:
- Worst over every output, every case: **9.56e-7** of max|ref|, at the T=1 cases.
- Chunk edges B=2 / B=1: ≤ 3.4e-7 / 9.5e-7.
- 2B heads (B=1, T=200, H=16, Dv=128): 2.6e-7.
- Variable-length batches: ≤ 9.6e-7.
- The 13-case corpus: ≤ 2.7e-7 vs the seam reference and ≤ 7.6e-7 vs the golden; the float64
  reference's own offset from the golden is ≤ 6.0e-7, which is the k re-normalization.
- Mirror o vs the repo rule: **3.87e-2**, against 1.27e-7 vs the published reference
  (`published_mirror_is_outside_the_bound_from_the_repo_rule`).

**The device tests** (`tests/device_gdn_published.rs`), every one `#[ignore]`, every name says
`published`:
- `device_checks_pass`, the runga hook;
- `matches_the_f64_reference_across_chunk_edges_b2` and `_b1`: T ∈ {1, 63, 64, 65, 130} with
  tessl's flags;
- `matches_the_f64_reference_at_qwen35_2b_heads`;
- `varlen_batch_matches_references_and_its_b1_launches_bitwise`: three batches, including H=16
  Dv=128;
- `matches_the_published_corpus`: 13 cases, forward and backward, golden and mirror; the corpus is
  `include_bytes!`-embedded and sha-checked against tessl's sums, because the box has no checkout;
- `forward_without_s_fin_writes_the_same_o_and_ckpt`;
- `is_bit_identical_over_five_runs_at_2b_heads_batched`;
- `refuses_what_the_kernels_cannot_run`;
- `timing_b1_vs_batched_report_only`.

Every f64 check also checks:
- sentinel prefill;
- bit equality with the mirror;
- two repeat runs.

## Fail-first (each mutation planted, run, then restored from a byte copy, sha re-checked)

| Mutation | Caught by | Evidence |
| --- | --- | --- |
| M1: mirror reads the **undecayed** state (the repo rule), in the forward and the recompute | 7 of 8 mirror tests fail. The 8th, the generator test, has no GDN in it. First failure: B=1 T=63 `o` at 5.3e-2 of max; the corpus at 2.8e-1 | `l-cuda-gdn-failfirst-m1-mirror-repo-read.txt` |
| M2: the CUDA source's forward reads before it decays | `the_published_rule_decays_before_the_read_in_the_forward_and_the_recompute` fails: "the read of S precedes its decay (the repo rule)" | `l-cuda-gdn-failfirst-m2-kernel-source-repo-read.txt` |
| M3: mirror writes the checkpoint **after** the token's decay | 6 tests fail on `ckpt` (up to 7.7e-1 of max). The repo-rule contrast (o only) and the generator test pass, as expected | `l-cuda-gdn-failfirst-m3-mirror-ckpt-after-decay.txt` |
| Reverted | all green again; sha256 equals the pre-mutation copy | `l-cuda-gdn-host-tests.txt` |

## Clippy, format, cross-build

- **Clippy.** `cargo clippy --offline --all-targets --features cuda -- -D warnings`, and the same
  without the feature, both in a fresh target dir: **exit 0** (`l-cuda-gdn-clippy.txt`). Earlier, M1's
  `k8_act.rs` tripped `excessive_precision`; I told M1, M1 fixed it, then I ran these.
  - **Late re-run, both modes: exit 0** (`l-cuda-gdn-late-rerun.txt`).
    - I touched my lib and test files first, so clippy re-linted every target instead of replaying
      a cached result.
    - Before that, two lints in M1's files had blocked the lint of the test targets:
      `k11_host.rs:338` `type_complexity` and `k11_golden.rs:715` `useless_conversion`. I told M1;
      M1 fixed both.
- **Format.**
  - `rustfmt --check` on my 5 src files: exit 0.
  - My 3 test files, checked as scratch copies with `reference` stubbed so rustfmt cannot reach the
    oracle's files: exit 0.
  - `cargo fmt --check` on the crate flags **only `src/npy.rs`**, another lane's new file
    (`l-cuda-gdn-fmt.txt`).
  - At the late re-run:
    - `rustfmt --check` on all 8 of my files: exit 0.
    - `cargo fmt --check` lists diffs only in M1's in-flight files: `src/k11_golden.rs`,
      `src/k11_host.rs`, `src/npy.rs`, `tests/fixture_pins.rs` and `tests/reference_k11_host.rs`.
    - None is in a GDN file or `lib.rs`.
  - I never ran plain `cargo fmt`.
- **Cross-build** (`l-cuda-gdn-cross-build.txt`). It uses M0's sysroot recipe in **my own target
  dir**, `/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-gdn`, because
  `cargo test --no-run` rebuilds `rung0`, and the shared dir holds the sha-pinned deployed one.
  - `cargo test --offline --release --no-run --target aarch64-unknown-linux-gnu --features cuda`:
    links 13 executables, exit 0.
  - **Final artifacts.** These come from the LATE rebuild at 20:32 CDT, after the late fixes. They
    supersede the `14e35026…` / `c0e0bb4c…` pair in this file's commit `5475de0`:
    - `.../release/deps/device_gdn_published-81ac041bb9f4a1f2`: sha256
      **`f0e88668a89f5665043c1f71b0faf480c7b69edd207e6a55952c4a5735cbefb0`**, ELF aarch64 PIE,
      10,367,712 bytes. It embeds the new kernel source: the ternary is present 3 times, and
      `min(T, t0` is absent [V, `rg -a`];
    - `.../release/deps/device_gdn_published_mirror-fcd093ce8b5195bd`: sha256
      `7205b5f61f92ec67da1f96c47d1372caac1f10c1a1a6a295afb7b6329a5ce828`, 9,455,520 bytes.
    - Source hashes at that build are at the end of `l-cuda-gdn-cross-build.txt`.
  - Libraries: `NEEDED libgcc_s.so.1, libm.so.6, libc.so.6`.
  - glibc versions GLIBC_2.17 … **2.39**. The 2.39 references are weak (`pidfd_spawnp`,
    `pidfd_getpid`, from std's test-harness spawn); the box has 2.39
    (`GAP-L-CUDA-GDN-TEST-BINARY-GLIBC239-2026-10-01`).
  - The live tree included other lanes' uncommitted files, so pin by these file hashes and do not
    rebuild elsewhere.

## The box command (after the post-F queue, single GPU, ≈ minutes, under rule 4's $20)

From the Mac, copy the test binary:

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    scp -i $KEY /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-gdn/aarch64-unknown-linux-gnu/release/deps/device_gdn_published-81ac041bb9f4a1f2 $BOX:/home/ubuntu/bin/ojas-gdn-published-device

On the box, check that `sha256sum /home/ubuntu/bin/ojas-gdn-published-device` equals
`f0e88668a89f5665043c1f71b0faf480c7b69edd207e6a55952c4a5735cbefb0`, then run:

    flock /home/ubuntu/queue/gpu.lock timeout 900 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib /home/ubuntu/bin/ojas-gdn-published-device --ignored --test-threads=1 --nocapture 2>&1 | tee /home/ubuntu/ojas-cuda/gdn-published-$(date -u +%Y%m%dT%H%M%SZ).log

- The timing test prints `GDN_PUBLISHED_TIMING {json}` lines:
  - one per shape, 4×2048 and 4×8192, at H=16 and Dv=128;
  - batched vs `B = 1`-serial fwd/bwd ms;
  - the projection `18 × (2·fwd + bwd)` beside PyTorch's 1.2–1.9 s.
- That is the number Fable's decision-6 trigger needs. Reading it is the lead's call; it is never
  a gate.
- It needs about 15 GiB free on the device.
- The rows go to `ledger/` as `quick` (rule 8).

## Messages exchanged (lanes coordinated directly, with the lead's leave)

- **To the lead:**
  - the exp status;
  - the lib.rs fmt-order question (answered: a second Edit, my lines only);
  - the runga hook.
- **With L-cuda-M1** (`a8ed369527080022d`):
  - prelude names, the `x > 0 → NaN` behaviour and its ulp (0.918), so I swapped to
    `qd_exp_nonpos`;
  - M1 made its prelude fully intrinsic, which removed the `--fmad` dependency;
  - the `k8_act` clippy lint, which M1 fixed;
  - the GDN module does not splice `device_prelude!` (it uses none of its names); M1 agreed.

**The runga hook** (signature sent to the lead):
- `gdn_smoke::gdn_published_checks(&CudaRuntime) -> Vec<Check>`: compile, then 6 smoke cases
  bitwise against the mirror plus a repeat run.
- Its float64 judgment is carried by the mirror suite, which holds the same 6 cases to the bound.

**The README.** Its "Device tests on the box" section does not list `device_gdn_published`. The
lead has asked L-cuda-M1, the README's owner, to add it, using the box command above.

**`src/lib.rs`** was edited again at 20:26 by another lane; its sha256 is now `840f3daf…`. My five
`pub mod` lines are unchanged [V, `rg -n gdn src/lib.rs`].

## Open (gap ids, all in `gaps.jsonl`)

This branch merged main at `c88caca`, picking up `e21066c`'s fix for the date-less
`GAP-L-CUDA-M0-LINUX-GLIBC239-NOT-RUN-2026-10-01` citation. The `gaps.jsonl` conflict was resolved as the union
of both sides: main's 20 records, then this lane's 9. Every line parses, and each of my 9 ids
appears exactly once. `python/tests/test_gaps_ledger.py` passes **10/10** on the merged tree [V].

- **Navigation:**
  - `GAP-L-CUDA-GDN-NO-LISTAGENTS-2026-10-01`
  - `GAP-L-CUDA-GDN-DEVMAP-OJAS-STALE-2026-10-01`
  - `GAP-L-CUDA-GDN-GITPULSE-LAPPI-UNTRUSTED-2026-10-01`
  - `GAP-L-CUDA-GDN-GITPULSE-OJAS-SAME-WORKTREE-BLIND-2026-10-01`
- **Device:**
  - `GAP-L-CUDA-GDN-DEVICE-NOT-RUN-2026-10-01`
  - `GAP-L-CUDA-GDN-BITWISE-MIRROR-ON-DEVICE-2026-10-01`
  - `GAP-L-CUDA-GDN-TRIGGER-TIMING-UNMEASURED-2026-10-01` (`GAP-L-CUDA-GDN-SPEED-INFERRED-2026-10-01`
    stays open with it)
  - `GAP-L-CUDA-GDN-TEST-BINARY-GLIBC239-2026-10-01`
- **Convergence:** `GAP-L-CUDA-GDN-REDUCTION-NOT-IN-SMALL-COMMON-2026-10-01`. The block/warp
  reduction is local because L-cuda-small's `small_common.rs` did not exist. It moves there with
  its exact tree kept.

**Not built, by scope:**
- K2(ii), the chunked form: only on the trigger.
- Wiring the batched launch into a step: the S lane's job. tessl's own step still runs one sequence
  per launch (`qwen35_train.rs:841-846`).

## First command for the next lane

Before the box run (CPU):

    cargo test --offline --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml --lib --test device_gdn_published_mirror published

After the box run:

    scp -i ~/.ssh/bharath_m5_macbook_pro.pem 'ubuntu@192.222.51.246:/home/ubuntu/ojas-cuda/gdn-published-*.log' /Users/bharath/qd-campaign/ && rg -n 'test result|FAILED|GDN_PUBLISHED_TIMING|panicked' /Users/bharath/qd-campaign/gdn-published-*.log
