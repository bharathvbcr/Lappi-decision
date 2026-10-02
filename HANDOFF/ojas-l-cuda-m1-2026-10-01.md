# HANDOFF: L-cuda-M1, milestone M1 of the CUDA Qwen3.5 provider (2026-10-01)

This is lane L-cuda-M1 of "train Lappi with ojas", working in
`/Users/bharath/Code/research/ojas/ojas-qwen35-cuda/`. M1 (`cuda-backend-scoping.md` §6.1) covers:
- K8, K11 and the tiny-fixture loader on top of M0;
- the `runga` binary, which runs every device check (M0 and M1, plus L-cuda-gdn's K2(i) and
  L-cuda-small's K3/K4/K6/K7/K9/K10) in one process;
- item 0 of the brief, which removed the `cargo fmt` hazard.

**All work was CPU-side, on a Mac with no NVIDIA GPU. No device code has run.**
- Every device test and every `runga` check is **NOT RUN**.
- **No ledger row was written**, because nothing was measured on a GPU (rule 5).

Labels: **[V]** I ran or read it (cited). **[I]** inferred, with the reasoning. **[U]**
unverified.

## The rung0 freeze, as ruled

- `src/bin/rung0.rs` is **byte-identical** to the M0 source snapshot [V]. Both sha256s are
  `1191975b…5826` (`AUDIT/ojas-training-2026-10-01/l-cuda-m1-cross-build.txt`, "rung0.rs against
  the M0 snapshot").
- The rung0 *binary* changes with any lib change. So the lead ruled that the snapshot
  (`AUDIT/ojas-training-2026-10-01/l-cuda-rung0-source-snapshot.tar.gz`, sha256
  `d536b031…4069`, committed at 4cfeaf9) is the record for the deployed `04ab105a` binary.
- A redeploy of rung0 gets a new sha and a new pin. The rung0 rebuilt today in this lane's target
  dir is `572d840d371f0dce509a23c314fce11fe759f358b2e28925c24ce7967f8e303d` [V]. It was **not**
  deployed.

## What was built

No git was run in ojas, so nothing in ojas is committed. The source state at build time is pinned
by sha256: `l-cuda-m1-cross-build.txt`, "this crate's sources". Prefixes below are from that list.

| File | Lines | sha256 | What |
| --- | ---: | --- | --- |
| `src/k8_act.rs` | 793 | `21edeca7` | The crate's one `exp`, `exp_nonpos`, `log`, softplus, sigmoid, SiLU, SiLU' (device prelude `act_prelude!` plus bit-identical host functions). Used by K2(i), K3, K4, K7 and K8 |
| `src/k8_plan.rs` | 564 | `7ab9538b` | K8 plans over `ColWindow`s; bitwise host references; activation sweep order; K8 cases |
| `src/k8_kernels.rs` | 191 | `2373887f` | Module `k8_swiglu`: `qd_swiglu_f32`, `_bf16`, `qd_swiglu_bwd_f32`, `_bwd_shared_f32`, `qd_residual_add_f32`, `qd_act_sweep_f32` |
| `src/k8.rs` (cuda) | 265 | `163600bd` | K8 launches |
| `src/k8_smoke.rs` (cuda) | 209 | `8330a36d` | K8 device checks (runga hook `k8_checks`) |
| `src/k11_host.rs` | 829 | `fdd2ef80` | AdamW scalars (f64, stored f32), `AdamwTable`, `AdamwBank` (per-entry step counts, active flags), the bit-exact f32 kernel-order emulation, the squared-norm partials emulation, `replay_f32`, synthetic bank cases |
| `src/k11_golden.rs` | 884 | `0f5858cc` | Both K11 goldens `include_bytes!`-embedded with their pins; `DecaySensitive`, `AdamwF`, `f_builder_of` |
| `src/k11_kernels.rs` | 157 | `06c2cf30` | Module `k11_adamw`: `qd_adamw_window_f32`, `qd_sq_partials_f32` (block fold = small_common's `qd_block_sum`) |
| `src/k11.rs` (cuda) | 163 | `c969a3e9` | `adamw_step` (one launch per active entry), `grad_sq_norm` |
| `src/k11_smoke.rs` (cuda) | 239 | `75ee057d` | K11 device checks (runga hook `k11_checks`), `run_device` |
| `src/tiny_fixture_published.rs` | 739 | `7d9d877a` | Tiny-fixture loader: embedded or `from_dir` (`ojas_io::open_nofollow`, bounded sizes); config via `ojas_io::parse_json_with` under explicit limits with every key named; BF16 safetensors; `.npy` gradients; device upload and runga's `loader_checks` |
| `src/report_cli.rs` | 505 | `ce3e3559` | The report and watchdog lifted from rung0, `prepare_report`, runga's arguments |
| `src/bin/runga.rs` (cuda) | 441 | `6e1f7259` | The rung (a) binary |
| `src/npy.rs` | 268 | `c32dca0f` | L-cuda-oracle's `.npy` reader, moved unchanged from `tests/reference/npy.rs` (now a one-line re-export) |
| `tests/device_k8.rs` (cuda) | 53 | `69ccea95` | 3 `#[ignore]` device tests |
| `tests/device_k11.rs` (cuda) | 152 | `6d19c371` | 4 `#[ignore]` device tests |
| `tests/reference_k11_host.rs` | 289 | `53729edb` | Tier 2 and tier 3 on the host, and the mutation ratios |
| `tests/fixture_pins.rs` | 160 | `16552833` | Both copied fixtures equal their pins and their live sources |
| `tests/fmt_boundary.rs` | 215 | `35ca3c82` | Item 0: no `#[path]` leaves the crate |

Copied fixtures, each byte-identical to its source and pinned:
- `tests/fixtures/adamw_decay_sensitive/`: L-oracle's 5 files. The manifest sha256 is
  `39882d3b…fe8e`.
- `tests/fixtures/qwen35_train_published/`: tessl's 31-file tiny fixture, pinned by
  `qwen35_train_published_SHA256SUMS`.
- `tests/fixtures/tessl/gdn_train_published.rs`: tessl's `tests/common/gdn_train.rs` (item 0).

Edits to shared files. Each is one concern and is behaviour-preserving where noted:
- `src/lib.rs`: only my own `pub mod` lines, one Edit at a time, with lib.rs re-read before each.
  The lines are `k11_golden`, `k11_host`, `k11_kernels`, `k8_act`, `k8_kernels`, `k8_plan`, `npy`,
  `report_cli` and `tiny_fixture_published`, plus `k11`, `k11_smoke`, `k8` and `k8_smoke` under
  `cuda`.
- `src/kernels.rs`: `#[macro_export]` on `device_prelude!` (lead-approved).
- `src/k0_plan.rs`: a new `ColWindow`; `CopyColsPlan` now delegates to it, with the same error
  text. L-cuda-small's `Window::check` delegates to it too [V, `src/small_common.rs:311`].
- `src/smoke.rs`: `twice` is now `pub`; new `m0_phases`, which is rung 0's M0 sequence for runga.
- `src/rung0_cli.rs`: `prepare_out` is a one-line delegation to `report_cli::prepare_report`
  (lead-approved).
- `Cargo.toml`: ojas-core removed, `ojas-io` added (lead-approved), `[[bin]] runga`.
- `Cargo.lock` was rewritten by cargo: ojas-io added, ojas-core is now only transitive. It is not
  committed anywhere. Its md5 at build time is `2c7ce4f0a22bfc10cfbec855c7c4d216`, and the diff
  against the snapshot's lock is in `l-cuda-m1-cross-build.txt`.
- `README.md`:
  - status;
  - the "What is here" rows;
  - a new "Runga on the box" section;
  - K8 and K11, L-cuda-gdn's and L-cuda-small's device tests under "Device tests on the box";
  - small's K10 marked **smoke-scale** (hidden 16, W 15.9 MB), not to be compared with the
    full-scale ~10–14 GB per call.
  - The README was edited (the smoke-scale note) after the evidence run. So its sha256 in
    `l-cuda-m1-cross-build.txt` (`a5935057`) is stale. The README is not a build input.
- `tests/reference/*` (oracle's) were rustfmt-only in item 0.

The oracle-file restructures in item 0:
- the tessl `#[path]` include became a pinned copy, `tests/fixtures/tessl/gdn_train_published.rs`;
- `npy` moved to the lib.

Neither changed a test's meaning [V]: the same reference tests pass.

**Not moved, by decision accepted by the lead:**
- the f64 AdamW reference stays in `tests/reference/adamw.rs`;
- `tests/device_k11.rs` reaches it through `mod reference;`;
- `runga` uses the embedded torch-f64 golden instead.

## K11: the three tiers, and why the f64 one is a sanity check only

1. **Device vs the f32 emulation, bit for bit** (is the kernel right). `k11_host` is the
   emulation:
   - tessl's kernel order, `qwen35_adamw.metal:58-66`;
   - scalars formed in f64 and stored as f32, `tessl/src/qwen35_adamw.rs:224-247`;
   - torch's two-form `lerp`;
   - per-entry `lr * lr_scale`, decoupled decay scaled by `lr * lr_scale` (D3), and a per-entry
     step count;
   - an entry with no gradient is untouched (D7).

   The squared norm: per 4096-element chunk, 256 threads each fmaf over 16 elements, folded by
   small_common's `qd_block_sum`/`block_sum`. That is per-block f32 partials, summed in f64 on
   the host in table order. **NOT RUN on a device.**
2. **The emulation vs L-oracle's decay-sensitive torch golden, held to ≤ 1e-6 absolute** (are the
   semantics right). This is L-oracle's pre-registered measure and bound [V,
   `l-cuda-m1-k11-host-tiers.txt`]:
   - floor: **max |w − torch| = 1.49e-8** at step 1, `layers.8.mlp.down_proj.weight`. Ratio
     1.5e-2 of the bound;
   - per-entry step counts equal torch's `adamw_steps_taken` for all 19 entries;
   - `layers.23.mlp.up_proj.weight` (never a gradient) is bit-identical to init at every step, in
     the emulation and in the golden.
   - Decay-sensitivity ratios: every pre-registered mutation, measured in **kernel-order
     arithmetic** through the library's own `entry_scalars`/`adamw_update_f32`. The two eps
     mutations use a test-local element function whose unmutated form is asserted bit-identical
     to the library over the whole replay:

     | Mutation | max abs | ratio over 1e-6 | at |
     | --- | ---: | ---: | --- |
     | D1 tessl default weight decay | 6.33e-4 | 632.5 | step 5, `norm.weight` |
     | D3 decay not lr-scaled | 5.00e-4 | 500.2 | step 5, `layers.3.self_attn.q_norm.weight` |
     | eps inside sqrt | 3.59e-2 | 35895.7 | step 5, `layers.9.linear_attn.dt_bias` |
     | eps before bias correction | 4.88e-3 | 4876.4 | step 5, `layers.9.linear_attn.dt_bias` |
     | lr_scale missing | 4.25e-2 | 42531.6 | step 5, `layers.0.input_layernorm.weight` |
     | D7 no-grad stepped as zero | 1.17e-2 | 11716.5 | step 4, `span_head.start_proj.weight` |
     | D7 model-wide step count | 2.86e-3 | 2864.7 | step 5, `span_head.start_proj.weight` |

     All 7 clear 100x. Each matches L-oracle's torch-arithmetic number to the printed digits.
3. **The device vs the adamw_f float64 golden, a sanity check only.**
   - On the host, the emulation is at params 2.83e-7 of max|p| (bound 2e-6) and grad_sq_norm
     3.40e-8 relative (bound 2e-6). Both bounds were derived in `src/k11_golden.rs` before the
     first run.
   - It cannot judge decay. At F's lr 1e-5 a decay slip moves `p` by about
     `0.9·lr·wd·|p|` = 9e-8 per step, under any f32 bound.
   - Also, `(1 − 1e-6·0.01) as f32 == 1.0`: the lower group does not decay at all in f32. And
     `(1 − 1e-5·0.01) as f32 == 1 − 2^-23` (test `f32_quantizes_the_decay_multiplier_at_fs_lr`).

**GAP-OJAS-K11-GOLDEN-RETYPES-F-HYPER-2026-10-01: resolved.**
- `gen_goldens.py` builds adamw_f through F's builder and records `f_builder`.
- All 193 goldens are byte-identical after regenerating (`l-cuda-m1-gen-goldens-run.txt`,
  `l-cuda-m1-goldens-manifest-diff.txt`).
- `f_builder_of` refuses the pre-fix manifest case, which has no `f_builder` (test
  `the_pre_fix_adamw_f_case_is_refused`). It also refuses six retypings of the live case.

## K8

`swiglu` f32 and bf16 out, `swiglu_bwd` into two buffers or two disjoint windows of one, and
`residual_add`. All are tessl's operand order
(`qwen35_mlp.metal:23-47,56-70`, `qwen35_bwd.metal:232-259`).
- Every float op is an explicitly rounded intrinsic. Every output is NaN-canonicalised
  (GAP-L-CUDA-M1-K8-NAN-CANONICAL-2026-10-01).
- Host tests [V]:
  - the left-to-right order is pinned by a value where the two orders differ;
  - canonical NaN on `inf*0`;
  - plan refusals: row overrun, buffer overrun, overlapping or different-stride shared outputs;
  - every case's references run.
- Device checks, **NOT RUN**:
  - all six kernels, bitwise, from sentinel-filled buffers, then a repeat;
  - three cases: 1x1, ragged 37x129 in a 300-wide fused row, and 64x6144 at the 2B's
    intermediate width, all carrying inf, NaN, ±0, subnormals and the exp edges;
  - the activation sweep: 7 functions over ~1M inputs plus the edges, bitwise.

## The tiny-fixture loader

`TinyFixture::embedded()` / `from_dir()` [V, 4 lib tests]:
- 27 BF16 parameters, each with its f32 gradient of the same shape, 70 token ids, and a finite
  f64 loss;
- the config's every key named, with an unknown key refused;
- refuses a missing or mis-shaped gradient, a truncated model file, a wrong-dtype loss, and a
  final-component symlink (via `ojas_io::open_nofollow`);
- `from_dir` equals `embedded`.

JSON is ojas-io's public reader with explicit `JsonLimits`, and there is no stopgap reader. The
`*_published` name (rule 9) rests on tessl's test docs:
GAP-L-CUDA-M1-TINY-FIXTURE-RULE-INFERRED-2026-10-01.

## runga

Phases:
1. open (4 GiB budget);
2. M0 (`smoke::m0_phases`);
3. K8;
4. K11;
5. the loader;
6. small.nvrtc, then K3, K4, K6, K7, K9, K10, each a phase (smoke-scale);
7. K2(i) `gdn_published_checks`;
8. libraries plus `runtime.no_leaked_buffers`.

Then, **report-only**: `extra.gdn_published_timing`.
- It runs at 4x8192, H=16, Dv=128, reps 3, on its own runtime opened after the first is dropped.
- The budget is `--gdn-timing-budget-gib`: default 24, min 16 (the ruling), max 96.
- It records `not_run` and the reason when:
  - `--no-gdn-timing` is passed;
  - fewer than 60 s of the cap are left;
  - the device has less memory than the budget;
  - its runtime cannot open;
  - an allocation is refused (Capacity).
- It records the budget used before and after. It is never in `overall()`.

Same watchdog and exit codes as rung0: 0, 1, 2, 3. A cap fire during timing exits 3 (the lead's
ruling), and every pass/fail result is already in the report. The header carries `golden_pins`
for the adamw_decay_sensitive (with the manifest), adamw_f and qwen35_train_published files.

On the Mac, runga refuses by name and exits 2, writing the report [V,
`l-cuda-m1-runga-mac-refusal.txt`]. A second run against the same `--out` is refused, and an
under-16 budget is refused. Its argument parser and 4 ported watchdog tests are in `report_cli`
(7 tests).

## Tests: run vs NOT RUN

| Suite | Result | Evidence |
| --- | --- | --- |
| Whole crate, no feature | **234 passed, 0 failed**, exit 0 | `l-cuda-m1-tests-host.txt` |
| Whole crate, `--features cuda` | **239 passed, 0 failed, 43 ignored**, exit 0. The 43 ignored are every lane's device tests | `l-cuda-m1-tests-cuda.txt` |
| This lane's new host tests | **60 ran and passed**: lib k11_host 8, k11_golden 4, k11_kernels 3, k8_act 12, k8_kernels 3, k8_plan 5, npy 2, tiny_fixture_published 4, report_cli 7; integration fixture_pins 4, reference_k11_host 4, fmt_boundary 4 | same |
| `tests/device_k8.rs` | **3 NOT RUN** (ignored, needs sm_90) | same |
| `tests/device_k11.rs` | **4 NOT RUN** | same |
| `runga` device checks | **NOT RUN**. Only the Mac refusal path ran | `l-cuda-m1-runga-mac-refusal.txt` |

Clippy and format [V, `l-cuda-m1-clippy-fmt.txt`]:
- `cargo clippy --all-targets --features cuda -D warnings` and the same without the feature, each
  in a fresh target dir: **exit 0, exit 0**.
- `cargo fmt --check`: **exit 0**. It was never run without `--check`.
- `rustfmt --check` file by file over all 93 `.rs` files: 1 diff, in
  `tests/fixtures/tessl/gdn_train_published.rs`. That file is tessl's byte-pinned copy and must
  not be formatted; its include carries `#[rustfmt::skip]`, which is why `cargo fmt --check` is
  clean.

Fail-first evidence:
- item 0's boundary test on the pre-fix tree flags 1 of 1 `#[path]`
  (`l-cuda-m1-failfirst-fmt-boundary.txt`);
- the F's-builder check refuses the pre-fix manifest case (unit test).

The other changes add new code with new tests, or are behaviour-preserving refactors:
- `ColWindow` (k0_plan's 6 tests pass unchanged);
- `prepare_out`'s delegation (rung0_cli's 4 tests and rung0's 4 watchdog tests pass unchanged);
- the K11 norm's move onto `qd_block_sum`. Before and after it, the host tier results are
  unchanged.

## Cross-build and artifacts

[V, `l-cuda-m1-cross-build.txt`]
- Setup: M0's sysroot recipe (`link.sh` sha256 in the file) and rustc/cargo 1.98.0. The target
  dir is `/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-m1`, this lane's own,
  so the deployed rung0 in the shared dir is untouched.
- Commands: `cargo build --release --target aarch64-unknown-linux-gnu --features cuda --bin runga
  --bin rung0` (exit 0), then `cargo test --release --no-run …` (exit 0; every test binary
  linked).
- **runga**: `aarch64-unknown-linux-gnu/release/runga`, 4,305,832 bytes, sha256
  **`bb3276e7a42ae3d0e20171ed78e0df537f2581d8f144d22b8b6c054a0814c805`**.
  - ELF aarch64 PIE, interpreter `/lib/ld-linux-aarch64.so.1`.
  - Its GLIBC symbol versions run up to 2.34; the box has 2.39.
- `deps/device_k8-ee452f644a37c50c`: sha256 `75f05d3a42c7964fc0143dc5149a02d37b2af04aaca3f7ea271cefcdb8074ff1`.
- `deps/device_k11-a91063d54a01db67`: sha256 `bc65b3bca3c48f1814185fc74859e6c623cffceda5bf3f9736d9b8b82c72ad90`.
  - Test binaries reference GLIBC 2.39 weakly through std's test harness, as
    GAP-L-CUDA-GDN-TEST-BINARY-GLIBC239-2026-10-01 records.
- rung0 (rebuilt, not deployed): `572d840d…303d`.
- The ojas-io and ojas-core source sha256s at build time are in each evidence file's first
  section. ojas-io was being edited by its lane during this work; the build used the listed
  hashes.
- The live tree included other lanes' uncommitted files. **Pin the file you copy by these
  hashes; do not rebuild elsewhere and expect them.**

**Unverified by name**, beyond "device NOT RUN" (both are under
GAP-L-CUDA-M1-DEVICE-NOT-RUN-2026-10-01):
- `runga` is the first binary to open a second `CudaRuntime` in one process after dropping the
  first, for the timing section. Its behaviour on the driver's primary context is untested.
- The K11 norm's move onto `qd_block_sum` has only the host mirror behind it.

## The box command for runga

It runs after the post-F queue. It is a single-GPU job of minutes, under rule 4's $20.

On the Mac:

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    shasum -a 256 /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-m1/aarch64-unknown-linux-gnu/release/runga
    scp -i $KEY /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-m1/aarch64-unknown-linux-gnu/release/runga $BOX:/home/ubuntu/bin/ojas-qwen35-cuda-runga

On the box, check that `sha256sum /home/ubuntu/bin/ojas-qwen35-cuda-runga` equals
`bb3276e7a42ae3d0e20171ed78e0df537f2581d8f144d22b8b6c054a0814c805`, then run:

    flock /home/ubuntu/queue/gpu.lock timeout 300 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/ojas-qwen35-cuda-runga --out /home/ubuntu/ojas-cuda/runga-$(date -u +%Y%m%dT%H%M%SZ); echo "runga exit $?"

The report is `<out>/runga-report.json`, `quick: true` (rule 8).
- Its checks are the pass/fail.
- `extra.gdn_published_timing` is the report-only input to Fable's decision-6 trigger. Reading
  it is the lead's call.
- If the timing says `not_run` with a Capacity reason, rerun with `--gdn-timing-budget-gib 40`.
  If you only want the pass/fail, pass `--no-gdn-timing`.

## Messages (lanes coordinated directly, with the lead's leave)

- **The lead:**
  - the ojas-io mid-edit build break (stopped until told);
  - the gdn hook wiring;
  - the f64 reference staying in tests;
  - the `prepare_out` delegation;
  - the K11 norm convergence;
  - small's hooks;
  - the K10 budget.

  All were accepted.
- **L-cuda-gdn:**
  - the clippy lints in my files, fixed and re-linted;
  - the runga wiring;
  - the `reference::npy` re-export.
- **L-cuda-small:**
  - `ColWindow`, which small adopted;
  - `smoke::twice`, which small kept its own `run_twice` for, with a reason;
  - the NaN policy, which small did not adopt: GAP-L-CUDA-SMALL-NAN-BITS-NOT-CANONICAL-2026-10-01.

## Open (every id in `gaps.jsonl`)

- **Navigation:**
  - GAP-L-CUDA-M1-GITPULSE-UNTRUSTED-2026-10-01
  - GAP-L-CUDA-M1-NO-LISTAGENTS-2026-10-01
- **Device:**
  - GAP-L-CUDA-M1-DEVICE-NOT-RUN-2026-10-01
  - GAP-L-CUDA-M1-GDN-TIMING-BUDGET-ESTIMATE-2026-10-01
- **Decisions for the lead:**
  - GAP-L-CUDA-M1-K8-NAN-CANONICAL-2026-10-01: two NaN policies in one crate; pick one.
  - GAP-L-CUDA-M1-TINY-FIXTURE-RULE-INFERRED-2026-10-01
- **Follow-ups:**
  - GAP-L-CUDA-M1-RUNG0-PRIVATE-WATCHDOG-COPY-2026-10-01: move rung0 onto `report_cli` and
    `m0_phases` at its next redeploy.
  - GAP-L-CUDA-M1-K8-ACT-TEST-TIME-2026-10-01
  - GAP-L-CUDA-M1-LIB-DOC-STALE-2026-10-01

**Closed here:**
- GAP-OJAS-K11-GOLDEN-RETYPES-F-HYPER-2026-10-01 (resolved);
- GAP-L-CUDA-M0-CARGO-FMT-WOULD-EDIT-TESSL-2026-10-01 (resolved);
- GAP-L-CUDA-M0-CRATE-NOT-IN-DEVMAP-2026-10-01 (resolved-with-residual: DevMap now indexes ojas
  at generation 3402, fresh, and finds this lane's symbols; cudarc is still not indexed).

## First command for the next lane

Before the box run (CPU):

    cargo test --offline --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml --test reference_k11_host --test fixture_pins -- --nocapture

After the box run:

    scp -i ~/.ssh/bharath_m5_macbook_pro.pem 'ubuntu@192.222.51.246:/home/ubuntu/ojas-cuda/runga-*/runga-report.json' /Users/bharath/qd-campaign/ && rg -o '"status":"[a-z_]+"|"summary":\{[^}]*\}' /Users/bharath/qd-campaign/runga-report.json
