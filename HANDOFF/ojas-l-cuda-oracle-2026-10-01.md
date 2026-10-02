# HANDOFF L-cuda-oracle (2026-10-01): float64 host references for the CUDA Qwen3.5 kernels

Lane L-cuda-oracle of "train Lappi with ojas" (rung a of the CUDA parity ladder: every kernel is held
to a float64 host reference before it is timed). Host-only Rust on the Mac. No GPU was used and no
ledger row exists: nothing here is a training run or a timing.

All code and fixtures are in `/Users/bharath/Code/research/ojas/ojas-qwen35-cuda/`, inside the
paths the brief assigned this lane: `tests/reference/**`, `tests/reference_*.rs` and `tests/fixtures/**`.
`Cargo.toml` was not touched (no dev-dependency was needed) and `src/**` was not touched.
Nothing in ojas is committed by this lane (the brief forbids git in ojas); the ojas coordinator owns that.
On resume after the usage-limit stop, `tests/` did not exist: no partial write had landed.

## Files written in ojas (all uncommitted there, by the brief's rule; the coordinator adopts them)

Everything below is under `ojas-qwen35-cuda/tests/`.

- `reference/`, 14 files: `mod.rs` and 13 modules.
  - References: `gdn_published`, `gates_published`, `conv1d`, `norms`, `attn_pieces`, `embed`, `ce`, `adamw`.
  - Helpers: `npy`, `sha256`, `rng`, `fd`, `goldens`.
- 5 test targets: `reference_gdn_published.rs`, `reference_gdn_published_vs_tessl.rs`,
  `reference_goldens_published.rs`, `reference_kernels.rs`, `reference_adamw.rs`.
- `fixtures/gdn/`, 108 files: tessl's published corpus, copied byte-identical, with its `SHA256SUMS` and `MANIFEST`.
- `fixtures/qwen35/`, 7 files: tessl's six `qwen35_rope_*` plus this lane's `qwen35_rope_SHA256SUMS`.
- `fixtures/gen_goldens.py` and `fixtures/goldens/`: 194 files, which are 193 goldens plus `manifest.json`.

## What was measured

Raw outputs are in `AUDIT/ojas-training-2026-10-01/`:
- `l-cuda-oracle-reference-suite-green.txt`: the full `cargo test` with every measurement printed;
- `l-cuda-oracle-clippy-all-targets-cuda.txt`: `cargo clippy --all-targets --features cuda -- -D warnings`,
  exit 0, no warnings (rust/clippy 1.98);
- `l-cuda-oracle-failfirst-*.txt`: the fail-first runs.

Tests: **33 tests in 5 targets, all pass**: `reference_gdn_published` 8, `reference_gdn_published_vs_tessl`
2, `reference_goldens_published` 3, `reference_kernels` 16, `reference_adamw` 4. The in-process comparison target is
pinned in-test to tessl `tests/common/gdn_train.rs` sha256 `dc455ac9…` (via `include_bytes!`), and the six copied
`qwen35_rope_*` files to `tests/fixtures/qwen35/qwen35_rope_SHA256SUMS`, so neither comparison target can move silently. L-cuda-M0's 62 unit
tests still pass alongside. Every target finishes in under 10 s in debug (`--test-threads=1`; measured 0.0-9.2 s across runs).

### Grades

- **golden**: an independent committed fixture, within a bound written in the test before its first run.
- **tessl in process**: tessl's own host reference compiled in unmodified.
- **derivative**: central differences, h = 1e-6 and bound 1e-7 (tessl `tests/qwen35_bwd.rs:69-81`). This
  checks a backward against its own forward only.

### The references

| K | Reference (`tests/reference/`) | Golden and bound | Measured worst | Grade |
| --- | --- | --- | --- | --- |
| K2 | `gdn_published.rs`: fwd + checkpoints `[B,H,ceil(T/64),Dk,Dv]`, bwd consuming them | (1) tessl's 13-case `gdn_published_*` corpus (numpy f64; copied byte-identical, checked in-test against tessl's `SHA256SUMS`), T=1,63,64,65,127,129,1000,8191, 1e-12 of max (`tessl/tests/gdn_fixtures.rs:63`); (2) torch f64 golden of transformers' `torch_recurrent_gated_delta_rule` (l2norm in kernel) + autograd, T=1,63,64,65,130, Dk=128, 1e-10 | (1) 2.45e-16 (L8191); (2) 2.23e-15 over o, final state, dq, dk, dv, dg, dbeta, ds0 | golden ×2 |
| K2 | same | tessl `gdn_train_f64`/`gdn_train_bwd_f64` in process, T=1,63,64,65,130 with tessl's flags (`tessl/tests/gdn_train.rs:365-371`), Dk=128 | **bit-identical**, all 8 outputs × 5 T | tessl in process |
| K2 | same | FD through the checkpointed backward at T=1,63,64,65,130 (tessl's own FD runs only T=7) | 9.98e-10 | derivative |
| K3 | `gates_published.rs` (`g = -exp(A_log) softplus(a+dt_bias)`, `beta = sigmoid(b)`) | `goldens/gates_published_*`, 1e-13 (includes softplus' linear branch and a=120) | 2.39e-16 | golden + derivative |
| K4 | `conv1d.rs` (causal depthwise K=4 + SiLU, zero state) | `goldens/conv1d_silu_*` (`F.conv1d`), 3 shapes incl. T<K-1, 1e-12 | 2.85e-16 | golden + derivative |
| K6 | `attn_pieces.rs`: q/k `(1+w)` norm + partial RoPE (cos/sin of the f32 angle) | tessl's transformers fixture `qwen35_rope_*` at positions 20000+, 5e-6 abs (`tessl/tests/qwen35_kernels.rs:192-212`); `goldens/qk_norm_rope_*` given the angle, 1e-12 | 7.08e-7 abs; 3.80e-16 | golden ×2 + derivative |
| K6 | `rope_angle_f32` | the golden's angles from transformers' own `Qwen3_5TextRotaryEmbedding` at the 2B config | 320/320 bit-identical | golden |
| K6 | `attn_pieces.rs`: output gate `o·sigmoid(gate)` | `goldens/attn_output_gate_*`, 1e-13 | 2.33e-16 | golden + derivative |
| K7 | `norms.rs`: `Qwen3_5RMSNorm` `(1+w)` and `Qwen3_5RMSNormGated` | `goldens/rms_norm_*` (d=64, 2048), `gated_rms_norm_*`, 1e-12 | 5.14e-16 | golden + derivative |
| K9 | `embed.rs` | `goldens/embed_*` (`F.embedding`, repeated, out-of-order ids), y exact, dtable 1e-13 | 0 (bit-exact) | golden |
| K10 | `ce.rs`: CE over chosen rows of a tied head | `goldens/ce_rows_*` (`F.cross_entropy`; mean, sum×0.7, a planted dominant logit), 1e-12 | 3.35e-16 | golden + derivative |
| K11 | `adamw.rs`: torch single-tensor AdamW with per-parameter wd and lr_scale, `grad_sq_norm` | `goldens/adamw_f_*`: 5 steps of `torch.optim.AdamW(foreach=False, fused=False)` over F's groups built by Lappi's own `layerwise_param_groups` + `apply_lr`, 1e-13 | params bit-identical every step; norm 3.4e-16 | golden |

**Validated against a golden: 8 of 8 requested references** (K2, K3, K4, K6, K7, K9, K10, K11).
**Unvalidated: 0.** Residual caveats are in the gaps below, chiefly that the K3 formula and the GDN
and norm bodies in the generator are this lane's float64 transcriptions of transformers' code.

K11 uses F's recipe as read from code: betas (0.9, `DEFAULT_BETA2` = 0.999), eps 1e-8, weight decay 0.01
(`python/qd_train/optim.py:63,336-338`), applied by torch to every group, so every parameter is
decayed; lr = base × lr_scale (`optim.py:224-241`); lr_scale 0.1 on `layers.<i>.` with i < 8
(`optim.py:247,250-316`). The reference's name predicate equals Lappi's on all 9 golden names, including
`layers.10` and `layers.8`. tessl's `tests/qwen35_adamw.rs` holds no committed golden: its reference is an
in-test formula with no `lr_scale`. L-oracle's `tools/qd_train_oracle_tiny.py` exists only in worktree
`agent-a4907ad4a70172990`, unmerged. Its AdamW group spec (`:479-507`, `:912-933`) was inspected by search, not reused: the golden
calls Lappi's `optim.py` directly instead.

### The goldens

`tests/fixtures/gen_goldens.py` writes `tests/fixtures/goldens/` (193 `.npy` files and one names list,
4.8 MB, plus `manifest.json`). The lead approved it on 2026-10-01 as a fixture generator. Its settings:
- float64, CPU, `torch.use_deterministic_algorithms(True)`, one thread, fixed seeds;
- torch 2.12.1, numpy 2.5.0, transformers 5.12.1, Python 3.14.7, all from `/Users/bharath/.venvs/ml`.

The manifest records the command line, the versions, every case's shapes and seeds, and every file's
sha256. It also records the generator's own sha256. `goldens::verify()` checks:
- every hash;
- that the directory holds exactly the pinned files;
- the generator's hash against the file on disk.

A regenerated golden without a manifest update therefore fails, and so does an edited generator that was not re-run.

Regenerating produced byte-identical `.npy` files: the aggregate hash `2eb45ae3…` was the same before and after.

The generator's own self-checks, recorded in the manifest:
- The GDN transcription against transformers' own float32 `torch_recurrent_gated_delta_rule`: ≤2.23e-7.
- The repo rule's divergence from the golden: 1.8e-2 to 3.6e-2.
- Both RMSNorm transcriptions against transformers' float32 modules: ≤1.43e-7.
- MRoPE collapse for text-only input. At the 2B config (`mrope_section [11,11,10]`, interleaved), transformers' rotary module gives cos/sin **bit-identical** (max abs 0.0) to the plain f32 angle at 10 positions up to 20001. This is evidence for `GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01`. That gap is not closed here; it is the lead's.

Rule 9 is enforced on all fixtures and tests:
- every GDN fixture, golden, test and function name says `published`;
- two tests assert it over both fixture directories and both manifests.

`tessl/tests/fixtures/qwen35/qwen35_gdn_*` was not used, because its names lack `published`. The lead agreed.

### Fail-first

Each run used a temporary edit. The edit was reverted, and the suite then ran green again. Raw output for each is in AUDIT.

| Mutation | Caught by |
| --- | --- |
| GDN reads the undecayed state (the repo rule) | corpus (2.03e-1), tessl in process (6.5e-2), FD |
| GDN beta gate dropped | corpus (8.9e-1), tessl in process, FD, the repo-rule contrast test |
| K7 `w` instead of `1 + w` | golden (7.3e-1), FD |
| K11 lr_scale not applied to the decay | golden (9.0e-8 vs bound 1e-13) |
| K11 eps inside the bias correction | golden (2.8e-10 vs bound 1e-13) |
| K6 angle formed in f64 instead of f32 | transformers fixture at position 20000 (4.3e-5 vs 5e-6) |

## For the kernel lane (how to use these)

- In a test: `mod reference;`, then e.g.
  `reference::gdn_published::{gdn_train_published_fwd_f64, gdn_train_published_bwd_f64, Inputs, Shape}`.
- The GDN backward takes the checkpoint tensor, so a kernel's own `ckpt` output can be checked as well as consumed.
- Bounds for f32 kernels against these references are the kernel lane's to set before its first run. tessl's are:
  - 1e-4 of max for GDN (`tessl/tests/gdn_train.rs:241`) and for the row-local ops (`qwen35_bwd.rs:58`);
  - 1e-5 abs + 1e-5 rel on the CE loss (`cross_entropy.rs:185`).
- K6: take the RoPE angle from the host, or use tessl's 4e-3 bound (`GAP-L-CUDA-ORACLE-ROPE-DEVICE-POW-2026-10-01`).
- K11 has **no `grad_scale` field**. A caller applying `clip_grad_norm_`'s coefficient scales the
  gradient tensors in place before `adamw_step_f64`, as torch does to `.grad`. tessl does it in f32
  before its step (`tessl/tests/qwen35_adamw.rs:89-94`). `grad_sq_norm_f64` is the squared norm that
  coefficient is computed from.
- **Rule 9 and K3 (the lead ruled 2026-10-01: rename).** The gates are part of the GDN operator,
  so their goldens, tests and reference now say `published`:
  - the goldens are `goldens/gates_published_*`, and the manifest case is `gates_published`;
  - the module is `reference/gates_published.rs`, with `gdn_gates_published_fwd_f64` / `_bwd_f64`;
  - the tests are `k3_gates_published_match_the_torch_golden` and
    `k3_gates_published_backward_is_the_derivative_away_from_the_threshold`.

  The rule-9 test now covers the gates as well (92 GDN goldens, 12 of them gates, all say `published`).
  - Regenerating gave **byte-identical** goldens apart from the rename: 193 files before and after;
    the 12 `gates_*` files map to the 12 `gates_published_*` files sha256 for sha256; nothing
    missing, nothing extra, no other file changed.
  - K3 measurements are unchanged (2.39e-16).
  - tessl's own `tests/gdn_gates.rs` names are tessl's to change.
- Not written by this lane, being outside its brief:
  - K0, K1 (GEMM), K5 (attention core) and K8 (swiglu, residual) references;
  - the whole-step reference against tessl's `qwen35_train/` fixture (§5.2).

## What changed (commits in this Lappi worktree, branch `worktree-agent-a7351a470e167004b`)

- `8fd9ea1`: interim. K2 handoff, GDN fail-first outputs, three gaps.
- `142faa2`: the handoff, the AUDIT outputs and five more gaps.
- `5b3f2e2`: in-test pins for tessl's `gdn_train.rs` and the RoPE copies; 33 tests.
- Merged to main at `6d1bac1`.
- The K3 rename commit (this note): the lead's rule-9 ruling.

## Open (gap ids)

- `GAP-L-CUDA-ORACLE-GITPULSE-TRUST-2026-10-01`: every GitPulse facet failed with `REPOSITORY_TRUST_REQUIRED`.
- `GAP-L-CUDA-ORACLE-LISTAGENTS-UNAVAILABLE-2026-10-01`: no `ListAgents` tool.
- `GAP-L-CUDA-ORACLE-QWEN35-GDN-FIXTURE-RULE-UNNAMED-2026-10-01`: tessl's transformers GDN fixture lacks `published`.
- `GAP-L-CUDA-ORACLE-TESSL-PATH-INCLUDE-2026-10-01`: the in-process target needs the sibling tessl checkout.
- `GAP-L-CUDA-ORACLE-GDN-2B-HEADS-NOT-RUN-2026-10-01`: H=16, Dv=128 not run in process (debug-time budget).
- `GAP-L-CUDA-ORACLE-GOLDEN-TRANSCRIPTIONS-2026-10-01`: generator transcriptions of transformers' f32-casting code.
- `GAP-L-CUDA-ORACLE-ROPE-DEVICE-POW-2026-10-01`: a device `powf` is a third angle implementation.
- `GAP-L-CUDA-ORACLE-DEVMAP-DIRECT-READS-2026-10-01`: sources were read directly; nothing is graph-confirmed.

## First command for the next lane

```
cargo test --offline --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml --test reference_gdn_published --test reference_gdn_published_vs_tessl --test reference_goldens_published --test reference_kernels --test reference_adamw
```
