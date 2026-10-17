# Delta report: ojas pin 132b46e -> 5b67cdd (2026-10-17)

For Fable's final ruling section 6 item 1 (`AUDIT/lora-2b-audit2-2026-10-17/fable-final-ruling.md`), which
the owner answered "procced with fable's advice" on 2026-10-17 (recorded in
`AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md`; Fable's recommendation was "now"). The previous
bump is Lappi commit 21fb7ad (bc498c4 -> 132b46e, 2026-10-15), whose shape this one follows.

Labels: **[V]** read or run this session, **[I]** inferred, **[U]** unverified. Heads: ojas local main
`5b67cdd8bd424a2fc5ff6d5b270142adec45ea8f` [V `git rev-parse`], Lappi `main` bffcf83 [V], tessl HEAD
42f1d52 with one dirty file [V `git status`; not pinned by Lappi, see section 5].

## 1. The range

`git -C /Users/bharath/Code/research/ojas log --oneline 132b46e..5b67cdd`: **99 commits** (75 non-merge,
24 merges) [V], all dated 2026-10-14 to 2026-10-16. 132b46e is an ancestor of 5b67cdd (`merge-base
--is-ancestor` exit 0, audit B). Neither commit is on `origin/main`, which is still bc498c4 per
`packed-refs` (GAP-BRANCH-PASS8-OJAS-ANCESTRY-UNCHECKABLE-2026-10-16, GAP-OJAS-PIN-STALE-EFA507F-DELTA-UNREADABLE-2026-10-14);
so, as with 21fb7ad, the lock's `url` does not serve this revision until ojas main is pushed [V gap text;
origin not re-queried this session].

Commits that touch the crates Lappi builds against (`ojas-core`, `ojas-io`, `ojas-data`, `ojas-qwen35`,
`ojas-model`; `*` marks them), from `git log --no-merges --name-only` [V]:

### 1a. ojas-qwen35 (the provider Lappi's `qd-train-metal` drives), 17 files, +1601/-578

| Commit | What |
|---|---|
| 6e1a937 | open checkpoints through tessl's `CheckpointView`, so sharded snapshots load |
| 5b9fd7a | state reads, `save_state` and `load_state` go through tessl's per-entry host API |
| 0c3fe16 | one owner for index validation; `load_state` validates before it writes and poisons only after a write |
| 7d9d5bc | read tensor dtype/shape through `CheckpointView::tensor_info`; **delete `shard_files`** |
| 077f60c | one home for the GPU tests' snapshot, npy and row helpers (`tests/support/mod.rs`) |
| 3cb7828 | the real 2B Tape step against transformers, without the provider (`tests/gpu_tape_2b.rs`) |
| 922309f | `gpu_real_2b_read_peaks` fits under memguard's 32 GiB and asserts the rise is the picked bytes |
| e2079ae | gdn corpus: tinyalpha stores the pre-clamp alpha; absent-key rule cited from transformers |
| 23f748f | G-IOREAD: stream typed reads, size-check `load_hf` from headers, charge LoRA merge (shared with ojas-io, ojas-model) |
| 3a9c55c | `load_hf` stores the tied table Bf16 on its own setting (shared with ojas-model) |

Source files changed: `src/config.rs` (+/-35), `src/names.rs` (146 lines, no `pub` item changed),
`src/state.rs` (245 lines; one new `pub fn load_state_from_index`), `src/step.rs` (4 lines: `Qwen35Step::open`
now opens the snapshot with `tessl::safetensors::CheckpointView::open` instead of `SafeTensors::open`) [V diff].

### 1b. ojas-core (`Budget`, `Tensor`, which Lappi imports), 5 files, +426/-95

| Commit | What |
|---|---|
| 2509603 | `Budget::check_room` is read-only and `charge` goes child-first, so a refusal never takes room on a shared ancestor (E6) |
| 02f4f4c | per-gradient flags on the fused cross-entropy (`CeGrads`; also autograd, cpu, metal, wgpu) |
| 1ea6c06 | a Bf16 tied table on the gather and the fused head (`refuse_bf16_table`; also cpu, metal) |

### 1c. ojas-io, 8 files, +1273/-15

| Commit | What |
|---|---|
| 5afa07c | parse and carry safetensors 0.8.0 dtype tags F8_E4M3FNUZ, F8_E5M2FNUZ, C64 |
| 23f748f, 66997a9, cacc1aa | G-IOREAD: streamed typed reads, header size checks, per-tensor `HfKind`, adversarial tests |
| 616f46f, dcfd76a, d9d2753 | SAVE replaces only empty or checkpoint directories (`replace_dir_with_if`); cancel; RESUME preflight |

### 1d. ojas-data, 8 files, +2340/-12

| Commit | What |
|---|---|
| 565a8ff | refuse a non-GPT-2 pre-tokenizer when loading from a directory (`check_gpt2_pre_tokenizer`, `load_hf_gpt2_dir`). **Adds `ojas-io` as a dependency of `ojas-data`** (`ojas-data/Cargo.toml`), which is why Lappi's `Cargo.lock` gains `"ojas-io"` under `ojas-data` |
| 5ea6d00, bb53eba, 2b01363, d02fe75 | image data path (MNIST idx, CIFAR-10 readers, `ImageBatchSampler`) and its adversarial tests |

### 1e. ojas-model (pulled in by ojas-qwen35), 32 files, +6022/-706

| Commit | What |
|---|---|
| 71b7e7a | qwen35 readout seam (`forward_hidden`, `gather_rows`, `RightPadded`) |
| bc08eeb | LoRA parity at scaling 1, the pre-registered run's alpha == r |
| 880da90 | `ParityModel` for the Metal and wgpu trainers, GQA/window parity |
| 0b2906b, aa8e2e2 | `Probe` forwarding proven by calling every op; `Gated` generated from the one method list |
| 1fef9ff, 5f3de7f, d253e1c, d02fe75, cacc1aa, 2b01363 | readout, LoRA-merge and adversarial tests |
| 23f748f, 3a9c55c, 616f46f, dcfd76a, d9d2753, 66997a9 | shared with 1a/1c |

### 1f. Not built by Lappi (listed for completeness)

ojas-metal (GDN scratch charged by tessl's figures c4281d0, oversize GDN workspace a0453c3, non-finite
V row d9e072b, OOM classification 979c126/e404142, partial-buffer wait 7a504a1, mnist step 0c2441e);
ojas-cpu (GDN forward follow-ups 6fc57c7/f8914b0/1b6748f, hybrid ops 293972a); ojas-wgpu (capped
workgroup harness 0648f9e/7003f66); ojas-cuda (d56e54a tessl MANIFEST pin, e404142); ojas-capi (SAVE,
cancel, RESUME preflight, Gated); ojas-infer/ojas-oracle (sampler and GQA goldens); go/; gemma-metal
(compare_intermediates port, act.h, ICB stub removal); bench/ (tied-table memory rows 07e6a3c; Go
summarisers); docs/, site/, tasks/ (the fine-tune track audit and Fable's 2026-10-15 rulings); .github.

## 2. API changes against what Lappi uses

Lappi's imports from ojas [V rg over `crates/`]: `ojas_qwen35::{ExternalGrad, Numerics, Precision,
Qwen35Step, Sequence, Snapshot, Which, TensorSpec, tower_tensors, Qwen35TextConfig}` (`qd-train-metal`
`src/provider.rs`, `src/run.rs`, tests `gpu_tiny`, `gpu_rung_b`, `names_2b`, `plan_groups`) and
`ojas_core::{Budget, Tensor}` (`src/provider.rs`, `tests/gpu_tiny.rs`).

- **No signature of those items changed** in the range: a diff of `ojas-qwen35/src` and `ojas-core/src`
  for `pub fn|struct|enum` lines naming them is empty [V grep over `git diff 132b46e 5b67cdd`].
- **Behaviour change under the same signature:** `Qwen35Step::open` opens the snapshot through tessl's
  `CheckpointView` (6e1a937, 7d9d5bc), so sharded safetensors snapshots load; the single-file 2B/0.8B
  snapshots Lappi pins are unaffected in interface. `shard_files` was deleted (7d9d5bc); Lappi never
  called it [V rg].
- `Budget` semantics (2509603): `check_room` is read-only and `charge` is child-first, so a refused
  charge no longer leaves room taken on a shared parent. Lappi's provider charges through `Budget`;
  the change is in the refusal path only [I from the commit message; not exercised by Lappi's tests
  beyond the suites below].
- **Manifest:** `ojas-data` now depends on `ojas-io` (565a8ff). The only `Cargo.lock` change the bump
  produces is `+ "ojas-io"` under `[[package]] name = "ojas-data"` [V `git diff Cargo.lock` after the
  cargo run under mac_heavy].
- No ojas crate adds a registry dependency in the range for the five crates [V `git diff` of their
  `Cargo.toml` files: only `ojas-data/Cargo.toml` changed].

## 3. What this delta carries for the LoRA-2B track

Per the ruling's evidence list, all on 5b67cdd: the readout seam (71b7e7a), the scaling-1 LoRA parity
(bc08eeb), the bf16 tied table (1ea6c06, 3a9c55c; rows 07e6a3c), the real-2B Tape gate test (3cb7828),
sharded snapshot loading and per-entry state (6e1a937, 5b9fd7a), and the GPT-2 pre-tokenizer refusal
(565a8ff). No `ojas-train` crate and no Tape `StepProvider` exist at 5b67cdd (`Cargo.toml` members; ruling
section 1) [V audit B, re-checked by `ls`]. The ruling's alternative (wait and bump once to the phase-A
sha) was not taken: the owner chose "now".

## 4. Verification at 5b67cdd (one `tools/mac_heavy.sh` lock window, label `lora2b-pin-verify`, 05:21:36Z-05:23:17Z)

| Check | Result [V log `scratchpad/pin_batch.log`] |
|---|---|
| `cargo clippy -p qd-train-metal -p qd-preflight --all-targets -- -D warnings` | Finished, 0 warnings, rc 0 (regenerated `Cargo.lock`: the one line above) |
| `make pin` | `pin PASS: ojas is at the locked commit 5b67cdd8bd424a2fc5ff6d5b270142adec45ea8f`, rc 0 |
| `cargo test -p qd-train-metal -p qd-preflight` (`QD_PYTHON` = the project `.venv`) | 18 binaries, **82 passed, 0 failed, 14 ignored** (10 `gpu_tiny` + 4 `gpu_rung_b`), rc 0 |
| `cargo test -p qd-train-metal --release --test gpu_tiny -- --ignored --test-threads=1` | **10 passed, 0 failed** (two of the ten are the SIGKILL tests' child entry points, which report "not run" and pass by design) |
| `python/tests/test_pin_gate.py` (`.venv`) | 7 passed |
| `gpu_rung_b` `--ignored` (4 tests) | **not run**, as in 21fb7ad |
| `make lint` | All checks passed (run outside the lock; ruff is not heavy) |

`qd-train-metal` is the only Lappi crate with an ojas path dependency (`crates/qd-train-metal/Cargo.toml:40-41`
[V rg over every `Cargo.toml`]); `qd-preflight` holds the pin gate. `qd-metal` depends on `qd-train`, not on
ojas, and was not rebuilt here.

## 5. Not pinned and not verified here

- tessl is built by path from `ojas-qwen35` (`../../tessl`) and is not named in `upstream.lock.json`; it was
  at 42f1d52 with one dirty tracked file during the runs above [V `git status`]. The runs' results hold for
  that tree, not for a pinned tessl.
- The ojas working tree had one modified tracked file (`.gitignore`) and untracked reports during the build
  [V `git status --short`]; the pin compares `HEAD` only, as GAP-LAPPI-PINS-NO-OJAS-SHA's residual says.
- No GPU memory or tokens/s row was produced; nothing here measures the LoRA step.
