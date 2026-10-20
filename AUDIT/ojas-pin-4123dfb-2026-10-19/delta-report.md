# Delta report: ojas pin 04277f8 -> 4123dfb (2026-10-19)

The owner approved this bump on 2026-10-19: `AUDIT/lora-2b-w2-2-2026-10-18/owner-decision-2026-10-19.md`, item 2a
("Pin 4123dfb (phase A+B) (Recommended)"), which supersedes item 2's earlier target. The previous bump is Lappi
commit c324c49 (5b67cdd -> 04277f8, 2026-10-17), whose shape this report follows
(`AUDIT/ojas-pin-04277f8-2026-10-17/delta-report.md`). The gate that requires this file is
`crates/qd-preflight/src/pin.rs` (`find_delta_report`): a directory `AUDIT/ojas-pin-<prefix>-<date>/` holding
`delta-report.md`, where `<prefix>` is at least 7 hex characters and a prefix of the locked commit. It checks the
path only; the report has no required fields.

Labels: **[V]** read or run this session, **[I]** inferred, **[P]** pending.

Constraint of this session: git is blocked in the ojas repo, so there was no `git log` or `git diff` there. The commit
list below comes from ojas's reflog file; the file-level diff is PENDING (section 4).

Heads: ojas local `main` is `4123dfbe570794d9b498c2bb0614990dee285ec0` [V: `ojas/.git/refs/heads/main`, HEAD is
`ref: refs/heads/main`]. Whether ojas `origin/main` has this commit was not checked, so the lock's `url` may not serve
this revision until ojas main is pushed [I, as for 6d7c807 and c324c49].

## 1. The range (from the reflog)

`ojas/.git/logs/refs/heads/main` lines 140-169, in order, each line moving local `main` one step from 04277f8 [V].
That is **30 reflog entries** (28 commits made directly on main, 2 merges), not the 29 the task brief gave [V, counted
from the file]. One correction to the brief, read from the reflog: the merge of `gp/a6-phase-b` is `fb057d4`
(line 168); `54fa635` is the commit before it, "ojas-metal: allocate the sq_norm parts buffer by SqNormPlan::parts_words"
(line 167). The owner-decision file's "merged at 54fa635" has the same slip [V].

A reflog on `main` lists first-parent steps only. The two merges (`gp/clip-sq-norm`, `gp/a6-phase-b`) bring in side
branch commits the reflog does not name, so **the true commit count of `04277f8..4123dfb` is larger than 30 and is not
known here** [I; PENDING `git rev-list --count`, section 4]. That 04277f8 is an ancestor of 4123dfb is supported by the
reflog chain starting at 04277f8 on line 140 [V], not by `git merge-base`.

| # | Commit (new sha) | Subject |
|---|---|---|
| 140 | ffc6f2e | ojas-train: crate shell with a hand-rolled FIPS 180-4 SHA-256 and its throughput bench |
| 141 | 3f87e76 | ojas-autograd: Tape::linear_cross_entropy_value, the fused CE with no gradient and no node |
| 142 | 07f60c2 | ojas-model: Qwen3.5 adapter bank, all-or-nothing AdamW, and the adapter checkpoint layout v1 |
| 143 | 4803dbd | ojas-oracle: mlx q8 g64 fake-quant tiny Qwen3.5 fixture generator and fq8 LoRA tests |
| 144 | 8e044d7 | ojas-model: merged bf16 LoRA weights for export (one rounding), by HF name |
| 145 | 525bab2 | ojas-model: AdamW vs torch.optim.AdamW over 100 steps on CPU and Metal (oracle and test) |
| 146 | 9ffc289 | ojas-model: memory pre-flight for the LoRA step, calibrated from the committed rows, plus a budgeted dry run |
| 147 | f0259b4 | ojas-metal: scan a frozen weight once (freeze_checked) and refuse writes to it; bench arms for the LoRA step A/B |
| 148 | b731d6d | docs: ojas-qwen35 README and dtype-policy match the committed 2B Tape, bf16 and LoRA rows |
| 149 | f4bea36 | docs: the Lappi LoRA training runbook for the Mac (bf16 and fake-quant q8 arms) |
| 150 | e85d7aa | ojas-model: generated qwen35_tiny_fq8 base and qwen35_tiny_lora_fq8 references |
| 151 | 03b450b | ojas-model: generated adamw_100 torch fixture |
| 152 | c695803 | site: register the adapter-checkpoint and runbook docs and regenerate the reference pages |
| 153 | e9547e1 | ojas-io: SHA-256 moves here with a streaming file digest; ojas-model pins the frozen base at load (load_hf_pinned) |
| 154 | 6f10af9 | ojas-io: the file digest refuses a FIFO before opening it and a file that changes length while hashed |
| 155 | befe116 | bench: clip_norm_only rows (ojas and torch), the norm pass of clip_grad_norm_full alone |
| 156 | cf89a2b | gitignore: ignore DevCouncil local state except .devcouncil/config.yaml |
| 157 | 43d5e82 | ojas-model: preflight dry_run refuses a request that overflows the budget (verify failure) |
| 158 | 0a17930 | ojas-model tests: three stale expectations brought to the intended behaviour (verify failures) |
| 159 | aaed8b3 | ojas-metal: MetalBackend::scale_grad scales in place, so a frozen buffer refuses it (verify failure) |
| 160 | 111097d | ojas-model tests: drop the unused Backend import in preflight.rs |
| 161 | 73cfe02 | merge gp/clip-sq-norm |
| 162 | dfd2419 | docs: quote the shipped sq_norm clip time; regenerate the site (from a797c1d) |
| 163 | d0fbe97 | ojas-io: sha256 uses as_chunks for its constant-size chunks (clippy failure) |
| 164 | 0e95fe7 | ojas-model: box AdapterError's quantizer records and use is_multiple_of/as_chunks (clippy failures) |
| 165 | 81a81ca | ojas-model tests: is_multiple_of in qwen35_optim (clippy failure) |
| 166 | 94aa3af | ojas-model tests: the last two clippy lints (single-element loop, useless vec!) |
| 167 | 54fa635 | ojas-metal: allocate the sq_norm parts buffer by SqNormPlan::parts_words |
| 168 | fb057d4 | merge gp/a6-phase-b |
| 169 | 4123dfb | site: regenerate the reference pages after the phase A+B merge |

`ojas-train/Cargo.toml` at HEAD says phase A moves Lappi's qd-train modules in with their history (edition 2024), and
reflog line 153 moves SHA-256 into `ojas-io` [V].

## 2. What Lappi builds against

Lappi's path dependencies on ojas, searched in every `crates/*/Cargo.toml` [V, `rg ojas --glob Cargo.toml`]:

| Lappi crate | ojas dependency |
|---|---|
| `crates/qd-train-metal` | `ojas-qwen35 = ../../../ojas/ojas-qwen35`, `ojas-core = ../../../ojas/ojas-core` |
| `crates/qd-train`, `crates/qd-export`, `crates/qd-metal` | none (a comment in qd-train's manifest names `ojas-train` as the human's later decision; qd-export and qd-metal depend on canonical tessl, not ojas) |

Items `qd-train-metal` imports from ojas [V, `crates/qd-train-metal/src/provider.rs:52-58`, `run.rs:29`]:
`ojas_core::{Budget, Tensor}`; `ojas_qwen35::{AdamWHyper, BankState, ExternalGrad, GroupSpec, LrRule, Numerics,
OptimizerPlan, Precision, Qwen35Error, Qwen35Step, Qwen35TextConfig, Select, Sequence, Snapshot, TensorSpec, WdRule,
Which, tower_tensors}`.

At ojas HEAD 4123dfb [V, read from `ojas-qwen35/src/lib.rs` and `ojas-core/src/lib.rs`]: every `ojas_qwen35` name
above is still re-exported from the crate root, and `ojas-core` still re-exports `Budget`. The compile in section 5 is
the check on the exact paths and signatures.

Manifest change that matters to the lockfile [V, `ojas-qwen35/Cargo.toml` at HEAD]: `ojas-qwen35` depends on
`ojas-core` and `ojas-io`, and on macOS on `tessl` (canonical `../../tessl`) and `ojas-model`. Lappi's committed
`Cargo.lock` already lists `ojas-qwen35` with exactly `ojas-core, ojas-io, ojas-model, tessl` [V, `Cargo.lock:653-660`],
so no lock change is expected from the dependency edges [I; the build in section 5 is the check].

`ojas-train` (new crate, edition 2024) holds the modules moved from Lappi's `qd-train`. Lappi does **not** depend on
it: moving `qd-train` onto `ojas-train` is Lappi handoff step 6 and is out of scope for this bump.
Lappi's `qd-train` is therefore unchanged by this bump and its tests do not exercise ojas.

## 3. Pin mechanics

`upstream.lock.json` `ojas.revision` 04277f8fc722f02dbbbb7ad8cbe93d540876ba7c -> 4123dfbe570794d9b498c2bb0614990dee285ec0.
`python/tests/test_pin_gate.py` reads the lock (`LOCKED = json.loads(LOCK.read_text())["ojas"]["revision"]`, line 31)
and has no hard-coded revision to change [V]. The closed allowance in `pin.rs` is not touched: a new pin needs a delta
report, not an allowance entry.

## 4. Source diff (owner-supplied command output)

**PENDING.** Not run here (git in the ojas repo is blocked). The owner supplies the output of:

```
git -C /Users/bharath/Code/research/ojas diff --name-only 04277f8fc722f02dbbbb7ad8cbe93d540876ba7c..4123dfbe570794d9b498c2bb0614990dee285ec0
git -C /Users/bharath/Code/research/ojas rev-list --count 04277f8fc722f02dbbbb7ad8cbe93d540876ba7c..4123dfbe570794d9b498c2bb0614990dee285ec0
git -C /Users/bharath/Code/research/ojas diff --name-only 04277f8fc722f02dbbbb7ad8cbe93d540876ba7c..4123dfbe570794d9b498c2bb0614990dee285ec0 -- '*/src/*' '*/Cargo.toml' Cargo.toml Cargo.lock
```

The third line answers whether any library source or manifest Lappi imports (`ojas-core`, `ojas-qwen35` and their
dependencies `ojas-io`, `ojas-model`) changed. Until it is pasted here, which files changed is unknown and nothing in
this report states it. The reflog subjects say library source did change (`ojas-io`, `ojas-model`, `ojas-metal`,
`ojas-autograd`, new `ojas-train`) [V: subjects only]; whether `ojas-core` or `ojas-qwen35` did is not known from them.

## 5. Verification

Run 2026-10-19/20 in the worktree of branch `w3-2-ojas-pin-4123dfb-2026-10-19`, with `upstream.lock.json` at 4123dfb,
each cargo job as `MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh <label> cargo ...`, one at a time. Counts are summed
from the `test result:` lines of each job's output [V].

| Check | Result |
|---|---|
| `cargo test -p qd-train` (label w3-2-qd-train) | RUN, exit 0. 26 suites, 260 passed, 0 failed, 5 ignored. |
| `cargo test -p qd-train-metal` (w3-2-qd-train-metal) | RUN, exit 0. 14 suites, 41 passed, 0 failed, 15 ignored. Ignored tests are GPU tests and were not run. |
| `cargo test --no-run -p qd-metal` (w3-2-qd-metal) | RUN, exit 0. Compiles only; no test executed. |
| `cargo test --no-run -p qd-export` (w3-2-qd-export) | RUN, exit 0. Compiles only; no test executed. |
| `make pin` / `qd-pin` after the bump | **PENDING (owner-run).** `make pin` was queued at the default load limit of 12 and never got the lock (load 14-100), so it did not run. The harness refuses `qd-pin` with a path outside the repo. |
| Q15 fails-before (`qd-pin` against the unmodified lock 04277f8, sibling 4123dfb) | **PENDING (owner-run).** Same harness refusal; not run. Expected mismatch is inferred from reading `pin.rs`, not observed. |
| `python/tests/test_pin_gate.py` | NOT RUN. |

Lappi breakage: none found. No Lappi source was changed. The one build effect is `Cargo.lock`: the build added
`ojas-train` (dependencies `ojas-core`, `ojas-io`) and an `ojas-train` edge under `ojas-model` [V, `git diff Cargo.lock`,
+9 lines]. That differs from the section 2 expectation of no lock change; it follows from `ojas-model` now depending on
`ojas-train`. It is committed with the bump, as the build produced it.

The pin is **not** verified by the gate until the owner-run `qd-pin` / `make pin` result is pasted here.
