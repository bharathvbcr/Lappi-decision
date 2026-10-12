# Fable: board-sweep ruling (2026-10-12, HEAD cedd684)

Asked by the lead (Opus) after four read-only reconcilers checked the 35 open GitPulse cards against
HEAD. Labels: **[V]** I read the line or ran the command this session; **[R]** a reconciler's or a
document's claim I did not re-run; **[I]** inferred; **[U]** unverified. I edited nothing but this
file; no `gaps.jsonl` append, no cargo, no GPU.

State at ruling time [V]: `git status` = `M gaps.jsonl` (1 added line); worktrees = main at cedd684,
`build/nulctl-red/wt` (detached b8f5ce2, 2 dirty files), `build/span-pin-8a531e8` (detached);
GitPulse insights snapshot (scratchpad/insights.json): one live claude session (the lead, pid 79674),
collisions `overlapping_files=0` across 3 scanned worktrees, codeintel `is_fresh=false` under
GitPulse's embedded analyzer while the `devmap` CLI says fresh (gen 3874). `origin/main` =
f7cb935, local main 2 ahead. tessl HEAD 0b1124f, clean, 163 commits past the pin head cf65d9d.
`hw.memsize` = 64 GiB.

What I did not consult: `ListAgents` (the lead owns that half; the lanes below must run it before
every write). Gap to append by the lead: `GAP-BOARD-SWEEP-RULING-NO-LISTAGENTS-2026-10-12`.

---

## 1. Priority order and closures

### Close now

| Card | Close as | Evidence |
|---|---|---|
| ft-5af5e8ee45e | **done** (criteria 3-4 met; 1-2 moved to ft-0f5a10c9f3e) | `admission.rs:245-280` picks the shape at line 3; `tests/admission.rs:198-345` incl. `documented_request()`; a1b1799 [R, reconciler B read both]. Residual: README still says "Multi-file diffs are refused at admission" — wording fix goes to lane 7, not a reason to keep the card. |
| ft-cbbc092be2f | **superseded** | Tier-B candidates are off for v5 (`campaign/v5-preregistered.json:229` [R]); the planned path `campaign/post-f-queue/perf_tierb_outcome.sh` does not exist (the script is `tools/perf_tierb_outcome.sh`) [R]; re-pointing it to current main would contradict its own pre-registered "phase-3 checkout" condition; the box is gone. Nothing to run. |
| ft-b99f46e3bd6 | **superseded** by `GAP-GITPULSE-INPROCESS-DEVMAP-CALLS-FRESH-STORE-OBSOLETE-2026-10-12` | schema 23/24 error gone (GitPulse opens the store, schema 27/27) [V insights.json]; plugin = binary 0.2.4 [R]; task tools present [R]. The residual (GitPulse embeds a different analyzer than the CLI) is a GitPulse-repo build and belongs on that board; here it is a one-line note: lanes use the `devmap` CLI and say so. |

Not closable yet though mostly done: ft-f8a5a19e51e (code merged 1980975; C2 waits on the human's
record), ft-be802e0780a (push happened; the gaps.jsonl commit is lane 7's; branch deletion is the
human's), ft-ab55b2509d0 (one deletion ruled below), ft-fc7b705092c (C3 met; C1/C2 lane 7).

### Human-only, raised now (only the first is crucial)

1. **Crucial — apply `docs/promotion-decisions.respec-2026-10-08.patch.json` into
   `docs/promotion-decisions.json`** (ft-89275cd4da5). Until then HEAD has 3 red tests and
   `qd_train.ledger verdict` refuses every record, v5's included [R reconciler C ran pytest: 3 failed,
   70 passed]. The lead drafts the commit message naming both gaps; the human commits.
2. Not crucial for the sweep: P1 box cost + 4B download (ft-12da8e67e32), v6 budget (ft-bd80c52e80e),
   caller-record admission and live check (ft-b4f737de235), served-population numbers going public
   (ft-1b5f5f56ec9 C1), pushing the 2 ahead commits and deleting `origin/outreach-2026-10-06` /
   deciding `claude/wizardly-pascal-6k1nci` (ft-be802e0780a), the repo-wide rustfmt moment (2c
   below), the crate's first `unsafe` (ft-afb31d47198), the dc-verify/dc-redact dependency
   (ft-ac6677c7f12 C1). The devtype.route ruling (2i) is mine to make and the human's to veto.

### Agent-executable work, in priority order

| # | Item ids | What | Why this rank |
|---|---|---|---|
| 1 | ft-0eb828f178f | Metal-aware memory guard in `tools/mac_heavy.sh` (2e) | Every Mac GPU job below runs under it; two OOMs and a watchdog panic already. Nothing heavy launches before it lands. |
| 2 | ft-315bcc7a04a | Wire `allocate.rs` (2f) | R9 step 2; v6's Mac-first chain starts here. 998 lines never compiled. |
| 3 | ft-a37989f8c09, ft-f137c48e269 (clippy half) | Revert the prefill kernel swap, remove the "halving" claim, fix the 2+1 clippy items, run the GPU pin (2b, 2c) | `make gates` is red on `clippy -D warnings`; the KNOWN-RED commit is public. |
| 4 | ft-2ed2fd55e93 (items 1,2,3,5), ft-bebe98461d1 | LoRA-2B pre-registration, torch LoRA reference arm, `ckpt_average` merged mode + `--span-head-seed`, the two approved downloads (2g, 2h) | Approved track; no heavy slot needed; unblocks the ojas side. |
| 5 | ft-3163ce2315f items 1*,2,3,6,7,8 | qd-train-metal driver fixes now; 4,5 during the move (2h) | Lappi-side files that survive the move. |
| 6 | ft-e0211ba7412 | Safetensors readers: Rust half with lane 2, Python half with lane 5 (2d) | Small, fail-first, removes a 1,043-node clone. |
| 7 | ft-d665ebab203, ft-a73b8f81b84 | Lint gate green, oracle regen under 3.14.8 (byte-identity condition), worktree-portable pytest | Cheap; the suite must run in pinned worktrees for every runner below. |
| 8 | ft-0f5a10c9f3e step 2, ft-2ed2fd55e93 item 4 | qd-metal span-head loader + `D17` additive readout in qd-export/qd-metal (2g) | Core family; serves v5, v6 and LoRA-2B heads alike. Wave 2 (same files as #3). |
| 9 | ft-a3b6c9ea916, ft-505105265aa | Scoring step with no optimizer + scoring footprint; one host copy per batch in `_decode` | Mac re-scores (span, Arm B) are budgeted as training today; verdict bytes must stay identical. |
| 10 | ft-2d0a5e4b27e, ft-b1dfb6da273 (prereg + driver + probe) | Caller-family builders; Arm B pre-registration and miner driver | R5/R4; the allocation is not pinned until their rows exist or Arm B reads `no_data`. |
| 11 | ft-b5110a5c0ed | Runner hardening committed under `tools/`; S8 email re-run (release, pre-written note) | Heavy; U4 criterion unchanged (rule 2). |
| 12 | ft-a4ef800cc7a 1,2,3(measure),5; ft-5c92df4cc91 (five records); ft-ab55b2509d0; ft-ad52698df9f; ft-be802e0780a (gaps commit); ft-fc7b705092c; ft-1b5f5f56ec9 wording (no numbers); ft-12da8e67e32 (b),(c) | Diagnoses, ledger hygiene, docs wording, P1 pre-registration from the DRAFT | S-M each, no heavy slot except the P1 data fact (light). |
| 13 | ft-ac6677c7f12 C2 | Content guard for `lappi.caller_record` lines in qd-train (and qd-prep's pool door) | Security-narrowing only; flips a limits-statement test deliberately. |
| 14 | ft-0f5a10c9f3e step 3-4, ft-a4ef800cc7a item 4 | Re-score with `--verdicts-out`, two-fold span fit, span entry in the release table; span arm (e) | Heavy, after #1, #8, #9. |
| 15 | ft-882ed8fcd29 C2, ft-afb31d47198 profile | Deferred past this sweep | Heavy, no v6 gate depends on either; C3/unsafe are human. |

---

## 2. Rulings

### 2a. ft-0f1944041b8 — fused CE bf16 logits; 36 fp32 tensors cast to bf16 at load

Facts [V]: `fused_ce.py:170` `torch.matmul(hidden_c, weight.t()).to(compute_dtype)` — with bf16
operands the product is bf16 and only then upcast; `:174` logsumexp runs on the rounded logits;
`:201` `probs.to(weight.dtype) @ weight` rounds the gradient's probabilities to bf16 too (the gap does
not mention this second rounding). `backbone.py:609,617` cast every tensor to the run dtype;
`:1254-1256` scores in the lm_head dtype (fp32 by default), so train rounds and score does not.
The local Qwen3.5-2B header (same text config as 2B-Base) stores exactly **36** F32 tensors:
`linear_attn.A_log` [16] and `linear_attn.norm.weight` [128] in each of 18 GDN layers [V, read the
safetensors header].

**Ruling: keep both as v5 for v6. Close both gaps with this ruling as evidence.**
- v6 is pre-registered as v5's recipe exactly (R8), read against v5-relative guards (choice ≥ 0.8278,
  span ≥ 0.9066) and an Arm B rule that compares `ood_abstain.in_distribution` to v5's same seed. A
  numerics change under those guards is a confound with no measurement behind it; the audit's own
  severity is "cosmetic"/"small, inferred". transformers does both roundings the same way [R].
- No measurement is required for "keep". A report-only measurement (tiny real model, fixed val slice,
  letter-margin delta for fp32-vs-bf16 logit matmul and kept-vs-cast 36 tensors) is permitted as a
  v7 input but is not on this sweep's list and gates nothing.

**For LoRA-2B (a different recipe family, already incomparable with v5 by D6):**
- The fused-CE question is moot: D2's objective is a 17-way CE over `embed[answer ids] + D17`,
  computed in f32 from the f32 residual stream (no bf16 rounding of those 17 logits). The report-only
  full-vocab letter CE may use the fused path; it is report-only.
- The 36 stored-F32 tensors stay **f32 resident** in the frozen base (they are the "norm parameters
  f32" of D3; they do not train). The torch reference arm must match whatever ojas's loader does, so
  `backbone.py`'s load gains an opt-in `keep_stored_dtype` name set, default off (v5/v6 bytes
  unchanged), and the LoRA-2B pre-registration records the per-name dtype map and its digests.
  Fail-first test: with the flag the 36 come back `float32`, without it `bfloat16`, and the v5 load
  path's state digest is unchanged. D7's (c′) identity-at-init bound then measures both stacks on the
  same resident dtypes.

### 2b. ft-a37989f8c09 — `flash_attn_rows` vs `attn_prefill_by_length`

Facts [V]: the swap is one call at `crates/qd-metal/src/model.rs:816-820` (plus the module doc at
`:16`); a766fcf called `nn::flash_attn_rows` with the same arguments plus `cfg.head_dim`. tessl's own
contract (`qwen35.rs`, `attn_prefill` doc): "The matrix units sum in a different order from the
scalar kernel, so the two agree to f32 rounding, not bit for bit." tessl's bench (M5 Pro, ABBA, 9
rounds, 2026-10-03 [R, tessl doc]): 0.116 vs 0.232 ms at T=200, 55.3 vs 176.2 ms at T=8192 per
launch. The pins (`tests/gpu.rs` `PINNED_V1`, `PIN_TESSL_HEAD` cf65d9d) are a property of (weights,
tessl kernels, digest code); tessl is now 163 commits past cf65d9d with GEMM/SiLU changes, so the
pins may fail on either kernel [I]. 9c563ed is on `origin/main` [R, reconciler D ran
`merge-base --is-ancestor`], so "fix before push" is moot. `docs/schema-api.md:471-472` still says
"halving kernel dispatch and GPU synchronization overhead" with no committed measurement [V].

**Ruling: keep `flash_attn_rows` — revert the six lines and the doc strings now; keep the
`decode_slots` batching; fix the clippy items; then run the GPU pin on current tessl.**
- Every released number (the 89/90 replay, the parity sets, the preview build at a766fcf) was
  produced on `flash_attn_rows`. The saving is ~0.7 s of ~2.3 s at 8,185 tokens and nothing at the
  124-400-token sizes a decision takes [R numbers]; serving correctness on the new kernel is
  unmeasured. A kernel change that moves every digest is a pre-registered change with its own parity
  row, not a side effect of a batching commit.
- After the revert: `bash tools/mac_heavy.sh gpu-pin cargo test --release -p qd-metal --test gpu --
  --ignored` (all 7). If the pins pass on tessl 0b1124f: done, no new pin, `PIN_TESSL_HEAD` is
  updated to 0b1124f in the same commit with the ledger row id of that run. If they fail: that is
  tessl drift, not Lappi's; the lane records a `GAP-QD-METAL-PINS-MOVED-WITH-TESSL-0b1124f-2026-10-12`
  with the four digests observed, and the re-pin + `qd-metal-parity` run becomes a human-owned item
  (public numbers rest on it). No re-pin on `attn_prefill_by_length` in this sweep.
- **The "halves" claim is removed now**, not hedged: `docs/schema-api.md:471-472` becomes "packs both
  equal-length passes into one forward pass; the dispatch saving is unmeasured." A measured claim may
  return with a committed benchmark (interleaved A/B, min-of-N) under its own card.

### 2c. ft-f137c48e269 — clippy fix vs repo-wide rustfmt

Facts [V]: `crates/qd-runtime/src/lib.rs:239-240` `#[cfg(test)] mod sha256_parallel_tests` precedes
`pub struct CounterRng` at `:415`; the Makefile already runs `cargo clippy --all-targets -- -D
warnings` inside `make gates` (`Makefile:128-130, 231-235`) and has no fmt gate; no `rustfmt.toml`,
no `rust-toolchain`.

**Ruling: do the clippy fix now; leave fmt entirely to the human; no scoped fmt gate.**
- Move the test module to the end of `lib.rs` (text move only; `CounterRng`'s determinism is pinned
  by `tests/render_contract.rs`). One commit, nothing else. Red: `cargo clippy -p qd-runtime
  --all-targets -- -D warnings` fails on HEAD; green after. Heavy slot (qd-runtime builds are light,
  but the whole-workspace clippy that `make gates` runs links tessl: run it once under
  `mac_heavy.sh -j 2` after lane 2 lands).
- A fmt gate "scoped to files already clean" is a check that cannot fail on the files that matter,
  reported beside gates that can — exactly the pattern CLAUDE.md forbids. A reformat touching ~1.4k
  hunks across every crate would also conflict with the two detached runner worktrees and any pinned
  tree a chain is running from. The human picks the moment (after this sweep's waves land, worktrees
  quiet); that commit contains the reformat and adds `fmt` to `make gates` in the same change, with
  the rustfmt version recorded in the commit message.

### 2d. ft-e0211ba7412 — one safetensors header reader per language

Facts [V]: `qd-export/src/safetensors.rs:150` refuses `header_len == 0`; `qd-metal/src/safetensors.rs:88`
accepts it; both bound at 100 MiB (`MAX_HEADER_BYTES`); Python `memory.py:392` bounds at 64 MiB and
refuses `n <= 0`; the two Rust `parse_header` bodies differ only in formatting, the dtype error text
and the dtype table (export: 16 dtypes; metal: BF16/F16/F32). qd-metal already depends on qd-train,
which depends on qd-export (`qd-train/src/ft_data.rs:26` [R]); qd-metal's Cargo comment (lines 36-47)
argues against a third-party crate, not against a workspace path.

**Ruling.**
- **Owner: `qd_export::safetensors`** (already upstream of qd-metal through qd-train; no third-party
  dependency; a workspace path edge, flagged in the commit message, no human ask). qd-metal's
  `SafeTensors` becomes a thin adapter: `SafeTensorsFile` underneath, `Refusal` mapped to
  `MetalError`, the public API (`names`, `info`, `expect`, `read_into`, `bf16`, `f32`, `u16_words`,
  `widen`) unchanged. qd-metal's `parse_header` is deleted in the same change.
- **Bound: 64 MiB everywhere.** Python's number is the one with a test behind it
  (`test_memory.py:818`); the 2B header is ~70 KB and `MAX_CHECKPOINT_TENSORS` (65,536) × a ~150-byte
  entry is ~10 MB, so 64 MiB refuses nothing real. Record the derivation next to the constant.
- **`header_len == 0` is refused everywhere.** An empty tensor set is not a model. The lane first
  checks with `rg -uu` and a test that no fixture or caller opens a zero-header file; if one does, it
  reports instead of keeping the looser bound.
- **Dtype: do not unify the enums.** The 16-dtype parse happens once in the owner; qd-metal's
  3-variant `Dtype` is a serving-side admission applied at `expect`/`bf16`/`f32` ("this reader loads
  BF16/F16/F32"), with the same refusal text it has now.
- **Tests:** one JSON fixture list of adversarial headers (oversized length, zero length, truncated
  file, bad offsets, overlapping offsets, non-object JSON, unknown dtype) under
  `crates/qd-export/tests/fixtures/`, read by both cargo tests and `python/tests/test_memory.py`, so
  Python and Rust refuse the same cases. Red on pre-change code: zero-length accepted by qd-metal; a
  65 MiB header accepted by both Rust readers.
- **Python half:** `run_control.py:1856-1866` and `tools/verify_checkpoint_inventory.py:56-63`
  delegate to `qd_train.memory.safetensors_header`; the sidecar's offset-vs-file-length check moves
  into a sibling `tensor_ranges(path)` in `memory.py` (one owner), and `verify_checkpoint_inventory`'s
  failure type changes from `SystemExit` to `CheckpointRefused` — say so in the commit.
- Lane split: Rust half in lane 2 (heavy: qd-metal links tessl), Python half in lane 5 (owns
  `memory.py` with the scoring footprint).

### 2e. ft-0eb828f178f — Metal-aware cap in `tools/mac_heavy.sh`

Facts: `tools/mac_heavy.sh:113-128` sums `ps` RSS over the tree and `:201-211` caps on it [V]. The
reconciler's measurement script (`scratchpad/fp.py`, `proc_pid_rusage` flavor 4 via ctypes) is
sound [V, read]; its numbers (RSS +0.04 GiB, `ri_phys_footprint` +3.06 GiB after a 3 GiB MPS alloc;
`footprint -p` agrees) are **[R]** — the lane re-runs it and records command + numbers in the gap, as
the card's first criterion demands. Machine: 64 GiB [V]; MPS ceiling 48 GiB [R, the OOM text]; the
OOM'd extracts sat at ~48 GiB Metal plus 20-24 GiB RSS, i.e. a footprint past physical memory [I].

**Ruling.**
- **Cap on `phys_footprint`, summed over the job's process tree**, read by one `python3 -I`
  helper per poll (`tools/mac_footprint.py`, stdlib `ctypes` + `proc_pid_rusage` RUSAGE_INFO_V4, all
  tree pids in one invocation; python3 is already a repo requirement; `footprint -p` walks VM regions
  and is too slow per poll). Summing over a tree can double-count shared IOKit memory across
  processes; a GPU job is one process and the cap is a safety bound, so over-counting is the right
  side to err on — note it in the script.
- **One cap, not two.** `MAC_HEAVY_MAX_FOOTPRINT_MB` replaces `MAC_HEAVY_MAX_RSS_MB`; a caller that
  still sets the old name exits 64 with a message naming the replacement (loud, not silently
  uncapped). Default **53,248 MiB (52 GiB)**: 64 GiB physical minus a 12 GiB floor for the OS, the
  harness, the MCP processes and a second light session. Both OOM'd extracts would have been stopped
  by it, which is the intent. A job that legitimately needs more sets the variable on its launch line
  and the launch line is recorded.
- **Fail closed, loud.** A reader failure on three consecutive polls stops the job and exits with a
  distinct code (71, "memory guard could not run; result does not count") — a running job with no
  guard is a check that could not run reported as passed. Keep every existing invariant: lock release
  on every path, SIGTERM then SIGKILL after 10 s, exit 70 "result does not count", grandchildren
  covered, the integer validation loop (add the new variable to it and to the exit-64 test).
- **`PYTORCH_MPS_HIGH_WATERMARK_RATIO` is not exported by `mac_heavy.sh`.** It is a torch-only knob
  on a wrapper that also runs cargo; its semantics in torch 2.12.1 are not to be coded from memory.
  A torch job that wants it sets it itself after reading the current docs and records the value in
  its ledger row; the footprint cap is the guard either way.
- **Tests:** `test_mac_heavy_lock.py` gains a footprint test using an injectable reader
  (`MAC_HEAVY_FOOTPRINT_CMD`) that reports over-cap → exit 70 with "footprint" in the message (red on
  the RSS-only script), a reader-failure test → exit 71, and an opt-in real-MPS test under
  `/Users/bharath/.venvs/ml` (3 GiB alloc, short). Swap the script in by rename while no job runs it
  (`ps aux | rg mac_heavy` first).

### 2f. ft-315bcc7a04a — `allocate.rs`

Facts [V]: `crates/qd-prep/src/lib.rs` has no `pub mod allocate`; `main.rs` has no `Allocate`
command; `allocate.rs` (998 lines, 386f9d7) and `tests/allocate.rs` exist; `data/decisions/v6-allocation.json`
pins 1 `as_built` + 7 `capped` pools with three dry-run pins.

**Ruling: wire it now as the one cap table** (`pub mod allocate;`, `Allocate { --config, --pool
NAME=DIR..., --out, --threads }` in `main.rs`), compile, run its unit tests, `tests/allocate.rs` and
`python/tests/test_allocate_pool.py` under the heavy slot. Expect compile errors; fix forward; a test
that encodes something R1/R2 contradict is reported, never weakened. No cap-value ruling is needed:
the values are R2's and are already in `v6-allocation.json`. Two sequencing rules: (i) the three
dry-run pins are replaced only after synth-jarvis/tools(/email) are rebuilt from committed code; (ii)
**no v6 pool is written `applied` until the caller families' rows exist and Arm B has rows or reads
`no_data`** (R5's deadline rule cuts the other way otherwise). Record R1/R9's reason for not moving
`decisions::run` onto `pool.rs` on the card and in the gap.

### 2g. ft-bebe98461d1 / ft-0f5a10c9f3e — the span head and the loader

Facts [V]: `ckpt_average.py:1224-1270` has no span-head-source flag; `AVERAGED_TREES` averages the
head; R7 pre-registers two v6 rows (averaged head on the shared init; fallback seed 0's head);
`span-refit-mac-preregistered.json` arms a-e exist, arm (a) is the validity gate, (b) = avg tower +
seed-1 head, (d) = seed 1 as trained; seed 1 is the only seed checkpoint on this Mac [R].

**Ruling.**
- **The flag is `--span-head-seed N`: the head is taken from the input whose checkpoint body says
  `seed == N`** (never from a file name); refused when no input or more than one input carries that
  seed, together with `--same-seed-trajectory`, and when the run is v6 and that seed's
  `span_head_init` manifest is not the shared one. Manifest records `span_head_source: {mode:
  "seed"|"averaged", seed, input, payload_digest}` so primary and fallback rows can never be pooled.
  It composes with `--from masters` and `--norm-preserving` (which then average the tower only).
  Refit heads are a different artifact (`span_refit_fit.py`); `ckpt_average` offers no option for
  them and refuses a mixed input by name.
- **Which head is served** is not re-decided here: for v6 it is R7 (primary averaged, fallback seed
  0); for the v0.1.x preview it is whatever the span-refit pre-registration's own reading rule
  selects among arms (b), (c), (d) once arm (a) passes its validity gate. Nothing in this sweep
  changes that rule (rule 2); the re-score and the fit (step 3) run after 2e lands.
- **Yes, agents build the qd-metal span-head loader now.** It serves any f32 `span_head.safetensors`
  (v5, v6 and LoRA-2B all export that), it is independent of which head ships, and its host-side
  parts (line-start → token rule incl. the whitespace-only last line, the bilinear scoring, the
  abstain row last, rows = line_count + 1 exactly) are testable on CPU fixtures against Python's
  `SpanPointerHead`. The one thing it must verify before trusting: the hidden state the query reads
  (final-norm vs pre-norm, GAP-G9A) against `backbone.hidden()` with a parity fixture. The GPU test
  is `#[ignore]` and runs in the heavy wave. Rule 7 holds: the loader admits; it emits no PASS.
- The D17 readout (2h) lands in the same lane and files: `score_answer_rows(..., embed[ids] + D17)`
  with a D17 = 0 bit-for-bit test against today's scores.

### 2h. ft-2ed2fd55e93 / ft-3163ce2315f — LoRA-2B and the trainer fixes

**Executable now (lane 4, no heavy slot):**
- `campaign/lora-2b-preregistered.json` from the D6 table and `human-decisions.md`, committed before
  any smoke is read. Digests that exist are pinned (span_head_init-seed0 file, D17 = 0); the LoRA-A
  digest is pinned as a *rule* (kaiming-uniform a=√5 from the run seed at the named shapes), with the
  value recorded in the row and checked against the rule. The file says in words: "not comparable to
  v5 rows; gate verdicts and val metrics on identical sets; Mac rows are quick and cannot promote;
  full-vocab letter CE is report-only." Amendments after a read are void.
- `python/qd_train/lora.py` (reference/oracle only, policy-allowed) + tiny-fixture tests: the (a′)
  bounds — no dW allocated, frozen leaves' gradients absent not zero, identity at B = 0. The (b′)/(d′)
  trajectory bounds wait for the ojas Tape `StepProvider` (not this repo).
- `ckpt_average` merged mode: averages `W' = W + (α/r)·B·A` in f32, rounds once; **refuses
  adapter-level inputs by name** with a typed error; lands with `--span-head-seed` (2g).
- The two downloads (approved 2026-10-09): fetch, record snapshot hashes in the pre-registration.
- D17 in qd-export/qd-metal: wave 2, lane 2b (2g).
- The mlx 8/4-bit measurement: not unless the human asks.

**ft-3163ce2315f — do not wait for the move.** The `ojas-train` move (gp-ft-trainer-loop) is
`ready`, unassigned, not started [R]. D8's "during the move" applied to the loop; the files below are
the Lappi-side driver and survive the move:
- now: `--seed` → data order (`run.rs:108-118`; test: two seeds, two orders), `--min-lr` flag calling
  `LrSchedule::real_ft_with_floor` (`run.rs:154`, `args.rs`), `--resume` (`run.rs:202`; `qd-train`'s
  resume machinery exists; test with a kill -9 on a tiny provider in the heavy wave), the ledger
  label (`run.rs:237`), and the oracle regen (item 8) under the byte-identity condition below.
- item 1 (lr_scale passthrough): now **iff** `MetalProvider`'s inner step is ojas-qwen35's (the
  reconciler's claim [R]); the lane reads `crates/qd-train-metal/Cargo.toml` first and keeps the
  per-entry table exact (`optimizer_table.rs`) and the span-head skip.
- items 4 and 5 (`trainer.rs` second epoch, end-of-run checkpoint after a group refusal): during the
  move, as D8 said — they are the loop.
- **Label: `optimizer_recipe = "f32"`** for the Metal full-FT path (AdamW over f32 parameters, no
  master/bf16 split); `"lora"` for LoRA-2B (D6). Both are added to Python's `OPTIMIZER_RECIPES` with a
  test that a Mac row is never pooled with `master`. The Python edit is lane 4's (it owns
  `real_ft_run.py` in wave 1).
- **Oracle regen (item 8 / ft-d665ebab203):** regenerate `trainer-oracle.json` under 3.14.8 and the
  two other oracle fixtures after the lint reflow; accept **only if every field but `python` (and
  `generator.script_sha256`/`argv` for the reformatted generators) is byte-identical**; any other
  difference stops the lane and is reported as a finding, not a golden to replace. Under that
  condition this is re-verification, not a new golden, and needs no human yes.

### 2i. ft-2d0a5e4b27e — devtype.route: the shape is closed

Facts [V]: DevType's own request builder `LappiRouteRequest.swift` (24cc8d8, 2026-10-06) pins
`task "devtype.route"`, question `Which DevType tool should answer this palette query?`, slot
`route`, options `[resolveDate, runMacro, findSnippet, none]`, context = the query bytes —
byte-identical to `docs/caller-contract.md` §4. `PaletteToolRouter.swift` exposes exactly three
tools; an empty reply is `none` (`LappiPaletteRouting.swift:20` `routedTool ?? "none"`). The
"13 + 3 layout, unverified" sentence in §4 is about tool-selection pools, not this family. The
2026-10-03 DRAFT listed options (a)/(b)/(c); R5 chose (a).

**Ruling: the four-option shape is final; nothing is open about it.** What remains is how rows are
made, and I fix that too: gold is the tool the deterministic resolver would run — `resolveDate` iff
the date parser accepts the phrase (forms from `PaletteToolRouter.swift:122-158`, copied into a
fixture with the DevType commit sha), `runMacro` iff a synthetic macro name matches, `findSnippet`
iff a synthetic snippet name matches, `none` otherwise (the generator's unable and direct classes,
R5). By construction, no model labels anything; real DevType queries are never read (held-out). The
builder is a `synth_tools` variant with ≥ 4 templates per class, 1 held out; 1,500 train / 150 val.
The byte-identity test renders one request and compares it to a fixture of
`LappiRouteRequest.line(...)`'s JSON at 24cc8d8 (sorted keys, unescaped slashes). The context bound
is `CommandPaletteCatalog.boundedQuery`'s cap — the lane reads it and pins it. The human may veto;
absent a veto this stands.

`gitpulse.commit_type`: R5 as ruled; GitPulse now sends the staged patch as exact bytes [R], so the
builder uses the commit's diff bytes unchanged and invalid UTF-8 is not a drop rule.

### 2j. What is wrong, dangerous, or unverified in the reports

1. **Reconciler D's footprint numbers are a claim.** The script is right; its output was not seen
   by me. The lane re-runs and records it (2e).
2. **"qd-metal is held by the training lane"** (cards ft-0f5a10c9f3e, ft-a37989f8c09,
   ft-e0211ba7412) is stale: insights show one live session and zero collisions [V]. The lanes still
   run `ListAgents` before writing.
3. **ft-ac6677c7f12 C2**: the test `a_record_file_copied_into_the_training_tree_is_not_caught_by_the_marker`
   asserts the hole [V]. Flipping it is allowed because it is a documented limit, not a gate; the
   replacement asserts the content guard refuses the copy and the marker test stays. The guard keys
   on the parsed `"record": "lappi.caller_record…"` prefix of any JSONL line within a bounded
   line/byte budget, refuses the whole file on an unparseable line beside a record line, is not
   path-dependent, and sits where JSONL data files open (qd-train's pool reader and qd-prep's input
   door). Security impact: narrows admission only.
4. **ft-ab55b2509d0**: `replay_texts_from_prompts` has two signals (`devmap dead` 0.40 and `rg -uu`
   definition-only) and a port-leftover intent [I]. I rule: delete it (this ruling is the owning
   lane's agreement), after the lane runs `rg -uu` across `~/Code/research/ojas` and
   `~/Code/research/tessl` as the out-of-repo check. `FileRecord` stays (constructed at
   `own_prose.rs:952`; the struct-literal miss is fixed upstream in DevCouncil ec243008 [R]).
5. **ft-a4ef800cc7a item 3**: the 16-option calibration miss is **not** added to v6's reading rule
   (R8 is pre-registered; a gate added after the fact is what rule 2 forbids). It is carried as a
   report-only named item with its ledger numbers re-read from rows. Item 1's string edit must not
   change `passed`/`value`/`n`; the lane checks whether `detail` text is hashed anywhere.
6. **ft-5c92df4cc91**: closing records 1-2 may cite code reads plus the 10-08 stress run
   (`qdprep-check.log` [R]); records 3-5 require the lane to open the cited logs/tests first. A
   commit message is not evidence.
7. **ft-12da8e67e32**: writing `campaign/next-train-first-box-preregistered.json` from the DRAFT
   (with the Mac data fact) is agent work; the launch is the human's. If the v5 shard set is not at
   the recorded path, the lane reports `not run`, never a guessed fact.
8. **ft-b5110a5c0ed**: the runner scripts live in gitignored `build/`; commit a reviewable copy under
   `tools/` (one owner) and run from a detached `git worktree add` at the pinned sha, HEAD check
   inside the lock, per-step timeout recorded as `timed_out`. The S8 email profile/size note is
   written before the run; U4's criterion is unchanged.
9. **ft-a3b6c9ea916**: keep one owner of step construction (`_real_step` with a `training: bool`),
   keep the width/eval_widths footprint bound that caught the 2026-09-30 refusal, and check with `rg`
   who reads `QwenDecisionStep.optimizer` before making it optional. Verdicts byte-identical on the
   tiny real model is the test.
10. **ft-505105265aa**: the batched host copy must keep `.double()` conversion order and compute
    `noul_probability` in the same dtype on the same device as today; parity = verdict JSONL bytes.
11. **ft-1b5f5f56ec9**: agents touch wording only (population labels, the stale "multi-file diffs
    refused" sentence, the six never-replayed families named); no number changes; the human's push is
    the publication step.
12. **gaps.jsonl** is shared by every lane: single-line `O_APPEND` + fsync only; lane 7 commits it
    once per wave after the three ledger tests pass on the exact bytes.

---

## 3. Lane plan (≤ 4 concurrent Sonnet lanes, one shared worktree, path-limited commits)

Rules for every lane: `ListAgents` and `gitpulse_collision_risk` before the first write; DevMap first,
`rg -uu` second, record what neither answered; no `git stash`, no `git add -A`; commit by explicit
path list only; every fix carries a test that is red on pre-fix code and the lane shows the red
output; a heavy slot = one `tools/mac_heavy.sh` job at a time, requested from the lead. The lead
serialises: **wave 1's heavy slots go 2e's real-MPS check (short) → lane 3 cargo (qd-prep,
qd-train-metal) → lane 2 cargo + GPU pin**; nothing else heavy until 2e has landed.

### Wave 1

| Lane | Cards | Files owned (nobody else writes these) | Red-first tests | Heavy slot |
|---|---|---|---|---|
| **1 guard + tooling** | ft-0eb828f178f; ft-d665ebab203 (lint + fixture regen); ft-a73b8f81b84 | `tools/mac_heavy.sh`, `tools/mac_footprint.py` (new), `python/tests/test_mac_heavy_lock.py`; `tools/qd_train_oracle_optimizer_table.py`, `tools/qd_train_oracle_head_digest.py`, `tools/qd_train_oracle_trainer.py` (docstring pin), `python/tests/test_perf_tierb_build.py`, `docs/outreach/evidence/served_families.py`, `crates/qd-train/tests/fixtures/*.json`, `crates/qd-train/tests/fixture_pins.rs`; `python/tests/conftest.py`, `test_gaps_writer.py`, `test_wire_golden_corpus.py`, `test_lint_gate.py` | footprint over-cap → exit 70 (red on RSS script); reader failure → exit 71; `MAC_HEAVY_MAX_RSS_MB` set → exit 64; `test_lint_gate` (red now); checkout-name assert replaced; `test_wire_golden_corpus` honours `QD_BIN` | only the short real-MPS check (ml venv, 3 GiB); `cargo test -p qd-train` against regenerated fixtures goes to wave 3 |
| **2 qd-metal / runtime Rust** | ft-a37989f8c09; ft-f137c48e269 (clippy half); ft-e0211ba7412 (Rust half) | `crates/qd-metal/src/model.rs`, `backend.rs`, `decision.rs`, `safetensors.rs`, `crates/qd-metal/Cargo.toml`, `crates/qd-metal/tests/gpu.rs` (pin head only), `crates/qd-export/src/safetensors.rs`, `crates/qd-export/tests/**`, `crates/qd-runtime/src/lib.rs`, `docs/schema-api.md` | zero-length header refused in qd-metal (red); 65 MiB header refused in both (red); shared fixture list refused identically; `clippy -p qd-runtime -D warnings` red → green; GPU pin run recorded either way | yes: `cargo test -p qd-export -p qd-runtime -p qd-metal`, then `gpu-pin … --ignored` (tens of minutes) |
| **3 qd-prep + trainer** | ft-315bcc7a04a; ft-3163ce2315f items 1*,2,3,6,7; ft-5c92df4cc91 (five qd-prep records) | `crates/qd-prep/src/lib.rs`, `main.rs`, `allocate.rs`, `crates/qd-prep/tests/allocate.rs`, `python/tests/test_allocate_pool.py`; `crates/qd-train-metal/src/{args.rs,run.rs,provider.rs}`, `crates/qd-train-metal/tests/**`, `crates/qd-train/src/ledger.rs` (label only); `gaps.jsonl` appends | `tests/allocate.rs` does not compile (red by construction) → passes; two seeds → two orders; `--min-lr` recorded; label ≠ `master`; `--resume` round-trip on the tiny provider | yes: `cargo test -p qd-prep`, `-p qd-train-metal`, `-p qd-train` (10-30 min each) |
| **4 LoRA-2B + ckpt_average** | ft-2ed2fd55e93 items 1,2,3,5; ft-bebe98461d1 | `campaign/lora-2b-preregistered.json` (new), `python/qd_train/lora.py` (new), `python/qd_train/backbone.py` (keep_stored_dtype flag only), `python/tests/test_lora_reference.py` (new), `python/tests/test_backbone_load_dtype.py` (new), `tools/ckpt_average.py`, `python/tests/test_ckpt_average_*.py`, `tools/real_ft_run.py` (manifest reader for the new `span_head_source` shape + `OPTIMIZER_RECIPES` "lora"/"f32"), `AUDIT/lora-2b-2026-10-09/downloads.md` (new, snapshot hashes) | adapter-level average refused (red: today it would average A and B); `--span-head-seed` picks the body's seed and refuses ambiguity (red); no dW allocated, identity at B = 0 (red: no lora module); 36 tensors f32 with the flag, bf16 without (red) | no (tiny fixtures; downloads are network, ~6 GB) |

### Wave 2 (after wave 1 commits land; lanes renumbered to keep four)

| Lane | Cards | Files owned | Red-first tests | Heavy slot |
|---|---|---|---|---|
| **2b span-head loader + D17** | ft-0f5a10c9f3e step 2; ft-2ed2fd55e93 item 4 | `crates/qd-metal/src/{weights.rs,model.rs,backend.rs,decision.rs}`, `crates/qd-metal/tests/**`, `crates/qd-export/src/export.rs`, new parity fixtures | D17 = 0 bit-for-bit vs today's scores; span decode on a fixture request answered (red: refused today); Rust span scores = Python `SpanPointerHead` on fixtures; hidden-state convention fixture | yes (qd-metal GPU `#[ignore]` tests) |
| **5 scoring Python** | ft-a3b6c9ea916; ft-e0211ba7412 (Python half); ft-505105265aa | `tools/real_ft_run.py`, `python/qd_train/backbone.py` (step construction), `python/qd_train/memory.py`, `python/qd_train/optim.py`, `python/qd_train/run_control.py`, `tools/verify_checkpoint_inventory.py`, their tests | no optimizer state after a scoring step (red); 9B scoring budget no longer refused for optimizer state (red); verdict bytes identical on the tiny model; Python readers refuse the shared fixture list (red for the two private readers) | no (tiny model in the ml venv) |
| **6 v6 data** | ft-2d0a5e4b27e; ft-b1dfb6da273 (prereg + driver + probe script); ft-b5110a5c0ed | `crates/qd-prep/src/commit_type.rs` (new), `synth_tools.rs`, `main.rs`/`lib.rs` (registration only), `data/synth/*`, `data/decisions/v6-allocation.json`, `python/qd_data/{sources.py,…}`, `campaign/v6-arm-b-mining-preregistered.json` (new), `tools/arm_b_mine.py` (new), `tools/stress_runner.sh` (new), `crates/qd-prep/tests/stress_v6.rs` (S8 profile path only) | rendered request = pinned app fixture bytes (red: no builder); `none` only from unable/direct; GitPulse repo refused as a producer; miner refuses held-out paths and v5-train duplicates; `timed_out` is its own state | yes: `cargo test -p qd-prep`, the S8 release re-run, then the 200-row Arm B probe (after 2e) |
| **7 diagnoses + ledger + docs** | ft-a4ef800cc7a 1,2,3m,5; ft-ab55b2509d0; ft-ad52698df9f; ft-be802e0780a (gaps commit); ft-fc7b705092c; ft-1b5f5f56ec9 wording; ft-5af5e8ee45e README; ft-12da8e67e32 (b),(c); ft-ac6677c7f12 C2 | `python/qd_train/needle.py`, `python/tests/test_needle_ft_contract.py`, `python/qd_train/replay.py`, `README.md`, `HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md` (population note), `AUDIT/decider-2b-comparison-2026-10-12.md` (new), `HANDOFF/ood-feature-space-feasibility-2026-10-12.md` (new), `campaign/next-train-first-box-preregistered.json` (new), `crates/qd-train/src/held_out.rs`, `crates/qd-train/tests/caller_records_are_held_out.rs`, `gaps.jsonl` | needle verdict names the failing bucket (red on the current string); a copied caller-record file refused by content (red: accepted today); gaps ledger tests on the exact committed bytes | `cargo test -p qd-train` for the content guard (short); the P1 data fact is light |

### Wave 3 (verification; every item a serialised heavy job, each a ledger row or a logged "not run")

1. Whole-workspace `cargo clippy --all-targets -- -D warnings` and `make gates`.
2. `cargo test -p qd-train` against the regenerated fixtures in a fresh worktree; the full pytest in
   that worktree (ft-a73b8f81b84's log).
3. Re-score with `--verdicts-out` on the head the span-refit pre-registration selects →
   `qd-calib-fit --population two-fold` → span entry in the release table (ft-0f5a10c9f3e 3-4).
4. Span arm (e) from `build/span-pin-8a531e8` (ft-a4ef800cc7a item 4).
5. Arm B: 200-row probe, then the scan in ≤ 6 h lock chunks under the pre-registered formula.
6. qd-train-metal `--resume` kill -9 test on a tiny MPS provider.

Deferred past the sweep: ft-882ed8fcd29 C2, ft-afb31d47198, ft-3163ce2315f items 4-5 (with the move),
the mlx quantisation measurement, anything that rents a box.
