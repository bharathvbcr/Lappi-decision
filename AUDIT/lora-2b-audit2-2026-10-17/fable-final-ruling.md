# Fable final ruling: LoRA-2B on the M5 Pro through the ojas Metal Tape, bf16 and quantised arms (2026-10-17)

Heads: ojas main 5b67cdd [V `git rev-parse`], Lappi main bffcf83 [V], tessl 42f1d52 [V]. Read-only: no code, brief, board or commit edited; no GPU job; no process touched. Labels: **[V]** read at the cited line or ran the command this session; **[A]** an audit's claim I did not re-read; **[I]** inferred; **[U]** unverified. Where I rely on an audit claim it is re-read here or carried as [A]; no [I] is laundered into [V].

Evidence re-read in code this session (the claims the ruling stands on):
- No `ojas-train` crate: `Cargo.toml:3-22` members, `git ls-files | rg '^ojas-train/'` = 0 [V].
- `DType = {F32, Bf16, F16, U32}`, `ojas-core/src/dtype.rs:5-10` [V]; no quantised type anywhere on the training path [A, audits A/C/D with `rg -uu`].
- Metal `adamw_step` is per tensor, check-then-write inside one call (`ojas-metal/src/backend.rs:2270-2300`) and `Trainer::apply` loops tensors (`ojas-model/src/trainer.rs:899-922`) [V]: not all-or-nothing across a list.
- No adapter read/write in ojas src: `rg -uu 'lora_A|lora_B'` over `ojas-model/ojas-qwen35/ojas-io/ojas-capi/ojas-data src` hits only the doc comment `lora.rs:289` [V].
- Readout seam on main: `forward_hidden` (:325), `forward_hidden_lora` (:338), `RightPadded` (:471), `gather_rows` (:621) in `ojas-model/src/qwen35/forward.rs`; `check_gpt2_pre_tokenizer` at `ojas-data/src/bpe.rs:614,653` [V]. D17, the 17-way CE and the report-only full-vocab CE exist only as test composition (`ojas-model/tests/qwen35_common/readout.rs`) [A].
- The memory rows are full-gradient, not LoRA (`ojas-qwen35/tests/gpu_memory_rows.rs:15-23` module doc) [V]. Rows: T=2,048 bf16/bf16-table 15.35 GB, 8.50 s; T=9,638 18.02 GB, 41.35 s; T=9,638 bf16/f32-table 18.97 GB, 40.60 s and 34.52 s (`bench/results/2026-10-14-bf16-tied-table/rows.jsonl`, `2026-10-11-bf16-frozen-base/rows.jsonl`) [V]. Tape gate: loss 4.54e-5 rel, worst grad 4.05e-3, peak 17.60 GB under a 28 GiB budget (`2026-10-15-qwen35-2b-tape-gate/README.md:23-37`) [V]. Wrapped-linear bf16 bytes 2,745,434,112 (`bf16_param_bytes`, tape-bf16 arm) [V].
- Lappi's loop feeds one sequence per `accumulate` call, Overwrite then Add into the bank; the step is one token-budget batch; non-finite loss refused before clip and AdamW (`crates/qd-train/src/trainer.rs:1-40, 745-800`) [V]. `ParamSpec {name, shape}` only (`step.rs:49-52`) [V]. v5 row 8079c966: `recipe.batch_tokens 35403`, 12,176 batches = 12,176 optimizer steps = 12,176 micro-batches per epoch, `plan_max_width 9638` (`ledger/h100x2-v5-2026-10-03.jsonl`) [V].
- `OPTIMIZER_RECIPES` is a closed dict `{bf16, master, kahan, f32, lora}` (`tools/real_ft_run.py:766-772`) pinned by `python/tests/test_real_ft_score_checkpoint.py:207` [V].
- `upstream.lock.json` pins ojas `132b46e`; ojas HEAD is `5b67cdd` [V file read]; the bump commit 21fb7ad is on Lappi main [V `git branch --contains`]. `make pin` failing at 5b67cdd is audit B's run [A]; I did not re-run it (it rewrites Cargo.lock).
- ojas `LoraKind::suffix` (`ojas-model/src/qwen35/params.rs:697-708`, `LORA_KINDS` :681-692) equals Lappi `LORA_TARGET_SUFFIXES` (`python/qd_train/lora.py:89-100`), ten for ten [V]: audit A's open [U] on the target list is closed, and the 150-linear count follows by construction (24x3 + 6x4 + 18x3).
- tessl `GEMM_MAX_M = 8` (`kernels/gemm_q4_mlx.metal:13`) [V]; gemma-metal bf16 rounding ties-up (`gemma-metal/src/quant.rs:572-576`) [V].
- Lappi board (stdio `gitpulse-mcp`): ft-2ed2fd55 "LoRA-2B lane (Lappi side)" review P1, criterion 5 already owns the mlx gate "if the human later asks"; ft-3163ce23 "Mac trainer blockers" review P1; ft-0eb828f1 mac_heavy Metal cap review P1 [V].
- Audit error ignored as instructed: "Lappi main 42f1d52" was tessl's HEAD.

---

## 1. Verdict and critical path

**Neither arm is trainable on this Mac today.** What is merged is every primitive the bf16 arm needs (frozen leaves, dX-only backward, bf16 lanes and tied table, CeGrads, LoRA library with scaling-1 parity, the readout seam, the real-2B Tape gate at 17.60 GB). What does not exist is the thing that runs them: no trainer drives the Qwen3.5 Tape (ojas-model's `Trainer` is nanolab-only; Lappi's loop has no LoRA-capable provider; `rg StepProvider` in ojas hits only ojas-cuda's unrelated trait) [V/A]; no adapter is saved or loaded anywhere [V]; D17 and the 17-way CE are test composition, not product code [A]; no LoRA step has been timed or measured [V]; Lappi's build is red on `pin` against ojas main [A]. The quantised arm has, in addition, no gate measurement, no decision record in Lappi, no pre-registration, and no quantised type or kernel [V/A]. Memory is not the blocker for either arm: the full-gradient upper bound at T=9,638 is 18.02 GB against memguard's 32 GiB per process [V]; time is (section 2).

**Ordered critical path to the first bf16 quick run** (the 0.8B-Base 60-step smoke; "owner" = card that owns the step):

| # | Step | Owner | Repo |
|---|---|---|---|
| 1 | LoRA memory and tokens/s rows (a `tape-lora` arm in `gpu_memory_rows.rs` composed from merged primitives; 2B-Base and 0.8B-Base at T=2,048 and 9,638; exact peak under a 28 GiB budget; tokens/s). Needs no trainer; it is the input to the step-count amendment (section 2) and closes GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01 for the LoRA step. | ft-2bbced18 crit 7 (+ new crit below) | ojas |
| 2 | Owner yes on the Lappi pin bump 132b46e -> 5b67cdd (or once, to the phase-A sha), with Cargo.lock regenerated (ojas-data now pulls ojas-io [A]). Until then Lappi gates are red on pin and `qd-train-metal` refuses or marks rows `unpinned ojas`. | owner (section 6); Lappi handoff `AUDIT/ojas-train-move-2026-10-14/fable-ruling.md` Q2 | Lappi |
| 3 | ojas-train phase A: move the eight modules + `groups.rs`, hand-rolled sha256, `pyjson` onto `ojas_io::JsonValue`, `ParamSpec` gains `trainable` + storage dtype. Lappi step 6 switches `qd-train` onto it and deletes the copies. | ft-b5e64557 crit 0, 2 (ojas); Lappi ft-3163ce23 / handoff step 6 (Lappi) | both |
| 4 | Phase B, the Tape `StepProvider`: one sequence per `accumulate` into an f32 adapter bank (sum across the batch's tapes), `hidden_at` + `HiddenGrad` callback through `forward_hidden_lora` + gather, `RowTargets` scale 0 = value only, `grad_sq_norm` over trainables, two-pass all-or-nothing AdamW, `step_count`, `read_parameters`, `save_state`/`load_state` = the adapter checkpoint (names, shapes, dtypes, moments, step, data digest, base sha). Dropout is 0 in the recipe, so no dropout key. | ft-b5e64557 crit 3, 6, 7, 8, 13 | ojas |
| 5 | Host-side D17 objective in Lappi's `qd-train` (R = E + D17 from the 17 embed rows read by name, 17-way CE, dz/dh/dD17, span head as today, report-only full-vocab CE through scale-0 `RowTargets`), with the committed fixture against `AnswerRowDelta`. **No card owns this** (ojas ft-9c4ae3c8 owns only the seam; Lappi ft-2ed2fd55 has no criterion for it). | new criterion on Lappi ft-2ed2fd55 (section 4, Lappi side) | Lappi |
| 6 | LoRA-aware parameter plan and ledger row writer in Lappi (adapters + D17 + span head at lr_scale 0.5, wd 0.01 on all, `optimizer_recipe "lora"`, the seven digests of `init.record_in_every_row`, quick reason). **No card owns the plan/ledger pieces.** | new criterion on Lappi ft-2ed2fd55 | Lappi |
| 7 | Torch reference-arm driver (shard loop, supervision, span head, D17, checkpoint writer in the `lora-merged` layout, ledger row). Needed for bound (d'). | Lappi ft-2ed2fd55 crit 1 (driver clause to add) | Lappi |
| 8 | 0.8B-Base 60-step smoke, both objectives, vs torch MPS bf16 and fp32 CPU. **First bf16 quick run.** | ft-da2c4ab1 crit 4 | both |
| 9 | Amendment 1 (step definition and N; section 2) -> reduced 2B bf16 run with kill -9 resume. | owner yes; ft-da2c4ab1 crit 5 (superseded text) + new crit | both |
| 10 | Adapter checkpoint -> Lappi converter into the `lora-merged` layout -> `ckpt_average --from lora-merged` (one seed: the one-seed D17 route) -> qd-export -> qd-metal score. **The converter has no owner.** | ft-58e316a9 (new crit, ojas layout contract) + new criterion on Lappi ft-2ed2fd55 (converter) | both |

**Ordered critical path to the first quantised quick run** (everything above, then):

| # | Step | Owner | Repo |
|---|---|---|---|
| Q1 | Stage 0, the D3 gate: `mx.quantize` 2B-Base to q8 g64 and q4 g64, >= 500 val prompts (never held-out), 17-row letter log-softmax bf16 vs q8 vs q4, bars 99.0% / 0.02 nats written before the run, mlx version named, numbers into a new `docs/fine-tune-plan.md`. The same tool writes the **fake-quant base file** (dequantize(quantize(W)) for the 150 wrapped linears, bf16; every other tensor the original) for 2B-Base and 0.8B-Base, with sha256. | Lappi ft-2ed2fd55 crit 5 (+ file clause to add) | Lappi |
| Q2 | Owner decision record in Lappi (the memory-need waiver in the owner's words; q8 only; q4 not overridden) and the quantised arm's sibling pre-registration (same D6 recipe; quantised set; mlx version; base-file sha256; recipe label; merge target), committed before any quantised result is read. | owner yes (section 6); ft-da2c4ab1 crit 7 names it; Lappi ft-2ed2fd55 new crit | Lappi |
| Q3 | ojas: the tiny-fixture parity over fake-quant weights (ft-a1a4ba4d crit 8) and a load path that pins the base by the fake-quant file's sha256. No quantised dtype, kernel or host quantiser is needed for this run. | ft-a1a4ba4d crit 8 + new crit; ft-b5e64557 crit 14 (narrowed) | ojas |
| Q4 | 0.8B fake-quant smoke vs the torch arm over the same fake-quant weights; then the reduced quantised 2B run; merge into dequant(Q(W)). **First quantised quick run = Q4's smoke.** | ft-da2c4ab1 crit 8, 9; ft-58e316a9 crit 6 | both |

Effort [I, one engineer, heavy runs under mac_heavy]: bf16 path steps 1-8 are the bulk (phase A + B are the two multi-day items); the quantised arm adds 3-5 days (Q1-Q3), nearly all Lappi-side, and zero kernel work.

---

## 2. What a step is; the window conflict; how the count changes

**Ruling: the pre-registration is internally consistent once its words are read against the code; audit D's [U] is resolved.** One step is one optimizer step = one token-budget batch: the planner packs rows of one bucket width so that rows x width <= 35,403 padded tokens (v5: 12,176 such batches per epoch, `plan_max_width 9638`, micro_batches == optimizer_steps [V ledger]); `grad_accum 1` means one such batch per step. The provider receives the batch **one sequence per launch** (`trainer.rs:745-780`: `BankMode::Overwrite` for the first sequence of the step, `Add` for the rest; gradients summed in the bank, as torch's per-row `backward()` would). "Micro-batch 1" in the memory rows names that per-launch width, not the batch. Consequences: (a) the step's device peak is the widest sequence's (T=9,638 row), plus an adapter-sized bank (~63 MB f32 for 15.7 M adapter params [I]) — memory does **not** scale with 35,403; (b) time does: each step processes ~35,403 padded tokens.

**The conflict is real and is time, not memory.** Measured full-gradient Tape rates (single step, warmup included, no optimizer): 203-246 tok/s at T=2,048 and 233-281 tok/s at T=9,638 [V rows] -> 126-174 s per 35,403-token step -> 200 steps = 7.0-9.7 h, 400 steps = 14-19 h. The LoRA step skips frozen dW and the frozen head's [V,d] gradient; its speed-up is unmeasured (audit D's "90-120 s" is [I]). Under no measured reading do 400 steps fit the approved 3-6 h window; 200 fits only if LoRA is >= 1.3x faster than full-gradient and the machine is quiet. Short-row buckets (mean row ~691 tokens) run at the lower per-token rate, so the realistic mean is nearer 160 s/step [I].

**What changes and through what process.** Rule 2 is not engaged: the step count of a `quick` Mac run is a schedule parameter, not a gate, threshold, seed count or held-out set; nothing is reclassified. The pre-registration's own `amendments_rule` allows a committed amendment before any result is read, and no LoRA-2B result exists (`rg optimizer_recipe.*lora ledger/` empty [A]). Its content is the owner's (D6, H4). I rule the default and put the yes/no to the owner (section 6):

- **Amendment 1** (one commit in Lappi, original text kept in history): (i) states the step definition above in words; (ii) the window binds: the reduced 2B run's step count N is fixed **from the measured LoRA tokens/s row** (ft-2bbced18 crit 7, at the run's bucket mix or the T=2,048 rate as the conservative figure) as the largest N <= 400 whose N x 35,403 tokens fit 80% of the approved window (6 h upper), with warmup max(1, N//20) and cosine over N; N and the row id are written into the amendment before step 0 and recorded in the row; (iii) `recipe.steps.mac` keeps "about 200 to 400" only as the original intent, struck. No floor on N: a quick run's job is the mechanics and a scored row; the owner may instead extend the window (the alternative in section 6).
- The 0.8B smoke (60 steps, ~35 min to 1 h at the measured rate scaled by params [I]) needs no change.

---

## 3. The quantised arm

**Scope ruling.** The owner's request stands as the owner's spending decision and is honoured as follows:

1. **The quantised arm is "LoRA over a q8 (MLX affine, g64) frozen base", and its first quick runs are run as fake-quant-at-load**: the bf16 trainer over a base file whose 150 wrapped linears are dequantize(quantize(W)) from mlx, pinned by sha256. This gives bit-for-bit the numerics a resident-q8 kernel path would give (dequant is deterministic; a dequant-to-bf16-scratch kernel produces the same bf16 values per use as dequant once at load), so the arm's scientific question — does LoRA over the q8 base train and score as the bf16 base does — is answered with **zero kernel work**. What it does not give is the memory saving: 1.29 GB (2.745 GB bf16 wrapped linears vs ~1.46 GB q8 g64 [V bytes; I arithmetic]) of an estimated 12-14 GB step; no pre-registered shape needs it (18.02 GB full-gradient upper bound at T=9,638 fits under 32 GiB [V]).
2. **Stages in this track:** audit C's stage 0 (the gate, Lappi) and the fake-quant base file (Lappi tool); ojas's fixture parity over fake-quant weights (ft-a1a4ba4d crit 8) and sha-pinned base load. **Stage 1** (an MLX-exact host quantiser and dequant oracle in Rust) is **not** on the first run's path: the base is a file pinned by sha256 exactly as the bf16 base is; stage 1 is needed only by stage 2. **Stage 2** (resident q8 storage, dequant-to-scratch kernel in canonical tessl, 1-2 weeks) and **stage 3** (fused dequant GEMM, 3-5 weeks) are deferred to a new backlog P2 card and start only on the owner's yes after the fake-quant reduced run is scored (section 6).
3. **The memory-need waiver.** My 2026-10-17 recording on ft-a1a4ba4d (criterion 5) is valid as the owner's decision — the "demonstrated memory need" was a build precondition in my D3, not a Lappi gate, and the owner owns the spend — but it is recorded only on the ojas board. Lappi rule 10 and the pre-registration's `not_decided_here` mean it must be written in Lappi in the owner's words before the quantised pre-registration is written: an owner-decision file (`AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md`, or an addendum to `AUDIT/lora-2b-2026-10-09/human-decisions.md`) carrying the arithmetic above. **The bf16 pre-registration is not amended for this**: its `base.precision` "No 8-bit or 4-bit ... not built here" and `not_decided_here` remain true of the bf16 arm; the quantised arm gets a **sibling** pre-registration (ft-da2c4ab1 crit 7 already says so). Audit B's "prereg amendment" is satisfied by the sibling file, not by editing the bf16 one.
4. **The quality gate stays as D3 set it** (argmax >= 99.0%, mean |dlogp| gold letter <= 0.02 nats, >= 500 val prompts, never held-out). q8 the format; **q4 is not built**: Unsloth's advice concerns bitsandbytes 4-bit (NF4) and is withdrawn for nothing else [A audit C fetch], but the owner has not given the written override and D3 requires it; q4 is measured in the gate (cheap, informative) and not built.
5. **Recipe label.** `OPTIMIZER_RECIPES` is closed and pinned [V]; adding a label is a Lappi code + test change, but rows must never pool: the quantised rows carry `optimizer_recipe "lora"` (the optimizer is identical) and a **new family id** `lora-2b-q8` with `base.format = "mlx-affine-q8-g64-fakequant"` and the base-file sha in the row, and a pooling rule in the sibling pre-registration that a reader pooling by `optimizer_recipe` must also key on the family id. If Lappi's ledger readers pool on `optimizer_recipe` alone, then add `"lora-q8"` to `OPTIMIZER_RECIPES` with the label test updated in the same commit — the Lappi lane decides which by reading its readers; either way the label is recorded before the first quantised row.
6. **Merge target for export:** dequant(Q(W)) + (alpha/r) B A — the base the adapters trained over — rounded once to bf16; merging into the original bf16 W is refused by base-sha mismatch. Release is a dense bf16 tower through qd-export/qd-metal unchanged; serving a quantised base is not in this track (D5 reversal not triggered). Manifest records format, group size, mlx version and the base file's sha256. Comparability with the bf16 arm is at the gates and val metrics only (the same sentence D6 wrote for v5).
7. **ft-a1a4ba4d status and priority:** `ready` is right for its ojas-side criterion 8 (can start now); **P1 stands** because the owner asked for it, but **severity high -> medium**: it is not on the bf16 critical path, needs no kernel work for its first run, and has no memory need. Its title and criteria 2-3 are re-scoped (section 4). Sequencing: no quantised run starts before the bf16 reduced run's row exists.

---

## 4. Board edits (machine-applicable)

Conventions: `set` = fields to set; `append` = criteria to add verbatim, in order; `mark` = prefix the existing criterion's text with the bracketed note shown (text otherwise unchanged; nothing deleted; no threshold moved). Agent ceiling is `review`; `done` is the owner's verify (memory: done gate is Manvi verify).

### ft-b5e64557f88bdb433700db571b7a78ec (ojas-train move + Tape StepProvider)
- set: status `ready`, priority `1`, severity `high` (unchanged).
- mark [7]: `[STALE CITE 2026-10-17: per-tensor check-then-write is now ojas-metal/src/backend.rs:2270-2300 (device.rs:3277-3287); the :2104/:2937 sites are linear_cross_entropy_mean]`
- append:
  1. `(Fable 2026-10-17) The Tape StepProvider's accumulate takes one sequence per call, as Lappi's loop feeds it (crates/qd-train/src/trainer.rs:745-780: BankMode::Overwrite for the first sequence of a step, Add for the rest): the adapters' gradients and the host entries' (D17, span head) are summed in an f32 bank across the sequences of one token-budget batch (35,403 padded tokens: rows of one bucket width, widths up to 9,638), one tape per sequence, no activation kept across sequences, so the step's device peak is the widest sequence's, not the batch's. A test with a 3-row batch of unequal widths matches a torch run summing per-row backward() calls within the (a') bound.`
  2. `(Fable 2026-10-17) RowTargets with scale 0 (the report-only full-vocab letter CE) returns the value, runs no backward and allocates no [V, d] gradient; a Budget-delta test pins it (the Tape fused_ce has no value-only call today: ojas-model/tests/qwen35_common/readout.rs records it through lin_ce).`
  3. `(Fable 2026-10-17) Criterion 7's all-or-nothing is a two-pass update inside the Tape StepProvider's adamw_step over the trainable list (check every tensor's step finite, then apply every one), not a change to Backend::adamw_step's per-tensor contract; a test injects a non-finite gradient in tensor k > 0 and asserts every tensor and every moment unchanged.`
  4. `(Fable 2026-10-17) Criterion 8's RNG state is the data/batch-order RNG and the consumed digest only: the recipe's LoRA dropout is 0 (Lappi campaign/lora-2b-preregistered.json recipe.lora.dropout), so no dropout key is checkpointed, and a non-zero dropout refuses to checkpoint until a key is stored. The checkpoint layout (tensor names, shapes, dtypes, base sha256, step, moments) is documented and versioned in ojas so Lappi's converter (ft-58e316a9) reads it by contract.`
  5. `(Fable 2026-10-17) Criterion 14 is narrowed for this track: the quantised arm's first runs load a fake-quant bf16 base file (ft-a1a4ba4d), so ParamSpec's storage dtype names bf16 and the checkpoint names the base by that file's sha256 plus the quantizer record (format, group, mlx version) copied from the arm's pre-registration; the block-quantised storage dtype lands with the stage-2 card.`

### ft-9c4ae3c8b3f95ce19204bbf1279c05cc (17-row readout)
- set: status `review`, priority `1` (unchanged).
- mark [0]: `[MET 2026-10-17: forward_hidden :325 / forward_hidden_lora :338, ojas-model/src/qwen35/forward.rs on main 5b67cdd]`
- mark [1]: `[MET 2026-10-17: gather_rows forward.rs:621, torch parity in tests/qwen35_readout.rs, Metal in qwen35_readout_metal.rs]`
- mark [2]: `[SUPERSEDED 2026-10-17 for product code by the 2026-10-15 host-side ruling: ojas's part (the seam and the test-composition parity, tests/qwen35_common/readout.rs) is met; the product D17 entry (R = E + D17 from the 17 embed rows read by name, dz, dh, dD17) is Lappi's objective (ft-2ed2fd55) with the fixture against AnswerRowDelta committed there]`
- mark [3]: `[SUPERSEDED 2026-10-17 by ft-b5e64557's scale-0 RowTargets criterion: the value-only full-vocab CE is the provider's; the test-only composition here stays as the parity pin]`
- mark [4]: `[MET in tests, not re-run by the 2026-10-16 audits: qwen35_readout.rs right-pad tests, RightPadded refuses left padding, holes, empty rows]`
- mark [5]: `[MET 2026-10-17: check_gpt2_pre_tokenizer, ojas-data/src/bpe.rs:614,653, on the directory entry point; the bare-vocab path is unverified]`
- append:
  1. `(Fable 2026-10-17) Review closes on a recorded run on main of ojas-model/tests/qwen35_readout.rs, qwen35_readout_metal.rs and the adversarial readout tests under tools/mac_heavy.sh (counts logged here), plus a test that load_hf_gpt2 on a bare vocab path (no tokenizer.json) is also refused or is documented as not an entry point for Qwen ids.`

### ft-da2c4ab1677189ec57271befbd0040c1 (LoRA-2B end-to-end acceptance)
- set: status `ready`, priority `1`, severity `high` (unchanged).
- mark [1]: `[MET WITH DEVIATION 2026-10-17: the gate ran on Qwen3.5-2B-Base against transformers' fp32 reference, not the instruct snapshot against the provider (Fable 2026-10-15 redesign); bench/results/2026-10-15-qwen35-2b-tape-gate]`
- mark [2]: `[MET 2026-10-17: Lappi 5b6d6cb, amended c61584b before any result]`
- mark [5]: `[SUPERSEDED 2026-10-17 in its "roughly 200-400 steps inside the approved 3-6 h window" clause by amendment 1 (next criteria): at the measured full-gradient Tape rate (203-281 tok/s, bench/results/2026-10-11-bf16-frozen-base and 2026-10-14-bf16-tied-table) a 35,403-token step is 126-174 s, so 400 steps cannot fit 6 h and 200 is marginal; the kill -9, export and scoring clauses stand]`
- append:
  1. `(Fable 2026-10-17) One step is one token-budget batch of 35,403 padded tokens (rows of one bucket width, widths up to 9,638, grad_accum 1), fed to the provider one sequence per launch with gradients summed (Lappi crates/qd-train/src/trainer.rs:745-780); "micro-batch 1" in the memory rows names that per-launch width. The step's device peak is the T=9,638 row's; its wall is ~35,403 tokens at the LoRA rate.`
  2. `(Fable 2026-10-17) Amendment 1 to Lappi campaign/lora-2b-preregistered.json, committed before any LoRA-2B result is read and after the owner's yes: (i) the step definition above in words; (ii) the window binds: the reduced 2B run's step count N is fixed from ft-2bbced18 criterion 7's measured LoRA tokens/s row as the largest N <= 400 whose N x 35,403 tokens fit 80% of the approved window, with warmup max(1, N//20) and cosine over N; N and the row id are written into the amendment before step 0 and into the ledger row. The owner may instead extend the window (owner item).`
  3. `(Fable 2026-10-17) Order of the quick runs: LoRA memory and tokens/s rows (ft-2bbced18 criterion 7) -> 0.8B bf16 smoke -> amendment 1 -> reduced 2B bf16 run -> quantised 0.8B smoke -> reduced quantised 2B run. No quantised run starts before the bf16 reduced run's ledger row exists.`
  4. `(Fable 2026-10-17) The first quantised quick runs are the bf16 trainer over a fake-quant base: a safetensors file whose 150 wrapped linears are dequantize(quantize(W)) from mlx (q8, group 64, bf16 scale and bias; mlx version recorded) and whose other tensors are the originals, written by the Lappi gate tool for 2B-Base and 0.8B-Base, pinned by sha256 in the quantised arm's sibling pre-registration and in the checkpoint's base name. No quantised dtype, kernel or host quantiser is on this run's path; a later resident-q8 (stage 2) run must reproduce this run's tiny-fixture loss bit for bit.`
  5. `(Fable 2026-10-17) Criterion 7's D3 addendum is a Lappi owner-decision file (AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md or an addendum to AUDIT/lora-2b-2026-10-09/human-decisions.md) in the owner's words: the memory-need waiver, q8 only, no 4-bit override, the memory arithmetic (bf16 wrapped linears 2.745 GB vs ~1.46 GB q8 g64, 1.29 GB of a 12-14 GB estimated step, no shape needs it). The bf16 pre-registration is not edited for the quantised arm.`
  6. `(Fable 2026-10-17) The quantised rows never pool with the bf16 rows: family id lora-2b-q8 with base.format and the base-file sha256 in every row; the label rule (keep optimizer_recipe "lora" and key readers on the family id, or add "lora-q8" to OPTIMIZER_RECIPES with python/tests/test_real_ft_score_checkpoint.py:207 updated in the same commit) is decided by the Lappi lane from its ledger readers and written into the sibling pre-registration before the first quantised row.`

### ft-2bbced18f11d70727f58e25696dcd5ed (LoRA step cost)
- set: status `backlog` -> `ready`, priority `1`, severity `medium` -> `high` (criterion 7 is now the next GPU job on the track and gates amendment 1).
- mark [4]: `[NOTE 2026-10-17: decided by criterion 7's row against the provider's 452 tok/s at T=8192 (Lappi AUDIT/ojas-training-2026-10-01/mac-8k-train-step-bench.log:35); the Tape's full-gradient 233-281 tok/s at T=9,638 already exceeds half of 452, so the D1 reversal is not triggered by the measured upper bound; the provider bf16 T=2048 row was SIGTERMed by memguard and is not required]`
- mark [8]: `[SUPERSEDED 2026-10-17 for this track: the admitted quantised base is a fake-quant bf16 file for the first runs, so its rows equal the bf16 rows by construction; resident-q8 rows belong to the stage-2 card]`
- append:
  1. `(Fable 2026-10-17) Criterion 7 does not wait for ojas-train: a tape-lora arm in ojas-qwen35/tests/gpu_memory_rows.rs composed from merged primitives (bind_frozen, forward_loss_lora or forward_hidden_lora + gather_rows, r = 16 alpha = 16 on the ten LoraKind suffixes, bf16 linears and bf16 table, per-tensor Backend::adamw_step over the adapters), 2B-Base and 0.8B-Base at T = 2,048 and T = 9,638, each row with exact device peak, first-step wall, and tokens/s as the mean of 3 steps after one warmup step, under a 28 GiB MetalBackend budget and Lappi-decision/tools/mac_heavy.sh, rows under bench/results/. Its tokens/s is the input to amendment 1 (ft-da2c4ab1). Host footprint (memguard's metric) is recorded beside the device peak.`

### ft-a1a4ba4d06d18ef307fa7776545e9a52 (quantised frozen base)
- set: status `ready` (unchanged), priority `1` (unchanged), severity `high` -> `medium`, title -> `Quantised LoRA arm (owner 2026-10-17): mlx quality gate first; the first quick runs train LoRA over a fake-quant q8 base file; resident block-quantised storage and dequant GEMM deferred to stage 2`
- mark [1]: `[SUPERSEDED 2026-10-17 in its memory-need clause by criterion 5 (owner waiver); the 4-bit written-override clause stands and has not been met]`
- mark [2]: `[DEFERRED 2026-10-17 to stage 2 (new card): not on the first quantised run's path]`
- mark [3]: `[DEFERRED 2026-10-17 to stage 2 (new card); gemma-metal's quantiser is not MLX-exact (ojas gemma-metal/src/quant.rs:346-448: scale (hi-lo)/15, bias lo, no edge/at_zero snapping; q8 signed int8 with f32 scale/zero) and is not reused]`
- append:
  1. `(Fable 2026-10-17) Scope for this track: the quantised arm's smoke and reduced run train the bf16 LoRA path over a fake-quant base file (dequantize(quantize(W)) for the 150 wrapped linears from mlx q8 g64 with bf16 scale and bias, written by the Lappi gate tool, pinned by sha256); ojas needs no quantised dtype, kernel or host quantiser for them. The memory arithmetic is recorded here: 2.745 GB bf16 wrapped linears vs ~1.46 GB q8 g64, a 1.29 GB saving of an estimated 12-14 GB step; no pre-registered shape needs it.`
  2. `(Fable 2026-10-17) Criterion 8's fixture: the tiny Qwen3.5 fixture's wrapped linears replaced by dequantize(quantize(W)) from mlx (q8 g64, bf16 scale and bias; mlx version and the quantiser call in the fixture manifest), torch fp32 one-step and 20-step references regenerated by the fixture generator, and the existing (a'), (b'), (c') LoRA tests run on it unchanged on CPU and Metal.`
  3. `(Fable 2026-10-17) The owner's waiver (criterion 5) is valid as the owner's decision but is recorded only on this board; it is written into Lappi in the owner's words before the sibling pre-registration (ft-da2c4ab1). q4 is measured in the gate and not built. Stage 2 (resident q8 storage, dequant-to-bf16-scratch kernel in canonical tessl, bit-identical loss to the fake-quant run) and stage 3 (fused dequant GEMM) start only on the owner's yes after the fake-quant reduced 2B run is scored.`

### ft-58e316a9bae421a3a2d71c26192aaea8 (export)
- set: status `backlog`, priority `1` (unchanged), severity `medium` (unchanged).
- mark [6]: `[DECIDED 2026-10-17: see the next criteria]`
- append:
  1. `(Fable 2026-10-17) The quantised arm's merge target is dequant(Q(W)) + (alpha/r) B A, the base the adapters trained over (the fake-quant file), rounded once to bf16 RNE; merging into the original bf16 W is refused with a typed error naming the base-sha mismatch. The release is a dense bf16 tower scored by qd-export and qd-metal unchanged; serving a quantised base is not in this track (D5's reversal is not triggered). The manifest records base format, group size, mlx version and the base file's sha256.`
  2. `(Fable 2026-10-17) The checkpoint-to-scorer bridge is Lappi's (ft-2ed2fd55): a converter reads ojas's adapter checkpoint (ft-b5e64557 criterion 13's documented layout) into the lora-merged layout tools/ckpt_average.py already accepts (frozen tower, lora_A/lora_B by module name, lora_r, lora_alpha, d17, span_head; ckpt_average.py:152-160) and closes GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED-2026-10-12 with a round-trip test; ojas's part here is a parity test that the merged tower from that route equals ojas's own per-site merge (lora.rs merge) within the stated bf16 bound.`

### ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 (frozen params and LoRA)
- set: status `review` (unchanged; owner verify to done), priority `1`.
- mark [3]: `[MET 2026-10-17: LoraKind::suffix (ojas-model/src/qwen35/params.rs:697-708) equals Lappi LORA_TARGET_SUFFIXES (python/qd_train/lora.py:89-100) ten for ten; 150 linears at 2B by construction; names read from the tensor index (lora_targets :771)]`
- mark [9]: `[HOMED 2026-10-17 at ft-2bbced18 criteria 3 and 4 (the end-to-end dW-skip share and the interleaved rows); criterion 2's benchmark clause is waived here]`

### ft-3cc9ed3f01f08ad71b71f403438c3aa0 (bf16 frozen base)
- set: status `review` (unchanged), priority `1`.
- mark [6]: `[HOMED 2026-10-17 at ft-2bbced18 criterion 7 and its 2026-10-17 criterion (the tape-lora arm); waived here]`
- append:
  1. `(Fable 2026-10-17) The tied-table lane is on main (merge 040e1ed, rows bench/results/2026-10-14-bf16-tied-table); the brief's "not on main" is stale. Review closes on criterion 8's documentation and line-reference clean-up.`

### ft-29462f23af79fa6d72825ecb7afd1944 (training-path correctness)
- set: status `review` (unchanged; ready for the owner's verify to done: Fable 2026-10-15 ruled "Board: done" after the recorded run of 2026-10-15), priority `0` -> `1` (nothing on the LoRA path waits on it; criterion 6's dtype-parse sweep is not on the path: Qwen weights are BF16/F32).

### ft-531216ed802203374c90becae0bea3f2 (2B Metal Tape gate)
- set: status `review` (unchanged), priority `0` -> `1`.
- mark [2]: `[PROVIDER LEG OFF THE PATH 2026-10-17: D1 keeps the provider full-FT; tessl's real_2b_step_matches_transformers reading gradients with less held is tessl's item and does not hold this card]`
- append:
  1. `(Fable 2026-10-17) Review closes on criterion 4's text and README amendment; criterion 3's tokens/s branch is decided by ft-2bbced18 criterion 7.`

### ft-9c13f47224dbb204fe9534c264500a5b (training semantics parity)
- set: status `ready`, priority `2` (unchanged).
- append:
  1. `(Fable 2026-10-17) Not on the LoRA-2B path: the Tape StepProvider runs under Lappi's loop (its own clip max_norm/(norm+1e-6), non-finite refusal before AdamW, grad_accum 1), not ojas-model's Trainer; criteria 5, 7 and 10 apply to the nanolab Trainer only.`

### ft-6a88d96ddc927ec872c63e0d33440e01 (step-cost follow-ups)
- no change.

### New card (ojas)
- title: `Stage 2, resident block-quantised frozen base: MLX-exact host quantiser and dequant oracle, U32+bf16 packed storage, dequant-to-bf16-scratch kernel in canonical tessl, loss bit-identical to the fake-quant run; fused dequant GEMM only if profiling says scratch traffic matters`
- status `backlog`, priority `2`, severity `low`, labels `fine-tune, quantization, metal, gated`, repositories `ojas`.
- criteria:
  1. `Starts only on the owner's yes after the fake-quant reduced 2B run (ft-da2c4ab1) is scored; the arithmetic it buys is ~1.29 GB at 2B (2.745 GB bf16 wrapped linears vs ~1.46 GB q8 g64), recorded with the ask.`
  2. `An MLX-exact affine quantiser and dequantiser in ojas-core (edge/side selection, at_zero snapping, eps 1e-7, w_max initialised to 0, round half away from zero, bf16 scale and bias via ojas_core::f32_to_bf16): quantised ints, scales, biases and dequantised values bit-exact against mx.quantize/mx.dequantize golden arrays from the Lappi gate tool for q8 and q4 on the six distinct 2B linear shapes (random, ties, all-negative group, constant group, zero group); NaN/Inf groups, group 0 and cols % group != 0 refused.`
  3. `Packed storage as a U32 tensor [out, in*bits/32] plus two Bf16 tensors [out, in/64] (no DType variant); load from an MLX-quantised safetensors; Budget-charged.`
  4. `A dequant_affine_{q8,q4}_bf16 elementwise kernel in canonical tessl (Lappi rule 6) and ojas-metal ops linear_forward_q / linear_backward_input_q (dequant to a reused bf16 scratch, then the existing GEMM); no weight gradient; CPU reference op for the oracle.`
  5. `Tape loss and adapter gradients over the resident-q8 base are bit-identical to the fake-quant bf16 base on the tiny fixture, CPU and Metal; the 2B rows (ft-2bbced18 criterion 8) report the saving and the dequant share of the step.`
  6. `A fused dequant GEMM (NT forward, NN dX, M up to 9,638) is built only if the stage-2 profile shows the scratch traffic above a stated share of the step; otherwise recorded as not needed.`
  7. `gemma-metal's quantiser delegates to the ojas-core owner or is left with its defects documented (bf16 ties-up rounding gemma-metal/src/quant.rs:572-576, silent NaN/Inf groups :452-461); one owner for host quantisation.`

### Lappi side (not the ojas board; for the Lappi lane, card ft-2ed2fd55e93dc1019b02deb7e1e2c754 unless stated)
- ft-2ed2fd55 crit 1: add the driver clause: `a driver under tools/ that runs the reference arm over the shards, supervision, span head and D17 with per-step loss logging, a checkpoint writer in the lora-merged layout and a ledger row; today nothing outside tests imports qd_train.lora`.
- ft-2ed2fd55 crit 5: add: `the gate tool also writes the fake-quant base files (2B-Base, 0.8B-Base) with sha256 and the mx.quantize/dequantize golden arrays; numbers into docs/fine-tune-plan.md; mlx version named; q4 measured, not built`.
- ft-2ed2fd55 new: `D17 as a host entry in qd-train's objective: 17 embed rows read by name from the checkpoint, R = E + D17, 17-way CE, dz = (softmax - onehot)/N, dh = dz R, dD17 = dz^T h, span-head gradient pre-summed where positions coincide, report-only full-vocab CE via scale-0 RowTargets; committed fixture against python/qd_train/lora.py AnswerRowDelta`.
- ft-2ed2fd55 new: `LoRA-aware parameter plan (adapters lr_scale 1.0, D17 and span head 0.5, wd 0.01 on all; no 0.1x split) and the "lora" ledger row writer with the seven digests of init.record_in_every_row and the quick reason; the LoRA-A digest recomputed by the attach_lora rule`.
- ft-2ed2fd55 new: `converter ojas adapter checkpoint -> lora-merged layout, round-trip test, closes GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED-2026-10-12`.
- ft-2ed2fd55 new: `amendment 1 (section 2) after the owner's yes; the quantised arm's owner-decision file and sibling pre-registration (section 3) before any quantised result`.
- ft-3163ce23 crit 8: mark stale: the --seed/--resume/--epochs/--min-lr fixes already landed (qd-train-metal/src/args.rs:105,214,245,484; run.rs:173-199; commit 6f88a00) [A audit B]; the pin bump and `ojas-train` switch remain.
- Lappi docs: CLAUDE.md "Environment facts" and tools/mac_heavy.sh header say nothing of memguard's 32 GiB per-process cap, which binds before mac_heavy's 52 GiB [A audit D]; add one line each. HANDOFF/ojas-train-move-2026-10-14.md "multi-epoch CLI not wired" and "pin PASS at bc498c4" are stale.
- gaps.jsonl: no record exists for the quantised-arm request, the missing gate tool, the missing torch driver, or the missing Rust LoRA provider [A audit B]; append GAP-LORA2B-QUANT-ARM-UNRECORDED-2026-10-17, GAP-LORA2B-TORCH-DRIVER-MISSING-2026-10-17, GAP-LORA2B-TAPE-PROVIDER-MISSING-2026-10-17 (rule 1/10).

---

## 5. Stale briefs

**Ruling: the board is canonical for status, priority, severity and criteria; the tracked `tasks/*.md` briefs are the long-form record and are synced to it, once, in one documentation-only commit on ojas main, after the section-4 edits land.** Do not re-import (`gitpulse_import_tasks` keys on `gp-` ids and would duplicate the `ft-` cards; memory note). The sync: (1) each brief's header `status`/`priority` set to the board's (today diverging: gp-ft-frozen-params-and-lora ready vs review; gp-ft-training-path-correctness review P0 vs the ruled P1; gp-qwen35-2b-metal-tape-run ready P0 vs review P1; gp-ft-step-memory-and-throughput backlog vs ready; gp-ft-quantized-frozen-base backlog P2 vs ready P1 with the new title); (2) a `### Fable ruling (2026-10-17)` section appended to each affected brief carrying the section-4 criteria verbatim with their item ids; (3) the stale claims audit A listed corrected in place with the original line struck and "corrected 2026-10-17 at 5b67cdd" (tied table "not on main"; "nothing implemented" in decision-head; `heads.rs` planned file superseded by the host-side ruling; "weights not on the machine"; the gp-uma CA-1/MD-1 items fixed on main; moved line cites). Going forward: a ruling recorded on a card is appended to its brief in the same change. The four untracked briefs from other lanes and the untracked `BUILD_REPORT*.md`/`AUDIT_REPORT.md` are not touched by this sync. A session rooted in ojas does it; no code in the commit.

---

## 6. Owner-only items (each a yes/no)

1. **Lappi ojas pin.** Bump `upstream.lock.json` 132b46e -> 5b67cdd now (99 commits; `make pin` fails today; Cargo.lock regenerated in the same commit because ojas-data now depends on ojas-io), with the delta report attached? **Yes/no.** (Alternative: wait and bump once to the ojas-train phase-A sha; Lappi gates stay red on pin meanwhile. My recommendation: now.)
2. **Amendment 1 to `campaign/lora-2b-preregistered.json`** (section 2: the step definition in words; N fixed from the measured LoRA tokens/s row under the approved window, cap 400, no floor), committed before any LoRA-2B result is read? **Yes/no.** If no: extend the approved Mac window so that 200 steps fit at the measured rate (7-10 h upper bound today)? **Yes/no.**
3. **Quantised arm, owner-decision record in Lappi** in your words (the memory-need waiver; q8 only; the arithmetic: 1.29 GB of a 12-14 GB step, no shape needs it; first runs over a fake-quant base file)? **Yes/no.**
4. **4-bit.** Do you override Unsloth's "no QLoRA on Qwen3.5" in writing and want q4 built if it clears the gate? **Yes/no.** (My recommendation: no; q4 is measured in the gate regardless.)
5. **Sibling pre-registration `campaign/lora-2b-q8-preregistered.json`** (same D6 recipe; quantised set = the 150 wrapped linears; tied table, norms, conv, A_log, dt_bias, in_proj_a/b unquantised; mlx version; base-file sha256; family id `lora-2b-q8`; merge target dequant(Q(W)); every row quick): approve its content once drafted? **Yes/no.**
6. **Extra Mac GPU time for the quantised arm** beyond the 2026-10-09 approval (the LoRA rows ~1 h are inside it; the quantised 0.8B smoke ~1 h and a second reduced 2B window of up to 6 h are not)? **Yes/no.**
7. **Stage 2** (resident q8 storage + dequant-to-scratch kernel in canonical tessl, 1-2 weeks, ~1.29 GB saving, no serving change) after the fake-quant reduced run is scored? **Yes/no.** (My recommendation: not now.)
8. **Verify to done** the review cards whose runs are recorded (ft-29462f23, ft-6fb212ce, ft-531216ed after its text amendment, ft-3cc9ed3f after its doc clean-up)? **Yes/no.** (Agents cannot; the done gate is yours.)
9. **peft** for the PEFT export round-trip test (D7, still unanswered): install, or keep the test "not run"? **Yes/no.**

What remains unverified in this ruling: every test and bench result quoted is a committed row or a card log, not re-run here (no GPU job was allowed); `make pin`'s failure and the Lappi door tests are audit B's runs; the LoRA step's speed-up over full-gradient is unmeasured (that is step 1 of the path); the bare-vocab GPT-2 refusal path and the Metal clip/AdamW behaviour under a mid-list non-finite are read, not run. DevMap was degraded on ojas and disagreeing on Lappi in every audit; nothing here is graph-confirmed.
