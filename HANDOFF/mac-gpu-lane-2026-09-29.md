# Mac GPU lane — handoff, 2026-09-29

Lane: everything the M5 Pro (64 GB, AC power) can run before renting one GH200 for the campaign
in `docs/train-plan-2026-09-28.md` ("The GH200 campaign"). Jobs ran one at a time on the GPU.

Labels: **[V]** measured here and in a ledger row, **[I]** inferred (arithmetic or reasoning),
**[U]** unverified. Every number below cites a row.

**Navigation.** The DevMap store is unavailable for this worktree (it indexes the qwen-decision
checkout; the gap was already recorded), so this lane used `rg` and Read; nothing here is
graph-confirmed. `ListAgents` was not available to this subagent; `ps` stood in for it
(GAP-MAC-GPU-LANE-HAS-NO-LISTAGENTS).

**Frozen code.** Other lanes were editing `python/qd_data` and `python/qd_train` throughout, and
every qd_data edit correctly invalidated the shard sets through the code fingerprint. On the
lead's instruction, every GPU job after that ran from a detached worktree at HEAD:
`scratchpad/gpu-lane/qd-head` (`c65d7da`). The rows carry a **clean** `code_commit c65d7da…`.
The rows were written outside that tree, because `ledger/*.jsonl` is not gitignored and would
make it dirty, and were then copied byte-for-byte (sha256 checked) into `ledger/`:

| file | rows |
| --- | --- |
| `ledger/mac-smoke-frozen-2026-09-29.jsonl` | 16 (jobs 1, 2) |
| `ledger/mac-teacher-2026-09-29.jsonl` | 2 (job 3) |
| `ledger/mac-mlx-2026-09-29.jsonl` | 7 (job 4) |
| `ledger/mac-torch-attrib-2026-09-29.jsonl` | 1 (job 5) |
| `ledger/mac-smoke-2026-09-29.jsonl` | 5, from the main tree **before** the freeze (dirty `code_commit`): shard builds e1133e80, 7381c57e, b2db1463, 71f0f4e8, and ft afcf38aa `killed` (terminated by me on moving to the frozen tree) |

All eight `mac-*-2026-09-29` chains verify (`Ledger.verify_chain`). Every row is `quick`.

## Job 1: trainer smoke on MPS [V]. The path runs end to end

`shard build → train → checkpoint → resume → epoch → score-val → rows`, on the real tower.

| step | row(s) | result |
| --- | --- | --- |
| shards, 2000 pairs, `--val-shards` | smoke 4c514fa5 | 3364 train / 184 val; remap `1b5e9101…`, the same as the GH200 set. 53.7 s |
| memorise arm, bf16, ≤300 wide, 2 passes, checkpoint every 20 | ft f0f0fa95, verdict 8e04efe1 | 86 steps in 959.8 s ≈ **11.2 s/step** (including four 8.03 GiB checkpoint writes of 7.7–18.9 s each, and load). Letter loss 1.614 → 1.224. Process peak footprint **26.3 GB** |
| **resume** from the step-80 checkpoint | ft 7a93bb47, verdict c477e677 | "resume: … at optimizer step 80". The remaining 6 steps give the **same `train.loss_log_digest`, the same final loss 1.224178 and identical verdict metrics** as the uninterrupted run, with the same protocol hash `a9c9d38b…`. **The resume is bit-exact on MPS.** Peak 34.1 GB |
| epoch arm on the 2000-pair set, bf16 | none (refused before a recorder existed) | `ValueError: this run is 1860 optimizer step(s) long and its moments would be torch.bfloat16, which does not survive it … exp_avg_sq stops moving at step 384`. Correct behaviour |
| the same with `--optimizer master` | memorise ft 4b3d565f + verdict cd8b448b; **no epoch row** | died at 2247 s: `time: command terminated abnormally`, peak footprint **80.8 GB** on 64 GB. The cause is most likely the kernel memory killer [I]. SIGKILL leaves no row (GAP-MAC-A-SIGKILLED-RUN-LEAVES-NO-ROW) |
| shards, 350 pairs (epoch = 377 steps, under 383) | smoke 0edf8de6 | 400 pairs → 417 steps (7b369923), 360 → 384 (efa8a036): both over the limit. 100 and 200 pairs fail val letter coverage ("val rows offer letter(s) ['E'] …"). Per the gap record GAP-PORTED-DECODE-KEYERROR-ON-A-LETTER-NO-ROW-USES-AS-GOLD, another lane has since fixed this in the main tree; I have not checked that [U] |
| **epoch arm + `--score-val`**, bf16 | ft **fd455c88**, eval **db3b971e** (memorise 83c56f6b / e98b2f18) | 352 of 377 steps in 1811.7 s ≈ **5.1 s/step**, `termination wall_clock_cap` (the tool's fixed 1800 s cap). Val (38 seqs): choice 10/17, score 11/17, span 0/4; `ece` not_run (<100 examples); `degenerate_head` ran and **failed** on choice.k2. Peak **45.8 GB** |

The exit codes are 1 from the tool's own claim checks: the memorise arm does not reach its floor in 2 passes. These are failed claims, not crashes.

**Regression gate:** `scratchpad/gpu-lane/SMOKE_CMD.sh` (7 steps, `CODE=` selects the checkout).
It is the measured runs above assembled into one script. Its step 2 drops `--epoch --score-val`
compared with the invocation that wrote f0f0fa95. That invocation's epoch arm was refused before
training, so the memorise rows are the same by construction. The script as a whole has **not**
been re-run in one piece [U].

## Job 2: A1 zero-shot baseline, rung 1 [V]

No entry point existed: `--score-val` only scores after the epoch arm. So `tools/mac_zero_shot.py`
reuses `real_ft_run.open_val_set`, `_decode`, `score_states` and `calibration_states` unchanged.
The decode is the scorer's own: the slot's rendered rows at `target_index`, which is 3 rows (A, B + Z)
for choice and 6 for score. That is `answer.rs`'s Letters query, not a "17-row" slice.

Eval **9004c495** (2000-pair val set, 184 seqs, base weights remapped to 28,239 rows):
**choice 43/90 (47.8%)** against the val majority 'A' at 46/90; **score 9/90 (10.0%)** against the
majority 'B' at 37/90. Span 1/4 comes from a *randomly initialised* span head, since the base model has none.
`ece` not_run (90 < 100); `degenerate_head` passed. Decode 111.4 s, 925 tok/s.
Re-run with the final (gate-fixed) tool: eval **3b970477**. It gives an **identical verdict digest `2c98c19ef585cc28`**
and identical val_top1. `wall_clock_source: caller`, and it carries `code_that_ran` + `decode_code_that_ran`.
Decode took 59.2 s this time (the 111.4 s run was under heavier swap [I]). The `--dump-items` re-run is byte-identical.

Only one val set exists that `open_val_set` accepts at the full size: the 2000-pair set. The
350-pair set's val (38 seqs) was scored only after training (db3b971e), not zero-shot.
The torch-bf16 drift in job 4 applies to these numbers.

## Job 3: teacher feasibility on the Mac [V, projections I]

`tools/mac_mlx_bench.py teacher`: 100 real val letter items (mean **555.8 tokens**), one prefill
each, logits of the option-letter rows at the last position only, batch 1, sequential.

| checkpoint | row | items/s | tok/s | MLX peak | 20k items [I] | 50k items [I] | top-1 vs gold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-4bit | c6d2170e | **0.674** | 374 | 16.0 GB | **8.2 h** | **20.6 h** | 55/100 |
| Qwen3.8-27B-nvfp4 | 8048bf34 | 0.649 | 361 | 15.9 GB | 8.6 h | 21.4 h | 55/100 |

- The projections assume the campaign's prompt lengths match these val items. They exclude batching and prefix reuse, which are plausible 2–4× levers [U].
- The 27B tokenizer re-encodes the 2B-decoded text to **identical ids on 100/100 items**, and every option letter is **one 27B token** [V]. So the 2B shard ids and letter slices carry over directly.
- `transformers 5.12.1` resolves the 27B config as `Qwen3_5Config`, `model_type qwen3_5`, 64 layers (16 full attention), with `Qwen3_5ForCausalLM` registered [V, config level]. Loading its bf16 weights under transformers was not tried; they are not cached [U].
- `incoai/Qwen3.8-27B-DFlash2` is a 5-layer `DFlash2DraftModel` with `dflash_config`/`num_target_layers`: a speculative draft, **not a teacher** [V].
- 4-bit soft labels come from a different model than bf16 soft labels. Their divergence from the bf16 teacher is unmeasured [U].

## Job 4: MLX against transformers on the 2B [V]

Parity on 50 val items, as the log-softmax over each item's decode rows, with an **fp32 head on both sides**. Rows are in `mac-mlx`:

| comparison | row | max \|Δ\| | median per-item max | argmax agree |
| --- | --- | --- | --- | --- |
| MLX fp32 vs torch fp32 | 8bcc3cec | **0.012** | 0.0027 | **50/50** |
| MLX bf16 vs torch fp32 | aeba026c | **0.078** | 0.029 | 49/50 |
| MLX bf16 vs torch bf16 | f3378d23 | 0.816 | 0.461 | 46/50 |
| torch bf16 vs torch fp32 (from the parity files of 59a44b97 / 5a2b96c6, computed offline) | — | 0.831 | 0.430 | 45/50 |

So **MLX matches transformers** (fp32 vs fp32), and **MLX bf16 is ~10× closer to the fp32 reference
than torch-MPS bf16**. Torch-MPS bf16 is the outlier (GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT; which op is responsible is not localized).
The outlier claim rests on rows, not only on the offline figure: f3378d23 (0.816) and aeba026c (0.078)
bound torch bf16 vs torch fp32 at ≥ 0.738 by the triangle inequality. Row 839b98ef (0.812, 43/50) and
c352cee1's parity file (`torch_parity.jsonl`) are superseded: both rounded the head to bf16. The `--dtype float32` MLX
load upcasts before `sanitize`, because mlx_lm's qwen3_5 `sanitize` stores `1 + w` for the norm weights in
the checkpoint dtype.

MLX prefill on the scoring path, 2B bf16, batch 1, real tokens (row **45595699**): **7,606 tok/s at 1k,
7,528 at 2k, 5,285 at 8k**. Peak 5.75 GB. The 8k input is concatenated prompts (synthetic length, real ids).

## Job 5: where torch's MPS time goes [V; shares I]

Row **c352cee1**, bf16, batch 1, real tokens, 10 reps after 3 warmup:

| mode | 1k s (tok/s) | 2k s (tok/s) |
| --- | --- | --- |
| `full_ce`: all-position full-vocab logits + CE | 0.439 (2,334) | 0.815 (2,513) |
| `last_only`: `labels=None`, last position, letter rows | 0.368 (2,785) | 0.657 (3,117) |
| `gdn_identity`: 18 GDN mixers → identity | 0.133 (7,692) | 0.254 (8,065) |

- **The GDN is ~64% of the forward at 1k and ~61% at 2k** [I, from the difference]. The full-vocab head + CE adds ~16–19% [I]. This confirms the AUDIT §3 ranking: fused GDN (K1) first.
- **torch bf16 matmul 4096×2048 @ 2048×6144: 30.6 TFLOP/s median (31.3 best)**. That is *above* tessl's 26.6, so torch/MPSGraph matmul is not the bottleneck [V]. This settles AUDIT §3.4's last [U].
- The torch forwards here are ~2.7× faster than AUDIT §1's 1.19 s/1k (which was taken on battery via another loader). The cause is not isolated [U].
- MLX's scoring prefill (7,606 tok/s) is 2.7× torch `last_only` at 1k and matches torch with the GDN removed [V; interpretation I].

## What this changes for the GH200 plan [I]

- A 27B 4-bit teacher on the Mac labels 20k items in ~8 h, or 50k in ~21 h, unattended, if 4-bit labels are acceptable. That could move teacher labelling off the rented box.
- The Mac cannot run a full-set epoch: bf16 is refused past 383 steps and master OOMs. Training stays on the GH200, as planned.
- Resume is bit-exact on MPS.
- Mac-side scoring should use MLX, or torch fp32, not torch-MPS bf16.

## Open

GAP-MAC-A-SIGKILLED-RUN-LEAVES-NO-ROW, GAP-MAC-EPOCH-ARM-CANNOT-RUN-ON-THE-2K-SET,
GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT, GAP-MAC-GPU-LANE-HAS-NO-LISTAGENTS (all appended to
`gaps.jsonl` via `qd_train.gaps.append_gap`). Also unmeasured: 4-bit vs bf16 teacher label divergence;
batched or prefix-shared teacher throughput; SDPA h256 memory growth (AUDIT §5 item 5); the Metal
profiler dispatch count (§5 item 1).

## New files (uncommitted, this lane's)

`tools/mac_zero_shot.py`, `tools/mac_bench_common.py`, `tools/mac_torch_attrib.py`,
`tools/mac_mlx_bench.py`. After the lead flagged gate failures, `make pytest` ran green: 1910 passed, 80 skipped.
After the later `--dtype`/parity edits, ruff on `tools/` is clean, and `test_tool_call_sites`,
`test_provenance_on_every_exit`, `test_lint_gate` and `test_ledger_provenance` pass (42 passed). The full suite
was not re-run after those edits. `python/tests/test_tool_call_sites.py` shows `M` from another lane's
`ft_linear_control.py` entry, not from this lane (my pin there was reverted). `QD_TOOLS`/`QD_REPO` point the
tools at a frozen checkout.

## Exact commands

    S=/private/tmp/claude-501/-Users-bharath-Code-research-Lappi-decision/752954f4-7f08-4414-bdac-8e49ae47f8b4/scratchpad/gpu-lane
    R=/Users/bharath/Code/research/Lappi-decision
    SNAP=/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
    H=/Users/bharath/.cache/huggingface/hub
    # jobs 1-2 (+ item dump):
    /bin/zsh $S/SMOKE_CMD.sh
    # job 3:
    QD_REPO=$S/qd-head /Users/bharath/.venvs/ml/bin/python $R/tools/mac_mlx_bench.py teacher --model $H/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff --items $S/val_letter_items.jsonl --header $S/frozen-shards-cpft-2000-val/shards/val/header.json --ledger $S/ledger-frozen/mac-teacher-2026-09-29.jsonl --base-tokenizer $SNAP/tokenizer.json --n 100
    # job 5 (+ torch parity refs):
    QD_REPO=$S/qd-head /Users/bharath/.venvs/ml/bin/python $R/tools/mac_torch_attrib.py --items $S/val_letter_items.jsonl --header $S/frozen-shards-cpft-2000-val/shards/val/header.json --snapshot $SNAP --ledger $S/ledger-frozen/mac-torch-attrib-2026-09-29.jsonl --parity-out $S/torch_parity.jsonl --lengths 1024 2048
    QD_REPO=$S/qd-head /Users/bharath/.venvs/ml/bin/python $R/tools/mac_torch_attrib.py ... --parity-only --dtype fp32 --parity-out $S/torch_parity_fp32.jsonl   # and --dtype bf16
    # job 4:
    QD_REPO=$S/qd-head /Users/bharath/.venvs/ml/bin/python $R/tools/mac_mlx_bench.py parity --dtype float32 --torch-parity $S/torch_parity_fp32.jsonl --model $SNAP --items $S/val_letter_items.jsonl --header $S/frozen-shards-cpft-2000-val/shards/val/header.json --ledger $S/ledger-frozen/mac-mlx-2026-09-29.jsonl --n 50
    QD_REPO=$S/qd-head /Users/bharath/.venvs/ml/bin/python $R/tools/mac_mlx_bench.py prefill --model $SNAP --items $S/val_letter_items.jsonl --header $S/frozen-shards-cpft-2000-val/shards/val/header.json --ledger $S/ledger-frozen/mac-mlx-2026-09-29.jsonl --lengths 1024 2048 8192

**Ephemeral:** `SMOKE_CMD.sh`, `val_letter_items.jsonl`, the parity files, the frozen shard sets, the
worktree and the ledger originals all live in the session scratchpad `$S`. The rows are safe in `ledger/`.
Whether `SMOKE_CMD.sh` gets a durable home is the lead's call.

**First command for the next lane:** build a fresh frozen worktree at the new HEAD (for example
`git worktree add --detach $S/qd-next HEAD`), then `CODE=$S/qd-next LEDGER=$S/ledger-next/mac-smoke.jsonl /bin/zsh $S/SMOKE_CMD.sh`.
Compare step 3's `train.loss_log_digest` to step 2's. The old worktree comes out with
`git worktree remove $S/qd-head`.
