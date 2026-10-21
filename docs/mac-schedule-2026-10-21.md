# Sequenced Mac schedule for P0, P0b and P2 (2026-10-21)

Card ft-5dfe48e7, criterion 3. Plan: `docs/lappi-v2-plan-2026-10-20.md` section 10 (phases) and
section 11 ("Mac schedule ... owed before P1 ends").

**Rule 5 applies to every number below.** A figure is either computed from a measured rate whose row
is named, or it is written "not measured" with the run that would measure it. Nothing is estimated
from memory. Hours are Mac GPU wall-clock hours on the M5 Pro (the rows' own device), one heavy job
at a time under `tools/mac_heavy.sh`.

## Measured rates this schedule uses

| Id | What it measures | Value | Row |
| --- | --- | --- | --- |
| R-LoRA-9638 | LoRA-2B step (r16/alpha16, 150 sites, bf16 base, checkpointing on, Qwen3.5-2B-Base b1485b2f), T = 9,638 | 261.607027 tok/s | `ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L4` (3 timed steps) |
| R-LoRA-2048 | same, T = 2,048 | 335.915287 tok/s | `ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L3` (3 timed steps) |
| R-score | `real_ft_run.py --score-val`, Qwen3.5-2B-Base (1,881,825,088 params), fp32, torch MPS, 4,385 val sequences | 1,362.8 s | ledger `20cf3845-6325-4dcd-be20-2047d9fa8905` (`ledger/mac-score-ckpt-2026-09-30.jsonl`) |
| R-score-gates | same scoring with all gates | 3,330.3 s | ledger `784868b3-8e17-424a-a93a-8d3a809fa8f3` (`ledger/mac-score-ckpt-allgates-2026-09-30.jsonl`) |
| R-MLX-prefill | MLX prefill, Qwen3.5-2B-Base, batch 1, L = 8,192 (L = 2,048) | 5,284.5 tok/s (7,528.4) | ledger `45595699-1d85-40f5-a87f-d2686104e9b6` (`ledger/mac-mlx-2026-09-29.jsonl`) |
| R-metal-8185 | `qd-metal` product path, 8,185-token prefix, min of 5 | 2,347.3 ms per decision | ledger `02156dc5-6d6d-4f5f-9d9c-d93f5eb1dfb1` (`ledger/mac-qd-metal-eb-2026-10-03.jsonl`) |

All six are `quick` rows or short benches (rule 8): they are throughput measurements, excluded from
every decision, and they are the only rates that exist. R-LoRA-9638 is the slowest measured LoRA rate
and is the one `campaign/lora-2b-preregistered.json` binds for its Mac step count
(`mac_steps_rule.tokens_per_s_row`); R-LoRA-2048 is the faster one. They bracket the run's bucket
mix, which no row measures.

Arithmetic (reproducible: tokens divided by rate): one v1 Mac step is 35,403 tokens
(`mac_step_definition`); 127 steps is 4,496,181 tokens.

## Order

The Mac has one heavy-job lock, so "parallel" in the plan's P0b means interleaved at job
boundaries, not simultaneous. The order is fixed by what each phase needs, not by calendar:

1. **P0, no GPU hours.** The four v1 open items in plan section 10 that are code or verification:
   LoRA checkpoint converter (not started), D3 quantisation gate tool (absent), ojas pin 4123dfb
   verification (owner-run), and the `test_containment_exclusions.py` rerun under `mac_heavy.sh`
   with `QD_PREP_BIN`. Hours: **not measured** (no row times any of them). The converter blocks
   scoring of any LoRA run (the run's release path is merge, ckpt_average, qd-export, qd-metal).
2. **P0, LoRA-2B Mac run(s).** Registered shape: n = 127 steps, quick, one seed per Mac row,
   cannot promote (`campaign/lora-2b-preregistered.json`, `steps.mac_steps_rule`, `controls`).
3. **P0b, rate probe first.** Then the frozen bake-off, once at least 30 held-out owner labels exist
   (plan section 10 entry condition).
4. **P1 ends** (CPU: generators, gates, pool, new pre-registration). The v2 pre-registration states
   tokens per seed; P2 hours are computed from it.
5. **P2, Vani-adapter seeds.** Qwen3.5-2B first (ruling 10).

No shadow window exists before P3, so the guard in `docs/mac-serving-windows-2026-10-21.md` does not
delay steps 2 to 5; it only has to hold when retraining follows P3.

## P0: v1 LoRA-2B on the Mac

| Quantity | At R-LoRA-9638 | At R-LoRA-2048 |
| --- | --- | --- |
| One run, 127 steps | 17,186.8 s = **4.774 h** | 13,384.9 s = **3.718 h** |
| Three seeds, back to back | 14.32 h | 11.15 h |
| Scoring one run (R-score, R-score-gates) | 1,362.8 s to 3,330.3 s (0.38 to 0.93 h) | same (scored in fp32 on a different checkpoint of the same architecture; the LoRA release path is merged bf16, so this is an order of magnitude, not a projection) |

The first row reproduces the pre-registration's fixed value (127 steps, 17,186 s against a 17,280 s
budget). Three seeds is a choice in this table, not a pre-registered Mac count: the prereg says a Mac
row is one seed and quick. Rule 8 and `controls.shuffled_label` mean none of these rows can promote;
the promotion-grade recipe is the full schedule (12,176 steps, 431,066,928 tokens):
**356.5 h per seed at R-LoRA-2048 to 457.7 h at R-LoRA-9638** (matches plan ruling 13's 357-458),
which is cloud work (see `docs/cloud-cost-estimate-2026-10-21.md`), not Mac work.

## P0b: bake-off (frozen checkpoints, option logits through transformers on MPS)

| Arm | Measured rate that applies | Projection |
| --- | --- | --- |
| 1. Qwen3.5-2B-Base | R-score (torch MPS, fp32, same model, but 4,385 val sequences, not postings); R-MLX-prefill and R-metal-8185 are other runtimes | **not projected**: no row scores the bake-off's rows (about 2.3k postings at up to about 8K tokens, plan section 6) through transformers on MPS |
| 2. Gemma 4 E2B Base | none | **not measured** |
| 3. Gemma 4 E4B Base | none | **not measured** |
| 4. Base vs IT, one family | none | **not measured** |

The run that would measure it: **P0b-rate**, one quick throughput row per arm on the bake-off's real
rows, prefill at 2K, 4K and 8K, 5 repetitions after 2 warmups (the shape of
`tools/mac_mlx_bench.py`'s rows), recording peak phys_footprint. It is a few minutes per arm, and it
sizes the bake-off before the pre-registration commits the row count. Two cautions that come from the
plan, not from a measurement: E4B LoRA at 8K is not runnable on the Mac today (plan section 4), and
`ledger/mac-torch-attrib-2026-09-29.jsonl` records that without `fla` transformers falls back to a
reference gated-delta-rule that is "much slower", so a torch-on-MPS Qwen rate may not represent the
Gemma arms either way.

## P2: Vani-adapter seeds on the Mac

Tokens per seed come from the v2 pre-registration, which does not exist yet (P1 writes it). The
conversion is fixed by the measured rates:

| Tokens per seed | Hours at R-LoRA-9638 | Hours at R-LoRA-2048 |
| --- | --- | --- |
| 1,000,000 | 1.062 h | 0.827 h |
| T (from the prereg) | T / 261.607027 / 3600 | T / 335.915287 / 3600 |

The number of runs is also fixed by the registered rules, not by this schedule: ruling 12 says a
real-row arm promotes only as a full-recipe, at least three-seed Mac run, and
`controls.shuffled_label` asks for three seeds carrying a shuffled-label control on a
promotion-grade campaign, so a promotable candidate is **at least 6 runs** (3 seeds plus 3 controls)
**plus scoring** (R-score-gates, 0.93 h each as an order of magnitude). Total P2 hours =
6 x (T / rate / 3600) + 6 x scoring. With T unknown this is **not projected**; with T = the v5 recipe
(431,066,928, an upper bound the plan itself says v2 will undercut) one seed alone is 356.5 to 457.7 h,
about 15 to 19 days of continuous Mac time, which is why the plan sets v2 as a one-epoch adapter over
a smaller pool. A projection waits for T.

Gemma E2B LoRA on the Mac: **not measured** (no kernel and no ojas step provider; plan section 4
adopts Gemma only after the bake-off wins). The rate that would measure it: the same `gpu_memory_rows`
tape-lora bench on E2B, T = 2,048 and 9,638, 3 timed steps.

## What this schedule does not cover

Criteria 5 and 6 (drift monitor, daemon priority classes) belong to a later lane. Cloud seeds are in
`docs/cloud-cost-estimate-2026-10-21.md`. Calendar dates are not given: they depend on when the owner
labels exist and on every other lane's use of the one heavy-job lock.
