# The human's answer on optimizing before the v5 launch (2026-10-03, ~21:05Z)

## What the human asked

At ~21:00Z, with the 2× H100 box up and the v5 build running, the human wrote:
"Optimize training pipeline before large training".

## What the lead found, and Fable's ruling

- **Where the time goes [V, ledger/gh200-p4-v4-2026-10-01.jsonl].** In F's v4 run on the GH200,
  the five `epoch` ft rows took 22.21 h of wall clock (median 266.4 min). `epoch-score-val`
  took 0.82 h and the needle control 0.26 h. Training is about 94% of the measured row wall clock.
  The pre-registered v5 estimate (launch.projected_cost_usd.check) is 7.0 h per seed:
  - 5.17 h of training;
  - 1.12 h for the score phase (about 16%);
  - 0.12 h for the needle control;
  - 0.1 h of prelude;
  - 0.4 h of trajectory scoring.
- **The step was already profiled and ruled on** (AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md):
  - The step is GPU-bound at 100% utilisation.
  - `torch.compile` gave nothing.
  - The feeder is not the bottleneck.
  - The remaining levers both change what is computed (Tier-B):
    - training without the padding mask (+20–40% per seed [I]). Its P2 equivalence screen read
      inconclusive on both shapes, so it was ruled off (C2a);
    - fused AdamW (~3%), turned off in the human's ~18:42Z answer (C2b).
- **Fable's ruling (~21:04Z):** re-opening either lever is the human's decision, so put it to the
  human as one question with the two options below. An optimization pass that silently flips a
  recorded decision is a recipe change under another name. Under the first option, the work is:
  - to measure seed 0's phases live, the score phase first: its GPU/Python split is unmeasured,
    and the code (`.double().tolist()` per row, log_softmax on CPU in float64) suggests it is
    Python-bound;
  - to make that measurement the first Rust-port target for v6;
  - to check whether the greedy two-lane schedule strands one GPU at the tail.

## The question, as asked (AskUserQuestion, ~21:05Z)

> Optimizing the training step was profiled and ruled on 2026-10-02. The GPU is already 100%
> busy, torch.compile gave nothing, and the data feeder isn't the bottleneck. The two remaining
> levers are both off for v5: training without the padding mask (+20–40% per seed; its
> equivalence screen came back inconclusive, so Fable ruled it off) and fused AdamW (~3%; turned
> off in your 18:42Z answer). The plan is 11 runs, about 74 GPU-hours, about 40 h of wall time,
> roughly $335 of box time. Which do you want?

The options:
1. **"Launch as pre-registered (Recommended)"**: launch as soon as the data is built (~23:30Z).
   While seed 0 runs, the lead measures each phase, especially the scoring phase (~16% of each
   seed, likely Python-bound), as the first Rust-port target for v6. The run's math is unchanged.
2. **"Re-open no-mask first"**: run a new equivalence screen on the idle box (~1 h, a few
   dollars). If it passes, amend the pre-registration to train without the padding mask (20–40%
   cheaper per seed); if it is inconclusive again, launch as pre-registered. This delays launch by
   about 1–2 h.

## The answer, verbatim

**"Launch as pre-registered (Recommended)"**

C2a and C2b stay off. The recipe and the pre-registration are unchanged by this answer. The
optimization work is measurement on seed 0 (read-only ssh; `nvidia-smi dmon` over the score
window), feeding the v6 plan.

## The lane-tail estimate [I]

The pre-registered per-run hours are 7.0 h for a v5 seed or an arm seed and 6.0 h for a J5′ seed.
Under the greedy two-lane rule:
- seeds 0–1, then seeds 2–3, then seed 4 beside arm seed 0 (the arm's room needs seeds 0–2's eval
  rows);
- then arm seeds 1–2;
- then J5′ seeds 0–1;
- then J5′ seed 2 alone, on one lane.

That is about 40 h of wall time, with one GPU idle for about 6 h at the end (~$25). If the arm
reads no_room, J5′ takes its slots and the plan is shorter. This is a scheduling note for v6. No
queue file changes.
