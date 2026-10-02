# Fable on the idle-GH200 experiment queue (2026-10-02)

The user's instruction (2026-10-02 ~01:00 UTC) was: "Train experiment lappi model when the gpu is idle." Advisor: claude-fable-5-1, through a read-only lane (task a74a51bb9ba068487). Recorded verbatim. The lead follows it under the user's standing instruction to follow Fable's recommendations. Nothing in it changes a gate.

I ran no GPU work and wrote no files. Fable's ruling is below. Evidence came from Read, rg and read-only ssh only. I made no DevMap or GitPulse call, so nothing here is graph-confirmed; one of the gaps at the end records this.

## Three premises in the brief are wrong

1. **The ≈94 h case is rule (ii), F seeds 3–4, not item 6.**
   - The source is `AUDIT/ojas-training-2026-10-01/fable-cuda-asks.md:32`. J6(f) runs in both branches; only its position moves (`box_q_j6f.sh` header).
   - The 73 h figure adds up caps. Measured, F seed 0 took about 5.8 h of wall clock: 17:55 → 23:43 in `/home/ubuntu/logs/q-f.log`. Its rows are ft `973cd4e3` (16,024 s), eval `f4feac15` (578 s) and needle control `a4f244f0` (187 s).
   - So the idle window starts earlier than Oct 5–6 (inferred).
   - Seed 1 was at step 2,616/9,683, 1.65 s/step, at 01:00 UTC (`train-s1.log`).
2. **The rung (d) torch arms do not need the Mac export to run.**
   - Only pass criteria 4–6 read the Mac artifact. The user already approved these arms for this slot (`HANDOFF/ojas-training-2026-10-01.md:29`, ask 5).
   - What blocks them is missing code. `--max-steps` and `--train-dtype` exist at neither `a502670` nor HEAD.
   - `--train-attention-mask` exists only at HEAD, so the arms must run from a lane based on main. That is safe: `git diff a502670 HEAD -- python/qd_data` is empty.
3. **F seed 0 fails a pre-registered bar that no falsifier covers.**
   - In `f4feac15`, `ood_abstain.unseen-language` is 20/60. The bar is "≥ 30/60 per category per seed" (`campaign/f-v4-preregistered.json:19`), but the falsifier at `:18` only names prose and scrambled.
   - Report it, but queue no training fix: G6 routes unseen-language to the `qd-lang` admission check, which is the human's call.

Also: F's controls log is `/home/ubuntu/logs/q-fcontrols.log`, not `q-f.log`.

**F seed 0 so far (`f4feac15`):**

| Metric | Value |
| --- | --- |
| choice | 9130/10985 |
| span | 6529/7238 |
| 8K needle worst bucket | 40/61 (80–100% depth) |
| 1K/2K/4K needle controls | 300/300 each (`a4f244f0`) |
| OOD prose / scrambled | 54/60, 58/60 |
| defect_class in-distribution abstention | 13/2304 |
| defect_class permutation | 2291/2304 |
| defect_class paired margin vs linear | +0.379 (`c89b89a1`) |

## Q1: the ranked queue

Nothing in this window can promote: all six entries in `docs/promotion-decisions.json` are still `"status": "open"`.

**1. `fsucc` (F′): a three-seed successor on a winning v4 ablation, conditional.**
- **Purpose:** decide whether F's recipe stands, or J6(f) / J6(d)-v4 replaces it. This is the only run here that yields a non-quick, rule-8-compliant artifact.
- **Trigger, pre-registered:** the J6(b) win definition (`f-v4-preregistered.json:13`), with F's three eval rows in place of J4's.
  - **J6(f):** the target is the 8K needle worst bucket alone, because 1K/2K/4K sit at 300/300 with no room to improve. Must not lose: the length controls and prose/scrambled OOD.
  - **J6(d)-v4:** the targets are the defect_class paired margin and `val_top1.choice`/`span`. Must not lose: OOD, needle, permutation and in-distribution abstention.
  - **Both fire on different metrics:** refused, and the human decides. Two one-seed wins are not combined into an untested recipe.
  - **A margin row is missing or did not run:** the check prints `refused`, never `quiet`.
- **Recipe:** `box_q_f.sh`'s argv with one change, seeds 0 1 2, from `qd-lane8` at `a502670`, with its own checkpoint directory and ledger.
  - F′(f) drops `--lower-layers-n 8 --lower-layers-lr-scale 0.1`.
  - F′(d) uses `--lr 3e-5 --beta2 0.95`.
  - Each seed also runs `--needle-control 1024,2048,4096` and `box_f_controls.sh`'s two CPU blocks, as F did.
- **Not quick.** Caps are 3 × 32,400 s plus 3 × 5,400 s.
- **Expected cost:** about 17–18 h. On the wall-clock basis the box bills, that is about $40. On the ledger-row basis it is about $32 (seed 0's ft row says $10.19).
- **Then `j5pp` (J5″):** `--shuffled-label` × 3 on F′, in `box_q_j5p.sh`'s form. Without it F′ is not a candidate. About 14.5 h, about $33 (inferred).
- **Code needed first:** a `qd-post-f-rules successor` subcommand, and a CPU control waiter for the two ablation arms (lane tasks below).

**2. `rungd`: rung (d) torch reference arms, T-bf16 then T-fp32.**
- **Purpose:** the torch reference for the Rust trainer's parity test. It also closes `GAP-OJAS-RUNGD-FP32-ARM-MEMORY-UNMEASURED-2026-10-01` and the fp32 AdamW eps/weight-decay read gap before the Mac window.
- **Recipe:** F's flags plus `--max-steps 200 --train-attention-mask none --deterministic`, seed 0, from a main-based lane.
  - T-fp32 adds `--train-dtype fp32` on the non-master AdamW path, and must assert eps 1e-8 and weight decay 0.01.
  - Unverified: commit `1cf8e6d` says no-mask runs "on SDPA flash only", and fp32 may refuse it. The lane has to settle that.
- **Quick.** About 1.5–2 h, about $4–5.
- **Reading:** pass criteria 0–3 of `fable-rung-d-resize.md` as written; criteria 4–6 wait for the Mac artifact.
- **Code needed first:** `--max-steps`, `--train-dtype fp32`, and the span-head init digest recorded. All additive, and entering the recipe only when switched on.

**3. `j6a`: J6(a) +replay on v4, seed 0, quick.**
- **Purpose:** the last phase-5 ablation agents can decide. Closes `GAP-PHASE4-REPLAY-SLICE-OVERLAPS-VAL` (`gaps.jsonl:596`).
- **What it needs:** a v4 rebuild with `--replay-shards` and a CLEAN attestation.
  - This is also a data change: replay-only rows leave gold training (`real_ft_run.py:3258-3263`).
  - Pin `--replay-weight` from the code, not from memory.
- **Cost:** cap 32,400 s; about 6 h, about $14.
- **Reading:** the same win definition, applied to the general-family metrics replay is meant to move (MMLU/CSQA permutation and abstention, MMLU's last-option bias), with no defect_class metric leaving F's envelope the wrong way.
- A win feeds the next re-plan; it does not start a successor run automatically.

**4. `cudadev`: the 11 ojas CUDA device tests marked NOT RUN by L-cuda-M0 (commit `7a360ee`).**
- Minutes, under $1.
- The box has only `ojas-qwen35-cuda-rung0` in `/home/ubuntu/bin`, so this is skipped unless cross-built, sha-pinned test binaries are shipped.

**Not queued:**
- J6(e): G1, the human's call.
- Soft-label or distillation training: no soft-label loss exists (`gaps.jsonl:573`) and there are no teacher labels.
- A 16K context arm: H4, the human's call, and it needs new data.
- More F seeds beyond what rule (ii) adds: rules 2 and 8.
- Any data change for unseen-language: G6, the human's call.

## Q2: waiter order and budget

`rung0.done` → `cudadev` → `rungd` → `fsucc` → `j5pp` → `j6a` → **STOP**. No waiter goes after `j6a`.

- `fsucc` also waits on `j6ctl.done` and on `=== F controls all done` in `q-fcontrols.log`.
- `j5pp` runs only if `fsucc` wrote three completed ft rows.
- `j6a` runs only if `fsucc` was quiet and a CLEAN attestation is on the box. Otherwise it logs that it was deferred to the re-plan.
- `j6ctl` is CPU-only. It waits on `j6f.done` and `j6dv4.done` and runs alongside the post-F items.

**Budget for the first 48 h (wall-clock basis, $2.29/h, inferred from the 5.8 h/seed rate):**

| Branch | Hours | Cost |
| --- | --- | --- |
| `fsucc` fires | ≈35 h (0.1 + 2 + 18 + 14.5) | ≈$80 |
| `fsucc` quiet | ≈8 h | ≈$19 |

If `fsucc` fires, the caps alone total about 61 h, but the actual run time fits inside 48 h. If it is quiet, re-plan at once using J7′'s avg / ens3 / avg-np rows. If only ens3 keeps OOD, the next GPU item is distillation, so start the soft-label-loss lane now on CPU.

## Q3: the Mac GPU

**No** Lappi training there beyond rung (d).
- At 452 tok/s a seed takes about 189 h (`fable-rung-d-resize.md:47`).
- The approval covers only rung (d) on F's recipe, and "full-schedule or 3-seed Mac runs need a separate yes" (`HANDOFF/ojas-training-2026-10-01.md:26`).
- The single-group arm is not F's recipe. It falls outside that approval and would use up the one agreed window without earning rung (d)'s claim.
- One small item fits existing scope: the kernel-diverse second arm for rung (b) (`GAP-L-ORACLE-RUNG-B-FLOOR-SHARES-KERNELS-2026-10-01`), on the tiny tower, in minutes.

## Q4: what needs the human's yes

- **Nothing new for the GH200 queue.**
  - F′ and J5″ are single-GPU runs, each capped at $20.61, under the human's existing words (`post_f_common.sh:69-70`).
  - Put one line in front of the human as a confirmation, not a question: about $80–95 in total if `fsucc` fires.
- **Still the human's, unchanged:**
  - the six promotion decisions — F's rows now exist to decide them against;
  - G1 (J6(e));
  - H4 (16K context);
  - G6 (unseen-language through admission).
- **No gate changes.** The successor rule picks a recipe; it is not a gate.

## Lane tasks, in order

1. Commit the F′ pre-registration (trigger, target metrics per arm, recipe, caps) to `campaign/` **before item 6's row exists**. Item 6 can run right after J5′.
   - Before committing, check that every key the rule reads is on each seed's rows: `needle_hunk_recall.depth.*`, `ood_abstain.<category>`, `permutation_consistency.family.code.defect_class`, `ood_abstain.in_distribution.family.code.defect_class`, `ece.family.code.defect_class.choice.k4`.
   - The lead also pins whether the envelope is F's three seeds or five if rule (ii) fires (my reading; Fable did not address it).
2. `tools/real_ft_run.py`: `--max-steps`, `--train-dtype fp32` and the head-init digest, each with a test that fails first.
3. `qd-post-f-rules successor`: tests on synthetic boundaries, plus a real-row test that J6(b) against J4 prints `fires`.
4. `box_q_j6ctl.sh`: CPU-only, niced, from `qd-lane10` at `3460afc` with `qd-prep-m1`, in `box_f_controls.sh`'s form, run on `/home/ubuntu/{j6f,j6d}-v4/verdicts.jsonl`.
5. The v4 replay set with a CLEAN attestation, copied to the box and checked with a prelude run.
6. Cross-built, sha-pinned CUDA device-test binaries.
7. The five waiters, dry-run in a container with slow fakes, as `HANDOFF/post-f-queue-2026-10-01.md` did.

## GAP ids to append

- `GAP-IDLE-QUEUE-SUCCESSOR-RULE-NOT-IN-CHECKER-2026-10-02`
- `GAP-F-SEED0-UNSEEN-LANGUAGE-BELOW-F2-BAR-2026-10-02`
- `GAP-RUNGD-TORCH-ARMS-NEED-MAX-STEPS-AND-TRAIN-DTYPE-2026-10-02` (include the flash-only no-mask caveat for fp32)
- `GAP-POST-F-QUEUE-DURATION-IS-CAPS-NOT-MEASURED-2026-10-02`
- `GAP-J6-V4-ARMS-HAVE-NO-LINEAR-CONTROL-ROW-2026-10-02`
- `GAP-IDLE-ADVICE-NAVIGATION-2026-10-02` (no DevMap or GitPulse call; rg, Read and read-only ssh only)

Main files:
- `/Users/bharath/Code/research/Lappi-decision/campaign/f-v4-preregistered.json`
- `/Users/bharath/Code/research/Lappi-decision/campaign/f-j7prime-preregistered.json`
- `/Users/bharath/Code/research/Lappi-decision/campaign/post-f-queue/post_f_common.sh`
- `/Users/bharath/Code/research/Lappi-decision/AUDIT/ojas-training-2026-10-01/fable-rung-d-resize.md`
- on the box: `/home/ubuntu/box_q_f.sh` and `/home/ubuntu/box_f_controls.sh`
