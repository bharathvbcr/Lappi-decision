# decider-2b vs Lappi: what it does differently, and whether the +0.110 is clean (2026-10-12)

ft-fc7b705092c. Research only: no training change, no run, no number of ours edited.
Labels: **[V]** read at the cited URL or file this session; **[R]** a repo document's claim I did
not re-run; **[I]** inferred; **[U]** unverified. The web pages were read through a fetch tool that
summarises the page, so quotes are the summariser's rendering of the page, not byte-exact; the
URLs are the evidence.

## The comparison being explained

decider-2b v11 (Mapika/decider-2b, rev 533964da, Apache-2.0, fine-tuned from Qwen3.5-2B-Base, the
same base and size as Lappi) leads Lappi v0.1 by **+0.110 [+0.071, +0.157]** paired on JevArena
`test-g6` (849 pairs, 168 prompt groups; JudgeBench +0.041, RewardBench 2 +0.074, RM-Bench
+0.200), a683c27 [R, `HANDOFF/merge-and-bench-2026-10-06.md` lines 265-266]. MPS float16 matches
the CPU float32 run (paired +0.001 [+0.000, +0.002]), so the gap is the same on either device
[R, `HANDOFF/decider-mps-2026-10-10.md`].

## What decider-2b does (sources)

Source: model card https://huggingface.co/Mapika/decider-2b (v11, dated 2026-09-24), the repo
https://github.com/Mapika/decider (recipe, `docs/RL.md`, `decider/data/mixture.py`).

**Training mix [V, card].**
- Supervised stages v1-v8: about 95 public decision datasets (intent, ticket routing, sentiment,
  NLI, reading comprehension, multiple-choice QA, pairwise response preference, tool selection),
  agent trajectories (AgentGym), web element choice (Mind2Web), game states.
- v10-v11 LoRA stage, 42,749 rows: 8,000 from ten generated decision families whose answers are
  computed by the generating code; 11,356 questions over business documents written by a larger
  model and kept when two independent answers agreed; 3,293 human-labelled rows (MMLU, ARC, BoolQ,
  MNLI, others); 20,100 replay rows from the public mixture.

**Objective [V, card].**
- Supervised: cross-entropy ("a proper scoring rule").
- RL stage (v8 to v10): 384 steps, PPO clipped surrogate plus a log-score term on stated beliefs,
  a KL limit to the v8 weights; rewards from live MiniWoB++ browser tasks and exact game
  probabilities, no gold labels.
- LoRA stage: cross-entropy on labelled rows plus KL(p_v10 || p_model) on replay rows.

**Calibration method [V, card].** Temperature scaling fitted by NLL (`decider.calibrate`); v11 has a
global temperature 1.145 and a per-answer-type map (choice 1.164, yes/no 1.624, score 1.124); v10
used one temperature, 1.30. The card says v11 "did not pass its pre-registered release rule" and was
released on the full comparison against v10.

## What of it is testable in v6 (and what is not)

Lappi already has: cross-entropy supervision, a per-entry temperature plus conformal calibration
table (`qd-calib-fit`), and code-computed labels by construction (`qd-mutate`).

| decider-2b element | Testable in v6? | How, and the limit |
|---|---|---|
| Generated families with code-computed answers | Already Lappi's design | Arm B / caller-family builders (ft-2d0a5e4b27e, ft-b1dfb6da273) |
| Document questions kept when two independent answers agree | Yes, as a data source | Arm B's mining; needs a teacher call budget, which the human has restricted ("no paid calls") [R, merge-and-bench] |
| Replay rows with a KL term to the previous checkpoint | Yes | Lappi has replay (`python/qd_train/replay.py`) but no KL-to-prior arm; a pre-registered recipe arm, report-only until the human decides |
| Per-answer-type temperature map | Partly present | Lappi fits per question-type entries (choice:3 .. choice:17); the 16-option miss (`HANDOFF/needle-end-of-context-2026-10-12.md` section 3) is where it fails |
| RL with outcome rewards (browser tasks, game probabilities) | No, not in v6's budget | Needs a pre-registered campaign and the human's decision; no Mac path |

The one structural difference worth reading before v6's data plan: its LoRA/replay stage with a KL
to the prior is a way to add data without drifting what already calibrated. That is a hypothesis
for a pre-registered arm [I], not a finding that it explains the +0.110.

## Benchmark overlap with its training: checked as far as it can be

What the sources say [V]:
- The card lists "QuALITY, XStoryCloze, RewardBench, Arena preferences, Hermes tool selection, and
  an abstention probe" as never seen in training. "RewardBench" has **no version**; our test-g6
  uses **RewardBench 2**, a different benchmark.
- Decontamination statement: "Every training row was checked against every evaluation file used for
  selection" and "No JevBench item and no Decision Index item was used for training". It names
  neither JevArena `test-g6`, RewardBench 2, RM-Bench nor JudgeBench, and does not say the check
  covered them.
- RM-Bench and JudgeBench are not mentioned on the card [V, search of the card text].
- `decider/data/mixture.py` puts two preference-like sources in the training split:
  `helpsteer2` and `helpsteer3_pref` (SCALE_TASKS) [V, file read through the fetch tool]. They
  are the same kind of data as the judge benchmarks' pairs. Whether any HelpSteer3 prompt or
  response appears in RewardBench 2, RM-Bench or JudgeBench is **[U]**: I did not compare rows.
- The v11 LoRA data names "human-labelled rows (MMLU, ARC, BoolQ, MNLI, and others)"; "others" is
  not enumerated.

**Verdict: overlap is neither found nor excluded. It is unverifiable from here, for these
reasons:** the card names no row-level decontamination against these three benchmarks, its
"RewardBench" is unversioned, and the ~95-dataset training registry and the v11 rows are not
downloaded here (only the model weights were), so a prompt-level comparison (the same normalised
8-gram method `qd-prep` uses for its own decontamination) cannot be run without fetching the data
registry. That comparison is the next step if the number is to be used publicly. The +0.110 is
therefore **uncaveated for lack of evidence of overlap, and unconfirmed as clean**.

Because no overlap was found, no caveat was added to any file this lane owns. Places that quote
the +0.110 and should carry "overlap with decider-2b's training is unverified (card names an
unversioned RewardBench; no statement covering RewardBench 2, RM-Bench or JudgeBench)" if it is used
publicly: `HANDOFF/merge-and-bench-2026-10-06.md` (lines 265-273 already say [unverified]),
`docs/outreach/README.md` and `compare_all.sh` in the bench directory, and the JevArena page. None
is owned by this lane.

## Upstream issue Mapika/decider#21: status

https://github.com/Mapika/decider/issues/21 (MPS: `scaled_softmax` returns NaN, a
`non_blocking=True` host-to-MPS copy of the temperature tensor, decider-ai >= 1.4.0) is **closed
(completed)**, closed by the maintainer on 2026-10-07 via commit 2d3ef87, fixed in decider-ai
1.8.2 [V, issue page]. Our comments (2026-10-11) confirmed the fix on transformers 5.12.1 and
5.19.0 and reported a full test-g6 MPS float16 run: 0.720 vs 0.719 CPU, 1,698/1,698 calls ok. Open
on our side only: two observations filed as notes (the `patch_mps()` warning never fires on
transformers 5.12.1; `mx.from_dlpack` missing so 1.8.2's MLX path falls back to PyTorch)
[V, issue; R, `HANDOFF/decider-mps-2026-10-10.md`]. The MPS GAP
(GAP-DECIDER-2B-MPS-PATH-GIVES-NO-USABLE-ANSWERS-2026-10-06) is outdated by that fix; its
record is not edited here.

## Sources

- https://huggingface.co/Mapika/decider-2b (card; raw README https://huggingface.co/Mapika/decider-2b/raw/main/README.md)
- https://github.com/Mapika/decider (recipe), `decider/data/mixture.py` via
  https://raw.githubusercontent.com/Mapika/decider/main/decider/data/mixture.py
- https://github.com/Mapika/decider/issues/21
- Local: `HANDOFF/merge-and-bench-2026-10-06.md`, `HANDOFF/decider-mps-2026-10-10.md`
