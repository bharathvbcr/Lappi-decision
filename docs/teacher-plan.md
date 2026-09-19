# The teacher, without a large local model

**Constraint (2026-09-19):** large LLMs are too slow on this Mac to iterate with. The plan puts
teacher rubric iteration on the Mac using MLX `Qwen3.8-27B-4bit` for 1-2 days. That is out.

This is not a workaround. **The Mac path was also wrong on the merits**, and removing it fixes a
latent validity bug — see §2.

---

## 1. What the teacher actually does

Three jobs, and only two were ever on the Mac:

| Job | Volume | Where the plan puts it | Affected? |
| --- | --- | --- | --- |
| **A. Rubric iteration** | ~50 items x N revisions | Mac, MLX 27B-4bit, 1-2 days | **Yes** |
| **B. kappa >= 0.6 vs 50 hand labels** | 50 items, one pass | Mac, then re-gated in the block | **Yes** |
| **C. Label ~80K real diffs** | 80K items | 8xH100, vLLM TP=8, 0.7 h, $21 | No |

Job C — the expensive one — is untouched. And the block **already** runs an automated kappa gate
against the 50 hand labels ("below 0.6 the pipeline switches to the mutation-heavy mixture and logs
it"). So kappa gets measured on Lambda no matter what.

**The Mac's only real job was to settle the rubric before paying for the block.** That is 50 items,
not 80,000 — a tiny amount of inference whose only problem is that it needs a big model.

## 2. The Mac path had a validity bug, not just a speed problem

The plan iterates the rubric and measures kappa with **MLX `Qwen3.8-27B-4bit`**, then labels the 80K
with **bf16/FP8 vLLM**. Those are different models.

kappa is an agreement statistic about *a specific labeller*. 4-bit quantization changes the label
distribution most on borderline cases — which is exactly where agreement is decided. A kappa measured
on the 4-bit MLX build is not evidence about the vLLM run that actually labels, and the plan never
says so. Measuring it on the serving stack is **strictly more correct**, independent of speed.

So moving this work is an improvement the plan should have made anyway.

## 3. The n=50 gate is underpowered — measured, not asserted

Simulating a teacher that truly agrees ~74% of the time (true kappa ~0.65), bootstrap CI over items:

| n | mean kappa | mean 95% CI width | fraction of draws where the CI lower bound clears 0.6 |
| --- | --- | --- | --- |
| **50** | 0.663 | **0.316** | **10%** |
| 100 | 0.658 | 0.227 | 15% |
| 200 | 0.647 | 0.160 | 22% |
| 300 | 0.652 | 0.132 | 35% |

A concrete n=50 draw: `kappa=0.678 [95% CI 0.512, 0.814]`.

**Reading:** at n=50 the interval is wide enough to contain both verdicts whenever the true value is
anywhere near the threshold. The gate reliably catches a *badly broken* rubric (kappa ~0.3) and
reliably passes an *excellent* one (kappa ~0.85). It cannot tell 0.55 from 0.65.

The threshold is **not** being changed — thresholds are read-only (rule 2). `kappa_gate()` now reports
the interval and says plainly when the sample does not settle the question, so a 0.62 cannot be read
as a clean pass. What to do about it is your call; §6 lists the options.

## 4. Options considered

| # | Option | Cost | Verdict |
| --- | --- | --- | --- |
| 1 | **Move A+B into Session 0** (already 3 h on 1xH100 PCIe, already loads the teacher via vLLM) | 3 h -> ~4-5 h = **+$3 to $6** | **Recommended** |
| 2 | Hosted API to *draft* rubric variants, validate on the real teacher | ~$0 at 50 items | **Recommended as an accelerator**, never as the validator |
| 3 | Hosted API as *the* teacher for all 80K | ~240M input tokens; $50-250+ | **No.** Costs more than the $21 vLLM line, breaks the pinned-weights protocol hash, and adds a redistribution question about the labels |
| 4 | A smaller local teacher | $0 | **No.** The rubric must be followable by the model that labels. Tuning it against a 4B misleads, and a weaker teacher lowers kappa |
| 5 | Drop the teacher, raise the mutation share | $0 | **Not yet** — but it is the existing fallback, see §7 |

## 5. Recommendation

**Reorder the work so the human effort lands where it is unavoidable, and delete the Mac inference
loop entirely.**

1. **Hand-label first.** You have to label 50 (ideally 300) diffs regardless; they are held-out and
   never trained on. Labelling is where the decision rules actually surface — you cannot write a good
   rubric before you have argued with 50 real cases.
2. **Write the rubric from that experience.** Drafting a rubric is *writing*, not inference. No model
   required. I can draft it from your labels and your disagreement notes.
3. **Validate in Session 0.** Extend Session 0 from 3 h to ~4-5 h. It already loads the teacher
   through vLLM; add the 50-item agreement pass and one or two rubric revisions. 50 items x a handful
   of revisions is minutes of H100 time.
4. **Optionally pre-screen with an API** between steps 2 and 3 to catch obvious rubric ambiguity for
   free — but kappa is only ever measured on the real teacher.

Net effect: **1-2 days of Mac work and a 15 GB local model removed; +$3-6 on Session 0; the kappa
measurement becomes valid.** Session 0 goes from $10 to ~$14-16; the program estimate is unchanged at
the dollar level that matters.

## 6. The decision left to you

The n=50 power problem in §3 is real whichever route the inference takes. Three honest responses:

- **Label 300 instead of 50** for the agreement set. You are already labelling 300 for held-out; using
  the same effort for both is the cheapest way to halve the interval. Note the sets must stay
  disjoint if you want both uses.
- **Keep 50 and accept a weaker guarantee**, recorded in the ledger as such, treating the gate as a
  broken-rubric detector rather than a fine discriminator.
- **Keep 50 and require the CI lower bound to clear 0.6** — much stricter than the plan intends
  (it needs a point estimate near 0.78), and would likely fail a serviceable rubric.

I have not chosen for you, and `kappa_gate()` implements none of them as a silent default: it reports
the point estimate, the interval, and whether the interval settles the question.

## 7. The existing fallback still stands

The teacher is **20% of the fine-tune mixture** — the smallest and most replaceable slice (40%
mutation, 20% teacher, 10% DevType intents, 30% open mixture). The plan already routes around a bad
teacher: below kappa 0.6 the pipeline switches to the mutation-heavy mixture and logs it.

What is lost without the teacher is **natural-distribution supervision**. Mutations are synthetic by
construction, which is exactly why the transfer gate exists ("a model trained on mutations only must
beat the char-n-gram baseline on the *natural* held-out set"). Drop the teacher entirely and training
becomes synthetic plus open-domain, with the only natural signal in a held-out set it must never see.

So: keep the teacher. Just stop iterating it on a laptop.
