<!--
DRAFT for a LinkedIn article, in the author's voice. Not posted: posting is the author's action.
Every number below is from the project's own records; the source file for each is listed in
docs/outreach/README.md ("Where each number comes from"). Links marked [pending] are private today.
-->

# Lappi v0.1: my first model, and the two bars it missed

Lappi is a 2-billion-parameter model that answers typed questions. You give it some context, a
question and a list of options. It picks one option, gives a score, or says it should not answer.
It never writes a sentence.

I released a preview of it on 6 October, 17 days after the first commit. It is not good enough to
call finished, and this post says where it falls short as plainly as where it works.

## What it does

A request is a small JSON object: the context as bytes, a question, and named slots. A choice slot
holds up to 16 options. The model is Qwen3.5-2B-Base, fine-tuned. Lappi does not generate an
answer: it reads one token, from a 17-row slice of the output layer. That is a letter for each
option plus one reserved row that means "I should not answer".

Here is a real request from the load test. The message was "how do i export my report as pdf?",
with six routing options. Lappi answered `inquiry` with a score of 0.986. On an M5 Pro the whole
decision takes about 72 ms for a short prompt and 2.3 s for an 8,000-token one. That is the same
cost as the base model, because only the weights changed.

Two design choices matter more than the size:

- **Abstaining is a row, not an option.** Most small decision models add a "none of these" answer.
  That answer competes with the real options for probability, so how often the model abstains gets
  tangled up with which labels are common. Lappi's abstain row competes in the same softmax but is
  never one of the options.
- **Every question is asked twice.** The second time the options are in a different order. If the
  two answers disagree, Lappi abstains. A model that changes its mind when you shuffle the options
  was not reading them.

## How the training differs

Most of the effort went into making the numbers hard to fool myself with.

- **Labels by construction.** The core task was "what kind of code change is this?". Its training
  rows come from a Rust mutation engine that edits real open-source files with tree-sitter. The
  engine knows exactly what it changed and on which lines, so the label is a fact about the edit.
  No model had to guess it.
- **Pre-registration.** Every training run had its bar written down and committed before its result
  was read. When a run missed, the miss stood.
- **Controls.** A copy trained on shuffled labels scored at chance on the four seeds where it ran,
  so the real scores are not leakage. A plain linear classifier is the baseline it has to beat: on
  one seed, Lappi beat it in 20 of 22 task families. The pooled version of that check could not
  run, which is one reason this release is a preview and not a promoted model.
- **Held-out families.** Two task families were held out of training completely. The trainer
  refuses to read their files rather than trusting a convention.
- **Five seeds, averaged.** The released weights are the mean of five training runs. Seeds disagreed
  about long context and about abstention, so one lucky seed would have said little.
- **Calibration on the margin.** Scores are fitted with split-conformal calibration per question
  size, not one global temperature. It mostly works (calibration error 0.006-0.038 on 8 of 9
  measured sizes) and fails on 16-option questions (0.079).

The recipe itself is plain: supervised fine-tuning only, no RL, one epoch, in PyTorch on two H100s.
I also have a Rust trainer on my own Metal kernels, but the shipped model was not trained with it.

## Why it took 17 days

The commit log has 1,087 commits. 1,060 of them were co-written with AI coding agents (Claude),
working in parallel under a rulebook I wrote. The top rule is that an agent may report that a gate failed but may never move a
threshold, drop a seed or shrink a test set to make it pass. Here is where the days went:

- **Cheapest model first.** Before touching a 2B model I trained a 606K-parameter byte-level model
  from scratch, to set a floor. A linear classifier beat it in 17 of 18 comparisons (best 81.0%
  against 88.5%). That result is what justified the bigger model.
- **The 2B would not train on my Mac.** A full run peaked at 80.8 GB on a 64 GB machine and was
  killed. Training moved to rented GPUs, first a GH200 and then two H100s. The final box billed
  about $463.
- **The training corpus went through five versions.** The project's own checks found 215
  validation questions overlapping MMLU and CommonsenseQA training rows. The abstain examples were
  rebuilt several times, because abstaining on one kind of strange input did not carry over to
  another.
- **Gates that would not move.** The run before the final one failed four of them. The final run
  failed out-of-distribution abstention on all five seeds. Some definitions did change, but only by
  a recorded decision of mine, never by an agent.
- **One machine, one heavy job.** My Mac kernel-panicked twice on 2 October. After that, heavy jobs
  ran one at a time.

## The bars it missed

Before averaging, I wrote down a bar: the average had to be no worse than the weakest seed on six
measures. It passed four and missed two.

- **Out-of-distribution abstention: 149 of 180.** The bar is 90%. Prose that is not code got 60 of
  60 abstentions, scrambled text 49 of 60, and unseen programming languages only 40 of 60. On the
  rest it answers, often confidently.
- **Span accuracy fell to 77%.** The span head says which line is the evidence. Averaging five seeds
  blended span heads that point in nearly unrelated directions. So this release does not serve span
  answers at all.

I shipped it anyway, as a preview, because it is my first model and I would rather show it with its
measurements than polish it in private.

## What it serves, and what it does not

- **It serves choice questions in 21 trained task families.** These are routing, policy, evidence,
  reading comprehension, NLI and science questions from open datasets. On the families it serves it
  gets 84.2% of validation questions right. At an 80% precision target, it answers about 91% of
  them at about 86% precision.
- **It does not serve the code-defect task it was built for.** That task's trained request asks for
  a label and a span together, and span is not served. A label-only request is a prompt it never
  saw. So the tool I meant it for, a code-review gate, cannot use this release.

## How it compares

I ran it on JevArena, a pairwise-judge benchmark: 849 pairs from JudgeBench, RM-Bench and
RewardBench 2, each asked in both orders, on local models only. This is not what Lappi was trained
for.

| Judge | Size | Domain macro accuracy | Order consistency |
|---|---|---|---|
| Lappi v0.1 | 2B | 0.534 | 0.716 |
| Qwen3.5-2B, as a chat judge | 2B | 0.522 | 0.523 |
| Gemma 4 E4B, no reasoning | 8B | 0.632 | 0.801 |

- **Against Qwen3.5-2B it is a tie.** The interval on the difference includes zero.
- **Gemma 4 E4B, about four times larger, is clearly better.**
- **Outside one domain (deciding when two answers are equally good), Lappi is near chance.**
- **On RM-Bench's code pairs it scored 0.111.** It called most pairs a tie or picked the broken
  program. That is a defect in the domain it was built for, and the next version has to fix it.
- A run of Gemma 4 E4B with reasoning on is still pending.

Other small "typed decision" models appeared the same week I started: TypeSafe's Jev, Bespoke's
Nimble, and decider-2b, which uses the same base model. Their published numbers are on different
tasks and metrics, and I have not run them head to head, so I am not ranking them against Lappi.
By their own documentation, none of them returns a span (which line is the evidence), and where
abstention is documented it is a catch-all answer.

## What I would tell myself on day one

- Write the bar down before you look. It is the only defence against a result you want to believe.
- Run the cheap baseline first. It decides whether the expensive model is worth building.
- Count what you actually serve, not what you trained.

The preview's weights, model card and source will be public soon. [pending: links]

The full JevArena results: [pending: link]
