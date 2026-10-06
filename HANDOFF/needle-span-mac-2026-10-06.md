# HANDOFF needle-span-mac, 2026-10-06

Lane: Mac, read-only analysis of existing artifacts. It covers the needle per-case diagnosis
(done) and the span-head refit (being designed; Fable is ruling). It follows
`AUDIT/training-audit-2026-10-06.md` section 6, items 1 and 2.

Nothing was trained, scored, rented or merged. No model was loaded. No gate, threshold or
held-out set was touched (rules 2 and 3). The needle work is file analysis of verdicts that v5's
eval rows already wrote.

Labels: **[V]** produced by the command below or read at file:line; **[I]** inferred;
**[U]** unknown.

## 1. Needle: what the v5 misses are

**Command:**
`python3 build/needle-span-mac-2026-10-06/needle_diag.py`, a throwaway analysis script
(Python policy category 3; it never ships). Its output is in
`build/needle-span-mac-2026-10-06/needle_diag.out`.

**Inputs** (sha256 prefix; each file carries `eval_row_id` per case):

| File | sha256 prefix | Eval row |
|---|---|---|
| `/Users/bharath/qd-campaign/verdicts-2026-10-05/v5/suite-verdicts-s0.jsonl` | `be77b06392d6b267` | 166f7ebd |
| `.../suite-verdicts-s1.jsonl` | `d6c93b3fe7650147` | e1bafd6f |
| `.../suite-verdicts-s2.jsonl` | `933e134c17fb0fcc` | 2af3c80d |
| `.../suite-verdicts-s3.jsonl` | `456845d55bb028e7` | e09b652a |
| `.../suite-verdicts-s4.jsonl` | `b0ec043cb629fe7a` | a51c5bf4 |
| `/Users/bharath/qd-campaign/preview-2026-10-05/v5-avg-verdicts/suite-verdicts-avg.jsonl` | `08bbe7c188b24dfc` | 6af73bef |

Every file has the same 300 needle cases (60 per language × 5 languages, about 8.1-8.2K tokens).

**Hypotheses, stated in the script before the misses were read:**
- (a) abstentions;
- (b) the needle is the last hunk;
- (c) wrong picks sit at a fixed offset;
- (d) misses concentrate by language or length.

**Results [V]:**

| Run | Misses | Abstained | Wrong pick | Offset (predicted − needle) | Languages of misses |
|---|---|---|---|---|---|
| seed 0 | 7 | 0 | 7 | −1: 6, −2: 1 | swift 4, typescript 3 |
| seed 1 | 0 | – | – | – | – |
| seed 2 | 26 | 0 | 26 | −1: 22, −2: 4 | swift 13, typescript 13 |
| seed 3 | 2 | 0 | 2 | −1: 2 | typescript 2 |
| seed 4 | 4 | 0 | 4 | −1: 4 | typescript 4 |
| average | 0 | – | – | – | – |

- **(a) is refuted.** There are no abstentions among the 39 misses.
- **(b) explains at most 2 of 39.** The needle is the last hunk in 2 of the 300 cases.
- **(c) holds strongly.** Every wrong pick is one or two hunks **before** the needle (34 at −1,
  5 at −2). None is later, and none is the final hunk.
- **(d) holds by language, not by length.**
  - All 39 misses are Swift or TypeScript. Rust, Go and Python have 0 misses on every seed.
  - Token lengths of misses and hits overlap: misses 8119-8190, hits 8113-8192.
  - The misses' depth fractions run 0.63-1.0, which is why they land in the 60-80% and 80-100%
    buckets.
- **Pointer detail [V, on seed 2's misses]:** seed 2 points at the **first line of the
  preceding hunk**. Seed 1 points at the needle's first line on the same case. The gap is 7
  lines in Swift and 8 in TypeScript, exactly one hunk. Example: `needle-swift-0189`, seed 2
  line 395 (hunk 78) vs seed 1 line 402 (hunk 79, the needle).
- **Overlap across seeds:** 6 of seed 0's 7 missed cases are among seed 2's 26. 4 of the 5
  cases missed by seeds 1/3/4 are among seed 2's. The same cases are hard across seeds.

**What this means:**
- **Not early context lost [V].** The misses are not in the early-context region, and they are
  not abstentions.
- **What fits [I]:** a confusion between adjacent hunks that depends on language and position.
  It appears late in the context and only for Swift and TypeScript.
- **The cause is [U].** Candidates: those languages' hunk-boundary tokenisation (for example a
  closing `}` or blank line merging into the next hunk's first token, the line-start-collapse
  family), or the filler and needle templates for those two languages.
- **Cheapest discriminating next check:** render one Swift or TypeScript miss case through
  `python/qd_train/needle.py`'s builder with the Qwen3.5 tokenizer, and inspect the tokens at
  the boundary between hunk k−1 and the needle against a Rust case. Fable is ruling on whether
  to do this before v6 design.
- **Records:** this refines GAP-V5-NEEDLE-END-OF-CONTEXT-MISSES-SEEDS-0-2-HAVE-NO-ID-2026-10-06,
  which the merge-and-bench lane appended. Its owner may add these per-case facts.

### 1b. Why Swift and TypeScript

This section was added after Fable's ruling; its parts are labelled.

- **Filler templates [V, `python/qd_train/needle.py:55-96`].** Rust, Go and Python each have
  two filler templates. TypeScript and Swift have one each.
- **The candidate state is a line's first token [V, `python/qd_train/heads.py:357-358`].** The
  span head scores each candidate line by the hidden state at that line's first token
  (`hidden.gather` at `plan.candidate_pos`). Under causal attention, that state has read only
  the prefix plus that one token.
- **Tokenizer check [V].**
  - Command: `/Users/bharath/.venvs/ml/bin/python build/needle-span-mac-2026-10-06/needle_boundary.py`.
    It loads only the tokenizer, with no model and no lock, and builds a heuristic-sized suite.
  - In all five languages, the needle's first line and the previous hunk's first line share
    their first two Qwen3.5 tokens: `@@` and `Ġ-`. They differ only from the third token on,
    in the line numbers.
  - So the candidate state at the needle's first line has seen nothing of the needle.
- **Inferred [I].** To point at the needle, the head can only match on the prefix: the state
  "just after hunk k−1". With one repeated template, consecutive prefixes differ only in
  randomly drawn names, numbers and function names. That fits the one-hunk-early misses
  confined to Swift and TypeScript. It is not proven: no ablation was run.
- **What follows for v6 [I].** Two findings point the same way: the free pass from the diff
  marker (GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD) and this needle failure. In both,
  the pointer scores lines by a state that has not read the line. A candidate represented by
  the line's **last** token, which has read the whole line, is the obvious alternative.
  - It is a training and serving contract change; Fable rules on it.
  - It can be tested cheaply on frozen features: gather end-of-line positions in the
    span-refit cache.
- **Suite finding, not a gate change (rule 2).** The single-template languages make the needle
  suite harder in a way the gate does not name. That is reported here; the gate's population is
  not changed.

## 2. Span-head refit: status

- **Available on the Mac [V]:**
  - the v5 average, with its mean span head;
  - v5 seed 1's checkpoint, with its own span head;
  - per-row verdicts for all five seeds.
  The other four seeds' weights are not on the Mac, so "mean of five heads' logits" cannot be
  computed here.
- **Head shape [V]:** `span_head.start_proj.weight` [2048, 2048], `end_proj.weight`
  [2048, 2048], and `abstain_start` / `abstain_end` [2048].
- **Design pending.** Fable is ruling on four questions:
  - feature extraction, torch on MPS vs qd-metal (which has no span or hidden-state output);
  - the fit;
  - the arms (shipped mean head, seed 1's head on the averaged tower, refit head);
  - the null (a random `+` line, per GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD).
- **Before any result is read,** a pre-registration goes into `campaign/`. Any run goes through
  `tools/mac_heavy.sh` and is marked `quick` (rule 8).
- **Pre-registered:** `campaign/span-refit-mac-preregistered.json`, sha256
  `87c02014e34dec619e536a221447bbffd80a724e7afe5a7c5b1598fcb5bd347c (corrected citation heads.py:357-358 before any result; the first draft hashed 836c5cf9…)`. It was written before any
  span feature was extracted or any arm scored, and it is uncommitted.
  - **Arms:**
    - (a) the shipped mean head on the averaged tower; this arm is the validity gate;
    - (b) seed 1's head on the averaged tower;
    - (c) a head refit on line-start features;
    - (d) seed 1's tower and head together;
    - (e) a head refit on each line's last token. This arm is not servable; it is a probe of
      "has the candidate state read the line".
  - **Population pin:** `bdd571b6…` (7,238 val span rows).
  - **Blocked on:** the Mac heavy lock. The merge-and-bench lane holds it, with two jobs queued.
    The extraction script is not yet written.

## Open

- The needle cause [U]; the next check is in section 1.
- The span refit: design, pre-registration, run.

## First command for the next lane

`python3 build/needle-span-mac-2026-10-06/needle_diag.py`, which reproduces section 1. The span
refit's first command is set after Fable's ruling.
