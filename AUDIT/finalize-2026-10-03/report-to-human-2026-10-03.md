# Report to the human, 2026-10-03 afternoon

This covers the delegated decisions, the decision data, hallucination and the reward system, and
the items only you can answer. Every number cites a ledger row or a file.

## Answers needed from you

1. **v5's data size.**
   - +15% train tokens over v4's 307M is the default being built. v5's projected cost goes from
     $137 to roughly $151.
   - The full approved decision set is about +30%, roughly $166-180.
   - The draft's launch rule already requires your yes on the final cost before any v5 GPU job,
     so you can answer then.
2. **The Decision Index panel rows.** May they be downloaded, so the new data can be checked
   against them row by row? Without them the check is by dataset id only, recorded as a
   limitation. The bench session will give the file names and sizes when it asks.
3. **The calibration fit, given "less hallucinating" and "reward system".** Your reward system is
   a calibrated abstain threshold at inference, set by the cost c you give a wrong answer
   against an abstention.
   - You scoped the calibration bind out of v5 in the bench session.
   - v4's verdicts say a threshold would **not** fix out-of-distribution answers: the model is
     confidently wrong there.
   - It **would** cut in-distribution wrong answers. At p_top >= 0.9, the defect family's 90 / 84
     / 79 wrong answers drop to 35 / 27 / 28, at 7-8% abstention
     (AUDIT/hallucination-2026-10-03/).
   - Say yes to reopen it, with a c. The fit's objective defect gets fixed first; that is CPU
     work. Or say no, and v5's noul-weight arm and noul data remain the only hallucination
     levers.
4. **Who owns the uncommitted edits in main's checkout?** These are `tokenizer.rs`, `backend.rs`,
   `serve.rs`, `service.rs`, `Cargo.lock` and others, dirty since Oct 1. The answer gates H1, M1,
   M2 and the qdm-digest-parallel merge.
5. **What did Apple Diagnostics report?** Or do you release that hold? The v5 data build waits
   on it.

## Your decision projects and the promotion population

The decided promotion population is `code.defect_class` only. Under it, the new decision
families train and report but cannot promote or block a release. They coexist in v5 that way. A
general bar would have to be set now, and MMLU fails it, so nothing would promote. Say if you
want one anyway.

## What was done under your delegation ("For decisions ask fable and proceed with them")

- **The six promotion decisions** are recorded in `docs/promotion-decisions.json`
  (AUDIT/finalize-2026-10-03/fable-delegated-decisions-ruling.md).
- **The promotion verdict applies them**: main 5e46424, v5-build 610a41b. Before this, the record
  was read but never applied.
- **The verdict also had a defect.** It refused every passing permutation or OOD gate as "a
  capped sample", because those gates count outcomes and it read them as coverage. It is fixed,
  under your delegation, and moves no threshold.
- **The honesty check still holds: no v4 row promotes.** The OOD suite half fails on every one:
  132/180, 56/180 and 68/180 for F seeds 0-2 (f4feac15, aeca8d69, 8c3a774a), against the 170 it
  needs.
- **The per-shape class share** that the decided degenerate-head rule reads is written by the
  trainer and recomputed by qd-gate-report: v5-build c59ccda.
- **The epoch arm** no longer evaluates the whole train plan for no row: v5-build 6f1546c, about
  57 min per run.
- **The no-mask arm is dropped**, and its pin is set at deploy (v5-build 3723bbf). The queue trim
  was skipped.

## The decision data (your answers in the bench session, Fable's design)

- **Sources:** the approved permissive set: Open-Jev v1.1, procedural-typed-decisions,
  typed-decisions (train split only), typed-decisions-synth and HelpSteer2, plus decider's
  teacher data under the repo's Apache-2.0. CC-BY-SA is allowed; those downloads are asked for
  separately.
- **Labels:**
  - Only rows a verifier agreed with, and only distribution labels with a clear mode (at least
    0.6).
  - Unverified teacher answers stay out.
  - Open-Jev's 1,500 customer-control rows of unverified licence are refused per row.
- **Shape:** every source becomes choice rows. Score slots and the soft-label loss wait for v6.
- **New val split:** at most 1,000 rows per new family and about 6,000 in total, split by
  question group.
- **Size:** a token budget of about +15%, filled in order of scarcity: decider routing and
  commands, Open-Jev policy, evidence and rubric, pairwise, then the rest, with NLI last and
  capped.
- **Reporting:** per-family val results are report-only. The defect family's effective token
  share is stated against v4's.
- **Who builds it:** the bench session writes the patches; the Rust normaliser is a `qd-prep`
  subcommand reusing the containment scanner. The lead reviews and commits them.

## Hallucination and the reward system

The full record is AUDIT/hallucination-2026-10-03/README.md.

- **On unknown input types the model is confidently wrong.**
  - The unseen-language and prose cases it answers sit at a median p_top of 0.94-0.98.
  - No threshold passes the gate's two bounds on any seed.
  - The fix is training data and loss weight, which v5 carries: real-file unseen-language noul
    rows (G6), own-repo prose noul rows, and the noul-weight arm.
- **A reward system has exactly two forms here.** With an abstain row, the reward (+1 correct, 0
  abstain, -c wrong) has a closed-form optimal policy, so RL adds nothing:
  - the training form is v5's noul-weight arm, which already exists;
  - the inference form is a calibrated threshold (item 3 above).
- **Prose abstention swings across seeds:** 54, 0 and 1 of 60 on F's seeds. v5 reports it per
  seed.
