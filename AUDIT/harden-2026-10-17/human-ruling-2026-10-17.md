# Human ruling, 2026-10-17: hardening against v1's mistakes

Recorded by session lappi-decision-72. The words below are the human's, copied verbatim.
No result of any LoRA-2B run existed when this was written: no `ft` or `eval` row in `ledger/`
names the LoRA-2B registration.

## The request

> Hadren everything such that I won't repeat the same mistakes I did with lappi v1; including data processing

"Lappi v1" here means the v0.1 preview (`HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md`).

## The three questions and the human's answers

The questions were asked through the session's question tool. Each answer is the option the human chose.

**Q1.** "LoRA-2B is preregistered on the v5r shards, which contain MMLU/CLINC test items. Once a 'no benchmark targets in train' guard exists, what should it do to this run?"

**Answer:** "Refuse; rebuild on v6 (Recommended)". The option read: "Fixes M6 at the source. Means an amendment to the LoRA-2B preregistration and a shard rebuild before the smoke run; the other session's lane is affected."

**Q2.** "Two of the new checks are gates, and rule 2 makes gates human-owned: the span head must beat the random-'+'-line null, and a decision record must be committed before the results it judges. How should they land?"

**Answer:** "Enforce both as gates (Recommended)". The option read: "You rule them in now; they bind promotion from the next registration on."

**Q3.** "How much should I build in this session?"

**Answer:** "Steps 1-4 now, step 5 after the pin commit (Recommended)". The option read: "All mechanical guards outside the other session's files; the LoRA launcher binding gets agreed with that session first."

The plan the steps refer to:
1. `make preflight-train` and `make preflight-release`, which run every guard as a tri-state.
2. Training guards: binding to the preregistration, a "started" row before the run, an explicit run cap, `mac_heavy`'s cap below memguard's, and `ckpt_average` refusing heads that were initialised independently.
3. Data guards: v6 dedupe and containment as the default for every rebuilder, a positive control in `qd-prep` `decisions.rs`, a composed-row `cluster_key`, and a check that examples match their manifest.
4. Release guards: every trained slot shape calibrated and passing, every served family and request shape replayed through the socket, metrics computed over the served population, and ledger-row citations in new handoffs.
5. The pin gate refuses a dirty ojas tree and requires a delta report. GPU pins join the gates.

## What this rules, and what it does not

- **v5r is refused for LoRA-2B.** `campaign/lora-2b-preregistered.json` gets a new `amendments[]` entry that rebuilds the run on v6-rule data. That entry cites this file. Amendment 1 (the step count) stands as is.
- **Two gates are added**, binding from the next registration on:
  - **Span null:** the span head's top-1 must beat the random-`+`-line null on the same rows.
  - **Decision-record ordering:** a gate's decision record must be committed before the first result it judges.
- **Thresholds.** The human set none beyond these two comparisons. No existing threshold, seed count or held-out set moves (rule 2).
