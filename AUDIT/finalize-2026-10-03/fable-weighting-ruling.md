# Fable's ruling on weighting v5 toward the human's own callers (2026-10-03)

## The request

The human, 2026-10-03: "Lappi should be a more specialized model, but also cover other decisions
to be competitive with its peer, have more weightage towards my use cases."

## What this file is, and what it cannot be

- **Fable's own text is not recoverable.** Fable ruled through the lead's advisor call at
  2026-10-03T15:12:22Z, and the session transcript stores that result redacted. Fable's words
  cannot be quoted.
- **The only written form is the lead's relay.** The lead relayed the ruling to the bench lane
  nine seconds later (15:12:31Z). That relay is reproduced verbatim below, and it is the record.
- **The record is a relay, not Fable's verbatim text.** Rule 10 needs this file. The other
  `fable-*-ruling.md` files quote Fable directly.

## The relay, verbatim (lead to bench, 2026-10-03T15:12:31Z)

> Lead → bench: Fable ruled on the weighting. The cap table is unchanged. Weight is a ratio over rows that exist, and of the human's caller decisions only DevCouncil's verdict has data: 84% of v4, protected by the two-thirds bound. Your decider routing and commands are already all-admitted. Generic Open-Jev stays as it is; it is the "competitive with peers" half.
>
> 1. **Yes to the domain-level stratum key**, `openjev.<family>/<domain>/<type>`. Numerically nothing moves, since those strata are all-admitted. It makes the dev-tool-shaped strata (shell_history, browser_tools, release_migration) measurable on val per domain.
>    - Make it one Rust change plus the re-survey, as you proposed.
>    - Pre-registration line: per-domain val accuracy for Open-Jev, report-only.
>    - The val draw stays stratified as before, by task × type within each family. Domain is a reporting key, so don't let it multiply the per-family val cap.
> 2. **The build hold lifts** once (a) my reviewer returns clean on your diff and (b) the domain key lands. I'll tell you which.
> 3. **HEAD moved to 7320707.** Your diff and the lock diff both `git apply --check` clean there; I checked. Rebase the domain-key change onto 7320707. Send it as a separate small diff if that's easier for you; I'll apply the main diff first.
> 4. **The untrained caller families go to v6:** DevCouncil relevance, GitPulse commit type, severity and DevType routing. Fable wants them pre-registered now so a lane can start. Nothing new for you in v5.

## Notes on the relay

- **Item 2 is only the bench patch's review condition.** It does not lift the v5 data build's
  other hold, which waits for the human's Apple Diagnostics answer. That hold stands.
- **The 84% figure** is v4's defect-family token share: 83.7%, 256.9M of 306.9M tokens
  (`AUDIT/v5-plan-2026-10-02/v4_token_accounting.json`). The two-thirds bound keeps v5's share at
  or above 55.8%.
- **code.commit_intent was added by the lead, not by Fable.** It is the fifth v6 family in
  `campaign/v6-caller-families.DRAFT.json`, because GitPulse's "does this draft message match the
  patch" is that family (train-plan C2).
