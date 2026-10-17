# Caller records are not admitted as training or eval data (decision, 2026-10-13)

Card: ft-b4f737de23516afeda535bc25af7bbb7. The user handed this decision over with "choose best", and Fable ruled on it in `AUDIT/board-sweep-2026-10-13/fable-owner-decisions-ruling.md`, item 3.

## Decision

1. **App records stay out of training and eval: declined.** The 2026-10-06 "Synthetic only" ruling stands (`docs/caller-contract.md`, `HANDOFF/caller-contract-2026-10-06.md`). It is the basis of the licence approvals the v6 data chain cites (`synthetic-by-rule`, human ruling of 2026-10-08).
2. **Commit the caller-contract work in the four repos: already done.** Lappi's part is 6f65501. For DevCouncil, GitPulse and DevType, "committed" is inferred from GitPulse's file lists. Whether those three are pushed is unverified.
3. **Live check on the real model: later.** It runs only once a v6 export lists `gitpulse.commit_type` or `devtype.route` in `trained_families`. Until then the check could only exercise the refusal path.

## Why

The decline costs nothing today. Admitting records would take three deliberate changes, and none of them is useful until v6 serves a caller family:
- a new `admission` value;
- a review step;
- a change to qd-train's rule-3 door.

## What enforces it now (verified by reading)

- `crates/qd-runtime/src/caller_record.rs:10`, `:31`, `:223`: `admission` has exactly one legal value, `not_admitted`, and any other value is refused.
- qd-train's content guard refuses any file that holds a `lappi.caller_record` line, wherever it lives (e12959e).
- The qd-prep input door refuses the same records by content (5715b38).
- Every Python path that opens rows on the way to shards goes through one caller-record content guard (e6e6974).

## To revisit

Revisiting needs all of:
- a written review step;
- a new admission value, with its tests written red-first;
- the human's explicit yes.

The privacy hardening on ft-ac6677c7 (one secret-pattern set, a held-out guard stronger than the path, visible drop counts) stays on that card.
