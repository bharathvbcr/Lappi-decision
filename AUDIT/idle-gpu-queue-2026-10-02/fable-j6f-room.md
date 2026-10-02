# Fable on J6(f)'s target under the ceiling (2026-10-02)

Asked by the lead at ~02:05 UTC 2026-10-02. The question was whether the exact block, 2*max - min >= 1 (spread >= 1 - max), changes the earlier 'record as a consequence, don't fix' ruling on GAP-SUCC-J6F-CANNOT-FIRE-WHEN-2MAX-MINUS-MIN-REACHES-1-2026-10-02. The lead's counterexample was max 46/61 and min 31/61: spread 0.246, so rule (ii) is silent, and 61/61 ties. Advisor: claude-fable-5-1, through a read-only lane (task af52d57cf952d6334). Recorded verbatim. The lead follows it under the user's standing instruction to follow Fable's recommendations. It changes no comparison and no gate: the checker refuses instead of printing quiet.

# Fable ruling: J6(f)'s target under the ceiling (2026-10-02)

Read-only lane. I ran nothing on the box and wrote no file. The repo's rule 2 is untouched: the successor rule picks a recipe and moves no gate (`campaign/f-successor-preregistered.json:3`, "decision").

## 1. Does this change the ruling? Yes.

The lead's arithmetic is exact. Through the checker's own count form (`crates/qd-runtime/src/bin/qd_post_f_rules.rs:1226`, higher-better clears iff `a·f·h + e·b·h > 2·g·b·f`) with candidate 61/61, min 31/61, max 46/61: `61·61² + 31·61² = 92·61²` against `2·46·61² = 92·61²`, a tie, so 61/61 does not clear (strict, prereg `:16`, R7 `:71`). Spread 15/61 = 0.246, so rule (ii) is silent. The block is `2·max − min ≥ 1`, i.e. `spread ≥ 1 − max`, and it does not coincide with rule (ii)'s `spread > 0.30`. The recorded premise says it did (`gaps.jsonl:790`, "Fable stated the trigger as rule (ii)'s spread > 0.30"; `HANDOFF/succ-rule-2026-10-02.md:274`). The sharper form: any F seed at 61/61 kills J6(f) at spread 0. This was never a spread phenomenon. I withdraw "record it as a consequence, don't fix it."

Today a blocked J6(f) prints `quiet` (`qd_post_f_rules.rs:3673`, `:3691`) and, with a J6(d)-v4 win, `fires:j6dv4` (`:3678`), the same words as "examined and did not clear." That is what `CLAUDE.md:86` forbids.

## 2. The fix: option (b), word `refused`. `target_clears` stays as written.

**Option (a) is rejected on substance.** In the lead's own case, `max + min(range, (1−max)/2)` = 46 + 7.5 fires at 54/61: eight items above max while F's seeds vary by fifteen. That is below seed noise, exactly what "by more than the seed range" (`campaign/f-v4-preregistered.json:13`) exists to reject. The `candidate = 1 when max < 1` clause fires on 61/61 against max 60/61, min 30/61: one item. Neither handles max = 1 (room 0), so it moves the boundary without closing the class, and it is a weaker rule chosen with one of three envelope points in hand.

**Option (b), but reuse `refused`, not a fifth word.** `post_f_common.sh:116` maps every word outside `fires|quiet|qualifies|fails` to `refused`, so `cannot-clear:j6f` reads as `refused` on the box anyway; `:75` and the test at `:2843` pin a closed four-word list; and the both-win case already uses `refused` + `refused_because` (`rule_successor`, `:1676-1681`). No fsucc waiter exists yet in `campaign/` (rg, default ignores; no file accepts `fires:j6f`).

**Scope: both arms, both directions, decided from the envelope alone.** It is a property of F's three rows, so it is computable at ~12 UTC before J6(f)'s row exists. J6(d)-v4's span target at 6529/7238 = 0.902 (`:107`) has a ceiling block at spread ≥ 0.098; far from noise, but the class must cover it. ECE is guard-only (`:24`), untouched.

**Exact text, three amendments, no comparison changes:**

Append to `comparison.target_clears` (`:16`):
> Room: the comparison is decidable only when the envelope leaves the bound room to clear. Higher-better, bound U (1 for counts and the paired margin): if 2*max - min >= U (counts: 2*g*f - e*h >= f*h), no candidate can clear. Lower-better, bound L (0 for counts): if 2*min - max <= L (counts: 2*e*h <= g*f), likewise. Decided from the envelope alone, before the arm's rows are read. A target with no room is 'cannot-clear' and refuses the decision (outcomes.refused (c)); it is never read as a target that was examined and did not clear.

Add to `outcomes.refused` (`:79`):
> (c) a target of either arm cannot clear because the envelope leaves it no room (comparison.target_clears, Room). The JSON carries cannot_clear: [{arm, target, max, min, bound}] and refused_because names them. Never quiet and never fires:<other arm> in case (c): a target that could not be examined is not one that was examined and missed, and a both-win cannot be ruled out (CLAUDE.md: a check that could not run must never report the same result as one that ran and passed).

Add to `readings`:
> R9_room: Fable 2026-10-02 withdraws GAP-SUCC-J6F-CANNOT-FIRE-WHEN-2MAX-MINUS-MIN-REACHES-1-2026-10-02's 'consequence, not fixed'. The block is 2*max - min >= 1 = spread >= 1 - max; it does not coincide with rule (ii)'s spread > 0.30 (46/61, 31/61: spread 0.246, 2*max - min = 1, 61/61 ties and does not clear) and holds at any spread when a seed is at 61/61. The comparison is unchanged; the checker refuses instead of printing quiet.

Checker: `j6f_cannot_fire_when_twice_max_minus_min_reaches_1` (`:3658`) is the fail-first; its `:3673`, `:3678`, `:3691` asserts become `refused` with `cannot_clear` naming `j6f` / `needle_8k_worst_bucket`; the `room` case (40/21/40, `:3692+`) stays `fires`. The gap record's status moves from open-as-consequence to fix-pre-registered.

## 3. Legitimacy: amend now.

Legitimate under the pre-registration's own clause (`:64`, `:157`: amend by commit on main before J6(f)-v4's eval row's `written_at`); the file is on main at `e9cff78` (`git log main --`). F seed 0's 40/61 was in the file at registration (`:93-94`), so only arithmetic is new, and no number in any comparison moves. Cleanest bar: land before F seed 1's eval row, since the refusal reads the envelope; seed 1 at step 3,017/9,683 at 01:13 UTC (`:2`) at 1.65 s/step puts that near 04:30 UTC (inferred). The binding bar is J6(f)'s eval row (`:157`).

## Labels

- **Not re-verified against a ledger row in this lane:** 40/61 and 6529/7238 are cited from `:93-94`, `:107` (citing `f4feac15`); `ledger/gh200-p4-v4-2026-10-01.jsonl` is absent locally and I did not touch the box (rule 5).
- **Graph-located:** `rule_successor` and `decide_arms` (`devmap_search`/`devmap_explore`; index generation 3016, `is_fresh: false`, `walk_incomplete` on explore, so the blast radius is a lower bound); the prereg surfaced as a File node. **Direct reads:** the prereg body, `fable-idle-queue.md`, `gaps.jsonl:790`, `HANDOFF/succ-rule-2026-10-02.md:88-100, 268-282`, `qd_post_f_rules.rs:1215-1260, 3650-3700`, `post_f_common.sh:108-130`, `CLAUDE.md:58, 74, 86`. GitPulse not called (REPOSITORY_TRUST_REQUIRED, per the brief).
