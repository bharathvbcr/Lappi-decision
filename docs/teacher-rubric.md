# Teacher rubric v1 — classifying an agent-authored diff

This is the prompt the teacher follows to label ~80K real AgentPack diffs. It is **v1, unvalidated**:
its kappa against the hand labels has not been measured, because that measurement belongs on the
serving stack (see `docs/teacher-plan.md`). Revise it from the disagreement list, not from taste.

Design constraints, because a 27B is not a frontier model:

- **Ordered decision rules, not a prose description.** A 27B follows "check A, then B" reliably and
  follows "weigh these considerations" badly.
- **Explicit precedence**, so a diff matching two classes has exactly one correct answer.
- **Hard labels only.** No confidence, no rationale in the output — the plan is explicit, and free
  text invites the model to talk itself out of a correct label.
- **Negative examples per class**, because the errors are near-misses, not wild guesses.

---

## The prompt

> You are labelling a single code diff written by an AI coding agent. Decide what the diff **does to
> the program's behaviour**. Answer with exactly one word from: `stub`, `logic`, `cosmetic`, `clean`.
> Output nothing else — no explanation, no punctuation.
>
> Apply these rules **in order**. The first rule that matches is the answer.
>
> **Rule 1 — `stub`.** The diff makes a function stop doing its work while still appearing to
> succeed. Answer `stub` if the changed code does any of:
>
> - replaces a body with an explicit not-implemented marker (`todo!()`, `unimplemented!()`, `pass`,
>   `raise NotImplementedError`, `panic("not implemented")`, `throw new Error("TODO")`);
> - returns a type's empty or zero value in place of doing the work — `Ok(())`, `nil`, `null`,
>   `None`, `[]`, `{}`, `0`, `""`, `false`, `undefined`;
> - returns a fixed literal that happens to match what a caller or a test expects, instead of
>   computing it;
> - returns early, before the real work, so the body below it cannot run.
>
> The last three matter most. They **look like working code**. A reviewer scanning for the word
> "TODO" will not see them, and they are the reason this task exists.
>
> **Rule 2 — `logic`.** The work is still attempted, but the diff changes behaviour in a way that
> looks like a defect rather than an intent. Answer `logic` if the diff:
>
> - inverts or negates a condition;
> - moves a loop or slice bound by one, or changes `<` to `<=` (or the reverse);
> - swaps two arguments of the same type at a call;
> - deletes an `else` branch or a case;
> - discards an error that was previously handled or propagated — `?` becoming
>   `.unwrap_or_default()`, a removed `if err != nil`, an emptied `catch`;
> - removes an `await`, a lock acquisition, or a synchronisation point;
> - changes a constant that affects behaviour.
>
> **Rule 3 — `cosmetic`.** The diff cannot change behaviour at all: renamed locals, reformatting,
> reordered imports within a group, comment or docstring edits, line wrapping. If you are not
> certain the behaviour is identical, this is not `cosmetic`.
>
> **Rule 4 — `clean`.** Everything else. The diff does real work and you found nothing wrong with it.
>
> `clean` means **"I found nothing"**, not "I verified this is correct". Use it whenever rules 1-3
> do not match, including when the diff is too large or too unfamiliar to judge.
>
> **Precedence.** A diff that both reformats and stubs a function is `stub`. Severity wins:
> `stub` > `logic` > `cosmetic` > `clean`.
>
> **Scope.** Judge only the lines the diff changes. Do not judge code that was already there, and do
> not reward or punish the commit message — a message that claims more than the diff delivers is
> evidence for rule 1, not a class of its own.

---

## Boundary cases the rubric must settle

These are the disagreements to expect. Each is stated as a rule so the teacher and the human resolve
them the same way.

| Case | Answer | Why |
| --- | --- | --- |
| Body replaced by `Ok(())`, and the function genuinely has nothing to do | `clean` | A no-op is only a stub when work was expected. If the signature and name promise nothing, there is nothing missing |
| `return nil` added at the top, real body left below | `stub` | `stub.early_return`. The dead body below is the tell |
| A test changed so it passes, without touching the code under test | `logic` | It discards a check that previously ran. Not `stub`: the test still does something |
| A deleted function | `clean` | Deletion is not a stub. If its callers now break, that is the compiler's finding, not this model's |
| A new function that is deliberately a placeholder in new scaffolding | `stub` | Still a stub. The label describes the code, not the intent |
| Renamed a local **and** changed a bound | `logic` | Precedence: severity wins over the cosmetic part |
| Reformatting that also reorders two statements | `logic` | Statement order can change behaviour, so it is not `cosmetic` by rule 3's certainty requirement |
| A diff too large to read in the window | `clean` | `clean` is "nothing found". Guessing on an unread diff is the error mode that poisons labels |
| Generated or vendored code | `clean` | Judging generated output teaches the model about the generator |

## Output and abstention

Hard label, one word. The teacher is **not** offered an abstain option, deliberately: the plan wants
hard labels only, and the abstention path belongs to `noul` in the served model, learned from the
open task mixture and the held-out families — not from teacher hedging.

The consequence is that teacher errors land in `clean`, which is the intended sink: `clean` is
already the noisy class ("some real agent diffs are stubs"), and the mutation engine supplies the
clean signal that the teacher cannot.

## How to revise this

1. Run the 50-item agreement pass on the real teacher.
2. Read `disagreements()` output, not the kappa. Kappa says *whether* to revise; the disagreement
   list says *what*.
3. For each disagreement, decide whether the **rubric** was ambiguous or the **teacher** was wrong.
   Only the first is fixable here. Add a row to the boundary-case table; do not add prose.
4. Re-measure. Record both kappa and its interval in the ledger.

Do not revise by adding emphasis ("carefully consider...", "it is very important..."). A 27B responds
to decision rules and worked boundary cases; emphasis costs tokens and buys nothing.

## The ceiling this is measured against

The teacher's kappa against a human is bounded above by that human's kappa against **themselves**.
`qd_label` re-presents ~10% of items later in a labelling session and reports intra-rater kappa; that
number is the denominator. A teacher at 0.62 against a human who self-agrees at 0.70 is close to the
ceiling; the same 0.62 against a human who self-agrees at 0.95 is not. Establish the ceiling before
judging the teacher — and note `docs/teacher-plan.md` §3: at n=50 neither number is measured
precisely enough to separate 0.55 from 0.65.
