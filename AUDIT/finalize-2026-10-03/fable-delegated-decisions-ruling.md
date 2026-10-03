# Fable's rulings on the open decisions, under the human's delegation (2026-10-03, ~13:15Z)

## The delegation

At about 13:14Z on 2026-10-03, the human wrote in the lead's session, verbatim:

> For decisions ask fable and proceed with them

It came after the lead's report, which listed the decisions waiting on the human. These were:
- whether "optimize" reopens model size, data or recipe;
- the H1/M1/M2 inference patches;
- the six open items in `docs/promotion-decisions.json`, and the calibration fit's conflict with
  the in-distribution cap;
- no-mask drop or amend;
- the Apple Diagnostics gate;
- the queue trim;
- the sha2 0.11 upgrade;
- box profiling runs.

The lead consulted the advisor (Fable) once on the whole list. The ruling follows verbatim. Every
record written under it names this file in its `decision_ref`.

## Fable's ruling, verbatim

The human delegated decisions, not facts. Two of your eight items are facts only they hold — who owns main's dirty files (gates H1/M1/M2 *and* the sha2 merge, since main's `Cargo.lock` is in the same dirty set), and the Diagnostics result (a safety hold they set in their own words). Those go back as two questions, not decisions. Everything else I rule on now.

Before any of it: record this ruling verbatim in `AUDIT/finalize-2026-10-03/` with the delegation quoted, so `decision_ref` has something to point at.

### Rulings

**1. Reopen size / data / recipe: no.** v5 is the experiment. "Optimize" and "proceed" don't say reopen.

**2. Promotion record — write it, under the delegation.** `decided_by` on every item: the human's sentence verbatim with timestamp, "through the lead, ruled by Fable." The file's own instruction is one commit per decision naming the gap; do that. Do not edit `who_edits`.

- `promotion_population`: code.defect_class; general families report-only. Record both readings' numbers from the brief.
- `average_may_promote`: false.
- `ece_population`: the same population — code.defect_class letter rows, per shape where shapes exist there, per-language report-only. One population decision drives all three in-distribution gates; a different ECE population would reintroduce the pooled-base-rate objection row 1 rejects. Readings: seed 0 0.0207, ens5 0.0135, avg 0.0498.
- `degenerate_head_floor`: retire the entropy half; keep max-class-share ≤ 0.95 on every slot shape. The floor penalizes calibrated confidence, which is the model's design goal; ECE already measures whether confidence is earned, and share catches the actual degeneracy (one class always). **This is the one ruling that flips v4 verdicts** — ens5, avg and seeds 1–4 go fail→pass on this control. Say so in `decision_ref`, with the k2/k16 entropies.
- `privileged_hunk_pass_rule`: retired for v5. 48% whole-file windows is the gap's own measurement; no eligibility share makes that a ceiling. Re-specification waits for a corpus with small hunks.
- `transfer_gate_definition`: retired as a promotion requirement for v5; the gap stays open as the research question. The record must state the scope consequence: promotion under it certifies the mutation-labelled distribution only, and every export/release note carries that line.

**Honesty check to put in the AUDIT record:** under this full record, every v4 row still fails `ood_abstain`'s suite half (132/180, 61/180), seed 0 also fails needle, and `paired_margin_vs_linear` is still not_run everywhere. The record rescues no v4 row. That's what separates it from gate-moving.

**Before writing the file:** `rg -n 'promotion-decisions' python/ crates/ tools/` on v5-build. If `promotion_verdict` or `qd_post_f_rules` reads it, confirm the reader accepts "retired" as a status and knows to drop a retired control from the required list at `ledger.py:148-160`. If nothing reads it, the record is documentation and the retirements need a code change on v5-build to take effect — that's a second deliverable, with a fail-first test, not an assumption.

**3. No-mask: drop.** Amending τ after reading P2 is the trap p2_amendment_2 named. Apply all three edits from `build/v5-c2a-off-draft.md` on v5-build now (the box binaries aren't built; the deploy commit is this one). Fill the placeholder with "Fable's ruling under the human's delegation of 2026-10-03," not "the lapse."

**4. sha2 0.11: approved.** `cargo update -p sha2` is the only lock change. Tell the bench. The merge still waits on main's `Cargo.lock` — same unknown-owner fact as H1.

**5. Queue trim: not without reading first.** You asked the human for the table because you didn't know how a `.queued` marker is consumed. Delegation doesn't change what you know. Read the waiter that consumes markers; if removal is a documented dequeue (not an edit of a running script), the only candidate is j6a, and only after its pre-registration shows it feeds no pin. rung0/cudadev/rungd are the ojas CUDA path the human approved 2026-10-01 — leave them. fsucc/F′/j5pp decide pins — leave them. Net trim is ≤6 h against a queue that doesn't reach v5 before j6g anyway; if the read is unclear, skip the trim and say so.

**6. Preamble profile: yes, on the Mac, not the box.** The ~6 min is CPU Python before the first GPU call (val set, second pass, suites, the train shard pass). The Mac has the v4 shards it built. cProfile a `--score-plan` startup under `mac_heavy.sh` and kill it at the first backbone load. No box run, no GPU contention, no human yes. The decode restructure: no run before v5 — there's no code to test.

**7. Calibration fit vs in-dist cap: open, human-owned, not pre-train.** Record only; no export ships a table until reconciled.

**8. H1/M1/M2 and Diagnostics: hold.** Facts, not decisions. One question each to the human.

### Order

1. AUDIT record of this ruling (verbatim, delegation quoted, honesty check included).
2. The reader check (`rg promotion-decisions`), then the six commits — or the code change first if the reader needs it.
3. No-mask drop on v5-build.
4. Tell the bench: sha2 yes; H1 held on the ownership fact.
5. Mac preamble profile.
6. Report: what's recorded, the two facts you need, the trim verdict.

## The honesty check, with its numbers

Under the full record above, the v4 rows read as follows. All values are ledger values:
- seed 0 is f4feac15 in `ledger/gh200-p4-v4-2026-10-01.jsonl`;
- ens5 is b45406b5 and avg is c962cdd9 in `ledger/gh200-p6-f-j7prime-2026-10-01.jsonl`.

| Gate or control | seed 0 | ens5 | avg |
|---|---|---|---|
| permutation_consistency, code.defect_class (floor 95%) | 99.44% pass | 99.44% pass | not read here |
| ood_abstain, in-distribution half, code.defect_class (cap: Wilson upper ≤ 0.05) | 13/2,304 pass | 13/2,304 pass | not read here |
| **ood_abstain, suite half (Wilson lower ≥ 0.90, so ≥ 170/180)** | **132/180 fail** | **61/180 fail** | **62/180 fail** |
| ece, code.defect_class k4 (≤ 0.05) | 0.0207 pass | 0.0135 pass | 0.0498 pass |
| degenerate_head, max-class-share ≤ 0.95 only | pass (largest top share 0.517) | pass (0.511) | pass (0.509) |
| needle_hunk_recall (worst depth bucket ≥ 0.95) | **0.656 fail** | 1.000 pass | 1.000 pass |
| paired_margin_vs_linear | **not_run** | **not_run** | **not_run** |
| shuffled_label | not_run (J5′ is writing it now) | not_run | not_run |

**No v4 row promotes under this record.** Seed 0 still fails the suite half and needle. ens5 and avg
still fail the suite half. paired_margin_vs_linear is not_run on all three. avg also stays out
under `average_may_promote: false`.

The only verdict the record flips is `degenerate_head`, from fail to pass on ens5, avg and F seeds
1–4. Their failing entropies were:
- ens5: k2 0.1415, k16 0.1383;
- avg: k2 0.0956, k16 0.1312;
- seed 1 (aeca8d69): k16 0.1337;
- seed 2 (8c3a774a): k2 0.0988, k16 0.1452;
- seed 3 (ebc83be6): k2 0.1433;
- seed 4 (d8c8300a): k2 0.1252, k16 0.1474.

Where the top-class shares come from: a failing slot's detail names only its entropy, so the shares
are the rows' `degenerate_head.family.*` metrics. Each of k2 (intent.in_scope) and k16
(intent.classification) is a single family, so its family metric is that slot's share. Across all
seven rows, the largest top share on any slot is 0.517 (k2), against the 0.95 bar.

## Follow-ups after the ruling (Fable's second pass, same session)

- **Ruling 3, the pin.** `V5_C2A=off` is not committed now. It is written at deploy with the
  other pins, because committed pins stay the literal `UNSET` and
  `python/tests/test_v5_queue_scripts.py` enforces that: setting it failed 44 of those tests. The
  drop itself is recorded in both pre-registrations (v5-build 3723bbf). Fable accepted this
  deviation from "apply all three edits now".
- **Ruling 5, the queue trim: skipped.**
  - There is no documented dequeue. j6a's waiter is already running. Its own run condition (the
    fsucc word) does not read its `.queued` marker, which only its dependents check.
  - j6a already carries the human's advance yes (`j6a-on-room-refusal-yes`).
- **How the record is applied.** The promotion verdict applies the record in
  `python/qd_train/ledger.py`. It does not rewrite the gates on rows: the ledger is
  append-only, and the rows keep their as-built measurements.
- **What the code will say about v4's `degenerate_head`.** The share-only rule reads a new
  structured metric, `degenerate_head.choice.kN.top_class_share`, which v4 rows do not carry.
  So under the verdict, v4's `degenerate_head` reads **not_run**, not pass. The "fail to pass"
  reading above was made by hand from the `degenerate_head.family.*` detail text, and the code
  does not parse text to reproduce it.

## The two facts returned to the human

1. **Who owns the uncommitted edits in main's checkout?** These are `tokenizer.rs`,
   `backend.rs`, `serve.rs`, `service.rs` and `Cargo.lock`, among others, all dirty since Oct 1
   10:38. The answer gates H1/M1/M2 and the qdm-digest-parallel merge.
2. **What did Apple Diagnostics report?** Or does the human release that hold? The v5 data build
   waits on it.
