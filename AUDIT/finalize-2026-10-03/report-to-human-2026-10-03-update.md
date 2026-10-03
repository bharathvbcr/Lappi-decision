# To the human: update, 2026-10-03 ~15:40Z

This follows `report-to-human-2026-10-03.md`. The items are in Fable's order: the six that need
your answer first, then what was done.

## Needs your answer

1. **What did Apple Diagnostics report?** I have asked three times. The v5 data build stays held
   until you answer or release the hold. Apart from that, the build waits only on my review of the
   bench's revised data patch and on the bench's domain-key change.

2. **Who owns main's uncommitted edits?** These 15 paths have been dirty in the main checkout
   since Oct 1, and none of the changes is mine:
   - `CLAUDE.md`, `Cargo.lock`, `README.md`;
   - `crates/qd-metal/src/{backend.rs,bin/parity.rs,tokenizer.rs}` and the untracked
     `crates/qd-metal/tests/fixtures/`;
   - `crates/qd-preflight/src/tristate.rs`;
   - `crates/qd-runtime/src/{serve.rs,service.rs}` and
     `crates/qd-runtime/tests/{common/mod.rs,lifecycle.rs}`;
   - `docs/{lappi.md,schedule-2026-09-28.md,train-plan-2026-09-28.md}`.

   Your answer gates H1, M1, M2 and the qdm-digest-parallel merge.

3. **The ship gate needs 300 hand-labelled held-out diffs, and they don't exist.**
   - `docs/train-plan-2026-09-28.md:267` at HEAD records this. Only you can label them.
   - The v6 draft fixes where they come from: only from held-out repos, and never touching the
     1,784 files whose prose is already in v5's data.

4. **Should the runtime refuse task ids that no release trains?**
   - **The doc bug, fixed.** The request documented for callers in `docs/schema-api.md` was not
     one the model was trained on: its task, question and slot names all differed from training.
     It is fixed on v5-build (0810487). A test now requires the doc's example to equal the
     request training builds.
   - **What remains is yours.** The runtime answers any task id, trained or not. An untrained id
     gets a confident-looking answer from a prompt shape the model never saw, which is the
     hallucination failure in product form.
   - **The proposal:** refuse any task id absent from the release's trained families, with a
     typed refusal that names those families. It changes what the product accepts, so it is your
     call (GAP-RUNTIME-ADMITS-TASKS-NO-RELEASE-FAMILY-TRAINS-2026-10-03).

5. **Do you want a clean Decision Index entry?**
   - The panel scores CLINC150's test split, and those items are Lappi training data. So
     Lappi's CLINC score on the index is not a reportable claim. CLINC is now marked
     non-reportable.
   - MMLU's test items are also training data. MMLU sits in the frozen panel, which is left out
     of the index, so it does not move Lappi's index score.
   - A clean entry means dropping CLINC's test-split rows from training. That is a v6 data-scope
     option, not a v5 change
     (GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03).

6. **May I download the ARC test splits?** They are used only to check that no ARC train row
   repeats one of the panel's ARC test items, and they are never trained on.
   - Source: `allenai/ai2_arc` at revision `210d026f`, licence cc-by-sa-4.0, sizes from the HF
     API listing.
   - `ARC-Challenge/test-00000-of-00001.parquet`: 203,808 B.
   - `ARC-Easy/test-00000-of-00001.parquet`: 346,257 B.

## Done

7. **The five approved downloads are in and verified.** Each one's size and sha256 match
   `/Users/bharath/qd-campaign/v5-ccbysa-panel-2026-10-03/fetch-record.jsonl`: methodology.json,
   BoolQ train, both ARC train files, and VitaminC train.

8. **More weight toward your use cases.** Weight can't fix zero. Of your callers' decisions, only
   DevCouncil's verdict (`code.defect_class`) has training rows: 84% of v4's tokens.
   - **v5:** the cap table is unchanged. A per-domain stratum key makes Open-Jev's dev-tool
     strata (shell history, browser tools, release and migration) measurable on val.
   - **v6:** your callers' untrained decisions are pre-registered as a draft
     (`campaign/v6-caller-families.DRAFT.json`, d10bda0). Fable reviewed it.
     - DevCouncil relevance (file, line range, file role), GitPulse commit type, scope and
       breaking change, and `code.commit_intent` are built by construction from your own repos'
       history.
     - Severity and DevType routing have no label source yet. The options are in the draft for
       Fable to rule on.
   - The ruling is recorded at `AUDIT/finalize-2026-10-03/fable-weighting-ruling.md`.
   - **A correction:** an earlier justification said DevType calls the code decision. It does
     not. DevType's decision is routing, and routing is untrained. The code-only promotion
     decision stands on its other two reasons (380f049, c55b53d).

9. **No caller logs real queries, and no caller calls Lappi yet.** "Your use cases" in the data
   are the plan's and the callers' code, not measured traffic. If DevCouncil, DevType or GitPulse
   can log queries, those logs become the held-out evaluation set. The README already reserves
   real DevType queries for that.

## State

- **The box** is busy at 93% GPU, running J5′, which ends ~23:30Z. Its monitor is re-armed.
- **v5 launch** still needs your go on the final plan and timeline.
