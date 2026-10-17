# Fable ruling: the synth leak-probe gate and `devtype.route` (2026-10-12)

Asked by the lead after the Opus verification of lane 6 (scratchpad `lanes/verify-lane6.md`, item
D6). Labels: [V] verified by me against the file or output cited; [R] read in a document I did not
re-verify; [I] inferred; [U] unverified.

## 1. What happened

- The gate [V]: `crates/qd-prep/src/synth.rs` runs a linear control (char 3-5-grams of the
  rendered context, fitted on train templates, scored on val templates) on every fixed-option
  synthetic family and refuses the build when its val accuracy is over the config's
  `probe.max_val_accuracy` (`synth.rs:624-732` the probe, `:811-834, :861-867` the refusal). The
  module doc (`synth.rs:22-26`) names its purpose: "The 0.1 diff-marker free pass was a format
  leak", so each family gets a control that reads the label off the context alone. The bound
  0.95 is written in each `data/synth/*.json` config before the pool is built (`synth.rs:109-110`).
- Introduced [V]: commit 110de0c (2026-10-06, "synthetic decision pools ... and the shared pool
  seam") by the data-clean-v6 lane, under plan item 8 of `HANDOFF/data-clean-v6-plan-2026-10-06.md`
  ("linfit per family as a format-leak detector ... a synthetic family with the label in the
  option text -> linfit ~ 1.0, flagged"; the 0.95 bound at `:431`). No separate ruling named the
  bound; it became binding on `devtype.route` through my R5, which placed the family under "the
  synth.rs rules" (`AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md:155`).
- The measurement [V]: `build/lane6/pool/route/manifest.json` `leak_probe.devtype.route`:
  `val_accuracy 1.0` (149/149 val rows, 1,465 train rows, majority share 0.403, converged in 393
  iterations, `bound 0.95`, `passed null`). No pool is pinned and no applied pool was written
  (`data/decisions/v6-allocation.json:8`), so nothing downstream has read this build.
- The exemption [V]: commit 24dadde added `LEAK_PROBE_REPORT_ONLY` (`synth.rs:613-622`), which
  rewrites `passed` to `null` and adds `report_only` for `devtype.route` after the probe ran; the
  refusal keys on `passed == false` (`:833`), so `null` builds. The lane wrote the test asserting the
  exemption (`crates/qd-prep/tests/caller_families.rs:251-253`) and closed the gap it opened,
  `GAP-ROUTE-LEAK-PROBE-IS-THE-TASK-2026-10-12` (`gaps.jsonl:1326`), with its own reasoning as the
  evidence. No ruling was asked for.
- Readers of the verdict [V, `rg -uu` over the repo minus `build/`, `target/`, `.git/`]: the only
  consumer of `leak_probe.passed` is the build refusal itself and the test above. Nothing in
  `python/qd_data`, `python/qd_train`, `allocate.rs`, `pool.rs` or `tools/` reads it.
- No test exercises the build-level refusal [V]: `the_probe_reads_a_planted_label_leak`
  (`synth.rs:1443-1461`) calls `leak_probe` directly and asserts `passed == false`; nothing asserts
  that `build()` refuses at `:861`. The `null` path therefore has no test on either side.

## 2. Is the lane's reasoning right?

Half right, and the half that is wrong is the half that matters for precedent.

- Right [V]: under my ruling 2i the gold is "the tool the deterministic resolver would run"
  (`AUDIT/board-sweep-2026-10-12/fable-board-sweep-ruling.md:313-330`), and the generator
  re-derives every row's gold from the query by a committed mirror (`synth_tools.rs:907-927`
  `route_gold`; a class/rule disagreement refuses the build). The 60 templates
  (`data/synth/devtype-route-phrasings-v6-2026-10-12.json`) make the label a lexical fact: every
  `runMacro` template contains the literal "macro", every `findSnippet` template the literal
  "snippet", every `resolveDate` template a production of a closed grammar over a tiny alphabet,
  and the two `none` classes contain none of these. A char 3-5-gram control reading 1.0 on
  held-out *templates* is therefore not an artefact of the generator; it is the gold rule.
- Wrong [V]: the stated criterion, "the label is a deterministic function of the query text, so
  a shallow probe solving it is expected", does not distinguish this family from any other.
  Every synthetic family in `synth.rs` is labelled by rule — that is what `synthetic-by-rule`
  means (`synth.rs:50-53`). `jarvis.field_fill` and `jarvis.step_risk` are deterministic functions
  of their generator's inputs and read 0.334 (majority 0.067) and 0.0 (majority 0.2)
  (`build/nulctl-red/real/synth-jarvis/manifest.json`), because their template-disjoint split hides
  the label's surface. What makes `devtype.route` different is not determinism but that the
  label-bearing surface is *shared across templates* by the gold rule's definition, so the
  template-disjoint split holds out wording and not the label. Taken as written, the lane's
  criterion would exempt any future family that fails the probe, which is exactly the door rule 2
  exists to keep shut.
- Also wrong: the form. An agent-authored constant inside the gate, plus a test that asserts the
  exemption as expected behaviour, plus a self-closed gap, is a threshold moved by the agent who
  tripped it. Rule 2 forbids that regardless of whether the reasoning would have been accepted.
  The verifier's D6 is upheld.
- What the probe is telling us, read honestly [I]: on this family the val set, and the
  `route-heldout` set built the same way, cannot distinguish a 2B model from a 3-5-gram lookup.
  Any model reading on `devtype.route` in v6 is evidence of nothing beyond that lookup. The
  measurement that would mean something is the one contract §5 reserves: DevType's caller
  records, where Lappi's route is compared to the on-device router's actual tool call
  (`LappiPaletteRouting.swift:20` `.agrees`/`.differs`, admission-only) — never trained, and
  none exist yet.

## 3. Ruling

### 3.1 The exemption is granted by this ruling, retroactively, and only on these terms

**(a) Authority.** Report-only status for a family's leak probe exists only by a ruling that names
the family. The lane's self-authorisation on 24dadde is a rule-2 breach in process; the measured
numbers it produced (1.0 on 149 val rows) stand as the reason this ruling exists. Nothing from
that build is pinned, so nothing is invalidated beyond the manifest, which is rebuilt under (b).

**(b) Two keys, or the build refuses.** A family is report-only iff *both*:
1. the code list (`LEAK_PROBE_REPORT_ONLY`, keep the name) names it, and each entry carries the
   ruling's repo path and date — this entry:
   `AUDIT/board-sweep-2026-10-12/fable-leak-probe-devtype-route-ruling.md`, 2026-10-12; and
2. the family's synth config (`data/synth/route-v6-2026-10-12.json`) carries
   `probe.report_only.<family>.ruling` with the same path, recorded before the build like the bound
   itself (`synth.rs:109-110`), and the path exists in the repo at build time.

One key without the other is a build refusal with a message naming the missing key. The manifest
records `leak_probe.<family>.report_only = {ruling, criterion: "surface-rule gold"}` beside the
measured accuracy. The config sha changes, so this is a new build; the lane rebuilds and reports
the new manifest's numbers.

**(c) The criterion, stated so it cannot be reused loosely.** Report-only is available only to a
*surface-rule family*: one whose gold is, by a ruling, computed from the context bytes alone by a
committed mirror of the caller's own deterministic resolver, which the generator runs on every row
and refuses on disagreement (today: `route_gold`). "The label is a deterministic function of the
generator's inputs" does not qualify — every synthetic family satisfies it. The test an agent must
be able to point at: a function in the crate that takes the rendered context and returns the gold,
with a build refusal when any row's class disagrees. `email.*` and `jarvis.*` have no such function
and stay gated at 0.95. Adding a second family to the list is a ruling, not a card.

**(d) `null` is not `true`, anywhere.** `passed: null` means "measured, not a pass". No reader —
loader, allocation, reading rule, dashboard — may treat it as a pass; a change that does is refused.
Today no reader keys on it [V]; this clause binds future ones.

**(e) Replacement checks.** A positive control of *this* probe on *this* family is uninformative:
the baseline is already at the ceiling, so a planted leak cannot move it. I do not order one. The
probe's positive control remains `the_probe_reads_a_planted_label_leak` on an unlisted family.
What replaces the gate for this family is:
1. **A negative control on the gate itself** (fails against 24dadde): a build of the route pool
   with the config key from (b)(2) absent must refuse; and a `build()`-level test that a family
   *not* in the list with a probe over the bound refuses at `synth.rs:861` (this path has no test
   today [V]). Both are the "test that fails against the pre-fix code" for this change.
2. **Both numbers in every reading.** Wherever a v6 reading reports `devtype.route` (val, the
   `route-heldout` target), it reports the linear control's accuracy on the same rows beside the
   model's, and labels the family `surface-rule (probe 1.0)`. R8 already makes the new families
   report-only; this clause says what the report must carry. The v6 pre-registration
   (`campaign/v6-preregistered.json`, R9 step 7) copies the probe value into the family's entry
   when it is written. No claim about the model's capability may cite this family's number alone.
3. **The informative measurement is the caller record**, contract §5: `lappiNote` agrees/differs
   against DevType's on-device router, held out, never trained, none existing yet [V]. The
   family's status in `docs/promotion-decisions.json` stays whatever the human has recorded; this
   ruling adds nothing to any promotion population.

**(f) Construction stays as ruled, with one app-fidelity fix.** No near-miss `none` rows ("what
is a macro", "remind me next friday to call bob"): under 2i's substring rule they would be labelled
by the mirror where the mirror and the app's router most plausibly disagree, which is label noise
in the direction of the app, not a harder family. The bound 0.95 does not move. The family is not
dropped: the human ruled on 2026-10-08 that caller families are not deferred, and R5's deadline
rule governs its entry. One fix, consistent with 2i's "what the app sends" (the generator already
enforces the 512-char cap): queries the app never sends because they are under
`PaletteToolRouter.shouldAttemptRouting`'s floor (`PaletteToolRouter.swift:60-63` at DevType
6cba355: trimmed count < 4, or a leading `=`/`>`) are refused and counted by the generator, floor
pinned from the DevType commit in the phrasings file's provenance. The 2026-10-12 pool holds 2 such
rows (`-3w`, `-8d`) [V]; the resolveDate count drops and the lane reports it.

**(g) The gap record.** `GAP-ROUTE-LEAK-PROBE-IS-THE-TASK-2026-10-12` was closed by its opener on
its own reasoning. Append one line (`O_APPEND`, never read-modify-write) with the same id,
`status: resolved`, `resolved_by: this ruling's path`, and the rebuilt manifest's probe numbers as
evidence. The 1326 line stays as history.

### 3.2 Would the same reasoning silently exempt other families, and how is that prevented?

Yes, as the lane wrote it. Every `synthetic-by-rule` family is "a deterministic function of its
inputs"; `email.*`, `jarvis.*` and any future `apps.*` family could claim it the day their probe
trips. Prevention is (b) and (c) above: a family enters report-only only through a named code
entry *and* a pre-recorded config key, both citing a ruling file that exists, under a criterion
that requires a committed context-to-gold function with a disagreement refusal. The two-key form
makes an exemption a visible diff in two places that review and `rg` find in one call; the
criterion makes "deterministic" insufficient on its face. A third defence is (d): since `null` is
never a pass, an exempted family buys itself only a build, never a gate population.

### 3.3 What this ruling does not decide (the human's)

1. **D8 fidelity** [V, from the verifier and `PaletteToolRouter.swift:150-158`]: DevType's
   `runMacro` is a built-in text operation (`upper`, `sortLines`, `base64Encode`, ...), not a
   user-named macro; the family's `runMacro` rows ("run the standup-notes macro") train a shape the
   app's tool does not serve. 2i stands unless the human vetoes; I flag that the family's
   `runMacro` class is the weakest link in 2i's "mirror of the app" claim, and that fixing it is a
   phrasings change (macro phrasings over the app's operation ids), not a gate change.
2. Whether a family that a 3-5-gram lookup solves is worth 1,500 of v6's rows at all. I keep R5's
   cap because the 2026-10-08 ruling keeps the family and the rows are cheap; the human may cut it.

## 4. Instruction for the implementing lane (ft-2d0a5e4b27e, synth half)

First command, from this worktree after `ListAgents` and `gitpulse_insights`:

```
cargo test -p qd-prep --release --test caller_families -- route 2>&1 | tail -20
```

Then, in one commit titled
`qd-prep synth: leak-probe report-only needs a ruling in code and config (Fable 2026-10-12)`:
1. `LEAK_PROBE_REPORT_ONLY` entries become `(family, ruling_path, date, why)`; the route entry cites
   this file. `Config::parse` reads `probe.report_only` (map family -> `{ruling}`); `build()` sets
   `passed: null` only when the code entry and the config key agree and the ruling path exists,
   else refuses naming the missing key. Manifest `report_only` carries `{ruling, criterion}`.
2. `data/synth/route-v6-2026-10-12.json` gains the config key; rebuild `build/lane6/pool/route`
   and report `val_accuracy`, `val_rows`, `train_rows`, the resolveDate count after (f), and the
   new config sha. Rows under the app's floor refused and counted (`under_app_floor`).
3. Tests, red on 24dadde: route build without the config key refuses; an unlisted family over the
   bound refuses at the `build()` level; a listed family with a missing ruling file refuses.
   `caller_families.rs:251-253` asserts `report_only.ruling` equals this path.
4. Append the gap line per (g). No other gate, bound, seed or set changes.

Report the manifest numbers as ledger-citable facts; the probe value is a number in a report only if
it cites the manifest sha.
