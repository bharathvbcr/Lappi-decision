# Fable's ruling on the pipeline, the delegated items and time-to-v5 (2026-10-03 ~15:50Z)

## The request

The human, 2026-10-03 ~15:41Z, in reply to `report-to-human-2026-10-03-update.md`:

> Download needed, ask Fable for the rest; make the training pipeline robust and purpose-built.
> Everyone are shipping models, I haven't even started reasonable training. Think hard and work
> hard and make the pipeline and data robust with tests or scripts and data splits and other
> innovative techniques to improve the pipeline. Ask fable.

- **"Download needed"** answers item 6. The two ARC test files were fetched at 15:42Z by
  `/Users/bharath/qd-campaign/v5-ccbysa-panel-2026-10-03/fetch-arc-test.sh`. Each one's size
  matched the HF API's, and both are recorded in `fetch-record.jsonl` with their sha256:
  - Challenge: 203,808 B, `62f03257…`;
  - Easy: 346,257 B, `4160597d…`.
- **"Ask Fable for the rest"** covers items 1–5. Fable sorted them by who can decide:
  - **Fable can rule:**
    - how the build proceeds without the Apple answer;
    - H1 on a branch;
    - item 4;
    - CLINC's version;
    - what the verifier checks;
    - the hardening order.
  - **Human yes still required, whatever Fable says:**
    - a second box;
    - any amendment to v5's pre-registered data or recipe;
    - any change to the ship gate's eval set (rule 2).

    Fable recommends; the lead relays.
  - **Facts only the human has:**
    - the Apple result;
    - who owns main's dirty files.

## The facts Fable ruled on (measured by the lead, read-only)

- **Time to v5 on the one GH200** [I, from the queue scripts' caps and F's cadence]:
  - Queue at 15:41Z: j5p (J5′) on seed 1 of 3, step 7,914 of 9,683.
  - Ahead of v5: nomask → j6f → tierb2 → j6dv4 → rung0 → cudadev → rungd → fsucc → j5pp → j6a →
    j6g → STOP.
  - About 56 GPU-h if fsucc is quiet, about 93 h if it fires (F′ ×3 and J5″ ×3).
  - So v5 seed 0 starts on this box around 2026-10-06 00:00Z at the earliest.
  - j6g, which decides v5's C1, is last in the queue.
- **Task-id admission** [V]: `crates/qd-runtime/src/admission.rs:54` (`admit`), called from
  `runtime.rs:341`. Both are clean on main.
  - `registry.rs` already has the typed refusal shape
    `RegisteredHeadMissing { task, available }`.
  - No release manifest records trained families today.

## Fable's ruling, verbatim

> **Rulings (within what a reviewer can decide):**
>
> 1. **Data build without the Apple answer.** The human was asked three times and answered "ask Fable" — treat that as the human declining to gate on it. Rule: one attempt on the Mac, under `mac_heavy.sh`, with the load gate read at start (109 peer sessions exist; the second panic was load ~30, not RSS). Record peak RSS and load during the build — both are in `amendments_pending` already. If the Mac panics a third time: hard stop, the build moves to box CPU, and that launch needs the human's yes. Put the box-CPU fallback in the report so the human can pre-approve it.
>
> 2. **H1 lands on a branch.** Apply in `h1-wt` (cut from HEAD, not from the dirty working tree), test there, commit there. Merge to main waits for the dirty-file owner. One cheap check you haven't done: `git diff` main's 15 dirty paths and grep the hunks for lane names, GAP ids or HANDOFF references — the content may name its owner without the human.
>
> 3. **The 300 diffs.** The gate stands as written; rule 2 binds Fable too. A by-construction set from held-out repos via `qd-mutate` is a report-only reading placed *beside* the gate, labelled as not-the-gate. The report to the human says plainly: the ship gate is unmet until they label or amend.
>
> 4. **Refuse untrained task ids: yes.** Release manifest gains `trained_families` (export writes it from the train manifest's family set); `admission::admit` refuses a task absent from it with a typed refusal mirroring `RegisteredHeadMissing { task, available }` — reuse that shape, don't add a second one. **Absent field → refuse** (a release that can't say what it trained can't admit anything). No caller is wired, so nothing breaks in production; the bench's v4 releases get the field by re-exporting the manifest, not retraining. Files are clean on main. Test fails against pre-fix code.
>
> 5. **CLINC → v6.** Your buckets overlap here: "CLINC in v5" *is* "a change to v5's pre-registered data," which you correctly put under the human. Resolve by ruling v6. Zero-cost prep now: write the CLINC test-split identity keys to a file so v6 drops them with one exclusion list.
>
> 6. **Verifier: extend, don't build.** `containment.rs` already takes N sets and N scans with an enforced flag, and writes attestation v2 with the splitter tri-states. The A7 count check is in the bench's v2 patch. So the only new thing is the **ARC test split as an enforced target set** in the containment request — a change to the Python request builder, Rust untouched. That is inside what the human approved by approving the download for that purpose; record it in `amendments_pending`. Add a one-line rsync guard refusing without a clean attestation sha. No new binary.
>
> 7. **Hallucination items split.** Label-noise audit as a *report* on v5's built data: yes, in the Mac lane after the build, feeds v6 exclusions. Hard negatives *added to train*: v6. Nothing else enters v5's data.
>
> 8. **Training canaries.** Before proposing NaN/grad-norm guards and checkpoint hashes, read `tools/real_ft_run.py` for what's already there — you haven't. Whatever is missing: a guard that fires must refuse with a ledger row, never skip silently; recipe-neutral when quiet; test included.
>
> **Recommendations the human decides (relay as recommendations):**
>
> - **A second single-GPU box for v5.** Yes. It's the only lever on "haven't started training" — v5 is otherwise behind 50–90 GPU-h of F diagnostics, with its gating item j6g *last* in the queue. Carry the real cost: ~$137 for the runs plus setup (base weights, env, v5-build tree — weights can rsync box-to-box rather than re-download, but say so explicitly since fetches need their yes). No v5 waiter has launched on the current box, so a second-box script is fresh, not an edit.
> - **Conditional defaults.** A second box can't wait for j6g. Recommend: all four conditionals taken as not-fired at launch, written into the DRAFT before launch as "undecided; taken as off; the queued rows are read afterward as report-only." That is an amendment — the human's yes, not Fable's.
>
> **Sequence:** bench v2 patch (live now; its full suite is *the* current Mac heavy job, so `mac_heavy.sh` will queue your cargo behind it) → your five-point review → ARC target set in the request builder → data build (ruling 1) → attestation + rsync guard + ledger row → item 4 in the lane while waiting on any of the above. Item 8's reads come first, cheap.
>
> **Housekeeping:** the monitor's output file is gone again; TaskStop b33t29mdw and re-arm now rather than at expiry.

## What this file does not change

- **No gate, threshold, seed rule or held-out path changes** (rules 2, 3).
- **The ship gate's 300 hand-labelled diffs stay unmet** until the human labels them or amends
  the gate.
- **These still need the human's yes:**
  - the second box;
  - the four conditional defaults;
  - a box-CPU build after a third Mac panic.
