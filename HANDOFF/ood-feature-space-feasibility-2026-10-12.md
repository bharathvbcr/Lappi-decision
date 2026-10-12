# HANDOFF OOD feature-space feasibility note, 2026-10-12

ft-a4ef800cc7a item 5; owner of `GAP-OOD-ABSTAIN-HAS-NO-FEATURE-SPACE-SCORE-2026-10-06` is v6
planning. This is a **scoping note**: nothing was run, nothing was extracted, no serving or gate
change is proposed or made (rule 2; the serving contract is the human's, audit section 7 item 5).
Source for the design: `AUDIT/training-audit-2026-10-06.md` section 4.1 and section 6 item 3.

Labels: **[V]** read at the named file this session; **[I]** inferred; **[U]** unknown.

## The question

Every abstain lever tried so far (noul weight, fitted margin) is an output-space score. Measured
[V, `gaps.jsonl` `GAP-OOD-CLINC-NOUL-DOES-NOT-TRANSFER-TO-CODE-SHAPED-OOD`, margin probe row
19d28ec3]: scrambled-code margins are at least 0.972 and unseen-language at least 0.995 against an
in-distribution p10 of 0.9996, so no margin threshold separates them. v5's OOD half is 149/180 for
the average (`HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md`), gate needs a Wilson lower
bound of 0.9. **Does a score computed on the model's hidden state separate those inputs where the
output does not?**

## Candidate features

All read from the frozen tower, no training, the gate's own request shape.

1. **Position:** the hidden state `QwenDecisionStep.hidden(batch)` (post-final-norm
   `last_hidden_state`, `python/qd_train/backbone.py:1212-1243`) at the token the choice head reads
   [I: the last prompt token for the letter slot; the exact index is the first thing the extract
   script must read out of the batch plan, not assume]. Width 2048 [V, span head shape in
   `HANDOFF/needle-span-mac-2026-10-06.md`], 8 KiB per row at fp32.
2. **Layer:** final-norm output first. Whether an intermediate layer or a layer mix separates
   better is **[U]**: there is no OOD literature on Gated DeltaNet hybrids (audit 4.1), and a mix
   is the X-Mahalanobis idea [audit 4.1, NeurIPS'25]. The note proposes two layers (final, and a
   middle full-attention layer) fixed in the pre-registration, not chosen by separation on the
   data (see selection leak below).
3. **Detectors**, ID-fitted only: Relative Mahalanobis (class-free: a background Gaussian is
   subtracted) and kNN-cosine on L2-normalised states (Ren et al. arXiv 2209.15558; Sun et al.
   arXiv 2204.06507, both cited in audit 4.1). Both are plain linear algebra on a few tens of
   thousands of 2048-vectors.
4. **Thresholding:** conformal p-values on a held-aside ID calibration split, so the
   in-distribution false-refusal rate is bounded by construction (Bates et al. arXiv 2104.08279,
   audit 4.1). The calibration split is not the 180-row gate population and not what training
   read.

## Eval set (the selection-leak rule)

The detector, layer and position must **not** be chosen by separation on the 180 gate rows
(`python/qd_train/ood.py`, 60 each of prose, unseen-language, scrambled): that is selection on
the gate population. So:

- **ID fit set:** a sample of the v5 train shard set (`/Users/bharath/qd-campaign/phase4-v5r-2026-10-03/shards/train`),
  held to the rows the tower trained on only in the sense that the fit reads features, not labels.
  [I] 5,000-10,000 choice rows are enough to fit a 2048-d Gaussian and a kNN index.
- **ID calibration split:** a disjoint slice of that shard set, or of val choice rows (the shipped
  calibration table is fitted on val with `--population all`, so a val-based split must be
  separated from it by fold; the existing two-fold split rule in
  `calib-two-fold/report.json` is `sha256(split_key || 0x00 || row_id)[0]` even).
- **Dev OOD set (new, not the gate's):** code-shaped OOD built by `qd-mutate` from train repos
  plus prose and unseen-language drawn from sources disjoint from `ood.py`'s (needle templates,
  `_UNSEEN` languages, the general record's val split are all spoken for). A few hundred rows per
  category is the target [I]. Building it is the main unbuilt piece.
- **Gate read, once, at the end:** the 180 gate rows are read after the pre-registration fixes
  detector, layer, position and threshold rule, and reported next to the existing output-space
  numbers, **report-only**. Held-out families are never read (rule 3).

Pre-register in `campaign/` before the gate rows are read (audit 6 item 3).

## Cost

- **Extraction (the heavy part):** a forward over each row at fp32 on the Mac. The existing span
  extract is the nearest measured reference: [R, reconciliation of ft-a4ef800cc7a] it ran more
  than 10 h of wall for the span arm's train set and reached 48 GiB footprint, which is what the
  `mac_heavy.sh` Metal-aware cap (ft-0eb828f178f) exists for. At that rate [I] 10,000 ID rows plus
  17,254 val rows plus ~1,000 dev OOD rows is of the order of several hours under
  `tools/mac_heavy.sh`, one lock. The 180 gate rows are negligible. Storage: 28 k rows x 8 KiB per
  layer = about 230 MB.
- **Fitting and scoring:** seconds to minutes on CPU (Mahalanobis: one 2048x2048 covariance; kNN:
  brute force over about 10 k vectors per query is fine).
- **Reuse:** `build/needle-span-mac-2026-10-06/span_refit_extract.py` already runs a tower over a
  shard set into a resumable cache and reuses `_real_step`, `ShardReader` and `assemble_batch`; it
  caches span-row states only, so the letter-slot state needs a small sibling extractor, not a new
  harness. No new dependency.
- **Wall-clock and memory cap, per `CLAUDE.md` rule 4:** a single-GPU-equivalent Mac job, no cloud
  spend; still run under the lock with a recorded cap.

## What a positive result would still need (plumbing, not done)

The score only moves a gate if it is computed everywhere the gate is:

1. Eval: the Python path (`tools/real_ft_run.py` `_decode` and `--ood`) must emit the state (or
   the score) per row, and `python/qd_train/ood.py` must count it.
2. Serving: `qd-runtime` and `qd-metal` must produce the same state and the ID statistics
   (mean, covariance, calibration quantile) must ship in the calibration table next to the
   existing entries. qd-metal has no hidden-state output today [V, `HANDOFF/needle-span-mac-2026-10-06.md`
   section 2]; that is the span-head loader's problem too (board-sweep wave 2).
3. Contract: a new abstain reason is a serving-contract change and the human's to approve.

Until all three exist, the feature-space score is a measurement, not an abstention.

## Open

- Which layer and position: [U], decided by pre-registration, not by data.
- Whether the Mac's fp32 states match what qd-metal would produce in serving: [U].
- The dev OOD set does not exist yet.

First command for the next lane: read `python/qd_train/backbone.py:1212-1243` and
`build/needle-span-mac-2026-10-06/span_refit_extract.py`, then draft
`campaign/ood-feature-space-preregistered.json` before any extraction.
