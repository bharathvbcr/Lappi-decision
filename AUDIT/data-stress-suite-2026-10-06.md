# Data-side stress suite (pre-registered 2026-10-06, before any item ran)

Owner: the data-clean-v6 session. Scope, as agreed with the training-plan session (cd80fb) and the
integration session (32beaa): `crates/qd-prep`, `crates/qd-mutate`, `python/qd_data`, `data/synth`,
`data/convert` and the v6 pools. Training-side and runtime suites belong to those sessions.

## Stopping rule

"Until complete confidence" needs a stopping rule, so here it is (Fable's ruling, 2026-10-06):

1. This list and its pass criteria are fixed before any item runs. Adding an item later is
   allowed, and the addition is dated. Removing an item, or loosening a criterion, is not
   (rule 2).
2. The suite runs **once**, under `tools/mac_heavy.sh`, on main after the four lanes merge.
3. Each item is reported tri-state: `passed`, `failed` (with output), or `not_run` (with the
   reason). A `not_run` is never counted as a pass.
4. A failure gets a fix and a test that fails against the pre-fix code. Then the **whole** suite
   runs once more. The suite is done when one complete run has no `failed` items. Every
   `not_run` stays listed with its reason.
5. Not allowed:
   - mutation testing (the Mac rule from 2026-10-02);
   - a new fuzzing crate without the human's yes (proptest and cargo-fuzz are new
     dependencies);
   - moving a bound, threshold or held-out set to make an item pass.
   Hypothesis, which the Python suite already uses, is allowed.

## Universal pass criteria (every item)

- **U1.** No panic. A Rust panic exits with 101, and that is a failure even when the input was
  garbage. Refusals are typed errors, exit non-zero, and name the bound or field.
- **U2.** A refusal writes nothing. There is no `out_dir`, or no file under it, and no
  `.partial` file is left behind.
- **U3.** The process RSS stays under the `mac_heavy` cap of 32 GiB. Every item also has a
  stated wall-clock bound.
- **U4.** Determinism. The same inputs and seed give byte-identical `examples.jsonl` and
  `manifest.json`, at any thread count.
- **U5.** No silent repair. An input the code alters to make acceptable must be counted under a
  named reason in the manifest.

## Items

| # | Target | Inputs | Expected |
|---|---|---|---|
| S1 | JSONL readers (`decisions::for_lines` and every pool or target reader) | empty file; torn last line (no `\n`); CRLF endings; UTF-8 BOM; non-UTF-8 bytes; NUL inside a string; a line of exactly `MAX_LINE_BYTES` and one of `+1`; JSON nested 200 deep; a duplicate `id` | Refused with the line number. CRLF, the BOM and the torn last line are each either refused or counted, never silently accepted. The exact-bound line passes; `+1` is refused. |
| S2 | Option and slot bounds (`decisions`, `synth::assemble`, `pool::examples`) | 0, 1, `MAX_OPTIONS` (16) and 17 options; duplicate options; an option of `MAX_OPTION_BYTES` and `+1`; a question at `MAX_QUESTION_BYTES` and `+1`; a context at `MAX_CONTEXT_BYTES` and `+1`; gold out of range; `noul` gold on a family that states `noul` as zero | Refused at `+1`, passed at the bound. A duplicate option is refused. |
| S3 | Containment (`containment`, `pool::decontaminate`) | a target with an unassigned code point (the recorded gap); an empty target set; threshold 0, 1 and NaN; `ngram_n` larger than every row; `ngram_n` 0 and `MAX_N + 1`; candidates where nothing survives | An empty target set is refused. NaN and out-of-range values are refused. Nothing surviving is refused. The unassigned code point is refused loudly, never skipped silently (the current behaviour, pinned until its gap is fixed). |
| S4 | `qd-synth-config/v1` | each required field missing; wrong types; `rows_per_template` 0 and over `MAX_ROWS_PER_TEMPLATE`; total over `MAX_ROWS`; `val_templates_per_class` ≥ templates; probe `dim` and `max_iter` over their bounds; a target pinned but not given, given but not pinned, or given twice; a sha mismatch on a target, catalog or phrasings file | Refused before any row is drafted (U2). |
| S5 | Tools data (`synth_tools`) | an empty catalog; a one-member group; repeated tool names in one app; a shared tool whose required params differ (the shipped case); duplicate phrasings, including case variants; an underspecified phrasing naming a parameter no catalog requires; an overlap median exactly at the bound and just over it; a catalog over `MAX_FILE_BYTES`; a catalog path that is a directory, missing, or a symlink to `/dev/zero` | Each refused with its reason. The exact bound passes, and just over is refused. The `/dev/zero` symlink is refused by the size bound, never read without end. |
| S6 | Output directory (`decisions::write_out`, `synth::run`, `convert`) | `out_dir` exists; `out_dir` is a file; the parent is missing; the parent is read-only; a kill during the write (SIGTERM after the first file) | Refused before writing. After a kill, the pool is either absent or marked incomplete; a reader never sees a half pool as complete. |
| S7 | Concurrency | two `qd-prep synth` runs to one `out_dir` at once; the lane-4 race shape (two tests sharing one temp input path), found statically across `crates/qd-prep/tests` and `python/tests` | One run refuses. No test shares a temp path keyed only on pid. |
| S8 | Determinism | each synth kind built twice with the same seed, at `--threads 1` and `--threads 8`; the decisions pool built twice | Byte-identical outputs (U4). |
| S9 | Scale, bounded | synth email at 100k rows; containment with 200k candidates × 50k targets; LSH at 1M keys | RSS and wall clock are recorded; each finishes under its stated bound (email ≤ 10 min, containment ≤ 20 min, LSH ≤ 10 min) and under U3. |
| S10 | Python loader door (`qd_data.decisions.load_decision_pool`) | a synth or convert pool whose family is unregistered; an unregistered licence id; a manifest whose sha256 of `examples.jsonl` does not match; a pool under a `heldout/` path | Each refused. A held-out path is refused by the path marker. |
| S11 | Held-out invariants (rule 3) | `heldout/natural-bugs/`, `heldout/tssb-3m/` and the caller records offered to every producer's input flags | Every producer refuses, or never reads them. `qd-train`'s path check refuses them. |
| S12 | Licence default-deny | a row with an unknown licence string; aliases differing only in case or whitespace; `NOASSERTION`; an empty licence string | Unknown is denied and counted. Aliases resolve or deny, never pass unclassified. |

## What this suite does not cover

- GPU paths, training steps and serving are other sessions' suites.
- Network fetches (`fetch_v6.py`, the GitHub licence lookups) are not re-run; their records
  are pinned by sha256.
- Semantic label quality: whether a rule-built gold is the right answer. That is measured by
  held-out evaluation, not by stress tests.
