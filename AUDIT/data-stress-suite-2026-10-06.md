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

## Clarifications (dated, recorded before the suite's run)

These were added on 2026-10-06 at about 20:30Z, after the tests were written and before any item ran on main. Each one interprets wording; none removes an item or moves a bound. The rulings are Fable's.

- **S1, "refused or counted".** A torn last line, CRLF endings and blank lines are counted, not refused. Every input `for_lines` reads is sha256-pinned, so refusing one of these would make a pinned, verified input unreadable for good: the only way to load it again would change its bytes and break the pin.
  - The count lands in the producer's manifest `inputs` map as `line_facts/<key>` (`decisions::record_input`). A clean file adds no entry. *Superseded at ~21:30Z, below: the entry is always written.*
  - A BOM fails the JSON parse and is refused with the line number.
  - An escaped NUL (`\u0000` in a string) is refused with the line number. *Superseded at ~21:30Z, below.*
  - A repeated target id is refused by `pool::read_targets` with both line numbers.
  - Not covered: the Python pool reader `qd_data.decisions._read_lines`.
- **S1 scoped by role (2026-10-06 ~21:30Z, the advisor's ruling).** The first S1 ruling did not say *whose* rows it governed, and the real views showed that it matters. Runner 6's conversions over them stopped on:
  - escaped NULs in source views: scirepeval search train, and llmail-inject phase 1 and 2. In llmail, 40 lines carry the `\u0000` escape, and 39 of them are real NUL rows (36 and 3, as measured by runner 6c); the other is a literal backslash in text;
  - a csn go train row of 9,945,948 bytes;
  - a SWE-bench **target** row of 117,859,127 bytes. This is `explosion__spaCy-1502`, whose `patch` field is a legitimate 102,783,478-byte diff.

  Escaped NULs also sit in target views: 2 in SWE-bench train and 1 in csn python test. `decisions::for_lines` now takes a role.
  - **Source** rows are rows that can become training examples. An escaped NUL, or a line past `MAX_LINE_BYTES` (8 MiB, unchanged), drops **that row**. The drop is counted per cause, with its first line numbers, in `line_facts/<key>`. The file is never refused for it, and nothing is dropped silently.
  - **Reference** rows are decontamination targets, held-out id sets, and the caches and decider files every row must be read from. They are **never dropped**: dropping one would shrink decontamination coverage, which is rule 2 in spirit.
    - An escaped NUL is accepted and counted. Strings do not truncate at NUL, and no surviving training row holds one, so a target n-gram spanning a NUL only over-covers.
    - A line past `MAX_REFERENCE_LINE_BYTES` (256 MiB, a finite bound of its own, sized from that 117,859,127-byte row) refuses the file.
  - `line_facts/<key>` is written for **every** input, clean or not. A count of zero then reads differently from an input that was never examined.

  Tests rewritten to this scope:
  - S1 "escaped NUL": a source row is dropped and counted; a target row is accepted and counted.
  - S1 "line bound": a source row past the bound is dropped and counted; a target row past `MAX_LINE_BYTES` is read.

  Both still fail on 22c38b6, which refused the over-bound target line and passed NULs on silently. The red check was rerun on them. A known loss: the 39 llmail-inject rows whose payloads hold NULs are an attack class, and dropping them is recorded in a gap, not only counted. *Superseded for the injection corpus on 2026-10-08, below.*
- **NUL payloads in the injection corpus (2026-10-08, Fable's data-recipe ruling, closing GAP-LLMAIL-INJECT-NUL-PAYLOAD-ROWS-DROPPED-2026-10-06).** The 39 llmail rows (36 in phase 1, 3 in phase 2) were read. Every NUL sits in the `text` field, and the NUL is the attack itself: a string terminator before a forged JSON close (`confirmation\x00"}]`), a separator before forged chat-template tokens (`\x00<|end|><|system|>`), or a break inside a subject or address (`System Update\x00.`). Dropping them removes the only examples of that technique.
  - **Ruling: encode.** `qd-prep convert injections` reads its sources as `Role::SourceNulEncoded`. A row holding a NUL is kept, and every U+0000 in it is written as U+2400 SYMBOL FOR NULL (`␀`, `decisions::NUL_PLACEHOLDER`), Unicode's visible symbol for NUL. The row is counted in `line_facts/<key>` as `nul_rows=N encoded lines [...]`, so the manifest still names every such line.
  - **Why U+2400 and not the six characters `\u0000`.** The literal text cannot be told apart from the one llmail row that already holds a backslash and `u0000` as text. `␀` does not occur in any llmail, deepset or gandalf view (an `rg -uu -F '␀'` search over the three on 2026-10-08 found none). U+2400 is not a `\w` character, so the containment scan splits words at it exactly as it splits them at a NUL.
  - **What still holds.** No training row holds a NUL. Every other source still drops a NUL row and counts it. That includes the 12 scirepeval search train rows: a NUL in an abstract is noise, not signal. References still accept a NUL and count it.
- **S3, the `MAX_N` fixture (2026-10-08, after the green run of the planted-copy control, `build/nulctl-red/green.log`).** `pool::decontaminate` now runs a planted-copy control per target set and refuses a set with no row of at least n words (GAP-SYNTH-POOLS-ZERO-EXCLUSION-NOT-POSITIVE-CONTROLLED-2026-10-06). That is this table's "`ngram_n` larger than every row: refused or counted", taking its first branch. `s3_ngram_n_zero_and_past_max_n_are_refused` also asserted that `n = MAX_N` passes, on 20-word rows: a row longer than every n is the other S3 case, so the assertion was refused for that reason, not for the bound. Its fixture at `MAX_N` now holds `MAX_N + 8` words per row. The assertion is unchanged, and 0 and `MAX_N + 1` are still refused on the 20-word rows.
- **S4, "refused before any row is drafted".** `synth::generate` drafts one row per template to count the templates; for the tools kind, those come from its data files. It refuses when templates × `rows_per_template` passes `MAX_ROWS`, before the pool's rows are drafted. That probe is 1/`rows_per_template` of the pool, and it is the only drafting before the refusal.
- **S5, the `/dev/zero` symlink.** It is refused as "not a regular file" (`files::open_regular`) before any size bound is consulted. That is stricter than "refused by the size bound", and it is never read.
- **S11, "every producer".** The producers of training rows are the three subcommands that write pools: `synth`, `convert` and `decisions`. Two other subcommands are not producers, and a held-out path given to either is not refused as held-out:
  - `dedupe` is a kernel over a request that Python builds. Deduplicating a held-out set is legitimate.
  - `own-repos` reads repositories under a scan root and *makes* the held-out split.

  `--target` files of the three producers are accepted under a held-out marker by design: decontaminating against held-out sets is how they stay out of training.

## What this suite does not cover

- GPU paths, training steps and serving are other sessions' suites.
- Network fetches (`fetch_v6.py`, the GitHub licence lookups) are not re-run; their records
  are pinned by sha256.
- Semantic label quality: whether a rule-built gold is the right answer. That is measured by
  held-out evaluation, not by stress tests.
