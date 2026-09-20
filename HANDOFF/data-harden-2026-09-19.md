# HANDOFF — DATA-HARDEN lane · 2026-09-19

Three `gaps.jsonl` records were assigned: the closed language set, the bidi/Trojan Source question,
and the LSH recall bound. All three are answered, implemented and tested. Two of the three turned
out to be pointing at a different defect than their title suggested, and the bidi one could not be
fixed the way its record proposed — the reasons are below, each with the measurement behind it.

Scope written: `python/qd_data/{render,mixture,minhash,dedupe}.py`,
`python/tests/test_{render,mixture,minhash,dedupe_split}.py`, seven appended `gaps.jsonl` records,
and this file. Nothing under `crates/`, `docs/`, `python/qd_train/` or `python/qd_label/` was
touched.

---

## What was measured

Every number here is from a run, with the command that produced it. Anything not run is named as
not run and says why.

| Check | Result | Command |
| --- | --- | --- |
| The four lane suites | **177 passed**, 2.83 s | `.venv/bin/python -m pytest python/tests/test_mixture.py python/tests/test_render.py python/tests/test_minhash.py python/tests/test_dedupe_split.py -p no:cacheprovider --no-header` |
| Lane suites + every downstream consumer | **332 passed, 1 skipped**, 15.9 s | the four above plus `test_manifest.py test_data_access.py test_shards.py test_loaders.py test_admission.py` |
| Whole Python suite | **1 failed, 922 passed, 4 skipped** | `.venv/bin/python -m pytest python/tests/ -p no:cacheprovider --no-header` |
| Whole suite, excluding the `qd_wire` lane in flight | **714 passed, 4 skipped** | as above with `--ignore` on the five `test_wire_*.py` files |
| Mutation sweep over the three fixes | **13 of 13 mutations killed**; every file restored byte-for-byte | scratchpad harness, table below |
| Derived vs measured LSH recall | derived **0.947049**, measured **0.9400** (0.55σ) | `pytest python/tests/test_minhash.py::test_measured_lsh_recall_at_the_threshold_matches_the_derived_bound` |
| Cross-language renderer constants | 9 surfaces compared, **all agree** | scripted parse of `crates/qd-runtime/src/render.rs` against `qd_data.render` |
| `ruff check` on the eight files written | 2 findings, both **pre-existing** | `ruff check --config pyproject.toml <the 8 files>` |

Per-file: `test_render.py` 79, `test_mixture.py` 45, `test_minhash.py` 29, `test_dedupe_split.py`
24 = 177. **44 of those 177 are new from this lane** (17 mixture, 20 render, 4 minhash, 3 dedupe),
counted with `pytest --collect-only -q -o 'addopts='`.

**The one whole-suite failure is not this lane's.** It is
`test_wire_gap_pins.py::test_the_rust_side_guard_does_not_yet_see_this_package`, in the `qd_wire`
lane's in-flight work. It **moved between three consecutive runs** during this session — 6 failures
in `test_wire_golden_corpus.py`, then 1 in `test_wire_answer_parser.py`, then 1 in
`test_wire_gap_pins.py` — which is what a lane actively editing looks like from outside. This lane
never imported, read or edited `python/qd_wire/`.

**The two ruff findings are pre-existing, and that is measured, not asserted.** `RUF022` (`__all__`
not sorted) fires on **12 files across `qd_data` and `qd_train`**, most never touched here;
`RUF023` is on `DeterministicRng.__slots__`, code this lane did not edit. Command:
`ruff check --config pyproject.toml --select RUF022,RUF023 python/qd_data/ python/qd_train/`.
The two findings this lane *did* introduce (`I001`, `SIM300`) are fixed.

**No ledger row.** This lane ran no model and no GPU suite. `python/qd_data/` is entirely untracked
against HEAD, so `git diff` shows nothing for it — the files are new in the working tree, not
modified, and that is why there are no commit hashes in this handoff.

---

## What changed, and why each record was about something other than its title

### 1. `code.language_id` — the refusal was fine; the *silence above it* was the defect

Record: **GAP-DATA-COMMITPACKFT-LANG-COVERAGE-NOW-FAILS-CLOSED** (supersedes
`GAP-DATA-COMMITPACKFT-LANG-SET-UNVERIFIED`).

Per row, the pipeline was already right and loud: `loaders.py:190` takes `lang` verbatim,
`mixture.py:277-283` refuses an out-of-set value with `reason_code="lang_not_in_option_set"`,
counted. Two things above it were not:

1. `refusals` is keyed by **source** and `n_input` counts raw rows read **once**, but
   `build_mixture` iterates those rows once per family — a commitpackft pull makes `3 * n_input`
   attempts. The superseded record's own workaround proposed dividing a reason count by `n_input`;
   that division answers a question nobody asked, and the per-family fraction was not recoverable
   from `MixtureResult` at all.
2. `build_mixture` returned `Ran(passed=True)` when a requested family produced **zero** examples.
   The pre-existing `test_refusal_counts_reach_the_result_rather_than_disappearing` proved it: five
   rows all `lang="Brainfuck"` wiped out `code.language_id` — a **held-out** family — and the test
   asserted `isinstance(status, Ran)`. Downstream everything then passes *vacuously*:
   `split._held_out_families_absent` finds no offenders because no rows of the family exist
   anywhere, the manifest records a clean stage, and the training door admits the snapshot. The
   abstention gate is measured on nothing while every report says the holdout was exercised.

Fixed: `MixtureResult.family_coverage: dict[family_id, TriState]` — `Ran(n=built,
n_total=attempted)` per family with the refusal codes in the detail, or `NotRun` when every attempt
was refused — reaching `to_json()` and so the manifest. Overall `status` is `NotRun` when any
requested family produced nothing.

**`NotRun` and not `Ran(passed=False)`, deliberately.** `qd_train/data_access.py:205` is
`if isinstance(manifest.status, NotRun):` and that is the only branch, so a failed-but-ran snapshot
walks straight through the training door. That hole is logged as its own record
(`GAP-TRAIN-DOOR-ADMITS-A-RAN-FAILED-SNAPSHOT`) and **was not fixed here** — `python/qd_train/` is
another lane.

No case-folding was added. Folding `"python"` into `"Python"` is silent bucketing: the gold label
would stop being the value the upstream row carried, and a pull whose spelling convention differs
from the 16-value set would look like a clean pull with a slightly different mixture.

One existing test was **edited, not weakened**: `test_refusal_counts_reach_the_result_rather_than_
disappearing` moved from a total to a partial refusal, keeping every assertion it made
(`refusals[...] == 5`, `n_total == n + 5`) while the total-refusal case moved to the new `NotRun`
test. Its `isinstance(status, Ran)` encoded the behaviour that was wrong; the docstring says so.

### 2. Bidi — the answer is "yes, escape", and `qd_data.render` is the wrong place to do it

Record: **GAP-DATA-RENDER-BIDI-DECIDED-REFUSE-AT-THE-CORPUS-BOUNDARY** (supersedes
`GAP-DATA-RENDER-BIDI-UNESCAPED`).

The superseded record rejected escaping because it "would corrupt legitimate right-to-left
content". **That reason is wrong.** Escaping is lossless — `unescape` is its exact inverse — so RTL
content survives as `‮` and is recovered byte-for-byte. The only cost is that a reviewer sees
the escape instead of the glyph, which is the entire point.

The decisive reason is a different one, and it is why this was never a one-line change:
`HEX_ESCAPED` is **hand-transcribed** into `crates/qd-runtime/src/render.rs:180`, and **nothing
pinned the two together**. The Rust byte-identity check,
`render_contract.rs::rust_renders_the_same_bytes_as_the_python_lane`, compares one frozen golden
produced by running the Python renderer once by hand; its sample context contains none of the
candidate characters, so a one-sided widening would have desynchronised the training renderer from
the serving renderer **silently**. Trading a display-integrity bug for a train/serve prompt drift
is a bad trade, and `crates/` belongs to another lane.

So the escape alphabet is **unchanged** (still the same 9 codepoints, still byte-identical to
Rust), and the protection lands where this lane owns both sides:

- `qd_data/render.py` gains `INVISIBLE_FORMAT_CHARS`: **the complete BMP Unicode general-category-Cf
  class, 43 codepoints**, enumerated and frozen, plus `first_invisible_format_char()`.
- `qd_data/mixture.py` refuses any row whose untrusted text carries one —
  `RowRefused(reason_code="invisible_format_characters")`, counted like every other refusal — at the
  two funnels every row passes: `_request` (context and every `ChoiceSlot` option) and `_row`
  (`repo_key`, `identity_key`, which reach the manifest raw because `canonical_json` uses
  `ensure_ascii=False`).

**Why 43 and not the eight the record named.** U+202A–U+202E and U+2066–U+2069 are not a class: the
same reordering is reachable with the implicit marks U+200E / U+200F / U+061C, and the same
"displays identically, tokenises differently" trick with the zero-width characters
(U+200B–U+200D, U+2060, U+FEFF) and U+00AD. Cf is the closed class containing all of them. Frozen
rather than computed from `unicodedata` at import, because deriving it live would make *which rows
are refused* — and therefore `data_snapshot_hash` — a property of the interpreter's Unicode version;
a test asserts the frozen table still equals the live one, so a Python upgrade fails loudly instead
of drifting. BMP-only because the escape form has four hex digits.

**Security impact.** This is Trojan Source (CVE-2021-42574). Every structural check in this lane —
option count, marker count, the letter map, the `\n` line count — passes on a row carrying a bidi
override, which is exactly why it needed a refusal of its own: the control it defeats is a human
reading rendered text. Concretely here, `code.commit_intent` asks whether a commit message describes
a diff and both sides are agent-authored, so a reviewer approving such a row, or a model card
quoting it, is shown a different program than the tokenizer was given. The widened class also
closes a live label bug: `mixture.py:255-260`'s decoy guard `claimed.strip() == raw.message.strip()`
is a string comparison, so a decoy differing from the true message by one U+200B passes the
"different from the true message" check while displaying identically, and is then labelled `"no"`.
**The residual exposure is serve time, not train time**: `qd-runtime` still renders such a request
raw.

Two more things fell out of this and are fixed:

- `python/tests/test_render.py::test_the_escape_alphabet_is_identical_to_the_rust_transcription`
  parses the `const HEX_ESCAPED: &[char]` table out of `render.rs` and pins the two alphabets. It
  refuses an empty or missing parse rather than passing vacuously, and it passes today.
- An import-time assert now refuses any non-BMP codepoint in `HEX_ESCAPED`. The escape emits `\u`
  plus **four** hex digits and `unescape` reads exactly four back, so a codepoint above U+FFFF would
  escape to five digits and the fifth would survive as literal text — a silently broken round-trip
  that was latent.

### 3. LSH recall — the miss rate was never the defect; reporting it as exhaustive was

Record: **GAP-DATA-LSH-RECALL-BOUND-NOW-STATED-AND-MEASURED** (supersedes
`GAP-DATA-LSH-RECALL-BOUND`).

**Derived.** `choose_bands(num_perm=128, threshold=0.8)` picks `b=16, r=8`, so per-pair recall is
`1-(1-0.8**8)**16` = **0.947049** at the threshold, 0.993842 at 0.85, 0.999877 at 0.90, and
1.000000 to six places at 0.95.

**Measured against a brute-force exhaustive pairwise baseline.** 300 independent document pairs
built at *exactly* J = 0.80 (16 shared shingles, 2 unique each side, union 20). Brute force over all
179,700 pairs of the 600 documents finds all 300. Banded LSH proposes **282**. Measured recall
**0.9400** against derived 0.947049 — a deviation of **0.55 binomial standard errors** (σ = 0.0129).
**18 genuine threshold pairs were never proposed.**

The error is one-sided and that is not the problem: every candidate is confirmed by exact Jaccard,
so dedupe can under-remove but never over-remove. The problem was that `DedupeReport` carried
`bands` and `rows_per_band` — the *cause* of the recall loss — and then
`Ran(n=len(units), n_total=len(units))`, which is complete coverage in the only dimension a machine
reads.

Fixed: `BandConfig.recall_at_threshold`; `DedupeReport.lsh_recall_at_threshold` and
`.n_possible_unit_pairs` as **derived properties**, not stored copies that could drift from the
banding that produced the candidate list; both in `to_json()` and so in the manifest; and the `Ran`
coverage pair moved from the unit dimension to the **pair** dimension —
`n=len(candidate_pairs)`, `n_total=C(n_units, 2)` — so `Ran.is_complete_coverage` is now `False` for
any real corpus. On a 24-row corpus the report reads: 0 of 2556 possible unit pairs compared, recall
0.947049, `is_complete_coverage=False`. A search that hit `max_candidate_pairs` stays `NotRun` — a
different fact from a completed probabilistic search, and the two must not be conflated.

Note for whoever reads the older test: `test_lsh_recall_equals_the_exhaustive_answer_on_a_small_
corpus` is sound, but its fixture sits at J ≥ 0.94 where the per-pair miss probability is 3e-7. It
is not evidence about behaviour at the threshold, and its name oversells it.

---

## Every new test was run against broken code

13 mutations, each breaking exactly one thing, each restored byte-for-byte afterwards (asserted, not
assumed). **All 13 were killed.** A test that was never run against broken code is not evidence.

| Mutation | Killed by |
| --- | --- |
| M1 `elif uncovered:` → `elif False:` | `test_a_family_wiped_out_by_out_of_set_langs_is_not_run_not_a_clean_pass` |
| M2 `n=n_built` → `n=n_attempted` (sample reported as population) | `test_per_family_coverage_carries_built_and_attempted_not_just_a_source_total` |
| M3 `if n_built == 0:` → `if n_built < 0:` | as M1 |
| M4 detector always returns `None` | 26 tests, first `test_every_bidi_control_is_in_the_refused_class[ALM]` |
| M5 drop U+061C from the frozen class | 2, incl. `test_the_frozen_format_table_still_matches_this_interpreters_unicode` |
| M6 add U+0001 to the Python `HEX_ESCAPED` only | `test_the_escape_alphabet_is_identical_to_the_rust_transcription` |
| M7 delete the option arm of `_request` | `test_an_invisible_character_in_an_option_is_refused_at_the_one_funnel` |
| M8 delete the `repo_key` check in `_row` | `test_an_invisible_character_in_a_repo_name_is_refused_before_the_manifest` |
| M9 detector → `ord(ch) > 127` (a non-ASCII filter) | `test_the_detector_returns_the_character_so_a_refusal_can_name_it`, `test_ordinary_non_ascii_text_is_not_swept_up_by_the_invisible_check` |
| M10 `recall_at_threshold` → `1.0` | 5, first `test_the_banding_states_its_recall_at_the_threshold` |
| M11 dedupe coverage back to the unit dimension | `test_the_dedupe_report_states_its_recall_bound_rather_than_implying_completeness` |
| M12 `choose_bands` → `b=num_perm, r=1` (never misses) | `test_measured_lsh_recall_at_the_threshold_matches_the_derived_bound` |
| M13 drop the bound from the report JSON | 2, first as M11 |

**One mutation had to be redone, and the reason matters.** M6 first used U+202E, which trips the
module-level `INVISIBLE_FORMAT_CHARS & HEX_ESCAPED` disjointness assert at *import* — a loud
failure, but the assert's, not the parity test's. Re-run with U+0001 (category Cc, outside the Cf
class) so the module still imports and only the parity test can catch it. It did, alone. A kill by
the wrong mechanism is not evidence that the test bites.

M9 exists specifically to guard the regression this fix could invite: the Cf class quietly widening
into a "reject anything unfamiliar" filter that drops legitimate Arabic, Hebrew, CJK and emoji
source files.

---

## What is open

| Gap id | What it blocks |
| --- | --- |
| `GAP-DATA-COMMITPACKFT-LANG-COVERAGE-NOW-FAILS-CLOSED` | The real `lang` distribution over 702k rows is **still unmeasured**. Carries a `NotRun` residual. |
| `GAP-DATA-RENDER-BIDI-DECIDED-REFUSE-AT-THE-CORPUS-BOUNDARY` | Serve-time handling of Cf characters. Needs `crates/qd-runtime`. |
| `GAP-DATA-LSH-RECALL-BOUND-NOW-STATED-AND-MEASURED` | The bound is on the estimator, not on any real corpus. |
| `GAP-RENDER-ESCAPE-ALPHABET-WAS-UNPINNED-ACROSS-LANGUAGES` | The Rust-side mirror of the parity guard. XLANG-RS is fixing the false comment at `render.rs:213`. |
| `GAP-RENDER-CROSSLANG-PAIRS-AGREE-TODAY-BUT-ARE-UNPINNED` | Eight of the nine cross-language surfaces are still unpinned (measured in agreement today). |
| `GAP-TRAIN-DOOR-ADMITS-A-RAN-FAILED-SNAPSHOT` | `qd_train/data_access.py:205`. Not this lane's file. |
| `GAP-DATA-HARDEN-GITPULSE-UNTRUSTED-AND-DEVMAP-STALE` | Collision checking for everything this lane wrote. |

### What could not be verified, and why

- **The commitpackft fraction. `NotRun`, never a pass.** No network on this host,
  `datasets-server.huggingface.co` unreachable, HuggingFace terms not accepted, `datasets` not
  installed. Nothing here measured the dataset; it changed what happens when the answer turns out to
  be bad. The 16-value option set is still a hand-made guess against an unobserved distribution, and
  16 is a hard cap — the generic route decodes over a 16-row `lm_head` slice, so a wider label set
  needs a different question shape, not a longer list.
- **Serve-time bidi behaviour** is untested and unchanged.
- **The Rust-side parity guard does not exist.** The Python test catches drift only when pytest
  runs; a `crates`-only change merged without it is still undetected.
- **The recall measurement bounds the estimator, not a corpus.** Seeded Monte Carlo on synthetic
  sets. No real corpus has been deduped, so the real near-duplicate population and whether
  `max_candidate_pairs` is large enough for it are unmeasured.
- **GitPulse: every facet returned `ok=false`** with `REPOSITORY_TRUST_REQUIRED` — called, not
  assumed. Nothing this lane wrote was collision-checked against other worktrees.
- **DevMap answered but degraded**: `is_fresh=false`, `degraded_reason="source tree differs from the
  indexed generation"`, and every edge walk returned `walk_incomplete` ("10,937 of 22,989 unresolved
  attribution sites have no indexed target"). Caller lists are lower bounds. The claim that most
  needed a complete one — that `_request` and `_row` are the **only** funnels constructing a
  `Request` and a `DataRow` in `mixture.py` — was verified by **ripgrep** over that file (one
  `Request(` site, one `DataRow(` site) and is **grep-derived, not graph-confirmed**.
- **The cross-language agreement measurement is a static comparison at one moment.** It runs no Rust
  and is wired into no suite, so it will not notice the next change.

---

## The exact first command for the next lane

```
.venv/bin/python -m pytest python/tests/test_mixture.py python/tests/test_render.py \
  python/tests/test_minhash.py python/tests/test_dedupe_split.py -p no:cacheprovider --no-header
```

Expect **177 passed**. If `test_the_escape_alphabet_is_identical_to_the_rust_transcription` is the
one that fails, `crates/qd-runtime/src/render.rs` and `python/qd_data/render.py` have drifted and
the model is being served a prompt shape it was not trained on — fix the drift, do not adjust the
test.

Then, before touching any of these files:

```
gitpulse_insights(repo_path="/Users/bharath/Code/research/qwen-decision")
```

It currently returns `REPOSITORY_TRUST_REQUIRED` on every facet. A human must trust the repository
in GitPulse first; until then, collision risk against other worktrees is unmeasured, not clean.
