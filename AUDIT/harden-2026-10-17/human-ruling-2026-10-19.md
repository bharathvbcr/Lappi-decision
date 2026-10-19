# Owner ruling, 2026-10-19: the scope of qd-train-data's exact_short tier

Rule 2 makes what a gate checks the owner's call. Fable's Q23 (fable-rulings-2026-10-19.md) found the tier under-specified and put the question to the owner. Session lappi-decision-ce asked it through AskUserQuestion. The owner's answer is recorded verbatim.

**Question.** The v6 shard gate's short-target check (exact_short in qd-train-data) fails the LoRA-2B data, mostly on template text rather than real contamination. It matches a short CLINC test utterance against every line of every training row, including answer-option lines and code lines, and it reads the text before template stripping. Of 4,833 hits, 3,558 are just the answer option 'goodbye'. Of the 1,275 left after stripping, only 92 are real CLINC overlap. The other 1,183 are code lines like `false` and generic options like 'great'. Which should the gate do?

**Options offered.**
1. Strip, whole-text match (Recommended): Fable's option 3.
2. Strip, keep per-line: Fable's option 2; drop all 1,275.
3. Leave the gate as is: Fable's option 1; drop all 4,833.

**Owner's answer:** "Strip, whole-text match (Recommended)"

**What it means, binding from this commit:**
- `exact_short` reads the template-stripped (strip v2) text, the same text the builder's containment rule is held to (python/qd_train/exclusions.py:14-18).
- A row is a hit only when its whole stripped slot text, normalised as train_data.rs:285-288 normalises, equals a short test target.
- A short target equal to one line inside a longer row is not a hit. Such matches are reported as a counted residual in the tier's coverage output (train_data.rs:25-26 already names embedded hits as residual), never silently dropped from the report.
- The tier's test at crates/qd-preflight/src/train_data.rs:812-827 is rewritten to this rule. Its fails-before is the current tree failing the new test.
- The builder and the gate share one matcher, owned by qd-prep (Fable Q22 (b)). The 92 CLINC rows (23 utterances × 4 families, including both digest duplicates) leave train through the existing exclusions seam.
- The two label-conflicting test items ("what's your designation" 11/75, "where did you grow up" 77/99) are recorded as noisy test items, report only. The test set is never edited (Fable Q23 (b)).
