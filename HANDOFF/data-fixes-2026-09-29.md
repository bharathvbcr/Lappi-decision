# data-fixes lane — 2026-09-29

CPU-only. No GPU used. DevMap has no store for this worktree and `ListAgents` was not
callable from this subagent (gaps already recorded by other lanes); navigation was `rg`/Read,
not the graph. Nothing committed.

## Measured (ledger row ids)

| row | file | what |
| --- | --- | --- |
| `901ac3a7` | `ledger/mac-defect-class-shards-2026-09-29.jsonl` | final **pre-general** shard rebuild (superseded by the lead's rebuild from a clean commit), all fixes incl. SQuAD title partition. `padding_waste` 4.72% (gate 15%, passed). quick. Its shard dirs predate the sequence-index `role` field and no longer read. |
| `b82bedad` | same file | rebuild after tasks 1-3, before the SQuAD partition. `padding_waste` 4.72%. Superseded by `901ac3a7`. |
| `0eb72e30` | `ledger/mac-length-control-2026-09-29.jsonl` | length-only control on `code.defect_class/defect_class`: 1,163/2,332 = 0.4987 val top-1 vs 0.4528 majority. quick. |

Final rebuild (`901ac3a7`), command:

    /Users/bharath/.venvs/ml/bin/python -u tools/real_tokenizer_pipeline.py --out <scratch>/data-fixes/rebuild2 \
      --defect-class data/pool/commitpackft-corpus-v2 --memo-limit 0 --val-shards --ledger <scratch ledger>

* train: 46,002 of 46,145 rows written, 85,014 of 91,550 (row, slot) sequences; 143 rows refused
  whole (render cap), 6,393 span slots refused on their own account (BPE line-start collapse).
* val: 2,422 of 2,466 rows, 4,475 of 4,798 sequences; 44 rows whole, 279 span slots.
* `code.defect_class` class balance as written — train 45,405 rows: stub 20,860 / logic 10,286 /
  clean 7,597 / cosmetic 6,662 (majority 0.4594); val 2,332: stub 1,056 / logic 516 / clean 416 /
  cosmetic 344 (majority **0.4528**, equal to the split's; A3's as-written was 0.4418).
* buckets `[193 … 1315, 4702 … 37098]`, `padding_waste` 0.047163.

## Changed

* `python/qd_data/mixture.py` — MMLU/CSQA `_dispatch` branches (lazy `general` import); SQuAD
  title routing in `build_mixture` (routed, not refused) + `rewrite_squad` fail-closed refusal.
* `python/qd_data/split.py` — `squad_title_family`, `SQUAD_ANSWERABILITY_TITLE_FRACTION = 0.2`,
  `content_disjoint_families` (in `SplitReport.status` when applicable).
* `python/qd_data/render.py` — `DeterministicRng.cyclic_permutation`, `second_pass_permutation`
  (canonical owner); `python/qd_data/defect_class.py` re-exports it.
* `python/qd_train/shards.py` — slot-scoped refusal; `sequence_index.json`
  (`SEQUENCE_INDEX_NAME`, format `qd-sequence-index/1`, atomic, hash pinned in header);
  `ShardReader.sequence_index` / `.slot_coverage`; `choose_buckets` = exact minimum-padding
  partition. `python/qd_train/artifacts.py` — `ShardHeader.sequence_index_hash`.
* `python/qd_train/baseline.py` — `Featurizer` protocol, `ContextLengthFeatures`,
  `RequestDoc.context`. `tools/ft_linear_control.py` — `LENGTH_ARM`, reported beside the gate.
* `tools/real_tokenizer_pipeline.py` — census mirrors slot scope; `defect_balance` counts a row
  unless its choice slot is gone; `shard_slots_written`, `val_shard_slots_written`,
  `refused_slot:*` metrics.
* Tests: new `test_general_mixture_integration.py`, `test_shard_sequence_index.py`,
  `test_length_control.py`, `test_squad_title_partition.py`; edits to `test_shards.py`,
  `test_stress_this_session.py`, `test_mixture.py`, `test_render.py`,
  `test_pipeline_defect_class.py`, `data_fixtures.py`. `AUDIT/shard-lengths-2026-09-29.json`.

Later the same day (lead queue):

* `python/qd_data/split.py` — `split()` honours `metadata[PINNED_SPLIT_KEY]`; an unknown
  value raises (`test_pinned_split.py`).
* `python/qd_data/mixture.py` — Part B: `build_mixture(..., clinc_domain_map=)`; the two-stage
  CLINC families refuse `no_clinc_domain_map` without one and are left out of the defaults.
* `python/qd_train/shards.py` — `write_shards(replay=)`: a gold set refuses `replay_only` rows
  and a replay set refuses anything else; `sequence_index.json` carries `role`
  (`gold` / `replay_only`). **Sets written before this (b82bedad, 901ac3a7 dirs) no longer
  read** — `role` is required.
* `python/qd_train/shards.py` — the span decode check's statement (1) compares the run of
  tokens that overlap a token's claimed span, so an honest byte-level split of a multi-byte
  character passes (`test_span_decode_multibyte.py`,
  `GAP-DATAFIX-SPAN-DECODE-CHECK-REJECTS-BYTE-SPLIT-CHARACTERS`).
* `tools/real_tokenizer_pipeline.py` — `--general-record` (every cache file checked against
  the fetch record's root, sha256 and row count; `REFUSED_READS` splits counted, not parsed),
  `--general-max-rows`, `--replay-shards` (replay slice as `shards/replay`, own manifest
  `data/pool/train-replay.json`, same remap). Tests in `test_pipeline_general.py`.
* `tools/replay_decontam.py` — unrenderable targets counted and skipped, fail closed on a split
  that renders nothing and on a non-replay shard set (`test_replay_decontam.py`).
* `python/qd_data/mixture.py` — `drop_contradictory_prompts`: every row of a rendered prompt
  that carries two golds is dropped, no winner, counted as `contradictory_prompt` (refusals and
  family coverage); `prompt_consistency` is the verdict on the emitted rows. Replaces
  refuse-the-whole-corpus (lead direction); the retired tests were rewritten and say so.
  `tools/real_tokenizer_pipeline.py` passes `PIPELINE_MAX_CONSISTENCY_ROWS = 1_000_000`
  (recipe key `max_consistency_rows`) and exits at stage 1 if the pass is NotRun.
  `GAP-DATAFIX-CONSISTENCY-PASS-NOT-RUN-ABOVE-BOUND-LET-CONTRADICTIONS-THROUGH`.

Two rebuilds with `--general-record --replay-shards` aborted at stage 6 and wrote **no ledger
row** (so none of their numbers are reportable): first on the byte-split decode check, then on
42 contradictory sequences in 21 groups (first: SQuAD `56cd97f8…` answerable vs `5a8dab7c…`
unanswerable, same question and context). Both causes are fixed above. The final rebuild is
the lead's, from a clean commit — command below.

## Open (gap ids)

* `GAP-A3-SECOND-PASS-PERMUTATION-NOT-WIRED-INTO-RENDER` — open: rung0 not replaced (506/3,000
  agreement = chance), no Rust parity and no fixture, FT eval does not call it.
* `GAP-A3-DEFECT-METADATA-DOES-NOT-REACH-THE-SHARD-SET` — untouched (another lane's).
* Pre-partition ledger rows built with SQuAD-shaped rows are not comparable on the two SQuAD
  families: the `real_tokenizer_pipeline` smoke rows in `gh200-2026-09-20/21`, `runs.jsonl`,
  `mac-shards-commitpackft-2026-09-22`, `mac-smoke-2026-09-29`, `shardset-v3/v4`,
  `mac-defect-class-pipeline-check-2026-09-29` (`8f8558a9`) and `b82bedad`. None ran
  `ood_abstain`.
* `SQUAD_ANSWERABILITY_TITLE_FRACTION = 0.2` is a lead decision under user delegation
  (2026-09-29), pinned by a test; revisit only as a decision, not as a tuning knob.
* Pipeline hardening for its next owner: stage 3's census already names FATAL slots that
  will abort stage 6, and the run still spends ~20 minutes reaching it. Fail fast there.

Closed: `GAP-A3-BPE-SPAN-COLLAPSE-DROPS-DEFECT-CLASS-LABELS`,
`GAP-DATAFIX-PADDING-WASTE-QUANTILE-TAIL`, `GAP-A3-CLEAN-DIFFS-ARE-LONGER-THAN-MUTATION-DIFFS`,
`GAP-DATA-SQUAD-SPAN-NOUL-LEAKS-HELD-OUT-ANSWERABILITY`,
`GAP-REPLAY-DECONTAM-TOOL-CRASHES-ON-UNRENDERABLE-FT-VAL-ROW`,
`GAP-DATAFIX-SPAN-DECODE-CHECK-REJECTS-BYTE-SPLIT-CHARACTERS`.

## First command for the next lane

The final rebuild, from the repo root at a clean commit (about 25-35 min, one core,
8.3-8.6 GB peak RSS; the aborted run took 1,492 s wall and 8.34 GB before stage 6):

    /Users/bharath/.venvs/ml/bin/python tools/real_tokenizer_pipeline.py \
      --out <out dir> --max-pairs 400 --rev HEAD --val-shards \
      --defect-class data/pool/commitpackft-corpus-v2 --defect-download data/pool/commitpackft \
      --memo-limit 0 \
      --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
      --replay-shards --ledger <ledger path>

Shard sets land at `<out>/shards/train`, `<out>/shards/val`, `<out>/shards/replay` (replay
manifest `<out>/data/pool/train-replay.json`). Then:

    /Users/bharath/.venvs/ml/bin/python tools/ft_linear_control.py --ledger <ft eval ledger> \
      --verdicts <verdicts.jsonl> --max-pairs <n> --rev <sha>

It now reports `length_control_top1.<task>` and `paired_margin_vs_length_control` beside
`paired_margin_vs_linear`; the go/no-go needs the 2B to beat both.
