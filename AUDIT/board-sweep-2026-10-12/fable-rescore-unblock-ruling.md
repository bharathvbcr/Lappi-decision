# Fable: re-score unblock ruling (2026-10-12, main a6551e1)

Asked by the lead for wave 3 items 3 and 4: (A) re-score v0.1's span with `--verdicts-out` and fit
the span entry two-fold (ft-0f5a10c9f3e steps 3-4); (B) run span arm (e) of
`campaign/span-refit-mac-preregistered.json` (ft-a4ef800cc7a item 4). Lane w3-rescore stopped
before any heavy job (`HANDOFF/span-rescore-2026-10-12.md`, `scratchpad/lanes/w3-rescore-plan.md`,
GAP-RESCORE-VAL-SHARD-STALE-VS-LOGIT-WRITING-CODE-2026-10-12,
GAP-SPAN-REFIT-ARMS-B-E-UNSCORED-NO-LEDGER-RECORDER-2026-10-12).

Labels: **[V]** I read the line or ran the command this session; **[R]** a document's claim I did not
re-run; **[I]** inferred; **[U]** unverified. I wrote nothing but this file: no gaps append, no cargo,
no GPU, no branch. Rules 2, 3, 5, 8 bind every line below. I did not run `ListAgents` or GitPulse
(the lane does, before every write).

---

## 0. Facts that decide the ruling

1. **The pin tree is the shard writer's code, byte for byte.** [V] `code_fingerprint` of
   `build/span-pin-8a531e8/python/qd_data` against the val header: changed `[]`, added `[]`,
   removed `[]`. Main drifts in 11 modules plus `caller_records.py`; `render.py`, `rows.py`,
   `schema.py`, `pool_builder.py`, `manifest.py` are unchanged on main. All three v5 sets
   (train `10808254…`, val `a485e7f7…`, train-no-decode `e8cffbdd…`) pin corpus_rev `8e6a0095`
   and snapshot `cb4b7d3a…`/`7d11986a…`.
2. **A `--score-val` run reconstructs labels; it does not merely read tokens.** [V]
   `tools/real_ft_run.py:4632-4700` rebuilds the split (`dedupe`, `split`, native MinHash via
   `QD_PREP_BIN`) and `pair_labels` (5151-5215) pairs by `(row_id, slot_name)` against
   `sequence_index.json`, refusing any disagreement. The span **gold** a verdict writes comes from
   `supervision.npz` (`ft_supervision` → `plan_span_batch`, `heads.py:184-193`), not from the rebuilt
   rows; the rebuild supplies identity, family, letters and `expected_abstain`.
3. **`allow_stale_code` exists in two places and both say what it is for.** [V]
   `python/qd_train/artifacts.py:1243-1284`: the deliberate exception "to inventory it, to compare it
   against its successor, to reproduce an old row"; reported either way, only declines to raise.
   `tools/perf_step.py:249-260` reaches it by monkeypatching `assert_shard_trainable`, with the
   reason in the comment: *"a throughput or same-batches parity measurement does not read its labels
   as labels. The production reader is never patched."* The Rust reader mirrors it
   (`crates/qd-train/src/shards.rs:653`). Python `ShardReader` exposes no such parameter, on purpose.
   A `--verdicts-out` re-score reads labels as labels (the gold indices are what `qd-calib-fit` fits
   on), so it is outside the flag's charter.
4. **a1b1799's verdict hunk fits the pin.** [V] `git apply --check` of `git show a1b1799 --
   tools/real_ft_run.py` against `build/span-pin-8a531e8`: exit 0. The hunk touches only `_decode`'s
   verdict dict, two call sites, and `VERDICT_DISTRIBUTION_KEYS` (41 lines; no scoring arithmetic).
   The pin's `heads.SpanPlan` carries `gold_start`/`gold_end` (`heads.py:137-138`), and its
   `real_ft_run` knows `--split-cache`, `--ft-ledger`, `--batch-tokens`, `--score-dtype`,
   `--devices`, `--wall-clock-cap-s`. `CAMPAIGN_DEVICES = {"cuda"}` (line 423), so a Mac row is
   `quick` by construction (precedent: row 20cf3845, 2026-09-30, device mps).
5. **Every input the label rebuild needs is on this Mac.** [V] The v5 argv is
   `build/v5-h100/score_avg.sh` (row 6af73bef's own command, main 4ebb790). Mac copies:
   exclusions `/Users/bharath/qd-campaign/v5-containment-predrops-2026-10-03/exclusions.txt`
   sha256 `c38dd2ca…` = the train header's `exclusions_sha256` (GAP-P1's "no copy on the Mac" is
   wrong: it searched the box's directory name `v5-exclusions-r-2026-10-03`); pre-dedupe drops
   `a8ab2a11…`; general record `~/.cache/qd-decision/general/fetch-record-2026-09-29.json`
   `a0841f0d…`; decisions pool `…/v5-decisions-data-2026-10-03/pool-v5-decisions-v4`; backbone
   snapshot `b1485b2f…`; `data/pool/{commitpackft-composed-v2,commitpackft,defect-noul-v3c}`.
   Not verified: a Mac `qd-prep` at v5's pinned commit (the box binary `900534f1…` is linux;
   `build/qd-prep-ca48960` exists, sha unchecked) [U].
6. **The rows the re-score cites are not in the checkout.** [V] The five v5 ft rows
   (9a00cfd0, b0d9454e, 8079c966, 1c8a0ab5, f2bd50e7; cuda, non-quick) and eval rows 6af73bef
   (avg, span 5,596) and e1bafd6f (seed 1, span 6,626) live only in
   `/Users/bharath/qd-campaign/ledgers-h100-2026-10-05/ledger/h100x2-v5-2026-10-03.jsonl` (16 rows)
   and `h100x2-v5avg-2026-10-05.jsonl` (1 row). `ledger/` has neither; `build/v6x/fixtures/v5rows.jsonl`
   is a test fixture. Scoring an average pairs it with the ft rows its manifest names through
   `--ft-ledger` (`_check_score_checkpoint_flags`, pin 9417-9435).
7. **B has no recorder.** [V] `span_refit_score.py:383-399` drafts rows and says so. The committed
   binders of exactly this shape are `tools/calib_fit_row.py` and `tools/gate_report_row.py`: a
   `RunRecorder` block around a subprocess, `quick`, recipe names `reported_eval_rows` /
   `fitted_eval_rows` and never `eval_row_id`, protocol copied from the eval row, every number a
   metric. Arm (a) ran gate-only: 5,596/7,238 exactly, cache fidelity 7,237/7,238
   (`results/span-refit-score.json`, prereg sha `b4a370ff…`, population pin `bdd571b6…`).
8. **Quick is not a refusal downstream.** [V] `calib_fit_row.py` requires exactly one eval row per
   verdict file and is itself always `quick` ("which table ships is the human's call, rule 2");
   `crates/qd-export/src/export.rs:624-660` validates the table's bytes, not its provenance. The
   v0.1 release table (`calib-all/table.json`, `7e56b34e…`) was fitted from 6af73bef with split key
   `lappi-v0.1-preview-2026-10-06` and has no span entry.

---

## 1. Blocker (1): **option (iii), as a committed branch.** Not (i), not (ii), not (iv).

**Ruling.** In the existing worktree `build/span-pin-8a531e8`, branch `span-pin-8a531e8-verdicts`
from 8a531e8 and commit exactly one change: a1b1799's `tools/real_ft_run.py` hunk (the scratch patch
the lane already extracted, applied with `git apply`). Message: "real_ft_run: a1b1799's verdict
writer on the v5 shard pin (Fable re-score ruling 2026-10-12)". No other file; no rebase; no push
needed. The re-score runs from that tree, and its eval row's `code_commit` is that sha.

**Why this is acceptable provenance.** The row names a real, non-dirty commit whose diff from
8a531e8 is 41 reviewable lines in the verdict writer and none in scoring (fact 4). The `qd_data`
that rebuilds the labels is the code that wrote the shard set (fact 1), which no option on main can
say. The tree already reproduced 5,596 exactly on this Mac (arm (a), fact 7). And the new row is
read only if its `val_top1.span` lands in **[5,581, 5,611]** (5,596 ± 15, the prereg's own
MPS-vs-CUDA tolerance); outside it, the row is void and a gap is recorded. That gate is the proof the
backport changed what is written and not what is scored.

**Why not (ii).** The flag's two owners state its charter (fact 3) and a verdict-writing re-score is
outside it. Threading it through Python `ShardReader` is a production-reader change on main with a
test, a widened door for every future caller, and weaker provenance: main's rebuild would run
`dedupe`/`split`/`mixture`/`defect_class` code that did not write the set, with `pair_labels` as the
only net. More code, less certainty. Refused.

**Why not (i).** Regeneration costs about an hour of the one lock (rows 849d3ae5 and c3bd0374:
2,729 s and 2,888 s), and even byte-identical rows produce a **different `shard_hash`**, because
`shard_hash` covers `code_fingerprint` (`artifacts.py:534-544`). It therefore reproduces neither the
prereg's `val_shard_hash` pin nor 6af73bef's reader, and the fingerprint-equality check the lead
proposed ("identical in every field that reaches the model") is answered more directly by fact 2:
tokens and golds come from `tokens.u32` and `supervision.npz`, which no code path re-derives at score
time. Regeneration answers a different, real question, *is v5's population reproducible under main's
`qd_data`?*, and belongs before v6's data build, not on this sweep. Follow-up, not a blocker.

**Why not (iv).** Nothing blocks execution once the branch exists; the only human decision is the
release table (section 3).

**Row checks the lane records.** `val_top1.span` in [5,581, 5,611]; verdict lines 24,492; span lines
7,238, each with `start_logits`, `end_logits`, `gold_start`, `gold_end` (`SPAN_FIT_FIELDS`,
`qd_calib_fit.rs:131`); the reader's code check `Ran passed=True` at that tree; `quick=True` with the
non-campaign-device reason, not any other.

## 2. Blocker (2): **commit a recorder; one row per arm; nothing reported before it exists.**

**Owner.** New `tools/span_refit_rows.py`, in the `calib_fit_row.py` / `gate_report_row.py` family
(Python: `qd_train.ledger.RunRecorder` is the ledger's one owner and a binding is not a hot path;
the policy's Rust/Go default does not apply to a ledger adapter). It computes nothing. It imports
nothing from `build/` (gitignored); it runs the scorer as a subprocess and pins it by sha256.

**Row shape** (`ledger/mac-span-refit-2026-10-06.jsonl`, the prereg's name):
- `run_kind="eval"`, `quick=True`, `quick_reason` = the prereg's status sentence; one
  `RunRecorder` block **per arm**, each block running `span_refit_score.py --arms <arm>` (the
  scorer gains that flag; it always computes arm (a) as the gate), so each row's wall clock is the
  recorder's own measurement of that scoring pass. The extraction and fit jobs are named in `notes`
  by their `mac_heavy` labels and are **not** priced into the row (the `_record_verdict` precedent:
  a row never repeats a parent's duration). `cost=None` (local device).
- `protocol`: copied from 6af73bef (data, tokenizer, backbone, seed), as `gate_report_row.py` does;
  `recipe_hash` the row's own.
- `recipe`: the prereg's list exactly (`tag span-refit-<arm>`, `device`, `score_dtype fp32`,
  `scored_checkpoint_sha256`, head `file_sha256` + `content_digest`, `val_shard_hash`, `fit
  {train shard_hash, n_sequences, seed, lambda, optimizer, iterations}` for (c)/(e), `prereg_sha256`),
  plus `script_sha256` for `span_refit_{common,extract,fit,score}.py`, `results_json_sha256`,
  `population_pin`, and `reported_eval_rows: ["6af73bef-…"]`. Never `eval_row_id`.
- `metrics`: `val_top1.span` `Ran(n, n_total)`; `val_top1.span.cell.<cell>` for the four cells;
  `null.<uniform_line|random_plus_line|always_abstain>.<cell>`; `cond_w_start`/`cond_w_end` on
  (c)/(e); `mcnemar.b_vs_c` on (c)'s row and `mcnemar.e_vs_c.<cell>` on (e)'s row, each carrying
  both discordant counts and the exact p; `reading` as the prereg's string. `passed=True` on every
  metric: a reading is not a verdict (rule 2).
- `gates`: exactly one, `span_refit.validity_gate` (arm (a) within ±15 of 5,596), on every row. It
  is pre-registered; recording its outcome is allowed, moving it is not.
- Refusals: prereg sha256 ≠ `b4a370ff…`; population pin ≠ `bdd571b6…`; a head file whose sha256 is
  not the one the scorer reports. If the gate fails: the arm-(a) row is written with the gate
  failed, no other row is written, a gap is appended (the prereg's "every arm is void").
- Test (red first): `python/tests/test_span_refit_rows.py` on a fixture results JSON: refuses a
  prereg mismatch; writes one row per arm; every row quick; recipe keys exact; no `eval_row_id`;
  the gate present on every row.

**Arm (e)** then runs as pre-registered: report-only (rule 8), reading cell
`qa.answer_span/pointing` against (c) by exact McNemar, void rule 0.5 % (census: 0 derivation
failures under amendment 1), never servable. GAP-QDM-REFIT-HEAD-HAS-NO-EXPORT-MODE-2026-10-12
stays open and is not this lane's.

## 3. Which head the v0.1 span entry belongs to

**The mean head v0.1 ships**: release `span_head.safetensors` `7a187a1e…`, which is arm (a)
(scorer content digest `52ca073e…`) on the averaged tower `0a19a1f6…`. A calibration table is bound
to one model's weights (`calib_fit_row.py`, "One model"), and the prereg's reading cannot change what
v0.1 shipped. So A fits on arm (a) **now**, without waiting on arms (b)-(e), and the entry is
labelled the v0.1 mean-head entry. It is report-only twice over (quick eval row, quick calibration
row) and enters the release table only by the human; the release notes' "do not serve the span
head" stands until the human says otherwise.

Ruling 2g is not re-decided: for **v0.1.x** the served head follows the prereg's reading among
(b), (c), (d) once B has rows. If that selects (b) or (d), the A command re-runs with
`--score-checkpoint` = `ckpt_average --span-head-seed 1`'s output (b) or the seed-1 checkpoint
`84edcc91…` (d), and the fit re-runs on those verdicts; (c) and (e) cannot be served until an export
mode exists. A table entry is per head; there is no entry that "follows" a selection.

## 4. Prerequisite under rule 5: the v5 rows come into `ledger/`

Copy `h100x2-v5-2026-10-03.jsonl` and `h100x2-v5avg-2026-10-05.jsonl` verbatim from
`/Users/bharath/qd-campaign/ledgers-h100-2026-10-05/ledger/` into `ledger/` (append-only files; the
source sha256s go in the handoff). Without them, the 5,596 target, the ±15 gate and the ft-row
pairing all cite files outside the checkout. Append a gap record correcting GAP-P1's exclusions
claim (fact 5); do not edit the old record.

## 5. Agent-executable now, and the human's

**Agent** (every heavy step one job at a time under `tools/mac_heavy.sh`; $0 local; rule 4 is not
triggered; `ListAgents` before every write; the pin worktree and `tools/span_refit_rows.py` are this
lane's files):
- A0 the branch and commit (section 1); A1 the two ledger copies (section 4); A2 the re-score (one
  heavy job; estimate 2-3 h on MPS fp32 at `--batch-tokens 4096`, unmeasured [I]; the split rebuild
  on CPU comes first and every pairing refusal fires before the tower loads); A3 the two-fold fit
  (CPU, light).
- B0 the recorder tool and its test (section 2); B1-B5 the pre-registered chain from
  `scratchpad/lanes/w3-rescore-plan.md` (train remainder, seed-1 val, fit c, fit e, score), then
  the five rows through B0; order A before B (one job, and its refusals are cheap and early).
- The handoff update (`HANDOFF/span-rescore-2026-10-12.md`: rows, commits, gap ids, first command).

**Human**: whether the span entry enters the release table (rule 2); whether v0.1.x changes its head
after B's reading; and a veto on the branch-as-provenance ruling in section 1, which I take as
accepted unless exercised.

## 6. The exact first command (A2), after A0 and A1

Absolute paths (the harness blocks `cd`; `score_avg.sh` relied on it). `QD_PREP_BIN` names a Mac
`qd-prep` built at v5's pinned commit, with its sha256 recorded in the handoff [U until built].

```
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh span-rescore-avg-val \
  env QD_PREP_BIN=<mac qd-prep at v5's pinned commit> \
  /Users/bharath/.venvs/ml/bin/python -u /Users/bharath/Code/research/Lappi-decision/build/span-pin-8a531e8/tools/real_ft_run.py \
  --out /Users/bharath/qd-campaign/phase4-v5r-2026-10-03 \
  --no-repo-history --rev 8e6a00952a09e31a09cc3f5270229a935b7235cc \
  --defect-class /Users/bharath/Code/research/Lappi-decision/data/pool/commitpackft-composed-v2 \
  --defect-download /Users/bharath/Code/research/Lappi-decision/data/pool/commitpackft \
  --defect-noul /Users/bharath/Code/research/Lappi-decision/data/pool/defect-noul-v3c \
  --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json --general-max-rows 200000 \
  --decisions-pool /Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4 \
  --drop-before-dedupe /Users/bharath/qd-campaign/v5-predrops-2026-10-03/pre-dedupe-drops.txt \
  --exclude-identity-keys /Users/bharath/qd-campaign/v5-containment-predrops-2026-10-03/exclusions.txt \
  --real-backbone /Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --devices mps --seeds 0 1 2 3 4 \
  --score-checkpoint /Users/bharath/qd-campaign/preview-2026-10-05/v5-avg/v5-avg-s0-4.safetensors \
  --score-val --score-dtype fp32 --batch-tokens 4096 \
  --ft-ledger /Users/bharath/Code/research/Lappi-decision/ledger/h100x2-v5-2026-10-03.jsonl \
  --split-cache /Users/bharath/Code/research/Lappi-decision/build/split-cache-ft \
  --wall-clock-cap-s 14400 \
  --ledger /Users/bharath/Code/research/Lappi-decision/ledger/mac-span-rescore-2026-10-12.jsonl \
  --verdicts-out /Users/bharath/qd-campaign/rescore-2026-10-12/verdicts-avg-mac.jsonl
```

No `--needle`, no `--ood`: the span entry needs the val pass only, and the lock is shared. A flag
the pin refuses is reported in the handoff, not worked around; a pairing refusal
(`pair_labels`, `check_exclusion_source`, `check_defect_source`) is the finding, with its message.

Then A3, against the `qd-calib-fit` at `target/release` (rebuilt if main moved; its sha256 in the
row's recipe as the tool records it):

```
/Users/bharath/.venvs/ml/bin/python /Users/bharath/Code/research/Lappi-decision/tools/calib_fit_row.py \
  --bin /Users/bharath/Code/research/Lappi-decision/target/release/qd-calib-fit \
  --verdicts /Users/bharath/qd-campaign/rescore-2026-10-12/verdicts-avg-mac.jsonl \
  --population two-fold --split-key lappi-v0.1-preview-2026-10-06 --name lappi-v0.1-span-meanhead-mac \
  --eval-ledger /Users/bharath/Code/research/Lappi-decision/ledger/mac-span-rescore-2026-10-12.jsonl \
  --ledger /Users/bharath/Code/research/Lappi-decision/ledger/mac-calib-span-2026-10-12.jsonl \
  --out-dir /Users/bharath/qd-campaign/rescore-2026-10-12/calib-span-two-fold
```

The split key is the release's own, so the span folds are the choice entries' folds. The table it
writes is report-only; the release table is untouched by the lane.

## 7. Gaps to close or append (by the lane, append-only)

- GAP-RESCORE-VAL-SHARD-STALE-VS-LOGIT-WRITING-CODE-2026-10-12: resolved by section 1 once the
  branch exists and the row lands inside the gate; cite the commit and the row.
- GAP-SPAN-REFIT-ARMS-B-E-UNSCORED-NO-LEDGER-RECORDER-2026-10-12: resolved by section 2 once five
  rows exist; cite the tool's commit and the row ids.
- New: GAP-P1's "exclusions list has no Mac copy" corrected (fact 5), and the v5 rows' absence from
  `ledger/` closed by section 4.
- Follow-up, not this sweep: regenerate v5's val set under main's `qd_data` and compare
  `data_snapshot_hash`, `sequence_index_hash`, `tokens.u32` and `supervision.npz` bytes, before any
  v6 data build reuses v5's splitter.

What I did not verify: that a Mac `qd-prep` at v5's pinned commit exists or builds [U]; the MPS wall
clock of a full-val `--score-val` on the 2B average [I]; that the pin's `check_defect_source` accepts
an absolute `--defect-class` path when the manifest recorded a relative one [U] — the first run says.
