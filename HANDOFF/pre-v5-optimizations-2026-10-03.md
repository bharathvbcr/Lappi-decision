# Pre-v5 optimization review across six dimensions (lead, 2026-10-03)

The human asked (~12:40Z): "Consider the model size, training data, training pipeline, inference
pipeline, functions, and role of the model. And ask fable and work on each optimizations before a
train."

**Fable's ruling (relayed in the session):** split the work by what each change touches.

1. **Anything that changes what v5 measures** is an amendment to a pre-registered experiment, not
   an optimization: model size, data content, recipe flags, gate or control definitions. v5
   carries only the conditionals its own rows decide:
   - C1 (decided by j6g);
   - C2a (nomask);
   - C2b (tierb2);
   - C3 (fsucc).

   Reopening any of this is the human's call, to be asked rather than assumed.
2. **Pipeline speed with measured quantities unchanged** is the work to do now, from measurement.
3. **Inference fixes** sit in files another lane holds uncommitted.

## The table

| Dimension | Fixed by pre-registration | Open, and who owns it | Evidence |
|---|---|---|---|
| **Model size** | Qwen3.5-2B-Base (24 layers, hidden 2048, vocab 248,320), the v5 DRAFT's recipe base | The human, if they want it reopened: a different size, or the cheaper ladder rungs 1–2 (`docs/lappi.md`, "The ladder") | Size is not what the failing gates point at. Refusal moved from 3–8 to 56–58/60 on scrambled at the same size when data was added (v3→v4). The code family is at 99.4% permutation consistency. The general families are bound by the base (MMLU 0.61, CSQA 0.76; `docs/lappi.md`, "What the 2B campaign has taught") |
| **Training data** | v5's sources (DRAFT `data.sources`): prose-noul 6,010; G6 real unseen languages; CLINC re-key; composed rows | None for agents. The "more data" contingency is pre-registered (v5-build a14ba6f). About 1,963 own-prose units remain, above the 1,500 floor | `AUDIT/finalize-2026-10-03/fable-data-and-promotion-ruling.md`; `promotion-decision-brief.md` |
| **Training pipeline** | Recipe flags. C2a (no-mask, −20/−40% train time) and C2b (fused AdamW) are decided by their own rows | **Fixed:** the whole-plan floor evaluation, about 57 min per epoch-arm run (below). **Open, lead:** the preamble's GPU idle, which needs a profiled box launch (human yes); and the decode at ~25% GPU | `GAP-EPOCH-ARM-EVALUATES-THE-WHOLE-TRAIN-PLAN-FOR-NO-ROW-2026-10-03`, `GAP-BOX-GPU-IDLE-REAL-FT-RUN-PREAMBLE-2026-10-03`, `GAP-SCORE-VAL-DECODE-RUNS-AT-A-QUARTER-OF-THE-GPU-2026-10-03` |
| **Inference pipeline** | — | **H1** (quadratic `encode_untrusted`): a patch is ready out of tree. It is held for the human's word, because main's `tokenizer.rs` is dirty with uncommitted work whose owner is unknown (mtime Oct 1 10:38). **M1, M2:** on hold. **qdm-digest-parallel → main:** waits on the human's direct sha2 0.11 yes, a clean main `Cargo.lock`, and v5 binaries | Bench patch `~/qd-campaign/lappi-bench-2026-10-03/h1/h1-tokenizer.diff` (sha256 bfd1bd16…). **The bench's report; the lead did not re-run it:** fail-first 12.42 s against a 2 s bound; fixed, about 21 ms; qd-metal 46 passed / 0 failed / 8 ignored. The bench lists as unverified: the real Qwen tokenizer was not exercised, debug profile only, the backend.rs callers were not run, clippy was not run. `AUDIT/mac-inference-audit-2026-10-03/relayed-findings.md` |
| **Functions** (slots, abstention, calibration) | The `noul` row, the slot kinds, and the gates as built | **Human:** the six open items in `docs/promotion-decisions.json`. **Human:** the conformal fit's coverage target against the in-distribution cap. As fitted it would abstain on 4,454 of 10,985 val rows (ab966074), and must be reconciled before any export ships a table | `AUDIT/finalize-2026-10-03/promotion-decision-brief.md` |
| **Role** (DevCouncil and DevType call the code decision) | — | **Human:** `promotion_population`, row 1 of the brief | Same brief |

## What was measured (read-only, box and ledger)

F seed 0 on the GH200 (`/home/ubuntu/logs/gpu.csv` at 15 s, file mtimes, and
`/home/ubuntu/p4-v4/train-s0.log`):

| Phase | Time | GPU |
|---|---|---|
| Startup (qd-prep minhash/LSH 53 s, then Python) | 17:55–18:01 | 0% |
| Train loop: 9,683 steps at 1.65 s/step, 20,175 positions/s, peak 61.6 GiB | 18:02–22:28 | 97–99% |
| Checkpoint written | 22:29:12–22:29:24 | — |
| Whole-plan floor evaluation (attribution inferred, not profiled) | ~22:30–23:26 | 94–100% |
| In-process scoring: val decode, permutation pass, needle, OOD (eval row `wall_clock_s` 578 s) | ~23:27–23:36 | ~25% |
| Needle length control | 23:40–23:43 | ~100% |

All five F seeds, ft row to eval row (`ledger/gh200-p4-v4-2026-10-01.jsonl`):

| Seed | ft row | eval row | train s | ft→eval s | scoring s | rest s |
|---|---|---|---|---|---|---|
| 0 | 973cd4e3 | f4feac15 | 16,024 | 4,052 | 578 | 3,475 |
| 1 | 95fa4854 | aeca8d69 | 15,981 | 4,062 | 586 | 3,477 |
| 2 | 32990e1a | 8c3a774a | 15,974 | 4,067 | 591 | 3,476 |
| 3 | 1845f471 | ebc83be6 | 15,982 | 4,080 | 601 | 3,479 |
| 4 | 62a11100 | d8c8300a | 15,989 | 4,083 | 605 | 3,478 |

## What changed

- **v5-build 6f1546c** (`tools/real_ft_run.py`, `tools/perf_parity.py`, two test modules, the
  DRAFT):
  - The epoch arm no longer runs `_evaluate`. The memorise arm and `perf_parity` still do.
  - `perf_parity`'s `evaluate_plan=True` is covered by import only: no test calls its `_train`
    path.
  - No recipe key moves. A forced on/off comparison shows no gate, control or metric changes.
  - Fail-first against a14ba6f: 2 failed, 1 passed (`build/evalplan-failfirst.log`).
  - After the change: the four direct modules gave 144 passed, 1 skipped
    (`build/evalplan-post.log`); the other 37 modules importing the tools gave 733 passed,
    26 skipped (data not on this host, or opt-in), 0 failed (`build/evalplan-blast.log`).
- **Saving:** about 57 GPU-min on each of v5's 9 epoch-arm runs. That is about 8.6 GPU-h, about
  $20 at $2.29/h, and about 8.6 h of wall clock, out of about 50 h. Seeds 3–4 would add 1.9 h if
  they run.
- **Already queued box runs keep paying it:** j5p, fsucc's F′ ×3, j5pp, j6a and j6g. Launched
  waiters are not edited.

## What is open

- `GAP-BOX-GPU-IDLE-REAL-FT-RUN-PREAMBLE-2026-10-03`: about 6 min per job. Needs a profiled launch
  (human yes).
- `GAP-SCORE-VAL-DECODE-RUNS-AT-A-QUARTER-OF-THE-GPU-2026-10-03`: about 10 min per scored seed.
  Any change needs a verdict-identity check on a real checkpoint.
- `GAP-DEVMAP-DATABASE-MALFORMED-2026-10-03`: every call-site claim this turn is from rg.
- H1/M1/M2, held for the human (see the table).
- Whether the human's message reopens model size, data or recipe.

## First command for the next lane

```bash
bash tools/mac_heavy.sh evalplan-recheck bash build/v5_build_pytest.sh python/tests/test_real_ft_shuffled_label.py -k whole_plan_evaluation
```
