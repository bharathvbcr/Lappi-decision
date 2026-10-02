# Needle and OOD failures on the phase-3 checkpoints — follow-up lane, 2026-09-30

Lane: wire and score every gate on the three saved phase-3 checkpoints, then act on the two
that failed (Fable decisions, approved by the user 2026-09-30). The GH200 (1×, $2.29/h) is
still rented. The user shuts it down; no agent terminates it.

## What was measured (ledger rows)

**All-gates re-scores**, fp32 on cuda at `63f5b68`, in `ledger/gh200-allgates-2026-09-30.jsonl`:

| Seed | Row | permutation_consistency | ece | needle_hunk_recall | ood_abstain | degenerate_head |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `58fd1532` | 2330/2332 pass | pass | FAIL: worst 0.000; 2/300 | FAIL: 22/180 | fail |
| 1 | `116d2af2` | 2331/2332 pass | pass | FAIL: worst 0.197; 64/300 | FAIL: 24/180 | fail |
| 2 | `efed7b19` | 2330/2332 pass | pass | FAIL: worst 0.000; 4/300 | FAIL: 12/180 | fail |

- **The OOD abstentions are all prose.** The scrambled and unseen-language categories abstain on 0–1 of 60. In-distribution abstention is 1–2 of 2,332.
- **The Mac re-score agrees with seed 0.** Row `784868b3` (seed 0, quick, MPS with the pure-torch fallback) in `ledger/mac-score-ckpt-allgates-2026-09-30.jsonl` matches `58fd1532` case for case: 235 needle cases abstained, 22 prose abstentions, 2 in-distribution. There is no device or kernel cause.

**Needle length control** — `ledger/gh200-needle-control-2026-09-30.jsonl`, row `5b22eeb8` (quick, its own family, at `7d166b0`):

| Target | Real tokens | Hunks per case | Worst bucket | Aggregate | Abstained |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1,156–1,373 (inside the 1,625 trained width) | 14–24 | 0.000 | 0.030 | 238/300 |
| 2048 | 2,053–2,410 | 26–44 | 0.000 | 0.010 | 239/300 |
| 4096 | 3,911–4,593 | 51–81 | 0.000 | 0.003 | 235/300 |
| 8192 | 7,577–8,821 | 100–158 | 0.000 | 0.007 | 235/300 |

- **The 8192 arm is the gate suite** (digest-checked) and reproduces `58fd1532` case for case.
- **The misses land far from the needle.** Raw verdicts are in `/Users/bharath/qd-campaign/gh200-needle-control-2026-09-30/suite-verdicts-s0.jsonl`. Of the cases that did not abstain, nearly every miss falls before the needle hunk, scattered over the fillers rather than at ±1:

  | Arm | Answered | Before the needle | After | Adjacent (±1) | Hit |
  | --- | --- | --- | --- | --- | --- |
  | 1024 | 62 | 50 | 3 | 1 | 9 |
  | 2048 | 61 | 57 | 1 | 3 | 3 |
  | 4096 | 65 | 63 | 1 | 1 | 1 |
  | 8192 | 65 | 62 | 1 | 0 | 2 |

  So it is not a mapping defect. It reads as the model pointing at the first defect-looking hunk; that is an inference, not tested.
- **Conclusion: length is not the cause; structure is.** Recall is ~0 even inside the trained width. A long-context stage alone will not move this gate.

## What changed (commits, local only)

| Commit | Change |
| --- | --- |
| `9955843` | `optimizer_spec`: a bf16 score of a master-trained checkpoint gets the master layout |
| `8b8678b` | promotion reads an eval row and the control rows naming it (`recipe.eval_row_id`) as one unit |
| `444bedc` | raw needle and OOD verdicts (`--suite-verdicts-out`); `--verdicts-out` carries `row_logits` |
| `7d166b0` | `--needle-control`: the gate's rule at other lengths, on a quick row of its own |
| `56b737f` | `qd-margin-probe` (Rust, in `qd-runtime`): margins via `calibration::calibrate`, a `noul_margin` fit on a val half, OOD/val abstention deltas |
| `a15f559` | gap records (below) |
| `0f4a18b`, `a151483` | ledger copies |
| `f011beb` | ruff E501 ×2; the row-writer census names `run_needle_control` |

- **Full suite on main at `f011beb`:** 2,708 passed, 14 skipped, 6 deselected. The 6 are the MPS tests, deselected because a peer session holds the Mac GPU; they were not run.
- **Failures before `f011beb`:** two, both fixed in it.
- **Cargo:** every workspace test passes.

### Building Rust for the box

The box has no Rust toolchain, and installing one would be a download. Instead, binaries are cross-built on the Mac:

- **Sysroot:** the box's own glibc 2.39 and gcc runtime files, copied from it, at `/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/`.
- **Linker:** `link.sh` (Apple clang driving rustup's bundled lld).
- **Verified:** `qd-margin-probe` built this way runs on the box.

    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" \
    cargo build --release -p <crate> --bin <bin> --target aarch64-unknown-linux-gnu \
      --target-dir /Users/bharath/qd-campaign/target-aarch64-linux

### Box layout

- **The lane checkout is `/home/ubuntu/qd-lane`.** It is a clean clone, with the main checkout's git-ignored `data/` files symlinked in (ignored, so the tree stays clean). Its runs therefore record a clean `code_commit`.
- **`/home/ubuntu/qwen-decision` stays at `63f5b68`** for the running OOD probe.

## In flight when this was written

- **OOD transfer probe.** `/home/ubuntu/box_ood_probe.sh` trains the phase-3 recipe plus clinc only, one seed, then scores with every gate.
  - Fetch record: `fetch-record-clinc-only-2026-09-30.json`.
  - Ledger: `/home/ubuntu/ledger/gh200-ood-probe-2026-09-30.jsonl`.
  - Its val set includes clinc val rows, so its `ood_abstain.in_distribution` population is not the phase-3 one.
- **Margin re-score.** `/home/ubuntu/box_margin_rescore.sh` re-scores the seed-0 checkpoint, fp32, with `--ood --verdicts-out --suite-verdicts-out`.
  - Outputs: `/home/ubuntu/margin/`.
  - Ledger: `/home/ubuntu/ledger/gh200-margin-rescore-2026-09-30.jsonl`.
  - These are the inputs `qd-margin-probe` reads.
- **Two Rust performance lanes.** Branches `perf-prelude-rust` (the scoring prelude) and `perf-pipeline-rust` (shard build), in agent worktrees.

## What is open (gap ids)

- `GAP-NEEDLE-FAILS-AT-8K-CAUSE-UNRESOLVED`: now answered as structure, not length (`5b22eeb8`). Revise it when the phase-4 data decision lands.
- `GAP-NEEDLE-FILLERS-ARE-NOT-BEHAVIOUR-NEUTRAL`: a suite-design question for the human.
- `GAP-OOD-ABSTAIN-FAILS-WITHOUT-LETTER-NOUL-SUPERVISION`: waiting on the clinc transfer probe.
- `GAP-PAIRED-MARGIN-IS-SCORED-ON-VAL-NOT-THE-NATURAL-HELD-OUT-SET`.
- `GAP-FT-EVAL-COMPUTES-NO-SHUFFLED-LABEL-PRIVILEGED-HUNK-OR-TRANSFER-CONTROL`.
- `GAP-ALLGATES-EVAL-ROWS-HAVE-NO-LINEAR-CONTROL-ROW`.
- `GAP-DEVMAP-NOT-USED-NEEDLE-OOD-LANE-2026-09-30`.

## First command for the next lane

Once `/home/ubuntu/margin/` holds both verdict files, copy them home and run the probe:

    scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246:/home/ubuntu/margin/*.jsonl /Users/bharath/qd-campaign/margin-2026-09-30/
    target/debug/qd-margin-probe --val /Users/bharath/qd-campaign/margin-2026-09-30/verdicts-s0.jsonl \
      --suite /Users/bharath/qd-campaign/margin-2026-09-30/suite-verdicts-s0.jsonl \
      --split-key margin-probe-2026-09-30 --out /Users/bharath/qd-campaign/margin-2026-09-30/report.json
