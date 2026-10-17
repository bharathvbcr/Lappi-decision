# HANDOFF ojas-train move, Lappi side, 2026-10-14 (lane: gitpulse-backlog-run)

Spec: `AUDIT/ojas-train-move-2026-10-14/fable-ruling.md` (Fable, on the user's "choose best"), built on
`AUDIT/ojas-train-move-2026-10-14/delta-report.md`. This file closes steps 1-4 and 6 of the ruling's
"Lappi session does now" list. Step 5's push is blocked (below). Labels: [V] run or read here.

## Start sha for the ojas lane

Lappi `main` at **30a069d** (local; not on origin, see "Push"). Fetch it from the local checkout:
`git fetch /Users/bharath/Code/research/Lappi-decision main`.

## What was measured

No ledger row: nothing in these steps trains or scores. The evidence is test and gate output on main [V]:
- `cargo test -p qd-train -p qd-train-metal -p qd-metal` (QD_PYTHON=/Users/bharath/.venvs/ml/bin/python):
  310 passed, 0 failed, 37 ignored. The 37 are the GPU/Metal `--ignored` suites: **not run**.
- `make fmt`, `make clippy`, `make pin` (~~pin PASS at ojas bc498c4~~) exit 0; `cargo doc -p qd-train` with
  `-D warnings` exits 0.
  **Correction 2026-10-17 (lane lora2b-audit2-2026-10-17):** the struck text was true on 2026-10-14 and is
  stale: `upstream.lock.json` was bumped bc498c4 -> 132b46e in 21fb7ad (2026-10-15, owner approved) and
  132b46e -> 5b67cdd on this lane's branch (owner's "procced with fable's advice", 2026-10-17, ruling
  section 6 item 1), with `make pin` PASS re-run at 5b67cdd. Between bumps `make pin` printed `pin FAILED`
  against ojas main, as designed (audit B, `AUDIT/lora-2b-audit2-2026-10-17/B-lappi-side.md` section 4).
- Moving-set import gate, empty: `rg 'crate::(shards|objective|span_head|ft_data|held_out|ledger|export)'`
  over `crates/qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson}.rs`.

## What changed (commits on main)

| Step | Commit | What |
|---|---|---|
| 1 quiesce | (message) | `ListAgents`: one peer (lappi-decision-9e, idle), told the claim on `crates/qd-train*`, `crates/qd-preflight`, `Makefile`, `upstream.lock.json`; `gitpulse_collision_risk`: no overlap on those paths |
| 2 pin | 7362473 | `upstream.lock.json` (ojas `bc498c4375c2e4753d784e6e9fc267a8d18b038a`); `qd-pin` in `crates/qd-preflight` (exit 0 / 1 / 3); `make pin` in `make gates`; `qd-train-metal` refuses another ojas commit unless `--unpinned-ojas`, which marks the row `unpinned ojas` |
| 3 card item 4 | 38db2ea, 732fede | `ResumeState::advance_to_epoch`; the consumed digest chains across epochs (`ConsumedPrefix::continuing`); epoch 0 bytes unchanged; `read` refuses an epoch / `epoch_link` mismatch |
| 4 seams | 96a732a, 30a069d | `ConsumedPrefix` in `run_control`; a shard batch folds with `Batch::fold_into` (an inherent impl on `ConsumedPrefix` would not compile once it is foreign); hand-written error impls replace `thiserror` in the moving set (13 message literals verified verbatim) |
| 6 costing | 03a6b43 | `AUDIT/ojas-train-move-2026-10-14/host-adamw-contingency.md`, doc only |

The moving set is `{step, trainer, schedule, recipe, adamw, mock, run_control, pyjson}`. It imports nothing
Lappi-owned. `thiserror` is gone from it; `serde_json` and `sha2` are still used (Q4: ojas-side work).

## What is open

- `GAP-LAPPI-PINS-NO-OJAS-SHA-2026-10-13`: resolved with residual. No ledger row names an ojas sha, so the
  baseline is verified only by the GPU rerun (ruling step 8).
- `GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01`: open until a step with moments is measured.
- Card ft-3163ce item 4: trainer-level done; ~~`qd-train-metal`'s `run()` still trains one epoch (records
  `passes: 1` honestly). A multi-epoch CLI is not wired.~~
  **Correction 2026-10-17 (lane lora2b-audit2-2026-10-17):** stale the same day it was written. Commit
  6f88a00 ("qd-train, qd-train-metal: multi-epoch driver and --epochs (card ft-3163ce item 4)") wired it:
  `--epochs N` and `--resume CKPT_DIR` in `crates/qd-train-metal/src/args.rs` (usage at :214, parsing at
  :245 and :484), `run.rs:340` calls `trainer::train_epochs`, and `tests/run_door.rs` covers resume under a
  changed recipe and a different epoch count [V read 2026-10-17]. The pin bump and the `ojas-train` switch
  (ft-3163ce criterion 8) remain open.
- **Push:** main is not on origin. GitHub push protection refused it over fake AWS-shaped test fixtures in older
  commits; the owner must allow them with the two unblock links, then `git push origin main` (no force).
- **Owner questions:** `sha2 = "0.11"` in ojas (unanswered, so the ojas lane hand-rolls SHA-256 per Q4); the
  later pin bump `bc498c4` -> the ojas-train sha (delta first, then the owner's yes); the 47 GB precondition.

## First command for the next lane (a session rooted in ojas, on card gp-ft-trainer-loop)

`ListAgents` in ojas, then `git fetch /Users/bharath/Code/research/Lappi-decision main` and confirm
`git merge-base --is-ancestor 30a069d FETCH_HEAD` exits 0 (this file's own commit sits on top). Then phase A
of the ruling's "Handed to ojas" section.
