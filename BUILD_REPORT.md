# BUILD_REPORT: GitPulse backlog, 2026-10-20/22 (session lappi-decision-0f)

The previous report (2026-10-17..19) is in git at 33adfc7.

Nothing has been pushed. On 2026-10-21 the owner said "commit and merge all and clean branches and next wave". The merges below follow from that instruction. Lanes still in flight merge only once they pass review.

## Method

- **Order.** There is no `AUDIT_REPORT.md`, so the order is priority, then severity, then dependencies. Fable ruled on every sequencing and judgment call (listed below).
- **Roles.**
  - Sonnet subagents implemented the lanes, at most 4 at a time. Twice a resume briefly made it 5; both times I paused a lane.
  - Opus (this session) wrote each brief and reviewed each diff.
  - A separate Sonnet attacked each lane before review.
- **Branching.** Each lane worked in its own worktree on a `lane/<card>` branch. Nothing merged before the owner's yes.
- **Heavy runs.** Every cargo and pytest job went through `tools/mac_heavy.sh`: cargo at `MAC_HEAVY_MAX_LOAD=48`, pytest at the default 12.
- **Load-48 grants.** Fable granted 48 to two pytest baselines, each with conditions: L2's and W3a's.
- **Host load.** Load ran 14-130 for most of the run, from other repos (tessl GPU runs, cargo slots held). Lock waits of 4-10 hours were normal, and several jobs expired unrun (exit 76). Those are reported as not run.
- **DevMap.** It answered status at generation 4003. Every impact and affected-tests walk came back `walk_incomplete` or `fail_closed`, so I read blast radius as unknown, not safe.
- **GitPulse.** Insights and collision risk were deferred under load from wave 2 on (`GAP-WAVE2-COLLISION-RISK-DEFERRED-UNDER-LOAD-2026-10-21`). Lanes owned disjoint files by assignment; that was not a clean scan.
- **`dev verify`: NOT RUN.** No card moved to done. Done is evidence-gated, and that gate belongs to the owner.

## Merged to main

| Merge | Card | What | Verified |
|---|---|---|---|
| 4c30649 | ft-d066187f (L1) | HANDOFF and gap only. The switch onto ojas-train is **blocked**: ojas-train at 4123dfb lacks Lappi's canonical LoRA-A draw order (amendment 4). | Verified by diff, not graph-confirmed. Filed ojas card ft-30dd709e (P0). |
| cdd7c4b | ft-57fc7484 b6, b2 (L4) | `tools/d3_quant_gate.py`: MLX affine fake-quant against mlx's own quantize, tri-state, threshold read from a pre-registration `d3_gate` block. One pinned `QD_PYTHON` resolver in qd-prep, qd-metal, qd-preflight and qd-export, with a pin test. | `make torch-pytest` (3 files) 75 passed. Cargo after-runs: qd-prep 4, qd-preflight 6, qd-metal 11, qd-export 5, all passed with `QD_PYTHON` unset. Fails-before ran for the resolver (preflight 1/5, metal 7/4) and the qd-export timeout (exit 101 at 20.15 s). |
| 9dc5aeb | ft-57fc7484 b5 (L5) | LoRA runs refused before any GPU work. The lora-row gate is in `Row::line_with`: closed label set, value validation, manifest sha. Python parity. The reference driver writes the Rust metric names (Fable). | `cargo test -p qd-train` 276 passed; `qd-train-metal` 49 passed; torch-pytest 318 passed; adversarial driver file 37 passed; 99 committed ledgers (2211 rows) verify. |
| 69127ec | ft-57fc7484 b1, b4, b7 (L2) | The three torch reds fixed at their cause: the reader names `header.json`, and the replay fixture is v6-valid. A headerless directory is refused. 10 gaps re-verified. | Fails-before on d09222d: 1 failed, then 2 failed. After, at load 12: 288 passed, 1 deselected (the MPS leg, not run). |
| f9b9376 | ft-5fdca0a2 (W2c) | Hardening-pass gap triage. Open went from 78 to 47 (the card said 74; the same rule gives 78 at d09222d). | `test_gaps_ledger` green. |
| 4172319 | ft-57fc7484 b3 (L3) and ft-badf64e2 (re-pin 4fba3b4) | Bounded containment scan: packed shingles, batched signing, a collision digest, per-batch canaries. An exact footprint tool (kernel lifetime max per process). | 3+3 interleaved v6 scans on the tip (Fable). All 6 exit 0 under the unchanged 28,672 MiB cap, with outputs byte-identical. After: parent 10.9-12.0 GiB, largest child 3.3 GiB (base parent 20.6-22.6 GiB). |
| 9795832, 1bfdee2 | post-merge | Lint gate green and one dangling gap reference corrected. L3's AUDIT evidence scripts got per-file ignores, by the existing ruling. | `test_gaps_ledger` + `test_lint_gate` 26 passed. `cargo test --workspace --offline --no-run` on merged main built 138 test executables, exit 0. |

Every merged lane was Opus-reviewed after an adversarial pass:

| Lane | Breaks | Outcome |
|---|---|---|
| L4 | 10 | All fixed. Among them: a stale PASS surviving a refused run, could-not-run exiting with FAIL's code, and a fourth `QD_PYTHON` reader outside the pin. |
| L5 | 4 real | All fixed. Among them: metric values were never checked, the gate could be bypassed through `append_line`, and Python accepted what Rust refused. |
| L2 | 2 | Fixed. |
| L3 | 6 | All fixed. Among them: the footprint sampler missed short-lived children, and two distinct shingle sets could silently merge. |

### Benchmark deltas

Only L3 touched a hot path, the v6 containment scan:

- **Memory.** Parent peak about 47% lower.
- **Wall time.** After is **+16.7% on medians, +28.4% min-of-3**. Base runs were 1169.8 / 1316.1 / 2189.6 s; after runs were 1502.1 / 1931.9 / 1535.8 s. The spreads were 77.5% and 28.0% under load 18-51, so **the 5% check is unresolved**, not passed.
- **Measured added cost.** The digest, packing and canaries add about 4%. The remaining gap is in the lsh stage, cause unidentified (code or host).
- **Justification.** Written against real numbers in `HANDOFF/l3-containment-scan-2026-10-20.md`. A quiet-host 3+3 would settle it, and the owner may ask for it.

No other lane touched a hot path. That is DevMap-incomplete and rg-confirmed, not graph-confirmed.

## In flight (code committed on the branch, verification queued behind the lock)

| Lane | Branch | Card | State |
|---|---|---|---|
| W2a | lane/ft-951c360d | ft-951c360d | All five v2 Vani gates written. Fails-before ran for the Python visibility test (ImportError) and qd-prep's gates (0/14). The after-runs are queued. **Merge waits on an owner decision** (below). |
| W2b | lane/ft-c5ea3e71-w2b | ft-c5ea3e71 items 1, 5, 6 | `--write-val-rows` producer (10 tests; 6 failed before, 10 pass after). The export gate asks the runtime's own lookup (fails-before 15/1, after 16). `verify_release_train_manifest` added. The final gaps and lint run is queued. |
| W3a | lane/ft-cc9fac73 | ft-cc9fac73 | All 17 v6-fallout reds diagnosed: 9 are fixtures moved to named legacy-v5 rules, 6 are stale expectations against the recorder's `started` row, 1 is a test-helper fix, and 1 is fixed by L2's merged commit. The after-runs and both flake arms are queued. |
| W3b | lane/ft-5dfe48e7 | ft-5dfe48e7 criteria 1-4 | `qd_serve_launchd.py` (temp dir and fake launchctl only; nothing installed). A mac_heavy training guard, opt-in via `MAC_HEAVY_TRAINING=1`. Mac schedule from measured rates. A cloud cost draft marked NEEDS HUMAN YES. The lock suite on the edited `mac_heavy.sh` is queued. |

Open beyond these lanes:

- **ft-57fc7484.** b5's producer needs ojas-model's LoRA provider to be reachable. b4's span-refit half needs the pinned scorer amended, which is the owner's call.
- **ft-c5ea3e71 items 3, 4, 7, 8.** Item 3 is the owner's. Items 4, 7 and 8 wait on W2a's and W2b's files. Item 8 (hardlinks) was ruled by Fable: refuse `st_nlink > 1`, and measure the on-disk count first.
- **Follow-ups found this run:**
  - qd-runtime admission does not bind family to slot shape (W2b item 5, narrowed).
  - `verify_release_train_manifest` has no caller yet (it goes in qd-release-check after W2a).
  - Training launchers do not set `MAC_HEAVY_TRAINING`.
  - A v6-valid fixture for `test_pre_dedupe_drops`.

## New tasks filed

- **ojas ft-30dd709e (P0).** ojas-train's `lora_adapter_layout` lacks the canonical LoRA-A draw order. It blocks ft-d066187f.
- **Lappi ft-cc9fac73 (P1).** 17 v6-default fallout reds (now lane W3a).

## Fable's decisions this run

1. **Wave 1.** Four lanes with exclusive file ownership. The qd-train moving set was frozen until ft-d066187f, then narrowed to the 8 modules once that card was blocked on ojas. Merges wait for the owner.
2. **Hardlinks (ft-c5ea3e71 item 8).** This is Fable's call, not the owner's: refuse `st_nlink > 1` after measuring the on-disk count.
3. **R2 caps and the Vani pre-registration.** These are owner decisions. An agent may draft them.
4. **Load grants.** L2's baseline and W3a's runs got load 48 with conditions: `QD_PREP_BIN` pre-built outside the lock, a stdlib per-file wrapper (counting from the lock hold), and timeouts reported as not run. The flake's alone arm waits for load below 12.
5. **L3 re-pin.** Running the scan characterization under named legacy-v5 rules is acceptable. The digest re-pin belongs to ft-badf64e2 and lands as its own commit.
6. **L3 evidence.** The sampled-max runs were not acceptance. Fable required 3+3 interleaved runs on the tip with the exact tool, and the 5% check is written against real numbers.
7. **LoRA metric names.** The Rust set is canonical. The driver writes it plus `train_manifest_sha256`. No committed row used the `init.*` names, so no legacy kind was added.

## Owner decisions pending

1. **W2a merge: `qd-release-check` now refuses releases it used to pass.**
   - It refuses a release without a train manifest matching its sha256.
   - It refuses BY-SA, research-only or `vani-agreed` rows unless the release is marked `private`. v5's `squad_v2` is CC BY-SA 4.0, so a v5 or v0.1 re-release must be marked private.
   - The question: merge as written, or scope the rule to v2 pools?
2. **L3's time-for-memory trade** (about +17% wall time for a 47% lower peak), or a quiet-host 3+3 first.
3. **Three adversarial branches** (`adv/ft-57fc7484-l2`, `-l4`, `-l5`). Their tests are on main. The blast-radius gate held `git branch -D` (5 commits would be unreachable): approve, or keep them.
4. **ojas.** The canonical-order fix (ojas ft-30dd709e) must be done from a session rooted in ojas. After it, the Lappi pin bump past 4123dfb needs your yes, and `make pin` is owner-run.
5. **D3 threshold.** Owner item 5: the q8 pre-registration carrying a machine-readable `d3_gate` block. Until it exists, the gate reports NOT_RUN.
6. **b4 span-refit.** Amending the sha-pinned `tools/span_refit_score.py`.
7. **Machine load.** Pausing other repos' heavy jobs would let the four queued lanes verify in hours instead of days.
8. **From earlier runs, still open:** ft-8139b5b1 (push main, held-out ratio, re-export allowlist, unowned worktrees). Item 13 there looks stale: a rebuilt v6 pool passes the length band (W2c).

## Not run

- **On merged main:** the full Python and torch suites. Only the gaps and lint tests and targeted files ran. L3's qd-prep-dependent tests (68 skipped without a binary) are queued as `main-postmerge-l3-tests`.
- **GPU-ignored tests:** `gpu_*` in qd-train-metal and qd-metal, and the MPS legs.
- **Real-model runs:** the D3 real-base verdict and any LoRA run.
- **`make pin` and the qd-pin checks:** the harness refuses paths outside the repo.
- **`dev verify`.**
- **Fails-before not run:** the qd-prep new worktree test, and the `GIT_DIR` and newline resolver hardening (L4).
