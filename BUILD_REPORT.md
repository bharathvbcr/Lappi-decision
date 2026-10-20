# BUILD_REPORT: GitPulse backlog, 2026-10-17/18 (session lappi-decision-10, formerly -34)

The previous report (2026-10-13) is in git at bffcf83.

Nothing has been pushed; local `main` is ahead of `origin/main`. A peer session, `lappi-decision-ce`, ran a hardening pass in parallel on the owner's instructions. Its merges are listed under "Shared main" and its own record is `HANDOFF/harden-2026-10-17.md`.

## Method

- **Order.** There is no `AUDIT_REPORT.md`, so I ordered the work by priority, then severity, then dependencies. Fable ruled on the order at 2733015. Almost every card was already in review. The one workable P1 card was ft-3b8126cd, the Lappi-side prerequisites of the LoRA-2B smoke.
- **Merge rule.** Fable's Q15 makes a fails-before that actually ran a precondition for merging (`AUDIT/harden-2026-10-17/fable-rulings-2026-10-18c.md:9`). Rust stays off `main` until every downstream crate compiles (Q1). Every merge below cites its runs.
- **Heavy runs.** Every cargo and torch run went through `tools/mac_heavy.sh`. Rust tests ran at `MAC_HEAVY_MAX_LOAD=48`, one crate per job (Fable Q8, `fable-rulings-2026-10-18b.md:9`). Machine load from other repos (tessl, GitPulse, Go `-race`) ran at 50–140 for most of the run, so lock waits of 2–8 hours were normal.
- **DevMap.** DevMap answered for status, impact and affected tests (generation 3933 and later). Every impact walk came back `walk_incomplete`, so I read blast radius as unknown, not as safe.
- **GitPulse.** The worktree and collision facets failed under load early in the run ("deferred under load") and recovered later. Two sessions shared this worktree, and I coordinated with the other through `SendMessage` with file claims.
- **`dev verify`: NOT RUN.** No card moved to done; done is evidence-gated, and that gate belongs to the owner.

## Cards moved

| Card | Now | Evidence |
|---|---|---|
| ft-3b8126cd (LoRA-2B smoke prerequisites, Lappi side) | review (revision 4) | Three of its four build items are on main: D17 host objective, LoRA parameter plan with the `"lora"` row, and the torch reference driver. The fourth, the converter, is **blocked** on ojas phase B. The card log lists the open items. |

No other card was moved. The other non-review cards each need an owner decision or a GPU or box run (ft-bd80c52e, ft-12da8e67, ft-afb31d47, ft-be802e07, ft-882ed8fc), or their workable parts were already met. Per Fable's 2026-10-17 staleness check: ft-0f5a10c9's loader half is met by `crates/qd-metal/src/span.rs` and its calibration criterion needs an MPS re-score; ft-ac6677c7 criterion 2 is met by `held_out.rs` and its other criteria are owner or app-side.

## What landed on main (mine)

| Merge | What | Verified |
|---|---|---|
| c324c49 (fast-forwarded to main by peer) | ojas pin 5b67cdd → 04277f8, delta report, LoRA-2B amendment 1 | **Pin verification NOT RUN when committed.** ojas has since moved to 0e95fe7, so the pin gate is red (owner decision 10). |
| fecfd5d | Wave 1. L1: D17 host objective (`d17.rs`, `LetterSpanObjective::with_d17`, torch-oracle fixture). L2: LoRA plan and `"lora"` row (`recipe.rs`, `lora_init.rs`, `ledger.rs lora_ft_row`). L3: torch driver `tools/lora_reference_run.py`. L3 adversarial tests. | `cargo test -p qd-train` 237 passed / 0 failed / 5 ignored. Fails-before: lora_init failed 2, lora_row failed 1. The integrator fixed a 24-ulp RNG mismatch (ATen accumulates in double), a fixture encoding error and a duplicated const. Merged before `qd-train-metal` ran, on the owner's direct "merge all"; it later ran green (37 passed / 0 failed, 14 GPU-ignored). |
| 99eb78b | Gaps and lint fixes from that merge | test_gaps_ledger went from 3 failed to 10 passed; ruff from 32 findings to 2 |
| 0a15658 | W2-1: `Status::Started` in the Rust mirror; `val_without_train` (Fable Q14 P2) and `train_manifest_sha256` as optional row fields | Rust: qd-train 77 passed / 0 failed. Rust fails-before: 1 failed. Python: 50 passed; fails-before: 16 failed. `qd-train-metal` and `qd-metal` compile (`--no-run`, exit 0). Every committed ledger still verifies, and `protocol_hash` is unchanged. |
| f22a5c7 | W2-4: `qd-train-metal` writes `train_manifest_sha256` (hashed before and after the read); `ledger::append_with` is the one locked append | qd-train and qd-train-metal: every target ok. GPU row test 1 passed; GPU fails-before on main's `run.rs`: `left: None`. `qd-metal` compile not run; inferred safe, since the change is an additive public fn and nothing glob-imports. |
| ca4616c, d969810 | A gap record for the LoRA row producer; Fable's W2-2 ruling recorded verbatim | — |
| 08436f8 | W2-2: driver applies v6 data rules and refuses `legacy_v5`; `qd-prereg-bind` at entry, where NOT_RUN refuses; 7 adversarial BREAKs fixed; both B017s pinned | Driver files on the main-merged branch: 60 passed / 1 skipped (MPS) / 1 strict-xfail. Fails-before: 13 failed. Ruff repo-wide: 0 findings. |

Lines: the wave-1 merge added about 33k lines, of which a 25.9k-line JSON oracle fixture is most. Nothing was removed beyond each lane's own replaced code.

### Adversarial outcomes (L3 driver)

- **Held-out symlink bypass:** fixed in the driver. The shared reader `shards.py` still has the hole; the peer's lane W9-c fixes it there and deletes the driver's copy.
- **Output clobbering:** three BREAKs, all fixed.
- **Malformed safetensors:** fixed.
- **Missing ledger row on A-digest drift:** fixed.
- **Loss log not fsynced:** fixed.
- **Read-before-validate, quick/promotion, determinism:** held.
- **Open:** the LoRA-A digest depends on module registration order. It is kept as `xfail(strict=True)` per Fable's ruling (`AUDIT/lora-2b-w2-2-2026-10-18/fable-ruling.md`). That test failure is how this report knows ojas and the oracle register qkv, z, out while real HF registers out first.

## Shared main (peer session, for context)

`lappi-decision-ce` merged: the human ruling (6b0ab85); LoRA-2B amendments 3 (v5r withdrawn) and later; hardening lanes L-data, L-ledger, L-ops and L-release; W2-a/b/c, W3, W4-b/d/e, W5-a/b/c, W6-a/b/c, W7-a/b, W8-a and W9-a. Its record is in `HANDOFF/harden-2026-10-17.md`.

## Benchmark deltas

None measured. No hot path was benchmarked. Fable Q17 rules timings taken under load 60–90 invalid as an A/B. `uniform_f32` changed precision, but that was a correctness fix, not a performance change.

## New tasks and gaps filed

- GAP-LORA-FT-ROW-HAS-NO-PRODUCER-AND-MUST-CARRY-TRAIN-MANIFEST-SHA-2026-10-18 (open)
- GAP-W2-2-LORA-A-DIGEST-DEPENDS-ON-MODULE-REGISTRATION-ORDER-2026-10-18 (open; owner amendment)
- GAP-LEDGER-TRAIN-MANIFEST-SHA256-FIELD-LANDED-PRODUCERS-NOT-WIRED-2026-10-18 (closed by W2-4 and W7-b)
- GAP-STARTED-STATUS-RUST-MIRROR-RESOLVED-CAMPAIGN-DRIVER-CORRECTED-2026-10-18 (resolved)
- The lane gaps GAP-L1-*, GAP-L2-*, GAP-L3-* and GAP-W2-2-*, with closures appended

No new board cards: every finding folded into an existing card or into a peer lane (W9-c).

## Fable's decisions this run

1. **Sequencing at 2733015.** Items 1–3 workable; the converter blocked; waves split by file.
2. **Q8 (load).** Serialized `cargo test -p` at load 48; pytest and torch stay at 12.
3. **Q15 (merge rule).** A fails-before that actually ran is required.
4. **W2-2.** The A-digest order is a rule defect: amend `lora-2b-preregistered.json:169` to layer index, then suffix order. Merge with a strict xfail meanwhile. The `shards.py` fix belongs to its owner and is not stop-ship.

## Owner decisions pending

1. **The ojas pin.** The sibling is at 0e95fe7 and the lock pins 04277f8, so every pin step is red. Bump with a delta report? (Also HANDOFF item 10.)
2. **The LoRA-A canonical order amendment** to `campaign/lora-2b-preregistered.json:169`. It must be committed before any LoRA-2B result is read.
3. **Re-export of existing models.** Since W7-a, export refuses ft rows without `train_manifest_sha256`, which covers v0.1, v5, the averaged models and the ft-0f5a10c9 span re-export. Allow them through an owner-signed (row id, manifest sha256) allowlist? (Also HANDOFF item 3.)
4. **Push.** `main` is far ahead of `origin` on a public repository, and nothing is pushed.
5. **Unknown-owner worktree** `build/main-gates-2026-10-15` (detached, 3 dirty files): keep or remove?
6. Still open from 2026-10-17 (Fable's final ruling §6): the quantised arm, 4-bit, extra Mac GPU time and peft.

## Addendum, 2026-10-19

- **W2-5** was merged as e8aade9 on the owner's "merge all". `tools/gen_ledger_schema_fixture.py` now hands RunRecorder its entry point, and its recipe spelling is pinned in `test_tool_call_sites.py`, with no exemption list.
  - Fails-before: 2 failed / 19 passed.
  - After: 48 passed.
  - Rust `cargo test -p qd-train --test ledger_schema` on main after the merge: 9 passed, 0 failed (main-w2-5-ledger-schema, 16:10Z).
- **W3-1, HELD**, branch `w3-1-lora-a-canonical-order-2026-10-19` (200a2d2, be52013).
  - It makes the LoRA-A draw order canonical in `lora.py` and `lora_init.rs`/`recipe.rs`/`ledger.rs`, and removes the strict-xfail.
  - The committed oracle digests are unchanged, because that fixture already uses the ojas order.
  - A model registered in HF order now matches the ojas-order digests at every seed.
  - **No test has run**: mac_heavy refused both cargo attempts at 8 GB free (exit 74).
  - It does not merge until the owner approves the amendment drafted in `AUDIT/lora-2b-w2-2-2026-10-18/amendment-draft-lora-a-order.md`.
- **Cards to review, with Fable's staleness findings recorded:** ft-0f5a10c9 and ft-ac6677c7. Neither has a criterion an agent can work.
- **Branches:** every lane branch of mine is merged and deleted, except W3-1, which is held.

## Wrap-up, 2026-10-19

**Owner decisions taken** (`AUDIT/lora-2b-w2-2-2026-10-18/owner-decision-2026-10-19.md`):
- The LoRA-A canonical-order amendment.
- The ojas pin to 4123dfb (phase A+B).

**What merged:**
- **0089cee:** LoRA-2B amendment 4. `init.lora_a.rule` now names the canonical order: layer index, then the suffix table.
- **d74bc85:** W3-2, the ojas pin 04277f8 → 4123dfb.
  - `qd-train`: 260 passed, 0 failed. `qd-train-metal`: 41 passed, 0 failed. `qd-metal` and `qd-export` compile (`--no-run`).
  - No Lappi source changed. `Cargo.lock` gains `ojas-train`.
  - **Not run:** `make pin` and the `qd-pin` checks. The harness refuses paths outside the repo, so the owner has to run those commands; they are in `AUDIT/ojas-pin-4123dfb-2026-10-19/delta-report.md` §4 and tracked as GAP-OJAS-PIN-4123DFB-SOURCE-DIFF-AND-PIN-GATE-PENDING-2026-10-19.
- **a5d1881:** W3-1, the canonical LoRA-A draw order in Python and Rust.
  - `qd-train`: 265 passed, 0 failed. `qd-train-metal`: 41 passed, 0 failed. Torch, on the merged tree: 91 passed, with the formerly strict-xfail order test now passing.
  - Fails-before: `main`'s `lora.py`, loaded by an explicit path, gives different digests for the two registration orders, and the new code gives equal ones.

**Card ft-3b8126cd:** ojas phase B now ships adapter checkpoint layout v1 (ojas 3f87e76). The blocked converter item is therefore workable in a new session. It was not started, per the owner's wrap-up instruction.

**Branches:** every lane branch and auto-branch of this session is merged and deleted. Left on purpose:
- `span-pin-8a531e8-verdicts`, still needed by card ft-0f5a10c9;
- `build/main-gates-2026-10-15`, whose owner is unknown, with 3 dirty files; the owner decides.

Nothing has been pushed; `main` is 510 commits ahead of `origin`.

## Not run

- `make pin` and `test_pin_gate` on main: red on the stale pin.
- The full Python and torch suites on the final main.
- Any real 0.8B or 2B model run.
- `qd-metal` tests: only `--no-run` was run for W2-1, and nothing for W2-4.
- `dev verify`.
- The v6 LoRA-2B shards are not built yet; the peer's W2-d lane builds them and writes amendment 4.
