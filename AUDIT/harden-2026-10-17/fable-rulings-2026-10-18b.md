# Fable rulings, 2026-10-18 (second round)

Asked by session lappi-decision-ce after the owner said again "Commit and merge all changes and next wave; for advice ask fable". Fable answered read-only at about 01:40Z. The body below is Fable's answer, copied verbatim. The first round is AUDIT/harden-2026-10-17/fable-rulings-2026-10-18.md.

Read-only, under the standing delegation. Navigation this round: `rg`/read only; DevMap not asked (the deferred tools were not loaded). Nothing below is graph-confirmed. State at 01:38Z: lock held by gdn-mem pid 97046 since 22:08Z, still waiting (gdn-mem.log 01:34Z: load 48.28); `vm.loadavg` 71/78/103; 18 cores, 64 GiB, `kern.memorystatus_vm_pressure_level` 1 (normal).

## Q8. Load stall: (b) narrowly, and yes, the lock-while-waiting is a defect

**Decision.** (b), bounded: after gdn-mem's lock frees, one serialized `cargo test -p <crate>` per lane under the lock with `MAC_HEAVY_MAX_LOAD=48`, `CARGO_BUILD_JOBS=2` (the default), footprint cap untouched, and the raised value on the recorded launch line. Not for pytest/torch suites or data builds. gdn-mem itself needs nothing: `MAX_WAIT` 14400 s from 22:08:21Z makes it exit 76 at 02:08Z and release.

**Reason.** 12 has no derivation anywhere (`rg MAX_LOAD` over AUDIT/HANDOFF/docs/gaps: only citations). mac_heavy.sh:7 names the 350-load panic; :27-29 says the second panic was at load ~30 with one job under the lock, caused by 36-120 GiB processes, i.e. memory, which the footprint cap (:30-47) and memguard (:48-57) now bound and the load gate never did. 48 is under 3 per core and is the load at which gdn-mem's own log last stood. A `cargo test -p` at `-j 2` adds about 2-4 to load; the storm is the sibling repos' and no Lappi setting reaches it (GAP-MAC-HEAVY-LOCK-AND-LOAD-GATE-BIND-LAPPI-ONLY-2026-10-18). Memory pressure is normal, so the guarded failure mode is not present.

**Defect.** mac_heavy.sh:164-180 takes the lock, then :270-286 waits on load holding it, so a job blocked by the machine blocks every other job for up to 4 h, and a later job with a different `MAX_LOAD` cannot run. Fix the structure: poll load *before* `mkdir`, re-check once after acquisition, and on busy `release()` and return to the outer loop; a job holds the lock only while it runs. Ship with a test in `python/tests/test_mac_heavy_lock.py` that a waiter at load > MAX_LOAD does not own `$LOCK/owner`. Install by temp file + `mv` (:72-77).

## Q9. Do not reinterpret v1 headers; fix exclusions.py the same way, and fix the Rust reader first

**Decision.** No to v1-on-16. Keep GAP-W4C-V1-HEADERS-STILL-REFUSED-UNDER-V1-RULE-2026-10-18 human-owned (rebuild or `allow_stale_code`).

**Reason.** `SHARD_CONTENT_MODULES` (fingerprint.py:76-93) is the import closure *at HEAD*. A v1 header did not record the closure at write time; a module that shaped rows then and left the closure since is invisible to the 16-list, so the comparison is sound only under an assumption the header cannot attest. 0 of 89 verify either way, so adopting it buys nothing and sets a precedent of reading a record under a later rule.

**Defect found (read, not run).** W4-c (0d498d2) changed no Rust. `crates/qd-train/src/shards.rs:699` still hashes every `*.py` and :700 `describe_drift` reports the unlisted modules as "added since" (:622-626); the header parser (:320-353) never reads `code_fingerprint_version`. Every v2 header the Python writer emits (shards.py:1805-1806) is refused by qd-train and qd-train-metal unless `allow_stale_code`. Highest-priority item in Q12.

**exclusions.py.** Yes, same class, not the same list: `containment_corpus` (exclusions.py:115) hashes `code_fingerprint()`, so the closure of *its* entry points (exclusions.py:53-56: config, rows, split, plus their imports) is the right set, derived and checked by the same AST test, with its own version field. Re-pinning: the digest in HANDOFF/prep3-2026-10-02.md and HANDOFF/v5-freeze-2026-10-03.md records v1 under the then-code; it already fails on any edit, and a v2 attestation is a new record, not a moved one. Nothing to re-pin; v1 attestations stay v1 (GAP-W4C-CONTAINMENT-ATTESTATION-STILL-FINGERPRINTS-WHOLE-QD-DATA-2026-10-18).

## Q10. W4-d histogram: nearly, one key short

**Decision.** Sufficient to revise K's magnitude; insufficient to revise K's *shape* (per-class vs. total) because `before`/`after` are keyed by `cluster_key` alone (generate.rs:124-125, :143, :152 on w4d-mutate-calib-2026-10-18) while the cap binds per `(key, class)` (:146). A function with 8 stub rows and one with 2 of each class land in the same bucket. Add `rows_per_key_class_before: BTreeMap<u64,u64>` from `kept`'s pre-cap counts; bounded like the others (≤ 33 buckets by the 32-per-file cap). `dropped_by_class` (:172) and `keys_seen` are right. Revising K stays a human step (GAP-W4D-MUTATE-CAP-K-AND-FIXTURES-2026-10-18).

## Q11. Keep 0.94; it is a refusal threshold, one-way under rule 2

**Decision.** Keep. It may be tightened by anyone; loosened only by the human citing a new OOM row. Not a promotion gate, but rule 2's spirit covers kill and admission criteria, and loosening admits runs that die without a ledger row.

**Reason.** memory.py:114-121: two OOMs, one device, one allocator; the predictor says so and requires a real step (gh200_rows.py). More measurement is the next rented box's first job (one more width), not a precondition. On a device other than that GH200 the fraction is an inference: `refuse_if_over_device` should print "fraction measured on GH200 only". GAP-MEMORY-PREDICTOR-UNDER-PREDICTED-EVERY-RECORDED-GH200-FOOTPRINT-2026-10-17 can close; GAP-MEMORY-HOST-CEILING-CHECK-COVERS-ONLY-TORCH-MPS-ADMISSION-2026-10-17 stays open.

## Q12. Next wave, ranked

1. **Rust shard reader refuses every v2 header** (Q9). Certain at the first qd-train open of a rebuilt set. Add `code_fingerprint_version` to the Rust header, branch on it, test: a v2 header opens; a v1 header still uses rule 1.
2. **Compile the held lanes.** W2-a (17cc878), W4-d (e66e576), W4-e (bc64367) are unbuilt; W4-d is the v6 data build and W4-e the release gate. Under Q8(b) after 02:08Z.
3. **`started` row unknown to the Rust mirror.** `rg Started crates/qd-train/src/ledger.rs` is empty on main and on wave-integration-2026-10-17 (GAP-STARTED-STATUS-UNKNOWN-TO-RUST-LEDGER-MIRROR-AND-CAMPAIGN-DRIVER-2026-10-17). Certain.
4. **Push main.** origin/main is 271 behind; every isolated lane cuts from it (GAP-AGENT-WORKTREE-ISOLATION-CUTS-FROM-STALE-ORIGIN-MAIN-2026-10-17). One command.
5. **mac_heavy lock-while-waiting** (Q8) plus the three unguarded paid-box launchers: `campaign/post-f-queue/{post_f_common,v5_common}.sh`, `tools/perf_tierb_outcome.sh` have no `require_driver` (GAP-LAUNCH-SCRIPTS-RUN-A-PAID-BOX-OUTSIDE-THE-DRIVER-2026-10-17).
6. **exclusions.py closure fingerprint** (Q9) and the W4-d per-class histogram (Q10). The release-check seam (GAP-QD-EXPORT-NO-TRACKED-PER-FAMILY-SLOT-SHAPE-REGISTRY-2026-10-17, GAP-QD-EXPORT-ENSEMBLE-TABLE-NOT-GATED-2026-10-17, GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17) stays last: release-time.

**Handoff.** Write HANDOFF/harden-2026-10-17.md now; none exists (`ls HANDOFF | rg harden`: only data-harden-2026-09-19). Rule 10: measured (no ledger rows: say so; the pytest counts with commit), changed (merge commits fd0a9d6, fa2e863, c115227, af4bf6d, 01d1ee3, df265a2), held (the three unbuilt tips with the reason), open gap ids (items 1-6 above, in full), and the first command: `bash tools/mac_heavy.sh w4e-cargo env MAC_HEAVY_MAX_LOAD=48 cargo test -p qd-export` after 02:08Z, preceded by `ListAgents`.

## Q13. commitpackft content overlap: partition by repo, refusal on in v6, human signs the ratio

(Asked as a follow-up after W3-a's report; Fable's answer, verbatim.)

**Decision.** Make the partition; it is the human's, not mine. Rule 2 names held-out sets, and this re-populates one: `code.language_id` rows would come only from the repos the hash assigns to it, so its row count and composition move. The SQuAD precedent agrees: `SQUAD_ANSWERABILITY_TITLE_FRACTION` is annotated "moving it re-partitions the held-out set, which rule 2 permits only by a human decision" (split.py:105-109) and was itself recorded as a lead decision under explicit user delegation (HANDOFF/data-fixes-2026-09-29.md:94). My ruling is therefore a proposal for the human to sign, with the mechanism fixed so only the number is open.

**How.** Unit = repository, not commit. `assign_repo` (split.py:175) already splits train/val/heldout by `repo_key`, and the family partition must not be finer than the split unit: two commits of one repo share files, so a commit-level partition leaves exactly the content overlap (identity_key, dedupe_text) that M12 measured. Mechanism = keyed hash of `(repo_key, seed)` into `code.language_id` versus `{commit_intent, change_scope}`, a sibling of `squad_title_family` (split.py:146-160), applied in `mixture._dispatch` (mixture.py:1677) exactly as line 638 applies the SQuAD owner. Ratio: propose **0.2** to the held-out family, the SQuAD figure and reason (it measures only, so takes the smaller share); the trained families keep 0.8 of the corpus. Record the number next to the SQuAD one and cite the human ruling.

**Refusal default.** On in v6: `refuse_task_holdout_content_overlap = True` (config.py:149) in the same change as the partition, never before it. A check that always reports and never refuses is M12's own finding restated; the partition is what makes "on" possible without failing every build. Keep the measured 5,148/5,148 in the handoff as the pre-fix characterization.

**Consequence.** GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN (replay refused by heldout containment) closes with it; GAP-M12-TASK-HOLDOUT-CONTENT-OVERLAP-IS-EVERY-COMMITPACKFT-TRAINING-ROW-2026-10-17 closes once the human signs the ratio. Every existing commitpackft shard set is then superseded, which is the v6 rebuild anyway.

**Evidence.** split.py:103-109, :146-160, :175-178; mixture.py:638, :1677; config.py:147-149; c015166; HANDOFF/data-fixes-2026-09-29.md:32, :94. Read, not graph-confirmed.

**Fable's correction to Q12's handoff command:** it should read `MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh w4e-cargo cargo test -p qd-export`. The variable is read by mac_heavy.sh:136, so it must be in the wrapper's environment, not the job's.

## Q14. The v6 length band: P1 option 1 is mine and I take it; P2 is yes; W2-d's brief holds

(Asked as a follow-up after W3-b's report. Fable's answer, verbatim.)

**P1. Decision.** Rebuild the pairwise rows under `pairwise_length_balance {low 0.45, high 0.55}` and re-pin the pool in `data/decisions/v6-allocation.json`; do not rule the band off. This is mine under delegation: the band is a data-construction rule from the plan I designed (HANDOFF/data-clean-v6-plan-2026-10-06.md:66, item 9; :7 "Decisions go to Fable; the human is asked only about cost and downloads"), applied to training rows. It touches no held-out set, no gate threshold and no seed, so rule 2 is not engaged; the only pairwise content that moves is the train/val stratification of `pairwise.helpfulness`, a 2% share (plan:23). Ruling the band off would reopen M24 and would be the human's.

**Conditions.** The rebuild changes the pairwise row count, so the R2 allocation figure for `pairwise.helpfulness` is re-read from the new manifest, with `len_ratio` and drop counts recorded (plan:221). Cost: CPU on the Mac under mac_heavy, $0, so no human ask.

**W2-d.** Its brief ("reproduce v5r's inputs with only the rules changed") is unchanged in meaning and the band is one of those rules: v5r's pool is the v5 input, the band is a v6 rule, and a rebuild that reads `DataConfig()` is the brief executed. What W2-d must not do is name `DataConfig.legacy_v5()` to get past the refusal; that would be the rule change silently undone. Until the pool is re-pinned, W2-d is blocked on this item, and the handoff says so.

**P2. Decision.** Yes: add an optional `val_without_train` mapping to `LedgerRow`, written by the producer and read by `_population_paired_margin`, so a not_run there names the cause. Schema change, so the lane owning `ledger.py` and its Rust mirror (`crates/qd-train/src/ledger.rs`) does both halves in one change with a parse test, the same shape as the `started` fix in Q12 item 3. It excludes nothing from the pool; `paired_margin_population` (human, 2026-10-08) is untouched.

**Evidence.** config.py:64, :143, :294, :301; v6-allocation.json:7, :10-13; w3b-proposals.md P1-P2; GAP-W3B-V5-DECISION-POOL-FAILS-THE-V6-LENGTH-BAND-2026-10-17; GAP-W3B-LEDGER-POPULATION-GATE-READING-DOES-NOT-NAME-VAL-WITHOUT-TRAIN-2026-10-17. Read, not graph-confirmed; the W2-d brief wording is the coordinator's, not read by me.

## Lead's notes (lappi-decision-ce)

- **Q14 P2 touches crates/qd-train/src/ledger.rs, which belongs to lappi-decision-10.** It goes to that session, together with the `Status::Started` change.

- **Q13's 0.2 ratio is waiting for the human's signature.** Nothing is partitioned, and the refusal stays off by default until it is signed.

- **Q12.4, pushing main**, publishes 271+ local commits to github.com/bharathvbcr/Lappi-decision. That is outward-facing, so it waits for the owner's explicit yes. It is not done under this delegation.
- **Q8's MAC_HEAVY_MAX_LOAD=48** applies only to serialized `cargo test -p <crate>` runs under the lock, after gdn-mem's lock frees, with the footprint cap untouched and the raised value recorded on the launch line. It does not cover pytest, torch suites or data builds.
- GAP-W4D-MUTATE-CAP-K-AND-FIXTURES-2026-10-18 lives on the unmerged branch w4d-mutate-calib-2026-10-18. Its line is copied into main's gaps.jsonl with this file, so the citation resolves.
