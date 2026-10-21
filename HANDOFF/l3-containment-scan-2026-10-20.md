# HANDOFF l3-containment-scan (lane L3, ft-57fc7484 b3 and ft-c5ea3e71 item 2)

Branch `lane/ft-57fc7484-l3`, from main d09222d. No merge, no push. Evidence copied to
`AUDIT/l3-containment-scan-2026-10-21/`; full logs and scratch are in the worktree's gitignored
`build/l3/`. This lane writes no ledger rows (it ran no training), so the numbers below cite gap
records in `gaps.jsonl` and the audit files, not ledger row ids.

## What was measured

### b3a: `python/tests/test_containment_exclusions.py`
- The 4 failures at `AUDIT/harden-2026-10-17/w2d-v6-rebuild-2026-10-20/mega-summary.txt:25-33`
  reproduce at c172ab7^ (= d2325a7) with a freshly built qd-prep: `4 failed, 25 passed, 3 skipped`.
  On main they are green. Cause: the fixture built under `DataConfig()`, which became v6 and refuses
  an MMLU upstream-test item as a row. Fixed by c172ab7 (test-only; `tools/containment_scan.py` was
  right). Under `tools/mac_heavy.sh` with `QD_PREP_BIN` set (qd-prep built from main in this
  worktree, `build/l3/b3a_before.log`): containment_exclusions + containment_strip +
  qd_prep_containment_parity = `58 passed, 8 skipped`.
- The torch-venv run (the Makefile's torch-pytest form) found 2 more reds on main that the torch-free
  venv hides (`torch-venv-two-reds-summary.txt`): `pipeline.run` defaults to v6 data rules and
  refuses the 60-pair corpus (no defect class); and the characterization digest had moved
  ff39da60 -> ea81e519. Fixed in 7a74991: `v6_data_rules=False` on the build calls, the drops test
  scans under `DataConfig.legacy_v5()`, and the pin is re-pinned with every moved byte accounted
  (171c879 reproduces ff39da60 on this host; 6 of 33 files differ, all JSON: header code
  fingerprints and shard_hash; manifest keys slot_shape, slot_shape_digest, val_without_train,
  task_holdout_content_disjoint). After: `32 passed, 0 skipped` (run directly, 44 s, not under the
  lock). Gap: GAP-L3-CONTAINMENT-EXCLUSIONS-TORCH-VENV-TWO-REDS-FIXED-2026-10-21. **The re-pin is a
  judgment the owner should look at.**
- Pinned behaviour, no fix needed: an interrupted scan leaves nothing that passes as complete
  (`test_containment_scan_interrupted.py`, 9 passed: a request cut at 0, 1, 8 bytes, 25/50/90 % or
  one byte short is refused by qd-prep with no attestation; a rerun refuses to overwrite).

### b3b: v6 scan footprint (GAP-L3-V6-SCAN-FOOTPRINT-BOUNDED-BY-PACKED-NATIVE-MINHASH-2026-10-21)
All under `tools/mac_heavy.sh` with the unchanged footprint cap (28,672 MiB), measured by
`tools/footprint_peak.py` (mac_footprint's ri_phys_footprint, tree sum, 2 s), v6 rules, the W2-d
inputs (pool sha 25a88f6e...). Timelines: `AUDIT/l3-containment-scan-2026-10-21/*.timeline.txt`.

| run | tree | exit | peak MiB | wall s |
| --- | --- | --- | --- | --- |
| base1 | a5c9a12 (main + footprint tool) | 0 | 22,957 | 440.3 |
| after1 | 1d05037 | 0 | 10,999 | 576.6 |
| after2 | 1d05037 | 0 | 10,854 | 482.1 |
| after3 | 1d05037 | 0 | 11,400 | 485.5 |

3 of 3 consecutive after-runs complete under memguard's 32 GiB line (and the cap). pairs.tsv,
exclusions.txt and attestation.json are byte-identical (`cmp`) between base1 and each after-run;
exclusions sha256 37b84644...3ff7 is also the sha W2-d's chain11 scan logged. Root cause: in base1
the Python parent was 20,417 MiB at the minhash and lsh stages (the lsh child adds ~2.5 GiB), the
frozenset-of-bytes shingle sets (116-160 B a shingle) plus one 7.3 KB tuple per signature in
`native_minhash`. Fix: packed int64 hashes (8 B a shingle), at most `batch_sets` (default 10,000;
1..1,000,000; validated) new sets per qd-prep call, a signature matrix instead of tuples.
What remains is qd_data's own O(corpus) state (peak 'python only' 10.7-10.9 GiB); changing it
re-fingerprints every pinned shard set and attestation, so it was not touched.
Benchmark: after2/after3 are +9.5 % / +10.3 % over base1 (n=1 base; after1 +31 % was the first run
on a loaded host; the after-runs' own spread is 17 %). The cost is packing (~35 us a set, ~18 s) and
~52 signing calls. Over 5 %, accepted for halving the peak; not distinguished from host noise.
Note: base1 on current main also passes the cap (22.4 GiB): the earlier 28.7-29.4 GiB kills predate
the shingle interning that is already on main.

### ft-c5ea3e71 item 2: containment attestation fingerprint
Already done at 5b1b1a6 on main (CONTAINMENT_CONTENT_MODULES, 9 modules, version 2).
`test_containment_code_fingerprint.py`: `16 passed` on main, collection error at 5b1b1a6^. Gap
record never closed; now closed (GAP-L3-CONTAINMENT-ATTESTATION-FINGERPRINT-CLOSURE-DONE-AT-5B1B1A6-2026-10-21).
**Opened, owner's call:** the closure is that of `exclusions.py`'s imports, not of what the scan
reads. Appending a comment to render.py, mixture.py, loaders.py or general.py does not move
`containment_corpus()` (probe `closure_probe.py`); split.py does. Option: containment rule 3 =
`SHARD_CONTENT_MODULES` (GAP-L3-CONTAINMENT-FINGERPRINT-CLOSURE-OMITS-ROW-BUILDING-MODULES-2026-10-21).

## What changed (commits on lane/ft-57fc7484-l3)
- a5c9a12 `tools/footprint_peak.py` + test (new). 4a14cc1 its test's name match fix.
- 1d05037 `tools/real_tokenizer_pipeline.py` `native_minhash` (packed sets, batches, SignatureRow,
  packed exact_jaccard) + `python/tests/test_native_minhash_packed.py` (18 tests).
- 7a74991 `test_containment_exclusions.py`. 3a3b16d `test_containment_scan_interrupted.py`.
- dc1c711, c311913, 9894135 `gaps.jsonl` (5 records appended, O_APPEND). 814 lines added, 42 removed.

Fails-before: `test_the_block_holds_hashes_not_shingles` is 27.8 MiB held on the pre-change code against a
6 MiB bound, 2.4 MiB after; the rest of the new file fails there at the `batch_sets` argument.
Existing native tests (minhash/lsh parity, composed files, memory): 39 passed, 4 skipped, unchanged.

## What is open
- 20 other tests fail identically on the pre-change tree (`failed-ids-on-base-tree-summary.txt`
  vs `subset-summary.txt`, 133 files, 3238 passed, 31 skipped): v6-default fallout in
  test_pipeline_val_shards (3), test_qd_prep_spancheck_parity (4), test_real_ft_general_record (2),
  test_split_cache_main, test_pre_dedupe_drops, test_real_ft_pieces, test_ft_linear_control_split_cache
  (3), test_option_control, test_batch_order, test_wall_clock_cap_required,
  test_real_ft_score_plan, test_ledger_provenance. Not this lane's; L2 owns the torch reds (b1).
  `test_ledger.py::test_an_interrupt_during_a_suite_takes_its_process_group_down` failed in the
  40-minute run and passes alone (load-sensitive, inferred).
- The scan inputs `commitpackft-composed-v2`, `defect-noul-v3b`, `own-prose-v1/units.jsonl`,
  `commitpackft-g6-v1` are not in the main checkout's `data/pool`; they were read from
  `/Users/bharath/qd-campaign/v5-data-2026-10-02/data/pool` through symlinks (`link_data.py`).
  The loaders check manifest sha256s, so a wrong file would have refused.
- Two lock holds were lost to my setup (main .venv has no torch; composed corpus missing in main).
  The first two chain attempts exited in seconds.

## First command for the next lane
`bash tools/mac_heavy.sh scan bash build/l3/baseline.sh` reruns after1-3 and base1 (needs the
worktree's `target/release/qd-prep`, `build/l3/{base,after}-tree` and linked data).
To read the numbers: `python3 -I build/l3/peaks.py base1 after1 after2 after3`.
