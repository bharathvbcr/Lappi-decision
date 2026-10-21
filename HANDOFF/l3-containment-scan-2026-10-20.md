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
  ff39da60 -> ea81e519. Fixed in 67d15db (this branch): `v6_data_rules=False` on the build calls,
  the drops test scans under `DataConfig.legacy_v5()`, and the characterization test's docstring
  names its rule set (explicit legacy v5; it characterizes the shard writer and exclusion flag,
  not the v6 rules; Fable ruling 2026-10-21). The commit message of 67d15db still says
  "re-pinned": its diff does not touch the digest. **The digest re-pin ff39da60 -> ea81e519 is NOT
  on this branch's own commits.** It is the one commit 4fba3b4 on `lane/ft-badf64e2-repin` (from
  d09222d; card ft-badf64e2 owns it), which this branch is rebased onto; the accounting is in the
  test docstring there (171c879 reproduces ff39da60 on this host; 6 of 33 files differ, all JSON:
  header code fingerprints and shard_hash; manifest keys slot_shape, slot_shape_digest,
  val_without_train, task_holdout_content_disjoint). On this branch's tip: `32 passed, 0 skipped`
  in the torch venv (run directly, 50 s, not under the lock). Gap:
  GAP-L3-CONTAINMENT-EXCLUSIONS-TORCH-VENV-TWO-REDS-FIXED-2026-10-21.
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
Benchmark: **the 5 % wall-time check is UNRESOLVED, not passed.** Base n=1 (440.3 s); after-runs
482.1 / 485.5 / 576.6 s, a 17 % spread among themselves, so the +9.5 % / +10.3 % (and +31 % for the
first run) deltas are inside the noise and neither confirm nor rule out a slowdown over 5 %. The
cost the change adds is packing (~35 us a set, ~18 s) and ~52 signing calls. Fable accepted this
without an interleaved A/B (ruling 2026-10-21); the owner may ask for a 3+3 interleaved A/B at
merge time (`build/l3/baseline.sh` is the harness shape).
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
Rebased 2026-10-21 onto `lane/ft-badf64e2-repin` (4fba3b4, the digest re-pin only), so every id
below is the rewritten one. Pre-rebase ids, which the gap records and the measured-tree names in
this file cite, map: a5c9a12 -> 3b12f62, 1d05037 -> 45e482e, dc1c711 -> 8f3cfbe,
7a74991 -> 67d15db, c311913 -> 90fe757, 3a3b16d -> 37cf63f, 9894135 -> 075d227,
4a14cc1 -> f034df9. `git diff 1d05037 45e482e -- tools` is empty: the scanned code is the same.
- 3b12f62 `tools/footprint_peak.py` + test (new). f034df9 its test's name match fix.
- 45e482e `tools/real_tokenizer_pipeline.py` `native_minhash` (packed sets, batches, SignatureRow,
  packed exact_jaccard) + `python/tests/test_native_minhash_packed.py` (18 tests).
- 67d15db `test_containment_exclusions.py` (rule set, no digest change). 37cf63f
  `test_containment_scan_interrupted.py`.
- 8f3cfbe, 90fe757, 075d227 `gaps.jsonl` (5 records appended, O_APPEND), plus the id-mapping
  record GAP-L3-COMMIT-IDS-REWRITTEN-BY-THE-REPIN-REBASE-2026-10-21.

Fails-before: `test_the_block_holds_hashes_not_shingles` is 27.8 MiB held on the pre-change code against a
6 MiB bound, 2.4 MiB after; the rest of the new file fails there at the `batch_sets` argument.
Existing native tests (minhash/lsh parity, composed files, memory): 39 passed, 4 skipped, unchanged.

## What is open
- 20 other tests fail identically on the pre-change tree (`failed-ids-on-base-tree-summary.txt`
  vs `subset-summary.txt`, 133 files, 3238 passed, 31 skipped): v6-default fallout in
  test_pipeline_val_shards (3), test_qd_prep_spancheck_parity (4), test_real_ft_general_record (2),
  test_split_cache_main, test_pre_dedupe_drops, test_real_ft_pieces, test_ft_linear_control_split_cache
  (3), test_option_control, test_batch_order, test_wall_clock_cap_required,
  test_real_ft_score_plan, test_ledger_provenance. Not this lane's. Per the lead (2026-10-21):
  three of them (two in test_real_ft_general_record, one in test_real_ft_score_plan) are L2's torch
  reds, now fixed; the other 17 are filed as their own card.
  `test_ledger.py::test_an_interrupt_during_a_suite_takes_its_process_group_down` failed in the
  40-minute run and passes alone (load-sensitive, inferred).
- The scan inputs `commitpackft-composed-v2`, `defect-noul-v3b`, `own-prose-v1/units.jsonl`,
  `commitpackft-g6-v1` are not in the main checkout's `data/pool`; they were read from
  `/Users/bharath/qd-campaign/v5-data-2026-10-02/data/pool` through symlinks (`link_data.py`).
  The loaders check manifest sha256s, so a wrong file would have refused.
- Two lock holds were lost to my setup (main .venv has no torch; composed corpus missing in main).
  The first two chain attempts exited in seconds.

## Adversarial round (lead, adv/ft-57fc7484-l3 b9b86fc, 9862a58; fixed in 974cb15)
Both adversarial files are cherry-picked (909f198, 9df6b69 on this branch); all BREAK tests pass.
Run: `make torch-pytest` narrowed to these files (the lane's tests, the three adv files, the
parity, composed-file, memory, fingerprint, mac_heavy_lock and gaps tests; `build/l3/adv_run.sh`)
under mac_heavy at its default load 12: `278 passed, 4 skipped` (benchmarks opt-in, real-MPS opt-in).
Only lint edits were made to the adversarial tests (unused noqa, one unused name).

1. **footprint_peak (breaks 1-3).** (a) Why `tools/mac_footprint.py` was not used as the sampler: it
   is a one-shot reader of the current footprint of pids it is handed (what the cap sums). I imported
   its `phys_footprint_bytes` and wrote the loop and the walk around it, and kept only the sampled
   maximum, which is a lower bound; I did not read the kernel's lifetime max. (b) One reader now:
   `mac_footprint.py` gained `rusage()` and `lifetime_max_phys_footprint_bytes()`;
   `footprint_peak.py` is the command runner and reporter on top of it (the adv tests, and a run
   that must launch a command, need a runner; the reader is not duplicated). (c) The tool reads
   each process's `ri_lifetime_max_phys_footprint` at every sample (exact for any process seen
   alive once) and takes `wait4`'s `ru_maxrss` of the command, which on macOS covers every reaped
   descendant (probe: idle root 16 MiB, root + 400 MiB grandchild that exited 417 MiB). The
   headline `peak` is the largest of those lower bounds and is printed as `LOWER BOUND`, with the
   sampled tree sum, one process's lifetime max, the reaped RSS and the sum of seen lifetime maxima
   separately. A tree sum cannot be exact: a child that lived and died between samples is in
   `ru_maxrss` (RSS, largest single process) but its overlap with the others is unknowable. (d)
   The timeline opens before the command starts. (e) The stage label parses the whole command line.
   The summary prints the largest lifetime max per stage (`python`, `qd-prep:minhash`,
   `qd-prep:lsh`, `qd-prep:containment`). **The 3-of-3 runs above used the older tool**: their
   peaks are sampled-tree lower bounds at 2 s; after3 never sampled a `qd-prep:minhash` child at
   all (`peaks.py after3`). The parent, alive for the whole run and above 8 GiB for minutes, was
   sampled many times; the short children are what the old tool could miss. Whether the three
   runs must be repeated with this tool is Fable's ruling; they were not re-run.
2. **Cross-set collision (break 4).** Each distinct set keeps a blake2b-128 content digest (sorted
   shingle bytes joined by 0xff, a byte UTF-8 never holds). An equal packed form with another digest
   is a refusal naming the collision, in the build and in `signature()` of a plain set. The
   docstring no longer says the canaries catch collisions; it says what nothing detects (one
   shingle colliding across two sets whose packed forms differ, about 1.7e-14 per compared pair, under
   3e-7 per v6 scan). Memory: 2.41 MiB on the 60-large-row fixture (2.40 before; bound 6); 1,549 B
   a set on 3,000 small rows (adv bound 2,500).
3. **Stale packed set (break 5).** `PackedShingles.block` is the owning block's token;
   `signature()` and `exact_jaccard` refuse another block's set.
4. **Every batch checked (break 6).** The first and last `NATIVE_MINHASH_CANARIES` (4) sets of every
   batch are re-signed by the reference and the packed Jaccard of those sets compared. At most 8
   reference signings a batch; a reply wrong only in sets between a batch's ends is still not
   caught. A single batch of at most 8 sets is checked whole, as before.
   Wall-time effect on a small fixture (4,000 distinct 420-word texts, 4 batches of 1,000, entering
   the block, interleaved, min of 3): 0.99 and 1.08 s before, 1.45 and 1.51 s after (about +0.4 to
   +0.5 s, +45 %): ~50 us a set for the digest (sort 37 us of it) and 32 reference signings at
   5.8 ms. Extrapolated to the v6 scan: 511,198 sets x 50 us is ~26 s, and 51 batches x 8 x
   5.8 ms ~2.4 s, so about +30 s on a ~440-480 s scan (+6 to +7 %), unmeasured at scale: the 5 %
   wall check stays UNRESOLVED and this adds to it.

## First command for the next lane
`bash tools/mac_heavy.sh scan bash build/l3/baseline.sh` reruns after1-3 and base1 (needs the
worktree's `target/release/qd-prep`, `build/l3/{base,after}-tree` and linked data).
To read the numbers: `python3 -I build/l3/peaks.py base1 after1 after2 after3`.
