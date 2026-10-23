# X6 fingerprint list (ft-2eb31e7d9863d70a25807b3425fa6057), 2026-10-23: STOPPED, no list change committed

Branch `lane/x6-fingerprint-list`. The fix the task describes (add `manifest.py` and `val_requests.py` to
`qd_data/fingerprint.py::SHARD_CONTENT_MODULES`) makes `_shard_code_check` refuse every pinned v2 shard set.
Rule 5 of the lane brief says stop and report; this lane did.

## Measured
- Red reproduced on main: `test_every_module_the_shard_builders_import_is_in_the_list` fails with
  `['manifest.py', 'val_requests.py'] shape shard bytes but are not fingerprinted` (1 failed, 10 passed).
- The list lives in `python/qd_data/fingerprint.py` (not shards.py or exclusions.py).
- `AUDIT/x6-fingerprint-list-2026-10-23/verify_pinned.py` reads only the header copies of the pinned v6 LoRA-2B
  sets (`build/shards-v6-lora2b-6bf4abfc/{train,val}/header.json`, shard_hash 94bef49a... and 58bd753f...; pinned
  by campaign/lora-2b, next-train-first-box and span-refit-mac pre-registrations).
  - `verify_pinned.BEFORE.txt` (unchanged list): `Ran passed=True`, 16 modules match.
  - `verify_pinned.WITH_LIST_FIX.txt` (list temporarily extended, then reverted): `Ran passed=False`,
    "added since: manifest.py, val_requests.py"; `verify_shard_contract` would raise unless `allow_stale_code=True`.
- Rebuild fingerprint after the fix would add `manifest.py` 76dd21734ad1fff97f2d2689cf29495c7ee73085ef91fa6a0a126becc3804e01
  and `val_requests.py` 8e4bde0e69f8759d8b48a0481cc17e678fdfdbfc97816931b941fae39c98fd4d to the 16 recorded digests
  (the other 16 are unchanged at this head); `shard_hash` moves with it.
- No committed fixture or golden digest was moved: nothing was re-pinned, because the list is unchanged.

## Open
- GAP-SHARD-FINGERPRINT-LIST-FIX-REFUSES-EVERY-PINNED-V2-SET-2026-10-23 (options in the gap record: a v3 rule whose
  check covers only the recorded keys for v2 headers, allow_stale_code on pinned sets, or rebuild plus amendment).
- Rust reader `crates/qd-train/src/shards.rs` not run against the change (unverified).

## First command for the next lane
`/Users/bharath/.venvs/ml/bin/python AUDIT/x6-fingerprint-list-2026-10-23/verify_pinned.py` after the owner picks an option.
