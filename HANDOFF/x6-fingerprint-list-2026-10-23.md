# X6 fingerprint list (ft-2eb31e7d9863d70a25807b3425fa6057), 2026-10-23

Branch `lane/x6-fingerprint-list`. First pass stopped (adding the two modules made the reader refuse the pinned v2
sets, GAP-SHARD-FINGERPRINT-LIST-FIX-REFUSES-EVERY-PINNED-V2-SET-2026-10-23). Fable ruled (a): fingerprint rule 3,
implemented here.

## What changed
- `python/qd_data/fingerprint.py`: `FINGERPRINT_VERSION = 3`; rule 2's 16-module list frozen as
  `SHARD_CONTENT_MODULES_V2`; `SHARD_CONTENT_MODULES` (rule 3) adds `manifest.py` and `val_requests.py` (the
  closure test derives them); `SHARD_CONTENT_MODULES_BY_RULE`, `KNOWN_FINGERPRINT_VERSIONS`,
  `not_covered_modules(rule)`, `shard_code_fingerprint(directory, rule)`.
- **The rule number is carried in `code_fingerprint_version`** (hash-covered, serialised when not 1). That is the
  field Fable's "`fingerprint_rule: 3`" refers to; no second field was added (lead accepted).
- `python/qd_train/artifacts.py`: readers branch on the recorded rule, never on module counts. A v2 header verifies
  over its own 16 modules and the verdict carries `not_covered: [manifest.py, val_requests.py]` (in
  `shard_code_current`'s detail and as a new check `shard_code_not_covered`, `Ran` with `value` = comma-joined list,
  `""` when none, `NotRun` for a header with no fingerprint).
- `crates/qd-train/src/shards.rs`: same branch, same check; reads the V2 and current tuples from the given
  `qd_data/fingerprint.py` by exact name. Parity: `crates/qd-train/tests/shard_code_rule.rs` embeds Python-written
  v1, v2 and v3 headers (`AUDIT/x6-fingerprint-list-2026-10-23/make_rust_fixtures.py`); the v2 header's shard_hash is
  unchanged (91cbf915...). `crates/qd-train-metal/tests/run_door.rs` moved to rule 3.
- Report-only is enforced: `python/qd_train/ledger.py` `_shard_not_covered_refusals`, called by `promotion_verdict`
  and `promotion_verdict_avg` (`byte_identity_path=`). Per split: `shard.code_not_covered` (train, judged against
  `recipe.shard_hash`) and `shard.val_code_not_covered` (val, `recipe.val_shard_hash`). A row refuses unless
  `AUDIT/x6-fingerprint-list-2026-10-23/byte_identity.json` has `identical: true`, a non-empty `audit` citation and
  THAT split's shard_hash. A `NotRun` verdict refuses. No such record exists.
- `tools/real_ft_run.py`: `_recorder` writes the train metric; `open_val_reader` puts the val metric in `_RUN_METRICS`
  (every later row of the run carries it). Rows written before this change carry neither metric, so they are not
  marked: the refusal covers rows from now on.
- Test edits that are not additions: `test_shards.py` stub of `shard_code_fingerprint` takes `rule`;
  `test_artifacts.py` expected check names gain `shard_code_not_covered`; `test_promotion_respec.py::_family` gains
  two optional parameters.

## Measured
- Pinned sets, header-only, branch reader: `AUDIT/x6-fingerprint-list-2026-10-23/verify_pinned.AFTER_V3.txt`: train
  94bef49a... and val 58bd753f... both `Ran passed=True` (16 modules match) and `not_covered: [manifest.py,
  val_requests.py]`. A rebuild now records 18 modules (rule 3).
- No committed fixture or golden digest moved (v1 fixtures untouched; the v2 mini header hash is identical).
- pytest (make torch-pytest under mac_heavy): after, 655 passed, 2 skipped, exit 0 (18 files: fingerprint, artifacts,
  shards, ledger, promotion, real_ft, containment). Before (main's code, new tests copied in): 8 failed, 87 passed,
  2 collection errors (ImportError of the new constants), exit 2. `build/x6_pytest.out`.
- cargo: `cargo test -p qd-train` exit 0, every result ok; `qd-train-metal --test run_door` 6 passed. Before (main's
  `shards.rs` swapped in): `shard_code_rule` does not compile (`not_covered_modules` unresolved), exit 101.
- Byte-identity rebuild: priced, not run (`byte-identity-rebuild-price.md`: pipeline 7299 s, chain 2 h 36 min, peak
  footprint never recorded, scan output gone). The sets stay report-only.

## Open
- GAP-SHARD-FINGERPRINT-LIST-FIX-REFUSES-EVERY-PINNED-V2-SET-2026-10-23 is resolved by the rule-3 design; its
  record is left as written.
- The rebuild and byte comparison against 94bef49a / 58bd753f; on identical bytes, add the `byte_identity.json`
  records (train and val); on a difference, rebuild and amend every pre-registration that pins them.

## First command for the next lane
`/Users/bharath/.venvs/ml/bin/python AUDIT/x6-fingerprint-list-2026-10-23/verify_pinned.py`
