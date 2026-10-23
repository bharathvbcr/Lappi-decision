# X7 containment-repin handoff, 2026-10-23

Branch `lane/x7-containment-repin` (from main 4908cda, the X6 merge). Not merged, not pushed.

## Measured

`test_a_build_without_the_flag_writes_what_it_wrote_before` moved from build_digest
`ea81e519...` to `431ebb22...` when X6 merged fingerprint rule 3 (ft-2eb31e7d).

The 60-pair corpus was built with the test's own call (`pipeline.run(max_pairs=60, ...,
v6_data_rules=False)`, `AUDIT/x7-containment-repin-2026-10-23/char_build.py`) from ce76aa3 (the
first parent of the X6 merge) and from 4908cda, on one host, one tokenizer, one qd-prep binary:

- ce76aa3 gives `ea81e519...` in two runs, so the host is not the difference.
- 4908cda gives `431ebb22...`.
- 33 files each, same names. 27 byte-identical; no non-JSON file differs (every token, offset,
  supervision, remap and index file is identical).
- Six JSON files differ (`compare.out`):
  - shards/{train,train-no-decode,val}/header.json: code_fingerprint gains `manifest.py` and
    `val_requests.py` (no existing module digest moved), code_fingerprint_version 2 -> 3,
    shard_hash (train and train-no-decode share one value), created_at.
  - data/{heldout,pool/train,pool/val}.json: provenance.built_at_utc only.
- Unexplained: 0. The digest was re-pinned and the eighth move is recorded in the test docstring.

## Changed

- `python/tests/test_containment_exclusions.py`: pinned digest and docstring accounting only.
- `AUDIT/x7-containment-repin-2026-10-23/`: char_build.py, compare.py, compare.out.
- `gaps.jsonl`: GAP-CONTAINMENT-CHARACTERIZATION-DIGEST-MOVED-BY-FINGERPRINT-RULE-3-2026-10-23
  (resolved).

## Other pinned digests

Searched `python/tests` and `crates/*/tests` for 64-hex strings (`rg -uu`). Candidates that pin a
shard build or fingerprint were run on this tree; none other than the one above moved. The
fixture-based Rust pins vendor their own shard sets and do not rebuild under rule 3.

## Not run

- `test_defect_class_own_repos`: skipped, the v5 defect corpus is git-ignored and not on this host.
- `test_pipeline_val_shards`: one test skipped, the qd-mutate defect corpus is not on this host.
- `cargo test -p qd-train --test shard_code_rule --test shard_header_parity --test
  retired_shard_sets --test v4 --test fixture_pins` ran under mac_heavy, exit 0: 14, 5, 4, 0 (2
  ignored), 13 passed. No Rust digest moved.
- After the audit scripts were lint-fixed (E402/E501 only: importlib instead of a late import, one
  wrapped print), they were not re-run; compare.out is from the run of the pre-lint-fix text. The
  re-run was queued behind the lock for hours under load and cancelled.

## First command for the next lane

`make torch-pytest 'TORCH_PYTEST_RUN=$(UV) run --no-project --python $(ML_PY) --with pytest --with hypothesis --with datasketch python -m pytest $(REPO)/python/tests/test_containment_exclusions.py -o addopts= -p no:cacheprovider'`
(needs `data/pool/commitpackft/{go,python,rust,typescript}.jsonl` and QD_PREP_BIN).
