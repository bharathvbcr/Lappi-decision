# Handoff: Arm B Swift rebuild (lane W4-1, ft-b1dfb6da273, 2026-10-13)

Ruling: `AUDIT/board-sweep-2026-10-13/fable-own-swift-clean-rows-ruling.md` (plus the structured
base pin of `fable-own-repos-diff-tie-ruling.md` s4). Branch `wave4-l1-own-swift-clean-rows`, base 174452f.

## Measured (ledger / gap ids)

- New Swift shard: ledger row `cd1b1168-e3e0-4f09-9eea-e3aef1fef644` (`ledger/arm-b-shards-2026-10-13.jsonl`,
  quick, smoke): 1,192 rows built, 123 refused (invisible_format_characters), 46 over max_seq_len 10240,
  1,146 rows, 2,292 sequences, shard_hash `2adfef2114508d80...`. Coverage gate reads passed=False because of the 46.
- `e1ccb673` is superseded (append-only; not edited). `165ba699` (TS) and `247678e1` (single-file) STAND:
  the re-run plan's materialised `examples.jsonl` are byte-identical to the previous materialisation
  (TS `2eebbdca...`, single-file `48bc7bc1...`, checked with `cmp`). Not rebuilt.
- Corpora: `own-swift-corpus-v2` 13,647 examples (clean 0, `clean_without_prior` 0) sha256 `02cbfdac...886`,
  manifest `47b35c83...162`; `own-swift-composed-v2-prefixed` N=1315 sha256 `d5e9341e...e26`, manifest
  `62338afa...25a`; each built twice, identical. Loads as pinned: 14,962 rows (13,647 + 1,315); as materialised: 1,315.
- Plan from the amendment commit: offered 13,032, kept 9,908 (was 9,651), candidates sha256 `1c2d0978...`.
- Gaps: `GAP-ARM-B-SWIFT-V2-REBUILT-E1CCB673-SUPERSEDED-2026-10-13`, `GAP-BOARD-SWEEP-TOOLING-GAPS-OWN-SWIFT-RULING-2026-10-13`.

## Changed (commits)

1. `2b2435f` qd-mutate compose refuses a single-version base with clean rows (names `--clean-permille 0`);
   counter `single_version_clean_examples_unused` removed; commit-mode compose over a pool with no
   `prior_source` now names `--single-version-records` (the clean-parity failure that did so is gone).
2. `b15d753` `arm_b_mine.py materialise` loads each pinned composed corpus through `load_defect_rows` first
   (`--download-root`, `selected_by_plan.loaded_rows`).
3. `3dd523f` pre-registration amendment (v2 pin, superseded v1 pin kept).
4. `1a946c1` structured `base_corpus` pin (path, sha256, rows). Pins W3-1 mirrors into the loader.
5. This commit: ledger row, gaps, handoff.

## Open

- MPS probe (not run), R3 shortfall (1,315 + 1,250 composed vs 8,000), ft-a59b211c (W3-1; merges after this lane).
- Pre-registration block at `candidates_sha256 0958ff52...` / `kept 9651` records the superseded plan; the
  current plan values are above.
- Data under `data/pool/` is gitignored except manifests; the shard and `.build-b` dirs are untracked.
  Copies of L3's corpora were used (TS, single-file, commitpackft download); L3's worktree was only read.

## Notes from review

- `materialise` loads each composed corpus as pinned with `repo_root=repo` but writes the planbase under
  `pool_dir`; the two differ if `--pool-dir` is not `<repo>/data/pool`, in which case the loader and the
  planbase read different base locations. Default use is consistent.
- The structured `base_corpus` pin in the pre-registration is read by W3-1's loader-side mirror test,
  not by `arm_b_mine.py`.
- Ledger row cd1b1168's `-dirty`: see `GAP-ARM-B-SWIFT-LEDGER-CD1B1168-CODE-COMMIT-DIRTY-2026-10-13`.
- `fable-own-repos-diff-tie-ruling.md` is W3-1's file and was removed from this branch.

## First command for the next lane

`QD_PREP_BIN=<repo>/target/release/qd-prep HF_HUB_OFFLINE=1 tools/mac_heavy.sh .venv/bin/python tools/real_ft_run.py --score-checkpoint ...`
over `data/pool/.build-b/shards/{ts-composed,single-file,swift-composed}` (swift = `cd1b1168`); see
`GAP-ARM-B-SHARDS-BUILT-PROBE-NOT-RUN-2026-10-13` for the probe's definition.
