# X2 pre-registration evidence (2026-10-22)

Lane X2, branch `lane/x2-prereg`, from main a012dac. Delegation: the owner ("spin agent and fix them",
2026-10-22); method ruled by Fable. Rule 2: no number was chosen, moved or added.

## Part A: `campaign/lora-2b-q8-preregistered.json`

- Thresholds: `argmax_agreement_min` 0.99, `mean_abs_delta_logp_max` 0.02, `min_prompts` 500, quantiser
  affine / 8 / 64, transcribed from `AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md` section 5,
  first bullet (blob `e2bbc86b609a503199fa28bc62d47b02d912c40d`, `git rev-parse HEAD:<path>` at a012dac).
  `python/tests/test_lora_2b_q8_prereg.py` reads the prose and the file and fails if they differ.
- Procedure (labelled `procedure` in the file): `base.format`, base-file sha256 in every row, pool key
  (family id, base.format, base-file sha256). The fake-quant file's sha256 does not exist yet and is not
  pre-filled.
- mlx `0.31.2`: `/Users/bharath/.venvs/ml/bin/python -c "import mlx.core as m; print(m.__version__)"`.
- Order check: `rg -uu -c "q8" ledger/` matched nothing (exit 1), so no q8 or lora-2b-q8 row exists in any
  ledger; none to mark quick or excluded. The commit precedes any such row.
- `bf16` file `campaign/lora-2b-preregistered.json` not edited.

## Part B: span-refit span null

- Old scorer sha256 `b88715afc031a2ef38aead1b0f0320d7d0763d4a990532ac8015c0652d7e07d5`, new
  `8dc59bd4df5f14e4bd106982f3d40119bf33380268e54c87520b166d1f77b2ed` (sha256 over file contents).
- Pre-registration `campaign/span-refit-mac-preregistered.json` amendment 2 (same commit) names both and what
  is newly emitted. The file's sha256 moves from `b4a370ff...` to `2252f630eeda39f907d03e339b6e2eea31f61fa932184281828d502f760c2cde`;
  `tools/span_refit_rows.py` `PREREG_SHA256` follows. Consequence: a refit head whose manifest records the old
  prereg sha is refused by the scorer and is refit.
- No span-refit row exists in `ledger/` (`rg -c span-refit ledger/*.jsonl` none), so nothing is reinterpreted;
  results JSONs from the old scorer are void for `span_beats_null` and rerun.
- Verifier: `LedgerRow.check_span_refit_span_null` (Ledger.append and LedgerRow.from_json). Python only; the Rust
  reader in `crates/qd-train/src/ledger.rs` does not carry it.
- Not run: any real span-refit run, hence any real-row `span_beats_null` verdict.
- Observed, not caused here: `test_span_refit_b_launcher.py` two tests exit 76 on the unmodified commit too
  (`GAP-SPAN-REFIT-B-LAUNCHER-TESTS-EXIT-76-2026-10-22`).
