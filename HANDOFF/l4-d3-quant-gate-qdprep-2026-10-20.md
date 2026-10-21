# L4 handoff: D3 quantisation gate tool (b6) and the qd-prep reds (b2), 2026-10-20/21

Lane L4 of ft-57fc7484 (revision 4). Branch `lane/ft-57fc7484-l4`, cut from main `d09222d`. No merge, no push.

## Measured (labels: V = read or run today, I = inferred, NOT RUN = did not happen)

### b6: tools/d3_quant_gate.py
- V: `make torch-pytest` with `TORCH_PYTEST_RUN` narrowed to `python/tests/test_d3_quant_gate.py`, under
  `tools/mac_heavy.sh` (label `l4-b6-torch-pytest`): 36 passed, exit 0, 2026-10-21T05:17Z.
- V: the hand oracle matches `mx.quantize` (mlx 0.31.2) on 2012 random groups, scale and bias bit for bit.
- V: `campaign/lora-2b-preregistered.json` carries no D3 bars. The tool reads a `d3_gate` block from a
  pre-registration and reports NOT_RUN (exit 3) without one. Nothing was invented
  (GAP-LORA2B-D3-THRESHOLD-NOT-MACHINE-READABLE-2026-10-21, owner human, owner item 5).
- NOT RUN: the real-base verdict. No model was loaded and nothing was downloaded.
- Not built: a ledger row (the verdict goes to a JSON file), and the 2B-Base / 0.8B-Base fake-quant base files.

### b2: qd-prep reds
- V, baseline on unmodified `d09222d`, no `QD_PYTHON` (mac_heavy label `l4-qdprep-base-noqdpython`,
  `cargo test -p qd-prep --no-fail-fast`, exit 101): lib 271 passed; `convert_licence_parity` 0 passed, 3 failed
  (`every_pending_id_is_one_python_does_not_admit`, `the_table_aliases_and_every_classification_match_python`,
  `concurrent_oracle_calls_never_share_an_input`; panic at `convert_licence_parity.rs:78`: the worktree has no
  `.venv`); `stress_v6` 52 passed, 2 failed, 3 ignored (the two S8 timeouts below); every other target passed.
- V, 2 S8 timeouts, pre-existing, out of scope: `s8_synth_email_...` and `s8_synth_jarvis_...` killed at
  `CLI_LIMIT` 1800 s (`stress_v6.rs:34`, panic at `:115`), debug build, load about 90. Owner card ft-b5110a5c;
  update appended to GAP-STRESS-V6-CLI-LIMIT-1800S-LOAD-SENSITIVE-2026-10-18. The bound is unchanged.
- V, fails-before for the new resolver test, lookup of the main checkout removed:
  qd-preflight `schema_compat` 1 passed 5 failed (new test fails at `schema_compat.rs:120`, candidates
  `[linked/.venv/bin/python, python3]`); qd-metal `ledger_rows` 7 passed 4 failed (new test fails at
  `ledger_rows.rs:113`).
- NOT RUN: the qd-prep fails-before run of the new test. I stopped that job (it had waited hours on the lock);
  the resolver text is identical in all three crates and `python/tests/test_qd_python_resolver_pinned.py` pins it.
- V: the pin test passes (2 passed) and fails when a copy drifts (checked with a one-line comment drift).
- NOT RUN (queued, blocked on the mac_heavy lock held by another session's `tessl-gpurun2`):
  the three after-runs with the fix and no `QD_PYTHON`, and the qd-prep run with
  `QD_PYTHON=/Users/bharath/Code/research/Lappi-decision/.venv/bin/python`. Until they run, the fix is unverified in
  cargo; it is not reported as passing.

## Changed (commits on lane/ft-57fc7484-l4)
- `1301511` tools/d3_quant_gate.py, python/tests/test_d3_quant_gate.py.
- The resolver commit that follows this handoff: the `qd-python-resolver` block in
  `crates/qd-prep/tests/convert_licence_parity.rs`, `crates/qd-metal/tests/ledger_rows.rs`,
  `crates/qd-preflight/tests/schema_compat.rs`, `python/tests/test_qd_python_resolver_pinned.py`, `gaps.jsonl`
  (3 appended lines), this file. Order: `QD_PYTHON`, this checkout's `.venv`, the main checkout's `.venv` via
  `git rev-parse --git-common-dir`, then `python3`; none usable is the existing loud panic naming `QD_PYTHON`.

## Open (gap ids)
- GAP-LORA2B-D3-THRESHOLD-NOT-MACHINE-READABLE-2026-10-21 (owner: human).
- GAP-STRESS-V6-CLI-LIMIT-1800S-LOAD-SENSITIVE-2026-10-18 (ft-b5110a5c).
- GAP-W6C-QDPREP-LICENCE-PARITY-TESTS-NEED-QD-PYTHON-IN-WORKTREE-2026-10-18: stays resolved-with-residual until the
  after-runs below pass; then append a closing line.
- GAP-LORA2B-QUANT-GATE-TOOL-MISSING-2026-10-17 now resolved-with-residual (real verdict, ledger row, base files).

## First command for the next lane
Read the three queued outputs (labels `l4-b2-after-prep`, `l4-b2-after-preflight`, `l4-b2-after-metal`) or rerun,
one crate per job:

    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-prep cargo test -p qd-prep --test convert_licence_parity
    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-preflight cargo test -p qd-preflight --test schema_compat
    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-metal cargo test -p qd-metal --test ledger_rows
