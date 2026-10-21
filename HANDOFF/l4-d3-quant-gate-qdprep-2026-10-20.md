# L4 handoff: D3 quantisation gate tool (b6) and the qd-prep reds (b2), 2026-10-20/21

Lane L4 of ft-57fc7484 (revision 4). Branch `lane/ft-57fc7484-l4`, cut from main `d09222d`. No merge, no push.
Labels: V = read or run today, I = inferred, NOT RUN = did not happen.

## Measured

### b6: tools/d3_quant_gate.py
- V: first `make torch-pytest` narrowed to `python/tests/test_d3_quant_gate.py`, under `tools/mac_heavy.sh`
  (label `l4-b6-torch-pytest`): 36 passed, exit 0, 2026-10-21T05:17Z. That was before the adversarial fixes.
- V (hand-assembled pytest path, not the project command): after the adversarial fixes the gate tests,
  `test_d3_quant_gate_adversarial.py` (commit `a26bc76`, from `925731a`) and the resolver pin test: 75 passed.
- V: the project-command rerun of those three files, `make torch-pytest` with `TORCH_PYTEST_RUN` narrowed, under
  `tools/mac_heavy.sh` at MAC_HEAVY_MAX_LOAD=48 (label `l4-b6-torch-pytest-2`): 75 passed, exit 0,
  2026-10-21T10:34Z.
- V: the hand oracle matches `mx.quantize` (mlx 0.31.2) on 2012 random groups.
- V: `campaign/lora-2b-preregistered.json` carries no D3 bars. The tool reads a `d3_gate` block and reports NOT_RUN
  (exit 3) without one (GAP-LORA2B-D3-THRESHOLD-NOT-MACHINE-READABLE-2026-10-21, owner human, owner item 5).
- Adversarial fixes (`856bdb5`): stale-PASS removal (NOT_RUN written first, final verdict by tmp + fsync +
  replace); every could-not-run path exits 3, an unexpected exception is NOT_RUN plus a traceback; the verdict
  records the pre-registration sha256; a held-out items path is REFUSED, exit 2, verdict without a tri-state;
  duplicate keys and overflowing numbers are refused; the step bound uses mlx's scale floor (1e-7);
  `fake_quant` refuses empty and non-bf16 input itself; `gold_row` is a non-bool int. bits 4 stays allowed: q4 is
  measured by the gate and never built (docstring).
- NOT RUN: the real-base verdict. No model was loaded and nothing was downloaded.
- Not built: a ledger row (the verdict goes to a JSON file); the 2B-Base and 0.8B-Base fake-quant base files.

### b2: qd-prep reds
- V, baseline on unmodified `d09222d`, no `QD_PYTHON` (label `l4-qdprep-base-noqdpython`,
  `cargo test -p qd-prep --no-fail-fast`, exit 101): lib 271 passed; `convert_licence_parity` 0 passed, 3 failed
  (`every_pending_id_is_one_python_does_not_admit`, `the_table_aliases_and_every_classification_match_python`,
  `concurrent_oracle_calls_never_share_an_input`; panic at `convert_licence_parity.rs:78`: no `.venv` in the
  worktree); `stress_v6` 52 passed, 2 failed, 3 ignored; every other target passed.
- V, the 2 S8 timeouts are pre-existing and out of scope: `s8_synth_email_...` and `s8_synth_jarvis_...` killed at
  `CLI_LIMIT` 1800 s (`stress_v6.rs:34`, panic at `:115`), debug build, load about 90. Owner card ft-b5110a5c; update
  appended to GAP-STRESS-V6-CLI-LIMIT-1800S-LOAD-SENSITIVE-2026-10-18. The bound is unchanged.
- V, fails-before for the first resolver test (main-checkout lookup removed): qd-preflight `schema_compat`
  1 passed 5 failed (new test fails at `schema_compat.rs:120`); qd-metal `ledger_rows` 7 passed 4 failed (new test
  fails at `ledger_rows.rs:113`).
- NOT RUN: the qd-prep fails-before run of the new test (stopped after hours in the queue).
- V, fails-before, qd-export hung interpreter (working tree only: `python_from` probe reverted to the unbounded
  call, new test kept; label `l4-fb-export-hung`, `cargo test -p qd-export --test ledger_real_producer
  a_hung_interpreter`, MAC_HEAVY_MAX_LOAD=48, exit 101, 2026-10-21T11:40Z): 0 passed, 1 failed in 20.15 s; the
  unbounded call waited out the 20 s stub and accepted it ("a hung interpreter must fail loudly, not be accepted",
  `ledger_real_producer.rs:250`).
- V, fails-before, later resolver changes (working tree only, qd-preflight `schema_compat.rs`: `GIT_DIR` and
  `GIT_WORK_TREE` removal, newline-only strip and `.git`-name check reverted, tests kept; label
  `l4-fb-preflight-resolver`, `cargo test -p qd-preflight --test schema_compat a_linked_worktree`, exit 101,
  2026-10-21T11:40Z): 0 passed, 1 failed; the separate-git-dir assertion at `schema_compat.rs:164` got
  `Some(".../qd-python-resolver-19125")`, expected `None`. Only the `.git`-name check is exercised by that
  assertion; the `GIT_DIR` and newline-only changes have no fails-before (NOT RUN).
- Both reverts were undone by hand (the `git checkout --` was held for a confirmation that timed out): `git status`
  showed only the untracked `scratch_site/`, `git diff` was empty, and the pin test passed (3 passed). Nothing
  reverted was committed.
- V, after the restore, qd-export hung-interpreter test (label `l4-fb-export-hung-after`, same command, exit 0,
  2026-10-21T12:25Z): 1 passed, 0 failed in 0.53 s (the stub was stopped inside its bound).
- NOT RUN (queued, `l4-fb-preflight-resolver-after`): the qd-preflight resolver test after the restore.
- V, qd-prep after-run with the final resolver block (label `l4-b2-after-prep`, `cargo test -p qd-prep --test
  convert_licence_parity`, no `QD_PYTHON`, MAC_HEAVY_MAX_LOAD=48, exit 0, 2026-10-21T10:32Z): 4 passed, 0 failed
  (the 3 former reds plus `a_linked_worktree_resolves_the_main_checkouts_venv`). Only this target ran; the full
  `cargo test -p qd-prep` (including the 2 pre-existing S8 timeouts) was not rerun.
- V, qd-preflight after-run (label `l4-b2-after-preflight-2`, `cargo test -p qd-preflight --test schema_compat`,
  no `QD_PYTHON`, exit 0, 2026-10-21T10:37Z): 6 passed, 0 failed (the 4 former reds, the missing-interpreter test
  and the new worktree test).
- V, qd-export after-run (label `l4-b2-after-export`, `cargo test -p qd-export --test ledger_real_producer`,
  no `QD_PYTHON`, exit 0, 2026-10-21T10:37Z): 5 passed, 0 failed, including
  `a_hung_interpreter_fails_loudly_within_the_bound` and the worktree test.
- V, qd-metal after-run (label `l4-b2-after-metal-2`, `cargo test -p qd-metal --test ledger_rows`, no
  `QD_PYTHON`, exit 0, 2026-10-21T10:37Z): 11 passed, 0 failed. The first qd-preflight and qd-metal jobs had
  expired at the 4 h wait limit (exit 76) and were re-queued with MAC_HEAVY_MAX_WAIT_S=86400.
- NOT RUN: the qd-prep run with `QD_PYTHON` set (refused by the command gate: a path outside the repository).
  All four after-runs without `QD_PYTHON` passed, so the worktree resolution works.
- V: pin test (`test_qd_python_resolver_pinned.py`) passes and fails on drift; it now requires every crate source
  that reads `QD_PYTHON` to be one of four pinned sites carrying the block (qd-prep, qd-metal, qd-preflight,
  qd-export). The adversarial file's `SITES` gained the qd-export site.

## Changed (commits on lane/ft-57fc7484-l4)
- `1301511` tools/d3_quant_gate.py, python/tests/test_d3_quant_gate.py.
- `959eb24` the resolver block in three crates, the pin test, gaps, this file.
- `a26bc76` adversarial tests (cherry-pick of `925731a`).
- `856bdb5` the D3 and resolver fixes from the adversarial pass; qd-export on the block.
- the commit that follows: qd-export's interpreter calls bounded by `output_bounded` (60 s), the hung-interpreter
  test, this file.

## Open (gap ids)
- GAP-LORA2B-D3-THRESHOLD-NOT-MACHINE-READABLE-2026-10-21 (owner: human).
- GAP-STRESS-V6-CLI-LIMIT-1800S-LOAD-SENSITIVE-2026-10-18 (ft-b5110a5c).
- GAP-W6C-QDPREP-LICENCE-PARITY-TESTS-NEED-QD-PYTHON-IN-WORKTREE-2026-10-18: stays resolved-with-residual until the
  after-runs pass; then append a closing line.
- GAP-LORA2B-QUANT-GATE-TOOL-MISSING-2026-10-17: resolved-with-residual (real verdict, ledger row, base files).
- Machine lock: another session's `tessl-gpurun2` held `mac_heavy` for over 3 h; owner item (machine load).

## First command for the next lane
One crate per job, with the lock free (labels as queued):

    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-prep cargo test -p qd-prep --test convert_licence_parity
    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-preflight cargo test -p qd-preflight --test schema_compat
    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-metal cargo test -p qd-metal --test ledger_rows
    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b2-after-export cargo test -p qd-export --test ledger_real_producer
    MAC_HEAVY_MAX_LOAD=48 bash tools/mac_heavy.sh l4-b6-torch-pytest-2 make torch-pytest 'TORCH_PYTEST_RUN=/Users/bharath/.local/bin/uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis --with datasketch python -m pytest python/tests/test_d3_quant_gate.py python/tests/test_d3_quant_gate_adversarial.py python/tests/test_qd_python_resolver_pinned.py -o addopts='
