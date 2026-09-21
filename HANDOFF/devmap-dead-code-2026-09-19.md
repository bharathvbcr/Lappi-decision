# Handoff — dead-code wiring and hardening (2026-09-19)

Branch `claude/devmap-dead-code-hardening-fc226b`. **Uncommitted in the worktree** at the
time of writing; see "What changed" for the file list.

## What was measured

No ledger rows: this lane ran no training, eval or GPU job, so it recorded none. The
measurements are test runs, and both numbers are stated before and after.

| Suite | Before | After |
| --- | --- | --- |
| `cargo test` (workspace) | **could not run** — manifest load failed | 22 unit + 5 integration, all pass |
| `pytest python/tests` | 124 pass | 140 pass |
| `ruff check` on the files this lane touched | 19 pre-existing | 19 pre-existing, 0 added |

The Rust "before" is the finding, not a caveat: the workspace had **never** built.

## What changed

Four defects, each verified against the unmodified code before the fix.

**1. The Rust preflight gate had never compiled.** `Cargo.toml` named workspace members
`crates/qd-mutate` and `crates/qd-runtime`, neither of which has ever existed. Cargo
fails to *load* a workspace with a missing member, so every invocation failed —
including `cargo test -p qd-preflight`. The gate added in 1149dc8 ("S1 environment, a
Rust preflight gate") was committed without ever being built. Fixed by listing only the
crate that exists; 27 tests run and pass now. → `Cargo.toml`

**2. The cross-language schema test reported green without running.**
`tests/schema_compat.rs` resolved exactly one interpreter, `<repo>/.venv/bin/python`,
which this repo does not have. Each test printed `SKIPPED` to stderr and `return`ed;
libtest has no "not run" status, so cargo recorded **4 passed** while the Rust and
Python tri-state encoders had never once been compared — and `cargo test` hides a
passing test's stderr, so the word SKIPPED reached nobody. Two further fail-open paths
(`.output().ok()?`, `from_slice(...).ok()?`) turned a spawn error or malformed output
into the same silent pass.

Now: candidates are `QD_PYTHON` → `<repo>/.venv/bin/python` → `python3`; a missing
interpreter **fails** with a remediation message; every error path panics; invocations
are bounded at 30 s. The comparison genuinely executes here for the first time, and the
two encoders agree. Demonstrated inversion, same host, `PATH` stripped of python:
pre-fix `4 passed`, post-fix `4 failed`. → `crates/qd-preflight/tests/schema_compat.rs`

**3. A capped sample promoted as if it were complete coverage.**
`Ledger.promotion_verdict` never looked at `n`/`n_total`, so a gate that passed on 1 of
1000 eligible items promoted exactly like one that passed on 1000 of 1000.
`Ran.is_complete_coverage` and `Ran.coverage_str` existed to prevent precisely this and
had **zero callers** — the schema's Coverage rule ("a report renders `n/n_total`") was
specified and then never enforced.

Now promotion condition 6: a `ran/passed` result stating `n < n_total` is itemized as a
refusal, coverage is rendered on every line of a verdict, and a `PROMOTE` states the
weakest coverage it promoted on. Unstated coverage (`n is None`) is deliberately *not*
treated as capped — many gates are a single observation with no population, and
refusing those would make promotion unreachable rather than honest. This makes
promotion strictly harder, never easier. → `python/qd_train/ledger.py`,
`docs/ledger-schema.md`

**4. `qd_label` shipped in no wheel.** `pyproject.toml` declared
`packages = ["python/qd_data", "python/qd_train"]`. `python/qd_data` does not exist and
hatchling drops a missing package silently rather than failing, so the built wheel
contained `qd_train` alone and the hand-labelling CLI that produces the held-out set
was absent. Verified by inspecting the wheel before and after. → `pyproject.toml`

**New: the entry point the schema names.** `docs/ledger-schema.md` referenced
`qd-ledger verify`, which nothing implemented — `verify_chain`, `promotion_verdict`,
`PromotionVerdict.__str__` and `coverage_str` were reachable only from tests, and
`qd_train` had no entry point at all. `python -m qd_train` now offers `verify`,
`families`, `promotion`, `show`. It **renders and does not judge**: output uses the
schema's vocabulary (`ran passed=true`, `not_run`) rather than a PASS of its own, the
only verdict shown is the one `promotion_verdict` computes, and it adds no condition
and cannot discharge a gate (repo rule 7). Exit codes separate "refuses to promote" (1)
from "could not read the ledger" (2). → `python/qd_train/__main__.py`

Files: `Cargo.toml`, `Cargo.lock` (new, tracked — `qd-preflight` is a binary and this
repo pins `stack/requirements.lock` for the same reason), `.gitignore`,
`crates/qd-preflight/tests/schema_compat.rs`, `docs/ledger-schema.md`, `gaps.jsonl`,
`pyproject.toml`, `python/qd_train/ledger.py`, `python/qd_train/__main__.py` (new),
`python/tests/test_ledger.py`, `python/tests/test_ledger_cli.py` (new).

Every fix ships a test that fails against the pre-fix code: the five coverage tests in
`test_ledger.py` were run against `git show HEAD:python/qd_train/ledger.py` and all five
failed, printing `PROMOTE … every gate and control ran and passed` while a control had
seen 2 of 300 items. `a_missing_interpreter_fails_rather_than_skipping` has nothing to
call pre-fix.

## What is open

- **`GAP-DEVMAP-DEAD-UNDERREPORTS-METHODS`** (new, `gaps.jsonl`). `devmap_dead_symbols`
  returned one row at confidence 0.4 with `walk_incomplete` set, and would have
  surfaced **none** of the four symbols wired here: Python resolution is 356 net
  permille with 1092 `uninferred_receiver` sites, so `obj.method()` is largely
  invisible and every method reads as callerless-and-unconfirmable. It also cannot see
  toolchain-level unreachability (defect 1) at all. Findings in this lane are
  **graph-informed, not graph-confirmed**: in-degree was computed directly over
  `generation_nodes`/`generation_edges`, then every candidate corroborated with
  `rg -uu`.
- **`EvalReport.summary()` and `NeedleReport.summary()` are still dead** —
  `python/qd_train/eval_harness.py:228` and `python/qd_train/needle.py:158`, zero call
  sites anywhere, including tests. Kept rather than deleted: both are the human-readable
  renderer for a result the S7 eval path will produce, and Chesterton's Fence says a
  plausible future caller is not proof of death. They should be wired by whoever builds
  the eval entry point, or removed then. Note both render tri-states via `{v}` (the raw
  dataclass repr) rather than `coverage_str()`, so they do not currently satisfy the
  schema's Coverage rule.
- `python/qd_data` and `crates/qd-{mutate,runtime}` remain unwritten. Both the wheel
  packages list and the cargo members list now name only what exists; add each in the
  commit that creates it.
- 19 pre-existing `ruff` findings in `ledger.py` / `test_ledger.py` (unused imports,
  `E501`, `SIM105`) were left alone as an unrelated drive-by. Repo-wide the count is 79.

## Exact first command for the next lane

```
cargo test --manifest-path /Users/bharath/Code/research/qwen-decision/Cargo.toml && \
uv run --no-project --python 3.14 --with pytest --with numpy -- \
  python -m pytest python/tests -c pyproject.toml --rootdir .
```

Both must be green before anything else is believed. To read a ledger once runs exist:

```
python -m qd_train --ledger ledger/runs.jsonl verify
python -m qd_train --ledger ledger/runs.jsonl families
python -m qd_train --ledger ledger/runs.jsonl promotion --seed-family <hash from families>
```
