# The one entry point for every gate in this repo.
#
# Why this file exists
# --------------------
# There is no git remote, no .github, no CI. Every gate here has always been run by
# hand, which is why they stopped being run. The repo has produced the same defect
# three times: the ledger went its whole history without a single row; `ruff>=0.6` sat
# declared in pyproject's `dev` extra and had never been installed, let alone run; and
# ruff's own isort was resolving `qd_data`/`qd_train`/`qd_wire`/`qd_label` as
# third-party, so I001 silently accepted 26 mis-grouped import blocks (3 findings
# before `[tool.ruff].src` was set, 29 after). A gate that exists only in
# configuration is not a gate.
#
# The three outcomes
# ------------------
# Matching `python/qd_train/tristate.py` and `docs/ledger-schema.md`, every gate here
# distinguishes three results, never two:
#
#   0  PASS    the gate ran and passed
#   1  FAILED  the gate ran and failed
#   3  NotRun  the gate could not run; the reason is printed
#
# A gate that cannot run NEVER reports the same result as one that ran and passed.
#
# One caveat, stated because it would otherwise be a silent lie: **GNU make collapses
# every failed recipe to its own exit status 2** (measured: a recipe ending `exit 3`
# yields `make: *** [lint] Error 3` and `$?` == 2). So `make <target>` exits 0 or 2,
# never 1 or 3. The three-way distinction survives in:
#   * each gate's printed `NotRun: <reason>` / `FAILED` line,
#   * the `RESULT:` line `make gates` prints last,
#   * the real exit status of the underlying commands, which `gates` captures itself
#     -- that is why `gates` runs the gate commands inline rather than recursing into
#     make, which would hand it 2 for both FAILED and NotRun and erase the distinction.
# `qd_train.ledger record` already returns 0/1/3 natively; call it directly if you need
# the code rather than the text.
#
# Two traps this file is written to avoid, both of which bit during its construction:
#
#  1. **Never pipe a gate's output.** `cargo test --workspace | tail -30` exits with
#     *tail's* status, so a workspace that does not compile reports success. That is
#     how this repo's currently-broken Rust build was first mis-read as green.
#  2. **Never narrow `ruff --select`.** RUF100 ("unused noqa") is evaluated against the
#     *enabled* rule set, so `--select F401,RUF100` declares every `# noqa: E402`
#     unused -- including load-bearing ones. Running that with `--fix` during this
#     lane stripped 12 live directives and took the tree from 0 E402 findings to 27.
#     The lint gate passes no `--select`. `python/tests/test_lint_gate.py` pins it.
#
# Usage:  make            # every gate, with a summary
#         make lint       # one gate
#         make help

REPO      := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV      := $(REPO)/.venv
PY        := $(VENV)/bin/python
RUFF      := $(VENV)/bin/ruff
LEDGER    := $(REPO)/ledger/runs.jsonl

# The torch environment, and the launcher that borrows it without modifying it.
#
# `$(PY)` deliberately has no torch: the core gate stays fast and torch-free. The cost
# of that, measured 2026-09-20, is that **five whole modules never ran in this gate** --
# test_heads.py, test_fused_ce.py, test_byte_train.py, test_byte_decider.py and
# test_remap_torch.py, plus part of test_byte_batch.py. They are exactly the model-path
# modules, and test_heads.py is what pins the abstain-row-last layout against
# `crates/qd-runtime/src/answer.rs`.
#
# **The coverage pair could not see it.** A module-level skip collapses N tests into ONE
# skipped item, so the row read `1316/1323` -- 99.5% -- while the true denominator was
# 1407 and real coverage was 93.5%. The denominator was computed in the same environment
# that caused the shortfall, which is why it looked almost complete.
#
# `uv run --no-project` borrows the ml venv and layers pytest on top for the duration of
# the run: nothing is installed into either venv and no project dependency is added.
# Overridable so this is not pinned to one machine.
ML_VENV   ?= /Users/bharath/.venvs/ml
ML_PY     := $(ML_VENV)/bin/python
UV        ?= /Users/bharath/.local/bin/uv

# The version uv.lock already resolved `ruff>=0.6` to. The standalone ruff on this host
# is a different release (0.15.20) and is NOT the gate: the gate is the locked version
# inside the project venv, so what runs is what the lockfile describes.
RUFF_PIN  := 0.16.8

NOT_RUN   := 3

TOOLCHAIN := cargo $(shell cargo --version 2>/dev/null | awk '{print $$2}' || echo absent) / CPython $(shell $(PY) -c 'import platform;print(platform.python_version())' 2>/dev/null || echo absent) (.venv) / ruff $(shell $(RUFF) --version 2>/dev/null | awk '{print $$2}' || echo absent)

# .pyc caching has produced three false mutation survivors in this repo.
export PYTHONDONTWRITEBYTECODE := 1

# --------------------------------------------------------------------------------
# Gate commands.
#
# `LINT_CMD` and the two `LEDGER_*` commands are single-line shell commands defined
# once and used twice: by their own target, and by `gates`, which needs the true
# 0/1/3 status that recursive make would destroy.
#
# The three *counted* suites are different, and the difference matters. `gates` does
# not invoke `cargo-test`, `pytest` or `torch-pytest` at all -- it runs them through
# `ledger-record`, which re-runs each one itself in order to parse its counts. So each
# counted suite has two entry points that both really execute it:
#
#   `make torch-pytest`   -> $(TORCH_PYTEST_CMD), for a person at a terminal
#   `make gates`          -> $(LEDGER_RECORD_CMD)'s --suite argument, which writes the row
#
# They were written out separately, and on 2026-09-21 they drifted: `--with datasketch`
# was added to the first and not the second, so `make torch-pytest` ran 1886 tests while
# the row from `make gates` reported 1858 under a `detail` field naming a command nobody
# had run. Neither count was wrong about its own run. The row was wrong about which run
# it was, which is worse, because the row is the only durable account.
#
# So the command is now spelled once, in a `*_RUN` variable, and both entry points are
# built from it. `python/tests/test_lint_gate.py` asserts that -- comparing argv, via
# make's own `-n` expansion -- because the comment above this one asserted the same
# property while it was false.
#
# The `*_RUN` payloads carry no quotes. `qd_train.ledger.parse_command` splits the
# recorded string with shlex and runs it as argv with no shell, so a path containing a
# space could not survive the recorder's side however the target were written; leaving
# both sides unquoted means they fail together rather than one silently differing.
# --------------------------------------------------------------------------------

# Whole repo, not just python/: stack/ and tools/ are Python too, and a gate scoped so
# it cannot fail is the defect this file closes. ruff honours .gitignore, so .venv/ and
# the sibling agent worktrees under .claude/worktrees/ are not scanned.
LINT_CMD = if [ ! -x "$(RUFF)" ]; then printf 'NotRun: lint - ruff is not installed in %s.\n' "$(VENV)"; printf "  'ruff>=0.6' IS declared in pyproject [project.optional-dependencies].dev, and\n"; printf '  uv.lock already resolves it to %s. Declared is not installed.\n' "$(RUFF_PIN)"; printf "  Install the gate:  uv pip install --python %s 'ruff==%s'\n" "$(PY)" "$(RUFF_PIN)"; printf '  This is NOT a pass: nothing was linted.\n'; exit $(NOT_RUN); fi; "$(RUFF)" check --config "$(REPO)/pyproject.toml" "$(REPO)"

# -D warnings: clippy exits 0 with any number of warnings otherwise, which is a gate
# that cannot fail.
CLIPPY_CMD = if ! command -v cargo > /dev/null 2>&1; then printf 'NotRun: clippy - cargo is not on PATH, so no Rust was examined.\n'; exit $(NOT_RUN); fi; if ! cargo clippy --version > /dev/null 2>&1; then printf 'NotRun: clippy - cargo is present but the clippy component is not installed.\n  Install it:  rustup component add clippy\n'; exit $(NOT_RUN); fi; cargo clippy --manifest-path "$(REPO)/Cargo.toml" --all-targets -- -D warnings

CARGO_TEST_RUN = cargo test --manifest-path $(REPO)/Cargo.toml --workspace
CARGO_TEST_CMD = if ! command -v cargo > /dev/null 2>&1; then printf 'NotRun: cargo-test - cargo is not on PATH, so no Rust test was executed.\n'; exit $(NOT_RUN); fi; $(CARGO_TEST_RUN)

# `-o addopts=` clears the `-q --strict-markers --strict-config` in pyproject. Passing
# another -q would make it -qq, which prints NO summary line -- and a green run with no
# counts is recorded by the ledger as NotRun, correctly, because a pass with no counts
# cannot be told from a suite that collected nothing.
PYTEST_RUN = $(PY) -m pytest $(REPO)/python/tests -o addopts=
PYTEST_CMD = if [ ! -x "$(PY)" ]; then printf 'NotRun: pytest - no interpreter at %s; the project venv is missing.\n' "$(PY)"; exit $(NOT_RUN); fi; if ! "$(PY)" -m pytest --version > /dev/null 2>&1; then printf 'NotRun: pytest - pytest is not installed in %s.\n' "$(VENV)"; exit $(NOT_RUN); fi; $(PYTEST_RUN)

# The same suite again, in an environment that has torch. NOT a substitute for the run
# above: that one answers "does the torch-free core still work without torch", this one
# answers "does the model path work at all". Both are real questions and the row carries
# both.
#
# A missing uv or ml venv makes this NotRun, never a pass -- the five modules would go
# back to being invisible, and invisible is what this suite exists to end.
#
# `--with datasketch` is there for a reason worth stating, because it looks like a stray
# dependency and is not. `datasketch` is declared in `[project.dependencies]` and is absent
# from the ml venv, so `test_minhash.py`'s module-level `importorskip` fired and the whole
# module -- 29 tests -- did not run here. Pytest counts a module-level skip as ONE, so this
# suite reported `1850 passed, 2 skipped` while 29 tests sat behind one of those two. The
# coverage pair in the ledger row said 1850/1852 and understated the un-run tests by 28:
# this repository's own "never present a capped sample as complete coverage", in the
# instrument built to prevent it.
#
# The tests themselves lose nothing by it -- they run in the torch-free suite, which has
# datasketch -- so this is a reporting fix rather than a coverage one. It is provisioned
# the same ephemeral way pytest and hypothesis already are; no virtualenv is modified.
# `python/tests/test_declared_dependencies.py` names the next one rather than letting it
# become another silent "1 skipped" -- and it is what caught the drift described above,
# by failing in the recorder's un-provisioned run while passing in the target's.
TORCH_PYTEST_RUN = $(UV) run --no-project --python $(ML_PY) --with pytest --with hypothesis --with datasketch python -m pytest $(REPO)/python/tests -o addopts=
TORCH_PYTEST_CMD = if [ ! -x "$(UV)" ]; then printf 'NotRun: torch-pytest - no uv at %s, so the torch suite could not be launched.\n  Override with: make UV=/path/to/uv\n  This is NOT a pass: the model-path modules were not run.\n' "$(UV)"; exit $(NOT_RUN); fi; if [ ! -x "$(ML_PY)" ]; then printf 'NotRun: torch-pytest - no interpreter at %s; the torch environment is missing.\n  Override with: make ML_VENV=/path/to/venv\n  This is NOT a pass: the model-path modules were not run.\n' "$(ML_PY)"; exit $(NOT_RUN); fi; PYTHONDONTWRITEBYTECODE=1 $(TORCH_PYTEST_RUN)

# The documented one-command path from docs/ledger-schema.md: runs each suite, writes
# one `build` row, prints the row id alone on stdout. Its own exit codes are already
# 0 / 1 / 3 with these same meanings.
#
# Only the two suites with parsers (cargo, pytest) are passed as --suite. ruff and
# clippy deliberately are not: neither prints a test summary, so run_suite would read a
# CLEAN ruff run as `not_run` (exit 0, no counts -- indistinguishable from having
# checked nothing) and a dirty one as `ran/failed`. Rather than teach the ledger a
# third parser, the lint gate is asserted inside the pytest suite itself
# (python/tests/test_lint_gate.py), where it is counted and reaches the row like any
# other test.
LEDGER_RECORD_CMD = if [ ! -x "$(PY)" ]; then printf 'NotRun: ledger-record - no interpreter at %s; no row was written.\n' "$(PY)"; exit $(NOT_RUN); fi; PYTHONPATH="$(REPO)/python" "$(PY)" -m qd_train.ledger record --ledger "$(LEDGER)" --repo "$(REPO)" --toolchain "$(TOOLCHAIN)" --suite "cargo_test_workspace=$(CARGO_TEST_RUN)" --suite "pytest_python_tests=$(PYTEST_RUN)" --suite "pytest_torch_python_tests=$(TORCH_PYTEST_RUN)"

LEDGER_VERIFY_CMD = if [ ! -x "$(PY)" ]; then printf 'NotRun: ledger-verify - no interpreter at %s; the chain was not checked.\n' "$(PY)"; exit $(NOT_RUN); fi; if [ ! -f "$(LEDGER)" ]; then printf 'NotRun: ledger-verify - no ledger at %s; there is no chain to check.\n  An absent ledger is not a verified one.\n' "$(LEDGER)"; exit $(NOT_RUN); fi; PYTHONPATH="$(REPO)/python" "$(PY)" -m qd_train.ledger verify --ledger "$(LEDGER)"

.PHONY: all gates help lint clippy cargo-test pytest torch-pytest ledger-record ledger-verify

all: gates

help:
	@echo 'Gate exit codes: 0 PASS, 1 FAILED, 3 NotRun.'
	@echo 'GNU make collapses any failure to 2, so `make X` gives 0 or 2; read the'
	@echo 'printed NotRun:/FAILED lines, or the RESULT: line from `make gates`.'
	@echo ''
	@echo '  gates          every gate below, with a summary; the default'
	@echo '  lint           ruff over the whole repo, full configured rule set'
	@echo '  clippy         cargo clippy --all-targets, warnings denied'
	@echo '  cargo-test     cargo test --workspace'
	@echo '  pytest         pytest with counts (torch-free venv)'
	@echo '  torch-pytest   the same suite where torch exists; +89 model-path tests'
	@echo '  ledger-record  run the counted suites, append one `build` row, print its id'
	@echo '  ledger-verify  recompute the ledger hash chain'

lint:
	@$(LINT_CMD)

clippy:
	@$(CLIPPY_CMD)

cargo-test:
	@$(CARGO_TEST_CMD)

pytest:
	@$(PYTEST_CMD)

torch-pytest:
	@$(TORCH_PYTEST_CMD)

ledger-record:
	@$(LEDGER_RECORD_CMD)

ledger-verify:
	@$(LEDGER_VERIFY_CMD)

# --------------------------------------------------------------------------------
# The aggregate.
#
# Runs every gate even after one fails -- a run that stops at the first red tells you
# about one gate, and this file exists because nobody knew the state of any of them.
# Each gate runs in its own subshell so its true 0/1/3 status is observable here.
# --------------------------------------------------------------------------------
gates:
	@tally=`mktemp`; fail=0; notrun=0; \
	for t in lint clippy ledger-record ledger-verify; do \
	  printf '\n======== %s ========\n' "$$t"; \
	  case "$$t" in \
	    lint)          ( $(LINT_CMD) );          rc=$$? ;; \
	    clippy)        ( $(CLIPPY_CMD) );        rc=$$? ;; \
	    ledger-record) ( $(LEDGER_RECORD_CMD) ); rc=$$? ;; \
	    ledger-verify) ( $(LEDGER_VERIFY_CMD) ); rc=$$? ;; \
	  esac; \
	  if [ $$rc -eq 0 ]; then \
	    printf '  %-15s PASS\n' "$$t" >> "$$tally"; \
	  elif [ $$rc -eq $(NOT_RUN) ]; then \
	    notrun=1; printf '  %-15s NotRun\n' "$$t" >> "$$tally"; \
	  else \
	    fail=1; printf '  %-15s FAILED (exit %s)\n' "$$t" "$$rc" >> "$$tally"; \
	  fi; \
	done; \
	printf '\n======== gate summary ========\n'; cat "$$tally"; rm -f "$$tally"; \
	if [ $$fail -ne 0 ]; then \
	  printf '\nRESULT: FAILED - at least one gate ran and failed.\n'; exit 1; \
	elif [ $$notrun -ne 0 ]; then \
	  printf '\nRESULT: NotRun - nothing failed, but a gate could not run.\n'; \
	  printf 'This is not a pass. An unexamined gate is not a clean one.\n'; \
	  exit $(NOT_RUN); \
	fi; \
	printf '\nRESULT: PASS - every gate ran and passed.\n'
