# HANDOFF — Stack, Ledger, Eval, Teacher · 2026-09-19

Covers the lanes carried directly rather than delegated: the ledger and eval harness, the
linear baseline, the teacher path, the needle-hunk suite, the preflight gate, and S1.

The Data, Mutate and Runtime lanes are in flight with their own agents and will write their own
handoffs.

---

## What was measured

Real numbers, from runs, not estimates. Anything not run is named as not run.

| Suite | Result | Command |
| --- | --- | --- |
| Python (`python/tests/`) | **124 passed**, 0 failed, in 2.10 s | `.venv/bin/python -m pytest python/tests/ -p no:cacheprovider` |
| `qd-preflight` (Rust) | **22 unit + 4 cross-language = 26 passed**, 0 failed | `cargo test -p qd-preflight` |
| `qd-mutate` (Rust) | **90 passed, 3 failed** — in flight, not mine | `cargo test -p qd-mutate` |
| tessl (reference, untouched) | 306 passed, 0 failed, 1 ignored, on-GPU | audited, not modified |
| S1 image build | **failing, 4 distinct causes found and fixed; rebuild in flight** | `podman build --platform linux/amd64 -f stack/Containerfile stack/` |
| fla fast-path gate | **NOT RUN** — needs CUDA; returns `not_run` here, `--require-fast` exits 1 | `stack/verify_fast_path.py` |
| CUDA preflight | **NOT RUN** — no driver on this host; `--require-all` exits 2 | `target/debug/qd-preflight` |

Measured incidentally, and load-bearing for two decisions:

- **kappa at n=50 is underpowered.** True kappa ~0.66 gives a 95% CI **0.316 wide**; the lower bound
  clears 0.6 in **10%** of draws. At n=300 the width is 0.132. The 0.6 threshold was **not** changed.
- **podman cross-arch works here.** `qemu-x86_64` registered; amd64 executes; **~1.4x** overhead
  (3.09 s vs 2.24 s on a compile + 2e8-iteration float loop, images cached).

## What changed

| Commit | Contents |
| --- | --- |
| `4cd4cfa` | Scaffold: `CLAUDE.md`, schema-api, ledger-schema, hardening, mutation-operators specs |
| `eb78d34` | `qd_train/{tristate,ledger,baseline,eval_harness,agreement}`, `qd_label/`, teacher plan + rubric |
| `1149dc8` | `crates/qd-preflight`, `stack/` (S1), `qd_train/needle.py` |
| *staged, uncommitted* | Teacher/trainer environment split. Message saved at `.git/COMMIT_MSG_TMP` |

**Uncommitted work is staged and safe.** `git -C` is blocked intermittently by the harness and `cd`
is blocked outright, so the final commit could not be made from this session. To land it:

```bash
cd ~/Code/research/qwen-decision && git commit -F .git/COMMIT_MSG_TMP && rm .git/COMMIT_MSG_TMP
```

## What is open

Gap ids in `gaps.jsonl` (17 records). The ones that block work:

| Gap | Blocks | Who resolves |
| --- | --- | --- |
| `GAP-GITPULSE-TRUST` | **The whole kernel lane.** tessl's worktree state is unverified; every facet returns `REPOSITORY_TRUST_REQUIRED`. Re-checked at handoff time, still untrusted | Human: open `~/Code/research/tessl` in GitPulse and trust it |
| AgentPack gate | The natural-distribution pool. Authenticated fetch 403s | Human: accept terms on HuggingFace |
| `GAP-S1-NO-DOCKER` (residual) | Nothing now — cross-arch resolved. Residual: VM has 1.89 GiB; `podman machine set --memory 8192` if the torch layer OOMs | Host owner |
| `GAP-TESSL-COPY-DRIFT-SEMANTIC` | K6's golden. **Canonical tessl carries the wrong GELU reference** and its test passes vacuously because random operands never reach the ±20 clamp | Kernel lane: derive from `gelu.h`, not from either test |
| MNLI / STS-B licences (`other` / `unknown`) | Two slots of the open mixture. ANLI is already out — `cc-by-nc-4.0` | Human call; wired as not-admitted, one-line switch |

Decision left open deliberately, in `docs/teacher-plan.md` §6: whether to label **300** instead of 50
for the agreement set, keep 50 and record the weaker guarantee, or keep 50 and require the CI lower
bound to clear 0.6. `kappa_gate()` implements none of them as a silent default.

## The first command for the next lane

**Kernel lane** — blocked until GitPulse trust. Once unblocked, the fixtures already exist:

```bash
ls ~/Code/research/tessl/tests/fixtures/gdn/   # 108 files, rule="published" in every name
cargo test --manifest-path ~/Code/research/tessl/Cargo.toml
```

Read `AUDIT/gdn-reference-and-contracts.md` before writing the f64 reference. Note three corrections
the plan gets wrong: **K7's alpha is `sigmoid(a_gate + decay_bias).clamp(1e-4, 1.0)`**, not the
`exp(-exp(A_log)·softplus(...))` in the plan — that is the Mamba2 decay from a different mixer in the
same file. The clamp is **already in the reference** and is load-bearing (unclamped diverges 3.99e-04
against a 5.8e-07 noise floor). And the published rule differs from `repo` at **one** assignment
(`mixers.py:734`), not two — patching "two places" creates a second source of truth.

**Human labelling** — unblocked now, and it is the long pole:

```bash
PYTHONPATH=~/Code/research/qwen-decision/python \
  ~/Code/research/qwen-decision/.venv/bin/python -m qd_label \
  --pool <pool.json> --store labels.jsonl label
```

Then `... ceiling` for intra-rater kappa — the bound the teacher's kappa is measured against, which
the plan never establishes.

**Session 0** — not yet. See below.

## Lambda: not needed yet

Weeks 1–3 are Mac-local at $0 and are not finished. Session 0 is the first rental and it should not
launch until:

1. the S1 image builds clean (in flight),
2. `qd-mutate`, `qd-data` and `qd-runtime` are green,
3. the 50 hand labels exist, and the rubric is drafted from them.

When it does launch, two changes to its scope, both argued in `docs/teacher-plan.md`:

- **+1–2 h for teacher rubric iteration and the kappa pass**, moving them off the Mac. ~$3–6. This is
  also a correctness fix: kappa measured on a 4-bit MLX build is not kappa for the bf16 vLLM run that
  labels the 80K, and the plan assumed they transfer.
- **Two images, not one** (`train.lock` CUDA 12, `teacher.lock` CUDA 13). One lockfile does not
  resolve; see `docs/plan-corrections.md` STACK-1.

Cost note for the block itself: the 40 h wall-clock cap is **$1,277 for the block alone** at
$31.92/h, so hitting the cap is also a ~15% overrun on the $1,250 estimate — inside the $2,000
ceiling, but it should be stated rather than discovered.
