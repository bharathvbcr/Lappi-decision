# HANDOFF — S1 images complete · 2026-09-19

Supersedes the S1 section of `stack-and-decision-2026-09-19.md`. Both S1 images now exist, both
build clean, and — this is the part that took ten attempts to get honest — **both gates have been
observed to actually run inside their images.**

---

## What was measured

Real numbers from runs. Anything not run is named as not run.

| Suite | Result | Command |
| --- | --- | --- |
| Rust workspace | **307 passed**, 0 failed, 0 ignored | `cargo test --workspace` |
| Python | **420 passed, 1 skipped**, 0 failed | `.venv/bin/python -m pytest python` |
| clippy, all targets | **0 warnings** | `cargo clippy --workspace --all-targets` |
| Production `unwrap`/`expect` | **0** | `tools/audit_unwrap.py <repo>` — note the path argument; it defaults to `.` and will scan all of `~/Code` without it |
| Render line-break probe | **PASS**, 12/12 blocked | `tools/probe_render_breaks.py` |
| Trainer image | `localhost/qd-train:s1`, 17.4 GB, 14/14 | build 10 |
| Teacher image | `localhost/qd-teacher:s1`, 10.7 GB, 14/14 | first attempt |
| Fast path live | **NOT RUN** — needs CUDA | Session 0 |
| Teacher readiness | **NOT RUN** — needs CUDA | Session 0 |

Verified *inside* each image, not inferred from the build log:

- **Trainer**: `uname -m` = `x86_64`, `torch-2.10.0+cu128`, `causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so`
  present — so the prebuilt wheel carried its compiled extension and nvcc never ran under qemu.
- **Teacher**: `x86_64`, `vllm-0.29.0`, `torch-2.13.0`, `transformers-5.17.0`, all matching `teacher.lock`.
- **Both gates**: report `state: not_run` with a specific reason, exit **0** without their
  `--require-*` flag and **1** with it. The container's own exit code, captured without a pipe.

## The commands

```bash
podman build --platform linux/amd64 -t qd-train:s1   -f stack/Containerfile          .
podman build --platform linux/amd64 -t qd-teacher:s1 -f stack/Containerfile.teacher  .
```

**The context is the repo root for both, not `stack/`.** That is load-bearing — see below.

## What changed, and why

Five defects, four of them in work I had already reported as done.

1. **`GAP-STACK-GATE-CANNOT-IMPORT`** — the ninth build tagged clean and the gate inside it died on
   `ModuleNotFoundError: No module named 'qd_train'`. Every property verified was a property of the
   artifact; none was a property of the artifact doing its job. Fixed by mirroring the repo layout
   into the image (`stack/` beside `python/` under `/opt/qd`) so the gate's
   `parents[1] / "python"` is right in both places, with no container-only branch to drift.
2. **`GAP-STACK-NO-TEACHER-IMAGE`** — `teacher.lock` existed at 196 pins with nothing consuming it.
   The tell was a dangling cross-reference to a README section 6 that did not exist. Session 0 loads
   the teacher through vLLM and could not have launched.
3. **`GAP-STACK-GATES-UNTESTED`** — the S1 status table claimed the `not_run` path was tested while
   no test referenced either gate script. Now 15 tests in `python/tests/test_stack_gates.py`, with
   the fail-open property **mutation-checked**: making `--require-ready` accept `NotRun` fails 2
   tests, confirmed by running the mutated source before reverting.
4. **`GAP-STACK-UV-CACHE-VAR-INERT`** — `PIP_NO_CACHE_DIR=1` is not read by `uv`; `UV_NO_CACHE` is.
   The wheel cache was being committed into the layer.
5. **`GAP-RUNTIME-DIGEST-SECOND-IMPL`** — `Context::digest` reimplemented `crate::sha256` inline.

## What is open

- **`GAP-DEVMAP-DEAD-FALSE-POSITIVES`** (`closed-no-defect`, but read it before using
  `devmap_dead_symbols` here). All 3 candidates were resolver artifacts, and the **0.90-confidence**
  one was the only one *without* a possibly-called warning — and the only one whose deletion would
  have done damage. Treat that output as a list to disprove, never a delete list.
- **The `~/Code` DevMap store** still returns `database disk image is malformed` to the MCP server
  while the SessionStart hook reads it at generation 172. Per-repo stores are unaffected —
  qwen-decision's own index was built for this audit and works. `AUDIT/devmap-false-corruption.md`
  stands: do not rebuild.
- **Kernel lane K1–K7** remains blocked on tessl GitPulse trust (user-side).
- ~~The 300-vs-50 hand-label decision~~ — **settled: 300.** The threshold was not moved; the sample
  size was raised, which makes the existing 0.6 gate mean something rather than making it easier to
  pass. Recorded in `docs/teacher-plan.md` §6.

  That decision turned §6's disjointness sentence into a load-bearing constraint over **600** items
  across two invocations, and nothing enforced it — `LabelSession` checked for duplicates *within* a
  pool and recorded `purpose` without acting on it. Now `disjoint_from` refuses at construction and
  names the colliding ids; the CLI takes a repeatable `--disjoint-from PATH` reading either a pool or
  a part-finished store. `GAP-LABEL-DISJOINTNESS-UNENFORCED`, mutation-checked.

## The exact first command for the next lane

Session 0 begins with the two gates, in this order, on the Lambda box:

```bash
podman run --rm --gpus all qd-teacher:s1 \
    python /opt/qd/stack/verify_teacher.py --require-ready --json
podman run --rm --gpus all qd-train:s1 \
    python /opt/qd/stack/verify_fast_path.py --require-fast --json
```

Both exit non-zero on `not_run` as well as on failure. **Neither has ever returned a pass**, on any
host, because neither has yet met a GPU — that is the correct state, and it is the reason the
`not_run` semantics were worth 15 tests.
