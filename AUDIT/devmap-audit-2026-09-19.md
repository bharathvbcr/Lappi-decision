# DevMap audit of qwen-decision — 2026-09-19

The plan's ask was a comprehensive DevMap audit of this repo. This is what the graph was asked, what
it answered, and what survived being checked.

## The index had to be built first

`devmap_status` on this repo returned *"no readable DevMap store"*. qwen-decision was created during
this session and had never been indexed — so every DevMap query against it would have answered
"nothing", which is indistinguishable from "clean" unless you check.

```bash
devmap build /Users/bharath/Code/research/qwen-decision
# Built generation #1 · 122 files · 1887 symbols · 6573 edges · 186ms
```

This is separate from the `~/Code` store, which still returns `database disk image is malformed` to
the MCP server while the SessionStart hook reads it fine at generation 172 — the stale-writer-lock
condition in `AUDIT/devmap-false-corruption.md`, unchanged and still not a reason to rebuild.

**Coverage, from `devmap_status`:** `not_parsed` 0, `parse_failed` 0, `call_blind` 0, `import_blind`
0, `discovery_refused` 0. `is_fresh` true. So the answers below are over the whole tree, not a
subset — which is the thing that has to be established before an empty result means anything.

## Finding 1 — no dead code, and the confidence ranking was inverted

`devmap_dead_symbols` returned exactly 3 candidates, `truncated: false`, `hidden: 0`.

**All 3 are resolver artifacts. None is dead.**

| Symbol | DevMap confidence | Reality |
| --- | --- | --- |
| `Service.wait_tick` | **0.90**, `is_exempt: false`, `exemption_reason: null` | Called at `service.rs:363` — 22 lines below its own definition |
| `CalibrationEntry.validate` | 0.40, flagged possibly-called | Called at `calibration.rs:96` and `:99` |
| `_TriStateBase.to_json` | 0.40, flagged possibly-called | Called throughout `python/` |

The resolver failures are ordinary and worth naming, because both languages here trip them:

- `service.wait_tick(tick)` is a method call through an **`Arc<Service>`** receiver; the auto-deref
  was not bound.
- `calibration.rs` defines **two** methods named `validate`; name matching could not pick one.
- `to_json` is **duck-typed** across many unrelated Python classes.

**The dangerous part is the ranking.** The two low-confidence entries carried an explicit
"an unresolved call site names this symbol" exemption. The 0.90 entry carried none — and it is the
one whose deletion does real damage: `wait_tick` is the condvar wait inside `run_idle_reaper`, so
removing it turns idle eviction into a busy-spin. **The only candidate with no warning attached was
the only one that mattered.**

What actually caught it was a second, independent signal, which is what the repo rule asks for:
the workspace is clippy-clean at 0 warnings, and rustc's own `dead_code` lint would fire on a
genuinely uncalled private method. A 0.90 "dead" score that rustc disagrees with is a claim about the
resolver, not the code.

Recorded as `GAP-DEVMAP-DEAD-FALSE-POSITIVES`.

## Finding 2 — one real duplicate implementation, now fixed

`devmap_clones` first returned **25 of 55 groups, `truncated: true`, `hidden: 30`**. Reporting on the
25 would have repeated this session's earlier `gate_up_gelu` mistake — reading a truncated result as
a complete one. Re-run through the CLI at `--budget 100000`: 55 of 55, `hidden: 0`, `truncated:
false`.

Of the 55 groups (39 `Structural`, 16 `Exact`), **30 lie entirely inside
`crates/qd-mutate/src/lang/`** — the per-language `Language` trait impls.

**One production defect, fixed:** `Context::digest` (`context.rs:166`) built the hasher inline —
byte-for-byte the body of `crate::sha256` (`lib.rs:82`), which its sibling `StateBuffer::digest`
already calls at `backend.rs:96`. Now calls `crate::sha256`; the orphaned `sha2` import was removed.
This hash seeds the option permutation the second-pass agreement check depends on, so a second
implementation drifting from the first is not cosmetic.

Fixed the class, not the case — the other 8 `Sha256::new()` sites were checked and are legitimately
separate: `DecisionRequest::digest`, `CounterRng::new` and `CounterRng::next_u64` are multi-update
with domain separators (`qd-request-v1\0`, `qd-runtime.CounterRng.v1\0`) and cannot use a single-shot
helper; `qd-mutate`'s two sites are likewise domain-separated; `qd-mutate::manifest::sha256_hex` and
`qd-preflight::artifacts::sha256_file` are those crates' own canonical helpers, and neither crate
depends on `qd-runtime`.

Recorded as `GAP-RUNTIME-DIGEST-SECOND-IMPL`. **This fix ships without a fail-first test**, and that
is deliberate: it is a refactor to the canonical owner with no behaviour change, so a test failing
against the pre-fix code cannot exist. Inventing one would assert a tautology. Said plainly rather
than quietly skipped.

## Finding 3 — the `lang/` duplication is left alone, on purpose

Twelve **Exact** (byte-identical) groups sit inside `crates/qd-mutate/src/lang/`, covering 32 method
bodies, the largest at 217 AST nodes: `comparison_ops` identical across Go/Rust/TypeScript,
`call_arguments` across Go/Rust, `comments` across Go/Python/TypeScript, and so on.

By volume this is the repo's largest duplication. It is **not** being unified, and the reason is the
repo's own "don't over-unify" rule: these are per-grammar tree-sitter node-kind queries. That
`Go.comparison_ops` and `Rust.comparison_ops` are byte-identical is a **coincidence of grammar
naming** — both grammars happen to call the node `binary_expression` — not a shared concept. Merging
them couples two grammars that are free to diverge at any upstream version bump, and the merged
helper acquires exactly the boolean-that-means-do-the-other-thing the rule warns about.

If it is ever unified, the seam is the *traversal*, with each language supplying its grammar's node
kinds as data — not a shared function with a language enum threaded through it.

Two smaller groups are worth a cheap fix later and are **not** done here, being test-only:
`Scripted.pooled_features` / `Wrapped.pooled_features` (45 nodes) and the matching `snapshot` pair
(40 nodes) are duplicate test doubles across `tests/answering_procedure.rs` and
`tests/common/mod.rs`, where one already lives in the shared module.

The four identical `fmt` bodies (`LangId`, `MutationClass`, `OpId`, `HashKind`) are idiomatic
`Display` impls and are correctly separate.

## What DevMap could not answer

Honoured rather than papered over, per rule 1:

- Every `dead_symbols` and `explore` answer carried **`walk_incomplete`**: *7444 of 14943 unresolved
  attribution sites have no indexed target*. That is why "0 callers" was treated as a hypothesis to
  disprove, and disproving it is what Finding 1 is.
- `explore` on `wait_tick` stopped at **depth 3** — the blast radius of 32 is a lower bound.
- `devmap_clones` covers **1488 signed symbols against 403 unsigned** (1891 total), so roughly 21% of
  symbols are outside clone detection entirely. The duplication figures are a floor, not a census.
