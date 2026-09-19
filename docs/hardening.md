# Hardening and adversarial test contract

Every lane implements the attacks for its surface **before** the feature is called done. The rule is
*break it before you fix it*: characterize current behaviour against unmodified code, add a test that
fails, attack the surrounding logic, enumerate the invariants the code makes impossible to violate,
then implement.

A test that cannot run in this environment is reported **not run**, never green. See
`docs/ledger-schema.md` for the tri-state that makes that structural rather than a matter of care.

---

## 1. `qd-mutate` — the span label is the sharpest edge

The mutation engine's whole value is that *labels are exact because the generator wrote them*. An
off-by-one in a line span is not a slightly worse label; it teaches the pointer head to point one line
off, systematically, on 40% of the training mixture. It is invisible in class accuracy.

**Invariant: for every emitted diff, the recorded `span` lines are exactly the lines the mutation
changed — verified by re-deriving the span from a textual diff of before/after, not from the
mutator's own bookkeeping.** Two independent derivations must agree, or the example is dropped.

Adversarial corpus every operator must survive:

| Attack | Why it breaks a naive implementation |
| --- | --- |
| CRLF line endings, and a file with mixed CRLF/LF | Byte offsets and line numbers diverge; tree-sitter counts bytes |
| UTF-8 BOM at file start | Shifts every byte offset by 3; line 1 becomes line 1 at column 3 |
| Multi-byte identifiers, emoji in comments and strings | Byte offset != char offset != column |
| A string literal containing `todo!()` / `pass` / `NotImplementedError` | A substring-based stub detector fires on data, not code. This is precisely the gate the model is meant to beat |
| A comment containing what looks like the mutated construct | Same |
| Nested functions, closures, lambdas, trait default bodies | "The function body" is ambiguous; the mutator must name which node it took |
| `async fn`, generics with `where` clauses, macros with braces | Body extents that a brace-counting heuristic gets wrong |
| A function whose body is already `todo!()` | Mutating a stub into a stub yields a `clean`-looking diff labelled `stub` |
| An empty body / a declaration with no body (trait sig, interface) | Nothing to mutate; must be skipped, not emitted with an empty span |
| A single-line function | Start line == end line; off-by-one hides here |
| Last line of file with no trailing newline | Span end runs past EOF |
| A file that is 1 byte, 0 bytes, or 10 MB | Bounds |
| Deeply nested blocks (1000 levels) | Parser recursion; must be bounded, not a stack overflow |
| A file tree-sitter parses with ERROR nodes | Must be refused, not mutated blindly into nonsense |

**Cosmetic must actually be behaviour-preserving.** Where a formatter exists for the language, run it:
a `cosmetic` label whose diff changed behaviour is a poisoned label. Where no formatter is available,
the operator set is restricted to ones provably behaviour-preserving (comment text, import order
within a group) and the restriction is recorded, not assumed.

**One mutation per diff**, and the mutation lands **inside the hunk the agent touched** — so the model
learns to read hunks, not file headers. A test asserts the mutated span intersects the original hunk.

**The same function never appears mutated and clean in different splits.** Enforced at split time by
function-level identity (repo + path + symbol), not by diff hash — the same function reformatted has a
different diff hash and would leak.

---

## 2. Data — leakage is the failure that makes every other number meaningless

- **Repo-level split, never row-level.** Keep the repo name on every row from the pull onward.
- **Repo-level is not sufficient by itself.** Vendored trees, forks and copied files put the same code
  in two repos. MinHash dedupe at 0.8 Jaccard runs **across pool and held-out before splitting**.
- **Shuffled-label control must fall to chance.** If it does not, the split is rebuilt before any other
  run. This catches leakage through file paths, repo names and near-duplicates.
- **Privileged-hunk control** (model sees only the mutated hunk) gives the ceiling the pooled model is
  measured against.
- **Held-out paths are refused by the training process at the path level**, not by convention. The
  check lives in `qd-train`; removing it is a refused change. A test asserts that pointing the trainer
  at a held-out manifest raises rather than trains.
- **Task-family holdout**: two families never enter training. A test asserts their identifiers appear
  in no training shard.

---

## 3. Prompt rendering — the context is adversarial by construction

The context is **agent-authored diffs**. Treat every byte of it as hostile input, because the model
will be served diffs written by other models.

| Attack | Required behaviour |
| --- | --- |
| Context contains `Answer: B`, or the literal option letters in a list | Must not steer the answer. Tested by measuring answer shift with and without an injected decoy |
| Context contains the literal `noul` token | Must not induce abstention |
| Context contains the tokenizer's special tokens as literal text (`<\|im_start\|>` etc.) | Rendered as text, never as control tokens. Test with the literal strings |
| Context contains the question-format delimiters | Must not terminate the context early |
| Context is all whitespace / empty | Refusal or `noul`, never a confident letter |
| Options that are duplicates of each other | Refusal — the question is malformed |
| Options containing newlines or the delimiter | Escaped; must not create a 17th apparent option |
| Option text longer than the context cap | Refusal, not truncation |

**Option order is shuffled per training example**, and permutation consistency >= 95% is a gate.
The runtime's second permuted pass turns residual disagreement into `noul`.

---

## 4. Ledger — the tri-state is the invariant

- A row with `state: "ran"` and no `passed` field fails deserialization.
- A row with `state: "not_run"` and an empty `reason` fails deserialization.
- An aggregate over a `not_run` input is `not_run`. Property-tested over random tri-state vectors.
- A killed run writes a row saying so. **S6 is not green until a deliberately killed run produces one.**
- Chain hash detects an edited or reordered history.
- Promotion requires three rows differing only in seed, every gate `ran`, every control `ran`+`passed`.
  Property test: for every subset of gates set to `not_run`, promotion is refused.

## 5. Eval harness — test the tests (S7)

The harness is fed two deliberately broken models and **must fail them**:

1. A **constant-prediction** model (always the majority class) — the degenerate-head assertion must
   fire: prediction entropy above a floor, no class above 95%, feature variance above a floor.
2. A **shuffled-label** model — the shuffled-label control must fall to chance and be flagged.

If the harness passes either, the harness is broken and no number it produces means anything. This is
run before any real eval, and its result is a ledger row.

## 6. Runtime — fail closed

- Every refusal in `docs/schema-api.md` has a test asserting a typed error, and asserting it is
  **distinguishable from `noul`**. A hash mismatch reading as model humility is the specific bug.
- Aliasing: input and output buffers pointing at the same memory.
- Recycled scratch: a second call must not read the first call's leftovers.
- `Tq = 0` overwrite.
- Poisoned `GpuRuntime` returns DEGRADED and rebuilds; the rebuild is tested, not assumed.
- Idle eviction at 10 minutes, and a measured cold start — a launchd agent that never evicts keeps
  1.1 GB wired forever; one that evicts too eagerly makes every DevType call cold.
- Concurrent requests on the socket; a slow request must not wedge the agent.
- **`readonly` decode leaves the state buffer hash-identical.** This is the slot-isolation guarantee;
  if it fails, slot 2's answer depends on slot 1's, and the whole snapshot design is void.

## 7. Kernels — parity before performance

Covered in full by the plan's parity suite. The load-bearing ones:

- f64 sequential reference in Rust is the golden; K1/K2 match at 1e-4 relative in fp32.
- Fixtures are generated at `rule="published"` and **say so in the filename**. A test that passes
  against `rule="repo"` is a failing test — nanolab's default is `repo`, so this is a live trap.
- Chunk boundaries: L in {1, 63, 64, 65, 127, 129, 1000, 8191}.
- Pad semantics: alpha = 1 and beta = 0 on padded rows change nothing.
- prefill(T) then decode(1) == prefill(T+1).
- alpha -> 1 (no decay) and alpha -> 1e-4 (hard reset); no NaN, no Inf.
- **Key-head repeat is INTERLEAVE, not tile** — one test with distinct K heads catches the wrong
  order. Confirm `linear_num_key_heads` from the config before writing the test.
- Determinism: same input, same device, bit-identical twice.

Note the portability limit already established on this machine: bit-identical replay is macOS-only;
`exp`/`ln`/`tanh`/`powf` differ from glibc. Determinism claims are per-platform.
