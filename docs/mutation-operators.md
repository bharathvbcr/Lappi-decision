# `qd-mutate` operator specification

Labels are exact **because the generator wrote them**. That is the only reason this data is worth
more than the teacher's. Every design choice below protects that property.

Target languages: Rust, Go, Swift, Python, TypeScript — the five slices pulled from the pool.

## Architecture

Operators are written **once** against a small language-agnostic facade, not five times per language.
A `Language` implementation supplies the node kinds; an operator asks the facade for what it needs.

```rust
trait Language {
    fn grammar(&self) -> tree_sitter::Language;

    /// Functions/methods with a real body. Excludes declarations without one
    /// (trait signatures, interface members, abstract methods) — there is
    /// nothing to stub, and emitting an empty span would be a poisoned label.
    fn function_bodies<'t>(&self, tree: &'t Tree, src: &[u8]) -> Vec<FunctionBody<'t>>;

    fn conditions<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn comparison_ops<'t>(&self, body: &FunctionBody<'t>) -> Vec<ComparisonSite<'t>>;
    fn call_arguments<'t>(&self, body: &FunctionBody<'t>) -> Vec<CallSite<'t>>;
    fn else_branches<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn error_propagations<'t>(&self, body: &FunctionBody<'t>) -> Vec<ErrorSite<'t>>;
    fn awaits_and_locks<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn integer_literals<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn local_bindings<'t>(&self, body: &FunctionBody<'t>) -> Vec<Binding<'t>>;
    fn comments<'t>(&self, body: &FunctionBody<'t>) -> Vec<Node<'t>>;
    fn import_block<'t>(&self, tree: &'t Tree) -> Option<ImportBlock<'t>>;

    /// Source text for `stub` bodies, and the zero value for the return type.
    fn panic_stub(&self) -> &'static [&'static str];
    fn default_return(&self, return_type: Option<&str>) -> Option<String>;

    /// Formatter for verifying `cosmetic` really is behaviour-preserving.
    /// `None` means no formatter is available, which RESTRICTS the cosmetic
    /// operator set rather than being assumed harmless.
    fn formatter(&self) -> Option<Formatter>;
}
```

A language whose facade returns an empty list for a construct simply yields no mutations of that kind.
That is correct and must be **recorded per language**, so a coverage report shows which operators
actually fired where — rather than a silent zero reading as "no opportunities".

## Operators

`silent` marks the stub operators dcverify's substring gate cannot see. These are weighted up: they
are the class that earns the model.

### stub

| Id | Operation | Languages | Notes |
| --- | --- | --- | --- |
| `stub.panic` | Body -> `todo!()` / `unimplemented!()` / `panic()` / `pass` / `raise NotImplementedError` / `throw new Error(...)` | all | The *visible* stub. A substring gate already catches this; included as the easy contrast class |
| `stub.default_return` **silent** | Body -> return the type's zero: `Ok(())`, `nil`, `[]`, `0`, `""`, `None`, `undefined` | all | Requires the return type. Where it cannot be resolved, the operator declines rather than guessing |
| `stub.hardcoded` **silent** | Body -> return a literal matching what the caller expects | all | Derived from a literal already present in the function or its tests |
| `stub.early_return` **silent** | Insert `return <default>` before the real work | all | Leaves the original body in place below — the shape that looks most like working code |

`stub.default_return` and `stub.early_return` are the operators the whole exercise is for. An agent's
`return Ok(())` is indistinguishable from real code by substring, and is the failure DevCouncil needs
caught.

### logic

One mutation per diff. Classic mutation-testing operators.

| Id | Operation | Notes |
| --- | --- | --- |
| `logic.negate_condition` | `if c` -> `if !c` | Skips conditions already negated at the top level, to avoid double-negation churn |
| `logic.off_by_one` | `n` -> `n+1` / `n-1` on a loop or slice bound | Bound position only, not any integer |
| `logic.swap_args` | Swap two call arguments **of the same type** | Same type or the swap does not compile. Where types are unavailable, restricted to arguments that are both simple identifiers of the same declared type |
| `logic.drop_else` | Remove an `else` branch | |
| `logic.widen_comparison` | `<` -> `<=`, `>` -> `>=` (and the reverse) | |
| `logic.swallow_error` | `?` -> `.unwrap_or_default()`; remove `if err != nil { return err }`; `try/catch` -> swallowed catch | The Go and Rust forms are the realistic agent failure |
| `logic.drop_await_or_lock` | Remove an `await` / a lock acquisition | Only where removal still parses |
| `logic.change_constant` | Perturb a numeric or string constant | Excludes 0/1 where the perturbation would be a no-op |

### cosmetic — behaviour-preserving **by verification, not by assertion**

| Id | Operation | Verification |
| --- | --- | --- |
| `cosmetic.rename_local` | Rename a local binding and all its uses **within the body** | Refused if the name is captured, shadowed, or referenced outside the body |
| `cosmetic.reformat` | Reformat via the language formatter | Only where `formatter()` is `Some`. Verified by formatting both sides to a canonical form and comparing |
| `cosmetic.reorder_imports` | Reorder within one import group | Refused where import order is semantic (Go side effects, Python circular) |
| `cosmetic.edit_comment` | Rewrite a comment or docstring | Always safe; the one operator available in every language |
| `cosmetic.wrap_line` | Wrap a long line | Refused inside a string literal or where the language is whitespace-sensitive (Python) |

**Where no formatter is available the operator set shrinks to `edit_comment` and, where provably safe,
`rename_local`.** The restriction is recorded in the manifest for that language. A `cosmetic` label
whose diff changed behaviour is a poisoned label, and poisoned `cosmetic` labels are worse than
missing ones: they teach the model that a behaviour change is cosmetic.

### clean

The original agent diff, unmodified. **Noisy by construction** — some real agent diffs *are* stubs.
The teacher labels reduce that noise; the model treats `clean` as "nothing found", not "verified good".

## Span labels — the exactness guarantee

Every mutation records the mutated line range, which is the `span` label.

**Two independent derivations must agree or the example is dropped:**

1. The mutator's own bookkeeping — the byte range it edited, converted to lines.
2. A textual diff of before/after, with the changed line range read back off the hunk.

They are computed by different code paths on purpose. Agreement is checked on **every** example, not
sampled: this is cheap, and it is the only thing standing between the pointer head and a systematic
one-line offset that no class metric would reveal.

Line numbers are **1-based, inclusive on both ends**, counted over the post-mutation file, in the
same convention the runtime's `span` slot returns. A test asserts the two conventions match by
round-tripping a known span through both.

## Placement rule

The mutation lands **inside the hunk the agent touched**, so the model learns to read hunks and not
file headers. Asserted per example: the mutated span intersects the original diff hunk. A mutation
site outside every hunk is skipped, not relocated.

## Split safety

**The same function never appears mutated and clean in different splits.** Identity is
`(repo, path, symbol_name, arity)` — *not* the diff hash, because the same function reformatted has a
different diff hash and would leak straight through a hash-based check.

## Determinism

Seeded with `rand_chacha` from an explicit seed recorded in the manifest. The same seed and the same
input pool produce byte-identical output. A test runs the generator twice and compares hashes — a
generator that is not reproducible cannot have its data snapshot hashed, and the protocol hash in the
ledger would be a fiction.

## Refusals

The generator **refuses**, and records the refusal with a reason, when:

- tree-sitter returns a tree containing `ERROR` nodes — mutating an unparsed file produces nonsense
  with a confident label.
- The file has a BOM or mixed line endings that would desynchronize byte and line offsets — normalize
  first, then mutate, and record that normalization happened.
- The function body is empty or absent.
- The operator's preconditions are unmet (no resolvable return type, a capture-shadowed rename).

Refusal counts are part of the manifest. A language with a high refusal rate is a language where the
mutation mixture is thinner than the headline count suggests, and the mixture weights must know that.
