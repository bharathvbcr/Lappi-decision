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
    fn import_block<'t>(&self, tree: &'t Tree, src: &'t str) -> Option<ImportBlock<'t>>;

    /// Source text for `stub` bodies, and the zero value for the return type.
    fn panic_stub(&self) -> &'static [&'static str];
    fn default_return(&self, return_type: Option<&str>) -> Option<String>;

    /// Formatter for verifying `cosmetic` really is behaviour-preserving.
    /// `None` means no formatter is available, which RESTRICTS the cosmetic
    /// operator set rather than being assumed harmless.
    fn formatter(&self) -> Option<Formatter>;
    /// A minimal file the formatter must accept. Resolution RUNS this: a path
    /// on disk is not evidence that the program formats anything.
    fn smoke_source(&self) -> &'static str;
    /// `Some(reason)` where import order is execution order for the whole
    /// language, which refuses `cosmetic.reorder_imports` here on every
    /// machine. Asked before any file is read; `import_block` answers the
    /// separate per-file question.
    fn import_order_is_semantic(&self) -> Option<&'static str>;
}
```

This sketch is the shape, not the signature list; `crates/qd-mutate/src/lang/mod.rs` is
authoritative.

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
| `cosmetic.reformat` | Apply one **layout-only** hunk of the formatter's whole-file pass | Only where `formatter()` is `Some`. Verified by formatting both sides to a canonical form and comparing. See *What `reformat` actually emits* below — it is narrower than "reformat" |
| `cosmetic.reorder_imports` | Reorder within one import group | Refused where import order is semantic. **Needs no formatter** — see below |
| `cosmetic.edit_comment` | Rewrite a comment or docstring | Always safe; the one operator available in every language |
| `cosmetic.wrap_line` | Wrap a long line | Refused inside a string literal or where the language is whitespace-sensitive (Python) |

**Where no formatter is available the operator set shrinks to `edit_comment`, `reorder_imports` where
the language's import order is not semantic, and `rename_local` where provably safe.** The
restriction is recorded in the manifest for that language. A `cosmetic` label whose diff changed
behaviour is a poisoned label, and poisoned `cosmetic` labels are worse than missing ones: they teach
the model that a behaviour change is cosmetic.

An earlier wording of that sentence named only `edit_comment` and `rename_local`, which contradicted
this document's own operator table (whose `reorder_imports` row states no formatter condition) and
`docs/hardening.md` §1 (which words the safe set as "comment text, import order within a group").
The code implemented the narrow wording and the disagreement was carried as
`GAP-MUTATE-DOC-CONFLICT-REORDER-IMPORTS`. It was settled by running it rather than by picking a
document, and the measurement is recorded in `HANDOFF/mutate-ops-2026-09-19.md`.

### Why `reorder_imports` takes no formatter

The formatter is the wrong axis for this operator, in **both** directions:

- It cannot establish the claim. `cosmetic.reformat` and `cosmetic.wrap_line` are verified by
  formatting both sides and comparing canonical forms. A reorder cannot be: it *changes* the
  canonical form by design, so the check would fail on every candidate. `OpId::verified_by_canonical_form`
  excludes the operator for exactly this reason, which means the formatter is never consulted about
  a reorder's safety even when one is installed.
- It cannot refute it either. Gating on a resolvable formatter deleted provably safe Rust and Swift
  candidates on a machine merely missing a tool, while on a machine that *had* the tool it waved
  through a TypeScript swap that nothing had checked.

The real axis is the language, and it is asked in two places for two different questions:

| Question | Asked by | When |
| --- | --- | --- |
| Is import order execution order **for this language**? | `Language::import_order_is_semantic` | before any file is read, so `Manifest::new` records the restriction as a property of the run |
| Does **this file's** group contain an order-sensitive item? | `Language::import_block().reorderable` | per file — a Rust glob, a Go blank import, a Swift `#if` |

Per language, as the code states it:

| Language | Order semantic for the language? | Why |
| --- | --- | --- |
| Rust | no | `use` binds a name and runs nothing; a glob is refused per file |
| Swift | no | a module is initialised on first use, not at the import statement; `#if` / `@_exported` refused per file |
| Go | no | the spec leaves the initialization order of *independent* imports unspecified, so no correct program may depend on it; a blank import is refused per file |
| Python | **yes** | imports execute in order; a circular-import workaround depends on it |
| TypeScript | **yes** | an ES module is evaluated when it is imported, in source order |

TypeScript's entry is a correction, not a restatement. The previous check refused only the clause-less
`import "./polyfill";` shape and declared everything else reorderable — but `import { a } from './a'`
evaluates `./a`'s module body just as surely, and a check confined to one file cannot show that two
module bodies do not interact. On this host the missing `prettier` hid that: the operator was refused
for the wrong reason and the unsound case never fired. On a host with `prettier` it would have.

### What `reformat` actually emits

`cosmetic.reformat` is **narrower than its name**, and the gap between the two is stated here rather
than left for a reader to infer from a label.

A whole-file formatter pass can move a token across a line boundary — `swift-format` pulling a `{` up
onto the signature line, `rustfmt` wrapping a long argument list *and* adding a trailing comma. The
line alignment splits that single move into two hunks, one removing the token and one adding it, and
applying either alone leaves a file that does not parse. So the operator emits only hunks whose
**non-whitespace content is unchanged**, which makes every candidate self-contained by construction.

Two consequences, both load-bearing:

1. A fixture whose only deviation from canonical form is a wrapped long line yields **no** reformat
   candidate. The operator is layout-only in the strict sense; it is not "whatever the formatter
   would do".
2. Hunks declined by that filter are **counted**, per language and per operator, in the manifest's
   `operators[].declined` table under the key `not_layout_only`. A body where the formatter had
   twelve hunks and all twelve were declined must not be recorded the same way as a body that was
   already canonical: the first means the operator ran and produced nothing, the second means there
   was nothing to do. `sites_found: 0` alone cannot tell them apart, and this document's own rule —
   *"rather than a silent zero reading as 'no opportunities'"* — is the one that would be broken.

Soundness is not resting on that filter. The proof remains the emit-time canonical-form comparison in
`generate::build_example`: both sides are formatted and compared, and a candidate whose canonical
forms differ is dropped as `CosmeticNotPreserving`. The filter exists so unusable candidates are not
generated in the first place, and so the narrowing is visible.

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
