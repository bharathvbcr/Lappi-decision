//! The generator: one pool record in, labelled examples out.
//!
//! This is the only module that chooses anything, and the only one that can drop an example. Both
//! facts are deliberate.
//!
//! # The span guarantee
//!
//! Every emitted example's span is computed **twice, by two code paths that share nothing but the
//! convention**:
//!
//! * derivation A — [`crate::span::span_from_byte_range`] over the byte range
//!   [`crate::edit::EditSet::apply`] reports it wrote;
//! * derivation B — [`crate::diffspan::span_from_text_diff`], which is handed the before and after
//!   text and has never heard of an edit.
//!
//! They must produce the same `LineSpan` or the example is dropped and counted. This is checked on
//! **every** example, not sampled. An off-by-one here is invisible in class accuracy — the model
//! would score exactly as well on the label it gets wrong — and it would systematically teach the
//! pointer head to point one line off across the mutated 40% of the mixture.
//!
//! # Determinism
//!
//! Each record gets its own `ChaCha20Rng`, seeded from `sha256(seed || record.id)`. No choice
//! depends on how many records came before, so the output is a pure function of (seed, pool) and
//! two runs are byte-identical. `tests/determinism.rs` runs the generator twice and compares
//! digests, because a generator that is not reproducible cannot have its snapshot hashed and the
//! protocol hash in the ledger would be a fiction.
//!
//! # Split safety
//!
//! Whether a record is mutated or left clean is a function of `(seed, repo, path)` — **not** of the
//! record, and not of a diff hash. A function identity is `(repo, path, symbol, arity)`, so two
//! records touching one file always land on the same side and the same function can never appear
//! both mutated and clean. The bookkeeping check in [`Generator::run`] exists to prove that rather
//! than to assume it.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diffspan;
use crate::fmt::{self, Availability, Resolved};
use crate::lang::{LangId, Language};
use crate::manifest::{sha256_hex, Manifest, PoolReport};
use crate::normalize::{self, Normalization};
use crate::ops::{self, Candidate, MutationClass, OpCtx, OpId};
use crate::parse::{parse_bounded, Refusal};
use crate::pool::{FunctionIdentity, Hunk, PoolRecord};
use crate::span::{self, LineSpan};

/// Function bodies one file may contribute. A generated or vendored file with tens of thousands of
/// bodies would otherwise dominate the mixture on its own.
pub const MAX_BODIES_PER_FILE: usize = 256;
/// Examples one file may contribute, for the same reason.
pub const MAX_EXAMPLES_PER_FILE: usize = 32;
/// Lines of unified-diff context carried on each example.
const DIFF_CONTEXT: usize = 3;

/// How the run was configured. Everything here lands in the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub seed: u64,
    /// Records in a thousand that are emitted `clean` instead of mutated.
    pub clean_permille: u32,
    /// Restrict to these languages. Empty means all five.
    pub languages: Vec<LangId>,
    /// Stop after this many examples. `None` is the whole pool.
    pub limit: Option<usize>,
    /// Examples one file may contribute.
    pub max_examples_per_file: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            seed: 0,
            // Roughly a fifth clean. The model treats `clean` as "nothing found", and the class is
            // noisy by construction — some real agent diffs *are* stubs — so it is a contrast set,
            // not a majority.
            clean_permille: 200,
            languages: Vec::new(),
            limit: None,
            max_examples_per_file: MAX_EXAMPLES_PER_FILE,
        }
    }
}

/// One labelled example. This is the JSONL row the training pipeline reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    pub pool_id: String,
    pub repo: String,
    pub path: String,
    pub language: LangId,
    pub class: MutationClass,
    /// `None` for `clean`: no operator ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    /// True when the operator is a stub a substring gate cannot see.
    pub silent: bool,
    /// **The span label.** 1-based, inclusive both ends, over `after`. `None` for `clean`, which
    /// points at nothing because nothing was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<LineSpan>,
    /// The same region in `before` coordinates, which is what the hunk-intersection rule is checked
    /// against. Carried so a consumer can re-check the placement rule without re-deriving it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_span: Option<LineSpan>,
    pub function: FunctionIdentity,
    /// The grammar's own name for the declaration node the mutator took. "The function body" is
    /// ambiguous for nested functions and closures, so the mutator names which node it took.
    pub node_kind: String,
    pub is_nested: bool,
    pub before: String,
    pub after: String,
    pub diff: String,
    /// What normalization the input needed. A consumer knows from this that `before` is not
    /// byte-identical to the file on disk.
    pub normalization: Normalization,
    /// The hunks the agent touched, when the pool carried them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hunks: Option<Vec<Hunk>>,
    /// False when the pool record carried no hunks, so the placement rule could not be applied.
    /// Never defaulted to true.
    pub hunk_constrained: bool,
    pub detail: String,
    pub seed: u64,
    pub tool_version: String,
}

/// The result of a run: the examples, and the manifest that describes how they came about.
pub struct Run {
    pub examples: Vec<Example>,
    pub manifest: Manifest,
}

impl Run {
    /// The examples as JSONL, and its digest — the two things the ledger needs.
    pub fn to_jsonl(&self) -> Result<(String, String), serde_json::Error> {
        let mut out = String::new();
        for example in &self.examples {
            out.push_str(&serde_json::to_string(example)?);
            out.push('\n');
        }
        let digest = sha256_hex(out.as_bytes());
        Ok((out, digest))
    }
}

/// Whether a record is mutated or left clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Mutated,
    Clean,
}

pub struct Generator {
    options: Options,
    formatters: Vec<(LangId, Availability)>,
    resolved: HashMap<LangId, Resolved>,
}

impl Generator {
    /// Probe every formatter once, up front.
    ///
    /// Once, because a formatter that appeared half-way through a run would make a language's
    /// cosmetic coverage depend on when each file was processed — and that is not a property a
    /// reproducible generator may have.
    pub fn new(options: Options) -> Self {
        // One pass, not two. Resolution now runs the formatter on a smoke input, so calling
        // `probe_all()` alongside this loop would have paid for every formatter twice and — worse —
        // left the recorded availability and the formatter actually used as two separate
        // measurements that could disagree.
        let mut resolved = HashMap::new();
        let mut formatters = Vec::with_capacity(LangId::ALL.len());
        for id in LangId::ALL {
            match fmt::resolve(crate::lang::for_id(id)) {
                Ok(r) => {
                    formatters.push((id, r.availability.clone()));
                    resolved.insert(id, r);
                }
                Err(a) => formatters.push((id, a)),
            }
        }
        Generator {
            formatters,
            options,
            resolved,
        }
    }

    /// The measured formatter availability this generator will use.
    pub fn formatters(&self) -> &[(LangId, Availability)] {
        &self.formatters
    }

    fn wants(&self, id: LangId) -> bool {
        self.options.languages.is_empty() || self.options.languages.contains(&id)
    }

    pub fn run(&self, records: &[PoolRecord], pool: PoolReport) -> Run {
        let mut manifest = Manifest::new(self.options.seed, pool, &self.formatters);
        let mut examples: Vec<Example> = Vec::new();
        // The split-safety proof. Keyed on the function identity, which does not move when the
        // layout does.
        let mut disposition_of: HashMap<FunctionIdentity, Disposition> = HashMap::new();

        for record in records {
            if let Some(limit) = self.options.limit
                && examples.len() >= limit
            {
                break;
            }
            let Some(language_id) = record.language() else {
                continue;
            };
            if !self.wants(language_id) {
                continue;
            }
            manifest.note_file(language_id);

            let (source, normalization) = normalize::normalize(&record.source);
            let language = crate::lang::for_id(language_id);
            let parsed = match parse_bounded(&language.grammar(), &source) {
                Ok(p) => p,
                Err(refusal) => {
                    manifest.note_file_refused(language_id, &refusal);
                    continue;
                }
            };

            let bodies = language.function_bodies(&parsed.tree, &source);
            if bodies.is_empty() {
                manifest.note_file_refused(language_id, &Refusal::NoFunctionBody);
                continue;
            }
            let bodies: Vec<_> = bodies.into_iter().take(MAX_BODIES_PER_FILE).collect();
            for body in &bodies {
                manifest.note_body(language_id, body.is_nested);
            }

            let disposition = self.disposition_of_record(record);
            let mut conflicted = false;
            for body in &bodies {
                let identity = record.identity(&body.name, body.arity);
                match disposition_of.get(&identity) {
                    Some(previous) if *previous != disposition => {
                        manifest.note_refusal(
                            language_id,
                            &Refusal::SplitConflict {
                                identity: identity.to_string(),
                            },
                        );
                        conflicted = true;
                    }
                    _ => {
                        disposition_of.insert(identity, disposition);
                    }
                }
            }
            if conflicted {
                continue;
            }

            let mut rng = record_rng(self.options.seed, &record.id);

            if disposition == Disposition::Clean {
                // `clean` is the original agent diff, unmodified — one example for the record, not
                // one per body. It is noisy by construction: some real agent diffs *are* stubs, and
                // the model must treat `clean` as "nothing found", never as "verified good".
                let first = &bodies[0];
                examples.push(Example {
                    id: format!("{}#clean", record.id),
                    pool_id: record.id.clone(),
                    repo: record.repo.clone(),
                    path: record.path.clone(),
                    language: language_id,
                    class: MutationClass::Clean,
                    operator: None,
                    silent: false,
                    span: None,
                    source_span: None,
                    function: record.identity(&first.name, first.arity),
                    node_kind: first.node_kind.to_string(),
                    is_nested: first.is_nested,
                    before: source.clone(),
                    after: source.clone(),
                    diff: String::new(),
                    normalization,
                    hunks: record.hunks.clone(),
                    hunk_constrained: record.hunks.is_some(),
                    detail: "unmodified agent diff".to_string(),
                    seed: self.options.seed,
                    tool_version: crate::TOOL_VERSION.to_string(),
                });
                manifest.note_clean(language_id);
                manifest.totals.examples += 1;
                manifest.totals.clean += 1;
                if record.hunks.is_none() {
                    manifest.totals.examples_without_hunk_constraint += 1;
                }
                continue;
            }

            let ctx = OpCtx::new(
                language,
                &parsed.tree,
                &source,
                self.resolved.get(&language_id),
            );
            let mut emitted_here = 0usize;

            for body in &bodies {
                if emitted_here >= self.options.max_examples_per_file {
                    break;
                }
                if let Some(limit) = self.options.limit
                    && examples.len() >= limit
                {
                    break;
                }

                // Enumerate every operator's candidates first, so the weighted choice is over what
                // is actually available for *this* body rather than over the operator list.
                let mut available: Vec<(OpId, Vec<Candidate>)> = Vec::new();
                for op in OpId::ALL {
                    match ops::candidates(op, &ctx, body) {
                        Ok(offered) => {
                            // Declines first, and unconditionally: an operator that offered
                            // nothing *because its filter rejected everything* is the case this
                            // records, and skipping the bookkeeping when the candidate list came
                            // back empty would drop exactly that case on the floor.
                            for (reason, count) in &offered.declined {
                                manifest.note_declined(language_id, op, reason, *count);
                            }
                            // An empty list with no declines is "this construct is not present
                            // here", which is correct and is recorded as `sites_found: 0` rather
                            // than as a refusal.
                            if !offered.candidates.is_empty() {
                                manifest.note_sites(
                                    language_id,
                                    op,
                                    offered.candidates.len() as u64,
                                );
                                available.push((op, offered.candidates));
                            }
                        }
                        Err(refusal) => manifest.note_refusal(language_id, &refusal),
                    }
                }
                if available.is_empty() {
                    continue;
                }

                let Some(index) = weighted_index(&available, &mut rng) else {
                    continue;
                };
                let (op, candidates) = &available[index];
                let pick = if candidates.len() == 1 {
                    0
                } else {
                    rng.random_range(0..candidates.len())
                };
                let candidate = &candidates[pick];

                match self.build_example(
                    record,
                    language_id,
                    language,
                    &ctx,
                    body,
                    *op,
                    candidate,
                    &source,
                    normalization,
                    emitted_here,
                ) {
                    Ok(example) => {
                        manifest.note_emitted(language_id, *op);
                        manifest.totals.examples += 1;
                        manifest.totals.mutated += 1;
                        if op.is_silent() {
                            manifest.totals.silent_stubs += 1;
                        }
                        if !example.hunk_constrained {
                            manifest.totals.examples_without_hunk_constraint += 1;
                        }
                        examples.push(example);
                        emitted_here += 1;
                    }
                    Err(refusal) => manifest.note_refusal(language_id, &refusal),
                }
            }
        }

        manifest.pool.records = records.len() as u64;
        let mut run = Run { examples, manifest };
        // The digest is over the JSONL a caller would write, so the manifest names the same bytes
        // the ledger will hash. A `Run` that is never written still carries it, rather than an
        // empty string that would read as "nothing was produced".
        if let Ok((_, digest)) = run.to_jsonl() {
            run.manifest.examples_sha256 = digest;
        }
        run
    }

    /// Mutated or clean, from `(seed, repo, path)`.
    ///
    /// Keyed on the **file**, not the record and not a diff hash. Two pool records touching the
    /// same file therefore land on the same side, which is what makes "the same function never
    /// appears mutated and clean" structural rather than bookkept — and a diff hash would not,
    /// because the same function reformatted hashes differently and would leak straight through.
    fn disposition_of_record(&self, record: &PoolRecord) -> Disposition {
        let mut hasher = Sha256::new();
        hasher.update(self.options.seed.to_le_bytes());
        hasher.update(b"disposition\0");
        hasher.update(record.repo.as_bytes());
        hasher.update([0u8]);
        hasher.update(record.path.as_bytes());
        let digest = hasher.finalize();
        let mut bucket = [0u8; 4];
        bucket.copy_from_slice(&digest[..4]);
        if u32::from_le_bytes(bucket) % 1000 < self.options.clean_permille {
            Disposition::Clean
        } else {
            Disposition::Mutated
        }
    }

    /// Apply one candidate and check everything that must hold before it may be emitted.
    #[allow(clippy::too_many_arguments)]
    fn build_example(
        &self,
        record: &PoolRecord,
        language_id: LangId,
        language: &dyn Language,
        ctx: &OpCtx<'_>,
        body: &crate::lang::FunctionBody<'_>,
        op: OpId,
        candidate: &Candidate,
        before: &str,
        normalization: Normalization,
        index: usize,
    ) -> Result<Example, Refusal> {
        let applied = candidate
            .edits
            .apply(before)
            .map_err(|e| Refusal::EditFailed {
                detail: e.to_string(),
            })?;
        let after = applied.text;

        // --- The placement rule, checked in the coordinates it is stated in -------------------
        // The hunks are lines of the *pre*-mutation file, so the site is converted to the same
        // coordinates. Comparing a post-mutation span against pre-mutation hunks would be off by
        // however many lines the mutation added, which is exactly the class of error this crate
        // exists to keep out.
        let (source_start, source_end) = candidate.edits.source_range();
        let source_span = span::span_from_byte_range(before, source_start, source_end);
        let hunk_constrained = record.hunks.is_some();
        if let Some(hunks) = &record.hunks
            && !hunks.iter().any(|h| source_span.intersects(h))
        {
            // Skipped, never relocated: a mutation moved into a hunk is a mutation nobody chose.
            return Err(Refusal::OutsideHunk {
                operator: op.as_str().to_string(),
            });
        }

        // --- The span guarantee: two derivations, one answer ---------------------------------
        let from_bytes = span::span_from_byte_range(&after, applied.start, applied.end);
        let Some(from_diff) = diffspan::span_from_text_diff(before, &after) else {
            // No textual difference at all. A mutation that changed nothing is not a mutation, and
            // an example claiming a span over an empty diff is a poisoned label. This is *not* a
            // span disagreement: the operators decline no-ops at candidate generation, and this is
            // the backstop, counted under its own key so it cannot dilute that alarm.
            return Err(Refusal::NoTextualChange {
                operator: op.as_str().to_string(),
            });
        };
        if from_bytes != from_diff {
            return Err(Refusal::SpanDisagreement {
                operator: op.as_str().to_string(),
                from_bytes: from_bytes.to_string(),
                from_diff: from_diff.to_string(),
            });
        }
        if !from_bytes.is_well_formed(span::total_lines(&after)) {
            return Err(Refusal::SpanDisagreement {
                operator: op.as_str().to_string(),
                from_bytes: from_bytes.to_string(),
                from_diff: format!("outside a {}-line file", span::total_lines(&after)),
            });
        }

        // --- The output must still be the language it claims to be ---------------------------
        // A mutation is allowed to change behaviour; that is the point. It is not allowed to stop
        // parsing. This is also what enforces the spec's "only where removal still parses" on
        // `logic.drop_await_or_lock`, without each operator having to prove it separately.
        if let Err(refusal) = parse_bounded(&language.grammar(), &after) {
            let (count, first_line) = match refusal {
                Refusal::ParseErrors { count, first_line } => (count, first_line),
                // Any other refusal on our own output — over the size cap, past the depth bound, a
                // parse budget — still means the output is unusable, and it is ours. It is counted
                // against the operator rather than against the input, which is the whole reason
                // this variant is distinct from `ParseErrors`.
                _ => (0, span::line_of(&after, applied.start)),
            };
            return Err(Refusal::MutationDidNotParse {
                operator: op.as_str().to_string(),
                count,
                first_line,
            });
        }

        // --- Cosmetic means behaviour-preserving, and that is verified ------------------------
        if op.verified_by_canonical_form() {
            let Some(resolved) = ctx.formatter else {
                return Err(Refusal::NoFormatter {
                    language: language_id.to_string(),
                    operator: op.as_str().to_string(),
                });
            };
            let before_canonical =
                ctx.canonical_src()
                    .map_err(|e| Refusal::CosmeticNotPreserving {
                        operator: op.as_str().to_string(),
                        detail: format!("could not canonicalise the input: {e}"),
                    })?;
            let after_canonical = fmt::canonical(resolved, &after).map_err(|e| {
                Refusal::CosmeticNotPreserving {
                    operator: op.as_str().to_string(),
                    detail: format!("could not canonicalise the output: {e}"),
                }
            })?;
            if before_canonical != after_canonical {
                return Err(Refusal::CosmeticNotPreserving {
                    operator: op.as_str().to_string(),
                    detail: "the formatter's canonical forms differ, so the edit changed more \
                             than layout"
                        .to_string(),
                });
            }
        }

        Ok(Example {
            id: format!("{}#{index}", record.id),
            pool_id: record.id.clone(),
            repo: record.repo.clone(),
            path: record.path.clone(),
            language: language_id,
            class: op.class(),
            operator: Some(op.as_str().to_string()),
            silent: op.is_silent(),
            span: Some(from_bytes),
            source_span: Some(source_span),
            function: record.identity(&body.name, body.arity),
            node_kind: body.node_kind.to_string(),
            is_nested: body.is_nested,
            diff: diffspan::unified(before, &after, DIFF_CONTEXT),
            before: before.to_string(),
            after,
            normalization,
            hunks: record.hunks.clone(),
            hunk_constrained,
            detail: candidate.detail.clone(),
            seed: self.options.seed,
            tool_version: crate::TOOL_VERSION.to_string(),
        })
    }
}

/// A record's own RNG, from `sha256(seed || record.id)`.
///
/// Per record and not per run, so the output does not depend on how many records came before —
/// which is what lets a caller pass `--limit`, or restrict to one language, and still get the same
/// examples for the records it did process.
fn record_rng(seed: u64, record_id: &str) -> ChaCha20Rng {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update([0u8]);
    hasher.update(record_id.as_bytes());
    let digest = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    ChaCha20Rng::from_seed(key)
}

/// Choose an operator, weighted by [`OpId::weight`].
///
/// The weights are what put the silent stubs — `stub.default_return`, `stub.hardcoded`,
/// `stub.early_return` — five times above `stub.panic`. They are the class a substring gate cannot
/// see, and the class the whole exercise is for.
fn weighted_index(available: &[(OpId, Vec<Candidate>)], rng: &mut ChaCha20Rng) -> Option<usize> {
    let total: u64 = available.iter().map(|(op, _)| u64::from(op.weight())).sum();
    if total == 0 {
        return None;
    }
    let mut draw = rng.random_range(0..total);
    for (index, (op, _)) in available.iter().enumerate() {
        let weight = u64::from(op.weight());
        if draw < weight {
            return Some(index);
        }
        draw -= weight;
    }
    // Unreachable while the weights sum to `total`; answering the last bucket rather than `None`
    // keeps a rounding surprise from silently dropping a body.
    Some(available.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, path: &str, source: &str, hunks: Option<Vec<Hunk>>) -> PoolRecord {
        PoolRecord {
            id: id.to_string(),
            repo: "o/r".to_string(),
            path: path.to_string(),
            language: None,
            source: source.to_string(),
            hunks,
        }
    }

    fn pool_report() -> PoolReport {
        PoolReport {
            path: String::new(),
            sha256: String::new(),
            records: 0,
            by_language: std::collections::BTreeMap::new(),
        }
    }

    const RUST_FILE: &str = "\
fn total(a: usize, b: usize) -> Result<usize, String> {
    let mut sum = 0;
    for i in 0..10 {
        if a < b {
            sum += i;
        } else {
            sum -= i;
        }
    }
    Ok(sum)
}
";

    #[test]
    fn the_same_seed_over_the_same_pool_is_byte_identical() {
        let records = vec![record("r1", "a.rs", RUST_FILE, None)];
        let options = Options {
            seed: 4242,
            ..Options::default()
        };
        let first = Generator::new(options.clone()).run(&records, pool_report());
        let second = Generator::new(options).run(&records, pool_report());
        let (a, da) = first.to_jsonl().expect("serialises");
        let (b, db) = second.to_jsonl().expect("serialises");
        assert_eq!(da, db, "two runs of one seed disagree");
        assert_eq!(a, b);
        assert!(!first.examples.is_empty(), "the fixture produced nothing");
    }

    #[test]
    fn a_different_seed_reaches_a_different_choice_somewhere() {
        let records: Vec<PoolRecord> = (0..40)
            .map(|i| record(&format!("r{i}"), &format!("a{i}.rs"), RUST_FILE, None))
            .collect();
        let a = Generator::new(Options {
            seed: 1,
            ..Options::default()
        })
        .run(&records, pool_report());
        let b = Generator::new(Options {
            seed: 2,
            ..Options::default()
        })
        .run(&records, pool_report());
        assert_ne!(
            a.to_jsonl().expect("a").0,
            b.to_jsonl().expect("b").0,
            "the seed is not reaching the choices"
        );
    }

    #[test]
    fn every_emitted_span_agrees_with_a_freshly_derived_one() {
        let records: Vec<PoolRecord> = (0..30)
            .map(|i| record(&format!("r{i}"), &format!("a{i}.rs"), RUST_FILE, None))
            .collect();
        let run = Generator::new(Options {
            seed: 99,
            ..Options::default()
        })
        .run(&records, pool_report());
        assert!(!run.examples.is_empty());
        for example in &run.examples {
            let Some(span) = example.span else {
                assert_eq!(example.class, MutationClass::Clean);
                continue;
            };
            assert_eq!(
                diffspan::span_from_text_diff(&example.before, &example.after),
                Some(span),
                "{}: the recorded span is not the one the texts imply",
                example.id
            );
            assert!(span.is_well_formed(span::total_lines(&example.after)));
        }
        assert_eq!(
            run.manifest.totals.span_disagreements, 0,
            "the fixture should not be producing disagreements"
        );
    }

    #[test]
    fn a_mutation_outside_every_hunk_is_skipped_not_relocated() {
        // The whole body sits on lines 2..10; a hunk on line 1 alone touches none of it.
        let records = vec![record(
            "r1",
            "a.rs",
            RUST_FILE,
            Some(vec![LineSpan::new(1, 1)]),
        )];
        let run = Generator::new(Options {
            seed: 7,
            clean_permille: 0,
            ..Options::default()
        })
        .run(&records, pool_report());
        assert!(
            run.examples.is_empty(),
            "emitted {:?}",
            run.examples.iter().map(|e| &e.detail).collect::<Vec<_>>()
        );
        assert!(run.manifest.totals.outside_hunk > 0, "the refusal was not counted");
    }

    #[test]
    fn a_mutation_inside_the_hunk_is_emitted_and_says_it_was_constrained() {
        let records = vec![record(
            "r1",
            "a.rs",
            RUST_FILE,
            Some(vec![LineSpan::new(1, 12)]),
        )];
        let run = Generator::new(Options {
            seed: 7,
            clean_permille: 0,
            ..Options::default()
        })
        .run(&records, pool_report());
        assert!(!run.examples.is_empty());
        for example in &run.examples {
            assert!(example.hunk_constrained);
            let source_span = example.source_span.expect("a mutated example has one");
            assert!(source_span.intersects(&LineSpan::new(1, 12)));
        }
        assert_eq!(run.manifest.totals.examples_without_hunk_constraint, 0);
    }

    #[test]
    fn a_pool_with_no_hunks_says_so_rather_than_claiming_the_constraint_held() {
        let records = vec![record("r1", "a.rs", RUST_FILE, None)];
        let run = Generator::new(Options {
            seed: 7,
            clean_permille: 0,
            ..Options::default()
        })
        .run(&records, pool_report());
        assert!(!run.examples.is_empty());
        assert!(run.examples.iter().all(|e| !e.hunk_constrained));
        assert_eq!(
            run.manifest.totals.examples_without_hunk_constraint,
            run.examples.len() as u64
        );
    }

    #[test]
    fn one_file_never_appears_both_mutated_and_clean() {
        // Twenty records over two files, each file appearing ten times.
        let records: Vec<PoolRecord> = (0..20)
            .map(|i| record(&format!("r{i}"), &format!("f{}.rs", i % 2), RUST_FILE, None))
            .collect();
        let run = Generator::new(Options {
            seed: 5,
            clean_permille: 500,
            ..Options::default()
        })
        .run(&records, pool_report());
        let mut seen: HashMap<FunctionIdentity, MutationClass> = HashMap::new();
        for example in &run.examples {
            let is_clean = example.class == MutationClass::Clean;
            match seen.get(&example.function) {
                Some(previous) => assert_eq!(
                    *previous == MutationClass::Clean,
                    is_clean,
                    "{} appears both mutated and clean",
                    example.function
                ),
                None => {
                    seen.insert(example.function.clone(), example.class);
                }
            }
        }
        assert_eq!(
            run.manifest.refusals.get("split_conflict"),
            None,
            "keying the disposition on (repo, path) should make this impossible"
        );
        assert!(seen.len() >= 2, "the fixture did not exercise both files");
    }

    #[test]
    fn a_limit_caps_the_output_without_changing_what_precedes_it() {
        let records: Vec<PoolRecord> = (0..10)
            .map(|i| record(&format!("r{i}"), &format!("a{i}.rs"), RUST_FILE, None))
            .collect();
        let full = Generator::new(Options {
            seed: 11,
            ..Options::default()
        })
        .run(&records, pool_report());
        let capped = Generator::new(Options {
            seed: 11,
            limit: Some(3),
            ..Options::default()
        })
        .run(&records, pool_report());
        assert!(capped.examples.len() <= 3);
        assert!(!capped.examples.is_empty());
        for (a, b) in full.examples.iter().zip(capped.examples.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.span, b.span);
        }
    }

    #[test]
    fn a_language_filter_keeps_the_other_languages_out() {
        let records = vec![
            record("r1", "a.rs", RUST_FILE, None),
            record("r2", "b.py", "def f(a: int) -> int:\n    x = 1\n    return x\n", None),
        ];
        let run = Generator::new(Options {
            seed: 3,
            clean_permille: 0,
            languages: vec![LangId::Python],
            ..Options::default()
        })
        .run(&records, pool_report());
        assert!(run.examples.iter().all(|e| e.language == LangId::Python));
        assert!(!run.examples.is_empty());
    }

    #[test]
    fn the_weighted_choice_favours_the_silent_stubs() {
        // Twelve hundred single-function files, each offering the full stub set. `stub.panic` and
        // the three silent stubs compete directly, and the ratio should follow the weights.
        let records: Vec<PoolRecord> = (0..400)
            .map(|i| record(&format!("r{i}"), &format!("a{i}.rs"), RUST_FILE, None))
            .collect();
        let run = Generator::new(Options {
            seed: 2026,
            clean_permille: 0,
            ..Options::default()
        })
        .run(&records, pool_report());
        let silent = run
            .examples
            .iter()
            .filter(|e| e.silent)
            .count();
        let panics = run
            .examples
            .iter()
            .filter(|e| e.operator.as_deref() == Some("stub.panic"))
            .count();
        assert!(
            silent > panics * 3,
            "silent {silent} vs stub.panic {panics}: the weights are not reaching the choice"
        );
        assert_eq!(run.manifest.totals.silent_stubs as usize, silent);
    }

    #[test]
    fn the_record_rng_depends_on_the_seed_and_the_record_and_nothing_else() {
        let a: u64 = record_rng(1, "r1").random();
        let b: u64 = record_rng(1, "r1").random();
        let c: u64 = record_rng(2, "r1").random();
        let d: u64 = record_rng(1, "r2").random();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
}
