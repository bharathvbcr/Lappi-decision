//! The manifest: what this run actually did, in numbers a later reader can act on.
//!
//! Three things in here are load-bearing rather than informational.
//!
//! **The seed and the hashes.** "The same seed and the same input pool produce byte-identical
//! output" is only a claim a consumer can check if the manifest names the seed, the pool it read
//! and the digest of what it wrote. A protocol hash in the ledger over output nobody can regenerate
//! is a fiction.
//!
//! **The refusal histogram.** A language with a high refusal rate has a thinner mixture than its
//! headline count suggests. Mixture weights that do not know that are set from a number the
//! generator knows to be misleading.
//!
//! **The per-operator coverage table, with `sites_found` beside `emitted`.** A language whose
//! facade finds no `await` yields no `logic.drop_await_or_lock`, and that is correct — but it must
//! not read the same as an operator that ran. `sites_found: 0` and `sites_found: 40, emitted: 0`
//! are different findings and the table carries both.
//!
//! There is deliberately **no timestamp**. A wall clock in here would make two runs of the same
//! seed over the same pool produce different manifests, which is the one property the determinism
//! test exists to assert.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fmt::Availability;
use crate::lang::LangId;
use crate::ops::OpId;
use crate::parse::Refusal;

/// Hex-encoded SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorReport {
    /// Candidates the operator offered across every body it was asked about. Zero here means the
    /// construct is not present in this language's slice of the pool.
    pub sites_found: u64,
    /// Candidates that became an emitted example.
    pub emitted: u64,
    /// Times the operator refused, by refusal key.
    pub refused: BTreeMap<String, u64>,
}

impl OperatorReport {
    pub fn total_refused(&self) -> u64 {
        self.refused.values().sum()
    }
}

/// What one language did in this run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageReport {
    pub language: LangId,
    /// Measured on this machine, not declared in the source. `not_found` is a different finding
    /// from `not_declared` and both are different from `found`.
    pub formatter: Availability,
    /// Operators unavailable here, and why. Populated whenever the cosmetic set shrinks, so the
    /// restriction is recorded rather than assumed.
    pub restricted_operators: BTreeMap<String, String>,
    pub files_seen: u64,
    pub files_refused: u64,
    pub bodies_seen: u64,
    /// Bodies that are lexically inside another body — a nested fn, a closure, a lambda. Counted
    /// because "the function body" is ambiguous for them and a consumer may want to know how much
    /// of the mixture came from inner ones.
    pub bodies_nested: u64,
    pub examples: u64,
    pub operators: BTreeMap<String, OperatorReport>,
    pub refusals: BTreeMap<String, u64>,
}

impl LanguageReport {
    fn new(language: LangId, formatter: Availability) -> Self {
        LanguageReport {
            language,
            formatter,
            restricted_operators: BTreeMap::new(),
            files_seen: 0,
            files_refused: 0,
            bodies_seen: 0,
            bodies_nested: 0,
            examples: 0,
            operators: BTreeMap::new(),
            refusals: BTreeMap::new(),
        }
    }

    pub fn total_refusals(&self) -> u64 {
        self.refusals.values().sum()
    }

    /// Operators that fired at least once here.
    pub fn operators_fired(&self) -> usize {
        self.operators.values().filter(|r| r.emitted > 0).count()
    }
}

/// What the run read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolReport {
    pub path: String,
    pub sha256: String,
    pub records: u64,
    pub by_language: BTreeMap<String, u64>,
}

/// Run-wide counts that must never be read off a per-language table by addition.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    pub examples: u64,
    pub mutated: u64,
    pub clean: u64,
    /// Emitted examples whose operator a substring gate cannot see. The number the whole exercise
    /// is for.
    pub silent_stubs: u64,
    /// Examples dropped because the two span derivations disagreed.
    ///
    /// **This is the number to watch.** It is not a nuisance counter: a rising one means the
    /// mutator's byte bookkeeping and the textual differ have drifted apart, and the labels that
    /// *did* agree were produced by the same two code paths. Zero is the expected value.
    pub span_disagreements: u64,
    /// Candidate mutations dropped because their site fell outside every hunk the agent touched.
    pub outside_hunk: u64,
    /// Emitted examples whose pool record carried no hunks, so the placement rule could not be
    /// applied. Carried separately: an unconstrained example is not a constrained one, and the sum
    /// of the two is not "examples that landed in the hunk".
    pub examples_without_hunk_constraint: u64,
    /// Mutations whose output no longer parsed. Ours, not the pool's.
    pub mutation_did_not_parse: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub tool_version: String,
    /// The `rand_chacha` seed every choice in the run came from.
    pub seed: u64,
    /// The per-record sub-seed derivation, named so a reader can reproduce it without the source.
    pub seed_derivation: String,
    pub pool: PoolReport,
    /// Digest of the emitted JSONL, byte for byte. Filled in once the output is written.
    pub examples_sha256: String,
    pub totals: Totals,
    pub languages: Vec<LanguageReport>,
    /// Every refusal in the run, by key. The per-language tables must agree with this; they are
    /// kept separately so a disagreement is visible rather than impossible.
    pub refusals: BTreeMap<String, u64>,
}

impl Manifest {
    pub fn new(seed: u64, pool: PoolReport, formatters: &[(LangId, Availability)]) -> Self {
        let languages = LangId::ALL
            .iter()
            .map(|id| {
                let availability = formatters
                    .iter()
                    .find(|(l, _)| l == id)
                    .map(|(_, a)| a.clone())
                    .unwrap_or(Availability::NotDeclared);
                let mut report = LanguageReport::new(*id, availability);
                // The shrink is recorded up front, from the measured availability — before a single
                // file is read, so it is a property of the run rather than of whichever file
                // happened to reach the operator first.
                if !report.formatter.is_usable() {
                    for op in OpId::ALL.into_iter().filter(|o| o.requires_formatter()) {
                        report.restricted_operators.insert(
                            op.as_str().to_string(),
                            format!(
                                "no usable formatter for {id} on this machine ({})",
                                report.formatter.key()
                            ),
                        );
                    }
                }
                let language = crate::lang::for_id(*id);
                if !language.line_wrapping_is_safe() {
                    report.restricted_operators.insert(
                        OpId::CosmeticWrapLine.as_str().to_string(),
                        format!("{id} terminates statements at line ends"),
                    );
                }
                report
            })
            .collect();
        Manifest {
            tool_version: crate::TOOL_VERSION.to_string(),
            seed,
            seed_derivation: "ChaCha20Rng::from_seed(sha256(seed.to_le_bytes() || record.id))"
                .to_string(),
            pool,
            examples_sha256: String::new(),
            totals: Totals::default(),
            languages,
            refusals: BTreeMap::new(),
        }
    }

    fn language_mut(&mut self, id: LangId) -> &mut LanguageReport {
        // `LangId::ALL` seeds every entry in `new`, so the fallback can only be reached if that
        // list and this lookup disagree — which the `every_language_has_a_report` test forbids.
        let index = self
            .languages
            .iter()
            .position(|r| r.language == id)
            .unwrap_or(0);
        &mut self.languages[index]
    }

    pub fn note_file(&mut self, id: LangId) {
        self.language_mut(id).files_seen += 1;
    }

    pub fn note_file_refused(&mut self, id: LangId, refusal: &Refusal) {
        self.language_mut(id).files_refused += 1;
        self.note_refusal(id, refusal);
    }

    pub fn note_body(&mut self, id: LangId, is_nested: bool) {
        let report = self.language_mut(id);
        report.bodies_seen += 1;
        if is_nested {
            report.bodies_nested += 1;
        }
    }

    pub fn note_sites(&mut self, id: LangId, op: OpId, count: u64) {
        self.language_mut(id)
            .operators
            .entry(op.as_str().to_string())
            .or_default()
            .sites_found += count;
    }

    pub fn note_emitted(&mut self, id: LangId, op: OpId) {
        let report = self.language_mut(id);
        report.examples += 1;
        report
            .operators
            .entry(op.as_str().to_string())
            .or_default()
            .emitted += 1;
    }

    pub fn note_clean(&mut self, id: LangId) {
        self.language_mut(id).examples += 1;
    }

    /// Record a refusal against a language, and against the operator when the refusal names one.
    pub fn note_refusal(&mut self, id: LangId, refusal: &Refusal) {
        let key = refusal.key().to_string();
        let operator = operator_of(refusal);
        {
            let report = self.language_mut(id);
            *report.refusals.entry(key.clone()).or_insert(0) += 1;
            if let Some(op) = &operator {
                *report
                    .operators
                    .entry(op.clone())
                    .or_default()
                    .refused
                    .entry(key.clone())
                    .or_insert(0) += 1;
            }
        }
        *self.refusals.entry(key).or_insert(0) += 1;
        match refusal {
            Refusal::SpanDisagreement { .. } => self.totals.span_disagreements += 1,
            Refusal::OutsideHunk { .. } => self.totals.outside_hunk += 1,
            Refusal::MutationDidNotParse { .. } => self.totals.mutation_did_not_parse += 1,
            _ => {}
        }
    }

    /// A one-screen coverage table. Printed by `qd-mutate generate` and by `qd-mutate coverage`.
    pub fn coverage_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "seed {}  tool {}  pool {} records  examples {}\n",
            self.seed, self.tool_version, self.pool.records, self.totals.examples
        ));
        out.push_str(&format!(
            "mutated {}  clean {}  silent-stub {}  span-disagreements {}  outside-hunk {}  \
             unconstrained {}\n",
            self.totals.mutated,
            self.totals.clean,
            self.totals.silent_stubs,
            self.totals.span_disagreements,
            self.totals.outside_hunk,
            self.totals.examples_without_hunk_constraint,
        ));
        for report in &self.languages {
            out.push_str(&format!(
                "\n[{}] formatter={} files={} refused={} bodies={} (nested {}) examples={} \
                 operators-fired={}/{}\n",
                report.language,
                report.formatter.key(),
                report.files_seen,
                report.files_refused,
                report.bodies_seen,
                report.bodies_nested,
                report.examples,
                report.operators_fired(),
                OpId::ALL.len(),
            ));
            for op in OpId::ALL {
                let key = op.as_str();
                let entry = report.operators.get(key);
                let (sites, emitted, refused) = entry
                    .map(|r| (r.sites_found, r.emitted, r.total_refused()))
                    .unwrap_or((0, 0, 0));
                let restriction = report
                    .restricted_operators
                    .get(key)
                    .map(|why| format!("  RESTRICTED: {why}"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "  {key:<28} sites {sites:>6}  emitted {emitted:>6}  refused {refused:>6}\
                     {restriction}\n"
                ));
            }
            if !report.refusals.is_empty() {
                out.push_str("  refusals:");
                for (key, count) in &report.refusals {
                    out.push_str(&format!(" {key}={count}"));
                }
                out.push('\n');
            }
        }
        out
    }
}

/// The operator a refusal names, when it names one.
fn operator_of(refusal: &Refusal) -> Option<String> {
    match refusal {
        Refusal::PreconditionUnmet { operator, .. }
        | Refusal::OutsideHunk { operator }
        | Refusal::SpanDisagreement { operator, .. }
        | Refusal::CosmeticNotPreserving { operator, .. }
        | Refusal::NoTextualChange { operator }
        | Refusal::MutationDidNotParse { operator, .. } => Some(operator.clone()),
        Refusal::NoFormatter { operator, .. } => Some(operator.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fmt::Availability;

    fn pool_report() -> PoolReport {
        PoolReport {
            path: "pool.jsonl".to_string(),
            sha256: "0".repeat(64),
            records: 0,
            by_language: BTreeMap::new(),
        }
    }

    #[test]
    fn every_language_has_a_report_and_the_lookup_finds_it() {
        let mut manifest = Manifest::new(7, pool_report(), &[]);
        assert_eq!(manifest.languages.len(), LangId::ALL.len());
        for id in LangId::ALL {
            manifest.note_body(id, false);
        }
        for id in LangId::ALL {
            let report = manifest
                .languages
                .iter()
                .find(|r| r.language == id)
                .expect("a report per language");
            assert_eq!(report.bodies_seen, 1, "{id} was credited to another language");
        }
    }

    #[test]
    fn a_missing_formatter_restricts_the_operators_before_a_file_is_read() {
        let formatters = vec![(
            LangId::TypeScript,
            Availability::NotFound {
                program: "prettier".to_string(),
                detail: "not on PATH".to_string(),
            },
        )];
        let manifest = Manifest::new(1, pool_report(), &formatters);
        let ts = manifest
            .languages
            .iter()
            .find(|r| r.language == LangId::TypeScript)
            .expect("typescript");
        assert!(ts.restricted_operators.contains_key("cosmetic.reformat"));
        assert!(ts.restricted_operators.contains_key("cosmetic.reorder_imports"));
        assert!(!ts.formatter.is_usable());
    }

    #[test]
    fn a_present_formatter_restricts_only_what_the_language_itself_forbids() {
        let formatters = vec![(
            LangId::Rust,
            Availability::Found {
                program: "rustfmt".to_string(),
                path: "/usr/bin/rustfmt".to_string(),
            },
        )];
        let manifest = Manifest::new(1, pool_report(), &formatters);
        let rust = manifest
            .languages
            .iter()
            .find(|r| r.language == LangId::Rust)
            .expect("rust");
        assert!(rust.restricted_operators.is_empty(), "{:?}", rust.restricted_operators);

        let python = manifest
            .languages
            .iter()
            .find(|r| r.language == LangId::Python)
            .expect("python");
        assert_eq!(
            python.restricted_operators.get("cosmetic.wrap_line").map(String::as_str),
            Some("python terminates statements at line ends")
        );
    }

    #[test]
    fn a_span_disagreement_is_counted_in_three_places_at_once() {
        let mut manifest = Manifest::new(1, pool_report(), &[]);
        manifest.note_refusal(
            LangId::Go,
            &Refusal::SpanDisagreement {
                operator: "stub.panic".to_string(),
                from_bytes: "3..=3".to_string(),
                from_diff: "4..=4".to_string(),
            },
        );
        assert_eq!(manifest.totals.span_disagreements, 1);
        assert_eq!(manifest.refusals.get("span_disagreement"), Some(&1));
        let go = manifest
            .languages
            .iter()
            .find(|r| r.language == LangId::Go)
            .expect("go");
        assert_eq!(go.refusals.get("span_disagreement"), Some(&1));
        assert_eq!(
            go.operators
                .get("stub.panic")
                .map(OperatorReport::total_refused),
            Some(1)
        );
    }

    #[test]
    fn sites_found_zero_reads_differently_from_emitted_zero() {
        let mut manifest = Manifest::new(1, pool_report(), &[]);
        manifest.note_sites(LangId::Rust, OpId::LogicSwapArgs, 40);
        let table = manifest.coverage_table();
        assert!(table.contains("logic.swap_args"), "{table}");
        let rust = manifest
            .languages
            .iter()
            .find(|r| r.language == LangId::Rust)
            .expect("rust");
        let swap = rust.operators.get("logic.swap_args").expect("recorded");
        assert_eq!((swap.sites_found, swap.emitted), (40, 0));
        // The operator with no sites at all has no entry, which the table renders as 0/0.
        assert!(!rust.operators.contains_key("logic.drop_await_or_lock"));
    }

    #[test]
    fn the_digest_is_stable_and_hex() {
        let a = sha256_hex(b"qd-mutate");
        assert_eq!(a.len(), 64);
        assert_eq!(a, sha256_hex(b"qd-mutate"));
        assert_ne!(a, sha256_hex(b"qd-mutate "));
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn a_manifest_carries_no_wall_clock() {
        let manifest = Manifest::new(99, pool_report(), &[]);
        let json = serde_json::to_string(&manifest).expect("serialises");
        for forbidden in ["timestamp", "generated_at", "created", "time"] {
            assert!(
                !json.contains(forbidden),
                "a `{forbidden}` field would make two runs of one seed differ"
            );
        }
    }
}
