//! The allowlist: which split units this generator may read, decided by the canonical split.
//!
//! The repo-level split (`qd_data.split.assign_repo`) and the SQuAD title partition
//! (`qd_data.split.squad_title_family`) are keyed blake2b hashes owned by Python. They are not
//! re-implemented here: a second implementation of "which titles are held out" is exactly the
//! copy that drifts, and the drift would put held-out content into training rows (CLAUDE.md
//! rule 3). `qd_data.defect_class.noul_allowlist` runs the canonical functions and writes this
//! file; this crate reads it, checks that the inputs it is about to read are the bytes the
//! allowlist was computed from, and emits rows only from the units it names.
//!
//! The allowlist is the generator's filter, not the rule-3 guarantee. The loader
//! (`qd_data.defect_class.load_noul_rows`) re-derives every row's split with the same canonical
//! functions at the training run's own seed and refuses the whole corpus on any row outside
//! `train`, so a hand-edited allowlist cannot reach a training process either.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "qd-noul-allowlist/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub seed: u64,
    pub train_fraction: f64,
    pub val_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Squad {
    /// File name of the SQuAD v2 train JSONL the titles were read from.
    pub file: String,
    pub sha256: String,
    /// `qd_data.sources`'s registered licence for `rajpurkar/squad_v2`.
    pub licence: String,
    /// Train-split `qa.answer_span` title -> its split unit (`squad-title:<title>`), which is the
    /// `repo_key` `qd_data.mixture.rewrite_squad` gives that title's rows.
    pub titles: BTreeMap<String, String>,
    pub excluded: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pool {
    /// File name of the commitpackft pool, re-anchored under `data/pool/` by the loader.
    pub file: String,
    pub sha256: String,
    pub records: u64,
    /// Pool id -> the licence the commitpackft download gives it, for every file of a
    /// train-split repo in one of the languages the defect corpus holds.
    pub files: BTreeMap<String, String>,
    pub excluded: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Templates {
    /// `sha256` of the template catalogue the units were listed from (`qd-noul-rows units`).
    pub catalogue_sha256: String,
    pub licence: String,
    pub licence_basis: String,
    /// Train-split template units.
    pub units: Vec<String>,
    pub excluded: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Allowlist {
    pub schema: String,
    pub split: Split,
    /// `qd_data.render.INVISIBLE_FORMAT_RANGES`: codepoints a row may not carry.
    pub invisible_format_ranges: Vec<(u32, u32)>,
    pub squad: Squad,
    pub pool: Pool,
    pub templates: Templates,
}

impl Allowlist {
    /// Refuse an allowlist that could not have come from the canonical step.
    pub fn check(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("schema {:?}, expected {SCHEMA:?}", self.schema));
        }
        if self.invisible_format_ranges.is_empty() {
            return Err(
                "no invisible-format ranges: the filter would pass every codepoint".to_string(),
            );
        }
        if let Some((lo, hi)) = self.invisible_format_ranges.iter().find(|(lo, hi)| lo > hi) {
            return Err(format!("invisible-format range {lo:#x}..{hi:#x} is backwards"));
        }
        for (what, sha) in [
            ("squad", &self.squad.sha256),
            ("pool", &self.pool.sha256),
            ("templates", &self.templates.catalogue_sha256),
        ] {
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("{what} sha256 {sha:?} is not a sha256"));
            }
        }
        Ok(())
    }
}
