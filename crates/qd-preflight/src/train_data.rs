//! `qd-train-data`'s checks: **no benchmark target is in the train set** (the human's Q1 ruling,
//! 2026-10-17, `AUDIT/harden-2026-10-17/human-ruling-2026-10-17.md`; mistakes M6/M7/M8, the v5r
//! shards that trained on MMLU and CLINC test items).
//!
//! A train set is a shard set (a directory: `header.json`, token ids, no text) or an
//! `examples.jsonl` (text). Four tiers, each a [`TriState`], and the verdict is their
//! [`aggregate`]: any hit is `FAILED`; otherwise any tier that could not run makes it `NOT_RUN`;
//! only all four ran and clean is `PASS`. A tier that could not run is never a pass, and the
//! coverage block says how many targets no tier could scan, never presenting a capped coverage as
//! complete.
//!
//! * `forbidden_family`: rows of a family the v6 rules empty of training rows
//!   ([`FORBIDDEN_FAMILIES`], Fable's R2: MMLU at zero). From the train manifest's entries (shard
//!   set) or the examples. No text needed.
//! * `clinc_digest`: a CLINC row's id carries the blake2b-8 digest of its stripped utterance
//!   (`qd_data.mixture.clinc_keys`); a target whose digest equals one is that very utterance. Exact,
//!   any length, no text needed, so a shard set can be checked for CLINC test items.
//! * `ngram`: word 8-gram containment of every target in each example, through qd-prep's
//!   positive-controlled scan ([`qd_prep::pool::scan`]): a scan that cannot demonstrate a planted
//!   hit is refused (`NOT_RUN`), never reported as zero hits. Needs text.
//! * `exact_short`: targets under n words have no n-gram, so the scan cannot see them (2,454 of
//!   CLINC's 5,500 test items, `GAP-CLINC-TEST-TARGETS-8GRAM-BLIND-2026-10-06`). Under the owner's
//!   ruling of 2026-10-19 (`AUDIT/harden-2026-10-17/human-ruling-2026-10-19.md`) a row is a hit
//!   only when its WHOLE strip-v2 text, normalised, equals a short target. The matcher is
//!   [`qd_prep::exact_short`], the one `qd-prep containment` (the builder) runs, so the gate and
//!   the builder cannot disagree. `--examples` must be the strip-v2 text (the exporter's
//!   `stripped.jsonl`); on unstripped prompts a whole-text match cannot happen by construction.
//!   Own planted-copy control. Needs text.
//!
//! What this does not establish: a *paraphrase* of a short target. A target equal to one line
//! inside a longer text is not a hit; the coverage block counts those rows and distinct targets
//! (`short_embedded_rows`, `short_embedded_targets`) so that residual is a number.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use qd_prep::blake2b::Keyed;
use qd_prep::decisions::{Candidate, Gold, TargetSet};
use qd_prep::exact_short::{ShortTargets, word_count};
use qd_prep::pool;
use serde::Deserialize;
use serde_json::{Value, json};

pub use crate::tristate::{TriState, aggregate};

use crate::pool_policy::{RunTarget, check_run_target, is_vani_agreed};

/// The exit code of a verdict that found a benchmark target in the train set.
pub const EXIT_FAILED: u8 = 1;
/// The exit code of a verdict that could not be reached (the Makefile's `NOT_RUN`).
pub const EXIT_NOT_RUN: u8 = 3;

/// The rule's word n and containment threshold (`qd_train.replay.DEFAULT_N`, `DEFAULT_THRESHOLD`).
pub const DEFAULT_N: u32 = 8;
pub const DEFAULT_THRESHOLD: f64 = 0.5;
/// Families v6 trains on zero rows of (`real_tokenizer_pipeline.V6_EMPTY_TRAIN_FAMILY`).
pub const FORBIDDEN_FAMILIES: [&str; 1] = ["knowledge.multiple_choice"];
/// The CLINC source whose row ids carry the utterance digest.
pub const CLINC_SOURCE: &str = "clinc/clinc_oos";
/// Most hit ids named per tier.
pub const SAMPLE_IDS: usize = 8;
/// Most planted copies the `exact_short` control plants (the n-gram control has qd-prep's bound).
pub const EXACT_CONTROL_MAX: usize = 4096;
/// Bound on the train manifest read.
const MAX_MANIFEST_BYTES: u64 = 8 << 30;

#[derive(Clone, Debug)]
pub struct Inputs {
    /// A shard set directory, or an `examples.jsonl`.
    pub train: PathBuf,
    /// The text of the rows, for a shard set (which holds none). Optional for a directory.
    pub examples: Option<PathBuf>,
    /// The train manifest; default `<set>/../../data/pool/train.json` for a shard set.
    pub manifest: Option<PathBuf>,
    /// `NAME=FILE` target sets, `{"id", "text"}` JSONL.
    pub targets: Vec<(String, PathBuf)>,
    pub forbid_families: Vec<String>,
    pub n: u32,
    pub threshold: f64,
    pub threads: usize,
    /// Where the run trains (`--run-target`). `None` is unstated, which refuses any `vani-agreed`
    /// row ([`tier_mac_only`]).
    pub run_target: Option<RunTarget>,
}

/// One tier's result: its tri-state, how many train rows it found, and a few of their ids.
#[derive(Clone, Debug)]
pub struct Tier {
    pub name: &'static str,
    pub state: TriState,
    pub hits: u64,
    pub sample_ids: Vec<String>,
}

impl Tier {
    fn ran(name: &'static str, hits: u64, ids: Vec<String>, detail: String) -> Self {
        let sample_ids = ids.into_iter().take(SAMPLE_IDS).collect();
        let state = if hits == 0 {
            TriState::pass(detail)
        } else {
            TriState::fail(detail)
        };
        Tier {
            name,
            state,
            hits,
            sample_ids,
        }
    }

    fn not_run(name: &'static str, reason: impl Into<String>) -> Self {
        Tier {
            name,
            state: TriState::not_run(reason),
            hits: 0,
            sample_ids: Vec::new(),
        }
    }
}

/// How much of the target set some tier could scan. `unscanned` is what no tier that ran covers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub target_sets: usize,
    pub targets: usize,
    /// Targets with at least n words: the n-gram tier's.
    pub ngram_scannable: usize,
    /// Targets under n words (and at least one): the exact tier's, and only its.
    pub short: usize,
    /// Targets with no word at all: nothing can match them.
    pub no_words: usize,
    /// Targets no tier that ran could scan.
    pub unscanned: usize,
    /// The exact tier's residual (owner ruling 2026-10-19): train rows that hold a short target
    /// as one line inside a longer text. Not hits; counted so the residual is a number.
    pub short_embedded_rows: u64,
    /// Distinct short targets among those embedded matches.
    pub short_embedded_targets: u64,
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub tiers: Vec<Tier>,
    pub coverage: Coverage,
    pub overall: TriState,
    pub target_sha256: BTreeMap<String, String>,
}

impl Verdict {
    pub fn exit_code(&self) -> u8 {
        match &self.overall {
            TriState::Ran { passed: true, .. } => 0,
            TriState::Ran { passed: false, .. } => EXIT_FAILED,
            TriState::NotRun { .. } => EXIT_NOT_RUN,
        }
    }

    pub fn result_line(&self) -> String {
        match &self.overall {
            TriState::Ran { passed: true, .. } => {
                "RESULT: PASS - every tier ran and no benchmark target is in the train set.".into()
            }
            TriState::Ran { passed: false, .. } => {
                "RESULT: FAILED - a benchmark target is in the train set.".into()
            }
            TriState::NotRun { .. } => {
                "RESULT: NotRun - nothing was found, but a tier could not run.".into()
            }
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "overall": serde_json::to_value(&self.overall).unwrap_or(Value::Null),
            "tiers": self.tiers.iter().map(|t| json!({
                "tier": t.name,
                "state": serde_json::to_value(&t.state).unwrap_or(Value::Null),
                "hits": t.hits, "sample_ids": t.sample_ids,
            })).collect::<Vec<_>>(),
            "coverage": {
                "target_sets": self.coverage.target_sets, "targets": self.coverage.targets,
                "ngram_scannable": self.coverage.ngram_scannable, "short": self.coverage.short,
                "no_words": self.coverage.no_words, "unscanned": self.coverage.unscanned,
                "short_embedded_rows": self.coverage.short_embedded_rows,
                "short_embedded_targets": self.coverage.short_embedded_targets,
            },
            "target_sha256": self.target_sha256,
        })
    }
}

/// A train row's identity as the manifest-level tiers read it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    id: String,
    source_id: String,
    family_id: String,
    /// The row is a `vani-agreed` row ([`is_vani_agreed`]).
    vani_agreed: bool,
}

#[derive(Deserialize)]
struct ManifestEntry {
    row_id: String,
    source_id: String,
    family_id: String,
    /// Optional: older manifests have neither. `source_id` alone already names a vani-agreed row.
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    licence_id: Option<String>,
}

#[derive(Deserialize)]
struct Manifest {
    data_snapshot_hash: String,
    n_rows: u64,
    entries: Vec<ManifestEntry>,
    /// Where the run this manifest feeds trains, when the manifest says (`"mac"` or `"cloud"`).
    #[serde(default)]
    run_target: Option<String>,
}

#[derive(Deserialize)]
struct Header {
    data_snapshot_hash: String,
}

/// The train manifest of a shard set and whether its hash is the header's. Every refusal is a
/// reason string: the manifest-level tiers are then `NOT_RUN`, never silently clean.
fn read_manifest(
    set: &Path,
    explicit: Option<&Path>,
) -> Result<(Vec<Entry>, Option<String>), String> {
    let header_path = set.join("header.json");
    let header: Header = read_json(&header_path)?;
    let manifest_path = match explicit {
        Some(p) => p.to_path_buf(),
        None => set
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("data").join("pool").join("train.json"))
            .ok_or_else(|| {
                format!(
                    "{}: no parent to find data/pool/train.json from",
                    set.display()
                )
            })?,
    };
    let manifest: Manifest = read_json(&manifest_path)?;
    if manifest.data_snapshot_hash != header.data_snapshot_hash {
        return Err(format!(
            "{} records data_snapshot_hash {} but {} pins {}: not this shard set's manifest",
            manifest_path.display(),
            manifest.data_snapshot_hash,
            header_path.display(),
            header.data_snapshot_hash
        ));
    }
    if manifest.n_rows != manifest.entries.len() as u64 {
        return Err(format!(
            "{}: n_rows {} but {} entries",
            manifest_path.display(),
            manifest.n_rows,
            manifest.entries.len()
        ));
    }
    let run_target = manifest.run_target;
    let entries = manifest
        .entries
        .into_iter()
        .map(|e| Entry {
            vani_agreed: is_vani_agreed(&e.source_id)
                || e.source.as_deref().is_some_and(is_vani_agreed)
                || e.licence_id.as_deref().is_some_and(is_vani_agreed),
            id: e.row_id,
            source_id: e.source_id,
            family_id: e.family_id,
        })
        .collect();
    Ok((entries, run_target))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if len > MAX_MANIFEST_BYTES {
        return Err(format!(
            "{}: {len} bytes; the bound is {MAX_MANIFEST_BYTES}",
            path.display()
        ));
    }
    serde_json::from_reader(BufReader::new(file)).map_err(|e| format!("{}: {e}", path.display()))
}

/// `qd_data.mixture.clinc_keys`'s digest: blake2b with an 8-byte output of the stripped
/// utterance, as lowercase hex.
pub fn clinc_digest(utterance: &str) -> Result<String, String> {
    let hasher = Keyed::new(&[], 8)?;
    Ok(hasher
        .digest(utterance.trim().as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The digest segment of a CLINC row id, `clinc:<family>:<digest>:<index>`.
fn clinc_row_digest(row_id: &str) -> Option<&str> {
    let mut parts = row_id.split(':');
    (parts.next()? == "clinc").then_some(())?;
    let _family = parts.next()?;
    let digest = parts.next()?;
    (digest.len() == 16).then_some(digest)
}

/// Rows of a forbidden family among `entries`.
fn tier_forbidden_family(entries: Option<&[Entry]>, why: &str, forbid: &[String]) -> Tier {
    let name = "forbidden_family";
    let Some(entries) = entries else {
        return Tier::not_run(name, why.to_owned());
    };
    let hit: Vec<&Entry> = entries
        .iter()
        .filter(|e| forbid.contains(&e.family_id))
        .collect();
    Tier::ran(
        name,
        hit.len() as u64,
        hit.iter().map(|e| e.id.clone()).collect(),
        format!(
            "{} of {} train rows are in {forbid:?} (v6 trains on none of them)",
            hit.len(),
            entries.len()
        ),
    )
}

/// CLINC rows whose utterance digest is a target's.
fn tier_clinc_digest(
    entries: Option<&[Entry]>,
    why: &str,
    targets: &[TargetSet],
) -> Result<Tier, String> {
    let name = "clinc_digest";
    let Some(entries) = entries else {
        return Ok(Tier::not_run(name, why.to_owned()));
    };
    let mut by_digest: HashSet<String> = HashSet::new();
    for (_, rows) in targets {
        for (_, text) in rows {
            by_digest.insert(clinc_digest(text)?);
        }
    }
    let clinc: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.source_id == CLINC_SOURCE)
        .collect();
    let hit: Vec<&Entry> = clinc
        .iter()
        .copied()
        .filter(|e| clinc_row_digest(&e.id).is_some_and(|d| by_digest.contains(d)))
        .collect();
    Ok(Tier::ran(
        name,
        hit.len() as u64,
        hit.iter().map(|e| e.id.clone()).collect(),
        format!(
            "{} of {} CLINC train rows are a target utterance (digest match, any length)",
            hit.len(),
            clinc.len()
        ),
    ))
}

/// Target sets with at least one row of n or more words: the n-gram tier's. A set whose every row
/// is shorter would fail qd-prep's positive control by design (nothing in it can ever match), so
/// it is left to the exact tier and counted there.
fn ngram_targets(targets: &[TargetSet], n: usize) -> Vec<TargetSet> {
    targets
        .iter()
        .filter(|(_, rows)| rows.iter().any(|(_, t)| word_count(t) >= n))
        .cloned()
        .collect()
}

fn tier_ngram(
    examples: Option<&[pool::ExampleText]>,
    why: &str,
    targets: &[TargetSet],
    inputs: &Inputs,
) -> Tier {
    let name = "ngram";
    let Some(examples) = examples else {
        return Tier::not_run(name, why.to_owned());
    };
    let usable = ngram_targets(targets, inputs.n as usize);
    if usable.is_empty() {
        return Tier::not_run(
            name,
            format!(
                "no target set has a row of {} or more words, so no n-gram scan can run",
                inputs.n
            ),
        );
    }
    let candidates: Vec<Candidate> = examples
        .iter()
        .map(|e| Candidate {
            id: e.id.clone(),
            source_id: "qd-train-data",
            family_id: e.family_id.clone(),
            stratum: String::new(),
            group_key: e.id.clone(),
            licence: String::new(),
            context: e.text.clone(),
            question: String::new(),
            slot_name: String::new(),
            options: Vec::new(),
            gold: Gold::Noul,
            label_basis: "",
            split: "train",
        })
        .collect();
    let tool = json!({"tool": "qd-train-data"});
    let scanned = match pool::scan(
        inputs.n,
        inputs.threshold,
        &tool,
        "train-examples",
        &candidates,
        &usable,
        inputs.threads,
    ) {
        Ok(s) => s,
        Err(reason) => return Tier::not_run(name, reason),
    };
    let kept: HashSet<&str> = scanned.clean.iter().map(|c| c.id.as_str()).collect();
    let hit: Vec<String> = candidates
        .iter()
        .filter(|c| !kept.contains(c.id.as_str()))
        .map(|c| c.id.clone())
        .collect();
    let control = scanned.report["positive_control"].clone();
    Tier::ran(
        name,
        hit.len() as u64,
        hit.clone(),
        format!(
            "{} of {} train rows contain a target at word {}-gram containment >= {}; by target {}; \
             positive control {control}",
            hit.len(),
            candidates.len(),
            inputs.n,
            inputs.threshold,
            scanned.report["excluded_by_target"]
        ),
    )
}

/// Ruling 12 (owner, 2026-10-20): real Vani rows never leave the Mac. Rows of the `vani-agreed`
/// source in a pool whose run is not stated to train on the Mac are a hit. The run's target is
/// the command line's (`--run-target`) and the manifest's own `run_target`; either saying `cloud`
/// is a cloud run, and an unstated target is read as not-Mac. Rows that cannot be read make the
/// tier `NOT_RUN`, never clean.
fn tier_mac_only(
    entries: Option<&[Entry]>,
    why: &str,
    flag: Option<RunTarget>,
    manifest: Option<&str>,
) -> Tier {
    let name = "mac_only";
    let Some(entries) = entries else {
        return Tier::not_run(name, why.to_owned());
    };
    let declared = match manifest.map(RunTarget::parse).transpose() {
        Ok(t) => t,
        Err(e) => return Tier::not_run(name, format!("the train manifest's run_target: {e}")),
    };
    let target = match (flag, declared) {
        (Some(RunTarget::Cloud), _) | (_, Some(RunTarget::Cloud)) => Some(RunTarget::Cloud),
        (Some(RunTarget::Mac), _) | (_, Some(RunTarget::Mac)) => Some(RunTarget::Mac),
        (None, None) => None,
    };
    let hit: Vec<&Entry> = entries.iter().filter(|e| e.vani_agreed).collect();
    match check_run_target(hit.len() as u64, target) {
        Ok(detail) => Tier::ran(name, 0, Vec::new(), detail),
        Err(detail) => Tier::ran(
            name,
            hit.len() as u64,
            hit.iter().map(|e| e.id.clone()).collect(),
            detail,
        ),
    }
}

/// What the exact tier left as a residual: rows in which a short target is one line inside a
/// longer text (not a hit under the owner's whole-text rule, 2026-10-19), and how many distinct
/// short targets those rows hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Embedded {
    pub rows: u64,
    pub targets: u64,
}

/// The exact-short tier, over the strip-v2 text of the train rows (`--examples`: the stripped
/// export, e.g. `tools/train_examples_export.py`'s `stripped.jsonl`). One matcher with the
/// builder's containment scan: [`qd_prep::exact_short`]. A row is a hit when its WHOLE text,
/// normalised, equals a short target; an embedded match is only counted ([`Embedded`]).
fn tier_exact_short(
    examples: Option<&[pool::ExampleText]>,
    why: &str,
    targets: &[TargetSet],
    n: usize,
) -> (Tier, Embedded) {
    let name = "exact_short";
    let Some(examples) = examples else {
        return (Tier::not_run(name, why.to_owned()), Embedded::default());
    };
    let flat: Vec<&str> = targets
        .iter()
        .flat_map(|(_, rows)| rows.iter().map(|(_, text)| text.as_str()))
        .collect();
    let short = ShortTargets::build(flat.iter().copied().enumerate(), n);
    if short.is_empty() {
        return (
            Tier::ran(
                name,
                0,
                Vec::new(),
                format!("no target is shorter than {n} words"),
            ),
            Embedded::default(),
        );
    }
    // The control: each (bounded) distinct short target planted as the whole text of a synthetic
    // example must be matched, or a zero here would say nothing about the train set.
    let keys: Vec<&str> = short.iter().map(|(k, _)| k).collect();
    let planted: Vec<&str> = if keys.len() <= EXACT_CONTROL_MAX {
        keys
    } else {
        (0..EXACT_CONTROL_MAX)
            .map(|k| keys[k * keys.len() / EXACT_CONTROL_MAX])
            .collect()
    };
    let missed: Vec<&&str> = planted
        .iter()
        .filter(|k| short.whole(k).is_none())
        .take(SAMPLE_IDS)
        .collect();
    if !missed.is_empty() {
        return (
            Tier::not_run(
                name,
                format!(
                    "positive control: planted short targets were not matched, first {missed:?}"
                ),
            ),
            Embedded::default(),
        );
    }
    let mut hit_ids = Vec::new();
    let mut targets_hit: BTreeSet<&str> = BTreeSet::new();
    let mut embedded = Embedded::default();
    let mut embedded_targets: BTreeSet<&str> = BTreeSet::new();
    for e in examples {
        if short.whole(&e.text).is_some() {
            hit_ids.push(e.id.clone());
            targets_hit.insert(&e.id);
            continue;
        }
        let inside = short.embedded(&e.text);
        if !inside.is_empty() {
            embedded.rows += 1;
            embedded_targets.extend(inside);
        }
    }
    embedded.targets = embedded_targets.len() as u64;
    (
        Tier::ran(
            name,
            hit_ids.len() as u64,
            hit_ids.clone(),
            format!(
                "{} of {} train rows have a whole stripped text equal to a short target \
                 (strip-v2 text, whole-text rule); positive control planted {} of {} distinct \
                 short targets; residual, not hits: {} rows hold a short target ({} distinct) as \
                 one line inside a longer text",
                hit_ids.len(),
                examples.len(),
                planted.len(),
                short.distinct(),
                embedded.rows,
                embedded.targets
            ),
        ),
        embedded,
    )
}

/// Run every tier and combine them. Inputs that cannot be read make every tier that needs them
/// `NOT_RUN` with the reason; a target set that cannot be read makes the whole verdict `NOT_RUN`.
pub fn check(inputs: &Inputs) -> Verdict {
    let n = inputs.n as usize;
    let mut target_sha256 = BTreeMap::new();
    let mut targets: Vec<TargetSet> = Vec::new();
    for (name, path) in &inputs.targets {
        match pool::read_target_file(name, path) {
            Ok((set, lines)) => {
                target_sha256.insert(name.clone(), lines.sha256.clone());
                targets.push(set);
            }
            Err(reason) => {
                return not_run_verdict(format!("cannot read target set {name}: {reason}"));
            }
        }
    }
    if targets.is_empty() {
        return not_run_verdict(
            "no target set given: a check against nothing is not a check".into(),
        );
    }
    let is_dir = inputs.train.is_dir();
    let mut entries: Option<Vec<Entry>> = None;
    let mut entries_why = String::new();
    let mut manifest_target: Option<String> = None;
    let mut examples: Option<Vec<pool::ExampleText>> = None;
    let mut text_why = String::new();
    let examples_path = if is_dir {
        inputs.examples.as_deref()
    } else {
        Some(inputs.train.as_path())
    };
    match examples_path {
        Some(path) => match pool::read_example_texts(path) {
            Ok((rows, _)) => examples = Some(rows),
            Err(reason) => text_why = format!("cannot read the examples: {reason}"),
        },
        None => {
            text_why = "a shard set holds token ids, not text: pass --examples (the pool's \
                        examples.jsonl) to run the text tiers"
                .into();
        }
    }
    if is_dir {
        match read_manifest(&inputs.train, inputs.manifest.as_deref()) {
            Ok((e, declared)) => {
                entries = Some(e);
                manifest_target = declared;
            }
            Err(reason) => entries_why = format!("cannot read the train manifest: {reason}"),
        }
    } else if let Some(rows) = &examples {
        entries = Some(
            rows.iter()
                .map(|r| Entry {
                    vani_agreed: is_vani_agreed(&r.source_id),
                    id: r.id.clone(),
                    source_id: r.source_id.clone(),
                    family_id: r.family_id.clone(),
                })
                .collect(),
        );
    } else {
        entries_why = text_why.clone();
    }

    let mut tiers = vec![tier_forbidden_family(
        entries.as_deref(),
        &entries_why,
        &inputs.forbid_families,
    )];
    tiers.push(
        tier_clinc_digest(entries.as_deref(), &entries_why, &targets)
            .unwrap_or_else(|reason| Tier::not_run("clinc_digest", reason)),
    );
    tiers.push(tier_ngram(examples.as_deref(), &text_why, &targets, inputs));
    let (exact_tier, embedded) = tier_exact_short(examples.as_deref(), &text_why, &targets, n);
    tiers.push(exact_tier);
    tiers.push(tier_mac_only(
        entries.as_deref(),
        &entries_why,
        inputs.run_target,
        manifest_target.as_deref(),
    ));

    let exact_ran = tiers[3].state.did_run();
    let mut coverage = Coverage {
        target_sets: targets.len(),
        short_embedded_rows: embedded.rows,
        short_embedded_targets: embedded.targets,
        ..Coverage::default()
    };
    for (_, rows) in &targets {
        for (_, text) in rows {
            coverage.targets += 1;
            match word_count(text) {
                0 => coverage.no_words += 1,
                w if w < n => coverage.short += 1,
                _ => coverage.ngram_scannable += 1,
            }
        }
    }
    coverage.unscanned = coverage.no_words
        + if exact_ran { 0 } else { coverage.short }
        + if tiers[2].state.did_run() {
            0
        } else {
            coverage.ngram_scannable
        };

    let parts: Vec<(&str, TriState)> = tiers.iter().map(|t| (t.name, t.state.clone())).collect();
    let mut overall = aggregate("qd-train-data", &parts);
    if overall.is_pass() && coverage.unscanned > 0 {
        overall = TriState::not_run(format!(
            "every tier ran clean, but {} of {} targets could not be scanned by any of them",
            coverage.unscanned, coverage.targets
        ));
    }
    Verdict {
        tiers,
        coverage,
        overall,
        target_sha256,
    }
}

fn not_run_verdict(reason: String) -> Verdict {
    Verdict {
        tiers: Vec::new(),
        coverage: Coverage::default(),
        overall: TriState::not_run(reason),
        target_sha256: BTreeMap::new(),
    }
}

/// Parse `NAME=FILE`.
pub fn parse_target(spec: &str) -> Result<(String, PathBuf), String> {
    match spec.split_once('=') {
        Some((name, file)) if !name.is_empty() && !file.is_empty() => {
            Ok((name.to_owned(), PathBuf::from(file)))
        }
        _ => Err(format!("--target {spec:?}: expected NAME=FILE")),
    }
}

/// Counts per tier for the printed report, in tier order.
pub fn summary_lines(v: &Verdict) -> Vec<String> {
    let mut out = Vec::new();
    for t in &v.tiers {
        let line = match &t.state {
            TriState::Ran { passed, detail, .. } => format!(
                "tier {}: {} hits={} {}{}",
                t.name,
                if *passed { "PASS" } else { "FAILED" },
                t.hits,
                if t.sample_ids.is_empty() {
                    String::new()
                } else {
                    format!("sample={:?} ", t.sample_ids)
                },
                detail
            ),
            TriState::NotRun { reason } => format!("tier {}: NOT_RUN {reason}", t.name),
        };
        out.push(line);
    }
    let c = &v.coverage;
    out.push(format!(
        "coverage: {} targets in {} sets; {} scannable by n-gram, {} shorter than n (exact tier \
         only), {} with no words; {} could not be scanned by any tier that ran",
        c.targets, c.target_sets, c.ngram_scannable, c.short, c.no_words, c.unscanned
    ));
    out.push(format!(
        "residual (not hits): {} train rows hold a short target ({} distinct) as one line inside \
         a longer text; the exact tier matches a row's whole stripped text only",
        c.short_embedded_rows, c.short_embedded_targets
    ));
    if let TriState::NotRun { reason } = &v.overall {
        out.push(format!("not run: {reason}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qd-train-data-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    const LONG: &str =
        "the quick brown fox jumps over the lazy dog and then runs far away into the forest";
    const OTHER: &str = "a completely different sentence about compilers parsers and type checkers that \
                         shares nothing with the other one at all here today";

    fn write_targets(dir: &Path) -> PathBuf {
        let path = dir.join("targets.jsonl");
        let rows = [
            json!({"id": "t-long", "text": LONG}),
            json!({"id": "t-short", "text": "Set a Timer for ten minutes"}),
            json!({"id": "t-other", "text": OTHER}),
        ];
        let body: Vec<String> = rows.iter().map(Value::to_string).collect();
        fs::write(&path, body.join("\n") + "\n").expect("targets");
        path
    }

    fn write_examples(dir: &Path, rows: &[(&str, &str, &str, &str)]) -> PathBuf {
        write_examples_with(dir, rows, &["A", "B"])
    }

    /// Examples whose text is exactly `context` (no option lines): the stripped export's shape.
    fn write_bare_examples(dir: &Path, rows: &[(&str, &str, &str, &str)]) -> PathBuf {
        write_examples_with(dir, rows, &[])
    }

    fn write_examples_with(
        dir: &Path,
        rows: &[(&str, &str, &str, &str)],
        options: &[&str],
    ) -> PathBuf {
        let path = dir.join("examples.jsonl");
        let body: Vec<String> = rows
            .iter()
            .map(|(id, source, family, context)| {
                json!({"id": id, "source_id": source, "family_id": family,
                       "context": context, "options": options})
                .to_string()
            })
            .collect();
        fs::write(&path, body.join("\n") + "\n").expect("examples");
        path
    }

    fn inputs(train: PathBuf, targets: PathBuf) -> Inputs {
        Inputs {
            train,
            examples: None,
            manifest: None,
            targets: vec![("bench".into(), targets)],
            forbid_families: FORBIDDEN_FAMILIES.iter().map(|s| (*s).to_owned()).collect(),
            n: DEFAULT_N,
            threshold: DEFAULT_THRESHOLD,
            threads: 2,
        }
    }

    fn tier<'a>(v: &'a Verdict, name: &str) -> &'a Tier {
        v.tiers
            .iter()
            .find(|t| t.name == name)
            .expect("tier present")
    }

    #[test]
    fn a_clean_examples_file_passes_only_when_every_tier_ran() {
        let dir = tmp("clean");
        let ex = write_examples(
            &dir,
            &[(
                "r1",
                "x/y",
                "fam.a",
                "nothing like any benchmark item in here at all",
            )],
        );
        let v = check(&inputs(ex, write_targets(&dir)));
        assert!(v.overall.is_pass(), "{:?}", v.overall);
        assert_eq!(v.exit_code(), 0);
        assert_eq!(v.coverage.unscanned, 0);
        assert_eq!(v.coverage.short, 1);
    }

    #[test]
    fn a_copied_long_target_fails_and_names_the_row() {
        let dir = tmp("long");
        let ex = write_examples(
            &dir,
            &[
                (
                    "r1",
                    "x/y",
                    "fam.a",
                    "unrelated words only here for the first row of the file",
                ),
                ("r2", "x/y", "fam.a", LONG),
            ],
        );
        let v = check(&inputs(ex, write_targets(&dir)));
        assert_eq!(v.exit_code(), EXIT_FAILED);
        assert_eq!(tier(&v, "ngram").sample_ids, vec!["r2".to_owned()]);
        assert_eq!(tier(&v, "ngram").hits, 1);
    }

    /// The 8-gram scan cannot see a target under 8 words; the exact tier must, on the WHOLE
    /// stripped text of a row (owner ruling 2026-10-19). This one pins that a whole-text row is
    /// still a hit; the next pins what stopped being one.
    #[test]
    fn a_row_whose_whole_stripped_text_is_a_short_target_is_caught_by_the_exact_tier() {
        let dir = tmp("short");
        let ex = write_bare_examples(
            &dir,
            &[("r1", "x/y", "fam.a", "Set a timer for TEN minutes!")],
        );
        let v = check(&inputs(ex, write_targets(&dir)));
        assert_eq!(v.exit_code(), EXIT_FAILED);
        assert_eq!(tier(&v, "ngram").hits, 0, "the n-gram scan is blind to it");
        assert_eq!(tier(&v, "exact_short").sample_ids, vec!["r1".to_owned()]);
        assert_eq!(v.coverage.short_embedded_rows, 0);
    }

    /// The owner's ruling of 2026-10-19 (`human-ruling-2026-10-19.md`, "Strip, whole-text
    /// match"): a short target equal to one line inside a longer text is NOT a hit. It is a
    /// counted residual in the coverage output. The per-line rule this replaces called it a hit
    /// (the test that stood here planted exactly this text and expected `EXIT_FAILED`).
    #[test]
    fn a_short_target_inside_a_longer_text_is_a_counted_residual_not_a_hit() {
        let dir = tmp("embedded");
        let ex = write_bare_examples(
            &dir,
            &[
                (
                    "r1",
                    "x/y",
                    "fam.a",
                    "which intent is this\n\nSet a timer for TEN minutes!",
                ),
                (
                    "r2",
                    "x/y",
                    "fam.a",
                    "no short target is anywhere in this row",
                ),
            ],
        );
        let v = check(&inputs(ex, write_targets(&dir)));
        assert_eq!(
            tier(&v, "exact_short").hits,
            0,
            "{:?}",
            tier(&v, "exact_short").state
        );
        assert_eq!(v.exit_code(), 0, "{:?}", v.overall);
        assert_eq!(v.coverage.short_embedded_rows, 1);
        assert_eq!(v.coverage.short_embedded_targets, 1);
        let json = v.to_json();
        assert_eq!(json["coverage"]["short_embedded_rows"], 1);
        assert!(
            summary_lines(&v)
                .iter()
                .any(|l| l.contains("residual (not hits): 1 train rows"))
        );
    }

    #[test]
    fn a_shard_set_without_text_is_not_run_not_pass() {
        let dir = tmp("shards");
        let set = dir.join("shards").join("train");
        fs::create_dir_all(&set).expect("set");
        fs::write(set.join("header.json"), r#"{"data_snapshot_hash": "abc"}"#).expect("header");
        let pool_dir = dir.join("data").join("pool");
        fs::create_dir_all(&pool_dir).expect("pool");
        fs::write(
            pool_dir.join("train.json"),
            json!({"data_snapshot_hash": "abc", "n_rows": 1, "entries": [
                {"row_id": "pool:1", "source_id": "x/y", "family_id": "fam.a"}]})
            .to_string(),
        )
        .expect("manifest");
        let v = check(&inputs(set, write_targets(&dir)));
        assert!(!v.overall.is_pass());
        assert_eq!(v.exit_code(), EXIT_NOT_RUN, "{:?}", v.overall);
        assert!(!tier(&v, "ngram").state.did_run());
        assert!(v.coverage.unscanned > 0);
    }

    #[test]
    fn a_shard_set_manifest_with_benchmark_rows_fails_without_any_text() {
        let dir = tmp("v5r");
        let set = dir.join("shards").join("train");
        fs::create_dir_all(&set).expect("set");
        fs::write(set.join("header.json"), r#"{"data_snapshot_hash": "abc"}"#).expect("header");
        let pool_dir = dir.join("data").join("pool");
        fs::create_dir_all(&pool_dir).expect("pool");
        let digest = clinc_digest("  Set a Timer for ten minutes ").expect("digest");
        fs::write(
            pool_dir.join("train.json"),
            json!({"data_snapshot_hash": "abc", "n_rows": 3, "entries": [
                {"row_id": "mmlu:knowledge.multiple_choice:ff:1", "source_id": "cais/mmlu",
                 "family_id": "knowledge.multiple_choice"},
                {"row_id": format!("clinc:intent.classification:{digest}:2"),
                 "source_id": CLINC_SOURCE, "family_id": "intent.classification"},
                {"row_id": "clinc:intent.classification:0000000000000000:3",
                 "source_id": CLINC_SOURCE, "family_id": "intent.classification"}]})
            .to_string(),
        )
        .expect("manifest");
        let v = check(&inputs(set, write_targets(&dir)));
        assert_eq!(v.exit_code(), EXIT_FAILED);
        assert_eq!(tier(&v, "forbidden_family").hits, 1);
        assert_eq!(tier(&v, "clinc_digest").hits, 1);
    }

    #[test]
    fn a_manifest_that_is_not_the_sets_is_not_run() {
        let dir = tmp("foreign");
        let set = dir.join("shards").join("train");
        fs::create_dir_all(&set).expect("set");
        fs::write(set.join("header.json"), r#"{"data_snapshot_hash": "abc"}"#).expect("header");
        let pool_dir = dir.join("data").join("pool");
        fs::create_dir_all(&pool_dir).expect("pool");
        fs::write(
            pool_dir.join("train.json"),
            json!({"data_snapshot_hash": "zzz", "n_rows": 0, "entries": []}).to_string(),
        )
        .expect("manifest");
        let v = check(&inputs(set, write_targets(&dir)));
        assert!(!tier(&v, "forbidden_family").state.did_run());
        assert_eq!(v.exit_code(), EXIT_NOT_RUN);
    }

    #[test]
    fn an_unreadable_target_set_is_not_run() {
        let dir = tmp("missing");
        let ex = write_examples(&dir, &[("r1", "x/y", "fam.a", "words words words")]);
        let v = check(&inputs(ex, dir.join("absent.jsonl")));
        assert_eq!(v.exit_code(), EXIT_NOT_RUN);
    }

    #[test]
    fn clinc_digest_matches_python_blake2b_8_of_the_stripped_utterance() {
        // python3 -c "import hashlib; print(hashlib.blake2b(b'set a timer', digest_size=8).hexdigest())"
        assert_eq!(
            clinc_digest(" set a timer ").expect("digest"),
            "eb804189f137d467"
        );
    }
}
