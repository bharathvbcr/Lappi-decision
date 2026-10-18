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
//!   CLINC's 5,500 test items, `GAP-CLINC-TEST-TARGETS-8GRAM-BLIND-2026-10-06`). They are matched
//!   as whole normalised lines of an example instead, with its own planted-copy control. Needs text.
//!
//! What this does not establish: a *paraphrase* of a short target, or one embedded inside a longer
//! line, is not matched. The coverage block counts the short targets so that residual is a number.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use qd_prep::blake2b::Keyed;
use qd_prep::decisions::{Candidate, Gold, TargetSet};
use qd_prep::pool;
use qd_prep::pyunicode;
use serde::Deserialize;
use serde_json::{Value, json};

pub use crate::tristate::{TriState, aggregate};

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
}

#[derive(Deserialize)]
struct ManifestEntry {
    row_id: String,
    source_id: String,
    family_id: String,
}

#[derive(Deserialize)]
struct Manifest {
    data_snapshot_hash: String,
    n_rows: u64,
    entries: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct Header {
    data_snapshot_hash: String,
}

/// The train manifest of a shard set and whether its hash is the header's. Every refusal is a
/// reason string: the manifest-level tiers are then `NOT_RUN`, never silently clean.
fn read_manifest(set: &Path, explicit: Option<&Path>) -> Result<Vec<Entry>, String> {
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
    Ok(manifest
        .entries
        .into_iter()
        .map(|e| Entry {
            id: e.row_id,
            source_id: e.source_id,
            family_id: e.family_id,
        })
        .collect())
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

fn normalised(text: &str) -> String {
    let lowered = pyunicode::lower(text);
    pyunicode::words(&lowered).join(" ")
}

fn word_count(text: &str) -> usize {
    let lowered = pyunicode::lower(text);
    pyunicode::words(&lowered).len()
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

/// Normalised short targets (1 to n-1 words) by their text, with the target they name.
fn short_targets(targets: &[TargetSet], n: usize) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, rows) in targets {
        for (id, text) in rows {
            let norm = normalised(text);
            let words = norm.split(' ').filter(|w| !w.is_empty()).count();
            if (1..n).contains(&words) {
                out.entry(norm).or_default().push(format!("{name}:{id}"));
            }
        }
    }
    out
}

/// The normalised short targets that are a whole line (or the whole text) of `text`.
fn exact_matches<'a>(text: &str, short: &'a BTreeMap<String, Vec<String>>) -> Vec<&'a str> {
    let mut found: BTreeSet<&str> = BTreeSet::new();
    let mut probe = |segment: &str| {
        if let Some((key, _)) = short.get_key_value(&normalised(segment)) {
            found.insert(key.as_str());
        }
    };
    probe(text);
    for line in text.lines() {
        probe(line);
    }
    found.into_iter().collect()
}

fn tier_exact_short(
    examples: Option<&[pool::ExampleText]>,
    why: &str,
    targets: &[TargetSet],
    n: usize,
) -> Tier {
    let name = "exact_short";
    let Some(examples) = examples else {
        return Tier::not_run(name, why.to_owned());
    };
    let short = short_targets(targets, n);
    if short.is_empty() {
        return Tier::ran(
            name,
            0,
            Vec::new(),
            format!("no target is shorter than {n} words"),
        );
    }
    // The control: each (bounded) short target planted as a line of a synthetic example must be
    // found by the matcher, or a zero here would say nothing about the train set.
    let keys: Vec<&String> = short.keys().collect();
    let planted: Vec<&String> = if keys.len() <= EXACT_CONTROL_MAX {
        keys
    } else {
        (0..EXACT_CONTROL_MAX)
            .map(|k| keys[k * keys.len() / EXACT_CONTROL_MAX])
            .collect()
    };
    let missed: Vec<&&String> = planted
        .iter()
        .filter(|k| {
            let probe = format!("some surrounding words before\n{k}\nand after it");
            !exact_matches(&probe, &short).contains(&k.as_str())
        })
        .take(SAMPLE_IDS)
        .collect();
    if !missed.is_empty() {
        return Tier::not_run(
            name,
            format!("positive control: planted short targets were not matched, first {missed:?}"),
        );
    }
    let mut hit_ids = Vec::new();
    let mut targets_hit: BTreeSet<&str> = BTreeSet::new();
    for e in examples {
        let found = exact_matches(&e.text, &short);
        if !found.is_empty() {
            hit_ids.push(e.id.clone());
            targets_hit.extend(found);
        }
    }
    Tier::ran(
        name,
        hit_ids.len() as u64,
        hit_ids.clone(),
        format!(
            "{} of {} train rows hold a short target ({} distinct) as a whole normalised line; \
             positive control planted {} of {} short targets",
            hit_ids.len(),
            examples.len(),
            targets_hit.len(),
            planted.len(),
            short.len()
        ),
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
            Ok(e) => entries = Some(e),
            Err(reason) => entries_why = format!("cannot read the train manifest: {reason}"),
        }
    } else if let Some(rows) = &examples {
        entries = Some(
            rows.iter()
                .map(|r| Entry {
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
    tiers.push(tier_exact_short(
        examples.as_deref(),
        &text_why,
        &targets,
        n,
    ));

    let exact_ran = tiers[3].state.did_run();
    let mut coverage = Coverage {
        target_sets: targets.len(),
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
        let path = dir.join("examples.jsonl");
        let body: Vec<String> = rows
            .iter()
            .map(|(id, source, family, context)| {
                json!({"id": id, "source_id": source, "family_id": family,
                       "context": context, "options": ["A", "B"]})
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

    /// The 8-gram scan cannot see a target under 8 words; the exact tier must.
    #[test]
    fn a_short_target_is_caught_by_the_exact_tier_not_the_ngram_scan() {
        let dir = tmp("short");
        let ex = write_examples(
            &dir,
            &[(
                "r1",
                "x/y",
                "fam.a",
                "which intent is this\n\nSet a timer for TEN minutes!",
            )],
        );
        let v = check(&inputs(ex, write_targets(&dir)));
        assert_eq!(v.exit_code(), EXIT_FAILED);
        assert_eq!(tier(&v, "ngram").hits, 0, "the n-gram scan is blind to it");
        assert_eq!(tier(&v, "exact_short").sample_ids, vec!["r1".to_owned()]);
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
