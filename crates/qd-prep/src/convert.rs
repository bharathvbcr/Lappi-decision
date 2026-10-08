//! `qd-prep convert`: the downloaded v6 datasets made into decision pools (lane 4, 2026-10-06;
//! Fable's ruling: Python is a thin view, every decision about a row is here).
//!
//! **Inputs and their trust chain.**
//! - The fetch record (`fetch-record-v6-2026-10-06.json`), pinned by sha256 in the config. Every
//!   entry names a file, its sha256 and its `use`: `train` or `target-only`.
//! - The view record `view_v6.py` writes beside the data. Each entry repeats the source's path,
//!   sha256 and `use`, which [`Views::load`] checks against the pinned fetch record, and names
//!   each JSONL view with its sha256 and row count. A view is read through
//!   [`decisions::for_lines`] and checked against that sha256 after the read.
//! - **A training reader only ever gets a `train` file** ([`Views::train_rows`] refuses anything
//!   else, and any path under `targets/`). Target files are read only by the decontamination
//!   path: [`Views::target_texts`] for the word n-gram containment scan
//!   ([`crate::pool::decontaminate`]) and [`Views::target_rows`] for the id-disjointness checks.
//!
//! **Every row** passes the licence gate ([`crate::convert_licence::gate`]); a refusal is counted
//! by reason, never dropped silently, and a dataset-level licence is stamped on every row of the
//! dataset. Options are shuffled per row with a seeded draw and the gold-position and gold-class
//! histograms are in the manifest. The split is by `group_key` with a seeded draw. Exact
//! duplicates (family, context, option set) are kept once, preferring the val copy; duplicates
//! that disagree on the gold are all dropped and counted. A pool over [`MAX_POOL_ROWS`] is
//! refused, not truncated: lower the config's caps.
//!
//! Pools: [`crate::convert_tools`] (When2Call, ToolACE), [`crate::convert_text`] (MNLI,
//! SciRepEval search), [`crate::convert_code`] (CodeSearchNet, and SWE-rebench's row filter, which
//! writes a filtered view, not a pool), [`crate::convert_inject`] (the injection-text corpus,
//! not a pool).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::convert_licence::{self, Verdict};
use crate::decisions::{self, Candidate, Gold, TargetSet};
use crate::heldout;
use crate::pool;
use crate::sha256::sha256_hex;

pub const CONFIG_SCHEMA: &str = "qd-convert-config/v1";
pub const VIEW_RECORD_SCHEMA: &str = "qd-view-record/v1";
/// `qd_data.decisions.MAX_POOL_ROWS`: the loader refuses a pool with more rows.
pub const MAX_POOL_ROWS: usize = 400_000;
/// Rows offered by one pool's sources, admitted or refused.
pub const MAX_OFFERED: usize = 8_000_000;
/// The fetch record's two uses.
pub const TRAIN: &str = "train";
pub const TARGET_ONLY: &str = "target-only";

/// The policy half: `data/convert/*.json`.
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub val_fraction: f64,
    pub ngram_n: u32,
    pub containment_threshold: f64,
    pub fetch_record_sha256: String,
    /// Per pool: its own parameters (caps, fractions, pins), read by the pool's module.
    pub pools: BTreeMap<String, Value>,
}

impl Config {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("config: {e}"))?;
        if decisions::get_str(&v, "schema", "config")? != CONFIG_SCHEMA {
            return Err(format!("config: schema is not {CONFIG_SCHEMA:?}"));
        }
        let unit_open = |x: f64, name: &str| -> Result<f64, String> {
            if x > 0.0 && x < 1.0 {
                Ok(x)
            } else {
                Err(format!("config: {name} {x} is outside (0, 1)"))
            }
        };
        let ngram_n = u32::try_from(decisions::get_u64(&v, "ngram_n", "config")?)
            .ok()
            .filter(|n| (1..=crate::containment::MAX_N).contains(n))
            .ok_or("config: ngram_n is outside 1..=64")?;
        let threshold = decisions::get_f64(&v, "containment_threshold", "config")?;
        if !(threshold > 0.0 && threshold <= 1.0) {
            return Err(format!(
                "config: containment_threshold {threshold} is outside (0, 1]"
            ));
        }
        let pools = decisions::get(&v, "pools", "config")?
            .as_object()
            .ok_or("config: \"pools\" is not an object")?
            .iter()
            .map(|(k, x)| (k.clone(), x.clone()))
            .collect();
        Ok(Config {
            seed: decisions::get_u64(&v, "seed", "config")?,
            val_fraction: unit_open(
                decisions::get_f64(&v, "val_fraction", "config")?,
                "val_fraction",
            )?,
            ngram_n,
            containment_threshold: threshold,
            fetch_record_sha256: decisions::get_str(&v, "fetch_record_sha256", "config")?
                .to_owned(),
            pools,
        })
    }

    /// The pool's own parameter object, or a refusal naming it.
    pub fn pool(&self, name: &str) -> Result<&Value, String> {
        self.pools
            .get(name)
            .ok_or_else(|| format!("config: no pools.{name}"))
    }
}

/// A fraction in (0, 1] from a pool's parameters.
pub fn fraction(p: &Value, key: &str, what: &str) -> Result<f64, String> {
    let x = decisions::get_f64(p, key, what)?;
    if x > 0.0 && x <= 1.0 {
        Ok(x)
    } else {
        Err(format!("{what}: {key} {x} is outside (0, 1]"))
    }
}

/// A count in 1..=max from a pool's parameters.
pub fn count(p: &Value, key: &str, what: &str, max: u64) -> Result<usize, String> {
    let n = decisions::get_u64(p, key, what)?;
    if (1..=max).contains(&n) {
        Ok(n as usize)
    } else {
        Err(format!("{what}: {key} {n} is outside 1..={max}"))
    }
}

/// One JSONL view of a fetched file.
#[derive(Clone, Debug)]
pub struct View {
    pub kind: String,
    pub path: PathBuf,
    pub sha256: String,
    pub rows: u64,
}

/// One fetched file, as the view record describes it and the fetch record confirms it.
#[derive(Clone, Debug)]
pub struct Entry {
    pub dataset: String,
    pub file: String,
    pub use_: String,
    pub source_path: PathBuf,
    pub source_sha256: String,
    pub views: Vec<View>,
}

/// The view record, checked against the pinned fetch record.
pub struct Views {
    pub entries: Vec<Entry>,
    /// `fetch_record` and `view_record` sha256s, for the manifest.
    pub digests: BTreeMap<String, String>,
    /// Entries an addendum moved away from training, never viewed: `"<dataset> <file>: <use>"`.
    pub not_viewed: Vec<String>,
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    crate::files::read_bounded(path, crate::files::MAX_RECORD_BYTES, "input")
}

/// Whether `path` lies under a `targets` directory: the fetch places every target-only file
/// there, so a training read of one is refused on its path as well as on its `use`.
fn under_targets(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == "targets")
}

impl Views {
    /// Read the view record and the fetch record it names. The fetch record must hash to
    /// `fetch_pin`; every view entry's dataset, file, source sha256 and `use` must match the
    /// fetch record's one entry for that file.
    pub fn load(view_record: &Path, fetch_pin: &str) -> Result<Self, String> {
        let vr_bytes = read(view_record)?;
        let vr: Value = serde_json::from_slice(&vr_bytes)
            .map_err(|e| format!("{}: {e}", view_record.display()))?;
        if decisions::get_str(&vr, "schema", "view record")? != VIEW_RECORD_SCHEMA {
            return Err(format!("view record: schema is not {VIEW_RECORD_SCHEMA:?}"));
        }
        if decisions::get_str(&vr, "fetch_record_sha256", "view record")? != fetch_pin {
            return Err(
                "view record: written from a fetch record other than the pinned one".into(),
            );
        }
        let fr_path = PathBuf::from(decisions::get_str(&vr, "fetch_record", "view record")?);
        let fr_bytes = read(&fr_path)?;
        let fr_sha = sha256_hex(&fr_bytes);
        decisions::check_pin(&fr_path, &fr_sha, fetch_pin, "the config")?;
        let fr: Value =
            serde_json::from_slice(&fr_bytes).map_err(|e| format!("{}: {e}", fr_path.display()))?;
        let mut fetched: BTreeMap<(&str, &str), (&str, &str, &str)> = BTreeMap::new();
        for e in fr.as_array().ok_or("fetch record: not a list")? {
            let key = (
                decisions::get_str(e, "dataset", "fetch record")?,
                decisions::get_str(e, "file", "fetch record")?,
            );
            let val = (
                decisions::get_str(e, "sha256", "fetch record")?,
                decisions::get_str(e, "use", "fetch record")?,
                decisions::get_str(e, "file_path", "fetch record")?,
            );
            if fetched.insert(key, val).is_some() {
                return Err(format!("fetch record: {} {} twice", key.0, key.1));
            }
        }
        let mut entries = Vec::new();
        let mut not_viewed = Vec::new();
        for e in decisions::get(&vr, "entries", "view record")?
            .as_array()
            .ok_or("view record: entries is not a list")?
        {
            let what = "view record entry";
            let dataset = decisions::get_str(e, "dataset", what)?;
            let file = decisions::get_str(e, "file", what)?;
            let (sha, use_, path) = fetched.get(&(dataset, file)).ok_or_else(|| {
                format!("view record: {dataset} {file} is not in the fetch record")
            })?;
            let source_sha = decisions::get_str(e, "source_sha256", what)?;
            let source_use = decisions::get_str(e, "use", what)?;
            let source_path = decisions::get_str(e, "source_path", what)?;
            // An entry a fetch-record addendum moved away from training (TSSB-3M to
            // `held-out-only`, 2026-10-06): its use and path differ from the record by design.
            // It must carry no view and the record's sha256; it is kept out of `entries`, so no
            // reader of either kind can reach it.
            if let Some(reason) = e.get("not_viewed") {
                let has_views = e
                    .get("views")
                    .and_then(Value::as_array)
                    .is_some_and(|v| !v.is_empty());
                if has_views
                    || source_sha != *sha
                    || source_use == TRAIN
                    || source_use == TARGET_ONLY
                {
                    return Err(format!(
                        "view record: {dataset} {file} is marked not_viewed ({reason}) but has \
                         views, a sha256 other than the record's, or use {source_use:?}"
                    ));
                }
                not_viewed.push(format!("{dataset} {file}: {source_use}"));
                continue;
            }
            if source_sha != *sha || source_use != *use_ || source_path != *path {
                return Err(format!(
                    "view record: {dataset} {file} says sha256 {source_sha}, use {source_use:?}, \
                     path {source_path}; the fetch record says {sha}, {use_:?}, {path}"
                ));
            }
            if *use_ != TRAIN && *use_ != TARGET_ONLY {
                return Err(format!("fetch record: {dataset} {file} has use {use_:?}"));
            }
            let mut views = Vec::new();
            for v in decisions::get(e, "views", what)?
                .as_array()
                .ok_or("views is not a list")?
            {
                views.push(View {
                    kind: decisions::get_str(v, "kind", what)?.to_owned(),
                    path: PathBuf::from(decisions::get_str(v, "path", what)?),
                    sha256: decisions::get_str(v, "sha256", what)?.to_owned(),
                    rows: decisions::get_u64(v, "rows", what)?,
                });
            }
            entries.push(Entry {
                dataset: dataset.to_owned(),
                file: file.to_owned(),
                use_: (*use_).to_owned(),
                source_path: PathBuf::from(source_path),
                source_sha256: source_sha.to_owned(),
                views,
            });
        }
        let mut digests = BTreeMap::new();
        digests.insert("fetch_record".to_owned(), fr_sha);
        digests.insert("view_record".to_owned(), sha256_hex(&vr_bytes));
        Ok(Views {
            entries,
            digests,
            not_viewed,
        })
    }

    fn entry(&self, dataset: &str, file: &str) -> Result<&Entry, String> {
        self.entries
            .iter()
            .find(|e| e.dataset == dataset && e.file == file)
            .ok_or_else(|| format!("view record: no entry for {dataset} {file}"))
    }

    fn view<'a>(e: &'a Entry, kind: &str) -> Result<&'a View, String> {
        e.views
            .iter()
            .find(|v| v.kind == kind)
            .ok_or_else(|| format!("view record: {} {} has no {kind} view", e.dataset, e.file))
    }

    /// The `rows` view of a `train` file: the only door a training reader has. Refuses any
    /// other `use`, and any source or view path under `targets/`.
    pub fn train_rows(&self, dataset: &str, file: &str) -> Result<&View, String> {
        let e = self.entry(dataset, file)?;
        if e.use_ != TRAIN {
            return Err(format!(
                "{dataset} {file}: use is {:?}; a training reader reads only `train` files \
                 (a target is read by the decontamination scan alone)",
                e.use_
            ));
        }
        let v = Self::view(e, "rows")?;
        if under_targets(&e.source_path) || under_targets(&v.path) {
            return Err(format!(
                "{dataset} {file}: a path under targets/ ({}); refused for training",
                v.path.display()
            ));
        }
        // Rule 3: a held-out file never reaches a training reader, whatever its recorded use.
        let what = format!("{dataset} {file}");
        heldout::refuse_training_input(&e.source_path, &format!("{what} source"))?;
        heldout::refuse_training_input(&v.path, &format!("{what} rows view"))?;
        Ok(v)
    }

    fn target_entry(&self, dataset: &str, file: &str) -> Result<&Entry, String> {
        let e = self.entry(dataset, file)?;
        if e.use_ != TARGET_ONLY {
            return Err(format!(
                "{dataset} {file}: use is {:?}, not {TARGET_ONLY:?}; a decontamination target \
                 must be an eval file",
                e.use_
            ));
        }
        Ok(e)
    }

    /// The full-row view of a target file, for an id-disjointness check.
    pub fn target_rows(&self, dataset: &str, file: &str) -> Result<TargetView<'_>, String> {
        Self::view(self.target_entry(dataset, file)?, "rows").map(TargetView)
    }

    /// The `{"id", "text"}` views of target files, as `pool::read_targets` reads them: the
    /// names and the pins (each view's recorded sha256) come from the view record.
    pub fn target_texts(
        &self,
        files: &[(&str, &str, &str)],
    ) -> Result<(Vec<TargetSet>, BTreeMap<String, String>), String> {
        let mut pins = BTreeMap::new();
        let mut given = Vec::new();
        for (name, dataset, file) in files {
            let v = Self::view(self.target_entry(dataset, file)?, "target-text")?;
            if pins.insert((*name).to_owned(), v.sha256.clone()).is_some() {
                return Err(format!("target set {name} named twice"));
            }
            given.push(((*name).to_owned(), v.path.clone()));
        }
        pool::read_targets(&pins, &given)
    }
}

/// Every JSON line of a source view, handed to `each`, then checked against the view's sha256.
/// Its rows can become training examples ([`decisions::Role::Source`]): one that cannot be
/// passed on is dropped and counted.
pub fn read_view(
    v: &View,
    each: impl FnMut(usize, Value) -> Result<(), String>,
) -> Result<decisions::Lines, String> {
    read_pinned(v, decisions::Role::Source, each)
}

/// [`read_view`] for a corpus where a NUL is the attack itself
/// ([`decisions::Role::SourceNulEncoded`]): a row holding one is kept with every NUL written as
/// [`decisions::NUL_PLACEHOLDER`], and counted.
pub fn read_view_nul_encoded(
    v: &View,
    each: impl FnMut(usize, Value) -> Result<(), String>,
) -> Result<decisions::Lines, String> {
    read_pinned(v, decisions::Role::SourceNulEncoded, each)
}

fn read_pinned(
    v: &View,
    role: decisions::Role,
    each: impl FnMut(usize, Value) -> Result<(), String>,
) -> Result<decisions::Lines, String> {
    let got = decisions::for_lines(&v.path, role, each)?;
    decisions::check_pin(&v.path, &got.sha256, &v.sha256, "the view record")?;
    Ok(got)
}

/// A target file's full-row view. Its rows are references, never dropped, so it is read only
/// through [`TargetView::read`] and cannot be handed to [`read_view`], which drops.
pub struct TargetView<'a>(&'a View);

impl TargetView<'_> {
    /// Every JSON line, handed to `each`, then checked against the view's sha256
    /// ([`decisions::Role::Reference`]).
    pub fn read(
        &self,
        each: impl FnMut(usize, Value) -> Result<(), String>,
    ) -> Result<decisions::Lines, String> {
        read_pinned(self.0, decisions::Role::Reference, each)
    }
}

/// A string field, or `None` when absent, null or not a string.
pub fn opt_str<'a>(r: &'a Value, key: &str) -> Option<&'a str> {
    r.get(key).and_then(Value::as_str)
}

/// The split of a group within a scope: `decisions`' `Ctx::split_of` rule (a seeded hash puts
/// `val_fraction` of the groups in val), cloned here because that one is private to it.
pub fn split_of(cfg: &Config, scope: &str, group: &str) -> &'static str {
    if decisions::unit_draw(cfg.seed, &["val", scope, group]) < cfg.val_fraction {
        "val"
    } else {
        "train"
    }
}

/// A row as a source module makes it, before the shuffle and the split.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: String,
    pub source_id: &'static str,
    pub family_id: &'static str,
    pub stratum: String,
    pub group_key: String,
    /// The licence as the source states it, for [`convert_licence::gate`]; `None` if it states
    /// none.
    pub licence: Option<String>,
    pub context: String,
    pub question: String,
    pub slot_name: &'static str,
    /// In the source's canonical order; shuffled per row.
    pub options: Vec<String>,
    pub gold: Gold,
    pub label_basis: &'static str,
}

/// What reading one source produced.
#[derive(Default, Debug)]
pub struct Tally {
    pub offered: usize,
    pub kept: usize,
    pub refused: BTreeMap<String, usize>,
    /// Kept rows by licence id.
    pub licence_admitted: BTreeMap<String, usize>,
    /// Kept rows whose licence is stated but not yet registered in Python.
    pub licence_pending: BTreeMap<String, usize>,
}

impl Tally {
    pub fn refuse(&mut self, reason: &str) {
        *self.refused.entry(reason.to_owned()).or_default() += 1;
    }

    pub fn json(&self) -> Value {
        json!({
            "offered": self.offered, "kept": self.kept, "refused": self.refused,
            "refused_total": self.refused.values().sum::<usize>(),
            "licence_admitted": self.licence_admitted, "licence_pending": self.licence_pending,
        })
    }
}

/// The rows a pool's sources offered, as candidates, and the account of every refusal.
pub struct Acc<'c> {
    pub cfg: &'c Config,
    pub rows: Vec<Candidate>,
    pub tallies: BTreeMap<&'static str, Tally>,
    offered: usize,
}

impl<'c> Acc<'c> {
    pub fn new(cfg: &'c Config) -> Self {
        Acc {
            cfg,
            rows: Vec::new(),
            tallies: BTreeMap::new(),
            offered: 0,
        }
    }

    /// Count a refusal of a row that never became a [`Row`] (a reader-level reason).
    pub fn refuse(&mut self, source: &'static str, reason: &str) -> Result<(), String> {
        self.bump(source)?;
        self.tallies.entry(source).or_default().refuse(reason);
        Ok(())
    }

    fn bump(&mut self, source: &'static str) -> Result<(), String> {
        self.offered += 1;
        if self.offered > MAX_OFFERED {
            return Err(format!("more than {MAX_OFFERED} rows offered to one pool"));
        }
        self.tallies.entry(source).or_default().offered += 1;
        Ok(())
    }

    /// Count one offered row, and keep it or count why not: the source's own refusal, the
    /// licence gate, the structural checks, and a code point the containment scan cannot read.
    /// The split is by `group_key` within `split_scope`; options are shuffled by the row id.
    pub fn offer(
        &mut self,
        source: &'static str,
        split_scope: &str,
        made: Result<Row, String>,
    ) -> Result<(), String> {
        self.bump(source)?;
        let r = match made {
            Ok(r) => r,
            Err(reason) => {
                self.tallies.entry(source).or_default().refuse(&reason);
                return Ok(());
            }
        };
        let verdict = match convert_licence::gate(r.licence.as_deref()) {
            Ok(v) => v,
            Err(reason) => {
                self.tallies.entry(source).or_default().refuse(&reason);
                return Ok(());
            }
        };
        let order: Vec<usize> = (0..r.options.len()).collect();
        let perm = crate::synth::Scope::new(self.cfg.seed, &r.id).shuffle("options", &order);
        let options: Vec<String> = perm.iter().map(|&i| r.options[i].clone()).collect();
        let gold = match r.gold {
            Gold::Option(g) => match perm.iter().position(|&i| i == g) {
                Some(p) => Gold::Option(p),
                None => {
                    self.tallies
                        .entry(source)
                        .or_default()
                        .refuse("gold_not_in_options");
                    return Ok(());
                }
            },
            Gold::Noul => Gold::Noul,
        };
        let c = Candidate {
            split: split_of(self.cfg, split_scope, &r.group_key),
            id: r.id,
            source_id: r.source_id,
            family_id: r.family_id.to_owned(),
            stratum: r.stratum,
            group_key: r.group_key,
            licence: verdict.id().to_owned(),
            context: r.context,
            question: r.question,
            slot_name: r.slot_name.to_owned(),
            options,
            gold,
            label_basis: r.label_basis,
        };
        let t = self.tallies.entry(source).or_default();
        if let Some(reason) = decisions::structural_refusal(&c) {
            t.refuse(reason);
            return Ok(());
        }
        let assigned = crate::pyunicode::check_assigned(&c.rendered_context()).is_ok()
            && c.options
                .iter()
                .all(|o| crate::pyunicode::check_assigned(o).is_ok());
        if !assigned {
            t.refuse("unassigned_code_point");
            return Ok(());
        }
        t.kept += 1;
        match verdict {
            Verdict::Admitted(id) => *t.licence_admitted.entry(id).or_default() += 1,
            Verdict::Pending(id) => *t.licence_pending.entry(id).or_default() += 1,
        }
        if self.rows.len() >= MAX_POOL_ROWS {
            return Err(format!(
                "more than {MAX_POOL_ROWS} rows kept: the loader refuses a pool that size; lower \
                 this pool's caps in the config (refused, never truncated)"
            ));
        }
        self.rows.push(c);
        Ok(())
    }
}

/// The gold's class: its option text, or `noul`.
pub fn gold_class(c: &Candidate) -> &str {
    match c.gold {
        Gold::Option(i) => &c.options[i],
        Gold::Noul => "noul",
    }
}

/// Drop duplicate ids (counted), then keep one row per exact content (family, rendered context,
/// option set), preferring the val copy; content whose copies disagree on the gold is dropped
/// whole. Returns the rows sorted by id and the counts.
pub fn dedupe(mut rows: Vec<Candidate>) -> (Vec<Candidate>, Value) {
    rows.sort_by(|a, b| a.id.cmp(&b.id).then(a.split.cmp(b.split)));
    let before = rows.len();
    rows.dedup_by(|b, a| a.id == b.id);
    let duplicate_ids = before - rows.len();
    let digest = |c: &Candidate| {
        let mut opts = c.options.clone();
        opts.sort();
        sha256_hex(
            format!(
                "{}\u{1f}{}\u{1f}{}",
                c.family_id,
                c.rendered_context(),
                opts.join("\u{1e}")
            )
            .as_bytes(),
        )
    };
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, c) in rows.iter().enumerate() {
        groups.entry(digest(c)).or_default().push(i);
    }
    let mut keep = vec![false; rows.len()];
    let (mut exact, mut conflicting) = (0usize, 0usize);
    for idx in groups.values() {
        let classes: BTreeSet<&str> = idx.iter().map(|&i| gold_class(&rows[i])).collect();
        if classes.len() > 1 {
            conflicting += idx.len();
            continue;
        }
        let pick = idx
            .iter()
            .copied()
            .find(|&i| rows[i].split == "val")
            .unwrap_or(idx[0]);
        keep[pick] = true;
        exact += idx.len() - 1;
    }
    let kept: Vec<Candidate> = rows
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c))
        .collect();
    (
        kept,
        json!({"duplicate_ids": duplicate_ids, "exact_duplicates": exact,
               "conflicting_gold_duplicates_dropped": conflicting}),
    )
}

/// Per family and split: rows by gold class. Shuffling hides a source whose gold never takes
/// one class (When2Call's train never has "answer directly"); this shows it.
pub fn gold_class_counts(rows: &[&Candidate]) -> Value {
    let mut out: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for c in rows {
        *out.entry(format!("{}/{}", c.family_id, c.split))
            .or_default()
            .entry(gold_class(c).to_owned())
            .or_default() += 1;
    }
    json!(out)
}

/// Target rows the containment scan cannot read (a code point unassigned in the tables'
/// Unicode version) are taken out of their set and named in the manifest; a scan against a row
/// it cannot tokenise would otherwise refuse the whole pool.
fn scannable_targets(targets: Vec<TargetSet>) -> (Vec<TargetSet>, Value) {
    let mut dropped = serde_json::Map::new();
    let kept = targets
        .into_iter()
        .map(|(name, rows)| {
            let (ok, bad): (Vec<_>, Vec<_>) = rows
                .into_iter()
                .partition(|(_, text)| crate::pyunicode::check_assigned(text).is_ok());
            if !bad.is_empty() {
                dropped.insert(
                    name.clone(),
                    json!({"rows": bad.len(), "ids": bad.iter().take(20).map(|(id, _)| id.clone()).collect::<Vec<_>>()}),
                );
            }
            (name, ok)
        })
        .collect();
    (kept, Value::Object(dropped))
}

/// What a pool module hands to [`build`] besides its rows.
pub struct PoolFacts {
    pub name: &'static str,
    /// Per source id: the licence note the manifest carries for every row of that source.
    pub licence_notes: BTreeMap<&'static str, String>,
    /// The pool's parameters as applied, and offered-vs-kept counts for each cap.
    pub caps: Value,
    /// Id-disjointness checks against target files, each with its own state.
    pub id_checks: Value,
    /// Anything else the manifest must say (the label rule, a parked part, an exposure note).
    pub notes: Value,
}

/// Dedupe, decontaminate against `targets`, and assemble the pool and its manifest.
pub fn build(
    acc: Acc<'_>,
    targets: Vec<TargetSet>,
    digests: BTreeMap<String, String>,
    facts: PoolFacts,
    threads: usize,
) -> Result<decisions::Built, String> {
    let cfg = acc.cfg;
    let tallies: serde_json::Map<String, Value> = acc
        .tallies
        .iter()
        .map(|(k, t)| ((*k).to_owned(), t.json()))
        .collect();
    let (rows, dedupe_report) = dedupe(acc.rows);
    if targets.is_empty() {
        return Err(format!(
            "pool {}: no target set; a scan against nothing is not a check",
            facts.name
        ));
    }
    let (targets, unscannable) = scannable_targets(targets);
    let scanned = pool::decontaminate(
        cfg.ngram_n,
        cfg.containment_threshold,
        &json!({"tool": "qd-prep convert", "pool": facts.name, "seed": cfg.seed}),
        "convert-candidates",
        &rows,
        &targets,
        threads,
    )?;
    let clean = &scanned.clean;
    let ex = pool::examples(clean)?;
    let mut yes_share = serde_json::Map::new();
    for split in ["train", "val"] {
        let side: Vec<&&Candidate> = clean.iter().filter(|c| c.split == split).collect();
        let yes = side.iter().filter(|c| gold_class(c) == "yes").count();
        if yes > 0 {
            yes_share.insert(split.to_owned(), json!(yes as f64 / side.len() as f64));
        }
    }
    let mut decon = scanned.report.clone();
    decon["target_rows_not_scannable"] = unscannable;
    let groups: BTreeSet<(&str, &str)> = clean
        .iter()
        .map(|c| (c.group_key.as_str(), c.split))
        .collect();
    let straddling = {
        let mut sides: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (g, s) in &groups {
            sides.entry(*g).or_default().insert(*s);
        }
        sides.values().filter(|s| s.len() > 1).count()
    };
    if straddling > 0 {
        return Err(format!(
            "pool {}: {straddling} group(s) on both sides of the split",
            facts.name
        ));
    }
    let manifest = json!({
        "schema": decisions::MANIFEST_SCHEMA, "mode": "build", "tool": "qd-prep convert",
        "tool_version": env!("CARGO_PKG_VERSION"), "pool": facts.name,
        "inputs": digests,
        "seed": cfg.seed, "val_fraction": cfg.val_fraction,
        "sources": tallies,
        "licence_notes": facts.licence_notes,
        "licence_pending_registration": convert_licence::PENDING.iter()
            .map(|(id, n)| ((*id).to_owned(), json!(n))).collect::<serde_json::Map<_, _>>(),
        "caps": facts.caps,
        "allocation": pool::allocation(false, "candidates: `caps` bounds what this pool converts \
                 per source, not what a mixture draws; the v6 allocation over every pool decides \
                 that, and the loader refuses this pool until then"),
        "dedupe": dedupe_report,
        "decontamination": decon,
        "id_checks": facts.id_checks,
        "selected": ex.selected,
        "noul_gold_selected": ex.noul,
        "gold_position": ex.gold_position,
        "val_without_train": ex.val_without_train,
        "gold_class": gold_class_counts(clean),
        "yes_share": yes_share,
        "notes": facts.notes,
        "examples": ex.rows, "examples_sha256": ex.sha256,
    });
    Ok(decisions::Built {
        summary: format!(
            "qd-prep convert {}: {} rows ({} excluded by the scan)",
            facts.name,
            ex.rows,
            rows.len() - clean.len()
        ),
        examples: ex.bytes,
        manifest,
        texts: None,
        containment: scanned.written,
    })
}

/// What the CLI passes.
pub struct Inputs {
    pub config: PathBuf,
    pub view_record: PathBuf,
    pub pool: String,
    /// CodeSearchNet's licence cache (`csn_licences.py`), pinned in the config.
    pub licence_cache: Option<PathBuf>,
}

/// The pools and outputs `qd-prep convert --pool` names.
pub const POOLS: [&str; 6] = [
    "tools",
    "mnli",
    "scirepeval",
    "csn",
    "swe-rebench-filter",
    "injections",
];

/// `qd-prep convert`: the config, the view record and one pool name in; `out_dir` out.
pub fn run(inputs: &Inputs, out_dir: &Path, threads: usize) -> Result<String, String> {
    if out_dir.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            out_dir.display()
        ));
    }
    heldout::refuse_training_input(&inputs.config, "--config")?;
    heldout::refuse_training_input(&inputs.view_record, "--view-record")?;
    if let Some(cache) = &inputs.licence_cache {
        heldout::refuse_training_input(cache, "--licence-cache")?;
    }
    let config_bytes = read(&inputs.config)?;
    let cfg = Config::parse(&config_bytes)?;
    let views = Views::load(&inputs.view_record, &cfg.fetch_record_sha256)?;
    let mut digests = views.digests.clone();
    digests.insert("config".to_owned(), sha256_hex(&config_bytes));
    let built = match inputs.pool.as_str() {
        "tools" => crate::convert_tools::build(&cfg, &views, digests, threads)?,
        "mnli" => crate::convert_text::build_mnli(&cfg, &views, digests, threads)?,
        "scirepeval" => crate::convert_text::build_scirepeval(&cfg, &views, digests, threads)?,
        "csn" => {
            let cache = inputs
                .licence_cache
                .as_deref()
                .ok_or("--licence-cache is required for csn")?;
            crate::convert_code::build_csn(&cfg, &views, cache, digests, threads)?
        }
        "swe-rebench-filter" => {
            return crate::convert_code::swe_rebench_filter(&cfg, &views, digests, out_dir);
        }
        "injections" => return crate::convert_inject::run(&cfg, &views, digests, out_dir),
        other => return Err(format!("--pool {other:?} is not one of {POOLS:?}")),
    };
    write_pool(out_dir, built, &views)
}

/// Writes a built pool, its manifest stating the entries an addendum moved out of reach, so a
/// source dropped from training is seen in every pool built from that view record.
fn write_pool(
    out_dir: &Path,
    mut built: decisions::Built,
    views: &Views,
) -> Result<String, String> {
    built.manifest["not_viewed"] = json!(views.not_viewed);
    decisions::write_out(out_dir, &built)?;
    Ok(format!("{} -> {}", built.summary, out_dir.display()))
}

/// Write `files` into a new directory, as `DIR.partial` renamed, so it appears whole or not at
/// all (the shape `decisions::write_out` gives a pool).
pub fn write_dir(out_dir: &Path, files: &[(&str, &[u8])]) -> Result<(), String> {
    crate::files::write_new_dir(out_dir, |partial| {
        files.iter().try_for_each(|(name, bytes)| {
            crate::files::write_synced_new(&partial.join(name), bytes)
        })
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn cfg() -> Config {
        Config {
            seed: 7,
            val_fraction: 0.2,
            ngram_n: 8,
            containment_threshold: 0.5,
            fetch_record_sha256: String::new(),
            pools: BTreeMap::new(),
        }
    }

    /// A scratch directory unique to the test.
    pub(crate) fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("qd-convert-{tag}-{}", std::process::id()));
        if d.exists() {
            std::fs::remove_dir_all(&d).unwrap();
        }
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Write `rows` as JSONL and return the [`View`] that names it.
    pub(crate) fn view_of(dir: &Path, name: &str, rows: &[Value]) -> View {
        let mut bytes = Vec::new();
        for r in rows {
            serde_json::to_writer(&mut bytes, r).unwrap();
            bytes.push(b'\n');
        }
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap();
        View {
            kind: "rows".into(),
            path,
            sha256: sha256_hex(&bytes),
            rows: rows.len() as u64,
        }
    }

    /// A fetch record and a view record over two files: one train, one under targets/.
    fn records(dir: &Path, lie_about_use: bool) -> (PathBuf, String) {
        let train_src = dir.join("ds__a/rev/train.parquet");
        let target_src = dir.join("targets/ds__a/rev/test.parquet");
        std::fs::create_dir_all(train_src.parent().unwrap()).unwrap();
        std::fs::create_dir_all(target_src.parent().unwrap()).unwrap();
        let train_view = view_of(
            train_src.parent().unwrap(),
            "train.jsonl",
            &[json!({"x": 1})],
        );
        let target_view = view_of(
            target_src.parent().unwrap(),
            "test.jsonl",
            &[json!({"x": 2})],
        );
        let fr = json!([
            {"dataset": "ds/a", "file": "train.parquet", "use": "train", "sha256": "aa",
             "file_path": train_src.display().to_string()},
            {"dataset": "ds/a", "file": "test.parquet", "use": "target-only", "sha256": "bb",
             "file_path": target_src.display().to_string()},
        ]);
        let fr_bytes = serde_json::to_vec(&fr).unwrap();
        let fr_path = dir.join("fetch.json");
        std::fs::write(&fr_path, &fr_bytes).unwrap();
        let fr_sha = sha256_hex(&fr_bytes);
        let v = |src: &Path, file: &str, sha: &str, use_: &str, view: &View| {
            json!({"dataset": "ds/a", "file": file, "use": use_, "source_sha256": sha,
                   "source_path": src.display().to_string(),
                   "views": [{"kind": "rows", "path": view.path.display().to_string(),
                              "sha256": view.sha256, "rows": view.rows}]})
        };
        let target_use = if lie_about_use {
            "train"
        } else {
            "target-only"
        };
        let vr = json!({"schema": VIEW_RECORD_SCHEMA, "fetch_record": fr_path.display().to_string(),
            "fetch_record_sha256": fr_sha, "entries": [
                v(&train_src, "train.parquet", "aa", "train", &train_view),
                v(&target_src, "test.parquet", "bb", target_use, &target_view)]});
        let vr_path = dir.join("view-record.json");
        std::fs::write(&vr_path, serde_json::to_vec(&vr).unwrap()).unwrap();
        (vr_path, fr_sha)
    }

    #[test]
    fn a_target_only_file_fed_to_a_training_reader_is_refused() {
        let dir = scratch("use");
        let (vr, pin) = records(&dir, false);
        let views = Views::load(&vr, &pin).unwrap();
        assert!(views.train_rows("ds/a", "train.parquet").is_ok());
        let err = views.train_rows("ds/a", "test.parquet").unwrap_err();
        assert!(err.contains("use is \"target-only\""), "{err}");
        // The decontamination path reads it; it refuses a train file as a target.
        assert!(views.target_rows("ds/a", "test.parquet").is_ok());
        assert!(views.target_rows("ds/a", "train.parquet").is_err());
    }

    #[test]
    fn a_view_record_that_relabels_a_target_as_train_is_refused() {
        let dir = scratch("lie");
        let (vr, pin) = records(&dir, true);
        let err = Views::load(&vr, &pin)
            .err()
            .expect("a relabelled use must be refused");
        assert!(err.contains("the fetch record says"), "{err}");
    }

    #[test]
    fn a_path_under_targets_is_refused_for_training_even_if_marked_train() {
        assert!(under_targets(Path::new("/x/v6/targets/ds/test.jsonl")));
        assert!(!under_targets(Path::new("/x/v6/views/ds/train.jsonl")));
        let e = Entry {
            dataset: "d".into(),
            file: "f".into(),
            use_: TRAIN.into(),
            source_path: PathBuf::from("/x/targets/d/f.parquet"),
            source_sha256: "s".into(),
            views: vec![View {
                kind: "rows".into(),
                path: PathBuf::from("/x/views/targets/d/f.jsonl"),
                sha256: "s".into(),
                rows: 1,
            }],
        };
        let views = Views {
            entries: vec![e],
            digests: BTreeMap::new(),
            not_viewed: Vec::new(),
        };
        assert!(views.train_rows("d", "f").unwrap_err().contains("targets/"));
    }

    /// A view record holding one train entry and one entry an addendum moved to held-out-only,
    /// recorded as `view_v6.py` records it (new path, amended use, no views).
    fn records_with_held_out(dir: &Path, held_out_views: bool, sha: &str) -> (PathBuf, String) {
        let train_src = dir.join("ds__a/rev/train.parquet");
        std::fs::create_dir_all(train_src.parent().unwrap()).unwrap();
        let train_view = view_of(
            train_src.parent().unwrap(),
            "train.jsonl",
            &[json!({"x": 1})],
        );
        let fr = json!([
            {"dataset": "ds/a", "file": "train.parquet", "use": "train", "sha256": "aa",
             "file_path": train_src.display().to_string()},
            {"dataset": "zenodo/TSSB-3M", "file": "tssb.zip", "use": "train", "sha256": "cc",
             "file_path": dir.join("zenodo__TSSB-3M/1/tssb.zip").display().to_string()},
        ]);
        let fr_bytes = serde_json::to_vec(&fr).unwrap();
        let fr_path = dir.join("fetch.json");
        std::fs::write(&fr_path, &fr_bytes).unwrap();
        let fr_sha = sha256_hex(&fr_bytes);
        let held_views = if held_out_views {
            json!([{"kind": "rows", "path": train_view.path.display().to_string(),
                    "sha256": train_view.sha256, "rows": 1}])
        } else {
            json!([])
        };
        let vr = json!({"schema": VIEW_RECORD_SCHEMA, "fetch_record": fr_path.display().to_string(),
            "fetch_record_sha256": fr_sha, "entries": [
                {"dataset": "ds/a", "file": "train.parquet", "use": "train", "source_sha256": "aa",
                 "source_path": train_src.display().to_string(),
                 "views": [{"kind": "rows", "path": train_view.path.display().to_string(),
                            "sha256": train_view.sha256, "rows": 1}]},
                {"dataset": "zenodo/TSSB-3M", "file": "tssb.zip", "use": "held-out-only",
                 "source_sha256": sha, "views": held_views,
                 "source_path": dir.join("heldout/tssb-3m/1/tssb.zip").display().to_string(),
                 "not_viewed": "held-out-only (addendum fetch.addendum-tssb-heldout.json)"}]});
        let vr_path = dir.join("view-record.json");
        std::fs::write(&vr_path, serde_json::to_vec(&vr).unwrap()).unwrap();
        (vr_path, fr_sha)
    }

    #[test]
    fn an_entry_moved_to_held_out_is_loaded_unreachable_and_counted() {
        let dir = scratch("heldout");
        let (vr, pin) = records_with_held_out(&dir, false, "cc");
        let views = Views::load(&vr, &pin).unwrap();
        assert!(views.train_rows("ds/a", "train.parquet").is_ok());
        assert!(views.train_rows("zenodo/TSSB-3M", "tssb.zip").is_err());
        assert!(views.target_rows("zenodo/TSSB-3M", "tssb.zip").is_err());
        assert_eq!(views.not_viewed, ["zenodo/TSSB-3M tssb.zip: held-out-only"]);
        // Fail-closed: a not-viewed entry with a view, or a sha256 other than the record's.
        let dir = scratch("heldout-views");
        let (vr, pin) = records_with_held_out(&dir, true, "cc");
        assert!(Views::load(&vr, &pin).is_err());
        let dir = scratch("heldout-sha");
        let (vr, pin) = records_with_held_out(&dir, false, "dd");
        assert!(Views::load(&vr, &pin).is_err());
    }

    #[test]
    fn a_pool_manifest_states_what_an_addendum_moved_out_of_reach() {
        let dir = scratch("heldout-manifest");
        let (vr, pin) = records_with_held_out(&dir, false, "cc");
        let views = Views::load(&vr, &pin).unwrap();
        let built = decisions::Built {
            examples: Vec::new(),
            manifest: json!({"pool": "p"}),
            texts: None,
            containment: crate::containment::Written {
                pairs_tsv: Vec::new(),
                exclusions: Vec::new(),
                attestation: Vec::new(),
                summary: String::new(),
            },
            summary: "p".to_owned(),
        };
        let out = dir.join("pool");
        write_pool(&out, built, &views).unwrap();
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            manifest["not_viewed"],
            json!(["zenodo/TSSB-3M tssb.zip: held-out-only"])
        );
    }

    /// Rule 3 at the training reader's door: a `train` entry whose source or view sits under a
    /// held-out marker is refused, so a mislabelled record cannot route held-out rows in.
    #[test]
    fn a_held_out_path_is_refused_for_training_even_if_marked_train() {
        let entry = |source: &str, view: &str| Entry {
            dataset: "d".into(),
            file: "f".into(),
            use_: TRAIN.into(),
            source_path: PathBuf::from(source),
            source_sha256: "s".into(),
            views: vec![View {
                kind: "rows".into(),
                path: PathBuf::from(view),
                sha256: "s".into(),
                rows: 1,
            }],
        };
        for (source, view) in [
            ("/x/heldout/tssb-3m/d.zip", "/x/views/d/f.jsonl"),
            ("/x/d/f.parquet", "/x/views/HeldOut/d/f.jsonl"),
        ] {
            let views = Views {
                entries: vec![entry(source, view)],
                digests: BTreeMap::new(),
                not_viewed: Vec::new(),
            };
            let err = views.train_rows("d", "f").unwrap_err();
            assert!(err.contains("held-out data"), "{source} {view}: {err}");
        }
        let views = Views {
            entries: vec![entry("/x/d/f.parquet", "/x/views/d/f.jsonl")],
            digests: BTreeMap::new(),
            not_viewed: Vec::new(),
        };
        assert!(views.train_rows("d", "f").is_ok());
    }

    #[test]
    fn the_shipped_config_parses_and_names_every_pool() {
        let c = Config::parse(include_bytes!(
            "../../../data/convert/convert-v6-2026-10-06.json"
        ))
        .unwrap();
        for p in POOLS {
            assert!(c.pool(p).is_ok(), "{p}");
        }
        assert_eq!(
            c.fetch_record_sha256,
            "72be4cdf4307d792edcfd3b46a4cf11fbf19850910569b70517c09e24221a702"
        );
    }

    #[test]
    fn a_fetch_record_other_than_the_pinned_one_is_refused() {
        let dir = scratch("pin");
        let (vr, _) = records(&dir, false);
        assert!(Views::load(&vr, &"0".repeat(64)).is_err());
    }

    #[test]
    fn a_view_whose_bytes_changed_is_refused_after_the_read() {
        let dir = scratch("viewsha");
        let mut v = view_of(&dir, "v.jsonl", &[json!({"a": 1})]);
        v.sha256 = "0".repeat(64);
        assert!(read_view(&v, |_, _| Ok(())).is_err());
    }

    pub(crate) fn row(id: &str, group: &str, gold: usize, licence: Option<&str>) -> Row {
        Row {
            id: id.into(),
            source_id: "t/src",
            family_id: "t.fam",
            stratum: "t.fam/x".into(),
            group_key: group.into(),
            licence: licence.map(str::to_owned),
            context: format!("context of {id} with enough words to scan"),
            question: "Which?".into(),
            slot_name: "answer",
            options: vec!["a".into(), "b".into(), "c".into()],
            gold: Gold::Option(gold),
            label_basis: "hard",
        }
    }

    #[test]
    fn an_unstated_licence_is_refused_and_counted_and_a_pending_one_is_kept_and_named() {
        let cfg = cfg();
        let mut acc = Acc::new(&cfg);
        acc.offer("t/src", "t.fam", Ok(row("r1", "g1", 0, None)))
            .unwrap();
        acc.offer("t/src", "t.fam", Ok(row("r2", "g2", 0, Some("odc-by-1.0"))))
            .unwrap();
        acc.offer("t/src", "t.fam", Ok(row("r3", "g3", 0, Some("cc-by-4.0"))))
            .unwrap();
        acc.offer("t/src", "t.fam", Ok(row("r4", "g4", 0, Some("gpl-3.0"))))
            .unwrap();
        let t = &acc.tallies["t/src"];
        assert_eq!(t.offered, 4);
        assert_eq!(t.kept, 2);
        assert_eq!(t.refused["licence_unstated"], 1);
        assert_eq!(t.refused["licence_needs_human_call:gpl-3.0"], 1);
        assert_eq!(t.licence_pending["odc-by-1.0"], 1);
        assert_eq!(t.licence_admitted["cc-by-4.0"], 1);
        let lic: Vec<&str> = acc.rows.iter().map(|c| c.licence.as_str()).collect();
        assert_eq!(lic, ["odc-by-1.0", "cc-by-4.0"]);
    }

    #[test]
    fn options_are_shuffled_per_row_and_the_gold_follows_its_option() {
        let cfg = cfg();
        let mut acc = Acc::new(&cfg);
        for i in 0..60 {
            acc.offer(
                "t/src",
                "t.fam",
                Ok(row(&format!("r{i}"), &format!("g{i}"), 0, Some("mit"))),
            )
            .unwrap();
        }
        let mut positions = BTreeSet::new();
        for c in &acc.rows {
            assert_eq!(gold_class(c), "a", "the gold must stay the option it was");
            if let Gold::Option(p) = c.gold {
                positions.insert(p);
            }
        }
        assert_eq!(
            positions.len(),
            3,
            "a source whose gold is always first must not stay first"
        );
    }

    #[test]
    fn the_option_order_is_synth_scope_s_draw_as_pinned() {
        // Pinned: the permutation the convert lane drew with its own copy of this shuffle
        // (Fisher-Yates, tag "options\u{1f}i", keyed % (i+1)) before it converged onto
        // synth::Scope, so the convergence moved no option.
        let mut v: Vec<usize> = (0..5).collect();
        for i in (1..5).rev() {
            let d = decisions::keyed(7, &["k", &format!("options\u{1f}{i}")]);
            let j = (u64::from_le_bytes(d[..8].try_into().unwrap()) % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
        let order: Vec<usize> = (0..5).collect();
        assert_eq!(
            crate::synth::Scope::new(7, "k").shuffle("options", &order),
            v
        );
    }

    #[test]
    fn groups_never_straddle_and_duplicates_disagreeing_on_gold_are_dropped() {
        let cfg = cfg();
        let mut acc = Acc::new(&cfg);
        for i in 0..40 {
            acc.offer(
                "t/src",
                "t.fam",
                Ok(row(
                    &format!("r{i}"),
                    &format!("g{}", i % 5),
                    0,
                    Some("mit"),
                )),
            )
            .unwrap();
        }
        let mut sides: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for c in &acc.rows {
            sides.entry(&c.group_key).or_default().insert(c.split);
        }
        assert!(sides.values().all(|s| s.len() == 1));
        let mut a = row("x1", "g", 0, Some("mit"));
        a.context = "same".into();
        let mut b = row("x2", "h", 1, Some("mit"));
        b.context = "same".into();
        let mut acc2 = Acc::new(&cfg);
        acc2.offer("t/src", "t.fam", Ok(a)).unwrap();
        acc2.offer("t/src", "t.fam", Ok(b)).unwrap();
        let (kept, report) = dedupe(acc2.rows);
        assert!(kept.is_empty());
        assert_eq!(report["conflicting_gold_duplicates_dropped"], 2);
    }

    #[test]
    fn a_pool_over_the_loader_s_row_bound_is_refused_not_truncated() {
        let cfg = cfg();
        let mut acc = Acc::new(&cfg);
        acc.rows = vec![
            Candidate {
                id: "x".into(),
                source_id: "t/src",
                family_id: "t.fam".into(),
                stratum: "t.fam/x".into(),
                group_key: "g".into(),
                licence: "mit".into(),
                context: "c".into(),
                question: "q".into(),
                slot_name: "answer".into(),
                options: vec!["a".into(), "b".into()],
                gold: Gold::Option(0),
                label_basis: "hard",
                split: "train",
            };
            MAX_POOL_ROWS
        ];
        let err = acc
            .offer("t/src", "t.fam", Ok(row("r", "g", 0, Some("mit"))))
            .unwrap_err();
        assert!(err.contains("refused, never truncated"));
    }

    #[test]
    fn a_row_with_an_unassigned_code_point_is_refused_and_counted() {
        let cfg = cfg();
        let mut acc = Acc::new(&cfg);
        let mut r = row("r", "g", 0, Some("mit"));
        r.context.push('\u{fffe}');
        acc.offer("t/src", "t.fam", Ok(r)).unwrap();
        assert_eq!(acc.tallies["t/src"].refused["unassigned_code_point"], 1);
    }
}
