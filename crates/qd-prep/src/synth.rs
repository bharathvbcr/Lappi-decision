//! `qd-prep synth`: decision rows synthesised by rule, for callers with no usable real data.
//!
//! - **Email sorter:** no labelled real-email corpus has a clean licence
//!   (AUDIT/data-licence-survey-2026-10-06.md, "Gaps with no clean source").
//! - **Jarvis:** no accessibility-tree element-selection set has a clean licence.
//! - **Privacy:** the human ruled on 2026-10-06 ("Synthetic only") that no real email, form or
//!   query enters training or eval. These generators read no mailbox, filter list or profile
//!   value.
//! - **Generator modules:** [`crate::synth_email`] and [`crate::synth_jarvis`]. Both write
//!   [`Draft`]s; this module turns them into the `qd-decisions/v1` pool that `qd-prep decisions`
//!   writes and `qd_data.decisions.load_decision_pool` reads.
//! - **Shared parts:** the pool reuses [`decisions::Candidate`], `example_json`, `write_out`,
//!   [`decisions::structural_refusal`] and the [`crate::containment`] scan.
//!
//! Guards against Lappi 0.1's mistakes, applied here and not left to each generator:
//!
//! - **Template-disjoint split.** Every draft names the template it was written from, and
//!   `group_key` is that template. Val holds whole templates, at least one per stratification
//!   class, so a val reading measures wording the model never trained on. A single-template
//!   class is refused ([`MIN_TEMPLATES_PER_CLASS`]). This is the generator form of v5's
//!   single-filler-template needle failure.
//! - **Label leak probe.** The 0.1 diff-marker free pass was a format leak, so each fixed-option
//!   family gets a linear control: char n-grams of the rendered context, fitted on train
//!   templates and scored on val templates. Its accuracy is in the manifest, and a build over
//!   the config's recorded bound is refused. Families whose options vary per row are reported
//!   `not_run` with the reason.
//! - **Option order.** Options are shuffled per row with a seeded draw (the PriDe lever), and the
//!   gold-position histogram is in the manifest. Per-class context lengths are there too, so a
//!   class cannot be told apart by length unnoticed.
//! - **Exact duplicates.** Rows with the same family, context and option set are kept once.
//!   The keep rule is split-aware (val over train), so a duplicate never removes a val row.
//! - **Decontamination.** Every row is scanned against the `--target` sets by the same word
//!   n-gram containment `qd-prep decisions` runs. Hits are excluded and counted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::containment;
use crate::decisions::{self, Candidate, Gold, TargetSet};
use crate::linfit::{self, Hyper, Selection};
use crate::ngram::{self, NGramSpec};
use crate::pool;
use crate::sha256::sha256_hex;

pub const CONFIG_SCHEMA: &str = "qd-synth-config/v1";
/// The manifest of an `--emit-heldout` directory.
pub const HELDOUT_SCHEMA: &str = "qd-synth-heldout/v1";
/// Not a licence any upstream granted: the rows are written by this crate's own rules. The
/// Python licence table refuses it (default deny) until it is registered there with the
/// human's 2026-10-06 ruling as its note, so the pool cannot enter a mixture unregistered.
pub const LICENCE: &str = "synthetic-by-rule";
pub const EMAIL_SOURCE: &str = "lappi/synth-email";
pub const JARVIS_SOURCE: &str = "lappi/synth-jarvis";
pub const TOOLS_SOURCE: &str = "lappi/synth-tools";
/// `qd_data.decisions.MAX_POOL_ROWS`: a pool past it is refused at load, so it is refused here.
pub const MAX_ROWS: usize = 400_000;
pub const MAX_ROWS_PER_TEMPLATE: usize = 10_000;
/// Fable's ruling (2026-10-06): at least four templates per label. One goes to val, so at
/// least three are left to train on.
pub const MIN_TEMPLATES_PER_CLASS: usize = 4;
/// The probe is a control, not a model: a few hundred full-batch steps on a small hashed space.
pub const MAX_PROBE_DIM: u32 = 1 << 20;
pub const MAX_PROBE_ITER: u32 = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Email,
    Jarvis,
    Tools,
}

impl Kind {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "email" => Ok(Self::Email),
            "jarvis" => Ok(Self::Jarvis),
            "tools" => Ok(Self::Tools),
            other => Err(format!(
                "config: kind {other:?} is not \"email\", \"jarvis\" or \"tools\""
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Jarvis => "jarvis",
            Self::Tools => "tools",
        }
    }

    pub fn source_id(self) -> &'static str {
        match self {
            Self::Email => EMAIL_SOURCE,
            Self::Jarvis => JARVIS_SOURCE,
            Self::Tools => TOOLS_SOURCE,
        }
    }
}

/// The linear leak probe's hyperparameters (`qd_train.baseline.LinearBaseline`'s), and the
/// bound its val accuracy may not exceed. The bound is recorded before any pool is built.
#[derive(Clone, Debug)]
pub struct Probe {
    pub n_min: u32,
    pub n_max: u32,
    pub dim: u32,
    pub max_iter: u32,
    pub tol: f64,
    pub lr: f64,
    pub l2_grid: Vec<f64>,
    pub max_val_accuracy: f64,
}

/// `data/synth/<name>.json`.
#[derive(Clone, Debug)]
pub struct Config {
    pub kind: Kind,
    pub seed: u64,
    pub rows_per_template: usize,
    /// Templates per stratification class that go to val.
    pub val_templates_per_class: usize,
    /// Templates per stratification class held out of the pool entirely (`heldout`, 0 = none):
    /// rendered only by [`emit_heldout`], as the kind's held-out evaluation set and the pool's
    /// decontamination target. They are drawn before val, so a config without the key splits
    /// exactly as before it existed.
    pub heldout_templates_per_class: usize,
    /// Email only: the share of rows that also get a label-flip injection variant.
    pub injection_rate: f64,
    pub ngram_n: u32,
    pub containment_threshold: f64,
    pub target_sha256: BTreeMap<String, String>,
    pub probe: Probe,
    /// Present iff `kind` is `tools`: the tool-selection generator's inputs and bounds.
    pub tools: Option<crate::synth_tools::ToolsConfig>,
}

impl Config {
    /// The config at `path` and its bytes. The tools config's data-file paths are relative to
    /// the config's own directory, so a committed config names no home path.
    pub fn load(path: &Path) -> Result<(Self, Vec<u8>), String> {
        crate::heldout::refuse_training_input(path, "--config")?;
        let bytes = crate::files::read_bounded(path, crate::files::MAX_RECORD_BYTES, "--config")?;
        let mut cfg = Self::parse(&bytes)?;
        if let Some(t) = cfg.tools.as_mut() {
            let dir = path.parent().unwrap_or(Path::new("."));
            t.catalog = dir.join(&t.catalog);
            t.phrasings = dir.join(&t.phrasings);
        }
        Ok((cfg, bytes))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        use decisions::{get, get_f64, get_str, get_u64, str_map};
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("config: {e}"))?;
        if get_str(&v, "schema", "config")? != CONFIG_SCHEMA {
            return Err(format!("config: schema is not {CONFIG_SCHEMA:?}"));
        }
        let p = get(&v, "probe", "config")?;
        let u32_in = |x: u64, name: &str, lo: u32, hi: u32| -> Result<u32, String> {
            u32::try_from(x)
                .ok()
                .filter(|n| (lo..=hi).contains(n))
                .ok_or_else(|| format!("config: {name} {x} is outside {lo}..={hi}"))
        };
        let l2_grid = get(p, "l2_grid", "probe")?
            .as_array()
            .ok_or("config: probe.l2_grid is not a list")?
            .iter()
            .map(|x| x.as_f64().filter(|x| x.is_finite() && *x >= 0.0))
            .collect::<Option<Vec<f64>>>()
            .ok_or("config: probe.l2_grid holds a value that is not a finite number >= 0")?;
        if l2_grid.is_empty() {
            return Err("config: probe.l2_grid is empty".into());
        }
        let probe = Probe {
            n_min: u32_in(
                get_u64(p, "n_min", "probe")?,
                "probe.n_min",
                1,
                ngram::MAX_ORDER,
            )?,
            n_max: u32_in(
                get_u64(p, "n_max", "probe")?,
                "probe.n_max",
                1,
                ngram::MAX_ORDER,
            )?,
            dim: u32_in(get_u64(p, "dim", "probe")?, "probe.dim", 16, MAX_PROBE_DIM)?,
            max_iter: u32_in(
                get_u64(p, "max_iter", "probe")?,
                "probe.max_iter",
                1,
                MAX_PROBE_ITER,
            )?,
            tol: get_f64(p, "tol", "probe")?,
            lr: get_f64(p, "lr", "probe")?,
            l2_grid,
            max_val_accuracy: get_f64(p, "max_val_accuracy", "probe")?,
        };
        NGramSpec::new(probe.n_min, probe.n_max, probe.dim)?;
        if probe.tol < 0.0 || probe.lr <= 0.0 {
            return Err("config: probe.tol must be >= 0 and probe.lr > 0".into());
        }
        if probe.max_val_accuracy <= 0.0 || probe.max_val_accuracy > 1.0 {
            return Err("config: probe.max_val_accuracy is outside (0, 1]".into());
        }
        let rows_per_template = get_u64(&v, "rows_per_template", "config")? as usize;
        if !(1..=MAX_ROWS_PER_TEMPLATE).contains(&rows_per_template) {
            return Err(format!(
                "config: rows_per_template {rows_per_template} is outside 1..={MAX_ROWS_PER_TEMPLATE}"
            ));
        }
        let val_templates_per_class = get_u64(&v, "val_templates_per_class", "config")? as usize;
        if !(1..MIN_TEMPLATES_PER_CLASS).contains(&val_templates_per_class) {
            return Err(format!(
                "config: val_templates_per_class {val_templates_per_class} must leave at least one \
                 of the {MIN_TEMPLATES_PER_CLASS} required templates to train on"
            ));
        }
        let heldout_templates_per_class = match v.get("heldout_templates_per_class") {
            None => 0,
            Some(n) => n
                .as_u64()
                .ok_or("config: heldout_templates_per_class is not a whole number")?
                as usize,
        };
        if heldout_templates_per_class + val_templates_per_class >= MIN_TEMPLATES_PER_CLASS {
            return Err(format!(
                "config: heldout_templates_per_class {heldout_templates_per_class} and \
                 val_templates_per_class {val_templates_per_class} must leave at least one of the \
                 {MIN_TEMPLATES_PER_CLASS} required templates to train on"
            ));
        }
        let injection_rate = get_f64(&v, "injection_rate", "config")?;
        if !(0.0..=1.0).contains(&injection_rate) {
            return Err(format!(
                "config: injection_rate {injection_rate} is outside [0, 1]"
            ));
        }
        let threshold = get_f64(&v, "containment_threshold", "config")?;
        if threshold <= 0.0 || threshold > 1.0 {
            return Err(format!(
                "config: containment_threshold {threshold} is outside (0, 1]"
            ));
        }
        let kind = Kind::parse(get_str(&v, "kind", "config")?)?;
        let tools = match (kind, v.get("tools")) {
            (Kind::Tools, Some(t)) => Some(crate::synth_tools::ToolsConfig::parse(t)?),
            (Kind::Tools, None) => return Err("config: kind tools needs a \"tools\" object".into()),
            (_, Some(_)) => return Err("config: a \"tools\" object is only for kind tools".into()),
            (_, None) => None,
        };
        Ok(Config {
            kind,
            seed: get_u64(&v, "seed", "config")?,
            rows_per_template,
            val_templates_per_class,
            heldout_templates_per_class,
            injection_rate,
            ngram_n: u32::try_from(get_u64(&v, "ngram_n", "config")?)
                .ok()
                .filter(|n| (1..=containment::MAX_N).contains(n))
                .ok_or("config: ngram_n is outside 1..=64")?,
            containment_threshold: threshold,
            target_sha256: str_map(&v, "target_sha256")?,
            probe,
            tools,
        })
    }
}

/// Seeded draws under a key: sha256 over the seed, the key and a tag, as
/// [`decisions::keyed`] makes every other choice in this crate.
#[derive(Clone, Debug)]
pub struct Scope {
    seed: u64,
    key: String,
}

impl Scope {
    pub fn new(seed: u64, key: &str) -> Self {
        Self {
            seed,
            key: key.to_owned(),
        }
    }

    pub fn unit(&self, tag: &str) -> f64 {
        decisions::unit_draw(self.seed, &[&self.key, tag])
    }

    /// Uniform over `0..n` (the modulo bias is below 2^-50 for any `n` used here).
    pub fn below(&self, tag: &str, n: usize) -> usize {
        assert!(n > 0, "a draw from an empty range");
        let d = decisions::keyed(self.seed, &[&self.key, tag]);
        (u64::from_le_bytes(d[..8].try_into().expect("8 bytes")) % n as u64) as usize
    }

    pub fn pick<'a, T>(&self, tag: &str, xs: &'a [T]) -> &'a T {
        &xs[self.below(tag, xs.len())]
    }

    /// Fisher-Yates under this scope.
    pub fn shuffle<T: Clone>(&self, tag: &str, xs: &[T]) -> Vec<T> {
        let mut v = xs.to_vec();
        for i in (1..v.len()).rev() {
            let j = self.below(&format!("{tag}\u{1f}{i}"), i + 1);
            v.swap(i, j);
        }
        v
    }
}

/// Fill `{name}` placeholders from `vars`. An unknown or unclosed placeholder is a bug in a
/// template: it is refused, never written into a row as a literal `{name}`.
pub fn fill(template: &str, vars: &BTreeMap<&str, String>) -> Result<String, String> {
    let mut out = String::with_capacity(template.len() + 32);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after
            .find('}')
            .ok_or_else(|| format!("template {template:?}: unclosed placeholder"))?;
        let name = &after[..close];
        let value = vars
            .get(name)
            .ok_or_else(|| format!("template {template:?}: no value for {{{name}}}"))?;
        out.push_str(value);
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// One row as a generator writes it, before the split, the option shuffle and the checks.
#[derive(Clone, Debug)]
pub struct Draft {
    pub family_id: &'static str,
    pub stratum: String,
    /// The template the row was written from; the split unit and the group key. Unique within
    /// a family and owned by one `template_class`.
    pub template: String,
    /// The class the template is stratified under for the split: every class gets val
    /// templates. The gold may still be `noul` (a control removed from a tree).
    pub template_class: String,
    pub context: String,
    pub question: String,
    pub slot_name: &'static str,
    /// In the generator's canonical order; shuffled per row by [`assemble`].
    pub options: Vec<String>,
    pub gold: Gold,
    /// Whether every row of this family offers the same option set, so a class-label probe
    /// is meaningful.
    pub fixed_options: bool,
}

impl Draft {
    /// The gold's class name: the option text or `noul`.
    fn gold_class(&self) -> String {
        match self.gold {
            Gold::Option(i) => self.options[i].clone(),
            Gold::Noul => "noul".to_owned(),
        }
    }
}

/// Each (family, template) and the side of the split it is on.
type Split = BTreeMap<(String, String), &'static str>;

/// The side of a template held out of the pool: never in `examples.jsonl`, rendered only by
/// [`emit_heldout`].
pub const HELDOUT: &str = "heldout";

/// The split of every template: per family and stratification class, the templates sorted by a
/// seeded key; the first `heldout_templates_per_class` are held out, the next
/// `val_templates_per_class` go to val, and the rest to train.
fn template_split(cfg: &Config, drafts: &[Draft]) -> Result<(Split, Value), String> {
    let mut by_class: BTreeMap<(&str, &str), BTreeSet<&str>> = BTreeMap::new();
    let mut owner: BTreeMap<(&str, &str), &str> = BTreeMap::new();
    for d in drafts {
        if let Some(prev) = owner.insert(
            (d.family_id, d.template.as_str()),
            d.template_class.as_str(),
        ) && prev != d.template_class
        {
            return Err(format!(
                "{}: template {:?} is stratified under both {prev:?} and {:?}; a template must \
                 sit on one side of the split",
                d.family_id, d.template, d.template_class
            ));
        }
        by_class
            .entry((d.family_id, d.template_class.as_str()))
            .or_default()
            .insert(d.template.as_str());
    }
    let mut split = BTreeMap::new();
    let mut report = serde_json::Map::new();
    for ((family, class), templates) in &by_class {
        if templates.len() < MIN_TEMPLATES_PER_CLASS {
            return Err(format!(
                "{family}/{class}: {} template(s); at least {MIN_TEMPLATES_PER_CLASS} are required \
                 so that val can hold one out",
                templates.len()
            ));
        }
        let mut order: Vec<&str> = templates.iter().copied().collect();
        order.sort_by_key(|t| decisions::keyed(cfg.seed, &["val-template", *family, *class, *t]));
        let h = cfg.heldout_templates_per_class;
        let held: Vec<&str> = order[..h].to_vec();
        let val: Vec<&str> = order[h..h + cfg.val_templates_per_class].to_vec();
        for t in &order {
            let side = if held.contains(t) {
                HELDOUT
            } else if val.contains(t) {
                "val"
            } else {
                "train"
            };
            split.insert(((*family).to_owned(), (*t).to_owned()), side);
        }
        report.insert(
            format!("{family}/{class}"),
            json!({"templates": templates.len(), "val_templates": val, "heldout_templates": held}),
        );
    }
    Ok((split, Value::Object(report)))
}

/// `(rows, min, max, sum)` of rendered-context bytes.
type LengthAcc = (usize, usize, usize, usize);

/// Per class: count and min / mean / max bytes of the rendered context, per split.
fn length_stats(rows: &[(&Draft, &'static str)]) -> Value {
    let mut acc: BTreeMap<(String, String, &str), LengthAcc> = BTreeMap::new();
    for (d, split) in rows {
        let n = d.question.len() + 2 + d.context.len();
        let e = acc
            .entry((d.family_id.to_owned(), d.gold_class(), *split))
            .or_insert((0, usize::MAX, 0, 0));
        e.0 += 1;
        e.1 = e.1.min(n);
        e.2 = e.2.max(n);
        e.3 += n;
    }
    let mut out = serde_json::Map::new();
    for ((family, class, split), (count, min, max, sum)) in acc {
        out.insert(
            format!("{family}/{class}/{split}"),
            json!({"rows": count, "min": min, "max": max, "mean": sum as f64 / count as f64}),
        );
    }
    Value::Object(out)
}

/// What [`assemble`] made: the rows, and the facts about them the manifest records.
pub struct Assembled {
    pub candidates: Vec<Candidate>,
    pub report: Value,
}

/// Split, shuffle, check and identify every draft of the pool (train and val). Refuses a class
/// with too few templates, a template owned by two classes and a pool over [`MAX_ROWS`]; counts
/// every row it drops, held-out-template rows included.
pub fn assemble(cfg: &Config, drafts: &[Draft]) -> Result<Assembled, String> {
    assemble_sides(cfg, drafts, false)
}

/// The held-out templates' rows, assembled exactly as the pool's are (same ids, shuffle and
/// structural checks), for [`emit_heldout`].
pub fn assemble_heldout(cfg: &Config, drafts: &[Draft]) -> Result<Assembled, String> {
    assemble_sides(cfg, drafts, true)
}

/// Families whose module states `noul` as a zero. A `noul` gold there is a generator defect:
/// refused and counted as `noul_on_a_family_stating_zero`, never admitted.
const NOUL_STATED_ZERO: [&str; 1] = [crate::synth_tools::FAMILY];

/// Both sides go through one dedupe in one order (held-out first, then val, then train), so a
/// pool row that duplicates a held-out row is the one dropped, and counted.
fn assemble_sides(cfg: &Config, drafts: &[Draft], heldout: bool) -> Result<Assembled, String> {
    if drafts.len() > MAX_ROWS {
        return Err(format!(
            "{} drafts; a pool holds at most {MAX_ROWS}",
            drafts.len()
        ));
    }
    let (split, split_report) = template_split(cfg, drafts)?;
    // The keep rule is split-aware: held-out drafts go first, then val, so a duplicate never
    // removes one of those in favour of a train row.
    let mut order: Vec<(&Draft, &'static str)> = drafts
        .iter()
        .map(|d| (d, split[&(d.family_id.to_owned(), d.template.clone())]))
        .collect();
    order.sort_by_key(|(_, s)| match *s {
        HELDOUT => 0,
        "val" => 1,
        _ => 2,
    });
    let source_id = cfg.kind.source_id();
    let mut seen = BTreeSet::new();
    let mut refused: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut kept_drafts: Vec<(&Draft, &'static str)> = Vec::new();
    let mut candidates = Vec::new();
    for (d, side) in order {
        let mut canonical = d.options.clone();
        canonical.sort();
        let digest = sha256_hex(
            format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{}",
                d.family_id,
                d.question,
                d.context,
                canonical.join("\u{1e}")
            )
            .as_bytes(),
        );
        if !seen.insert(digest.clone()) {
            *refused.entry("exact_duplicate").or_default() += 1;
            continue;
        }
        if (side == HELDOUT) != heldout {
            if side == HELDOUT {
                *refused.entry("heldout_template").or_default() += 1;
            }
            continue;
        }
        if d.gold == Gold::Noul && NOUL_STATED_ZERO.contains(&d.family_id) {
            *refused.entry("noul_on_a_family_stating_zero").or_default() += 1;
            continue;
        }
        let perm = Scope::new(cfg.seed, &digest)
            .shuffle("options", &(0..d.options.len()).collect::<Vec<_>>());
        let options: Vec<String> = perm.iter().map(|&i| d.options[i].clone()).collect();
        let gold = match d.gold {
            Gold::Option(g) => Gold::Option(
                perm.iter()
                    .position(|&i| i == g)
                    .ok_or("gold lost in shuffle")?,
            ),
            Gold::Noul => Gold::Noul,
        };
        let c = Candidate {
            id: format!(
                "synth-{}:{}:{}",
                cfg.kind.name(),
                d.family_id,
                &digest[..16]
            ),
            source_id,
            family_id: d.family_id.to_owned(),
            stratum: d.stratum.clone(),
            group_key: d.template.clone(),
            licence: LICENCE.to_owned(),
            context: d.context.clone(),
            question: d.question.clone(),
            slot_name: d.slot_name.to_owned(),
            options,
            gold,
            label_basis: "by_construction",
            split: side,
        };
        if let Some(reason) = decisions::structural_refusal(&c) {
            *refused.entry(reason).or_default() += 1;
            continue;
        }
        kept_drafts.push((d, side));
        candidates.push(c);
    }
    let report = json!({
        "templates": split_report,
        "refused": refused,
        "context_bytes_by_class": length_stats(&kept_drafts),
    });
    Ok(Assembled { candidates, report })
}

/// The leak probe for one fixed-option family: a linear control on char n-grams of the
/// rendered context, fitted on the train rows (its L2 picked on a seeded tenth of them) and
/// scored on the val rows, which are other templates.
pub fn leak_probe(
    cfg: &Config,
    family: &str,
    rows: &[&Candidate],
    threads: usize,
) -> Result<Value, String> {
    let mut classes: Vec<String> = rows
        .iter()
        .map(|c| match c.gold {
            Gold::Option(i) => c.options[i].clone(),
            Gold::Noul => "noul".to_owned(),
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    classes.sort();
    let class_of = |c: &Candidate| -> u32 {
        let name = match c.gold {
            Gold::Option(i) => c.options[i].as_str(),
            Gold::Noul => "noul",
        };
        classes
            .iter()
            .position(|x| x == name)
            .expect("every class was collected") as u32
    };
    let train: Vec<&Candidate> = rows
        .iter()
        .copied()
        .filter(|c| c.split == "train")
        .collect();
    let val: Vec<&Candidate> = rows.iter().copied().filter(|c| c.split == "val").collect();
    if classes.len() < 2 || train.len() < 4 || val.is_empty() {
        return Ok(json!({
            "state": "not_run",
            "reason": format!("{} class(es), {} train and {} val rows: nothing to fit or score",
                classes.len(), train.len(), val.len()),
        }));
    }
    let spec = NGramSpec::new(cfg.probe.n_min, cfg.probe.n_max, cfg.probe.dim)?;
    let docs = |cs: &[&Candidate]| cs.iter().map(|c| c.rendered_context()).collect::<Vec<_>>();
    let (train_docs, val_docs) = (docs(&train[..]), docs(&val[..]));
    let x = ngram::transform(
        spec,
        &train_docs.iter().map(String::as_str).collect::<Vec<_>>(),
        threads,
    )?;
    let eval_x = ngram::transform(
        spec,
        &val_docs.iter().map(String::as_str).collect::<Vec<_>>(),
        threads,
    )?;
    let y: Vec<u32> = train.iter().map(|c| class_of(c)).collect();
    let k = classes.len();
    let scope = Scope::new(cfg.seed, &format!("probe\u{1f}{family}"));
    let order = scope.shuffle("order", &(0..train.len()).collect::<Vec<_>>());
    let n_val = (train.len() / 10).max(1);
    let hyper = Hyper {
        n_classes: k,
        max_iter: cfg.probe.max_iter,
        tol: cfg.probe.tol,
        lr: cfg.probe.lr,
        l2_grid: cfg.probe.l2_grid.clone(),
    };
    let w0 = vec![0.0; x.n_cols * k];
    let fitted = linfit::fit(
        &x,
        &y,
        &order,
        n_val,
        &w0,
        &eval_x,
        &hyper,
        Selection::Top1,
        threads,
    )?;
    let mut correct = 0usize;
    for (r, c) in val.iter().enumerate() {
        let z = &fitted.eval_logits[r * k..(r + 1) * k];
        let best = (1..k).fold(0, |b, j| if z[j] > z[b] { j } else { b });
        if best as u32 == class_of(c) {
            correct += 1;
        }
    }
    let mut majority: BTreeMap<u32, usize> = BTreeMap::new();
    for c in &val {
        *majority.entry(class_of(c)).or_default() += 1;
    }
    let majority = majority.values().copied().max().unwrap_or(0);
    let accuracy = correct as f64 / val.len() as f64;
    Ok(json!({
        "state": "ran",
        "classes": k,
        "train_rows": train.len(),
        "val_rows": val.len(),
        "val_correct": correct,
        "val_accuracy": accuracy,
        "val_majority_share": majority as f64 / val.len() as f64,
        "bound": cfg.probe.max_val_accuracy,
        "passed": accuracy <= cfg.probe.max_val_accuracy,
        "selected_l2": fitted.grid[fitted.selected].l2,
        "converged": fitted.refit.converged,
        "iterations": fitted.refit.iterations,
    }))
}

/// What the CLI passes.
pub struct Inputs {
    pub config: PathBuf,
    pub targets: Vec<(String, PathBuf)>,
}

/// What a generator made: its drafts, the facts it alone can report (the manifest's
/// `generator` block), and the sha256 of every input file it read beyond the config.
pub struct Generated {
    pub drafts: Vec<Draft>,
    pub report: Value,
    pub inputs: BTreeMap<String, String>,
}

/// The generator for `cfg.kind`.
pub fn generate(cfg: &Config) -> Result<Generated, String> {
    // The row budget, before the pool's rows are drafted: one row per template is drafted to
    // count the templates (the tools kind's come from its data files), and a config whose
    // templates x rows_per_template is past MAX_ROWS is refused there, not after drafting it.
    if cfg.rows_per_template > 1 {
        let mut one = cfg.clone();
        one.rows_per_template = 1;
        let probe = generate(&one)?;
        let templates = probe
            .drafts
            .iter()
            .map(|d| (d.family_id, d.template.as_str()))
            .collect::<BTreeSet<_>>()
            .len();
        let at_least = templates.saturating_mul(cfg.rows_per_template);
        if at_least > MAX_ROWS {
            return Err(format!(
                "{templates} templates x rows_per_template {} = {at_least} rows; a pool holds at \
                 most {MAX_ROWS} (refused before drafting them)",
                cfg.rows_per_template
            ));
        }
    }
    let plain = |drafts| Generated {
        drafts,
        report: Value::Null,
        inputs: BTreeMap::new(),
    };
    match cfg.kind {
        Kind::Email => Ok(plain(crate::synth_email::drafts(cfg)?)),
        Kind::Jarvis => Ok(plain(crate::synth_jarvis::drafts(cfg)?)),
        Kind::Tools => crate::synth_tools::generate(cfg),
    }
}

/// The pool, the manifest and the containment record, from drafts already generated.
pub fn build(
    cfg: &Config,
    config_sha256: &str,
    drafts: &[Draft],
    targets: &[TargetSet],
    mut digests: BTreeMap<String, String>,
    generator_report: &Value,
    threads: usize,
) -> Result<decisions::Built, String> {
    let assembled = assemble(cfg, drafts)?;
    let scanned = pool::decontaminate(
        cfg.ngram_n,
        cfg.containment_threshold,
        &json!({"tool": "qd-prep synth", "kind": cfg.kind.name(), "seed": cfg.seed}),
        "synth-candidates",
        &assembled.candidates,
        targets,
        threads,
    )?;
    let clean = &scanned.clean;

    let fixed: BTreeMap<&str, bool> = drafts
        .iter()
        .map(|d| (d.family_id, d.fixed_options))
        .collect();
    let mut probes = serde_json::Map::new();
    let mut over_bound = Vec::new();
    for (family, is_fixed) in &fixed {
        let rows: Vec<&Candidate> = clean
            .iter()
            .copied()
            .filter(|c| c.family_id == *family)
            .collect();
        let probe = if *is_fixed {
            leak_probe(cfg, family, &rows, threads)?
        } else {
            json!({"state": "not_run", "reason": "options differ per row (enumerated elements), so \
                   a class-label probe has no fixed class set; the per-option control \
                   (qd_train.option_control) over the built mixture covers it"})
        };
        if probe.get("passed") == Some(&Value::Bool(false)) {
            over_bound.push((*family).to_owned());
        }
        probes.insert((*family).to_owned(), probe);
    }

    let ex = pool::examples(clean)?;
    digests.insert("config".to_owned(), config_sha256.to_owned());
    let manifest = json!({
        "schema": decisions::MANIFEST_SCHEMA, "mode": "build", "tool": "qd-prep synth",
        "tool_version": env!("CARGO_PKG_VERSION"), "kind": cfg.kind.name(),
        "source_id": cfg.kind.source_id(), "licence": LICENCE,
        "privacy": "synthetic only (human ruling 2026-10-06): no mailbox, filter list, profile \
                    value or scenario input is read; senders are on reserved example domains",
        "inputs": digests,
        "selected": ex.selected,
        "noul_gold_selected": ex.noul,
        "gold_position": ex.gold_position,
        "val_without_train": ex.val_without_train,
        "assembly": assembled.report,
        "generator": generator_report,
        "decontamination": scanned.report,
        "leak_probe": probes,
        "allocation": pool::allocation(false, "candidates: no cap table has drawn from these \
                 rows; the v6 allocation over every pool (its cap table and the defect-share \
                 bound) decides what enters a mixture, and the loader refuses this pool until then"),
        "examples": ex.rows, "examples_sha256": ex.sha256,
    });
    if !over_bound.is_empty() {
        return Err(format!(
            "leak probe over its bound ({}) for {over_bound:?}: a linear control on the context \
             alone reads the label on held-out templates, so the generator leaks it. Probe: {}",
            cfg.probe.max_val_accuracy,
            Value::Object(probes)
        ));
    }
    Ok(decisions::Built {
        summary: format!(
            "qd-prep synth {}: {} rows ({} noul gold)",
            cfg.kind.name(),
            ex.rows,
            ex.noul
        ),
        examples: ex.bytes,
        manifest,
        texts: None,
        containment: scanned.written,
    })
}

/// `qd-prep synth`: the config and targets in, `DIR/{examples.jsonl, manifest.json,
/// containment/}` out.
pub fn run(inputs: &Inputs, out_dir: &Path, threads: usize) -> Result<String, String> {
    if out_dir.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            out_dir.display()
        ));
    }
    let (cfg, config_bytes) = Config::load(&inputs.config)?;
    let (targets, mut digests) = pool::read_targets(&cfg.target_sha256, &inputs.targets)?;
    let generated = generate(&cfg)?;
    digests.extend(generated.inputs);
    let built = build(
        &cfg,
        &sha256_hex(&config_bytes),
        &generated.drafts,
        &targets,
        digests,
        &generated.report,
        threads,
    )?;
    decisions::write_out(out_dir, &built)?;
    Ok(format!("{} -> {}", built.summary, out_dir.display()))
}

/// What [`emit_heldout`] writes, before it is written.
pub struct HeldOut {
    /// Decision rows of the held-out templates, `split` = [`HELDOUT`]: the evaluation set.
    pub examples: Vec<u8>,
    /// `{"id", "text"}` rows, the text the containment scan compares: the pool's target set.
    pub targets: Vec<u8>,
    pub manifest: Value,
    pub rows: usize,
}

/// The held-out templates' rows of `cfg`'s kind. Refuses a config that holds no template out.
///
/// A target's text is its row's context: the instance, never what every row of the kind shares.
/// The question and the option list are the same in every row of a kind, so a target carrying
/// them is contained in every candidate, and the pool's scan excludes them all
/// (GAP-SYNTH-HELDOUT-TARGET-TEXT-CARRIED-SHARED-SCAFFOLDING-2026-10-06).
pub fn held_out(cfg: &Config, config_sha256: &str) -> Result<HeldOut, String> {
    if cfg.heldout_templates_per_class == 0 {
        return Err(format!(
            "config ({}) holds no template out: heldout_templates_per_class is 0",
            cfg.kind.name()
        ));
    }
    let generated = generate(cfg)?;
    let a = assemble_heldout(cfg, &generated.drafts)?;
    if a.candidates.is_empty() {
        return Err("no held-out row survived assembly".into());
    }
    let mut examples = Vec::new();
    let mut targets = Vec::new();
    for c in &a.candidates {
        let line = |v: Value, out: &mut Vec<u8>| -> Result<(), String> {
            serde_json::to_writer(&mut *out, &v).map_err(|e| e.to_string())?;
            out.push(b'\n');
            Ok(())
        };
        line(decisions::example_json(c), &mut examples)?;
        line(json!({"id": c.id, "text": c.context}), &mut targets)?;
    }
    let mut inputs = generated.inputs;
    inputs.insert("config".to_owned(), config_sha256.to_owned());
    let manifest = json!({
        "schema": HELDOUT_SCHEMA, "tool": "qd-prep synth --emit-heldout",
        "tool_version": env!("CARGO_PKG_VERSION"), "kind": cfg.kind.name(),
        "source_id": cfg.kind.source_id(), "licence": LICENCE,
        "never_train": "held-out evaluation set (CLAUDE.md rule 3): its templates are excluded \
                        from every pool built from this config, and its path carries a held-out \
                        marker. A pool reads targets.jsonl only as a decontamination target.",
        "inputs": inputs,
        "heldout_templates_per_class": cfg.heldout_templates_per_class,
        "rows": a.candidates.len(),
        "examples_sha256": sha256_hex(&examples),
        "targets_sha256": sha256_hex(&targets),
        "assembly": a.report,
    });
    Ok(HeldOut {
        examples,
        targets,
        manifest,
        rows: a.candidates.len(),
    })
}

/// `qd-prep synth --emit-heldout`: `DIR/{examples.jsonl, targets.jsonl, manifest.json}` out,
/// where `DIR` carries a held-out path marker. Pin `targets.jsonl`'s sha256 in the config's
/// `target_sha256` and pass it as `--target` when the pool is built.
pub fn emit_heldout(config: &Path, out_dir: &Path) -> Result<String, String> {
    crate::heldout::check_held_out_path(out_dir)?;
    let partial = crate::files::partial_path(out_dir)?;
    for p in [out_dir, partial.as_path()] {
        if p.exists() {
            return Err(format!("{} exists; refusing to overwrite it", p.display()));
        }
    }
    let (cfg, config_bytes) = Config::load(config)?;
    let h = held_out(&cfg, &sha256_hex(&config_bytes))?;
    let mut manifest = serde_json::to_vec_pretty(&h.manifest).map_err(|e| e.to_string())?;
    manifest.push(b'\n');
    crate::files::write_new_dir(out_dir, |partial| {
        // A symlinked parent could land the files somewhere unmarked; check where they go.
        let real =
            std::fs::canonicalize(partial).map_err(|e| format!("{}: {e}", partial.display()))?;
        crate::heldout::check_held_out_path(&real)?;
        crate::files::write_synced_new(&partial.join("examples.jsonl"), &h.examples)?;
        crate::files::write_synced_new(&partial.join("targets.jsonl"), &h.targets)?;
        crate::files::write_synced_new(&partial.join("manifest.json"), &manifest)
    })?;
    Ok(format!(
        "qd-prep synth {} held-out: {} rows -> {}",
        cfg.kind.name(),
        h.rows,
        out_dir.display()
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn cfg(kind: Kind) -> Config {
        Config {
            kind,
            seed: 20261006,
            rows_per_template: 6,
            val_templates_per_class: 1,
            heldout_templates_per_class: 0,
            injection_rate: 0.1,
            ngram_n: 8,
            containment_threshold: 0.5,
            target_sha256: BTreeMap::new(),
            probe: Probe {
                n_min: 3,
                n_max: 5,
                dim: 1 << 14,
                max_iter: 300,
                tol: 1e-4,
                lr: 0.05,
                l2_grid: vec![1e-4, 1e-3],
                max_val_accuracy: 0.95,
            },
            tools: None,
        }
    }

    fn draft(template: &str, class: &str, gold: usize, context: &str) -> Draft {
        Draft {
            family_id: "t.family",
            stratum: "t.family/choice".to_owned(),
            template: template.to_owned(),
            template_class: class.to_owned(),
            context: context.to_owned(),
            question: "Which?".to_owned(),
            slot_name: "answer",
            options: vec!["alpha".into(), "beta".into(), "gamma".into()],
            gold: Gold::Option(gold),
            fixed_options: true,
        }
    }

    #[test]
    fn fill_refuses_a_placeholder_it_has_no_value_for() {
        let mut v = BTreeMap::new();
        v.insert("a", "x".to_owned());
        assert_eq!(fill("{a}-{a}", &v).unwrap(), "x-x");
        assert!(fill("{b}", &v).is_err());
        assert!(fill("{a", &v).is_err());
    }

    #[test]
    fn a_class_with_fewer_than_four_templates_is_refused() {
        let ds: Vec<Draft> = (0..3)
            .map(|i| draft(&format!("t{i}"), "alpha", 0, &format!("row {i}")))
            .collect();
        let Err(err) = assemble(&cfg(Kind::Email), &ds) else {
            panic!("not refused")
        };
        assert!(err.contains("at least 4 are required"), "{err}");
    }

    #[test]
    fn a_template_owned_by_two_classes_is_refused() {
        let mut ds: Vec<Draft> = (0..4)
            .map(|i| draft(&format!("t{i}"), "alpha", 0, &format!("a {i}")))
            .collect();
        ds.push(draft("t0", "beta", 1, "b 0"));
        let Err(err) = assemble(&cfg(Kind::Email), &ds) else {
            panic!("not refused")
        };
        assert!(err.contains("stratified under both"), "{err}");
    }

    #[test]
    fn val_holds_whole_templates_and_every_class_has_one() {
        let mut ds = Vec::new();
        for (ci, class) in ["alpha", "beta"].iter().enumerate() {
            for t in 0..5 {
                for r in 0..3 {
                    ds.push(draft(
                        &format!("{class}-t{t}"),
                        class,
                        ci,
                        &format!("{class} {t} {r}"),
                    ));
                }
            }
        }
        let a = assemble(&cfg(Kind::Email), &ds).unwrap();
        let mut side: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for c in &a.candidates {
            side.entry(c.group_key.as_str())
                .or_default()
                .insert(c.split);
        }
        assert!(
            side.values().all(|s| s.len() == 1),
            "a template straddles the split: {side:?}"
        );
        for class in ["alpha", "beta"] {
            let val = a
                .candidates
                .iter()
                .filter(|c| c.split == "val" && c.group_key.starts_with(class))
                .count();
            assert_eq!(val, 3, "{class}: one val template of three rows");
        }
    }

    fn two_class_drafts() -> Vec<Draft> {
        let mut ds = Vec::new();
        for (ci, class) in ["alpha", "beta"].iter().enumerate() {
            for t in 0..5 {
                for r in 0..3 {
                    ds.push(draft(
                        &format!("{class}-t{t}"),
                        class,
                        ci,
                        &format!("{class} {t} {r}"),
                    ));
                }
            }
        }
        ds
    }

    #[test]
    fn the_seeded_order_puts_held_out_first_then_val_and_none_held_out_splits_as_before() {
        let ds = two_class_drafts();
        // The order the split draws from, recomputed from its definition: per class, templates
        // sorted by the seeded key. Without a held-out template val is order[0], as before the
        // key existed; with one, held-out is order[0] and val order[1].
        let order = |class: &str| {
            let mut ts: Vec<String> = (0..5).map(|t| format!("{class}-t{t}")).collect();
            ts.sort_by_key(|t| {
                decisions::keyed(20261006, &["val-template", "t.family", class, t.as_str()])
            });
            ts
        };
        let side_of = |a: &Assembled, template: &str| -> BTreeSet<&'static str> {
            a.candidates
                .iter()
                .filter(|c| c.group_key == template)
                .map(|c| c.split)
                .collect()
        };
        let none = assemble(&cfg(Kind::Email), &ds).unwrap();
        let mut c = cfg(Kind::Email);
        c.heldout_templates_per_class = 1;
        let one = assemble(&c, &ds).unwrap();
        let held = assemble_heldout(&c, &ds).unwrap();
        for class in ["alpha", "beta"] {
            let o = order(class);
            assert_eq!(side_of(&none, &o[0]), BTreeSet::from(["val"]), "{class}");
            assert_eq!(side_of(&none, &o[1]), BTreeSet::from(["train"]), "{class}");
            assert_eq!(side_of(&held, &o[0]), BTreeSet::from([HELDOUT]), "{class}");
            assert!(
                side_of(&one, &o[0]).is_empty(),
                "{class}: held out of the pool"
            );
            assert_eq!(side_of(&one, &o[1]), BTreeSet::from(["val"]), "{class}");
        }
        assert!(none.report["refused"].get("heldout_template").is_none());
    }

    #[test]
    fn held_out_templates_never_reach_the_pool_and_are_counted() {
        let ds = two_class_drafts();
        let mut c = cfg(Kind::Email);
        c.heldout_templates_per_class = 1;
        let pool = assemble(&c, &ds).unwrap();
        let held = assemble_heldout(&c, &ds).unwrap();
        // One template of three rows per class is held out.
        assert_eq!(held.candidates.len(), 6);
        assert!(held.candidates.iter().all(|c| c.split == HELDOUT));
        assert_eq!(pool.report["refused"]["heldout_template"], 6);
        let held_templates: BTreeSet<&str> = held
            .candidates
            .iter()
            .map(|c| c.group_key.as_str())
            .collect();
        let pool_templates: BTreeSet<&str> = pool
            .candidates
            .iter()
            .map(|c| c.group_key.as_str())
            .collect();
        assert!(held_templates.is_disjoint(&pool_templates));
        assert_eq!(pool.candidates.len() + held.candidates.len(), ds.len());
        // Val still holds one whole template per class, drawn from what is left.
        for class in ["alpha", "beta"] {
            let val = pool
                .candidates
                .iter()
                .filter(|c| c.split == "val" && c.group_key.starts_with(class))
                .count();
            assert_eq!(val, 3, "{class}");
        }
    }

    #[test]
    fn a_pool_row_duplicating_a_held_out_row_is_the_one_dropped() {
        let mut ds = two_class_drafts();
        let mut c = cfg(Kind::Email);
        c.heldout_templates_per_class = 1;
        let held = assemble_heldout(&c, &ds).unwrap();
        // A train-side draft with a held-out row's exact text, under another template.
        let h = &held.candidates[0];
        let class = if h.group_key.starts_with("alpha") {
            "alpha"
        } else {
            "beta"
        };
        let ci = usize::from(class == "beta");
        let text = ds
            .iter()
            .find(|d| d.template == h.group_key)
            .map(|d| d.context.clone())
            .unwrap();
        let other = ds
            .iter()
            .map(|d| d.template.clone())
            .find(|t| t.starts_with(class) && *t != h.group_key)
            .unwrap();
        ds.push(draft(&other, class, ci, &text));
        let pool = assemble(&c, &ds).unwrap();
        assert!(pool.candidates.iter().all(|p| p.id != h.id));
        assert_eq!(pool.report["refused"]["exact_duplicate"], 1);
        assert_eq!(assemble_heldout(&c, &ds).unwrap().candidates.len(), 6);
    }

    /// A held-out target is decontaminated by its row's own content. The v1 email set put the
    /// question and the option list, which every email row shares, into each target, and the
    /// pool's scan then excluded every candidate
    /// (GAP-SYNTH-HELDOUT-TARGET-TEXT-CARRIED-SHARED-SCAFFOLDING-2026-10-06). The controls: a
    /// candidate carrying a held-out row's email is excluded, and the genuine candidates, drafted
    /// from other templates, are not all excluded.
    #[test]
    fn a_held_out_target_is_its_rows_content_so_the_scan_still_tells_rows_apart() {
        let c = Config {
            heldout_templates_per_class: 1,
            ..cfg(Kind::Email)
        };
        let h = held_out(&c, "config-sha").unwrap();
        let targets: Vec<(String, String)> = h
            .targets
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .map(|l| {
                let v: Value = serde_json::from_slice(l).unwrap();
                let field = |k: &str| v[k].as_str().unwrap().to_owned();
                (field("id"), field("text"))
            })
            .collect();
        let g = generate(&c).unwrap();
        let a = assemble(&c, &g.drafts).unwrap();
        let held = assemble_heldout(&c, &g.drafts).unwrap();
        let mut copy = a.candidates[0].clone();
        copy.id = "control/carries-a-held-out-email".to_owned();
        copy.context = held.candidates[0].context.clone();
        let mut candidates = a.candidates.clone();
        candidates.push(copy.clone());
        let set: TargetSet = ("email-heldout".to_owned(), targets.clone());
        let s = pool::decontaminate(
            c.ngram_n,
            c.containment_threshold,
            &json!({"tool": "test"}),
            "synth-candidates",
            &candidates,
            &[set],
            1,
        )
        .unwrap_or_else(|e| panic!("every candidate matched a held-out target: {e}"));
        let clean: BTreeSet<&str> = s.clean.iter().map(|x| x.id.as_str()).collect();
        assert!(
            !clean.contains(copy.id.as_str()),
            "a candidate carrying a held-out email was kept"
        );
        let kept = a
            .candidates
            .iter()
            .filter(|x| clean.contains(x.id.as_str()))
            .count();
        assert!(
            kept > 0,
            "all {} genuine candidates were excluded: the targets match what every row shares, \
             not the rows",
            a.candidates.len()
        );
        for (id, text) in &targets {
            assert!(
                !text.contains(&copy.question),
                "{id} carries the shared question"
            );
            for o in &copy.options {
                assert!(
                    !text.contains(o.as_str()),
                    "{id} carries the shared option {o:?}"
                );
            }
        }
    }

    #[test]
    fn a_config_must_leave_a_train_template_after_val_and_held_out() {
        let shipped: Value = serde_json::from_slice(include_bytes!(
            "../../../data/synth/email-v6-2026-10-06.json"
        ))
        .unwrap();
        let with = |h: u64| {
            let mut v = shipped.clone();
            v["heldout_templates_per_class"] = json!(h);
            Config::parse(&serde_json::to_vec(&v).unwrap())
        };
        let val = shipped["val_templates_per_class"].as_u64().unwrap();
        let most = MIN_TEMPLATES_PER_CLASS as u64 - 1 - val;
        assert_eq!(
            with(most).unwrap().heldout_templates_per_class,
            most as usize
        );
        let err = with(most + 1).err().unwrap();
        assert!(err.contains("at least one"), "{err}");
        let mut v = shipped.clone();
        v["heldout_templates_per_class"] = json!("one");
        assert!(Config::parse(&serde_json::to_vec(&v).unwrap()).is_err());
    }

    /// Before the probe, a config past the row budget was refused by `assemble`, after every
    /// row (~650k for email) had been drafted in memory.
    #[test]
    fn a_config_past_the_row_budget_is_refused_before_its_rows_are_drafted() {
        let mut c = cfg(Kind::Email);
        c.rows_per_template = MAX_ROWS_PER_TEMPLATE;
        let Err(err) = generate(&c) else {
            panic!("not refused")
        };
        assert!(err.contains("refused before drafting them"), "{err}");
        assert!(
            err.contains(&format!("a pool holds at most {MAX_ROWS}")),
            "{err}"
        );
        c.rows_per_template = 2;
        assert!(generate(&c).is_ok());
    }

    #[test]
    fn emitting_held_out_rows_needs_a_marked_directory_and_a_held_out_template() {
        let dir = std::env::temp_dir().join(format!(
            "qd-prep-synth-emit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("email.json");
        let shipped: Value = serde_json::from_slice(include_bytes!(
            "../../../data/synth/email-v6-2026-10-06.json"
        ))
        .unwrap();
        let mut none = shipped.clone();
        none["heldout_templates_per_class"] = json!(0);
        std::fs::write(&config, serde_json::to_vec(&none).unwrap()).unwrap();
        let err = emit_heldout(&config, &dir.join("unmarked")).unwrap_err();
        assert!(err.contains("no path segment"), "{err}");
        let err = emit_heldout(&config, &dir.join("heldout").join("email")).unwrap_err();
        assert!(err.contains("holds no template out"), "{err}");
        assert!(!dir.join("heldout").join("email").exists());
        assert!(!dir.join("heldout").join("email.partial").exists());
    }

    #[test]
    fn options_are_shuffled_and_the_gold_follows_its_option() {
        let mut ds = Vec::new();
        for t in 0..4 {
            for r in 0..40 {
                ds.push(draft(&format!("t{t}"), "alpha", 0, &format!("ctx {t} {r}")));
            }
        }
        let a = assemble(&cfg(Kind::Email), &ds).unwrap();
        let mut positions = BTreeSet::new();
        for c in &a.candidates {
            let Gold::Option(i) = c.gold else {
                panic!("gold is an option")
            };
            assert_eq!(c.options[i], "alpha");
            positions.insert(i);
        }
        assert_eq!(
            positions.len(),
            3,
            "the gold lands in every position over 160 rows"
        );
    }

    #[test]
    fn a_duplicate_keeps_its_val_copy() {
        let mut ds = Vec::new();
        for t in 0..4 {
            ds.push(draft(&format!("t{t}"), "alpha", 0, &format!("ctx {t}")));
        }
        let c = cfg(Kind::Email);
        let a = assemble(&c, &ds).unwrap();
        let val_t = a
            .candidates
            .iter()
            .find(|c| c.split == "val")
            .unwrap()
            .group_key
            .clone();
        let train_t = a
            .candidates
            .iter()
            .find(|c| c.split == "train")
            .unwrap()
            .group_key
            .clone();
        // The same row text under a train template and under the val template.
        let mut dup = ds.clone();
        dup.push(draft(&train_t, "alpha", 0, "shared"));
        dup.push(draft(&val_t, "alpha", 0, "shared"));
        let b = assemble(&c, &dup).unwrap();
        let shared: Vec<&Candidate> = b
            .candidates
            .iter()
            .filter(|c| c.context == "shared")
            .collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].split, "val");
        assert_eq!(b.report["refused"]["exact_duplicate"], 1);
    }

    /// The break-it-first pair's first half: a pool whose context names its label reads
    /// ~1.0 on held-out templates, so the probe does detect a planted leak.
    #[test]
    fn the_probe_reads_a_planted_label_leak() {
        let classes = ["alpha", "beta", "gamma"];
        let mut ds = Vec::new();
        for (ci, class) in classes.iter().enumerate() {
            for t in 0..5 {
                for r in 0..12 {
                    let ctx = format!("message {t}-{r} about nothing in particular. label={class}");
                    ds.push(draft(&format!("{class}-t{t}"), class, ci, &ctx));
                }
            }
        }
        let c = cfg(Kind::Email);
        let a = assemble(&c, &ds).unwrap();
        let rows: Vec<&Candidate> = a.candidates.iter().collect();
        let p = leak_probe(&c, "t.family", &rows, 2).unwrap();
        assert_eq!(p["state"], "ran");
        assert!(p["val_accuracy"].as_f64().unwrap() >= 0.95, "{p}");
        assert_eq!(p["passed"], false);
    }
}
