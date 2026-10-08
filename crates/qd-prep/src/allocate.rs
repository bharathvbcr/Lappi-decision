//! `qd-prep allocate`: N `qd-decisions/v1` build pools in, one applied pool out (Fable's v6
//! ruling R1 and R2, `AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md`).
//!
//! The loader, the pipeline and the trainer stay single-pool. A candidate pool (`qd-prep
//! convert`, `qd-prep synth`) is refused by `qd_data.decisions.load_decision_pool` until a cap
//! table has drawn from it; this is the one place that draws, over every pool at once, so no
//! family enters a mixture weighted by how many rows its generator or source happened to yield
//! (GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06).
//!
//! - A pool drawn `as_built` (v5's) passes through whole: every row, its split, its bytes.
//! - A pool drawn `capped` has its rows refused by licence where the config says so (R2: CSN's
//!   cc-by-sa-4.0 rows), deduplicated exactly across every capped pool (the val copy kept), and
//!   drawn under one cap table keyed by family: [`largest_remainder`] shares a family's cap
//!   over its strata by train availability, and [`decisions::select`] draws each stratum and
//!   the val rows (cap and floor per family, a total) in seeded order, as `qd-prep decisions`
//!   draws v5.
//! - The rows are written through [`pool::examples`], so the structural gate and the row format
//!   have one owner. Every input row is checked to be exactly what that writer writes, and every
//!   input file to be exactly those rows' bytes, so an `as_built` row comes out byte for byte.
//!
//! Nothing here scans for contamination: each input pool ran its own scan, and the manifest
//! carries each pool's record. Near-duplicate dedupe and the defect-share bound are the
//! pipeline's (R1).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Value, json};

use crate::decisions::{self, Candidate, Gold, Role};
use crate::sha256::{Sha256, sha256_hex};
use crate::{files, heldout, pool};

/// The config's `schema`.
pub const CONFIG_SCHEMA: &str = "qd-allocation-config/v1";
/// The manifest's `tool`.
pub const TOOL: &str = "qd-prep allocate";
/// A pool whose rows pass through untouched.
pub const DRAW_AS_BUILT: &str = "as_built";
/// A pool whose rows the cap table draws from.
pub const DRAW_CAPPED: &str = "capped";
/// `qd_data.decisions.MAX_POOL_ROWS`: the loader reads a pool whole or refuses it, so a pool
/// past it is refused here rather than written.
pub const MAX_POOL_ROWS: usize = 400_000;
/// `qd_data.decisions.MAX_POOL_ROW_BYTES`: a row the loader would refuse is refused here.
pub const MAX_POOL_ROW_BYTES: usize = 1 << 20;
/// Rows read across every input pool. The real inputs of 2026-10-08 hold about 993,000.
pub const MAX_INPUT_ROWS: usize = 4_000_000;
/// Input pools in one config.
pub const MAX_POOLS: usize = 64;
/// Distinct `source_id` and `label_basis` values across every input: they are interned once
/// each, so this bounds what interning keeps.
const MAX_INTERNED: usize = 4_096;
/// Split ranks for the dedupe keep rule: held-out, then val, then train
/// (`qd_data.dedupe`'s `split_priority`). A pool row is never held-out ([`pool::examples`]
/// refuses one), so the top rank is named for the rule, not reached.
const SPLIT_RANK: [(&str, u8); 3] = [("heldout", 2), ("val", 1), ("train", 0)];

/// What the CLI passes.
pub struct Inputs {
    pub config: PathBuf,
    /// `(name, dir)` per `--pool NAME=DIR`.
    pub pools: Vec<(String, PathBuf)>,
}

/// One pool's entry in the config.
#[derive(Clone, Debug)]
pub struct PoolSpec {
    pub examples_sha256: String,
    pub as_built: bool,
}

/// `data/decisions/v6-allocation.json`.
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub val_cap_per_family: usize,
    pub val_floor_per_family: usize,
    pub val_cap_total: usize,
    pub pools: BTreeMap<String, PoolSpec>,
    /// Train rows per family, over every capped pool.
    pub train_caps: BTreeMap<String, usize>,
    /// Per family, the licences whose rows are refused.
    pub refused_licences: BTreeMap<String, BTreeSet<String>>,
    /// Families with zero train rows that come through no pool, each with its reason, recorded
    /// for the pipeline to assert (R2: `knowledge.multiple_choice`).
    pub stated_zero: BTreeMap<String, String>,
}

/// Keys a config may carry besides the required ones: prose for the reader.
const CONFIG_PROSE_KEYS: [&str; 2] = ["basis", "notes"];
const CONFIG_KEYS: [&str; 9] = [
    "schema",
    "seed",
    "val_cap_per_family",
    "val_floor_per_family",
    "val_cap_total",
    "pools",
    "train_caps",
    "refused_licences",
    "stated_zero_train_families",
];

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn object<'a>(v: &'a Value, key: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    decisions::get(v, key, "config")?
        .as_object()
        .ok_or_else(|| format!("config: {key:?} is not an object"))
}

impl Config {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("config: {e}"))?;
        let top = v.as_object().ok_or("config: not a JSON object")?;
        // Every key is required or prose: a misspelt optional key would otherwise be a rule
        // silently not applied.
        let unknown: Vec<&String> = top
            .keys()
            .filter(|k| !CONFIG_KEYS.contains(&k.as_str()) && !CONFIG_PROSE_KEYS.contains(&k.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(format!("config: unknown key(s) {unknown:?}"));
        }
        if decisions::get_str(&v, "schema", "config")? != CONFIG_SCHEMA {
            return Err(format!("config: schema is not {CONFIG_SCHEMA:?}"));
        }
        let count = |key: &str| -> Result<usize, String> {
            usize::try_from(decisions::get_u64(&v, key, "config")?)
                .map_err(|_| format!("config: {key} does not fit a usize"))
        };
        let mut pools = BTreeMap::new();
        for (name, p) in object(&v, "pools")? {
            let what = format!("config pools.{name}");
            let o = p.as_object().ok_or_else(|| format!("{what}: not an object"))?;
            if let Some(k) = o.keys().find(|k| !["examples_sha256", "draw", "note"].contains(&k.as_str())) {
                return Err(format!("{what}: unknown key {k:?}"));
            }
            let sha = decisions::get_str(p, "examples_sha256", &what)?;
            if !is_sha256_hex(sha) {
                return Err(format!("{what}: examples_sha256 {sha:?} is not 64 lower-case hex digits"));
            }
            let as_built = match decisions::get_str(p, "draw", &what)? {
                DRAW_AS_BUILT => true,
                DRAW_CAPPED => false,
                other => {
                    return Err(format!(
                        "{what}: draw {other:?} is neither {DRAW_AS_BUILT:?} nor {DRAW_CAPPED:?}"
                    ));
                }
            };
            pools.insert(name.clone(), PoolSpec { examples_sha256: sha.to_owned(), as_built });
        }
        if pools.is_empty() || pools.len() > MAX_POOLS {
            return Err(format!("config: {} pools; 1 to {MAX_POOLS} are allowed", pools.len()));
        }
        let train_caps = object(&v, "train_caps")?
            .iter()
            .map(|(k, x)| {
                x.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .map(|n| (k.clone(), n))
                    .ok_or_else(|| format!("config: train_caps.{k} is not a non-negative integer"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut refused_licences = BTreeMap::new();
        for (family, list) in object(&v, "refused_licences")? {
            let arr = list
                .as_array()
                .ok_or_else(|| format!("config: refused_licences.{family} is not a list"))?;
            let set = arr
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("config: refused_licences.{family} holds a non-string"))
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            if set.is_empty() || set.len() != arr.len() {
                return Err(format!(
                    "config: refused_licences.{family} is empty or repeats a licence"
                ));
            }
            // A licence rule applies to drawn rows only: an as-built pool passes untouched.
            if !train_caps.contains_key(family) {
                return Err(format!(
                    "config: refused_licences names {family:?}, which train_caps does not cap; \
                     a licence is refused only in a capped family"
                ));
            }
            refused_licences.insert(family.clone(), set);
        }
        let stated_zero = decisions::str_map(&v, "stated_zero_train_families")?;
        if let Some(f) = stated_zero.keys().find(|f| train_caps.contains_key(*f)) {
            return Err(format!(
                "config: {f:?} is both a stated zero and in train_caps; a stated zero comes \
                 through no pool"
            ));
        }
        let c = Config {
            seed: decisions::get_u64(&v, "seed", "config")?,
            val_cap_per_family: count("val_cap_per_family")?,
            val_floor_per_family: count("val_floor_per_family")?,
            val_cap_total: count("val_cap_total")?,
            pools,
            train_caps,
            refused_licences,
            stated_zero,
        };
        if c.val_floor_per_family > c.val_cap_per_family {
            return Err("config: val_floor_per_family exceeds val_cap_per_family".into());
        }
        Ok(c)
    }
}

/// `source_id` and `label_basis` as the `&'static str` a [`Candidate`] holds: each distinct
/// value is leaked once, and their number is bounded.
struct Interner(Mutex<BTreeSet<&'static str>>);

impl Interner {
    fn get(&self, s: &str) -> Result<&'static str, String> {
        let mut set = self.0.lock().map_err(|_| "interner lock poisoned".to_owned())?;
        if let Some(hit) = set.get(s) {
            return Ok(hit);
        }
        if set.len() >= MAX_INTERNED {
            return Err(format!(
                "more than {MAX_INTERNED} distinct source_id/label_basis values across the pools"
            ));
        }
        let leaked: &'static str = Box::leak(s.to_owned().into_boxed_str());
        set.insert(leaked);
        Ok(leaked)
    }
}

/// One pool row as a [`Candidate`], refused unless [`decisions::example_json`] writes exactly
/// `v` back: so a row with a field the writer does not write, or one it would write
/// differently, is refused rather than altered on the way through.
fn candidate(v: &Value, what: &str, interner: &Interner) -> Result<Candidate, String> {
    let s = |key: &str| decisions::get_str(v, key, what);
    let rendered = s("context")?;
    // `example_json` writes `question + "\n\n" + context`; splitting at the first blank line
    // gives back a pair that renders the same string, whatever the question held.
    let (question, context) = rendered
        .split_once("\n\n")
        .ok_or_else(|| format!("{what}: context has no question paragraph"))?;
    let options = decisions::get(v, "options", what)?
        .as_array()
        .ok_or_else(|| format!("{what}: \"options\" is not a list"))?
        .iter()
        .map(|o| o.as_str().map(str::to_owned).ok_or_else(|| format!("{what}: an option is not a string")))
        .collect::<Result<Vec<_>, _>>()?;
    let noul = decisions::get(v, "gold_noul", what)?
        .as_bool()
        .ok_or_else(|| format!("{what}: \"gold_noul\" is not a bool"))?;
    let gold = match (noul, decisions::get(v, "gold_option", what)?) {
        (true, Value::Null) => Gold::Noul,
        (false, Value::String(g)) => Gold::Option(
            options
                .iter()
                .position(|o| o == g)
                .ok_or_else(|| format!("{what}: gold_option {g:?} is not an option"))?,
        ),
        _ => return Err(format!("{what}: gold_noul and gold_option disagree")),
    };
    let split = match s("split")? {
        "train" => "train",
        "val" => "val",
        other => return Err(format!("{what}: split {other:?}; a pool holds only train and val rows")),
    };
    let c = Candidate {
        id: s("id")?.to_owned(),
        source_id: interner.get(s("source_id")?)?,
        family_id: s("family_id")?.to_owned(),
        stratum: s("stratum")?.to_owned(),
        group_key: s("group_key")?.to_owned(),
        licence: s("licence")?.to_owned(),
        context: context.to_owned(),
        question: question.to_owned(),
        slot_name: s("slot_name")?.to_owned(),
        options,
        gold,
        label_basis: interner.get(s("label_basis")?)?,
        split,
    };
    if let Some(reason) = decisions::structural_refusal(&c) {
        return Err(format!("{what}: {reason}"));
    }
    if decisions::example_json(&c) != *v {
        return Err(format!(
            "{what}: the row is not what the pool seam writes (a field added, missing or \
             retyped); it cannot pass through unaltered"
        ));
    }
    Ok(c)
}

/// One input pool, read whole.
struct PoolRead {
    rows: Vec<Candidate>,
    examples: decisions::Lines,
    manifest_sha256: String,
    /// The input manifest's `decontamination` and `allocation` blocks, carried into the record.
    decontamination: Value,
    allocation: Value,
}

/// Read and check one pool: no held-out path; its manifest a `qd-decisions/v1` build naming
/// these examples; the examples' sha256 the config's pin; every row the pool seam's bytes.
fn read_pool(
    name: &str,
    dir: &Path,
    spec: &PoolSpec,
    interner: &Interner,
) -> Result<PoolRead, String> {
    let what = format!("--pool {name}");
    let examples_path = dir.join("examples.jsonl");
    let manifest_path = dir.join("manifest.json");
    for p in [dir, examples_path.as_path(), manifest_path.as_path()] {
        heldout::refuse_training_input(p, &what)?;
    }
    let manifest_bytes = files::read_bounded(&manifest_path, files::MAX_RECORD_BYTES, &what)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(decisions::MANIFEST_SCHEMA)
        || manifest.get("mode").and_then(Value::as_str) != Some("build")
    {
        return Err(format!(
            "{}: not a {:?} build manifest",
            manifest_path.display(),
            decisions::MANIFEST_SCHEMA
        ));
    }
    let mut rows = Vec::new();
    // The digest of every row as the pool seam writes it, one `\n` after each. Equal to the
    // file's own digest only when the file is exactly those bytes: no blank line, no CRLF, no
    // torn last line, no row serialised another way.
    let mut canonical = Sha256::new();
    let examples = decisions::for_lines(&examples_path, Role::Reference, |n, v| {
        let at = format!("{}:{n}", examples_path.display());
        let bytes = serde_json::to_vec(&v).map_err(|e| format!("{at}: {e}"))?;
        if bytes.len() + 1 > MAX_POOL_ROW_BYTES {
            return Err(format!("{at}: a row over {MAX_POOL_ROW_BYTES} bytes"));
        }
        canonical.update(&bytes);
        canonical.update(b"\n");
        if rows.len() >= MAX_INPUT_ROWS {
            return Err(format!("{at}: more than {MAX_INPUT_ROWS} rows"));
        }
        rows.push(candidate(&v, &at, interner)?);
        Ok(())
    })?;
    let canonical = canonical.hex();
    decisions::check_pin(
        &examples_path,
        &examples.sha256,
        &spec.examples_sha256,
        &format!("the config (pool {name})"),
    )?;
    decisions::check_pin(
        &examples_path,
        &examples.sha256,
        decisions::get_str(&manifest, "examples_sha256", &manifest_path.display().to_string())?,
        &format!("its manifest {}", manifest_path.display()),
    )?;
    if canonical != examples.sha256 {
        return Err(format!(
            "{}: its bytes are not its rows as the pool seam writes them (sha256 {} read, {} \
             rewritten): a blank line, a CRLF, a torn last line or another serialisation, so its \
             rows cannot be passed on byte for byte",
            examples_path.display(),
            examples.sha256,
            canonical
        ));
    }
    let recorded = manifest.get("examples").and_then(Value::as_u64);
    if recorded != Some(rows.len() as u64) {
        return Err(format!(
            "{}: {} rows, its manifest records {recorded:?}",
            examples_path.display(),
            rows.len()
        ));
    }
    Ok(PoolRead {
        rows,
        examples,
        manifest_sha256: sha256_hex(&manifest_bytes),
        decontamination: manifest.get("decontamination").cloned().unwrap_or(Value::Null),
        allocation: manifest.get("allocation").cloned().unwrap_or(Value::Null),
    })
}

/// Every pool, read on up to `threads` threads. The result is in `order`, whatever the thread
/// count; the first refusal in that order is the one reported.
fn read_pools(
    order: &[(&str, &Path, &PoolSpec)],
    threads: usize,
) -> Result<Vec<PoolRead>, String> {
    let interner = Interner(Mutex::new(BTreeSet::new()));
    let workers = threads.clamp(1, order.len().max(1));
    let mut slots: Vec<Option<Result<PoolRead, String>>> = (0..order.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let interner = &interner;
                scope.spawn(move || {
                    (w..order.len())
                        .step_by(workers)
                        .map(|i| {
                            let (name, dir, spec) = order[i];
                            (i, read_pool(name, dir, spec, interner))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(done) => {
                    for (i, r) in done {
                        slots[i] = Some(r);
                    }
                }
                Err(_) => return Err("a pool reader panicked".to_owned()),
            }
        }
        Ok(())
    })?;
    slots
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err("a pool was not read".to_owned())))
        .collect()
}

/// `n` shared over `avail` by largest remainder: each share is `n * a / sum` rounded down,
/// and the rows left go one each to the largest remainders, ties to the earlier entry. When
/// `n` covers the sum, every entry gets all it has. Exact integer arithmetic; the same rule as
/// [`decisions::select`]'s val apportionment, which does it in floating point.
pub fn largest_remainder(n: usize, avail: &[usize]) -> Vec<usize> {
    let sum: usize = avail.iter().sum();
    if n >= sum {
        return avail.to_vec();
    }
    let (n, sum) = (n as u128, sum as u128);
    let mut take: Vec<usize> = avail.iter().map(|a| (n * *a as u128 / sum) as usize).collect();
    let rem: Vec<u128> = avail.iter().map(|a| n * *a as u128 % sum).collect();
    let mut rest = n as usize - take.iter().sum::<usize>();
    let mut by_rem: Vec<usize> = (0..avail.len()).collect();
    by_rem.sort_by(|&a, &b| rem[b].cmp(&rem[a]).then(a.cmp(&b)));
    for k in by_rem {
        if rest == 0 {
            break;
        }
        if take[k] < avail[k] {
            take[k] += 1;
            rest -= 1;
        }
    }
    take
}

/// The exact-dedupe key: the row's family, group and rendered text (context and options),
/// length-prefixed and hashed. `qd_data`'s `identity_key` less its split segment: the split is
/// what the keep rule chooses between, so it cannot be part of the key.
fn identity_key(c: &Candidate) -> [u8; 32] {
    let mut h = Sha256::new();
    let rendered = c.rendered_context();
    let parts = [c.family_id.as_str(), c.group_key.as_str(), rendered.as_str()];
    for p in parts.iter().copied().chain(c.options.iter().map(String::as_str)) {
        h.update(&(p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    h.finish()
}

fn split_rank(split: &str) -> u8 {
    SPLIT_RANK
        .iter()
        .find(|(s, _)| *s == split)
        .map_or(0, |(_, r)| *r)
}

/// The family a stratum belongs to, as [`decisions::select`] reads it.
fn stratum_family(stratum: &str) -> &str {
    stratum.split('/').next().unwrap_or(stratum)
}

/// What the run writes, before it is written.
pub struct Built {
    pub examples: Vec<u8>,
    pub manifest: Value,
    pub summary: String,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    train: usize,
    val: usize,
}

impl Tally {
    fn add(&mut self, split: &str) {
        if split == "val" {
            self.val += 1;
        } else {
            self.train += 1;
        }
    }

    fn json(self) -> Value {
        json!({"train": self.train, "val": self.val})
    }
}


/// One capped family's draw, for the manifest.
#[derive(Default)]
struct FamilyDraw {
    cap: usize,
    /// Per stratum: `(available, stratum cap, taken)`, train and val.
    strata: BTreeMap<String, (Tally, usize, Tally)>,
}

impl FamilyDraw {
    fn json(&self) -> Value {
        let (mut avail, mut taken) = (Tally::default(), Tally::default());
        let mut val_untrained = 0usize;
        let strata: serde_json::Map<String, Value> = self
            .strata
            .iter()
            .map(|(s, (a, cap, t))| {
                avail.train += a.train;
                avail.val += a.val;
                taken.train += t.train;
                taken.val += t.val;
                if *cap == 0 {
                    val_untrained += a.val;
                }
                (
                    s.clone(),
                    json!({"available": a.json(), "train_cap": cap, "taken": t.json()}),
                )
            })
            .collect();
        json!({
            "cap": self.cap,
            "available": avail.json(),
            "taken": taken.json(),
            // A cap over what the family has takes all of it; the shortfall is named.
            "train_shortfall": self.cap.saturating_sub(avail.train),
            // `decisions::select` draws no val row from a stratum whose train cap is 0, so val
            // never measures a stratum train does not contain.
            "val_rows_in_strata_capped_at_zero": val_untrained,
            "strata": strata,
        })
    }
}

/// Build the applied pool from `inputs`.
pub fn build(inputs: &Inputs, threads: usize) -> Result<Built, String> {
    heldout::refuse_training_input(&inputs.config, "--config")?;
    let config_bytes = files::read_bounded(&inputs.config, files::MAX_RECORD_BYTES, "--config")?;
    let cfg = Config::parse(&config_bytes)?;

    // The pools given are exactly the pools configured, each once.
    let mut given: BTreeMap<&str, &Path> = BTreeMap::new();
    for (name, dir) in &inputs.pools {
        if given.insert(name.as_str(), dir.as_path()).is_some() {
            return Err(format!("--pool {name} is given twice"));
        }
    }
    let extra: Vec<&str> = given
        .keys()
        .filter(|n| !cfg.pools.contains_key(**n))
        .copied()
        .collect();
    let absent: Vec<&String> = cfg
        .pools
        .keys()
        .filter(|n| !given.contains_key(n.as_str()))
        .collect();
    if !extra.is_empty() || !absent.is_empty() {
        return Err(format!(
            "the --pool set is not the config's pools: given and not configured {extra:?}, \
             configured and not given {absent:?}"
        ));
    }
    let order: Vec<(&str, &Path, &PoolSpec)> = cfg
        .pools
        .iter()
        .map(|(n, spec)| (n.as_str(), given[n.as_str()], spec))
        .collect();
    let mut reads = read_pools(&order, threads)?;
    let total_read: usize = reads.iter().map(|r| r.rows.len()).sum();
    if total_read > MAX_INPUT_ROWS {
        return Err(format!(
            "{total_read} rows across the pools; the bound is {MAX_INPUT_ROWS}"
        ));
    }

    // Which pools carry which families, and the family rules over them.
    let mut family_pools: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (p, r) in reads.iter().enumerate() {
        let fams: BTreeSet<&str> = r.rows.iter().map(|c| c.family_id.as_str()).collect();
        for f in fams {
            family_pools.entry(f.to_owned()).or_default().push(p);
        }
    }
    for (family, in_pools) in &family_pools {
        let names: Vec<&str> = in_pools.iter().map(|p| order[*p].0).collect();
        let as_built = in_pools.iter().filter(|p| order[**p].2.as_built).count();
        if cfg.stated_zero.contains_key(family) {
            return Err(format!(
                "{family:?} is a stated zero (it comes through no pool), but pool(s) {names:?} \
                 carry it"
            ));
        }
        if as_built > 0 && in_pools.len() > 1 {
            return Err(format!(
                "family {family:?} is in pools {names:?}, one of them as built: an as-built \
                 family passes through whole from one pool, so it is neither drawn nor passed \
                 through twice"
            ));
        }
        if as_built > 0 && cfg.train_caps.contains_key(family) {
            return Err(format!(
                "family {family:?} has a train cap but comes only through the as-built pool \
                 {names:?}, whose rows pass untouched"
            ));
        }
        if as_built == 0 && !cfg.train_caps.contains_key(family) {
            return Err(format!(
                "family {family:?} (pool(s) {names:?}) has no entry in train_caps. Every drawn \
                 family is capped explicitly (0 excludes it), so nothing enters or drops \
                 silently"
            ));
        }
    }

    // The record of each input, then its rows: as-built ones aside, capped ones through the
    // licence refusals and the exact dedupe across every capped pool.
    let mut digests: BTreeMap<String, String> = BTreeMap::new();
    digests.insert("config".to_owned(), sha256_hex(&config_bytes));
    let mut pools_record = serde_json::Map::new();
    let mut as_built_rows: Vec<(usize, usize, Candidate)> = Vec::new();
    let mut refused: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    // (pool, line index, row, dropped) of every capped row past the licence rule.
    let mut kept: Vec<(usize, usize, Candidate, bool)> = Vec::new();
    let mut by_key: BTreeMap<[u8; 32], usize> = BTreeMap::new();
    let mut dedupe_by_family: BTreeMap<String, Tally> = BTreeMap::new();
    let (mut cross_pool, mut within_pool) = (0usize, 0usize);
    for (p, r) in reads.iter_mut().enumerate() {
        let (name, dir, spec) = order[p];
        let mut fam_split: BTreeMap<&str, Tally> = BTreeMap::new();
        for c in &r.rows {
            fam_split.entry(c.family_id.as_str()).or_default().add(c.split);
        }
        pools_record.insert(
            name.to_owned(),
            json!({
                "draw": if spec.as_built { DRAW_AS_BUILT } else { DRAW_CAPPED },
                "path": dir.display().to_string(),
                "rows": r.rows.len(),
                "rows_by_family_split": fam_split.iter().map(|(k, t)| ((*k).to_owned(), t.json()))
                    .collect::<serde_json::Map<_, _>>(),
                "examples_sha256": r.examples.sha256,
                "manifest_sha256": r.manifest_sha256,
                "input_allocation": r.allocation,
                "decontamination": r.decontamination,
            }),
        );
        decisions::record_input(
            &mut digests,
            format!("pool/{name}/examples.jsonl"),
            r.examples.clone(),
        );
        digests.insert(format!("pool/{name}/manifest.json"), r.manifest_sha256.clone());
        let rows = std::mem::take(&mut r.rows);
        if spec.as_built {
            as_built_rows.extend(rows.into_iter().enumerate().map(|(i, c)| (p, i, c)));
            continue;
        }
        for (i, c) in rows.into_iter().enumerate() {
            if stratum_family(&c.stratum) != c.family_id {
                return Err(format!(
                    "pool {name} row {}: stratum {:?} is not under its family {:?}; the caps \
                     read a stratum's family from its first segment",
                    c.id, c.stratum, c.family_id
                ));
            }
            if cfg
                .refused_licences
                .get(&c.family_id)
                .is_some_and(|l| l.contains(&c.licence))
            {
                *refused
                    .entry(c.family_id.clone())
                    .or_default()
                    .entry(c.licence.clone())
                    .or_default() += 1;
                continue;
            }
            let key = identity_key(&c);
            let Some(k) = by_key.get(&key).copied() else {
                by_key.insert(key, kept.len());
                kept.push((p, i, c, false));
                continue;
            };
            // The more protected split survives; among equals the earlier row (pool order,
            // then file order), so the survivor is a function of the inputs alone.
            let new_wins = split_rank(c.split) > split_rank(kept[k].2.split);
            let (winner_pool, loser_pool, loser) = if new_wins {
                (p, kept[k].0, &kept[k].2)
            } else {
                (kept[k].0, p, &c)
            };
            if loser_pool == winner_pool {
                within_pool += 1;
            } else {
                cross_pool += 1;
            }
            dedupe_by_family
                .entry(loser.family_id.clone())
                .or_default()
                .add(loser.split);
            if new_wins {
                kept[k].3 = true;
                by_key.insert(key, kept.len());
                kept.push((p, i, c, false));
            }
        }
    }
    drop(reads);
    let mut origins: Vec<(usize, usize)> = Vec::new();
    let mut cands: Vec<Candidate> = Vec::new();
    for (p, i, c, gone) in kept {
        if !gone {
            origins.push((p, i));
            cands.push(c);
        }
    }

    // Per family, its cap shared over its strata by train availability.
    let mut draws: BTreeMap<String, FamilyDraw> = BTreeMap::new();
    for c in &cands {
        let d = draws.entry(c.family_id.clone()).or_default();
        d.strata.entry(c.stratum.clone()).or_default().0.add(c.split);
    }
    let empty: Vec<&String> = cfg
        .train_caps
        .iter()
        .filter(|(f, cap)| **cap > 0 && !draws.contains_key(*f))
        .map(|(f, _)| f)
        .collect();
    if !empty.is_empty() {
        return Err(format!(
            "train_caps gives a non-zero cap to {} famil(ies) with no candidate rows in any \
             capped pool (after licence refusals and dedupe): {empty:?}. A family no pool \
             carries is listed with cap 0",
            empty.len()
        ));
    }
    let mut stratum_caps: BTreeMap<String, usize> = BTreeMap::new();
    for (family, d) in draws.iter_mut() {
        d.cap = cfg.train_caps[family];
        let avail: Vec<usize> = d.strata.values().map(|(a, _, _)| a.train).collect();
        for ((stratum, entry), cap) in d
            .strata
            .iter_mut()
            .zip(largest_remainder(d.cap, &avail))
        {
            entry.1 = cap;
            stratum_caps.insert(stratum.clone(), cap);
        }
    }
    // `decisions::select` reads the seed, the val caps and train_caps (per stratum here); the
    // other fields are the v5 source reader's and select never reads them.
    let select_cfg = decisions::Config {
        seed: cfg.seed,
        val_fraction: 0.0,
        mode_threshold: 0.0,
        pairwise_length_band: None,
        ngram_n: 0,
        containment_threshold: 0.0,
        fetch_record_sha256: String::new(),
        decider_sha256: BTreeMap::new(),
        target_sha256: BTreeMap::new(),
        val_cap_per_family: cfg.val_cap_per_family,
        val_floor_per_family: cfg.val_floor_per_family,
        val_cap_total: cfg.val_cap_total,
        train_caps: stratum_caps,
    };
    let chosen = decisions::select(&select_cfg, &cands)?;

    // The pool: every as-built row and every drawn row, in pool order then file order.
    let mut out: Vec<((usize, usize), &Candidate)> = as_built_rows
        .iter()
        .map(|(p, i, c)| ((*p, *i), c))
        .chain(chosen.iter().map(|&j| (origins[j], &cands[j])))
        .collect();
    out.sort_by_key(|(at, _)| *at);
    if out.len() > MAX_POOL_ROWS {
        return Err(format!(
            "{} rows would be written; the loader reads at most {MAX_POOL_ROWS}",
            out.len()
        ));
    }
    let mut ids: BTreeMap<&str, usize> = BTreeMap::new();
    for (at, c) in &out {
        if let Some(first) = ids.insert(c.id.as_str(), at.0) {
            return Err(format!(
                "id {:?} is in pool {} and pool {}: a pool row id names one row",
                c.id, order[first].0, order[at.0].0
            ));
        }
    }
    let refs: Vec<&Candidate> = out.iter().map(|(_, c)| *c).collect();
    let ex = pool::examples(&refs)?;

    let mut rows_by_family: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut text_bytes: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &refs {
        rows_by_family
            .entry(c.family_id.as_str())
            .or_default()
            .add(c.split);
        *text_bytes.entry(c.family_id.as_str()).or_default() += c.text().len();
    }
    for &j in &chosen {
        let c = &cands[j];
        if let Some(d) = draws.get_mut(&c.family_id)
            && let Some(s) = d.strata.get_mut(&c.stratum)
        {
            s.2.add(c.split);
        }
    }
    let refused_total: usize = refused.values().flat_map(BTreeMap::values).sum();
    let dropped_total = cross_pool + within_pool;
    let manifest = json!({
        "schema": decisions::MANIFEST_SCHEMA, "mode": "build", "tool": TOOL,
        "tool_version": env!("CARGO_PKG_VERSION"), "seed": cfg.seed,
        "inputs": digests,
        "pools": pools_record,
        "train_caps": cfg.train_caps,
        "val_caps": {"per_family": cfg.val_cap_per_family,
                     "floor_per_family": cfg.val_floor_per_family,
                     "total": cfg.val_cap_total,
                     "applies_to": "capped pools; an as-built pool's val rows pass untouched"},
        "caps_applied": draws.iter().map(|(f, d)| (f.clone(), d.json()))
            .collect::<serde_json::Map<_, _>>(),
        "listed_zero_without_candidates": cfg.train_caps.iter()
            .filter(|(f, cap)| **cap == 0 && !draws.contains_key(*f)).map(|(f, _)| f)
            .collect::<Vec<_>>(),
        "stated_zero_train_families": cfg.stated_zero,
        "refusals": {"licence": refused, "total": refused_total},
        "dedupe": {
            "key": "family_id, group_key, rendered context and options (qd_data's identity_key \
                    less its split), sha256, across every capped pool",
            "keep_rule": "split_priority: held-out, then val, then train; among equals the \
                          earlier row in pool then file order",
            "dropped_total": dropped_total,
            "dropped_cross_pool": cross_pool,
            "dropped_within_pool": within_pool,
            "dropped_by_family_split": dedupe_by_family.iter()
                .map(|(f, t)| (f.clone(), t.json())).collect::<serde_json::Map<_, _>>(),
            "near_duplicates": "not searched here: the pipeline's dedupe under --v6-data-rules (R1)",
        },
        "rows_by_family_split": rows_by_family.iter().map(|(f, t)| ((*f).to_owned(), t.json()))
            .collect::<serde_json::Map<_, _>>(),
        "as_built_rows": as_built_rows.len(),
        "drawn_rows": chosen.len(),
        "drawn_from_candidates": cands.len(),
        "selected": ex.selected,
        "noul_gold_selected": ex.noul,
        "gold_position": ex.gold_position,
        "val_without_train": ex.val_without_train,
        "text_bytes_by_family": text_bytes,
        "text_bytes_by_family_note": "report only: a pre-estimate. The defect-share bound is \
                                      measured and enforced in the pipeline (R1)",
        "decontamination_note": "allocate runs no scan; each input pool's own record is under \
                                 pools.NAME.decontamination",
        "examples": ex.rows, "examples_sha256": ex.sha256,
        "allocation": pool::allocation(true, "qd-prep allocate: the pinned config's train_caps \
                 (per family, shared over strata by largest remainder), val caps and floors \
                 drew the capped pools; the as-built pools passed through whole"),
    });
    let summary = format!(
        "qd-prep allocate: {} rows ({} as built, {} drawn of {} candidates); {refused_total} \
         refused by licence; {dropped_total} dropped as exact duplicates ({cross_pool} across \
         pools)",
        ex.rows,
        as_built_rows.len(),
        chosen.len(),
        cands.len()
    );
    Ok(Built {
        examples: ex.bytes,
        manifest,
        summary,
    })
}

/// `qd-prep allocate`: build the pool, then write `OUT/{examples.jsonl, manifest.json}` whole
/// or not at all.
pub fn run(inputs: &Inputs, out_dir: &Path, threads: usize) -> Result<String, String> {
    // Training data under a held-out marker is refused by every reader; refused here first.
    heldout::refuse_training_input(out_dir, "--out")?;
    let partial = files::partial_path(out_dir)?;
    for p in [out_dir, partial.as_path()] {
        if p.exists() {
            return Err(format!("{} exists; refusing to overwrite it", p.display()));
        }
    }
    let built = build(inputs, threads)?;
    let mut manifest = serde_json::to_vec_pretty(&built.manifest).map_err(|e| e.to_string())?;
    manifest.push(b'\n');
    files::write_new_dir(out_dir, |dir| {
        files::write_synced_new(&dir.join("examples.jsonl"), &built.examples)?;
        files::write_synced_new(&dir.join("manifest.json"), &manifest)
    })?;
    Ok(format!(
        "{} on {threads} thread(s) -> {}",
        built.summary,
        out_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_remainder_gives_the_leftover_to_the_largest_remainders_ties_first() {
        assert_eq!(largest_remainder(25, &[60, 30, 10]), [15, 8, 2]);
        assert_eq!(largest_remainder(100, &[60, 30, 10]), [60, 30, 10]);
        assert_eq!(largest_remainder(500, &[60, 30, 10]), [60, 30, 10]);
        assert_eq!(largest_remainder(0, &[60, 30, 10]), [0, 0, 0]);
        assert_eq!(largest_remainder(2, &[1, 1, 1]), [1, 1, 0]);
        assert_eq!(largest_remainder(1, &[0, 5]), [0, 1]);
        let got = largest_remainder(15_000, &[40_000, 30_000, 20_000, 11_111, 7]);
        assert_eq!(got.iter().sum::<usize>(), 15_000);
    }

    #[test]
    fn the_config_refuses_an_unknown_key_and_a_bad_pin_or_draw() {
        let base = json!({
            "schema": CONFIG_SCHEMA, "seed": 1, "val_cap_per_family": 10,
            "val_floor_per_family": 1, "val_cap_total": 100,
            "pools": {"p": {"examples_sha256": "a".repeat(64), "draw": "capped"}},
            "train_caps": {"f": 1}, "refused_licences": {}, "stated_zero_train_families": {},
        });
        assert!(Config::parse(&serde_json::to_vec(&base).unwrap()).is_ok());
        let with = |k: &str, v: Value| {
            let mut c = base.clone();
            c[k] = v;
            Config::parse(&serde_json::to_vec(&c).unwrap())
        };
        assert!(with("train_cap", json!({})).unwrap_err().contains("unknown key"));
        assert!(
            with("pools", json!({"p": {"examples_sha256": "A".repeat(64), "draw": "capped"}}))
                .is_err()
        );
        assert!(
            with("pools", json!({"p": {"examples_sha256": "a".repeat(64), "draw": "all"}}))
                .is_err()
        );
        assert!(with("refused_licences", json!({"g": ["x"]})).is_err(), "uncapped family");
        assert!(with("stated_zero_train_families", json!({"f": "why"})).is_err());
        assert!(with("val_floor_per_family", json!(11)).is_err());
    }

    #[test]
    fn a_row_the_seam_would_not_write_back_is_refused() {
        let interner = Interner(Mutex::new(BTreeSet::new()));
        let row = json!({
            "id": "a", "source_id": "s", "family_id": "f", "stratum": "f/x", "split": "train",
            "group_key": "g", "licence": "mit", "context": "Q?\n\nbody", "slot_name": "answer",
            "options": ["x", "y"], "gold_option": "y", "gold_noul": false, "label_basis": "hard",
        });
        let c = candidate(&row, "t", &interner).unwrap();
        assert_eq!(c.gold, Gold::Option(1));
        assert_eq!(decisions::example_json(&c), row);
        let mut extra = row.clone();
        extra["len_ratio"] = json!(1.5);
        assert!(candidate(&extra, "t", &interner).unwrap_err().contains("pool seam"));
        let mut held = row.clone();
        held["split"] = json!("heldout");
        assert!(candidate(&held, "t", &interner).is_err());
        let mut bad_gold = row;
        bad_gold["gold_option"] = json!("z");
        assert!(candidate(&bad_gold, "t", &interner).is_err());
    }
}
