//! `qd-prep containment`: the v5 decontamination scan (Fable's 2026-10-02 ruling, Q2 "The
//! rule, end to end"), as `qd_train.replay.decontaminate` defines it, with the COMPLETE pair
//! list.
//!
//! A source row hits a target row when it contains at least `threshold` of the target row's
//! word n-grams: `re.findall(r"\w+", text.lower())`, every run of `n` consecutive words joined
//! by one space, each hashed with `blake2b(digest_size=8)` ([`crate::pyunicode`] is Python's
//! `\w` and `lower()`; [`crate::blake2b`] its BLAKE2b). Containment is the count of the target
//! row's DISTINCT n-grams the source row also has, over the target row's distinct count, as a
//! float64 quotient compared with `>=` -- the oracle's `c / size[t] >= threshold`, bit for bit.
//!
//! What the oracle returns per call, this returns per SCAN (a source set against a target
//! set), so one request can carry the whole rule: train against every val row and every
//! held-out row, enforced or reported, and val against held-out, reported. The pair list is
//! every (source row, target row) at or over the threshold -- not the best target per row --
//! in the oracle's order: source sets in the order their first scan names them, each set's rows
//! by key (bytewise, which is Python's code-point order), then that source's scans in request
//! order, then target rows by key. `exclusions.txt` is the identity keys of the source rows
//! of every ENFORCED scan with a pair, byte-sorted, one per line, LF.
//!
//! Keys (i) exact identity/content equality and (iii) MinHash near-duplicates are the
//! splitter's (`qd_data.split`'s `identity_disjoint` and `near_duplicate_disjoint`, computed
//! with this crate's `minhash` and `lsh`); the request carries their tri-states and the
//! attestation repeats them, and it is clean only when both ran and passed.
//!
//! Request (`QDPCTIN1`), little-endian; `STR` is `len u32` then that many UTF-8 bytes:
//!
//! ```text
//! magic 8 | n u32 | threshold f64 | corpus STR (a JSON object, written into the attestation)
//!         | export STR (a JSON object: what the exporter left out and why, written beside it)
//!         | n_checks u32, per check: name STR, state u8 (0 not run, 1 ran and failed,
//!           2 ran and passed), detail STR
//!         | n_sets u32, per set: name STR
//!         | n_scans u32, per scan: source u32, target u32, enforced u8
//!         | n_rows u64, per row: set u32, key STR, identity_key STR, family STR, text STR
//! ```
//!
//! Output: a directory holding `pairs.tsv`, `exclusions.txt` and `attestation.json`
//! (version 2), written as `DIR.partial` and renamed, so it appears whole or not at all.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use crate::blake2b::Keyed;
use crate::pyunicode;
use crate::sha256::{Sha256, sha256_hex};
use crate::wire::Cursor;

pub const INPUT_MAGIC: &[u8; 8] = b"QDPCTIN1";
/// A full v5 request is every rendered train, val and held-out slot: a few GB of text.
pub const MAX_INPUT_BYTES: u64 = 32 << 30;
/// `qd_train.replay.MAX_ROWS_PER_SIDE`: rows per set.
pub const MAX_ROWS_PER_SET: u64 = 2_000_000;
/// The longest word n-gram accepted; the rule's is 8.
pub const MAX_N: u32 = 64;
/// Sets, scans and splitter checks per request: the rule has a handful of each.
pub const MAX_SETS: u32 = 64;
pub const MAX_CHECKS: u32 = 64;
/// One string's bytes; a rendered prompt is at most a few hundred kilobytes.
pub const MAX_STR_BYTES: u32 = 64 << 20;
pub const ATTESTATION_VERSION: u32 = 2;
pub const PAIRS_NAME: &str = "pairs.tsv";
pub const EXCLUSIONS_NAME: &str = "exclusions.txt";
pub const ATTESTATION_NAME: &str = "attestation.json";
pub const PAIRS_HEADER: &str = "source_set\tsource_key\tsource_identity_key\tsource_family\t\
                                target_set\ttarget_key\ttarget_family\tshared\ttarget_ngrams\n";
/// Source rows per work item.
const CHUNK_ROWS: usize = 256;

/// A splitter post-condition as the request states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckState {
    NotRun,
    Failed,
    Passed,
}

#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub state: CheckState,
    pub detail: String,
}

#[derive(Clone, Copy, Debug)]
pub struct Scan {
    pub source: usize,
    pub target: usize,
    pub enforced: bool,
}

#[derive(Clone, Debug)]
pub struct Row<'b> {
    pub key: &'b str,
    pub identity: &'b str,
    pub family: &'b str,
    pub text: &'b str,
}

/// A parsed, validated request. Rows are grouped by set, each set's rows sorted by key.
#[derive(Debug)]
pub struct Request<'b> {
    pub n: usize,
    pub threshold: f64,
    pub corpus: &'b str,
    pub export: &'b str,
    pub checks: Vec<Check>,
    pub set_names: Vec<&'b str>,
    pub scans: Vec<Scan>,
    pub sets: Vec<Vec<Row<'b>>>,
}

fn take_str<'b>(c: &mut Cursor<'b>, what: &str) -> Result<&'b str, String> {
    let len = c.u32(what)?;
    if len > MAX_STR_BYTES {
        return Err(format!("{what}: {len} bytes; the bound is {MAX_STR_BYTES}"));
    }
    let range = c.take(len as usize, what)?;
    std::str::from_utf8(&c.buf[range]).map_err(|e| format!("{what} is not UTF-8: {e}"))
}

/// A name or key that will be a TSV field and a line of `exclusions.txt`.
fn field<'b>(s: &'b str, what: &str) -> Result<&'b str, String> {
    if s.is_empty() {
        return Err(format!("{what} is empty"));
    }
    if let Some(c) = s.chars().find(|c| matches!(c, '\t' | '\n' | '\r')) {
        return Err(format!(
            "{what} {s:?} holds {c:?}, which a TSV field or a line of exclusions.txt cannot"
        ));
    }
    Ok(s)
}

/// A string the attestation embeds as it came: one JSON object on one line, as Python's
/// `json.dumps` writes it (no raw control character, so no line break).
fn json_object<'b>(s: &'b str, what: &str) -> Result<&'b str, String> {
    if !(s.starts_with('{') && s.ends_with('}')) || s.chars().any(|ch| u32::from(ch) < 0x20) {
        return Err(format!(
            "{what} must be one JSON object on one line (json.dumps)"
        ));
    }
    Ok(s)
}

/// Parse and validate a request. Nothing is hashed until all of it has been read.
pub fn parse(buf: &[u8]) -> Result<Request<'_>, String> {
    if buf.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; the bound is {MAX_INPUT_BYTES}",
            buf.len()
        ));
    }
    let mut c = Cursor::new(buf);
    let magic = c.take(8, "the magic")?;
    if &buf[magic] != INPUT_MAGIC {
        return Err(format!(
            "input does not start with {:?}",
            String::from_utf8_lossy(INPUT_MAGIC)
        ));
    }
    let n = c.u32("n")?;
    if !(1..=MAX_N).contains(&n) {
        return Err(format!("n {n} is outside 1..={MAX_N}"));
    }
    let raw = c.take(8, "threshold")?;
    let threshold = f64::from_le_bytes(buf[raw].try_into().map_err(|_| "threshold")?);
    // The oracle's own refusal: `not (0.0 < threshold <= 1.0)`.
    if !(threshold > 0.0 && threshold <= 1.0) {
        return Err(format!(
            "threshold {threshold} is not a containment test (0 < t <= 1)"
        ));
    }
    let corpus = json_object(take_str(&mut c, "corpus")?, "corpus")?;
    let export = json_object(take_str(&mut c, "export")?, "export")?;
    let n_checks = c.u32("n_checks")?;
    if n_checks > MAX_CHECKS {
        return Err(format!("{n_checks} checks; the bound is {MAX_CHECKS}"));
    }
    let mut checks = Vec::new();
    let mut check_names = HashSet::new();
    for _ in 0..n_checks {
        let name = field(take_str(&mut c, "check name")?, "check name")?;
        if !check_names.insert(name) {
            return Err(format!("check {name:?} is stated twice"));
        }
        let state = match c.take(1, "check state").map(|r| buf[r.start])? {
            0 => CheckState::NotRun,
            1 => CheckState::Failed,
            2 => CheckState::Passed,
            s => return Err(format!("check {name:?}: state {s} is not 0, 1 or 2")),
        };
        let detail = take_str(&mut c, "check detail")?;
        checks.push(Check {
            name: name.to_string(),
            state,
            detail: detail.to_string(),
        });
    }
    let n_sets = c.u32("n_sets")?;
    if n_sets == 0 || n_sets > MAX_SETS {
        return Err(format!("{n_sets} sets; 1..={MAX_SETS} are accepted"));
    }
    let mut set_names = Vec::new();
    for _ in 0..n_sets {
        let name = field(take_str(&mut c, "set name")?, "set name")?;
        if set_names.contains(&name) {
            return Err(format!("set {name:?} is named twice"));
        }
        set_names.push(name);
    }
    let n_scans = c.u32("n_scans")?;
    if n_scans == 0 || n_scans > MAX_SETS * MAX_SETS {
        return Err(format!(
            "{n_scans} scans: a decontamination against nothing is not a check"
        ));
    }
    let mut scans: Vec<Scan> = Vec::new();
    for _ in 0..n_scans {
        let source = c.u32("scan source")? as usize;
        let target = c.u32("scan target")? as usize;
        let enforced = match c.take(1, "scan enforced").map(|r| buf[r.start])? {
            0 => false,
            1 => true,
            e => return Err(format!("scan enforced flag {e} is not 0 or 1")),
        };
        if source >= set_names.len() || target >= set_names.len() || source == target {
            return Err(format!(
                "scan {source} -> {target} does not name two different sets of {}",
                set_names.len()
            ));
        }
        if scans
            .iter()
            .any(|s| s.source == source && s.target == target)
        {
            return Err(format!(
                "scan {} -> {} is stated twice",
                set_names[source], set_names[target]
            ));
        }
        scans.push(Scan {
            source,
            target,
            enforced,
        });
    }
    let enforced_sources: HashSet<usize> = scans
        .iter()
        .filter(|s| s.enforced)
        .map(|s| s.source)
        .collect();
    if enforced_sources.len() > 1 {
        return Err(
            "enforced scans name more than one source set; exclusions.txt excludes rows of one"
                .to_string(),
        );
    }
    let n_rows = c.u64("n_rows")?;
    if n_rows > MAX_ROWS_PER_SET * u64::from(n_sets) {
        return Err(format!(
            "{n_rows} rows; the bound is {MAX_ROWS_PER_SET} per set"
        ));
    }
    let mut sets: Vec<Vec<Row<'_>>> = vec![Vec::new(); set_names.len()];
    for _ in 0..n_rows {
        let set = c.u32("row set")? as usize;
        let key = field(take_str(&mut c, "row key")?, "row key")?;
        let identity = field(take_str(&mut c, "row identity_key")?, "row identity_key")?;
        let family = field(take_str(&mut c, "row family")?, "row family")?;
        let text = take_str(&mut c, "row text")?;
        let rows = sets
            .get_mut(set)
            .ok_or_else(|| format!("row {key:?} names set {set} of {}", set_names.len()))?;
        if rows.len() as u64 >= MAX_ROWS_PER_SET {
            return Err(format!(
                "set {:?} holds more than {MAX_ROWS_PER_SET} rows",
                set_names[set]
            ));
        }
        pyunicode::check_assigned(text).map_err(|e| format!("row {key:?}: {e}"))?;
        rows.push(Row {
            key,
            identity,
            family,
            text,
        });
    }
    if c.at != buf.len() {
        return Err(format!(
            "{} bytes follow the last of {n_rows} rows; refusing a file that is not exactly \
             what its header describes",
            buf.len() - c.at
        ));
    }
    for (rows, name) in sets.iter_mut().zip(&set_names) {
        rows.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
        if let Some(w) = rows.windows(2).find(|w| w[0].key == w[1].key) {
            return Err(format!("set {name:?} holds key {:?} twice", w[0].key));
        }
    }
    Ok(Request {
        n: n as usize,
        threshold,
        corpus,
        export,
        checks,
        set_names,
        scans,
        sets,
    })
}

/// A `u64` that is already a hash: `HashMap` keys pass straight through.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type GramMap<V> = HashMap<u64, V, BuildHasherDefault<PassThrough>>;

/// `word_ngrams(text, n)`: the distinct n-gram digests, sorted. Each digest is the 8-byte
/// BLAKE2b read big-endian, a bijection of the oracle's `bytes`.
pub fn word_ngrams(text: &str, n: usize, hasher: &Keyed) -> Result<Vec<u64>, String> {
    let lowered = pyunicode::lower(text);
    let words = pyunicode::words(&lowered);
    if words.len() < n {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(words.len() + 1 - n);
    let mut joined = String::new();
    for window in words.windows(n) {
        joined.clear();
        for (i, w) in window.iter().enumerate() {
            if i > 0 {
                joined.push(' ');
            }
            joined.push_str(w);
        }
        out.push(hasher.u64_be(joined.as_bytes())?);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// `target_digest`: sha256 over `key \x1f text \x1e` for every row in key order.
fn set_digest(rows: &[Row<'_>]) -> String {
    let mut h = Sha256::new();
    for r in rows {
        h.update(r.key.as_bytes());
        h.update(b"\x1f");
        h.update(r.text.as_bytes());
        h.update(b"\x1e");
    }
    h.hex()
}

/// One target set's n-gram index: which rows hold each n-gram, and each row's count.
struct Index {
    rows_of: GramMap<Vec<u32>>,
    size: Vec<u32>,
    indexed: usize,
}

fn build_index(rows: &[Row<'_>], n: usize, threads: usize) -> Result<Index, String> {
    let grams = per_row(rows, n, threads)?;
    let mut rows_of: GramMap<Vec<u32>> = GramMap::default();
    let mut size = Vec::with_capacity(rows.len());
    let mut indexed = 0;
    for (i, g) in grams.iter().enumerate() {
        size.push(u32::try_from(g.len()).map_err(|_| "a row holds over 2^32 n-grams")?);
        if !g.is_empty() {
            indexed += 1;
        }
        for &d in g {
            rows_of.entry(d).or_default().push(i as u32);
        }
    }
    Ok(Index {
        rows_of,
        size,
        indexed,
    })
}

/// Every row's distinct n-grams, in row order, on up to `threads` threads.
fn per_row(rows: &[Row<'_>], n: usize, threads: usize) -> Result<Vec<Vec<u64>>, String> {
    let mut out: Vec<Vec<u64>> = vec![Vec::new(); rows.len()];
    let hasher = Keyed::new(&[], 8)?;
    let error: Mutex<Option<String>> = Mutex::new(None);
    let items: Vec<(usize, &mut [Vec<u64>])> = out
        .chunks_mut(CHUNK_ROWS)
        .enumerate()
        .map(|(i, c)| (i * CHUNK_ROWS, c))
        .collect();
    run_parallel(items, threads, |(start, slots)| {
        for (j, slot) in slots.iter_mut().enumerate() {
            match word_ngrams(rows[start + j].text, n, &hasher) {
                Ok(g) => *slot = g,
                Err(e) => {
                    *error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
                    return;
                }
            }
        }
    });
    match error.into_inner().unwrap_or_else(|p| p.into_inner()) {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// Run `work` over `items` on up to `threads` threads, each taking the next item in turn.
/// Each item owns disjoint output, so scheduling cannot change any result.
fn run_parallel<T: Send>(items: Vec<T>, threads: usize, work: impl Fn(T) + Sync) {
    let workers = threads.clamp(1, items.len().max(1));
    let queue = Mutex::new(items.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let item = queue.lock().unwrap_or_else(|p| p.into_inner()).next();
                    match item {
                        Some(t) => work(t),
                        None => break,
                    }
                }
            });
        }
    });
}

/// One pair at or over the threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub scan: usize,
    pub source_row: usize,
    pub target_row: usize,
    pub shared: u32,
    pub target_ngrams: u32,
}

/// What one set is, as the attestation reports it.
#[derive(Clone, Debug)]
pub struct SetReport {
    pub rows: usize,
    pub indexed: usize,
    pub too_short: usize,
    pub digest: String,
}

/// The scan's answer, before it is written.
#[derive(Debug)]
pub struct Found {
    pub pairs: Vec<Pair>,
    pub sets: Vec<SetReport>,
}

impl Request<'_> {
    /// Source sets in the order their first scan names them.
    fn source_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        for s in &self.scans {
            if !order.contains(&s.source) {
                order.push(s.source);
            }
        }
        order
    }

    /// The complete pair list and every set's counts. Does not depend on `threads`.
    pub fn scan(&self, threads: usize) -> Result<Found, String> {
        let targets: Vec<usize> = {
            let mut t: Vec<usize> = self.scans.iter().map(|s| s.target).collect();
            t.sort_unstable();
            t.dedup();
            t
        };
        let mut indexes: BTreeMap<usize, Index> = BTreeMap::new();
        for &t in &targets {
            indexes.insert(t, build_index(&self.sets[t], self.n, threads)?);
        }
        let mut pairs = Vec::new();
        let mut source_grams: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for source in self.source_order() {
            let rows = &self.sets[source];
            let scans: Vec<(usize, &Scan)> = self
                .scans
                .iter()
                .enumerate()
                .filter(|(_, s)| s.source == source)
                .collect();
            let grams = per_row(rows, self.n, threads)?;
            source_grams.insert(source, grams.iter().map(Vec::len).collect());
            let mut chunks: Vec<Vec<Pair>> = vec![Vec::new(); rows.len().div_ceil(CHUNK_ROWS)];
            let items: Vec<(usize, &mut Vec<Pair>)> = chunks.iter_mut().enumerate().collect();
            let threshold = self.threshold;
            let indexes = &indexes;
            let grams = &grams;
            let scans = &scans;
            run_parallel(items, threads, |(chunk, out)| {
                let mut counts: HashMap<u32, u32> = HashMap::new();
                let lo = chunk * CHUNK_ROWS;
                let hi = (lo + CHUNK_ROWS).min(grams.len());
                for (r, g) in grams.iter().enumerate().take(hi).skip(lo) {
                    if g.is_empty() {
                        continue;
                    }
                    for &(scan_i, scan) in scans {
                        let index = &indexes[&scan.target];
                        counts.clear();
                        for d in g {
                            if let Some(ts) = index.rows_of.get(d) {
                                for &t in ts {
                                    *counts.entry(t).or_insert(0) += 1;
                                }
                            }
                        }
                        // Target rows are in key order, so their indices sort as their keys.
                        let mut hit: Vec<(u32, u32)> = counts
                            .iter()
                            .map(|(&t, &c)| (t, c))
                            .filter(|&(t, c)| {
                                f64::from(c) / f64::from(index.size[t as usize]) >= threshold
                            })
                            .collect();
                        hit.sort_unstable();
                        out.extend(hit.into_iter().map(|(t, c)| Pair {
                            scan: scan_i,
                            source_row: r,
                            target_row: t as usize,
                            shared: c,
                            target_ngrams: index.size[t as usize],
                        }));
                    }
                }
            });
            pairs.extend(chunks.into_iter().flatten());
        }
        let mut sets = Vec::with_capacity(self.sets.len());
        for (i, rows) in self.sets.iter().enumerate() {
            let indexed = match (indexes.get(&i), source_grams.get(&i)) {
                (Some(ix), _) => ix.indexed,
                (None, Some(g)) => g.iter().filter(|&&l| l > 0).count(),
                (None, None) => per_row(rows, self.n, threads)?
                    .iter()
                    .filter(|g| !g.is_empty())
                    .count(),
            };
            sets.push(SetReport {
                rows: rows.len(),
                indexed,
                too_short: rows.len() - indexed,
                digest: set_digest(rows),
            });
        }
        Ok(Found { pairs, sets })
    }
}

// --- writing ------------------------------------------------------------------------------

/// A JSON string literal: `"`, `\` and the control characters escaped, everything else as is.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON value, written with two-space indentation and keys in insertion order.
enum Json {
    Bool(bool),
    Int(u64),
    Float(f64),
    Str(String),
    /// Bytes the request vouched for as one JSON object, written as they came.
    Verbatim(String),
    Arr(Vec<Json>),
    Obj(Vec<(&'static str, Json)>),
    /// An object whose keys are data (set names, families), not field names.
    Map(Vec<(String, Json)>),
}

impl Json {
    fn write(&self, out: &mut String, depth: usize) {
        let pad = |d: usize| "  ".repeat(d);
        match self {
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            // Debug keeps a fractional part ("1.0"), so Python reads a float back.
            Json::Float(f) => out.push_str(&format!("{f:?}")),
            Json::Str(s) => out.push_str(&json_str(s)),
            Json::Verbatim(s) => out.push_str(s),
            Json::Arr(items) if items.is_empty() => out.push_str("[]"),
            Json::Arr(items) => {
                out.push_str("[\n");
                for (i, v) in items.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    v.write(out, depth + 1);
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push(']');
            }
            Json::Obj(fields) => Self::write_fields(
                fields.iter().map(|(k, v)| (*k, v)),
                fields.len(),
                out,
                depth,
            ),
            Json::Map(fields) => Self::write_fields(
                fields.iter().map(|(k, v)| (k.as_str(), v)),
                fields.len(),
                out,
                depth,
            ),
        }
    }

    fn write_fields<'a>(
        fields: impl Iterator<Item = (&'a str, &'a Json)>,
        len: usize,
        out: &mut String,
        depth: usize,
    ) {
        if len == 0 {
            out.push_str("{}");
            return;
        }
        out.push_str("{\n");
        for (i, (k, v)) in fields.enumerate() {
            out.push_str(&"  ".repeat(depth + 1));
            out.push_str(&json_str(k));
            out.push_str(": ");
            v.write(out, depth + 1);
            out.push_str(if i + 1 < len { ",\n" } else { "\n" });
        }
        out.push_str(&"  ".repeat(depth));
        out.push('}');
    }
}

/// The three files, as bytes, and a summary line.
pub struct Written {
    pub pairs_tsv: Vec<u8>,
    pub exclusions: Vec<u8>,
    pub attestation: Vec<u8>,
    pub summary: String,
}

impl Request<'_> {
    pub fn render(&self, found: &Found) -> Written {
        let set = |i: usize| self.set_names[i];
        // pairs.tsv
        let mut pairs_tsv = String::from(PAIRS_HEADER);
        for p in &found.pairs {
            let scan = &self.scans[p.scan];
            let s = &self.sets[scan.source][p.source_row];
            let t = &self.sets[scan.target][p.target_row];
            pairs_tsv.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                set(scan.source),
                s.key,
                s.identity,
                s.family,
                set(scan.target),
                t.key,
                t.family,
                p.shared,
                p.target_ngrams
            ));
        }
        // exclusions.txt: the enforced scans' source identity keys, byte-sorted, unique.
        let mut excluded: Vec<&str> = found
            .pairs
            .iter()
            .filter(|p| self.scans[p.scan].enforced)
            .map(|p| self.sets[self.scans[p.scan].source][p.source_row].identity)
            .collect();
        excluded.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        excluded.dedup();
        let mut exclusions = String::new();
        for k in &excluded {
            exclusions.push_str(k);
            exclusions.push('\n');
        }
        let excluded_set: HashSet<&str> = excluded.iter().copied().collect();

        // Per scan, and per (scan, source family).
        let mut scan_json = Vec::new();
        let mut family_json = Vec::new();
        let mut remaining = Vec::new();
        let mut reasons: Vec<String> = Vec::new();
        for (i, scan) in self.scans.iter().enumerate() {
            let mine: Vec<&Pair> = found.pairs.iter().filter(|p| p.scan == i).collect();
            let source_rows: HashSet<usize> = mine.iter().map(|p| p.source_row).collect();
            let target_rows: HashSet<usize> = mine.iter().map(|p| p.target_row).collect();
            // Re-counted from the pairs, not assumed: a pair whose source row survives the
            // exclusion is overlap the build would still train on.
            let left = mine
                .iter()
                .filter(|p| !excluded_set.contains(self.sets[scan.source][p.source_row].identity))
                .count() as u64;
            let mut by_family: BTreeMap<&str, (HashSet<usize>, u64)> = BTreeMap::new();
            for p in &mine {
                let e = by_family
                    .entry(self.sets[scan.source][p.source_row].family)
                    .or_default();
                e.0.insert(p.source_row);
                e.1 += 1;
            }
            for (family, (rows, n_pairs)) in by_family {
                family_json.push(Json::Obj(vec![
                    ("source", Json::Str(set(scan.source).to_string())),
                    ("target", Json::Str(set(scan.target).to_string())),
                    ("source_family", Json::Str(family.to_string())),
                    ("source_rows_hit", Json::Int(rows.len() as u64)),
                    ("pairs", Json::Int(n_pairs)),
                ]));
            }
            scan_json.push(Json::Obj(vec![
                ("source", Json::Str(set(scan.source).to_string())),
                ("target", Json::Str(set(scan.target).to_string())),
                ("enforced", Json::Bool(scan.enforced)),
                ("pairs", Json::Int(mine.len() as u64)),
                ("source_rows_hit", Json::Int(source_rows.len() as u64)),
                ("target_rows_hit", Json::Int(target_rows.len() as u64)),
                ("remaining_hits", Json::Int(left)),
            ]));
            if scan.enforced {
                remaining.push((set(scan.target).to_string(), Json::Int(left)));
                if left > 0 {
                    reasons.push(format!(
                        "{} -> {}: {left} pair(s) survive the exclusion",
                        set(scan.source),
                        set(scan.target)
                    ));
                }
                let (src, tgt) = (&found.sets[scan.source], &found.sets[scan.target]);
                if tgt.indexed == 0 {
                    reasons.push(format!(
                        "enforced target {:?} has no row with a {}-gram: nothing was compared",
                        set(scan.target),
                        self.n
                    ));
                }
                if src.indexed == 0 {
                    reasons.push(format!(
                        "source {:?} has no row with a {}-gram: nothing was checked",
                        set(scan.source),
                        self.n
                    ));
                }
            }
        }
        if !self.scans.iter().any(|s| s.enforced) {
            reasons.push("no scan is enforced: an attestation that excludes nothing".to_string());
        }
        let mut checks = Vec::new();
        for c in &self.checks {
            let body = match c.state {
                CheckState::NotRun => {
                    reasons.push(format!("splitter check {:?} did not run", c.name));
                    Json::Obj(vec![
                        ("state", Json::Str("not_run".to_string())),
                        ("reason", Json::Str(c.detail.clone())),
                    ])
                }
                CheckState::Failed | CheckState::Passed => {
                    let passed = c.state == CheckState::Passed;
                    if !passed {
                        reasons.push(format!("splitter check {:?} failed", c.name));
                    }
                    Json::Obj(vec![
                        ("state", Json::Str("ran".to_string())),
                        ("passed", Json::Bool(passed)),
                        ("detail", Json::Str(c.detail.clone())),
                    ])
                }
            };
            checks.push((c.name.clone(), body));
        }
        if self.checks.is_empty() {
            reasons.push(
                "no splitter check was stated: keys (i) and (iii) are unattested".to_string(),
            );
        }
        let excluded_from = self
            .scans
            .iter()
            .find(|s| s.enforced)
            .map(|s| set(s.source).to_string());
        let mut enforced_targets = Vec::new();
        let mut unenforced_targets = Vec::new();
        let mut report_only = Vec::new();
        for s in &self.scans {
            let name = Json::Str(set(s.target).to_string());
            match (&excluded_from, s.enforced) {
                (_, true) => enforced_targets.push(name),
                (Some(src), false) if src == set(s.source) => unenforced_targets.push(name),
                _ => report_only.push(Json::Obj(vec![
                    ("source", Json::Str(set(s.source).to_string())),
                    ("target", name),
                ])),
            }
        }
        let sets = Json::Map(
            found
                .sets
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    (
                        set(i).to_string(),
                        Json::Obj(vec![
                            ("rows", Json::Int(r.rows as u64)),
                            ("indexed", Json::Int(r.indexed as u64)),
                            ("too_short", Json::Int(r.too_short as u64)),
                            ("digest", Json::Str(r.digest.clone())),
                        ]),
                    )
                })
                .collect(),
        );
        let clean = reasons.is_empty();
        let pairs_sha256 = sha256_hex(pairs_tsv.as_bytes());
        let exclusions_sha256 = sha256_hex(exclusions.as_bytes());
        let attestation = Json::Obj(vec![
            ("version", Json::Int(u64::from(ATTESTATION_VERSION))),
            ("tool", Json::Str("qd-prep containment".to_string())),
            (
                "rule",
                Json::Str(
                    "a source row hits a target row when it contains at least `threshold` of \
                     the target row's distinct lower-cased \\w+ word n-grams, blake2b-64 \
                     hashed (qd_train.replay.decontaminate, every pair, not the best per row); \
                     keys (i) and (iii) are the splitter's checks below"
                        .to_string(),
                ),
            ),
            ("n", Json::Int(self.n as u64)),
            ("threshold", Json::Float(self.threshold)),
            (
                "unicode_version",
                Json::Str(pyunicode::unicode_version().to_string()),
            ),
            ("corpus", Json::Verbatim(self.corpus.to_string())),
            ("export", Json::Verbatim(self.export.to_string())),
            ("splitter_checks", Json::Map(checks)),
            ("sets", sets),
            ("scans", Json::Arr(scan_json)),
            ("hits_by_source_family", Json::Arr(family_json)),
            (
                "excluded_from",
                excluded_from.map_or(Json::Str(String::new()), Json::Str),
            ),
            ("enforced_targets", Json::Arr(enforced_targets)),
            ("unenforced_targets", Json::Arr(unenforced_targets)),
            ("report_only_scans", Json::Arr(report_only)),
            ("pairs_file", Json::Str(PAIRS_NAME.to_string())),
            ("n_pairs", Json::Int(found.pairs.len() as u64)),
            ("pairs_sha256", Json::Str(pairs_sha256)),
            ("exclusions_file", Json::Str(EXCLUSIONS_NAME.to_string())),
            ("n_exclusions", Json::Int(excluded.len() as u64)),
            ("exclusions_sha256", Json::Str(exclusions_sha256)),
            ("remaining_hits", Json::Map(remaining)),
            ("clean", Json::Bool(clean)),
            (
                "not_clean_because",
                Json::Arr(reasons.into_iter().map(Json::Str).collect()),
            ),
        ]);
        let mut body = String::new();
        attestation.write(&mut body, 0);
        body.push('\n');
        let summary = format!(
            "qd-prep containment: {} pair(s), {} identity key(s) excluded, {}",
            found.pairs.len(),
            excluded.len(),
            if clean { "CLEAN" } else { "NOT CLEAN" }
        );
        Written {
            pairs_tsv: pairs_tsv.into_bytes(),
            exclusions: exclusions.into_bytes(),
            attestation: body.into_bytes(),
            summary,
        }
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.sync_all().map_err(|e| format!("{}: {e}", path.display()))
}

/// Write the three files into `out_dir`, which must not exist: into `out_dir.partial` first,
/// then renamed, so a reader never sees a directory with some of them.
pub fn write_dir(out_dir: &Path, w: &Written) -> Result<(), String> {
    if out_dir.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            out_dir.display()
        ));
    }
    // DIR.partial beside DIR: appended, not `with_extension`, which would turn `scan.v5`
    // into `scan.partial` and let two out dirs share one.
    let mut partial_name = out_dir
        .file_name()
        .ok_or_else(|| format!("{} names no directory", out_dir.display()))?
        .to_os_string();
    partial_name.push(".partial");
    let partial = out_dir.with_file_name(partial_name);
    std::fs::create_dir(&partial).map_err(|e| format!("{}: {e}", partial.display()))?;
    write_file(&partial.join(PAIRS_NAME), &w.pairs_tsv)?;
    write_file(&partial.join(EXCLUSIONS_NAME), &w.exclusions)?;
    write_file(&partial.join(ATTESTATION_NAME), &w.attestation)?;
    std::fs::rename(&partial, out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))
}

/// `qd-prep containment`: request bytes in, the three files and a summary line out.
pub fn run(buf: &[u8], threads: usize) -> Result<Written, String> {
    let request = parse(buf)?;
    let found = request.scan(threads)?;
    Ok(request.render(&found))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        out.extend_from_slice(v.as_bytes());
    }

    type TestRow<'a> = (u32, &'a str, &'a str, &'a str, &'a str);

    fn request(
        sets: &[&str],
        scans: &[(u32, u32, bool)],
        checks: &[(&str, u8)],
        rows: &[TestRow<'_>],
    ) -> Vec<u8> {
        let mut out = INPUT_MAGIC.to_vec();
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&0.5f64.to_le_bytes());
        s(&mut out, "{\"rev\": \"x\"}");
        s(&mut out, "{}");
        out.extend_from_slice(&(checks.len() as u32).to_le_bytes());
        for (name, state) in checks {
            s(&mut out, name);
            out.push(*state);
            s(&mut out, "detail");
        }
        out.extend_from_slice(&(sets.len() as u32).to_le_bytes());
        for name in sets {
            s(&mut out, name);
        }
        out.extend_from_slice(&(scans.len() as u32).to_le_bytes());
        for (src, tgt, enf) in scans {
            out.extend_from_slice(&src.to_le_bytes());
            out.extend_from_slice(&tgt.to_le_bytes());
            out.push(u8::from(*enf));
        }
        out.extend_from_slice(&(rows.len() as u64).to_le_bytes());
        for (set, key, identity, family, text) in rows {
            out.extend_from_slice(&set.to_le_bytes());
            s(&mut out, key);
            s(&mut out, identity);
            s(&mut out, family);
            s(&mut out, text);
        }
        out
    }

    const LONG_A: &str = "the quick brown fox jumps over the lazy dog and then runs far away \
                          into the deep dark forest beyond the river bend";
    const LONG_B: &str = "a completely different sentence about compilers parsers and type \
                          checkers that shares nothing with the other one at all here";

    #[test]
    fn a_source_row_holding_two_targets_lists_both_and_is_excluded_once() {
        let both = format!("{LONG_A} and also {LONG_B}");
        let rows = [
            (0, "t1", "id-1", "fam.x", both.as_str()),
            (
                0,
                "t2",
                "id-2",
                "fam.y",
                "nothing in common with any target row",
            ),
            (1, "va", "vid-a", "fam.x", LONG_A),
            (1, "vb", "vid-b", "fam.y", LONG_B),
            (2, "ha", "hid-a", "fam.x", LONG_A),
        ];
        let buf = request(
            &["train", "val", "heldout-family:qa.answerability"],
            &[(0, 1, true), (0, 2, false)],
            &[("identity_disjoint", 2), ("near_duplicate_disjoint", 2)],
            &rows,
        );
        let w = run(&buf, 3).expect("runs");
        let tsv = String::from_utf8(w.pairs_tsv).expect("utf8");
        let lines: Vec<&str> = tsv.lines().skip(1).collect();
        assert_eq!(lines.len(), 3, "{tsv}");
        assert!(lines[0].starts_with("train\tt1\tid-1\tfam.x\tval\tva\t"));
        assert!(lines[1].starts_with("train\tt1\tid-1\tfam.x\tval\tvb\t"));
        assert!(
            lines[2].starts_with("train\tt1\tid-1\tfam.x\theldout-family:qa.answerability\tha\t")
        );
        assert_eq!(w.exclusions, b"id-1\n");
        let att = String::from_utf8(w.attestation).expect("utf8");
        assert!(att.contains("\"clean\": true"), "{att}");
        assert!(
            att.contains("\"remaining_hits\": {\n    \"val\": 0\n  }"),
            "{att}"
        );
    }

    #[test]
    fn the_answer_does_not_depend_on_the_thread_count() {
        let mut rows: Vec<(u32, String, String, String, String)> = Vec::new();
        for i in 0..700 {
            let text = if i % 7 == 0 {
                format!("row {i} {LONG_A}")
            } else {
                format!(
                    "row {i} words {} {}",
                    i * 3,
                    LONG_B.split(' ').nth(i % 20).unwrap_or("")
                )
            };
            rows.push((
                0,
                format!("k{i:04}"),
                format!("id{}", i / 3),
                "f".into(),
                text,
            ));
        }
        rows.push((1, "v".into(), "vid".into(), "f".into(), LONG_A.into()));
        let refs: Vec<TestRow<'_>> = rows
            .iter()
            .map(|(a, b, c, d, e)| (*a, b.as_str(), c.as_str(), d.as_str(), e.as_str()))
            .collect();
        let buf = request(&["train", "val"], &[(0, 1, true)], &[("x", 2)], &refs);
        let one = run(&buf, 1).expect("runs");
        for threads in [2, 5, 64] {
            let many = run(&buf, threads).expect("runs");
            assert_eq!(many.pairs_tsv, one.pairs_tsv, "{threads} threads");
            assert_eq!(many.attestation, one.attestation, "{threads} threads");
        }
        assert_eq!(
            String::from_utf8(one.pairs_tsv)
                .expect("utf8")
                .lines()
                .count()
                - 1,
            100,
            "rows 0, 7, ..., 693 contain the val row"
        );
    }

    #[test]
    fn a_failed_or_missing_splitter_check_or_an_empty_target_is_not_clean() {
        let rows = [(0, "t", "id", "f", LONG_A), (1, "v", "vid", "f", LONG_B)];
        let not_clean = |checks: &[(&str, u8)], rows: &[TestRow<'_>]| {
            let w = run(
                &request(&["train", "val"], &[(0, 1, true)], checks, rows),
                1,
            )
            .expect("runs");
            String::from_utf8(w.attestation)
                .expect("utf8")
                .contains("\"clean\": false")
        };
        assert!(!not_clean(&[("a", 2)], &rows));
        assert!(not_clean(&[("a", 1)], &rows), "failed");
        assert!(not_clean(&[("a", 0)], &rows), "not run");
        assert!(not_clean(&[], &rows), "unstated");
        let short_target = [
            (0, "t", "id", "f", LONG_A),
            (1, "v", "vid", "f", "too short"),
        ];
        assert!(
            not_clean(&[("a", 2)], &short_target),
            "nothing to compare against"
        );
    }

    #[test]
    fn malformed_requests_are_refused() {
        let rows = [(0, "t", "id", "f", LONG_A), (1, "v", "vid", "f", LONG_B)];
        let good = request(&["train", "val"], &[(0, 1, true)], &[("a", 2)], &rows);
        assert!(parse(&good).is_ok());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(parse(&trailing).unwrap_err().contains("follow the last"));
        let tab = [(0, "t\tx", "id", "f", LONG_A)];
        let err = parse(&request(&["train", "val"], &[(0, 1, true)], &[], &tab)).unwrap_err();
        assert!(err.contains("TSV"), "{err}");
        let dup = [(0, "t", "id", "f", LONG_A), (0, "t", "id2", "f", LONG_B)];
        let err = parse(&request(&["train", "val"], &[(0, 1, true)], &[], &dup)).unwrap_err();
        assert!(err.contains("twice"), "{err}");
        let self_scan = request(&["train", "val"], &[(0, 0, true)], &[], &rows);
        assert!(
            parse(&self_scan)
                .unwrap_err()
                .contains("two different sets")
        );
        let two_sources = request(&["a", "b", "c"], &[(0, 2, true), (1, 2, true)], &[], &[]);
        assert!(
            parse(&two_sources)
                .unwrap_err()
                .contains("more than one source")
        );
        let unassigned = [(0, "t", "id", "f", "a\u{0378}b")];
        let err = parse(&request(
            &["train", "val"],
            &[(0, 1, true)],
            &[],
            &unassigned,
        ))
        .unwrap_err();
        assert!(err.contains("unassigned"), "{err}");
        let mut zero_threshold = good.clone();
        zero_threshold[12..20].copy_from_slice(&0.0f64.to_le_bytes());
        assert!(parse(&zero_threshold).unwrap_err().contains("threshold"));
    }

    #[test]
    fn json_strings_escape_what_json_requires_and_nothing_else() {
        assert_eq!(json_str("a\"b\\c\nd\u{1}é"), "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }
}
