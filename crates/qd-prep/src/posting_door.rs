//! The content-level door for job postings (v2 plan section 6, gate 1).
//!
//! `crates/qd-train/src/held_out.rs` is a path door: it stops a training process opening a file
//! that sits under a held-out marker. A reposted near-duplicate of a held-out posting sits under
//! no marker, so it needs a content door beside the path door. Vani exports the owner-labelled
//! postings' keys and their normalised-text MinHash signatures to a held-out path
//! ([`DEFAULT_SIGNATURES_PATH`]); this module refuses any train row that is within the v5
//! decontamination rule of one of them.
//!
//! **The rule is v5's, not a new one.** v5's decontamination (`campaign/v5-preregistered.json`,
//! Fable Q2; `qd_train.replay`) is: lower-cased `\w+` words, 8-grams, a row is a hit when it
//! contains at least 0.5 of the held-out row's n-grams ([`CONTAINMENT_THRESHOLD`]). v5's dedupe
//! (`qd_data.config.DataConfig`) also drops pairs at MinHash Jaccard 0.8
//! ([`JACCARD_THRESHOLD`]) over 128 permutations ([`NUM_PERM`]). The door applies both over the
//! same unit: the same normalisation ([`crate::containment::word_ngrams`], so a whitespace or
//! case change, punctuation or a moved sentence changes nothing), and the held-out side is a
//! signature, so containment is estimated from it: with `J` the estimated Jaccard of the two
//! n-gram sets and `m`, `k` their sizes (the row's and the held-out row's distinct n-grams),
//! `|A and B| = J (m + k) / (1 + J)` and containment is that over `k`.
//!
//! The thresholds are constants here because this crate cannot import Python; the tests read
//! `python/qd_train/replay.py` and `python/qd_data/config.py` and fail if either moved. They
//! are never read from the signature file: a file that could carry its own threshold could
//! loosen the door.
//!
//! **Fails closed.** A signature file that is missing, not a regular file, over its bound,
//! corrupt (a digest over every entry is checked), of another format, n or permutation count, or
//! empty refuses the whole pool: [`PostingDoor::load`] returns an error and no row is judged. A
//! row with no 8-gram (fewer than 8 words) cannot be compared and is refused, not admitted. The
//! file must sit under a held-out path marker, so the training process's path door refuses it.
//!
//! **What it does not establish.** A MinHash estimate over 128 permutations has a standard error
//! of at most 0.044 (`sqrt(J(1-J)/128)`), so a pair truly just under a threshold can be refused
//! and one just over can be admitted. Where a text is held, `qd-prep containment` is exact.
//! Reading the signatures is a decontamination read, as `--target` files are
//! ([`crate::heldout`]): the training process never opens them.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use serde_json::{Value, json};

use crate::blake2b::Keyed;
use crate::containment::word_ngrams;
use crate::files;
use crate::heldout::check_held_out_path;
use crate::minhash::{Family, MERSENNE_PRIME};
use crate::pyunicode;
use crate::sha256::sha256_hex;

/// The format string of the signature file.
pub const FORMAT: &str = "qd-posting-heldout.v1";
/// Where Vani's export lives: under the held-out root (`data/heldout`), so the training door
/// refuses it. `crates/qd-train/src/held_out.rs` pins the same path in a test.
pub const DEFAULT_SIGNATURES_PATH: &str = "data/heldout/vani-postings/signatures.json";
/// v5's decontamination word n (`qd_train.replay.DEFAULT_N`).
pub const NGRAM_N: usize = 8;
/// v5's decontamination containment threshold (`qd_train.replay.DEFAULT_THRESHOLD`).
pub const CONTAINMENT_THRESHOLD: f64 = 0.5;
/// v5's dedupe Jaccard threshold (`qd_data.config.DataConfig.dedupe_threshold`).
pub const JACCARD_THRESHOLD: f64 = 0.8;
/// v5's MinHash permutation count (`qd_data.config.DataConfig.num_perm`).
pub const NUM_PERM: usize = 128;
/// The signature file is read within this bound.
pub const MAX_FILE_BYTES: u64 = 1 << 30;
/// Most held-out postings one file may name. Vani's ledger holds about 2.3k postings; the
/// comparison is a scan, so the bound is what keeps a row's cost bounded.
pub const MAX_ENTRIES: usize = 1 << 16;
/// A posting longer than this is refused unsigned: it is not a posting.
pub const MAX_POSTING_BYTES: usize = 1 << 20;

/// One held-out posting: its key, its distinct 8-gram count and its signature.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    key: String,
    n_ngrams: u64,
    signature: Vec<u64>,
}

/// Why a row is refused by the door.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// The row's posting key is a held-out posting's key.
    KeyHeldOut { key: String },
    /// The row is within v5's decontamination rule of a held-out posting.
    NearDuplicate {
        held_out_key: String,
        jaccard: f64,
        containment: f64,
    },
    /// The row cannot be compared (too short, an unassigned code point, over the bound). Not a
    /// pass: an unscannable row is refused.
    Unscannable(String),
}

impl Refusal {
    /// A short machine-readable kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Refusal::KeyHeldOut { .. } => "held_out_key",
            Refusal::NearDuplicate { .. } => "near_duplicate_of_held_out",
            Refusal::Unscannable(_) => "unscannable",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::KeyHeldOut { key } => {
                write!(f, "posting key {key:?} is a held-out posting's key")
            }
            Refusal::NearDuplicate {
                held_out_key,
                jaccard,
                containment,
            } => write!(
                f,
                "within the v5 decontamination rule of held-out posting {held_out_key:?}: \
                 estimated jaccard {jaccard:.3} (refused from {JACCARD_THRESHOLD}), containment \
                 {containment:.3} (refused from {CONTAINMENT_THRESHOLD})"
            ),
            Refusal::Unscannable(why) => write!(f, "cannot be compared with the held-out set: {why}"),
        }
    }
}

/// The held-out postings' keys and signatures, and the family that signs a row to compare.
pub struct PostingDoor {
    seed: u64,
    family: Family,
    hasher: Keyed,
    entries: Vec<Entry>,
    keys: HashSet<String>,
}

impl PostingDoor {
    fn with_entries(seed: u64, entries: Vec<Entry>) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("no held-out posting: a door with nothing behind it is not a door".into());
        }
        if entries.len() > MAX_ENTRIES {
            return Err(format!(
                "{} held-out postings; the bound is {MAX_ENTRIES}",
                entries.len()
            ));
        }
        let mut keys = HashSet::with_capacity(entries.len());
        for e in &entries {
            if e.key.trim().is_empty() {
                return Err("a held-out posting has an empty key".into());
            }
            if !keys.insert(e.key.clone()) {
                return Err(format!("held-out posting key {:?} repeats", e.key));
            }
            if e.n_ngrams == 0 {
                return Err(format!("held-out posting {:?} has no 8-gram to compare", e.key));
            }
            if e.signature.len() != NUM_PERM {
                return Err(format!(
                    "held-out posting {:?}: signature has {} values, not {NUM_PERM}",
                    e.key,
                    e.signature.len()
                ));
            }
            if let Some(v) = e.signature.iter().find(|v| **v >= MERSENNE_PRIME) {
                return Err(format!(
                    "held-out posting {:?}: signature value {v} is not a residue mod 2**61 - 1",
                    e.key
                ));
            }
        }
        Ok(Self {
            seed,
            family: Family::from_seed(NUM_PERM, seed)?,
            hasher: Keyed::new(&[], 8)?,
            entries,
            keys,
        })
    }

    /// Distinct 8-gram count and signature of a posting's text.
    fn sign(&self, text: &str) -> Result<(u64, Vec<u64>), String> {
        if text.len() > MAX_POSTING_BYTES {
            return Err(format!(
                "{} bytes of text; the bound is {MAX_POSTING_BYTES}",
                text.len()
            ));
        }
        pyunicode::check_assigned(text)?;
        let grams = word_ngrams(text, NGRAM_N, &self.hasher)?;
        if grams.is_empty() {
            return Err(format!("fewer than {NGRAM_N} words, so no {NGRAM_N}-gram"));
        }
        let bytes: Vec<[u8; 8]> = grams.iter().map(|g| g.to_be_bytes()).collect();
        let mut signature = vec![0u64; NUM_PERM];
        self.family
            .signature(bytes.iter().map(|b| b.as_slice()), &mut signature)?;
        Ok((grams.len() as u64, signature))
    }

    /// The door's own export: sign each held-out posting `(key, text)`. Refuses an empty set, a
    /// repeated key and a posting with no 8-gram.
    pub fn build(seed: u64, postings: &[(String, String)]) -> Result<Self, String> {
        let signer = Self {
            seed,
            family: Family::from_seed(NUM_PERM, seed)?,
            hasher: Keyed::new(&[], 8)?,
            entries: Vec::new(),
            keys: HashSet::new(),
        };
        let mut entries = Vec::with_capacity(postings.len());
        for (key, text) in postings {
            let (n_ngrams, signature) = signer
                .sign(text)
                .map_err(|e| format!("held-out posting {key:?}: {e}"))?;
            entries.push(Entry {
                key: key.clone(),
                n_ngrams,
                signature,
            });
        }
        Self::with_entries(seed, entries)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The digest over the header and every entry in order; the file records it.
    fn digest(seed: u64, entries: &[Entry]) -> String {
        let mut text = format!("{FORMAT}|{NGRAM_N}|{NUM_PERM}|{seed}\x1d");
        for e in entries {
            let sig: Vec<String> = e.signature.iter().map(|v| format!("{v:016x}")).collect();
            text.push_str(&format!("{}\x1f{}\x1f{}\x1e", e.key, e.n_ngrams, sig.join(",")));
        }
        sha256_hex(text.as_bytes())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "format": FORMAT,
            "ngram_n": NGRAM_N,
            "num_perm": NUM_PERM,
            "seed": self.seed,
            "digest": Self::digest(self.seed, &self.entries),
            "entries": self.entries.iter().map(|e| json!({
                "key": e.key,
                "n_ngrams": e.n_ngrams,
                "signature": e.signature.iter().map(|v| format!("{v:016x}")).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    }

    /// Write the signature file, refused unless `path` sits under a held-out marker and does not
    /// exist.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        check_held_out_path(path)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        files::write_new_file(path, self.to_json().to_string().as_bytes())
    }

    /// Read and verify a signature file. Every failure is an error; none is a door that admits.
    pub fn load(path: &Path) -> Result<Self, String> {
        check_held_out_path(path)
            .map_err(|e| format!("the signature file must sit under a held-out path: {e}"))?;
        let bytes = files::read_bounded(path, MAX_FILE_BYTES, "held-out posting signatures")?;
        Self::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// [`Self::load`] from bytes: the checks, without the path and the read.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let doc: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("corrupt: not JSON: {e}"))?;
        let field = |k: &str| doc.get(k).ok_or_else(|| format!("corrupt: no {k:?}"));
        let format = field("format")?.as_str();
        if format != Some(FORMAT) {
            return Err(format!("wrong version: format is {format:?}, this reader reads {FORMAT:?}"));
        }
        for (name, want) in [("ngram_n", NGRAM_N as u64), ("num_perm", NUM_PERM as u64)] {
            let got = field(name)?.as_u64();
            if got != Some(want) {
                return Err(format!(
                    "wrong version: {name} is {got:?}, the v5 rule this door applies needs {want}"
                ));
            }
        }
        let seed = field("seed")?
            .as_u64()
            .ok_or("corrupt: seed is not an unsigned integer")?;
        let recorded = field("digest")?
            .as_str()
            .ok_or("corrupt: digest is not a string")?
            .to_owned();
        let raw = field("entries")?
            .as_array()
            .ok_or("corrupt: entries is not an array")?;
        if raw.len() > MAX_ENTRIES {
            return Err(format!("{} entries; the bound is {MAX_ENTRIES}", raw.len()));
        }
        let mut entries = Vec::with_capacity(raw.len());
        for (i, e) in raw.iter().enumerate() {
            let key = e
                .get("key")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("corrupt: entry {i} has no string key"))?;
            let n_ngrams = e
                .get("n_ngrams")
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("corrupt: entry {i} has no n_ngrams"))?;
            let sig = e
                .get("signature")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("corrupt: entry {i} has no signature"))?;
            let signature = sig
                .iter()
                .map(|v| {
                    v.as_str()
                        .filter(|s| s.len() == 16)
                        .and_then(|s| u64::from_str_radix(s, 16).ok())
                        .ok_or_else(|| format!("corrupt: entry {i} has a signature value that is not 16 hex digits"))
                })
                .collect::<Result<Vec<u64>, String>>()?;
            entries.push(Entry {
                key: key.to_owned(),
                n_ngrams,
                signature,
            });
        }
        let computed = Self::digest(seed, &entries);
        if computed != recorded {
            return Err(format!(
                "corrupt: the entries hash to {computed} but the file records {recorded}"
            ));
        }
        Self::with_entries(seed, entries)
    }

    /// Judge one train row: its posting key and the posting's text (before redaction: the
    /// held-out signature was made from the unredacted text).
    pub fn check(&self, posting_key: &str, text: &str) -> Result<(), Refusal> {
        if self.keys.contains(posting_key) {
            return Err(Refusal::KeyHeldOut {
                key: posting_key.to_owned(),
            });
        }
        let (m, signature) = self.sign(text).map_err(Refusal::Unscannable)?;
        let mut worst: Option<Refusal> = None;
        let mut worst_containment = -1.0f64;
        for held in &self.entries {
            let same = signature
                .iter()
                .zip(&held.signature)
                .filter(|(a, b)| a == b)
                .count();
            let jaccard = same as f64 / NUM_PERM as f64;
            let k = held.n_ngrams as f64;
            let shared = jaccard * (m as f64 + k) / (1.0 + jaccard);
            let containment = (shared / k).min(1.0);
            if (containment >= CONTAINMENT_THRESHOLD || jaccard >= JACCARD_THRESHOLD)
                && containment > worst_containment
            {
                worst_containment = containment;
                worst = Some(Refusal::NearDuplicate {
                    held_out_key: held.key.clone(),
                    jaccard,
                    containment,
                });
            }
        }
        worst.map_or(Ok(()), Err)
    }

    /// The keys, for a report.
    pub fn keys(&self) -> BTreeSet<&str> {
        self.entries.iter().map(|e| e.key.as_str()).collect()
    }
}

/// `qd-prep posting-door-build`: `postings` is JSONL, one `{"key", "text"}` per line, the
/// owner-labelled postings Vani exports; `out` is the signature file (under a held-out path).
pub fn run_build(postings: &Path, out: &Path, seed: u64) -> Result<String, String> {
    check_held_out_path(out)?;
    let bytes = files::read_bounded(postings, MAX_FILE_BYTES, "--postings")?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("--postings is not UTF-8: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line)
            .map_err(|e| format!("--postings line {}: not JSON: {e}", i + 1))?;
        let get = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("--postings line {}: no string {k:?}", i + 1))
        };
        rows.push((get("key")?, get("text")?));
    }
    let door = PostingDoor::build(seed, &rows)?;
    door.write(out)?;
    Ok(format!(
        "qd-prep posting-door-build: {} held-out postings, seed {seed} -> {}",
        door.len(),
        out.display()
    ))
}
