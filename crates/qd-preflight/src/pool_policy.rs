//! Who may receive the adapter a pool trains, and where that pool may train (v2 plan section 6,
//! gates 4 and 5; owner rulings 6, 7 and 12 of 2026-10-20).
//!
//! Three kinds of row are scoped out of a **public** adapter: share-alike licences (CC BY-SA), a
//! research-only grant, and `vani-agreed` rows (real teacher readings on real postings). One kind
//! never leaves the **Mac**: `vani-agreed`. This module is the single reader of those two rules
//! for the preflight binaries; the Python side's twin is `qd_data.licences.private_only_reason`
//! and the two are checked against the same markers in the tests.
//!
//! * `adapter_visibility` is a field of the pool (train) manifest: `"public"` or `"private"`. An
//!   **absent** field is read as public. That is the stricter reading: a manifest that forgot the
//!   field is refused for a restricted row, where the opposite default would let that row reach a
//!   published adapter. An unrecognised value is refused, not guessed.
//! * A pool is scanned from its `entries` (`source_id`, `licence_id` per row) and its
//!   `licence_histogram`. A manifest whose rows cannot be read is never "clean": the callers map
//!   a [`PolicyError`] to `FAIL` (release) or `NOT_RUN` (train data), never to a pass.
//!
//! Nothing here widens access: every function either leaves a verdict as it was or refuses.

use qd_prep::vani_pool::VANI_AGREED_SOURCE;
use serde_json::Value;

/// Most row ids a finding names.
pub const SAMPLE_IDS: usize = 8;

/// A licence id carrying this (after `_` and space become `-`) is a CC BY-SA licence.
const SHARE_ALIKE_MARKER: &str = "by-sa";
/// A licence id carrying this is a research-only grant (`qd_data.licences.RESEARCH_ONLY_MARKERS`).
const RESEARCH_ONLY_MARKER: &str = "research-only";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

/// Where a run trains. A run whose target is not stated is neither: see [`check_run_target`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunTarget {
    Mac,
    Cloud,
}

impl RunTarget {
    pub fn parse(raw: &str) -> Result<Self, PolicyError> {
        match raw {
            "mac" => Ok(RunTarget::Mac),
            "cloud" => Ok(RunTarget::Cloud),
            other => Err(PolicyError(format!(
                "run target must be 'mac' or 'cloud', got {other:?}"
            ))),
        }
    }
}

/// A manifest the policy cannot be read from. Never a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The manifest's `adapter_visibility`: absent is public, a bad value is an error.
pub fn parse_visibility(field: Option<&Value>) -> Result<Visibility, PolicyError> {
    match field {
        None | Some(Value::Null) => Ok(Visibility::Public),
        Some(Value::String(s)) if s == "public" => Ok(Visibility::Public),
        Some(Value::String(s)) if s == "private" => Ok(Visibility::Private),
        Some(other) => Err(PolicyError(format!(
            "adapter_visibility must be \"public\" or \"private\" (or absent, read as public), got {other}"
        ))),
    }
}

/// A licence id with `_` and spaces read as `-`, lower-cased and trimmed.
fn licence_key(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c == '_' || c.is_whitespace() { '-' } else { c })
        .collect()
}

pub fn is_share_alike(licence_id: &str) -> bool {
    licence_key(licence_id).contains(SHARE_ALIKE_MARKER)
}

pub fn is_research_only(licence_id: &str) -> bool {
    licence_key(licence_id).contains(RESEARCH_ONLY_MARKER)
}

/// A source id (or `source`, or licence id) that names the real-Vani row source: the agreed
/// source itself or a sub-source under it (`vani-agreed/role_fit`, `vani-agreed:seniority`).
pub fn is_vani_agreed(source: &str) -> bool {
    let key = source.trim().to_lowercase();
    key.strip_prefix(VANI_AGREED_SOURCE)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(['/', ':', '.', '-', '_', ' ']))
}

/// A count and a few ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub count: u64,
    pub sample: Vec<String>,
}

impl Tally {
    fn add(&mut self, id: &str) {
        self.count += 1;
        if self.sample.len() < SAMPLE_IDS {
            self.sample.push(id.to_owned());
        }
    }
}

/// What a pool holds that a public adapter or a cloud run may not.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Findings {
    pub rows: u64,
    pub share_alike: Tally,
    pub research_only: Tally,
    pub vani_agreed: Tally,
}

impl Findings {
    /// True when a public adapter may not be trained on this pool.
    pub fn restricted_for_public(&self) -> bool {
        self.share_alike.count + self.research_only.count + self.vani_agreed.count > 0
    }

    fn describe(&self) -> String {
        format!(
            "{} share-alike (BY-SA) {:?}, {} research-only {:?}, {} vani-agreed {:?}, of {} rows",
            self.share_alike.count,
            self.share_alike.sample,
            self.research_only.count,
            self.research_only.sample,
            self.vani_agreed.count,
            self.vani_agreed.sample,
            self.rows
        )
    }
}

fn str_field<'a>(entry: &'a Value, key: &str) -> Option<&'a str> {
    entry.get(key).and_then(Value::as_str)
}

/// Scan a train manifest (`Manifest.to_json()`): every entry's `source_id`, `source` and
/// `licence_id`, and the `licence_histogram` keys with a non-zero count. A manifest with no
/// `entries` array, no entry, or an entry that is not an object with a string `licence_id` and
/// `source_id` cannot be scanned and is refused: a row the scan skipped is a row it did not clear.
pub fn scan_manifest(doc: &Value) -> Result<Findings, PolicyError> {
    let entries = doc
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| PolicyError("the manifest has no entries array to scan".into()))?;
    if entries.is_empty() {
        return Err(PolicyError(
            "the manifest has no entries: a pool of nothing names nothing to clear".into(),
        ));
    }
    let mut found = Findings {
        rows: u64::try_from(entries.len()).map_err(|_| PolicyError("entry count overflows".into()))?,
        ..Findings::default()
    };
    for (i, entry) in entries.iter().enumerate() {
        let (Some(source), Some(licence)) =
            (str_field(entry, "source_id"), str_field(entry, "licence_id"))
        else {
            return Err(PolicyError(format!(
                "entry {i} has no string source_id and licence_id; a row that cannot be read is not a row that was cleared"
            )));
        };
        let id = str_field(entry, "row_id").map_or_else(|| format!("#{i}"), str::to_owned);
        if is_share_alike(licence) {
            found.share_alike.add(&id);
        }
        if is_research_only(licence) {
            found.research_only.add(&id);
        }
        let named_source = str_field(entry, "source");
        if is_vani_agreed(source) || is_vani_agreed(licence) || named_source.is_some_and(is_vani_agreed) {
            found.vani_agreed.add(&id);
        }
    }
    // The histogram is the manifest's own account of what it holds; a licence it counts that the
    // entries do not show is a disagreement the scan must not clear.
    if let Some(hist) = doc.get("licence_histogram").and_then(Value::as_object) {
        for (licence, n) in hist {
            let nonzero = n.as_u64().is_none_or(|n| n > 0);
            if !nonzero {
                continue;
            }
            if is_share_alike(licence) && found.share_alike.count == 0 {
                found.share_alike.add(&format!("licence_histogram[{licence}]"));
            }
            if is_research_only(licence) && found.research_only.count == 0 {
                found.research_only.add(&format!("licence_histogram[{licence}]"));
            }
        }
    }
    Ok(found)
}

/// `Ok(detail)` when a pool with these findings may train an adapter of this visibility; `Err`
/// names what a public adapter may not carry.
pub fn check_visibility(findings: &Findings, visibility: Visibility) -> Result<String, String> {
    match visibility {
        Visibility::Private => Ok(format!(
            "adapter_visibility is private; the pool holds {}",
            findings.describe()
        )),
        Visibility::Public if findings.restricted_for_public() => Err(format!(
            "adapter_visibility is public (or absent, which reads as public), and the pool holds rows a published adapter may not carry: {}",
            findings.describe()
        )),
        Visibility::Public => Ok(format!(
            "adapter_visibility is public and the pool holds no BY-SA, research-only or vani-agreed row ({} rows)",
            findings.rows
        )),
    }
}

/// Ruling 12: real Vani rows never leave the Mac. `target` is `None` when the run did not say
/// where it trains, which is read as not-Mac: unproven is refused.
pub fn check_run_target(vani_agreed_rows: u64, target: Option<RunTarget>) -> Result<String, String> {
    match (vani_agreed_rows, target) {
        (0, _) => Ok("no vani-agreed row is in the pool".to_owned()),
        (n, Some(RunTarget::Mac)) => Ok(format!(
            "{n} vani-agreed rows, and the run is stated to train on the Mac"
        )),
        (n, Some(RunTarget::Cloud)) => Err(format!(
            "{n} vani-agreed rows in a cloud run: real Vani rows never leave the Mac (ruling 12)"
        )),
        (n, None) => Err(format!(
            "{n} vani-agreed rows and the run does not state where it trains (--run-target mac): \
             an unproven target is read as cloud, because real Vani rows never leave the Mac (ruling 12)"
        )),
    }
}
