//! The licence policy `qd-prep convert` applies to every row, as a mirror of
//! `python/qd_data/licences.py` (Python is the oracle, per this crate's rule; the parity test
//! `tests/convert_licence_parity.rs` runs Python's `classify` on every string this table is fed
//! and fails on any difference).
//!
//! Mirrored and nothing more: the table's ids and tiers, `_ALIASES`, `normalise_licence` and the
//! segment-exact non-commercial fallback. An alias Python lacks is NOT added here, so a display
//! name such as `BSD 3-Clause "New" or "Revised" License` (SWE-rebench) is `needs_human_call` in
//! both until `qd_data.licences` registers it.
//!
//! One addition the loader already implies: a dataset-level licence the conversion can state but
//! Python has not registered yet ([`PENDING`]) keeps its rows in the pool, stamped with that id.
//! `qd_data.decisions.load_decision_pool` then default-denies them (`admit_licence`) until the
//! id is registered with a human's note -- the `synthetic-by-rule` precedent of plan section 8.
//! The parity test fails the day Python admits a pending id, so the two lists cannot drift.

/// `qd_data.licences.LicenceTier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Allow,
    NeedsHumanCall,
    Disqualifying,
}

impl Tier {
    /// Python's `LicenceTier.value`.
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Allow => "allow",
            Tier::NeedsHumanCall => "needs_human_call",
            Tier::Disqualifying => "disqualifying",
        }
    }
}

use Tier::{Allow as A, Disqualifying as D, NeedsHumanCall as H};

/// `qd_data.licences.LICENCE_POLICY`: every id and its tier, in the table's order.
pub const POLICY: [(&str, Tier); 24] = [
    ("mit", A),
    ("apache-2.0", A),
    ("bsd-2-clause", A),
    ("bsd-3-clause", A),
    ("isc", A),
    ("cc0-1.0", A),
    ("unlicense", A),
    ("cc-by-3.0", A),
    ("cc-by-4.0", A),
    ("cc-by-sa-3.0", A),
    ("cc-by-sa-4.0", A),
    ("owner-granted", A),
    ("agpl-3.0", H),
    ("lgpl-2.1", H),
    ("epl-1.0", H),
    ("mpl-2.0", H),
    ("artistic-2.0", H),
    ("unknown", H),
    ("other", H),
    ("c-uda", H),
    ("cc-by-nc-4.0", D),
    ("cc-by-nc-sa-4.0", D),
    ("cc-by-nc-nd-4.0", D),
    ("cc-by-nc-3.0", D),
];

/// `qd_data.licences._ALIASES`.
pub const ALIASES: [(&str, &str); 10] = [
    ("apache 2.0", "apache-2.0"),
    ("apache-2", "apache-2.0"),
    ("apache2.0", "apache-2.0"),
    ("bsd-2", "bsd-2-clause"),
    ("bsd-3", "bsd-3-clause"),
    ("cc0", "cc0-1.0"),
    ("cc-0", "cc0-1.0"),
    ("mit license", "mit"),
    ("the unlicense", "unlicense"),
    ("public domain", "cc0-1.0"),
];

/// Dataset-level licences this conversion states but `qd_data.licences` has not registered.
/// Each is `needs_human_call` in Python (the parity test checks it), so rows carrying one are
/// built but cannot be loaded until a human registers the id. `(id, the note for the manifest)`.
pub const PENDING: [(&str, &str); 2] = [
    (
        "odc-by-1.0",
        "allenai/scirepeval: ODC-BY 1.0 aggregate licence; attribution required (cite \
         SciRepEval, Singh et al. 2023, and the Allen Institute for AI). The search config has \
         no external constituent per the SciRepEval GitHub README table. Not yet in \
         qd_data.licences: the loader refuses these rows until it is registered",
    ),
    (
        "oanc",
        "nyu-mll/multi_nli non-fiction genres (government, slate, telephone, travel): the Open \
         American National Corpus licence, which the MultiNLI paper describes as allowing free \
         use, modification and sharing; the fiction genre (which includes CC-BY-SA-3.0 text) is \
         refused per row. Not yet in qd_data.licences: the loader refuses these rows until it \
         is registered",
    ),
];

/// Python's `str.split()` separators for ASCII and the C0 separators it adds (U+001C..U+001F).
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `qd_data.licences.normalise_licence`: casefold, collapse whitespace, then the alias table.
/// `str.casefold()` and [`crate::pyunicode::lower`] agree on ASCII; a non-ASCII licence string
/// is lowered the Python `lower()` way, and the parity test feeds only strings this crate meets.
pub fn normalise(raw: &str) -> String {
    let words: Vec<&str> = raw.split(is_py_space).filter(|w| !w.is_empty()).collect();
    let key = crate::pyunicode::lower(&words.join(" "));
    match ALIASES.iter().find(|(k, _)| *k == key) {
        Some((_, v)) => (*v).to_owned(),
        None => key,
    }
}

/// `qd_data.licences.is_non_commercial`: a `nc` segment, never a substring (`unlicense`).
pub fn is_non_commercial(id: &str) -> bool {
    normalise(id).split('-').any(|s| s == "nc")
}

/// `qd_data.licences.classify`: the normalised id and its tier. An id outside the table is
/// `needs_human_call`, or `disqualifying` when it carries a non-commercial segment.
pub fn classify(raw: &str) -> (String, Tier) {
    let key = normalise(raw);
    if let Some((_, t)) = POLICY.iter().find(|(id, _)| *id == key) {
        return (key, *t);
    }
    if is_non_commercial(&key) {
        return (key, D);
    }
    (key, H)
}

/// What the conversion does with a row's licence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// On Python's allowlist: the row is admitted and loadable.
    Admitted(String),
    /// Stated, in [`PENDING`]: the row is built, and refused by the loader until registered.
    Pending(String),
}

impl Verdict {
    pub fn id(&self) -> &str {
        match self {
            Verdict::Admitted(s) | Verdict::Pending(s) => s,
        }
    }
}

/// The licence gate every converted row passes. `None` (the source states no licence for the
/// row) is refused as `licence_unstated`; every other refusal names the id and its tier, so the
/// manifest's count says why. Never silent: the caller counts the `Err` as a refusal.
pub fn gate(raw: Option<&str>) -> Result<Verdict, String> {
    let raw = match raw.map(str::trim) {
        None | Some("") => return Err("licence_unstated".to_owned()),
        Some(r) => r,
    };
    let (id, tier) = classify(raw);
    match tier {
        A => Ok(Verdict::Admitted(id)),
        H if PENDING.iter().any(|(p, _)| *p == id) => Ok(Verdict::Pending(id)),
        t => Err(format!("licence_{}:{id}", t.as_str())),
    }
}

/// The manifest note for a pending id.
pub fn pending_note(id: &str) -> Option<&'static str> {
    PENDING.iter().find(|(p, _)| *p == id).map(|(_, n)| *n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unstated_licence_is_refused_with_its_own_reason() {
        assert_eq!(gate(None), Err("licence_unstated".to_owned()));
        assert_eq!(gate(Some("  ")), Err("licence_unstated".to_owned()));
    }

    #[test]
    fn aliases_and_whitespace_are_python_s_and_no_more() {
        assert_eq!(classify("  MIT   License "), ("mit".to_owned(), A));
        assert_eq!(classify("The Unlicense"), ("unlicense".to_owned(), A));
        // Not a Python alias, so not one here: a human registers it, the mirror follows.
        assert_eq!(classify("MIT License (Expat)").1, H);
        assert_eq!(classify("BSD 3-Clause \"New\" or \"Revised\" License").1, H);
        assert_eq!(classify("BSD License").1, H);
        assert_eq!(classify("Apache License 2.0").1, H);
    }

    #[test]
    fn non_commercial_is_disqualifying_by_segment_not_substring() {
        assert_eq!(classify("cc-by-nc-sa-3.0").1, D);
        assert_eq!(classify("unlicense").1, A);
        assert!(
            gate(Some("CC-BY-NC-4.0"))
                .unwrap_err()
                .starts_with("licence_disqualifying")
        );
    }

    #[test]
    fn a_pending_dataset_licence_is_kept_and_named_never_admitted() {
        assert_eq!(
            gate(Some("ODC-BY-1.0")),
            Ok(Verdict::Pending("odc-by-1.0".to_owned()))
        );
        assert_eq!(gate(Some("oanc")), Ok(Verdict::Pending("oanc".to_owned())));
        assert!(
            gate(Some("other"))
                .unwrap_err()
                .starts_with("licence_needs_human_call")
        );
        assert!(pending_note("oanc").is_some());
    }
}
