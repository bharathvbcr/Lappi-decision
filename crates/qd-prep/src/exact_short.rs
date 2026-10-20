//! The exact whole-text tier for short targets: ONE matcher for the builder and the gate.
//!
//! The word n-gram scan (`containment`) cannot see a target under n words: it has no n-gram
//! (`GAP-CLINC-TEST-TARGETS-8GRAM-BLIND-2026-10-06`: 2,343 of CLINC's 5,500 test utterances).
//! This tier is the owner's ruling of 2026-10-19
//! (`AUDIT/harden-2026-10-17/human-ruling-2026-10-19.md`, Fable Q22/Q23): a source row is a hit
//! of a short target (1 to n-1 words) exactly when its WHOLE template-stripped text,
//! lower-cased and reduced to its `\w+` words joined by one space ([`normalised`]), equals the
//! target's. A target equal to one line inside a longer row is not a hit; it is an *embedded*
//! match ([`ShortTargets::embedded`]), reported as a counted residual and never dropped
//! silently.
//!
//! `qd-prep containment` runs it on every source row too short to have an n-gram (a row that
//! can equal a short target has fewer than n words, so it has none), writing its pairs and its
//! exclusion keys through the same `pairs.tsv` / `exclusions.txt` seam as the n-gram scan.
//! `qd-train-data` calls the same functions over the train set's text. The text both read is
//! the strip-v2 text (`qd_train.containment_strip`); this module does not strip.

use std::collections::BTreeMap;

use crate::pyunicode;

/// The rule's name, written into the containment attestation (`exact_short.rule`) and required
/// by `qd_train.exclusions.read_exclusions`, so a list made without this tier is refused.
pub const EXACT_SHORT_RULE: &str = "exact-short-v1: a source row whose whole template-stripped \
    text, lower-cased and reduced to its \\w+ words joined by one space, equals a target of 1 to \
    n-1 words is a hit (pairs carry shared=0 target_ngrams=0); a target equal to one line inside \
    a longer row is an embedded match, reported by qd-train-data and never a hit";

/// `text` lower-cased and reduced to its `\w+` words joined by one space.
pub fn normalised(text: &str) -> String {
    let lowered = pyunicode::lower(text);
    pyunicode::words(&lowered).join(" ")
}

/// How many `\w+` words `text` has.
pub fn word_count(text: &str) -> usize {
    let lowered = pyunicode::lower(text);
    pyunicode::words(&lowered).len()
}

/// A target set's short targets (1 to n-1 words), by normalised text.
#[derive(Clone, Debug, Default)]
pub struct ShortTargets {
    /// Normalised text -> the target rows (indices into the set) holding it, ascending.
    by_text: BTreeMap<String, Vec<usize>>,
    n: usize,
}

impl ShortTargets {
    /// The short targets among `texts` (`(row index, text)`), for n-grams of `n` words.
    pub fn build<'a>(texts: impl IntoIterator<Item = (usize, &'a str)>, n: usize) -> Self {
        let mut by_text: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, text) in texts {
            let norm = normalised(text);
            let words = norm.split(' ').filter(|w| !w.is_empty()).count();
            if (1..n).contains(&words) {
                by_text.entry(norm).or_default().push(i);
            }
        }
        ShortTargets { by_text, n }
    }

    /// The n this was built for.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Short target rows (not distinct texts).
    pub fn rows(&self) -> usize {
        self.by_text.values().map(Vec::len).sum()
    }

    /// Distinct short target texts.
    pub fn distinct(&self) -> usize {
        self.by_text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_text.is_empty()
    }

    /// Every distinct normalised short target text and its rows, in text order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[usize])> {
        self.by_text.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// The target rows whose normalised text equals the whole of `text`'s: the hit.
    pub fn whole(&self, text: &str) -> Option<&[usize]> {
        self.by_text.get(&normalised(text)).map(Vec::as_slice)
    }

    /// The normalised short targets that are a whole line (or the whole text) of `text`: what
    /// the superseded per-line rule called a hit. Today only a residual count.
    pub fn embedded<'a>(&'a self, text: &str) -> Vec<&'a str> {
        let mut found: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut probe = |segment: &str| {
            if let Some((key, _)) = self.by_text.get_key_value(&normalised(segment)) {
                found.insert(key.as_str());
            }
        };
        probe(text);
        for line in text.lines() {
            probe(line);
        }
        found.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_lowers_and_drops_punctuation() {
        assert_eq!(
            normalised("  Set a Timer, for TEN minutes! "),
            "set a timer for ten minutes"
        );
        assert_eq!(normalised("what's up"), "what s up");
        assert_eq!(word_count("what's up"), 3);
    }

    #[test]
    fn whole_matches_only_the_whole_text_and_embedded_matches_a_line() {
        let targets = ShortTargets::build(
            [
                (0, "Thank you!"),
                (1, "a much longer target of many words here today"),
            ],
            8,
        );
        assert_eq!(targets.distinct(), 1);
        assert_eq!(targets.whole("thank  YOU"), Some(&[0][..]));
        assert_eq!(targets.whole("please thank you"), None);
        let text = "a longer row\nthank you\nmore text";
        assert_eq!(targets.whole(text), None);
        assert_eq!(targets.embedded(text), vec!["thank you"]);
    }

    #[test]
    fn a_target_of_n_or_more_words_is_not_short_and_an_empty_one_is_nothing() {
        let t = ShortTargets::build([(0, "one two three"), (1, "..."), (2, "x")], 3);
        assert_eq!(t.rows(), 1, "only 'x' is under 3 words and non-empty");
        assert_eq!(t.whole("one two three"), None);
    }
}
