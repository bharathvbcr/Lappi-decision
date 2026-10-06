//! The seam every v6 pool producer writes through, after its own rows are made: the pinned
//! decontamination targets read, the containment scan, the examples file and the facts a
//! manifest records about the rows (strata, `noul` count, gold positions). A producer builds
//! [`Candidate`]s its own way (`qd-prep synth` by rule, `qd-prep convert` from downloaded
//! sources) and hands them here, so the scan and the row format have one owner.
//!
//! `qd-prep decisions` predates this module and keeps its own copy of the same steps (its
//! `run`), which also applies the cap table; folding it in here is the rebuild lane's work
//! (GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::containment;
use crate::decisions::{self, Candidate, Gold, TargetSet};
use crate::sha256::sha256_hex;

/// Every `--target NAME=FILE` set, read as `{"id", "text"}` JSONL and checked against the
/// config's sha256 pin. The given names must be exactly the pinned names; an empty set is
/// refused, because a scan against nothing is not a check.
pub fn read_targets(
    pins: &BTreeMap<String, String>,
    given: &[(String, PathBuf)],
) -> Result<(Vec<TargetSet>, BTreeMap<String, String>), String> {
    let names: BTreeSet<&String> = given.iter().map(|(n, _)| n).collect();
    let pinned: BTreeSet<&String> = pins.keys().collect();
    if names != pinned || names.len() != given.len() {
        return Err(format!(
            "targets {names:?} are not exactly the config's pinned {pinned:?}, each once"
        ));
    }
    let mut out = Vec::new();
    let mut digests = BTreeMap::new();
    for (name, path) in given {
        let mut rows = Vec::new();
        let got = decisions::for_lines(path, |n, r| {
            let what = format!("{name}:{n}");
            rows.push((
                decisions::get_str(&r, "id", &what)?.to_owned(),
                decisions::get_str(&r, "text", &what)?.to_owned(),
            ));
            Ok(())
        })?;
        decisions::check_pin(
            path,
            &got,
            &pins[name],
            &format!("the config (target {name})"),
        )?;
        if rows.is_empty() {
            return Err(format!(
                "target set {name} is empty: a scan against nothing is not a check"
            ));
        }
        digests.insert(format!("target/{name}"), got);
        out.push((name.clone(), rows));
    }
    Ok((out, digests))
}

/// The rows that survived the scan, the containment record and the manifest's account of it.
pub struct Scanned<'c> {
    pub clean: Vec<&'c Candidate>,
    pub written: containment::Written,
    pub report: Value,
}

/// Scan `candidates` against every target set by word n-gram containment (the scan
/// `qd-prep decisions` runs) and drop every hit. Refuses when nothing survives.
pub fn decontaminate<'c>(
    ngram_n: u32,
    threshold: f64,
    tool: &Value,
    candidate_set: &str,
    candidates: &'c [Candidate],
    targets: &[TargetSet],
    threads: usize,
) -> Result<Scanned<'c>, String> {
    let request_bytes = decisions::containment_request_with(
        ngram_n,
        threshold,
        tool,
        candidate_set,
        candidates,
        targets,
    );
    let request = containment::parse(&request_bytes)?;
    let found = request.scan(threads)?;
    let mut hit: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for p in &found.pairs {
        let scan = request.scans[p.scan];
        let key = request.sets[scan.source][p.source_row].key;
        hit.entry(request.set_names[scan.target].to_owned())
            .or_default()
            .insert(key.to_owned());
    }
    let excluded: BTreeSet<&str> = hit.values().flatten().map(String::as_str).collect();
    let clean: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| !excluded.contains(c.id.as_str()))
        .collect();
    if clean.is_empty() {
        return Err("no row survived decontamination: a pool of nothing is refused".into());
    }
    let report = json!({
        "n": ngram_n, "threshold": threshold,
        "excluded_by_target": hit.iter().map(|(k, v)| (k.clone(), json!(v.len())))
            .collect::<serde_json::Map<_, _>>(),
        "excluded_total": excluded.len(),
        // A row shorter than n words has no n-grams: it cannot be matched, and an unscanned
        // row must not read as clean.
        "rows_too_short_by_set": request.set_names.iter().zip(&found.sets)
            .map(|(name, s)| ((*name).to_owned(), json!({"rows": s.rows, "too_short": s.too_short})))
            .collect::<serde_json::Map<_, _>>(),
    });
    Ok(Scanned {
        clean,
        written: request.render(&found),
        report,
    })
}

/// `examples.jsonl`'s bytes and what the manifest records about them.
pub struct Examples {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub rows: usize,
    /// Per stratum: `{"train", "val"}` row counts.
    pub selected: Value,
    pub noul: usize,
    /// Per family: rows by the gold option's position, or `noul`. A source whose gold sits in
    /// one position is visible here (the PriDe lever).
    pub gold_position: Value,
    /// Strata and families with val rows and no train rows
    /// ([`decisions::val_strata_without_train`]): a val row there measures something the model
    /// never trained on. Report only; nothing is dropped (rule 2).
    pub val_without_train: Value,
}

/// Serialise `rows` in the `qd-decisions/v1` row format (`decisions::example_json`).
pub fn examples(rows: &[&Candidate]) -> Result<Examples, String> {
    let mut bytes = Vec::new();
    let mut selected: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut gold_position: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut noul = 0usize;
    for c in rows {
        serde_json::to_writer(&mut bytes, &decisions::example_json(c))
            .map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        let s = selected.entry(c.stratum.as_str()).or_default();
        if c.split == "val" {
            s.1 += 1;
        } else {
            s.0 += 1;
        }
        let pos = match c.gold {
            Gold::Option(i) => i.to_string(),
            Gold::Noul => {
                noul += 1;
                "noul".to_owned()
            }
        };
        *gold_position
            .entry(c.family_id.as_str())
            .or_default()
            .entry(pos)
            .or_default() += 1;
    }
    let avail: BTreeMap<String, decisions::Avail> = selected
        .iter()
        .map(|(k, (t, v))| ((*k).to_owned(), decisions::Avail { train: *t, val: *v }))
        .collect();
    Ok(Examples {
        val_without_train: decisions::val_without_train_json(&avail),
        sha256: sha256_hex(&bytes),
        bytes,
        rows: rows.len(),
        selected: selected
            .iter()
            .map(|(k, (t, v))| ((*k).to_owned(), json!({"train": t, "val": v})))
            .collect::<serde_json::Map<_, _>>()
            .into(),
        noul,
        gold_position: json!(gold_position),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, context: &str, gold: Gold, split: &'static str) -> Candidate {
        Candidate {
            id: id.to_owned(),
            source_id: "t/source",
            family_id: "t.family".to_owned(),
            stratum: "t.family/x".to_owned(),
            group_key: id.to_owned(),
            licence: "mit".to_owned(),
            context: context.to_owned(),
            question: "Which?".to_owned(),
            slot_name: "answer".to_owned(),
            options: vec!["alpha".into(), "beta".into()],
            gold,
            label_basis: "hard",
            split,
        }
    }

    #[test]
    fn a_row_containing_a_target_is_dropped_and_counted() {
        let target_text = "the quick brown fox jumps over the lazy dog near the river bank today";
        let rows = vec![
            cand("a", target_text, Gold::Option(0), "train"),
            cand(
                "b",
                "an entirely unrelated sentence about compilers and parsers and grammars here",
                Gold::Option(1),
                "val",
            ),
        ];
        let targets: Vec<TargetSet> = vec![(
            "t".to_owned(),
            vec![("t1".to_owned(), target_text.to_owned())],
        )];
        let s =
            decontaminate(8, 0.5, &json!({"tool": "test"}), "rows", &rows, &targets, 1).unwrap();
        let ids: Vec<&str> = s.clean.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["b"]);
        assert_eq!(s.report["excluded_total"], 1);
        assert_eq!(s.report["excluded_by_target"]["t"], 1);
    }

    #[test]
    fn nothing_surviving_is_refused() {
        let text = "the quick brown fox jumps over the lazy dog near the river bank today";
        let rows = vec![cand("a", text, Gold::Option(0), "train")];
        let targets: Vec<TargetSet> =
            vec![("t".to_owned(), vec![("t1".to_owned(), text.to_owned())])];
        assert!(decontaminate(8, 0.5, &json!({}), "rows", &rows, &targets, 1).is_err());
    }

    #[test]
    fn examples_count_strata_noul_and_gold_positions() {
        let rows = [
            cand("a", "x", Gold::Option(1), "train"),
            cand("b", "y", Gold::Noul, "val"),
        ];
        let refs: Vec<&Candidate> = rows.iter().collect();
        let e = examples(&refs).unwrap();
        assert_eq!(e.rows, 2);
        assert_eq!(e.noul, 1);
        assert_eq!(e.selected["t.family/x"]["train"], 1);
        assert_eq!(e.selected["t.family/x"]["val"], 1);
        assert_eq!(e.gold_position["t.family"]["1"], 1);
        assert_eq!(e.gold_position["t.family"]["noul"], 1);
        assert_eq!(e.bytes.iter().filter(|b| **b == b'\n').count(), 2);
        assert_eq!(e.sha256, sha256_hex(&e.bytes));
    }

    /// A stratum with val rows and no train row is reported at the seam, so every producer
    /// (synth and convert, not only decisions) names it; nothing is dropped.
    #[test]
    fn a_val_only_stratum_is_reported_and_kept() {
        let mut only_val = cand("c", "z", Gold::Option(0), "val");
        only_val.stratum = "t.family/val-only".to_owned();
        let rows = [
            cand("a", "x", Gold::Option(1), "train"),
            cand("b", "y", Gold::Option(0), "val"),
            only_val,
        ];
        let refs: Vec<&Candidate> = rows.iter().collect();
        let e = examples(&refs).unwrap();
        assert_eq!(e.rows, 3);
        assert_eq!(
            e.val_without_train,
            json!({"strata": {"t.family/val-only": 1}, "families": {}})
        );
    }

    #[test]
    fn a_target_set_not_pinned_or_given_twice_is_refused() {
        let pins: BTreeMap<String, String> = [("a".to_owned(), "0".repeat(64))].into();
        let twice = [
            ("a".to_owned(), PathBuf::from("/x")),
            ("a".to_owned(), PathBuf::from("/y")),
        ];
        assert!(read_targets(&pins, &twice).is_err());
        let other = [("b".to_owned(), PathBuf::from("/x"))];
        assert!(read_targets(&pins, &other).is_err());
    }
}
