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
        // An id names one row: the scan reports exclusions by id, so a repeated one would make
        // two rows indistinguishable in the record. Refused with both line numbers.
        let mut first_line: BTreeMap<String, usize> = BTreeMap::new();
        let got = decisions::for_lines(path, decisions::Role::Reference, |n, r| {
            let what = format!("{name}:{n}");
            let id = decisions::get_str(&r, "id", &what)?.to_owned();
            if let Some(first) = first_line.insert(id.clone(), n) {
                return Err(format!(
                    "{}:{n}: target set {name}: id {id:?} repeats line {first}",
                    path.display()
                ));
            }
            rows.push((id, decisions::get_str(&r, "text", &what)?.to_owned()));
            Ok(())
        })?;
        decisions::check_pin(
            path,
            &got.sha256,
            &pins[name],
            &format!("the config (target {name})"),
        )?;
        if rows.is_empty() {
            return Err(format!(
                "target set {name} is empty: a scan against nothing is not a check"
            ));
        }
        decisions::record_input(&mut digests, format!("target/{name}"), got);
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
    // Refused here, not only by `read_targets`: this is `pub`, and a caller that built its
    // target sets another way must not get a scan against nothing reported as a scan.
    if targets.is_empty() {
        return Err("no target set: a decontamination against nothing is not a check".into());
    }
    if let Some((name, _)) = targets.iter().find(|(_, rows)| rows.is_empty()) {
        return Err(format!(
            "target set {name} is empty: a scan against nothing is not a check"
        ));
    }
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

/// The manifest's `allocation.state` when a cap table chose the rows (`qd-prep decisions`).
pub const ALLOCATION_APPLIED: &str = "applied";
/// The manifest's `allocation.state` for a candidate pool no cap table has drawn from yet
/// (`qd-prep synth`, `qd-prep convert`).
pub const ALLOCATION_NOT_APPLIED: &str = "not_applied";

/// The manifest's `allocation` block. `qd_data.decisions.load_decision_pool` admits only
/// [`ALLOCATION_APPLIED`], so a candidate pool cannot enter a mixture until an allocation over
/// every pool has drawn from it: otherwise families would be weighted by how many rows a
/// generator or source happens to yield
/// (GAP-V6-THREE-POOL-PRODUCERS-CAPS-APPLIED-IN-DECISIONS-ONLY-2026-10-06).
pub fn allocation(applied: bool, detail: &str) -> Value {
    let state = if applied {
        ALLOCATION_APPLIED
    } else {
        ALLOCATION_NOT_APPLIED
    };
    json!({"state": state, "detail": detail})
}

/// Serialise `rows` in the `qd-decisions/v1` row format (`decisions::example_json`).
pub fn examples(rows: &[&Candidate]) -> Result<Examples, String> {
    let mut bytes = Vec::new();
    let mut selected: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut gold_position: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut noul = 0usize;
    for c in rows {
        // The last gate before bytes: a row a producer let through malformed is refused here by
        // name, never written (and never a panic on an out-of-range gold).
        if let Some(reason) = decisions::structural_refusal(c) {
            return Err(format!(
                "pool row {}: {reason}; every row is checked before it is written",
                c.id
            ));
        }
        if c.split != "train" && c.split != "val" {
            return Err(format!(
                "pool row {}: split {:?}; a pool holds only train and val rows",
                c.id, c.split
            ));
        }
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
    fn a_repeated_target_id_is_refused_with_both_line_numbers() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("qd-prep-pool-dup-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.jsonl");
        let body = b"{\"id\":\"a\",\"text\":\"x\"}\n{\"id\":\"b\",\"text\":\"y\"}\n{\"id\":\"a\",\"text\":\"z\"}\n";
        std::fs::write(&path, body).unwrap();
        let pins: BTreeMap<String, String> = [("t".to_owned(), sha256_hex(body))].into();
        let Err(err) = read_targets(&pins, &[("t".to_owned(), path)]) else {
            panic!("not refused")
        };
        assert!(
            err.contains("t.jsonl:3:") && err.contains("repeats line 1"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_scan_with_no_target_set_or_an_empty_one_is_refused() {
        let rows = vec![cand("a", "some text to scan", Gold::Option(0), "train")];
        let Err(err) = decontaminate(8, 0.5, &json!({}), "rows", &rows, &[], 1) else {
            panic!("not refused")
        };
        assert!(err.contains("against nothing"), "{err}");
        let empty: Vec<TargetSet> = vec![("t".to_owned(), Vec::new())];
        let Err(err) = decontaminate(8, 0.5, &json!({}), "rows", &rows, &empty, 1) else {
            panic!("not refused")
        };
        assert!(err.contains("target set t is empty"), "{err}");
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

    /// The last gate before bytes refuses by name what a producer let through: before this
    /// check an out-of-range gold panicked in `example_json`'s index (exit 101), and a held-out
    /// row was written as train.
    #[test]
    fn examples_refuse_a_malformed_or_held_out_row_by_name() {
        let bad_gold = cand("g", "x", Gold::Option(5), "train");
        let Err(err) = examples(&[&bad_gold]) else {
            panic!("not refused")
        };
        assert!(err.contains("pool row g: gold_not_in_options"), "{err}");
        let mut many = cand("m", "x", Gold::Option(0), "train");
        many.options = (0..=decisions::MAX_OPTIONS)
            .map(|i| format!("o{i}"))
            .collect();
        let Err(err) = examples(&[&many]) else {
            panic!("not refused")
        };
        assert!(err.contains("too_many_options"), "{err}");
        let held = cand("h", "x", Gold::Option(0), "heldout");
        let Err(err) = examples(&[&held]) else {
            panic!("not refused")
        };
        assert!(err.contains("split \"heldout\""), "{err}");
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
