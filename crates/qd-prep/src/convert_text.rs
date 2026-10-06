//! Text pools: MNLI (a three-way NLI choice) and SciRepEval `search` (pointwise relevance).
//!
//! **MNLI** (`nyu-mll/multi_nli` train). The licence filter is per row: genre `fiction` includes
//! CC-BY-SA-3.0 text and is refused (`licence_fiction_genre`); the four other genres are the
//! Open ANC's text, stamped `oanc` (pending registration in `qd_data.licences`). A genre outside
//! the five is refused (`unknown_genre`): its licence cannot be stated. Labels map 0/1/2 to
//! entailment/neutral/contradiction (the card's `class_label` names); anything else is refused.
//! `group_key` is `promptID`, so the hypotheses written for one premise never straddle the split;
//! the stratum carries the genre so one can be capped or refused later without a rebuild.
//!
//! **SciRepEval `search`** (ODC-BY 1.0, stamped `odc-by-1.0`, pending registration; the
//! attribution is in every manifest). Each row is a query with ten candidates scored 0..4 by
//! relevance; measured 2026-10-06 on one row group, 903 of 1,000 queries have exactly one
//! positive. v1's negatives are the same query's score-0 candidates. **Capped at read time**: a
//! seeded draw admits `query_fraction` of the queries before any candidate is materialised, then
//! each admitted query keeps at most `max_positives_per_query` positives and
//! `negatives_per_positive` score-0 candidates per kept positive (seeded). A query with no
//! positive is skipped and counted. Cross-query BM25 hard negatives are a follow-up (std-only
//! Rust), not built here.
//!
//! Id-disjointness, enforced: a train query whose id is a query of `search/evaluation` or of
//! `scirepeval_test/search`, and a train candidate whose paper id is a candidate there, are
//! refused and counted. `scirepeval_test/nfcorpus` and `trec_covid` hold ids from other
//! corpora (MED-*, CORD uids) and no text, so no check of them can run: reported `not_run`.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::convert::{self, Acc, Config, PoolFacts, Row, Views};
use crate::decisions::{self, Gold};

pub const MNLI: &str = "nyu-mll/multi_nli";
pub const MNLI_FILE: &str = "data/train-00000-of-00001.parquet";
pub const MNLI_FAMILY: &str = "mnli.nli";
pub const MNLI_OPTIONS: [&str; 3] = ["entailment", "neutral", "contradiction"];
pub const MNLI_QUESTION: &str =
    "Does the premise entail the hypothesis, contradict it, or neither (neutral)?";
/// The OANC genres of MNLI's train split; `fiction` is refused on licence.
pub const MNLI_GENRES: [&str; 4] = ["government", "slate", "telephone", "travel"];

pub const SCIREPEVAL: &str = "allenai/scirepeval";
pub const SCIREPEVAL_TEST: &str = "allenai/scirepeval_test";
pub const SCI_FAMILY: &str = "scirepeval.search_rel";
pub const SCI_SHARDS: usize = 14;
pub const SCI_INSTRUCTION: &str = "Instruction: Given a scientific literature search query, decide whether this paper is relevant to it.";
pub const SCI_QUESTION: &str = "Is this paper relevant to the search query?";
pub const YES_NO: [&str; 2] = ["yes", "no"];

/// Every MNLI row, as a row or the reason it is refused.
pub fn mnli_row(r: &Value) -> Result<Row, String> {
    let genre = convert::opt_str(r, "genre").ok_or("malformed")?;
    if genre == "fiction" {
        return Err("licence_fiction_genre".into());
    }
    if !MNLI_GENRES.contains(&genre) {
        return Err("unknown_genre".into());
    }
    let pair = convert::opt_str(r, "pairID").ok_or("malformed")?;
    let prompt = r
        .get("promptID")
        .and_then(Value::as_i64)
        .ok_or("malformed")?;
    let premise = convert::opt_str(r, "premise").ok_or("malformed")?;
    let hypothesis = convert::opt_str(r, "hypothesis").ok_or("malformed")?;
    let label = r.get("label").and_then(Value::as_i64).ok_or("malformed")?;
    let gold = usize::try_from(label)
        .ok()
        .filter(|l| *l < 3)
        .ok_or("unknown_label")?;
    Ok(Row {
        id: format!("mnli:{pair}"),
        source_id: MNLI,
        family_id: MNLI_FAMILY,
        stratum: format!("{MNLI_FAMILY}/{genre}"),
        group_key: prompt.to_string(),
        licence: Some("oanc".to_owned()),
        context: format!(
            "Premise: {}\nHypothesis: {}",
            premise.trim(),
            hypothesis.trim()
        ),
        question: MNLI_QUESTION.to_owned(),
        slot_name: "relation",
        options: MNLI_OPTIONS.map(str::to_owned).to_vec(),
        gold: Gold::Option(gold),
        label_basis: "hard",
    })
}

fn ids_of(
    views: &Views,
    dataset: &str,
    file: &str,
    mut take: impl FnMut(&Value),
) -> Result<String, String> {
    convert::read_view(views.target_rows(dataset, file)?, |_, r| {
        take(&r);
        Ok(())
    })
}

pub fn build_mnli(
    cfg: &Config,
    views: &Views,
    mut digests: BTreeMap<String, String>,
    threads: usize,
) -> Result<decisions::Built, String> {
    let val_files = [
        (
            "mnli-validation-matched",
            "data/validation_matched-00000-of-00001.parquet",
        ),
        (
            "mnli-validation-mismatched",
            "data/validation_mismatched-00000-of-00001.parquet",
        ),
    ];
    let mut val_prompts = BTreeSet::new();
    for (_, f) in val_files {
        let got = ids_of(views, MNLI, f, |r| {
            if let Some(p) = r.get("promptID").and_then(Value::as_i64) {
                val_prompts.insert(p);
            }
        })?;
        digests.insert(format!("target-rows/{MNLI}/{f}"), got);
    }
    let mut acc = Acc::new(cfg);
    let mut prompt_hits = 0usize;
    let got = convert::read_view(views.train_rows(MNLI, MNLI_FILE)?, |_, r| {
        let made = match r.get("promptID").and_then(Value::as_i64) {
            Some(p) if val_prompts.contains(&p) => {
                prompt_hits += 1;
                Err("prompt_in_validation".to_owned())
            }
            _ => mnli_row(&r),
        };
        acc.offer(MNLI, MNLI_FAMILY, made)
    })?;
    digests.insert(format!("{MNLI}/{MNLI_FILE}"), got);
    let refs: Vec<(&str, &str, &str)> = val_files.iter().map(|(n, f)| (*n, MNLI, *f)).collect();
    let (targets, td) = views.target_texts(&refs)?;
    digests.extend(td);
    let facts = PoolFacts {
        name: "mnli",
        licence_notes: BTreeMap::from([(
            MNLI,
            crate::convert_licence::pending_note("oanc")
                .expect("oanc is pending")
                .to_owned()
                + "; attribution: Williams, Nangia and Bowman, MultiNLI, NAACL 2018",
        )]),
        caps: json!({"per_row_caps": "none: every admitted row is a candidate"}),
        id_checks: json!({"validation_prompt_ids": {"state": "ran", "enforced": true,
            "validation_prompts": val_prompts.len(), "train_rows_refused": prompt_hits}}),
        notes: json!({"licence_filter": "per row: genre fiction refused (licence_fiction_genre)",
                      "labels": "0 entailment, 1 neutral, 2 contradiction (card class_label)"}),
    };
    convert::build(acc, targets, digests, facts, threads)
}

/// The SciRepEval read-time caps.
#[derive(Clone, Copy, Debug)]
pub struct SciCaps {
    pub query_fraction: f64,
    pub max_positives: usize,
    pub negatives_per_positive: usize,
}

/// What the read-time caps did, for the manifest.
#[derive(Default, Debug)]
pub struct SciCounts {
    pub queries_seen: usize,
    pub queries_not_drawn: usize,
    pub queries_in_eval: usize,
    pub queries_without_positive: usize,
    pub candidates_seen_in_drawn_queries: usize,
    pub positives_capped: usize,
    pub negatives_capped: usize,
}

/// The ids of the eval queries and their candidate papers.
#[derive(Default, Debug)]
pub struct EvalIds {
    pub queries: BTreeSet<String>,
    pub papers: BTreeSet<String>,
}

fn sci_row(qid: &str, query: &str, c: &Value) -> Result<Row, String> {
    let doc = convert::opt_str(c, "doc_id").ok_or("malformed")?;
    let title = convert::opt_str(c, "title")
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or("no_title")?;
    let abstract_ = convert::opt_str(c, "abstract")
        .map(str::trim)
        .filter(|a| !a.is_empty());
    let score = c.get("score").and_then(Value::as_u64).ok_or("malformed")?;
    Ok(Row {
        id: format!("scirepeval:{qid}:{doc}"),
        source_id: SCIREPEVAL,
        family_id: SCI_FAMILY,
        stratum: format!("{SCI_FAMILY}/pointwise"),
        group_key: qid.to_owned(),
        licence: Some("odc-by-1.0".to_owned()),
        context: format!(
            "{SCI_INSTRUCTION}\nQuery: {}\n\nTitle: {title}\nAbstract: {}",
            query.trim(),
            abstract_.unwrap_or("(none)")
        ),
        question: SCI_QUESTION.to_owned(),
        slot_name: "relevant",
        options: YES_NO.map(str::to_owned).to_vec(),
        gold: Gold::Option(if score > 0 { 0 } else { 1 }),
        label_basis: "graded_relevance_score",
    })
}

/// One SciRepEval query: drawn or not, then its capped candidates offered.
pub fn sci_query(
    acc: &mut Acc<'_>,
    caps: SciCaps,
    eval: &EvalIds,
    counts: &mut SciCounts,
    r: &Value,
) -> Result<(), String> {
    counts.queries_seen += 1;
    let seed = acc.cfg.seed;
    let (Some(qid), Some(query), Some(cands)) = (
        convert::opt_str(r, "doc_id"),
        convert::opt_str(r, "query"),
        r.get("candidates").and_then(Value::as_array),
    ) else {
        return acc.refuse(SCIREPEVAL, "malformed_query");
    };
    if decisions::unit_draw(seed, &["scirepeval-query", qid]) >= caps.query_fraction {
        counts.queries_not_drawn += 1;
        return Ok(());
    }
    if eval.queries.contains(qid) {
        counts.queries_in_eval += 1;
        for _ in cands {
            acc.refuse(SCIREPEVAL, "query_in_eval")?;
        }
        return Ok(());
    }
    counts.candidates_seen_in_drawn_queries += cands.len();
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for c in cands {
        let paper_ids = [
            convert::opt_str(c, "doc_id").map(str::to_owned),
            c.get("corpus_id")
                .and_then(Value::as_u64)
                .map(|n| n.to_string()),
        ];
        if paper_ids.iter().flatten().any(|p| eval.papers.contains(p)) {
            acc.refuse(SCIREPEVAL, "paper_in_eval")?;
            continue;
        }
        let doc = convert::opt_str(c, "doc_id").unwrap_or("");
        let key = decisions::keyed(seed, &["scirepeval-candidate", qid, doc]);
        match c.get("score").and_then(Value::as_u64) {
            Some(0) => neg.push((key, c)),
            Some(_) => pos.push((key, c)),
            None => acc.refuse(SCIREPEVAL, "malformed")?,
        }
    }
    if pos.is_empty() {
        counts.queries_without_positive += 1;
        return Ok(());
    }
    pos.sort_by_key(|(k, _)| *k);
    neg.sort_by_key(|(k, _)| *k);
    let kp = pos.len().min(caps.max_positives);
    let kn = neg.len().min(kp * caps.negatives_per_positive);
    counts.positives_capped += pos.len() - kp;
    counts.negatives_capped += neg.len() - kn;
    for (_, c) in pos.iter().take(kp).chain(neg.iter().take(kn)) {
        acc.offer(SCIREPEVAL, SCI_FAMILY, sci_row(qid, query, c))?;
    }
    Ok(())
}

pub fn build_scirepeval(
    cfg: &Config,
    views: &Views,
    mut digests: BTreeMap<String, String>,
    threads: usize,
) -> Result<decisions::Built, String> {
    let p = cfg.pool("scirepeval")?;
    let caps = SciCaps {
        query_fraction: convert::fraction(p, "query_fraction", "pools.scirepeval")?,
        max_positives: convert::count(p, "max_positives_per_query", "pools.scirepeval", 10)?,
        negatives_per_positive: convert::count(p, "negatives_per_positive", "pools.scirepeval", 9)?,
    };
    let mut eval = EvalIds::default();
    let eval_file = "search/evaluation-00000-of-00001.parquet";
    let got = ids_of(views, SCIREPEVAL, eval_file, |r| {
        if let Some(q) = convert::opt_str(r, "doc_id") {
            eval.queries.insert(q.to_owned());
        }
        for c in r
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(d) = convert::opt_str(c, "doc_id") {
                eval.papers.insert(d.to_owned());
            }
            if let Some(n) = c.get("corpus_id").and_then(Value::as_u64) {
                eval.papers.insert(n.to_string());
            }
        }
    })?;
    digests.insert(format!("target-rows/{SCIREPEVAL}/{eval_file}"), got);
    let test_file = "search/test-00000-of-00001.parquet";
    let got = ids_of(views, SCIREPEVAL_TEST, test_file, |r| {
        if let Some(q) = convert::opt_str(r, "query_id") {
            eval.queries.insert(q.to_owned());
        }
        if let Some(c) = convert::opt_str(r, "cand_id") {
            eval.papers.insert(c.to_owned());
        }
    })?;
    digests.insert(format!("target-rows/{SCIREPEVAL_TEST}/{test_file}"), got);

    let mut acc = Acc::new(cfg);
    let mut counts = SciCounts::default();
    for i in 0..SCI_SHARDS {
        let file = format!("search/train-{i:05}-of-{SCI_SHARDS:05}.parquet");
        let got = convert::read_view(views.train_rows(SCIREPEVAL, &file)?, |_, r| {
            sci_query(&mut acc, caps, &eval, &mut counts, &r)
        })?;
        digests.insert(format!("{SCIREPEVAL}/{file}"), got);
    }
    let refs = [
        ("scirepeval-search-evaluation", SCIREPEVAL, eval_file),
        (
            "beir-scidocs-corpus",
            "BeIR/scidocs",
            "corpus/corpus-00000-of-00001.parquet",
        ),
        (
            "beir-scidocs-queries",
            "BeIR/scidocs",
            "queries/queries-00000-of-00001.parquet",
        ),
    ];
    let (targets, td) = views.target_texts(&refs)?;
    digests.extend(td);
    let facts = PoolFacts {
        name: "scirepeval",
        licence_notes: BTreeMap::from([(
            SCIREPEVAL,
            crate::convert_licence::pending_note("odc-by-1.0")
                .expect("odc-by-1.0 is pending")
                .to_owned(),
        )]),
        caps: json!({
            "query_fraction": caps.query_fraction, "max_positives_per_query": caps.max_positives,
            "negatives_per_positive": caps.negatives_per_positive,
            "queries_seen": counts.queries_seen, "queries_not_drawn": counts.queries_not_drawn,
            "queries_without_positive": counts.queries_without_positive,
            "candidates_seen_in_drawn_queries": counts.candidates_seen_in_drawn_queries,
            "positives_capped": counts.positives_capped, "negatives_capped": counts.negatives_capped,
            "coverage": "a seeded sample of queries, capped per query: not the whole search config",
        }),
        id_checks: json!({
            "search_evaluation_and_scirepeval_test_search": {"state": "ran", "enforced": true,
                "eval_queries": eval.queries.len(), "eval_papers": eval.papers.len(),
                "queries_refused": counts.queries_in_eval,
                "rule": "a train query id among the eval/test query ids, or a train candidate's \
                         doc_id/corpus_id among their candidate ids, is refused"},
            "scirepeval_test_nfcorpus_trec_covid": {"state": "not_run",
                "reason": "ids from other corpora (MED-*, CORD uids) and no text: no id or text \
                           check can match them against S2 papers"},
        }),
        notes: json!({
            "negatives": "v1: the same query's score-0 candidates; BM25 cross-query hard negatives \
                          are a follow-up (std-only Rust), not built",
            "label": "yes when the candidate's graded score is > 0",
        }),
    };
    convert::build(acc, targets, digests, facts, threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mnli_fiction_is_refused_on_licence_and_labels_map_by_name() {
        let r = |genre: &str, label: i64| {
            json!({"promptID": 7, "pairID": "7e", "premise": "A man sleeps.",
            "hypothesis": "A person rests.", "genre": genre, "label": label, "premise_parse": "(x)"})
        };
        assert_eq!(
            mnli_row(&r("fiction", 0)).unwrap_err(),
            "licence_fiction_genre"
        );
        assert_eq!(mnli_row(&r("poetry", 0)).unwrap_err(), "unknown_genre");
        assert_eq!(mnli_row(&r("travel", -1)).unwrap_err(), "unknown_label");
        let ok = mnli_row(&r("travel", 2)).unwrap();
        assert_eq!(
            ok.options[match ok.gold {
                Gold::Option(i) => i,
                Gold::Noul => panic!(),
            }],
            "contradiction"
        );
        assert_eq!(ok.group_key, "7");
        assert_eq!(ok.licence.as_deref(), Some("oanc"));
        assert_eq!(ok.stratum, "mnli.nli/travel");
    }

    fn query(qid: &str, scores: &[u64]) -> Value {
        let cands: Vec<Value> = scores.iter().enumerate().map(|(i, s)| json!({
            "doc_id": format!("{qid}-{i}"), "title": format!("Paper {i} on topic {qid}"),
            "abstract": "We study things.", "corpus_id": 1000 + i as u64, "venue": "V", "year": 2020.0,
            "author_names": ["A"], "n_citations": 1, "n_key_citations": 0, "score": s})).collect();
        json!({"query": format!("query {qid}"), "doc_id": qid, "candidates": cands})
    }

    #[test]
    fn scirepeval_caps_per_query_at_read_and_refuses_eval_ids() {
        let cfg = crate::convert::tests::cfg();
        let caps = SciCaps {
            query_fraction: 1.0,
            max_positives: 1,
            negatives_per_positive: 2,
        };
        let mut eval = EvalIds::default();
        eval.queries.insert("qe".into());
        eval.papers.insert("1003".into());
        let mut acc = Acc::new(&cfg);
        let mut counts = SciCounts::default();
        sci_query(
            &mut acc,
            caps,
            &eval,
            &mut counts,
            &query("q1", &[1, 0, 0, 2, 0, 0, 0, 0, 0, 0]),
        )
        .unwrap();
        sci_query(&mut acc, caps, &eval, &mut counts, &query("qe", &[1, 0])).unwrap();
        sci_query(&mut acc, caps, &eval, &mut counts, &query("q0", &[0, 0, 0])).unwrap();
        // q1: candidate 3 (corpus_id 1003) is an eval paper; 1 positive kept of 1, 2 negatives of 8.
        let t = &acc.tallies[SCIREPEVAL];
        assert_eq!(t.refused["paper_in_eval"], 1);
        assert_eq!(t.refused["query_in_eval"], 2);
        assert_eq!(acc.rows.len(), 3);
        assert_eq!(
            acc.rows
                .iter()
                .filter(|c| crate::convert::gold_class(c) == "yes")
                .count(),
            1
        );
        assert_eq!(counts.negatives_capped, 6);
        assert_eq!(counts.queries_without_positive, 1);
        assert!(
            acc.rows
                .iter()
                .all(|c| c.group_key == "q1" && c.licence == "odc-by-1.0")
        );
        assert!(acc.rows[0].context.starts_with(SCI_INSTRUCTION));
    }

    #[test]
    fn scirepeval_query_draw_happens_before_any_candidate_is_kept() {
        let cfg = crate::convert::tests::cfg();
        let caps = SciCaps {
            query_fraction: 0.3,
            max_positives: 2,
            negatives_per_positive: 1,
        };
        let mut acc = Acc::new(&cfg);
        let mut counts = SciCounts::default();
        for i in 0..200 {
            sci_query(
                &mut acc,
                caps,
                &EvalIds::default(),
                &mut counts,
                &query(&format!("q{i}"), &[1, 0]),
            )
            .unwrap();
        }
        assert_eq!(counts.queries_seen, 200);
        let drawn = 200 - counts.queries_not_drawn;
        assert!((30..=90).contains(&drawn), "{drawn}");
        assert_eq!(acc.rows.len(), drawn * 2);
    }
}
