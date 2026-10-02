//! Rung (b)'s data path on the CPU: L-oracle's 20 batches, rebuilt as the shard reader's
//! `Batch` and passed through `RealBatch` (`ft_supervision` + `plan_span_batch`, the glue a v4
//! run uses), give exactly the letter targets and span-head row layouts the torch reference
//! trained on (`plan.json`, written by Python's own `plan_span_batch`).

mod tiny_published;

use qd_train::ft_data::RealBatch;
use qd_train::objective::FtBatch;
use qd_train::trainer::ConsumedBatch;
use tiny_published::{json, Sequences};

#[test]
fn the_rung_b_batches_supervise_what_the_torch_reference_supervised() {
    let seqs = Sequences::load();
    let plan = json("batches/plan.json");
    assert_eq!(plan["grad_accum"], 1);
    let batches = plan["batches"].as_array().unwrap();
    assert_eq!(batches.len(), 20);
    let (mut letters, mut spans, mut abstaining) = (0usize, 0usize, 0usize);
    for (i, b) in batches.iter().enumerate() {
        let rb = RealBatch::new(seqs.batch(b)).unwrap_or_else(|e| panic!("batch {i}: {e}"));
        assert_eq!(rb.index(), i as u64);
        let rows: Vec<usize> = b["rows"].as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect();
        assert_eq!(rb.n_rows(), rows.len());
        let mut n_letter = 0;
        for (r, &s) in rows.iter().enumerate() {
            assert_eq!(rb.row_tokens(r), seqs.sequence(s), "batch {i} row {r}: tokens");
            if let Some(l) = rb.letter(r) {
                let ti = seqs.target_index[s];
                assert_eq!((l.position, l.target), (ti as u32, seqs.sequence(s)[ti as usize + 1]), "batch {i} row {r}");
                n_letter += 1;
            }
        }
        assert_eq!(n_letter, b["n_supervised"].as_u64().unwrap() as usize, "batch {i}: letter rows");
        let want = b["span_plan"].as_array().cloned().unwrap_or_default();
        let got: Vec<usize> = (0..rb.n_rows()).filter(|&r| rb.span(r).is_some()).collect();
        assert_eq!(got.len(), want.len(), "batch {i}: span rows");
        assert_eq!(want.len(), b["n_spans"].as_u64().unwrap() as usize);
        for w in &want {
            let r = w["batch_row"].as_u64().unwrap() as usize;
            let s = rb.span(r).unwrap_or_else(|| panic!("batch {i} row {r} is a span row in plan.json"));
            let cands: Vec<u32> = w["candidate_pos"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap() as u32).collect();
            assert_eq!(s.query_index(), w["query_index"].as_u64().unwrap() as u32, "batch {i} row {r}: query");
            assert_eq!(s.candidates(), cands.as_slice(), "batch {i} row {r}: candidates");
            assert_eq!(s.gold_start(), w["gold_start_row"].as_u64().unwrap() as usize, "batch {i} row {r}: gold start");
            assert_eq!(s.gold_end(), w["gold_end_row"].as_u64().unwrap() as usize, "batch {i} row {r}: gold end");
            assert_eq!(s.is_abstaining(), w["abstaining"].as_bool().unwrap(), "batch {i} row {r}: abstaining");
            assert_eq!(s.runtime_rows(), w["runtime_rows"].as_u64().unwrap() as usize, "batch {i} row {r}: rows");
            abstaining += usize::from(s.is_abstaining());
        }
        letters += n_letter;
        spans += want.len();
    }
    // The manifest's coverage, recomputed from what the Rust glue decided.
    let cov = &json("manifest.json")["coverage"];
    assert_eq!(letters as u64, cov["letter_rows"].as_u64().unwrap());
    assert_eq!((spans - abstaining) as u64, cov["pointing_span_rows"].as_u64().unwrap());
    assert_eq!(abstaining as u64, cov["abstaining_span_rows"].as_u64().unwrap());
}
