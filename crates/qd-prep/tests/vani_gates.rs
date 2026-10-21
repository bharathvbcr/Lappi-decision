//! The v2 gates before any real Vani row pool (v2 plan section 6), through the `qd-prep` binary:
//!
//! * the **agreement filter**: a row is admitted only where two teachers read one posting and
//!   agree; one teacher, one reading or a disagreeing pair is refused;
//! * the **content door**: a reposted near-duplicate of a held-out posting is refused and an
//!   unrelated posting is admitted; a missing, corrupt or wrong-version signature file refuses
//!   the whole pool;
//! * the **redactor**: a planted email, phone number or signature-block name does not reach a row,
//!   and a row that still carries one is refused.
//!
//! Every case drives the binary, so each one fails on the code before these gates existed (there
//! was no `vani-pool` subcommand). The thresholds the door applies are checked against the Python
//! files that define them, because this crate cannot import Python.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-prep");

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn scratch(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!(
        "qd-vg-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&p).expect("scratch");
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().expect("qd-prep runs")
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

/// A held-out posting: sixteen distinct sentences, about 170 words.
const HELD_SENTENCES: [&str; 16] = [
    "Northwind Freight is hiring a staff platform engineer to own our dispatch routing service.",
    "You will design the queueing layer that assigns drivers to loads across three regions.",
    "The team ships weekly and keeps an on call rotation shared by six engineers.",
    "We value written design reviews and small reversible changes over large rewrites.",
    "Experience with Rust or Go and a deep understanding of distributed consensus is expected.",
    "You have operated a production database through a painful migration and can tell the story.",
    "This role is hybrid with two days a week in our Rotterdam office near the harbour.",
    "We sponsor work visas for the right candidate after a successful probation period.",
    "Compensation includes equity, a learning budget and generous parental leave for everyone.",
    "Our interview process has four stages and finishes within two weeks of the first call.",
    "You will mentor two junior engineers and help shape the hiring bar for the whole group.",
    "The tech stack covers Kafka, Postgres, Kubernetes and a lot of well tested Rust services.",
    "We care about accessibility and run regular reviews of our driver facing mobile screens.",
    "Success in the first year means a stable routing core and a clear path to multi region failover.",
    "Applications are reviewed by engineers rather than by a keyword filter or an automated score.",
    "Tell us about the hardest outage you ever debugged and what you changed afterwards.",
];

/// A posting about something else entirely.
const OTHER_SENTENCES: [&str; 14] = [
    "Harbourlight Clinics seeks a pediatric nurse practitioner for its weekend outreach programme.",
    "Candidates hold an active state licence and two years of paediatric community experience.",
    "The schedule rotates between the mobile van and the permanent clinic on Saturdays and Sundays.",
    "Duties include immunisation clinics, growth assessments and coaching for first time parents.",
    "We reimburse continuing education and provide malpractice cover from the first day onward.",
    "The position reports to the director of nursing and works closely with two social workers.",
    "Bilingual candidates who speak Portuguese or Vietnamese are strongly encouraged to apply here.",
    "Shifts are ten hours and include one paid hour for charting and a protected lunch break.",
    "On site parking and a transit subsidy are available for staff travelling from outside town.",
    "Our patients are families who often have no other regular access to preventive medicine.",
    "You will keep accurate records in our electronic chart and join the monthly case conference.",
    "Relocation help is not offered but we will gladly share contacts for local rental agents.",
    "The clinic closes for two weeks in the winter and everyone takes the same holiday together.",
    "Please describe a time you earned a worried parent's trust during a stressful appointment.",
];

fn posting(sentences: &[&str]) -> String {
    sentences.join(" ")
}

fn held_text() -> String {
    posting(&HELD_SENTENCES)
}

/// The held-out posting reposted: every word kept, one sentence moved, case and spacing changed.
fn repost_text() -> String {
    let mut sentences = HELD_SENTENCES.to_vec();
    let moved = sentences.remove(4);
    sentences.insert(11, moved);
    let joined = posting(&sentences);
    format!("  {}\n\n{}  ", joined.to_uppercase(), "").replace(". ", ".  \n")
}

fn other_text() -> String {
    posting(&OTHER_SENTENCES)
}

/// A signature file under a held-out path, built by the binary from one held-out posting.
fn build_door(dir: &Path) -> PathBuf {
    let postings = dir.join("held-postings.jsonl");
    std::fs::write(
        &postings,
        json!({"key": "held-1", "text": held_text()}).to_string() + "\n",
    )
    .unwrap();
    let out = dir.join("heldout").join("vani-postings").join("signatures.json");
    let o = run(&["posting-door-build", "--postings", s(&postings), "--out", s(&out)]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    out
}

fn pair(key: &str, text: &str, readings: &[(&str, &str)]) -> Value {
    json!({
        "posting_key": key, "slot": "seniority", "text": text,
        "readings": readings.iter().map(|(t, l)| json!({"teacher": t, "label": l})).collect::<Vec<_>>(),
    })
}

fn agreed(key: &str, text: &str) -> Value {
    pair(key, text, &[("d1-ultra", "staff_plus"), ("haiku", "staff_plus")])
}

struct Pool {
    code: Option<i32>,
    stderr: String,
    out_dir: PathBuf,
}

impl Pool {
    fn rows(&self) -> Vec<Value> {
        self.lines("rows.jsonl")
    }
    fn refusals(&self) -> Vec<Value> {
        self.lines("refusals.jsonl")
    }
    fn lines(&self, name: &str) -> Vec<Value> {
        std::fs::read_to_string(self.out_dir.join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}; stderr: {}", self.stderr))
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
    fn kinds(&self) -> Vec<String> {
        self.refusals()
            .iter()
            .map(|r| r["kind"].as_str().unwrap().to_owned())
            .collect()
    }
}

fn pool(dir: &Path, signatures: &Path, pairs: &[Value]) -> Pool {
    let pairs_path = dir.join("pairs.jsonl");
    std::fs::write(
        &pairs_path,
        pairs.iter().map(Value::to_string).collect::<Vec<_>>().join("\n") + "\n",
    )
    .unwrap();
    let out_dir = dir.join("out");
    let o = run(&[
        "vani-pool", "--pairs", s(&pairs_path), "--signatures", s(signatures), "--out-dir", s(&out_dir),
    ]);
    Pool {
        code: o.status.code(),
        stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
        out_dir,
    }
}

// ---------------------------------------------------------------------------------------------
// 1. The agreement filter

#[test]
fn a_pair_is_admitted_only_where_two_teachers_agree() {
    let d = scratch("agree");
    let door = build_door(&d);
    let other = other_text();
    let p = pool(
        &d,
        &door,
        &[
            agreed("ok-1", &other),
            pair("dis-1", &other.replace("Harbourlight", "Harbourlight Two"), &[("d1-ultra", "staff_plus"), ("haiku", "senior")]),
            pair("same-1", &other.replace("Harbourlight", "Harbourlight Three"), &[("haiku", "senior"), ("Haiku ", "senior")]),
            pair("one-1", &other.replace("Harbourlight", "Harbourlight Four"), &[("d1-ultra", "senior")]),
            pair("three-1", &other.replace("Harbourlight", "Harbourlight Five"), &[("a", "x"), ("b", "x"), ("c", "x")]),
            pair("blank-1", &other.replace("Harbourlight", "Harbourlight Six"), &[("d1-ultra", " "), ("haiku", " ")]),
        ],
    );
    assert_eq!(p.code, Some(0), "{}", p.stderr);
    let rows = p.rows();
    assert_eq!(rows.len(), 1, "{rows:?}");
    let row = &rows[0];
    assert_eq!(row["source"], "vani-agreed");
    assert_eq!(row["source_id"], "vani-agreed");
    assert_eq!(row["teachers"], json!(["d1-ultra", "haiku"]));
    assert_eq!(row["readings"], json!(["staff_plus", "staff_plus"]));
    assert_eq!(row["label"], "staff_plus");
    assert_eq!(
        p.kinds(),
        vec!["disagree", "same_teacher", "reading_count", "reading_count", "empty_field"]
    );
}

#[test]
fn a_posting_read_twice_is_one_row_not_two() {
    let d = scratch("dup");
    let door = build_door(&d);
    let p = pool(&d, &door, &[agreed("ok-1", &other_text()), agreed("ok-1", &other_text())]);
    assert_eq!(p.rows().len(), 1);
    assert_eq!(p.kinds(), vec!["duplicate"]);
}

// ---------------------------------------------------------------------------------------------
// 2. The content door

#[test]
fn a_reposted_near_duplicate_of_a_held_out_posting_is_refused_and_an_unrelated_one_admitted() {
    let d = scratch("door");
    let door = build_door(&d);
    let p = pool(
        &d,
        &door,
        &[
            agreed("repost-1", &repost_text()),
            agreed("exact-1", &held_text()),
            agreed("unrelated-1", &other_text()),
        ],
    );
    assert_eq!(p.code, Some(0), "{}", p.stderr);
    let ids: Vec<String> = p.rows().iter().map(|r| r["posting_key"].as_str().unwrap().to_owned()).collect();
    assert_eq!(ids, vec!["unrelated-1"], "refusals: {:?}", p.refusals());
    assert_eq!(p.kinds(), vec!["near_duplicate_of_held_out", "near_duplicate_of_held_out"]);
}

#[test]
fn a_row_with_a_held_out_key_is_refused_whatever_its_text() {
    let d = scratch("key");
    let door = build_door(&d);
    let p = pool(&d, &door, &[agreed("held-1", &other_text())]);
    assert_eq!(p.rows().len(), 0);
    assert_eq!(p.kinds(), vec!["held_out_key"]);
}

#[test]
fn a_posting_too_short_to_compare_is_refused_not_admitted() {
    let d = scratch("short");
    let door = build_door(&d);
    let p = pool(&d, &door, &[agreed("short-1", "senior platform role in Rotterdam")]);
    assert_eq!(p.rows().len(), 0);
    assert_eq!(p.kinds(), vec!["unscannable"]);
}

#[test]
fn a_missing_corrupt_or_wrong_version_signature_file_refuses_the_whole_pool() {
    let d = scratch("closed");
    let good = build_door(&d);
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&good).unwrap()).unwrap();

    let heldout = d.join("heldout").join("bad");
    std::fs::create_dir_all(&heldout).unwrap();
    let write = |name: &str, v: &Value| {
        let p = heldout.join(name);
        std::fs::write(&p, v.to_string()).unwrap();
        p
    };
    let mut flipped = doc.clone();
    let sig = flipped["entries"][0]["signature"][5].as_str().unwrap().to_owned();
    flipped["entries"][0]["signature"][5] = json!(format!("{:016x}", u64::from_str_radix(&sig, 16).unwrap() ^ 1));
    let mut wrong_format = doc.clone();
    wrong_format["format"] = json!("qd-posting-heldout.v0");
    let mut wrong_n = doc.clone();
    wrong_n["ngram_n"] = json!(5);
    let mut wrong_perm = doc.clone();
    wrong_perm["num_perm"] = json!(64);
    let mut empty = doc.clone();
    empty["entries"] = json!([]);
    let cases: Vec<(&str, PathBuf)> = vec![
        ("missing", heldout.join("absent.json")),
        ("flipped digit", write("flipped.json", &flipped)),
        ("wrong format", write("format.json", &wrong_format)),
        ("wrong ngram_n", write("n.json", &wrong_n)),
        ("wrong num_perm", write("perm.json", &wrong_perm)),
        ("no entries", write("empty.json", &empty)),
    ];
    std::fs::write(heldout.join("garbage.json"), "not json at all").unwrap();
    let mut cases = cases;
    cases.push(("not json", heldout.join("garbage.json")));
    for (name, path) in cases {
        let work = scratch("closed-run");
        let p = pool(&work, &path, &[agreed("ok-1", &other_text())]);
        assert!(p.code != Some(0), "{name}: the pool must be refused, exit {:?}", p.code);
        assert_ne!(p.code, Some(101), "{name}: a refusal is an error, not a panic: {}", p.stderr);
        assert!(!p.out_dir.exists(), "{name}: nothing may be written when the door cannot open");
    }
}

#[test]
fn a_signature_file_outside_a_held_out_path_is_not_a_door() {
    let d = scratch("path");
    let good = build_door(&d);
    let copy = d.join("elsewhere.json");
    std::fs::copy(&good, &copy).unwrap();
    let p = pool(&d, &copy, &[agreed("ok-1", &other_text())]);
    assert!(p.code != Some(0), "{}", p.stderr);
    assert!(p.stderr.contains("held-out"), "{}", p.stderr);
    assert!(!p.out_dir.exists());

    let postings = d.join("postings.jsonl");
    std::fs::write(&postings, json!({"key": "k", "text": held_text()}).to_string() + "\n").unwrap();
    let o = run(&["posting-door-build", "--postings", s(&postings), "--out", s(&d.join("sigs.json"))]);
    assert!(!o.status.success(), "build must refuse an output with no held-out marker");
}

#[test]
fn the_door_applies_the_thresholds_v5_defines() {
    let read = |rel: &str| std::fs::read_to_string(repo().join(rel)).unwrap();
    let replay = read("python/qd_train/replay.py");
    let config = read("python/qd_data/config.py");
    let has = |text: &str, needle: &str| text.contains(needle);
    assert!(has(&replay, &format!("DEFAULT_N: Final[int] = {}", qd_prep::posting_door::NGRAM_N)));
    assert!(has(
        &replay,
        &format!("DEFAULT_THRESHOLD: Final[float] = {}", qd_prep::posting_door::CONTAINMENT_THRESHOLD)
    ));
    assert!(has(
        &config,
        &format!("dedupe_threshold: float = {}", qd_prep::posting_door::JACCARD_THRESHOLD)
    ));
    assert!(has(&config, &format!("num_perm: int = {}", qd_prep::posting_door::NUM_PERM)));
    assert!(has(&config, "seed: int = 20260919"));
    // The default signature path is under the held-out root the Python config names.
    assert!(has(&config, "held_out_roots: tuple[Path, ...] = (Path(\"data/heldout\"),)"));
    assert!(qd_prep::posting_door::DEFAULT_SIGNATURES_PATH.starts_with("data/heldout/"));
    assert!(qd_prep::heldout::check_held_out_path(Path::new(
        qd_prep::posting_door::DEFAULT_SIGNATURES_PATH
    ))
    .is_ok());
}

// ---------------------------------------------------------------------------------------------
// 3. The redactor

fn with_signature(extra: &str) -> String {
    format!("{} {extra}", other_text())
}

#[test]
fn a_planted_email_phone_and_name_do_not_reach_a_row() {
    let d = scratch("redact");
    let door = build_door(&d);
    let text = with_signature(
        "Send a CV to talent.lead@harbourlight-clinics.org or call +44 20 7946 0958.\n\nBest regards,\nMarguerite Okonkwo-Reyes\nHead of People",
    );
    let p = pool(&d, &door, &[agreed("redact-1", &text)]);
    assert_eq!(p.code, Some(0), "{}", p.stderr);
    let rows = p.rows();
    assert_eq!(rows.len(), 1, "refusals: {:?}", p.refusals());
    let context = rows[0]["context"].as_str().unwrap();
    for planted in ["talent.lead", "harbourlight-clinics.org", "7946", "0958", "Marguerite", "Okonkwo"] {
        assert!(!context.contains(planted), "{planted:?} reached the row: {context}");
    }
    assert!(!context.contains('@'));
    for placeholder in ["[EMAIL]", "[PHONE]", "[NAME]"] {
        assert!(context.contains(placeholder), "{placeholder} missing: {context}");
    }
    assert_eq!(rows[0]["redaction"], json!({"emails": 1, "phones": 1, "names": 1}));
    // The whole row, not just the text: nothing the redactor removed is anywhere in it.
    let whole = rows[0].to_string();
    assert!(!whole.contains("talent.lead") && !whole.contains("Marguerite"), "{whole}");
}

#[test]
fn obfuscated_unicode_and_international_forms_are_removed() {
    let d = scratch("obf");
    let door = build_door(&d);
    let text = with_signature(
        "Write to recruiting [at] harbourlight [dot] org, or to j\u{f6}rg\u{ff20}klinik.de, or ring 07911 123456 or +91 98765 43210. Thanks,\n\nJos\u{e9} O'Neil",
    );
    let p = pool(&d, &door, &[agreed("obf-1", &text)]);
    let rows = p.rows();
    assert_eq!(rows.len(), 1, "refusals: {:?}", p.refusals());
    let context = rows[0]["context"].as_str().unwrap();
    for planted in ["recruiting", "klinik", "07911", "98765", "O'Neil", "[at]"] {
        assert!(!context.contains(planted), "{planted:?} reached the row: {context}");
    }
}

#[test]
fn a_row_that_still_carries_one_after_redaction_is_refused() {
    let d = scratch("residual");
    let door = build_door(&d);
    // An at sign no email detector can parse still counts as an email: the row is refused.
    let p = pool(
        &d,
        &door,
        &[
            agreed("stray-1", &with_signature("Reach me at jane@@harbourlight")),
            agreed("local-1", &with_signature("Mail ops@localhost for access")),
        ],
    );
    assert_eq!(p.rows().len(), 0, "{:?}", p.rows());
    assert_eq!(p.kinds(), vec!["residual_pii", "residual_pii"]);
}

#[test]
fn a_secret_in_a_row_is_refused_not_redacted() {
    let d = scratch("secret");
    let door = build_door(&d);
    let p = pool(
        &d,
        &door,
        &[agreed("secret-1", &with_signature("Use the token ghp_abcdefghijklmnopqrstuvwxyz0123 to clone"))],
    );
    assert_eq!(p.rows().len(), 0);
    assert_eq!(p.kinds(), vec!["secret"]);
}

#[test]
fn a_row_redaction_would_empty_is_refused() {
    let d = scratch("emptied");
    let door = build_door(&d);
    let text = "a@b.co c@d.org e@f.net g@h.io i@j.dev k@l.app m@n.com o@p.net q@r.org s@t.io";
    let p = pool(&d, &door, &[agreed("emptied-1", text)]);
    assert_eq!(p.rows().len(), 0);
    assert_eq!(p.kinds(), vec!["emptied_by_redaction"]);
}

#[test]
fn a_posting_with_no_personal_data_passes_through_unchanged() {
    let d = scratch("clean");
    let door = build_door(&d);
    let text = other_text();
    let p = pool(&d, &door, &[agreed("clean-1", &text)]);
    let rows = p.rows();
    assert_eq!(rows.len(), 1, "refusals: {:?}", p.refusals());
    assert_eq!(rows[0]["context"], text);
    assert_eq!(rows[0]["redaction"], json!({"emails": 0, "phones": 0, "names": 0}));
}

#[test]
fn the_pairs_input_may_not_be_held_out_data() {
    let d = scratch("pairs-path");
    let door = build_door(&d);
    let held = d.join("heldout");
    std::fs::create_dir_all(&held).unwrap();
    let pairs = held.join("pairs.jsonl");
    std::fs::write(&pairs, agreed("ok-1", &other_text()).to_string() + "\n").unwrap();
    let out_dir = d.join("out");
    let o = run(&["vani-pool", "--pairs", s(&pairs), "--signatures", s(&door), "--out-dir", s(&out_dir)]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("held-out"));
    assert!(!out_dir.exists());
}

// ---------------------------------------------------------------------------------------------
// Door cost (not a gate): wall time of the content door over a Vani-sized held-out set.

#[test]
#[ignore = "bench: run with --ignored --nocapture"]
fn bench_door_over_2300_held_out_postings_and_5000_rows() {
    use qd_prep::posting_door::PostingDoor;
    let seeded = |i: u64, salt: u64| -> String {
        (0..150u64)
            .map(|w| {
                let mut x = (salt ^ (i << 8) ^ w).wrapping_add(0x9E37_79B9_7F4A_7C15);
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                format!("w{}", (x ^ (x >> 27)) % 20_000)
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let held: Vec<(String, String)> = (0..2300).map(|i| (format!("h{i}"), seeded(i, 1))).collect();
    let t = std::time::Instant::now();
    let door = PostingDoor::build(20_260_919, &held).expect("builds");
    println!("door build: {} postings in {:.2} s", door.len(), t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let mut refused = 0usize;
    for i in 0..5000u64 {
        // Every 100th row is a held-out posting's text, so the door has hits to find.
        let text = if i % 100 == 0 { held[(i / 100) as usize].1.clone() } else { seeded(i, 2) };
        if door.check(&format!("r{i}"), &text).is_err() {
            refused += 1;
        }
    }
    let secs = t.elapsed().as_secs_f64();
    println!("door check: 5000 rows in {secs:.2} s ({:.1} rows/s), {refused} refused", 5000.0 / secs);
    assert_eq!(refused, 50);
}
