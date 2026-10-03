//! The N-tower ensemble: member releases exported by the real exporter, an ensemble directory
//! written by `export_ensemble`, opened by the runtime's own reader (`Ensemble::open`) and served
//! through `Runtime::from_ensemble`. Every way a member can be missing, swapped or disagree with
//! the others refuses the whole load; nothing serves N-1 towers.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use qd_export::{
    EnsembleRequest, EnsembleSummary, ExportSummary, RefusalKind, export, export_ensemble,
};
use qd_runtime::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, SlotQuery, StateSnapshot,
};
use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::ensemble::ensemble_weight_hash;
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::{
    CALIBRATION_FILE, CONFIG_FILE, ENSEMBLE_DECODE, ENSEMBLE_MANIFEST_FILE, Ensemble,
    MANIFEST_FILE, ReleaseRefusalKind, Tower,
};
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{Response, SlotKind};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

const WIDTH: u64 = 1105;
const SOURCE: &str = "test: the width the members were trained at";
/// The prompt format a v5 checkpoint's source manifest states, and the one this runtime serves.
/// Written as the wire number, not `render::PROMPT_FORMAT`, so the file pins the contract.
const V5_PROMPT_FORMAT: u64 = 2;

/// The ensemble's table, fitted on the mean decode; not the reference table.
fn ensemble_table() -> CalibrationTable {
    CalibrationTable::new("ensemble-under-test")
        .with_letters(
            SlotKind::Choice,
            5,
            CalibrationEntry {
                temperature: 1.21,
                conformal_quantile: 0.09,
                noul_margin: 0.02,
            },
        )
        .with_span(CalibrationEntry {
            temperature: 1.0,
            conformal_quantile: 0.0,
            noul_margin: 0.0,
        })
}

/// One seed's release: the standard fixture with one tower tensor regenerated from `seed`, so
/// every member is a different tower under one config and tokenizer. Its source manifest states
/// prompt format 2, and the export is checked to have stamped exactly that.
fn member_release(
    seed: u64,
    config: &Value,
    tokenizer_suffix: &str,
) -> (common::Fixture, ExportSummary) {
    member_release_of_format(seed, config, tokenizer_suffix, Some(V5_PROMPT_FORMAT))
}

/// [`member_release`] from a source manifest stating `prompt_format` (none when `None`, as
/// every checkpoint averaged before the format existed). The release is stamped with the
/// source's format, absent = 1.
fn member_release_of_format(
    seed: u64,
    config: &Value,
    tokenizer_suffix: &str,
    prompt_format: Option<u64>,
) -> (common::Fixture, ExportSummary) {
    let mut tensors = common::standard_tensors();
    let (name, shape) = tensors
        .iter()
        .find(|(name, _)| name.starts_with("tower.") && name.contains("layernorm"))
        .map(|(name, t)| (name.clone(), t.shape.clone()))
        .expect("the standard fixture has a layernorm");
    tensors.insert(name, common::bf16_tensor(&shape, 7000 + seed));
    let fx = common::build(&tensors, config, |m| {
        if let Some(format) = prompt_format {
            m["prompt_format"] = json!(format);
        }
    });
    if !tokenizer_suffix.is_empty() {
        let tokenizer = common::tiny_tokenizer_json() + tokenizer_suffix;
        std::fs::write(fx.snapshot.join("tokenizer.json"), tokenizer).unwrap();
    }
    let summary = export(&fx.request()).expect("a member release exports");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(fx.out.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(manifest["format"], json!("qd-release.v2"));
    assert_eq!(
        manifest["expected_identity"]["prompt_format"],
        json!(prompt_format.unwrap_or(1)),
        "the member is stamped with its source's prompt format, absent = 1"
    );
    (fx, summary)
}

fn seed(n: u64) -> (common::Fixture, ExportSummary) {
    member_release(n, &common::tiny_config(), "")
}

struct Written {
    // Held so the member releases outlive the ensemble written from them.
    _members: Vec<common::Fixture>,
    summaries: Vec<ExportSummary>,
    dir: common::TempDir,
    table_path: PathBuf,
}

impl Written {
    fn out(&self) -> PathBuf {
        self.dir.0.join("ensemble")
    }
}

fn members(n: u64) -> (Vec<common::Fixture>, Vec<ExportSummary>) {
    (0..n).map(seed).unzip()
}

fn table_file(dir: &Path) -> PathBuf {
    let path = dir.join("ensemble_table.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&ensemble_table()).unwrap()).unwrap();
    path
}

fn request(paths: Vec<PathBuf>, table: &Path, out: &Path) -> EnsembleRequest {
    EnsembleRequest {
        trained_widths: vec![WIDTH; paths.len()],
        members: paths,
        trained_width_source: SOURCE.to_string(),
        calibration: table.to_path_buf(),
        out: out.to_path_buf(),
    }
}

/// Three seeds, written as one ensemble.
fn written() -> (Written, EnsembleSummary) {
    let (fixtures, summaries) = members(3);
    let dir = common::TempDir::new("ensemble");
    let table_path = table_file(&dir.0);
    let written = Written {
        _members: fixtures,
        summaries,
        dir,
        table_path,
    };
    let paths = written._members.iter().map(|f| f.out.clone()).collect();
    let summary = export_ensemble(&request(paths, &written.table_path, &written.out()))
        .expect("three agreeing seeds write an ensemble");
    (written, summary)
}

fn rewrite_ensemble_manifest(out: &Path, edit: impl FnOnce(&mut Value)) {
    let path = out.join(ENSEMBLE_MANIFEST_FILE);
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut doc);
    std::fs::write(&path, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
}

#[track_caller]
fn open_refused(out: &Path, kind: ReleaseRefusalKind, needle: &str) {
    match Ensemble::open(out) {
        Err(err) => {
            assert_eq!(err.kind, kind, "{err}");
            assert!(err.detail.contains(needle), "{needle:?} not in {err}");
        }
        Ok(_) => panic!("an ensemble that must be refused opened"),
    }
}

// -- writer and reader agree ---------------------------------------------------------------------

#[test]
fn three_seeds_write_an_ensemble_the_runtime_reader_opens() {
    let (w, summary) = written();
    let ensemble = Ensemble::open(&w.out()).expect("the written ensemble opens");
    let member_hashes: Vec<&str> = w.summaries.iter().map(|s| s.weight_hash.as_str()).collect();
    assert_eq!(ensemble.towers().len(), 3);
    for (tower, member) in ensemble.towers().iter().zip(&w.summaries) {
        assert_eq!(tower.weight_hash(), member.weight_hash);
        assert_eq!(tower.tokenizer_hash(), member.tokenizer_hash);
        assert_eq!(tower.config_sha256(), member.files[CONFIG_FILE]);
    }
    assert_eq!(ensemble.weight_hash(), ensemble_weight_hash(&member_hashes));
    assert_eq!(ensemble.weight_hash(), summary.weight_hash);
    assert_eq!(summary.member_weight_hashes, member_hashes);
    assert_eq!(ensemble.trained_width(), WIDTH);
    assert_eq!(ensemble.calibration(), &ensemble_table());
    assert_eq!(summary.calibration_hash, ensemble_table().hash());

    let manifest: Value =
        serde_json::from_slice(&std::fs::read(w.out().join(ENSEMBLE_MANIFEST_FILE)).unwrap())
            .unwrap();
    assert_eq!(manifest["decode"], Value::from(ENSEMBLE_DECODE));
    assert_eq!(manifest["trained_width_source"], Value::from(SOURCE));
    assert_eq!(
        manifest["expected_identity"]["trained_width"],
        Value::from(WIDTH)
    );
    // Stamped from the members' own releases, which the writer opened.
    assert_eq!(
        manifest["expected_identity"]["prompt_format"],
        json!(V5_PROMPT_FORMAT)
    );
}

// -- the prompt format ---------------------------------------------------------------------------

/// Members exported from v4 checkpoints (source manifests that state no format, so stamped 1) do
/// not open under this runtime, so the writer refuses them rather than writing an ensemble the
/// runtime would then refuse -- or, worse, one that served format-2 prompts to v4 towers.
#[test]
fn members_exported_from_v4_checkpoints_are_refused_by_the_writer() {
    let (a, _) = member_release_of_format(0, &common::tiny_config(), "", None);
    let (b, _) = member_release_of_format(1, &common::tiny_config(), "", None);
    let dir = common::TempDir::new("ensemble-v4");
    let table = table_file(&dir.0);
    let out = dir.0.join("ensemble");
    export_refused(
        &request(vec![a.out.clone(), b.out.clone()], &table, &out),
        "prompt_format",
    );

    // One v4 member among format-2 ones is enough.
    let (c, _) = seed(2);
    export_refused(
        &request(vec![c.out.clone(), a.out.clone()], &table, &out),
        "member 1",
    );
}

#[test]
fn an_ensemble_manifest_without_the_members_prompt_format_is_refused() {
    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["expected_identity"]
            .as_object_mut()
            .unwrap()
            .remove("prompt_format");
    });
    let err = Ensemble::open(&w.out()).expect_err("an ensemble that states no prompt format");
    assert_eq!(err.kind.as_str(), "prompt_format", "{err}");
    assert!(err.detail.contains("expected_identity.prompt_format"), "{err}");

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| m["expected_identity"]["prompt_format"] = json!(1));
    let err = Ensemble::open(&w.out()).expect_err("an ensemble stating another prompt format");
    assert_eq!(err.kind.as_str(), "prompt_format", "{err}");
    assert!(err.detail.contains("prompt_format 1"), "{err}");
}

// -- the writer refuses what cannot be averaged honestly -----------------------------------------

#[track_caller]
fn export_refused(req: &EnsembleRequest, needle: &str) {
    match export_ensemble(req) {
        Err(err) => {
            assert_eq!(err.kind, RefusalKind::Ensemble, "{err}");
            assert!(err.detail.contains(needle), "{needle:?} not in {err}");
        }
        Ok(_) => panic!("an ensemble that must be refused was written"),
    }
    assert!(
        !req.out.exists(),
        "a refused ensemble left {} behind",
        req.out.display()
    );
}

#[test]
fn the_writer_refuses_a_count_width_or_source_it_cannot_stand_behind() {
    let (fixtures, _) = members(3);
    let dir = common::TempDir::new("ensemble-w");
    let table = table_file(&dir.0);
    let out = dir.0.join("ensemble");
    let paths: Vec<PathBuf> = fixtures.iter().map(|f| f.out.clone()).collect();

    export_refused(&request(paths[..1].to_vec(), &table, &out), "2 to");

    let mut req = request(paths.clone(), &table, &out);
    req.trained_widths.pop();
    export_refused(&req, "each member states its own");

    let mut req = request(paths.clone(), &table, &out);
    req.trained_widths[2] = 2048;
    export_refused(&req, "member 2 was trained at width 2048");

    let mut req = request(paths.clone(), &table, &out);
    req.trained_widths[0] = 0;
    export_refused(&req, "0 tokens");

    let mut req = request(paths.clone(), &table, &out);
    req.trained_width_source = "  ".to_string();
    export_refused(&req, "need a source");
}

#[test]
fn the_writer_refuses_a_missing_repeated_or_disagreeing_member() {
    let (fixtures, _) = members(2);
    let dir = common::TempDir::new("ensemble-m");
    let table = table_file(&dir.0);
    let out = dir.0.join("ensemble");
    let a = fixtures[0].out.clone();
    let b = fixtures[1].out.clone();

    export_refused(
        &request(vec![a.clone(), dir.0.join("no-such-release")], &table, &out),
        "member 1",
    );
    export_refused(
        &request(vec![a.clone(), b.clone(), a.clone()], &table, &out),
        "are one tower",
    );

    let (other_tokenizer, _) = member_release(5, &common::tiny_config(), "\n");
    export_refused(
        &request(vec![a.clone(), other_tokenizer.out.clone()], &table, &out),
        "tokenizer_hash",
    );

    let mut config = common::tiny_config();
    config["text_config"]["rope_parameters"]["rope_theta"] = 1_000_000.into();
    let (other_config, _) = member_release(6, &config, "");
    export_refused(
        &request(vec![a, b, other_config.out.clone()], &table, &out),
        "config_sha256",
    );
}

// -- the reader refuses the whole load for one bad member ----------------------------------------

#[test]
fn a_missing_member_tower_refuses_the_whole_ensemble() {
    let (w, _) = written();
    std::fs::remove_dir_all(w.out().join("tower-1")).unwrap();
    open_refused(&w.out(), ReleaseRefusalKind::Manifest, "ensemble member 1");
    open_refused(
        &w.out(),
        ReleaseRefusalKind::Manifest,
        "never served with the rest",
    );

    // The manifest drops the member instead: the members no longer give the bound weight hash.
    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["members"].as_array_mut().unwrap().remove(2);
    });
    open_refused(
        &w.out(),
        ReleaseRefusalKind::Manifest,
        "expected_identity.weight_hash",
    );

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["members"].as_array_mut().unwrap().truncate(1);
    });
    open_refused(&w.out(), ReleaseRefusalKind::Manifest, "2 to");
}

#[test]
fn a_member_whose_config_was_swapped_refuses_the_whole_ensemble() {
    let (w, _) = written();
    let mut other = common::tiny_config();
    other["text_config"]["rope_parameters"]["rope_theta"] = 1_000_000.into();
    std::fs::write(
        w.out().join("tower-2").join(CONFIG_FILE),
        serde_json::to_string_pretty(&other).unwrap(),
    )
    .unwrap();
    open_refused(
        &w.out(),
        ReleaseRefusalKind::ConfigMismatch,
        "ensemble member 2",
    );
}

#[test]
fn a_member_that_is_not_the_tower_the_manifest_names_is_refused() {
    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["members"][1]["weights_sha256"] = Value::from("ab".repeat(32));
    });
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MemberMismatch,
        "weights_sha256",
    );

    // Two members' directories swapped: each opens, neither is the tower its entry names.
    let (w, _) = written();
    let out = w.out();
    std::fs::rename(out.join("tower-0"), out.join("swap")).unwrap();
    std::fs::rename(out.join("tower-1"), out.join("tower-0")).unwrap();
    std::fs::rename(out.join("swap"), out.join("tower-1")).unwrap();
    open_refused(
        &out,
        ReleaseRefusalKind::MemberMismatch,
        "ensemble member 0",
    );
}

/// Replace member `i`'s directory with `release` and re-record its entry to match, so only the
/// reader's cross-member checks stand between it and serving.
fn substitute_member(out: &Path, i: usize, release: &Path) {
    let target = out.join(format!("tower-{i}"));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::create_dir(&target).unwrap();
    for entry in std::fs::read_dir(release).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
    }
    let tower = Tower::open(&target).unwrap();
    rewrite_ensemble_manifest(out, |m| {
        let member = &mut m["members"][i];
        member["release_manifest_sha256"] = Value::from(tower.manifest_sha256());
        member["weights_sha256"] = Value::from(tower.weights_sha256());
        member["weight_hash"] = Value::from(tower.weight_hash());
        member["config_sha256"] = Value::from(tower.config_sha256());
        member["tokenizer_hash"] = Value::from(tower.tokenizer_hash());
    });
}

#[test]
fn members_that_disagree_on_tokenizer_config_or_width_are_refused_by_the_reader() {
    let (w, _) = written();
    let (other, _) = member_release(5, &common::tiny_config(), "\n");
    substitute_member(&w.out(), 1, &other.out);
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MembersDisagree,
        "tokenizer_hash",
    );

    let (w, _) = written();
    let mut config = common::tiny_config();
    config["text_config"]["rope_parameters"]["rope_theta"] = 1_000_000.into();
    let (other, _) = member_release(6, &config, "");
    substitute_member(&w.out(), 2, &other.out);
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MembersDisagree,
        "config_sha256",
    );

    let (w, _) = written();
    substitute_member(&w.out(), 2, &w._members[0].out);
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MembersDisagree,
        "are one tower",
    );

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| m["members"][1]["trained_width"] = json!(4096));
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MembersDisagree,
        "trained at width 4096",
    );
}

#[test]
fn an_ensemble_manifest_without_a_width_a_decode_or_a_contained_dir_is_refused() {
    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["members"][0]
            .as_object_mut()
            .unwrap()
            .remove("trained_width");
    });
    open_refused(&w.out(), ReleaseRefusalKind::Manifest, "trained_width");

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["expected_identity"]["trained_width"] = json!(WIDTH + 1);
    });
    open_refused(
        &w.out(),
        ReleaseRefusalKind::Manifest,
        "expected_identity.trained_width",
    );

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["decode"] = Value::from("mean over members of the probabilities");
    });
    open_refused(
        &w.out(),
        ReleaseRefusalKind::Manifest,
        "does not serve one recorded",
    );

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["members"][0]["dir"] = Value::from("../tower-0")
    });
    open_refused(
        &w.out(),
        ReleaseRefusalKind::Manifest,
        "one plain path component",
    );

    let (w, _) = written();
    rewrite_ensemble_manifest(&w.out(), |m| m["format"] = Value::from("qd-ensemble.v0"));
    open_refused(&w.out(), ReleaseRefusalKind::Manifest, "qd-ensemble");
}

#[test]
fn an_ensemble_whose_table_is_not_the_written_one_is_refused() {
    let (w, _) = written();
    std::fs::write(
        w.out().join(CALIBRATION_FILE),
        serde_json::to_vec(&CalibrationTable::reference()).unwrap(),
    )
    .unwrap();
    open_refused(
        &w.out(),
        ReleaseRefusalKind::CalibrationMismatch,
        "not the table",
    );

    let (w, _) = written();
    std::fs::remove_file(w.out().join(CALIBRATION_FILE)).unwrap();
    match Ensemble::open(&w.out()) {
        Err(err) => assert_eq!(err.kind, ReleaseRefusalKind::CalibrationMissing, "{err}"),
        Ok(_) => panic!("an ensemble without its table opened"),
    }

    // A member's own release manifest edited after the ensemble was written.
    let (w, _) = written();
    let path = w.out().join("tower-0").join(MANIFEST_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&path, bytes).unwrap();
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MemberMismatch,
        "release_manifest_sha256",
    );
}

// -- serving -------------------------------------------------------------------------------------

/// The reference backend's behaviour under the identity a backend loaded from one member tower
/// would report.
struct LoadedFrom {
    inner: ReferenceBackend,
    identity: BackendIdentity,
}

impl LoadedFrom {
    fn tower(summary: &ExportSummary, calibration_hash: &str) -> Arc<dyn DecisionBackend> {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        identity.weight_hash = summary.weight_hash.clone();
        identity.tokenizer_hash = summary.tokenizer_hash.clone();
        identity.calibration_hash = calibration_hash.to_string();
        Arc::new(Self { inner, identity })
    }
}

impl DecisionBackend for LoadedFrom {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.inner.prefill(prefix)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.inner.snapshot(handle)
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        self.inner.decode_slot(snapshot, query, mode)
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

fn loaded(summaries: &[ExportSummary], calibration_hash: &str) -> Vec<Arc<dyn DecisionBackend>> {
    summaries
        .iter()
        .map(|s| LoadedFrom::tower(s, calibration_hash))
        .collect()
}

fn pinned_request(
    weight_hash: &str,
    calibration_hash: &str,
) -> qd_runtime::schema::DecisionRequest {
    let context = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";
    let value = json!({
        "schema_version": 1, "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context), "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": [{"name": "verdict", "type": "choice",
                   "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
        "expect": {"weight_hash": weight_hash, "calibration_hash": calibration_hash},
    });
    match parse_line(&serde_json::to_vec(&value).unwrap()) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted: {other:?}"),
    }
}

#[test]
fn a_runtime_built_from_the_ensemble_answers_under_the_ensemble_identity_and_table() {
    let (w, summary) = written();
    let ensemble = Ensemble::open(&w.out()).unwrap();
    let table = ensemble_table().hash();
    let runtime = Runtime::from_ensemble(
        &ensemble,
        loaded(&w.summaries, &table),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("every member loaded its tower");
    assert_eq!(runtime.calibration(), &ensemble_table());
    match runtime.answer(&pinned_request(&summary.weight_hash, &table), None) {
        Response::Ok(_) => {}
        other => panic!("the ensemble's own identity, pinned, must be answered: {other:?}"),
    }
    // A caller that pinned one seed's tower is not answered by the ensemble.
    match runtime.answer(&pinned_request(&w.summaries[0].weight_hash, &table), None) {
        Response::Refused(envelope) => assert_eq!(envelope.refusal.kind(), "hash_mismatch"),
        other => panic!("a pin of one member was answered by the ensemble: {other:?}"),
    }
}

#[track_caller]
fn not_built(result: Result<Runtime, BackendError>, needles: &[&str]) {
    match result {
        Err(BackendError::Unavailable { detail }) => {
            for needle in needles {
                assert!(detail.contains(needle), "{needle:?} not in {detail}");
            }
        }
        Err(other) => panic!("expected unavailable, got {other:?}"),
        Ok(_) => panic!("a runtime was built around an ensemble it must refuse"),
    }
}

#[test]
fn a_runtime_is_not_built_from_fewer_swapped_or_miscalibrated_members() {
    let (w, _) = written();
    let ensemble = Ensemble::open(&w.out()).unwrap();
    let table = ensemble_table().hash();
    let build = |members| {
        Runtime::from_ensemble(&ensemble, members, HeadRegistry::new(), RenderCaps::DEFAULT)
    };

    not_built(
        build(loaded(&w.summaries[..2], &table)),
        &["partial ensemble"],
    );

    let mut swapped = loaded(&w.summaries, &table);
    swapped.swap(0, 1);
    not_built(
        build(swapped),
        &["ensemble member 0", "identity_mismatch", "weight_hash"],
    );

    let reference = CalibrationTable::reference().hash();
    not_built(
        build(loaded(&w.summaries, &reference)),
        &["ensemble member 0", "calibration_hash"],
    );
}

// -- the binary ----------------------------------------------------------------------------------

const BIN: &str = env!("CARGO_BIN_EXE_qd-export-ensemble");

fn run(members: &[&Path], widths: &[u64], table: &Path, out: &Path) -> std::process::Output {
    let mut cmd = Command::new(BIN);
    for (member, width) in members.iter().zip(widths) {
        cmd.arg("--member").arg(member);
        cmd.arg("--trained-width").arg(width.to_string());
    }
    // A width without a member: clap pairs nothing, the writer counts.
    for width in widths.iter().skip(members.len()) {
        cmd.arg("--trained-width").arg(width.to_string());
    }
    cmd.arg("--trained-width-source")
        .arg(SOURCE)
        .arg("--calibration")
        .arg(table)
        .arg("--out")
        .arg(out);
    cmd.output().expect("the binary runs")
}

#[test]
fn the_binary_writes_what_the_reader_opens_and_exits_2_on_a_refusal() {
    let (fixtures, summaries) = members(3);
    let dir = common::TempDir::new("ensemble-bin");
    let table = table_file(&dir.0);
    let paths: Vec<&Path> = fixtures.iter().map(|f| f.out.as_path()).collect();

    let out = dir.0.join("ensemble");
    let ok = run(&paths, &[WIDTH; 3], &table, &out);
    let stdout = String::from_utf8(ok.stdout).unwrap();
    assert_eq!(
        ok.status.code(),
        Some(0),
        "{stdout}{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let ensemble = Ensemble::open(&out).expect("the binary's ensemble opens");
    assert!(
        stdout.contains(&format!("\nweight_hash: {}\n", ensemble.weight_hash())),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("calibration_hash: {}", ensemble_table().hash())),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("trained_width: {WIDTH}")),
        "{stdout}"
    );
    for (i, s) in summaries.iter().enumerate() {
        assert!(
            stdout.contains(&format!("member {i} weight_hash: {}", s.weight_hash)),
            "{stdout}"
        );
    }

    for (members, widths) in [(&paths[..1], vec![WIDTH]), (&paths[..2], vec![WIDTH; 3])] {
        let out = dir.0.join("refused");
        let refused = run(members, &widths, &table, &out);
        let stderr = String::from_utf8(refused.stderr).unwrap();
        assert_eq!(refused.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("qd-export-ensemble: refused"), "{stderr}");
        assert!(
            !out.exists(),
            "a refused ensemble left {} behind",
            out.display()
        );
    }
}

// -- trained families (Fable's pipeline ruling, item 4) ------------------------------------------

/// Set a member release's `trained_families`, as a release exported from another train manifest
/// would carry it.
fn set_trained_families(release: &Path, families: Value) -> String {
    let path = release.join(MANIFEST_FILE);
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    doc["trained_families"] = families;
    let bytes = serde_json::to_vec_pretty(&doc).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    qd_runtime::hex(&qd_runtime::sha256(&bytes))
}

#[test]
fn the_writer_refuses_members_trained_on_different_families() {
    let (fixtures, _) = members(2);
    let dir = common::TempDir::new("ensemble-tf");
    let table = table_file(&dir.0);
    let out = dir.0.join("ensemble");
    set_trained_families(&fixtures[0].out, json!(["code.defect_class"]));
    set_trained_families(
        &fixtures[1].out,
        json!(["code.change_scope", "code.defect_class"]),
    );
    let paths = fixtures.iter().map(|f| f.out.clone()).collect();
    export_refused(&request(paths, &table, &out), "trained_families");
}

#[test]
fn the_reader_refuses_members_trained_on_different_families() {
    let (w, _) = written();
    let sha = set_trained_families(&w.out().join("tower-1"), json!(["code.change_scope"]));
    rewrite_ensemble_manifest(&w.out(), |m| {
        m["members"][1]["release_manifest_sha256"] = Value::from(sha);
    });
    open_refused(
        &w.out(),
        ReleaseRefusalKind::MembersDisagree,
        "trained_families",
    );
}

#[test]
fn a_runtime_built_from_the_ensemble_refuses_a_task_its_members_did_not_train() {
    let (fixtures, summaries) = members(3);
    for fx in &fixtures {
        set_trained_families(&fx.out, json!(["code.defect_class"]));
    }
    let dir = common::TempDir::new("ensemble-tf-serve");
    let table_path = table_file(&dir.0);
    let out = dir.0.join("ensemble");
    let paths = fixtures.iter().map(|f| f.out.clone()).collect();
    let summary = export_ensemble(&request(paths, &table_path, &out))
        .expect("three members trained on one family set write an ensemble");
    let ensemble = Ensemble::open(&out).expect("the ensemble opens");
    let table = ensemble_table().hash();
    let runtime = Runtime::from_ensemble(
        &ensemble,
        loaded(&summaries, &table),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("every member loaded its tower");
    // `pinned_request` asks `devcouncil.verdict`.
    match runtime.answer(&pinned_request(&summary.weight_hash, &table), None) {
        Response::Refused(envelope) => {
            assert_eq!(envelope.refusal.kind(), "task_not_trained");
            let value = serde_json::to_value(&envelope.refusal).unwrap();
            assert_eq!(value["available"], json!(["code.defect_class"]));
        }
        other => panic!("the ensemble answered a task its members did not train: {other:?}"),
    }
}
