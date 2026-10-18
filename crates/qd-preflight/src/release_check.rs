//! `qd-release-check`: is this release one that has been shown to serve what it admits?
//!
//! The v0.1 preview was loaded through the socket on 90 rows of 15 families. It admitted 23. Its
//! core family, `code.defect_class`, could not be served; six admitted families were never sent
//! through the runtime; the notes quoted headline numbers over rows the release does not serve.
//! Nothing checked any of that, so this does. Inputs are a release directory, a replay results
//! file produced by [`replay`] (one command, through `qd oneshot --socket`, the product path), and
//! the repository's documents.
//!
//! PASS only if all of:
//!
//! * (a) every admitted family, for every request shape the release records it trained
//!   (`trained_slot_shapes` in `release_manifest.json`), has at least `min_answered` DISTINCT
//!   requests that the socket path ANSWERED (`status: ok`, not degraded). A refusal is a recorded
//!   response, not a served request. A release that records no shapes cannot be checked and fails.
//! * (b) every documented example request (a full request in a ```json block of
//!   `docs/schema-api.md` or `docs/caller-contract.md`, and every row of the caller contract's
//!   request-shapes table) was replayed and was either answered or refused with a refusal kind the
//!   corpus `fixtures/wire/refusal-<kind>.json` documents.
//! * (c) the replay's `weight_hash` and `calibration_hash` (the build's own `status` reply) equal
//!   the release's `expected_identity`.
//!
//! Exit codes (see the binary): 0 PASS, 1 FAIL, 3 NOT_RUN (an input is missing or unreadable). A
//! check that could not run is never a pass.
//!
//! Headline figures ([`Metrics`]) are computed over the admitted families only, with numerator and
//! denominator, and the all-rows figure is reported under a name that says it is not the release's.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const RESULTS_FORMAT: &str = "qd-replay.v1";
const MANIFEST_FILE: &str = "release_manifest.json";
const DOC_FILES: [&str; 2] = ["docs/schema-api.md", "docs/caller-contract.md"];
/// A reply line is bounded by the runtime's own cap (`qd_runtime::wire::MAX_PAYLOAD_BYTES`).
const MAX_REPLY_BYTES: usize = 1_572_864;
const MAX_RESULTS_BYTES: u64 = 1 << 30;

// ---------------------------------------------------------------------------------------------
// Verdicts

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    NotRun,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub verdict: Verdict,
    pub detail: String,
}

impl Check {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, verdict: Verdict::Pass, detail: detail.into() }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, verdict: Verdict::Fail, detail: detail.into() }
    }
    fn not_run(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, verdict: Verdict::NotRun, detail: detail.into() }
    }
}

/// Fail beats not-run beats pass: a demonstrated failure is never hidden behind a check that
/// could not run, and a check that could not run is never a pass.
pub fn overall(checks: &[Check]) -> Verdict {
    if checks.iter().any(|c| c.verdict == Verdict::Fail) {
        Verdict::Fail
    } else if checks.iter().any(|c| c.verdict == Verdict::NotRun) {
        Verdict::NotRun
    } else {
        Verdict::Pass
    }
}

pub fn exit_code(v: Verdict) -> u8 {
    match v {
        Verdict::Pass => 0,
        Verdict::Fail => 1,
        Verdict::NotRun => 3,
    }
}

pub fn result_line(v: Verdict) -> &'static str {
    match v {
        Verdict::Pass => "RESULT: PASS",
        Verdict::Fail => "RESULT: FAIL",
        Verdict::NotRun => "RESULT: NOT_RUN",
    }
}

// ---------------------------------------------------------------------------------------------
// The release

#[derive(Debug, Clone)]
pub struct ReleaseView {
    pub dir: PathBuf,
    pub trained_families: Vec<String>,
    /// `None` when the manifest records none: an old release, or one not written by the gated
    /// exporter.
    pub trained_slot_shapes: Option<BTreeMap<String, Vec<Vec<String>>>>,
    pub excluded_families: Vec<(String, String)>,
    pub weight_hash: Option<String>,
    pub calibration_hash: Option<String>,
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("{} is not JSON", path.display()))
}

impl ReleaseView {
    pub fn load(dir: &Path) -> Result<Self> {
        let m = read_json(&dir.join(MANIFEST_FILE))?;
        let families = m
            .get("trained_families")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("{MANIFEST_FILE} has no trained_families: the release admits no task"))?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| anyhow!("trained_families holds a non-string")))
            .collect::<Result<Vec<_>>>()?;
        let shapes = match m.get("trained_slot_shapes") {
            None | Some(Value::Null) => None,
            Some(v) => {
                let obj = v.as_object().ok_or_else(|| anyhow!("trained_slot_shapes is not an object"))?;
                let mut out = BTreeMap::new();
                for (family, shapes) in obj {
                    let arr = shapes.as_array().ok_or_else(|| anyhow!("trained_slot_shapes[{family}] is not an array"))?;
                    let mut mine = Vec::new();
                    for shape in arr {
                        let keys = shape
                            .as_array()
                            .ok_or_else(|| anyhow!("trained_slot_shapes[{family}] holds a non-array shape"))?
                            .iter()
                            .map(|k| k.as_str().map(str::to_string).ok_or_else(|| anyhow!("a shape key is not a string")))
                            .collect::<Result<Vec<_>>>()?;
                        let mut keys = keys;
                        keys.sort();
                        mine.push(keys);
                    }
                    out.insert(family.clone(), mine);
                }
                Some(out)
            }
        };
        let excluded = m
            .get("excluded_families")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| {
                        Some((e.get("family")?.as_str()?.to_string(), e.get("reason")?.as_str()?.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let id = |k: &str| m.pointer(&format!("/expected_identity/{k}")).and_then(Value::as_str).map(str::to_string);
        Ok(Self {
            dir: dir.to_path_buf(),
            trained_families: families,
            trained_slot_shapes: shapes,
            excluded_families: excluded,
            weight_hash: id("weight_hash"),
            calibration_hash: id("calibration_hash"),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Requests and their shapes

/// The sorted slot keys of a request, as the calibration table keys them: `choice:N` with N the
/// options plus the reserved `noul` row, `score:N` likewise, `span`.
pub fn request_shape(req: &Value) -> Option<Vec<String>> {
    let slots = req.get("slots")?.as_array()?;
    let mut keys = Vec::with_capacity(slots.len());
    for s in slots {
        match s.get("type")?.as_str()? {
            "choice" => keys.push(format!("choice:{}", s.get("options")?.as_array()?.len() + 1)),
            "score" => keys.push(format!("score:{}", s.get("bins")?.as_u64()? + 1)),
            "span" => keys.push("span".to_string()),
            _ => return None,
        }
    }
    keys.sort();
    Some(keys)
}

fn canonical(v: &Value) -> String {
    // serde_json's Map is ordered by key (no preserve_order feature here), so this is canonical.
    v.to_string()
}

fn request_hash(req: &Value) -> String {
    let d = Sha256::digest(canonical(req).as_bytes());
    d.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------------------------
// Documented examples

#[derive(Debug, Clone)]
pub enum DocExample {
    /// A whole request in a ```json block.
    Full { source: String, request: Value },
    /// A row of the caller contract's request-shapes table: task, question and slots, no context.
    Row { source: String, task: String, question: String, slots: Vec<Value> },
}

impl DocExample {
    pub fn source(&self) -> &str {
        match self {
            DocExample::Full { source, .. } | DocExample::Row { source, .. } => source,
        }
    }

    /// Does `req` replay this example?
    pub fn matches(&self, req: &Value) -> bool {
        match self {
            DocExample::Full { request, .. } => canonical(request) == canonical(req),
            DocExample::Row { task, question, slots, .. } => {
                req.get("task").and_then(Value::as_str) == Some(task)
                    && req.get("question").and_then(Value::as_str) == Some(question)
                    && req.get("slots").and_then(Value::as_array).is_some_and(|s| slot_sig(s) == slot_sig(slots))
            }
        }
    }
}

fn slot_sig(slots: &[Value]) -> Vec<(String, String, Vec<String>)> {
    slots
        .iter()
        .map(|s| {
            let get = |k: &str| s.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            let opts = s
                .get("options")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|o| o.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            (get("name"), get("type"), opts)
        })
        .collect()
}

fn backticks(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut parts = s.split('`');
    parts.next();
    while let (Some(inside), _) = (parts.next(), parts.next()) {
        out.push(inside);
    }
    out
}

fn parse_table_slots(cell: &str) -> Option<Vec<Value>> {
    let mut slots = Vec::new();
    for part in cell.split(';') {
        let ticks = backticks(part);
        let name = *ticks.first()?;
        let rest = part.split('`').nth(2)?.trim();
        if rest.starts_with("span") {
            slots.push(json!({"name": name, "type": "span"}));
        } else if rest.starts_with("choice") {
            let list = ticks.get(1)?.trim().strip_prefix('[')?.strip_suffix(']')?;
            let options: Vec<&str> = list.split(',').map(str::trim).collect();
            slots.push(json!({"name": name, "type": "choice", "options": options}));
        } else {
            return None;
        }
    }
    (!slots.is_empty()).then_some(slots)
}

/// Every documented example request in one markdown file.
pub fn extract_examples(text: &str, file: &str) -> Vec<DocExample> {
    let mut out = Vec::new();
    let mut in_json = false;
    let mut block = String::new();
    let mut start = 0usize;
    let mut table: Option<bool> = None; // Some(true) once a header with task and question was seen
    for (i, line) in text.lines().enumerate() {
        let n = i + 1;
        if let Some(rest) = line.trim_start().strip_prefix("```") {
            if in_json {
                if let Ok(v) = serde_json::from_str::<Value>(&block) {
                    let is_request = v.get("task").is_some() && v.get("slots").is_some() && v.get("schema_version").is_some();
                    if is_request {
                        out.push(DocExample::Full { source: format!("{file}:{start}"), request: v });
                    }
                }
                in_json = false;
                block.clear();
            } else if rest.trim() == "json" {
                in_json = true;
                start = n;
            }
            continue;
        }
        if in_json {
            block.push_str(line);
            block.push('\n');
            continue;
        }
        if !line.trim_start().starts_with('|') {
            table = None;
            continue;
        }
        let cells: Vec<&str> = line.trim().trim_matches('|').split('|').map(str::trim).collect();
        if cells.len() >= 4 && cells[1].contains("task") && cells[2].contains("question") && cells[3].contains("slots") {
            table = Some(true);
            continue;
        }
        if table == Some(true) && cells.len() >= 4 && !cells[0].starts_with("---") {
            let task = backticks(cells[1]).first().map(|s| s.to_string());
            let question = backticks(cells[2]).first().map(|s| s.to_string());
            let slots = parse_table_slots(cells[3]);
            if let (Some(task), Some(question), Some(slots)) = (task, question, slots) {
                out.push(DocExample::Row { source: format!("{file}:{n}"), task, question, slots });
            }
        }
    }
    out
}

pub fn load_examples(repo: &Path) -> Result<Vec<DocExample>> {
    let mut all = Vec::new();
    for f in DOC_FILES {
        let text = std::fs::read_to_string(repo.join(f)).with_context(|| format!("reading {f}"))?;
        let found = extract_examples(&text, f);
        if found.is_empty() {
            bail!("{f}: no documented example request found; the extractor and the document disagree");
        }
        all.extend(found);
    }
    Ok(all)
}

/// The refusal kinds the corpus documents: `fixtures/wire/refusal-<kind>.json`.
pub fn documented_refusals(repo: &Path) -> Result<BTreeSet<String>> {
    let dir = repo.join("fixtures/wire");
    let mut kinds = BTreeSet::new();
    for e in std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let name = e?.file_name().to_string_lossy().into_owned();
        if let Some(k) = name.strip_prefix("refusal-").and_then(|n| n.strip_suffix(".json")) {
            kinds.insert(k.to_string());
        }
    }
    if kinds.is_empty() {
        bail!("{} holds no refusal-<kind>.json: the documented refusal set is empty", dir.display());
    }
    Ok(kinds)
}

// ---------------------------------------------------------------------------------------------
// Replay results

#[derive(Debug, Clone)]
pub struct Replayed {
    pub id: String,
    pub source: String,
    pub request: Value,
    pub exit_code: Option<i64>,
    pub served_by: String,
    pub reply: Option<Value>,
}

impl Replayed {
    pub fn status(&self) -> Option<&str> {
        self.reply.as_ref()?.get("status")?.as_str()
    }
    /// The model was asked and answered (possibly by abstaining) and the answer is not degraded.
    pub fn answered(&self) -> bool {
        self.status() == Some("ok")
            && self.reply.as_ref().and_then(|r| r.get("degraded")).and_then(Value::as_bool) == Some(false)
    }
    pub fn refusal_kind(&self) -> Option<&str> {
        if self.status() != Some("refused") {
            return None;
        }
        self.reply.as_ref()?.pointer("/refusal/kind")?.as_str()
    }
    pub fn via_socket(&self) -> bool {
        self.served_by == "socket"
    }
}

#[derive(Debug, Clone, Default)]
pub struct Results {
    pub weight_hash: Option<String>,
    pub calibration_hash: Option<String>,
    pub records: Vec<Replayed>,
}

pub fn load_results(path: &Path) -> Result<Results> {
    let len = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?.len();
    if len > MAX_RESULTS_BYTES {
        bail!("{} is {len} bytes, over the {MAX_RESULTS_BYTES}-byte bound", path.display());
    }
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Results::default();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).with_context(|| format!("{}:{}: not JSON", path.display(), i + 1))?;
        match v.get("kind").and_then(Value::as_str) {
            Some("header") => {
                if v.get("format").and_then(Value::as_str) != Some(RESULTS_FORMAT) {
                    bail!("{}:{}: header format is not {RESULTS_FORMAT}", path.display(), i + 1);
                }
                let s = |k: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
                out.weight_hash = s("weight_hash");
                out.calibration_hash = s("calibration_hash");
            }
            Some("replay") => {
                let request = v.get("request").cloned().ok_or_else(|| anyhow!("{}:{}: no request", path.display(), i + 1))?;
                out.records.push(Replayed {
                    id: v.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                    source: v.get("source").and_then(Value::as_str).unwrap_or("").to_string(),
                    request,
                    exit_code: v.get("exit_code").and_then(Value::as_i64),
                    served_by: v.get("served_by").and_then(Value::as_str).unwrap_or("unknown").to_string(),
                    reply: v.get("reply").filter(|r| !r.is_null()).cloned(),
                });
            }
            other => bail!("{}:{}: unknown record kind {other:?}", path.display(), i + 1),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The checks

/// Per (family, shape): distinct answered requests and distinct refused ones.
#[derive(Debug, Clone, Serialize)]
pub struct Coverage {
    pub family: String,
    pub shape: Vec<String>,
    pub answered: usize,
    pub refused: usize,
    pub refusal_kinds: Vec<String>,
}

pub fn coverage(release: &ReleaseView, results: &Results) -> Option<Vec<Coverage>> {
    let shapes = release.trained_slot_shapes.as_ref()?;
    let mut out = Vec::new();
    for family in &release.trained_families {
        for shape in shapes.get(family).cloned().unwrap_or_default() {
            let mut answered = BTreeSet::new();
            let mut refused = BTreeSet::new();
            let mut kinds = BTreeSet::new();
            for r in results.records.iter().filter(|r| r.via_socket()) {
                if r.request.get("task").and_then(Value::as_str) != Some(family) || request_shape(&r.request).as_ref() != Some(&shape) {
                    continue;
                }
                let h = request_hash(&r.request);
                if r.answered() {
                    answered.insert(h);
                } else if let Some(k) = r.refusal_kind() {
                    refused.insert(h);
                    kinds.insert(k.to_string());
                }
            }
            out.push(Coverage {
                family: family.clone(),
                shape,
                answered: answered.len(),
                refused: refused.len(),
                refusal_kinds: kinds.into_iter().collect(),
            });
        }
    }
    Some(out)
}

pub struct Inputs<'a> {
    pub release: &'a ReleaseView,
    pub results: &'a Results,
    pub examples: &'a [DocExample],
    pub refusals: &'a BTreeSet<String>,
    pub min_answered: usize,
}

pub fn run_checks(inp: &Inputs<'_>) -> (Vec<Check>, Option<Vec<Coverage>>) {
    let mut checks = Vec::new();
    let r = inp.release;
    let res = inp.results;

    // (c) identity.
    let mut bad = Vec::new();
    for (what, release, replay) in [
        ("weight_hash", &r.weight_hash, &res.weight_hash),
        ("calibration_hash", &r.calibration_hash, &res.calibration_hash),
    ] {
        match (release, replay) {
            (Some(a), Some(b)) if a == b => {}
            (Some(a), Some(b)) => bad.push(format!("{what}: release {a}, replay {b}")),
            (None, _) => bad.push(format!("{what}: the release records none")),
            (_, None) => bad.push(format!("{what}: the replay recorded none (no status reply from the warm build)")),
        }
    }
    checks.push(if bad.is_empty() {
        Check::pass("c_identity", "replay weight_hash and calibration_hash equal the release's")
    } else {
        Check::fail("c_identity", bad.join("; "))
    });

    // Path: a replay that fell back in-process is not the product path.
    let sidecar = res.records.iter().filter(|x| !x.via_socket()).count();
    checks.push(if res.records.is_empty() {
        Check::fail("replay_path", "the results file holds no replayed request")
    } else if sidecar > 0 {
        Check::fail(
            "replay_path",
            format!("{sidecar} of {} replays were not served by the socket; they are excluded from every count", res.records.len()),
        )
    } else {
        Check::pass("replay_path", format!("all {} replays were served by the socket", res.records.len()))
    });

    // (a) coverage.
    let cov = coverage(r, res);
    checks.push(match &cov {
        None => Check::fail(
            "a_coverage",
            "the release records no trained_slot_shapes, so what it admits cannot be enumerated; re-export with the gated qd-export",
        ),
        Some(cov) => {
            let short: Vec<String> = cov
                .iter()
                .filter(|c| c.answered < inp.min_answered)
                .map(|c| {
                    format!(
                        "{} {:?}: {} answered (<{}), {} refused {:?}",
                        c.family, c.shape, c.answered, inp.min_answered, c.refused, c.refusal_kinds
                    )
                })
                .collect();
            let unshaped: Vec<&String> = r
                .trained_families
                .iter()
                .filter(|f| r.trained_slot_shapes.as_ref().is_none_or(|s| s.get(*f).is_none_or(Vec::is_empty)))
                .collect();
            if !unshaped.is_empty() {
                Check::fail("a_coverage", format!("admitted families with no recorded slot shape: {unshaped:?}"))
            } else if short.is_empty() {
                Check::pass(
                    "a_coverage",
                    format!("{} (family, shape) pairs each have >= {} distinct answered requests", cov.len(), inp.min_answered),
                )
            } else {
                Check::fail("a_coverage", format!("{} of {} (family, shape) pairs short: {}", short.len(), cov.len(), short.join(" | ")))
            }
        }
    });

    // (b) documented examples.
    let mut missing = Vec::new();
    let mut wrong = Vec::new();
    for ex in inp.examples {
        let hits: Vec<&Replayed> = res.records.iter().filter(|x| x.via_socket() && ex.matches(&x.request)).collect();
        if hits.is_empty() {
            missing.push(ex.source().to_string());
            continue;
        }
        let ok = hits.iter().any(|h| {
            h.answered() || h.refusal_kind().is_some_and(|k| inp.refusals.contains(k))
        });
        if !ok {
            let seen: Vec<String> = hits
                .iter()
                .map(|h| h.status().map_or("no status".to_string(), |s| format!("{s}/{}", h.refusal_kind().unwrap_or("-"))))
                .collect();
            wrong.push(format!("{} -> {seen:?}", ex.source()));
        }
    }
    checks.push(if inp.examples.is_empty() {
        Check::not_run("b_documented_examples", "no documented example request was extracted")
    } else if missing.is_empty() && wrong.is_empty() {
        Check::pass("b_documented_examples", format!("{} documented examples replayed, each answered or documentedly refused", inp.examples.len()))
    } else {
        Check::fail("b_documented_examples", format!("not replayed: {missing:?}; neither answered nor a documented refusal: {wrong:?}"))
    });

    (checks, cov)
}

// ---------------------------------------------------------------------------------------------
// Served-population metrics

#[derive(Debug, Clone, Default, Serialize)]
pub struct Fraction {
    pub numerator: u64,
    pub denominator: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Population {
    pub families: Vec<String>,
    pub choice_top1: Fraction,
    /// Rows the calibrated rule answers (neither the noul row nor the margin abstained).
    pub coverage: Fraction,
    /// Correct among answered.
    pub precision_on_answered: Fraction,
    pub rows_without_calibration_row: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub target_precision: Option<f64>,
    /// The release's headline: rows of the admitted families only.
    pub admitted_population: Population,
    /// Admitted families that also passed check (a): what the release has been shown to serve.
    pub shown_served_population: Population,
    /// Rows of families the release does not admit, excluded from both figures above.
    pub excluded_rows: u64,
    pub excluded_families: Vec<String>,
    /// Every row, whether or not the release admits it. Not the release's figure.
    #[serde(rename = "NOT_THE_RELEASES_all_rows_choice_top1")]
    pub not_the_releases_all_rows_choice_top1: Fraction,
}

/// Compute the headline figures from the eval harness's verdict rows (`family_id`, `kind`,
/// `correct`, `row_id`) joined to the fit's `rows.jsonl` (`row_id`, `scored`,
/// `abstained_noul_row_or_margin`, `correct`).
pub fn metrics(
    admitted: &BTreeSet<String>,
    shown_served: &BTreeSet<String>,
    verdicts: &Path,
    calib_rows: Option<&Path>,
    target_precision: Option<f64>,
    eval_row_id: Option<&str>,
) -> Result<Metrics> {
    let mut abst: BTreeMap<(String, String), bool> = BTreeMap::new();
    if let Some(p) = calib_rows {
        for (i, line) in std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?.lines().enumerate() {
            let v: Value = serde_json::from_str(line).with_context(|| format!("{}:{}", p.display(), i + 1))?;
            if v.get("scored").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            if let (Some(row), Some(slot), Some(a)) = (
                v.get("row_id").and_then(Value::as_str),
                v.get("slot_name").and_then(Value::as_str),
                v.get("abstained_noul_row_or_margin").and_then(Value::as_bool),
            ) {
                abst.insert((row.to_string(), slot.to_string()), a);
            }
        }
    }
    let mut adm = Population { families: admitted.iter().cloned().collect(), ..Default::default() };
    let mut shown = Population { families: shown_served.iter().cloned().collect(), ..Default::default() };
    let mut all = Fraction::default();
    let mut excluded = 0u64;
    let mut excluded_families = BTreeSet::new();
    let text = std::fs::read_to_string(verdicts).with_context(|| format!("reading {}", verdicts.display()))?;
    let mut saw_family = false;
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).with_context(|| format!("{}:{}", verdicts.display(), i + 1))?;
        if v.get("kind").and_then(Value::as_str) != Some("choice") {
            continue;
        }
        if let Some(want) = eval_row_id
            && v.get("eval_row_id").and_then(Value::as_str) != Some(want)
        {
            continue;
        }
        let family = v.get("family_id").and_then(Value::as_str);
        let correct = v.get("correct").and_then(Value::as_bool);
        let (Some(family), Some(correct)) = (family, correct) else {
            bail!("{}:{}: a choice verdict without family_id or correct", verdicts.display(), i + 1);
        };
        saw_family = true;
        all.denominator += 1;
        all.numerator += u64::from(correct);
        let in_admitted = admitted.contains(family);
        if !in_admitted {
            excluded += 1;
            excluded_families.insert(family.to_string());
            continue;
        }
        let key = (
            v.get("row_id").and_then(Value::as_str).unwrap_or("").to_string(),
            v.get("slot_name").and_then(Value::as_str).unwrap_or("").to_string(),
        );
        let answered = abst.get(&key).map(|a| !a);
        for (pop, member) in [(&mut adm, true), (&mut shown, shown_served.contains(family))] {
            if !member {
                continue;
            }
            pop.choice_top1.denominator += 1;
            pop.choice_top1.numerator += u64::from(correct);
            match answered {
                None => pop.rows_without_calibration_row += 1,
                Some(a) => {
                    pop.coverage.denominator += 1;
                    pop.coverage.numerator += u64::from(a);
                    if a {
                        pop.precision_on_answered.denominator += 1;
                        pop.precision_on_answered.numerator += u64::from(correct);
                    }
                }
            }
        }
    }
    if !saw_family {
        bail!("{}: no choice verdict rows with family_id", verdicts.display());
    }
    Ok(Metrics {
        target_precision,
        admitted_population: adm,
        shown_served_population: shown,
        excluded_rows: excluded,
        excluded_families: excluded_families.into_iter().collect(),
        not_the_releases_all_rows_choice_top1: all,
    })
}

// ---------------------------------------------------------------------------------------------
// The replay driver

pub struct ReplayConfig {
    pub qd: PathBuf,
    pub socket: PathBuf,
    pub release: PathBuf,
    pub repo: PathBuf,
    pub request_files: Vec<PathBuf>,
    pub socket_timeout: Duration,
    pub max_requests: usize,
}

/// Run `cmd` with `stdin`, killing the child we started if it outlives `limit`.
fn run_bounded(mut cmd: Command, stdin: &[u8], limit: Duration) -> Result<(Option<i32>, Vec<u8>, Vec<u8>)> {
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().with_context(|| format!("spawning {:?}", cmd.get_program()))?;
    let mut sin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
    let payload = stdin.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = sin.write_all(&payload);
    });
    let mut so = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut se = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
    let out = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = so.by_ref().take(MAX_REPLY_BYTES as u64 + 1).read_to_end(&mut b);
        b
    });
    let err = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = se.by_ref().take(1 << 20).read_to_end(&mut b);
        b
    });
    let deadline = Instant::now() + limit;
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break Some(s);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let _ = writer.join();
    let stdout = out.join().map_err(|_| anyhow!("stdout reader panicked"))?;
    let stderr = err.join().map_err(|_| anyhow!("stderr reader panicked"))?;
    Ok((status.map(|s| s.code().unwrap_or(-1)), stdout, stderr))
}

fn oneshot(cfg: &ReplayConfig, line: &[u8]) -> Result<(Option<i32>, String, Option<Value>, f64)> {
    let mut cmd = Command::new(&cfg.qd);
    cmd.arg("oneshot")
        .arg("--socket")
        .arg(&cfg.socket)
        .arg("--socket-timeout-ms")
        .arg(cfg.socket_timeout.as_millis().max(1).to_string());
    let t = Instant::now();
    let (code, stdout, stderr) = run_bounded(cmd, line, cfg.socket_timeout * 3 + Duration::from_secs(5))?;
    let secs = t.elapsed().as_secs_f64();
    let stderr = String::from_utf8_lossy(&stderr);
    let served_by = stderr
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("qd oneshot: served by "))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let reply = if stdout.len() > MAX_REPLY_BYTES {
        None
    } else {
        serde_json::from_slice::<Value>(stdout.split(|b| *b == b'\n').next().unwrap_or(&[])).ok()
    };
    Ok((code, served_by, reply, secs))
}

/// Replay the documented examples and every request in `cfg.request_files` through
/// `qd oneshot --socket`, then ask the warm build for its identity the same way. Returns the
/// JSONL results text.
pub fn replay(cfg: &ReplayConfig) -> Result<String> {
    let examples = load_examples(&cfg.repo)?;
    let full: Vec<&Value> = examples
        .iter()
        .filter_map(|e| match e {
            DocExample::Full { request, .. } => Some(request),
            DocExample::Row { .. } => None,
        })
        .collect();
    let mut work: Vec<(String, Value)> = Vec::new();
    for ex in &examples {
        match ex {
            DocExample::Full { source, request } => work.push((format!("doc:{source}"), request.clone())),
            DocExample::Row { source, task, question, slots } => {
                // A row carries no context. One that a full example already covers is replayed
                // there; the others get a neutral context, so what is measured is admission of
                // the task and shape.
                if full.iter().any(|f| ex.matches(f)) {
                    continue;
                }
                let context = format!("example context for {task}");
                work.push((
                    format!("doc:{source}"),
                    json!({
                        "schema_version": 1, "task": task,
                        "context_b64": b64(context.as_bytes()), "context_len": context.len(),
                        "question": question, "slots": slots, "route": "generic",
                        "example_id": format!("qd-release-check:{task}:doc-row"), "metadata": {},
                    }),
                ));
            }
        }
    }
    for f in &cfg.request_files {
        let text = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
        for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).with_context(|| format!("{}:{}: not JSON", f.display(), i + 1))?;
            work.push((format!("file:{}:{}", f.display(), i + 1), v));
        }
    }
    if work.len() > cfg.max_requests {
        bail!("{} requests is over --max-requests {}", work.len(), cfg.max_requests);
    }

    let mut lines = Vec::with_capacity(work.len() + 1);
    for (n, (source, request)) in work.iter().enumerate() {
        let (code, served_by, reply, secs) = oneshot(cfg, canonical(request).as_bytes())?;
        lines.push(json!({
            "kind": "replay", "id": format!("r{:06}", n + 1), "source": source, "request": request,
            "exit_code": code, "served_by": served_by, "reply": reply, "seconds": secs,
        }));
    }
    let (code, served_by, status, _) = oneshot(cfg, br#"{"op":"status"}"#)?;
    let backend = status.as_ref().and_then(|s| s.get("backend")).filter(|b| !b.is_null());
    let hash = |k: &str| backend.and_then(|b| b.get(k)).and_then(Value::as_str).map(str::to_string);
    let header = json!({
        "kind": "header", "format": RESULTS_FORMAT,
        "release_dir": cfg.release.display().to_string(), "socket": cfg.socket.display().to_string(),
        "status_exit_code": code, "status_served_by": served_by, "status_reply": status,
        "weight_hash": hash("weight_hash"), "calibration_hash": hash("calibration_hash"),
        "n_replays": lines.len(),
    });
    let mut out = String::new();
    out.push_str(&header.to_string());
    out.push('\n');
    for l in lines {
        out.push_str(&l.to_string());
        out.push('\n');
    }
    Ok(out)
}

fn b64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].iter().enumerate() {
            if i <= c.len() {
                s.push(T[((n >> shift) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_is_canonical_with_padding() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foob"), "Zm9vYg==");
    }

    #[test]
    fn a_table_row_and_a_full_block_are_both_extracted() {
        let md = "| Decision point | `task` | `question` | slots | context |\n| --- | --- | --- | --- | --- |\n| A | `x.y` | `Q?` | `c` choice `[a, b]`; `s` span | ctx |\n\n```json\n{\"schema_version\":1,\"task\":\"x.y\",\"slots\":[]}\n```\n";
        let ex = extract_examples(md, "t.md");
        assert_eq!(ex.len(), 2);
        assert!(matches!(&ex[0], DocExample::Row { task, slots, .. } if task == "x.y" && slots.len() == 2));
        assert!(matches!(&ex[1], DocExample::Full { .. }));
    }

    #[test]
    fn shapes_count_the_reserved_row() {
        let req = json!({"slots": [{"name": "a", "type": "choice", "options": ["x", "y", "z", "w"]}, {"name": "s", "type": "span"}]});
        assert_eq!(request_shape(&req), Some(vec!["choice:5".to_string(), "span".to_string()]));
    }
}
