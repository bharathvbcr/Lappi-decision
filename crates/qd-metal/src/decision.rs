//! `qd-metal-bench --decision`: the latency of one real decision, A/B over flags.
//!
//! Fable's Mac-inference ruling (`AUDIT/fable-optimize-2026-10-02/fable-improve-and-mac-inference-ruling.md`,
//! Q2(c)) asks every optimization to be proven by one benchmark of *the real request shape*:
//! prefill + 2 decodes + digests + readback, at T = 512 / 2048 / 8192 and k = 4, interleaved A/B,
//! min-of-7 and median, with dispatch and sync-wait counts, written as a `quick` ledger row.
//!
//! # What one decision is here
//!
//! One `choice` slot of `k` options, answered the way `qd_runtime::answer` answers it through
//! [`crate::backend::MetalBackend`] (its worker's `prefill` and `decode_slot`):
//!
//! ```text
//! prefill(prefix)               -> PrefixState            Model::prefill
//! digest(state)                                            host SHA-256 of every state buffer
//! for the pass, then step 4's permuted pass (different suffix, same snapshot):
//!     run(suffix, from state, read-only) + score(k + 1 rows)  the readback
//!     digest(state)                                        must equal the first: read-only held
//! ```
//!
//! The prompt is the runtime's own: a wire request parsed by `qd_runtime::wire::parse_line`,
//! rendered by `qd_runtime::render::render`, the second suffix from `second_pass_permutation` +
//! `permuted_slot_suffix` exactly as `answer.rs` builds it. The suffix tokens are those of
//! `prefix + suffix` after the prefix's, refused if the boundary merges, as the backend does.
//!
//! This drives [`Model`] and not `MetalBackend`: the backend caches prefills by prompt digest
//! (`prompt_digest`, the worker's `prefill`), so every timed iteration after the first would skip
//! the prefill, and
//! its `MetalConfig` cannot carry the A/B flags (`backend.rs` is another session's file today).
//!
//! # T
//!
//! `T` is the **prefix** length in tokens, the prefill: the context is the largest whole number
//! of lines of a fixed source text (this crate's `model.rs`, repeated) whose rendered prefix is at
//! most `T` tokens. The row records the prefix and both suffix lengths actually run.
//!
//! # Timing
//!
//! Wall clock per phase, with a `synchronize` after the prefill so the prefill's GPU time and the
//! first digest's host time are separated. That sync costs nothing extra: the digest's first
//! buffer map commits and waits for the same work. Counts come from `tessl::infer_trace`
//! (dispatches, barriers, commits, cold allocations, sync-wait microseconds), reset per decision.
//! A host buffer map commits and waits, so `commits` counts the host syncs of a decision.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use qd_runtime::render::{RenderCaps, permuted_slot_suffix, render, second_pass_permutation};
use qd_runtime::wire::{Incoming, parse_line};
use qd_train::ledger::{Environment, Protocol, Row, Status, WallClockSource};
use qd_train::tristate::TriState;
use serde_json::{Value, json};

use crate::error::{MetalError, Result};
use crate::ledger::{self, Provenance, TreeState};
use crate::model::Model;
use crate::tokenizer::QwenTokenizer;

pub const DEFAULT_T: [usize; 3] = [512, 2048, 8192];
pub const DEFAULT_K: usize = 4;
pub const DEFAULT_ITERS: usize = 7;
pub const DEFAULT_WARMUP: usize = 2;

/// The options a slot of `k` offers: the four defect classes the product asks first, then
/// neutral labels up to the contract's 16.
const OPTIONS: [&str; 16] = [
    "stub", "logic", "cosmetic", "clean", "option e", "option f", "option g", "option h",
    "option i", "option j", "option k", "option l", "option m", "option n", "option o", "option p",
];
const TASK: &str = "code.defect_class";
const QUESTION: &str = "What kind of change is this diff?";
const SLOT: &str = "defect_class";
/// The context's source text. Throughput does not depend on which tokens; real code keeps the
/// tokenizer's work and the prompt's shape realistic. It is frozen: the `model.rs` that rows
/// 147da0cc and 28505f4c read (3d68484 + the tested 1a diff), so a later row feeds the same ids
/// (`data_snapshot_hash` 3d38c835...) whatever model.rs has become. Reading the live model.rs
/// made every edit to it a new input. A different context is a new file (`-v2`), never an edit
/// of this one; `tests::the_bench_context_is_the_frozen_v1_file` pins its sha256, and every row
/// records it as `context_sha256`.
const CONTEXT_SOURCE: &str = include_str!("../fixtures/decision-context-v1.txt");
const CONTEXT_SOURCE_NAME: &str = "crates/qd-metal/fixtures/decision-context-v1.txt";

/// One arm of the A/B: the one flag it varies, every other flag at the product's setting, so an
/// arm is fully determined by its name. No flag is left to vary: the `embed=host|device` and
/// `digest=serial|parallel` arms went with the paths they compared (their rows keep their
/// `embed_*` / `digest_*` keys), and [`Arm::Product`] runs the product as it is, one arm whose
/// samples are checked bit for bit against each other. A next flag, if one is approved, is a
/// variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Product,
}

impl Arm {
    pub fn name(&self) -> String {
        match self {
            Arm::Product => "product".to_string(),
        }
    }

    /// `product`, the only arm there is.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "product" => Ok(Arm::Product),
            _ => Err(MetalError::Input(format!(
                "arm {s:?} is not `product`, the one arm this bench runs"
            ))),
        }
    }

    /// The metric-key form of the name: `product`.
    fn key(&self) -> String {
        self.name().replace('=', "_")
    }
}

/// Where the row goes. A run without one says so on purpose; it is never the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowTarget {
    Ledger(PathBuf),
    /// `--no-ledger`: a development run. It prints `NO ROW WRITTEN` and its numbers are not
    /// citable (rule 5).
    None,
}

/// The tessl runtime a run opens: one per process, so it is a run flag (`--runtime`), not an
/// arm. qd-metal reads no GPU timestamps, so the two differ only in the CounterHeap work tessl
/// does at each commit (tessl runtime.rs `new_inference`: "host encode tax").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeKind {
    /// `GpuRuntime::new`, timestamps on: what `MetalBackend`'s worker opens. The default.
    Timestamps,
    /// `GpuRuntime::new_inference`, no CounterHeap timestamps.
    Inference,
}

impl RuntimeKind {
    pub fn name(&self) -> &'static str {
        match self {
            RuntimeKind::Timestamps => "timestamps",
            RuntimeKind::Inference => "inference",
        }
    }

    /// `timestamps|inference`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "timestamps" => Ok(RuntimeKind::Timestamps),
            "inference" => Ok(RuntimeKind::Inference),
            _ => Err(MetalError::Input(format!(
                "--runtime {s:?} is not `timestamps|inference`"
            ))),
        }
    }

    /// Open the runtime as `MetalBackend`'s worker does, with this constructor.
    pub fn open(&self) -> Result<std::sync::Arc<tessl::GpuRuntime>> {
        let rt = match self {
            RuntimeKind::Timestamps => tessl::GpuRuntime::new(),
            RuntimeKind::Inference => tessl::GpuRuntime::new_inference(),
        }
        .map_err(MetalError::Gpu)?;
        rt.set_async_encode(true).map_err(MetalError::Gpu)?;
        Ok(rt)
    }

    /// The recipe's `runtime`. The default's text is the one every earlier row recorded.
    fn recipe(&self) -> &'static str {
        match self {
            RuntimeKind::Timestamps => {
                "tessl::GpuRuntime::new + set_async_encode(true), as MetalBackend's worker"
            }
            RuntimeKind::Inference => {
                "tessl::GpuRuntime::new_inference + set_async_encode(true): no CounterHeap timestamps; \
                 MetalBackend's worker opens ::new"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionArgs {
    pub ts: Vec<usize>,
    pub k: usize,
    pub iters: usize,
    pub warmup: usize,
    pub arms: Vec<Arm>,
    pub runtime: RuntimeKind,
    pub row: RowTarget,
    pub snapshot: Option<PathBuf>,
}

fn parse_list(v: &str, what: &str) -> Result<Vec<usize>> {
    let out: Vec<usize> = v
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| MetalError::Input(format!("{what}: {s:?} is not a count")))
        })
        .collect::<Result<_>>()?;
    if out.is_empty() || out.contains(&0) {
        return Err(MetalError::Input(format!(
            "{what} must be positive counts, got {v:?}"
        )));
    }
    // A repeated T would run twice and write the same `decision.t<T>.*` keys, the second run
    // silently replacing the first while `t_coverage` still matched the recipe.
    if let Some((i, x)) = out.iter().enumerate().find(|&(i, x)| out[..i].contains(x)) {
        return Err(MetalError::Input(format!(
            "{what}: {x} is given twice (position {i}) in {v:?}"
        )));
    }
    Ok(out)
}

fn parse_one(v: &str, what: &str) -> Result<usize> {
    match parse_list(v, what)?.as_slice() {
        [one] => Ok(*one),
        _ => Err(MetalError::Input(format!(
            "{what} takes one count, got {v:?}"
        ))),
    }
}

/// Parse the arguments after `--decision`:
///
/// ```text
/// [T=512,2048,8192] [k=4] [--iters 7] [--warmup 2] [--arms product] [--runtime timestamps]
/// [--snapshot DIR] (--ledger ledger/mac-qd-metal-<date>.jsonl | --no-ledger)
/// ```
pub fn parse_args(args: &[String]) -> Result<DecisionArgs> {
    let mut ts = DEFAULT_T.to_vec();
    let mut k = DEFAULT_K;
    let mut iters = DEFAULT_ITERS;
    let mut warmup = DEFAULT_WARMUP;
    let mut arms = vec![Arm::Product];
    let mut runtime = RuntimeKind::Timestamps;
    let mut ledger: Option<PathBuf> = None;
    let mut no_ledger = false;
    let mut snapshot = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| MetalError::Input(format!("{flag} needs a value")))
        };
        if let Some(v) = a.strip_prefix("T=") {
            ts = parse_list(v, "T")?;
        } else if let Some(v) = a.strip_prefix("k=") {
            k = parse_one(v, "k")?;
        } else {
            match a.as_str() {
                "--iters" => iters = parse_one(&value("--iters")?, "--iters")?,
                "--warmup" => {
                    let v = value("--warmup")?;
                    warmup = v.parse().map_err(|_| {
                        MetalError::Input(format!("--warmup: {v:?} is not a count"))
                    })?;
                }
                "--arms" => {
                    arms = value("--arms")?
                        .split(',')
                        .map(|s| Arm::parse(s.trim()))
                        .collect::<Result<_>>()?;
                }
                "--runtime" => runtime = RuntimeKind::parse(&value("--runtime")?)?,
                "--ledger" => ledger = Some(PathBuf::from(value("--ledger")?)),
                "--no-ledger" => no_ledger = true,
                "--snapshot" => snapshot = Some(PathBuf::from(value("--snapshot")?)),
                other => return Err(MetalError::Input(format!("unknown argument {other:?}"))),
            }
        }
    }
    if !(qd_runtime::schema::MIN_OPTIONS..=qd_runtime::schema::MAX_OPTIONS).contains(&k) {
        return Err(MetalError::Input(format!(
            "k = {k}: a choice slot has {}..={} options",
            qd_runtime::schema::MIN_OPTIONS,
            qd_runtime::schema::MAX_OPTIONS
        )));
    }
    if arms.is_empty() {
        return Err(MetalError::Input("--arms names no arm".into()));
    }
    for (i, a) in arms.iter().enumerate() {
        if arms[..i].contains(a) {
            return Err(MetalError::Input(format!(
                "arm {} is given twice",
                a.name()
            )));
        }
    }
    // An A/B varies one flag: two arms naming different flags each differ from the product in a
    // different flag, so their delta is neither flag's. (Unreachable while `product` is the only arm;
    // it holds the rule for the next flag.)
    if let Some(a) = arms
        .iter()
        .find(|a| std::mem::discriminant(*a) != std::mem::discriminant(&arms[0]))
    {
        return Err(MetalError::Input(format!(
            "arms {} and {} vary different flags; an A/B varies one",
            arms[0].name(),
            a.name()
        )));
    }
    let row = match (ledger, no_ledger) {
        (Some(p), false) => {
            ledger::check_ledger_path(&p)?;
            RowTarget::Ledger(p)
        }
        (None, true) => RowTarget::None,
        (Some(_), true) => {
            return Err(MetalError::Input(
                "--ledger and --no-ledger contradict each other".into(),
            ));
        }
        (None, false) => {
            return Err(MetalError::Input(
                "say where the row goes: --ledger ledger/mac-qd-metal-<date>.jsonl, or --no-ledger \
                 for a development run whose numbers are not citable"
                    .into(),
            ));
        }
    };
    Ok(DecisionArgs {
        ts,
        k,
        iters,
        warmup,
        arms,
        runtime,
        row,
        snapshot,
    })
}

/// The token ids of one decision at one `T`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPrompt {
    pub target: usize,
    pub context_lines: usize,
    pub prefix: Vec<u32>,
    /// The pass and step 4's permuted pass: each suffix's tokens after the prefix's.
    pub passes: [Vec<u32>; 2],
    /// Decode rows per pass: `k` options + noul.
    pub rows: usize,
}

fn request_json(context: &str, k: usize) -> Value {
    json!({
        "schema_version": 1,
        "task": TASK,
        "context_b64": qd_runtime::b64::encode(context.as_bytes()),
        "context_len": context.len(),
        "question": QUESTION,
        "slots": [{"name": SLOT, "type": "choice", "options": &OPTIONS[..k]}],
        "route": "generic",
    })
}

/// The rendered prefix and the two suffixes for a context of `lines` source lines.
fn context(lines: usize) -> String {
    CONTEXT_SOURCE
        .lines()
        .cycle()
        .take(lines)
        .flat_map(|l| [l, "\n"])
        .collect()
}

/// The most source lines whose context fits the runtime's context cap: the render would refuse
/// one more, so the search never asks for it.
fn max_context_lines(cap_bytes: usize) -> usize {
    // Every line costs at least its newline, so at most `cap_bytes` lines can fit.
    let mut bytes = 0usize;
    let mut fit = 0usize;
    for l in CONTEXT_SOURCE.lines().cycle().take(cap_bytes) {
        bytes += l.len() + 1;
        if bytes > cap_bytes {
            break;
        }
        fit += 1;
    }
    fit
}

fn rendered(lines: usize, k: usize) -> Result<(String, [String; 2])> {
    let context = context(lines);
    let line = serde_json::to_vec(&request_json(&context, k))
        .map_err(|e| MetalError::Input(format!("request: {e}")))?;
    let request = match parse_line(&line) {
        Ok(Incoming::Request(r)) => r,
        Ok(_) => {
            return Err(MetalError::Input(
                "the bench request parsed as a control op".into(),
            ));
        }
        Err(refusal) => {
            return Err(MetalError::Input(format!(
                "the bench request was refused: {refusal:?}"
            )));
        }
    };
    let caps = RenderCaps::DEFAULT;
    let prompt = render(&request, &caps)
        .map_err(|r| MetalError::Input(format!("render refused the bench request: {r:?}")))?;
    let first = prompt
        .slot(SLOT)
        .ok_or_else(|| MetalError::Input("the rendered prompt has no slot".into()))?
        .suffix
        .clone();
    let perm = second_pass_permutation(&request.digest(), SLOT, k);
    let second = permuted_slot_suffix(&request.slots[0], &caps, &perm, 0)
        .map_err(|r| MetalError::Input(format!("the permuted pass was refused: {r:?}")))?
        .suffix;
    Ok((prompt.prefix, [first, second]))
}

/// The continuation tokens of `suffix` after `prefix_ids`, refused if `prefix + suffix` does not
/// tokenize as the prefix's tokens followed by more (the check `backend.rs` makes).
fn continuation(
    tok: &QwenTokenizer,
    prefix: &str,
    prefix_ids: &[u32],
    suffix: &str,
) -> Result<Vec<u32>> {
    let all = tok.encode(&format!("{prefix}{suffix}"))?;
    if all.len() <= prefix_ids.len() || all[..prefix_ids.len()] != prefix_ids[..] {
        return Err(MetalError::Input(format!(
            "prefix + suffix does not tokenize as the prefix's {} tokens followed by the suffix's",
            prefix_ids.len()
        )));
    }
    Ok(all[prefix_ids.len()..].to_vec())
}

/// The largest context (in whole source lines) whose rendered prefix is at most `target` tokens.
pub fn build_prompt(tok: &QwenTokenizer, target: usize, k: usize) -> Result<DecisionPrompt> {
    let tokens_at =
        |lines: usize| -> Result<usize> { Ok(tok.encode(&rendered(lines, k)?.0)?.len()) };
    if tokens_at(1)? > target {
        return Err(MetalError::Input(format!(
            "T = {target}: the request's fixed text alone is longer than that"
        )));
    }
    // Every line adds at least one token (a newline), so `target` lines is an upper bound; so is
    // the runtime's context cap, which `render` would refuse past.
    let (mut lo, mut hi) = (
        1usize,
        target.min(max_context_lines(RenderCaps::DEFAULT.max_context_bytes)),
    );
    if hi < 1 {
        return Err(MetalError::Input(
            "not one source line fits the runtime's context cap".into(),
        ));
    }
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if tokens_at(mid)? <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let (prefix_text, suffixes) = rendered(lo, k)?;
    let prefix = tok.encode(&prefix_text)?;
    let passes = [
        continuation(tok, &prefix_text, &prefix, &suffixes[0])?,
        continuation(tok, &prefix_text, &prefix, &suffixes[1])?,
    ];
    if passes[0] == passes[1] {
        return Err(MetalError::Input(
            "the permuted pass rendered the same tokens as the first: it would not test anything"
                .into(),
        ));
    }
    Ok(DecisionPrompt {
        target,
        context_lines: lo,
        prefix,
        passes,
        rows: k + 1,
    })
}

/// The text `p` was tokenized from: the rendered prefix and the two passes' suffixes, as a
/// `DecisionBackend` caller hands them over (the product path tokenizes them itself).
pub fn prompt_text(p: &DecisionPrompt, k: usize) -> Result<(String, [String; 2])> {
    rendered(p.context_lines, k)
}

/// sha256 over every id the bench feeds, length-prefixed: the row's `data_snapshot_hash`.
pub fn inputs_digest(prompts: &[DecisionPrompt]) -> String {
    let mut acc = b"qd-metal.decision-bench.inputs.v1\0".to_vec();
    for p in prompts {
        for ids in [&p.prefix, &p.passes[0], &p.passes[1]] {
            acc.extend_from_slice(&(ids.len() as u64).to_le_bytes());
            for id in ids.iter() {
                acc.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    qd_runtime::hex(&qd_runtime::sha256(&acc))
}

/// One timed decision.
#[derive(Debug, Clone)]
pub struct Sample {
    pub total_ms: f64,
    pub prefill_ms: f64,
    /// The two passes' run + score (the readback).
    pub decode_ms: [f64; 2],
    /// The digest after the prefill, then after each pass.
    pub digest_ms: [f64; 3],
    pub counts: tessl::infer_trace::Snapshot,
    /// Both passes' logits, for the cross-arm bit-identity check.
    pub logits: Vec<f32>,
    pub state_digest: [u8; 32],
    pub state_bytes: usize,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Run one decision on `model` as it is configured.
pub fn run_decision(model: &Model, p: &DecisionPrompt, answers: &[u32]) -> Result<Sample> {
    tessl::infer_trace::reset_token_counters();
    let t0 = Instant::now();
    let (out, state) = model.prefill(&p.prefix)?;
    drop(out);
    model
        .runtime()
        .synchronize()
        .map_err(|e| MetalError::Gpu(format!("synchronize after the prefill: {e}")))?;
    let prefill_ms = ms(t0);
    let td = Instant::now();
    let d0 = state.digest()?;
    let mut digest_ms = [ms(td), 0.0, 0.0];
    let mut decode_ms = [0.0; 2];
    let mut logits = Vec::with_capacity(2 * answers.len());
    if let Some((ids, seq)) = crate::backend::equal_length_batch(&p.passes[0], &p.passes[1]) {
        // Both suffixes share one 128-row GEMM tile. Measured on the 2B weights:
        // two 60-token runs 49.7 ms, one batch 28.1 ms, logprobs bit-identical.
        // The second row is `2 * seq - 1`. Refuse before the forward when it does not fit:
        // the same check `decode_batch` makes. A wrapping multiply would score the wrong row.
        let Some(score_at) = crate::backend::batch_score_rows(seq) else {
            return Err(MetalError::Input(format!(
                "batched continuation of {seq} tokens has no in-range score row"
            )));
        };
        let tp = Instant::now();
        let out = model.run(&ids, 2, seq, Some(&state), false, None)?;
        let scores = model.score(&out, &score_at, answers)?;
        let elapsed = ms(tp);
        decode_ms = [elapsed, 0.0];
        if let Some(bad) = scores.logits.iter().find(|x| !x.is_finite()) {
            return Err(MetalError::Gpu(format!(
                "batched passes: non-finite answer logit {bad}"
            )));
        }
        if scores.logits.len() != 2 * answers.len() {
            return Err(MetalError::Gpu(format!(
                "batched passes returned {} logits, want {}",
                scores.logits.len(),
                2 * answers.len()
            )));
        }
        logits.extend_from_slice(&scores.logits);
        let td = Instant::now();
        let d = state.digest()?;
        digest_ms[1] = ms(td);
        if d != d0 {
            return Err(MetalError::State(format!(
                "batched passes changed the prefix state they read: digest {} -> {}",
                qd_runtime::hex(&d0),
                qd_runtime::hex(&d)
            )));
        }
    } else {
        for (i, cont) in p.passes.iter().enumerate() {
            let s = u32::try_from(cont.len())
                .map_err(|_| MetalError::Input("suffix too long".into()))?;
            let tp = Instant::now();
            let out = model.run(cont, 1, s, Some(&state), false, None)?;
            let scores = model.score(&out, &[s - 1], answers)?;
            decode_ms[i] = ms(tp);
            if let Some(bad) = scores.logits.iter().find(|x| !x.is_finite()) {
                return Err(MetalError::Gpu(format!(
                    "pass {i}: non-finite answer logit {bad}"
                )));
            }
            logits.extend_from_slice(&scores.logits);
            let td = Instant::now();
            let d = state.digest()?;
            digest_ms[i + 1] = ms(td);
            if d != d0 {
                return Err(MetalError::State(format!(
                    "pass {i} changed the prefix state it read: digest {} -> {}; a read-only \
                     continuation wrote the snapshot",
                    qd_runtime::hex(&d0),
                    qd_runtime::hex(&d)
                )));
            }
        }
    }
    let total_ms = ms(t0);
    Ok(Sample {
        total_ms,
        prefill_ms,
        decode_ms,
        digest_ms,
        counts: tessl::infer_trace::snapshot(),
        logits,
        state_digest: d0,
        state_bytes: state.nbytes(),
    })
}

pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

pub fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Every timed sample of one arm at one `T`.
#[derive(Debug, Clone)]
pub struct ArmResult {
    pub arm: Arm,
    pub samples: Vec<Sample>,
}

/// One `T`'s results, every arm.
#[derive(Debug, Clone)]
pub struct TResult {
    pub prompt: DecisionPrompt,
    pub arms: Vec<ArmResult>,
}

/// The order arms run in at iteration `i`: rotated each iteration, so no arm always runs first
/// (or always right after another one's allocations).
pub fn arm_order(n_arms: usize, i: usize) -> Vec<usize> {
    (0..n_arms).map(|j| (i + j) % n_arms).collect()
}

/// Warm up every arm, then `iters` interleaved rounds. Every sample of every arm must give the
/// same prefix-state digest and the same logits bits as the first arm's first sample when the
/// arms are expected to agree bitwise; that is checked by the caller from the samples.
pub fn run_t(
    model: &Model,
    prompt: &DecisionPrompt,
    answers: &[u32],
    arms: &[Arm],
    warmup: usize,
    iters: usize,
) -> Result<TResult> {
    for _ in arms {
        for _ in 0..warmup {
            run_decision(model, prompt, answers)?;
        }
    }
    let mut results: Vec<ArmResult> = arms
        .iter()
        .map(|&arm| ArmResult {
            arm,
            samples: Vec::with_capacity(iters),
        })
        .collect();
    for i in 0..iters {
        for j in arm_order(arms.len(), i) {
            results[j]
                .samples
                .push(run_decision(model, prompt, answers)?);
        }
    }
    Ok(TResult {
        prompt: prompt.clone(),
        arms: results,
    })
}

/// Whether every sample of every arm produced the first sample's logits and state digest, bit
/// for bit. The product repeats bit for bit (every sample of rows 147da0cc, 28505f4c and
/// dc51c827 agreed, and the GPU pin test repeats each digest), so a difference is a defect.
pub fn bit_identical(t: &TResult) -> (bool, usize, usize) {
    let Some(first) = t.arms.first().and_then(|a| a.samples.first()) else {
        return (false, 0, 0);
    };
    let mut n = 0usize;
    let mut same = 0usize;
    for s in t.arms.iter().flat_map(|a| &a.samples) {
        n += 1;
        let logits_same = s.logits.len() == first.logits.len()
            && s.logits
                .iter()
                .zip(&first.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if logits_same && s.state_digest == first.state_digest {
            same += 1;
        }
    }
    (same == n && n > 0, same, n)
}

/// What the row records beside the results.
#[derive(Debug, Clone, PartialEq)]
pub struct RunContext {
    pub device: String,
    pub snapshot: PathBuf,
    pub vocab: usize,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub load_s: f64,
    pub wall_clock_s: f64,
    pub provenance: Provenance,
    /// tessl's state after the last timed decision; must equal `provenance.tessl`.
    pub tessl_after: TreeState,
}

fn tri(passed: bool, value: Value, detail: impl Into<String>) -> TriState {
    TriState::ran(passed, value).with_detail(detail)
}

/// The recipe the row stores and hashes.
pub fn recipe(args: &DecisionArgs, snapshot: &Path, vocab: usize) -> Result<Value> {
    Ok(json!({
        "tool": "crates/qd-metal/src/bin/bench.rs --decision",
        "mode": "decision",
        "t_targets": args.ts,
        "t_is": "prefix (prefill) tokens; the context is the largest whole number of source lines that fits",
        "k": args.k,
        "iters": args.iters,
        "warmup": args.warmup,
        "arms": args.arms.iter().map(Arm::name).collect::<Vec<_>>(),
        "interleaved": "every round runs every arm once, the order rotated each round",
        "request": {"task": TASK, "question": QUESTION, "slot": SLOT, "options": &OPTIONS[..args.k], "route": "generic"},
        "context_source": CONTEXT_SOURCE_NAME,
        "context_sha256": qd_runtime::hex(&qd_runtime::sha256(CONTEXT_SOURCE.as_bytes())),
        "decision": "prefill + digest, then 2 read-only passes (run + score k+1 rows) each followed by a digest",
        "backbone_snapshot": snapshot.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        "backbone_vocab": vocab,
        "runtime": args.runtime.recipe(),
    }))
}

/// The `throughput` row. Fails closed: a non-finite statistic is refused by the row writer.
pub fn build_row(args: &DecisionArgs, results: &[TResult], ctx: &RunContext) -> Result<Row> {
    let recipe = recipe(args, &ctx.snapshot, ctx.vocab)?;
    let prompts: Vec<DecisionPrompt> = results.iter().map(|t| t.prompt.clone()).collect();
    let protocol = Protocol {
        data_snapshot_hash: inputs_digest(&prompts),
        tokenizer_hash: ctx.tokenizer_hash.clone(),
        backbone_commit: ledger::backbone_commit(&ctx.snapshot, ctx.vocab)?,
        recipe_hash: ledger::recipe_hash(&recipe)?,
        seed: 0,
    };
    let mut m: BTreeMap<String, TriState> = ctx.provenance.metrics();
    let tessl_held = ctx.tessl_after == ctx.provenance.tessl;
    m.insert(
        "tessl_unchanged_during_run".into(),
        tri(
            tessl_held,
            Value::from(ctx.tessl_after.digest()),
            format!("after the last decision: {}", ctx.tessl_after.describe()),
        ),
    );
    m.insert(
        "load_s".into(),
        tri(true, ledger::float(ctx.load_s)?, "Model::load wall clock"),
    );
    m.insert(
        "weight_hash".into(),
        tri(
            true,
            Value::from(ctx.weight_hash.as_str()),
            "the loader's hash over every tensor it read",
        ),
    );
    m.insert(
        "device".into(),
        tri(
            true,
            Value::from(ctx.device.as_str()),
            "tessl GpuRuntime device",
        ),
    );
    for t in results {
        let p = &t.prompt;
        let tk = format!("decision.t{}", p.target);
        m.insert(
            format!("{tk}.tokens"),
            tri(
                true,
                json!({"prefix": p.prefix.len(), "pass0": p.passes[0].len(), "pass1": p.passes[1].len(), "context_lines": p.context_lines}),
                "prefix (prefill) tokens and each pass's suffix tokens actually run",
            ),
        );
        let (same, n_same, n) = bit_identical(t);
        m.insert(
            format!("{tk}.arms_bit_identical"),
            if n == 0 {
                TriState::not_run("no arm produced a sample at this T, so nothing was compared")
            } else {
                tri(
                    same,
                    Value::from(n_same as u64),
                    "samples whose logits (both passes) and prefix-state digest equal the first \
                     arm's first sample bit for bit; the product repeats bit for bit, so every \
                     sample must",
                )
                .with_coverage(n_same as u64, n as u64)
            },
        );
        for a in &t.arms {
            let ak = format!("{tk}.{}", a.arm.key());
            let col =
                |f: &dyn Fn(&Sample) -> f64| -> Vec<f64> { a.samples.iter().map(f).collect() };
            let total = col(&|s: &Sample| s.total_ms);
            let n = a.samples.len() as u64;
            let stats = |name: &str, v: &[f64], what: &str| -> Result<(String, TriState)> {
                Ok((
                    format!("{ak}.{name}"),
                    tri(
                        true,
                        json!({"min": ledger::float(min(v))?, "median": ledger::float(median(v))?}),
                        format!("{what}; min and median of {n} interleaved samples (ms)"),
                    )
                    .with_coverage(n, n),
                ))
            };
            for (k, v) in [
                stats(
                    "total_ms",
                    &total,
                    "one decision end to end: prefill, 3 digests, 2 passes with readback",
                )?,
                stats(
                    "prefill_ms",
                    &col(&|s: &Sample| s.prefill_ms),
                    "prefill incl. embedding gather, to GPU completion",
                )?,
                stats(
                    "decode_ms",
                    &col(&|s: &Sample| s.decode_ms[0] + s.decode_ms[1]),
                    "both passes: run + score readback",
                )?,
                stats(
                    "digest_ms",
                    &col(&|s: &Sample| s.digest_ms.iter().sum()),
                    "the 3 host SHA-256 state digests",
                )?,
            ] {
                m.insert(k, v);
            }
            let c = |f: &dyn Fn(&tessl::infer_trace::Snapshot) -> u64| -> Vec<f64> {
                a.samples.iter().map(|s| f(&s.counts) as f64).collect()
            };
            m.insert(
                format!("{ak}.counts"),
                tri(
                    true,
                    json!({
                        "dispatches": ledger::float(median(&c(&|s: &tessl::infer_trace::Snapshot| s.dispatches)))?,
                        "barriers": ledger::float(median(&c(&|s: &tessl::infer_trace::Snapshot| s.barriers)))?,
                        "commits": ledger::float(median(&c(&|s: &tessl::infer_trace::Snapshot| s.commits)))?,
                        "cold_allocs": ledger::float(median(&c(&|s: &tessl::infer_trace::Snapshot| s.cold_allocs)))?,
                        "sync_wait_us": ledger::float(median(&c(&|s: &tessl::infer_trace::Snapshot| s.sync_wait_us)))?,
                    }),
                    "tessl::infer_trace per decision, median over the samples; a host buffer map \
                     commits and waits, so `commits` counts host syncs",
                )
                .with_coverage(n, n),
            );
            if let Some(s) = a.samples.first() {
                m.insert(
                    format!("{ak}.state_mb"),
                    tri(
                        true,
                        ledger::float(s.state_bytes as f64 / 1e6)?,
                        "PrefixState device bytes",
                    ),
                );
            }
        }
    }
    let all_same = results.iter().all(|t| bit_identical(t).0);
    // Every T the recipe names must have run: a run cut short (bench.rs stops when tessl moves)
    // is a capped sample, never a complete row, even if tessl later reads unchanged again.
    let ran_ts: Vec<usize> = results.iter().map(|t| t.prompt.target).collect();
    let all_ts = ran_ts == args.ts;
    m.insert(
        "t_coverage".into(),
        tri(
            all_ts,
            json!(ran_ts),
            format!(
                "the Ts that ran, against the recipe's t_targets {:?}",
                args.ts
            ),
        )
        .with_coverage(ran_ts.len() as u64, args.ts.len() as u64),
    );
    let status = if tessl_held && all_ts && all_same {
        Status::Completed
    } else {
        Status::Failed
    };
    Ok(Row {
        run_kind: "throughput".into(),
        protocol,
        status,
        quick_reason:
            "a latency benchmark: one process, base weights, a fixed synthetic request per T, \
                       interleaved samples; rule 8: quick, excluded from every decision"
                .into(),
        code_commit: ctx.provenance.code_commit.clone(),
        env: Environment {
            torch: "n/a: Rust binary, no torch in the process".into(),
            transformers_sha: "n/a: Rust binary, no transformers in the process".into(),
            device: "metal".into(),
            host: ctx.provenance.host.clone(),
        },
        metrics: m,
        noul_rate: TriState::not_run("a benchmark decodes no verdicts against a gate"),
        wall_clock_s: ctx.wall_clock_s,
        wall_clock_source: WallClockSource::Caller,
        notes: format!(
            "qd-metal decision latency on {} (weight hash {}). Samples bit-identical at every T: \
             {all_same}.{}{}{}",
            ctx.snapshot.display(),
            ctx.weight_hash,
            if tessl_held {
                ""
            } else {
                " tessl changed during the run: the A/B is discarded (status failed) and must be re-run."
            },
            if all_ts {
                String::new()
            } else {
                format!(
                    " Only {} of {} Ts ran ({ran_ts:?} of {:?}): a capped sample, status failed.",
                    ran_ts.len(),
                    args.ts.len(),
                    args.ts
                )
            },
            if all_same {
                ""
            } else {
                " Samples disagreed bit for bit: status failed."
            },
        ),
        recipe,
    })
}

/// One timed product-path decision through `MetalBackend` (item B), each call timed from the
/// caller's side, job hop included.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductSample {
    pub total_ms: f64,
    pub prefill_ms: f64,
    pub snapshot_ms: f64,
    pub decode_ms: [f64; 2],
}

/// Item B at one T: the product path against `run_decision` on the same ids, in one process,
/// and the checks that make the two comparable.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductPathT {
    pub prompt: DecisionPrompt,
    /// `run_decision` totals before and after the product phase.
    pub model_before_ms: Vec<f64>,
    pub model_after_ms: Vec<f64>,
    pub product: Vec<ProductSample>,
    /// The backend's host tokenization alone: the prefix once, prefix + suffix per pass.
    pub encode_ms: Vec<f64>,
    /// Product samples whose letter logits equal the Model path's bit for bit.
    pub logits_same: usize,
    /// Whether the two Model phases agreed bit for bit.
    pub model_phases_same: bool,
    /// Timed prefills that missed the cache (each got a new entry).
    pub fresh_prefills: usize,
}

/// The product-path recipe: one fixed procedure, so only the Ts, `k` and the counts vary.
pub fn product_recipe(
    ts: &[usize],
    k: usize,
    warmup: usize,
    iters: usize,
    snapshot: &Path,
    vocab: usize,
) -> Value {
    json!({
        "tool": "crates/qd-metal/tests/gpu.rs gpu_product_path_host_cost_vs_model",
        "mode": "product-path",
        "label": "item B: MetalBackend product path vs Model-driven decision",
        "t_targets": ts,
        "k": k,
        "iters": iters,
        "warmup": warmup,
        "phases": "Model, then MetalBackend, then Model, in one process with one model loaded at a time",
        "model": "decision::run_decision: prefill + digest, 2 read-only passes each followed by a digest",
        "product": "MetalBackend with the committed backend.rs/tokenizer.rs: prefill (tokenize, prompt \
                    sha256, prefill, digest) + snapshot + 2 read-only decode_slot (re-tokenize prefix + \
                    suffix, run, score, digest); max_entries 1, so every prefill misses the cache",
        "request": {"task": TASK, "question": QUESTION, "slot": SLOT, "options": &OPTIONS[..k], "route": "generic"},
        "context_source": CONTEXT_SOURCE_NAME,
        "context_sha256": qd_runtime::hex(&qd_runtime::sha256(CONTEXT_SOURCE.as_bytes())),
        "backbone_snapshot": snapshot.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        "backbone_vocab": vocab,
        "runtime": RuntimeKind::Timestamps.recipe(),
    })
}

/// Item B's row. Completed only when every T ran and, at every T, the product's logits equalled
/// the Model path's, the two Model phases agreed, every timed prefill missed the cache, and tessl
/// held; the delta itself is reported, never judged.
pub fn build_product_row(
    ts: &[usize],
    k: usize,
    warmup: usize,
    results: &[ProductPathT],
    ctx: &RunContext,
) -> Result<Row> {
    let iters = results.first().map_or(0, |t| t.product.len());
    let recipe = product_recipe(ts, k, warmup, iters, &ctx.snapshot, ctx.vocab);
    let prompts: Vec<DecisionPrompt> = results.iter().map(|t| t.prompt.clone()).collect();
    let protocol = Protocol {
        data_snapshot_hash: inputs_digest(&prompts),
        tokenizer_hash: ctx.tokenizer_hash.clone(),
        backbone_commit: ledger::backbone_commit(&ctx.snapshot, ctx.vocab)?,
        recipe_hash: ledger::recipe_hash(&recipe)?,
        seed: 0,
    };
    let mut m: BTreeMap<String, TriState> = ctx.provenance.metrics();
    let tessl_held = ctx.tessl_after == ctx.provenance.tessl;
    m.insert(
        "tessl_unchanged_during_run".into(),
        tri(
            tessl_held,
            Value::from(ctx.tessl_after.digest()),
            format!("after the last phase: {}", ctx.tessl_after.describe()),
        ),
    );
    m.insert(
        "load_s".into(),
        tri(
            true,
            ledger::float(ctx.load_s)?,
            "the first Model phase's Model::load wall clock",
        ),
    );
    m.insert(
        "weight_hash".into(),
        tri(
            true,
            Value::from(ctx.weight_hash.as_str()),
            "the loader's hash over every tensor it read",
        ),
    );
    m.insert(
        "device".into(),
        tri(
            true,
            Value::from(ctx.device.as_str()),
            "tessl GpuRuntime device",
        ),
    );
    let mut checks_held = true;
    for t in results {
        let p = &t.prompt;
        let tk = format!("product_path.t{}", p.target);
        let n = t.product.len();
        let same = n > 0 && t.logits_same == n && t.model_phases_same;
        let fresh = n > 0 && t.fresh_prefills == n;
        checks_held &= same && fresh;
        m.insert(
            format!("{tk}.tokens"),
            tri(
                true,
                json!({"prefix": p.prefix.len(), "pass0": p.passes[0].len(), "pass1": p.passes[1].len(), "context_lines": p.context_lines}),
                "prefix (prefill) tokens and each pass's suffix tokens",
            ),
        );
        m.insert(
            format!("{tk}.logits_bit_identical"),
            tri(
                same,
                json!({"product_samples": t.logits_same, "model_phases_agree": t.model_phases_same}),
                "product samples whose letter logits (both passes) equal the Model path's bit for bit, \
                 and whether the two Model phases agreed: the two paths did the same work",
            )
            .with_coverage(t.logits_same as u64, n as u64),
        );
        m.insert(
            format!("{tk}.fresh_prefills"),
            tri(
                fresh,
                Value::from(t.fresh_prefills as u64),
                "timed prefills that missed the cache (a new entry each)",
            )
            .with_coverage(t.fresh_prefills as u64, n as u64),
        );
        let before = min(&t.model_before_ms);
        let after = min(&t.model_after_ms);
        let model_min = before.min(after);
        let col =
            |f: &dyn Fn(&ProductSample) -> f64| -> Vec<f64> { t.product.iter().map(f).collect() };
        let product_total = col(&|s: &ProductSample| s.total_ms);
        let product_min = min(&product_total);
        m.insert(
            format!("{tk}.model_ms"),
            tri(
                true,
                json!({"min": ledger::float(model_min)?, "min_before": ledger::float(before)?, "min_after": ledger::float(after)?,
                       "drift": ledger::float((before - after).abs())?}),
                format!("run_decision total, min of {} per Model phase; drift is |before - after|", t.model_before_ms.len()),
            ),
        );
        m.insert(
            format!("{tk}.product_ms"),
            tri(
                true,
                json!({
                    "min": ledger::float(product_min)?,
                    "median": ledger::float(median(&product_total))?,
                    "prefill_min": ledger::float(min(&col(&|s: &ProductSample| s.prefill_ms)))?,
                    "snapshot_min": ledger::float(min(&col(&|s: &ProductSample| s.snapshot_ms)))?,
                    "decode0_min": ledger::float(min(&col(&|s: &ProductSample| s.decode_ms[0])))?,
                    "decode1_min": ledger::float(min(&col(&|s: &ProductSample| s.decode_ms[1])))?,
                }),
                format!("MetalBackend calls timed from the caller, min (and total median) of {n}"),
            )
            .with_coverage(n as u64, n as u64),
        );
        m.insert(
            format!("{tk}.delta_ms"),
            tri(
                true,
                ledger::float(product_min - model_min)?,
                "product min total - Model min total: what the product path adds; report only (Fable ruling 2, step 6a)",
            ),
        );
        m.insert(
            format!("{tk}.host_encode_ms"),
            tri(
                true,
                ledger::float(min(&t.encode_ms))?,
                "QwenTokenizer::encode alone, as the backend calls it: the prefix once, prefix + suffix per pass; min",
            ),
        );
    }
    let ran_ts: Vec<usize> = results.iter().map(|t| t.prompt.target).collect();
    let all_ts = ran_ts == ts;
    m.insert(
        "t_coverage".into(),
        tri(
            all_ts,
            json!(ran_ts),
            format!("the Ts that ran, against the recipe's t_targets {ts:?}"),
        )
        .with_coverage(ran_ts.len() as u64, ts.len() as u64),
    );
    let status = if tessl_held && all_ts && checks_held {
        Status::Completed
    } else {
        Status::Failed
    };
    Ok(Row {
        run_kind: "throughput".into(),
        protocol,
        status,
        quick_reason:
            "a latency measurement: one process, base weights, a fixed synthetic request per T; \
                       rule 8: quick, excluded from every decision"
                .into(),
        code_commit: ctx.provenance.code_commit.clone(),
        env: Environment {
            torch: "n/a: Rust binary, no torch in the process".into(),
            transformers_sha: "n/a: Rust binary, no transformers in the process".into(),
            device: "metal".into(),
            host: ctx.provenance.host.clone(),
        },
        metrics: m,
        noul_rate: TriState::not_run("a benchmark decodes no verdicts against a gate"),
        wall_clock_s: ctx.wall_clock_s,
        wall_clock_source: WallClockSource::Caller,
        notes: format!(
            "Item B, product path vs Model-driven decision on {} (weight hash {}). Caveats: it times the \
             committed backend.rs/tokenizer.rs (QwenTokenizer::encode), not main's uncommitted \
             encode_untrusted, so it does not measure H1; it runs the base snapshot, so its absolute \
             ms are not comparable to rows on the release weights; the reading is only the \
             in-process delta. Identity checks held at every T: {checks_held}.{}{}",
            ctx.snapshot.display(),
            ctx.weight_hash,
            if tessl_held {
                ""
            } else {
                " tessl changed during the run: status failed."
            },
            if all_ts {
                ""
            } else {
                " Not every T ran: a capped sample, status failed."
            },
        ),
        recipe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::equal_length_batch;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn equal_length_passes_pack_as_one_batch_and_unequal_ones_do_not() {
        let (ids, seq) = equal_length_batch(&[1, 2, 3], &[4, 5, 6]).unwrap();
        assert_eq!(seq, 3);
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);
        // Neither side is padded out to the other. A pad would attend.
        assert!(equal_length_batch(&[1, 2], &[3]).is_none());
        assert!(equal_length_batch(&[3], &[1, 2]).is_none());
        assert!(equal_length_batch(&[], &[]).is_none());
        assert!(equal_length_batch(&[1], &[]).is_none());
        assert!(equal_length_batch(&[], &[1]).is_none());
    }

    /// `run_decision` scores a packed pair with `batch_score_rows` before `model.run`.
    /// `2 * seq - 1` wraps in `u32` for `seq >= 2^31`; that must be `None`, not a row index.
    #[test]
    fn the_bench_score_row_does_not_wrap() {
        use crate::backend::batch_score_rows;

        assert_eq!(batch_score_rows(1), Some([0, 1]));
        assert_eq!(batch_score_rows(3), Some([2, 5]));
        assert_eq!(batch_score_rows(0), None, "seq 0 would underflow the score index");
        assert_eq!(batch_score_rows(1 << 31), None, "2 * seq must not wrap");
        assert_eq!(batch_score_rows(u32::MAX), None);
        let half = u32::MAX / 2;
        let second = u32::try_from(u64::from(half) * 2 - 1).unwrap();
        assert_eq!(batch_score_rows(half), Some([half - 1, second]));
    }

    #[test]
    fn the_ruling_spelling_parses_and_defaults_are_the_rulings() {
        let a = parse_args(&strs(&[
            "T=512,2048,8192",
            "k=4",
            "--ledger",
            "/r/ledger/mac-qd-metal-2026-10-02.jsonl",
        ]))
        .unwrap();
        assert_eq!(a.ts, vec![512, 2048, 8192]);
        assert_eq!((a.k, a.iters, a.warmup), (4, 7, 2));
        assert_eq!(
            a.arms.iter().map(Arm::name).collect::<Vec<_>>(),
            ["product"]
        );
        assert_eq!(
            a.row,
            RowTarget::Ledger(PathBuf::from("/r/ledger/mac-qd-metal-2026-10-02.jsonl"))
        );
        let b = parse_args(&strs(&["--no-ledger"])).unwrap();
        assert_eq!(b.ts, DEFAULT_T.to_vec());
        assert_eq!(b.row, RowTarget::None);
    }

    #[test]
    fn a_run_must_say_where_its_row_goes() {
        let e = parse_args(&strs(&["T=512"])).unwrap_err().to_string();
        assert!(e.contains("--ledger"), "{e}");
        assert!(
            parse_args(&strs(&[
                "--no-ledger",
                "--ledger",
                "/r/ledger/mac-qd-metal-x.jsonl"
            ]))
            .is_err()
        );
        // Never a campaign ledger or the trainer's.
        assert!(parse_args(&strs(&["--ledger", "/r/ledger/runs.jsonl"])).is_err());
        assert!(parse_args(&strs(&["--ledger", "/r/ledger/mac-ojas-x.jsonl"])).is_err());
    }

    #[test]
    fn bad_arguments_are_refused_not_defaulted() {
        for bad in [
            &["T=0", "--no-ledger"][..],
            &["T=", "--no-ledger"],
            &["T=512,x", "--no-ledger"],
            &["k=1", "--no-ledger"],
            &["k=17", "--no-ledger"],
            &["k=4,5", "--no-ledger"],
            &["--iters", "0", "--no-ledger"],
            &["--iters", "--no-ledger"],
            &["--arms", "product,product", "--no-ledger"],
            &["--arms", "", "--no-ledger"],
            &["--arms", "digest=device", "--no-ledger"],
            &["--wat", "--no-ledger"],
        ] {
            assert!(parse_args(&strs(bad)).is_err(), "{bad:?} was accepted");
        }
        let a = parse_args(&strs(&[
            "--iters",
            "3",
            "--warmup",
            "0",
            "--arms",
            "product",
            "--no-ledger",
        ]))
        .unwrap();
        assert_eq!((a.iters, a.warmup), (3, 0));
        assert_eq!(a.arms, vec![Arm::Product]);
    }

    /// Fail-first (Fable ruling 2, #6): a repeated T used to parse, run twice and write one set of
    /// `decision.t<T>.*` keys, the second run replacing the first under a passing `t_coverage`.
    #[test]
    fn a_repeated_t_is_refused() {
        for bad in [
            &["T=512,512", "--no-ledger"][..],
            &["T=131,409,131", "--no-ledger"],
        ] {
            let e = parse_args(&strs(bad))
                .expect_err(&format!("{bad:?} was accepted"))
                .to_string();
            assert!(e.contains("given twice"), "{e}");
        }
        assert_eq!(
            parse_args(&strs(&["T=131,409", "--no-ledger"])).unwrap().ts,
            vec![131, 409]
        );
    }

    /// The one-thread digest and the device embedding gather are gone, and with them the
    /// `digest=` and `embed=` arms: asking for one is refused, naming the one arm there is, never
    /// silently run as the product.
    #[test]
    fn the_retired_arms_are_refused() {
        for bad in [
            "digest=serial,digest=parallel",
            "digest=serial",
            "embed=host,embed=device",
            "embed=host",
            "embed=device",
            "product,embed=host",
            "embed=host,digest=parallel",
        ] {
            let e = parse_args(&strs(&["--arms", bad, "--no-ledger"]))
                .expect_err(&format!("--arms {bad} was accepted"))
                .to_string();
            assert!(e.contains("is not `product`, the one arm"), "{e}");
        }
    }

    #[test]
    fn interleaving_rotates_so_no_arm_always_runs_first() {
        assert_eq!(arm_order(2, 0), [0, 1]);
        assert_eq!(arm_order(2, 1), [1, 0]);
        assert_eq!(arm_order(3, 2), [2, 0, 1]);
        let firsts: std::collections::BTreeSet<usize> =
            (0..7).map(|i| arm_order(2, i)[0]).collect();
        assert_eq!(firsts.len(), 2);
    }

    #[test]
    fn statistics_are_min_and_median() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0, 7.0, 6.0];
        assert_eq!(min(&v), 1.0);
        assert_eq!(median(&v), 4.0);
        assert_eq!(median(&[1.0, 3.0]), 2.0);
        assert!(median(&[]).is_nan());
    }

    /// `--runtime` picks the constructor and the row says which. The default is the worker's and
    /// keeps the text every earlier row recorded, so a default run's recipe does not move.
    #[test]
    fn the_runtime_is_a_run_flag_defaulting_to_the_workers() {
        let at = |a: &[&str]| parse_args(&strs(a));
        let default = at(&["--no-ledger"]).unwrap();
        assert_eq!(default.runtime, RuntimeKind::Timestamps);
        let inf = at(&["--runtime", "inference", "--no-ledger"]).unwrap();
        assert_eq!(inf.runtime, RuntimeKind::Inference);
        for bad in [
            &["--runtime", "fast", "--no-ledger"][..],
            &["--runtime", "", "--no-ledger"],
            &["--runtime"],
        ] {
            assert!(at(bad).is_err(), "{bad:?} was accepted");
        }
        let snap = Path::new("/hf/snapshots/b1485b2f");
        let r = recipe(&default, snap, 248_320).unwrap();
        let worker = "tessl::GpuRuntime::new + set_async_encode(true), as MetalBackend's worker";
        assert_eq!(r["runtime"], worker);
        let r = recipe(&inf, snap, 248_320).unwrap();
        let inference = r["runtime"].as_str().unwrap();
        assert!(inference.starts_with("tessl::GpuRuntime::new_inference"));
    }

    /// The context is the frozen v1 file, byte for byte, and the row says which: an edit to the
    /// fixture fails here rather than silently feeding later rows different ids.
    #[test]
    fn the_bench_context_is_the_frozen_v1_file() {
        const V1_SHA256: &str = "32549c5c4ada77173dd0c5339992c31b087de3c8b0eed3ced74009b68785ebc0";
        let sha = qd_runtime::hex(&qd_runtime::sha256(CONTEXT_SOURCE.as_bytes()));
        assert_eq!(sha, V1_SHA256);
        assert_eq!(CONTEXT_SOURCE.lines().count(), 1150);
        let args = parse_args(&strs(&["--no-ledger"])).unwrap();
        let r = recipe(&args, Path::new("/hf/snapshots/b1485b2f"), 248_320).unwrap();
        assert_eq!(r["context_source"], CONTEXT_SOURCE_NAME);
        assert!(CONTEXT_SOURCE_NAME.ends_with("fixtures/decision-context-v1.txt"));
        assert_eq!(r["context_sha256"], V1_SHA256);
    }

    #[test]
    fn the_context_search_never_asks_render_for_more_than_its_cap() {
        let n = max_context_lines(RenderCaps::DEFAULT.max_context_bytes);
        assert!(context(n).len() <= RenderCaps::DEFAULT.max_context_bytes);
        assert!(context(n + 1).len() > RenderCaps::DEFAULT.max_context_bytes);
        assert!(rendered(n, 4).is_ok());
        assert_eq!(max_context_lines(0), 0);
    }

    #[test]
    fn the_second_pass_is_the_runtimes_permuted_suffix() {
        let (prefix, [a, b]) = rendered(20, 4).unwrap();
        assert!(prefix.starts_with(qd_runtime::render::M_BEGIN));
        assert!(prefix.contains(QUESTION));
        assert_ne!(a, b, "the permuted pass must render a different suffix");
        for s in [&a, &b] {
            assert!(s.ends_with(qd_runtime::render::M_ANSWER), "{s}");
            for o in &OPTIONS[..4] {
                assert!(s.contains(o), "{o} missing from {s}");
            }
        }
    }
}
