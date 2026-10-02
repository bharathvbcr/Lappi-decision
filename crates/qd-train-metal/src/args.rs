//! The command line. Portable, and run before anything is read or opened: every refusal here
//! happens before the shard door, the snapshot and the GPU.
//!
//! Every value of the recipe is a required flag. The binary has no defaults of its own,
//! because a default here would be a recipe choice nobody made. The exception is what F's
//! recipe pins and this trainer cannot change: one pass over the plan in the reader's order,
//! `grad_accum` 1, AdamW's betas, eps and weight decay, and the clip at 1.0 (qd-train's
//! `recipe`).
//!
//! **Refused by name**, so a flag from `tools/real_ft_run.py` cannot be dropped silently:
//!
//! * `--lower-layers-n` and `--lower-layers-lr-scale`. tessl has no per-entry learning rate
//!   (the lead's ruling 6, 2026-10-01). The refusal stays until tessl's `lr_scale` lands; it is
//!   never worked around.
//! * `--passes`, `--max-width`, `--option-permutation-seed` and `--replay-shards`, real_ft's
//!   source transforms. F's epoch arm uses none of them: it trains `plan_all` with `passes=1`
//!   (`tools/real_ft_run.py` `_train(plan=plan_all, passes=1)` in the `--epoch` branch). Its
//!   `--max-width` filters only the probe and memorise arms' `plan_small`. F's flags
//!   (`campaign/f-v4-preregistered.json`) set no permutation seed and no replay. With one pass,
//!   `_train`'s re-indexing across passes is the identity, so the reader's own batch order is
//!   F's. A recipe that needs a transform is refused rather than run without it
//!   (`GAP-L-DATA-REAL-FT-SOURCE-TRANSFORMS-NOT-PORTED-2026-10-01`).
//! * The door's escape hatches (`--allow-stale-code`, `--allow-rev-mismatch`,
//!   `--allow-not-run-snapshot`). This binary trains only from a shard set that passes every
//!   check.

use std::path::PathBuf;

use qd_train::ledger::check_ledger_path;
use qd_train::run_control::WallClockCap;
use qd_train::trainer::EtaRule;

/// The GEMM operand precision (ojas-qwen35's `Numerics`), which the recipe must name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operands {
    /// Exact f32 products: the tight arm.
    ExactF32,
    /// bf16 operands, f32 accumulation: the analogue of torch's bf16 training.
    Bf16,
}

impl Operands {
    /// The name the row's recipe records (`FtRecipe::operands`).
    pub fn as_str(self) -> &'static str {
        match self {
            Operands::ExactF32 => "exact_f32",
            Operands::Bf16 => "bf16",
        }
    }
}

/// One training run, as the command line states it.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainArgs {
    /// A Hugging Face snapshot directory (`config.json` and the weights). Its final component is
    /// the row's `backbone_snapshot`.
    pub snapshot: PathBuf,
    /// The directory the held-out roots are anchored under (the v4 out directory).
    pub data_root: PathBuf,
    /// The train shard set.
    pub shards: PathBuf,
    /// The pool manifest the shard set was written from.
    pub manifest: PathBuf,
    /// `python/qd_data`, whose fingerprint the shard header pins.
    pub qd_data: PathBuf,
    /// The corpus revision the shard set must have been written at.
    pub expect_rev: Option<String>,
    /// The span head's initial weights (`span_head.*` or bare names, f32).
    pub head_init: PathBuf,
    /// A new or empty directory for the export, its manifest and any checkpoint.
    pub out: PathBuf,
    /// `ledger/mac-ojas-*.jsonl`.
    pub ledger: PathBuf,
    /// The repository whose commit the row records.
    pub repo: PathBuf,
    /// The protocol seed: the row's, the export's name's and the head init's.
    pub seed: u64,
    pub lr: f64,
    /// Optimizer steps: the first `steps` batches of the epoch plan, one per step, and a
    /// schedule over exactly that many.
    pub steps: u64,
    pub batch_tokens: u64,
    pub span_weight: f64,
    pub cap_s: f64,
    pub eta: Option<EtaRule>,
    pub operands: Operands,
    pub checkpoint_every: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArgsError {
    /// The command line is malformed: an unknown or repeated flag, a missing value, a bad number.
    Usage(String),
    /// The command line asks for something this trainer refuses to run.
    Refused(String),
}

impl std::fmt::Display for ArgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgsError::Usage(m) => write!(f, "usage: {m}\n\n{USAGE}"),
            ArgsError::Refused(m) => write!(f, "refused: {m}"),
        }
    }
}

impl std::error::Error for ArgsError {}

pub const USAGE: &str = "qd-train-metal \
--snapshot DIR --data-root DIR --shards DIR --manifest FILE --qd-data DIR [--expect-rev REV] \
--head-init FILE --out DIR --ledger ledger/mac-ojas-*.jsonl --repo DIR \
--seed N --lr X --steps N --batch-tokens N --span-weight X --cap-s X --operands exact_f32|bf16 \
[--eta-at-step N --eta-margin-s X] [--checkpoint-every N]";

const TAKES_VALUE: [&str; 20] = [
    "--snapshot",
    "--data-root",
    "--shards",
    "--manifest",
    "--qd-data",
    "--expect-rev",
    "--head-init",
    "--out",
    "--ledger",
    "--repo",
    "--seed",
    "--lr",
    "--steps",
    "--batch-tokens",
    "--span-weight",
    "--cap-s",
    "--operands",
    "--eta-at-step",
    "--eta-margin-s",
    "--checkpoint-every",
];

/// The refusal for a layer-wise learning rate (the lead's ruling 6).
pub const LR_SCALE_REFUSAL: &str = "a layer-wise learning rate needs a per-entry lr_scale in tessl's \
Qwen35Model::adamw_step (qwen35_adamw.rs), which tessl does not have yet. Folding the scale into one \
rate or running two updates would change the optimizer under test, so this trainer refuses it until \
tessl's lr_scale lands (ruling 6, 2026-10-01). Without it this binary runs the single-group \
optimizer, which is not rung (d)";

fn refused_flag(flag: &str) -> Option<String> {
    match flag {
        "--lower-layers-n" | "--lower-layers-lr-scale" => Some(format!("{flag}: {LR_SCALE_REFUSAL}")),
        "--passes" | "--max-width" | "--option-permutation-seed" | "--replay-shards" => Some(format!(
            "{flag}: real_ft's source transforms are not ported. F's epoch arm uses none of them \
             (plan_all, passes=1, no permutation, no replay), so this trainer runs the reader's \
             batch order as is, and a recipe that needs one is refused rather than run without it"
        )),
        "--allow-stale-code" | "--allow-rev-mismatch" | "--allow-not-run-snapshot" => Some(format!(
            "{flag}: the shard door's escape hatches are not offered; this trainer reads only a \
             shard set that passes every check"
        )),
        _ => None,
    }
}

fn number<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, ArgsError> {
    raw.parse::<T>()
        .map_err(|_| ArgsError::Usage(format!("{flag} {raw:?} is not a valid number")))
}

/// Parse `args` (without the program name). Refuses before anything is read.
pub fn parse(args: &[String]) -> Result<TrainArgs, ArgsError> {
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_owned(), Some(v.to_owned())),
            _ => (arg.clone(), None),
        };
        if let Some(why) = refused_flag(&flag) {
            return Err(ArgsError::Refused(why));
        }
        if flag == "--help" || flag == "-h" {
            return Err(ArgsError::Usage("help requested".into()));
        }
        if !TAKES_VALUE.contains(&flag.as_str()) {
            return Err(ArgsError::Usage(format!("unknown argument {arg:?}")));
        }
        let value = match inline {
            Some(v) => v,
            None => it
                .next()
                .cloned()
                .ok_or_else(|| ArgsError::Usage(format!("{flag} needs a value")))?,
        };
        if value.is_empty() {
            return Err(ArgsError::Usage(format!("{flag} has an empty value")));
        }
        if seen.iter().any(|(f, _)| *f == flag) {
            return Err(ArgsError::Usage(format!("{flag} is given twice")));
        }
        seen.push((flag, value));
    }
    let get = |flag: &str| seen.iter().find(|(f, _)| f == flag).map(|(_, v)| v.as_str());
    let req = |flag: &str| get(flag).ok_or_else(|| ArgsError::Usage(format!("{flag} is required")));
    let path = |flag: &str| req(flag).map(PathBuf::from);

    let lr: f64 = number("--lr", req("--lr")?)?;
    if !(lr.is_finite() && lr > 0.0) {
        return Err(ArgsError::Usage(format!("--lr {lr} must be positive and finite")));
    }
    let steps: u64 = number("--steps", req("--steps")?)?;
    if steps == 0 {
        return Err(ArgsError::Usage("--steps must be at least 1".into()));
    }
    let batch_tokens: u64 = number("--batch-tokens", req("--batch-tokens")?)?;
    if batch_tokens == 0 {
        return Err(ArgsError::Usage("--batch-tokens must be at least 1".into()));
    }
    let span_weight: f64 = number("--span-weight", req("--span-weight")?)?;
    if !(span_weight.is_finite() && span_weight > 0.0) {
        return Err(ArgsError::Usage(format!("--span-weight {span_weight} must be positive and finite")));
    }
    let cap_s: f64 = number("--cap-s", req("--cap-s")?)?;
    WallClockCap::new(cap_s).map_err(|e| ArgsError::Usage(format!("--cap-s: {e}")))?;
    let operands = match req("--operands")? {
        "exact_f32" => Operands::ExactF32,
        "bf16" => Operands::Bf16,
        other => return Err(ArgsError::Usage(format!("--operands {other:?}: exact_f32 or bf16"))),
    };
    let eta = match (get("--eta-at-step"), get("--eta-margin-s")) {
        (None, None) => None,
        (Some(s), Some(m)) => {
            let rule = EtaRule {
                at_step: number("--eta-at-step", s)?,
                margin_s: number("--eta-margin-s", m)?,
            };
            rule.validate(steps, cap_s).map_err(|e| ArgsError::Usage(e.to_string()))?;
            Some(rule)
        }
        _ => return Err(ArgsError::Usage("--eta-at-step and --eta-margin-s come together".into())),
    };
    let checkpoint_every = match get("--checkpoint-every") {
        None => None,
        Some(raw) => {
            let n: u64 = number("--checkpoint-every", raw)?;
            if n == 0 {
                return Err(ArgsError::Usage("--checkpoint-every must be at least 1".into()));
            }
            Some(n)
        }
    };
    let ledger = path("--ledger")?;
    check_ledger_path(&ledger).map_err(|e| ArgsError::Refused(e.to_string()))?;
    Ok(TrainArgs {
        snapshot: path("--snapshot")?,
        data_root: path("--data-root")?,
        shards: path("--shards")?,
        manifest: path("--manifest")?,
        qd_data: path("--qd-data")?,
        expect_rev: get("--expect-rev").map(str::to_owned),
        head_init: path("--head-init")?,
        out: path("--out")?,
        ledger,
        repo: path("--repo")?,
        seed: number("--seed", req("--seed")?)?,
        lr,
        steps,
        batch_tokens,
        span_weight,
        cap_s,
        eta,
        operands,
        checkpoint_every,
    })
}
