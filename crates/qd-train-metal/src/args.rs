//! The command line. Portable, and run before anything is read or opened: every refusal here
//! happens before the shard door, the snapshot and the GPU.
//!
//! Every value of the recipe is a required flag. The binary has no defaults of its own,
//! because a default here would be a recipe choice nobody made. The exception is what F's
//! recipe pins and this trainer cannot change: one pass over the plan in the reader's order,
//! `grad_accum` 1, AdamW's betas, eps and weight decay, and the clip at 1.0 (qd-train's
//! `recipe`).
//!
//! **The layer-wise split** (`--lower-layers-n`, `--lower-layers-lr-scale`, together) is
//! accepted: ojas-qwen35's `OptimizerPlan` carries a learning-rate scale per entry into tessl's
//! `adamw_step_scaled` (Fable's board-sweep ruling 2h, 2026-10-12, retiring the lead's ruling 6
//! of 2026-10-01). The split is never folded into one rate or run as two updates.
//!
//! **Refused by name**, so a flag from `tools/real_ft_run.py` cannot be dropped silently:
//!
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

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;

use qd_train::ft_data::HeadInitPin;
use qd_train::ledger::{check_ledger_path, prompt_format_extra};
use qd_train::recipe::LowerLayers;
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
    /// What pins it (Amendment 2 (iv)): the file's sha256 and its content digest
    /// (`run_control._sidecar_digest` over `qd-tensor-ref-v1`), as its tracked manifest
    /// records them. The run refuses a file that matches either one alone, or neither.
    pub head_init_pin: HeadInitPin,
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
    /// The cosine floor when `--min-lr` is given; otherwise `lr / 10` (`LrSchedule::real_ft`).
    /// v5's recipe ran `--min-lr 0`.
    pub min_lr: Option<f64>,
    /// A `ckpt-<step>` directory of an earlier run of this same recipe to resume from.
    pub resume: Option<PathBuf>,
    /// F's layer-wise split: decoder layers `0 .. n - 1` at `lr_scale` times the schedule.
    pub lower_layers: Option<LowerLayers>,
}

impl TrainArgs {
    /// The seed the epoch plan is drawn at: the run's own `--seed`, as `real_ft_run.py
    /// --batch-order seed` does, so two seeds are two batch orders. (Rows before this change
    /// planned every seed at `DataConfig().seed`; the row's recipe now says `batch_order: seed`
    /// so they hash apart.) A resume refuses a checkpoint of another seed for the same reason.
    pub fn data_seed(&self) -> u64 {
        self.seed
    }

    /// What a checkpoint root records of this command line's recipe (`recipe.json` beside the
    /// `ckpt-*` directories): every setting that shapes the optimizer trajectory, which the
    /// checkpoint itself does not store. Precision is `f32`, the only one the provider opens.
    pub fn checkpoint_recipe(&self) -> Value {
        serde_json::json!({
            "format": "qd-train-metal-recipe/v1",
            "precision": "f32",
            "operands": self.operands.as_str(),
            "seed": self.seed,
            "batch_order": "seed",
            "lr": self.lr,
            "min_lr": self.min_lr,
            "steps": self.steps,
            "batch_tokens": self.batch_tokens,
            "span_weight": self.span_weight,
            "lower_layers": self.lower_layers.map(|l| serde_json::json!({"n": l.n, "lr_scale": l.lr_scale})),
        })
    }

    /// Refuse a resume whose recipe is not the one `saved` (read from the checkpoint root's
    /// `recipe.json`) records. Absent (`Null`) is refused: a checkpoint without its recipe was
    /// not written by this binary, so what it was trained under cannot be stated.
    pub fn check_resume_recipe(&self, saved: &Value) -> Result<(), String> {
        let now = self.checkpoint_recipe();
        if *saved == now {
            return Ok(());
        }
        let differs: Vec<&str> = match (saved.as_object(), now.as_object()) {
            (Some(was), Some(is)) => is
                .iter()
                .filter(|(k, v)| was.get(*k) != Some(*v))
                .map(|(k, _)| k.as_str())
                .collect(),
            _ => vec!["recipe.json is missing or not an object"],
        };
        Err(format!(
            "the checkpoint was written under another recipe ({differs:?} differ): a resume \
             continues the same optimizer trajectory or it is refused"
        ))
    }

    /// The recipe keys this command line adds beyond the fixed ones, as `real_ft_run.py`
    /// names them: `batch_order` (always `"seed"`: the plan is drawn at `--seed`), `min_lr`
    /// when `--min-lr` was given, and the shard set's `prompt_format` when it is not 1. Each
    /// moves the recipe hash, so a row with another batch order or floor never pools with one
    /// that lacks the key.
    pub fn recipe_extra(&self, prompt_format: u64) -> BTreeMap<String, Value> {
        let mut extra = prompt_format_extra(prompt_format);
        extra.insert("batch_order".to_owned(), Value::from("seed"));
        if let Some(floor) = self.min_lr {
            extra.insert("min_lr".to_owned(), Value::from(floor));
        }
        extra
    }
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
--head-init FILE --head-init-sha256 HEX --head-init-content-digest HEX \
--out DIR --ledger ledger/mac-ojas-*.jsonl --repo DIR \
--seed N --lr X --steps N --batch-tokens N --span-weight X --cap-s X --operands exact_f32|bf16 \
[--eta-at-step N --eta-margin-s X] [--checkpoint-every N] [--min-lr X] [--resume CKPT_DIR] \
[--lower-layers-n N --lower-layers-lr-scale X]";

const TAKES_VALUE: [&str; 26] = [
    "--head-init-sha256",
    "--head-init-content-digest",
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
    "--min-lr",
    "--resume",
    "--lower-layers-n",
    "--lower-layers-lr-scale",
];

fn refused_flag(flag: &str) -> Option<String> {
    match flag {
        "--passes" | "--max-width" | "--option-permutation-seed" | "--replay-shards" => {
            Some(format!(
                "{flag}: real_ft's source transforms are not ported. F's epoch arm uses none of them \
             (plan_all, passes=1, no permutation, no replay), so this trainer runs the reader's \
             batch order as is, and a recipe that needs one is refused rather than run without it"
            ))
        }
        "--allow-stale-code" | "--allow-rev-mismatch" | "--allow-not-run-snapshot" => {
            Some(format!(
                "{flag}: the shard door's escape hatches are not offered; this trainer reads only a \
             shard set that passes every check"
            ))
        }
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
    let get = |flag: &str| {
        seen.iter()
            .find(|(f, _)| f == flag)
            .map(|(_, v)| v.as_str())
    };
    let req = |flag: &str| get(flag).ok_or_else(|| ArgsError::Usage(format!("{flag} is required")));
    let path = |flag: &str| req(flag).map(PathBuf::from);

    let lr: f64 = number("--lr", req("--lr")?)?;
    if !(lr.is_finite() && lr > 0.0) {
        return Err(ArgsError::Usage(format!(
            "--lr {lr} must be positive and finite"
        )));
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
        return Err(ArgsError::Usage(format!(
            "--span-weight {span_weight} must be positive and finite"
        )));
    }
    let cap_s: f64 = number("--cap-s", req("--cap-s")?)?;
    WallClockCap::new(cap_s).map_err(|e| ArgsError::Usage(format!("--cap-s: {e}")))?;
    let operands = match req("--operands")? {
        "exact_f32" => Operands::ExactF32,
        "bf16" => Operands::Bf16,
        other => {
            return Err(ArgsError::Usage(format!(
                "--operands {other:?}: exact_f32 or bf16"
            )));
        }
    };
    let eta = match (get("--eta-at-step"), get("--eta-margin-s")) {
        (None, None) => None,
        (Some(s), Some(m)) => {
            let rule = EtaRule {
                at_step: number("--eta-at-step", s)?,
                margin_s: number("--eta-margin-s", m)?,
            };
            rule.validate(steps, cap_s)
                .map_err(|e| ArgsError::Usage(e.to_string()))?;
            Some(rule)
        }
        _ => {
            return Err(ArgsError::Usage(
                "--eta-at-step and --eta-margin-s come together".into(),
            ));
        }
    };
    let checkpoint_every = match get("--checkpoint-every") {
        None => None,
        Some(raw) => {
            let n: u64 = number("--checkpoint-every", raw)?;
            if n == 0 {
                return Err(ArgsError::Usage(
                    "--checkpoint-every must be at least 1".into(),
                ));
            }
            Some(n)
        }
    };
    let min_lr = match get("--min-lr") {
        None => None,
        Some(raw) => {
            let floor: f64 = number("--min-lr", raw)?;
            if !(floor.is_finite() && floor >= 0.0 && floor <= lr) {
                return Err(ArgsError::Usage(format!(
                    "--min-lr {floor} must be finite, at least 0 and at most --lr {lr}"
                )));
            }
            Some(floor)
        }
    };
    let lower_layers = match (get("--lower-layers-n"), get("--lower-layers-lr-scale")) {
        (None, None) => None,
        (Some(n), Some(scale)) => {
            let n: usize = number("--lower-layers-n", n)?;
            let lr_scale: f64 = number("--lower-layers-lr-scale", scale)?;
            if n == 0 {
                return Err(ArgsError::Usage(
                    "--lower-layers-n must be at least 1; zero is the plain optimizer, so give no split".into(),
                ));
            }
            if !(lr_scale.is_finite() && lr_scale > 0.0) {
                return Err(ArgsError::Usage(format!(
                    "--lower-layers-lr-scale {lr_scale} must be finite and positive"
                )));
            }
            Some(LowerLayers { n, lr_scale })
        }
        _ => {
            return Err(ArgsError::Usage(
                "--lower-layers-n and --lower-layers-lr-scale come together".into(),
            ));
        }
    };
    let ledger = path("--ledger")?;
    check_ledger_path(&ledger).map_err(|e| ArgsError::Refused(e.to_string()))?;
    let sha256_hex = |flag: &str| -> Result<String, ArgsError> {
        let v = req(flag)?;
        if v.len() == 64 && v.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            Ok(v.to_owned())
        } else {
            Err(ArgsError::Usage(format!(
                "{flag} {v:?} is not 64 lowercase hex characters"
            )))
        }
    };
    let head_init_pin = HeadInitPin {
        sha256: sha256_hex("--head-init-sha256")?,
        content_digest: sha256_hex("--head-init-content-digest")?,
    };
    Ok(TrainArgs {
        snapshot: path("--snapshot")?,
        data_root: path("--data-root")?,
        shards: path("--shards")?,
        manifest: path("--manifest")?,
        qd_data: path("--qd-data")?,
        expect_rev: get("--expect-rev").map(str::to_owned),
        head_init: path("--head-init")?,
        head_init_pin,
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
        min_lr,
        resume: get("--resume").map(PathBuf::from),
        lower_layers,
    })
}
