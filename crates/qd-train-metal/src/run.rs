//! One run, end to end. Everything that can be refused on the host is refused before the GPU
//! is opened:
//!
//! 0. ojas is at the commit `upstream.lock.json` pins, or `--unpinned-ojas` marks the row ([`crate::pin`]);
//! 1. the output directory is new or empty;
//! 2. the train shard set opens through the held-out door, with no escape hatch, and is the
//!    `train` split;
//! 3. every epoch's plan (`reader.plan(batch_tokens, --seed, e)` for `e` in `0..--epochs`:
//!    `real_ft_run.py --batch-order seed`, so two seeds are two orders) has at least `steps`
//!    batches;
//! 4. the snapshot's config and weights header pass ojas-qwen35's checks, its directory name
//!    can be a `backbone_snapshot`, and qd-export's `Layout` reads its config;
//! 5. the shard set's remap is the identity over the snapshot's vocabulary ([`crate::remap`]);
//! 6. the span head's initial weights load at the tower's width.
//!
//! Then the ft row's id is minted, the provider opens, and qd-train's loop runs the first
//! `steps` batches of each epoch with `grad_accum` 1 and `LrSchedule::real_ft(lr, steps * epochs)`
//! (one cosine over the whole run; `--min-lr` gives the floor; `--resume` continues from a
//! checkpoint of the same recipe, mid-epoch included). The trained
//! masters are read back once and exported twice: the scored bf16 file and the f32 masters.
//! Last, one `quick` row goes to the `mac-ojas` ledger. A run that fails at any point writes no
//! row, so it is rerun rather than remembered.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use ojas_qwen35::Snapshot;
use qd_export::layout::Layout;
use qd_train::export::{
    self, ExportSummary, METAL_DEVICE, METAL_TRAINER, ManifestInfo, NamedTensor, Precision,
};
use qd_train::files::OsFiles;
use qd_train::ft_data::{HostSpanHead, RealBatch, load_span_head_pinned};
use qd_train::held_out::DataConfig;
use qd_train::ledger::{
    self, Environment, FtRecipe, FtRow, METAL_FULL_FT_OPTIMIZER_RECIPE, Protocol, RunFacts, Status,
    WallClockSource,
};
use qd_train::objective::LetterSpanObjective;
use qd_train::recipe::{self, LowerLayers, OptimizerRecipe};
use qd_train::run_control::{MonotonicClock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::shards::{ShardOpen, ShardReader};
use qd_train::step::{ParamSpec, StepProvider};
use qd_train::trainer::{
    self, CheckpointPolicy, EpochSource, Hooks, HostParams, Progress, ResumeState, Termination,
    TrainConfig, TrainError,
};
use qd_train::tristate::TriState;

use crate::args::TrainArgs;
use crate::pin;
use crate::provider::Qwen35Provider;
use crate::remap::require_identity_remap;

/// The file beside a run's `ckpt-*` directories that records the recipe they were trained under.
pub const RECIPE_FILE: &str = "recipe.json";

/// What a finished run wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub row_id: String,
    pub termination: Termination,
    pub optimizer_steps: u64,
    pub export: ExportSummary,
    pub masters: ExportSummary,
}

#[derive(Debug)]
pub enum RunError {
    Refused(String),
    Failed(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Refused(m) => write!(f, "refused: {m}"),
            RunError::Failed(m) => write!(f, "failed: {m}"),
        }
    }
}

impl std::error::Error for RunError {}

fn refused<T>(m: impl Into<String>) -> Result<T, RunError> {
    Err(RunError::Refused(m.into()))
}

fn failed<E: std::fmt::Display>(what: &'static str) -> impl FnOnce(E) -> RunError {
    move |e| RunError::Failed(format!("{what}: {e}"))
}

/// The arm's quick reason. Every row this binary writes is quick (rule 8).
fn quick_reason(
    steps: u64,
    epochs: u64,
    epoch_batches: usize,
    lower_layers: Option<LowerLayers>,
) -> String {
    let optimizer = match lower_layers {
        None => {
            "single-group optimizer: F's layer-wise split is not applied, so this is not rung (d)"
                .to_owned()
        }
        Some(l) => format!(
            "two-group optimizer: decoder layers below {} at {}x the schedule, through ojas-qwen35's \
             per-entry lr_scale",
            l.n, l.lr_scale
        ),
    };
    let passes = if epochs > 1 {
        format!("{epochs} epochs of ")
    } else {
        String::new()
    };
    format!(
        "Metal/tessl trainer (ojas-qwen35), 1 seed, {passes}{steps} of {epoch_batches} epoch batches, {optimizer}"
    )
}

fn check_out_dir(out: &Path) -> Result<(), RunError> {
    match fs::read_dir(out) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                return refused(format!(
                    "{} is not empty; a run writes into a new directory",
                    out.display()
                ));
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(out).map_err(|e| RunError::Failed(format!("{}: {e}", out.display())))
        }
        Err(e) => Err(RunError::Failed(format!("{}: {e}", out.display()))),
    }
}

/// sha256 of the file at `path`'s bytes, lowercase hex (Python's `data_access._file_sha256`).
fn file_sha256_hex(path: &Path) -> Result<String, RunError> {
    let digest = qd_export::safetensors::sha256_file(path)
        .map_err(|e| RunError::Refused(format!("{}: cannot hash the train manifest: {e}", path.display())))?;
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Run `read` (the one thing that opens the train manifest) between two hashes of the manifest
/// file's bytes, as `data_access.open_training_data` does, and return what it read with the
/// digest. A file whose bytes changed in between is refused: the digest a row records must be
/// the file `read` read.
pub fn read_with_manifest_sha256<T>(
    manifest: &Path,
    read: impl FnOnce() -> Result<T, RunError>,
) -> Result<(T, String), RunError> {
    let before = file_sha256_hex(manifest)?;
    let value = read()?;
    if file_sha256_hex(manifest)? != before {
        return refused(format!(
            "{}: the manifest's bytes changed between the hash and the read, refusing, so a \
             recorded train_manifest_sha256 is the file read",
            manifest.display()
        ));
    }
    Ok((value, before))
}

/// The optional row fields this run records: the train manifest file's sha256.
pub fn row_extras(train_manifest_sha256: &str) -> ledger::RowExtras {
    ledger::RowExtras {
        train_manifest_sha256: Some(train_manifest_sha256.to_owned()),
        ..ledger::RowExtras::default()
    }
}

/// [`ledger::append`] with `extras` ([`FtRow::line_with`]): the same `mac-ojas-*` path rule, the
/// same locked, chained, fsynced append.
pub fn append_row(
    path: &Path,
    row: &FtRow,
    extras: &ledger::RowExtras,
    row_id: String,
    written_at: String,
) -> Result<ledger::Stamp, ledger::LedgerError> {
    ledger::check_ledger_path(path)?;
    ledger::append_line(path, row_id, written_at, |stamp| row.line_with(stamp, extras))
}

/// Run `a` to completion. See the module docs for the order of checks.
pub fn run(a: &TrainArgs) -> Result<RunSummary, RunError> {
    // ---- host-side refusals, before the GPU ------------------------------------------------
    // The ojas pin first: a run at an unlocked ojas writes nothing unless --unpinned-ojas.
    let pin = pin::check_ojas(
        Path::new(pin::LOCK_PATH),
        Path::new(pin::OJAS_DIR),
        a.unpinned_ojas,
    )
    .map_err(RunError::Refused)?;
    check_out_dir(&a.out)?;
    let config = DataConfig::default();
    let data_seed = a.data_seed();
    let mut opts = ShardOpen::new(config, a.data_root.clone(), a.qd_data.clone());
    opts.expect_rev = a.expect_rev.clone();
    opts.files = Arc::new(OsFiles);
    let (reader, manifest_sha256) = read_with_manifest_sha256(&a.manifest, || {
        ShardReader::open(&a.shards, &a.manifest, &opts)
            .map_err(|e| RunError::Refused(format!("the shard door: {e}")))
    })?;
    let header = reader.header().clone();
    if header.split != "train" {
        return refused(format!(
            "{} is the {:?} split; training reads only train",
            a.shards.display(),
            header.split
        ));
    }
    let n = usize::try_from(a.steps).map_err(|e| RunError::Refused(format!("--steps: {e}")))?;
    // Every epoch's plan is drawn now, so a short one refuses before the GPU opens, not after
    // the earlier epochs have trained. Epoch `e`'s order is a function of (--seed, e).
    let mut epoch_batches = 0usize;
    let mut width = 0u64;
    for epoch in 0..a.epochs {
        let plans = reader
            .plan(a.batch_tokens, data_seed, epoch)
            .map_err(failed("the epoch plan"))?;
        if plans.len() < n {
            return refused(format!(
                "epoch {epoch} has {} batches at batch_tokens {}; {} steps of one batch each do \
                 not fit",
                plans.len(),
                a.batch_tokens,
                a.steps
            ));
        }
        if epoch == 0 {
            epoch_batches = plans.len();
        }
        width = plans[..n]
            .iter()
            .map(|p| p.width as u64)
            .max()
            .unwrap_or(0)
            .max(width);
    }
    let total_steps = a
        .steps
        .checked_mul(a.epochs)
        .ok_or_else(|| RunError::Refused(format!("--steps x --epochs {} overflows", a.epochs)))?;

    let snapshot = Snapshot::from_dir(&a.snapshot)
        .map_err(|e| RunError::Refused(format!("the snapshot: {e}")))?;
    let backbone_snapshot = a
        .snapshot
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| {
            RunError::Refused(format!(
                "{} has no directory name to record",
                a.snapshot.display()
            ))
        })?
        .to_owned();
    let cfg_text = fs::read_to_string(&snapshot.config_path).map_err(failed("config.json"))?;
    let mut layout = Layout::from_config_json(&cfg_text)
        .map_err(|e| RunError::Refused(format!("qd-export's layout: {e}")))?;
    let vocab = require_identity_remap(&reader, &OsFiles, u64::from(snapshot.config.vocab))
        .map_err(RunError::Refused)?;
    layout.vocab = usize::try_from(vocab).map_err(|e| RunError::Refused(e.to_string()))?;
    let hidden = snapshot.config.hidden as usize;
    if layout.hidden != hidden {
        return refused(format!(
            "qd-export reads hidden {} but ojas-qwen35 reads {hidden}",
            layout.hidden
        ));
    }
    let (head, head_sha, head_content) =
        load_span_head_pinned(&a.head_init, hidden, &a.head_init_pin)
            .map_err(|e| RunError::Refused(format!("the head init: {e}")))?;
    let head_file = a
        .head_init
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| RunError::Refused(format!("{} has no file name", a.head_init.display())))?
        .to_owned();
    let schedule = match a.min_lr {
        Some(floor) => LrSchedule::real_ft_with_floor(a.lr, total_steps, floor),
        None => LrSchedule::real_ft(a.lr, total_steps),
    }
    .map_err(failed("the schedule"))?;
    let train_cfg = TrainConfig {
        schedule,
        cap: WallClockCap::new(a.cap_s).map_err(failed("the cap"))?,
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: a.lr,
            beta2: recipe::DEFAULT_BETA2,
            lower_layers: a.lower_layers,
        },
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        epochs: a.epochs,
        seed: a.seed,
        checkpoint: a.checkpoint_every.map(|every| CheckpointPolicy {
            dir: a.out.join("checkpoints"),
            every,
        }),
        eta: a.eta,
    };
    let resume = match &a.resume {
        Some(dir) => {
            let recipe_path = dir
                .parent()
                .map(|root| root.join(RECIPE_FILE))
                .ok_or_else(|| {
                    RunError::Refused(format!(
                        "--resume {} has no parent directory",
                        dir.display()
                    ))
                })?;
            let saved = match fs::read(&recipe_path) {
                Ok(bytes) => serde_json::from_slice(&bytes)
                    .map_err(|e| RunError::Refused(format!("{}: {e}", recipe_path.display())))?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
                Err(e) => return Err(RunError::Failed(format!("{}: {e}", recipe_path.display()))),
            };
            a.check_resume_recipe(&saved)
                .map_err(|e| RunError::Refused(format!("--resume {}: {e}", dir.display())))?;
            Some(
                ResumeState::read(dir)
                    .map_err(|e| RunError::Refused(format!("--resume {}: {e}", dir.display())))?,
            )
        }
        None => None,
    };
    if let Some(policy) = &train_cfg.checkpoint {
        // The recipe the checkpoints below were trained under, for a later `--resume` to check.
        fs::create_dir_all(&policy.dir)
            .map_err(|e| RunError::Failed(format!("{}: {e}", policy.dir.display())))?;
        let mut text =
            serde_json::to_vec_pretty(&a.checkpoint_recipe()).map_err(failed("the recipe"))?;
        text.push(b'\n');
        fs::write(policy.dir.join(RECIPE_FILE), text).map_err(|e| {
            RunError::Failed(format!("{}: {e}", policy.dir.join(RECIPE_FILE).display()))
        })?;
    }
    let row_id = ledger::uuid4().map_err(failed("the row id"))?;

    // ---- the GPU --------------------------------------------------------------------------
    let mut provider =
        Qwen35Provider::open(&snapshot, a.operands).map_err(failed("Qwen35Provider::open"))?;
    let provider_text = provider.describe();
    let mut objective: LetterSpanObjective<RealBatch, HostSpanHead> = LetterSpanObjective::new(
        HostSpanHead::new(head).map_err(failed("the head"))?,
        a.span_weight,
    )
    .map_err(failed("the objective"))?;
    // Epoch `e`: the first `steps` batches of the plan drawn at (--seed, e), one per step.
    let epoch_source = |epoch: u64| {
        let batches = reader
            .batches(a.batch_tokens, data_seed, epoch)
            .map_err(|e| TrainError::Io(format!("the batches of epoch {epoch}: {e}")))?
            .take(n)
            .map(|b| {
                b.map_err(|e| TrainError::Io(e.to_string()))
                    .and_then(RealBatch::new)
            });
        Ok(EpochSource {
            batches,
            count: a.steps,
        })
    };
    let mut on_progress = |p: &Progress| {
        eprintln!(
            "step {}/{} lr {:e} loss {:.6} norm {:.4} clip {:.4} elapsed {:.1}s",
            p.record.optimizer_step + 1,
            p.total_steps,
            p.record.lr,
            p.record.loss,
            p.record.grad_norm,
            p.record.clip_coefficient,
            p.elapsed_s
        );
    };
    let mut hooks = Hooks {
        on_progress: Some(&mut on_progress),
    };
    let clock = MonotonicClock::new();
    let result = trainer::train_epochs(
        &mut provider,
        &mut objective,
        epoch_source,
        &train_cfg,
        &clock,
        resume,
        &mut hooks,
    )
    .map_err(failed("training"))?;

    // ---- the export -----------------------------------------------------------------------
    let values = provider
        .read_parameters()
        .map_err(failed("reading the trained masters"))?;
    let specs: Vec<ParamSpec> = provider.parameters().to_vec();
    let tower: Vec<NamedTensor<'_>> = specs
        .iter()
        .zip(&values)
        .map(|(spec, v)| NamedTensor {
            spec: spec.clone(),
            values: v,
        })
        .collect();
    let head = objective.head();
    let head_tensors: Vec<NamedTensor<'_>> = head
        .entries()
        .into_iter()
        .zip(head.values())
        .map(|(spec, values)| NamedTensor { spec, values })
        .collect();
    let recipe = FtRecipe {
        trainer: METAL_TRAINER.into(),
        tag: export::EXPORT_TAG.into(),
        device: METAL_DEVICE.into(),
        lr: a.lr,
        passes: a.epochs,
        batches: a.steps,
        width,
        span_weight: a.span_weight,
        shard_hash: header.shard_hash().map_err(failed("shard_hash"))?,
        backbone_snapshot,
        backbone_vocab: vocab,
        backbone_params: snapshot.config.parameter_count(),
        attn_implementation: "sdpa".into(),
        optimizer_recipe: METAL_FULL_FT_OPTIMIZER_RECIPE.into(),
        wall_clock_cap_s: Some(a.cap_s),
        batch_tokens: Some(a.batch_tokens),
        no_memorise: true,
        beta2: None,
        lower_layers: a.lower_layers.map(|l| (l.n as u64, l.lr_scale)),
        provider: provider_text.clone(),
        operands: a.operands.as_str().into(),
        optimizer_groups: if a.lower_layers.is_some() {
            "two"
        } else {
            "single"
        }
        .into(),
        train_attention_mask: "padding".into(),
        deterministic: false,
        extra: a.recipe_extra(header.prompt_format),
    };
    let protocol = Protocol::for_recipe(
        &recipe,
        &header.data_snapshot_hash,
        &header.tokenizer_hash,
        a.seed,
    )
    .map_err(failed("the protocol"))?;
    let info = ManifestInfo {
        trainer: METAL_TRAINER.into(),
        device: METAL_DEVICE.into(),
        seed: a.seed,
        optimizer_step: result.optimizer_steps,
        ft_row_id: row_id.clone(),
        vocab_size: vocab,
        span_weight: a.span_weight,
        schedule: schedule.to_json().map_err(failed("the schedule's JSON"))?,
        provider: provider_text.clone(),
        operands: a.operands.as_str().into(),
        recipe_hash: protocol.recipe_hash.clone(),
        loss_log_digest: result
            .loss_log_digest()
            .map_err(failed("the loss log digest"))?,
        consumed_digest: result.consumed_digest.clone(),
        head_init_digest: Some(head_sha.clone()),
        prompt_format: header.prompt_format,
    };
    let scored = export::export(
        &a.out,
        &tower,
        &head_tensors,
        Some(&layout),
        &info,
        Precision::Bf16,
    )
    .map_err(failed("the bf16 export"))?;
    let masters = export::export(
        &a.out,
        &tower,
        &head_tensors,
        Some(&layout),
        &info,
        Precision::F32Masters,
    )
    .map_err(failed("the f32 masters export"))?;

    // ---- the row --------------------------------------------------------------------------
    let facts = RunFacts {
        cap_s: a.cap_s,
        provider: &provider_text,
        head_init: Some((&head_file, &head_sha)),
        eta: a.eta,
    };
    let mut metrics = ledger::ft_metrics(&result, &facts).map_err(failed("the metrics"))?;
    metrics.insert(
        "deterministic_kernels".into(),
        TriState::not_run("one run; repeat-run equality was not measured by this run"),
    );
    // Amendment 2 (iv): the digest a torch arm records for the head it built from seed 0, under
    // the same name, so gate 0 compares the two before anything else.
    metrics.insert(
        "train.span_head_init_digest".into(),
        TriState::ran(true, head_content.as_str()).with_detail(format!(
            "run_control._sidecar_digest over qd-tensor-ref-v1 of {head_file}'s span_head.* (float32), \
             recomputed at load and equal to the pinned value; file sha256 {head_sha}"
        )),
    );
    metrics.insert(
        "train.vocab_remap".into(),
        TriState::ran(true, "identity").with_detail(format!(
            "{} re-read through the door and every id maps to itself over {vocab} ids, the snapshot's \
             vocabulary: the tower is trained as loaded",
            reader.root().join("remap.npz").display()
        )),
    );
    let resumed = a.resume.as_ref().map_or_else(String::new, |d| {
        format!(
            " Resumed from checkpoint {} (recipe checked equal to its recipe.json).",
            d.display()
        )
    });
    let row = FtRow {
        protocol,
        recipe,
        status: Status::Completed,
        quick_reason: pin::mark_quick_reason(
            quick_reason(a.steps, a.epochs, epoch_batches, a.lower_layers),
            &pin,
        ),
        code_commit: ledger::git_commit(&a.repo),
        env: Environment::rust_metal(ledger::hostname()),
        metrics,
        wall_clock_s: result.wall_clock_s,
        wall_clock_source: WallClockSource::Recorder,
        notes: format!(
            "qd-train-metal. Batches: the first {} of reader.batches(batch_tokens={}, seed={data_seed}, \
             epoch=e) for each of {} epoch(s), the plan drawn at the run's --seed (batch_order seed), \
             one per optimizer step. \
             Schedule real_ft(lr, {}): warmup max(1, steps/20), min lr {}. Optimizer: AdamW betas \
             (0.9, 0.999), eps 1e-8, weight decay 0.01 on every tower and head entry (tessl applies it \
             as f32(0.01) = 0.00999999977648258, 2.2e-8 relative), clip 1.0 over tower and head \
             together; {}. Exports: {} (scored, bf16 tower) and {} (f32 masters, never scored).",
            a.steps,
            a.batch_tokens,
            a.epochs,
            total_steps,
            a.min_lr
                .map_or_else(|| "lr/10".to_owned(), |f| f.to_string()),
            match a.lower_layers {
                None => "one learning rate for every entry".to_owned(),
                Some(l) => format!(
                    "decoder layers below {} at {}x the schedule, the rest and the span head at 1x",
                    l.n, l.lr_scale
                ),
            },
            scored.weights.display(),
            masters.weights.display()
        ) + &resumed,
    };
    let written_at = ledger::isoformat_utc(SystemTime::now()).map_err(failed("written_at"))?;
    append_row(
        &a.ledger,
        &row,
        &row_extras(&manifest_sha256),
        row_id.clone(),
        written_at,
    )
    .map_err(failed("the ledger append"))?;
    Ok(RunSummary {
        row_id,
        termination: result.termination,
        optimizer_steps: result.optimizer_steps,
        export: scored,
        masters,
    })
}
