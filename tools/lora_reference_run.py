"""The LoRA-2B torch reference arm's driver (card ft-3b8126cd564d889af2b0169943020aa7).

Runs ``qd_train.lora``'s plain-torch LoRA arm over whichever train shard set it is handed
(``--out`` is required; no corpus is assumed), as the ojas leg will:
the same shard reader and batch order (``--batch-order seed``: the plan is drawn at the training
seed), the same supervision (``qd_train.trainer.ft_supervision``), the 17-row D17 readout and the
span pointer head from the pinned span-head init, the schedule ``qd_train.lora.lora_schedule``
(warmup ``max(1, steps // 20)``, cosine to 0), and AdamW at the pre-registered rates. It is the
reference and oracle for bound (d') of ``campaign/lora-2b-preregistered.json`` and nothing more:
every row it writes is ``quick`` and cannot promote (rule 8).

Two device/dtype pairs only, the two torch legs of the 0.8B smoke:

* ``--device mps --dtype bf16``: the frozen base resident in bf16, LoRA, D17 and span head in f32;
* ``--device cpu --dtype fp32``: the tight pin.

Anything else is refused before a path is opened.

What it writes, all of it only when the schedule ran to its last step:

* ``--loss-log``: one JSON line per optimizer step (``step``, ``lr``, ``loss_total``,
  ``loss_choice_17way``, ``loss_span``, ``loss_vocab_ce`` (report-only, no gradient), ``grad_norm``
  before clipping, ``batch_index``). It is written as ``<path>.partial`` and renamed on completion,
  so a killed or capped run leaves no file that looks finished.
* ``--checkpoint``: a ``run_control.Checkpoint`` whose ``model_state`` is the ``lora-merged`` layout
  ``tools/ckpt_average.py --from lora-merged`` reads.
* one ledger row (run kind ``ft``, ``quick``, ``optimizer_recipe "lora"``) through
  ``qd_train.ledger.RunRecorder``, carrying the seven digests of the pre-registration's
  ``init.record_in_every_row``. A run that raises or is terminated writes its row as failed or
  killed, by the recorder, and no checkpoint.

Rule 3: every path argument goes through ``qd_train.data_access.assert_path_not_held_out`` before
anything is opened, and the shard reader's own door (``assert_shard_trainable``) runs before a
token is read.

Run with the ml venv::

    /Users/bharath/.venvs/ml/bin/python tools/lora_reference_run.py --help
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Final

# Same two lines as tools/real_ft_run.py, for the same reasons (its comments carry them): the
# repo's python/ and tools/ on the path, and the MPS allocator's ceiling set before torch loads.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")

import torch

# Imported from tools/real_ft_run.py rather than copied (a refactor of that file is out of scope):
# plan_seed_for, plan_order_digest, read_span_head_init, load_span_head_init, span_head_digest,
# vocab_letter_ids, max_steps_reason, MAX_WALL_CLOCK_CAP_S are public; _protocol, _cost and
# _resolve_batch_tokens are its private helpers, imported anyway so the protocol hash, the cost
# estimate and the batch-token rule are the ones every other ft row used.
from real_ft_run import (
    BATCH_ORDER_SEED,
    MAX_WALL_CLOCK_CAP_S,
    REPO,
    _cost,
    _protocol,
    _resolve_batch_tokens,
    load_span_head_init,
    max_steps_reason,
    plan_order_digest,
    plan_seed_for,
    read_span_head_init,
    span_head_digest,
    vocab_letter_ids,
)
from real_tokenizer_pipeline import data_rules

from qd_data.config import DataConfig
from qd_data.schema import NOUL_LETTER, OPTION_LETTERS
from qd_train import lora
from qd_train.backbone import BackboneContractViolation, load_text_tower, remap_text_tower
from qd_train.data_access import _file_sha256, assert_path_not_held_out
from qd_train.heads import SpanPointerHead
from qd_train.ledger import (
    SHARD_NOT_COVERED_METRIC,
    Environment,
    Ledger,
    RunRecorder,
    shard_coverage_state,
)
from qd_train.memory import ADAMW_BF16, ADAMW_FP32, CheckpointRefused, spec_from_checkpoint
from qd_train.run_control import RunControl, WallClockCap
from qd_train.shards import ShardReader, assert_shard_files_trainable
from qd_train.trainer import train_ft
from qd_train.tristate import NotRun, Ran

TOOL: Final[str] = "tools/lora_reference_run.py"
#: The only pairs this driver runs: the 0.8B smoke's two torch legs (bound d').
DEVICE_DTYPES: Final[tuple[tuple[str, str], ...]] = (("mps", "bf16"), ("cpu", "fp32"))
#: Answer rows in D17 order: sixteen option letters, then noul (the ojas leg reads it).
ANSWER_LETTERS: Final[tuple[str, ...]] = (*OPTION_LETTERS, NOUL_LETTER)
#: The attention kernel, named as real_ft_run names it.
ATTN_IMPLEMENTATION: Final[str] = "sdpa"
#: The environment variable naming the ``qd-prereg-bind`` binary: an absolute path, never searched
#: for (the convention ``QD_PREP_BIN`` sets). ``LoraRunConfig.prereg_bind_bin`` beats it.
PREREG_BIND_ENV: Final[str] = "QD_PREREG_BIND_BIN"
#: The pre-registration every run of this driver is bound to.
PREREG_PATH: Final[Path] = REPO / "campaign" / "lora-2b-preregistered.json"
#: Seconds the binding check may take: it reads two small files and runs a few ``git`` commands.
PREREG_BIND_TIMEOUT_S: Final[float] = 120.0


class LoraDriverRefusal(ValueError):
    """The run was asked for something this driver refuses, before any work was paid for."""


class LegacyDataRulesRefusal(LoraDriverRefusal):
    """The config carries v5's data rules; the human refused v5 for LoRA-2B on 2026-10-17."""


class PreregBindRefusal(LoraDriverRefusal):
    """``qd-prereg-bind`` did not exit 0 with ``RESULT: PASS``: FAILED and NOT_RUN alike refuse
    step 0 (pre-registration amendment 3)."""


class RunIncomplete(RuntimeError):
    """The loop stopped before its schedule ended; its row is written, no checkpoint or log is."""


@dataclasses.dataclass(frozen=True)
class LoraRunConfig:
    """Everything a run needs, with no defaults for the facts that decide what it is."""

    out: Path
    backbone: Path
    tokenizer_json: Path
    span_head_init: Path
    span_head_init_manifest: Path
    #: The train manifest the shard set was written from. Its file's sha256 is hashed before and
    #: after the shard door reads it and recorded as the row's ``train_manifest_sha256``
    #: (qd-train-metal's W2-4 does the same), which the release export requires.
    manifest: Path
    ledger: Path
    loss_log: Path
    checkpoint: Path
    seed: int
    device: str
    dtype: str
    max_steps: int
    wall_clock_cap_s: float
    batch_tokens: int | None = None
    expect_rev: str | None = None
    expect_base_sha256: str | None = None
    tag: str = "lora-ref"
    #: The data rules the shard set is read under; ``None`` is ``DataConfig()`` (v6).
    data_config: DataConfig | None = None
    #: The ``qd-prereg-bind`` binary; ``None`` reads :data:`PREREG_BIND_ENV`.
    prereg_bind_bin: Path | None = None


@dataclasses.dataclass(frozen=True)
class LoraRunResult:
    row_id: str | None
    optimizer_steps: int
    termination: str
    checkpoint: Path
    loss_log: Path
    digests: dict[str, str | int]


def check_device_dtype(device: str, dtype: str) -> None:
    if (device, dtype) not in DEVICE_DTYPES:
        raise LoraDriverRefusal(
            f"--device {device!r} --dtype {dtype!r} is not one of {list(DEVICE_DTYPES)}: the "
            "reference arm's torch legs are MPS bf16 and the fp32 CPU pin, and nothing else"
        )


def check_bounds(cfg: LoraRunConfig) -> None:
    if isinstance(cfg.seed, bool) or cfg.seed < 0:
        raise LoraDriverRefusal(f"--seed must be a non-negative integer, got {cfg.seed!r}")
    if isinstance(cfg.max_steps, bool) or cfg.max_steps < 1:
        raise LoraDriverRefusal(f"--max-steps must be at least 1, got {cfg.max_steps!r}")
    if not 0.0 < cfg.wall_clock_cap_s <= MAX_WALL_CLOCK_CAP_S:
        raise LoraDriverRefusal(
            f"--wall-clock-cap-s must be in (0, {MAX_WALL_CLOCK_CAP_S}], got {cfg.wall_clock_cap_s}"
        )
    for what, path in (("--loss-log", cfg.loss_log), ("--checkpoint", cfg.checkpoint)):
        # lexists: a dangling symlink is not `exists()`, and `open('w')` would follow it and
        # create its target, a write outside every validated path.
        if os.path.lexists(path) or os.path.lexists(f"{path}.partial"):
            raise LoraDriverRefusal(f"{what} {path} already exists: refusing to overwrite a result")


def check_not_held_out(cfg: LoraRunConfig, config: DataConfig) -> None:
    """Rule 3 over every path the run reads or writes, before any of them is opened."""
    for _what, path in (
        ("--out", cfg.out),
        ("--out/shards/train", cfg.out / "shards" / "train"),
        ("--backbone", cfg.backbone),
        ("--tokenizer-json", cfg.tokenizer_json),
        ("--span-head-init", cfg.span_head_init),
        ("--span-head-init-manifest", cfg.span_head_init_manifest),
        ("--manifest", cfg.manifest),
        ("--ledger", cfg.ledger),
        ("--loss-log", cfg.loss_log),
        ("--checkpoint", cfg.checkpoint),
    ):
        assert_path_not_held_out(Path(path), config=config, repo_root=REPO)
    # Each shard FILE too, by the reader's own pre-read check (the one implementation; it reads
    # no token). A missing directory is the reader's refusal to make, later.
    shard_dir = cfg.out / "shards" / "train"
    if shard_dir.is_dir():
        assert_shard_files_trainable(shard_dir, config=config, repo_root=REPO)


def answer_ids(reader: ShardReader, tokenizer_json: Path) -> list[int]:
    """The 17 answer rows' post-remap token ids, in :data:`ANSWER_LETTERS` order."""
    letters = vocab_letter_ids(reader, tokenizer_json=tokenizer_json, letter_id={})
    missing = [letter for letter in ANSWER_LETTERS if letter not in letters]
    if missing:
        raise LoraDriverRefusal(
            f"the shard set's remap dropped answer letter(s) {missing}; D17 needs all "
            f"{len(ANSWER_LETTERS)} rows"
        )
    return [letters[letter] for letter in ANSWER_LETTERS]


def quick_reasons(cfg: LoraRunConfig, plan_batches: int) -> list[str]:
    reasons = [
        "one seed (rule 8)",
        "LoRA reference arm: a reference and oracle for bound d', never a promotion run",
        f"device {cfg.device!r} is not a campaign device: a smoke of the path",
    ]
    if cfg.max_steps < plan_batches:
        reasons.append(max_steps_reason(cfg.max_steps))
    return reasons


def _stat(text: str, value: str | int, detail: str | int) -> Ran:
    return Ran(passed=True, value=value, detail=f"{text}: {detail}")


def read_with_manifest_sha256(manifest: Path, read):
    """``(read(), sha256)``: ``read`` (the one thing that opens the train manifest) runs between
    two hashes of the manifest file's bytes, as ``data_access.open_training_data`` and
    qd-train-metal's ``read_with_manifest_sha256`` do. A file whose bytes changed in between is
    refused: the digest a row records must be the file ``read`` read."""
    try:
        before = _file_sha256(manifest)
        value = read()
        after = _file_sha256(manifest)
    except OSError as exc:
        raise LoraDriverRefusal(f"{manifest}: cannot hash the train manifest: {exc}") from exc
    if after != before:
        raise LoraDriverRefusal(
            f"{manifest}: the manifest's bytes changed between the hash and the read, refusing, "
            "so a recorded train_manifest_sha256 is the file read"
        )
    return value, before


def param_groups_summary(groups: Sequence[dict[str, object]]) -> str:
    """``lora.param_groups`` as Rust's ``lora_ft_row`` writes it: ``name xN at lr_scale S``,
    comma-separated, one per optimizer group."""
    return ", ".join(
        f"{g['name']} x{len(g['params'])} at lr_scale {g['lr_scale']!r}"  # type: ignore[arg-type]
        for g in groups
    )


def check_data_rules(config: DataConfig) -> DataConfig:
    """``config`` if it is already under v6's data rules, else a typed refusal (v5r is refused for
    LoRA-2B: human ruling 2026-10-17). ``data_rules`` is the one seam that says what v6 is."""
    v6 = data_rules(config, v6=True)
    if config != v6:
        raise LegacyDataRulesRefusal(
            "the data config is not under v6's data rules (v5's dedupe keep rule, no LSH "
            "prefilter, or benchmark evaluation splits trainable); v5 is refused for LoRA-2B, "
            "which is rebuilt on v6"
        )
    return v6


def _prereg_bind_binary(cfg: LoraRunConfig) -> Path:
    named = cfg.prereg_bind_bin if cfg.prereg_bind_bin is not None else os.environ.get(
        PREREG_BIND_ENV
    )
    if not named:
        raise PreregBindRefusal(
            f"NOT_RUN: {PREREG_BIND_ENV} is unset and no prereg_bind_bin was given, so the "
            "pre-registration binding was not checked"
        )
    binary = Path(named)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise PreregBindRefusal(f"NOT_RUN: {binary} is not an executable file")
    return binary


def check_prereg_bound(cfg: LoraRunConfig) -> None:
    """Run ``qd-prereg-bind`` over the pre-registration and the train shard set, before any tensor
    is read. Only exit 0 with a final ``RESULT: PASS`` line passes; FAILED (1), NOT_RUN (3), a bad
    command line (2), any other status, a missing binary and a timeout all refuse."""
    binary = _prereg_bind_binary(cfg)
    argv = [
        str(binary), "--prereg", str(PREREG_PATH),
        "--shard-dir", str(cfg.out / "shards" / "train"), "--repo", str(REPO),
    ]
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=PREREG_BIND_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise PreregBindRefusal(
            f"NOT_RUN: qd-prereg-bind timed out after {PREREG_BIND_TIMEOUT_S}s"
        ) from exc
    except OSError as exc:
        raise PreregBindRefusal(f"NOT_RUN: qd-prereg-bind could not be launched: {exc}") from exc
    lines = done.stdout.strip().splitlines()
    last = lines[-1].strip() if lines else ""
    if done.returncode == 0 and last == "RESULT: PASS":
        return
    verdict = {1: "FAILED", 3: "NOT_RUN", 2: "usage error"}.get(done.returncode, "unexpected exit")
    raise PreregBindRefusal(
        f"{verdict}: qd-prereg-bind exit {done.returncode}, last line {last!r} "
        f"(RESULT: PASS is required)\n{done.stdout.strip()}\n{done.stderr.strip()}"
    )


def validate_inputs(cfg: LoraRunConfig) -> DataConfig:
    """The single entry check, before any file is opened or tensor read: device/dtype pair,
    bounds, outputs not yet present, v6 data rules, rule 3 over every path, and the
    pre-registration binding (``qd-prereg-bind``, run as a subprocess, last because it is the only
    check that spawns a process). Returns the v6 ``DataConfig`` it checked with."""
    check_device_dtype(cfg.device, cfg.dtype)
    check_bounds(cfg)
    config = check_data_rules(cfg.data_config if cfg.data_config is not None else DataConfig())
    check_not_held_out(cfg, config)
    check_prereg_bound(cfg)
    return config


def run(cfg: LoraRunConfig) -> LoraRunResult:
    """Run the reference arm once. See the module docstring."""
    config = validate_inputs(cfg)
    if cfg.device == "cpu":
        # Bit-identical reruns on the CPU pin (the reference's whole point).
        torch.use_deterministic_algorithms(True)

    # --- data: the reader's own door runs before a token is read -------------------------------
    reader, manifest_sha256 = read_with_manifest_sha256(
        cfg.manifest,
        lambda: ShardReader(
            cfg.out / "shards" / "train", config=config, repo_root=cfg.out,
            expect_rev=cfg.expect_rev,
        ),
    )
    if reader.remap is None:
        raise LoraDriverRefusal(
            f"{reader.header.shard_hash()}: the shard set carries no remap table; training would "
            "index the wrong embedding row for every token"
        )
    widest = int(max(reader.header.buckets))
    batch_tokens, _ = _resolve_batch_tokens(cfg.batch_tokens, widest=widest)
    plan_seed = plan_seed_for(BATCH_ORDER_SEED, seed=cfg.seed, config=config)
    plan_batches = len(reader._plan(batch_tokens=batch_tokens, seed=plan_seed, epoch=0))
    if cfg.max_steps > plan_batches:
        raise LoraDriverRefusal(
            f"--max-steps {cfg.max_steps} is past the epoch's {plan_batches} batches"
        )
    order_digest = plan_order_digest(reader, batch_tokens=batch_tokens, plan_seed=plan_seed)
    batches = itertools.islice(
        reader.batches(batch_tokens=batch_tokens, seed=plan_seed, epoch=0), cfg.max_steps
    )
    init = read_span_head_init(cfg.span_head_init, cfg.span_head_init_manifest)

    # --- model: frozen base (stored-F32 tensors kept), remap, adapters --------------------------
    try:
        stored = lora.stored_dtypes(cfg.backbone)
    except (BackboneContractViolation, CheckpointRefused) as exc:
        raise LoraDriverRefusal(
            f"the snapshot's safetensors headers are unreadable: {exc}"
        ) from exc
    if cfg.expect_base_sha256 is not None and stored.safetensors_sha256 != cfg.expect_base_sha256:
        raise LoraDriverRefusal(
            f"the snapshot's safetensors sha256 is {stored.safetensors_sha256}, not the "
            f"pinned {cfg.expect_base_sha256}"
        )
    tower = load_text_tower(
        cfg.backbone,
        gradient_checkpointing=True,
        # Never built: the tower is loaded as a scoring tower (training=False), because the
        # footprint model prices a full fine-tune and this arm trains adapters.
        optimizer=ADAMW_BF16 if cfg.dtype == "bf16" else ADAMW_FP32,
        attn_implementation=ATTN_IMPLEMENTATION,
        device=cfg.device,
        dtype=cfg.dtype,
        rows=1,
        width=widest,
        spec=spec_from_checkpoint(cfg.backbone),
        keep_stored_dtype=stored.f32_names,
        training=False,
    )
    tower = remap_text_tower(tower, reader.remap)
    model = tower.model
    ids = answer_ids(reader, cfg.tokenizer_json)
    torch.manual_seed(cfg.seed)
    adapter = lora.attach_lora(model, seed=cfg.seed)
    a_digest = lora.lora_a_digest(model)
    rule_digest = lora.lora_a_rule_digest(lora.adapted_in_features(model), seed=cfg.seed)
    # The three pre-training digest checks are collected and raised once the recorder exists, so a
    # refusal after the expensive load leaves a failed row rather than nothing.
    drift: list[str] = []
    if a_digest != rule_digest:
        drift.append(
            f"the LoRA-A digest {a_digest} is not the pre-registered rule's {rule_digest} at seed "
            f"{cfg.seed}: the row would be void"
        )
    hidden = int(tower.hidden_size)
    span_head = SpanPointerHead(hidden).to(device=tower.device, dtype=torch.float32)
    head_digest = load_span_head_init(types.SimpleNamespace(span_head=span_head), init)
    if head_digest != span_head_digest(types.SimpleNamespace(span_head=span_head)):
        drift.append("the span head's digest moved between the load and the check")

    cfg.loss_log.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{cfg.loss_log}.partial")
    step = lora.LoraReferenceStep(
        model,
        answer_ids=ids,
        span_head=span_head,
        total_steps=cfg.max_steps,
        max_width=widest,
        vocab_size=tower.vocab_size,
        prompt_format=reader.header.prompt_format,
        log_path=partial,
    )
    d17_digest = lora.d17_init_digest(hidden)
    if lora._tensor_ref(step.delta.delta).digest() != d17_digest:
        drift.append("D17 did not start at the pre-registered all-zero float32 tensor")

    digests = {
        "span_head_init_content_digest": head_digest,
        "d17_init_digest": d17_digest,
        "lora_a_digest": a_digest,
        "base_snapshot": tower.snapshot.name,
        "base_safetensors_sha256": stored.safetensors_sha256,
        "dtype_map_sha256": stored.dtype_map_sha256,
        "plan_seed": plan_seed,
        "train_shard_hash": reader.header.shard_hash(),
        "train_manifest_sha256": manifest_sha256,
    }

    # --- ledger -------------------------------------------------------------------------------
    cap_s = float(cfg.wall_clock_cap_s)
    recipe: dict[str, object] = {
        "tool": TOOL,
        "tag": cfg.tag,
        "family": "lora-2b",
        "device": cfg.device,
        "dtype": cfg.dtype,
        "optimizer_recipe": "lora",
        "lora_r": adapter.r,
        "lora_alpha": adapter.alpha,
        "adapted_linears": len(adapter.module_names),
        "lr": lora.LORA_LR,
        "head_lr_scale": lora.HEAD_LR_SCALE,
        "weight_decay": lora.LORA_WEIGHT_DECAY,
        "max_grad_norm": step.max_grad_norm,
        "span_weight": step.span_weight,
        "objective": "choice_17way_over_answer_rows+span",
        "grad_accum": 1,
        "max_steps": cfg.max_steps,
        "batch_order": BATCH_ORDER_SEED,
        "batch_tokens": batch_tokens,
        "gradient_checkpointing": True,
        "shard_hash": reader.header.shard_hash(),
        "span_head_init": init.recipe(),
        "backbone_snapshot": tower.snapshot.name,
        "backbone_vocab": tower.vocab_size,
        "attn_implementation": tower.attn_implementation,
        "kept_stored_f32_tensors": len(stored.f32_names),
    }
    if reader.header.prompt_format != 1:
        recipe["prompt_format"] = reader.header.prompt_format
    cost = _cost(device=cfg.device, cap_s=cap_s)
    recorder = RunRecorder(
        Ledger(cfg.ledger),
        protocol=_protocol(reader=reader, seed=cfg.seed, recipe=recipe),
        run_kind="ft",
        repo=REPO,
        env=Environment.detect(device=cfg.device),
        wall_clock_s=None,
        cost=cost,
        recipe=recipe,
        quick=True,
        quick_reason="; ".join(quick_reasons(cfg, plan_batches)),
        train_manifest_sha256=manifest_sha256,
        notes=(
            f"{TOOL} [{cfg.tag}] -- the LoRA-2B torch reference arm (qd_train.lora): frozen "
            f"{cfg.dtype} base from {cfg.backbone.name}, f32 adapters, D17 and span head, on "
            f"{cfg.device}; batches 0..{cfg.max_steps - 1} of the epoch planned at seed "
            f"{plan_seed}."
        ),
        entry_point=Path(__file__),
    )
    formulas = {
        # The names below are qd_train::ledger::lora_ft_row's (Fable's ruling, 2026-10-21: the Rust
        # set is canonical), checked by ledger.LedgerRow.check_optimizer_label.
        "train.span_head_init_digest": ("run_control._sidecar_digest over span_head.* TensorRefs",
                                        head_digest),
        "lora.d17_init_digest": ("TensorRef.digest of float32 zeros [17, hidden]", d17_digest),
        "lora.a_init_digest": ("lora.lora_a_digest; equals lora.lora_a_rule_digest at the seed",
                               a_digest),
        "lora.base_safetensors_sha256": ("sha256 of the snapshot's safetensors file",
                                         stored.safetensors_sha256),
        "lora.dtype_map_sha256": ("lora.StoredDtypes.dtype_map_sha256", stored.dtype_map_sha256),
        "corpus.plan_seed": ("the seed the plan was drawn at (--batch-order seed)", plan_seed),
        "lora.param_groups": (
            "lora.lora_param_groups, one group per optimizer group", param_groups_summary(
                lora.lora_param_groups(model, delta=step.delta, span_head=span_head)
            ),
        ),
        "init.base_snapshot": ("the snapshot directory's name", tower.snapshot.name),
        "init.f32_names_sha256": ("lora.StoredDtypes.f32_names_sha256", stored.f32_names_sha256),
        "corpus.plan_order_digest": ("real_ft_run.plan_order_digest", order_digest),
        "corpus.train_shard_hash": ("ShardHeader.shard_hash()", reader.header.shard_hash()),
    }
    for name, (what, value) in formulas.items():
        recorder.metric(name, _stat(what, value, value))
    # What the train set's fingerprint cannot vouch for (`ledger._shard_not_covered_refusals`):
    # this row is quick today, and does not rely on that to stay unpromotable.
    recorder.metric(SHARD_NOT_COVERED_METRIC, shard_coverage_state(reader.checks))
    recorder.metric(
        "device_budget",
        NotRun(
            reason="memory.py's footprint prices a full fine-tune; the LoRA step's memory row is "
            "ojas's (ft-2bbced18 criterion 7), so no budget was applied"
        ),
    )
    if drift:
        step.close_log()
        partial.unlink(missing_ok=True)
        with recorder:  # writes the started row and a failed row; nothing was trained
            raise LoraDriverRefusal("; ".join(drift))
    control = RunControl(
        schedule=lora.lora_schedule(cfg.max_steps), cap=WallClockCap(cap_s=cap_s), cost=cost,
        grad_accum=1, checkpoint_every=0,
    )
    try:
        result = train_ft(batches, epoch=0, step=step, control=control, recorder=recorder)
    finally:
        step.close_log()
    if result.termination != "steps_exhausted":
        raise RunIncomplete(
            f"stopped by {result.termination!r} after {result.optimizer_steps} of "
            f"{cfg.max_steps} steps (row {result.row_id}); no checkpoint and no finished log"
        )
    # The log is finished first: a checkpoint must never exist beside a log still named .partial
    # (ckpt_average would read it as a complete run). If the checkpoint then fails to write, the
    # log goes back to .partial, so nothing that looks finished is left.
    partial.replace(cfg.loss_log)
    try:
        result.checkpoint.write(cfg.checkpoint)
    except BaseException:
        cfg.loss_log.replace(partial)
        raise
    return LoraRunResult(
        row_id=result.row_id,
        optimizer_steps=result.optimizer_steps,
        termination=result.termination,
        checkpoint=cfg.checkpoint,
        loss_log=cfg.loss_log,
        digests=digests,
    )


def parse_args(argv: Sequence[str] | None) -> LoraRunConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True,
                   help="the corpus dir: the train shard set is OUT/shards/train (as real_ft_run)")
    p.add_argument("--backbone", type=Path, required=True, help="the Qwen3.5 snapshot directory")
    p.add_argument("--tokenizer-json", type=Path, required=True)
    p.add_argument("--span-head-init", type=Path, required=True)
    p.add_argument("--span-head-init-manifest", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True,
                   help="the train manifest the shard set was written from; its file sha256 is "
                        "recorded as the row's train_manifest_sha256")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--loss-log", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--wall-clock-cap-s", type=float, required=True)
    p.add_argument("--batch-tokens", type=int, default=None)
    p.add_argument("--expect-rev", default=None)
    p.add_argument("--expect-base-sha256", default=None)
    p.add_argument("--tag", default="lora-ref")
    p.add_argument("--prereg-bind-bin", type=Path, default=None,
                   help=f"the qd-prereg-bind binary (default: ${PREREG_BIND_ENV})")
    a = p.parse_args(argv)
    return LoraRunConfig(
        out=a.out, backbone=a.backbone, tokenizer_json=a.tokenizer_json,
        span_head_init=a.span_head_init, span_head_init_manifest=a.span_head_init_manifest,
        manifest=a.manifest, ledger=a.ledger, loss_log=a.loss_log, checkpoint=a.checkpoint,
        seed=a.seed,
        device=a.device, dtype=a.dtype, max_steps=a.max_steps,
        wall_clock_cap_s=a.wall_clock_cap_s, batch_tokens=a.batch_tokens,
        expect_rev=a.expect_rev, expect_base_sha256=a.expect_base_sha256, tag=a.tag,
        prereg_bind_bin=a.prereg_bind_bin,
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    result = run(cfg)
    print(json.dumps({
        "row_id": result.row_id, "optimizer_steps": result.optimizer_steps,
        "termination": result.termination, "checkpoint": str(result.checkpoint),
        "loss_log": str(result.loss_log), "digests": result.digests,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
