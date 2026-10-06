"""``tools/real_ft_run.py``: the additive flags rung (d)'s GH200 torch reference arms need.

Rung (d) (``AUDIT/ojas-training-2026-10-01/fable-rung-d-resize.md``) compares the Rust trainer
over tessl with two PyTorch arms on F's recipe: T-bf16 (F's dtype) and T-fp32 (the tight
reference). Fable's idle-GPU ruling (``AUDIT/idle-gpu-queue-2026-10-02/fable-idle-queue.md``,
Q1 item 2) names what blocked them: ``--max-steps``, ``--train-dtype fp32`` on the non-master
AdamW path, and the span-head init digest (Amendment 2 (iv), ``fable-advice.md`` Q3), with the
pre-registered ``--span-head-init PATH`` fallback.

What is pinned here:

* every new flag lands in the recipe -- and so in ``recipe_hash`` and every row scored after
  it -- only when on, and a run with all of them off writes the rows it wrote before
  (``test_with_every_new_flag_off...``: an opt-in differential against the base commit);
* ``--max-steps N`` trains the first N batches of the epoch arm's own order, under the
  recipe's schedule formula recomputed at N steps, and every row it writes -- or that is later
  scored from it -- is quick (rule 8);
* ``--train-dtype fp32`` builds ``torch.optim.AdamW`` with the master path's eps and weight
  decay, read off the optimizer's own groups, and refuses ``--train-attention-mask none``,
  which on CUDA has no fp32 kernel;
* the span head's init digest is the manifest's formula, ``81bd953e...`` for the tiny h64
  head at seed 0 in-tree, ``1c30346c...`` for the 2B's h2048 head (opt-in).

Torch-gated, like ``test_real_ft_protocol.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import inspect
import io
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
import test_backbone as tb  # noqa: E402
from test_real_ft_shuffled_label import (  # noqa: E402
    REV,
    _defect_rows,
    _offsets,
    _remap,
    _tokenize,
)

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.defect_class import DEFECT_FAMILY_ID, DEFECT_SOURCE_ID  # noqa: E402
from qd_data.manifest import build_manifests  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_data.split import HELD_OUT, split  # noqa: E402
from qd_train.ledger import Ledger, LedgerRow  # noqa: E402
from qd_train.optim import MasterWeightAdamW  # noqa: E402
from qd_train.run_control import LRSchedule, _sidecar_digest  # noqa: E402
from qd_train.shards import write_shards  # noqa: E402

FIXTURES = REPO / "crates" / "qd-train" / "tests" / "fixtures"
H64_FILE = FIXTURES / "span-head-init-seed0-h64.safetensors"
H64_MANIFEST = FIXTURES / "span-head-init-seed0-h64.manifest.json"
H2048_MANIFEST = FIXTURES / "span-head-init-seed0.manifest.json"
#: The content digests L-oracle's manifests record, restated so a manifest edited to match a
#: drifted head cannot pass this file.
H64_DIGEST = "81bd953e1f27e8e2b703264df663f15f239327c9523d3e85973687009db471d6"
H2048_DIGEST = "1c30346c9ea8c1fe12cbc41d556587187ad5a27fe165ac7bc9c69d6bcd7def68"
#: The opt-in h2048 test reads the file L-oracle wrote in the main checkout (ignored, 33.6 MB).
H2048_ENV = "QD_SPAN_HEAD_INIT_H2048"
#: The opt-in byte-identity differential compares against the tool at this git ref.
BASE_ENV = "QD_BYTE_IDENTITY_BASE"
TINY_VOCAB = 256  # the corpus's identity remap is 256 byte ids wide
TINY_HIDDEN = 64  # the width of L-oracle's tracked h64 span-head fixture


# --- the toy corpus and the two backbones it trains ------------------------------------------


@dataclasses.dataclass(frozen=True)
class Corpus:
    out: Path
    train: list[DataRow]
    val: list[DataRow]
    snapshot: Path


def _write_corpus(out: Path, n_repos: int) -> tuple[list[DataRow], list[DataRow]]:
    """Train and val code.defect_class shard sets over the byte tokenizer, as the shuffled-
    label fixture writes them, at ``n_repos`` repositories (four rows each)."""
    config = DataConfig()
    mixture = build_mixture(
        {DEFECT_SOURCE_ID: _defect_rows(n_repos)}, config=config, families=[DEFECT_FAMILY_ID]
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        path = out / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    train = list(split_report.rows_by_split["train"])
    val = list(split_report.rows_by_split["val"])
    assert train and val
    for name, rows in (("train", train), ("val", val)):
        write_shards(
            paths[name], rows, out_dir=out / "shards" / name, remap=_remap(),
            tokenize=_tokenize, token_offsets=_offsets, config=config, repo_root=out,
            corpus_rev=REV,
        )
    return train, val


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    out = tmp_path_factory.mktemp("rungd-corpus")
    train, val = _write_corpus(out, 48)
    snapshot = out / "tiny-snapshot-h64"
    tb._write_tiny_snapshot(snapshot, vocab=TINY_VOCAB, hidden=TINY_HIDDEN)
    return Corpus(out=out, train=train, val=val, snapshot=snapshot)


def _patch_module(mp: pytest.MonkeyPatch, module: Any, corpus: Corpus) -> None:
    """What ``main`` reads from outside the shard set, answered for this corpus -- the
    shuffled-label fixture's ``_patch``, for any copy of the tool (the base commit's too)."""
    rows = (corpus.train, corpus.val)
    mp.setattr(module, "ft_split_rows", lambda **kw: rows)
    mp.setattr(module, "resolve_rev", lambda repo, rev: rev)
    mp.setattr(module, "check_defect_source", lambda out, *, defect_class: None)
    mp.setattr(
        module, "corpus_facts",
        lambda out, **kw: module.CorpusFacts(snapshot_not_run=None, history_rows={}),
    )
    mp.setattr(
        module, "_probe_one",
        lambda device, rows, width, hidden, heads: {
            "device": device, "rows": rows, "width": width, "ok": True, "wall_s": 0.0,
            "why": "stubbed in-process",
        },
    )


def _tiny_backbone(mp: pytest.MonkeyPatch, hidden: int = TINY_HIDDEN) -> None:
    """``_real_step`` loads with the 2B ModelSpec, which refuses a tiny tower; the spec is the
    only thing replaced (as ``tiny_master_checkpoints`` does) -- the remap, the step, the
    optimizer and the loop all run as written."""
    import qd_train.backbone as backbone

    real_load = backbone.load_text_tower
    spec = tb._tiny_spec(TINY_VOCAB, hidden)
    mp.setattr(backbone, "load_text_tower", lambda *a, **k: real_load(*a, **{**k, "spec": spec}))


@contextlib.contextmanager
def _deterministic_restored() -> Iterator[None]:
    """``--deterministic`` calls ``torch.use_deterministic_algorithms(True)``, which is process-
    global; put back whatever was in force so no later test inherits it."""
    before = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before)


#: The one claim of the epoch arm a run of a few steps cannot be held to: it compares the
#: letter loss of the first batch with the last, which over 1-3 different batches is a
#: statement about the batches, not about training.
_SHORT_RUN_CLAIM = "the letter loss did not fall"


def _run(
    corpus: Corpus, ledger: Path, *extra: str, real: bool = False, module: Any = rft,
    hidden: int = TINY_HIDDEN, short: bool = False,
) -> list[LedgerRow]:
    """One epoch arm at seed 0 on CPU, rows appended to ``ledger``. Exit 0, or -- for a
    ``short`` run -- exit 1 whose only failed claim is :data:`_SHORT_RUN_CLAIM`."""
    argv = [
        "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
        "--epoch", "--no-memorise", "--ledger", str(ledger),
        *(["--real-backbone", str(corpus.snapshot)] if real else []),
        *extra,
    ]
    out = io.StringIO()
    with (
        pytest.MonkeyPatch.context() as mp,
        _deterministic_restored(),
        contextlib.redirect_stdout(out),
    ):
        _patch_module(mp, module, corpus)
        if real:
            _tiny_backbone(mp, hidden)
        code = module.main(argv)
    text = out.getvalue()
    print(text)
    failed = [line for line in text.splitlines() if line.startswith("  - ")]
    if short and code == 1:
        assert failed and all(_SHORT_RUN_CLAIM in line for line in failed), failed
    else:
        assert code == 0, failed
    return Ledger(ledger).rows()


def _only(rows: list[LedgerRow], kind: str) -> LedgerRow:
    found = [r for r in rows if r.run_kind == kind]
    assert len(found) == 1, [r.run_kind for r in rows]
    return found[0]


def _value(row: LedgerRow, name: str) -> Any:
    return row.metrics[name].to_json()["value"]


# --- 1. the recipe: each new piece only when on -----------------------------------------------

_OFF = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
        "permutation": None, "replay": None}


def test_the_new_pieces_add_no_recipe_key_when_off():
    """So a run that uses none of them hashes exactly as every row before them did."""
    assert rft._recipe_pieces(**_OFF, max_steps=None, train_dtype="bf16", span_head_init=None) == {}


def test_each_new_piece_lands_in_the_recipe_under_a_mirrored_key_when_on():
    init = {"sha256": "a" * 64, "content_digest": "b" * 64}
    pieces = rft._recipe_pieces(**_OFF, max_steps=200, train_dtype="fp32", span_head_init=init)
    assert pieces == {"max_steps": 200, "train_dtype": "fp32", "span_head_init": init}
    # Mirrored into the verdict, eval and score rows (test_real_ft_pieces pins the mechanism).
    assert sorted(set(pieces) - set(rft.RECIPE_PIECE_KEYS)) == []


def test_an_unknown_train_dtype_is_refused_by_the_recipe():
    with pytest.raises(ValueError, match="train_dtype"):
        rft._recipe_pieces(**_OFF, train_dtype="fp16")


# --- 2. rule 8: --max-steps is a truncated schedule wherever its model goes --------------------

_CLEAN = rft.CorpusFacts(snapshot_not_run=None, history_rows={})


def _campaign_reasons(**kw: Any) -> list[str]:
    return rft.quick_reasons(tag="epoch", device="cuda", real_backbone=True, corpus=_CLEAN, **kw)


def test_a_max_steps_run_is_quick_for_a_truncated_schedule():
    """Rule 8. The loop ends ``steps_exhausted`` -- the schedule IS N steps long -- so the
    termination cannot say it was truncated; the flag does."""
    assert _campaign_reasons() == []
    assert _campaign_reasons(max_steps=None) == []
    reasons = _campaign_reasons(max_steps=200, termination="steps_exhausted")
    assert reasons == [rft.max_steps_reason(200)]
    assert "truncated schedule (--max-steps 200)" in reasons[0]


def _single_seed_model(recipe: dict[str, Any]) -> Any:
    ft = {
        "row_id": "f" * 8 + "-0000-4000-8000-000000000000",
        "metrics": {"train.termination": {"value": "steps_exhausted"}},
        "recipe": recipe,
    }
    args = SimpleNamespace(score_checkpoint=Path("epoch-seed0-cuda.json"))
    return rft.scored_model(args, ft, {"sidecar": {"digest": "d" * 64}})


def _model_reasons(model: Any) -> list[str]:
    return rft.model_reasons(
        model, device="cuda",
        reasons_for=lambda tag, device, termination=None: rft.quick_reasons(
            tag=tag, device=device, real_backbone=True, corpus=_CLEAN, termination=termination,
        ),
    )


def test_a_checkpoint_scored_later_from_a_max_steps_ft_row_is_quick():
    """The torch ``.json`` route reads only the ft row's termination, which a truncated
    schedule leaves at ``steps_exhausted``: without this, T-fp32's val row scored on the
    GH200 (criterion 6) would be quick=False for a model 2% trained."""
    assert _model_reasons(_single_seed_model({"lr": 1e-5})) == []
    assert _model_reasons(_single_seed_model({"lr": 1e-5, "max_steps": 200})) == [
        rft.max_steps_reason(200)
    ]


@pytest.mark.parametrize("bad", [True, "200", 0, -3, 2.0])
def test_a_malformed_max_steps_on_an_ft_row_is_refused_not_ignored(bad):
    with pytest.raises(SystemExit, match="max_steps"):
        _single_seed_model({"max_steps": bad})


# --- 3. argv refusals, before the shard set is read -------------------------------------------


def _tripwire(*args: object, **kwargs: object) -> None:
    raise AssertionError("an argv refusal read the shard set first")


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--epoch", "--max-steps", "1"], "--max-steps must be at least 2"),
        (["--max-steps", "5"], "--max-steps truncates the epoch arm"),
        (["--epoch", "--max-steps", "5", "--score-val", "--score-checkpoint", "c.json",
          "--ft-ledger", "l.jsonl", "--ft-row-id", "abcdefgh"], "trains nothing"),
        (["--epoch", "--no-memorise", "--score-val", "--max-steps", "5",
          "--shuffled-label", "abcdefgh"], "--shuffled-label"),
        (["--epoch", "--train-dtype", "fp32"], "--train-dtype fp32 needs --real-backbone"),
        (["--epoch", "--real-backbone", "snap", "--train-dtype", "fp32",
          "--train-attention-mask", "none"], "no fp32 kernel"),
        (["--epoch", "--real-backbone", "snap", "--optimizer", "master", "--fused-adamw",
          "--train-dtype", "fp32"], "--fused-adamw"),
        (["--epoch", "--span-head-init", str(H64_FILE)], "--span-head-init-manifest"),
        (["--epoch", "--span-head-init-manifest", str(H64_MANIFEST)], "without --span-head-init"),
        (["--epoch", "--span-head-init", str(H64_FILE), "--span-head-init-manifest",
          str(H64_MANIFEST)], "needs --real-backbone"),
        (["--epoch", "--span-head-init-digest"], "needs --real-backbone"),
    ],
)
def test_a_rungd_flag_that_would_determine_nothing_is_refused_at_argv_time(
    flags, match, tmp_path, monkeypatch
):
    monkeypatch.setattr(rft, "ShardReader", _tripwire)
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), "--rev", REV, "--seeds", "0", "--devices", "cpu",
                  *flags])


# --- 4. --max-steps end to end on the CPU stand-in ----------------------------------------------


def test_max_steps_trains_the_first_n_batches_under_a_schedule_recomputed_at_n(
    corpus, tmp_path, monkeypatch
):
    """The epoch arm's plan sliced to its first N batches, before ``steps = len(plan)``, so the
    schedule, the loop and the row all see N; the run and its eval row are quick (rule 8)."""
    seen: list[Any] = []
    consumed: list[Any] = []
    real_control, real_train_ft = rft._control, rft.train_ft

    def spy(steps: int, **kw: Any) -> Any:
        control = real_control(steps, **kw)
        seen.append(control)
        return control

    def tap(batches: Any, **kw: Any) -> Any:
        def through() -> Iterator[Any]:
            for batch in batches:
                consumed.append(batch.tokens.copy())
                yield batch

        return real_train_ft(through(), **kw)

    monkeypatch.setattr(rft, "_control", spy)
    monkeypatch.setattr(rft, "train_ft", tap)
    full = _only(_run(corpus, tmp_path / "full.jsonl"), "ft")
    n_batches = int(_value(full, "train.optimizer_steps"))
    assert n_batches > 3
    full_order = list(consumed)
    seen.clear()
    consumed.clear()

    rows = _run(corpus, tmp_path / "cut.jsonl", "--max-steps", "3", "--score-val", short=True)
    ft, ev = _only(rows, "ft"), _only(rows, "eval")
    # The first three batches of the epoch arm's own order, and nothing after them.
    assert len(consumed) == 3
    assert all(
        (a == b).all() and a.shape == b.shape
        for a, b in zip(consumed, full_order[:3], strict=True)
    )
    assert _value(ft, "train.optimizer_steps") == 3
    # The existing vocabulary: the schedule of 3 steps ran out, which is what happened.
    assert _value(ft, "train.termination") == "steps_exhausted"
    assert _value(ft, "corpus.plan_batches") == 3
    assert ft.recipe["max_steps"] == 3 and ft.recipe["batches"] == 3
    lr = float(ft.recipe["lr"])
    assert [c.schedule for c in seen] == [
        LRSchedule(peak_lr=lr, total_steps=3, warmup_steps=1, min_lr=lr / 10)
    ]
    assert ft.quick and rft.max_steps_reason(3) in str(ft.quick_reason)
    assert ev.quick and rft.max_steps_reason(3) in str(ev.quick_reason)
    assert ev.recipe["max_steps"] == 3
    assert "max_steps" not in full.recipe


def test_rungd_schedule_is_the_recipes_formula_at_200_steps():
    """Fable's design: LRSchedule(peak_lr=1e-5, total_steps=200, warmup_steps=10, min_lr=1e-6),
    the min-lr formula lr/10 per GAP-LTRAINER-RUNGD-MIN-LR-FORMULA-VS-LITERAL-2026-10-01."""
    control = rft._control(200, device="cpu", lr=1e-5, checkpoint_every=0)
    assert control.schedule == LRSchedule(
        peak_lr=1e-5, total_steps=200, warmup_steps=10, min_lr=1e-5 / 10
    )


def test_max_steps_at_or_past_the_epoch_is_not_a_truncation_and_is_refused(corpus, tmp_path):
    ledger = tmp_path / "l.jsonl"
    with pytest.raises(ValueError, match="not a truncation"):
        _run(corpus, ledger, "--max-steps", "100000")
    assert not ledger.exists() or Ledger(ledger).rows() == []


# --- 5. --train-dtype fp32 on the real (tiny) tower ---------------------------------------------


def _master_eps_wd() -> tuple[float, float]:
    sig = inspect.signature(MasterWeightAdamW.__init__)
    return sig.parameters["eps"].default, sig.parameters["weight_decay"].default


def test_the_master_paths_eps_and_weight_decay_are_the_ones_f_ran():
    """F's MasterWeightAdamW takes eps and weight_decay from its own defaults
    (build_optimizer passes neither): the values T-fp32 must match."""
    assert _master_eps_wd() == (1e-8, 0.01)


def test_the_fp32_adamw_check_reads_the_optimizers_own_groups():
    p = [torch.nn.Parameter(torch.zeros(3))]
    assert rft.fp32_adamw_problems(torch.optim.AdamW(p, lr=1e-5), beta2=0.999) == []
    assert any(
        "eps" in x for x in rft.fp32_adamw_problems(
            torch.optim.AdamW(p, lr=1e-5, eps=1e-6), beta2=0.999
        )
    )
    assert any(
        "weight_decay" in x for x in rft.fp32_adamw_problems(
            torch.optim.AdamW(p, lr=1e-5, weight_decay=0.0), beta2=0.999
        )
    )
    assert any(
        "betas" in x for x in rft.fp32_adamw_problems(
            torch.optim.AdamW(p, lr=1e-5), beta2=0.95
        )
    )
    assert any(
        "torch.optim.AdamW" in x
        for x in rft.fp32_adamw_problems(torch.optim.SGD(p, lr=1e-5), beta2=0.999)
    )


def test_train_dtype_fp32_trains_an_fp32_tower_on_plain_adamw(corpus, tmp_path):
    rows = _run(
        corpus, tmp_path / "fp32.jsonl", "--optimizer", "master", "--lower-layers-n", "1",
        "--train-dtype", "fp32", "--max-steps", "2", real=True, short=True,
    )
    ft = _only(rows, "ft")
    assert ft.status == "completed"
    assert ft.recipe["train_dtype"] == "fp32" and ft.recipe["max_steps"] == 2
    # F's recipe name stays; optimizer_spec("fp32", "master") is ADAMW_FP32, as scoring a
    # master checkpoint at --score-dtype fp32 has always resolved it.
    assert ft.recipe["optimizer_recipe"] == "master"
    path = json.loads(_value(ft, "train.path"))
    assert path["optimizer"] == "AdamW"
    assert "dtype=fp32." in ft.notes
    assert _value(ft, "train.optimizer_steps") == 2


def test_optimizer_kahan_trains_the_bf16_tower_on_the_compensated_recipe(corpus, tmp_path):
    """``--optimizer kahan`` end to end: the recipe names it, the step builds
    KahanBf16AdamW over a bf16 tower, and the row is quick (``--max-steps``)."""
    rows = _run(
        corpus, tmp_path / "kahan.jsonl", "--optimizer", "kahan", "--max-steps", "2",
        real=True, short=True,
    )
    ft = _only(rows, "ft")
    assert ft.status == "completed"
    assert ft.recipe["optimizer_recipe"] == "kahan"
    path = json.loads(_value(ft, "train.path"))
    assert path["optimizer"] == "KahanBf16AdamW"
    assert _value(ft, "train.optimizer_steps") == 2


def test_train_dtype_fp32_refuses_an_optimizer_whose_groups_drift(corpus, tmp_path, monkeypatch):
    """The eps/weight-decay check runs on what was BUILT, so a builder that changed them is
    refused before step 0 -- not discovered as a trajectory that does not track."""
    import qd_train.backbone as backbone

    real_build = backbone.build_optimizer

    def drifted(params: Any, **kw: Any) -> Any:
        opt = real_build(params, **kw)
        for g in opt.param_groups:
            g["eps"] = 1e-6
        return opt

    monkeypatch.setattr(backbone, "build_optimizer", drifted)
    ledger = tmp_path / "l.jsonl"
    with pytest.raises(SystemExit, match="eps"):
        _run(corpus, ledger, "--train-dtype", "fp32", "--max-steps", "2", real=True)
    assert not ledger.exists() or Ledger(ledger).rows() == []


# --- 6. the span head's init digest and the --span-head-init fallback ---------------------------


def _step_for(snapshot: Path, *, seed: int, dtype: str = "bf16", hidden: int = TINY_HIDDEN) -> Any:
    """A real QwenDecisionStep over the tiny tower, as ``_real_step`` builds one."""
    from qd_train.backbone import QwenDecisionStep, load_text_tower

    optimizer = rft.optimizer_spec(dtype, "master")
    tower = load_text_tower(
        snapshot, gradient_checkpointing=True, optimizer=optimizer, attn_implementation="sdpa",
        dtype=dtype, rows=1, width=64, spec=tb._tiny_spec(TINY_VOCAB, hidden),
    )
    return QwenDecisionStep(tower, seed=seed, lr=1e-5, total_steps=2, max_width=64)


def test_the_head_digest_is_the_checkpoints_own_formula(corpus):
    """One formula for the manifest, the Rust trainer and the arm: ``_sidecar_digest`` over
    ``state()['span_head']`` named ``span_head.<k>`` -- pinned here so the head-only helper
    (which skips copying the whole tower to host at init) cannot drift from it."""
    step = _step_for(corpus.snapshot, seed=3)
    from_state = _sidecar_digest(
        {f"span_head.{k}": v for k, v in step.state()["span_head"].items()}
    )
    assert rft.span_head_digest(step) == from_state


def test_the_tiny_h64_head_at_seed_0_is_the_tracked_fixtures(corpus):
    assert rft.span_head_digest(_step_for(corpus.snapshot, seed=0)) == H64_DIGEST
    assert json.loads(H64_MANIFEST.read_text())["content_digest"]["value"] == H64_DIGEST


def test_the_digest_flag_records_the_init_digest_on_the_ft_row_only_when_on(corpus, tmp_path):
    on = _only(_run(corpus, tmp_path / "on.jsonl", "--span-head-init-digest", "--max-steps", "2",
                    real=True, short=True), "ft")
    assert _value(on, "train.span_head_init_digest") == H64_DIGEST
    # A recording option, not a recipe input: it moves no hash.
    assert "span_head_init" not in on.recipe
    off = _only(
        _run(corpus, tmp_path / "off.jsonl", "--max-steps", "2", real=True, short=True), "ft"
    )
    assert "train.span_head_init_digest" not in off.metrics
    assert on.protocol.recipe_hash == off.protocol.recipe_hash


def test_span_head_init_loads_the_file_into_the_head_the_optimizer_steps(corpus):
    """Loaded in place after the optimizer exists: the head's fp32 parameters are their own
    masters (optim.py), so the copy is what the first step updates."""
    init = rft.read_span_head_init(H64_FILE, H64_MANIFEST)
    step = _step_for(corpus.snapshot, seed=1)
    assert rft.span_head_digest(step) != H64_DIGEST
    head_params = {id(p) for p in step.span_head.parameters()}
    assert rft.load_span_head_init(step, init) == H64_DIGEST
    assert rft.span_head_digest(step) == H64_DIGEST
    masters = [m for m in step.optimizer._masters if id(m) in head_params]
    assert len(masters) == len(head_params)
    want = {f"span_head.{k}": v for k, v in step.span_head.state_dict().items()}
    from safetensors.torch import load_file

    for name, t in load_file(str(H64_FILE)).items():
        assert torch.equal(want[name], t)


def test_span_head_init_end_to_end_records_the_file_in_the_recipe(corpus, tmp_path):
    ft = _only(_run(
        corpus, tmp_path / "init.jsonl", "--span-head-init", str(H64_FILE),
        "--span-head-init-manifest", str(H64_MANIFEST), "--max-steps", "2", real=True, short=True,
    ), "ft")
    manifest = json.loads(H64_MANIFEST.read_text())
    assert ft.recipe["span_head_init"] == {
        "sha256": manifest["file"]["sha256"], "content_digest": H64_DIGEST,
    }
    assert _value(ft, "train.span_head_init_digest") == H64_DIGEST


def _tampered(tmp_path: Path, *, file: bytes | None = None,
              edit: Any = None) -> tuple[Path, Path]:
    path = tmp_path / "head.safetensors"
    path.write_bytes(H64_FILE.read_bytes() if file is None else file)
    manifest = json.loads(H64_MANIFEST.read_text())
    if edit is not None:
        edit(manifest)
    mpath = tmp_path / "head.manifest.json"
    mpath.write_text(json.dumps(manifest))
    return path, mpath


def _flip_last_byte(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 0x01])


@pytest.mark.parametrize(
    ("file", "edit", "match"),
    [
        (_flip_last_byte, None, "sha256"),
        (None, lambda m: m["content_digest"].update(value="0" * 64), "content digest"),
        (None, lambda m: m["file"].update(sha256="0" * 64), "sha256"),
        (None, lambda m: m.pop("content_digest"), "content_digest"),
        (None, lambda m: m["construction"].update(seed=True), "seed"),
    ],
)
def test_a_span_head_init_that_does_not_match_its_manifest_is_refused(tmp_path, file, edit, match):
    data = H64_FILE.read_bytes()
    path, mpath = _tampered(tmp_path, file=None if file is None else file(data), edit=edit)
    with pytest.raises(SystemExit, match=match):
        rft.read_span_head_init(path, mpath)


def test_a_file_whose_bytes_and_manifest_both_changed_still_fails_the_content_digest(tmp_path):
    """sha256 over the bytes is updated to match a tampered file; the content digest, over
    the tensors, still names the head that was pre-registered."""
    tampered = _flip_last_byte(H64_FILE.read_bytes())
    path, mpath = _tampered(
        tmp_path, file=tampered,
        edit=lambda m: m["file"].update(sha256=hashlib.sha256(tampered).hexdigest()),
    )
    with pytest.raises(SystemExit, match="content digest"):
        rft.read_span_head_init(path, mpath)


def test_a_span_head_init_for_another_seed_is_refused_at_argv_time(corpus, tmp_path):
    path, mpath = _tampered(tmp_path, edit=lambda m: m["construction"].update(seed=1))
    with pytest.raises(SystemExit, match="seed"):
        _run(corpus, tmp_path / "l.jsonl", "--span-head-init", str(path),
             "--span-head-init-manifest", str(mpath), real=True)


def test_a_span_head_init_for_another_width_is_refused_by_name(tmp_path):
    """The h64 file into a hidden-32 tower: refused with the shapes, before load_state_dict
    raises its own less specific error."""
    snapshot = tmp_path / "h32"
    tb._write_tiny_snapshot(snapshot, vocab=TINY_VOCAB, hidden=32)
    step = _step_for(snapshot, seed=0, hidden=32)
    with pytest.raises(SystemExit, match="shape"):
        rft.load_span_head_init(step, rft.read_span_head_init(H64_FILE, H64_MANIFEST))


@pytest.mark.skipif(
    not os.environ.get(H2048_ENV),
    reason=(
        f"opt-in: set {H2048_ENV} to L-oracle's span_head_init-seed0.safetensors (main checkout, "
        "data/checkpoints/span-head-init/, ignored). Builds a hidden-2048 stand-in tower on CPU"
    ),
)
def test_the_2b_head_at_seed_0_is_the_pre_registered_h2048_fixture(tmp_path):
    """Amendment 2 (iv): the torch arms build their head from seed 0 and record the digest;
    gate 0 compares it with the Rust trainer's. Built through QwenDecisionStep's own path at
    hidden 2048, as L-oracle's generator does, then the fallback file read and loaded."""
    from qd_train_oracle_span_head_init import STAND_IN_LAYERS, STAND_IN_PLAN
    from qd_train_oracle_tiny import make_step, write_snapshot

    snapshot = tmp_path / "snapshot-h2048"
    write_snapshot(snapshot, STAND_IN_LAYERS, 2048)
    _, step = make_step(
        snapshot=snapshot, layers=STAND_IN_LAYERS, dtype="bf16", plan=STAND_IN_PLAN, steps=1,
        lr=1e-5, seed=0, lower_layers_n=0, lower_lr_scale=1.0, hidden=2048,
    )
    assert rft.span_head_digest(step) == H2048_DIGEST
    init = rft.read_span_head_init(Path(os.environ[H2048_ENV]), H2048_MANIFEST)
    assert init.content_digest == H2048_DIGEST
    assert rft.load_span_head_init(step, init) == H2048_DIGEST


# --- 7. byte identity: every new flag off writes the rows the base commit wrote ------------------

#: Fields that differ between any two runs by construction, never by recipe: the row's own
#: identity and position in its ledger, its timing, and the digest of the tool's own source
#: (which is the change under test).
_VOLATILE = ("row_id", "written_at", "prev_row_hash", "wall_clock_s")
_VOLATILE_METRICS = ("code_that_ran",)


def _normalised(rows: list[LedgerRow]) -> list[dict[str, Any]]:
    """Each row as JSON with this ledger's row ids replaced by their position, so an eval row
    citing its ft row compares equal across two ledgers."""
    text = json.dumps([r.to_json() for r in rows], sort_keys=True)
    for i, r in enumerate(rows):
        text = text.replace(r.row_id, f"<row {i}>")
    out = json.loads(text)
    for row in out:
        for key in _VOLATILE:
            row.pop(key)
        for key in _VOLATILE_METRICS:
            row["metrics"].pop(key)
    return out


def _base_module(ref: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    source = subprocess.run(
        ["git", "show", f"{ref}:tools/real_ft_run.py"], cwd=REPO, capture_output=True,
        check=True,
    ).stdout
    path = tmp_path / "real_ft_run_base.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("real_ft_run_base", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered while it runs (its dataclasses resolve their annotations through
    # sys.modules) and removed after the test.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Its REPO is derived from its own path; the rows are about this repository.
    module.REPO = rft.REPO
    return module


_ARMS = {
    "stand-in": ([], False),
    # F-shaped on the tiny tower: master, layer-wise lr, no mask, deterministic kernels.
    "real-tiny": (["--optimizer", "master", "--lower-layers-n", "1",
                   "--lower-layers-lr-scale", "0.1", "--train-attention-mask", "none",
                   "--deterministic"], True),
}


@pytest.mark.skipif(
    not os.environ.get(BASE_ENV),
    reason=(
        f"opt-in: set {BASE_ENV} to the git ref the new flags were added on top of (e21066c). "
        "A differential against a fixed base fails by design once the tool legitimately "
        "changes, so it is run for the change, not on every suite"
    ),
)
@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_with_every_new_flag_off_the_rows_are_the_base_commits(
    arm, corpus, tmp_path, monkeypatch
):
    flags, real = _ARMS[arm]
    base = _base_module(os.environ[BASE_ENV], tmp_path, monkeypatch)
    flags = [*flags, "--score-val"]
    new_rows = _run(corpus, tmp_path / "new.jsonl", *flags, real=real)
    base_rows = _run(corpus, tmp_path / "base.jsonl", *flags, real=real, module=base)
    assert [r.run_kind for r in new_rows] == [r.run_kind for r in base_rows] == ["ft", "eval"]
    new, old = _normalised(new_rows), _normalised(base_rows)
    for a, b in zip(new, old, strict=True):
        assert a["recipe"] == b["recipe"]
        assert a["protocol_hash"] == b["protocol_hash"]
        assert sorted(a["metrics"]) == sorted(b["metrics"])
    assert new == old
    print(
        f"{arm}: {len(new)} rows identical to {os.environ[BASE_ENV]}'s after dropping "
        f"{list(_VOLATILE)} and metrics {list(_VOLATILE_METRICS)}; recipe_hash "
        + ", ".join(r.protocol.recipe_hash[:16] for r in new_rows)
    )


def test_argparse_namespace_for_old_check_piece_flags_callers_is_untouched():
    """The new checks live in their own function: ``_check_piece_flags``'s callers build bare
    Namespaces without the new attributes, and must keep working."""
    ns = argparse.Namespace(
        lower_layers_n=0, lower_layers_lr_scale=None, beta2=None, checkpoint_skip_layers=0,
        fused_adamw=False, optimizer="bf16", train_attention_mask="padding",
        option_permutation_seed=None, tokenizer_json=None, real_backbone=None, epoch=False,
        replay_shards=None, replay_attestation=None, replay_cache=None, replay_weight=None,
        replay_every=rft.DEFAULT_REPLAY_EVERY, verdicts_out=None, score_val=False,
        suite_verdicts_out=None, needle=False, ood=False,
    )
    rft._check_piece_flags(ns)


