"""The LoRA reference arm's driver, ``tools/lora_reference_run.py`` (tiny model, CPU, no download).

The pre-registration's bound (d') compares the ojas leg with this arm step by step, so what is
pinned here is what that comparison leans on: the loss log has one line per step, the checkpoint is
the ``lora-merged`` layout ``tools/ckpt_average.py`` reads, the same seed gives the same bits, the
base never moves, held-out paths and unregistered device/dtype pairs are refused before anything is
read, and every row is quick.

Torch-gated. Run with the ml venv::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python /Users/bharath/.venvs/ml/bin/python \\
        --with pytest --with hypothesis python -m pytest \\
        python/tests/test_lora_reference_driver.py -o addopts= -q
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")
pytest.importorskip("safetensors", reason="safetensors is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import lora_reference_run as drv  # noqa: E402
import test_backbone as tb  # noqa: E402
import test_real_ft_rungd_flags as rg  # noqa: E402
from test_real_ft_family_metrics import _tokenizer_json  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_train import lora  # noqa: E402
from qd_train.ledger import Ledger  # noqa: E402
from qd_train.run_control import Checkpoint  # noqa: E402

STEPS = 3


@dataclasses.dataclass(frozen=True)
class Env:
    out: Path
    snapshot: Path
    tokenizer: Path


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Env:
    out = tmp_path_factory.mktemp("lora-driver-corpus")
    rg._write_corpus(out, 48)
    snapshot = out / "tiny-snapshot-h64"
    tb._write_tiny_snapshot(snapshot, vocab=rg.TINY_VOCAB, hidden=rg.TINY_HIDDEN)
    return Env(out=out, snapshot=snapshot, tokenizer=_tokenizer_json(out / "tokenizer.json"))


@pytest.fixture(autouse=True)
def tiny_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """``spec_from_checkpoint``'s measured spec describes the 2B; the tiny tower is loaded with its
    own, the one replacement ``test_real_ft_scoring_step`` makes. Nothing else is patched."""
    real = drv.load_text_tower
    spec = tb._tiny_spec(rg.TINY_VOCAB, rg.TINY_HIDDEN)
    monkeypatch.setattr(drv, "load_text_tower", lambda *a, **k: real(*a, **{**k, "spec": spec}))
    monkeypatch.setattr(drv, "spec_from_checkpoint", lambda _path: spec)


def _cfg(env: Env, work: Path, **over) -> drv.LoraRunConfig:
    base = dict(
        out=env.out, backbone=env.snapshot, tokenizer_json=env.tokenizer,
        span_head_init=rg.H64_FILE, span_head_init_manifest=rg.H64_MANIFEST,
        ledger=work / "ledger.jsonl", loss_log=work / "loss.jsonl",
        checkpoint=work / "ckpt" / "lora.json", seed=0, device="cpu", dtype="fp32",
        max_steps=STEPS, wall_clock_cap_s=600.0, expect_rev=rg.REV,
    )
    base.update(over)
    return drv.LoraRunConfig(**base)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _weights(path: Path, *subtrees: str):
    weights, _ = Checkpoint.read_weights(path, subtrees=subtrees)
    return weights


def _tensor(ref) -> torch.Tensor:
    return torch.frombuffer(bytearray(ref.data), dtype=getattr(torch, ref.dtype)).reshape(ref.shape)


# 1 -------------------------------------------------------------------------------------------


def test_a_tiny_run_writes_one_loss_line_per_step_and_one_ledger_row(env, tmp_path):
    result = drv.run(_cfg(env, tmp_path))
    log = _lines(tmp_path / "loss.jsonl")
    assert result.optimizer_steps == STEPS and result.termination == "steps_exhausted"
    assert [line["step"] for line in log] == list(range(STEPS))
    schedule = lora.lora_schedule(STEPS)
    for i, line in enumerate(log):
        assert set(line) == {
            "step", "lr", "loss_total", "loss_choice_17way", "loss_span", "loss_vocab_ce",
            "grad_norm", "batch_index",
        }
        assert line["lr"] == schedule.lr_at(i)
        assert line["grad_norm"] > 0.0 and line["loss_total"] > 0.0
    assert any(line["loss_vocab_ce"] is not None for line in log)
    assert not (tmp_path / "loss.jsonl.partial").exists()
    (row,) = Ledger(tmp_path / "ledger.jsonl").rows()
    assert row.row_id == result.row_id and row.status == "completed"
    assert row.recipe["optimizer_recipe"] == "lora" and row.run_kind == "ft"
    assert row.metrics["train.optimizer_steps"].value == STEPS


def test_the_row_carries_the_seven_digests(env, tmp_path):
    result = drv.run(_cfg(env, tmp_path))
    (row,) = Ledger(tmp_path / "ledger.jsonl").rows()
    cfg = _cfg(env, tmp_path)
    dig = result.digests
    assert dig["span_head_init_content_digest"] == rg.H64_DIGEST
    assert dig["d17_init_digest"] == lora.d17_init_digest(rg.TINY_HIDDEN)
    stored = lora.stored_dtypes(cfg.backbone)
    assert dig["dtype_map_sha256"] == stored.dtype_map_sha256
    assert dig["base_safetensors_sha256"] == stored.safetensors_sha256
    for metric, key in (
        ("init.span_head_digest", "span_head_init_content_digest"),
        ("init.d17_digest", "d17_init_digest"),
        ("init.lora_a_digest", "lora_a_digest"),
        ("init.base_snapshot", "base_snapshot"),
        ("init.base_safetensors_sha256", "base_safetensors_sha256"),
        ("init.dtype_map_sha256", "dtype_map_sha256"),
        ("corpus.plan_seed", "plan_seed"),
        ("corpus.train_shard_hash", "train_shard_hash"),
    ):
        assert row.metrics[metric].value == dig[key], metric


def test_the_a_digest_moves_with_the_seed(env, tmp_path):
    a0 = drv.run(_cfg(env, tmp_path / "a", seed=0)).digests["lora_a_digest"]
    a1 = drv.run(_cfg(env, tmp_path / "b", seed=1)).digests["lora_a_digest"]
    assert a0 != a1


# 2 -------------------------------------------------------------------------------------------


def test_the_checkpoint_round_trips_through_ckpt_averages_lora_merged_reader(env, tmp_path):
    paths = [drv.run(_cfg(env, tmp_path / f"s{s}", seed=s)).checkpoint for s in (0, 1)]
    out = tmp_path / "avg" / "m.safetensors"
    assert ckpt_average.main([*map(str, paths), "--out", str(out), "--from", "lora-merged"]) == 0
    from safetensors.torch import load_file

    averaged = load_file(str(out))
    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["from"] == "lora-merged"
    assert manifest["lora"]["r"] == lora.LORA_RANK and manifest["lora"]["d17"] is True
    assert manifest["lora"]["adapted_modules"] > 0

    states = [_weights(p, "tower", "lora_A", "lora_B", "d17", "span_head") for p in paths]
    scaling = lora.LORA_ALPHA / lora.LORA_RANK
    modules = sorted(states[0]["lora_A"])
    for module in modules:
        merged = []
        for st in states:
            w = _tensor(st["tower"][f"{module}.weight"]).to(torch.float32)
            a, b = _tensor(st["lora_A"][module]), _tensor(st["lora_B"][module])
            assert a.dtype == b.dtype == torch.float32
            merged.append(w + scaling * (b @ a))  # W + (alpha/r) B A
        want = torch.stack([m.to(torch.float64) for m in merged]).mean(0).to(torch.float32)
        assert torch.equal(averaged[f"tower.{module}.weight"], want), module
    assert "d17.delta" in averaged and averaged["d17.delta"].shape[0] == lora.D17_ROWS
    # B moved (the adapters trained), so the merge is not the base.
    assert any(float(_tensor(states[0]["lora_B"][m]).abs().max()) > 0 for m in modules)


def test_the_checkpoint_tower_is_the_dense_base_under_the_dense_names(env, tmp_path):
    result = drv.run(_cfg(env, tmp_path))
    tower = _weights(result.checkpoint, "tower")["tower"]
    assert not [k for k in tower if ".base." in k or "lora_" in k]
    assert "layers.0.mlp.gate_proj.weight" in tower


# 3 -------------------------------------------------------------------------------------------


def _forbid(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a held-out run reached the shard reader or the tower")

    monkeypatch.setattr(drv, "ShardReader", boom)
    monkeypatch.setattr(drv, "load_text_tower", boom)


@pytest.mark.parametrize(
    "where",
    [REPO / "data" / "heldout" / "corpus", Path("/x/heldout/corpus"), Path("/x/held_out/c")],
)
def test_a_held_out_path_is_refused_before_any_tensor_is_read(env, tmp_path, monkeypatch, where):
    _forbid(monkeypatch)
    with pytest.raises(HeldOutViolation):
        drv.run(_cfg(env, tmp_path, out=where))


def test_a_held_out_ledger_or_checkpoint_path_is_refused_too(env, tmp_path, monkeypatch):
    _forbid(monkeypatch)
    with pytest.raises(HeldOutViolation):
        drv.run(_cfg(env, tmp_path, checkpoint=REPO / "data" / "heldout" / "lora.json"))
    with pytest.raises(HeldOutViolation):
        drv.run(_cfg(env, tmp_path, backbone=Path("/tmp/heldout/snapshot")))


# 4 -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "dtype"), [("cpu", "bf16"), ("mps", "fp32"), ("cuda", "bf16"), ("cpu", "fp16")]
)
def test_a_bad_device_dtype_pair_is_refused_before_anything_is_read(
    env, tmp_path, monkeypatch, device, dtype
):
    _forbid(monkeypatch)
    with pytest.raises(drv.LoraDriverRefusal, match="not one of"):
        drv.run(_cfg(env, tmp_path, device=device, dtype=dtype))
    assert not (tmp_path / "ledger.jsonl").exists()


# 5 -------------------------------------------------------------------------------------------


def test_two_runs_at_one_seed_are_bit_identical_on_cpu(env, tmp_path):
    one = drv.run(_cfg(env, tmp_path / "one", seed=2))
    two = drv.run(_cfg(env, tmp_path / "two", seed=2))
    log_one = (tmp_path / "one" / "loss.jsonl").read_bytes()
    assert log_one == (tmp_path / "two" / "loss.jsonl").read_bytes()
    assert one.digests == two.digests
    keys = ("tower", "lora_A", "lora_B", "d17", "span_head")
    a, b = _weights(one.checkpoint, *keys), _weights(two.checkpoint, *keys)
    for tree in keys:
        assert {k: v.digest() for k, v in a[tree].items()} == {
            k: v.digest() for k, v in b[tree].items()
        }, tree
    other = drv.run(_cfg(env, tmp_path / "three", seed=3))
    assert (tmp_path / "three" / "loss.jsonl").read_bytes() != log_one
    assert other.digests["lora_a_digest"] != one.digests["lora_a_digest"]


# 6 -------------------------------------------------------------------------------------------


def test_the_base_is_bit_identical_after_training(env, tmp_path):
    from safetensors import safe_open

    result = drv.run(_cfg(env, tmp_path))
    tower = _weights(result.checkpoint, "tower")["tower"]
    checked = 0
    with safe_open(str(env.snapshot / "model.safetensors"), framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118 - safe_open's handle is not a mapping
            short = name.removeprefix("model.language_model.")
            if short == "embed_tokens.weight":
                continue  # sliced to the shard set's vocabulary; its rows are checked by remap
            assert short in tower, short
            assert torch.equal(_tensor(tower[short]), handle.get_tensor(name)), short
            checked += 1
    assert checked > 20


# 7 -------------------------------------------------------------------------------------------


def test_every_row_is_quick_and_cannot_promote(env, tmp_path):
    drv.run(_cfg(env, tmp_path))
    ledger = Ledger(tmp_path / "ledger.jsonl")
    rows = ledger.rows()
    assert rows
    for row in rows:
        assert row.quick is True and row.quick_reason
        assert "one seed" in row.quick_reason and "never a promotion run" in row.quick_reason
        verdict = ledger.promotion_verdict(row.protocol.hash_without_seed())
        assert verdict.promoted is False
        assert any("quick" in reason for reason in verdict.reasons)


# bounded, fail closed -------------------------------------------------------------------------


def test_a_run_the_cap_stops_writes_its_row_and_nothing_that_looks_finished(env, tmp_path):
    with pytest.raises(drv.RunIncomplete, match="wall_clock_cap"):
        drv.run(_cfg(env, tmp_path, wall_clock_cap_s=1e-9))
    assert not (tmp_path / "ckpt" / "lora.json").exists()
    assert not (tmp_path / "loss.jsonl").exists()
    (row,) = Ledger(tmp_path / "ledger.jsonl").rows()
    assert row.quick and row.metrics["train.termination"].value == "wall_clock_cap"


def test_bounds_and_existing_outputs_are_refused(env, tmp_path):
    for over, match in (
        ({"max_steps": 0}, "at least 1"),
        ({"max_steps": 10_000}, "past the epoch"),
        ({"wall_clock_cap_s": 0.0}, "wall-clock-cap"),
        ({"wall_clock_cap_s": 1e12}, "wall-clock-cap"),
        ({"seed": -1}, "seed"),
    ):
        with pytest.raises(drv.LoraDriverRefusal, match=match):
            drv.run(_cfg(env, tmp_path, **over))
    drv.run(_cfg(env, tmp_path))
    with pytest.raises(drv.LoraDriverRefusal, match="already exists"):
        drv.run(_cfg(env, tmp_path))


def test_a_wrong_pinned_base_sha_is_refused_before_the_tower_loads(env, tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "load_text_tower", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(drv.LoraDriverRefusal, match="safetensors sha256"):
        drv.run(_cfg(env, tmp_path, expect_base_sha256="0" * 64))


# the MPS bf16 leg, on the tiny model (skipped where MPS is absent) ---


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device on this host")
def test_the_mps_bf16_leg_runs_with_a_bf16_base_f32_stored_tensors_and_f32_adapters(env, tmp_path):
    from safetensors.torch import load_file, save_file

    snap = tmp_path / "snap-bf16"
    snap.mkdir()
    for f in env.snapshot.iterdir():
        if f.name != "model.safetensors":
            (snap / f.name).write_bytes(f.read_bytes())
    stored = load_file(str(env.snapshot / "model.safetensors"))
    keep = ("linear_attn.A_log", "linear_attn.norm.weight")
    save_file(
        {k: (v if k.endswith(keep) else v.to(torch.bfloat16)).contiguous()
         for k, v in stored.items()},
        str(snap / "model.safetensors"),
    )
    result = drv.run(_cfg(env, tmp_path, backbone=snap, device="mps", dtype="bf16"))
    assert result.optimizer_steps == STEPS
    assert all(line["loss_total"] == line["loss_total"] for line in _lines(tmp_path / "loss.jsonl"))
    states = _weights(result.checkpoint, "tower", "lora_A", "lora_B")
    tower = states["tower"]
    assert tower["layers.0.mlp.gate_proj.weight"].dtype == "bfloat16"
    a_log = next(k for k in tower if k.endswith("linear_attn.A_log"))
    assert tower[a_log].dtype == "float32"
    assert {r.dtype for r in states["lora_A"].values()} == {"float32"}


# v6 data rules and the pre-registration binding (card ft-3b8126cd, lane W2-2) ----------------------
#
# Both are checked by validate_inputs, the one pre-read entry point. The binding calls
# qd-prereg-bind as a subprocess; no tiny fixture matches the real pre-registration's shard
# hash, so every test below reaches the same call with a fake binary named by
# QD_PREREG_BIND_BIN (or LoraRunConfig.prereg_bind_bin), never a skip flag.


def _fake_bind(where: Path, *, code: int, result: str | None, sleep_s: float = 0.0) -> Path:
    """A stand-in qd-prereg-bind: logs its argv, optionally sleeps, prints a RESULT line."""
    where.mkdir(parents=True, exist_ok=True)
    script = where / "qd-prereg-bind"
    tail = f"echo 'RESULT: {result}'\n" if result is not None else ""
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{where / "argv.txt"}"\n'
        + (f"sleep {sleep_s}\n" if sleep_s else "")
        + "echo 'one line of report'\n"
        + tail
        + f"exit {code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.fixture(autouse=True)
def passing_prereg_bind(tmp_path_factory, monkeypatch) -> Path:
    """Every test but the binding's own runs against a binary that passes."""
    script = _fake_bind(tmp_path_factory.mktemp("bind-pass"), code=0, result="PASS")
    monkeypatch.setenv("QD_PREREG_BIND_BIN", str(script))
    return script


def _nothing_opened(monkeypatch) -> None:
    _forbid(monkeypatch)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("a tensor file was read"))


def test_a_v5_rule_config_is_refused_before_any_tensor_is_read(env, tmp_path, monkeypatch):
    """Q15: on the tree before this change the driver ran under any DataConfig it was handed."""
    _nothing_opened(monkeypatch)
    monkeypatch.setattr(drv, "DataConfig", lambda: DataConfig.legacy_v5())
    with pytest.raises(drv.LegacyDataRulesRefusal, match="v5"):
        drv.run(_cfg(env, tmp_path))
    assert not (tmp_path / "ledger.jsonl").exists()


def test_a_v6_config_passes_validation_and_is_returned_as_v6(env, tmp_path):
    config = drv.validate_inputs(_cfg(env, tmp_path))
    assert config == drv.data_rules(DataConfig(), v6=True)
    assert config.benchmark_eval_splits_are_targets is True


@pytest.mark.parametrize(
    ("code", "result", "needle"),
    [
        (1, "FAILED", "FAILED"),
        (3, "NOT_RUN", "NOT_RUN"),
        (2, None, "usage"),
        (0, None, "RESULT"),
        (0, "FAILED", "RESULT"),
        (0, "NOT_RUN", "RESULT"),
        (7, "PASS", "exit"),
    ],
    ids=["failed", "not-run", "bad-usage", "exit0-no-result", "exit0-says-failed",
         "exit0-says-not-run", "odd-exit-with-pass-line"],
)
def test_a_prereg_bind_that_does_not_pass_refuses_before_any_tensor_is_read(
    env, tmp_path, monkeypatch, code, result, needle
):
    _nothing_opened(monkeypatch)
    bind = _fake_bind(tmp_path / "bind", code=code, result=result)
    monkeypatch.setenv("QD_PREREG_BIND_BIN", str(bind))
    with pytest.raises(drv.PreregBindRefusal, match=needle):
        drv.run(_cfg(env, tmp_path))
    assert not (tmp_path / "ledger.jsonl").exists()


def test_a_prereg_bind_that_passes_is_called_with_the_prereg_and_the_train_shard_set(
    env, tmp_path, monkeypatch
):
    bind = _fake_bind(tmp_path / "bind", code=0, result="PASS")
    monkeypatch.setenv("QD_PREREG_BIND_BIN", str(bind))
    drv.run(_cfg(env, tmp_path))
    argv = (tmp_path / "bind" / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv[argv.index("--prereg") + 1] == str(REPO / "campaign" / "lora-2b-preregistered.json")
    assert argv[argv.index("--shard-dir") + 1] == str(env.out / "shards" / "train")
    assert argv[argv.index("--repo") + 1] == str(REPO)


def test_a_prereg_bind_that_hangs_is_refused_at_its_bound(env, tmp_path, monkeypatch):
    _nothing_opened(monkeypatch)
    bind = _fake_bind(tmp_path / "bind", code=0, result="PASS", sleep_s=30)
    monkeypatch.setenv("QD_PREREG_BIND_BIN", str(bind))
    monkeypatch.setattr(drv, "PREREG_BIND_TIMEOUT_S", 0.5)
    with pytest.raises(drv.PreregBindRefusal, match="timed out"):
        drv.run(_cfg(env, tmp_path))


def test_a_missing_prereg_bind_binary_is_not_run_and_refuses(env, tmp_path, monkeypatch):
    _nothing_opened(monkeypatch)
    monkeypatch.setenv("QD_PREREG_BIND_BIN", str(tmp_path / "no-such-binary"))
    with pytest.raises(drv.PreregBindRefusal, match="NOT_RUN"):
        drv.run(_cfg(env, tmp_path))
    monkeypatch.delenv("QD_PREREG_BIND_BIN")
    with pytest.raises(drv.PreregBindRefusal, match="QD_PREREG_BIND_BIN is unset"):
        drv.run(_cfg(env, tmp_path))


def test_the_prereg_bind_field_beats_the_environment(env, tmp_path, monkeypatch):
    _nothing_opened(monkeypatch)
    failing = _fake_bind(tmp_path / "bad", code=1, result="FAILED")
    with pytest.raises(drv.PreregBindRefusal, match="FAILED"):
        drv.run(_cfg(env, tmp_path, prereg_bind_bin=failing))
