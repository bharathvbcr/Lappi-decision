"""Adversarial tests for ``tools/lora_reference_run.py`` (lane L3 review, 2026-10-17).

Each test names an attack. A test that FAILS here is a confirmed break of the driver; a test that
passes is a held attack kept as a regression guard. Run under ``tools/mac_heavy.sh``.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import lora_reference_run as drv  # noqa: E402
import test_real_ft_rungd_flags as rg  # noqa: E402
from test_lora_reference_driver import (  # noqa: E402,F401
    _cfg,
    env,
    passing_prereg_bind,
    tiny_spec,
)

from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_train import lora  # noqa: E402
from qd_train.ledger import Ledger  # noqa: E402

# 1 held-out ----------------------------------------------------------------------------------


def test_symlink_into_a_heldout_dir_is_refused(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    real = tmp_path / "heldout" / "c"
    real.mkdir(parents=True)
    (tmp_path / "clean").symlink_to(real)
    with pytest.raises(HeldOutViolation):
        drv.validate_inputs(_cfg(env, tmp_path, out=tmp_path / "clean"))


def test_dotdot_traversal_into_heldout_is_refused(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    (tmp_path / "heldout" / "c").mkdir(parents=True)
    (tmp_path / "a").mkdir()
    with pytest.raises(HeldOutViolation):
        drv.validate_inputs(_cfg(env, tmp_path, out=tmp_path / "a" / ".." / "heldout" / "c"))


@pytest.mark.parametrize("seg", ["HeldOut", "HELD-OUT", "Held_Out"])
def test_case_variants_of_the_marker_are_refused(env, tmp_path, seg):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    with pytest.raises(HeldOutViolation):
        drv.validate_inputs(_cfg(env, tmp_path, tokenizer_json=tmp_path / seg / "tok.json"))


def test_loss_log_partial_sibling_in_a_heldout_dir_is_refused(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    with pytest.raises(HeldOutViolation):
        drv.validate_inputs(_cfg(env, tmp_path, loss_log=tmp_path / "heldout" / "loss.jsonl"))


def test_a_shard_file_symlinked_into_heldout_is_refused(env, tmp_path, monkeypatch):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    """Shared gap: the Rust door resolves only the directory too (files.rs reads by plain open),
    so this is not a Rust-refuses/Python-accepts break, but rule 3 is bypassed by indirection."""
    out = tmp_path / "corpus"
    shutil.copytree(env.out / "shards" / "train", out / "shards" / "train")
    held = tmp_path / "heldout"
    held.mkdir()
    tok = out / "shards" / "train" / "tokens.u32"
    shutil.move(str(tok), held / "tokens.u32")
    tok.symlink_to(held / "tokens.u32")
    # The reader is another lane's; the driver's own held-out check resolves each shard file.
    with pytest.raises(HeldOutViolation):
        drv.validate_inputs(_cfg(env, tmp_path / "w", out=out))


# 3 output clobbering -------------------------------------------------------------------------


def test_a_dangling_symlink_at_the_loss_log_partial_is_refused(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    """``Path.exists`` is False for a dangling link, so ``open('w')`` would follow it and create
    its target: a write outside the validated paths."""
    target = tmp_path / "elsewhere" / "file"
    target.parent.mkdir()
    (tmp_path / "loss.jsonl.partial").symlink_to(target)
    with pytest.raises(drv.LoraDriverRefusal):
        drv.validate_inputs(_cfg(env, tmp_path))


def test_a_dangling_symlink_at_the_checkpoint_is_refused(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "lora.json").symlink_to(tmp_path / "nowhere")
    with pytest.raises(drv.LoraDriverRefusal):
        drv.validate_inputs(_cfg(env, tmp_path))


def test_a_checkpoint_never_exists_without_its_finished_loss_log(env, tmp_path, monkeypatch):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    """Kill/failure between ``checkpoint.write`` and ``partial.replace``: the checkpoint would look
    complete to ckpt_average while the loss log is still ``.partial``."""
    real = os.replace

    def flaky(src, dst, *a, **k):
        if str(src).endswith("loss.jsonl.partial"):
            raise OSError("simulated kill before the log rename")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError):
        drv.run(_cfg(env, tmp_path))
    assert not (tmp_path / "ckpt" / "lora.json").exists()


def test_a_write_failure_midway_leaves_no_readable_checkpoint(env, tmp_path, monkeypatch):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    from qd_train.run_control import Checkpoint

    real = Checkpoint.write

    def die_after_sidecar(self, path):
        real_replace = os.replace
        calls = {"n": 0}

        def boom(src, dst, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # the JSON's rename (the sidecar's was first)
                raise OSError("kill")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(os, "replace", boom)
        try:
            return real(self, path)
        finally:
            monkeypatch.setattr(os, "replace", real_replace)

    monkeypatch.setattr(Checkpoint, "write", die_after_sidecar)
    with pytest.raises(OSError):
        drv.run(_cfg(env, tmp_path))
    ck = tmp_path / "ckpt" / "lora.json"
    assert not ck.exists()
    # A lone path is refused for having fewer than two inputs before the file is looked at, which
    # would pass for any file. Two inputs, one absent, with the span-head arguments the averager
    # demands for two inputs, so the read of the absent file is what refuses.
    monkeypatch.setattr(Checkpoint, "write", real)
    good = drv.run(_cfg(env, tmp_path / "good")).checkpoint
    out = tmp_path / "m.safetensors"
    with pytest.raises(FileNotFoundError):
        ckpt_average.main([str(good), str(ck), "--out", str(out), "--from", "lora-merged",
                           "--span-head-seed", "0", "--ft-ledger", str(tmp_path / "ledger.jsonl"),
                           "--ft-row-ids", "a", "b"])
    assert not out.exists()


# 4 quick -------------------------------------------------------------------------------------


def test_three_seeds_of_one_protocol_still_cannot_promote(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    for s in (0, 1, 2):
        drv.run(_cfg(env, tmp_path / f"s{s}", seed=s, ledger=tmp_path / "ledger.jsonl"))
    ledger = Ledger(tmp_path / "ledger.jsonl")
    rows = ledger.rows()
    assert len(rows) == 3 and all(r.quick for r in rows)
    for r in rows:
        assert ledger.promotion_verdict(r.protocol.hash_without_seed()).promoted is False


def test_a_full_epoch_run_is_still_quick(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    drv.run(_cfg(env, tmp_path, max_steps=3))
    (row,) = Ledger(tmp_path / "ledger.jsonl").rows()
    assert row.quick is True


# 5 digests -----------------------------------------------------------------------------------


def _tiny_two_orders():
    from torch import nn

    class Attn(nn.Module):
        """Named ``self_attn`` so the pair's names end with the suffixes lora.py adapts."""

        def __init__(self, first_q: bool):
            super().__init__()
            if first_q:
                self.q_proj, self.o_proj = nn.Linear(8, 8, bias=False), nn.Linear(16, 8, bias=False)
            else:
                self.o_proj, self.q_proj = nn.Linear(16, 8, bias=False), nn.Linear(8, 8, bias=False)

    class Block(nn.Module):
        def __init__(self, first_q: bool):
            super().__init__()
            self.self_attn = Attn(first_q)

    class M(nn.Module):
        def __init__(self, first_q: bool):
            super().__init__()
            self.b = Block(first_q)

    return M(True), M(False)


def test_the_rule_digest_is_independent_of_the_attached_a(env):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    m, _ = _tiny_two_orders()
    lora.attach_lora(m, seed=3)
    with torch.no_grad():
        next(p for n, p in lora.lora_parameters(m) if n.endswith("lora_A.weight")).add_(1.0)
    assert lora.lora_a_digest(m) != lora.lora_a_rule_digest(lora.adapted_in_features(m), seed=3)


def test_the_a_digest_does_not_depend_on_module_registration_order():
    """The pre-registration pins named_modules() order, so a same-architecture model registered in
    another order (an ojas-side enumeration) silently gets a different A and digest; the driver
    compares the digest only against a rule digest fed from the same order."""
    m1, m2 = _tiny_two_orders()
    lora.attach_lora(m1, seed=3)
    lora.attach_lora(m2, seed=3)
    assert lora.lora_a_digest(m1) == lora.lora_a_digest(m2)


def test_a_nonzero_d17_init_aborts_the_run(env, tmp_path, monkeypatch):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    real = lora.AnswerRowDelta.__init__

    def init(self, *a, **k):
        real(self, *a, **k)
        with torch.no_grad():
            self.delta.add_(0.5)

    monkeypatch.setattr(lora.AnswerRowDelta, "__init__", init)
    with pytest.raises(drv.LoraDriverRefusal, match="D17"):
        drv.run(_cfg(env, tmp_path))
    assert not (tmp_path / "ckpt" / "lora.json").exists()


def test_a_drifted_a_aborts_the_run_and_leaves_a_row(env, tmp_path, monkeypatch):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    real = lora.attach_lora

    def drifted(model, **k):
        out = real(model, **k)
        with torch.no_grad():
            next(p for n, p in lora.lora_parameters(model) if n.endswith("lora_A.weight")).add_(1)
        return out

    monkeypatch.setattr(lora, "attach_lora", drifted)
    with pytest.raises(drv.LoraDriverRefusal, match="LoRA-A digest"):
        drv.run(_cfg(env, tmp_path))
    assert (tmp_path / "ledger.jsonl").exists(), "an aborted run wrote no row at all"


# 6 determinism -------------------------------------------------------------------------------


def test_a_cpu_run_does_not_leave_deterministic_mode_on_for_the_process(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    before = torch.are_deterministic_algorithms_enabled()
    drv.run(_cfg(env, tmp_path))
    assert torch.are_deterministic_algorithms_enabled() == before


def test_interleaved_seeds_in_one_process_do_not_leak(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    a = drv.run(_cfg(env, tmp_path / "a", seed=5))
    drv.run(_cfg(env, tmp_path / "b", seed=6))
    torch.manual_seed(999)
    c = drv.run(_cfg(env, tmp_path / "c", seed=5))
    a_log = (tmp_path / "a" / "loss.jsonl").read_bytes()
    assert a_log == (tmp_path / "c" / "loss.jsonl").read_bytes()
    assert a.digests == c.digests


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device on this host")
def test_mps_bf16_first_step_loss_tracks_the_cpu_pin(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    cpu = drv.run(_cfg(env, tmp_path / "cpu", seed=1))
    mps = drv.run(_cfg(env, tmp_path / "mps", seed=1, device="mps", dtype="bf16"))
    assert cpu.digests["lora_a_digest"] == mps.digests["lora_a_digest"]
    cpu_log = (tmp_path / "cpu" / "loss.jsonl").read_text().splitlines()
    mps_log = (tmp_path / "mps" / "loss.jsonl").read_text().splitlines()
    l_cpu = [json.loads(x)["loss_total"] for x in cpu_log]
    l_mps = [json.loads(x)["loss_total"] for x in mps_log]
    assert all(abs(a - b) <= 0.05 for a, b in zip(l_cpu, l_mps, strict=True)), (l_cpu, l_mps)


# 7 malformed inputs --------------------------------------------------------------------------


def _bad_snapshot(env, dst: Path, blob: bytes) -> Path:  # noqa: F811 - the fixture imported from test_lora_reference_driver
    shutil.copytree(env.snapshot, dst)
    (dst / "model.safetensors").write_bytes(blob)
    return dst


@pytest.mark.parametrize(
    "blob",
    [
        struct.pack("<Q", 2**62) + b"{}",
        struct.pack("<Q", 2**64 - 1),
        struct.pack("<Q", 4) + b"\xff\xfe\xfd\xfc",
        b"abc",
    ],
    ids=["huge-len", "u64-max", "non-utf8", "truncated"],
)
def test_a_malformed_safetensors_header_is_a_typed_refusal(env, tmp_path, blob):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    snap = _bad_snapshot(env, tmp_path / "snap", blob)
    with pytest.raises(drv.LoraDriverRefusal) as exc:
        drv.run(_cfg(env, tmp_path / "w", backbone=snap))
    assert type(exc.value) is not UnicodeDecodeError
    assert not (tmp_path / "w" / "ckpt").exists()


def test_a_nan_in_the_base_aborts_with_no_checkpoint_and_a_failed_row(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    from safetensors.torch import load_file, save_file

    snap = tmp_path / "snap"
    shutil.copytree(env.snapshot, snap)
    tensors = load_file(str(snap / "model.safetensors"))
    key = next(k for k in tensors if k.endswith("layers.0.mlp.gate_proj.weight"))
    tensors[key][0, 0] = float("nan")
    save_file(tensors, str(snap / "model.safetensors"), metadata={"format": "pt"})
    # The NaN reaches the loss first, and the accumulation group's rule 3 stops the run there with
    # the trainer's own violation type (before lora.py's non-finite grad-norm check is reached).
    from qd_train.trainer import TrainerContractViolation

    with pytest.raises(TrainerContractViolation, match="non-finite loss"):
        drv.run(_cfg(env, tmp_path / "w", backbone=snap))
    assert not (tmp_path / "w" / "ckpt" / "lora.json").exists()
    assert not (tmp_path / "w" / "loss.jsonl").exists()
    rows = Ledger(tmp_path / "w" / "ledger.jsonl").rows()
    assert rows and all(r.status != "completed" for r in rows)


def test_an_empty_shard_set_is_a_typed_refusal(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    out = tmp_path / "empty"
    try:
        rg._write_corpus(out, 0)
    except Exception:  # the writer itself refuses an empty corpus: nothing to attack
        pytest.skip("an empty shard set cannot be written")
    with pytest.raises(drv.LoraDriverRefusal):
        drv.run(_cfg(env, tmp_path / "w", out=out))


def test_unicode_and_space_paths_run(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    w = tmp_path / "wörk dir é"
    res = drv.run(_cfg(env, w))
    assert res.optimizer_steps == 3 and res.checkpoint.exists()


# 8 loss log ----------------------------------------------------------------------------------


def test_the_loss_log_is_fsynced_each_step(env, tmp_path, monkeypatch):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    real_log_fsyncs = []
    orig_apply = lora.LoraReferenceStep.apply

    def watch(self, *, lr):
        n = len(calls)
        orig_apply(self, lr=lr)
        real_log_fsyncs.append(len(calls) - n)

    monkeypatch.setattr(lora.LoraReferenceStep, "apply", watch)
    drv.run(_cfg(env, tmp_path))
    assert all(n >= 1 for n in real_log_fsyncs), real_log_fsyncs


def test_one_log_line_per_step_with_no_extra_lines(env, tmp_path):  # noqa: F811 - the fixture imported from test_lora_reference_driver
    drv.run(_cfg(env, tmp_path))
    lines = (tmp_path / "loss.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(x) for x in lines)
