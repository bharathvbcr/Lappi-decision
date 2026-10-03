"""The Rust release exporter against a Python oracle: transformers loads what it wrote.

A tiny Qwen3.5 tower is trained into two real checkpoints and averaged by
``tools/ckpt_average.py`` exactly as the phase-4 average is (``test_backbone``'s
``_written_checkpoint``, unchanged). ``qd-export`` -- the Rust binary, named by
``QD_EXPORT_BIN`` -- turns the average into a release, and ``tools/export_letter_parity.py``
loads that release with ``transformers`` and compares it to the tower:

* every tensor must be torch's own bf16 cast of the tower's, bit for bit -- which checks the
  Rust round-to-nearest-even against torch's on every weight of an fp32 tower;
* letter logits through the release must equal, **exactly**, letter logits through the
  bf16-cast tower (identical weights, identical code, CPU, deterministic algorithms);
* against the uncast tower the letter logits differ by what the bf16 cast costs: zero for a
  bf16 tower, and for an fp32 tower the measured difference is required to be non-zero (the
  cast did something) and under ``CAST_LOGIT_ATOL``, stated below with its arithmetic.

Torch-gated like ``test_backbone``, and skipped (not passed) when ``QD_EXPORT_BIN`` is unset::

    cargo build -p qd-export
    QD_EXPORT_BIN=$PWD/target/debug/qd-export uv run --no-project \\
        --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \\
        --with datasketch python -m pytest python/tests/test_export_oracle.py -o addopts= -q
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")
pytest.importorskip("safetensors", reason="safetensors is an optional 'mac' extra")
pytest.importorskip("tokenizers", reason="tokenizers comes with transformers")

from test_backbone import (  # noqa: E402
    TINY_HIDDEN,
    TINY_VOCAB,
    _ckpt_average,
    _master_checkpoint,
    _tiny_tower,
    _written_checkpoint,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import export_letter_parity as parity  # noqa: E402

from qd_train.backbone import text_tensor_index  # noqa: E402

#: For an fp32 tower, |logit(release) - logit(tower)| over the 17 letters. Each bf16 weight
#: is within a relative 2^-9 (~2.0e-3) of its fp32 source. Measured on this fixture
#: 2026-10-01: letter logits at most 0.59 in magnitude, and the cast moved them by at most
#: 6.6e-4 -- about |logit| x 2^-9, as a first-order perturbation should. 1e-2 is ~15x that.
#: This bound is for THIS fixture only and says nothing about the 2B model, whose difference
#: the box run reports and does not gate.
CAST_LOGIT_ATOL = 1e-2


def _tiny_tokenizer_json() -> str:
    """A WordLevel tokenizer whose 17 answer letters are single tokens (ids 1..26), all inside
    the tiny 64-row vocabulary -- the same one ``crates/qd-export/tests/common`` uses."""
    vocab = {"[UNK]": 0, **{chr(ord("A") + i): i + 1 for i in range(26)}}
    return json.dumps(
        {
            "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
            "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None, "decoder": None,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
        }
    )


#: The families the oracle's train manifest holds, sorted: what the release records as
#: ``trained_families`` (``qd-export --train-manifest``, Fable's pipeline ruling, item 4).
TRAINED_FAMILIES = ["code.defect_class", "intent.in_scope"]


def _train_manifest(dest: Path) -> Path:
    """A train-split data manifest in ``qd_data.manifest.Manifest.to_json``'s shape, one row
    per family. The exporter reads only its split, families and integrity fields."""
    entries = [
        {
            "row_id": f"oracle-row-{i}", "content_hash": f"{i:064x}", "split": "train",
            "source_id": "test/source", "host": "test", "family_id": family,
            "repo_key": f"repo-{i}", "identity_key": f"identity-{i}", "licence_id": "MIT",
            "obligations": [],
        }
        for i, family in enumerate(TRAINED_FAMILIES)
    ]
    manifest = {
        "manifest_format_version": 1, "split": "train", "data_snapshot_hash": "cd" * 32,
        "n_rows": len(entries), "held_out_families": ["code.language_id", "qa.answerability"],
        "entries": entries,
    }
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dest


def _export_bin() -> Path:
    raw = os.environ.get("QD_EXPORT_BIN")
    if not raw:
        pytest.skip("QD_EXPORT_BIN is unset: build crates/qd-export and point it at qd-export")
    path = Path(raw)
    if not path.is_file():
        pytest.fail(f"QD_EXPORT_BIN={raw} is not a file")
    return path


def _base_snapshot(tower_snapshot: Path, dest: Path) -> Path:
    """The tiny tower's own config with the two fields the real Qwen3.5-2B-Base config states
    and transformers' saved tiny config does not, plus the four tokenizer files.

    * top-level ``tie_word_embeddings: true`` -- ``Qwen3_5Config`` defaults the top-level flag
      to false while the text config says true, and qd-metal reads the top-level one first;
    * ``text_config.attn_output_gate: true`` -- transformers 5.12.1 has no such field and
      always builds the gated ``q_proj`` (``modeling_qwen3_5.py:658``, ``* 2``), so the tiny
      tower IS gated; the real config says so explicitly and qd-metal requires it.

    Neither changes a tensor: both describe the model the tiny tower already is.
    """
    dest.mkdir()
    config = json.loads((tower_snapshot / "config.json").read_text())
    assert config["text_config"]["tie_word_embeddings"] is True
    assert config["text_config"].get("attn_output_gate", True) is True
    config["tie_word_embeddings"] = True
    config["text_config"]["attn_output_gate"] = True
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    (dest / "tokenizer.json").write_text(_tiny_tokenizer_json())
    (dest / "tokenizer_config.json").write_text('{"model_max_length": 64}')
    (dest / "vocab.json").write_text('{"A": 1}')
    (dest / "merges.txt").write_text("#version: 0.2\n")
    return dest


def _checkpoint(tmp_path: Path, *, seed: int, dtype: str) -> Path:
    """A real checkpoint of a tiny tower in ``dtype``.

    bf16 is ``test_backbone._written_checkpoint`` itself (the fp32-master recipe). Its
    ``MasterWeightAdamW`` refuses an fp32 tower -- a master copy of fp32 parameters changes
    nothing -- so the fp32 tower is the same construction under plain AdamW, ``_tiny_tower``'s
    own default: it stands for an average taken over fp32 weights, which is what ckpt_average
    writes when the tower it averages is fp32.
    """
    if dtype == "bf16":
        return _written_checkpoint(tmp_path, seed=seed, dtype="bf16")[0]
    from qd_train.backbone import QwenDecisionStep
    from qd_train.run_control import Checkpoint, LossLog, LRSchedule, Position

    tower, _ = _tiny_tower(tmp_path / f"s{seed}")
    assert str(tower.model.get_input_embeddings().weight.dtype) == "torch.float32"
    step = QwenDecisionStep(tower, seed=seed, lr=1e-3, total_steps=4, max_width=64)
    ckpt = Checkpoint(
        position=Position(epoch=0, index=1),
        optimizer_step=1,
        seed=seed,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=4, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed:064x}",
        model_state=step.state(),
    )
    path = tmp_path / f"ckpt-{seed}-fp32.json"
    ckpt.write(path)
    return path


def _rows(n: int, seed: int) -> list[tuple[list[int], int]]:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        length = 8 + 5 * i
        tokens = torch.randint(0, TINY_VOCAB, (length,), generator=g).tolist()
        rows.append((tokens, length - 1))
    return rows


def _average_and_export(
    tmp_path: Path, dtype: str, source: str = "tower"
) -> tuple[Path, Path, Path]:
    """Two checkpoints -> ``ckpt_average`` -> ``qd-export``. Returns (avg, base, release).

    ``source`` is ``ckpt_average --from``: ``masters`` is J7's average (Fable D1), so the
    manifest that average writes is the one the exporter must accept. Its inputs are
    ``test_backbone._master_checkpoint``'s: trained a few steps on batches of their own, since
    an untrained tower's layers can hold byte-identical tensors (``dt_bias``), whose masters
    the average rightly refuses to tell apart."""
    exe = _export_bin()
    if source == "masters":
        a = _master_checkpoint(tmp_path / "s1", seed=1)[0]
        b = _master_checkpoint(tmp_path / "s2", seed=2)[0]
    else:
        a = _checkpoint(tmp_path, seed=1, dtype=dtype)
        b = _checkpoint(tmp_path, seed=2, dtype=dtype)
    avg = tmp_path / "avg" / "avg.safetensors"
    argv = [str(a), str(b), "--out", str(avg), "--ft-row-ids", "r1", "r2", "--from", source]
    assert _ckpt_average().main(argv) == 0
    base = _base_snapshot(tmp_path / "s1" / "snapshot", tmp_path / "base")
    pin = hashlib.sha256((base / "tokenizer.json").read_bytes()).hexdigest()
    release = tmp_path / "release"
    done = subprocess.run(
        [
            str(exe), "--source", str(avg), "--base-snapshot", str(base),
            "--tokenizer-sha256", pin, "--expect-vocab-size", str(TINY_VOCAB),
            "--train-manifest", str(_train_manifest(tmp_path / "train.json")),
            "--out", str(release),
        ],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr
    return avg, base, release


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_transformers_loads_the_rust_export_as_the_tower(tmp_path, dtype):
    avg, base, release = _average_and_export(tmp_path, dtype)

    # transformers' own loader path reads the release: the sidecar does not confuse it.
    index = text_tensor_index(release)
    assert {p.name for p, _ in index.values()} == {"model.safetensors"}

    report = parity.compare(
        source=avg, release=release, base_snapshot=base, rows=_rows(6, seed=7), device="cpu"
    )
    assert report["weights"]["mismatched"] == [], report["weights"]
    assert report["weights"]["span_head_tensors"] == 4
    assert report["weights"]["tower_tensors"] == len(index)
    assert report["logits"]["release_vs_tower_bf16"]["max_abs"] == 0.0
    assert report["passed"] is True
    cast_cost = report["logits"]["release_vs_tower"]["max_abs"]
    if dtype == "bf16":
        assert cast_cost == 0.0
    else:
        assert 0.0 < cast_cost < CAST_LOGIT_ATOL, cast_cost

    manifest = json.loads((release / "release_manifest.json").read_text())
    avg_manifest = avg.with_name(avg.name + ".manifest.json")
    assert manifest["source"]["ft_row_ids"] == ["r1", "r2"]
    want = hashlib.sha256(avg_manifest.read_bytes()).hexdigest()
    assert manifest["source"]["manifest_sha256"] == want
    assert manifest["conversion"]["hidden_size"] == TINY_HIDDEN
    assert manifest["trained_families"] == TRAINED_FAMILIES


def test_an_average_of_the_fp32_masters_exports_and_loads_as_the_tower(tmp_path):
    """J7 averages the masters and exports that file: the two lanes' manifests must meet.
    The average is cast to bf16 once inside ckpt_average, so the release is its tower bit
    for bit and the cast costs nothing further."""
    avg, base, release = _average_and_export(tmp_path, "bf16", source="masters")
    avg_manifest = json.loads(avg.with_name(avg.name + ".manifest.json").read_text())
    assert avg_manifest["from"] == "masters"

    report = parity.compare(
        source=avg, release=release, base_snapshot=base, rows=_rows(6, seed=7), device="cpu"
    )
    assert report["weights"]["mismatched"] == [], report["weights"]
    assert report["logits"]["release_vs_tower_bf16"]["max_abs"] == 0.0
    assert report["logits"]["release_vs_tower"]["max_abs"] == 0.0
    assert report["passed"] is True
    manifest = json.loads((release / "release_manifest.json").read_text())
    assert manifest["source"]["ft_row_ids"] == ["r1", "r2"]


def test_the_oracle_refuses_a_release_that_is_not_the_tower(tmp_path):
    """The oracle is not vacuous: one flipped bit in one tensor fails both gates."""
    avg, base, release = _average_and_export(tmp_path, "bf16")
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(release / "model.safetensors"))
    key = f"{parity.TEXT_PREFIX}layers.0.mlp.down_proj.weight"
    flipped = tensors[key].clone()
    flipped.view(torch.int16)[0, 0] ^= 0x0100
    tensors[key] = flipped
    save_file(tensors, str(release / "model.safetensors"), metadata={"format": "pt"})

    report = parity.compare(
        source=avg, release=release, base_snapshot=base, rows=_rows(6, seed=7), device="cpu"
    )
    assert any("layers.0.mlp.down_proj.weight" in m for m in report["weights"]["mismatched"])
    assert report["logits"]["release_vs_tower_bf16"]["max_abs"] > 0.0
    assert report["passed"] is False
    with pytest.raises(parity.ParityRefusal, match="vacuously"):
        parity.compare(source=avg, release=release, base_snapshot=base, rows=[], device="cpu")
