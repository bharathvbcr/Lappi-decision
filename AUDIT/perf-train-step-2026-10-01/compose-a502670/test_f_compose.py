"""Does --checkpoint-skip-layers compose with --lower-layers-n 8 --lower-layers-lr-scale 0.1?

Run against main a502670's own code (git archive into the scratchpad), not this branch:
  (a) argv: F's exact box_q_f.sh argv, +/- `--checkpoint-skip-layers 6`, through a502670's
      real_ft_run.main(), stopped at the first shard-set read;
  (b) training: a502670's `_real_step` (the function F builds its step with) on the tiny
      Qwen3.5 tower, master optimizer, bf16, lower_layers_n=2 / 0.1, skip 0 vs skip N where
      a skipped full-attention layer sits inside the lower group (as 3 and 7 do in F), then
      train_ft over 6 span-carrying batches. Compared bit-for-bit on CPU.

Reproduce (Mac, CPU; the tree is main a502670, not the branch this file is committed on):
  mkdir -p <dir>/a502670 && git archive --output=<dir>/a.tar a502670 python tools pyproject.toml
  tar -xf <dir>/a.tar -C <dir>/a502670 && cp test_f_compose.py <dir>/
  uv run --no-project --python ~/.venvs/ml/bin/python --with pytest --with hypothesis \
    --with datasketch python -m pytest <dir>/test_f_compose.py -o addopts= -q -s \
    -p no:cacheprovider --rootdir <dir>
"""
import sys
from pathlib import Path
from types import SimpleNamespace

A = Path(__file__).resolve().parent / "a502670"
for p in (A / "python" / "tests", A / "tools", A / "python"):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402
import real_ft_run as rft  # noqa: E402
import test_backbone as tb  # noqa: E402
import torch  # noqa: E402
from test_real_ft_shuffled_label import _remap  # noqa: E402

import qd_train.backbone as backbone  # noqa: E402
from qd_train.memory import OptimizerSpec  # noqa: E402
from qd_train.trainer import train_ft  # noqa: E402

assert Path(rft.__file__).resolve().is_relative_to(A), rft.__file__
assert Path(backbone.__file__).resolve().is_relative_to(A), backbone.__file__

SNAP = Path(
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)


class _Stop(Exception):
    pass


#: Captured once: a second monkeypatched wrapper must not wrap the first.
_REAL_CHECK = rft._check_piece_flags


def _f_argv(tmp: Path, *extra: str) -> list[str]:
    """box_q_f.sh's seed loop line for seed 0, with box paths moved under tmp."""
    rec = tmp / "fetch-record.json"
    return [
        "--out", str(tmp / "phase4-v4"), "--no-repo-history",
        "--rev", "881ab304f15ea13529002391dda8520c2ea47af4",
        "--defect-class", "data/pool/commitpackft-composed-v1",
        "--defect-download", "data/pool/commitpackft",
        "--defect-noul", "data/pool/defect-noul-v3b",
        "--general-record", str(rec), "--general-max-rows", "200000",
        "--real-backbone", str(SNAP),
        "--optimizer", "master", "--lr", "1e-5", "--devices", "cuda", "--seeds", "0",
        "--epoch", "--no-memorise", "--batch-tokens", "35403",
        "--lower-layers-n", "8", "--lower-layers-lr-scale", "0.1", *extra,
        "--checkpoint-dir", str(tmp / "ckpt"), "--checkpoint-every", "100000",
        "--score-val", "--needle", "--ood", "--ood-general-record", str(rec),
        "--verdicts-out", str(tmp / "v.jsonl"), "--suite-verdicts-out", str(tmp / "sv.jsonl"),
        "--wall-clock-cap-s", "32400", "--instance", "lambda-1xgh200", "--usd-per-hour", "2.29",
        "--ledger", str(tmp / "ledger.jsonl"), "--approved-by", "compose check (not a run)",
    ]


def _argv_outcome(tmp: Path, monkeypatch, *extra: str) -> tuple[str, dict]:
    seen: dict = {}

    def check(args):
        _REAL_CHECK(args)
        seen.update(lower=args.lower_layers_n, scale=args.lower_layers_lr_scale,
                    skip=args.checkpoint_skip_layers)

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(rft, "_check_piece_flags", check)
    for name in ("ShardReader", "ft_split_rows", "_probe_one"):
        monkeypatch.setattr(rft, name, stop)
    monkeypatch.setattr(rft, "resolve_rev", lambda repo, rev: rev)
    monkeypatch.setattr(rft, "check_defect_source", lambda *a, **k: None)
    monkeypatch.setattr(rft, "corpus_facts", lambda *a, **k: rft.CorpusFacts(None, {}))
    monkeypatch.setattr(rft, "general_record_datasets", lambda record: None)
    (tmp / "fetch-record.json").write_text("{}", encoding="utf-8")
    try:
        rft.main(_f_argv(tmp, *extra))
    except _Stop:
        return "reached the first data read", seen
    except SystemExit as exc:
        return f"SystemExit: {exc}", seen
    return "returned", seen


def test_f_argv_with_and_without_skip_6_both_pass_every_argv_check(tmp_path, monkeypatch):
    if not SNAP.is_dir():
        pytest.skip(f"{SNAP} is not on this host")
    (tmp_path / "plain").mkdir()
    (tmp_path / "skip").mkdir()
    plain = _argv_outcome(tmp_path / "plain", monkeypatch)
    skip =_argv_outcome(tmp_path / "skip", monkeypatch, "--checkpoint-skip-layers", "6")
    print("plain:", plain, "\nskip :", skip)
    assert plain[0] == skip[0] == "reached the first data read"
    assert plain[1] == {"lower": 8, "scale": 0.1, "skip": 0}
    assert skip[1] == {"lower": 8, "scale": 0.1, "skip": 6}


def test_the_argv_harness_does_reach_the_skip_refusal(tmp_path, monkeypatch):
    """Control: the same harness does see a refusal, so a pass above is not vacuous."""
    if not SNAP.is_dir():
        pytest.skip(f"{SNAP} is not on this host")
    (tmp_path / "neg").mkdir()
    outcome, _ = _argv_outcome(tmp_path / "neg", monkeypatch, "--checkpoint-skip-layers", "-1")
    assert "must not be negative" in outcome


def test_the_skip_6_set_overlaps_the_lower_8_group_exactly_at_3_and_7():
    types = (["linear_attention"] * 3 + ["full_attention"]) * 6
    skip = backbone.uncheckpointed_layers(6, types)
    assert skip == (3, 7, 11, 15, 19, 23)
    assert [i for i in skip if i < 8] == [3, 7]


def _f_like(where: Path, monkeypatch, *, skip: int, lower: int, n: int = 6) -> dict:
    snap = where / "snap"
    tb._write_tiny_snapshot(snap, vocab=256)
    spec = tb._tiny_spec(256)
    real_load = backbone.load_text_tower
    monkeypatch.setattr(
        backbone, "load_text_tower", lambda *a, **k: real_load(*a, **{**k, "spec": spec})
    )
    master = OptimizerSpec(*tb.MASTER_SPEC_ARGS, keeps_fp32_master=True)
    plan = [tb._ft_batch(i) for i in range(n)]
    step, tower, _ = rft._real_step(
        backbone=snap, reader=SimpleNamespace(remap=_remap()), plan=plan, device="cpu",
        dtype="bf16", spec=master, attn_implementation="sdpa", seed=0, lr=1e-3,
        total_steps=n, span_weight=1.0, width=6, lower_layers_n=lower,
        lower_lr_scale=0.1 if lower else 1.0, checkpoint_skip_layers=skip,
    )
    monkeypatch.setattr(backbone, "load_text_tower", real_load)
    flags = [bool(layer.gradient_checkpointing) for layer in step.tower.model.layers]
    groups = [(g.get("name"), g.get("lr_scale")) for g in step.optimizer.param_groups]
    result = train_ft(
        (tb._ft_batch(i) for i in range(n)), epoch=0, step=step, control=tb._control(n),
        recorder=tb._recorder(where / "rec"),
    )
    return {
        "digest": result.loss_log.digest(),
        "losses": result.loss_log.losses(),
        "letter": list(step.letter_log),
        "span": list(step.span_log),
        "consumed": result.checkpoint.consumed_digest,
        "weights": {k: v.detach().clone() for k, v in step.tower.model.state_dict().items()},
        "head": {k: v.detach().clone() for k, v in step.span_head.state_dict().items()},
        "skip_layers": tuple(tower.checkpoint_skip_layers),
        "flags": flags,
        "groups": groups,
    }


def _same(a: dict, b: dict) -> bool:
    keys = ("digest", "losses", "letter", "span", "consumed")
    if any(a[k] != b[k] for k in keys):
        return False
    return all(
        torch.equal(a[t][k], b[t][k]) for t in ("weights", "head") for k in a[t]
    ) and a["weights"].keys() == b["weights"].keys()


@pytest.mark.parametrize("skip", [2, tb.TINY_LAYERS])
def test_skip_with_the_layerwise_split_leaves_the_trajectory_bit_identical(
    tmp_path, monkeypatch, skip
):
    lower = 2
    base = _f_like(tmp_path / "base", monkeypatch, skip=0, lower=lower)
    other = _f_like(tmp_path / f"skip{skip}", monkeypatch, skip=skip, lower=lower)
    overlap = [i for i in other["skip_layers"] if i < lower]
    print(f"skip={skip} layers={other['skip_layers']} flags={other['flags']} "
          f"in lower group={overlap} groups={other['groups']}")
    assert overlap, "a skipped layer must sit inside the lower group, as 3 and 7 do in F"
    assert ("base_lower", 0.1) in other["groups"] and ("base", 1.0) in other["groups"]
    assert other["flags"] == [i not in other["skip_layers"] for i in range(tb.TINY_LAYERS)]
    assert any(x > 0.0 for x in base["span"]), "the span channel must be exercised"
    assert _same(base, other), "skip moved a bit with the split on"
    # The split is live: the same run without it lands elsewhere, so the identity above is
    # not two runs that both ignored lower_layers_n.
    flat = _f_like(tmp_path / f"flat{skip}", monkeypatch, skip=skip, lower=0)
    assert flat["digest"] != other["digest"]
