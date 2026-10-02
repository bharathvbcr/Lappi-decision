"""Decay-sensitive per-entry AdamW golden: rung (a) test 2 of ojas plan Q3 Amendment 2 (ii).

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_adamw.py \\
        --out crates/qd-train/tests/fixtures/adamw-decay-sensitive

A **fixture generator** (user policy: Python only as an oracle). L-trainer's per-entry AdamW
is checked against what it writes, at <= 1e-6 absolute per element.

Every setting comes from ``preregistration.json`` in ``--out``, committed before this script
first ran. The script refuses a pre-registration it cannot read whole; it never chooses a
setting itself.

The golden is F's optimizer, built by F's own code with nothing retyped:

* ``qd_train.optim.layerwise_param_groups`` splits the entries at ``lower_layers_n = 8`` and
  ``real_ft_run.RSI_LOWER_LR_SCALE``;
* ``qd_train.optim.build_optimizer`` with ``real_ft_run.optimizer_spec("bf16", "master")``
  returns ``MasterWeightAdamW``, as ``QwenDecisionStep`` builds it. That means bf16 tower
  entries with fp32 masters, and fp32 span-head entries that are their own masters;
* ``qd_train.optim.apply_lr`` runs before every step.

eps, weight decay and betas are read off the built optimizer, never passed in. That is the
point of ``GAP-OJAS-K11-GOLDEN-RETYPES-F-HYPER-2026-10-01``.

**Mutations.** Each pre-registered mutation is a flag on one float32 restatement of torch's
single-tensor AdamW (``torch/optim/adam.py``, the ``capturable=False`` branch: decoupled decay,
``lerp`` for m, ``mul``/``addcmul`` for v, bias corrections as Python floats, denominator
``sqrt(v) / sqrt(bc2) + eps``).

The unmutated restatement's gap to the torch golden is reported as the instrument's floor. A
mutant's ratio means something only when that floor is far below the bound.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import real_ft_run  # noqa: E402
import torch  # noqa: E402

# The one restatement of tessl's decay mask, and the fixture helpers, from the rung (b) oracle:
# imported rather than copied, so the two fixtures cannot disagree about either.
from qd_train_oracle_tiny import (  # noqa: E402
    _git_head,
    _version,
    save_safetensors,
    sha256_file,
    tessl_excluded_from_weight_decay,
)

from qd_train.optim import (  # noqa: E402
    DEFAULT_BETA2,
    LR_SCALE_KEY,
    MasterWeightAdamW,
    apply_lr,
    build_optimizer,
    layerwise_param_groups,
)

PREREGISTRATION: Final[str] = "preregistration.json"
SPAN_PREFIX: Final[str] = "span_head."
LIVE_DTYPES: Final[dict[str, torch.dtype]] = {"bf16": torch.bfloat16, "f32": torch.float32}
#: Every key the pre-registration must carry; a missing one is a refusal, not a default.
SETTING_KEYS: Final[tuple[str, ...]] = ("lr", "steps", "lower_layers_n", "data_seed")
MUTATIONS: Final[tuple[str, ...]] = (
    "D1_tessl_default_weight_decay",
    "D3_decay_not_lr_scaled",
    "eps_inside_sqrt",
    "eps_before_bias_correction",
    "lr_scale_missing",
    "D7_no_grad_stepped_as_zero",
    "D7_model_wide_step_count",
)

Trajectory = dict[int, dict[str, torch.Tensor]]


def read_preregistration(out: Path) -> dict[str, Any]:
    """The committed pre-registration, every field this script reads present and typed."""
    path = out / PREREGISTRATION
    if not path.is_file():
        raise SystemExit(f"{path}: no pre-registration; the golden is generated from one only")
    pre = json.loads(path.read_text(encoding="utf-8"))
    settings = pre.get("settings", {})
    missing = [k for k in SETTING_KEYS if k not in settings]
    if missing:
        raise SystemExit(f"{path}: settings lack {missing}")
    ids = [m.get("id") for m in pre.get("mutations", [])]
    if tuple(ids) != MUTATIONS:
        raise SystemExit(
            f"{path}: mutations are {ids}; this generator implements exactly {list(MUTATIONS)}"
        )
    for e in pre.get("entries", []):
        for key in ("name", "shape", "live_dtype", "expected_group", "base", "grad_scale"):
            if key not in e:
                raise SystemExit(f"{path}: entry {e.get('name')!r} lacks {key!r}")
        if e["live_dtype"] not in LIVE_DTYPES:
            raise SystemExit(f"{path}: entry {e['name']!r} live_dtype {e['live_dtype']!r}")
        if e["name"].startswith(SPAN_PREFIX) != (e["live_dtype"] == "f32"):
            raise SystemExit(f"{path}: {e['name']!r}: the span head is f32, the tower bf16")
    return pre


def make_inputs(
    pre: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    """Init and per-step gradients as the pre-registration describes them, in f32 (bf16-exact
    for tower entries: what the live bf16 parameter and its bf16 grad upcast to)."""
    settings = pre["settings"]
    gen = torch.Generator().manual_seed(int(settings["data_seed"]))
    init: dict[str, torch.Tensor] = {}
    for e in pre["entries"]:
        shape = tuple(e["shape"])
        if e["base"] == "zeros":
            value = torch.zeros(shape)
        else:
            value = float(e["base"]) + 0.1 * torch.randn(shape, generator=gen)
        init[e["name"]] = value.to(LIVE_DTYPES[e["live_dtype"]]).to(torch.float32)
    grads: dict[int, dict[str, torch.Tensor]] = {}
    for t in range(1, int(settings["steps"]) + 1):
        grads[t] = {}
        for e in pre["entries"]:
            if t in e["grad_steps"]:
                g = float(e["grad_scale"]) * torch.randn(tuple(e["shape"]), generator=gen)
                grads[t][e["name"]] = g.to(LIVE_DTYPES[e["live_dtype"]]).to(torch.float32)
    return init, grads


def golden(
    pre: Mapping[str, Any],
    init: Mapping[str, torch.Tensor],
    grads: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """F's optimizer over the entries, stepped ``steps`` times. Returns the per-step masters and
    moments, the per-entry table read off the built optimizer, and how torch dispatched."""
    from torch.optim.optimizer import _default_to_fused_or_foreach

    settings = pre["settings"]
    entries = pre["entries"]
    live = {
        e["name"]: torch.nn.Parameter(init[e["name"]].to(LIVE_DTYPES[e["live_dtype"]]).clone())
        for e in entries
    }
    named = [(n, p) for n, p in live.items() if not n.startswith(SPAN_PREFIX)]
    extra = [p for n, p in live.items() if n.startswith(SPAN_PREFIX)]
    groups = layerwise_param_groups(
        named,
        lower_layers_n=int(settings["lower_layers_n"]),
        lower_lr_scale=real_ft_run.RSI_LOWER_LR_SCALE,
        extra=extra,
    )
    lr = float(settings["lr"])
    steps = int(settings["steps"])
    opt = build_optimizer(
        groups,
        spec=real_ft_run.optimizer_spec("bf16", "master"),
        lr=lr,
        total_steps=steps,
        beta2=DEFAULT_BETA2,
    )
    if not isinstance(opt, MasterWeightAdamW):
        raise SystemExit(f"F's builder returned {type(opt).__name__}, not MasterWeightAdamW")
    master_of = {id(p): m for p, m in zip(opt._live, opt._masters, strict=True)}
    group_of: dict[int, dict[str, Any]] = {}
    for g in opt.param_groups:
        for m in g["params"]:
            group_of[id(m)] = g
    table: dict[str, dict[str, Any]] = {}
    for e in entries:
        g = group_of[id(master_of[id(live[e["name"]])])]
        if g.get("name") != e["expected_group"]:
            raise SystemExit(
                f"{e['name']}: F's builder put it in {g.get('name')!r}, the pre-registration "
                f"expects {e['expected_group']!r}"
            )
        table[e["name"]] = {
            "shape": list(e["shape"]),
            "live_dtype": e["live_dtype"],
            "group": g["name"],
            "lr_scale": float(g.get(LR_SCALE_KEY, 1.0)),
            "weight_decay": float(g["weight_decay"]),
            "eps": float(g["eps"]),
            "betas": [float(b) for b in g["betas"]],
            "grad_steps": list(e["grad_steps"]),
            "tessl_default_excludes": tessl_excluded_from_weight_decay(e["name"]),
        }
    inner = opt._inner
    fused, foreach = _default_to_fused_or_foreach(
        [p for g in inner.param_groups for p in g["params"]], differentiable=False, use_fused=False
    )
    weights: Trajectory = {}
    exp_avg: Trajectory = {}
    exp_avg_sq: Trajectory = {}
    for t in range(1, steps + 1):
        for name, p in live.items():
            g = grads[t].get(name)
            p.grad = None if g is None else g.to(p.dtype).clone()
        apply_lr(opt, lr)
        opt.step()
        opt.zero_grad(set_to_none=True)
        weights[t] = {n: master_of[id(p)].detach().clone() for n, p in live.items()}
        exp_avg[t], exp_avg_sq[t] = {}, {}
        for n, p in live.items():
            state = inner.state.get(master_of[id(p)], {})
            if "exp_avg" in state:
                exp_avg[t][n] = state["exp_avg"].detach().clone()
                exp_avg_sq[t][n] = state["exp_avg_sq"].detach().clone()
    for n, p in live.items():
        state = inner.state.get(master_of[id(p)], {})
        table[n]["adamw_steps_taken"] = int(state["step"]) if "step" in state else 0
    return {
        "weights": weights,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "table": table,
        "optimizer": f"{type(opt).__module__}.{type(opt).__qualname__}",
        "inner": f"{type(inner).__module__}.{type(inner).__qualname__}",
        "torch_dispatch_on_this_host": {"fused": bool(fused), "foreach": bool(foreach)},
    }


def restated(
    pre: Mapping[str, Any],
    init: Mapping[str, torch.Tensor],
    grads: Mapping[int, Mapping[str, torch.Tensor]],
    table: Mapping[str, Mapping[str, Any]],
    mutation: str | None,
) -> Trajectory:
    """torch's single-tensor AdamW (capturable=False) restated in f32 per entry, with one
    pre-registered mutation switched on, or none."""
    if mutation is not None and mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}")
    settings = pre["settings"]
    lr = float(settings["lr"])
    w = {n: v.clone() for n, v in init.items()}
    m = {n: torch.zeros_like(v) for n, v in init.items()}
    v2 = {n: torch.zeros_like(v) for n, v in init.items()}
    count = dict.fromkeys(init, 0)
    out: Trajectory = {}
    for t in range(1, int(settings["steps"]) + 1):
        for n in init:
            row = table[n]
            g = grads[t].get(n)
            if g is None:
                if mutation != "D7_no_grad_stepped_as_zero":
                    continue
                g = torch.zeros_like(w[n])
            count[n] += 1
            step_n = t if mutation == "D7_model_wide_step_count" else count[n]
            scale = 1.0 if mutation == "lr_scale_missing" else float(row["lr_scale"])
            group_lr = lr * scale  # apply_lr: float(lr) * scale
            wd = float(row["weight_decay"])
            if mutation == "D1_tessl_default_weight_decay" and row["tessl_default_excludes"]:
                wd = 0.0
            decay_lr = lr if mutation == "D3_decay_not_lr_scaled" else group_lr
            beta1, beta2 = (float(b) for b in row["betas"])
            eps = float(row["eps"])
            if wd != 0:
                w[n].mul_(1 - decay_lr * wd)
            m[n].lerp_(g, 1 - beta1)
            v2[n].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bias_correction1 = 1 - beta1**step_n
            bias_correction2 = 1 - beta2**step_n
            step_size = group_lr / bias_correction1
            bias_correction2_sqrt = bias_correction2**0.5
            if mutation == "eps_inside_sqrt":
                denom = (v2[n] / bias_correction2).add_(eps).sqrt_()
            elif mutation == "eps_before_bias_correction":
                denom = (v2[n].sqrt() + eps) / bias_correction2_sqrt
            else:
                denom = (v2[n].sqrt() / bias_correction2_sqrt).add_(eps)
            w[n].addcdiv_(m[n], denom, value=-step_size)
        out[t] = {n: x.clone() for n, x in w.items()}
    return out


def gap(a: Trajectory, b: Trajectory) -> dict[str, Any]:
    """The pre-registered measure: max over steps, entries and elements of |a - b|, absolute."""
    worst = (0.0, 0, "")
    for t in b:
        for n in b[t]:
            d = float((a[t][n] - b[t][n]).abs().max())
            if d > worst[0]:
                worst = (d, t, n)
    return {"max_abs": worst[0], "at_step": worst[1], "entry": worst[2]}


def flatten(prefix: str, traj: Mapping[int, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    return {f"{prefix}.step{t}.{n}": x for t, row in traj.items() for n, x in row.items()}


def build(out: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    pre = read_preregistration(out)
    bound = float(pre["measure"]["bound"])
    init, grads = make_inputs(pre)
    ref = golden(pre, init, grads)
    floor = gap(restated(pre, init, grads, ref["table"], None), ref["weights"])
    floor_ok = floor["max_abs"] <= bound / 100
    by_id = {mm["id"]: mm for mm in pre["mutations"]}
    mutations: dict[str, Any] = {}
    for mid in MUTATIONS:
        mg = gap(restated(pre, init, grads, ref["table"], mid), ref["weights"])
        ratio = mg["max_abs"] / bound
        mutations[mid] = {
            **mg,
            "ratio_over_bound": ratio,
            "predicted_ratio_at_least": by_id[mid]["predicted_ratio_at_least"],
            "clears_100x": ratio >= 100.0,
            "meets_prediction": ratio >= float(by_id[mid]["predicted_ratio_at_least"]),
        }
    never = [n for n, row in ref["table"].items() if not row["grad_steps"]]
    untouched = all(torch.equal(ref["weights"][max(ref["weights"])][n], init[n]) for n in never)

    save_safetensors(
        out / "inputs.safetensors",
        {**{f"init.{n}": x for n, x in init.items()}, **flatten("grad", grads)},
        "init.<entry> (f32; bf16-exact for tower entries) and grad.step<t>.<entry> for every "
        "step that entry has a gradient; an absent grad.step<t>.<entry> is grad None",
    )
    save_safetensors(
        out / "golden.safetensors",
        {
            **flatten("w", ref["weights"]),
            **flatten("exp_avg", ref["exp_avg"]),
            **flatten("exp_avg_sq", ref["exp_avg_sq"]),
        },
        "w.step<t>.<entry>: the fp32 master after optimizer step t; exp_avg/exp_avg_sq.step<t>"
        ".<entry>: its moments, present once the entry has stepped",
    )
    (out / "table.json").write_text(json.dumps(ref["table"], indent=1) + "\n")
    manifest: dict[str, Any] = {
        "what": "decay-sensitive per-entry AdamW golden (ojas plan Q3 Amendment 2 (ii), rung (a) "
        "test 2), F's optimizer built by F's builder at the pre-registered setting",
        "preregistration": {
            "file": PREREGISTRATION,
            "sha256": sha256_file(out / PREREGISTRATION),
        },
        "generator": {
            "script": "tools/qd_train_oracle_adamw.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "command": "PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python "
            "tools/qd_train_oracle_adamw.py --out crates/qd-train/tests/fixtures/"
            "adamw-decay-sensitive",
            "git_head": _git_head(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": _version("safetensors"),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "optimizer": {
            "class": ref["optimizer"],
            "inner": ref["inner"],
            "torch_dispatch_on_this_host": ref["torch_dispatch_on_this_host"],
            "lr": float(pre["settings"]["lr"]),
            "steps": int(pre["settings"]["steps"]),
            "lower_layers_n": int(pre["settings"]["lower_layers_n"]),
            "lower_lr_scale": real_ft_run.RSI_LOWER_LR_SCALE,
            "beta2": DEFAULT_BETA2,
        },
        "measure": pre["measure"],
        "instrument_floor": {**floor, "within_bound_over_100": floor_ok},
        "mutations": mutations,
        "all_mutations_clear_100x": all(r["clears_100x"] for r in mutations.values()),
        "d7_no_gradient_entries_bit_identical_to_init": {"entries": never, "holds": untouched},
        "contents": {
            "preregistration.json": "the settings, entries, measure, criteria and mutations, "
            "committed before this generator first ran",
            "inputs.safetensors": "init.<entry> and grad.step<t>.<entry>, f32",
            "golden.safetensors": "w/exp_avg/exp_avg_sq.step<t>.<entry>, f32, t = 1..steps",
            "table.json": "per entry, read off the built optimizer: group, lr_scale, "
            "weight_decay, eps, betas, the steps it has a gradient on, torch's per-entry step "
            "count after the run, and whether tessl's default decay mask excludes it",
        },
        "files": {},
    }
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            manifest["files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if not floor_ok:
        raise SystemExit(
            f"the f32 restatement is {floor['max_abs']:.3e} from the torch golden, over the "
            f"pre-registered {bound / 100:.0e}: every ratio below is unreliable"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args.out)
    floor = manifest["instrument_floor"]["max_abs"]
    print(f"instrument floor {floor:.3e}")
    for mid, row in manifest["mutations"].items():
        print(
            f"  {mid}: {row['max_abs']:.3e} = {row['ratio_over_bound']:.1f}x "
            f"(predicted >= {row['predicted_ratio_at_least']}) at step {row['at_step']} "
            f"{row['entry']}"
        )
    print(f"all clear 100x: {manifest['all_mutations_clear_100x']}")
    return 0 if manifest["all_mutations_clear_100x"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
