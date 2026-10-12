"""Fit arm (c) or arm (e) of campaign/span-refit-mac-preregistered.json on cached train features.

Throwaway analysis (house Python policy, category 3): it never ships. A separate process
from the extraction: the tower is never loaded here.

The pre-registered fit, exactly:

* score of candidate j: ``(W h_q) . h_j``; abstain: ``u . h_q`` with ``u = W^T a`` (the
  convex reparametrisation: logits linear in W and u); softmax over n_lines + 1; mean CE over
  the start and the end pointers (2N decisions);
* full-rank ``W_start, W_end`` [H, H] and ``u_start, u_end`` [H];
* ridge ``lambda * (||W||^2 + ||u||^2)`` summed over both pointers; lambda from
  {1e-4, 1e-3, 1e-2, 1e-1} chosen by (unregularised) CE on the train-internal 10,000 / 2,000
  split the extraction recorded -- never val;
* torch.optim.LBFGS, full batch, max 500 iterations per lambda; final fit on all 12,000 at
  the chosen lambda; recovery ``a = W^{-T} u``; cond(W) reported; the RECOVERED (W, a) head
  is what is written and scored.

Arm (c) reads ``cand_start`` (line starts, as served); arm (e) reads ``cand_end`` (each line's
last token). Both from the same averaged-tower pass, same subsample, same grid, same rule.

The gradient is analytic (the objective is a sum of per-row log-sum-exps of linear scores),
computed in bounded row chunks; span_refit_test.py checks it against autograd.

Output: ``span_head.safetensors`` holding ``span_head.start_proj.weight``,
``span_head.end_proj.weight``, ``span_head.abstain_start``, ``span_head.abstain_end`` (float32,
the averaged file's layout and dtypes) and ``span_head.manifest.json`` in the shape
tools/real_ft_run.py ``read_span_head_init`` accepts (so the arm-(c) head is a
``--span-head-init`` file), plus the fit's record.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_refit_common as C  # noqa: E402

C.repo_imports()

LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
N_INNER_FIT, N_INNER_SELECT = 10_000, 2_000
MAX_ITER = 500


@dataclass
class PointerData:
    """Rows of one fit set: query states, ragged candidates, gold columns for both pointers."""

    hq: torch.Tensor          # [N, H] fp32
    cand: torch.Tensor        # [M, H] fp32
    offsets: torch.Tensor     # [N + 1] int64
    row_of: torch.Tensor      # [M] int64
    gold_start: torch.Tensor  # [N] int64, n_cand for an abstaining row
    gold_end: torch.Tensor    # [N] int64
    n_cand: torch.Tensor      # [N] int64

    @property
    def n(self) -> int:
        return int(self.hq.shape[0])

    def to(self, device: str) -> "PointerData":
        return PointerData(*(getattr(self, f).to(device) for f in (
            "hq", "cand", "offsets", "row_of", "gold_start", "gold_end", "n_cand")))


def make_data(hq: torch.Tensor, cand_rows: list[torch.Tensor], gold_start, gold_end) -> PointerData:
    counts = torch.as_tensor([c.shape[0] for c in cand_rows], dtype=torch.int64)
    offsets = torch.zeros(len(cand_rows) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, 0)
    row_of = torch.repeat_interleave(torch.arange(len(cand_rows)), counts)
    return PointerData(
        hq=hq.float().contiguous(), cand=torch.cat(cand_rows, 0).float().contiguous(),
        offsets=offsets, row_of=row_of,
        gold_start=torch.as_tensor(gold_start, dtype=torch.int64),
        gold_end=torch.as_tensor(gold_end, dtype=torch.int64), n_cand=counts,
    )


def pointer_ce(
    W: torch.Tensor, u: torch.Tensor, d: PointerData, gold: torch.Tensor, *,
    grad: bool, chunk_rows: int = 1024,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    """Summed CE of one pointer over all rows of ``d``, and (optionally) its gradient in W, u.

    Logits per row: ``s_m = (W h_q) . c_m`` over its candidates, then ``a = u . h_q`` last.
    """
    total = 0.0
    dW = torch.zeros_like(W) if grad else None
    du = torch.zeros_like(u) if grad else None
    for lo in range(0, d.n, chunk_rows):
        hi = min(d.n, lo + chunk_rows)
        m_lo, m_hi = int(d.offsets[lo]), int(d.offsets[hi])
        hq = d.hq[lo:hi]
        cc = d.cand[m_lo:m_hi]
        r = d.row_of[m_lo:m_hi] - lo
        q = hq @ W.T
        s = (q[r] * cc).sum(1)
        a = hq @ u
        mx = a.scatter_reduce(0, r, s, reduce="amax", include_self=True)
        ex = torch.exp(s - mx[r])
        ea = torch.exp(a - mx)
        se = ea.index_add(0, r, ex)
        lse = mx + torch.log(se)
        g = gold[lo:hi]
        abst = g == d.n_cand[lo:hi]
        local = (d.offsets[lo:hi] - m_lo) + torch.where(abst, torch.zeros_like(g), g)
        gold_score = torch.where(abst, a, s[local.clamp(max=max(s.shape[0] - 1, 0))])
        total += float((lse - gold_score).sum().double())
        if grad:
            ds = ex / se[r]
            pt = torch.nonzero(~abst).flatten()
            ds = ds.index_add(0, local[pt], -torch.ones(pt.shape[0], dtype=ds.dtype, device=ds.device))
            da = ea / se - abst.to(ea.dtype)
            dq = torch.zeros_like(q).index_add_(0, r, ds.unsqueeze(1) * cc)
            dW += dq.T @ hq
            du += hq.T @ da
    return total, dW, du


def objective(params: list[torch.Tensor], d: PointerData, lam: float, *, grad: bool):
    """(mean CE over 2N pointer decisions) + lam * sum of squared norms; with its gradient."""
    Ws, We, us, ue = params
    ls, gWs, gus = pointer_ce(Ws, us, d, d.gold_start, grad=grad)
    le, gWe, gue = pointer_ce(We, ue, d, d.gold_end, grad=grad)
    scale = 1.0 / (2 * d.n)
    ce = (ls + le) * scale
    reg = sum(float((p.double() ** 2).sum()) for p in params)
    value = ce + lam * reg
    grads = None
    if grad:
        grads = [g * scale + 2 * lam * p for g, p in zip((gWs, gWe, gus, gue), params)]
    return value, ce, grads


def fit(d: PointerData, lam: float, *, hidden: int, device: str, max_iter: int = MAX_ITER,
        history_size: int = 100) -> tuple[list[torch.Tensor], dict[str, Any]]:
    params = [torch.zeros((hidden, hidden), device=device), torch.zeros((hidden, hidden), device=device),
              torch.zeros(hidden, device=device), torch.zeros(hidden, device=device)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, history_size=history_size,
                            line_search_fn="strong_wolfe")
    t0 = time.monotonic()
    last: dict[str, float] = {}

    def closure():
        with torch.no_grad():
            value, ce, grads = objective([p.detach() for p in params], d, lam, grad=True)
            for p, g in zip(params, grads):
                p.grad = g
        last["value"], last["ce"] = value, ce
        return torch.tensor(value, dtype=torch.float32)

    opt.step(closure)
    state = opt.state[opt._params[0]]
    gnorm = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params)))
    record = {
        "lambda": lam, "n_iter": int(state.get("n_iter", -1)),
        "func_evals": int(state.get("func_evals", -1)), "max_iter": max_iter,
        "hit_max_iter": int(state.get("n_iter", -1)) >= max_iter,
        "final_objective": last.get("value"), "final_train_ce": last.get("ce"),
        "final_grad_norm": gnorm, "wall_s": round(time.monotonic() - t0, 1),
        "optimizer": {"name": "torch.optim.LBFGS", "lr": 1.0, "history_size": history_size,
                      "line_search_fn": "strong_wolfe", "tolerance_grad": 1e-7,
                      "tolerance_change": 1e-9, "init": "zeros"},
    }
    return [p.detach() for p in params], record


def recover(W: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """``a = W^{-T} u`` (servable abstain vector), in float64; cond(W) and the residual."""
    Wd, ud = W.double().cpu(), u.double().cpu()
    a = torch.linalg.solve(Wd.T, ud)
    resid = float(torch.linalg.norm(Wd.T @ a - ud) / max(float(torch.linalg.norm(ud)), 1e-300))
    return a.float(), {"cond": float(torch.linalg.cond(Wd)), "recovery_relative_residual": resid}


# --- cache -> data -----------------------------------------------------------------------------


def load_fit_cache(root: Path, arm: str, *, allow_smoke: bool) -> tuple[C.Cache, dict[str, Any]]:
    prereg = C.load_prereg()
    cache = C.open_cache(root)
    lay = cache.layout
    if lay.get("split") != "train" or lay.get("tower") != "avg":
        raise SystemExit(f"{root}: the fit reads the averaged tower's TRAIN cache only (rule 3); "
                         f"this is tower={lay.get('tower')} split={lay.get('split')}")
    if lay["shard_hash"] != prereg["inputs"]["train_shard_for_fit"]["shard_hash"]:
        raise SystemExit(f"{root}: shard {lay['shard_hash']} is not the prereg's train shard")
    if lay["weights"]["sha256"] != prereg["inputs"]["averaged_tower"]["sha256"]:
        raise SystemExit(f"{root}: features are not from the prereg's averaged tower")
    if not cache.complete():
        raise SystemExit(f"{root}: {len(cache.rows)} of {cache.n_rows} rows extracted")
    smoke = lay.get("limit") is not None
    if smoke and not allow_smoke:
        raise SystemExit(f"{root} is a --limit smoke cache; pass --smoke to fit it as a smoke")
    if not smoke and cache.n_rows != int(prereg["fit_set"]["n_sequences"]):
        raise SystemExit(f"{root}: {cache.n_rows} rows, the prereg fits 12,000")
    return cache, prereg


def build_data(cache: C.Cache, arm: str, slots: list[int]) -> PointerData:
    rows = cache.rows_in_layout_order()
    offs = cache.layout["cand_offsets"]
    hq_all = cache.hq()
    mm = cache.cands("start" if arm == "c" else "end")
    hq = torch.from_numpy(np.asarray(hq_all[slots], dtype=np.float32))
    cand_rows = [C.from_bf16_bits(mm[offs[s]:offs[s + 1]]) for s in slots]
    return make_data(hq, cand_rows, [rows[s]["gold_start"] for s in slots],
                     [rows[s]["gold_end"] for s in slots])


def write_head(out: Path, params: list[torch.Tensor], a_s: torch.Tensor, a_e: torch.Tensor,
               manifest_extra: dict[str, Any]) -> Path:
    from safetensors.torch import save
    from qd_train.run_control import _sidecar_digest, tensor_refs_from_safetensors

    Ws, We = params[0].cpu().float().contiguous(), params[1].cpu().float().contiguous()
    tensors = {
        "span_head.start_proj.weight": Ws, "span_head.end_proj.weight": We,
        "span_head.abstain_start": a_s.cpu().float().contiguous(),
        "span_head.abstain_end": a_e.cpu().float().contiguous(),
    }
    payload = save(tensors)
    path = out / "span_head.safetensors"
    path.write_bytes(payload)
    refs = tensor_refs_from_safetensors(payload)
    manifest = {
        "what": "span-refit head (campaign/span-refit-mac-preregistered.json); a "
                "--span-head-init file (tools/real_ft_run.py read_span_head_init)",
        "file": {"path": str(path), "bytes": len(payload),
                 "sha256": C.hashlib.sha256(payload).hexdigest()},
        "content_digest": {"value": _sidecar_digest(refs),
                           "formula": "run_control._sidecar_digest over TensorRef digests",
                           "names": "span_head.*", "dtype": "float32"},
        "tensors": {name: {"dtype": ref.dtype, "shape": list(ref.shape), "nbytes": len(ref.data),
                           "digest": ref.digest()} for name, ref in sorted(refs.items())},
        "construction": {"seed": int(manifest_extra["seed"])},
        **manifest_extra,
    }
    mpath = out / "span_head.manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return mpath


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=("c", "e"), required=True)
    p.add_argument("--cache", type=Path, default=C.CACHE / "avg-train")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--smoke", action="store_true", help="fit a --limit cache; output marked smoke")
    args = p.parse_args(argv)
    if args.max_iter != MAX_ITER and not args.smoke:
        raise SystemExit("the prereg fixes max 500 L-BFGS iterations; --max-iter is smoke-only")
    t0 = time.monotonic()
    cache, prereg = load_fit_cache(args.cache, args.arm, allow_smoke=args.smoke)
    out = args.out or (C.BUILD / "heads" / (f"arm-{args.arm}" + ("-smoke" if args.smoke else "")))
    out.mkdir(parents=True, exist_ok=True)
    lay = cache.layout
    slot_of = {int(s): k for k, s in enumerate(lay["order"])}
    if args.smoke:
        k = cache.n_rows
        inner_fit = list(range(0, k - max(1, k // 6)))
        inner_sel = list(range(k - max(1, k // 6), k))
    else:
        split = lay["selection"]["inner_split"]
        inner_fit = sorted(slot_of[i] for i in split["inner_fit"])
        inner_sel = sorted(slot_of[i] for i in split["inner_select"])
        if len(inner_fit) != N_INNER_FIT or len(inner_sel) != N_INNER_SELECT or set(inner_fit) & set(inner_sel):
            raise SystemExit("the cache's inner split is not 10,000 / 2,000 disjoint rows")
    status_failed = sum(1 for r in cache.rows if r["end_status"] != "ok")
    hidden = cache.hidden
    d_fit = build_data(cache, args.arm, inner_fit).to(args.device)
    d_sel = build_data(cache, args.arm, inner_sel).to(args.device)
    grid = []
    for lam in LAMBDA_GRID:
        params, rec = fit(d_fit, lam, hidden=hidden, device=args.device, max_iter=args.max_iter)
        with torch.no_grad():
            _, sel_ce, _ = objective(params, d_sel, 0.0, grad=False)
        rec["select_ce"] = sel_ce
        grid.append(rec)
        print(json.dumps({k: rec[k] for k in ("lambda", "n_iter", "final_train_ce", "select_ce",
                                              "wall_s")}), flush=True)
        del params
    chosen = min(grid, key=lambda r: (r["select_ce"], r["lambda"]))["lambda"]
    del d_fit, d_sel
    d_all = build_data(cache, args.arm, list(range(cache.n_rows))).to(args.device)
    params, final = fit(d_all, chosen, hidden=hidden, device=args.device, max_iter=args.max_iter)
    a_s, rs = recover(params[0], params[2])
    a_e, re_ = recover(params[1], params[3])
    inputs = {name: C.sha256_file(cache.root / name) for name in
              ("layout.json", "rows.jsonl", "hq.f32",
               "cand_start.bf16" if args.arm == "c" else "cand_end.bf16")}
    extra = {
        "arm": args.arm, "smoke": bool(args.smoke),
        "candidates": "line starts (plan.candidate_pos)" if args.arm == "c"
                      else "each line's last token (arm-(e) derivation); NOT SERVABLE",
        "prereg": {"path": str(C.PREREG), "sha256": C.prereg_sha256()},
        "inputs_sha256": inputs, "cache": str(cache.root),
        "train_shard_hash": lay["shard_hash"], "tower_sha256": lay["weights"]["sha256"],
        "n": cache.n_rows, "seed": int(prereg["fit_set"]["seed"]),
        "lambda_grid": grid, "lambda": chosen, "final_fit": final,
        "iterations": final["n_iter"],
        "cond_W_start": rs["cond"], "cond_W_end": re_["cond"],
        "recovery_relative_residual": {"start": rs["recovery_relative_residual"],
                                       "end": re_["recovery_relative_residual"]},
        "rows_with_arm_e_end_status_failed": status_failed,
        "wall_clock_s": round(time.monotonic() - t0, 1),
        "max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 2),
    }
    mpath = write_head(out, params, a_s, a_e, extra)
    from real_ft_run import read_span_head_init

    read_span_head_init(out / "span_head.safetensors", mpath)  # refuses unless it round-trips
    print(f"wrote {out} (lambda {chosen}, cond W_start {rs['cond']:.3g}, W_end {re_['cond']:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
