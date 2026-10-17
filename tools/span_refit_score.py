"""Score the span-refit arms on cached val features (campaign/span-refit-mac-preregistered.json).

Throwaway analysis (house Python policy, category 3): it never ships. No tower is loaded.

Counted exactly as tools/real_ft_run.py ``_decode`` counts a span row (1541-1579, the rule of
row 6af73bef): each pointer's top is the argmax over the runtime's rows (line starts, then the
abstention last; first maximum on a tie, as torch.argmax returns it); the row abstained iff
EITHER top is the abstention row; correct iff abstained == expected and, on a pointing row,
both tops are the gold columns.

Order of reading, as pre-registered:

1. the population pin over the cached val rows, else refuse;
2. arm (a), the validity gate: 5,596 of 7,238 (row 6af73bef) +/- 15. The GATE number is the
   exact one -- the averaged tower's own head decoded on fp32 hidden states during the
   extraction (``live_top_own_head``) -- because the bf16 cache rounds the candidate vectors
   and the +/-15 tolerance was measured for MPS-vs-CUDA, not for cache rounding. The cached
   (bf16) arm-(a) number is reported beside it as the cache's fidelity. Outside the gate:
   every arm is void, nothing else is computed, exit 3;
3. arms (b), (c), (e) on the cached features; (d) on the seed-1-tower cache when given;
4. per cell (family x pointing/abstain), the exact nulls, the reading thresholds and the
   paired exact McNemar tests the prereg names.

Arm (e) is void, and is not scored, when its derivation fails on more than 0.5% of the
qa.answer_span/pointing rows (prereg amendment 1; the extraction records each row's
``end_status`` under the amended rule). Every cell's failure count is reported beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_refit_common as C  # noqa: E402

C.repo_imports()

ROW_6AF73BEF = 5596
GATE_TOLERANCE = 15
TARGET_6626 = 6626  # row e1bafd6f, arm (d)'s reference
RESTORES = 6562     # 0.9066 x 7,238
ABOVE_MEAN = 5667   # 5,596 + 1.96 binomial sd
ARM_E_VOID_FRACTION = 0.005


# --- heads -------------------------------------------------------------------------------------


def head_tensors_from_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    want = ("start_proj.weight", "end_proj.weight", "abstain_start", "abstain_end")
    if sorted(state) != sorted(want):
        raise SystemExit(f"a span head holds {sorted(state)}, not {sorted(want)}")
    return {k: v.detach().to("cpu", torch.float32) for k, v in state.items()}


def content_digest(head: dict[str, torch.Tensor]) -> str:
    """The head's content digest by the checkpoint's own formula (run_control._sidecar_digest
    over TensorRefs named span_head.<k>), as real_ft_run.span_head_digest computes it."""
    from qd_train.run_control import TensorRef, _sidecar_digest

    refs = {}
    for name, t in head.items():
        host = t.contiguous()
        refs[f"span_head.{name}"] = TensorRef(
            dtype=str(host.dtype).removeprefix("torch."), shape=tuple(host.shape),
            data=host.reshape(-1).view(torch.uint8).numpy().tobytes())
    return _sidecar_digest(refs)


def shipped_head(prereg: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The averaged file's span_head.* tensors, after the file's sha256 matches the prereg."""
    from safetensors import safe_open

    spec = prereg["inputs"]["averaged_tower"]
    path = Path(spec["path"])
    got = C.sha256_file(path)
    if got != spec["sha256"]:
        raise SystemExit(f"{path} hashes to {got}, not the prereg's {spec['sha256']}")
    with safe_open(str(path), framework="pt") as fh:
        state = {k.removeprefix("span_head."): fh.get_tensor(k)
                 for k in fh.keys() if k.startswith("span_head.")}
    head = head_tensors_from_state(state)
    return head, {"source": str(path), "file_sha256": got, "content_digest": content_digest(head)}


def seed1_head(prereg: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from qd_train.backbone import revive_tensors
    from qd_train.run_control import Checkpoint
    from span_refit_extract import seed1_json

    body = seed1_json(prereg)
    sub, meta = Checkpoint.read_weights(body, subtrees=("span_head",))
    head = head_tensors_from_state(revive_tensors(sub["span_head"], where="span_head"))
    return head, {"source": str(body), "payload_digest": meta["payload_digest"],
                  "content_digest": content_digest(head)}


def refit_head(head_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from real_ft_run import read_span_head_init

    path, mpath = head_dir / "span_head.safetensors", head_dir / "span_head.manifest.json"
    init = read_span_head_init(path, mpath)  # sha256 + content digest + names + dtype
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("smoke"):
        raise SystemExit(f"{head_dir} is a smoke fit; it scores no arm")
    if manifest["prereg"]["sha256"] != C.prereg_sha256():
        raise SystemExit(f"{head_dir} was fit under prereg {manifest['prereg']['sha256']}, and "
                         f"the file now hashes to {C.prereg_sha256()}")
    head = head_tensors_from_state(dict(init.tensors))
    return head, {"source": str(path), "file_sha256": init.sha256,
                  "content_digest": init.content_digest, "lambda": manifest["lambda"],
                  "iterations": manifest["iterations"], "cond_W_start": manifest["cond_W_start"],
                  "cond_W_end": manifest["cond_W_end"], "fit_n": manifest["n"],
                  "fit_seed": manifest["seed"], "train_shard_hash": manifest["train_shard_hash"],
                  "hit_max_iter": manifest["final_fit"]["hit_max_iter"]}


# --- scoring -------------------------------------------------------------------------------------


class ValCache:
    def __init__(self, root: Path, prereg: dict[str, Any], *, tower: str) -> None:
        cache = C.open_cache(root)
        lay = cache.layout
        if lay.get("split") != "val" or lay.get("tower") != tower:
            raise SystemExit(f"{root}: expected the {tower} tower's val cache, got "
                             f"tower={lay.get('tower')} split={lay.get('split')}")
        if lay["shard_hash"] != prereg["inputs"]["val_shard"]["shard_hash"]:
            raise SystemExit(f"{root}: not the prereg's val shard")
        if not cache.complete():
            raise SystemExit(f"{root}: {len(cache.rows)} of {cache.n_rows} rows extracted")
        self.rows = cache.rows_in_layout_order()
        pin = C.population_pin(f"{r['row_id']}|{r['slot_name']}" for r in self.rows)
        if pin != C.POPULATION_PIN:
            raise SystemExit(
                f"{root}: the cached val rows hash to {pin}, not the population pin "
                f"{C.POPULATION_PIN}. Refusing: an arm scored on another population is not "
                "the pre-registered measurement")
        self.cells = [f"{r['family']}/{'abstain' if r['abstaining'] else 'pointing'}" for r in self.rows]
        want = prereg["population"]["cells"]
        got = {c: self.cells.count(c) for c in want}
        if got != want:
            raise SystemExit(f"{root}: cells {got}, the prereg pins {want}")
        self.cache = cache
        self.offsets = np.asarray(lay["cand_offsets"], dtype=np.int64)
        self.n_cand = np.asarray([r["n_candidates"] for r in self.rows])
        self.abstaining = np.asarray([bool(r["abstaining"]) for r in self.rows])
        self.gold = np.asarray([[r["gold_start"], r["gold_end"]] for r in self.rows])
        self._feat: dict[str, torch.Tensor] = {}
        self.hq = torch.from_numpy(np.array(cache.hq(), dtype=np.float32, copy=True))

    def features(self, which: str) -> torch.Tensor:
        if which not in self._feat:
            self._feat[which] = C.from_bf16_bits(np.asarray(self.cache.cands(which)))
        return self._feat[which]

    def correct_from_tops(self, tops: np.ndarray) -> np.ndarray:
        return np.asarray([
            C.span_correct(int(tops[i, 0]), int(tops[i, 1]), n_candidates=int(self.n_cand[i]),
                           abstaining=bool(self.abstaining[i]), gold_start=int(self.gold[i, 0]),
                           gold_end=int(self.gold[i, 1]))
            for i in range(len(self.rows))
        ])

    def cached_tops(self, head: dict[str, torch.Tensor], which: str) -> np.ndarray:
        feats = self.features(which)
        s = C.head_tops(head["start_proj.weight"], head["abstain_start"], self.hq, feats, self.offsets)
        e = C.head_tops(head["end_proj.weight"], head["abstain_end"], self.hq, feats, self.offsets)
        return np.stack([s, e], axis=1)

    def live_tops(self, key: str) -> np.ndarray:
        if any(key not in r for r in self.rows):
            raise SystemExit(f"{self.cache.root}: rows carry no {key}")
        return np.asarray([r[key] for r in self.rows], dtype=np.int64)

    def per_cell(self, correct: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"overall": {"n": int(correct.sum()), "n_total": int(correct.size)}}
        for cell in C.CELLS:
            m = np.asarray([c == cell for c in self.cells])
            out[cell] = {"n": int(correct[m].sum()), "n_total": int(m.sum())}
        return out


def reading(n: int) -> str:
    if n >= RESTORES:
        return "restores (>= 6,562)"
    if n >= ABOVE_MEAN:
        return "above mean head (>= 5,667)"
    return "does not"


def nulls(vc: ValCache) -> dict[str, Any]:
    uniform = {cell: 0.0 for cell in C.CELLS}
    plus = 0.0
    plus_rows = no_plus = 0
    for r, cell in zip(vc.rows, vc.cells):
        if r["abstaining"]:
            continue
        single = r["gold_start"] == r["gold_end"]
        uniform[cell] += (1.0 / r["n_candidates"]) if single else 0.0
        if r["family"] == "code.defect_class":
            plus_rows += 1
            cols = r["plus_columns"]
            if not cols:
                no_plus += 1
            elif single and r["gold_start"] in cols:
                plus += 1.0 / len(cols)
    abstain = int(vc.abstaining.sum())
    return {
        "uniform_line": {"rule": "start=end uniform over the row's lines, never abstain; exact "
                                 "expectation", "by_cell": uniform,
                         "overall": sum(uniform.values())},
        "random_plus_line": {"rule": "code.defect_class pointing rows only: start=end uniform "
                                     "over the row's '+' lines; exact expectation",
                             "cell": "code.defect_class/pointing", "expected_correct": plus,
                             "rows": plus_rows, "rows_with_no_plus_line": no_plus},
        "always_abstain": {"n": abstain, "prereg_states": 1881, "agrees": abstain == 1881},
    }


def mcnemar(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is not None:
        x, y = x[mask], y[mask]
    b = int((x & ~y).sum())
    c = int((~x & y).sum())
    p = C.exact_mcnemar(b, c)
    try:
        from scipy.stats import binomtest
        p_scipy = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    except ImportError:
        p_scipy = None
    return {"n": int(x.size), "first_only": b, "second_only": c, "p_exact_two_sided": p,
            "p_scipy_binomtest": p_scipy, "significant_at_0.05": p < 0.05,
            "first": int(x.sum()), "second": int(y.sum())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--val-cache", type=Path, default=C.CACHE / "avg-val")
    p.add_argument("--seed1-val-cache", type=Path, default=None, help="arm (d)")
    p.add_argument("--head-c", type=Path, default=None)
    p.add_argument("--head-e", type=Path, default=None)
    p.add_argument("--gate-only", action="store_true", help="arm (a) and stop")
    p.add_argument("--arms", nargs="+", choices=("a", "b", "c", "d", "e"), default=None,
                   help="write only these arms (arm a is always kept: it is the gate); every arm "
                        "is still computed, so McNemar tests that need (c) keep working")
    p.add_argument("--out", type=Path, default=C.BUILD / "results" / "span-refit-score.json")
    args = p.parse_args(argv)
    t0 = time.monotonic()
    prereg = C.load_prereg()
    prereg_sha = C.prereg_sha256()
    vc = ValCache(args.val_cache, prereg, tower="avg")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "prereg": {"path": str(C.PREREG), "sha256": prereg_sha},
        "population_pin": C.POPULATION_PIN, "val_shard_hash": vc.cache.layout["shard_hash"],
        "extraction": {"device": vc.cache.layout.get("device"),
                       "tower": vc.cache.layout["weights"]},
        "quick": True,
        "quick_reason": "prereg status: quick, report-only (rule 8): one tower, one fit seed, a "
                        "train subsample",
        "arms": {},
    }

    # Arm (a): the gate, then its cache fidelity.
    head_a, meta_a = shipped_head(prereg)
    live_a = vc.correct_from_tops(vc.live_tops("live_top_own_head"))
    cached_tops_a = vc.cached_tops(head_a, "start")
    cached_a = vc.correct_from_tops(cached_tops_a)
    n_live = int(live_a.sum())
    gate = abs(n_live - ROW_6AF73BEF) <= GATE_TOLERANCE
    agree = int((vc.live_tops("live_top_own_head") == cached_tops_a).all(axis=1).sum())
    result["arms"]["a"] = {
        "what": "shipped mean span head on the averaged tower", "head": meta_a,
        "gate": {"value_fp32_live": n_live, "target": ROW_6AF73BEF, "tolerance": GATE_TOLERANCE,
                 "passed": gate},
        "fp32_live": vc.per_cell(live_a), "bf16_cache": vc.per_cell(cached_a),
        "cache_fidelity": {"rows_whose_both_tops_agree_live_vs_cache": agree,
                           "of": len(vc.rows)},
        "reading": reading(n_live),
    }
    if not gate:
        result["void"] = (f"arm (a) decoded {n_live} of 7,238, outside 5,596 +/- 15: every arm "
                          "is void, a gap is recorded, nothing else is read (prereg arms.a.role)")
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(result["void"])
        return 3
    if args.gate_only:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"arm (a) gate passed: {n_live} (cache {int(cached_a.sum())}); wrote {args.out}")
        return 0

    correct: dict[str, np.ndarray] = {"a": cached_a}
    head_b, meta_b = seed1_head(prereg)
    correct["b"] = vc.correct_from_tops(vc.cached_tops(head_b, "start"))
    live_b = vc.correct_from_tops(vc.live_tops("live_top_seed1_head"))
    result["arms"]["b"] = {"what": "v5 seed-1 span head on the averaged tower", "head": meta_b,
                           "bf16_cache": vc.per_cell(correct["b"]),
                           "fp32_live": vc.per_cell(live_b),
                           "reading": reading(int(correct["b"].sum()))}
    if args.head_c is not None:
        head_c, meta_c = refit_head(args.head_c)
        correct["c"] = vc.correct_from_tops(vc.cached_tops(head_c, "start"))
        result["arms"]["c"] = {"what": "refit head, line-start candidates (servable)",
                               "head": meta_c, "bf16_cache": vc.per_cell(correct["c"]),
                               "reading": reading(int(correct["c"].sum()))}
    else:
        result["arms"]["c"] = {"not_run": "no --head-c given"}

    # Amendment 1's void rule: failures in the reading cell (qa.answer_span/pointing) against
    # 0.5% of that cell; every cell's failure count is reported beside the result.
    fail_by_cell = {cell: sum(1 for r, c in zip(vc.rows, vc.cells)
                              if c == cell and r["end_status"] != "ok") for cell in C.CELLS}
    qa_rows = vc.cells.count("qa.answer_span/pointing")
    failed = fail_by_cell["qa.answer_span/pointing"]
    e_void = failed > ARM_E_VOID_FRACTION * qa_rows
    result["arms"]["e"] = {"validity": {
        "rule": "prereg amendment 1: void iff derivation failures exceed 0.5% of "
                "qa.answer_span/pointing rows",
        "failures_by_cell": fail_by_cell, "qa_pointing_rows": qa_rows, "void": e_void,
        "rows_whose_last_line_used_amendment_1": sum(
            1 for r in vc.rows if r.get("last_rule") == "amendment-1")}}
    if e_void:
        result["arms"]["e"]["void"] = (
            f"the end-of-line derivation failed on {failed} of {qa_rows} qa.answer_span/pointing "
            "rows (> 0.5%): arm (e) is void; the count is reported, never read")
    elif args.head_e is not None:
        head_e, meta_e = refit_head(args.head_e)
        tops_e = vc.cached_tops(head_e, "end")
        ce = vc.correct_from_tops(tops_e)
        bad = np.asarray([r["end_status"] != "ok" for r in vc.rows])
        ce = ce & ~bad  # a row whose end could not be derived is counted wrong for (e)
        correct["e"] = ce
        result["arms"]["e"].update({"what": "refit head, last-token candidates (NOT SERVABLE)",
                                    "head": meta_e, "bf16_cache": vc.per_cell(ce),
                                    "rows_counted_wrong_for_failed_end": int(bad.sum()),
                                    "reading": reading(int(ce.sum()))})
    else:
        result["arms"]["e"]["not_run"] = "no --head-e given"

    if args.seed1_val_cache is not None:
        vd = ValCache(args.seed1_val_cache, prereg, tower="seed1")
        if [r["row_id"] + "|" + r["slot_name"] for r in vd.rows] != [
                r["row_id"] + "|" + r["slot_name"] for r in vc.rows]:
            raise SystemExit("the seed-1 val cache orders its rows differently")
        live_d = vd.correct_from_tops(vd.live_tops("live_top_own_head"))
        cached_d = vd.correct_from_tops(vd.cached_tops(head_b, "start"))
        result["arms"]["d"] = {"what": "seed-1 tower + seed-1 head (reference)", "head": meta_b,
                               "tower": vd.cache.layout["weights"],
                               "fp32_live": vd.per_cell(live_d), "bf16_cache": vd.per_cell(cached_d),
                               "reference_target": TARGET_6626,
                               "reading": reading(int(live_d.sum()))}
    else:
        result["arms"]["d"] = {"not_run": "no --seed1-val-cache given (run only after (a) passes)"}

    result["nulls"] = nulls(vc)
    tests: dict[str, Any] = {}
    if "c" in correct:
        tests["b_vs_c_all_rows (prereg)"] = mcnemar(correct["b"], correct["c"])
        tests["a_vs_c_all_rows (descriptive, not pre-registered)"] = mcnemar(correct["a"], correct["c"])
    tests["a_vs_b_all_rows (descriptive, not pre-registered)"] = mcnemar(correct["a"], correct["b"])
    if "e" in correct and "c" in correct:
        for cell in C.CELLS:
            m = np.asarray([c == cell for c in vc.cells])
            tag = "e_vs_c_" + cell + (" (prereg: the read cell)" if cell == "qa.answer_span/pointing"
                                      else " (reported, not read)")
            tests[tag] = mcnemar(correct["e"], correct["c"], m)
    result["mcnemar"] = tests
    result["wall_clock_s"] = round(time.monotonic() - t0, 1)
    result["ledger_row_drafts"] = ledger_drafts(result, vc, prereg_sha)
    if args.arms is not None:
        keep = set(args.arms) | {"a"}
        result["arms"] = {k: v for k, v in result["arms"].items() if k in keep}
        result["arms_written"] = sorted(keep)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v.get("reading") if isinstance(v, dict) else v)
                      for k, v in result["arms"].items()}, indent=1))
    print(f"wrote {args.out}")
    return 0


def ledger_drafts(result: dict[str, Any], vc: ValCache, prereg_sha: str) -> list[dict[str, Any]]:
    """What each arm's ledger row would carry. Drafts only: ledger/ is outside this lane's
    writable directory, so qd_train.ledger.RunRecorder is NOT called here."""
    drafts = []
    for arm, block in result["arms"].items():
        if "head" not in block:
            continue
        cache_key = "fp32_live" if arm in ("a", "d") else "bf16_cache"
        overall = block[cache_key]["overall"]
        drafts.append({
            "status": "DRAFT, not written to ledger/ (RunRecorder not called)",
            "quick": True, "quick_reason": result["quick_reason"],
            "recipe": {"tag": f"span-refit-{arm}", "device": vc.cache.layout.get("device"),
                       "score_dtype": "fp32",
                       "scored_checkpoint_sha256": (block.get("tower") or vc.cache.layout["weights"])["sha256"],
                       "head": block["head"], "val_shard_hash": vc.cache.layout["shard_hash"],
                       "prereg_sha256": prereg_sha},
            "metrics": {"val_top1.span": overall, "per_cell": block[cache_key],
                        "nulls": result.get("nulls")},
            "wall_clock_s": result.get("wall_clock_s"),
        })
    return drafts


if __name__ == "__main__":
    sys.exit(main())
