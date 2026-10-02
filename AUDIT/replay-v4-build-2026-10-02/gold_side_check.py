"""Gold-side containment check on v4, giving E_val (Fable's J6(a) ruling, Q6 item 1).

Not a training process: like tools/replay_decontam.py, it rebuilds the v4 splits with
real_ft_run.ft_splits, renders rows and compares text, and trains nothing. It reads the
held-out split as a comparison target only (rule 3 binds training processes).

Every MMLU/CSQA train row of v4 (knowledge.multiple_choice + commonsense.multiple_choice,
23,819 expected), rendered through qd_data.render + qd_train.replay.prompt_content exactly
as replay_decontam.row_texts renders a target, is compared against every val and held-out
row with qd_train.replay.decontaminate (word 8-grams, containment >= 0.5). Every pair at or
over the threshold is written (DecontamReport.pairs, the full list, not the best target per
row). E_val is the set of val ``row_id#slot`` keys hit by any train row, written in the form
qd-gate-report --exclude-rows reads: UTF-8, LF, one key per line, sorted, no duplicates.

Usage: python gold_side_check.py --lane LANE_ROOT --out-dir DIR [--replay-hits FILE]
Run with QD_PREP_BIN set (ft_splits signs with qd-prep).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

V4_REV = "881ab304f15ea13529002391dda8520c2ea47af4"
REC = Path("/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json")
#: v4's val shard set, whose sequence index says which val (row_id, slot) F's eval decodes.
V4_VAL_INDEX = Path(
    "/Users/bharath/qd-campaign/phase4-v4-2026-10-01/shards/val/sequence_index.json"
)
EXPECTED_TRAIN_MC = 23_819


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--replay-hits", type=Path, default=None,
                    help="decontam 1's --hits-out list: its val pairs must all appear here")
    args = ap.parse_args()
    lane = args.lane.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; refusing to overwrite a pinned result")
    t0 = time.monotonic()
    os.chdir(lane)
    sys.path.insert(0, str(lane / "python"))
    sys.path.insert(0, str(lane / "tools"))

    import replay_decontam
    from real_ft_run import ft_splits, replay_corpus_identity
    from repo_git import resolve_rev

    from qd_data.config import DataConfig
    from qd_data.general import REPLAY_FAMILIES
    from qd_train.replay import DEFAULT_N, DEFAULT_THRESHOLD, decontaminate

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=lane, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=lane, capture_output=True,
                           text=True, check=True).stdout.strip()
    config = DataConfig()
    rev = resolve_rev(lane, V4_REV)
    corpus_kw = dict(
        commitpackft=None, max_pairs=400, rev=rev, config=config,
        defect_class=Path("data/pool/commitpackft-composed-v1"),
        defect_download=Path("data/pool/commitpackft"), defect_max_rows=None,
        repo_history=False, general_record=REC, general_max_rows=200_000,
        defect_noul=Path("data/pool/defect-noul-v3b"),
    )
    splits = ft_splits(**corpus_kw)
    train_mc = [r for r in splits["train"] if r.family_id in REPLAY_FAMILIES]
    if len(train_mc) != EXPECTED_TRAIN_MC:
        raise SystemExit(
            f"{len(train_mc)} MMLU/CSQA train rows, not v4's {EXPECTED_TRAIN_MC}: not v4's corpus"
        )
    identity = {r.row_id: r.identity_key for r in train_mc}
    family = {r.row_id: r.family_id for r in train_mc}
    train_texts, train_unrenderable = replay_decontam.row_texts(train_mc)
    if train_unrenderable:
        raise SystemExit(f"{len(train_unrenderable)} MMLU/CSQA train rows do not render")
    targets, unrenderable = replay_decontam.target_texts(splits)
    report = decontaminate(train_texts, targets, n=DEFAULT_N, threshold=DEFAULT_THRESHOLD)

    pairs = []
    for p in report.pairs:
        row_id, slot = p.replay_row.rsplit("#", 1)
        pairs.append({
            "train_key": p.replay_row, "row_id": row_id, "slot_name": slot,
            "identity_key": identity[row_id], "family_id": family[row_id],
            "target": p.target, "target_row": p.target_row, "shared": p.shared,
            "target_ngrams": p.target_ngrams, "containment": p.containment,
        })
    e_val = sorted({p.target_row for p in report.pairs if p.target == "val"})
    val_index = json.loads(V4_VAL_INDEX.read_text(encoding="utf-8"))
    decoded = {f"{s['row_id']}#{s['slot_name']}" for s in val_index["sequences"]}
    not_decoded = sorted(k for k in e_val if k not in decoded)

    out_dir.mkdir(parents=True)
    pairs_path = out_dir / "gold-side-pairs.json"
    pairs_path.write_text(json.dumps({
        "what": "every (v4 MMLU/CSQA train row, val or held-out row) pair at or over the "
                "threshold; replay_row in DecontamReport is the train key row_id#slot",
        "n": report.n, "threshold": report.threshold, "pairs": pairs,
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    e_val_path = out_dir / "e-val.txt"
    e_val_path.write_bytes("".join(k + "\n" for k in e_val).encode("utf-8"))

    cross = None
    if args.replay_hits is not None:
        hits = json.loads(args.replay_hits.read_text(encoding="utf-8"))
        gold_val = {(q["row_id"], q["target_row"]) for q in pairs if q["target"] == "val"}
        replay_val = {(q["row_id"], q["target_row"]) for q in hits["pairs"] if q["target"] == "val"}
        cross = {
            "replay_hits": str(args.replay_hits), "replay_hits_sha256": _sha(args.replay_hits),
            "replay_val_pairs": len(replay_val),
            "replay_val_pairs_missing_here": len(replay_val - gold_val),
            "replay_rows_hit": hits["replay_rows_hit"],
        }

    by_target_rows = {
        name: len({q["train_key"] for q in pairs if q["target"] == name}) for name in targets
    }
    summary = {
        "tool": "AUDIT/replay-v4-build-2026-10-02/gold_side_check.py",
        "lane_commit": commit, "lane_dirty": bool(dirty),
        "corpus": replay_corpus_identity(
            rev=rev, max_pairs=400, commitpackft=None, defect_class=corpus_kw["defect_class"],
            defect_max_rows=None, repo_history=False, general_record=REC,
            general_max_rows=200_000, defect_noul=corpus_kw["defect_noul"],
        ),
        "split_rows": {k: len(v) for k, v in splits.items()},
        "train_mc_rows": len(train_mc),
        "train_mc_by_family": {f: sum(1 for r in train_mc if r.family_id == f)
                               for f in sorted({r.family_id for r in train_mc})},
        "train_texts": len(train_texts),
        "train_rows_too_short": report.replay_rows_too_short,
        "train_rows_checked": report.replay_rows_checked,
        "targets": report.targets,
        "unrenderable_targets": {k: len(v) for k, v in unrenderable.items()},
        "hits_rows": report.hits,
        "train_keys_with_a_pair_by_target": by_target_rows,
        "train_keys_with_any_pair": len({q["train_key"] for q in pairs}),
        "pairs_total": len(pairs),
        "pairs_by_target": {name: sum(1 for q in pairs if q["target"] == name) for name in targets},
        "e_val_size": len(e_val),
        "e_val_by_family_prefix": {
            pre: sum(1 for k in e_val if k.startswith(pre + ":")) for pre in ("mmlu", "csqa")
        },
        "e_val_keys_not_in_v4_val_shards": len(not_decoded),
        "e_val_keys_not_in_v4_val_shards_list": not_decoded,
        "v4_val_index_sha256": _sha(V4_VAL_INDEX),
        "heldout_target_rows_hit": len(
            {q["target_row"] for q in pairs if q["target"] == "heldout"}
        ),
        "pairs_sha256": _sha(pairs_path),
        "e_val_sha256": _sha(e_val_path),
        "cross_check_with_replay_hits": cross,
        "wall_s": round(time.monotonic() - t0, 1),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("e_val_keys_not_in_v4_val_shards_list", "targets")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
