"""Evidence script for lane L-v5plan (2026-10-02). Throwaway accounting, never shipped.

Measures F's v4 train shard set (/Users/bharath/qd-campaign/phase4-v4-2026-10-01/shards/train,
shard_hash 8bcf56ad...) by category: sequences, real tokens, and positions at bucket width (each
sequence padded to the smallest header bucket that holds it, as the batch planner pads). Reads
offsets.npy, sequence_index.json and header.json only; tokens.u32 is not read.

Then projects v5's tokens and positions per epoch under the plan's data changes (HANDOFF
v5-plan-2026-10-02.md section 1). Every projected number is labelled inferred; the measured
v4 numbers reproduce row d96409bd's shard_sequences (363,950 sequences, 306,926,895 tokens)
and padding_waste (323,073,710 positions) metrics, which is the check that the measurement is
the shard set F trained on.
"""

import bisect
import collections
import json
import sys
from pathlib import Path

import numpy as np

SHARDS = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01/shards/train")
OUT = Path(__file__).with_name("v4_token_accounting.json")


def category(row_id: str) -> str:
    if ":compose:" in row_id:
        return "defect.composed"
    head = row_id.split(":", 2)
    prefix, fam = head[0], (head[1] if len(head) > 1 else "")
    if fam == "code.defect_class":
        return "defect.noul" if "noul" in row_id else f"defect.{prefix}"
    return f"{fam}"


def main() -> int:
    header = json.loads((SHARDS / "header.json").read_text())
    buckets = list(header["buckets"])
    index = json.loads((SHARDS / "sequence_index.json").read_text())
    seqs = index["sequences"]
    offsets = np.load(SHARDS / "offsets.npy")
    lengths = np.diff(offsets).astype(np.int64)
    if len(lengths) != len(seqs):
        raise SystemExit(f"{len(lengths)} offsets vs {len(seqs)} index entries")
    by = collections.defaultdict(lambda: {"sequences": 0, "tokens": 0, "positions": 0})
    rows = collections.defaultdict(set)
    composed_len_hist = collections.Counter()
    for s, n in zip(seqs, lengths.tolist(), strict=False):
        cat = category(s["row_id"])
        b = buckets[bisect.bisect_left(buckets, n)]
        rec = by[cat]
        rec["sequences"] += 1
        rec["tokens"] += n
        rec["positions"] += b
        rows[cat].add(s["row_id"])
        if cat == "defect.composed":
            composed_len_hist[f"{1000 * (n // 1000):05d}"] += 1
    total = {k: sum(v[k] for v in by.values()) for k in ("sequences", "tokens", "positions")}
    measured = {
        "shard_dir": str(SHARDS),
        "buckets": buckets,
        "by_category": {k: {**v, "rows": len(rows[k])} for k, v in sorted(by.items())},
        "total": total,
        "padding_share": 1 - total["tokens"] / total["positions"],
        "composed_sequence_length_hist_1k": dict(sorted(composed_len_hist.items())),
    }

    # -- v5 projection (inferred) ------------------------------------------------------------
    comp = by["defect.composed"]
    comp_rows = len(rows["defect.composed"])
    v4_comp_mean_tokens_per_seq = comp["tokens"] / comp["sequences"]
    seq_per_comp_row = comp["sequences"] / comp_rows
    # v4 composed real lengths run ~0.9k-7.9k (estimate band 850-7,300, hard 7,450 est).
    # v5 stretches the band so real lengths run ~0.9k-9.9k (est 850-9,100, hard 9,400 est):
    # the mean scales with the band's midpoint under a uniform target.
    v4_band = (900, 7_936)
    v5_band = (900, 9_900)
    stretch = (sum(v5_band) / 2) / (sum(v4_band) / 2)
    v5_comp_rows = 25_000
    v5_comp_tokens = v5_comp_rows * seq_per_comp_row * v4_comp_mean_tokens_per_seq * stretch
    non_comp_tokens = total["tokens"] - comp["tokens"]
    non_comp_seqs = total["sequences"] - comp["sequences"]
    # Added prose-noul rows: 4,000 (2,000 own-repo, 2,000 MMLU/CSQA contrast), measured at the
    # v4 noul rows' own mean tokens per sequence and sequences per row.
    noul_cats = [
        k for k in by if k.startswith("defect.") and k not in ("defect.composed", "defect.qdm")
    ]
    noul_tok = sum(by[k]["tokens"] for k in noul_cats)
    noul_seq = sum(by[k]["sequences"] for k in noul_cats)
    noul_rows = sum(len(rows[k]) for k in noul_cats)
    add_noul_rows = 4_000
    add_noul_tokens = (
        add_noul_rows * (noul_seq / noul_rows) * (noul_tok / noul_seq) if noul_rows else 0
    )
    # CLINC re-key: 4 families x 134 oos rows leave train (clinc_oos_rekey.json); ~2 seqs? no:
    # each CLINC row is one choice sequence. Measured mean tokens per CLINC sequence.
    clinc = [k for k in by if k.startswith("intent.")]
    clinc_mean = sum(by[k]["tokens"] for k in clinc) / sum(by[k]["sequences"] for k in clinc)
    clinc_delta_tokens = -4 * 134 * clinc_mean
    # E_val exclusions: 1,196 MMLU/CSQA train rows (gold-side summary), one sequence each.
    mc = [k for k in by if k.endswith("multiple_choice")]
    mc_mean = sum(by[k]["tokens"] for k in mc) / sum(by[k]["sequences"] for k in mc)
    eval_delta_tokens = -1_196 * mc_mean
    v5_tokens = (
        non_comp_tokens + v5_comp_tokens + add_noul_tokens + clinc_delta_tokens + eval_delta_tokens
    )
    pad = measured["padding_share"]
    v5_positions = v5_tokens / (1 - pad)
    f_steps = 9_683
    f_pos = total["positions"]
    projection = {
        "label": "inferred: v4 measurements scaled by the plan's declared deltas",
        "composed": {
            "v4_rows": comp_rows,
            "v4_sequences_per_row": seq_per_comp_row,
            "v4_mean_tokens_per_sequence": v4_comp_mean_tokens_per_seq,
            "v4_real_band": v4_band,
            "v5_real_band": v5_band,
            "length_stretch_factor": stretch,
            "v5_rows": v5_comp_rows,
            "v5_tokens": v5_comp_tokens,
        },
        "non_composed_v4_tokens": non_comp_tokens,
        "non_composed_v4_sequences": non_comp_seqs,
        "added_noul": {"rows": add_noul_rows, "tokens": add_noul_tokens,
                       "v4_noul_rows": noul_rows, "v4_noul_categories": noul_cats},
        "clinc_rekey_tokens_delta": clinc_delta_tokens,
        "e_val_exclusion_tokens_delta_lower_bound": eval_delta_tokens,
        "v5_tokens_per_epoch": v5_tokens,
        "v5_positions_per_epoch_at_v4_padding": v5_positions,
        "ratio_to_v4_positions": v5_positions / f_pos,
        "v5_steps_at_f_positions_per_step": f_steps * v5_positions / f_pos,
    }
    out = {"measured_v4": measured, "projected_v5": projection}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
