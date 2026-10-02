"""STOP gates for the J6(a) v4 replay builds (HANDOFF/replay-v4-plan-2026-10-02.md section 6,
campaign/j6a-preregistered.json arm.declared_data_delta). Reads manifests' and headers'
recorded fields only; for the held-out manifest it reads only data_snapshot_hash.

Usage: python gates.py BUILD_DIR LEDGER [--probe PROBE_DIR --hits HITS_JSON]
Prints one line per gate, PASS or FAIL, then a JSON summary; exits 1 on any FAIL.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

V4 = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01")
V4_ROW = "d96409bd-4890-4d7f-9e7c-ca4cafbdb9a8"
V4_LEDGER = Path("/Users/bharath/qd-campaign/ledger-mac-phase4-v4-2026-10-01.jsonl")
WANT = {
    "val_shard_hash": "ef06ab99eef107dd608424c80117b81b3986ce5ea827d2d4f47f990d960b1bc5",
    "val_manifest": "69d45fd04fad01751b3f102f09041c0ba67176e44725dbeb977d550f4b0abbd4",
    "remap": "e7d0890c547b43811f93e50dc0e582ba35eb2e75706ee0ad27186d6a89d502ab",
    "replay_manifest": "19a772846da4b759cf6509dca9be642bf8d01ddccb2015e8215f42532946ba84",
    "replay_rows": 3568,
    "train_rows": 289_142 - 3568,
}

ap = argparse.ArgumentParser()
ap.add_argument("build", type=Path)
ap.add_argument("ledger", type=Path)
ap.add_argument("--probe", type=Path, default=None, help="build 1's dir, for build 2")
ap.add_argument("--hits", type=Path, default=None, help="decontam 1's hit list, for build 2")
args = ap.parse_args()
B = args.build
fails: list[str] = []
summary: dict[str, object] = {}


def gate(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        fails.append(name)


def header(p: Path) -> dict:
    return json.loads((p / "header.json").read_text(encoding="utf-8"))


def manifest(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


val_h = header(B / "shards/val")
gate("val shard_hash", val_h["shard_hash"] == WANT["val_shard_hash"], val_h["shard_hash"])
val_m = manifest(B / "data/pool/val.json")
gate("val manifest data_snapshot_hash", val_m["data_snapshot_hash"] == WANT["val_manifest"],
     val_m["data_snapshot_hash"])
ho_new = manifest(B / "data/heldout/heldout.json")["data_snapshot_hash"]
ho_v4 = manifest(V4 / "data/heldout/heldout.json")["data_snapshot_hash"]
gate("held-out manifest data_snapshot_hash == v4's", ho_new == ho_v4, f"{ho_new} vs v4 {ho_v4}")
train_h = header(B / "shards/train")
replay_h = header(B / "shards/replay")
for name, h in (("train", train_h), ("val", val_h), ("replay", replay_h)):
    gate(f"{name} remap_hash", h["remap_hash"] == WANT["remap"], h["remap_hash"])
rep_m = manifest(B / "data/pool/train-replay.json")
train_m = manifest(B / "data/pool/train.json")
v4_train = manifest(V4 / "data/pool/train.json")
v4_pairs = {(e["row_id"], e["content_hash"]) for e in v4_train["entries"]}
new_pairs = {(e["row_id"], e["content_hash"]) for e in train_m["entries"]}
rep_ids = {e["row_id"] for e in rep_m["entries"]}

if args.probe is None:
    gate("replay manifest data_snapshot_hash",
         rep_m["data_snapshot_hash"] == WANT["replay_manifest"], rep_m["data_snapshot_hash"])
    gate("replay manifest rows", rep_m["n_rows"] == WANT["replay_rows"], str(rep_m["n_rows"]))
    drawn = rep_ids
else:
    probe_rep = manifest(args.probe / "data/pool/train-replay.json")
    drawn = {e["row_id"] for e in probe_rep["entries"]}
    hits = json.loads(args.hits.read_text(encoding="utf-8"))
    hit_rows = {p["row_id"] for p in hits["pairs"]}
    h = hits["replay_rows_hit"]
    summary["h"] = h
    gate("hit list h == distinct hit rows", h == len(hit_rows), f"h={h}, rows={len(hit_rows)}")
    gate("replay rows == 3,568 - h", rep_m["n_rows"] == WANT["replay_rows"] - h,
         f"{rep_m['n_rows']} vs {WANT['replay_rows']} - {h}")
    gate("replay rows == build 1's replay rows minus the hit rows", rep_ids == drawn - hit_rows,
         f"{len(rep_ids)} vs {len(drawn - hit_rows)}")
    probe_train_h = header(args.probe / "shards/train")
    gate("train shard_hash == build 1's", train_h["shard_hash"] == probe_train_h["shard_hash"],
         f"{train_h['shard_hash']} vs {probe_train_h['shard_hash']}")
    probe_train_m = manifest(args.probe / "data/pool/train.json")
    gate("train manifest hash == build 1's",
         train_m["data_snapshot_hash"] == probe_train_m["data_snapshot_hash"],
         train_m["data_snapshot_hash"])

v4_ids = {r for r, _ in v4_pairs}
gate("every drawn replay row is a v4 train row", drawn <= v4_ids, f"{len(drawn - v4_ids)} not")
want_pairs = {p for p in v4_pairs if p[0] not in drawn}
gate("train.json == v4 train minus the 3,568 drawn row_ids (row_id, content_hash)",
     new_pairs == want_pairs, f"{len(new_pairs)} entries; diff {len(new_pairs ^ want_pairs)}")
gate("train rows == 285,574", train_m["n_rows"] == WANT["train_rows"], str(train_m["n_rows"]))
gate("train manifest hash differs from F's ea3215c4",
     train_m["data_snapshot_hash"] != v4_train["data_snapshot_hash"], train_m["data_snapshot_hash"])

rows = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines() if line]
row = rows[-1]
v4_row = next(json.loads(line) for line in V4_LEDGER.read_text(encoding="utf-8").splitlines()
              if json.loads(line)["row_id"] == V4_ROW)
shared = sorted(set(row["recipe"]) & set(v4_row["recipe"]) - {"rev"})
differ = [k for k in shared if row["recipe"][k] != v4_row["recipe"][k]]
gate("recipe keys shared with d96409bd are equal", not differ, f"differ: {differ}")
gate("recipe rev", row["recipe"]["rev"] == v4_row["recipe"]["rev"], row["recipe"]["rev"])
added = sorted(set(row["recipe"]) - set(v4_row["recipe"]))
removed = sorted(set(v4_row["recipe"]) - set(row["recipe"]))
gate("recipe adds only the replay keys", set(added) <= {"replay_shards", "replay_exclude_sha256"}
     and not removed, f"added {added}, removed {removed}")
summary.update({
    "ledger_row": row["row_id"], "code_commit": row["code_commit"], "status": row["status"],
    "wall_clock_s": row["wall_clock_s"], "recipe_hash": row["protocol"]["recipe_hash"],
    "protocol_data_snapshot_hash": row["protocol"]["data_snapshot_hash"],
    "train_manifest_hash": train_m["data_snapshot_hash"], "train_shard_hash": train_h["shard_hash"],
    "train_sequences": train_h["n_sequences"], "replay_shard_hash": replay_h["shard_hash"],
    "replay_sequences": replay_h["n_sequences"],
    "replay_manifest_hash": rep_m["data_snapshot_hash"],
    "replay_rows": rep_m["n_rows"], "val_shard_hash": val_h["shard_hash"],
    "recipe_added": added,
    "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
})
print(json.dumps(summary, indent=2))
print("GATES:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
