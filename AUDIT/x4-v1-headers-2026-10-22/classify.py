"""Read-only: turn readers.json into readers.md and verdicts.json (retire / hold per set)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

A = Path(sys.argv[1]) / "AUDIT/x4-v1-headers-2026-10-22"
rows = json.loads((A / "readers.json").read_text())
sets = {s["dir"]: s for s in json.loads((A / "sets.json").read_text())}

#: file -> (class, citation). A reader is classified by the file that names the set.
CLASS: dict[str, tuple[str, str]] = {
    "campaign/v5-preregistered.json": ("historical", "v5 run complete; its rows are ledger/h100x2-v5-2026-10-03.jsonl"),
    "campaign/j6a-preregistered.json": ("historical", "quick, one seed, 'starts nothing automatically' (its own decision text); v4-era"),
    "campaign/j6g-preregistered.json": ("historical", "quick, one seed, 'starts nothing automatically' (its own decision text); v4-era"),
    "campaign/lora-2b-preregistered.json": ("withdrawn", "amendments[3] (2026-10-17) withdraws the v5r train and val shards as this registration's data; amendment 4 pins the v6-rule rebuild"),
    "campaign/next-train-first-box-preregistered.json": ("live-pending, quick-only", "P1 data_fact reads v5r train; status 'PRE-REGISTERED, NOT LAUNCHED'; 'every row it writes is quick (rule 8)'; method pins a worktree build of the writing code (line 43), not allow_stale_code"),
    "campaign/span-refit-mac-preregistered.json": ("live-pending, quick-only", "status 'quick, report-only. Cannot promote (rule 8)'; amendment 2 (2026-10-22) not yet run; pins v5r val+train by path and shard_hash"),
    "campaign/post-f-queue/box_q_j6g.sh": ("historical", "j6g queue script (quick, v4-era); shard_hash pin"),
    "campaign/post-f-queue/j6g_prelude_mac.py": ("historical", "j6g prelude (quick, v4-era); shard_hash pin"),
    "crates/qd-preflight/src/prereg_bind.rs": ("historical", "const HASH inside the #[cfg(test)] module (line 361): a test fixture value, opens no set"),
    "crates/qd-runtime/src/bin/qd_post_f_rules.rs": ("historical", "TIERB_*/J6G_* consts compare ledger-row identities of finished GH200 rows; open no set"),
    "tools/qd_train_oracle_trainer.py": ("historical", "oracle fixture value for tier-B trainer parity; opens no set"),
    "tools/real_ft_run.py": ("dormant", "COMPOSED_SLICE_SHARD_HASH (line 8633) is a pin on a caller-supplied slice_out; no default path to the qd-campaign set; a rebuild would also move the pin (code owned elsewhere)"),
}
TEST_FILES = {
    "crates/qd-runtime/tests/post_f_rules_tierb.rs",
    "crates/qd-train/tests/fixtures/trainer-oracle.json",
    "crates/qd-train/tests/ledger_oracle.rs",
    "crates/qd-train/tests/lora_row.rs",
}

out = [
    "# X4 readers table (lane x4-v1-headers, 2026-10-22)",
    "",
    "Source of truth: `readers.json` (made by `readers.py`, one streaming pass over every tracked text file and every text file under `~/qd-campaign`; a set is hit by a `qd-campaign/<rel>` path mention or a hex run beginning with its shard_hash[:16]). `sets.json` is `enumerate.py` over the 89 headers. Both fall back to no code-intelligence index (text scan, `git ls-files`, not DevMap); this is a text scan, not a graph answer, and it only sees text files <= 200 MB.",
    "",
    "Findings that decide the table:",
    "",
    "1. All 89 headers are fingerprint v1 (every `qd_data/*.py`); all 89 drift at HEAD (0 verify). `shard_hash` covers `code_fingerprint`, so ANY rebuild moves `shard_hash`, whatever the payload.",
    "2. Every machine reader of these sets pins the set by `shard_hash` (ledger identity checks, pre-registrations, test constants). A rebuilt set cannot satisfy a hash pin, so a rebuild serves a reader only if that reader is re-pinned. Re-pinning a pre-registration is an amendment (lead, 2026-10-22).",
    "3. No tracked code opens a `~/qd-campaign` shard set by default path. Path mentions in live files: `campaign/j6a`, `next-train-first-box`, `span-refit-mac` only.",
    "4. held_out.rs / shards.rs path rules: `held_out.rs` refuses path segments `heldout`/`held_out`/`held-out` and configured roots; none of the 89 shard dirs has such a segment, so those rules neither read nor bar any of them. `shards.rs` (Rust) parses `code_fingerprint` and enforces staleness only through `ShardOpenOptions.allow_stale_code` (default false); that is the read-only check this lane leaves alone.",
    "",
    "Classes: live (a decision still being made reads it), live-pending quick-only (pre-registered, cannot promote, rule 8), historical (the decision is made; the citation is identity), withdrawn, dormant.",
    "",
    "## Per-set verdicts",
    "",
    "| set (under ~/qd-campaign) | shard_hash[:12] | readers (class) | verdict |",
    "| --- | --- | --- | --- |",
]
verdicts = []
for r in rows:
    h = r["hits"]
    rel = r["rel"]
    sh = r["shard_hash"][:12]
    classes: dict[str, set[str]] = {}
    for f in h["live"] + h["test"]:
        name = f.split(" ")[0]
        c = CLASS.get(name, ("test-pin" if name in TEST_FILES else "UNCLASSIFIED", ""))[0]
        classes.setdefault(c, set()).add(name)
    if h["ledger"]:
        classes.setdefault("historical", set()).add(f"{len(h['ledger'])} ledger file(s)")
    if "shards-tiny" in rel:
        verdict = "retire: no reader (scratch copy of tracked fixture crates/qd-train/tests/fixtures/shards-tiny, same shard_hash; the tracked fixture is the reader and lives outside ~/qd-campaign)"
        why = "copy of tracked fixture"
    elif "live-pending, quick-only" in classes:
        verdict = "HOLD: live-pending quick-only reader; not rebuilt, not retired; lead's call"
        why = "hold"
    else:
        bits = sorted(c for c in classes if c != "UNCLASSIFIED")
        if "UNCLASSIFIED" in classes:
            raise SystemExit(f"unclassified reader for {rel}: {classes['UNCLASSIFIED']}")
        if bits:
            verdict = "retire: readers are only " + "/".join(bits) + " (every one pins by shard_hash; a rebuild cannot serve it)"
        else:
            verdict = "retire: no reader"
        why = "retire"
    desc = "; ".join(f"{c}: {', '.join(sorted(v))}" for c, v in sorted(classes.items())) or "none"
    ext = len(h["external"])
    out.append(f"| `{rel}` | {sh} | {desc}{'; ' + str(ext) + ' file(s) under ~/qd-campaign (logs/ledgers/manifests of past runs, historical)' if ext else ''} | {verdict} |")
    verdicts.append({"rel": rel, "dir": r["dir"], "shard_hash": r["shard_hash"], "verdict": why, "detail": verdict})
(A / "verdicts.json").write_text(json.dumps(verdicts, indent=1))

n = {k: sum(1 for v in verdicts if v["verdict"] == k) for k in ("retire", "hold", "copy of tracked fixture")}
out += [
    "",
    f"Totals: {len(verdicts)} sets; retire {n['retire'] + n['copy of tracked fixture']} ({n['copy of tracked fixture']} of them scratch copies of the tracked fixture); hold {n['hold']}.",
    "",
    "## Reader citations (classes)",
    "",
]
for f, (c, cite) in sorted(CLASS.items()):
    out.append(f"- `{f}`: **{c}**. {cite}")
for f in sorted(TEST_FILES):
    out.append(f"- `{f}`: **test-pin**. Hash constants in tier-B/oracle tests; open no `~/qd-campaign` set.")
(A / "readers.md").write_text("\n".join(out) + "\n")
print(n)
