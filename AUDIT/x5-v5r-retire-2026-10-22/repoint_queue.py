"""Repoint campaign/next-train-first-box-queue-2026-10-08.json from the v5 data to the v6 shards.

Run once from the repo root: python repoint_queue.py. It rewrites the file with the same
2-space layout; the round trip of the unmodified file is checked first, so the diff is only
what is edited here.
"""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path("campaign/next-train-first-box-queue-2026-10-08.json")
text = PATH.read_text(encoding="utf-8")
queue = json.loads(text)
if json.dumps(queue, indent=2, ensure_ascii=False) + "\n" != text:
    raise SystemExit("the file does not round-trip; refusing to rewrite its layout")

V6_EXCLUSIONS_SHA = "37b84644f5718869ea57372681a21a6ad6b8105a398ee45f1cffa143688a3ff7"

queue["_comment"] = [
    *queue["_comment"],
    "REPOINTED 2026-10-22 (Fable: 'repoint the queue now'; campaign/next-train-first-box-"
    "preregistered.json promoted.amendment_2). The v5r shard set is retired: contaminated "
    "(LoRA-2B amendment 3). Arms now read the v6-rule rebuild pinned by LoRA-2B's entry n = 5 "
    "(train shard_hash 94bef49aa2c1..., val 58bd753fc933...): {v6_data_dir} replaces "
    "{v5_data_dir}, --exclude-identity-keys is the v6 scan's list (pinned below), the v5 "
    "pre-dedupe drop list is NOT passed (the v6 split-aware keep rule supersedes it, "
    "as in that entry), --legacy-v5-data-rules is absent so the rebuild is under v6's rules, "
    "and {decisions_pool} must be pool-v6-decisions-v1 (see that entry for its examples "
    "sha256). The comparison of the arms' shard_hash with v5's "
    "108082545a66... / c38dd2ca3e8f... is DROPPED; the reading's expected shard_hash is the v6 "
    "train 94bef49a....",
    "HELD: the queue is not launchable. See 'hold'.",
]
queue["hold"] = {
    "status": "held",
    "since": "2026-10-22",
    "why": "Repointing was done from the text of LoRA-2B's entry n = 5, not from a run: "
           "whether the box's real_ft_run, given {v6_data_dir} (a directory laid out as "
           "<out>/shards/{train,val} plus the manifests real_ft_run reads) and these inputs, "
           "reproduces the v6 shard_hash 94bef49a... has not been run. The directory the v6 set "
           "was built in no longer exists, and main's copy holds only the shards.",
    "released_by": "fill the placeholder v6_shards_verified with the path of a run that "
                   "reproduced train 94bef49a... and val 58bd753f... from these argv, then --plan",
    "enforced_by": "the driver refuses an unfilled placeholder (tools/campaign_driver.py "
                   "_placeholders), so no unit launches while v6_shards_verified is null",
}
ph = queue["placeholders"]
ph["v6_data_dir"] = ph.pop("v5_data_dir")
ph.pop("drop_before_dedupe")
ph["v6_shards_verified"] = None

for phase in queue["phases"]:
    for unit in phase["units"]:
        argv = unit["argv"]
        if "--drop-before-dedupe" in argv:
            i = argv.index("--drop-before-dedupe")
            del argv[i : i + 2]
        unit["argv"] = [a.replace("{v5_data_dir}", "{v6_data_dir}") for a in argv]

queue["pins"] = [p for p in queue["pins"] if p["path"] != "{drop_before_dedupe}"]
queue["pins"].append({"path": "{exclusions}", "sha256": V6_EXCLUSIONS_SHA})

PATH.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
