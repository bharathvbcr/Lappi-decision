"""A gap lane L-prep3 found in passing, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. It refuses to append a second copy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

GAP = "GAP-CONTAINMENT-CORPUS-IDENTITY-DEFECT-DIRS-UNHASHED-2026-10-02"

if any(r.get("id") == GAP for r in read_gaps()):
    sys.exit(f"{GAP} is already in gaps.jsonl; refusing a duplicate")

append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "lane": "L-prep3",
        "owner": "lead",
        "status": "open",
        "tool": (
            "read: tools/real_tokenizer_pipeline.py:1742-1782 (corpus_identity), "
            "tools/containment_scan.py main (the replay_corpus_identity call), "
            "python/qd_train/exclusions.py read_exclusions (the corpus check)"
        ),
        "question": (
            "read_exclusions refuses a containment list whose attestation names another corpus. "
            "Does the corpus identity it compares pin every input that turns into rows?"
        ),
        "answer": (
            "Not all of them [V, read]. corpus_identity names --defect-class by its directory "
            "NAME only (defect_class.name, e.g. 'commitpackft-composed-v1'), not by content, "
            "and does not name --defect-download at all; --defect-noul is pinned by its "
            "manifest's examples_sha256, the general record by its sha256, and qd_data's code "
            "by qd_data_fingerprint_sha256. So a full containment scan run against a different "
            "composed corpus in a same-named directory, or a different --defect-download, "
            "yields a list the v5 build's hook accepts although it describes other rows. Until "
            "the identity pins them, the lead matches --defect-class and --defect-download to "
            "the v5 build by hand (HANDOFF/prep3-2026-10-02.md names the flags). Not changed "
            "here: tools/real_tokenizer_pipeline.py is another lane's file."
        ),
    }
)
print(f"appended {GAP}")
