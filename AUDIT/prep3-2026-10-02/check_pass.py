"""Fable's CLINC strip ruling section 4, checked item by item (lane L-prep3, 2026-10-02).

Throwaway analysis; it moves nothing. Input: the three version-2 subsample scans, written by
``tools/containment_scan.py`` with ``--request-out SCAN.request.bin``. The expected numbers
are the ruling's (AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md section 4) and are not
this script's to move (rule 2). Pass iff every item holds at every size:

1. CLINC (intent.*) excluded keys 0 of 246 / 464 / 1089 distinct train CLINC identity keys,
   no ``clinc-intent:`` key in ``exclusions.txt``, and 0 enforced pairs (train -> val or
   train -> heldout) with an intent.* row on either side.
2. The non-intent subset of ``exclusions.txt`` byte-identical to version 1's: the keys not
   starting with ``clinc-intent:``, in file order, each followed by LF (the trailing LF is
   hashed): 8 / 16 / 41 keys, sha256 as the ruling gives them.
3. Every intent.* family-slot: the question line stripped, every row checked equal to its
   context (``closed_vocabulary.rows_checked_equal_to_context == rows``), and the four
   intent.* texts of each identity key identical in the request; ``too_short_after_strip``
   of EACH intent.* family-slot, per set (train/val/heldout), 102/4/2, 197/9/7, 451/21/34;
   ``key_ii_blind`` per set the same, every blind key a CLINC key, its list hashing to the
   recorded sha256. (The per-set total over every family-slot,
   ``too_short_after_strip_by_set``, counts slot texts: about four per CLINC key. It is
   reported, not compared.) The too_short counts are also recomputed from the request.
4. CLEAN, ``remaining_hits`` 0 on every enforced target, and identity_disjoint,
   near_duplicate_disjoint and repo_disjoint each ran and passed; the strip applied at
   version 2 under this checkout's ``STRIP_RULE``.
5. Reported, not compared: the side-by-side version-2 numbers per family
   (``AUDIT/prep2-2026-10-02/family_rates.py``'s groups), pairs, keys, too-short per set per
   key, distinct option values per (2b) family-slot.

    python AUDIT/prep3-2026-10-02/check_pass.py p01=SCAN_DIR p02=SCAN_DIR p05=SCAN_DIR

Output: one JSON object on stdout; exit 0 iff every item passed at every size given, 1 if
any failed, 2 if a size other than the three was named.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "AUDIT" / "prep2-2026-10-02"))
sys.path.insert(0, str(REPO / "python"))

from family_rates import MAGIC, _str, rates  # noqa: E402

from qd_train.containment_strip import (  # noqa: E402
    CLOSED_VOCABULARY_FAMILIES,
    STRIP_RULE,
    STRIP_VERSION,
)

#: The ruling's section 4 numbers, per subsample (read-only).
EXPECTED: dict[str, dict[str, Any]] = {
    "p01": {"clinc_train_keys": 246, "non_intent_keys": 8,
            "non_intent_sha256": "2f172e00166496dc1ca66b568ffd303c1694d8e167e53c825b419057046aae31",
            "too_short": {"train": 102, "val": 4, "heldout": 2}},
    "p02": {"clinc_train_keys": 464, "non_intent_keys": 16,
            "non_intent_sha256": "6f4e9364f30da9468553651df787c1a8b1791a4ecf673d6a201a53a90b79c268",
            "too_short": {"train": 197, "val": 9, "heldout": 7}},
    "p05": {"clinc_train_keys": 1089, "non_intent_keys": 41,
            "non_intent_sha256": "889e0ecf85f395af7b762980bc38da6e933f9a27e38ff757fe8c462c335d71ac",
            "too_short": {"train": 451, "val": 21, "heldout": 34}},
}
SETS = ("train", "val", "heldout")
ENFORCED = ("val", "heldout")
SPLITTER_CHECKS = ("identity_disjoint", "near_duplicate_disjoint", "repo_disjoint")
CLINC_PREFIX = "clinc-intent:"
N = 8
WORD = re.compile(r"\w+")


def request_rows(request: Path) -> Iterator[tuple[str, str, str, str, str]]:
    """``(set, key, identity, family, text)`` of every row of a ``QDPCTIN1`` request."""
    with request.open("rb") as fh:
        if fh.read(8) != MAGIC:
            raise SystemExit(f"{request}: not a QDPCTIN1 request")
        fh.read(12)
        _str(fh)
        _str(fh)
        (n_checks,) = struct.unpack("<I", fh.read(4))
        for _ in range(n_checks):
            _str(fh)
            fh.read(1)
            _str(fh)
        (n_sets,) = struct.unpack("<I", fh.read(4))
        names = [_str(fh) for _ in range(n_sets)]
        (n_scans,) = struct.unpack("<I", fh.read(4))
        fh.read(9 * n_scans)
        (n_rows,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n_rows):
            (i,) = struct.unpack("<I", fh.read(4))
            yield names[i], _str(fh), _str(fh), _str(fh), _str(fh)


def _too_short(text: str) -> bool:
    return len(WORD.findall(text.lower())) < N


def check(size: str, scan: Path) -> dict[str, Any]:
    want = EXPECTED[size]
    att = json.loads((scan / "attestation.json").read_text(encoding="utf-8"))
    strip = att["export"]["template_strip"]
    excluded = (scan / "exclusions.txt").read_text(encoding="utf-8").splitlines()
    request = scan.with_name(scan.name + ".request.bin")

    clinc_train: set[str] = set()
    texts: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    short: Counter[tuple[str, str]] = Counter()
    for set_name, _key, identity, family, text in request_rows(request):
        if family not in CLOSED_VOCABULARY_FAMILIES:
            continue
        if set_name == "train":
            clinc_train.add(identity)
        if family in texts[(set_name, identity)]:
            raise SystemExit(f"{scan}: {identity} asked twice by {family} in {set_name}")
        texts[(set_name, identity)][family] = text
        short[(family, set_name)] += _too_short(text)
    differing = sorted(f"{s}/{k}" for (s, k), by in texts.items() if len(set(by.values())) > 1)

    # 1
    pairs_intent = 0
    with (scan / "pairs.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[0] == "train" and f[4] in ENFORCED and (
                f[3] in CLOSED_VOCABULARY_FAMILIES or f[6] in CLOSED_VOCABULARY_FAMILIES
            ):
                pairs_intent += 1
    clinc_excluded = len(clinc_train & set(excluded))
    item1 = {
        "clinc_train_keys": len(clinc_train),
        "clinc_train_keys_excluded": clinc_excluded,
        "clinc_intent_keys_in_exclusions": sum(k.startswith(CLINC_PREFIX) for k in excluded),
        "enforced_pairs_with_an_intent_row": pairs_intent,
    }
    item1["pass"] = (
        item1["clinc_train_keys"] == want["clinc_train_keys"] and clinc_excluded == 0
        and item1["clinc_intent_keys_in_exclusions"] == 0 and pairs_intent == 0
    )

    # 2
    non_intent = [k for k in excluded if not k.startswith(CLINC_PREFIX)]
    digest = hashlib.sha256("".join(f"{k}\n" for k in non_intent).encode("utf-8")).hexdigest()
    item2 = {"non_intent_keys": len(non_intent), "sha256": digest,
             "expected_keys": want["non_intent_keys"], "expected_sha256": want["non_intent_sha256"]}
    item2["pass"] = (
        len(non_intent) == want["non_intent_keys"] and digest == want["non_intent_sha256"]
    )

    # 3
    groups = strip["family_slots"]
    intent_groups = {name: g for name, g in groups.items()
                     if name.split("/")[0] in CLOSED_VOCABULARY_FAMILIES}
    per_slot = {}
    slot_ok = len(intent_groups) == len(CLOSED_VOCABULARY_FAMILIES)
    for name, g in sorted(intent_groups.items()):
        family = name.split("/")[0]
        cv = g.get("closed_vocabulary") or {}
        by_set = {s: g["by_set"].get(s, {}).get("too_short_after_strip") for s in SETS}
        recomputed = {s: short[(family, s)] for s in SETS}
        ok = (
            g["stripped"]["question"] is not None
            and cv.get("rows_checked_equal_to_context") == g["rows"]
            and by_set == want["too_short"]
            and recomputed == want["too_short"]
        )
        slot_ok = slot_ok and ok
        per_slot[name] = {
            "rows": g["rows"], "question_stripped": g["stripped"]["question"],
            "rows_checked_equal_to_context": cv.get("rows_checked_equal_to_context"),
            "distinct_option_values": cv.get("distinct_option_values"),
            "too_short_after_strip_by_set": by_set,
            "too_short_recomputed_from_request": recomputed, "pass": ok,
        }
    blind = strip.get("key_ii_blind", {})
    blind_n = {s: blind.get(s, {}).get("n") for s in SETS}
    blind_hash_ok = all(
        hashlib.sha256("".join(f"{k}\n" for k in b["identity_keys"]).encode("utf-8")).hexdigest()
        == b["sha256"] and len(b["identity_keys"]) == b["n"]
        for b in blind.values()
    )
    blind_non_clinc = sorted(k for s in SETS for k in blind.get(s, {}).get("identity_keys", [])
                             if not k.startswith(CLINC_PREFIX))
    item3 = {
        "intent_family_slots": per_slot,
        "intent_texts_differing_across_families": len(differing),
        "intent_texts_differing_examples": differing[:5],
        "key_ii_blind_by_set": blind_n,
        "key_ii_blind_lists_hash_to_their_sha256": blind_hash_ok,
        "key_ii_blind_non_clinc_keys": blind_non_clinc,
        "too_short_after_strip_by_set_slot_texts_all_families": strip.get(
            "too_short_after_strip_by_set"),
        "expected_per_intent_slot_and_key_ii_blind": want["too_short"],
    }
    item3["pass"] = (slot_ok and not differing and blind_n == want["too_short"]
                     and blind_hash_ok and not blind_non_clinc)

    # 4
    checks = att["splitter_checks"]
    item4 = {
        "clean": att["clean"], "remaining_hits": att["remaining_hits"],
        "not_clean_because": att["not_clean_because"],
        "splitter_checks": {c: checks.get(c) for c in SPLITTER_CHECKS},
        "strip": {"applied": strip["applied"], "version": strip["version"],
                  "rule_is_this_checkouts": strip["rule"] == STRIP_RULE},
    }
    item4["pass"] = (
        att["clean"] is True
        and set(att["remaining_hits"]) == set(ENFORCED)
        and all(v == 0 for v in att["remaining_hits"].values())
        and all(isinstance(checks.get(c), dict) and checks[c].get("state") == "ran"
                and checks[c].get("passed") is True for c in SPLITTER_CHECKS)
        and strip["applied"] is True and strip["version"] == STRIP_VERSION
        and strip["rule"] == STRIP_RULE
    )

    # 5 (reported)
    r = rates(scan)
    item5 = {
        "groups": r["groups"],
        "pairs": att["n_pairs"],
        "identity_keys_excluded": att["n_exclusions"],
        "enforced_pairs_by_source_family": r["enforced_pairs"],
        "too_short_per_set_per_key_key_ii_blind": blind_n,
        "distinct_option_values_per_2b_slot": {
            name: v["distinct_option_values"] for name, v in per_slot.items()
        },
        "sets": r["sets"],
    }
    return {"size": size, "scan": str(scan), "1": item1, "2": item2, "3": item3, "4": item4,
            "5": item5, "pass": item1["pass"] and item2["pass"] and item3["pass"]
            and item4["pass"]}


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    results = []
    for arg in argv:
        size, _, path = arg.partition("=")
        if size not in EXPECTED or not path:
            print(f"{arg!r}: expected one of {sorted(EXPECTED)}=SCAN_DIR", file=sys.stderr)
            return 2
        results.append(check(size, Path(path)))
    sizes = sorted(r["size"] for r in results)
    out = {"sizes": sizes, "all_three_sizes": sizes == sorted(EXPECTED),
           "pass": all(r["pass"] for r in results) and sizes == sorted(EXPECTED),
           "results": results}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
