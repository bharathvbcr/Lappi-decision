"""The two zero-checks of Fable's CLINC strip ruling section 2 (lane L-prep3, 2026-10-02).

Throwaway analysis over a version-2 scan; it moves nothing. Under strip version 2 every intent.*
slot text is the utterance, so a val or held-out CLINC utterance under 8 words is invisible to
key (ii). Those keys are covered by keys (i) and (iii) only; the ruling adds no key and asks
instead for two counts over them, here and again on the full scan:

* **exact**: how many of the key-(ii)-blind val / held-out keys' utterances, lower-cased and
  cut to ``\\w+`` words joined by one space, equal a train utterance cut the same way;
* **subsequence**: how many are a contiguous word-subsequence of any train utterance (the
  space-padded cut is a substring of the space-padded cut of some train utterance).

"Train utterance" means a CLINC train utterance: the text of a train row of an intent.*
family, which version 2 makes the bare utterance (as ``AUDIT/prep2-2026-10-02/
clinc_oracle.py`` reads intent.domain's). The definitions are that oracle's, so the counts
are comparable with its 0/0 at 1/2/5%.

The key-(ii)-blind keys are recomputed from the request -- identity keys all of whose slot
texts in the set have fewer than 8 ``\\w+`` words -- and compared with the attestation's
``export.template_strip.key_ii_blind`` (count and sha256). A nonzero count is a ``GAP-`` for
the human, not a rule change; this script only reports it.

    python AUDIT/prep3-2026-10-02/key_ii_blind_zero_checks.py SCAN_DIR [SCAN_DIR ...]

Each SCAN_DIR was written with ``--request-out SCAN_DIR.request.bin``. Output: one JSON list on
stdout. Exit 0 when every count was computed and the blind sets agree with the attestation
(the counts themselves may be nonzero; they are reported, not gated), 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_pass import CLINC_PREFIX, request_rows  # noqa: E402

N = 8
WORD = re.compile(r"\w+")
TARGETS = ("val", "heldout")
INTENT_PREFIX = "intent."


def _cut(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def zero_checks(scan: Path) -> dict[str, Any]:
    att = json.loads((scan / "attestation.json").read_text(encoding="utf-8"))
    recorded = att["export"]["template_strip"]["key_ii_blind"]
    request = scan.with_name(scan.name + ".request.bin")
    # set -> identity -> every slot text too short so far
    all_short: dict[str, dict[str, bool]] = defaultdict(dict)
    # set -> identity -> the intent.* utterances (one, under version 2)
    utterances: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for set_name, _key, identity, family, text in request_rows(request):
        short = len(WORD.findall(text.lower())) < N
        seen = all_short[set_name]
        seen[identity] = seen.get(identity, True) and short
        if family.startswith(INTENT_PREFIX):
            utterances[set_name][identity].add(text)
    train = {_cut(u) for texts in utterances["train"].values() for u in texts}
    padded_train = [f" {t} " for t in train]
    out: dict[str, Any] = {"scan": str(scan), "train_clinc_utterances": len(train)}
    agree = True
    for target in TARGETS:
        blind = sorted(k for k, s in all_short[target].items() if s)
        digest = hashlib.sha256("".join(f"{k}\n" for k in blind).encode("utf-8")).hexdigest()
        rec = recorded.get(target, {})
        matches = rec.get("n") == len(blind) and rec.get("sha256") == digest
        agree = agree and matches
        clinc = [k for k in blind if k.startswith(CLINC_PREFIX)]
        multi = sorted(k for k in clinc if len(utterances[target][k]) != 1)
        exact: list[str] = []
        sub: list[str] = []
        for k in clinc:
            for u in utterances[target][k]:
                cut = _cut(u)
                if cut in train:
                    exact.append(k)
                if any(f" {cut} " in t for t in padded_train):
                    sub.append(k)
        agree = agree and not multi
        out[target] = {
            "key_ii_blind": len(blind),
            "key_ii_blind_matches_attestation": matches,
            "clinc_keys": len(clinc),
            "non_clinc_keys": sorted(set(blind) - set(clinc)),
            "clinc_keys_without_exactly_one_utterance": multi,
            "exact_match_in_train": len(exact),
            "contiguous_word_subsequence_of_a_train_utterance": len(sub),
            "exact_keys": sorted(set(exact)),
            "subsequence_keys": sorted(set(sub)),
            "subsequence_utterances": sorted(
                {u for k in sub for u in utterances[target][k]}),
        }
    out["computed_and_agrees_with_attestation"] = agree
    return out


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    results = [zero_checks(Path(a)) for a in argv]
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return 0 if all(r["computed_and_agrees_with_attestation"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
