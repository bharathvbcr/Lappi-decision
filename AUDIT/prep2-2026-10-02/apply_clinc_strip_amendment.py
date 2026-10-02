"""Apply Fable's CLINC strip ruling (section 1) to the v5 DRAFT's containment rule.

The lead ran this once on 2026-10-02, before any v5 build, full scan or list existed. The script
replaces the strip paragraph of `data.decontamination.rule`, from "Constant template text is
stripped" through "...renames this file).", with Fable's text from
AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md (the block-quoted paragraph of section 1),
plus a citation. The DRAFT is ASCII-only, so two characters are transliterated:
'§' (section sign) becomes 'section ' and '—' (em dash) becomes '--'. Nothing else
changes. It also updates the two amendments_pending/build_order lines that name the strip and
appends to amendments_applied. A second run fails, because the old text is gone.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "campaign" / "v5-preregistered.DRAFT.json"
RULING = ROOT / "AUDIT" / "prep2-2026-10-02" / "fable-clinc-strip-ruling.md"

quoted = [ln for ln in RULING.read_text(encoding="utf-8").splitlines() if ln.startswith("> Constant template")]
if len(quoted) != 1:
    sys.exit(f"expected one quoted amendment paragraph, found {len(quoted)}")
new_para = quoted[0][2:].replace("§", "section ").replace("—", "--")
if any(ord(c) > 127 for c in new_para):
    sys.exit("amendment text still has non-ASCII characters")
new_para += (
    " (Fable's post-F ruling 3(a), amended by Fable's CLINC strip ruling, "
    "AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md section 1, after lane L-prep2's subsample "
    "re-run showed the version-1 strip raise CLINC exclusion to 63-66%; the human ratifies by the "
    "commit that renames this file)."
)

text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
rule = d["data"]["decontamination"]["rule"]
start = rule.find("Constant template text is stripped")
end_marker = "the human ratifies by the commit that renames this file)."
end = rule.find(end_marker)
if start < 0 or end < 0 or rule.count(end_marker) != 1:
    sys.exit("could not locate the version-1 strip paragraph exactly once")
end += len(end_marker)
new_rule = rule[:start] + new_para + rule[end:]


def q(v):
    return json.dumps(v, ensure_ascii=True)


def sub(old: str, new: str) -> None:
    global text
    if text.count(old) != 1:
        sys.exit(f"expected one match for {old[:100]!r}, found {text.count(old)}")
    text = text.replace(old, new)


sub(q(rule), q(new_rule))

pending = d["amendments_pending"]
old_p = next(p for p in pending if p.startswith("the template strip's subsample re-run"))
sub(
    q(old_p),
    q(
        "the template strip's subsample re-run under STRIP_VERSION 2: the pass condition of "
        "AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md section 4 at 1/2/5% (CLINC 0 keys and 0 "
        "enforced intent.* pairs; the non-intent exclusions byte-identical to version 1's by the three "
        "pinned sha256; too_short_after_strip and key_ii_blind per set; CLEAN with the three splitter "
        "checks), then on the full scan the CLINC rate, the key_ii_blind counts and the two zero-checks "
        "(exact lower-cased match and contiguous word-subsequence) for the key-(ii)-blind val/held-out keys"
    ),
)
bo0 = d["build_order"][0]
sub(
    q(bo0),
    q(
        bo0.replace(
            "the template strip and its subsample re-run (L-prep2) read before the full scan",
            "the version-2 template strip (Fable's CLINC strip ruling) and its subsample re-run meeting "
            "that ruling's section 4 pass condition before the full scan",
        )
    ),
)
applied = d["amendments_applied"]
sub(
    q(applied),
    q(
        applied
        + " Then on 2026-10-02 (~16:20 UTC), Fable's CLINC strip ruling "
        "(AUDIT/prep2-2026-10-02/fable-clinc-strip-ruling.md, source sha256 ec2c1e6d...) replaced the "
        "strip paragraph of data.decontamination.rule (version 2: question line in every family, "
        "family-constant options, and every option value of the four intent.* families; key (ii)-blind "
        "short utterances reported, no new key), through "
        "AUDIT/prep2-2026-10-02/apply_clinc_strip_amendment.py."
    ),
)

out = json.loads(text)
assert "STRIP_VERSION 2" in out["data"]["decontamination"]["rule"]
assert "renames this file)" in out["data"]["decontamination"]["rule"]
assert all(ord(c) < 128 for c in text)
DRAFT.write_text(text, encoding="utf-8")
print("amended", DRAFT)
