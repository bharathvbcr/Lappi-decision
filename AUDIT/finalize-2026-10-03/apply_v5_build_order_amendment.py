"""Apply Fable's build-order ruling (2026-10-03 ~20:05Z) to the v5 DRAFT.

    python3 AUDIT/finalize-2026-10-03/apply_v5_build_order_amendment.py <DRAFT path>

<DRAFT path> is a checkout's campaign/v5-preregistered.DRAFT.json, after the launch amendment
(apply_v5_launch_amendment.py). The ruling and its conditions are in
AUDIT/finalize-2026-10-03/build-order-ruling-2026-10-03.md.

It changes build_order[2] and [3]. The containment scan runs first, on the v5 corpus flags
directly. Then one build runs with --exclude-identity-keys, and A7 runs on that build. The
standalone pre-exclusion build is dropped. No gate, threshold or population moves.

Textual edits keep the file's layout. A second run fails, because build_order[2] would already
name the ruling.
"""

import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
DRAFT = Path(sys.argv[1]).resolve()
RULING = "AUDIT/finalize-2026-10-03/build-order-ruling-2026-10-03.md"
text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
order = d["build_order"]
if RULING in order[2]:
    sys.exit("already applied")
if "approved" not in d["launch"]:
    sys.exit("the launch amendment is not applied; apply apply_v5_launch_amendment.py first")
if not (order[2].startswith("2. Build on the Mac (CPU): tools/real_tokenizer_pipeline.py")
        and order[3].startswith("3. Decontam: qd-prep containment")):
    sys.exit(f"build_order[2]/[3] are not the ones this amendment replaces: {order[2][:60]!r}, "
             f"{order[3][:60]!r}")


def q(v):
    return json.dumps(v, ensure_ascii=True)


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        sys.exit(f"anchor not found exactly once ({text.count(old)}): {old[:80]!r}")
    text = text.replace(old, new)


FLAGS = (
    "--rev <build commit> --no-repo-history --defect-class data/pool/commitpackft-composed-v2 "
    "--defect-download data/pool/commitpackft --defect-noul data/pool/defect-noul-v3c "
    "--general-record a0841f0d... --general-max-rows 200000 --decisions-pool <the rebuilt "
    "decision pool, at its report's examples sha256>"
)
new2 = (
    "2. Decontam first (Fable, ~20:05Z 2026-10-03, " + RULING + "): qd-prep containment "
    "(tools/containment_scan.py, template strip version 2, L-prep's request form) on the v5 "
    "corpus flags directly, the same list the build takes (" + FLAGS + "; no --max-pairs, no "
    "--defect-max-rows); exclusions.txt and attestation.json; then "
    "AUDIT/prep2-2026-10-02/family_rates.py and AUDIT/prep3-2026-10-02/"
    "key_ii_blind_zero_checks.py, whose two zero-checks must read 0 for val and held-out (a "
    "nonzero is a GAP for the human, not a build). The scan's exit 0, a CLEAN attestation, is "
    "the only pass."
)
new3 = (
    "3. The one build on the Mac (CPU): tools/real_tokenizer_pipeline.py with the scan's corpus "
    "flags plus v5's argv (--max-seq-len 10240, --vocab full, --val-shards, "
    "--span-collapse-policy refuse-gold, --memo-limit 0) and --exclude-identity-keys "
    "<scan>/exclusions.txt; contrast rows are derived after its dedupe, split and exclusions. "
    "The standalone pre-exclusion build is dropped: everything it produced, this build produces. "
    "A7 runs on this build (tools/v5_a7_check.py --decisions-pool). A7 reads val and held-out "
    "only, and the pipeline refuses an exclusion key that is not a train row, so an A7 missing "
    "key is a dedupe knock-out: exclude that row and rebuild (data.sources' A7 rule)."
)
replace_once(q(order[2]), q(new2))
replace_once(q(order[3]), q(new3))
note = (
    " On 2026-10-03 (~20:05Z) Fable's build-order ruling was applied "
    "(AUDIT/finalize-2026-10-03/apply_v5_build_order_amendment.py; " + RULING + "). "
    "build_order[2] became the containment scan on the corpus flags, and build_order[3] the one "
    "build with --exclude-identity-keys and A7 on it. The standalone pre-exclusion build is "
    "dropped. No gate, threshold or population moves."
)
replace_once(q(d["amendments_applied"]), q(d["amendments_applied"] + note))

after = json.loads(text)
assert after["build_order"][2] == new2 and after["build_order"][3] == new3
assert after["build_order"][:2] == order[:2] and after["build_order"][4:] == order[4:]
assert after["amendments_applied"].endswith(note)
assert {k: v for k, v in after.items() if k not in ("build_order", "amendments_applied")} == {
    k: v for k, v in d.items() if k not in ("build_order", "amendments_applied")}
DRAFT.write_text(text, encoding="utf-8")
print(f"applied to {DRAFT}: build_order[2] scan first, [3] one build with exclusions and A7")
