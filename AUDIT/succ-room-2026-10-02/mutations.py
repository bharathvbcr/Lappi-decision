"""Mutation pass over qd-post-f-rules' successor rule (lane L-room, 2026-10-02).

Throwaway audit tooling, not shipped code. It is committed because the previous lane's runner
(HANDOFF/succ-rule-2026-10-02.md, 18 of 18) was never committed and had to be reconstructed
from the descriptions in AUDIT/idle-gpu-queue-2026-10-02/succ-rule-mutations.txt.

Form, as before: apply one source mutation (an exact string that must occur exactly once), run
`cargo test -p qd-runtime --bin qd-post-f-rules`, record the result and the failing tests,
restore the source byte for byte and check its sha256. A mutation that does not compile is
reported as such and is not counted as caught.

Run from the worktree root:
    /Users/bharath/.venvs/ml/bin/python AUDIT/succ-room-2026-10-02/mutations.py
"""

import hashlib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "crates/qd-runtime/src/bin/qd_post_f_rules.rs"
OUT = ROOT / "AUDIT/succ-room-2026-10-02/mutations.txt"
CMD = ["cargo", "test", "-p", "qd-runtime", "--bin", "qd-post-f-rules"]
TIMEOUT_S = 900

# (id, description, old, new). M1-M18 reconstruct the previous lane's list; N1-N8 are new.
MUTATIONS = [
    ("M1", "R7 target tie fires (higher: > -> >=)",
     "Dir::Higher => a * f * h + e * b * h > 2 * g * b * f,",
     "Dir::Higher => a * f * h + e * b * h >= 2 * g * b * f,"),
    ("M2", "R7 target tie fires (lower: > -> >=)",
     "Dir::Lower => 2 * e * b * h > a * f * h + g * b * f,",
     "Dir::Lower => 2 * e * b * h >= a * f * h + g * b * f,"),
    ("M3", "R7 guard at its bound loses (higher)",
     "Dir::Higher => cmp_val(c, min)? == std::cmp::Ordering::Less,",
     "Dir::Higher => cmp_val(c, min)? != std::cmp::Ordering::Greater,"),
    ("M4", "R7 guard at its bound loses (lower)",
     "Dir::Lower => cmp_val(c, max)? == std::cmp::Ordering::Greater,",
     "Dir::Lower => cmp_val(c, max)? != std::cmp::Ordering::Less,"),
    ("M5", "R3 any target suffices",
     "!cleared.is_empty() && cleared.iter().all(|&c| c)",
     "cleared.iter().any(|&c| c)"),
    ("M6", "both arms winning picks J6(f) instead of refusing",
     "(true, false) => Some((",
     "(true, _) => Some(("),
    ("M7", "R4 struck: no unseen-language guard on J6(d)-v4",
     "guards: &[PROSE, SCRAMBLED, UNSEEN, ID_ABSTAIN_DC],",
     "guards: &[PROSE, SCRAMBLED, ID_ABSTAIN_DC],"),
    ("M8", "R5 as first registered (controls guard J6(d)-v4), i.e. the amendment not applied",
     "guards: &[PROSE, SCRAMBLED, UNSEEN, ID_ABSTAIN_DC],",
     "guards: &[CONTROL_1K, CONTROL_2K, CONTROL_4K, PROSE, SCRAMBLED, UNSEEN, ID_ABSTAIN_DC],"),
    ("M9", "R5 amendment half-applied: J6(d)-v4 still requires its needle-control row",
     "            arm.reads_control(),\n            arm.reads_margin(),",
     "            true,\n            arm.reads_margin(),"),
    ("M10", "R1 struck: per-arm guards only, no base list",
     "for m in BASE_MUST_NOT_LOSE.iter().chain(self.guards) {",
     "for m in self.guards.iter() {"),
    ("M11", "envelope pin: any three seeds",
     "        seeds == F_SEEDS,\n        \"the envelope is {ENVELOPE_PIN}",
     "        seeds.len() == F_SEEDS.len(),\n        \"the envelope is {ENVELOPE_PIN}"),
    ("M12", "R8 off: no arm identity check",
     "fn arm_identity(arm: &Arm, ft: &Row, reference: &Row) -> Result<()> {\n",
     "fn arm_identity(arm: &Arm, ft: &Row, reference: &Row) -> Result<()> {\n    if arm.name != \"\" {\n        return Ok(());\n    }\n"),
    ("M13", "margin grid check off",
     "        close(value, exact.f64()),\n",
     "        true || close(value, exact.f64()),\n"),
    ("M14", "letter-control row selected without the per-family key",
     "&& r.get(&[\"metrics\", MARGIN_KEY]).is_some()",
     "&& (true || r.get(&[\"metrics\", MARGIN_KEY]).is_some())"),
    ("M15", "an arm whose rows cannot be read counts as not winning (missing -> quiet)",
     "        Err(e) => {\n            add(format!(\"(b) {e}\"));\n            None\n        }",
     "        Err(_) => Some((\"quiet\", \"mutated\")),"),
    ("M16", "ECE direction flipped (higher is better)",
     "    source: Source::Float(ECE_DC_KEY),\n    dir: Dir::Lower,",
     "    source: Source::Float(ECE_DC_KEY),\n    dir: Dir::Higher,"),
    ("M17", "in-distribution abstention direction flipped",
     "    \"ood_abstain.in_distribution.family.code.defect_class\",\n    Dir::Lower,",
     "    \"ood_abstain.in_distribution.family.code.defect_class\",\n    Dir::Higher,"),
    ("M18", "eval comparability (val set / suites) not checked",
     "&[\"val_shard_hash\", \"needle\", \"ood\"],",
     "&[],"),
    ("N1", "R9_room off: no_room never blocks",
     "            Ok(match dir {\n                Dir::Higher => (2 * g * f - e * h) * v >= u * f * h,",
     "            Ok(false && match dir {\n                Dir::Higher => (2 * g * f - e * h) * v >= u * f * h,"),
    ("N2", "R9_room higher form at equality has room (>= -> >)",
     "Dir::Higher => (2 * g * f - e * h) * v >= u * f * h,",
     "Dir::Higher => (2 * g * f - e * h) * v > u * f * h,"),
    ("N3", "R9_room lower form at equality has room (<= -> <)",
     "Dir::Lower => (2 * e * h - g * f) * v <= u * f * h,",
     "Dir::Lower => (2 * e * h - g * f) * v < u * f * h,"),
    ("N4", "R9_room lower-better uses the higher-better form",
     "Dir::Lower => (2 * e * h - g * f) * v <= u * f * h,",
     "Dir::Lower => (2 * g * f - e * h) * v >= u * f * h,"),
    ("N5", "room decided and listed but (c) does not refuse",
     "        .filter_map(Room::refusal)\n",
     "        .filter_map(|r| r.refusal().filter(|s| s.starts_with(\"(b)\")))\n"),
    ("N6", "R9_room applied to J6(f)'s targets only",
     "        .flat_map(|(arm, _)| arm.targets.iter().map(|m| Room::of(arm, *m, &envelope)))",
     "        .filter(|(arm, _)| arm.name == \"j6f\")\n        .flat_map(|(arm, _)| arm.targets.iter().map(|m| Room::of(arm, *m, &envelope)))"),
    ("N7", "an arm-row refusal propagates and drops the room already decided",
     "    let judged = judge_arms(inputs, arm_ledger, &envelope, arms);",
     "    let judged = Ok(judge_arms(inputs, arm_ledger, &envelope, arms)?);"),
    ("N8", "R9_room lower bound L = 1 instead of 0",
     "            name: \"L\",\n            value: Rat { num: 0, den: 1 },",
     "            name: \"L\",\n            value: Rat { num: 1, den: 1 },"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_tests() -> tuple[int, str, list[str], bool]:
    p = subprocess.run(CMD, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S)
    text = p.stdout + p.stderr
    result = re.findall(r"^test result: .*$", text, re.M)
    failed = sorted(set(re.findall(r"^test tests::(\S+) \.\.\. FAILED$", text, re.M)))
    compiled = bool(result)
    return p.returncode, result[-1] if result else "(no test result: did not compile)", failed, compiled


def main() -> int:
    original = SRC.read_bytes()
    digest = sha(original)
    lines = [f"source {SRC.relative_to(ROOT)} sha256 {digest}", f"command: {' '.join(CMD)}", ""]
    code, result, failed, _ = run_tests()
    lines.append(f"[unmutated] exit {code}: {result}")
    if code != 0:
        lines.append("unmutated source fails; no mutation result is meaningful")
        OUT.write_text("\n".join(lines) + "\n")
        return 1
    caught = not_compiled = 0
    text = original.decode()
    try:
        for mid, desc, old, new in MUTATIONS:
            count = text.count(old)
            if count != 1:
                lines.append(f"\n{mid} {desc}: NOT APPLIED, pattern occurs {count} times")
                continue
            SRC.write_text(text.replace(old, new))
            code, result, failed, compiled = run_tests()
            SRC.write_bytes(original)
            lines.append(f"\n{mid} {desc}: exit {code}: {result}")
            if not compiled:
                not_compiled += 1
                lines.append("    did not compile: not counted as caught")
                continue
            if code != 0 and failed:
                caught += 1
            else:
                lines.append("    SURVIVED")
            lines.extend(f"    FAILED {t}" for t in failed)
    finally:
        SRC.write_bytes(original)
    restored = SRC.read_bytes()
    code, result, _, _ = run_tests()
    lines.append(
        f"\n[restored, sha256 {sha(restored)}, equal: {restored == original}] exit {code}: {result}"
    )
    lines.append(
        f"mutations caught: {caught} of {len(MUTATIONS)}"
        + (f" ({not_compiled} did not compile)" if not_compiled else "")
    )
    OUT.write_text("\n".join(lines) + "\n")
    print(lines[-1])
    return 0 if caught == len(MUTATIONS) and restored == original else 1


if __name__ == "__main__":
    sys.exit(main())
