"""Mutation pass over qd-post-f-rules' successor and j6a rules (lane L-room, 2026-10-02).

Throwaway audit tooling, not shipped code. The form is AUDIT/succ-room-2026-10-02/mutations.py's:
apply one source mutation (an exact string that must occur exactly once), run
`cargo test -p qd-runtime --bin qd-post-f-rules`, record the result and the failing tests,
restore the source byte for byte and check its sha256. A mutation that does not compile is
reported as such and is not counted as caught.

M1-M18 and N1-N8 are the succ-room list, with the patterns that the j6a refactor moved
(M6, M12, M15, N5, N7) re-pointed at the same logic; J0-J20 are new, on the j6a code and on
the claim that arm_identity's snapshot equality is not loosened (J0).

Run from the worktree root (`--check` only counts each pattern):
    /Users/bharath/.venvs/ml/bin/python AUDIT/j6a-rule-2026-10-02/mutations.py [--check]
"""

import hashlib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "crates/qd-runtime/src/bin/qd_post_f_rules.rs"
OUT = ROOT / "AUDIT/j6a-rule-2026-10-02/mutations.txt"
CMD = ["cargo", "test", "-p", "qd-runtime", "--bin", "qd-post-f-rules"]
TIMEOUT_S = 900

MUTATIONS = [
    # --- the successor (succ-room's list) ---
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
     "Some(&[true, false]) => Some((",
     "Some(&[true, _]) => Some(("),
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
     "fn arm_identity(arm: &Arm, ft: &Row, reference: &Row) -> Result<Value> {\n",
     "fn arm_identity(arm: &Arm, ft: &Row, reference: &Row) -> Result<Value> {\n    if arm.name != \"\" {\n        return Ok(Value::Null);\n    }\n"),
    ("M13", "margin grid check off",
     "        close(value, exact.f64()),\n",
     "        true || close(value, exact.f64()),\n"),
    ("M14", "letter-control row selected without the per-family key",
     "&& r.get(&[\"metrics\", MARGIN_KEY]).is_some()",
     "&& (true || r.get(&[\"metrics\", MARGIN_KEY]).is_some())"),
    ("M15", "an arm whose rows cannot be read counts as not winning (missing -> quiet)",
     "        Err(e) => s.refuse(format!(\"{} {e}\", labels.unreadable)),",
     "        Err(_) => s.wins = Some(vec![false, false]),"),
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
    ("N5", "room decided and listed but no-room does not refuse",
     "for reason in decided.room.iter().filter_map(|r| r.refusal(labels)) {",
     "for reason in decided\n        .room\n        .iter()\n        .filter_map(|r| r.refusal(labels).filter(|s| !s.contains(\"cannot clear\")))\n    {"),
    ("N6", "R9_room applied to J6(f)'s targets only",
     "        .flat_map(|(arm, _)| arm.targets.iter().map(|m| Room::of(arm, *m, &envelope)))",
     "        .filter(|(arm, _)| arm.name == \"j6f\")\n        .flat_map(|(arm, _)| arm.targets.iter().map(|m| Room::of(arm, *m, &envelope)))"),
    ("N7", "an arm-row refusal propagates and drops the room already decided",
     "    let judged = judge_arms(inputs, arm_ledger, &envelope, arms, identity);",
     "    let judged = Ok(judge_arms(inputs, arm_ledger, &envelope, arms, identity)?);"),
    ("N8", "R9_room lower bound L = 1 instead of 0",
     "            name: \"L\",\n            value: Rat { num: 0, den: 1 },",
     "            name: \"L\",\n            value: Rat { num: 1, den: 1 },"),
    # --- j6a ---
    ("J0", "arm_identity's snapshot equality loosened (the check Fable says must stand)",
     "data.is_some() && data == reference.str_at(&[\"protocol\", \"data_snapshot_hash\"]),",
     "true || data == reference.str_at(&[\"protocol\", \"data_snapshot_hash\"]),"),
    ("J1", "J6(a) on F's snapshot accepted",
     "f_data.is_some() && data != f_data,",
     "f_data.is_some(),"),
    ("J2", "J6(a)'s snapshot not checked against build 2's pin",
     "data == Some(pins.data_snapshot_hash.as_str()),",
     "true || data == Some(pins.data_snapshot_hash.as_str()),"),
    ("J3", "J6(a) quick not required",
     "ft.get(&[\"quick\"]) == Some(&Value::Bool(true)),",
     "true || ft.get(&[\"quick\"]) == Some(&Value::Bool(true)),"),
    ("J4", "J6(a) code_commit not checked",
     "ft.str_at(&[\"code_commit\"]) == Some(J6A_CODE_COMMIT),",
     "true || ft.str_at(&[\"code_commit\"]) == Some(J6A_CODE_COMMIT),"),
    ("J5", "a replay key on F's recipe accepted",
     "!base.contains_key(key),",
     "true || !base.contains_key(key),"),
    ("J6", "the replay keys' values not checked",
     "let ok = match key {",
     "let ok = true || match key {"),
    ("J7", "R1 off: batches / width compared",
     "        .chain(&DERIVED_RECIPE_KEYS)\n",
     ""),
    ("J8", "an undeclared recipe difference ignored",
     "if declared.contains(key.as_str()) {",
     "if true || declared.contains(key.as_str()) {"),
    ("J9", "J6(a)'s shard_hash not checked against its pin",
     "got.get(\"shard_hash\").and_then(Value::as_str) == Some(pins.shard_hash.as_str()),",
     "true || got.get(\"shard_hash\").and_then(Value::as_str) == Some(pins.shard_hash.as_str()),"),
    ("J10", "pending amendments accepted (the flags stand in for the file)",
     "let filed = amendments_of(p)?;",
     "let filed = amendments_of(p).unwrap_or_else(|_| cli.clone());"),
    ("J11", "an extra key in amendments accepted",
     "        extra.is_empty(),\n        \"the pre-registration's amendments carry",
     "        true || extra.is_empty(),\n        \"the pre-registration's amendments carry"),
    ("J12", "the flags not checked against the file's pins",
     "f[key] == c[key],",
     "true || f[key] == c[key],"),
    ("J13", "h not bounded by the replay set",
     "self.h <= REPLAY_SET_ROWS,",
     "true || self.h <= REPLAY_SET_ROWS,"),
    ("J14", "a pin's sha256 form not checked",
     "v.len() == 64 &&",
     "true || v.len() == 64 &&"),
    ("J15", "J6(a) refusals under the successor's labels",
     "const J6A_LABELS: Labels = Labels {\n    unreadable: \"(a)\",\n    no_room: \"(b)\",\n};",
     "const J6A_LABELS: Labels = SUCC_LABELS;"),
    ("J16", "the pre-registration's structured fields not checked against the checker",
     "    j6a_agrees(&prereg).map_err(|e| format!(\"{} {e}\", J6A_LABELS.unreadable))?;\n",
     "    let _ = j6a_agrees(&prereg);\n"),
    ("J17", "a J6(a) target dropped",
     "        ID_ABSTAIN_KNOWLEDGE,\n        ID_ABSTAIN_COMMONSENSE,\n    ],\n    guards: &[ID_ABSTAIN_DC, PROSE, SCRAMBLED, UNSEEN],",
     "        ID_ABSTAIN_KNOWLEDGE,\n    ],\n    guards: &[ID_ABSTAIN_DC, PROSE, SCRAMBLED, UNSEEN],"),
    ("J18", "a J6(a) guard dropped (unseen-language)",
     "guards: &[ID_ABSTAIN_DC, PROSE, SCRAMBLED, UNSEEN],",
     "guards: &[ID_ABSTAIN_DC, PROSE, SCRAMBLED],"),
    ("J19", "pending amendments not listed when an arm row failed first",
     "    if let Err(e) = &pins {\n",
     "    if let (false, Err(e)) = (true, &pins) {\n"),
    ("J21", "pending amendments dropped when F's envelope does not resolve",
     "        Err(p) => format!(\n            \"{}; {} {p}\",",
     "        Err(_) => no_envelope(J6A_LABELS, &e),\n        Err(p) => format!(\n            \"{}; {} {p}\","),
    ("J20", "J6(a) judged under R8 instead of its own identity",
     "        replay_identity(arm, ft, reference, pins)\n",
     "        let _ = pins;\n        arm_identity(arm, ft, reference)\n"),
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
    text = original.decode()
    if "--check" in sys.argv:
        bad = [(m, text.count(old)) for m, _, old, _ in MUTATIONS if text.count(old) != 1]
        print("patterns not occurring exactly once:", bad or "none", f"({len(MUTATIONS)} checked)")
        return 1 if bad else 0
    digest = sha(original)
    lines = [f"source {SRC.relative_to(ROOT)} sha256 {digest}", f"command: {' '.join(CMD)}", ""]
    code, result, failed, _ = run_tests()
    lines.append(f"[unmutated] exit {code}: {result}")
    if code != 0:
        lines.append("unmutated source fails; no mutation result is meaningful")
        OUT.write_text("\n".join(lines) + "\n")
        return 1
    caught = not_compiled = 0
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
