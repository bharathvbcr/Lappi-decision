"""Mutation pass over qd-post-f-rules' v5 rules (lane L-v5-rules, 2026-10-02).

Throwaway audit tooling, not shipped code, in the form of AUDIT/j6a-rule-2026-10-02/mutations.py:
apply one source mutation (an exact string that must occur exactly once), run the v5 tests
(`--test post_f_rules_v5`) and the bin's own unit tests, record the result and the failing tests,
restore the source byte for byte and check its sha256. A mutation that does not compile is
reported as such and is not counted as caught.

Run from the worktree root, under the machine's heavy-job lock (`--check` only counts patterns):
    bash tools/mac_heavy.sh "L-v5-rules mutations" \
        /Users/bharath/.venvs/ml/bin/python AUDIT/v5-rules-2026-10-02/mutations.py
"""

import hashlib
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "crates/qd-runtime/src/bin/qd_post_f_rules.rs"
OUT = ROOT / "AUDIT/v5-rules-2026-10-02/mutations.txt"
CMD = [
    "cargo", "test", "-j", "4", "--manifest-path", str(ROOT / "Cargo.toml"), "-p", "qd-runtime",
    "--test", "post_f_rules_v5", "--bin", "qd-post-f-rules",
]
TIMEOUT_S = 900

MUTATIONS = [
    # --- count forms and their boundaries ---
    ("V1", "a count form's floor is strict (R9, seed_holds: >= -> >)",
     "if self.at_least { l >= r } else { l <= r }",
     "if self.at_least { l > r } else { l <= r }"),
    ("V2", "a count form's ceiling is strict (absolute guard: <= -> <)",
     "if self.at_least { l >= r } else { l <= r }",
     "if self.at_least { l >= r } else { l < r }"),
    ("V3", "R9 continues on either count instead of both",
     "let go = span_ok && needle_ok;",
     "let go = span_ok || needle_ok;"),
    ("V4", "R9's words swapped",
     "PAUSE_WORDS[usize::from(!go)]",
     "PAUSE_WORDS[usize::from(go)]"),
    ("V5", "R9 accepts a ceiling form",
     "span.at_least && needle.at_least,",
     "true,"),
    # --- room ---
    ("V6", "room needs v5 to hold on fewer than K seeds (<= -> <)",
     "held.holding() <= rule.room_at_most",
     "held.holding() < rule.room_at_most"),
    ("V7", "--room prints room only when every target has room",
     "let any_room = room.iter().any(|(_, r)| *r);",
     "let any_room = room.iter().all(|(_, r)| *r);"),
    ("V8", "--room's words swapped",
     "ROOM_WORDS[usize::from(!any_room)]",
     "ROOM_WORDS[usize::from(any_room)]"),
    # --- targets ---
    ("V9", "a target clears when the arm holds on as many seeds as v5",
     "let clears = arm_n == n && arm_n > v5_n;",
     "let clears = arm_n == n && arm_n >= v5_n;"),
    ("V10", "a target clears without every arm seed holding",
     "let clears = arm_n == n && arm_n > v5_n;",
     "let clears = arm_n > v5_n;"),
    ("V11", "a no-room target read as a guard tolerates one seed short",
     "let loses = arm_n < n;",
     "let loses = arm_n + 1 < n;"),
    ("V12", "seed_holds counts over any number of cases",
     "f.n == rule.bar.n,",
     "true,"),
    # --- guards ---
    ("V13", "a higher-better guard reads the arm's best seed",
     "Dir::Higher => arm_min,",
     "Dir::Higher => arm_max,"),
    ("V14", "a lower-better guard reads the arm's best seed",
     "Dir::Lower => arm_max,",
     "Dir::Lower => arm_min,"),
    ("V15", "the absolute guard never loses",
     ".filter(|(_, _, ok)| !ok)",
     ".filter(|(_, _, ok)| !ok && false)"),
    # --- identity and comparability ---
    ("V16", "the added keys' values are not checked",
     "if !ok {",
     "if false && !ok {"),
    ("V17", "a recipe key other than the added ones may differ",
     "if !rule.added.iter().any(|(k, _)| k == key) && got.get(key) != base.get(key) {",
     "if !rule.added.iter().any(|(k, _)| k == key) && false {"),
    ("V18", "code_commit / data snapshot / shard hash not compared",
     "if want.is_none() || got != want {",
     "if want.is_none() {"),
    ("V19", "an arm row that is quick is accepted",
     'if ft.get(&["quick"]) != Some(&Value::Bool(false)) {',
     'if ft.get(&["quick"]) == Some(&Value::Null) {'),
    ("V20", "the arm's eval rows are not held comparable to v5 seed 0's",
     'comparable(&rows, reference, &format!("arm seed {seed}"))?;',
     "let _ = reference;"),
    ("V21", "v5's envelope is not checked as one configuration",
     "one_configuration(&env, ft_rows)?;",
     "let _ = &env;"),
    ("V22", "seeds34 under v5's pre-registration skips one_configuration",
     "    one_configuration(&ledger, ft_rows)?;\n    let seeds = seeds_in",
     "    let seeds = seeds_in"),
    # --- the pre-registration ---
    ("V23", "a DRAFT is read as a rule",
     '!p.contains_key("draft"),',
     "true,"),
    ("V24", "a listed direction that disagrees is accepted",
     "dir_word(m.dir) == direction,",
     "true,"),
    ("V25", "seed_holds' printed bar is not checked against F2",
     "printed == f2_bar,",
     "true,"),
    ("V26", "the eval-row recipe keys are not checked against comparable's",
     "keys == COMPARABLE_EVAL_KEYS,",
     "true,"),
    ("V27", "the absolute guard is not checked against F3",
     "ag_form.is_bound(pct, 100),",
     "true,"),
    ("V28", "seeds34's seed set is not checked against seeds.v5",
     "seeds == reading.seeds,",
     "true,"),
    # --- the pairing check (Fable's seed-order ruling) and the arm's one configuration ---
    ("V29", "the pairing check accepts two different orders",
     "(Ok(a), Ok(b)) if a == b => Some(a),",
     "(Ok(a), Ok(_)) => Some(a),"),
    ("V30", "the pairing check accepts an absent or unread order",
     "for e in [a.err(), b.err()].into_iter().flatten() {",
     "for e in [a.err(), b.err()].into_iter().flatten().filter(|_| false) {"),
    ("V31", "the pairing check reads only the arm's side",
     "match (paired(ft), paired(v5)) {",
     "match (paired(ft), paired(ft)) {"),
    ("V32", "the paired metric's name is not read from the pre-registration",
     'row.ran("metrics", &rule.paired)?',
     'row.ran("metrics", "corpus.plan_order_digest")?'),
    ("V33", "the arm's ft rows are not checked as one configuration",
     "let (_, arm_recipe_hash, _) = one_configuration(&ledger, arm_ft)?;",
     "let arm_recipe_hash = rule.ft_tag.clone();"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_tests() -> tuple[int, str, list[str], bool]:
    p = subprocess.run(CMD, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S)
    text = p.stdout + p.stderr
    result = re.findall(r"^test result: .*$", text, re.M)
    failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED$", text, re.M)))
    compiled = bool(result)
    summary = " | ".join(result) if result else "(no test result: did not compile)"
    return p.returncode, summary, failed, compiled


def main() -> int:
    original = SRC.read_bytes()
    text = original.decode()
    bad = [(m, text.count(old)) for m, _, old, _ in MUTATIONS if text.count(old) != 1]
    if "--check" in sys.argv:
        print("patterns not occurring exactly once:", bad or "none", f"({len(MUTATIONS)} checked)")
        return 1 if bad else 0
    if bad:
        print("refused: patterns not occurring exactly once:", bad)
        return 1
    digest = sha(original)
    lines = [f"source {SRC.relative_to(ROOT)} sha256 {digest}", f"command: {' '.join(CMD)}", ""]
    code, result, failed, _ = run_tests()
    lines.append(f"[unmutated] exit {code}: {result}")
    if code != 0:
        lines.append(f"  failing before any mutation: {failed}")
        OUT.write_text("\n".join(lines) + "\n")
        return 1
    caught = 0
    try:
        for name, what, old, new in MUTATIONS:
            SRC.write_text(text.replace(old, new, 1))
            code, result, failed, compiled = run_tests()
            if not compiled:
                verdict = "DID NOT COMPILE (not counted)"
            elif code != 0 and failed:
                verdict = "caught"
                caught += 1
            else:
                verdict = "SURVIVED"
            lines.append(f"{name} {verdict}: {what}")
            lines.append(f"    {result}")
            if failed:
                lines.append(f"    failed: {', '.join(failed)}")
    finally:
        SRC.write_bytes(original)
    assert sha(SRC.read_bytes()) == digest, "source not restored"
    lines.append("")
    lines.append(f"caught {caught} of {len(MUTATIONS)}; source restored, sha256 {digest}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
