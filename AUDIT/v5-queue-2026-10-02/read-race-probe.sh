#!/bin/bash
# Probe the real qd-post-f-rules build (v5-build-wt debug, read-only) on ledgers with a malformed
# line, to ground the waiters' read-race match in the binary's own output. Writes only under
# this directory.
set -u
BIN=/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt/target/debug/qd-post-f-rules
D=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-afa17f069a43eb74a/build/l-v5-queue/race
ROW='{"row_id": "aaaaaaaa-1111-2222-3333-444444444444", "completed": true}'
rm -f "$D"/*.jsonl "$D"/*.json "$D"/*.err
printf '%s\n%s' "$ROW" '{"row_id": "half' > "$D/last.jsonl"
printf '%s\n%s\n%s\n' "$ROW" '{"row_id": "bad' "$ROW" > "$D/inner.jsonl"
printf '%s\n' "$ROW" > "$D/clean.jsonl"
echo "sha256 $(shasum -a 256 "$BIN" | cut -c1-64)"
for L in last inner; do
  out=$("$BIN" ft-rows --ledger "$D/$L.jsonl" --ft-row 0=aaaaaaaa-1111-2222-3333-444444444444 2> "$D/$L-ftrows.err")
  echo "== ft-rows on $L: exit $? stdout '$out'; last stderr line:"
  tail -n 1 "$D/$L-ftrows.err"
done
out=$("$BIN" seeds34 --f-ledger "$D/last.jsonl" --ft-row 0=aaaaaaaa-1111-2222-3333-444444444444 --ft-row 1=bbbbbbbb-1111-2222-3333-444444444444 --ft-row 2=cccccccc-1111-2222-3333-444444444444 --out "$D/last-seeds34.json" 2> "$D/last-seeds34.err")
echo "== seeds34 on last: exit $? stdout '$out'; last stderr line:"
tail -n 1 "$D/last-seeds34.err"
echo "== its --out JSON 'refused':"
grep '"refused"' "$D/last-seeds34.json"
out=$("$BIN" seeds34 --f-ledger "$D/last.jsonl" --ft-row 0=aaaaaaaa-1111-2222-3333-444444444444 --ft-row 1=bbbbbbbb-1111-2222-3333-444444444444 --ft-row 2=cccccccc-1111-2222-3333-444444444444 --out "$D/last-seeds34.json" 2> "$D/again.err")
echo "== the same --out again: exit $? stdout '$out'; last stderr line:"
tail -n 1 "$D/again.err"
