"""Find the fixture inputs that restore fixtures/wire/'s answer shapes under prompt format 2.

Throwaway analysis (lane L-v5-fmt, 2026-10-02); it ships nothing. The reference backend's
answers are a function of the rendered prompt's bytes (crates/qd-runtime/src/reference.rs:104-112),
so prompt format 2 re-rolled every answer fixture, and three of them -- example-three-slots,
score-only, span-only -- went from answered to abstained. This searches, deterministically, for
the smallest change to their CONTEXT bytes that shows each fixture's format-1 shape again:

* example-three-slots and score-only share one context: SAMPLE_CONTEXT with ``// {i}`` appended
  to its first line. Wanted: verdict and severity answered, evidence abstained (three-slots) and
  severity answered (score-only).
* span-only: its four-line context with `` {j}`` appended to its first line. Wanted: evidence
  answered, as {start_line, end_line}.

The question is not touched: docs/schema-api.md quotes it. Candidate 0 is the unchanged input,
shown for the record. Each candidate is answered by the real ``qd oneshot --reference-backend``,
the same runtime `fixtures.rs` builds (ReferenceBackend::new(true), CalibrationTable::reference(),
RenderCaps::DEFAULT; crates/qd-runtime/src/runtime.rs:95-96).

    python AUDIT/v5-fmt-2026-10-02/fixture_search.py <path to target/debug/qd>
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys

QUESTION = "Does this diff implement what the commit message claims?"
THREE = [
    {"name": "verdict", "type": "choice", "options": ["stub", "logic", "cosmetic", "clean"]},
    {"name": "severity", "type": "score", "bins": 5},
    {"name": "evidence", "type": "span"},
]
SCORE = [{"name": "severity", "type": "score", "bins": 5}]
SPAN = [{"name": "evidence", "type": "span"}]
MAX_CANDIDATES = 200


def code_context(i: int) -> bytes:
    first = "fn add(a: i32, b: i32) -> i32 {" + ("" if i == 0 else f" // {i}")
    return f"{first}\n    todo!()\n}}\n".encode()


def span_context(j: int) -> bytes:
    first = "alpha" + ("" if j == 0 else f" {j}")
    return f"{first}\nbeta\ngamma\ndelta\n".encode()


def ask(qd: str, context: bytes, slots: list[dict[str, object]]) -> dict[str, object]:
    request = {
        "schema_version": 1,
        "task": "devcouncil.verdict",
        "context_b64": base64.b64encode(context).decode(),
        "context_len": len(context),
        "question": QUESTION,
        "slots": slots,
        "route": "generic",
    }
    done = subprocess.run(
        [qd, "oneshot", "--reference-backend"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    reply = json.loads(done.stdout)
    if reply.get("status") != "ok":
        raise SystemExit(f"not an answer: {done.stdout} {done.stderr}")
    return reply


def shape(reply: dict[str, object]) -> dict[str, str]:
    slots = reply["slots"]
    assert isinstance(slots, dict)
    return {name: ("abstained" if s["noul"] else "answered") for name, s in sorted(slots.items())}


def search(qd: str, label: str, make, wants: list[tuple[list[dict[str, object]], dict[str, str]]]):
    for i in range(MAX_CANDIDATES):
        context = make(i)
        shapes = [shape(ask(qd, context, slots)) for slots, _ in wants]
        ok = all(got == want for got, (_, want) in zip(shapes, wants, strict=True))
        verdict = "PICK" if ok else "no"
        print(f"{label} candidate {i}: context={context!r} shapes={shapes} -> {verdict}")
        if ok and i > 0:
            return i, context
    raise SystemExit(f"{label}: no candidate in 0..{MAX_CANDIDATES} shows the wanted shape")


def main() -> None:
    qd = sys.argv[1]
    i, code = search(
        qd,
        "three-slots+score-only",
        code_context,
        [
            (THREE, {"evidence": "abstained", "severity": "answered", "verdict": "answered"}),
            (SCORE, {"severity": "answered"}),
        ],
    )
    j, span = search(qd, "span-only", span_context, [(SPAN, {"evidence": "answered"})])
    print(f"\nPICKED three-slots+score-only i={i}: {code!r}")
    print(f"PICKED span-only j={j}: {span!r}")
    for label, context, slots in (("three-slots", code, THREE), ("score-only", code, SCORE),
                                  ("span-only", span, SPAN)):
        print(f"{label} reply: {json.dumps(ask(qd, context, slots)['slots'], sort_keys=True)}")


if __name__ == "__main__":
    main()
