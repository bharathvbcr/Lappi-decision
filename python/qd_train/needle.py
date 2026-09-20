"""The needle-hunk recall test: one of the four gates that decide whether the 2B ships.

A whole-diff read is the plan's central bet — *"With a verify-time latency budget of
seconds for DevCouncil, the model reads a diff to 8K in one prefill; hunk pooling is
the fallback above that, not the design."* That bet rests on the model still being able
to **find** a hunk that appeared early in an 8K context.

For a hybrid with 18 recurrent layers, the failure mode is specific and directional:
the GDN state is a fixed-size summary, so information about early tokens degrades as
later tokens overwrite it. A model that has lost the first 2K still scores well on an
**aggregate** recall number, because most needles are not at the start.

**So recall is always reported by depth, never as one number.** An aggregate is
computed but is explicitly not the gate; `recall_by_depth` is. This is the same reason
the plan keeps the next attention layer above L* when pruning — and this suite is what
would catch it if that did not work.

Wilson intervals rather than normal approximation, following nanolab's `mqar_suite.py`:
at recall near 1.0, which is where a passing model sits, the normal interval runs past
1 and understates uncertainty exactly where the decision is made.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from .tristate import NotRun, Ran, TriState

__all__ = [
    "DepthBucket",
    "NeedleCase",
    "NeedleReport",
    "build_suite",
    "score_suite",
    "wilson_interval",
]

# Behaviour-neutral filler hunks, one per pool language. The haystack must look like
# the real input distribution: filler that is obviously not code would let a model
# find the needle by noticing it is the only plausible hunk.
_FILLER: dict[str, tuple[str, ...]] = {
    "rust": (
        "@@ -{a},6 +{a},7 @@ impl {name} {{\n"
        "     pub fn {fn}(&self, input: &str) -> Result<usize> {{\n"
        "-        let parsed = input.trim().parse::<usize>()?;\n"
        "+        let parsed = input.trim().parse::<usize>().unwrap_or(0);\n"
        "         Ok(parsed + self.offset)\n     }}\n",
        "@@ -{a},4 +{a},5 @@ fn {fn}(items: &[Item]) -> Vec<Summary> {{\n"
        "     items.iter()\n+        .filter(|i| i.active)\n"
        "         .map(Summary::from)\n         .collect()\n }}\n",
    ),
    "go": (
        "@@ -{a},5 +{a},6 @@ func ({r} *{name}) {fn}(ctx context.Context) error {{\n"
        "     rows, err := {r}.db.QueryContext(ctx, q)\n     if err != nil {{\n"
        "-        return err\n+        return fmt.Errorf(\"{fn}: %w\", err)\n     }}\n",
        "@@ -{a},3 +{a},4 @@ func {fn}(xs []int) int {{\n"
        "     total := 0\n     for _, x := range xs {{\n+        if x < 0 {{ continue }}\n"
        "         total += x\n     }}\n",
    ),
    "python": (
        "@@ -{a},4 +{a},5 @@ class {name}:\n"
        "     def {fn}(self, payload: dict) -> dict:\n"
        "-        return {{k: v for k, v in payload.items()}}\n"
        "+        return {{k: v for k, v in payload.items() if v is not None}}\n",
        "@@ -{a},3 +{a},4 @@ def {fn}(values):\n"
        "     scaled = [v * 2 for v in values]\n+    scaled.sort()\n     return scaled\n",
    ),
    "typescript": (
        "@@ -{a},5 +{a},6 @@ export class {name} {{\n"
        "   async {fn}(id: string): Promise<Record<string, unknown>> {{\n"
        "-    const res = await fetch(`/api/${{id}}`);\n"
        "+    const res = await fetch(`/api/${{encodeURIComponent(id)}}`);\n"
        "     return res.json();\n   }}\n",
    ),
    "swift": (
        "@@ -{a},4 +{a},5 @@ extension {name} {{\n"
        "     func {fn}(_ items: [Item]) -> [Summary] {{\n"
        "-        items.map {{ Summary($0) }}\n"
        "+        items.filter(\\.isActive).map {{ Summary($0) }}\n     }}\n",
    ),
}

# The needle: a silent stub. Deliberately NOT a `todo!()` — a marker-shaped needle is
# findable by lexical match, which measures the tokenizer rather than recall.
_NEEDLE: dict[str, str] = {
    "rust": "@@ -{a},7 +{a},3 @@ impl {name} {{\n"
            "     pub fn {fn}(&self, items: &[Item]) -> Result<usize> {{\n"
            "-        let mut total = 0;\n-        for item in items {{\n"
            "-            total += item.weight * self.scale;\n-        }}\n-        Ok(total)\n"
            "+        Ok(0)\n     }}\n",
    "go": "@@ -{a},6 +{a},3 @@ func ({r} *{name}) {fn}(items []Item) (int, error) {{\n"
          "-    total := 0\n-    for _, it := range items {{\n"
          "-        total += it.Weight\n-    }}\n"
          "-    return total, nil\n+    return 0, nil\n }}\n",
    "python": "@@ -{a},6 +{a},2 @@ class {name}:\n"
              "     def {fn}(self, items):\n-        total = 0\n-        for it in items:\n"
              "-            total += it.weight\n-        return total\n+        return 0\n",
    "typescript": "@@ -{a},6 +{a},3 @@ export class {name} {{\n"
                  "   {fn}(items: Item[]): number {{\n-    let total = 0;\n"
                  "-    for (const it of items) total += it.weight;\n-    return total;\n"
                  "+    return 0;\n   }}\n",
    "swift": "@@ -{a},6 +{a},3 @@ extension {name} {{\n"
             "     func {fn}(_ items: [Item]) -> Int {{\n-        var total = 0\n"
             "-        for it in items {{ total += it.weight }}\n-        return total\n"
             "+        return 0\n     }}\n",
}

_NAMES = ("Indexer", "Resolver", "Scheduler", "Collector", "Validator", "Encoder", "Planner")
_FNS = ("total_weight", "compute_score", "collect_items", "resolve_all", "summarize")


@dataclass(frozen=True, slots=True)
class NeedleCase:
    case_id: str
    language: str
    context: str
    needle_index: int          # which hunk (0-based) is the needle
    n_hunks: int
    depth_fraction: float      # 0.0 = very start of the context, 1.0 = very end
    needle_start_line: int     # 1-based, inclusive — matches the `span` slot convention
    needle_end_line: int
    approx_tokens: int

    @property
    def depth_bucket(self) -> str:
        edges = ((0.2, "0-20%"), (0.4, "20-40%"), (0.6, "40-60%"), (0.8, "60-80%"))
        for edge, label in edges:
            if self.depth_fraction < edge:
                return label
        return "80-100%"


@dataclass(frozen=True, slots=True)
class DepthBucket:
    label: str
    correct: int
    total: int
    lo: float
    hi: float

    @property
    def recall(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(slots=True)
class NeedleReport:
    by_depth: list[DepthBucket] = field(default_factory=list)
    aggregate_correct: int = 0
    aggregate_total: int = 0

    @property
    def aggregate_recall(self) -> float:
        return self.aggregate_correct / self.aggregate_total if self.aggregate_total else 0.0

    def summary(self) -> str:
        lines = [
            f"needle-hunk recall: {self.aggregate_recall:.3f} "
            f"({self.aggregate_correct}/{self.aggregate_total}) AGGREGATE — not the gate",
            "  by depth (this is the gate):",
        ]
        for b in self.by_depth:
            lines.append(
                f"    {b.label:>8}: {b.recall:.3f} ({b.correct}/{b.total}) "
                f"[95% CI {b.lo:.3f}, {b.hi:.3f}]"
            )
        return "\n".join(lines)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because a passing model sits near recall
    1.0, where the normal interval runs past 1 and understates uncertainty exactly
    where the decision is being made. Follows nanolab's `mqar_suite._wilson`.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _approx_tokens(text: str) -> int:
    """Rough token count for sizing the haystack.

    Deliberately a heuristic and named as one: the real tokenizer is not available
    here, and a case built to ~8K by this estimate must be re-measured with the real
    tokenizer before any claim about "8K" is made. Code tokenizes at roughly 3
    characters per token.
    """
    return max(1, len(text) // 3)


def build_suite(
    *,
    target_tokens: int = 8192,
    cases_per_depth: int = 20,
    languages: tuple[str, ...] = ("rust", "go", "python", "typescript", "swift"),
    seed: int = 0,
) -> list[NeedleCase]:
    """Build needles at controlled depths through a ~`target_tokens` haystack.

    Depths are swept uniformly rather than sampled, so every bucket is populated by
    construction. A randomly-placed needle set leaves the early buckets thin, which is
    precisely where a recurrent model fails and precisely where the estimate would
    then be least certain.
    """
    if target_tokens < 256:
        raise ValueError(f"target_tokens must be at least 256, got {target_tokens}")
    if cases_per_depth < 1:
        raise ValueError(f"cases_per_depth must be >= 1, got {cases_per_depth}")
    unknown = set(languages) - set(_FILLER)
    if unknown:
        raise ValueError(f"no filler hunks for language(s): {sorted(unknown)}")

    rng = random.Random(seed)
    cases: list[NeedleCase] = []
    depths = [i / (cases_per_depth * 5 - 1) for i in range(cases_per_depth * 5)]

    for n, depth in enumerate(depths):
        lang = languages[n % len(languages)]
        filler_pool = _FILLER[lang]

        # Grow a haystack of filler hunks until it is about the target size.
        hunks: list[str] = []
        tokens = 0
        while tokens < target_tokens:
            tpl = filler_pool[rng.randrange(len(filler_pool))]
            hunk = tpl.format(
                a=rng.randint(10, 900),
                name=rng.choice(_NAMES),
                fn=rng.choice(_FNS),
                r=rng.choice(("s", "c", "m")),
            )
            hunks.append(hunk)
            tokens += _approx_tokens(hunk)

        needle = _NEEDLE[lang].format(
            a=rng.randint(10, 900), name=rng.choice(_NAMES), fn=rng.choice(_FNS),
            r=rng.choice(("s", "c", "m")),
        )
        position = min(len(hunks), max(0, round(depth * len(hunks))))
        hunks.insert(position, needle)

        before = "".join(hunks[:position])
        start_line = before.count("\n") + 1
        end_line = start_line + needle.rstrip("\n").count("\n")
        context = "".join(hunks)

        cases.append(
            NeedleCase(
                case_id=f"needle-{lang}-{n:04d}-"
                        f"{hashlib.sha256(context.encode()).hexdigest()[:8]}",
                language=lang,
                context=context,
                needle_index=position,
                n_hunks=len(hunks),
                depth_fraction=position / max(len(hunks) - 1, 1),
                needle_start_line=start_line,
                needle_end_line=end_line,
                approx_tokens=_approx_tokens(context),
                )
        )
    return cases


def score_suite(
    cases: list[NeedleCase],
    predictions: dict[str, int | None],
    *,
    min_recall: float,
    min_per_bucket: int = 5,
) -> tuple[NeedleReport, TriState]:
    """Score predicted needle hunk indices, bucketed by depth.

    `predictions` maps `case_id` to the predicted hunk index, or `None` for abstain.
    A case with no prediction at all is **not** scored as wrong — it is missing data,
    and the gate returns `not_run` rather than quietly treating an unevaluated suite
    as a failing one.

    `min_recall` is **required and has no default**, deliberately. A gate function with
    a default threshold is a gate that passes when the caller forgets to set one, which
    is the same fail-open shape as the fla log-grep this codebase already removed. The
    plan does not state a numeric threshold for this test, so the caller states it and
    the ledger records which value was used.
    """
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError(f"min_recall must be a recall in [0, 1], got {min_recall}")
    if not cases:
        return NeedleReport(), NotRun(reason="needle suite was empty; nothing was measured")

    missing = [c.case_id for c in cases if c.case_id not in predictions]
    if missing:
        return NeedleReport(), NotRun(
            reason=(
                f"{len(missing)} of {len(cases)} cases have no prediction "
                f"(e.g. {missing[0]}). An unevaluated suite is not a failing one."
            )
        )

    buckets: dict[str, list[bool]] = {}
    for c in cases:
        buckets.setdefault(c.depth_bucket, []).append(predictions[c.case_id] == c.needle_index)

    report = NeedleReport()
    order = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    thin: list[str] = []
    for label in order:
        hits = buckets.get(label)
        if not hits:
            continue
        k, n = sum(hits), len(hits)
        if n < min_per_bucket:
            thin.append(f"{label} has only {n}")
        lo, hi = wilson_interval(k, n)
        report.by_depth.append(DepthBucket(label=label, correct=k, total=n, lo=lo, hi=hi))
        report.aggregate_correct += k
        report.aggregate_total += n

    if thin:
        return report, NotRun(
            reason=(
                f"depth buckets too thin to read ({'; '.join(thin)}; need {min_per_bucket}). "
                "The early buckets are where a recurrent model fails, so a thin early "
                "bucket is the one that must not be guessed at."
            )
        )

    # The gate is the WEAKEST depth bucket, not the average. A model that has lost the
    # first 2K of context still posts a strong aggregate.
    worst = min(report.by_depth, key=lambda b: b.recall)
    passed = worst.recall >= min_recall
    verdict = "" if passed else (
        f" -- FAILS: depth {worst.label} is below the {min_recall:.2f} threshold. "
        "A recurrent model losing early context looks exactly like this while its "
        "aggregate still reads well."
    )
    return report, Ran(
        passed=passed,
        value=worst.recall,
        n=report.aggregate_total,
        n_total=report.aggregate_total,
        detail=(
            f"worst depth bucket {worst.label} at {worst.recall:.3f} "
            f"[95% CI {worst.lo:.3f}, {worst.hi:.3f}] vs threshold {min_recall:.2f}; "
            f"aggregate {report.aggregate_recall:.3f} (reported, not the gate){verdict}"
        ),
    )
