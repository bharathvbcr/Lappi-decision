"""The ``ood_abstain`` gate: does the decision model abstain on inputs it was never built for?

``docs/build-order-2026-09-19.md`` asks for the abstain rate "in-domain and OOD reported
separately" and defines nothing else; no plan text defines OOD or a threshold. The contract
here was approved by the human on 2026-09-30 on a Fable recommendation
(GAP-OOD-ABSTAIN-HAS-NO-CONTRACT):

* **The OOD set is generated**, as the needle suite is. The held-out families are not OOD by
  content (``code.language_id`` rows are built from the same commits as the training rows)
  and reading them for a gate would need a rule-3 ruling. Three categories, each asked as a
  ``code.defect_class`` request -- the question the model was trained on, over a context it
  was not: English prose from the general records' **val** split (MMLU/CSQA questions),
  code in languages the mutation pool never held, and scrambled code.
* **Scored on the choice slot** under the runtime's own rule for a generic-route choice
  (``crates/qd-runtime/src/answer.rs``, ``abstain_rule``): abstain when either pass answers
  ``noul``, when the permuted second pass names a different option, or when the calibrated
  margin is below ``noul_margin``. The margin half needs a fitted calibration table, which
  does not exist yet, so it is recorded as its own ``not_run`` metric and the gate is scored
  on the halves the runtime can apply today. The span slot is not the gate: abstaining there
  is the trained meaning of ``clean``, so it would measure "found no defect".
* **Paired with an in-distribution bound.** A model that always abstains would pass an OOD
  rate alone, so the gate also bounds abstention on val choice rows, none of whose golds is
  ``noul``.

Thresholds are read-only to an agent (rule 2): they change only by the human's say-so.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from .needle import _FILLER, wilson_interval
from .tristate import NotRun, Ran, TriState

if TYPE_CHECKING:
    from qd_data.config import DataConfig
    from qd_data.rows import DataRow

__all__ = [
    "OOD_CASES_PER_CATEGORY",
    "OOD_CATEGORIES",
    "OOD_MAX_IN_DISTRIBUTION_ABSTAIN",
    "OOD_MIN_ABSTAIN",
    "OodCase",
    "build_ood_suite",
    "ood_defect_row",
    "ood_gate",
]

#: The Wilson LOWER bound of the OOD abstain rate must reach this.
OOD_MIN_ABSTAIN: Final[float] = 0.90
#: The Wilson UPPER bound of the in-distribution (val choice) abstain rate must not exceed this.
OOD_MAX_IN_DISTRIBUTION_ABSTAIN: Final[float] = 0.05
#: Per category: the count the human approved for the needle suite, for the same reason --
#: fewer and a rate near the floor carries an interval too wide to read against it.
OOD_CASES_PER_CATEGORY: Final[int] = 60
OOD_CATEGORIES: Final[tuple[str, ...]] = ("prose", "unseen-language", "scrambled")

#: Hunks in languages the ``code.defect_class`` mutation pool does not hold (its rows are
#: python, go, rust and typescript). Behaviour-neutral, like the needle filler: the question
#: is whether the model notices the input is not its kind, not whether it finds a bug.
_UNSEEN: dict[str, tuple[str, str]] = {
    "c": (
        "@@ -{a},6 +{a},7 @@ static int {fn}(const struct {name} *s, size_t n)\n"
        " {{\n     int total = 0;\n     for (size_t i = 0; i < n; i++) {{\n"
        "-        total += s[i].weight;\n+        total += s[i].weight * s[i].scale;\n"
        "     }}\n     return total;\n",
        "c",
    ),
    "java": (
        "@@ -{a},5 +{a},6 @@ public final class {name} {{\n"
        "     public List<Item> {fn}(List<Item> items) {{\n"
        "-        return items.stream().collect(Collectors.toList());\n"
        "+        return items.stream()\n+            .filter(Item::isActive)\n"
        "+            .collect(Collectors.toList());\n     }}\n",
        "java",
    ),
    "ruby": (
        "@@ -{a},4 +{a},5 @@ class {name}\n   def {fn}(items)\n"
        "-    items.map(&:weight).sum\n+    items.select(&:active?).map(&:weight).sum\n"
        "   end\n+  alias_method :total, :{fn}\n end\n",
        "rb",
    ),
    "haskell": (
        "@@ -{a},4 +{a},5 @@ module {name} where\n"
        " {fn} :: [Item] -> Int\n-{fn} = sum . map weight\n"
        "+{fn} = sum . map weight . filter active\n+\n"
        " data Item = Item {{ weight :: Int, active :: Bool }}\n",
        "hs",
    ),
    "sql": (
        "@@ -{a},4 +{a},5 @@ CREATE VIEW {fn} AS\n SELECT {name_l}.id,\n"
        "-       SUM(w.weight) AS total\n+       SUM(w.weight) AS total,\n"
        "+       COUNT(*) AS n\n FROM {name_l} JOIN weights w ON w.owner = {name_l}.id\n"
        " GROUP BY {name_l}.id;\n",
        "sql",
    ),
    "lua": (
        "@@ -{a},5 +{a},6 @@ local {name} = {{}}\n function {name}.{fn}(items)\n"
        "   local total = 0\n-  for _, it in ipairs(items) do total = total + it.weight end\n"
        "+  for _, it in ipairs(items) do\n+    if it.active then total = total + it.weight end\n"
        "+  end\n   return total\n",
        "lua",
    ),
}

_NAMES = ("Indexer", "Resolver", "Scheduler", "Collector", "Validator", "Encoder", "Planner")
_FNS = ("total_weight", "compute_score", "collect_items", "resolve_all", "summarize")
_EXT = {"python": "py", "go": "go", "rust": "rs", "typescript": "ts"}


@dataclass(frozen=True, slots=True)
class OodCase:
    case_id: str
    category: str
    #: The context's language, or ``prose``. Carried as the row's language metadata.
    language: str
    path: str
    context: str


def _hunks(rng: random.Random, templates: Sequence[str], n: int, **extra: str) -> str:
    out = []
    for _ in range(n):
        name = rng.choice(_NAMES)
        out.append(
            templates[rng.randrange(len(templates))].format(
                a=rng.randint(10, 900), name=name, name_l=name.lower(), fn=rng.choice(_FNS),
                r=rng.choice(("s", "c", "m")), **extra,
            )
        )
    return "".join(out)


def build_ood_suite(
    prose: Sequence[str], *, per_category: int = OOD_CASES_PER_CATEGORY, seed: int = 0
) -> list[OodCase]:
    """``per_category`` cases of each of :data:`OOD_CATEGORIES`, deterministic in ``seed``.

    ``prose`` is the candidate prose pool (the general records' val-split questions); the
    cases are drawn from it by a seeded sample, so the same pool and seed give the same
    suite. Refuses a pool smaller than ``per_category`` rather than repeating texts.
    """
    if per_category < 1:
        raise ValueError(f"per_category must be >= 1, got {per_category}")
    pool = sorted({p.strip() for p in prose if p.strip()})
    if len(pool) < per_category:
        raise ValueError(
            f"the prose pool holds {len(pool)} distinct texts; {per_category} are needed"
        )
    rng = random.Random(f"qd_train.ood.v1:{seed}")
    cases: list[OodCase] = []

    def add(category: str, language: str, path: str, context: str) -> None:
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:8]
        cases.append(
            OodCase(
                case_id=f"ood-{category}-{len(cases):04d}-{digest}",
                category=category, language=language, path=path, context=context,
            )
        )

    for text in rng.sample(pool, per_category):
        add("prose", "prose", "notes.txt", text)

    languages = sorted(_UNSEEN)
    for i in range(per_category):
        language = languages[i % len(languages)]
        template, ext = _UNSEEN[language]
        add("unseen-language", language, f"src/{language}_{i}.{ext}",
            _hunks(rng, (template,), rng.randint(2, 4)))

    trained = sorted(_EXT)
    for i in range(per_category):
        language = trained[i % len(trained)]
        lines = _hunks(rng, _FILLER[language], rng.randint(2, 4)).rstrip("\n").split("\n")
        rng.shuffle(lines)
        add("scrambled", language, f"src/scrambled_{i}.{_EXT[language]}", "\n".join(lines) + "\n")
    return cases


def ood_defect_row(case: OodCase, *, config: DataConfig) -> DataRow:
    """``case`` as a ``code.defect_class`` request, built by ``rewrite_defect_class`` itself.

    The row's golds (``clean``, span abstains) exist only because the rewriter needs one to
    render: the gate reads the choice slot's DECODE, never its gold.
    """
    from qd_data.defect_class import DEFECT_FAMILY_ID, DefectRow
    from qd_data.mixture import rewrite_defect_class

    raw = DefectRow(
        example_id=case.case_id, pool_id="ood-suite", repo=f"ood-suite/{case.case_id}",
        path=case.path, symbol="ood", arity=1, language=case.language,
        mutation_class="clean", operator="ood", diff=case.context, diff_span=None,
        span_refusal=None, licence="MIT",
    )
    return rewrite_defect_class(raw, family_id=DEFECT_FAMILY_ID, index=0, config=config)


def ood_gate(
    *, ood_abstained: int, ood_total: int, in_abstained: int, in_total: int
) -> TriState:
    """``ood_abstain`` from the two counts, each read at its conservative Wilson bound."""
    if ood_total == 0:
        return NotRun(reason="the OOD suite produced no scored choice row")
    if in_total == 0:
        return NotRun(reason="no in-distribution choice row was scored to bound")
    ood_lo, _ = wilson_interval(ood_abstained, ood_total)
    _, in_hi = wilson_interval(in_abstained, in_total)
    passed = ood_lo >= OOD_MIN_ABSTAIN and in_hi <= OOD_MAX_IN_DISTRIBUTION_ABSTAIN
    return Ran(
        passed=passed,
        value=ood_abstained / ood_total,
        n=ood_abstained,
        n_total=ood_total,
        detail=(
            f"OOD: abstained on {ood_abstained} of {ood_total} "
            f"({ood_abstained / ood_total:.1%}, Wilson lower {ood_lo:.3f} vs floor "
            f"{OOD_MIN_ABSTAIN}); in-distribution: {in_abstained} of {in_total} val choice "
            f"rows ({in_abstained / in_total:.1%}, Wilson upper {in_hi:.3f} vs cap "
            f"{OOD_MAX_IN_DISTRIBUTION_ABSTAIN}). Rule: either pass answers noul, or the "
            "permuted second pass names another option. The calibrated-margin half is not "
            "applied (no fitted calibration table): it can only add abstentions, so this "
            "OOD rate is a lower bound and this in-distribution rate is too"
        ),
    )
