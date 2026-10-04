"""Score an FT eval row against its char-n-gram linear control: ``paired_margin_vs_linear``.

## Why this is post hoc

Every FT eval row ``tools/real_ft_run.py --score-val`` has written carries
``paired_margin_vs_linear: not_run`` (``ledger/gh200-ft-commitpackft-2026-09-22.jsonl``): the
run has an accuracy and no verdict. The gate needs two things the GPU run does not need to
own -- a linear control fitted on the FT train split, which is CPU work that saturated 64
cores on the GH200 on 2026-09-22, and the model's **per-row** correctness on the val split.
This tool reads the second from the verdicts file the run writes (``--verdicts-out``), fits
the first here, pairs them by row key, and appends the gate as a new ``eval`` row.

## What it refuses, and why each refusal is not a pass

* **No verdicts file.** The historical FT rows cannot be retrofitted: ``_decode`` computed the
  per-row verdicts and ``_record_score`` kept only the counts. Nothing here reconstructs them.
* **A verdicts file that is not this eval row's.** Every line names its ``eval_row_id`` and
  ``seed``, and the per-kind totals and correct counts must equal the eval row's own
  ``val_top1.*`` metrics. A file paired with the wrong row would produce a margin against a
  model nobody trained.
* **Different populations.** :func:`qd_train.eval_harness.paired_margin_by_key` pairs by
  key and refuses anything but identical key sets.
* **An unconverged control, or one with a single class.** Either is recorded ``not_run``:
  a model cannot beat a control that never trained.

## Its input is the model's input

The control reads ``render(...).prompt_for(slot)`` -- every byte the model conditions on --
through :func:`qd_train.baseline.request_texts`. One control is fitted per task
(``family_id/slot_name``), because one task's labels are not candidate answers to another,
and each task's labels are chosen by :func:`qd_train.baseline.control_label_space`: the gold
**value** where every row offers one option set (the letter is a per-example permutation
artefact there), the gold **letter** where the options are the row's own (CommonsenseQA,
MMLU, CLINC's intent sets), whose values are not classes any other row shares. The val rows'
``linear_control_top1`` detail names which. Span slots are not scored: a pointer's answer is
a line pair a bag of n-grams cannot produce, and the row says so.

## The rows

``real_ft_run.ft_split_rows`` rebuilds the train/val split exactly as the run did. It is
imported **by name** and a missing function is a refusal: a second copy of the rebuild here
would be free to drift into scoring a different split while still printing a margin.

## The operator-holdout arm

``--hold-out-operator OP --operator-key KEY`` drops every train row whose
``metadata[KEY] == OP`` before fitting, which is the control for an FT arm trained on the
same reduced set (the rung-0 design in ``AUDIT/operator-holdout-model.md``). Validation is
never filtered. The eval row's recipe must name the same holdout, or the control is not that
arm's opponent. Beside the gate it reports the margins the rung-0 report reads: on the
operator's own val rows, on its **siblings** (same task and gold value, other operator --
the rows that separate "lost the generator" from "the prior moved"), and overall.

## The per-option control (``--option-control``)

A second, separately named control for the tasks whose rows offer their own options (Fable H,
2026-10-01; :mod:`qd_train.option_control`): one binary scorer over each row's shown options,
the best of them its answer. The letter-labelled control above stays the general families'
control of record; this one is **report-only**. ``--option-control`` fits only it and writes it
on its OWN supplement row, joined to the eval row by ``eval_row_id`` like the gate's row, under
keys of its own -- ``linear_option_control.*`` and ``paired_margin_vs_linear_option.*`` -- and
never a gate, so neither row can be read as the other and promotion reads it as nothing.
Adopting it as a gate's comparator is the human's decision under G1. Tasks whose rows share
one option set (``code.defect_class``, ``intent.domain``, ``intent.in_scope``) are recorded as
not scored by it, with the reason: ``control_label_space`` is the test, as it is for the label.

## The engine

Both controls -- the n-gram gate's and the length control -- are featurised and fitted by
``qd-prep linfit`` (``crates/qd-prep``) through :mod:`linear_control_native`, never by
``qd_train.baseline`` in this process: the Python fit of one full-mixture eval row ran 2.3 h+ on
the GH200 without finishing. ``qd_train.baseline`` is unchanged and is the parity oracle; on the
Mac the binary reproduces its sparse operand bit for bit. ``QD_PREP_BIN`` must name the binary
and is checked before the split is rebuilt; there is no Python fallback. The row's recipe names
the engine and the sha256 of the binary that ran.

## The split cache (``--split-cache DIR --split-cache-shards DIR``)

The split rebuild is most of a call's wall clock before the first fit: ~200 s on the Mac on v5's
data, ~325 s on the H100 box, where v5's controls are about 16 calls over one split.
``--split-cache DIR`` uses the cache ``real_ft_run.py --split-cache`` uses
(``tools/split_cache.py``), with its key, checks and bounds; none of them is restated here. An
entry whose key matches and which passes its checks is read back; otherwise the rebuild runs and
is stored.

* **The partial.** The rebuild is one ``functools.partial`` of ``ft_split_rows``. Its keywords
  are what the uncached call passes and what the cache keys.
* **The extra inputs.** What the rebuild reads beyond its arguments is named by
  ``real_ft_run.split_rebuild_inputs``, reached by name as ``ft_split_rows`` is.
* **The shard set.** ``--split-cache-shards DIR`` is the shard set's root: ``real_ft_run``'s
  ``--out``, which the control is not otherwise given. Every cached row is checked against its
  ``data/pool/{train,val}.json``. Before the rebuild, its ``train.json`` must name the eval row's
  ``data_snapshot_hash``.
* **Off by default, and not recipe.** With the flag, the row carries the cache's state as a
  metric: ``split_cache`` on the letter row, ``linear_option_control.split_cache`` on the option
  row, every one of whose keys carries that prefix. Without it, the row is what it was.

RUN (on the machine that has torch; it imports the FT runner for the split)
---
    QD_PREP_BIN=<abs path> /Users/bharath/.venvs/ml/bin/python tools/ft_linear_control.py \\
      --ledger ledger/<file>.jsonl --verdicts <verdicts.jsonl> \\
      --commitpackft data/pool/commitpackft --max-pairs 2000 --rev <sha>
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
import types
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import linear_control_native as native  # noqa: E402
import split_cache  # noqa: E402
from repo_git import require_full_sha  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_train.baseline import (  # noqa: E402
    CharNGramHasher,
    ContextLengthFeatures,
    Featurizer,
    LabelSpace,
    LinearBaseline,
    RequestDoc,
    control_label,
    control_label_space,
    fit_budget_refusal,
    request_texts,
)
from qd_train.control_cache import control_key, load_control, store_control  # noqa: E402
from qd_train.eval_harness import paired_margin_by_key  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    RunRecorder,
)
from qd_train.option_control import OptionScorer, read_option_rows  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState, aggregate  # noqa: E402

#: The gate this tool exists to evaluate, spelled once.
GATE: Final[str] = "paired_margin_vs_linear"

#: The per-option control's metric prefixes. Every key its row carries starts with one of these
#: two, and neither is a gate or control name the ledger joins (``REQUIRED_GATES``/``_CONTROLS``).
OPTION_ARM: Final[str] = "linear_option_control"
OPTION_MARGIN: Final[str] = "paired_margin_vs_linear_option"
#: What the per-option row is, stated on it.
OPTION_REPORT_ONLY: Final[str] = (
    "report-only (Fable H, 2026-10-01): the per-option linear control for tasks whose rows offer "
    "their own options, beside the letter control of record; adopting it as a gate's comparator "
    "is the human's decision under G1"
)

#: The control's iteration budget. The same number as ``rung0_linear_control`` and
#: ``rung0_real_run.LINEAR_CONTROL_MAX_ITER``; restated rather than imported because both of
#: those import torch at module scope, and ``test_ft_linear_control`` pins the equality where
#: torch exists. Raising it is allowed; lowering it to fit a cap weakens the opponent.
#: 6,000 until 2026-10-02; 8,000 since, with ``qd_train.baseline.step_size`` halving the step
#: every 500 iterations past 6,000, because F seed 0's intent.domain control sat in a limit
#: cycle at the constant step (c89b89a1) and converges at 6,093 under the schedule
#: (HANDOFF/prep-containment-2026-10-02.md). Every fit that converged within 6,000 is
#: unchanged bit for bit.
DEFAULT_MAX_ITER: Final[int] = 8_000

#: Letter kinds the eval row reports as ``val_top1.<kind>``; the verdicts must agree with them.
LETTER_KIND_NAMES: Final[tuple[str, ...]] = ("choice", "score")

#: ``--split-cache``'s state, in the form ``real_ft_run.py`` writes: ``hit``, ``miss`` or
#: ``corrupt``, with the key. A metric, never a recipe key, so a cached and an uncached control of
#: one eval row keep one recipe_hash. The letter row carries it under this name. The option row
#: carries it under ``OPTION_ARM``'s prefix, as it does every key. Without the flag, neither row
#: carries it.
SPLIT_CACHE_METRIC: Final[str] = "split_cache"

Key = tuple[str, str]


class Refused(SystemExit):
    """A precondition failed. Exits non-zero with the reason; no row is written."""


# --- verdicts --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdicts:
    """One eval row's per-row model correctness, keyed the way the control is keyed."""

    eval_row_id: str
    seed: int
    #: ``(row_id, slot_name)`` when the file carries slot names, else ``(row_id, kind)``.
    correct: dict[Key, bool]
    kind_of: dict[Key, str]
    by_slot_name: bool
    span_rows: int
    #: Span rows' correctness and whether their gold abstains, by the same key. ``None``
    #: when any span line lacks ``expected_abstain`` (files written before it existed).
    span_correct: dict[Key, bool] | None = None
    span_expected_abstain: dict[Key, bool] | None = None


def load_verdicts(path: Path) -> Verdicts:
    """Parse the ``--verdicts-out`` JSONL, refusing anything that could mispair.

    Keyed by ``(row_id, slot_name)`` when every line has a slot name, else by
    ``(row_id, kind)`` -- and then a row with two slots of one kind cannot be told apart, so
    a duplicate key is a refusal rather than a last-one-wins overwrite.
    """
    if not path.is_file():
        raise Refused(
            f"no verdicts file at {path}. It is written by tools/real_ft_run.py "
            "--score-val --verdicts-out; a run that did not write one has no per-row "
            "correctness to pair, and nothing here reconstructs it"
        )
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise Refused(f"{path} holds no verdicts")
    #: (eval_row_id, seed, row_id, kind, slot_name or None, correct), each field checked.
    records: list[tuple[str, int, str, str, str | None, bool]] = []
    #: Per record, the span line's ``expected_abstain``, or None where absent.
    abstains: list[bool | None] = []
    for n, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Refused(f"{path}:{n}: not JSON ({exc})") from exc
        if not isinstance(rec, dict):
            raise Refused(f"{path}:{n}: a verdict must be an object")
        for field, typ in (("eval_row_id", str), ("row_id", str), ("kind", str)):
            if not isinstance(rec.get(field), typ):
                raise Refused(f"{path}:{n}: {field!r} missing or not a {typ.__name__}")
        if not isinstance(rec.get("correct"), bool):
            raise Refused(f"{path}:{n}: 'correct' must be a JSON boolean")
        seed = rec.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise Refused(f"{path}:{n}: 'seed' must be an integer")
        slot = rec.get("slot_name")
        if slot is not None and not isinstance(slot, str):
            raise Refused(f"{path}:{n}: 'slot_name' must be a string when present")
        expected = rec.get("expected_abstain")
        if expected is not None and not isinstance(expected, bool):
            raise Refused(f"{path}:{n}: 'expected_abstain' must be a JSON boolean when present")
        records.append(
            (rec["eval_row_id"], seed, rec["row_id"], rec["kind"], slot, rec["correct"])
        )
        abstains.append(expected)

    eval_ids = {r[0] for r in records}
    seeds = {r[1] for r in records}
    if len(eval_ids) != 1 or len(seeds) != 1:
        raise Refused(
            f"{path} mixes eval rows {sorted(eval_ids)} / seeds {sorted(seeds)}; one file is "
            "one eval row's verdicts"
        )
    with_slot = [r[4] is not None for r in records]
    if any(with_slot) and not all(with_slot):
        raise Refused(f"{path}: some verdicts carry slot_name and some do not")
    by_slot_name = all(with_slot)

    correct: dict[Key, bool] = {}
    kind_of: dict[Key, str] = {}
    span_rows = 0
    span_correct: dict[Key, bool] = {}
    span_expected: dict[Key, bool] = {}
    span_complete = True
    for (_, _, row_id, kind, slot, hit), expected in zip(records, abstains, strict=True):
        if kind == "span":
            span_rows += 1
            span_key: Key = (row_id, slot if (by_slot_name and slot is not None) else kind)
            if span_key in span_correct:
                raise Refused(f"{path}: two span verdicts share key {span_key}")
            span_correct[span_key] = hit
            if expected is None:
                span_complete = False
            else:
                span_expected[span_key] = expected
            continue
        if kind not in LETTER_KIND_NAMES:
            raise Refused(f"{path}: unknown kind {kind!r} on row {row_id}")
        key: Key = (row_id, slot if (by_slot_name and slot is not None) else kind)
        if key in correct:
            raise Refused(
                f"{path}: two verdicts share key {key}. Without slot names a row with two "
                "slots of one kind cannot be paired; the run must write slot_name"
            )
        correct[key] = hit
        kind_of[key] = kind
    return Verdicts(
        eval_row_id=eval_ids.pop(), seed=seeds.pop(), correct=correct, kind_of=kind_of,
        by_slot_name=by_slot_name, span_rows=span_rows,
        span_correct=span_correct if span_complete else None,
        span_expected_abstain=span_expected if span_complete else None,
    )


def check_against_eval_row(verdicts: Verdicts, row: LedgerRow) -> None:
    """The file must be this row's: same id, same seed, same per-kind totals and hits."""
    if verdicts.eval_row_id != row.row_id:
        raise Refused(
            f"the verdicts are eval row {verdicts.eval_row_id}'s and --eval-row is {row.row_id}"
        )
    if verdicts.seed != row.protocol.seed:
        raise Refused(f"verdicts seed {verdicts.seed} != eval row seed {row.protocol.seed}")
    if row.run_kind != "eval":
        raise Refused(f"row {row.row_id} is run_kind {row.run_kind!r}, not 'eval'")
    for kind in LETTER_KIND_NAMES:
        state = row.metrics.get(f"val_top1.{kind}")
        mine = [k for k, v in verdicts.kind_of.items() if v == kind]
        hits = sum(verdicts.correct[k] for k in mine)
        if not isinstance(state, Ran):
            if mine:
                raise Refused(
                    f"the verdicts hold {len(mine)} {kind} row(s) and the eval row scored none"
                )
            continue
        if (state.n_total, state.n) != (len(mine), hits):
            raise Refused(
                f"val_top1.{kind} on the eval row is {state.n}/{state.n_total} and the "
                f"verdicts say {hits}/{len(mine)}: this file is not that run's decode"
            )


def find_row(rows: Sequence[LedgerRow], row_id: str) -> LedgerRow:
    """The one row whose id starts with ``row_id``, or a refusal."""
    hits = [r for r in rows if r.row_id.startswith(row_id)]
    if len(hits) != 1:
        raise Refused(f"--eval-row {row_id!r} matches {len(hits)} row(s); need exactly one")
    return hits[0]


# --- the control -----------------------------------------------------------------------


def doc_key(doc: RequestDoc, *, by_slot_name: bool) -> Key:
    return (doc.row_id, doc.slot_name if by_slot_name else doc.kind)


@dataclass(frozen=True)
class HoldOut:
    """Which training rows the arm and its control never saw."""

    key: str
    operator: str


def apply_holdout(train: list[RequestDoc], hold: HoldOut | None) -> tuple[list[RequestDoc], int]:
    """``(kept, removed)``. Refuses a key no row carries or an operator no row has.

    Both are silent no-ops otherwise: the control would be fitted on the full set, beat the
    arm on data the arm never saw, and the report would call it a holdout.
    """
    if hold is None:
        return train, 0
    if not any(hold.key in d.metadata for d in train):
        raise Refused(
            f"no training row carries metadata {hold.key!r}, so --hold-out-operator "
            f"{hold.operator!r} would remove nothing. The family that records operators "
            "(A3's code.defect_class) must be in this split"
        )
    kept = [d for d in train if d.metadata.get(hold.key) != hold.operator]
    removed = len(train) - len(kept)
    if removed == 0:
        raise Refused(f"no training row has {hold.key}={hold.operator!r}")
    return kept, removed


@dataclass(frozen=True)
class ControlArm:
    """One control: what it is called in the row, how it featurizes, what it reads."""

    name: str
    make_features: Callable[[], Featurizer]
    text_of: Callable[[RequestDoc], str]
    #: Only the n-gram arm is cached: its fit is the expensive one, and the cache key's
    #: ``hasher_params`` names n-gram parameters, which the length arm does not have.
    cacheable: bool


#: The gate's opponent: every byte of the prompt, as hashed char n-grams.
NGRAM_ARM: Final[ControlArm] = ControlArm(
    name="linear_control", make_features=CharNGramHasher, text_of=lambda d: d.text,
    cacheable=True,
)
#: GAP-A3-CLEAN-DIFFS-ARE-LONGER-THAN-MUTATION-DIFFS: the context's size and nothing else.
#: Reported beside the gate, never folded into it -- the gate is the n-gram margin.
LENGTH_ARM: Final[ControlArm] = ControlArm(
    name="length_control", make_features=ContextLengthFeatures, text_of=lambda d: d.context,
    cacheable=False,
)
#: The metric name for the model's paired margin over the length-only control.
LENGTH_MARGIN: Final[str] = "paired_margin_vs_length_control"

#: The span rows' opponent, since the n-gram control cannot produce a line pair: a head
#: that abstains on every row, correct exactly where the gold abstains. A constant, not a
#: linear control, and named as one -- it is a metric beside the gate, never inside it.
ABSTAIN_MARGIN: Final[str] = "paired_margin_vs_abstain_constant"


def abstain_constant_margin(verdicts: Verdicts, *, seed: int) -> TriState:
    """The model's span rows against always abstaining, paired by row."""
    if verdicts.span_rows == 0:
        return NotRun(reason="the verdicts hold no span rows")
    if verdicts.span_correct is None or verdicts.span_expected_abstain is None:
        return NotRun(
            reason=(
                f"{verdicts.span_rows} span verdict(s) without expected_abstain: the file "
                "predates it, so what an always-abstaining head scores is not known"
            )
        )
    return paired_margin_by_key(
        verdicts.span_correct, verdicts.span_expected_abstain, seed=seed
    )


@dataclass(frozen=True)
class TaskControl:
    task: str
    correct: dict[Key, bool]
    convergence: TriState
    accuracy: TriState
    fitted_s: float
    cached: bool


def fit_task(
    task: str,
    train: list[RequestDoc],
    val: list[RequestDoc],
    *,
    by_slot_name: bool,
    seed: int,
    max_iter: int,
    dense_budget_bytes: int | None,
    max_fit_minutes: float | None,
    cache_dir: Path | None,
    engine: Path,
    arm: ControlArm = NGRAM_ARM,
) -> TaskControl:
    """One task's control, scored on that task's val rows. Never raises for a weak fit:
    an unconverged or single-class control comes back ``not_run`` with its reason.

    The fit runs in ``engine`` (``qd-prep linfit``, :mod:`linear_control_native`); the
    ``LinearBaseline`` built here supplies every choice the fit makes and speaks for the
    result through ``convergence()``, but is never fitted in Python. ``dense_budget_bytes``
    now only prices the ``max_fit_minutes`` projection, which is calibrated on the Python
    engine and so over-states the native one -- the safe direction for a refusal."""
    val_docs = [arm.text_of(d) for d in val]
    keys = [doc_key(d, by_slot_name=by_slot_name) for d in val]
    try:
        space = control_label_space(train, val)
    except ValueError as exc:
        reason = f"task {task}: no label space every row shares: {exc}"
        return TaskControl(task, {}, NotRun(reason=reason), NotRun(reason=reason), 0.0, False)
    train_labels = [control_label(d, space) for d in train]
    val_golds = [control_label(d, space) for d in val]
    classes = sorted(set(train_labels))
    if len(classes) < 2:
        reason = (
            f"task {task}: the training split holds {len(train)} row(s) with class(es) "
            f"{classes}; a control cannot be fitted on fewer than two classes"
        )
        return TaskControl(task, {}, NotRun(reason=reason), NotRun(reason=reason), 0.0, False)
    train_docs = [arm.text_of(d) for d in train]
    empty = sum(1 for t in (*train_docs, *val_docs) if not t)
    if empty:
        reason = (
            f"task {task}: {empty} doc(s) carry no context for the {arm.name}; scoring them "
            "as zero-length would make a constant, easily beaten control"
        )
        return TaskControl(task, {}, NotRun(reason=reason), NotRun(reason=reason), 0.0, False)
    features = arm.make_features()
    model = LinearBaseline(
        hasher=features, seed=seed, max_iter=max_iter, dense_budget_bytes=dense_budget_bytes
    )

    if cache_dir is not None and arm.cacheable and isinstance(features, CharNGramHasher):
        cache_key = control_key(
            train_docs=train_docs, train_labels=train_labels, val_docs=val_docs, seed=seed,
            max_iter=max_iter,
            hasher_params=(features.n_min, features.n_max, features.dim),
            l2_grid=model.l2_grid, tol=model.tol, lr=model.lr,
        )
    else:
        cache_key = None
    if cache_dir is not None and cache_key is not None:
        hit = load_control(cache_dir, cache_key, expected_n=len(val_docs))
        if hit is not None:
            correct = dict(zip(keys, (bool(x) for x in hit.correct), strict=True))
            return TaskControl(
                task, correct,
                Ran(passed=True, value=hit.final_grad_norm,
                    detail=f"cached: converged in {hit.iterations} iterations at l2={hit.l2}"),
                _accuracy(task, correct, space), hit.fitted_s, True,
            )

    refusal = fit_budget_refusal(
        model.projected_fit_seconds(train_docs, n_classes=len(classes)), max_fit_minutes
    )
    if refusal is not None:
        raise Refused(f"task {task}: {refusal}")
    started = time.monotonic()
    result = native.fit(
        engine, model, train_docs, train_labels, val_docs,
        timeout_s=(
            native.DEFAULT_FIT_TIMEOUT_S if max_fit_minutes is None else max_fit_minutes * 60
        ),
    )
    fitted_s = time.monotonic() - started
    print(f"task {task} ({arm.name}): {result.summary}", file=sys.stderr, flush=True)
    model.fit_ = result.fit
    convergence = model.convergence()
    if not (isinstance(convergence, Ran) and convergence.passed):
        reason = convergence.reason if isinstance(convergence, NotRun) else convergence.detail
        return TaskControl(task, {}, convergence, NotRun(reason=reason), fitted_s, False)
    predicted = result.predictions
    hits = [p == gold for p, gold in zip(predicted, val_golds, strict=True)]
    fit = model.fit_
    if fit is None:  # pragma: no cover - convergence() refused an unfitted model above
        raise RuntimeError("a converged control has no fit")
    if cache_dir is not None and cache_key is not None:
        store_control(
            cache_dir, cache_key, hits, fitted_s=fitted_s, n_train=len(train_docs),
            l2=fit.l2, iterations=fit.iterations, final_grad_norm=fit.final_grad_norm,
        )
    correct = dict(zip(keys, hits, strict=True))
    return TaskControl(
        task, correct, convergence, _accuracy(task, correct, space, arm.name), fitted_s, False
    )


def _accuracy(
    task: str, correct: Mapping[Key, bool], space: LabelSpace, arm: str = "linear_control"
) -> TriState:
    if not correct:
        return NotRun(reason=f"task {task} has no val rows")
    hits = sum(correct.values())
    return Ran(
        passed=True, value=hits / len(correct), n=hits, n_total=len(correct),
        detail=f"{arm} top-1 on task {task}'s val rows, labelled by {space}",
    )


@dataclass(frozen=True)
class ControlScore:
    gate: TriState
    metrics: dict[str, TriState]
    train_rows: int
    removed_by_holdout: int
    fit_seconds: float


def score_against_control(
    train: list[RequestDoc],
    val: list[RequestDoc],
    verdicts: Verdicts,
    *,
    seed: int,
    max_iter: int = DEFAULT_MAX_ITER,
    dense_budget_bytes: int | None = None,
    max_fit_minutes: float | None = None,
    cache_dir: Path | None = None,
    hold: HoldOut | None = None,
    engine: Path | None = None,
) -> ControlScore:
    """The gate and its supporting metrics. Pure apart from the optional cache.

    The gate is the paired margin over every letter row pooled; it is ``not_run`` if any
    task's control did not run, because a pooled margin over the tasks that happened to fit
    would be a capped sample reported as complete coverage. Beside it -- metrics, never the
    gate -- each family's margin, ``paired_margin_vs_linear.{kind}.{family}`` and
    ``paired_margin_vs_length_control.{kind}.{family}``: that family's rows against that
    family's own control, ``not_run`` only for a family whose control did not run, so one
    family's unconverged fit cannot hide another's measured margin.

    ``engine`` is the ``qd-prep`` binary that fits every control; ``None`` reads it from
    ``QD_PREP_BIN`` and refuses, before anything is fitted, when that is unset.
    """
    engine = native.prep_binary() if engine is None else engine
    train, removed = apply_holdout(train, hold)
    by_task_train: dict[str, list[RequestDoc]] = {}
    for d in train:
        by_task_train.setdefault(d.task, []).append(d)
    by_task_val: dict[str, list[RequestDoc]] = {}
    for d in val:
        by_task_val.setdefault(d.task, []).append(d)

    metrics: dict[str, TriState] = {}
    control: dict[Key, bool] = {}
    fit_seconds = 0.0
    task_states: dict[str, TriState] = {}
    for task in sorted(by_task_val):
        tc = fit_task(
            task, by_task_train.get(task, []), by_task_val[task],
            by_slot_name=verdicts.by_slot_name, seed=seed, max_iter=max_iter,
            dense_budget_bytes=dense_budget_bytes, max_fit_minutes=max_fit_minutes,
            cache_dir=cache_dir, engine=engine,
        )
        fit_seconds += tc.fitted_s
        metrics[f"linear_control_convergence.{task}"] = tc.convergence
        metrics[f"linear_control_top1.{task}"] = tc.accuracy
        task_states[task] = tc.convergence
        control.update(tc.correct)

    # The length-only control, per task, beside the n-gram one and never inside the gate.
    length_control: dict[Key, bool] = {}
    length_states: dict[str, TriState] = {}
    for task in sorted(by_task_val):
        lc = fit_task(
            task, by_task_train.get(task, []), by_task_val[task],
            by_slot_name=verdicts.by_slot_name, seed=seed, max_iter=max_iter,
            dense_budget_bytes=dense_budget_bytes, max_fit_minutes=max_fit_minutes,
            cache_dir=None, engine=engine, arm=LENGTH_ARM,
        )
        fit_seconds += lc.fitted_s
        metrics[f"length_control_convergence.{task}"] = lc.convergence
        metrics[f"length_control_top1.{task}"] = lc.accuracy
        length_states[task] = lc.convergence
        length_control.update(lc.correct)
    length_ran = aggregate(length_states, name="length_control_convergence")
    if not (isinstance(length_ran, Ran) and length_ran.passed):
        why = length_ran.reason if isinstance(length_ran, NotRun) else length_ran.detail
        metrics[LENGTH_MARGIN] = NotRun(reason=f"a task's length control did not run: {why}")
    else:
        metrics[LENGTH_MARGIN] = paired_margin_by_key(verdicts.correct, length_control, seed=seed)

    ran = aggregate(task_states, name="linear_control_convergence")
    if not (isinstance(ran, Ran) and ran.passed):
        reason = ran.reason if isinstance(ran, NotRun) else ran.detail
        gate: TriState = NotRun(reason=f"a task's control did not run: {reason}")
    else:
        gate = paired_margin_by_key(verdicts.correct, control, seed=seed)

    for kind in LETTER_KIND_NAMES:
        keys = [k for k, v in verdicts.kind_of.items() if v == kind]
        if keys and isinstance(gate, Ran):
            metrics[f"{GATE}.{kind}"] = _subset_margin(verdicts.correct, control, keys, seed)
    # Per family, beside the pooled margins and never instead of them: a pooled not_run must
    # not hide a family whose control ran (J4's rows d597ee7d/7921ae18 carried no defect_class
    # margin because intent.domain's control did not converge).
    metrics.update(_family_margins(GATE, val, verdicts, control, task_states, seed=seed))
    metrics.update(
        _family_margins(LENGTH_MARGIN, val, verdicts, length_control, length_states, seed=seed)
    )
    metrics[f"{GATE}.span"] = NotRun(
        reason=(
            f"{verdicts.span_rows} span row(s) not scored: a pointer's answer is a line "
            "pair the char-n-gram control cannot produce"
        )
    )
    metrics[f"{ABSTAIN_MARGIN}.span"] = abstain_constant_margin(verdicts, seed=seed)
    if hold is not None:
        metrics.update(_holdout_metrics(val, verdicts, control, hold, seed=seed, gate=gate))
    return ControlScore(gate, metrics, len(train), removed, fit_seconds)


def task_family(task: str) -> str:
    """The family of a ``family_id/slot_name`` task (``RequestDoc.task``)."""
    return task.rsplit("/", 1)[0]


def _population_refusal(val: Sequence[RequestDoc], verdicts: Verdicts) -> str | None:
    """Why a subset of the model's rows cannot be attributed, or ``None``.

    A margin over a subset -- one family's rows, the per-row-option tasks' rows -- finds that
    subset among the rebuilt split's docs. A model row the split does not hold belongs to no
    subset and would drop out of every one without a word, so unless the two are one
    population no subset margin is computed.
    """
    split = {doc_key(d, by_slot_name=verdicts.by_slot_name) for d in val}
    if split == set(verdicts.correct):
        return None
    return (
        f"the rebuilt split's {len(split)} val row(s) and the verdicts' {len(verdicts.correct)} "
        f"letter row(s) are different populations ({len(set(verdicts.correct) - split)} only "
        f"in the verdicts, {len(split - set(verdicts.correct))} only in the split), so which "
        "of the model's rows belong to which subset is not known"
    )


def _family_margins(
    prefix: str, val: Sequence[RequestDoc], verdicts: Verdicts, control: Mapping[Key, bool],
    states: Mapping[str, TriState], *, seed: int,
) -> dict[str, TriState]:
    """``{prefix}.{kind}.{family}``: each family's rows of one letter kind against that family's
    own control (its tasks', fitted as ``fit_task`` fits them), ``not_run`` with the reason
    when any of the family's controls did not run."""
    by_family: dict[str, list[RequestDoc]] = {}
    for d in val:
        by_family.setdefault(task_family(d.task), []).append(d)
    refusal = _population_refusal(val, verdicts)
    out: dict[str, TriState] = {}
    for family, docs in sorted(by_family.items()):
        tasks = sorted({d.task for d in docs})
        ran = aggregate({t: states[t] for t in tasks}, name=f"family {family}'s controls")
        for kind in LETTER_KIND_NAMES:
            keys = [doc_key(d, by_slot_name=verdicts.by_slot_name) for d in docs
                    if d.kind == kind]
            if not keys:
                continue
            name = f"{prefix}.{kind}.{family}"
            if refusal is not None:
                out[name] = NotRun(reason=refusal)
            elif not (isinstance(ran, Ran) and ran.passed):
                why = ran.reason if isinstance(ran, NotRun) else ran.detail
                out[name] = NotRun(reason=f"family {family}'s control did not run: {why}")
            else:
                out[name] = _subset_margin(verdicts.correct, control, keys, seed)
    return out


def _subset_margin(
    model: Mapping[Key, bool], control: Mapping[Key, bool], keys: Sequence[Key], seed: int
) -> TriState:
    if not keys:
        return NotRun(reason="the subset holds no rows")
    missing = [k for k in keys if k not in model or k not in control]
    if missing:
        return NotRun(reason=f"{len(missing)} subset row(s) unscored by one arm")
    return paired_margin_by_key(
        {k: model[k] for k in keys}, {k: control[k] for k in keys}, seed=seed
    )


def _holdout_metrics(
    val: list[RequestDoc], verdicts: Verdicts, control: Mapping[Key, bool], hold: HoldOut,
    *, seed: int, gate: TriState,
) -> dict[str, TriState]:
    """Own rows, siblings and their counts: what ``operator_holdout_report`` reads."""
    by = verdicts.by_slot_name
    own = [d for d in val if d.metadata.get(hold.key) == hold.operator]
    classes = {(d.task, d.value) for d in own}
    siblings = [
        d for d in val
        if d.metadata.get(hold.key) not in (None, hold.operator) and (d.task, d.value) in classes
    ]
    out: dict[str, TriState] = {}
    for name, docs in (("own", own), ("siblings", siblings)):
        keys = [doc_key(d, by_slot_name=by) for d in docs]
        if not isinstance(gate, Ran):
            out[f"{GATE}.holdout.{name}"] = NotRun(reason="the pooled gate did not run")
        else:
            out[f"{GATE}.holdout.{name}"] = _subset_margin(verdicts.correct, control, keys, seed)
        for arm, table in (("model", verdicts.correct), ("control", control)):
            scored = [table[k] for k in keys if k in table]
            out[f"holdout.{name}.{arm}_top1"] = (
                Ran(passed=True, value=sum(scored) / len(scored), n=sum(scored),
                    n_total=len(scored), detail=f"{hold.key}={hold.operator!r} {name} rows")
                if scored else NotRun(reason=f"no {name} rows were scored by the {arm}")
            )
    return out


# --- the per-option control ---------------------------------------------------------------


@dataclass(frozen=True)
class OptionTaskControl:
    task: str
    #: Whether the task's rows offer their own options, the only tasks this control scores.
    applies: bool
    correct: dict[Key, bool]
    convergence: TriState
    accuracy: TriState
    chance: TriState
    fitted_s: float


def fit_option_task(
    task: str,
    train: list[RequestDoc],
    val: list[RequestDoc],
    *,
    by_slot_name: bool,
    seed: int,
    max_iter: int,
    max_fit_minutes: float | None,
    engine: Path,
) -> OptionTaskControl:
    """One task's per-option control, scored on that task's val rows. Never raises for a
    weak fit: a task it does not apply to, cannot read or cannot fit comes back ``not_run``
    with the reason, and a val row whose shown options cannot be read is left unscored (and
    named), never scored wrong."""

    def not_run(reason: str, *, applies: bool = True) -> OptionTaskControl:
        state = NotRun(reason=reason)
        return OptionTaskControl(task, applies, {}, state, state, state, 0.0)

    try:
        space = control_label_space(train, val)
    except ValueError as exc:
        return not_run(f"task {task}: whether its rows offer their own options is undecided: {exc}")
    if space != "letter":
        return not_run(
            f"task {task}: every row offers one option set, so its control is the {space}-"
            f"labelled linear control; the {OPTION_ARM} scores only tasks whose rows offer "
            "their own options",
            applies=False,
        )
    train_rows, train_bad = read_option_rows(train)
    val_rows, val_bad = read_option_rows(val)
    if not val_rows:
        first = f"; first: {val_bad[0][1]}" if val_bad else ""
        return not_run(
            f"task {task}: none of its {len(val)} val row(s) shows options that can be read"
            f"{first}"
        )
    if len(train_rows) < 4:
        first = f"; first unreadable: {train_bad[0][1]}" if train_bad else ""
        return not_run(
            f"task {task}: {len(train_rows)} readable training row(s) of {len(train)}; the "
            f"carve needs at least 4{first}"
        )
    scorer = OptionScorer(seed=seed, max_iter=max_iter)
    refusal = fit_budget_refusal(scorer.projected_fit_seconds(train_rows), max_fit_minutes)
    if refusal is not None:
        raise Refused(f"task {task}: {refusal}")
    started = time.monotonic()
    try:
        result = native.fit_options(
            engine, scorer, train_rows, val_rows,
            timeout_s=(
                native.DEFAULT_FIT_TIMEOUT_S if max_fit_minutes is None else max_fit_minutes * 60
            ),
        )
    except ValueError as exc:
        return not_run(f"task {task}: {exc}")
    fitted_s = time.monotonic() - started
    print(f"task {task} ({OPTION_ARM}): {result.summary}", file=sys.stderr, flush=True)
    scorer.fit_ = result.fit
    convergence = scorer.convergence()
    if not (isinstance(convergence, Ran) and convergence.passed):
        reason = convergence.reason if isinstance(convergence, NotRun) else convergence.detail
        state = NotRun(reason=reason)
        return OptionTaskControl(task, True, {}, convergence, state, state, fitted_s)
    hits = [p == r.gold for p, r in zip(result.predictions, val_rows, strict=True)]
    correct = {
        doc_key(r.doc, by_slot_name=by_slot_name): hit
        for r, hit in zip(val_rows, hits, strict=True)
    }
    left_out = "".join([
        f"; {len(val_bad)} val row(s) whose shown options could not be read are NOT scored "
        f"(first: {val_bad[0][1]})" if val_bad else "",
        f"; {len(train_bad)} unreadable training row(s) were not fitted" if train_bad else "",
    ])
    accuracy = Ran(
        passed=True, value=sum(hits) / len(hits), n=sum(hits), n_total=len(hits),
        detail=(
            f"{OPTION_ARM} top-1 on task {task}'s val rows: the best of each row's shown "
            f"options, noul included, by one binary scorer over n-grams of (question, option) "
            f"and of the option ({len(train_rows)} training rows){left_out}"
        ),
    )
    chance = Ran(
        passed=True, value=sum(1 / len(r.options) for r in val_rows) / len(val_rows),
        detail=(
            f"a uniform choice among each of the {len(val_rows)} scored val rows' shown "
            "options, noul included, averaged over the rows"
        ),
    )
    return OptionTaskControl(task, True, correct, convergence, accuracy, chance, fitted_s)


@dataclass(frozen=True)
class OptionControlScore:
    metrics: dict[str, TriState]
    train_rows: int
    fit_seconds: float


def score_against_option_control(
    train: list[RequestDoc],
    val: list[RequestDoc],
    verdicts: Verdicts,
    *,
    seed: int,
    max_iter: int = DEFAULT_MAX_ITER,
    max_fit_minutes: float | None = None,
    engine: Path | None = None,
) -> OptionControlScore:
    """The per-option control's metrics: per task, and the paired margin pooled over the
    per-row-option tasks' choice rows. No gate -- the row is report-only.

    The pooled margin is ``not_run`` when any per-row-option task's control did not run, as the
    gate is: a margin over the tasks that happened to fit is a capped sample. It is paired by
    ``paired_margin_by_key``, which refuses when the two arms scored different rows, so a val
    row the control could not read leaves the margin ``not_run`` rather than smaller.
    """
    engine = native.prep_binary() if engine is None else engine
    by = verdicts.by_slot_name
    by_task_train: dict[str, list[RequestDoc]] = {}
    for d in train:
        by_task_train.setdefault(d.task, []).append(d)
    by_task_val: dict[str, list[RequestDoc]] = {}
    for d in val:
        by_task_val.setdefault(d.task, []).append(d)

    metrics: dict[str, TriState] = {}
    control: dict[Key, bool] = {}
    population: list[Key] = []
    states: dict[str, TriState] = {}
    fit_seconds, train_rows = 0.0, 0
    refusal = _population_refusal(val, verdicts)
    for task in sorted(by_task_val):
        task_train = by_task_train.get(task, [])
        oc = fit_option_task(
            task, task_train, by_task_val[task], by_slot_name=by, seed=seed,
            max_iter=max_iter, max_fit_minutes=max_fit_minutes, engine=engine,
        )
        metrics[f"{OPTION_ARM}.top1.{task}"] = oc.accuracy
        if not oc.applies:
            continue
        fit_seconds += oc.fitted_s
        train_rows += len(task_train)
        metrics[f"{OPTION_ARM}.convergence.{task}"] = oc.convergence
        metrics[f"{OPTION_ARM}.chance.{task}"] = oc.chance
        states[task] = oc.convergence
        control.update(oc.correct)
        keys = [doc_key(d, by_slot_name=by) for d in by_task_val[task] if d.kind == "choice"]
        population.extend(keys)
        if refusal is not None:
            metrics[f"{OPTION_MARGIN}.choice.{task}"] = NotRun(reason=refusal)
        elif isinstance(oc.convergence, Ran) and oc.convergence.passed:
            metrics[f"{OPTION_MARGIN}.choice.{task}"] = _subset_margin(
                verdicts.correct, oc.correct, keys, seed
            )
        else:
            metrics[f"{OPTION_MARGIN}.choice.{task}"] = NotRun(
                reason=f"task {task}'s {OPTION_ARM} did not run"
            )

    pooled = f"{OPTION_MARGIN}.choice"
    ran = aggregate(states, name=f"{OPTION_ARM}.convergence")
    if not states:
        metrics[pooled] = NotRun(
            reason=f"no task in this split offers per-row options; the {OPTION_ARM} scored nothing"
        )
    elif not (isinstance(ran, Ran) and ran.passed):
        why = ran.reason if isinstance(ran, NotRun) else ran.detail
        metrics[pooled] = NotRun(reason=f"a per-row-option task's control did not run: {why}")
    elif refusal is not None:
        metrics[pooled] = NotRun(reason=refusal)
    else:
        margin = _subset_margin(verdicts.correct, control, population, seed)
        if isinstance(margin, Ran):
            margin = Ran(
                passed=margin.passed, value=margin.value, n=margin.n, n_total=margin.n_total,
                detail=(
                    f"{margin.detail}; over the {len(population)} choice row(s) of the "
                    f"per-row-option tasks {sorted(states)}, and none of the split's other "
                    f"{len(val) - len(population)} letter row(s)"
                ),
            )
        metrics[pooled] = margin
    return OptionControlScore(metrics, train_rows, fit_seconds)


# --- main ------------------------------------------------------------------------------


def _note(args: argparse.Namespace) -> str:
    """``--note``, as the row's notes end with it: a statement for the reader, never a key."""
    return f". {args.note}" if args.note else ""


def _runner() -> types.ModuleType:
    """``tools/real_ft_run.py``, the owner of the split rebuild, imported by name, or a refusal."""
    try:
        import real_ft_run
    except SystemExit as exc:  # real_ft_run refuses to import without torch
        raise Refused(f"tools/real_ft_run.py could not be imported: {exc}") from exc
    return real_ft_run


def split_rows_function() -> Callable[..., tuple[list[DataRow], list[DataRow]]]:
    """``real_ft_run.ft_split_rows``, by name, or a refusal. Never a local copy."""
    fn = getattr(_runner(), "ft_split_rows", None)
    if not callable(fn):
        raise Refused(
            "tools/real_ft_run.py has no ft_split_rows(). This tool rebuilds the split by "
            "calling the run's own function and refuses to carry a copy of it, because a "
            "copy that drifted would score the control on a different split and still print "
            "a margin. It is owned by the lane that owns real_ft_run.py."
        )
    return fn


def split_rebuild_inputs_function() -> Callable[..., tuple[dict[str, Path], dict[str, object]]]:
    """``real_ft_run.split_rebuild_inputs``, by name, or a refusal. Never a local copy.

    It names what the rebuild reads beyond its arguments, which ``--split-cache`` keys beside
    them. A list kept here would miss the next input the runner's rebuild learns to read, and a
    cache keyed without that input would hand back rows from before it changed."""
    fn = getattr(_runner(), "split_rebuild_inputs", None)
    if not callable(fn):
        raise Refused(
            "tools/real_ft_run.py has no split_rebuild_inputs(). --split-cache keys the split "
            "rebuild by every input it reads, and only the runner that owns the rebuild can "
            "name them; this tool refuses to carry its own list. Run without --split-cache."
        )
    return fn


def check_shard_set(shards: Path, row: LedgerRow) -> None:
    """Refuse a ``--split-cache-shards`` that is not the eval row's shard set, before the rebuild.

    ``split_cache`` checks every cached row against the shard set's manifests, and the rebuilt
    rows too before it stores them. Give it another build's manifests and every call is a miss
    whose rows fail that check and are never stored: the whole rebuild on every call, and only
    a metric would say so. The train manifest's ``data_snapshot_hash`` is the shard header's
    (``real_ft_run.corpus_facts`` refuses otherwise), which is the eval row's protocol's
    (``real_ft_run._protocol``), so the pairing is checked here, where it costs a manifest
    read (0.8 s on v5's 270 MB train.json).
    """
    try:
        manifests = split_cache.read_manifests(shards)
    except split_cache.KeyUnavailable as exc:
        raise Refused(f"--split-cache-shards {shards}: {exc}") from exc
    if manifests.data_snapshot_hash != row.protocol.data_snapshot_hash:
        raise Refused(
            f"--split-cache-shards {shards}: its data/pool/train.json records data_snapshot_hash "
            f"{manifests.data_snapshot_hash}, but eval row {row.row_id}'s model was trained on "
            f"{row.protocol.data_snapshot_hash}. It is not that row's shard set, so no cached "
            "row could be checked against it"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, required=True, help="where the eval row is")
    parser.add_argument(
        "--eval-row", default=None,
        help=(
            "the eval row's id (a prefix is ok). Default: the eval_row_id the verdicts file "
            "names, so a campaign config can name the file without knowing the id in advance"
        ),
    )
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument(
        "--write-ledger", type=Path, default=None,
        help="where to append the gate row; default: --ledger, beside the row it scores",
    )
    parser.add_argument("--commitpackft", type=Path, default=None)
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="exactly as the run was given it. Required, except under --no-repo-history "
             "without --commitpackft, where it bounds nothing and is refused",
    )
    parser.add_argument(
        "--no-repo-history", dest="repo_history", action="store_false",
        help="exactly as the run was given it: the split holds no row from this repository's "
             "git history",
    )
    parser.add_argument("--rev", required=True, help="a full 40-character sha")
    parser.add_argument(
        "--defect-class", type=Path, default=None,
        help="the code.defect_class corpus, exactly as the run was given it: it is part of how "
             "the run built its split, so a control rebuilt without it scores a different split",
    )
    parser.add_argument("--defect-download", type=Path, default=None)
    parser.add_argument("--defect-max-rows", type=int, default=None)
    parser.add_argument(
        "--defect-noul", type=Path, default=None,
        help="exactly as the run was given it: the code.defect_class noul corpus is in the "
             "split, so a control rebuilt without it would pair a different val set",
    )
    parser.add_argument(
        "--general-record", type=Path, default=None,
        help="the general-family fetch record, exactly as the run was given it: the split "
             "holds MMLU/CSQA/CLINC/SQuAD rows only through it",
    )
    parser.add_argument("--general-max-rows", type=int, default=None)
    parser.add_argument(
        "--decisions-pool", type=Path, default=None,
        help="the general-decision pool, exactly as the run was given it: the split holds "
             "the pool's families only through it",
    )
    parser.add_argument(
        "--replay-partition", action="store_true",
        help="exactly as the run was given it: the replay-only rows left the gold train "
             "split, so the control is not fitted on them",
    )
    parser.add_argument(
        "--exclude-identity-keys", type=Path, default=None,
        help="exactly as the run was given it: the qd-prep containment exclusions.txt whose "
             "train rows the run never saw, so the control is not fitted on them either. "
             "Checked against the eval row's exclusions_sha256, in both directions",
    )
    parser.add_argument(
        "--drop-before-dedupe", type=Path, default=None, dest="pre_dedupe_drops",
        help="exactly as the run was given it: the pre-dedupe drop list whose train rows "
             "left the corpus before dedupe. Its sha256 is part of the corpus the exclusion "
             "list's attestation names, so a rebuild without it is refused there",
    )
    parser.add_argument("--control-cache", type=Path, default=None)
    parser.add_argument(
        "--split-cache", type=Path, default=None, metavar="DIR",
        help=(
            "read the split rebuild's result (the train and val rows: ~200 s of each call on "
            "v5's data on the Mac) from DIR when an entry's key matches and the entry passes its "
            "checks; otherwise rebuild and store it there. The cache real_ft_run.py "
            "--split-cache uses (tools/split_cache.py): the key is every input the rebuild "
            "reads, by content, and the code that reads them. Needs --split-cache-shards. Off "
            "by default. Not recipe: the row carries a split_cache metric (hit, miss or "
            "corrupt, with the key). DIR must be owned by the running user, not group- or "
            "world-writable and not a symlink; it is created 0700 when absent. Not "
            "--control-cache, which caches the letter control's fits"
        ),
    )
    parser.add_argument(
        "--split-cache-shards", type=Path, default=None, metavar="DIR",
        help=(
            "with --split-cache: the eval row's shard set, the directory real_ft_run.py was "
            "given as --out. Every cached row is checked against its data/pool/{train,val}.json, "
            "and before the rebuild its train.json must name the eval row's data_snapshot_hash"
        ),
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument(
        "--note", default="",
        help="appended to the row's notes (not its recipe): e.g. that a re-run written to a "
             "report-only ledger is not the row a gate reads unless the human says so",
    )
    parser.add_argument(
        "--dense-budget-gb", type=float, default=24.0,
        help="only prices the --max-fit-minutes projection (calibrated on the Python engine, "
             "so it over-states the native one); the qd-prep fit's arithmetic is the "
             "reference's sparse operand's whatever this is",
    )
    parser.add_argument(
        "--max-fit-minutes", type=float, default=None,
        help="refuse a task whose projected fit exceeds this, and kill a qd-prep fit that runs "
             f"past it; default: no refusal and a {native.DEFAULT_FIT_TIMEOUT_S / 3600:g} h kill",
    )
    parser.add_argument("--hold-out-operator", default="")
    parser.add_argument("--operator-key", default="operator")
    parser.add_argument(
        "--option-control", action="store_true",
        help=f"fit only the per-option control and write it on a row of its own: {OPTION_ARM}.* "
             f"and {OPTION_MARGIN}.* metrics, no gate ({OPTION_REPORT_ONLY})",
    )
    args = parser.parse_args(argv)
    if args.split_cache_shards is not None and args.split_cache is None:
        parser.error("--split-cache-shards without --split-cache reads nothing")
    if args.split_cache is not None:
        if args.split_cache_shards is None:
            raise Refused(
                "--split-cache needs --split-cache-shards: the eval row's shard set "
                "(real_ft_run.py's --out), whose data/pool/{train,val}.json every cached row is "
                "checked against"
            )
        # Refused in seconds, before the ledger, the verdicts or the split are read: the cache
        # reads pickles. real_ft_run.py makes the same check at the same point.
        split_cache.check_dir(args.split_cache)
    if args.option_control and args.hold_out_operator:
        raise Refused(
            "--option-control scores the tasks whose rows offer their own options; an "
            "operator-holdout arm is a code.defect_class diagnostic it has no opponent for"
        )
    if args.option_control and args.control_cache is not None:
        raise Refused(
            "--control-cache caches the letter control's fits; the per-option control is not "
            "cached, so the flag would read and write nothing"
        )
    if args.defect_class is None and (
        args.defect_download is not None or args.defect_max_rows is not None
    ):
        parser.error("--defect-download/--defect-max-rows without --defect-class read nothing")
    if args.general_record is None and (
        args.general_max_rows is not None or args.replay_partition
    ):
        parser.error(
            "--general-max-rows/--replay-partition without --general-record read nothing"
        )
    max_pairs_bounds_nothing = not args.repo_history and args.commitpackft is None
    if max_pairs_bounds_nothing and args.max_pairs is not None:
        raise Refused(
            "--max-pairs bounds the repository-history rows and the --commitpackft sample; "
            "under --no-repo-history without --commitpackft it bounds nothing"
        )
    if not max_pairs_bounds_nothing and args.max_pairs is None:
        raise Refused("--max-pairs is required: it decides which rows the split holds")
    # This tool writes a ledger row, and the split it rebuilds is named by the revision.
    try:
        rev = require_full_sha(args.rev)
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    # Unread by the rebuild when it bounds nothing; ft_split_rows takes an int.
    max_pairs = 0 if args.max_pairs is None else args.max_pairs

    ledger = Ledger(args.ledger)
    verdicts = load_verdicts(args.verdicts)
    row = find_row(ledger.rows(), args.eval_row or verdicts.eval_row_id)
    check_against_eval_row(verdicts, row)
    recorded_holdout = str((row.recipe or {}).get("hold_out_operator", ""))
    if recorded_holdout != args.hold_out_operator:
        raise Refused(
            f"the eval row's model was trained with hold_out_operator={recorded_holdout!r} "
            f"and this control would hold out {args.hold_out_operator!r}: it would not be "
            "that arm's opponent"
        )
    # The eval row carries the run's exclusion list (real_ft_run's RECIPE_PIECE_KEYS); a
    # control fitted with another list, or none, is fitted on rows the model never saw.
    exclusions_sha256 = ""
    if args.exclude_identity_keys is not None:
        try:
            exclusions_sha256 = hashlib.sha256(
                args.exclude_identity_keys.read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise Refused(
                f"--exclude-identity-keys {args.exclude_identity_keys}: unreadable ({exc})"
            ) from exc
    recorded_exclusions = str((row.recipe or {}).get("exclusions_sha256", ""))
    if recorded_exclusions != exclusions_sha256:
        raise Refused(
            f"the eval row's model was trained with exclusion list "
            f"{recorded_exclusions or 'none'} and this control would apply "
            f"{exclusions_sha256 or 'none'}: it would be fitted on a different train split"
        )
    if args.split_cache_shards is not None:
        check_shard_set(args.split_cache_shards, row)
    hold = HoldOut(args.operator_key, args.hold_out_operator) if args.hold_out_operator else None
    # Before the split rebuild, which is most of a run's wall clock before the first fit: a
    # control that cannot be fitted is refused while that costs nothing.
    engine = native.prep_binary()

    config = DataConfig()
    # One set of keywords for the rebuild and for the split cache's key, so what is keyed is what
    # is rebuilt: real_ft_run.main's own form.
    rebuild = functools.partial(
        split_rows_function(),
        commitpackft=args.commitpackft, max_pairs=max_pairs, rev=rev, config=config,
        defect_class=args.defect_class, defect_download=args.defect_download,
        defect_max_rows=args.defect_max_rows, repo_history=args.repo_history,
        general_record=args.general_record, general_max_rows=args.general_max_rows,
        replay_partition=args.replay_partition, defect_noul=args.defect_noul,
        exclude_identity_keys=args.exclude_identity_keys, decisions_pool=args.decisions_pool,
        pre_dedupe_drops=args.pre_dedupe_drops,
    )
    split_cache_state: TriState | None = None
    if args.split_cache is None:
        train_rows, val_rows = rebuild()
    else:
        extra_inputs, extra_facts = split_rebuild_inputs_function()(
            general_record=args.general_record, exclude_identity_keys=args.exclude_identity_keys,
            repo_history=args.repo_history,
        )
        train_rows, val_rows, split_cache_state = split_cache.cached_split_rows(
            args.split_cache, kwargs=rebuild.keywords, rebuild=rebuild,
            extra_inputs=extra_inputs, extra_facts=extra_facts, out=args.split_cache_shards,
            config=config, repo=REPO,
        )
    # Rule 3 through this door too. A control fitted on a held-out family would not train a
    # model, but it would set the bar the model is measured against with data the model may
    # never see -- the same violation wearing a different hat (fit_linear_control.py).
    held = sorted({r.family_id for r in (*train_rows, *val_rows)
                   if config.is_held_out_family(r.family_id)})
    if held:
        raise Refused(
            f"the split handed to the control holds rows of held-out famil(ies) {held}; "
            "rule 3 refuses them to anything that trains or sets a training bar"
        )
    train_docs, train_excluded = request_texts(train_rows, seed=config.seed)
    val_docs, val_excluded = request_texts(val_rows, seed=config.seed)
    print(f"eval row {row.row_id} seed {row.protocol.seed}: {len(verdicts.correct)} letter "
          f"verdict(s), {verdicts.span_rows} span")
    print(f"control docs: {len(train_docs)} train ({len(train_excluded)} excluded), "
          f"{len(val_docs)} val ({len(val_excluded)} excluded); tasks "
          f"{dict(Counter(d.task for d in val_docs))}")

    recipe = {
        "tool": "tools/ft_linear_control.py", "eval_row_id": row.row_id,
        "max_iter": args.max_iter, "hold_out_operator": args.hold_out_operator,
        "operator_key": args.operator_key if hold else "", "key": "slot_name"
        if verdicts.by_slot_name else "kind", "max_pairs": args.max_pairs, "rev": rev,
        "repo_history": args.repo_history,
        "defect_class": None if args.defect_class is None else args.defect_class.name,
        "defect_max_rows": args.defect_max_rows,
        # Which engine fitted the controls, by the bytes that ran. The Python engine wrote no
        # such key; every row with it was fitted by qd-prep, whose sparse-operand arithmetic is
        # the reference's (linear_control_native). A cache hit is keyed without the engine, so
        # a cached task may have been fitted by either (HANDOFF/perf-linear-control-rust-*).
        "control_engine": native.ENGINE_NAME,
        "control_engine_sha256": native.engine_sha256(engine),
    }
    if args.general_record is not None:
        # Only when used, so every gate row written before these inputs existed hashes as
        # it did. The record is named by its sha256, as the pipeline's recipe names it.
        recipe["general_record_sha256"] = hashlib.sha256(
            args.general_record.read_bytes()
        ).hexdigest()
        # Resolved, as the pipeline's recipe and the replay attestation record it, so the
        # default is a number rather than a null that means "whatever the default was".
        import real_tokenizer_pipeline as pipeline

        recipe["general_max_rows"] = (
            pipeline.DEFAULT_GENERAL_MAX_ROWS if args.general_max_rows is None
            else args.general_max_rows
        )
    if args.replay_partition:
        recipe["replay_partition"] = True
    if exclusions_sha256:
        # Only when used, as real_ft_run's recipe pieces name it.
        recipe["exclusions_sha256"] = exclusions_sha256
    if args.defect_noul is not None:
        # Only when used, as the pipeline's and real_ft_run's recipes record it: by the noul
        # corpus's examples sha256, so a control with the noul rows never hashes as one without.
        recipe["defect_noul_examples_sha256"] = str(json.loads(
            (args.defect_noul / "manifest.json").read_text(encoding="utf-8")
        )["examples_sha256"])
    if args.decisions_pool is not None:
        # Only when used, as the pipeline's recipe records it: by the pool's examples sha256,
        # so a control fitted with the pool's rows never hashes as one without.
        recipe["decisions_pool_examples_sha256"] = str(json.loads(
            (args.decisions_pool / "manifest.json").read_text(encoding="utf-8")
        )["examples_sha256"])
    if args.pre_dedupe_drops is not None:
        # Only when used, as the pipeline's recipe records it.
        recipe["pre_dedupe_drops_sha256"] = hashlib.sha256(
            args.pre_dedupe_drops.read_bytes()
        ).hexdigest()
    quick = bool(row.quick) or hold is not None
    quick_reason = (
        f"inherits eval row {row.row_id[:8]}'s quick flag ({row.quick_reason})"
        if row.quick else "an operator-holdout arm is a diagnostic; it promotes nothing"
    ) if quick else None
    if args.option_control:
        return write_option_row(
            args, row, verdicts, train_docs, val_docs, recipe=recipe, quick=quick,
            quick_reason=quick_reason, engine=engine, split_cache_state=split_cache_state,
        )
    # The recorder WRAPS the fit: on a rented box this CPU work is billed with the instance,
    # and a fit killed outside a block would leave no row saying it ran.
    with RunRecorder(
        Ledger(args.write_ledger or args.ledger), entry_point=Path(__file__),
        # The eval row's protocol, so this gate row joins that configuration's seed family.
        protocol=Protocol(**row.protocol.to_json()), run_kind="eval", repo=REPO,
        env=Environment.detect(device="cpu"), wall_clock_s=None, cost=None,
        recipe=recipe, quick=quick, quick_reason=quick_reason,
        notes=(f"tools/ft_linear_control.py: paired_margin_vs_linear for eval row {row.row_id}"
               f" over {len(verdicts.correct)} letter rows{_note(args)}"),
    ) as recorder:
        started = time.monotonic()
        result = score_against_control(
            train_docs, val_docs, verdicts, seed=row.protocol.seed, max_iter=args.max_iter,
            dense_budget_bytes=int(args.dense_budget_gb * 1024**3),
            max_fit_minutes=args.max_fit_minutes, cache_dir=args.control_cache, hold=hold,
            engine=engine,
        )
        recorder.measured(time.monotonic() - started)
        recorder.metric("scored_eval_row_id", Ran(passed=True, value=row.row_id,
                                                  detail="the eval row these verdicts are"))
        if split_cache_state is not None:
            recorder.metric(SPLIT_CACHE_METRIC, split_cache_state)
        recorder.metric("control_train_rows", Ran(
            passed=True, value=result.train_rows, detail=(
                f"letter slots the control was fitted on; the holdout removed "
                f"{result.removed_by_holdout}")))
        for name, state in result.metrics.items():
            recorder.metric(name, state)
        recorder.gate(GATE, result.gate)
    written = recorder.row
    print(f"{GATE}: {result.gate}")
    print(f"wrote row {written.row_id if written else '?'} to {args.write_ledger or args.ledger}")
    return 0 if isinstance(result.gate, Ran) else 3


def write_option_row(
    args: argparse.Namespace, row: LedgerRow, verdicts: Verdicts,
    train_docs: list[RequestDoc], val_docs: list[RequestDoc], *, recipe: dict[str, object],
    quick: bool, quick_reason: str | None, engine: Path, split_cache_state: TriState | None,
) -> int:
    """The per-option control's own supplement row: metrics only, never a gate.

    Joined to the eval row by ``eval_row_id`` as the gate's row is, so promotion reads the
    three as one unit; every key here is the per-option control's own, and the ledger joins
    only gates and controls, so this row changes no gate or control the unit is judged by. It
    inherits the eval row's ``quick`` flag exactly as the gate's row does -- a report-only row
    must not become a promotion blocker -- and, like any row of the unit, a failed run blocks
    promotion until it is rerun.
    """
    recipe = {
        **recipe,
        "control": OPTION_ARM,
        "report_only": True,
        "comparator_decision": "the human's, under G1 (Fable H, 2026-10-01)",
        # The interpretation taken of Fable's "(question + option) pair text", stated where the
        # numbers are: the row's question is the prompt's context region, not the whole prompt,
        # whose remainder is constant within a task.
        "option_question_text": "the rendered prompt's context region",
    }
    with RunRecorder(
        Ledger(args.write_ledger or args.ledger), entry_point=Path(__file__),
        protocol=Protocol(**row.protocol.to_json()), run_kind="eval", repo=REPO,
        env=Environment.detect(device="cpu"), wall_clock_s=None, cost=None,
        recipe=recipe, quick=quick, quick_reason=quick_reason,
        notes=(f"tools/ft_linear_control.py --option-control for eval row {row.row_id}: "
               f"{OPTION_REPORT_ONLY}{_note(args)}"),
    ) as recorder:
        started = time.monotonic()
        result = score_against_option_control(
            train_docs, val_docs, verdicts, seed=row.protocol.seed, max_iter=args.max_iter,
            max_fit_minutes=args.max_fit_minutes, engine=engine,
        )
        recorder.measured(time.monotonic() - started)
        recorder.metric(f"{OPTION_ARM}.scored_eval_row_id", Ran(
            passed=True, value=row.row_id, detail="the eval row these verdicts are"))
        if split_cache_state is not None:
            recorder.metric(f"{OPTION_ARM}.{SPLIT_CACHE_METRIC}", split_cache_state)
        recorder.metric(f"{OPTION_ARM}.train_rows", Ran(
            passed=True, value=result.train_rows,
            detail="training letter slots of the per-row-option tasks"))
        for name, state in result.metrics.items():
            recorder.metric(name, state)
    written = recorder.row
    pooled = result.metrics[f"{OPTION_MARGIN}.choice"]
    print(f"{OPTION_MARGIN}.choice (report-only): {pooled}")
    print(f"wrote row {written.row_id if written else '?'} to {args.write_ledger or args.ledger}")
    return 0 if isinstance(pooled, Ran) else 3


if __name__ == "__main__":
    raise SystemExit(main())
