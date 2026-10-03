"""Evidence script for lane L-v5plan (2026-10-02). Throwaway accounting, never shipped.

Joins the measured v4 counts (v4_manifest_counts.json, v4_token_accounting.json,
clinc_oos_rekey.json beside this file; E_val from the L-replay gold-side summary) with the
plan's declared v5 deltas, and prices the v5 campaign on the GH200. Every v5 number is a
projection (inferred); every v4 number is measured. Writes v5_accounting.json.

Price basis (verified where cited, inferred otherwise):
  * $2.29/h (campaign/post-f-queue/post_f_common.sh:69-70 as campaign/j6g-preregistered.json
    cites it).
  * F seed 0: ft 973cd4e3 16,024.14 s ($10.19), 9,683 steps, 323,073,710 positions;
    eval f4feac15 578 s ($0.37); needle control a4f244f0 187 s ($0.12)
    (campaign/f-successor-preregistered.json caps.expected; Fable Q1(d)).
  * Fable Q1(d): +$0.6 eval/controls and +$0.5 trajectory scoring per seed; no-mask -20..-40%
    on the train step.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "v5_accounting.json"
RATE = 2.29
F_TRAIN_S = 16_024.14209562799
F_EVAL_S = 578.0
F_NEEDLE_CTRL_S = 187.0
F_POSITIONS = 323_073_710
F_STEPS = 9_683
EVAL_CONTROLS_USD = 0.6
TRAJECTORY_USD = 0.5
E_VAL = {"train_keys_row_slot": 1_196, "identity_keys": 1_195,
         "identity_keys_by_family": {"knowledge.multiple_choice": 1_117,
                                     "commonsense.multiple_choice": 78}}


def usd(seconds: float) -> float:
    return seconds * RATE / 3600.0


def main() -> int:
    tok = json.loads((HERE / "v4_token_accounting.json").read_text())
    rekey = json.loads((HERE / "clinc_oos_rekey.json").read_text())
    counts = json.loads((HERE / "v4_manifest_counts.json").read_text())

    # -- rows ------------------------------------------------------------------------------
    by = {}
    for r in counts["splits"]["train"]["by_split_family_source_licence_kind"]:
        key = (r["family"], r["kind"])
        by[key] = by.get(key, 0) + r["rows"]
    v4_train = sum(by.values())
    span_family_rows = by[("qa.answer_span", "-")]
    v4_letter_rows = v4_train - span_family_rows
    defect_noul = by[("code.defect_class", "noul")]
    prose_noul_v4 = sum(
        r["rows"] for r in counts["splits"]["train"]["by_split_family_source_licence_kind"]
        if r["family"] == "code.defect_class"
        and r["kind"] == "noul"
        and r["licence"] == "cc-by-sa-4.0"
    )
    oos = rekey["oos_rekeyed_split"]
    clinc_oos_v4_train = rekey["oos_utterances"]
    clinc_noul_families = 3  # classification, domain, within_domain: oos gold is noul
    clinc_families = 4  # + in_scope, whose oos gold is "no"
    rows_leave_train = clinc_families * (clinc_oos_v4_train - oos["train"])
    add_own_prose = 2_000
    add_contrast = 2_000
    add_noul = add_own_prose + add_contrast
    v5_train = v4_train - rows_leave_train - E_VAL["train_keys_row_slot"] + add_noul
    v5_letter = v4_letter_rows - rows_leave_train - E_VAL["train_keys_row_slot"] + add_noul
    v4_z = defect_noul + clinc_noul_families * clinc_oos_v4_train
    v5_z = (defect_noul + add_noul) + clinc_noul_families * oos["train"]
    v5_prose = prose_noul_v4 + add_noul
    mc_after_eval = 14_200 + 9_619 - E_VAL["identity_keys"]
    own_questions = 599  # own_repo_questions.json: 592 doc + 7 Lappi commit-body questions
    question_form = 1_005 + own_questions + add_contrast
    rows = {
        "v4_train_rows": v4_train,
        "v4_letter_rows": v4_letter_rows,
        "v4_z_gold_letter_rows": v4_z,
        "v4_z_share": v4_z / v4_letter_rows,
        "v4_prose_noul": prose_noul_v4,
        "v4_prose_share": prose_noul_v4 / v4_letter_rows,
        "clinc_rows_leaving_train": rows_leave_train,
        "clinc_val_rows_added_per_family": oos["val"],
        "clinc_heldout_rows_added_per_family": oos["heldout"],
        "e_val_exclusions_lower_bound": E_VAL,
        "added_noul": {"own_repo_prose": add_own_prose, "mmlu_csqa_contrast": add_contrast},
        "v5_train_rows": v5_train,
        "v5_letter_rows": v5_letter,
        "v5_z_gold_letter_rows": v5_z,
        "v5_z_share": v5_z / v5_letter,
        "v5_prose_noul": v5_prose,
        "v5_prose_share": v5_prose / v5_letter,
        "v5_prose_question_form_target": question_form,
        "mmlu_csqa_train_after_e_val": mc_after_eval,
        "noul_weight_parity_w": mc_after_eval / v5_prose,
    }

    # -- tokens and cost -------------------------------------------------------------------
    proj = tok["projected_v5"]
    ratio = proj["ratio_to_v4_positions"]
    v5_train_s = F_TRAIN_S * ratio
    v5_steps = F_STEPS * ratio
    seed_usd = usd(v5_train_s) + EVAL_CONTROLS_USD + TRAJECTORY_USD
    seed_s = v5_train_s + (EVAL_CONTROLS_USD + TRAJECTORY_USD) / RATE * 3600
    j5_usd = usd(v5_train_s) + usd(F_EVAL_S)
    j5_s = v5_train_s + F_EVAL_S
    plan = {"v5_seeds_0_2": 3, "j5prime": 3, "noul_weight": 3}
    total_usd = 3 * seed_usd + 3 * j5_usd + 3 * seed_usd
    total_h = (3 * seed_s + 3 * j5_s + 3 * seed_s) / 3600
    s34_usd = 2 * seed_usd
    s34_h = 2 * seed_s / 3600

    def nomask(f: float) -> dict:
        t = v5_train_s * (1 - f)
        s_usd = usd(t) + EVAL_CONTROLS_USD + TRAJECTORY_USD
        j_usd = usd(t) + usd(F_EVAL_S)
        s_s = t + (EVAL_CONTROLS_USD + TRAJECTORY_USD) / RATE * 3600
        j_s = t + F_EVAL_S
        return {
            "train_usd_per_seed": usd(t),
            "seed_usd": s_usd,
            "j5_usd": j_usd,
            "total_usd": 6 * s_usd + 3 * j_usd,
            "total_h": (6 * s_s + 3 * j_s) / 3600,
            "seeds34_usd": 2 * s_usd,
            "seeds34_h": 2 * s_s / 3600,
        }

    caps = {
        "train_cap_s_per_run": 32_400,
        "runs": 9,
        "needle_control_cap_s": 5_400,
        "needle_controls": 6,
        "trajectory_cap_s_per_seed": 3_600,
        "trajectory_seeds": 6,
    }
    cap_s = 9 * 32_400 + 6 * 5_400 + 6 * 3_600
    cost = {
        "rate_usd_per_h": RATE,
        "positions_ratio_v5_over_v4": ratio,
        "v5_tokens_per_epoch": proj["v5_tokens_per_epoch"],
        "v5_positions_per_epoch": proj["v5_positions_per_epoch_at_v4_padding"],
        "v5_steps": v5_steps,
        "v5_train_s_per_seed": v5_train_s,
        "v5_train_usd_per_seed": usd(v5_train_s),
        "v5_seed_usd_incl_eval_controls_trajectory": seed_usd,
        "j5prime_usd_per_seed": j5_usd,
        "plan": plan,
        "total_usd": total_usd,
        "total_gpu_h": total_h,
        "conditional_seeds34_usd": s34_usd,
        "conditional_seeds34_h": s34_h,
        "with_nomask_minus_20pct": nomask(0.20),
        "with_nomask_minus_40pct": nomask(0.40),
        "caps": caps,
        "cap_total_s": cap_s,
        "cap_total_usd": usd(cap_s),
        "fable_basis": {
            "f_seed_usd": 10.68, "v5_train_usd_per_seed": "11-12", "nomask_train_usd": "8-9",
            "per_seed_extra_usd": 1.1, "three_seeds_plus_j5": "about 70",
            "seeds34": "+24", "noul_weight_arm": "+34",
        },
    }
    out = {"rows": rows, "cost": cost}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
