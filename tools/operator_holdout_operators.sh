# The operators the operator-holdout experiment runs over, in one place. SOURCED, never
# executed, by launch_operator_holdout_model.sh (the model arms) and by
# fit_operator_holdout_controls.sh (the controls those arms are measured against), so the two
# cannot disagree about which training sets exist.
#
# They did disagree. The launcher took QD_HOLDOUT_OPERATORS from 7ae12bc on, while the fit
# script kept its own hardcoded list of the three defaults. The three minor operators then
# ran as model arms with no way to fit their opponents short of editing the fit script, and
# all 48 of their holdout and size-matched rows came back paired_margin_vs_linear: not_run.
#
# The defaults are the dominant operator in each class. Measured as a share of the 37385
# training rows (from each holdout arm's own "holding out" line): stub.panic 16037 (42.9%),
# logic.change_constant 5326 (14.2%), cosmetic.rename_local 3349 (9.0%). Holding out the
# biggest operator in a class is the weakest version of the test, because it moves the class
# prior hardest -- the confound the sibling metric then has to rule out. The minor operators
# barely move it: logic.negate_condition 1995 (5.3%), cosmetic.edit_comment 1837 (4.9%),
# stub.default_return 670 (1.8%).
#
#   QD_HOLDOUT_OPERATORS="logic.negate_condition cosmetic.edit_comment stub.default_return" \
#     bash tools/launch_operator_holdout_model.sh <40-char-sha>
#
# `-`, not `:-`. With the colon an override that is SET BUT EMPTY -- QD_HOLDOUT_OPERATORS=
# "$LIST" with $LIST accidentally blank -- falls back to the defaults and silently runs the
# three dominant operators instead of the ones asked for. Without it, only an UNSET variable
# takes the defaults, and a blank one reaches the refusal below.
OPERATORS="${QD_HOLDOUT_OPERATORS-stub.panic logic.change_constant cosmetic.rename_local}"
if [ -z "${OPERATORS// /}" ]; then
  echo "refusing to start: QD_HOLDOUT_OPERATORS is set but empty. That would run nothing" >&2
  echo "and still print the DONE marker, which a watcher cannot tell from a finished run." >&2
  exit 2
fi
