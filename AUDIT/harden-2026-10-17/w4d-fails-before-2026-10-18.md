# W4-d fails-before, run (2026-10-18)

Fable Q15: a fails-before must have run. One mac_heavy window (`w4d-proof`, `MAC_HEAVY_MAX_LOAD=48`, lock held 13:04:41Z, finished exit 0 at 13:07:27Z), driven by scratch `build/w4d_proof.sh` (gitignored) over two detached scratch worktrees, one at the base `af4bf6d` and one at the branch head `d96704b`. Both were removed afterwards. Output below is copied from `build/w4d-proof.log`, abridged only by dropping cargo's compile lines.

## (a) cluster_key and the per-function cap

Fixture: the cap unit test's pool, 40 records of one single-function Rust file, `--seed 3 --clean-permille 0`.

Pre-fix `qd-mutate generate` (af4bf6d):

```
rows=40 rows_with_cluster_key=0
max rows of one class from one function=26
  function=total/2 class=stub rows=26
  function=total/2 class=logic rows=12
  function=total/2 class=cosmetic rows=2
```

Branch, `--max-per-function 1000` (uncapped). The rows are the pre-fix rows plus the key, byte for byte:

```
pre-fix rows sha256=bfad4e70b55fb31bb233d06c6cd6bbd09e7114ebb347eb4369cc45277ade026f
branch rows without cluster_key sha256=bfad4e70b55fb31bb233d06c6cd6bbd09e7114ebb347eb4369cc45277ade026f
IDENTICAL
rows=40 rows_with_cluster_key=40
```

Branch, `--max-per-function 2`:

```
rows=6 rows_with_cluster_key=6 distinct_keys=1
max rows of one class per cluster_key=2 (K=2)
function_cap: {"dropped_by_class": {"logic": 10, "stub": 24}, "keys_seen": 1, "max_per_function": 2, "rows_after": 6, "rows_before": 40, "rows_per_key_after": {"6": 1}, "rows_per_key_before": {"40": 1}, "rows_per_key_class_before": {"12": 1, "2": 1, "26": 1}}
rows_before - rows_after = 34, sum(dropped_by_class) = 34
CHECKS OK: cap holds at 2, drops recorded and equal to the difference
```

## (b) span ECE, through the real qd-calib-fit

One synthetic span verdict file per case (430 lines each: 30 letter lines, which qd-calib-fit needs, and 400 span lines whose pointers put a confidence on a winner that hits at a fixed rate). Calibrated: confidence equals hit rate in every group. Miscalibrated: confidence says nothing about hits.

`qd-calib-fit` at af4bf6d, calibrated input:

```
span.fitted=True span.scored keys=['abstained_by_margin', 'abstained_by_rule', 'answered', 'answered_correct', 'answered_precision', 'gold_abstains', 'how', 'n_not_scored', 'n_scored', 'pointer_conformal_coverage', 'served_accuracy', 'served_correct', 'verdict_correct']
  /span/scored/ece_after is ABSENT: the report records no span check
```

`qd-calib-fit` at d96704b:

```
calibrated:    /span/scored/ece_after = {"bins": 15, "n": 800, "passed": true, "populated_bins": 4, "state": "ran", "threshold": 0.05, "value": 5.407491010522847e-09}
miscalibrated: /span/scored/ece_after = {"bins": 15, "n": 800, "passed": false, "populated_bins": 2, "state": "ran", "threshold": 0.05, "value": 0.37550860800252617}
```

`qd-export` on the `span` section of those real reports (the rest of the report is the gate test's fixed letter entries):

```
d96704b  OLD qd-calib-fit report, calibrated span:     exit=2  refused: ... code.defect_class/span: the fit's report records no ece_after for this entry; an unrecorded check is not a passed one
d96704b  NEW qd-calib-fit report, calibrated span:     exit=0
d96704b  NEW qd-calib-fit report, miscalibrated span:  exit=2  refused: ... the fit's recorded calibration check failed (ECE 0.3755, passed=false, threshold 0.05)
af4bf6d  OLD qd-calib-fit report, calibrated span:     exit=2  refused: ... records no ece_after for this entry
af4bf6d  NEW qd-calib-fit report, calibrated span:     exit=0
af4bf6d  NEW qd-calib-fit report, miscalibrated span:  exit=2  refused: ... ECE 0.3755, passed=false
```

(The proof log's third line of each block has a shorter label than the table above: it names the report, not the export. The first column here is which tree's `qd-export` ran.)

Reading: the gate did not change between af4bf6d and d96704b (the old export admits the new calibrated report and refuses the miscalibrated one). The defect was the writer: the old `qd-calib-fit` wrote no span check, so the gate refused every fitted span. That is what the fix changed.

## Not covered

* The span input is synthetic, built so the expected verdicts are known; it is not a real scoring run's verdicts.
* The `qd-export` fixture is the gate test's (fixed letter entries and table); only the `span` section comes from the real reports.
