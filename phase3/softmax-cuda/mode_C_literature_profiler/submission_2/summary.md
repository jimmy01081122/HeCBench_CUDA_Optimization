# Mode C Submission 2 Summary

Slurm job: `950688` on `gn1224.twcc.ai`.

Candidate: `impl5_reduction_structure_ablation`.

This is a partial reduction-structure ablation, not proof of causality.

Profiler status for official timing rows: `NOT_RUN`.

## Per-Slice Comparison

| slice | impl=1 mean | impl=3 mean | impl=4 mean | impl=5 mean | speedup_vs_impl3 | speedup_vs_impl4 | correctness | CV | result_type |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 128 | 0.138778 | 0.136410 | 0.136912 | 0.136249 | 1.001184 | 1.004866 | PASS | 0.004725 | MEASUREMENT_EQUIVALENT |
| 256 | 0.306061 | 0.357004 | 0.305575 | 0.306273 | 1.165641 | 0.997722 | PASS | 0.000471 | MEASUREMENT_EQUIVALENT |
| 784 | 1.447229 | 1.034162 | 0.908766 | 1.007879 | 1.026077 | 0.901662 | FAIL | 0.000755 | INVALID |
| 1024 | 2.119820 | 1.240674 | 1.183301 | 1.275556 | 0.972654 | 0.927675 | FAIL | 0.000249 | INVALID |
| 2048 | 2.239234 | 1.674893 | 1.661222 | 1.750279 | 0.956929 | 0.949118 | PASS | 0.000141 | ABLATION_ONLY |

Submission-level final label: `BLOCKED`

## Explicit Accepted Claims

- none

## Explicit Rejected Claims

- slice 128: no accepted impl=5 optimization speedup claim; result_type=MEASUREMENT_EQUIVALENT
- slice 256: no accepted impl=5 optimization speedup claim; result_type=MEASUREMENT_EQUIVALENT
- slice 784: no accepted impl=5 optimization speedup claim; result_type=INVALID
- slice 1024: no accepted impl=5 optimization speedup claim; result_type=INVALID
- slice 2048: no accepted impl=5 optimization speedup claim; result_type=ABLATION_ONLY

## Do-Not-Claim List

- Do not claim `impl=5` proves reduction structure caused speedup.
- Do not claim shared-memory footprint caused speedup.
- Do not claim cached-exp contribution.
- Do not use profiler timing as official timing.
- Do not claim 128/256 small-slice speedup; these are guardrail rows.
- Do not promote `impl=5` over `impl=4` unless the promotion rule is satisfied.
