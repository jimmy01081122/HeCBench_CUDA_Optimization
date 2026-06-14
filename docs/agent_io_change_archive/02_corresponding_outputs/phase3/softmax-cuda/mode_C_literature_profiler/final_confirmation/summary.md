# Mode C Final Confirmation Summary

Slurm job: `950691` on `gn1224.twcc.ai`.

Candidate: `impl4_shape_specialized_large_reduce`.

`impl=5` is mentioned only as a blocked ablation artifact and was not run in final confirmation.

Profiler status for official timing rows: `NOT_RUN`.

## Per-Slice Table

| slice | impl=1 mean | impl=3 mean | impl=4 mean | speedup_vs_impl3 | speedup_vs_impl1 | correctness | CV | result_type |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 128 | 0.137974 | 0.136672 | 0.136548 | 1.000906 | 1.010441 | PASS | 0.005285 | MEASUREMENT_EQUIVALENT |
| 256 | 0.305982 | 0.306187 | 0.305269 | 1.003007 | 1.002335 | PASS | 0.002846 | MEASUREMENT_EQUIVALENT |
| 784 | 1.448263 | 1.031688 | 0.908544 | 1.135540 | 1.594048 | PASS | 0.000143 | MODE_C_CANDIDATE |
| 1024 | 2.107634 | 1.240982 | 1.183307 | 1.048740 | 1.781139 | PASS | 0.000211 | MODE_C_CANDIDATE |
| 2048 | 2.238517 | 1.674963 | 1.661275 | 1.008239 | 1.347470 | PASS | 0.000069 | MEASUREMENT_EQUIVALENT |

Mode C final label: `SUCCESS_WITH_ADDITIONAL_SPEEDUP`

Final confirmation status: `CONFIRMED`

## Accepted Claims

- slice 784: impl=4 measured speedup_vs_impl3=1.135540
- slice 1024: impl=4 measured speedup_vs_impl3=1.048740

## Rejected Claims

- slice 128: no accepted impl=4 additional speedup claim; result_type=MEASUREMENT_EQUIVALENT
- slice 256: no accepted impl=4 additional speedup claim; result_type=MEASUREMENT_EQUIVALENT
- slice 2048: no accepted impl=4 additional speedup claim; result_type=MEASUREMENT_EQUIVALENT

## Do-Not-Claim List

- Do not claim small-slice Mode C speedup for 128 or 256.
- Do not use speedup_vs_impl1 as the main Mode C metric.
- Do not use profiler timing as official timing.
- Do not claim profiler-supported causality.
- Do not claim cached-exp contribution.
- Do not claim shared-memory-footprint causality.
- Do not claim reduction-structure causality.
- Do not promote `impl=5`; it remains a blocked ablation artifact.
- Do not hide per-slice regression or measurement equivalence behind aggregate success.
