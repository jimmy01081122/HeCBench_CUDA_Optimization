# Mode C Submission 1 Summary

Slurm job: 950665 on `gn1230.twcc.ai`.

Candidate: `impl4_shape_specialized_large_reduce`.

Profiler status: `NOT_RUN`.

## Per-Slice Results

| slice | impl=1 mean ms | impl=3 mean ms | impl=4 mean ms | speedup_vs_impl3 | correctness | validity | result_type |
|---:|---:|---:|---:|---:|---|---|---|
| 128 | 0.138020 | 0.157587 | 0.137648 | 1.144855 | PASS | VALID | MEASUREMENT_EQUIVALENT |
| 256 | 0.306415 | 0.307093 | 0.306706 | 1.001262 | PASS | VALID | MEASUREMENT_EQUIVALENT |
| 784 | 1.449216 | 1.028795 | 0.909558 | 1.131093 | PASS | VALID | MODE_C_CANDIDATE |
| 1024 | 2.109019 | 1.241707 | 1.183763 | 1.048949 | PASS | VALID | MODE_C_CANDIDATE |
| 2048 | 2.237945 | 1.675165 | 1.661960 | 1.007945 | PASS | VALID | MEASUREMENT_EQUIVALENT |

## Interpretation

- Primary Mode C comparison is `impl=4` vs `impl=3`.
- Small slices 128 and 256 dispatch to unchanged `impl=1`; no Mode C optimization speedup is claimed for them.
- Large slices 784 and 1024 have valid additional speedup vs `impl=3` in this Submission 1 run.
- Large slice 2048 is measurement-equivalent vs `impl=3`, not a Mode C speedup claim.
- Profiler was not run, so there is no profiler-supported bottleneck conclusion.
- No cached-exp causality claim is made because no ablation was run.
- Submission 1 label before human audit: `SUCCESS_WITH_ADDITIONAL_SPEEDUP`.
