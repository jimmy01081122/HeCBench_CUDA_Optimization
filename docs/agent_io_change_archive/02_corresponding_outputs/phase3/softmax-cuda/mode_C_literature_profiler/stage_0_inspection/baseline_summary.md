# Stage 0 Baseline Summary

Inspected Mode B artifacts:
- final `results.csv`
- final `round_summary.md`
- final `auditor_report.csv`
- final `contradiction_check.csv`
- Round 1 summary and patch summary
- Round 2 summary and patch summary

Path note:
- The prompt listed round summaries directly under `round_1/` and `round_2/`.
  The actual files are under `round_1/docs/` and `round_2/doc/`.

Mode B final accepted candidate:
- `impl3_shape_dispatch_impl1_small_impl2_large`

Final Mode B aggregate results:

| slice | dispatch | impl=3 mean ms | paired impl=1 mean ms | speedup vs impl=1 | result |
|---:|---:|---:|---:|---:|---|
| 128 | impl=1 | 0.134574 | 0.134869 | 1.002197 | MEASUREMENT_EQUIVALENT |
| 256 | impl=1 | 0.321505 | 0.321793 | 1.000895 | MEASUREMENT_EQUIVALENT |
| 784 | impl=2 | 1.036402 | 1.442716 | 1.392043 | PARAM_TUNE |
| 1024 | impl=2 | 1.238443 | 2.104045 | 1.698944 | PARAM_TUNE |
| 2048 | impl=2 | 1.672904 | 2.237452 | 1.337466 | PARAM_TUNE |

Auditor and contradiction checks:
- Final auditor checks: PASS.
- Final contradiction checks: PASS.
- Profiler status: NOT_RUN.

Important interpretation boundaries:
- `impl=3` is a shape-aware dispatch result.
- The small-slice rows are not valid speedup claims because they dispatch to
  unchanged `impl=1`.
- Large-slice gains are valid Mode B dispatch-policy gains, but cannot be
  attributed solely to cached exponentials without an ablation.
- A Mode C speedup claim must compare against `impl=3`, not against the
  original `mode_C/main.cu` copy.
