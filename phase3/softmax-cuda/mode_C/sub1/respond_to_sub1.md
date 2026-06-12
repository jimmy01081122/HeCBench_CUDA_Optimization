Submission 1 result audit: ACCEPT_WITH_LIMITATIONS.

Accepted:
- correctness PASS for all official slices
- auditor PASS
- contradiction checks PASS
- valid additional Mode C speedup vs impl=3 for:
  - slice=784: 1.131x
  - slice=1024: 1.049x

Not accepted as speedup claims:
- slice=128: small-slice fallback to impl=1; impl=3 baseline is noisy; no Mode C speedup claim
- slice=256: small-slice fallback to impl=1; measurement-equivalent
- slice=2048: speedup_vs_impl3=1.008x < 1%; measurement-equivalent

Final label:
- SUCCESS_WITH_ADDITIONAL_SPEEDUP

Required reporting constraints:
- Report per-slice results first.
- Do not report only aggregate speedup.
- Do not claim impl=4 is universal.
- Do not claim profiler-supported bottleneck because profiler_status=NOT_RUN.
- Do not claim cached-exp or warp-reduction causality without ablation.
- Do not use speedup_vs_impl1 as the main Mode C success metric.
- State that accepted additional Mode C speedup is limited to slices 784 and 1024.