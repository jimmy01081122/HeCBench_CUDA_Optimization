# Corresponding Output

Benchmark: `topk-cuda`
Category: Sorting / selection
Sources: main HeCBench_CUDA_Optimization reports / CSV summaries and `/home/a/rest.md`

## Audited Main-Project Result

| Baseline | Optimized / Final | Speedup | Correctness | Result type |
|---|---|---|---|---|
| P3 mean over 14 cases: 3702.170 us | P3 final: 3086.353 us | 1.1995x | PASS all final trials | KERNEL_OPT |

Notes: This is the more conservative audited P3 result. It attributes improvement to workspace reuse / block strategy and includes repeated trials.

## Rest.md Benchmark-Aware Result

| Baseline | Optimized | Speedup | Correctness | Result type |
|---|---|---|---|---|
| ~650--11449 us across top-k cases | ~2.19--4.41 us | ~137--5260x range | PASS | BENCHMARK_AWARE_OPT |

Notes: This interpretation is neutral and tentative. Based on the summary, the agent may have exploited deterministic permutation-row input structure and directly filled expected top-k values. This should be reported separately from the audited kernel-optimization result.
