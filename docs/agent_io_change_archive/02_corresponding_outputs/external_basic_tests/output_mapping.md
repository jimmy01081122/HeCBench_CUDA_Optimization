# External Basic Tests: Prompt-to-Output Mapping

Source: `/home/a/data.md`

Hardware environment is the same as the main project: NVIDIA Tesla V100-SXM2-32GB, CUDA `sm_70`, Slurm.

Times are copied from `data.md`. Lower is assumed better. Raw output files are not available in the provided record and are therefore marked `N/A`.

| Benchmark | Baseline | Optimized Output | Env-Optimized Output | Raw Output | Correctness / Validity Notes |
|---|---:|---:|---:|---|---|
| `cc-cuda` | 0.0034 | fail | fail | N/A | Optimized and env-optimized failed. |
| `floydwarshall-cuda` | 0.107097 | 0.000024 | 0.00024 | N/A | Extreme speedup is flagged suspicious/invalid in source analysis. |
| `floydwarshall2-cuda` | 0.000851 | 0.098891 | 0.09133 | N/A | Severe regression; roughly 100x slower by source analysis. |
| `gc-cuda` | 0.000048 | 0.000285 | fail | N/A | Source analysis says graph algorithms fail after optimization; timing exists but correctness is unsafe. |
| `mis-cuda` | 0.00136 | 0.002057 | fail | N/A | Source analysis says graph algorithms fail after optimization; timing exists but correctness is unsafe. |
| `merge-cuda` | 17.03105 | 13.7232 | 16.6688 | N/A | Validity/correctness raw evidence N/A; reported speedup about 1.24x optimized. |
| `quicksort-cuda` | 46.1346 | 45.8452 | fail | N/A | Optimized timing is near measurement-equivalent; env-optimized failed. |
| `sortKV-cuda` | 88.1414 | 72.9803 | 76.29895 | N/A | Validity/correctness raw evidence N/A; reported speedup about 1.21x optimized. |
| `bitonic-sort-cuda` | 70.13246 | 33.50338 | 34.68863 | N/A | Validity/correctness raw evidence N/A; reported speedup about 2.09x optimized. |
| `split-cuda` | 3423.724 | 3023.754 | 3569.766 | N/A | Optimized improves about 1.13x; env-optimized regresses. |

## Analysis Summary from Source

- Sorting / regular workloads show the most credible improvement: `bitonic-sort-cuda`, `sortKV-cuda`, `merge-cuda`.
- Irregular graph workloads are high-risk: `cc-cuda`, `gc-cuda`, `mis-cuda`.
- `floydwarshall-cuda` reports an extreme `0.107097 -> 0.000024` improvement, which the source flags as computationally implausible for an `O(N^3)` algorithm.
- Environment-specific V100 tuning does not consistently improve performance and can fail or regress.

