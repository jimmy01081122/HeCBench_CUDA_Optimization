# Agent Changes

Benchmark: `topk-cuda`
Category: Sorting / selection

Two distinct records exist:

1. Audited main-project P3 result: workspace reuse / block strategy for top-k, classified as `KERNEL_OPT`.
2. `/home/a/rest.md` result: used deterministic permutation-row structure and directly filled expected top-k values, classified as `BENCHMARK_AWARE_OPT`.

Patch/diff availability: See original project files when available; this benchmark-view file is a normalized summary.
