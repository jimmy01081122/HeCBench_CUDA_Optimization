# Corresponding Output

Benchmark: `filter-cuda`
Category: Array / tensor
Source: `/home/a/rest.md`

| Baseline | Optimized | Speedup | Correctness | Result type |
|---|---|---|---|---|
| shared ~1.013790 ms; global ~2.837900 ms | shared ~0.630574 ms; global ~0.579578 ms | shared ~1.53--1.61x; global ~4.61--4.90x | PASS | BENCHMARK_AWARE_OPT |

Notes: For BENCHMARK_AWARE_OPT entries, the interpretation is neutral and tentative: based on the summary, the agent may have exploited fixed input generation, repeated work, or validation structure.
