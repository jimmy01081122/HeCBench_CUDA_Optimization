# Corresponding Output

Benchmark: `softmax-cuda`
Category: ML kernel
Source: main HeCBench_CUDA_Optimization reports / CSV summaries

| Baseline | Optimized / Final | Speedup | Correctness | Result type |
|---|---|---|---|---|
| P3 baseline slice=784 impl=1: 1.450565 ms | P3 final: 0.995243 ms; Mode B/Mode C further shape-aware results | P3 1.4575x; Mode B large slices 1.3375--1.6989x; Mode C additional 1.0487--1.1355x | PASS | KERNEL_OPT / PARAM_TUNE |
