# HeCBench moe-cuda Optimization Summary
Token : 75,416 used
## Experiment Setup

- Benchmark: MoE gating softmax + top-k expert selection
- Hardware: NVIDIA V100, `sm_70`
- Fixed input: `32768` tokens, `384` experts, repeat `1000`
- Tested top-k values: `1`, `2`, `4`, `8`
- Each submitted version was evaluated with 10 runs per top-k.
- Correctness constraint: all versions preserved the original reference check, tolerance, input size, and repeat count.

## Submission Overview

| Version | Job ID | Strategy | PASS | FAIL |
|---|---:|---|---:|---:|
| V1 | 944980 | Fused softmax + top-k for top-k 1/2/4/8 | 40 | 0 |
| V2 | 944983 | Fused top-k 1/2/4, original two-kernel path for top-k 8 | 40 | 0 |
| V3 | 944997 | Dedicated top-k 1 kernel, fused top-k 2/4, original path for top-k 8 | 40 | 0 |

## Average Kernel Time

All values are average execution time of kernels in microseconds.

| top-k | Baseline | V1 | V2 | V3 |
|---:|---:|---:|---:|---:|
| 1 | 311.860419 | 239.392831 | 239.096091 | 168.573779 |
| 2 | 395.990085 | 345.344208 | 345.435580 | 344.903119 |
| 4 | 599.585193 | 561.996075 | 562.304511 | 561.396344 |
| 8 | 1112.569775 | 1229.952942 | 949.442499 | 949.431598 |

## V1 Analysis

V1 introduced a fused `moeSoftmaxTopK<TPB, TOPK>` kernel for all judged top-k values. The fused kernel computes the row maximum, softmax denominator, cached per-expert softmax probabilities, and top-k selection within one block per token.

This was effective for small top-k because it removed the full global softmax workspace write/read and reduced two timed kernel launches to one. Compared with baseline:

| top-k | Baseline | V1 | Change |
|---:|---:|---:|---:|
| 1 | 311.860419 | 239.392831 | -23.24% |
| 2 | 395.990085 | 345.344208 | -12.79% |
| 4 | 599.585193 | 561.996075 | -6.27% |
| 8 | 1112.569775 | 1229.952942 | +10.55% |

The key finding from V1 is that full fusion is not universally beneficial. For top-k 8, repeated in-block top-k reductions became more expensive than the original implementation's two-kernel approach. This made V1 slower than baseline for top-k 8 despite reducing global memory traffic.

## V2 Analysis

V2 used V1's measured result to choose a hybrid strategy:

- Keep fused softmax + top-k for top-k 1, 2, and 4.
- Restore the original two-kernel path for top-k 8.

This corrected the V1 top-k 8 regression. Relative to V1, top-k 8 improved from `1229.952942 us` to `949.442499 us`, a `22.81%` reduction.

V2 also remained faster than baseline for all four top-k values:

| top-k | Baseline | V2 | Change |
|---:|---:|---:|---:|
| 1 | 311.860419 | 239.096091 | -23.33% |
| 2 | 395.990085 | 345.435580 | -12.77% |
| 4 | 599.585193 | 562.304511 | -6.22% |
| 8 | 1112.569775 | 949.442499 | -14.66% |

The V2 result suggests that the best structure is top-k dependent: low top-k benefits from fusion, while top-k 8 benefits from the original softmax materialization plus top-k scan.

## V3 Analysis

V3 specialized the top-k 1 case further with `moeSoftmaxTop1<TPB>`. Since top-k 1 only needs the maximum expert and its softmax probability, the kernel avoids the V2 fused path's 384-entry shared softmax cache and extra top-k reduction.

The top-k 1 kernel performs:

- one block reduction to find the maximum expert, with smaller-index tie behavior,
- one block reduction for the softmax denominator,
- one output write for the selected expert.

For top-k 1, the selected maximum expert has probability:

```text
exp(row_max - row_max) / sum(exp(input - row_max))
= 1 / sum(exp(input - row_max))
```

This preserves correctness while reducing work.

V3 improvements relative to V2:

| top-k | V2 | V3 | Change |
|---:|---:|---:|---:|
| 1 | 239.096091 | 168.573779 | -29.50% |
| 2 | 345.435580 | 344.903119 | -0.15% |
| 4 | 562.304511 | 561.396344 | -0.16% |
| 8 | 949.442499 | 949.431598 | -0.00% |

V3 final improvements relative to baseline:

| top-k | Baseline | V3 | Speedup | Time Reduction |
|---:|---:|---:|---:|---:|
| 1 | 311.860419 | 168.573779 | 1.850x | 45.95% |
| 2 | 395.990085 | 344.903119 | 1.148x | 12.90% |
| 4 | 599.585193 | 561.396344 | 1.068x | 6.37% |
| 8 | 1112.569775 | 949.431598 | 1.172x | 14.66% |

## Research Conclusions

The three submissions show that the optimal kernel structure depends strongly on top-k.

1. Fusion is profitable for small top-k.

   For top-k 2 and 4, fused softmax + top-k consistently reduces runtime by removing global workspace traffic and eliminating one kernel launch. The benefit decreases as top-k grows because top-k selection requires more repeated reductions.

2. Full fusion is harmful for top-k 8.

   V1 showed that fusing top-k 8 caused a regression from baseline. The extra per-token reductions and synchronization overhead outweighed the memory traffic saved by avoiding the global softmax workspace.

3. Top-k 1 deserves a separate algorithm.

   V3 demonstrated the largest improvement by recognizing that top-k 1 does not need a full softmax vector. Finding the max expert first allows the selected probability to be computed directly from the denominator. This reduced top-k 1 runtime from `239.096091 us` in V2 to `168.573779 us` in V3.

4. The final implementation is a hybrid dispatch.

   The best measured strategy is:

   - top-k 1: dedicated `moeSoftmaxTop1<TPB>`
   - top-k 2 and 4: fused `moeSoftmaxTopK<TPB, TOPK>`
   - top-k 8: original `moeSoftmax` + `moeTopK`

This final V3 strategy passes all correctness checks and gives the strongest measured performance across the fixed benchmark set.

## Result Files

- V1 result: `results/moe_cuda_result_944980.txt`
- V2 result: `results/moe_cuda_result_944983.txt`
- V3 result: `results/moe_cuda_result_944997.txt`
- V1 note: `results/CG_V1_doc.md`
- V2 note: `results/CG_V2_doc.md`
- V3 note: `results/CG_V3_doc.md`
