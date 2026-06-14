# topk-cuda Agent Summary

## Environment

- Date: 2026-06-09
- Node: gn1223
- GPU visibility: CUDA_VISIBLE_DEVICES=0
- CUDA module: cuda/12.8
- nvcc: CUDA 12.8, V12.8.61
- Build command: `make clean || true && make ARCH=sm_70`
- Benchmark command: `./main 3072 100`
- Prompt level: P3
- Submission limit: 5 optimization submissions; baseline did not count.

## Baseline Result

- Job: 948627
- Correctness: 14 PASS, 0 FAIL
- Mean metric across 14 hidden_size/topk cases: 3702.17 us
- Status: valid baseline

## Submission History

| Submission | Job | Variant | Result type | Status | Mean us | Notes |
|---:|---:|---|---|---|---:|---|
| 1 | 948629 | cached workspace | KERNEL_OPT | accepted | 3101.51 | Removed repeated workspace allocation/free from timed top-k calls. |
| 2 | 948631 | cached workspace + block 512 | REGRESSION | rejected | 3123.74 | Valid correctness, but mean slower than submission 1. |
| 3 | 948632 | cached workspace + hybrid block 512/1024 | KERNEL_OPT | accepted final trial 1 | 3083.34 | Uses block 512 for len <= 4096, block 1024 otherwise. |
| 4 | 948633 | final repeat 2 | KERNEL_OPT | accepted final trial 2 | 3081.38 | Repeat of submission 3 for variance. |
| 5 | 948636 | final repeat 3 | KERNEL_OPT | accepted final trial 3 | 3094.34 | Repeat of submission 3 for variance. |

## Correctness Table

| Run | PASS | FAIL |
|---|---:|---:|
| baseline 948627 | 14 | 0 |
| attempt1 948629 | 14 | 0 |
| attempt2 948631 | 14 | 0 |
| attempt3 948632 | 14 | 0 |
| attempt4 948633 | 14 | 0 |
| attempt5 948636 | 14 | 0 |

## Final Performance

Final candidate is attempts 3-5 combined.

| Case | Final mean us |
|---|---:|
| hidden=3072, topk=2048 | 327.106 |
| hidden=3072, topk=1024 | 288.104 |
| hidden=4096, topk=2048 | 377.290 |
| hidden=4096, topk=1024 | 360.796 |
| hidden=8192, topk=2048 | 676.894 |
| hidden=8192, topk=1024 | 584.434 |
| hidden=16384, topk=2048 | 1136.486 |
| hidden=16384, topk=1024 | 1233.497 |
| hidden=32768, topk=2048 | 3018.069 |
| hidden=32768, topk=1024 | 2876.211 |
| hidden=65536, topk=2048 | 5623.489 |
| hidden=65536, topk=1024 | 5434.458 |
| hidden=131072, topk=2048 | 10754.300 |
| hidden=131072, topk=1024 | 10517.821 |

## Variance Statistics

- Trial means: 3083.34 us, 3081.38 us, 3094.34 us
- Mean: 3086.353 us
- Min: 3081.380 us
- Max: 3094.340 us
- Sample stddev: 6.986 us
- Coefficient of variation: 0.226%
- Baseline mean: 3702.17 us
- Final speedup: 1.1995x
- Time reduction: 16.63%

## Profiler / Measurement Notes

- Nsight profiler data was not collected because the 5 allowed optimization submissions were used for the accepted candidate search and the required 3 final trials.
- CUB temporary storage notes: CUB `BlockScan` and `BlockReduce` temp storage remains shared-memory scoped inside the one-block radix kernels. The high-value change was reusing the global radix workspace instead of allocating/freeing it on every timed call.
- Workspace notes: maximum cached workspace for hidden_size 131072 and batch 3072 is about 201 MB; the cache grows only when a larger workspace is required.
- Register pressure, occupancy, and memory throughput were not directly profiled; attempt 2/3 block-size tuning served as an empirical occupancy/register-pressure probe. Block 512 helped small rows, while block 1024 remained better for larger rows.

## Result Files

- CSV: `result/topk-cuda_results.csv`
- Raw baseline: `result/topk-cuda_result_948627.txt`
- Raw final trials: `result/topk-cuda_result_attempt3_cached_workspace_hybrid512_1024_948632.txt`, `result/topk-cuda_result_attempt4_final_repeat2_948633.txt`, `result/topk-cuda_result_attempt5_final_repeat3_948636.txt`

## Final Conclusion

SUCCESS

The final accepted KERNEL_OPT candidate preserves all correctness checks, runs the full hidden_size/topk matrix, and improves the measured mean from 3702.17 us to 3086.353 us across three final trials.


TOKENS : 72,850 used