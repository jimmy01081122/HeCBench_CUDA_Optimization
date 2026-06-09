# moe-cuda P3 Optimization Summary

## Environment

- Date: 2026-06-09
- Benchmark path: `/home/r14525078/p3/HeCBench/src/moe-cuda`
- Result path: `/home/r14525078/p3/HeCBench/src/moe-cuda/result`
- GPU jobs submitted through Slurm only.
- Account: `ACD115083`
- Partition used: `gp1d`
- CUDA module: `cuda/12.8`
- NVCC: CUDA compilation tools release 12.8, V12.8.61
- Target arch: `sm_70`

## Prompt Level And Limit

- Prompt level: P3
- Optimization submission limit: 3
- Optimization submissions used: 1
- Baseline does not count as a submission.

## Baseline Result

- Job: `948395`
- Node: `gn1226.twcc.ai`
- Output files:
  - `result/moe-cuda_948395.out`
  - `result/moe-cuda_948395.err`
  - `result/moe-cuda_result_948395.txt`
- Stderr: CUDA module load banner only; no fatal errors.
- Correctness: PASS for all official cases.

| case | correctness | average kernel time us |
|---|---:|---:|
| topk=1 | PASS | 285.113678 |
| topk=2 | PASS | 347.580872 |
| topk=4 | PASS | 519.113708 |
| topk=8 | PASS | 949.848694 |

## Submission History

### Submission 1: fused_smallk

- Job: `948399`
- Node: `gn1226.twcc.ai`
- Modification: added `moeSoftmaxTopKSmallK<256>` and routed `topk <= 4` through a fused softmax+topk kernel. `topk=8` remains on the original separate softmax and topk kernels.
- Hypothesis: removing one launch and avoiding full softmax workspace global write/read improves small top-k cases.
- Expected improvement: strongest for topk=1 and topk=2, smaller for topk=4, neutral for topk=8.
- Validation target: official cases `32768 tokens, 384 experts, topk 1/2/4/8, repeat 1000`, all PASS.
- Output files:
  - `result/moe-cuda_948399.out`
  - `result/moe-cuda_948399.err`
  - `result/moe-cuda_result_948399.txt`
- Stderr: CUDA module load banner only; no fatal errors.
- Result: accepted.
- Result type: `KERNEL_OPT`; topk=8 classified as `MEASUREMENT_EQUIVALENT`.

## Correctness Table

| run | PASS cases | FAIL cases | notes |
|---|---:|---:|---|
| baseline job 948395 | 4 | 0 | one official run for topk 1/2/4/8 |
| submission 1 job 948399 | 12 | 0 | three trials for all four official topk cases |
| profiler job 948404 | 4 | 0 | repeat 1 profiler run for all four topk cases |

## Performance Table

| case | baseline us | accepted mean us | min us | max us | stddev us | cv % | speedup vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| topk=1 | 285.113678 | 216.274740 | 206.436874 | 235.898926 | 16.995063 | 7.858090 | 31.829394% |
| topk=2 | 347.580872 | 293.344187 | 293.326904 | 293.377655 | 0.028989 | 0.009882 | 18.489095% |
| topk=4 | 519.113708 | 490.138082 | 490.042969 | 490.278442 | 0.124086 | 0.025317 | 5.911727% |
| topk=8 | 949.848694 | 950.246786 | 949.975342 | 950.475342 | 0.252744 | 0.026598 | -0.041894% |

The accepted candidate improves topk 1/2/4 by more than 1%. Topk=8 is within 1% and is not claimed as a speedup.

## Profiler And Measurement Notes

- Profiler job: `948404`
- Node: `gn1221.twcc.ai`
- Output files:
  - `result/moe-cuda_profile_948404.out`
  - `result/moe-cuda_profile_948404.err`
  - `result/moe-cuda_profile_948404.txt`
- Note: an earlier Slurm submit returned a socket timeout but later materialized as duplicate profiler job `948405` on `gn1222.twcc.ai`. Its raw outputs are preserved under `result/moe-cuda_profile_948405.*`; it also reported PASS for all four profiled cases. The primary profiler data below uses job `948404`.
- `ncu`, `nvprof`, and `nsys` were available in the CUDA 12.8 job environment.
- Nsight Compute metrics collected: `sm__warps_active.avg.pct_of_peak_sustained_active`, `dram__bytes.sum`, `lts__t_bytes.sum`, and `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`.
- Resource usage from `cuobjdump --dump-resource-usage ./main`:
  - fused `moeSoftmaxTopKSmallK<256>`: 32 registers, 88 bytes static shared memory per block.
  - original `moeSoftmax<256>`: 30 registers, 52 bytes static shared memory per block.
  - original `moeTopK<256>`: 32 registers, 80 bytes static shared memory per block.
- Launch count per benchmark iteration:
  - topk=1/2/4 accepted path: 1 fused kernel launch.
  - topk=8 retained path: 2 launches, `moeSoftmax` then `moeTopK`.

Representative profiler observations from timed/profile launches:

| case | kernel path | occupancy metric % | dram bytes | L2 bytes | shared wavefront metric |
|---|---|---:|---:|---:|---:|
| topk=1 | fused | 95.85-95.95 | 50.684768M-51.066720M | 53.865856M-53.866176M | 9.170573M-9.171100M |
| topk=2 | fused | 97.01-97.06 | 51.030656M-51.431072M | 56.995968M-57.010912M | 12.809120M-12.809201M |
| topk=4 | fused | 97.92-97.95 | 51.784576M-52.195552M | 63.181408M-63.218016M | 20.437621M-20.439999M |
| topk=8 | original softmax | 93.00-93.17 | 97.517088M-97.932256M | 102.318464M-102.382144M | 5.923562M-5.937641M |
| topk=8 | original topk | 98.42 | 53.112416M-53.140544M | 75.758272M-75.777344M | 28.899155M-28.899176M |

Profiler timing is inflated by instrumentation and is not used for speedup.

## CSV

- CSV result file: `result/moe-cuda_results.csv`

## Contradiction Check

- PASS/FAIL counts were checked from raw output.
- Speedup uses measured baseline job `948395`, not an estimate.
- Only accepted submission `948399` is used for final performance.
- No correctness failures were observed; all official nonzero cases passed.
- Inputs and repeat count for final performance remained `32768 384 topk 1/2/4/8 repeat 1000`.


TOKENS : 41,203 used