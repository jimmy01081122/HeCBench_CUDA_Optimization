# moe-align CUDA Optimization Summary

## Environment

- Benchmark path: `/home/r14525078/p3/HeCBench/src/moe-align-cuda`
- Result path: `/home/r14525078/p3/HeCBench/src/moe-align-cuda/result`
- Prompt level: P3
- Submission limit: 5 optimization submissions; used 4 optimization submissions plus baseline and profiler-only jobs.
- GPU/node observations: baseline on gn1222, optimization/profiler runs on gn1115, CUDA_VISIBLE_DEVICES=0.
- CUDA module: cuda/12.8
- NVCC: CUDA compilation tools release 12.8, V12.8.61
- Build command: `make clean || true && make ARCH=sm_70`
- Benchmark command for scored runs: `./main 100000`

## Source Changes

Final accepted source changes in `main.cu`:

- Cached the large-path `cumsum_buff` device allocation across calls instead of allocating/freeing every repeat.
- Replaced the per-call `cudaMemset` with stream-ordered `cudaMemsetAsync` before the dependent kernels.
- Preserved all input sizes, repeat count, validation checks, and tolerance/semantics.

Backups created before source edits:

- `main.cu.bak_agent`
- `main.cu.attempt1.bak_agent`

## Baseline Result

- Job: 948025
- Node: gn1222
- Correctness: PASS=30, FAIL=0
- Mean latency over 30 official cases: 19.366169 us
- Status: valid baseline

## Submission History

| Submission | Job | Variant | Correctness | Mean latency (us) | Result type | Accepted | Notes |
|---:|---:|---|---|---:|---|---|---|
| 1 | 948048 | cached_cumsum | PASS=30, FAIL=0 | 16.833719 | PARAM_TUNE | yes | Removed repeated cumsum allocation/free overhead. |
| 2 | 948049 | small_batch_128 | PASS=30, FAIL=0 | 21.573584 | REGRESSION | no | Correct but slower; not used for final speedup. |
| 3 | 948050 | cached_cumsum repeat | PASS=30, FAIL=0 | 16.842155 | PARAM_TUNE | yes | Repeat trial for variance. |
| 4 | 948052 | cached_cumsum repeat | PASS=30, FAIL=0 | 16.836133 | PARAM_TUNE | yes | Repeat trial for variance. |

## Correctness Table

Every scored baseline/optimization run reported 30 PASS and 0 FAIL over the full official matrix: tokens={(1, 3, 256, 4096, 8192)}, topk={(2, 3, 4)}, experts={(32, 128)}, block_size=32.

## Performance Table

- Baseline mean: 19.366169 us
- Final accepted trial means: 16.833719, 16.842155, 16.836133 us
- Final accepted mean of means: 16.837336 us
- Final accepted min/max trial mean: 16.833719 / 16.842155 us
- Trial stddev: 0.004345 us
- Coefficient of variation: 0.0258%
- Speedup versus measured baseline: 13.06% latency reduction

Full per-case CSV with explicit correctness/status fields: `result/moe-align_results.csv`.

## Profiler / Measurement Notes

Profiler-only jobs were separate from scored performance runs and used `./main 1` to keep Nsight Compute practical. Job 948055 with `--set speedOfLight` reported no section metrics available. Job 948057 with explicit metrics succeeded for 12 representative launches.

Representative Nsight Compute metrics from job 948057:

- `moe_align_block_size_kernel`: active warps about 49.14-49.18% of peak sustained active; DRAM throughput about 0.14-0.15% of peak sustained elapsed on sampled launches.
- `count_and_sort_expert_tokens_kernel`: active warps about 7.32-8.13%; DRAM throughput about 0.08-0.09%.
- `moe_align_block_size_small_batch_expert_kernel`: active warps about 5.29-5.81%; DRAM throughput about 0.14-0.15%.

The scored benchmark is dominated by launch/synchronization and host-side allocation overhead for these small kernels, so low memory-throughput percentages are expected from the sampled launches.

## Contradiction Check

- Raw scored outputs counted PASS=30 and FAIL=0 for baseline and accepted final trials.
- Attempt 2 is rejected and excluded from final speedup.
- Speedup uses measured baseline job 948025 and measured accepted candidate jobs 948048, 948050, 948052.
- No case was skipped, waived, or reduced in scored runs; repeat remained 100000 for scored runs.

## Result Classification

- Primary result type: PARAM_TUNE
- Final conclusion label: SUCCESS

## Next Optimization Recommendations

- Move repeated `cudaDeviceSynchronize` out of the per-call path only if the benchmark/API contract allows timing a batched loop with one final sync; this would be a larger measurement-semantics decision.
- Explore fusing the large path's count/align and sort work for medium/large token counts to reduce launch count.
- Avoid the 128-expert small-batch extension tested in job 948049 unless the shared-memory layout is redesigned; it was correctness-valid but slower.

72,786 used