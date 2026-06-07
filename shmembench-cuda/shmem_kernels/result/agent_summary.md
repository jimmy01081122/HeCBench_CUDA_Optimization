# shmembench-cuda Agent Summary

## 1. Environment

- GPU model: NVIDIA Tesla V100-SXM2-32GB
- CUDA_VISIBLE_DEVICES: `0`
- Number of GPUs: 1
- nvcc version: CUDA 12.8, V12.8.61
- CUDA arch: `sm_70`
- Node: `gn1222.twcc.ai`
- Slurm settings: `-A ACD115083`, `-N 1`, `--ntasks-per-node=1`, `--gpus-per-node=1`, `-t 00:10:00`
- Module: `cuda/12.8`

## 2. Benchmark Characterization

- `shmembench-cuda` is a shared-memory bandwidth microbenchmark from the gpumembench style of tests.
- Kernel pattern: each thread initializes six `float4` values in `__shared__` memory, repeatedly swaps pairs of `float4` values for `TOTAL_ITERATIONS=1024`, uses `__syncthreads()` between swap phases, then writes one reduced `float4` per thread to global memory.
- It primarily measures shared-memory vector swap throughput plus synchronization/instruction overhead, not application-level memory bandwidth.
- It includes a checksum check; after instrumentation this is explicit as `CHECKSUM,...,status=PASS/FAIL`.
- Timing method after Submission 2: CUDA events around repeated kernel launches, with 5 warmup launches and 5 timing trials.
- Performance metric: average/min/max kernel time and derived throughput in GB/s.
- Bytes formula retained from original code: `(6 + 4*5*TOTAL_ITERATIONS + 6) * size * sizeof(float)`, where `size=VECTOR_SIZE=1048576`.

## 3. Baseline

- Baseline job id: `946712`
- Build: PASS with `make ARCH=sm_70`
- Run: PASS
- Correctness: PASS by absence of `checksum failed`; original benchmark did not print an explicit PASS
- Baseline performance: `6.550231 ms`, `13121.63 GB/sec`
- Failure reason: none
- Notes: Makefile default `ARCH=sm_60` and typoed `${LFLAGS}` were real infrastructure issues, but the baseline script’s `ARCH=sm_70` override allowed build success.

## 4. Submission History

| Submission | Job id | Modification | Hypothesis | Result | Correctness | Performance |
|---:|---:|---|---|---|---|---|
| 1 | 946714 | Fixed Makefile default `sm_70`, link flags, and explicit object build rules. | Build should not depend on script-only arch override or empty typoed link flags. | PASS. | Checksum did not fail. | `6.547937 ms`, `13126.22 GB/s`. |
| 2 | 946717 | Added CUDA event timing, warmup, 5 trials, avg/min/max, explicit checksum status, and structured `RESULT`. | CPU chrono timing around launches was less trustworthy. | PASS. | Explicit checksum PASS. | avg `6508.801937 us`, `13205.145647 GB/s`. |
| 3 | 946722 | Tried block-size sweep: 128, 256, 512. | Block size might expose occupancy/synchronization limits. | INVALID: block 128 changed deterministic checksum and failed before later variants. | FAIL for block 128. | Invalid; not used. |
| 4 | 946728 | Restored valid block 256 and set `cudaFuncCachePreferShared`. | Prefer-shared cache config might slightly help a shared-memory-heavy kernel. | PASS. | Explicit checksum PASS. | avg `6511.340618 us`, `13199.997145 GB/s`; measurement-equivalent to Submission 2. |
| 5 | 946735 | Final confirmation and CSV generation. | Current valid event-timed block-256 benchmark remains stable and parseable. | PASS. | Explicit checksum PASS. | avg `6509.391308 us`, `13203.950032 GB/s`. |

## 5. Performance Table

| test_name | block | stride / pattern | avg_us | min_us | max_us | bandwidth_GBps | correctness | status |
|---|---:|---|---:|---:|---:|---:|---|---|
| baseline_original | 256 | barriered float4 swaps | 6550.231 | n/a | n/a | 13121.63 | PASS, implicit | PASS |
| event_timed | 256 | barriered float4 swaps | 6508.801937 | 6490.023136 | 6538.070679 | 13205.145647 | PASS | PASS |
| block_sweep | 128 | barriered float4 swaps | 6434.020 | 6424.208 | 6463.988 | invalid | FAIL | INVALID |
| prefer_shared_final | 256 | barriered float4 swaps | 6509.391308 | 6493.736744 | 6537.559986 | 13203.950032 | PASS | PASS |

Final CSV: `result/shmembench_results_946735.csv`

## 6. Optimization Analysis

- Effective changes: Makefile cleanup and CUDA event timing improved reproducibility and reporting.
- Measurement-equivalent changes: `cudaFuncCachePreferShared` produced no meaningful performance change versus Submission 2.
- Invalid attempt: changing `BLOCK_SIZE` without updating/deriving the checksum changed benchmark semantics as observed by checksum failure, so those performance values are not valid.
- Hardware-limit interpretation: reported shared-memory bandwidth is an internal operation-count metric for repeated shared-memory swaps, not external DRAM bandwidth. The value is plausible for a microbenchmark but should not be treated as application throughput.
- Likely bottlenecks: synchronization and instruction overhead inside the repeated swap loop, plus shared-memory bank/access behavior. No profiler/Nsight metrics were collected, so bank conflict details are inferred from code structure rather than measured counters.

## 7. Limitations

- Single node only.
- Single GPU only, as requested for priority testing.
- Submit limit respected: baseline + 5 optimization submissions.
- No Nsight Compute / profiler metrics.
- Parameter sweep was attempted but invalidated by checksum semantics.
- The checksum is tied to the original block-256 configuration; future sweeps need an analytical or per-configuration reference checksum before results can be valid.

## 8. Final Conclusion

SUCCESS: correctness PASS and performance measured for the valid original block-256 shared-memory benchmark. The main work was build repair, timing/output repair, and validation; no significant performance optimization was demonstrated, and the final performance is measurement-equivalent to the best event-timed result.
