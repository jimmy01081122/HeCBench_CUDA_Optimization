# softmax-cuda Agent Summary

## Environment
- Date: 2026-06-09 Asia/Taipei
- Node: gn1222
- GPU: Tesla V100-SXM2-32GB
- CUDA module: cuda/12.8
- nvcc: CUDA 12.8, V12.8.61
- Build: `make clean || true && make ARCH=sm_70`
- Benchmark sweep: `./main 100000 <slice> <impl> 100` for slices 64, 128, 256, 512, 784, 1024, 2048 and implementations 0/1

## Prompt Protocol
- Prompt level: P3
- Optimization submission limit: 5, baseline excluded
- Baseline job: 948563
- Final accepted job: 948573

## Baseline Result
- Status: valid baseline
- Correctness: 14 PASS, 0 FAIL
- stderr: empty
- Primary required case, slice 784 implementation 1: 1.450565 ms

## Submission History
| index | job | variant | status | result_type | notes |
|---:|---:|---|---|---|---|
| 1 | 948565 | cache numerator to reduce optimized-kernel expf calls | rejected | REGRESSION | PASS but impl 1 regressed for most slices |
| 2 | 948566 | block-per-slice dispatch >=512 | rejected | BUILD_FIX | compile failed: `CUDART_INF_F` undefined |
| 3 | 948568 | block-per-slice dispatch >=512 | accepted | KERNEL_OPT | PASS and large-slice speedup |
| 4 | 948571 | tune threshold to >=256 | rejected | PARAM_TUNE | PASS but slice 256 regressed to 0.503766 ms |
| 5 | 948573 | accepted kernel with 3 trials | accepted final | KERNEL_OPT | 42 PASS, 0 FAIL |

## Correctness Table
| job | required outputs | PASS | FAIL | valid |
|---:|---:|---:|---:|---|
| 948563 | 14 | 14 | 0 | yes |
| 948565 | 14 | 14 | 0 | yes |
| 948566 | 0 | 0 | 0 | no |
| 948568 | 14 | 14 | 0 | yes |
| 948571 | 14 | 14 | 0 | yes |
| 948573 | 42 | 42 | 0 | yes |

## Performance Table
| slice | impl | baseline ms | final mean ms | min | max | stddev | speedup vs baseline | classification |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 0 | 1.562978 | 1.670418 | 1.576444 | 1.734823 | 0.083227 | 0.936x | REGRESSION |
| 64 | 1 | 0.073627 | 0.073166 | 0.072757 | 0.073481 | 0.000371 | 1.006x | MEASUREMENT_EQUIVALENT |
| 128 | 0 | 4.371422 | 4.399024 | 4.389662 | 4.404030 | 0.008114 | 0.994x | MEASUREMENT_EQUIVALENT |
| 128 | 1 | 0.136389 | 0.136874 | 0.136631 | 0.137127 | 0.000248 | 0.996x | MEASUREMENT_EQUIVALENT |
| 256 | 0 | 15.592968 | 15.618146 | 15.600123 | 15.637608 | 0.018784 | 0.998x | MEASUREMENT_EQUIVALENT |
| 256 | 1 | 0.305836 | 0.306225 | 0.305878 | 0.306587 | 0.000355 | 0.999x | MEASUREMENT_EQUIVALENT |
| 512 | 0 | 32.658607 | 32.777074 | 32.770187 | 32.788342 | 0.009838 | 0.996x | MEASUREMENT_EQUIVALENT |
| 512 | 1 | 0.841365 | 0.634466 | 0.634294 | 0.634553 | 0.000149 | 1.326x | KERNEL_OPT |
| 784 | 0 | 54.675507 | 54.741344 | 54.714527 | 54.790619 | 0.042729 | 0.999x | MEASUREMENT_EQUIVALENT |
| 784 | 1 | 1.450565 | 0.995243 | 0.994848 | 0.995810 | 0.000503 | 1.457x | KERNEL_OPT |
| 1024 | 0 | 64.667046 | 64.636904 | 64.596954 | 64.674850 | 0.038987 | 1.000x | MEASUREMENT_EQUIVALENT |
| 1024 | 1 | 2.109552 | 1.209571 | 1.209448 | 1.209758 | 0.000165 | 1.744x | KERNEL_OPT |
| 2048 | 0 | 137.948685 | 138.251913 | 137.702271 | 138.680191 | 0.500129 | 0.998x | MEASUREMENT_EQUIVALENT |
| 2048 | 1 | 4.417161 | 3.555559 | 3.532282 | 3.581077 | 0.024475 | 1.242x | KERNEL_OPT |

## Variance Statistics
- Final accepted candidate used 3 trials in one Slurm job.
- impl 1 slice 512: mean 0.634466 ms, min 0.634294, max 0.634553, stddev 0.000149, CV 0.023%
- impl 1 slice 784: mean 0.995243 ms, min 0.994848, max 0.995810, stddev 0.000503, CV 0.051%
- impl 1 slice 1024: mean 1.209571 ms, min 1.209448, max 1.209758, stddev 0.000165, CV 0.014%
- impl 1 slice 2048: mean 3.555559 ms, min 3.532282, max 3.581077, stddev 0.024475, CV 0.688%

## Profiler / Measurement Notes
- No separate profiler Slurm run was launched because the five-submission optimization budget was fully used by baseline-following attempts and the required 3-trial final validation.
- Occupancy expectation: the final block-per-slice kernel uses 256 threads and one 256-float shared array per block, so shared memory use is about 1 KiB/block and should not be the occupancy limiter on V100.
- expf instruction note: rejected attempt 1 reduced exponentials from two per element to one per element but added global output traffic and regressed, indicating the original large-slice path was not helped by that tradeoff.
- Shared memory note: accepted kernel uses shared memory only for block reductions of max and sum; baseline warp kernel uses cooperative-groups warp reductions without explicit shared memory.
- Memory throughput note: accepted block kernel keeps two input-read passes and one output-write pass like the original, but increases parallelism within each large slice. The measured gains at 512+ are consistent with reduced per-slice serial work rather than reduced memory traffic.

## Contradiction Check
- Raw final output count: 42 PASS, 0 FAIL.
- Summary correctness tables match raw output counts.
- Speedups use measured baseline job 948563, not estimates.
- Rejected attempts 948565, 948566, and 948571 are not used as final metrics.
- Since no case failed in final job 948573, the final correctness statement is all final validation cases PASS.

## Final Conclusion
FINAL_RESULT: ACCEPTED_KERN_OPT

The final candidate dispatches implementation 1 to the original warp-per-slice kernel for slices below 512 and to a new block-per-slice reduction kernel for slices 512 and larger. The primary required optimized case, slice 784, improved from 1.450565 ms to a 3-trial mean of 0.995243 ms, a 1.457x speedup, with all final validation cases passing.
