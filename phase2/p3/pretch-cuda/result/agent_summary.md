# prefetch-cuda P3 Summary

## Environment

- Benchmark path: `/home/r14525078/p3/HeCBench/src/prefetch-cuda`
- Result path: `/home/r14525078/p3/HeCBench/src/prefetch-cuda/result`
- Prompt level: P3
- Submission limit: 5 optimization submissions; used 3
- GPU node: `gn1222`
- GPU: Tesla V100-SXM2-32GB
- CUDA_VISIBLE_DEVICES: `0`
- Module: `cuda/12.8`
- nvcc: CUDA compilation tools release 12.8, V12.8.61
- Build command: `make clean || true && make ARCH=sm_70`
- Benchmark commands: `./main 10`, `./main 100`

## Baseline Result

- Job: `948608`
- Raw files:
  - `result/prefetch-cuda_948608.out`
  - `result/prefetch-cuda_948608.err`
  - `result/prefetch-cuda_result_948608.txt`
- Correctness: 40 PASS, 0 FAIL
- Status: valid baseline

| Case | Mode | Mean avg_ms | Min | Max | Stddev |
|---|---:|---:|---:|---:|---:|
| repeat=10 | with_prefetch | 6.192024 | 6.098832 | 6.310051 | 0.079166 |
| repeat=10 | without_prefetch | 12.365682 | 12.217469 | 12.504829 | 0.088478 |
| repeat=100 | with_prefetch | 1.681913 | 1.680878 | 1.682529 | 0.000458 |
| repeat=100 | without_prefetch | 2.145355 | 2.106580 | 2.164934 | 0.017962 |

## Submission History

| Submission | Job | Modification | Hypothesis | Result | Accepted |
|---:|---:|---|---|---|---|
| 1 | 948609 | Set both modes to 512-thread blocks | Fewer blocks may reduce scheduling overhead | Correct; no-prefetch improved, prefetch repeat=100 was measurement-equivalent | Yes, but not final |
| 2 | 948610 | Set both modes to 1024-thread blocks | Larger blocks may further improve memory-kernel throughput | Correct; no-prefetch improved, but prefetch repeat=100 regressed >1% | No |
| 3 | 948611 | Keep prefetch at 256-thread blocks; set no-prefetch to 1024-thread blocks | Preserve prefetch behavior while improving demand-paging path | Correct; no-prefetch improved strongly, prefetch stayed equivalent/slightly improved | Yes, final |

## Final Candidate

- Job: `948611`
- Result type: `PARAM_TUNE`
- Source change: in `main.cu`, only the no-prefetch `naive()` kernel launch block size changed from `256` to `1024`.
- Correctness: 40 PASS, 0 FAIL

| Case | Mode | Baseline avg_ms | Final avg_ms | Speedup | Classification |
|---|---|---:|---:|---:|---|
| repeat=10 | with_prefetch | 6.192024 | 6.109064 | 1.34% | PARAM_TUNE |
| repeat=10 | without_prefetch | 12.365682 | 10.187994 | 17.61% | PARAM_TUNE |
| repeat=100 | with_prefetch | 1.681913 | 1.684764 | -0.17% | MEASUREMENT_EQUIVALENT |
| repeat=100 | without_prefetch | 2.145355 | 1.921765 | 10.42% | PARAM_TUNE |

## Variance Statistics

The final accepted candidate includes 10 timing samples for each required case/mode from the benchmark's repeated mode loop.

| Case | Mode | Mean avg_ms | Min | Max | Stddev |
|---|---|---:|---:|---:|---:|
| repeat=10 | with_prefetch | 6.109064 | 6.096423 | 6.153045 | 0.015511 |
| repeat=10 | without_prefetch | 10.187994 | 9.958567 | 10.381825 | 0.138326 |
| repeat=100 | with_prefetch | 1.684764 | 1.682055 | 1.698291 | 0.004781 |
| repeat=100 | without_prefetch | 1.921765 | 1.898970 | 1.940632 | 0.012673 |

## Profiler And Measurement Notes

- Profiler job: `948612`
- Raw files:
  - `result/prefetch-cuda_profile_948612.out`
  - `result/prefetch-cuda_profile_948612.err`
  - `result/prefetch-cuda_profile_result_948612.txt`
  - `result/prefetch-cuda_nsys_948612.nsys-rep`
  - `result/prefetch-cuda_nsys_948612.sqlite`
- Nsight Systems was available after loading `cuda/12.8`.
- Profile command: `nsys profile --trace=cuda --stats=true --force-overwrite=true --output=... ./main 10`
- Unified-memory migration summary from Nsight Systems:
  - Unified Host-to-Device: 10,719.371 MB across 209,744 transfers
  - Unified Device-to-Host: 5,368.709 MB across 30,720 transfers
  - `cudaMemPrefetchAsync` API total: 85.431347 ms across 200 calls
  - `cudaDeviceSynchronize` API total: 2,111.497987 ms across 220 calls

## CSV

- CSV written to `result/prefetch-cuda_results.csv`.

## Contradiction Check

- Baseline raw output: 40 PASS, 0 FAIL.
- Final candidate raw output: 40 PASS, 0 FAIL.
- Speedups use measured baseline job `948608`, not estimates.
- Rejected submission `948610` is not used as the final result.
- No tests were skipped, waived, shrunk, or tolerance-loosened.

## Final Conclusion

ACCEPTED PARAM_TUNE. The final candidate preserves both prefetch and no-prefetch modes. It improves the no-prefetch demand-paging path by 17.61% for `./main 10` and 10.42% for `./main 100`, while prefetch mode is unchanged to slightly improved within the measurement-equivalent band for repeat=100.

TOKENS : 53,292 used