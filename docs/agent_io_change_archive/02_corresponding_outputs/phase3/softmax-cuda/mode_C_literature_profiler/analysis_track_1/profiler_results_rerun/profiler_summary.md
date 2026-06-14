# Profiler Summary Rerun

Profiler rerun job: `950686` on `gn1224.twcc.ai`.

This was an analysis-only profiler rerun. It does not count as Submission 2.

Execution safeguards:

- repeat_for_profiler=10
- launch_skip=2
- launch_count=1
- explicit metrics were requested
- `.ncu-rep` files were imported with `ncu --import --page details`
- official_timing_used=false for every row
- no 128/256 profiling
- profiler timing is not official speedup timing

## Previous Failed/Not-Accepted Profiler Cases

- Job `950683`: feasibility test failed before profiling because the existing binary was missing on the compute node. This was not a profiler availability result.
- Job `950678`: profiler attached and wrote `.ncu-rep` files, but the direct `--set default` stdout reported no metrics in selected sections; the result was marked PARTIAL and not accepted as profiler evidence.

## Rerun Status Table

| slice | impl | status | registers/thread | dynamic shared mem | static shared mem | waves/SM | profiler duration |
|---:|---:|---|---|---|---|---|---|
| 784 | 3 | AVAILABLE | register/thread 18 | Kbyte/block 4.16 | byte/block 0 | 156.25 | ms 1.09 |
| 784 | 4 | AVAILABLE | register/thread 18 | Kbyte/block 3.26 | byte/block 0 | 156.25 | us 938.24 |
| 1024 | 3 | AVAILABLE | register/thread 18 | Kbyte/block 5.12 | byte/block 0 | 156.25 | ms 1.29 |
| 1024 | 4 | AVAILABLE | register/thread 18 | Kbyte/block 4.22 | byte/block 0 | 156.25 | ms 1.20 |
| 2048 | 3 | AVAILABLE | register/thread 18 | Kbyte/block 9.22 | byte/block 0 | 78.12 | ms 1.66 |
| 2048 | 4 | AVAILABLE | register/thread 18 | Kbyte/block 8.32 | byte/block 0 | 78.12 | ms 1.64 |

## Interpretation Limits

- Metrics are profiler diagnostics only.
- Profiler timing is not official timing.
- Do not start Submission 2 until these artifacts are audited.
