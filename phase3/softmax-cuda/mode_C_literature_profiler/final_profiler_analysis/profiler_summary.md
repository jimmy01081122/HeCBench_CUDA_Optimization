# Mode C Final Profiler Summary

Profiler job: `950695` on `gn1225.twcc.ai`.

This is analysis-only. Profiler timing is not official timing, and `official_timing_used=false` for every row.

Execution safeguards:

- repeat_for_profiler=10
- launch_skip=2
- launch_count=1
- kernel filters: `softMax3` for `impl=3`, `softMax4` for `impl=4`
- profiled only slices 784, 1024, and 2048
- did not profile slices 128 or 256

## Resource Table

| slice | impl | status | registers/thread | dynamic shared memory | static shared memory | waves/SM | profiler timing |
|---:|---:|---|---|---|---|---|---|
| 784 | 3 | AVAILABLE | register/thread 18 | Kbyte/block 4.16 | byte/block 0 | 156.25 | ms 1.10 |
| 784 | 4 | AVAILABLE | register/thread 18 | Kbyte/block 3.26 | byte/block 0 | 156.25 | us 938.11 |
| 1024 | 3 | AVAILABLE | register/thread 18 | Kbyte/block 5.12 | byte/block 0 | 156.25 | ms 1.30 |
| 1024 | 4 | AVAILABLE | register/thread 18 | Kbyte/block 4.22 | byte/block 0 | 156.25 | ms 1.20 |
| 2048 | 3 | AVAILABLE | register/thread 18 | Kbyte/block 9.22 | byte/block 0 | 78.12 | ms 1.66 |
| 2048 | 4 | AVAILABLE | register/thread 18 | Kbyte/block 8.32 | byte/block 0 | 78.12 | ms 1.64 |

## Impl=3 vs Impl=4 Resource Differences

- slice 784: registers/thread register/thread 18 -> register/thread 18; waves/SM 156.25 -> 156.25; dynamic shared memory decreased by 0.90 Kbyte/block in impl=4; official speedup_vs_impl3=1.135540
- slice 1024: registers/thread register/thread 18 -> register/thread 18; waves/SM 156.25 -> 156.25; dynamic shared memory decreased by 0.90 Kbyte/block in impl=4; official speedup_vs_impl3=1.048740
- slice 2048: registers/thread register/thread 18 -> register/thread 18; waves/SM 78.12 -> 78.12; dynamic shared memory decreased by 0.90 Kbyte/block in impl=4; official speedup_vs_impl3=1.008239

## Missing Profiler Evidence

- memory_throughput is unavailable in this diagnostic collection.
- warp_execution_efficiency is unavailable in this diagnostic collection.
- instruction mix and math/special-function summaries are unavailable.
- stall or scheduler summaries are unavailable.
- profiler timing is diagnostic only and is not official speedup evidence.
