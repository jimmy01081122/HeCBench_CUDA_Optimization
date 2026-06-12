# Profiler Plan

## Status

Profiler appears feasible after loading `cuda/12.8`, where `ncu` resolves to:

```text
/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.8/bin/ncu
```

Do not execute this plan without explicit human approval.

## Purpose

Use Nsight Compute for explanation only:

- understand why `impl=4` improves 784 and 1024 vs `impl=3`
- understand why 2048 is only measurement-equivalent
- inform whether Submission 2 should target reduction overhead, shared memory
  footprint, block size, or 2048-specific behavior

Profiler timing must not be used for official speedup.

## Proposed Comparison

Run profiler through sbatch only.

Recommended cases:

| slice | numSlice | benchmark repeat | profiler target launches | implementations |
|---:|---:|---:|---:|---|
| 784 | 100000 | 100 | 1-3 launches after warmup | `impl=3`, `impl=4` |
| 1024 | 100000 | 100 | 1-3 launches after warmup | `impl=3`, `impl=4` |
| 2048 | 50000 | 100 | 1-3 launches after warmup | `impl=3`, `impl=4` |

Each profiler run should store stdout, stderr, and report files separately from
official timing artifacts.

## Execution Safeguards

The profiler must not profile all `repeat=100` launches. Profiling every launch
would be slow, would inflate profiler overhead, and would not improve official
timing evidence.

Use one of these controls:

- kernel filtering:
  - profile `softMax3` for `impl=3` large-slice runs
  - profile `softMax4` for `impl=4` large-slice runs
- launch-count control:
  - use `--launch-skip` to skip early warmup launches
  - use `--launch-count` to capture only 1-3 kernel launches

Recommended command pattern:

```text
ncu --target-processes all \
    --kernel-name softMax3 \
    --launch-skip 5 \
    --launch-count 1 \
    --set default \
    --export <report>.ncu-rep \
    ./main <numSlice> <sliceSize> 3 100
```

Use the matching `softMax4` kernel filter for `impl=4`.

If `--kernel-name` matching is unreliable because of mangled CUDA kernel names,
the sbatch should rely on launch-skip and launch-count control and record the
exact kernel names observed in the text output.

Profiler output timing is diagnostic only. It must not be copied into
`results.csv` as official timing and must not be used to compute
`speedup_vs_impl3`.

## Metrics To Collect If Available

- achieved occupancy
- register usage
- shared memory usage
- memory throughput
- warp execution efficiency
- instruction mix
- math or special-function indicators
- scheduler or stall indicators, if available in the selected section set

## Suggested Output Directory

```text
/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_raw
```

## Suggested sbatch Shape

The profiler job should:

- `module purge`
- `module load cuda/12.8`
- build with `make ARCH=sm_70`
- run only the six profiler commands listed above
- avoid profiling all `repeat=100` launches
- use kernel filtering or launch-count control
- save report files such as `slice_784_impl_3.ncu-rep`
- save text summaries such as `slice_784_impl_3.txt`
- record environment metadata
- generate `profiler_summary.csv`

## profiler_summary.csv Schema

Create:

```text
/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_summary.csv
```

Required columns:

```text
benchmark
mode
analysis_track
profiler_job_id
sliceSize
numSlice
repeat
impl
kernel_filter
launch_skip
launch_count
profiler_status
ncu_version
hostname
gpu_name
cuda_version
report_path
stdout_path
stderr_path
achieved_occupancy
registers_per_thread
static_shared_memory_bytes
dynamic_shared_memory_bytes
memory_throughput
warp_execution_efficiency
instruction_mix_summary
math_special_function_summary
profiler_timing_ms
official_timing_used
notes
```

Required values:

- `benchmark=softmax-cuda`
- `mode=Mode_C`
- `analysis_track=analysis_track_1`
- `official_timing_used=false`
- `profiler_status=NOT_RUN`, `AVAILABLE`, `UNAVAILABLE`, `FAILED`, or `PARTIAL`

If a metric is unavailable, write `n/a` and explain in `notes`.

## Interpretation Rules

- If profiler succeeds, use metrics to form hypotheses for Submission 2.
- If profiler is unavailable or permission-limited, record
  `profiler_status=UNAVAILABLE` and continue with ablation planning.
- Do not make profiler-supported claims unless the relevant metric exists.
- Do not use profiler timing for official speedup.
- Do not decide or start Submission 2 until profiler artifacts and
  `profiler_summary.csv` have been audited by the human planner.

## Decision Use

Profiler evidence should guide whether Submission 2 is:

- a targeted ablation,
- a 2048-specific candidate,
- block-size/resource tuning,
- or skipped in favor of final confirmation.

The direction remains deferred until profiler results are reviewed and accepted.
