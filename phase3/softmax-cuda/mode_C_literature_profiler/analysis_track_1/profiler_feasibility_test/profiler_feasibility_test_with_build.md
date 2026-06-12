# Profiler Feasibility Test With Build

Slurm job: `950684` on `gn1224.twcc.ai`.

Status: `AVAILABLE`

This was a profiler usability test only. It is not official timing and does not
count as Submission 2. Source code was not modified.

## Diagnostic Case

- numSlice=1000
- sliceSize=784
- impl=4
- repeat=1
- kernel filter: `softMax4`
- launch_skip=0
- launch_count=1
- official_timing_used=false

## Environment

- GPU: `Tesla V100-SXM2-32GB`
- CUDA: `V12.8.61`
- ncu: `NVIDIA (R) Nsight Compute Command Line Profiler Copyright (c) 2018-2024 NVIDIA Corporation Version 2025.1.0.0 (build 35237751) (public-release) `
- ncu return code: `0`
- actual kernel observed: `softMax4`
- build log: `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test/build.log`

## Metrics Requested

- `launch__registers_per_thread`
- `launch__shared_mem_per_block_static`
- `launch__shared_mem_per_block_dynamic`
- `sm__cycles_elapsed.avg`

Metrics seen in direct profiler stdout:

- none; direct stdout only showed attach/profiling/report lines.

Metrics recovered from `.ncu-rep` via `ncu --import --page details`:

- `launch__registers_per_thread`: register/thread 18
- `launch__shared_mem_per_block_dynamic`: Kbyte/block 3.26
- `launch__shared_mem_per_block_static`: byte/block 0
- `sm__cycles_elapsed.avg`: cycle 15663.38

## Artifacts

- stdout: `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test/raw_with_build/softmax4_metric_test.stdout`
- stderr: `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test/raw_with_build/softmax4_metric_test.stderr`
- report: `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_feasibility_test/raw_with_build/softmax4_metric_test.ncu-rep`

## Interpretation

Profiler metric collection is usable when `.ncu-rep` is imported with `ncu --import --page details`. Direct stdout is insufficient for metric extraction in this environment. Future profiler runs should save `.ncu-rep` and export/import the report to text or CSV before drawing any profiler-supported conclusions.

## Final Feasibility Conclusion

Profiler usable: YES.

Conditions:

- run through sbatch only;
- use explicit metrics or a verified section set;
- save `.ncu-rep`;
- run `ncu --import --page details` or equivalent export to extract metrics;
- do not use profiler timing as official timing.
