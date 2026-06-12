Perform Mode C final profiler analysis for the final confirmation result.

This is analysis-only.
Do not modify source.
Do not add a new candidate.
Do not run Submission 4.
Do not compute official speedup from profiler timing.
Do not run ./main on the login node.
All profiler execution must use sbatch.

Purpose:
Analyze the final confirmation result for impl4_shape_specialized_large_reduce.

Final confirmation accepted claims:
- slice=784:  speedup_vs_impl3=1.135540
- slice=1024: speedup_vs_impl3=1.048740

Not accepted as Mode C speedup:
- slice=128
- slice=256
- slice=2048, because speedup_vs_impl3=1.008239 < 1.01

Profiler analysis questions:
1. What resource differences exist between impl=3 and impl=4 for 784?
2. What resource differences exist between impl=3 and impl=4 for 1024?
3. What resource differences exist between impl=3 and impl=4 for 2048?
4. Can profiler explain why 784/1024 improve but 2048 is measurement-equivalent?
5. What profiler evidence is missing?

Profile only:
- slice=784: impl=3 and impl=4
- slice=1024: impl=3 and impl=4
- slice=2048: impl=3 and impl=4

Do not profile:
- slice=128
- slice=256

Use profiler diagnostic repeat, not official repeat:
- repeat_for_profiler=10
- launch_skip=2 or 3
- launch_count=1

Use kernel filtering if possible:
- softMax3 for impl=3
- softMax4 for impl=4

If kernel filtering fails:
- use launch skip/count fallback
- record actual kernel names observed
- do not profile all repeat=100 launches

Profiler output directory:
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_profiler_analysis

Required outputs:
1. final_profiler_analysis/run.slurm
2. final_profiler_analysis/profiler_summary.csv
3. final_profiler_analysis/profiler_summary.md
4. final_profiler_analysis/raw/
5. final_profiler_analysis/ncu_reports/
6. final_profiler_analysis/final_profiler_interpretation.md

profiler_summary.csv columns:
benchmark,mode,stage,profiler_job_id,sliceSize,numSlice,repeat_for_profiler,impl,kernel_filter,launch_skip,launch_count,profiler_status,ncu_version,hostname,gpu_name,cuda_version,report_path,stdout_path,stderr_path,achieved_occupancy,registers_per_thread,static_shared_memory_bytes,dynamic_shared_memory_bytes,waves_per_sm,memory_throughput,warp_execution_efficiency,instruction_mix_summary,math_special_function_summary,stall_or_scheduler_summary,profiler_timing_ms,official_timing_used,notes

Required fixed values:
- benchmark=softmax-cuda
- mode=Mode_C
- stage=final_profiler_analysis
- official_timing_used=false

Allowed profiler_status:
- AVAILABLE
- PARTIAL
- UNAVAILABLE
- FAILED
- NOT_RUN

Interpretation rules:
- Do not make profiler-supported claims unless the relevant metric exists.
- If only resource allocation metrics are available, label evidence as LIMITED_PROFILER_EVIDENCE.
- Do not claim shared-memory footprint caused speedup unless supported by stronger evidence or ablation.
- Do not claim reduction-structure causality.
- Do not claim cached-exp causality.
- Do not use profiler timing for official speedup.
- Do not change final confirmation speedup values.

Final profiler interpretation must include:
1. What profiler supports.
2. What profiler does not support.
3. Whether profiler helps explain 784.
4. Whether profiler helps explain 1024.
5. Whether profiler helps explain 2048.
6. Whether further ablation would be needed.
7. Paper-safe wording.
8. Do-not-claim list.

Stop after producing profiler analysis artifacts.