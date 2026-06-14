# Profiler Summary

Profiler job: `950678` on `gn1225.twcc.ai`.

This was an analysis-only profiler run. It does not count as Submission 2.

Execution safeguards:

- repeat_for_profiler=10
- launch_skip=2
- launch_count=1
- official_timing_used=false for every row
- no 128/256 profiling
- profiler timing is not official speedup timing

## Status Table

| slice | impl | kernel_filter | profiler_status | report | notes |
|---:|---:|---|---|---|---|
| 784 | 3 | softMax3 | PARTIAL | /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_raw/slice_784_impl_3.ncu-rep | filtered; actual_kernel=softMax3; ncu report generated but no metrics to collect found in selected sections; diagnostic_app_avg_ms_under_profiler=53.282917; not official timing |
| 784 | 4 | softMax4 | PARTIAL | /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_raw/slice_784_impl_4.ncu-rep | filtered; actual_kernel=softMax4; ncu report generated but no metrics to collect found in selected sections; diagnostic_app_avg_ms_under_profiler=49.694729; not official timing |
| 1024 | 3 | softMax3 | PARTIAL | /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_raw/slice_1024_impl_3.ncu-rep | filtered; actual_kernel=softMax3; ncu report generated but no metrics to collect found in selected sections; diagnostic_app_avg_ms_under_profiler=50.420567; not official timing |
| 1024 | 4 | softMax4 | PARTIAL | /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_raw/slice_1024_impl_4.ncu-rep | filtered; actual_kernel=softMax4; ncu report generated but no metrics to collect found in selected sections; diagnostic_app_avg_ms_under_profiler=48.959095; not official timing |
| 2048 | 3 | softMax3 | PARTIAL | /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_raw/slice_2048_impl_3.ncu-rep | filtered; actual_kernel=softMax3; ncu report generated but no metrics to collect found in selected sections; diagnostic_app_avg_ms_under_profiler=52.085854; not official timing |
| 2048 | 4 | softMax4 | PARTIAL | /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_raw/slice_2048_impl_4.ncu-rep | filtered; actual_kernel=softMax4; ncu report generated but no metrics to collect found in selected sections; diagnostic_app_avg_ms_under_profiler=49.304703; not official timing |

## Interpretation Limits

- Nsight Compute attached and produced `.ncu-rep` files for all six requested large-slice runs.
- Kernel filtering worked for `softMax3` and `softMax4`.
- The selected `--set default` sections emitted `No metrics to collect found in sections`; therefore metric fields are `n/a` and profiler evidence is PARTIAL rather than a usable bottleneck explanation.
- Use profiler artifacts for audit only unless metrics are recovered by an approved follow-up import or profiler rerun.
- Do not use profiler timing for official speedup.
- Do not start Submission 2 until these artifacts are audited.

---
在 Mode C Submission 1 之後，本研究嘗試進行 profiler-only analysis，以比較 large slices 上 Mode B `impl=3` 與 Mode C `impl=4` 的差異。Nsight Compute 能夠附加到目標 kernel，且六組 `impl=3` / `impl=4` large-slice profiling run 均產生 `.ncu-rep` 檔案。然而，所選 profiler sections 未產生所需的效能指標，包括 occupancy、memory throughput、warp execution behavior、instruction mix 或 resource-usage indicators。因此，此 profiler run 應分類為 `PARTIAL`，不能支撐 profiler-based bottleneck conclusion。Profiler timing 僅作診斷用途，未用於 official speedup claim。基於此不完整的 profiler evidence，下一步較合理的方向是 reduction-structure ablation，而不是 profiler-informed optimization。