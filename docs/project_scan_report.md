# Project Scan Report

## 1. Scan Metadata
- **Date/Time**: 2026-06-11T18:59:46+08:00
- **Current Directory**: `/home/a/PP`
- **Repository Root**: [PP](file:///home/a/PP)
- **Git Status**: 
  - Deleted: `phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/advice`
  - Untracked:
    - `phase3/softmax-cuda/mode_B_human_guided/final/state_report.md`
    - `phase3/softmax-cuda/mode_B_human_guided/final/temp_commition.md`
  - Recent Commit: `59990fe (HEAD -> main, origin/main)` "phase3 softmax modeB r1,2,f done" (Committed: Jun 11)
- **Command List Used**:
  - `pwd`
  - `ls -la`
  - `find`
  - `wc -l`
  - `grep`
  - `diff`
  - `view_file` (API)
  - `list_dir` (API)

## 2. Scan Scope and Output Limits Applied
- **Read-Only Scan**: Checked `/home/a/PP/phase3`, `/home/a/PP/phase2`, `/home/a/PP/evaluation_summary`, and `/home/a/PP/BASIC` directories. No writing, compilation, binary runs, or Slurm job submissions were performed.
- **Output Limits Applied**:
  - Markdown files previewed up to 40 lines (very short files viewed in full).
  - CSV files analyzed by column headers, row counts, and only the first 5 rows displayed.
  - Raw stdout/stderr logs sampled to verify PASS/FAIL status, timing metrics, and compiler/job output.

## 3. Top-Level Structure
- **Repository Root Path**: `/home/a/PP`
- **Top-Level Files**:
  - [README.md](file:///home/a/PP/README.md) (Planning context and repository guidelines)
  - [abstract.md](file:///home/a/PP/abstract.md) (Abstract of the human-AI collaborative workspace research)
  - [cuda_benchmarks.txt](file:///home/a/PP/cuda_benchmarks.txt) (Official benchmark parameters and sweep configs)
  - [phase3workflow.md](file:///home/a/PP/phase3workflow.md) (Main specification document for HeCBench Phase 3)
- **Top-Level Folders**:
  - `BASIC/`: Contains the original naive and optimized CUDA benchmark source trees.
  - `docs/`: Holds workflow planning and design documentation.
  - `evaluation_summary/`: Contains tables and CSV data summarizing Phase 2 results.
  - `phase2/`: Archive folders for Phase 2 runs, prompt templates, and reports.
  - `phase3/`: Active workspace containing `mode_A_agent_only` and `mode_B_human_guided` directories.
- **Git Repo existence**: Yes, `.git/` folder present. Commit logs show the progression from default setup, Mode A completion, Mode B robust baseline initialization, and finally Mode B `softmax-cuda` rounds and final confirmation.

## 4. Source Inventory
- **softmax-cuda**:
  - Final Source File: [final/main.cu](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/main.cu) (Exists)
  - Round 1 Source File: [rounds/round_1/src/main.cu](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/src/main.cu) (Exists)
  - Round 2 Source File: [rounds/round_2/src/main.cu](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/src/main.cu) (Exists)
  - **Implementation Variants in Source**:
    - `impl=0`: Naive reference (`softMax` kernel)
    - `impl=1`: Existing optimized baseline (`softMax2` kernel using cooperative groups tiled partitioning)
    - `impl=2`: Compound block-per-slice + shared-memory cached exponentials (`softMax3` kernel)
    - `impl=3`: Shape-aware dispatch candidate (dispatch logic in `main.cu`)
  - **Dispatch Behavior**: 
    - Slices 784, 1024, and 2048 dispatch to `softMax3` (`impl=2`) using static shared memory sized to `sizeof(float) * (sliceSize + BLOCK_SIZE)`.
    - Slices 128 and 256 dispatch to `softMax2` (`impl=1`).
    - Matches dispatch rules: `128 -> impl=1`, `256 -> impl=1`, `784 -> impl=2`, `1024 -> impl=2`, `2048 -> impl=2`.
- **topk-cuda**: No source files exist in `phase3/topk-cuda/`. The baseline compilation compiled files in [BASIC/topk-cuda/](file:///home/a/PP/BASIC/topk-cuda/).
- **shmembench-cuda**: No source files exist in `phase3/shmembench-cuda/`. The baseline compilation compiled files in [BASIC/shmembench-cuda/](file:///home/a/PP/BASIC/shmembench-cuda/).

## 5. Experiment Artifact Inventory
- **softmax-cuda (Mode B)**:
  - **Round 1**: Artifacts under [rounds/round_1/](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/). Includes `build.log`, `results.csv`, `auditor_report.csv`, and `raw/` directory containing 30 stdout/stderr logs. Candidate `impl=2` rejected because of 1 trial correctness check failure on `slice=256` and regression on `slice=128`.
  - **Round 2**: Artifacts under [rounds/round_2/](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/). Candidate `impl=3` shape-aware dispatch passed all correctness checks.
  - **Final**: Artifacts under [final/](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/). Includes final confirmation run logs, results, auditor outputs, environment metadata, and raw output files. Slurm Job 949717 completed successfully.
- **Robust Baselines (Mode B)**:
  - **softmax-cuda**: [robust_baseline/](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/robust_baseline/). Contains 3 trials per case for optimized `impl=1`.
  - **topk-cuda**: [robust_baseline/](file:///home/a/PP/phase3/topk-cuda/mode_B_human_guided/robust_baseline/). Contains 7 trials per case (run up front to address high CV).
  - **shmembench-cuda**: [robust_baseline/](file:///home/a/PP/phase3/shmembench-cuda/mode_B_human_guided/robust_baseline/). Contains block sizes 128, 256, 512, 1024.
- **Mode A (Agent-only)**:
  - **softmax-cuda**: [mode_A_agent_only/](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/). Contains baseline/final trials, results, and environment metadata.
  - **topk-cuda**: [mode_A_agent_only/](file:///home/a/PP/phase3/topk-cuda/mode_A_agent_only/).
  - **shmembench-cuda**: [mode_A_agent_only/](file:///home/a/PP/phase3/shmembench-cuda/mode_A_agent_only/).
- **Redundant Directory**:
  - `phase3/mode_A_agent_only` is a byte-for-byte duplicate of `phase3/topk-cuda/mode_A_agent_only`.

## 6. CSV Schema and Result Files

### A. Results CSV Schema (33 Fields)
`benchmark,mode,round_id,human_decision,variant,impl,dispatch_selected_impl,baseline_impl,numSlice,sliceSize,repeat,trial_id,time_ms,baseline_time_ms,speedup_vs_impl1,correctness_status,measurement_validity,speedup_claim_valid,result_type,mean_ms,min_ms,max_ms,stddev_ms,cv,raw_stdout_path,raw_stderr_path,build_log_path,slurm_job_id,hostname,gpu_name,cuda_version,profiler_status,notes`

### B. CSV Metadata Checklist
1. **softmax-cuda Mode B Final Results**:
   - Path: [final/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/results.csv)
   - Row Count: 31 (1 header + 30 data rows)
   - Column Names: Matches official 33-column schema exactly.
   - Missing/Extra Columns: None.
   - First 5 Rows Sample:
     ```csv
     benchmark,mode,round_id,human_decision,variant,impl,dispatch_selected_impl,baseline_impl,numSlice,sliceSize,repeat,trial_id,time_ms,baseline_time_ms,speedup_vs_impl1,correctness_status,measurement_validity,speedup_claim_valid,result_type,mean_ms,min_ms,max_ms,stddev_ms,cv,raw_stdout_path,raw_stderr_path,build_log_path,slurm_job_id,hostname,gpu_name,cuda_version,profiler_status,notes
     softmax-cuda,Mode_B,final,Approved,paired_impl1_baseline,1,1,1,100000,128,100,1,0.134951,0.134951,n/a,PASS,VALID,false,BASELINE,0.134869,0.134827,0.134951,0.000071,0.000525,.../final/raw/slice_128_impl_1_trial_1.stdout,.../final/raw/slice_128_impl_1_trial_1.stderr,.../final/build.log,949717,gn1288.twcc.ai,Tesla V100-SXM2-32GB,V12.8.61,NOT_RUN,paired impl=1 baseline measured in same Slurm final confirmation job
     softmax-cuda,Mode_B,final,Approved,paired_impl1_baseline,1,1,1,100000,128,100,2,0.134827,0.134827,n/a,PASS,VALID,false,BASELINE,0.134869,0.134827,0.134951,0.000071,0.000525,.../final/raw/slice_128_impl_1_trial_2.stdout,.../final/raw/slice_128_impl_1_trial_2.stderr,.../final/build.log,949717,gn1288.twcc.ai,Tesla V100-SXM2-32GB,V12.8.61,NOT_RUN,paired impl=1 baseline measured in same Slurm final confirmation job
     softmax-cuda,Mode_B,final,Approved,paired_impl1_baseline,1,1,1,100000,128,100,3,0.134830,0.134830,n/a,PASS,VALID,false,BASELINE,0.134869,0.134827,0.134951,0.000071,0.000525,.../final/raw/slice_128_impl_1_trial_3.stdout,.../final/raw/slice_128_impl_1_trial_3.stderr,.../final/build.log,949717,gn1288.twcc.ai,Tesla V100-SXM2-32GB,V12.8.61,NOT_RUN,paired impl=1 baseline measured in same Slurm final confirmation job
     softmax-cuda,Mode_B,final,Approved,impl3_shape_dispatch_impl1_small_impl2_large,3,1,1,100000,128,100,1,0.134521,0.134951,1.003197,PASS,VALID,false,MEASUREMENT_EQUIVALENT,0.134574,0.134521,0.134635,0.000057,0.000427,.../final/raw/slice_128_impl_3_trial_1.stdout,.../final/raw/slice_128_impl_3_trial_1.stderr,.../final/build.log,949717,gn1288.twcc.ai,Tesla V100-SXM2-32GB,V12.8.61,NOT_RUN,final confirmation: shape-aware dispatch selected unchanged impl=1; result_type fixed to MEASUREMENT_EQUIVALENT and no speedup claim
     ```

2. **softmax-cuda Mode B Round 2 Results**:
   - Path: [rounds/round_2/data/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/data/results.csv)
   - Row Count: 31 (1 header + 30 data rows)
   - Column Names: Matches official 33-column schema exactly.
   - Missing/Extra Columns: None.

3. **softmax-cuda Mode B Round 1 Results**:
   - Path: [rounds/round_1/data/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/data/results.csv)
   - Row Count: 31 (1 header + 30 data rows)
   - Column Names: 32 columns.
   - Missing Columns: `dispatch_selected_impl`.

4. **softmax-cuda Mode B Robust Baseline Results**:
   - Path: [robust_baseline/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/robust_baseline/results.csv)
   - Row Count: 6 (1 header + 5 baseline rows)
   - Column Names: 27 columns (Legacy schema).
   - Missing Columns: Lacks trial-level time fields.

5. **softmax-cuda Mode A Agent-only Results**:
   - Path: [mode_A_agent_only/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/results.csv)
   - Row Count: 21 (1 header + 20 data rows)
   - Column Names: 23 columns (Older schema).
   - Missing Columns: Lacks correctness_status, validity, and speedup_claim_valid.

6. **Phase 2 Evaluation Summary CSVs**:
   - [benchmark_summary_used.csv](file:///home/a/PP/evaluation_summary/data/benchmark_summary_used.csv): 26 rows. Header: `benchmark,agent,case,baseline_metric,optimized_metric,metric_unit,speedup,correctness,best_strategy,source`
   - [contradiction_check.csv](file:///home/a/PP/evaluation_summary/data/contradiction_check.csv): 6 rows. Header: `benchmark,prompt_level,issue_type,description,severity,source_file`
   - [invalid_results.csv](file:///home/a/PP/evaluation_summary/data/invalid_results.csv): 11 rows. Header: `benchmark,prompt_level,case,reason,source_file,notes`
   - [phase2_level_summary_used.csv](file:///home/a/PP/evaluation_summary/data/phase2_level_summary_used.csv): 31 rows. Header: `level,benchmark,baseline_metric,final_metric,metric_unit,speedup,correctness,result_type,status,notes,source`

## 7. Raw Output and Correctness Evidence
- **softmax-cuda (Mode B Final stdout sample)**:
  - [raw/slice_128_impl_3_trial_1.stdout](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/raw/slice_128_impl_3_trial_1.stdout) contains:
    ```
    Average kernel execution time: 0.134521 (ms)
    PASS
    ```
  - PASS/FAIL lines: `PASS`
  - Timing lines: `Average kernel execution time: 0.134521 (ms)`
- **softmax-cuda (Mode B Round 1 correctness failure sample)**:
  - [raw/slice_256_impl_2_trial_1.stdout](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/raw/slice_256_impl_2_trial_1.stdout) contains correctness failure:
    ```
    Average kernel execution time: 0.593995 (ms)
    @index 12290 host: 0.004004 device: 0.000000
    FAIL
    ```
- **topk-cuda (Mode B Baseline stdout sample)**:
  - [trial_1.txt](file:///home/a/PP/phase3/topk-cuda/mode_B_human_guided/robust_baseline/trial_1.txt) lists all 14 shapes, e.g.:
    ```
    batch size: 3072, hidden size: 3072, topk: 2048
    Average execution time of topk : 681.591736 (us)
    PASS
    ```
- **shmembench-cuda (Mode B Baseline stdout sample)**:
  - [block_256_trial_1.txt](file:///home/a/PP/phase3/shmembench-cuda/mode_B_human_guided/robust_baseline/block_256_trial_1.txt) outputs the checksum validation status (`PASS`). Diagnostic block sizes (128, 512, 1024) output checksum failure (`FAIL`).

## 8. Auditor and Contradiction Evidence
- **Auditor Script**: [self_consistency_auditor.py](file:///home/a/PP/phase3/tools/self_consistency_auditor.py)
- **Checked Files**: `contradiction_check.csv` and `auditor_report.csv` in `final/`, `round_1/data/`, `round_2/data/`, and all benchmark baseline directories.
- **Results**: All 10 checked rules are marked as **PASS** with note "no issues" (e.g. [final/contradiction_check.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/contradiction_check.csv)).
- **Summary**: Contradiction check outputs exist for each completed phase and are backed by files. No self-consistency failures were flagged.

## 9. Slurm and Build Log Evidence
- **Job/Log Summary**:

| Directory / Job | Slurm Job ID | Hostname | GPU Name | CUDA Version | Login-Node Run? | Errors? | sbatch used? |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- | :---: |
| softmax Mode B Baseline | 949640 | `gn1221.twcc.ai` | Tesla V100-SXM2-32GB | 12.8 | No | None | Yes |
| softmax Mode B Round 1 | 949687 | `gn1221.twcc.ai` | Tesla V100-SXM2-32GB | 12.8 | No | `slice=256` correctness FAIL | Yes |
| softmax Mode B Round 2 | 949703 | `gn1221.twcc.ai` | Tesla V100-SXM2-32GB | 12.8 | No | None | Yes |
| softmax Mode B Final | 949717 | `gn1228.twcc.ai` | Tesla V100-SXM2-32GB | 12.8 | No | None | Yes |
| topk Mode B Baseline | 949641 | `gn1102.twcc.ai` | Tesla V100-SXM2-32GB | 12.8 | No | None | Yes |
| shmembench Mode B Baseline | 949642 | `gn1102.twcc.ai` | Tesla V100-SXM2-32GB | 12.8 | No | diagnostic FAILs (128, 512, 1024) | Yes |

- **Official Cases Checked**:
  - `softmax-cuda` sweeps: 128, 256, 784, 1024, 2048.
  - `topk-cuda` sweeps: 14 shape configurations.
  - `shmembench-cuda` sweeps: block sizes 128, 256, 512, 1024.
- **Login-Node Check**: Environment logs and Slurm redirect structures confirm sbatch was strictly utilized. No login-node executions of GPU benchmark binaries exist in the logs.

## 10. Mode Status Matrix

| Benchmark | Mode A Status | Mode B Status | Mode C Status | Latest Accepted Round | Latest Candidate | Correctness Status | Measurement Validity | Result Type | Final Confirmation | Missing Artifacts | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :--- | :--- |
| **softmax-cuda** | `SUCCESS` | `SUCCESS` | `NOT_FOUND` | Round 2 | `impl3_shape_dispatch_impl1_small_impl2_large` | `PASS` | `VALID` | `PARAM_TUNE` / `MEASUREMENT_EQUIVALENT` | `SUCCESS` (Job 949717) | Stale top-level `agent_summary.md` and `results.csv` on local | Shape-aware dispatch candidate accepted. Large slices (784, 1024, 2048) achieved 1.34x - 1.70x speedup. |
| **topk-cuda** | `SUCCESS` | `BASELINE_ONLY` | `NOT_FOUND` | None | None | `PASS` | `VALID` / `CAUTION` | `BASELINE` | None | Optimization rounds (1-6), final confirmation, and `phase3` source code | Optimization stopped before Round 1. 7 trials run up front to suppress CV variance. |
| **shmembench-cuda** | `INVALID` | `BASELINE_ONLY` | `NOT_FOUND` | None | None | `PASS` (for 256), `FAIL` (diagnostic) | `VALID` (for 256), `DIAGNOSTIC_FAIL` | `BASELINE` | None | Optimization rounds (1-6), final confirmation, and `phase3` source code | Optimization stopped before Round 1. Diagnostic cases (128, 512, 1024) failed correctness. |

## 11. softmax-cuda Detailed Status
1. **Mode A Status & Artifacts**: Completed successfully (`SUCCESS`). Stable baseline comparison established under Job `949514`. Artifacts located in [mode_A_agent_only/](file:///home/a/PP/phase3/softmax-cuda/mode_A_agent_only/).
2. **Mode B Round 1 Artifacts & Conclusion**: Completed. Candidate `impl2` (compound block cached-exp) was rejected because `slice=256` had a correctness check failure (trial 1) and `slice=128` regressed.
3. **Mode B Round 2 Artifacts & Conclusion**: Completed. Candidate `impl3` (shape-aware dispatch) passed all correctness checks. 
4. **Final Confirmation**: Completed. Slurm Job 949717 on `gn1288.twcc.ai`.
5. **Final Confirmation Pass**: Yes, labeled `SUCCESS` in the final summary.
6. **impl=3 exists in source**: Yes, verified in [final/main.cu](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/main.cu).
7. **Source Dispatch Check**: Confirmed. `sliceSize == 784 \|\| sliceSize == 1024 \|\| sliceSize == 2048` dispatches to cached-exp `softMax3` (`impl=2`), whereas small slices dispatch to existing optimized `softMax2` (`impl=1`).
8. **Evidence of Mode C**: None.
9. **Missing Files**:
   - The top-level files [mode_B_human_guided/agent_summary.md](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/agent_summary.md) and [mode_B_human_guided/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/results.csv) on local are stale. They only display robust baseline info ("stopped before Round 1") and were not updated with the server's final run outputs.
   - [reports/mode_B_report.md](file:///home/a/PP/phase3/reports/mode_B_report.md) is stale; it only reports the initial robust baseline status before any rounds were run.

## 12. topk-cuda Detailed Status
- **Source Paths**: None exist under `phase3/topk-cuda/`. Active files reside in [BASIC/topk-cuda/](file:///home/a/PP/BASIC/topk-cuda/).
- **Mode A Artifacts**: Results and summary logs completed under Job `949515`.
- **Robust Baseline Remeasurement**: Completed under Job `949641`. Seven trials were run up front to suppress CV variance.
- **High-CV Issues Documented**: Yes. [reports/mode_A_report.md](file:///home/a/PP/phase3/reports/mode_A_report.md) identifies CV variance of up to 54.6% in Mode A as a pseudo-speedup trap.
- **Mode B Artifacts**: Baseline only (stopped before Round 1).
- **Mode C Artifacts**: None.
- **Final Confirmation Artifacts**: None.
- **Missing Files**: Optimization rounds (1-6), plan, final confirmation, and `phase3` source code files.

## 13. shmembench-cuda Detailed Status
- **Source Paths**: None exist under `phase3/shmembench-cuda/`. Active files reside in [BASIC/shmembench-cuda/](file:///home/a/PP/BASIC/shmembench-cuda/).
- **Mode A Artifacts**: Completed under Job `949516` (labeled `INVALID` due to required sweep correctness failure).
- **Mode B Artifacts**: Baseline only (stopped before Round 1).
- **Mode C Artifacts**: None.
- **Final Confirmation Artifacts**: None.
- **Official Validated Comparison**: Yes, `block_size=256`, `variant=original` exists and passed correctness check.
- **Diagnostic Sweeps**: Yes, block sizes 128, 512, 1024 are present and labeled `DIAGNOSTIC_FAIL`.
- **Missing Files**: Optimization rounds, plan, final confirmation, and `phase3` source code files.

## 14. Mode C Artifact Detection
- **Mode C Status**: `NOT_FOUND` / `NOT_STARTED`. No directories or files related to Mode C exist in the repository.

## 15. Missing Artifacts
- **softmax-cuda**:
  - Synced top-level [mode_B_human_guided/agent_summary.md](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/agent_summary.md) and [mode_B_human_guided/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/results.csv) reflecting Round 2 and final confirmation outputs.
- **topk-cuda & shmembench-cuda**:
  - Source code files (`main.cu`, `Makefile`) inside `phase3` directories.
  - Mode B optimization rounds (1-6) and final confirmation files.
  - Plans and intervention logs for Mode B rounds.

## 16. Risks and Inconsistencies
- **Stale Summaries on Local**: The top-level [reports/mode_B_report.md](file:///home/a/PP/phase3/reports/mode_B_report.md) and benchmark-level `agent_summary.md` files are out-of-sync with the rounds executed for `softmax-cuda`. They indicate optimization is stopped before Round 1.
- **Duplicate Directories**: `phase3/mode_A_agent_only` is a redundant, identical copy of `phase3/topk-cuda/mode_A_agent_only`.
- **Legacy CSV Schema Usage**: All baseline results CSVs and Mode A results CSVs use legacy 23-column or 27-column formats rather than the official 33-column Phase 3 schema.
- **Correctness Swings in topk-cuda**: The high CV of ~54.6% in Mode A baseline highlights the risk of claiming pseudo-speedups if the number of trials is not increased (resolved in Mode B baseline by running 7 trials).

## 17. Files Recommended for Human Review
1. **To Send to the Main Planner**:
   - [phase3/reports/mode_A_report.md](file:///home/a/PP/phase3/reports/mode_A_report.md): Contains the comprehensive report of the Mode A experiments, CV variance risks, and benchmark limitations.
   - [phase3/softmax-cuda/mode_B_human_guided/final/round_summary.md](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/round_summary.md): Details the final confirmation results for `softmax-cuda` under Mode B.
   - [phase3/softmax-cuda/mode_B_human_guided/final/results.csv](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/final/results.csv): Contains the 30 rows of final confirmation trials under the 33-column schema.
2. **Stale/Contradictory Files**:
   - [phase3/softmax-cuda/mode_B_human_guided/agent_summary.md](file:///home/a/PP/phase3/softmax-cuda/mode_B_human_guided/agent_summary.md): Contradicts completed rounds; needs synchronization.
   - [reports/mode_B_report.md](file:///home/a/PP/phase3/reports/mode_B_report.md): Stale summary.

## 18. Do-Not-Claim List
- **Do not claim** that Mode C is started or completed.
- **Do not claim** that shape-aware dispatch is a universal kernel optimization (it dispatches small slices to existing optimized baseline).
- **Do not claim** any speedup for `slice=128` or `256` under candidate `impl3` (they dispatch to baseline `impl=1` and are measurement-equivalent).
- **Do not claim** that `topk-cuda` or `shmembench-cuda` Mode B optimizations have been run (stopped at robust baseline).
- **Do not claim** that the 1.4x speedup for `topk-cuda` in Mode A is a valid optimization (it was a pseudo-speedup due to high CV in the baseline).
- **Do not claim** profiler-supported bottlenecks for Mode B (profiler was not run).

## 19. Summary for Research Planner
- **softmax-cuda**: Mode B is a complete success up to final confirmation. The shape-aware dispatch candidate `impl3` resolves the correctness failure in `slice=256` and regression in `slice=128` by dispatching small slices to `impl=1`, while large slices achieve ~1.34x - 1.70x speedups.
- **topk-cuda & shmembench-cuda**: Mode B is set up with correct robust baselines (7 trials run for topk to suppress CV, diagnostic cases isolated for shmembench), but optimization rounds have not yet been started.
- **Action Required**: Sync server-updated files to local, clean up duplicate directories, and proceed to Mode B Round 1 for `topk-cuda` and `shmembench-cuda`.
