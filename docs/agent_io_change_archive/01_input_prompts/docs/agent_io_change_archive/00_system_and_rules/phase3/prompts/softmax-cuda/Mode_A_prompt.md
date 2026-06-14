# Phase 3 Mode A Prompt: softmax-cuda (Agent-Only Baseline)

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as a strict experimental protocol under Phase 3 (Mode A: Agent-only baseline).

## Prompt Metadata
- benchmark: softmax-cuda
- canonical_name: softmax-cuda
- benchmark_category: AI Primitive / Kernel Optimization
- prompt_level: P3
- experimental_mode: Mode A (Agent-only baseline, no human intervention)
- target_agent: server-side coding agent
- submission_limit: 5
- baseline_counts_as_submission: false
- required_gpus: 1
- requires_mpi: false
- requires_nccl: false
- expected_metric: avg_ms by slice size and implementation
- correctness_required: true

## Mode A Definition
Mode A is an agent-only baseline and optimization workflow. There is no interactive human approval during execution.

The agent must still document its plan before every optimization submission, including:
- modification
- hypothesis
- expected improvement
- validation target

However, the agent must not wait for human feedback. All `human_decision` fields must be set to `None_Agent_Only`.

## Environment
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
```bash
module purge
module load cuda/12.8
```

## Paths
- benchmark_path: `/home/r14525078/HeCBench/src/softmax-cuda`
- result_path: `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_A_agent_only`

## Result Directory Layout
Create the following directories under `result_path`:
```text
baseline/
rounds/
rounds/round_1/
rounds/round_2/
rounds/round_3/
rounds/round_4/
rounds/round_5/
final/
logs/
```
All raw `.out`, `.err`, `.txt`, CSV, and summaries must be stored under these directories.

## Benchmark-Specific Requirements
- Benchmark naive and optimized implementations; preserve full slice-size sweep.
- Dispatch policy must depend on slice size; do not assume one kernel dominates all shapes.
- Shapes to test: slice = 128, 256, 784, 1024, 2048.
- Grid parameters: batch = 100000 for slice <= 1024; batch = 50000 for slice = 2048; repeat = 100.
- Preserve checksum or reference validation. If the benchmark has no explicit correctness check, mark correctness as `NOT_PROVIDED`; do not fabricate PASS.
- Because `correctness_required=true`, `correctness=NOT_PROVIDED` cannot be treated as SUCCESS. If the original benchmark lacks explicit correctness validation and the agent cannot add non-invasive validation, the final conclusion must be PARTIAL_SUCCESS or INVALID, not SUCCESS.
- Official baseline variant is `original`.
- The likely result type is `KERNEL_OPT` or `PARAM_TUNE` or `MEASUREMENT_EQUIVALENT`, but the final result type must be determined from actual measured evidence.
- Profiler must be attempted for the final accepted candidate. If `ncu` is unavailable or permission denied, set `profiler_available=False` and record the reason.

## Official Sweep Definition
The official comparison sweep for softmax-cuda is:
- variant=original
- slice_size = 128, 256, 784, 1024, 2048
- batch_size = 100000 for slice_size <= 1024; batch_size = 50000 for slice_size = 2048; repeat = 100

Optional variants such as custom dispatch policies or thread configurations are optimization variants. They must be labeled explicitly and must not replace the official original baseline rows.
Optional variants may use a subset of shapes for diagnosis, but the final accepted candidate must be compared against the original official sweep. Optional variants must never replace the original baseline rows.

Official speedup must be computed against the best valid `variant=original` baseline unless otherwise justified in `agent_summary.md`.

## Server Run Instructions
Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.
Recommended workflow:
```bash
cd /home/r14525078/HeCBench/src/softmax-cuda
mkdir -p /home/r14525078/HeCBench/phase3/softmax-cuda/mode_A_agent_only
module purge
module load cuda/12.8
sbatch run_softmax_cuda.slurm
```

The following commands must be placed inside the Slurm script. Do not execute them directly on the login node:
```bash
make clean || true
make ARCH=sm_70
./main 100000 784 0 100
./main 100000 784 1 100
```
If `run_softmax_cuda.slurm` does not exist, create it. The script must request 1 GPU, print environment metadata, build the benchmark, run the commands, and tee output to a result `.txt` file under `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_A_agent_only/baseline`.

## Hard Rules
1. Baseline does not count toward submission limit.
2. After baseline, at most 5 optimization sbatch submissions.
3. Before each submission, state: modification, hypothesis, expected improvement, and validation target.
4. After each submission, read and summarize: result `.out`, `.err`, `.txt`, and CSV.
5. Do not delete correctness checks. Do not loosen tolerance or modify CPU verification to match GPU output.
6. Do not execute GPU benchmark binaries directly on the login node. All correctness validation, timing, profiling, MPI/NCCL, and GPU execution must be done via sbatch. 禁止在 login node 直接執行 ./main 或 any GPU benchmark binary。所有 correctness validation、timing、profiler、MPI/NCCL/GPU execution 必須透過 sbatch 進行。若需要快速檢查 usage，只能讀 source 或使用 --help 類無 GPU 操作，不得啟動 GPU kernel。
7. Official sweep cases must remain fixed and must not be reduced, skipped, or replaced: slice = 128, 256, 784, 1024, 2048. Do not remove slow or failing cases to improve average speedup.
8. If correctness FAIL, the result is invalid. If only size 0 PASS, the result is invalid.
9. If baseline is invalid, speedup must be n/a.
10. If improvement is below 1%, classify as MEASUREMENT_EQUIVALENT or marginal, not significant.
11. ENV_FIX, MEASURE_FIX, and TOPOLOGY_MEASURE must not be described as kernel-level optimization.
12. Nsight Compute / ncu is optional but must be attempted. If ncu is unavailable or permission denied: set profiler_available=False in CSV, record the reason in profiler_summary.md, and continue with correctness-gated timing measurements/variance analysis. Do not stop the experiment or repeatedly retry.
13. Run the self-consistency auditor after Mode A completes. If contradictions are found, mark the result invalid until corrected. Do not manually override contradiction_check.csv without documenting the reason.
14. Because `correctness_required=true`, `correctness=NOT_PROVIDED` cannot be treated as SUCCESS. If the original benchmark lacks explicit correctness validation and the agent cannot add non-invasive validation, the final conclusion must be PARTIAL_SUCCESS or INVALID, not SUCCESS.
15. Preserve full raw output.

## Required Result Types
Classify every attempt using one of:
`BASELINE`, `KERNEL_OPT`, `PARAM_TUNE`, `MEASURE_FIX`, `BUILD_FIX`, `ENV_FIX`, `CORRECT_FIX`, `TOPOLOGY_MEASURE`, `NO_EFFECT`, `REGRESSION`, `MEASUREMENT_EQUIVALENT`, `INVALID`.

## CSV Result Schema
Generate or maintain a CSV under result path aligning with the Phase 3 result schema:
`benchmark,mode,round,job_id,node,case,variant,metric_name,metric_value,unit,baseline_metric,speedup,correctness,status,result_type,mean,min,max,stddev,cv,profiler_available,human_decision,notes`

Baseline CSV rows must use:
- `mode=Mode_A`
- `round=baseline`
- `human_decision=None_Agent_Only`
- `baseline_metric=n/a`
- `speedup=n/a`
- `result_type=BASELINE`

Optimization rows use:
- `round=1,2,3,...` (corresponding to the optimization round index)

Final confirmation rows use:
- `round=final`

If profiler is unavailable/fails, set `profiler_available` to `False` and note the exact error (e.g., `ncu unavailable: command not found` or `ncu unavailable: permission denied`) in the `notes` column. Do not omit human_decision; set it strictly to None_Agent_Only.

## Variance / Repeated Trials
For official speedup calculation, both the baseline and final accepted candidate must have at least 3 trials. If the baseline was initially measured with only one trial, run a repeated baseline confirmation before reporting final speedup. Report mean, min, max, stddev, and CV (coefficient of variation).

## Profiler / Measurement Notes
Profiler must be attempted for the final accepted candidate. If `ncu` is unavailable or permission denied, set `profiler_available=False` in CSV, document the reason in `profiler_summary.md` (and final report limitations) and continue. Required profiler focus: occupancy, expf instruction reduction, shared memory, and memory throughput.
Attempt Nsight Compute profiling at most once for the final accepted candidate. If it fails due to missing command, permission denial, or unavailable hardware counters, do not spend additional submissions retrying profiler. Record the failure and continue.

## Contradiction Check
Before writing the final conclusion, run or verify with the self-consistency auditor.
If `self_consistency_auditor.py` already exists, use it. If it does not exist, create a minimal auditor or manually generate `contradiction_check.csv` with these checks:
1. correctness != PASS → status must not be SUCCESS
2. baseline invalid → speedup must be n/a
3. speedup < 1% → result_type must be MEASUREMENT_EQUIVALENT or marginal
4. profiler_available=False → no profiler-supported claim may appear
5. rejected attempts must not be used as final result
6. missing official cases → status must be PARTIAL_SUCCESS, not SUCCESS

## Final Output
Write `agent_summary.md` in the result path with: environment, prompt level and submission limit, baseline result, submission history, accepted/rejected attempts, correctness table, performance table, variance statistics, profiler/measurement notes, result type classification, and final conclusion label (`SUCCESS`, `PARTIAL_SUCCESS`, `INVALID`, `BLOCKED`).

Conclusion label definitions:
- SUCCESS:
  All official original sweep cases pass correctness, final CSV exists, contradiction_check.csv exists, and final result is valid.
- PARTIAL_SUCCESS:
  The benchmark runs and produces valid timing, but profiler is unavailable, optional variants fail, or some non-critical metadata is incomplete.
- INVALID:
  Correctness fails, baseline is invalid, official cases are missing without explanation, or speedup is computed from invalid data.
- BLOCKED:
  Build, Slurm allocation, environment, or runtime failures prevent valid measurement.

## Acceptance Criteria
Mode A is acceptable only if:
1. Baseline was executed through sbatch.
2. No GPU benchmark binary was executed on the login node.
3. Baseline CSV row exists.
4. At least one final CSV row exists.
5. All CSV rows follow the Phase 3 schema.
6. `human_decision=None_Agent_Only` for every row.
7. All official block-size sweep cases are either PASS, FAIL, or explicitly marked invalid; none are silently skipped.
8. Correctness FAIL results are not used in final performance claims.
9. If profiler is unavailable, `profiler_available=False` is recorded with reason.
10. `contradiction_check.csv` exists.
11. `agent_summary.md` exists.
12. Final conclusion is one of:
    - SUCCESS
    - PARTIAL_SUCCESS
    - INVALID
    - BLOCKED
