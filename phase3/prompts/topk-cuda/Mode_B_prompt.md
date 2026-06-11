# Phase 3 Mode B Prompt: topk-cuda (Human-Guided Optimization)

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as a strict experimental protocol under Phase 3 (Mode B: Human-in-the-loop Guided Optimization).

## Prompt Metadata
- benchmark: topk-cuda
- canonical_name: topk-cuda
- benchmark_category: AI Primitive / Kernel Optimization
- prompt_level: P3
- experimental_mode: Mode B (Human-in-the-loop guided optimization, with interactive checkpoints)
- target_agent: server-side coding agent
- submission_limit: 6 optimization rounds + 1 final confirmation
- baseline_counts_as_submission: false
- correctness_required: true

## Mode B Definition
Mode B is a human-guided baseline and optimization workflow. Every optimization round requires interactive human approval before submitting execution.

The agent must document its plan before every optimization submission in `plan.md`, including:
- bottleneck hypothesis
- proposed modification
- expected improvement
- risk
- validation target

The agent **MUST NOT** proceed with sbatch execution until the human reviewer reviews and marks the plan as `Approved` in `decision_log.md` and `human_intervention_log.md`. If the reviewer requests modifications (`Revise`) or rejects the direction (`Rejected`), the agent must adapt or rollback accordingly.

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

## Paths & Metadata Inputs
- benchmark_path: `/home/r14525078/HeCBench/src/topk-cuda`
- result_path: `/home/r14525078/HeCBench/phase3/topk-cuda/mode_B_human_guided`
- official_sweeps_path: `/home/r14525078/HeCBench/phase3/metadata/official_sweeps.yaml`
- result_schema_path: `/home/r14525078/HeCBench/phase3/metadata/result_schema.csv`
- auditor_script_path: `/home/r14525078/HeCBench/phase3/tools/self_consistency_auditor.py`
- mode_A_report_path: `/home/r14525078/HeCBench/phase3/reports/mode_A_report.md`

## Result Directory Layout
Create the following directories under `result_path`:
```text
robust_baseline/
rounds/
rounds/round_1/
rounds/round_2/
rounds/round_3/
rounds/round_4/
rounds/round_5/
rounds/round_6/
final/
logs/
```
All plans, patches, raw logs, CSV rows, and auditor reports must be stored under these directories per round.

## Benchmark-Specific Requirements & Official Sweep
- **Baseline definition**:
  - `baseline = Phase 3 official topk implementation before Mode B modification`.
- **Required Sweep cases**:
  - hidden_size = 3072, 4096, 8192, 16384, 32768, 65536, 131072.
  - topk = 1024, 2048 (total 14 cases).
  - **trials >= 5 (or 7 trials if CV > 15% or remeasured)**.
- **Tuning Directions**:
  - **Warmup policy**: First run a warm-up phase to avoid cold-start overheads and reduce measurement noise.
  - **Repeated baseline**: Ensure CV is checked. If CV is high (>15%), remeasure or label it as `NOISY`.
  - **Noisy case policy**: Speedup comparison is only valid for cases marked `VALID` or `CAUTION` in `measurement_validity`. High-noise cases must be excluded from speedup claims until remeasured.
  - **Optimization targets**:
    - CUB workspace reuse.
    - Eliminating repeated `cudaMalloc`/`cudaFree` allocations in the timed loop.
    - Sweep block sizes: 256, 512, 1024.
    - Shape-aware dispatch.
    - Reducing synchronization barriers.
    - Fine-tuning strategies per hidden_size/topk pair.

## Server Run Instructions
Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.
The following commands must be placed inside the Slurm script. Do not execute them directly on the login node:
```bash
make clean || true
make ARCH=sm_70
./main 3072 100
```
If `run_topk_cuda.slurm` does not exist, create it. The script must request 1 GPU, print environment metadata, build the benchmark, run the commands, and tee output to a result `.txt` file under `robust_baseline/` or `rounds/round_N/`.

## Hard Rules
1. Robust Baseline must be established first. **Both baseline and final accepted candidates must have at least 5 trials.**
2. After baseline, at most 6 optimization rounds.
3. Every round must start with writing a plan in `plan.md` and waiting for human approval. **Without human approval, sbatch execution is prohibited.**
4. After sbatch execution, read and summarize results, and run `self_consistency_auditor.py` on the result CSV row.
5. If correctness is not `PASS` for all 14 cases, the result is invalid. If CV > 15%, the result is marked `NOISY` and cannot be claimed as a valid speedup.
6. If auditor checks fail, the result cannot be promoted to the final phase.
7. Preserve full raw output.

## Required Result Types
Classify every attempt using one of:
`BASELINE`, `KERNEL_OPT`, `PARAM_TUNE`, `MEASURE_FIX`, `BUILD_FIX`, `ENV_FIX`, `CORRECT_FIX`, `TOPOLOGY_MEASURE`, `NO_EFFECT`, `REGRESSION`, `MEASUREMENT_EQUIVALENT`, `INVALID`.

## CSV Result Schema & Mapping
CSV rows must align with `result_schema.csv`:
`benchmark,mode,round,job_id,node,case,variant,metric_name,metric_value,unit,baseline_metric,speedup,correctness,status,result_type,mean,min,max,stddev,cv,profiler_available,human_decision,correctness_status,measurement_validity,speedup_claim_valid,require_remeasurement,notes`

Mode B CSV mapping rules:
- `mode` must be `Mode_B`.
- `round` must be `baseline` (for baseline), `1,2,3...` (for rounds), or `final` (for final confirmation).
- `human_decision` must be set to `Approved`, `Rejected`, `Revise`, or `Stop` based on the reviewer's decision for that round.

## Variance / Repeated Trials
For official speedup calculation, both the baseline and final accepted candidate must have at least 5 trials (or 7 if CV > 15%). Report mean, min, max, stddev, and CV.

## Profiler / Measurement Notes
Profiler must be attempted for the final accepted candidate. If `ncu` is unavailable or permission denied, set `profiler_available=False` in CSV, document the reason in `profiler_summary.md` (and final report limitations) and continue. Required profiler focus: occupancy, register pressure, memory throughput, and CUB temporary storage.
Attempt Nsight Compute profiling at most once for the final accepted candidate. If it fails due to missing command, permission denial, or unavailable hardware counters, do not spend additional submissions retrying profiler. Record the failure and continue.

## Contradiction Check (Auditor Rules)
After each round, run `self_consistency_auditor.py` which enforces:
- Rule V1: If CV > 15%, set `measurement_validity=NOISY`.
- Rule V2: If CV > 15% and speedup > 1.05, set `speedup_claim_valid=false`.
- Rule V2b: If speedup < 1.01, set `result_type=MEASUREMENT_EQUIVALENT` and `speedup_claim_valid=false`.
- Rule V2c: If no source code change occurred, `speedup_claim_valid` must be `false` unless explicitly labeled `BASELINE_REMEASUREMENT` or `BASELINE_COMPARISON`.
- Rule V3: softmax-cuda impl0_to_impl1 comparison must be `BASELINE_COMPARISON`, not `AGENT_OPT`.
- Rule V4: shmembench-cuda non-256 block size FAIL must be labeled `DIAGNOSTIC_FAIL`.
- Rule V5: If official validated baseline is missing, `speedup=n/a`.
- Rule V6: If optional variant replaces original baseline, mark `INVALID`.
- Rule V7: If correctness_status != PASS, then `speedup` must be `n/a` and `speedup_claim_valid=false`.
- Rule V8: If no source change occurred, `result_type` cannot be `KERNEL_OPT`.
- Rule V9: If speedup < 1.01, set `result_type=MEASUREMENT_EQUIVALENT` and `speedup_claim_valid=false`.
- Rule V10: If speedup < 1.0, set `result_type=REGRESSION` and `speedup_claim_valid=false`.

## Round Artifacts
Each round must produce under `rounds/round_N/`:
- `plan.md` (hypothesis, proposed modification, expected improvement, risk, validation target)
- `patch_summary.md` (code diff)
- `run.slurm` (Slurm run script)
- `result.out` and `result.err` (raw output files)
- `results.csv` (CSV data row)
- `auditor_report.csv` (auditor output file)
- `round_summary.md` (summary of outcomes and metrics)

## Final Output
Write `agent_summary.md` in the result path containing: robust baseline results, round-by-round logs, accepted/rejected modifications, performance table, variance statistics, auditor results, and final conclusion label (`SUCCESS`, `PARTIAL_SUCCESS`, `INVALID`, `BLOCKED`).

Conclusion label definitions:
- SUCCESS:
  All official original sweep cases pass correctness, final CSV exists, contradiction_check.csv exists, and final result is valid.
- PARTIAL_SUCCESS:
  The benchmark runs and produces valid timing, but profiler is unavailable, optional variants fail, or some non-critical metadata is incomplete.
- INVALID:
  Correctness fails, baseline is invalid, official cases are missing without explanation, or speedup is computed from invalid data.
- BLOCKED:
  Build, Slurm allocation, environment, or runtime failures prevent valid measurement.
