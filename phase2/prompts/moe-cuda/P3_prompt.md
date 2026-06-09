# P3 Strong Prompt: moe-cuda

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as an experimental protocol, not a casual request.

## Prompt Metadata

- benchmark: moe-cuda
- canonical_name: moe-cuda
- benchmark_category: moe_inference
- prompt_level: P3
- target_agent: server-side coding agent
- submission_limit: 3
- baseline_counts_as_submission: false
- required_gpus: 1
- requires_mpi: false
- requires_nccl: false
- expected_metric: Average execution time of kernels us for topk 1/2/4/8
- correctness_required: true

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

- benchmark_path: `/home/r14525078/HeCBench/src/moe-cuda`
- result_path: `/home/r14525078/HeCBench/src/moe-cuda/result`

## Benchmark-Specific Requirements

- Fixed cases: 32768 tokens, 384 experts, topk 1/2/4/8, repeat 1000.
- Hybrid dispatch should be evaluated per top-k; full fusion can regress for topk=8.
- profiler requirement: Collect launch count, occupancy, global-memory traffic, and shared-memory usage for best candidate.
- expected primary result type: KERNEL_OPT

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/HeCBench/src/moe-cuda
mkdir -p /home/r14525078/HeCBench/src/moe-cuda/result
module purge
module load cuda/12.8
sbatch run_moe_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/HeCBench/src/moe-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
./main 32768 384 1 1000
./main 32768 384 2 1000
./main 32768 384 4 1000
./main 32768 384 8 1000
```

If `run_moe_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Hard Rules

1. Baseline does not count toward submission limit.
2. After baseline, at most 3 optimization sbatch submissions.
3. Before each submission, state:
   - modification
   - hypothesis
   - expected improvement
   - validation target
4. After each submission, read and summarize:
   - result `.out`
   - result `.err`
   - result `.txt`
   - generated CSV, if any
5. Do not delete correctness checks.
6. Do not loosen tolerance.
7. Do not modify CPU/reference validation to match GPU output.
8. Do not shrink input or skip official cases to fake speedup.
9. If correctness FAIL, the metric is invalid.
10. If only size 0 PASS, the result is invalid.
11. If output is missing, stderr has fatal error, or a case is waived/skipped, the result is invalid.
12. If improvement is below 1%, mark it as `MEASUREMENT_EQUIVALENT`, not a real speedup.
13. Do not claim all tests PASS if any case failed.
14. Preserve full raw output.

## Baseline Requirements

Run baseline before source modification.

Save:

- `result/moe-cuda_<jobid>.out`
- `result/moe-cuda_<jobid>.err`
- `result/moe-cuda_result_<jobid>.txt`

Record:

- job_id
- node
- CUDA_VISIBLE_DEVICES
- `nvcc --version`
- loaded modules
- benchmark command
- correctness
- primary metric
- full case list tested

If baseline fails, classify the failure:

- BUILD_FAIL
- COMPILE_FAIL
- RUNTIME_FAIL
- ENV_FAIL
- CORRECTNESS_FAIL
- NO_VALID_NONZERO_RESULT
- NO_PERFORMANCE_METRIC
- TIMEOUT

If baseline has no valid metric, do not compute speedup. Optimize toward correctness or measurement recovery first.

## Optimization Submission Rules

For each optimization submission:

1. Create backups before editing changed files using `.bak_agent`.
2. State the hypothesis before `sbatch`.
3. Submit exactly one sbatch job for that attempt.
4. Read out/err/result immediately after completion.
5. Classify result as accepted or rejected.
6. If rejected, preserve the reason and do not use its metric in final speedup.

## Correctness Gate

A result is valid only when correctness is PASS for every required case.

Invalid cases:

- correctness FAIL
- only size 0 PASS
- skipped or waived tests
- output missing
- stderr fatal error
- benchmark semantics changed
- input size/repeat reduced for final result

## Required Result Types

Classify every attempt using one of:

- KERNEL_OPT
- PARAM_TUNE
- MEASURE_FIX
- BUILD_FIX
- ENV_FIX
- CORRECT_FIX
- TOPOLOGY_MEASURE
- NO_EFFECT
- REGRESSION
- MEASUREMENT_EQUIVALENT

## CSV Result Schema

Generate or maintain a CSV under result path with at least:

```csv
benchmark,job_id,node,prompt_level,submission_index,variant,case,metric_name,metric_value,metric_unit,correctness,status,result_type,accepted,reject_reason,notes
```

If the benchmark naturally has multiple dimensions, encode them in `case`, for example `size_bytes=...`, `topk=...`, `slice=...`, or `num_gpus=...`.

## Variance / Repeated Trials

Final accepted candidate must include at least 3 trials when feasible. Report:

- mean
- min
- max
- stddev or coefficient of variation

If trial count cannot be increased due to submission or queue constraints, explain why in `agent_summary.md`.

## Profiler / Measurement Notes

For the final accepted candidate, collect profiler data if available without exceeding practical limits. If profiler cannot be run, include a measurement note explaining why. Required or preferred profiler focus:

Collect launch count, occupancy, global-memory traffic, and shared-memory usage for best candidate.

## Contradiction Check

Before writing the final conclusion:

1. Count PASS and FAIL cases from raw output.
2. Check that summary text matches those counts.
3. Check that speedup uses real measured baseline, not estimated baseline.
4. Check that rejected attempts are not used as final results.
5. If any case failed, do not write "all tests PASS".

## Final Output

Write `agent_summary.md` in the result path with:

- environment
- prompt level and submission limit
- baseline result
- submission history
- accepted/rejected attempts
- correctness table
- performance table
- variance statistics
- profiler or measurement notes
- result type classification
- final conclusion label:
  - SUCCESS
  - PARTIAL_SUCCESS
  - INVALID
  - BLOCKED
- next optimization recommendations
