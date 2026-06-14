# Input Prompt

Benchmark: `moe-align-cuda`
Category: ML kernel

## Source prompt files

### `phase2/p1/moe-align-cuda/prompt.md`
```text
# P1 Weak Prompt: moe-align

Please optimize the CUDA benchmark at:

`/home/r14525078/p1/HeCBench/src/moe-align-cuda`

Goal:
- Improve performance for `mean latency over tokens/topk/experts/block_size combinations`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/p1/HeCBench/src/moe-align-cuda
mkdir -p /home/r14525078/p1/HeCBench/src/moe-align-cuda/result
module purge
module load cuda/12.8
sbatch run_moe_align_cuda.slurm
```

If `run_moe_align_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
```

### `phase2/p2/moe-align-cuda/prompt.md`
```text
# P2 Medium Prompt: moe-align

You are a CUDA performance engineer.

## Benchmark

- benchmark: moe-align
- benchmark path: `/home/r14525078/p2/HeCBench/src/moe-align-cuda`
- result path: `/home/r14525078/p2/HeCBench/src/moe-align-cuda/result`
- category: moe_alignment
- expected metric: mean latency over tokens/topk/experts/block_size combinations

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
- required GPUs: 1
- requires MPI: false
- requires NCCL: false

## Goal

1. Establish a real baseline before modifying source code.
2. Improve performance while keeping correctness PASS.
3. Save raw output and summarize the final valid result.

## Rules

1. Run baseline before source changes.
2. Baseline does not count as an optimization submission.
3. After baseline, at most 5 optimization sbatch submissions.
4. Do not remove or weaken correctness checks.
5. Do not shrink input size or skip cases to fake speedup.
6. Save `.out`, `.err`, and result `.txt` files under the result directory.
7. Report job id, node, correctness, metric, speedup, and whether the result is accepted or rejected.

## Benchmark-Specific Notes

- Compare full parameter matrix and explicitly include correctness/status fields in CSV.
- Existing comparison CSV lacked explicit correctness field; Phase 2 must fix that.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p2/HeCBench/src/moe-align-cuda
mkdir -p /home/r14525078/p2/HeCBench/src/moe-align-cuda/result
module purge
module load cuda/12.8
sbatch run_moe_align_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p2/HeCBench/src/moe-align-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
# Run the official moe-align parameter sweep from the Slurm script.
```

If `run_moe_align_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
```

### `phase2/p3/moe-align-cuda/prompt.md`
```text
# P3 Strong Prompt: moe-align

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as an experimental protocol, not a casual request.

## Prompt Metadata

- benchmark: moe-align
- canonical_name: moe-align
- benchmark_category: moe_alignment
- prompt_level: P3
- target_agent: server-side coding agent
- submission_limit: 5
- baseline_counts_as_submission: false
- required_gpus: 1
- requires_mpi: false
- requires_nccl: false
- expected_metric: mean latency over tokens/topk/experts/block_size combinations
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

- benchmark_path: `/home/r14525078/p3/HeCBench/src/moe-align-cuda`
- result_path: `/home/r14525078/p3/HeCBench/src/moe-align-cuda/result`

## Benchmark-Specific Requirements

- Compare full parameter matrix and explicitly include correctness/status fields in CSV.
- Existing comparison CSV lacked explicit correctness field; Phase 2 must fix that.
- profiler requirement: Collect occupancy and memory-throughput notes for final kernels if available.
- expected primary result type: PARAM_TUNE

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p3/HeCBench/src/moe-align-cuda
mkdir -p /home/r14525078/p3/HeCBench/src/moe-align-cuda/result
module purge
module load cuda/12.8
sbatch run_moe_align_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p3/HeCBench/src/moe-align-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
# Run the official moe-align parameter sweep from the Slurm script.
```

If `run_moe_align_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Hard Rules

1. Baseline does not count toward submission limit.
2. After baseline, at most 5 optimization sbatch submissions.
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

- `result/moe-align_<jobid>.out`
- `result/moe-align_<jobid>.err`
- `result/moe-align_result_<jobid>.txt`

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

Collect occupancy and memory-throughput notes for final kernels if available.

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
```
