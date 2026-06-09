#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "phase2"
TOP_PROMPTS = ROOT / "prompts"


BENCHMARKS = [
    {
        "benchmark": "p2p-cuda",
        "canonical_name": "p2p-cuda",
        "category": "multi_gpu_interconnect",
        "benchmark_path": "/home/r14525078/HeCBench/src/p2p-cuda",
        "result_path": "/home/r14525078/HeCBench/src/p2p-cuda/result",
        "required_gpus": "4",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "P2P bandwidth GB/s and correctness over directional GPU pairs",
        "run_hint": "Use sbatch and preserve full topology/pair sweep. Do not report only the best pair.",
        "notes": "Topology-aware measurement; improvement below 1% should be marked measurement-equivalent.",
        "result_type_hint": "TOPOLOGY_MEASURE",
        "profiler_hint": "Record nvidia-smi topo -m; profiler optional unless kernel changes are made.",
    },
    {
        "benchmark": "shmembench-cuda",
        "canonical_name": "shmembench-cuda",
        "category": "shared_memory",
        "benchmark_path": "/home/r14525078/HeCBench/src/shmembench-cuda",
        "result_path": "/home/r14525078/HeCBench/src/shmembench-cuda/result",
        "required_gpus": "1",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "shared-memory bandwidth GB/s and kernel avg/min/max us",
        "run_hint": "Run ./main 1000 or the benchmark's official repeat. Preserve checksum validation.",
        "notes": "Block-size sweeps are valid only if checksum/reference remains valid for that configuration.",
        "result_type_hint": "PARAM_TUNE",
        "profiler_hint": "Collect Nsight Compute shared-memory bank conflict and occupancy metrics for final candidate if available.",
    },
    {
        "benchmark": "topk-cuda",
        "canonical_name": "topk-cuda",
        "category": "ai_topk",
        "benchmark_path": "/home/r14525078/HeCBench/src/topk-cuda",
        "result_path": "/home/r14525078/HeCBench/src/topk-cuda/result",
        "required_gpus": "1",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "average top-k execution time us across full hidden_size/topk matrix",
        "run_hint": "Run the full hidden sizes and topk values. Do not skip OOM/slow cases in final.",
        "notes": "Workspace reuse and radix-selection block-size tuning are known high-value hypotheses.",
        "result_type_hint": "KERNEL_OPT",
        "profiler_hint": "Collect occupancy, register pressure, memory throughput, and CUB temporary storage notes.",
    },
    {
        "benchmark": "allreduce-cuda",
        "canonical_name": "allreduce-cuda",
        "category": "mpi_collective",
        "benchmark_path": "/home/r14525078/HeCBench/src/allreduce-cuda",
        "result_path": "/home/r14525078/HeCBench/src/allreduce-cuda/result",
        "required_gpus": "2",
        "requires_mpi": "true",
        "requires_nccl": "false",
        "module": "nvhpc-24.11_hpcx-2.20_cuda-12.6",
        "submission_limit": "5",
        "expected_metric": "allreduce us/iteration by buffer size",
        "run_hint": "Use 2 MPI ranks / 2 GPUs. If default UCX/GDRCopy fails, record tuned UCX launcher as environment metadata.",
        "notes": "Known valid launcher: UCX_TLS=self,shm,cuda_copy,cuda_ipc with --mca coll ^hcoll,ucc --mca pml ucx.",
        "result_type_hint": "ENV_FIX",
        "profiler_hint": "Profiler not required for launcher repair; collect MPI/UCX environment metadata.",
    },
    {
        "benchmark": "moe-cuda",
        "canonical_name": "moe-cuda",
        "category": "moe_inference",
        "benchmark_path": "/home/r14525078/HeCBench/src/moe-cuda",
        "result_path": "/home/r14525078/HeCBench/src/moe-cuda/result",
        "required_gpus": "1",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "3",
        "expected_metric": "Average execution time of kernels us for topk 1/2/4/8",
        "run_hint": "Fixed cases: 32768 tokens, 384 experts, topk 1/2/4/8, repeat 1000.",
        "notes": "Hybrid dispatch should be evaluated per top-k; full fusion can regress for topk=8.",
        "result_type_hint": "KERNEL_OPT",
        "profiler_hint": "Collect launch count, occupancy, global-memory traffic, and shared-memory usage for best candidate.",
    },
    {
        "benchmark": "pingpong-cuda",
        "canonical_name": "pingpong-cuda",
        "category": "mpi_nccl_pingpong",
        "benchmark_path": "/home/r14525078/HeCBench/src/pingpong-cuda",
        "result_path": "/home/r14525078/HeCBench/src/pingpong-cuda/result",
        "required_gpus": "2",
        "requires_mpi": "true",
        "requires_nccl": "true",
        "module": "nvhpc-24.11_hpcx-2.20_cuda-12.6",
        "submission_limit": "5",
        "expected_metric": "MPI and NCCL one-way transfer time and GB/s by size",
        "run_hint": "Use 2 MPI ranks / 2 GPUs. Compare MPI and NCCL without claiming general collective superiority.",
        "notes": "Record UCX_TLS and NCCL environment. Full size sweep required.",
        "result_type_hint": "MEASURE_FIX",
        "profiler_hint": "Profiler optional; transport metadata and topology are required.",
    },
    {
        "benchmark": "simpleMultiDevice-cuda",
        "canonical_name": "simpleMultiDevice-cuda",
        "category": "multi_gpu_scaling",
        "benchmark_path": "/home/r14525078/HeCBench/src/simpleMultiDevice-cuda",
        "result_path": "/home/r14525078/HeCBench/src/simpleMultiDevice-cuda/result",
        "required_gpus": "2-4",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "total_us, h2d_us, kernel_us, d2h_us, correctness diff",
        "run_hint": "Test at least 2 GPUs; if available also test 4 GPUs. Separate copy and kernel timing.",
        "notes": "End-to-end speedup may be H2D-copy-limited.",
        "result_type_hint": "KERNEL_OPT",
        "profiler_hint": "Collect copy/kernel split; profiler optional unless kernel is changed.",
    },
    {
        "benchmark": "moe-align",
        "canonical_name": "moe-align",
        "category": "moe_alignment",
        "benchmark_path": "/home/r14525078/HeCBench/src/moe-align",
        "result_path": "/home/r14525078/HeCBench/src/moe-align/result",
        "required_gpus": "1",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "mean latency over tokens/topk/experts/block_size combinations",
        "run_hint": "Compare full parameter matrix and explicitly include correctness/status fields in CSV.",
        "notes": "Existing comparison CSV lacked explicit correctness field; Phase 2 must fix that.",
        "result_type_hint": "PARAM_TUNE",
        "profiler_hint": "Collect occupancy and memory-throughput notes for final kernels if available.",
    },
    {
        "benchmark": "prefetch-cuda",
        "canonical_name": "prefetch-cuda",
        "category": "unified_memory",
        "benchmark_path": "/home/r14525078/HeCBench/src/prefetch-cuda",
        "result_path": "/home/r14525078/HeCBench/src/prefetch-cuda/result",
        "required_gpus": "1",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "avg_ms for with_prefetch and without_prefetch modes",
        "run_hint": "Preserve both prefetch and no-prefetch modes; do not only report the faster mode.",
        "notes": "Separate prefetch API overhead from demand-paging penalty.",
        "result_type_hint": "PARAM_TUNE",
        "profiler_hint": "Collect unified-memory migration/page fault information if available.",
    },
    {
        "benchmark": "softmax-cuda",
        "canonical_name": "softmax-cuda",
        "category": "softmax_kernel",
        "benchmark_path": "/home/r14525078/HeCBench/src/softmax-cuda",
        "result_path": "/home/r14525078/HeCBench/src/softmax-cuda/result",
        "required_gpus": "1",
        "requires_mpi": "false",
        "requires_nccl": "false",
        "module": "cuda/12.8",
        "submission_limit": "5",
        "expected_metric": "avg_ms by slice size and implementation",
        "run_hint": "Benchmark naive and optimized implementations; preserve full slice-size sweep.",
        "notes": "Dispatch policy may depend on slice size; do not assume one kernel dominates all shapes.",
        "result_type_hint": "KERNEL_OPT",
        "profiler_hint": "Collect occupancy, expf instruction reduction, shared memory, and memory throughput notes.",
    },
]


RUN_SCRIPTS = {
    "p2p-cuda": "run_p2p_cuda.slurm",
    "shmembench-cuda": "run_shmembench_cuda.slurm",
    "topk-cuda": "run_topk_cuda.slurm",
    "allreduce-cuda": "run_allreduce_cuda.slurm",
    "moe-cuda": "run_moe_cuda.slurm",
    "pingpong-cuda": "run_pingpong_cuda.slurm",
    "simpleMultiDevice-cuda": "run_simpleMultiDevice_cuda.slurm",
    "moe-align": "run_moe_align_cuda.slurm",
    "prefetch-cuda": "run_prefetch.slurm",
    "softmax-cuda": "run_softmax_cuda.slurm",
}


BASELINE_COMMANDS = {
    "p2p-cuda": "make clean || true && make ARCH=sm_70\n# Run the official full P2P/topology sweep from the Slurm script.",
    "shmembench-cuda": "make clean || true && make ARCH=sm_70\n./main 1000",
    "topk-cuda": "make clean || true && make ARCH=sm_70\n./main 3072 100",
    "allreduce-cuda": "make clean || true && make ARCH=sm_70\nUCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main",
    "moe-cuda": "make clean || true && make ARCH=sm_70\n./main 32768 384 1 1000\n./main 32768 384 2 1000\n./main 32768 384 4 1000\n./main 32768 384 8 1000",
    "pingpong-cuda": "make clean || true && make ARCH=sm_70\nUCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-mpi\nUCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-nccl",
    "simpleMultiDevice-cuda": "make clean || true && make ARCH=sm_70 MAX_GPU=4\n./main 1000",
    "moe-align": "make clean || true && make ARCH=sm_70\n# Run the official moe-align parameter sweep from the Slurm script.",
    "prefetch-cuda": "make clean || true && make ARCH=sm_70\n./main 10\n./main 100",
    "softmax-cuda": "make clean || true && make ARCH=sm_70\n./main 100000 784 0 100\n./main 100000 784 1 100",
}


def run_instruction_block(b, detail="medium"):
    script = RUN_SCRIPTS[b["benchmark"]]
    command = BASELINE_COMMANDS[b["benchmark"]]
    if detail == "short":
        return f"""## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd {b['benchmark_path']}
mkdir -p {b['result_path']}
{module_block(b['module'])}
sbatch {script}
```

If `{script}` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.
"""
    return f"""## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd {b['benchmark_path']}
mkdir -p {b['result_path']}
{module_block(b['module'])}
sbatch {script}
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `{b['result_path']}`:

```bash
{command}
```

If `{script}` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`{b['required_gpus']}`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.
"""


def ensure_dirs():
    for rel in ["prompt_templates", "prompts", "metadata", "scripts", "reports"]:
        (PHASE2 / rel).mkdir(parents=True, exist_ok=True)
    for b in BENCHMARKS:
        (PHASE2 / "prompts" / b["benchmark"]).mkdir(parents=True, exist_ok=True)
        (TOP_PROMPTS / b["benchmark"]).mkdir(parents=True, exist_ok=True)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_write(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def module_block(module):
    return f"module purge\nmodule load {module}"


def p1_prompt(b):
    return f"""# P1 Weak Prompt: {b['benchmark']}

Please optimize the CUDA benchmark at:

`{b['benchmark_path']}`

Goal:
- Improve performance for `{b['expected_metric']}`.
- Keep correctness.

{run_instruction_block(b, "short")}

Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
"""


def p2_prompt(b):
    return f"""# P2 Medium Prompt: {b['benchmark']}

You are a CUDA performance engineer.

## Benchmark

- benchmark: {b['benchmark']}
- benchmark path: `{b['benchmark_path']}`
- result path: `{b['result_path']}`
- category: {b['category']}
- expected metric: {b['expected_metric']}

## Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
```bash
{module_block(b['module'])}
```
- required GPUs: {b['required_gpus']}
- requires MPI: {b['requires_mpi']}
- requires NCCL: {b['requires_nccl']}

## Goal

1. Establish a real baseline before modifying source code.
2. Improve performance while keeping correctness PASS.
3. Save raw output and summarize the final valid result.

## Rules

1. Run baseline before source changes.
2. Baseline does not count as an optimization submission.
3. After baseline, at most {b['submission_limit']} optimization sbatch submissions.
4. Do not remove or weaken correctness checks.
5. Do not shrink input size or skip cases to fake speedup.
6. Save `.out`, `.err`, and result `.txt` files under the result directory.
7. Report job id, node, correctness, metric, speedup, and whether the result is accepted or rejected.

## Benchmark-Specific Notes

- {b['run_hint']}
- {b['notes']}

{run_instruction_block(b)}

## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
"""


def p3_prompt(b):
    return f"""# P3 Strong Prompt: {b['benchmark']}

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as an experimental protocol, not a casual request.

## Prompt Metadata

- benchmark: {b['benchmark']}
- canonical_name: {b['canonical_name']}
- benchmark_category: {b['category']}
- prompt_level: P3
- target_agent: server-side coding agent
- submission_limit: {b['submission_limit']}
- baseline_counts_as_submission: false
- required_gpus: {b['required_gpus']}
- requires_mpi: {b['requires_mpi']}
- requires_nccl: {b['requires_nccl']}
- expected_metric: {b['expected_metric']}
- correctness_required: true

## Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
```bash
{module_block(b['module'])}
```

## Paths

- benchmark_path: `{b['benchmark_path']}`
- result_path: `{b['result_path']}`

## Benchmark-Specific Requirements

- {b['run_hint']}
- {b['notes']}
- profiler requirement: {b['profiler_hint']}
- expected primary result type: {b['result_type_hint']}

{run_instruction_block(b)}

## Hard Rules

1. Baseline does not count toward submission limit.
2. After baseline, at most {b['submission_limit']} optimization sbatch submissions.
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

- `result/{b['benchmark']}_<jobid>.out`
- `result/{b['benchmark']}_<jobid>.err`
- `result/{b['benchmark']}_result_<jobid>.txt`

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

{b['profiler_hint']}

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
"""


TEMPLATE_P1 = """# P1 Weak Prompt Template

Please optimize the CUDA benchmark at:

`<benchmark_path>`

Goal:
- Improve performance.
- Keep correctness.

Server run instructions:

```bash
cd <benchmark_path>
mkdir -p <result_path>
module purge
module load <module>
sbatch <run_script>.slurm
```

Do not run GPU benchmarks directly on the login node.

Please inspect the code, make changes, run the benchmark, and report the result.
"""


TEMPLATE_P2 = """# P2 Medium Prompt Template

You are a CUDA performance engineer.

Benchmark path:
`<benchmark_path>`

Result path:
`<result_path>`

Goal:
- Establish baseline.
- Improve performance.
- Keep correctness PASS.

Rules:
1. Run baseline before modifying source.
2. Save raw output.
3. Do not remove correctness checks.
4. At most `<N>` optimization submissions after baseline.
5. Report job id, correctness, metric, and speedup.

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083

Server run instructions:

```bash
cd <benchmark_path>
mkdir -p <result_path>
module purge
module load <module>
sbatch <run_script>.slurm
```

The Slurm script must build the benchmark, run the official baseline command, and save raw output to `<result_path>`.

Final output:
- agent_summary.md
"""


TEMPLATE_P3 = """# P3 Strong Prompt Template

You are a CUDA performance engineer conducting a reproducible optimization experiment.

Required sections:

1. Prompt Metadata
2. Environment
3. Paths
4. Benchmark-Specific Requirements
5. Server Run Instructions
6. Hard Rules
7. Baseline Requirements
8. Optimization Submission Rules
9. Correctness Gate
10. Required Result Types
11. CSV Result Schema
12. Variance / Repeated Trials
13. Profiler / Measurement Notes
14. Contradiction Check
15. Final Output

The final result is valid only if every required correctness case PASSes and raw output is preserved.
"""


def write_templates():
    write(PHASE2 / "prompt_templates/P1_weak_prompt_template.md", TEMPLATE_P1)
    write(PHASE2 / "prompt_templates/P2_medium_prompt_template.md", TEMPLATE_P2)
    write(PHASE2 / "prompt_templates/P3_strong_prompt_template.md", TEMPLATE_P3)


def write_prompts():
    for b in BENCHMARKS:
        base = PHASE2 / "prompts" / b["benchmark"]
        prompts = {
            "P1_prompt.md": p1_prompt(b),
            "P2_prompt.md": p2_prompt(b),
            "P3_prompt.md": p3_prompt(b),
        }
        for name, text in prompts.items():
            write(base / name, text)
            write(TOP_PROMPTS / b["benchmark"] / name, text)


def write_metadata():
    taxonomy_fields = [
        "benchmark", "canonical_name", "category", "benchmark_path", "result_path",
        "required_gpus", "requires_mpi", "requires_nccl", "module",
        "submission_limit", "expected_metric", "result_type_hint", "notes",
    ]
    csv_write(PHASE2 / "metadata/benchmark_taxonomy.csv", taxonomy_fields, [
        {k: b[k] for k in taxonomy_fields} for b in BENCHMARKS
    ])

    matrix_fields = ["benchmark", "category", "P1", "P2", "P3", "notes"]
    csv_write(PHASE2 / "metadata/prompt_assignment_matrix.csv", matrix_fields, [
        {
            "benchmark": b["benchmark"],
            "category": b["category"],
            "P1": "yes",
            "P2": "yes",
            "P3": "yes",
            "notes": b["notes"],
        }
        for b in BENCHMARKS
    ])

    write(PHASE2 / "metadata/benchmark_taxonomy.json", json.dumps(BENCHMARKS, ensure_ascii=False, indent=2))


def score_prompt_text(text):
    checks = {
        "task_clarity": bool(re.search(r"benchmark|Goal|expected metric", text, re.I)),
        "environment": "Environment" in text and "V100" in text,
        "baseline": "baseline" in text.lower(),
        "correctness": "correctness" in text.lower() and ("FAIL" in text or "invalid" in text.lower()),
        "raw_output": "raw output" in text.lower() or ".out" in text,
        "submission_limit": ("submission" in text.lower() or "submissions" in text.lower()) and ("limit" in text.lower() or "at most" in text.lower() or "最多" in text),
        "csv": "csv" in text.lower(),
        "contradiction": "Contradiction" in text or "contradiction" in text.lower(),
        "variance": "trial" in text.lower() or "stddev" in text.lower() or "variance" in text.lower(),
        "profiler": "profiler" in text.lower() or "Nsight" in text,
    }
    score = 0
    score += 10 if checks["task_clarity"] else 0
    score += 10 if checks["environment"] else 0
    score += 10 if checks["baseline"] else 0
    score += 15 if checks["correctness"] else 0
    score += 10 if checks["raw_output"] else 0
    score += 10 if checks["submission_limit"] else 0
    score += 10 if checks["csv"] else 0
    score += 10 if checks["contradiction"] else 0
    score += 10 if checks["variance"] else 0
    score += 5 if checks["profiler"] else 0
    return score, checks


def write_inventory():
    rows = []
    for b in BENCHMARKS:
        for level in ["P1", "P2", "P3"]:
            rel = Path("prompts") / b["benchmark"] / f"{level}_prompt.md"
            text = (PHASE2 / rel).read_text(encoding="utf-8")
            score, checks = score_prompt_text(text)
            rows.append({
                "benchmark": b["benchmark"],
                "prompt_level": level,
                "prompt_file": rel.as_posix(),
                "lines": str(text.count("\n") + 1),
                "chars": str(len(text)),
                "mentions_correctness": str("correctness" in text.lower()),
                "mentions_baseline": str("baseline" in text.lower()),
                "mentions_sbatch": str("sbatch" in text.lower()),
                "mentions_submission_limit": str(checks["submission_limit"]),
                "mentions_raw_output": str(checks["raw_output"]),
                "mentions_csv": str(checks["csv"]),
                "mentions_contradiction_check": str(checks["contradiction"]),
                "mentions_variance": str(checks["variance"]),
                "mentions_profiler": str(checks["profiler"]),
                "prompt_score": str(score),
                "main_strength": "strong protocol" if score >= 85 else ("engineering prompt" if score >= 45 else "weak prompt"),
                "main_gap": "none" if score >= 85 else "missing one or more reproducibility controls",
            })
    fields = [
        "benchmark", "prompt_level", "prompt_file", "lines", "chars",
        "mentions_correctness", "mentions_baseline", "mentions_sbatch",
        "mentions_submission_limit", "mentions_raw_output", "mentions_csv",
        "mentions_contradiction_check", "mentions_variance", "mentions_profiler",
        "prompt_score", "main_strength", "main_gap",
    ]
    csv_write(PHASE2 / "metadata/prompt_inventory_phase2.csv", fields, rows)


def write_docs():
    readme = """# Phase 2 Prompt Specification Package

This directory contains the Phase 2 prompt experiment package.

Goal:
- compare P1/P2/P3 prompt constraint levels
- run the same HeCBench benchmark tasks under different prompt protocols
- collect raw outputs, CSV summaries, and agent summaries for later analysis

Directory map:

- `prompt_templates/`: generic P1/P2/P3 templates
- `prompts/`: 30 benchmark-specific prompts
- `metadata/`: taxonomy, prompt inventory, assignment matrix
- `scripts/`: local validation utilities
- `reports/`: operation plan and acceptance criteria

Server-side workflow:

1. Copy `phase2/` to `/home/r14525078/HeCBench/phase2`.
2. For each benchmark and prompt level, provide the corresponding prompt to the server-side agent.
3. Run experiments on the server using Slurm only.
4. Preserve result directories and raw output.
5. Return `phase2_results/`, result CSV files, and agent summaries for final analysis.
"""
    write(PHASE2 / "README.md", readme)
    write(TOP_PROMPTS / "README.md", """# Prompt Files

This portable directory contains the 30 benchmark-specific Phase 2 prompts.

Structure:

- `<benchmark>/P1_prompt.md`: weak prompt
- `<benchmark>/P2_prompt.md`: medium prompt
- `<benchmark>/P3_prompt.md`: strong reproducible protocol prompt

The full experiment package, including metadata, scripts, templates, and reports, is under `/home/a/PP/phase2`.
""")

    schema = """# Prompt Schema

Each P3 prompt must contain:

- Prompt Metadata
- Environment
- Paths
- Benchmark-Specific Requirements
- Hard Rules
- Baseline Requirements
- Optimization Submission Rules
- Correctness Gate
- Required Result Types
- CSV Result Schema
- Variance / Repeated Trials
- Profiler / Measurement Notes
- Contradiction Check
- Final Output

Required CSV schema:

```csv
benchmark,job_id,node,prompt_level,submission_index,variant,case,metric_name,metric_value,metric_unit,correctness,status,result_type,accepted,reject_reason,notes
```
"""
    write(PHASE2 / "prompt_schema.md", schema)

    rubric = """# Prompt Quality Rubric

Total: 100 points.

| Item | Points |
|---|---:|
| Task clarity | 10 |
| Environment specificity | 10 |
| Baseline requirement | 10 |
| Correctness gate | 15 |
| Raw output requirement | 10 |
| Submission limit | 10 |
| CSV / machine-readable output | 10 |
| Contradiction check | 10 |
| Variance / repeated trials | 10 |
| Profiler requirement | 5 |

Interpretation:

- 0-35: weak prompt
- 40-75: medium engineering prompt
- 80-100: strong reproducible experiment protocol
"""
    write(PHASE2 / "prompt_quality_rubric.md", rubric)

    plan = """# PHASE2_PLAN

Phase 2 evaluates whether prompt constraint level changes correctness, performance, and auditability of AI-assisted CUDA optimization.

Experiment matrix:

- 10 benchmarks
- 3 prompt levels
- 30 total prompts

Prompt levels:

- P1: weak, casual optimization request
- P2: medium, basic engineering constraints
- P3: strong, reproducible experimental protocol

Data to collect per run:

- prompt file used
- source diff or changed file list
- Slurm job ids
- `.out`, `.err`, result `.txt`
- CSV metrics if generated
- `agent_summary.md`
- final accepted/rejected status
"""
    write(PHASE2 / "reports/PHASE2_PLAN.md", plan)

    acceptance = """# PHASE2_ACCEPTANCE

A server-side experiment is acceptable only if:

1. The prompt file used is recorded.
2. Baseline is attempted and classified.
3. Raw `.out`, `.err`, and result `.txt` are preserved.
4. Correctness is explicitly reported.
5. FAIL/skipped/waived cases are not counted as success.
6. Speedup uses measured baseline only.
7. Agent summary includes accepted and rejected attempts.
8. P3 runs include contradiction check, variance notes, and profiler/measurement notes.

When data is returned, include the whole result tree rather than only final tables.
"""
    write(PHASE2 / "reports/PHASE2_ACCEPTANCE.md", acceptance)

    server = """# SERVER_SIDE_TODO

This is the checklist for running Phase 2 on the server.

## 1. Copy Files

Copy the entire `phase2/` directory to:

```bash
/home/r14525078/HeCBench/phase2
```

Each benchmark prompt is under:

```bash
phase2/prompts/<benchmark>/P1_prompt.md
phase2/prompts/<benchmark>/P2_prompt.md
phase2/prompts/<benchmark>/P3_prompt.md
```

## 2. Run Experiments

For each benchmark:

1. Start with `P1_prompt.md`.
2. Then run `P2_prompt.md`.
3. Then run `P3_prompt.md`.
4. Use a clean copy or clean git branch/worktree per prompt level when possible.
5. Do not mix outputs from different prompt levels.

Recommended result layout:

```text
phase2_results/
├── <benchmark>/
│   ├── P1/
│   │   ├── prompt_used.md
│   │   ├── agent_summary.md
│   │   ├── raw/
│   │   ├── csv/
│   │   └── patch_or_diff/
│   ├── P2/
│   └── P3/
```

## 3. Required Files To Return

Return these files after experiments:

```text
phase2_results/
phase2/metadata/prompt_inventory_phase2.csv
phase2/metadata/prompt_assignment_matrix.csv
phase2/metadata/benchmark_taxonomy.csv
```

For each benchmark/prompt level, include:

- `prompt_used.md`
- `agent_summary.md`
- all Slurm `.out`
- all Slurm `.err`
- benchmark result `.txt`
- generated `.csv`
- source diff or changed file list
- notes for failed/blocked runs

## 4. Minimum Run Priority

If queue time is limited, run this priority order first:

1. `softmax-cuda`: clear kernel optimization case
2. `topk-cuda`: workspace/radix optimization case
3. `moe-cuda`: prompt-sensitive top-k strategy case
4. `allreduce-cuda`: environment repair case
5. `pingpong-cuda`: MPI/NCCL measurement case
6. `shmembench-cuda`: profiler/bank-conflict case
7. `simpleMultiDevice-cuda`
8. `prefetch-cuda`
9. `p2p-cuda`
10. `moe-align`

## 5. After Completion

Compress the result bundle:

```bash
tar -czf phase2_results_$(date +%Y%m%d_%H%M%S).tar.gz phase2_results phase2/metadata
```

Send the tarball back for analysis.
"""
    write(PHASE2 / "reports/SERVER_SIDE_TODO.md", server)
    write(TOP_PROMPTS / "SERVER_SIDE_TODO.md", server)

    summary = """# PHASE2_SUMMARY

This file is intentionally left as the post-experiment summary target.

After server-side runs are returned, fill this report with:

- P1/P2/P3 correctness rates
- speedup distributions
- invalid result counts
- contradiction counts
- raw-output completeness
- prompt score vs result quality
- benchmark-specific findings
"""
    write(PHASE2 / "reports/PHASE2_SUMMARY.md", summary)


CHECK_SCRIPT = r'''#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["P1_prompt.md", "P2_prompt.md", "P3_prompt.md"]
missing = []
for bench_dir in sorted((root / "prompts").iterdir()):
    if not bench_dir.is_dir():
        continue
    for name in required:
        path = bench_dir / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(root)))
if missing:
    print("Missing prompt files:")
    for item in missing:
        print(item)
    raise SystemExit(1)
print("All benchmark prompt files exist.")
'''

SCORE_SCRIPT = r'''#!/usr/bin/env python3
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]

def score(text):
    checks = [
        (10, bool(re.search(r"benchmark|Goal|expected metric", text, re.I))),
        (10, "Environment" in text and "V100" in text),
        (10, "baseline" in text.lower()),
        (15, "correctness" in text.lower() and ("FAIL" in text or "invalid" in text.lower())),
        (10, "raw output" in text.lower() or ".out" in text),
        (10, ("submission" in text.lower() or "submissions" in text.lower()) and ("limit" in text.lower() or "at most" in text.lower() or "最多" in text)),
        (10, "csv" in text.lower()),
        (10, "contradiction" in text.lower()),
        (10, "trial" in text.lower() or "stddev" in text.lower() or "variance" in text.lower()),
        (5, "profiler" in text.lower() or "nsight" in text.lower()),
    ]
    return sum(points for points, ok in checks if ok)

for path in sorted((root / "prompts").glob("*/*_prompt.md")):
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"{path.relative_to(root)},{score(text)}")
'''

INVENTORY_SCRIPT = r'''#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
generator = root.parent / "docs" / "generate_phase2_prompts.py"
if generator.exists():
    subprocess.check_call([sys.executable, str(generator)])
    print("Regenerated phase2 metadata.")
else:
    print("Generator not found; use existing metadata.")
'''


def write_scripts():
    write(PHASE2 / "scripts/check_prompt_requirements.py", CHECK_SCRIPT)
    write(PHASE2 / "scripts/score_prompt_quality.py", SCORE_SCRIPT)
    write(PHASE2 / "scripts/generate_prompt_inventory.py", INVENTORY_SCRIPT)


def main():
    ensure_dirs()
    write_templates()
    write_prompts()
    write_metadata()
    write_docs()
    write_scripts()
    write_inventory()
    print(f"Wrote Phase 2 prompt package to {PHASE2}")
    print("Prompts:", len(BENCHMARKS) * 3)


if __name__ == "__main__":
    main()
