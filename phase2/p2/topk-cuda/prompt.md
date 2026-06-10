# P2 Medium Prompt: topk-cuda

You are a CUDA performance engineer.

## Benchmark

- benchmark: topk-cuda
- benchmark path: `/home/r14525078/p2/HeCBench/src/topk-cuda`
- result path: `/home/r14525078/p2/HeCBench/src/topk-cuda/result`
- category: ai_topk
- expected metric: average top-k execution time us across full hidden_size/topk matrix

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

- Run the full hidden sizes and topk values. Do not skip OOM/slow cases in final.
- Workspace reuse and radix-selection block-size tuning are known high-value hypotheses.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p2/HeCBench/src/topk-cuda
mkdir -p /home/r14525078/p2/HeCBench/src/topk-cuda/result
module purge
module load cuda/12.8
sbatch run_topk_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p2/HeCBench/src/topk-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
./main 3072 100
```

If `run_topk_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
