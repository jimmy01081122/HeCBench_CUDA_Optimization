# P2 Medium Prompt: simpleMultiDevice-cuda

You are a CUDA performance engineer.

## Benchmark

- benchmark: simpleMultiDevice-cuda
- benchmark path: `/home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda`
- result path: `/home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda/result`
- category: multi_gpu_scaling
- expected metric: total_us, h2d_us, kernel_us, d2h_us, correctness diff

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
- required GPUs: 2-4
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

- Test at least 2 GPUs; if available also test 4 GPUs. Separate copy and kernel timing.
- End-to-end speedup may be H2D-copy-limited.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda
mkdir -p /home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda/result
module purge
module load cuda/12.8
sbatch run_simpleMultiDevice_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p2/HeCBench/src/simpleMultiDevice-cuda/result`:

```bash
make clean || true && make ARCH=sm_70 MAX_GPU=4
./main 1000
```

If `run_simpleMultiDevice_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`2-4`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
