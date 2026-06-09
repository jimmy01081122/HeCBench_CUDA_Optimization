# P1 Weak Prompt: moe-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/p1/HeCBench/src/moe-cuda`

Goal:
- Improve performance for `Average execution time of kernels us for topk 1/2/4/8`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/p1/HeCBench/src/moe-cuda
mkdir -p /home/r14525078/p1/HeCBench/src/moe-cuda
module purge
module load cuda/12.8
sbatch run_moe_cuda.slurm
```

If `run_moe_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
