# P1 Weak Prompt: pingpong-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/HeCBench/src/pingpong-cuda`

Goal:
- Improve performance for `MPI and NCCL one-way transfer time and GB/s by size`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/HeCBench/src/pingpong-cuda
mkdir -p /home/r14525078/HeCBench/src/pingpong-cuda/result
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
sbatch run_pingpong_cuda.slurm
```

If `run_pingpong_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
