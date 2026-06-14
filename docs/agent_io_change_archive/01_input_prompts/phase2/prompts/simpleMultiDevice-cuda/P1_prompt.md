# P1 Weak Prompt: simpleMultiDevice-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/HeCBench/src/simpleMultiDevice-cuda`

Goal:
- Improve performance for `total_us, h2d_us, kernel_us, d2h_us, correctness diff`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/HeCBench/src/simpleMultiDevice-cuda
mkdir -p /home/r14525078/HeCBench/src/simpleMultiDevice-cuda/result
module purge
module load cuda/12.8
sbatch run_simpleMultiDevice_cuda.slurm
```

If `run_simpleMultiDevice_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
