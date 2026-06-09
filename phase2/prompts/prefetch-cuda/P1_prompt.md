# P1 Weak Prompt: prefetch-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/HeCBench/src/prefetch-cuda`

Goal:
- Improve performance for `avg_ms for with_prefetch and without_prefetch modes`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/HeCBench/src/prefetch-cuda
mkdir -p /home/r14525078/HeCBench/src/prefetch-cuda/result
module purge
module load cuda/12.8
sbatch run_prefetch.slurm
```

If `run_prefetch.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
