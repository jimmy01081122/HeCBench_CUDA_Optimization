# P1 Weak Prompt: p2p-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/p1/HeCBench/src/p2p-cuda`

Goal:
- Improve performance for `P2P bandwidth GB/s and correctness over directional GPU pairs`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/p1/HeCBench/src/p2p-cuda
mkdir -p /home/r14525078/p1/HeCBench/src/p2p-cuda/result
module purge
module load cuda/12.8
sbatch run_p2p_cuda.slurm
```

If `run_p2p_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
