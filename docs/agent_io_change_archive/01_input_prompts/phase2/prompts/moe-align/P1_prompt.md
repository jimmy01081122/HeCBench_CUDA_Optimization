# P1 Weak Prompt: moe-align

Please optimize the CUDA benchmark at:

`/home/r14525078/HeCBench/src/moe-align`

Goal:
- Improve performance for `mean latency over tokens/topk/experts/block_size combinations`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/HeCBench/src/moe-align
mkdir -p /home/r14525078/HeCBench/src/moe-align/result
module purge
module load cuda/12.8
sbatch run_moe_align_cuda.slurm
```

If `run_moe_align_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
