# P1 Weak Prompt Template

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
