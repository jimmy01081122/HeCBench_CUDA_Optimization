# P2 Medium Prompt Template

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
