# PHASE2_PLAN

Phase 2 evaluates whether prompt constraint level changes correctness, performance, and auditability of AI-assisted CUDA optimization.

Experiment matrix:

- 10 benchmarks
- 3 prompt levels
- 30 total prompts

Prompt levels:

- P1: weak, casual optimization request
- P2: medium, basic engineering constraints
- P3: strong, reproducible experimental protocol

Data to collect per run:

- prompt file used
- source diff or changed file list
- Slurm job ids
- `.out`, `.err`, result `.txt`
- CSV metrics if generated
- `agent_summary.md`
- final accepted/rejected status
