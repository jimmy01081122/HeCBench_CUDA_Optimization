# Phase 2 Prompt Specification Package

This directory contains the Phase 2 prompt experiment package.

Goal:
- compare P1/P2/P3 prompt constraint levels
- run the same HeCBench benchmark tasks under different prompt protocols
- collect raw outputs, CSV summaries, and agent summaries for later analysis

Directory map:

- `prompt_templates/`: generic P1/P2/P3 templates
- `prompts/`: 30 benchmark-specific prompts
- `metadata/`: taxonomy, prompt inventory, assignment matrix
- `scripts/`: local validation utilities
- `reports/`: operation plan and acceptance criteria

Server-side workflow:

1. Copy `phase2/` to `/home/r14525078/HeCBench/phase2`.
2. For each benchmark and prompt level, provide the corresponding prompt to the server-side agent.
3. Run experiments on the server using Slurm only.
4. Preserve result directories and raw output.
5. Return `phase2_results/`, result CSV files, and agent summaries for final analysis.
