# SERVER_SIDE_TODO

This is the checklist for running Phase 2 on the server.

## 1. Copy Files

Copy the entire `phase2/` directory to:

```bash
/home/r14525078/HeCBench/phase2
```

Each benchmark prompt is under:

```bash
phase2/prompts/<benchmark>/P1_prompt.md
phase2/prompts/<benchmark>/P2_prompt.md
phase2/prompts/<benchmark>/P3_prompt.md
```

## 2. Run Experiments

For each benchmark:

1. Start with `P1_prompt.md`.
2. Then run `P2_prompt.md`.
3. Then run `P3_prompt.md`.
4. Use a clean copy or clean git branch/worktree per prompt level when possible.
5. Do not mix outputs from different prompt levels.

Recommended result layout:

```text
phase2_results/
├── <benchmark>/
│   ├── P1/
│   │   ├── prompt_used.md
│   │   ├── agent_summary.md
│   │   ├── raw/
│   │   ├── csv/
│   │   └── patch_or_diff/
│   ├── P2/
│   └── P3/
```

## 3. Required Files To Return

Return these files after experiments:

```text
phase2_results/
phase2/metadata/prompt_inventory_phase2.csv
phase2/metadata/prompt_assignment_matrix.csv
phase2/metadata/benchmark_taxonomy.csv
```

For each benchmark/prompt level, include:

- `prompt_used.md`
- `agent_summary.md`
- all Slurm `.out`
- all Slurm `.err`
- benchmark result `.txt`
- generated `.csv`
- source diff or changed file list
- notes for failed/blocked runs

## 4. Minimum Run Priority

If queue time is limited, run this priority order first:

1. `softmax-cuda`: clear kernel optimization case
2. `topk-cuda`: workspace/radix optimization case
3. `moe-cuda`: prompt-sensitive top-k strategy case
4. `allreduce-cuda`: environment repair case
5. `pingpong-cuda`: MPI/NCCL measurement case
6. `shmembench-cuda`: profiler/bank-conflict case
7. `simpleMultiDevice-cuda`
8. `prefetch-cuda`
9. `p2p-cuda`
10. `moe-align`

## 5. After Completion

Compress the result bundle:

```bash
tar -czf phase2_results_$(date +%Y%m%d_%H%M%S).tar.gz phase2_results phase2/metadata
```

Send the tarball back for analysis.
