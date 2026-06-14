# External Basic Tests: Input/System Prompts

Source: `/home/a/data.md`

## System Prompt

N/A

The provided record does not include a separate chat/API-level system prompt. The hardware execution environment is stated and is the same as the main HeCBench CUDA optimization project:

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm

## Input Prompt: Optimized

```text
請幫我優化這份程式碼，需輸出相同的資料，請給我完整的程式碼。
```

## Input Prompt: Env-Optimized

```text
以下是我的環境，請再進行優化，給我完整的程式碼。
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
```

## Benchmarks Covered

- `cc-cuda`
- `floydwarshall-cuda`
- `floydwarshall2-cuda`
- `gc-cuda`
- `mis-cuda`
- `merge-cuda`
- `quicksort-cuda`
- `sortKV-cuda`
- `bitonic-sort-cuda`
- `split-cuda`

