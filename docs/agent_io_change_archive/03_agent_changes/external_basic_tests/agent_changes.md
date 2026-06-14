# External Basic Tests: Agent Changes

Source: `/home/a/data.md`

The provided record states that another person asked an AI agent to return complete optimized code for each benchmark, first with a generic optimization prompt and then with a V100/Slurm environment-aware prompt. However, the actual modified source files, diffs, patch summaries, and raw agent responses are not included in the provided data.

| Benchmark | Optimized Agent Change | Env-Optimized Agent Change | Patch / Diff Available | Notes |
|---|---|---|---|---|
| `cc-cuda` | N/A | N/A | N/A | Both optimized versions failed. |
| `floydwarshall-cuda` | N/A | N/A | N/A | Reported extreme speedup is suspicious; likely semantic or execution error. |
| `floydwarshall2-cuda` | N/A | N/A | N/A | Optimized and env-optimized versions regress heavily. |
| `gc-cuda` | N/A | N/A | N/A | Graph optimization likely broke convergence/frontier/atomic semantics. |
| `mis-cuda` | N/A | N/A | N/A | Graph optimization likely broke convergence/frontier/atomic semantics. |
| `merge-cuda` | N/A | N/A | N/A | Source analysis attributes improvement to regular memory access and parallelism, but exact code changes are unavailable. |
| `quicksort-cuda` | N/A | N/A | N/A | Env-optimized failed; optimized timing is near measurement-equivalent. |
| `sortKV-cuda` | N/A | N/A | N/A | Source analysis suggests memory coalescing/block tuning/global memory reduction, but exact changes are unavailable. |
| `bitonic-sort-cuda` | N/A | N/A | N/A | Source analysis suggests regular sorting kernels benefit most; exact changes are unavailable. |
| `split-cuda` | N/A | N/A | N/A | Env-optimized regressed, likely due to over-tuning occupancy or ignoring memory-bound behavior. |

## Missing Information

- Original input source files: N/A
- AI-produced complete optimized code: N/A
- AI-produced complete env-optimized code: N/A
- Raw execution logs: N/A
- Correctness output per benchmark: N/A except explicit `fail` labels and source narrative
- Patch summaries: N/A
- Trial count / variance / CV: N/A

