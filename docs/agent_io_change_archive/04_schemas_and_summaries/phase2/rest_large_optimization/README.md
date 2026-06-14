# Rest.md Phase2 Large Optimization Archive

Source file: `/home/a/rest.md`

This folder organizes the additional Phase2 benchmark summaries that were not part of the original ten-benchmark prompt-constraint table. The source reports P1/P2/P3 runs for ten benchmarks:

`adam-cuda`, `adjacent-cuda`, `dropout-cuda`, `filter-cuda`, `minmax-cuda`, `nonzero-cuda`, `randomAccess-cuda`, `reverse-cuda`, `scan-cuda`, and `topk-cuda`.

## Files

| File | Purpose |
|---|---|
| `rest_phase2_summary.csv` | Machine-readable benchmark-level summary extracted from `/home/a/rest.md`. |
| `REST_PHASE2_ORGANIZED_SUMMARY_ZH.md` | Chinese organized interpretation for paper/report use. |

## Archive Interpretation

The source gives compact summaries rather than full raw Slurm logs. Therefore:

- system prompt: `N/A` unless present elsewhere in the archive.
- raw output: summarized from `/home/a/rest.md`.
- source diff: `N/A` unless present elsewhere in the project.
- correctness: copied from `/home/a/rest.md`.
- result type: copied from `/home/a/rest.md`, with `BENCHMARK_AWARE_OPT` kept separate from conventional `KERNEL_OPT`.

Benchmark-centric versions are also available under:

`docs/agent_io_change_archive/benchmark_view/<benchmark>/`
