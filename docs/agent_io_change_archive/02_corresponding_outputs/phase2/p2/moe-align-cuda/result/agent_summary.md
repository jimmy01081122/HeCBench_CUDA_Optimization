# moe-align CUDA Optimization Summary

## Baseline

- Job: 948013
- Node: gn1222
- Correctness: PASS on 30/30 parameter cases
- Metric: 19.504113 us mean latency

## Optimization Submissions

| Tag | Job | Node | Correctness | Mean latency (us) | Speedup | Status |
|---|---:|---|---|---:|---:|---|
| opt1 | 948014 | gn1222 | PASS 30/30 | 16.899124 | 1.154149x | accepted |
| opt2 | 948015 | gn1222 | PASS 30/30 | 22.465978 | 0.868162x | rejected |
| opt3 | 948016 | gn1222 | PASS 30/30 | 16.992179 | 1.147829x | rejected |
| opt4 | 948019 | gn1222 | PASS 30/30 | 15.939143 | 1.223661x | accepted |

## Final Result

- Best accepted submission: opt4 / job 948019
- Final correctness: PASS on 30/30 parameter cases
- Final metric: 15.939143 us mean latency
- Final speedup: 1.223661x
- Raw outputs: `baseline-948013.txt`, `opt1-948014.txt`, `opt2-948015.txt`, `opt3-948016.txt`, `opt4-948019.txt`
- Slurm stdout/stderr files are saved in this result directory for each job.
- Per-case CSV with explicit correctness/status fields: `comparison.csv`

## Best Strategy

The best valid strategy caches the non-small-path `cumsum` device workspace instead of allocating and freeing it on every measured wrapper call, and removes a redundant `cudaMemset` because the alignment kernel writes the full prefix buffer before the sort kernel consumes it. Attempts to broaden the small-batch path to 128 experts and reduce fill threads both passed correctness but regressed mean latency, so they were rejected.




Implemented and benchmarked the CUDA optimization workflow for `moe-align-cuda`.

Best accepted result is `opt4`:

- Baseline job `948013` on `gn1222`: PASS 30/30, mean `19.504113 us`
- Final job `948019` on `gn1222`: PASS 30/30, mean `15.939143 us`
- Speedup: `1.223661x`

What changed in [main.cu](/home/r14525078/p2/HeCBench/src/moe-align-cuda/main.cu:349): cached the non-small-path `cumsum` device buffer across calls and removed a redundant `cudaMemset`; the alignment kernel writes the prefix buffer before the sort kernel reads it.

Artifacts written under [result](/home/r14525078/p2/HeCBench/src/moe-align-cuda/result):

- `baseline-948013.txt`
- `opt1-948014.txt`
- `opt2-948015.txt`
- `opt3-948016.txt`
- `opt4-948019.txt`
- `comparison.csv`
- `agent_summary.md`

I also added [run_moe_align_cuda.slurm](/home/r14525078/p2/HeCBench/src/moe-align-cuda/run_moe_align_cuda.slurm) for the required Slurm-only runs.


TOKENS : 49,271 used