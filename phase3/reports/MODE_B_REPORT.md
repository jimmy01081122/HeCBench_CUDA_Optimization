# Phase 3 Mode B Robust Baseline Report

Mode B has been started, but guided optimization has not begun. Per the Mode B protocol, the required robust baselines were remeasured first and Round 1 optimization is stopped pending human review.

## Execution

- softmax-cuda: Slurm job 949640 on gn1221.twcc.ai
- topk-cuda: Slurm job 949641 on gn1102.twcc.ai
- shmembench-cuda: Slurm job 949642 on gn1102.twcc.ai

All GPU execution was performed through Slurm. No GPU benchmark binary was run directly on the login node.

## Robust Baseline Summary

### softmax-cuda

- Baseline definition: existing optimized implementation, `impl=1`.
- Official cases: slice sizes 128, 256, 784, 1024, 2048.
- Trials: 3 per case.
- Correctness: PASS for all five official cases.
- Measurement validity: VALID for all five cases.
- Auditor: PASS.
- Conclusion: SUCCESS; ready for human review before any optimization round.

### topk-cuda

- Baseline definition: official topk implementation before Mode B modification.
- Official cases: 14 hidden_size/topk combinations.
- Trials: 7 per case, run up front because Mode A exposed high-CV pseudo-speedups.
- Correctness: PASS for all 14 cases.
- Measurement validity: 12 VALID, 2 CAUTION, 0 NOISY.
- Auditor: PASS.
- Conclusion: SUCCESS; ready for human review before any optimization round.

### shmembench-cuda

- Official baseline: `variant=original`, `block_size=256`.
- Diagnostic cases: `block_size=128`, `512`, `1024`.
- Trials: 3 per case.
- Official correctness: PASS for `block_size=256`.
- Official measurement validity: VALID.
- Diagnostic status: `128` and `512` failed checksum and are labeled `DIAGNOSTIC_FAIL`; `1024` failed build due shared-memory usage and is labeled `DIAGNOSTIC_FAIL`.
- Auditor: PASS.
- Conclusion: SUCCESS for the official Mode B baseline; diagnostics preserved and excluded from speedup claims.

## Human Review Gate

No Mode B optimization round has been submitted. Before Round 1, the reviewer should inspect:

- `phase3/softmax-cuda/mode_B_human_guided/robust_baseline/results.csv`
- `phase3/topk-cuda/mode_B_human_guided/robust_baseline/results.csv`
- `phase3/shmembench-cuda/mode_B_human_guided/robust_baseline/results.csv`

If approved, the next step is to write `rounds/round_1/plan.md` for each benchmark and wait for explicit human approval before any optimization sbatch.
