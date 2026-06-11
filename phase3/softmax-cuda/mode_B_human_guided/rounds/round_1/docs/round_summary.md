# softmax-cuda Mode B Round 1 Summary

Slurm job: 949687 on `gn1221.twcc.ai`

Candidate: `impl2_block_cached_exp_compound`

Attribution: this is a compound candidate combining block-per-slice row parallelism and shared-memory cached exponentials. Results must not be attributed solely to cached exponentials.

Profiler status: `NOT_RUN`

## Per-Slice Outcome

| slice | paired impl=1 mean ms | impl=2 mean ms | candidate correctness | measurement_validity | result_type | speedup notes |
|---:|---:|---:|---|---|---|---|
| 128 | 0.135152 | 0.554750 | PASS 3/3 | VALID | REGRESSION | 0.243x-0.244x, slower by far more than 1% |
| 256 | 0.323384 | 0.594147 | PASS 2/3, FAIL 1/3 | INVALID | INVALID | speedup=n/a because one official trial failed correctness |
| 784 | 1.434026 | 1.108087 | PASS 3/3 | VALID | KERNEL_OPT | 1.292x-1.296x per paired trial |
| 1024 | 2.068956 | 1.300902 | PASS 3/3 | VALID | KERNEL_OPT | 1.590x-1.591x per paired trial |
| 2048 | 2.212359 | 1.680560 | PASS 3/3 | VALID | KERNEL_OPT | 1.300x-1.325x per paired trial |

## Decision

The candidate cannot be promoted as a full valid optimization because one official slice (`slice=256`) has a correctness failure. The failed trial is preserved in `results.csv` with `correctness_status=FAIL`, `measurement_validity=INVALID`, `result_type=INVALID`, and `speedup_claim_valid=false`.

The candidate also regresses `slice=128`, so successful larger-slice results must be treated as partial improvement only. No average is used to hide the failure or regression.

## Artifacts

- `results.csv`: per-trial paired `impl=1` and `impl=2` rows using the approved Round 1 schema.
- `raw/`: raw stdout and stderr for every slice/trial/implementation.
- `build.log`: build output.
- `auditor_report.csv`: self-consistency auditor output.
- `result.out` and `result.err`: Slurm stdout/stderr.

Auditor status: all checks PASS.
