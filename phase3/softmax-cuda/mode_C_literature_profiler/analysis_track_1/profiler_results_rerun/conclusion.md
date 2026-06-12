# Profiler Rerun Conclusion

## Task Status

Profiler rerun completed.

This task was profiler-only:

- no Submission 2 was started
- no optimization candidate was added
- source code was not modified
- `./main` was not run on the login node
- profiler timing is not official timing

## Current Rerun

Slurm job:

- `950686`

Execution constraints:

- `repeat_for_profiler=10`
- `launch_skip=2`
- `launch_count=1`
- slices profiled: 784, 1024, 2048
- implementations profiled: `impl=3`, `impl=4`
- official_timing_used=false

Result:

- all 6 profiler rows are `AVAILABLE`
- `.ncu-rep` files were generated
- `ncu --import --page details` successfully recovered metrics

## Previous Failed / Not-Accepted Cases

Previous feasibility failure:

- Job `950683`
- Result: failed before profiler execution
- Reason: existing binary was missing on the compute node
- Interpretation: this was not a profiler availability result

Previous profiler-only run not accepted:

- Job `950678`
- Result: `PARTIAL`
- Reason: Nsight Compute attached and generated `.ncu-rep` files, but direct
  stdout from `--set default` did not expose the requested metrics
- Interpretation: not accepted as profiler evidence because metrics were not
  extracted

Corrective action in this rerun:

- built the existing source inside sbatch to ensure `main` exists
- requested explicit metrics
- imported each `.ncu-rep` via `ncu --import --page details`
- parsed metrics into `profiler_summary.csv`

## Rerun Evidence

| slice | impl | status | registers/thread | dynamic shared memory | waves/SM | profiler duration |
|---:|---:|---|---|---|---|---|
| 784 | 3 | AVAILABLE | 18 | 4.16 Kbyte/block | 156.25 | 1.09 ms |
| 784 | 4 | AVAILABLE | 18 | 3.26 Kbyte/block | 156.25 | 938.24 us |
| 1024 | 3 | AVAILABLE | 18 | 5.12 Kbyte/block | 156.25 | 1.29 ms |
| 1024 | 4 | AVAILABLE | 18 | 4.22 Kbyte/block | 156.25 | 1.20 ms |
| 2048 | 3 | AVAILABLE | 18 | 9.22 Kbyte/block | 78.12 | 1.66 ms |
| 2048 | 4 | AVAILABLE | 18 | 8.32 Kbyte/block | 78.12 | 1.64 ms |

## Direction

Recommended next direction after human audit:

- `Submission 2 = reduction-structure ablation`

Reason:

- profiler metrics are now available for all representative large-slice cases
- `impl=4` uses less dynamic shared memory than `impl=3`
- profiler timing is diagnostic only and must not be used as official speedup
- ablation remains the clearest next step for attribution without overclaiming
  profiler-supported causality

Submission 2 remains deferred until explicit human approval.
