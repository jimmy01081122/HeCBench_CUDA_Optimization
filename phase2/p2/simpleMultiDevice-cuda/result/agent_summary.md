# simpleMultiDevice-cuda Agent Summary

## Baseline

- Status: PASS
- Job id: 948649
- Node: gn1228
- GPUs: 4
- Result file: `baseline_948649.txt`
- Slurm stdout/stderr: `simpleMultiDevice-cuda-948649.out`, `simpleMultiDevice-cuda-948649.err`
- Average total_us: 5621.596680
- Correctness diff: 5.724980E-07

## Optimization Submissions

| Submission | Job id | Node | Strategy | Correctness | total_us | h2d_us | kernel_us | d2h_us | Speedup | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 948649 | gn1228 | Original source, `ARCH=sm_70 MAX_GPU=4` | PASS, diff 5.724980E-07 | 5621.596680 | n/a | n/a | n/a | 1.000000 | Baseline |
| opt1_block_reduce | 948652 | gn1228 | Block-level reduction, 1024 blocks/GPU, fewer D2H partials, phase event timing | PASS, diff 1.427544E-07 | 5561.413086 | 5444.657715 | 62.834869 | 10.396240 | 1.010822 | Accepted |
| opt2_write_combined | 948654 | gn1228 | Added write-combined pinned input allocation on top of opt1 | PASS, diff 1.427544E-07 | 5566.998047 | 5443.934570 | 62.699677 | 10.164328 | 1.009808 | Rejected, slower than opt1 |

## Final Result

- Accepted result: `opt1_block_reduce`
- Final correctness: PASS
- Final diff: 1.427544E-07
- Final total_us: 5561.413086
- Final h2d_us: 5444.657715
- Final kernel_us: 62.834869
- Final d2h_us: 10.396240
- Final speedup vs baseline: 1.010822x

## Best Strategy

The baseline reduction launched only 32 blocks per GPU and copied one partial sum per thread back to the host. The accepted version increases parallelism to 1024 blocks per GPU and performs an in-block shared-memory reduction, so each block writes one partial. This reduces kernel time and D2H volume while preserving the original H2D transfer and full CPU correctness check. The final metric remains H2D-copy-limited, so the total speedup is modest.

TOKENS : 37,194 used 