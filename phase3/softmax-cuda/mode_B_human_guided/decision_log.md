# Decision Log

- Robust baseline execution: Approved by protocol; no optimization round executed.
- Round 1: Approved candidate `impl2_block_cached_exp_compound`; executed as Slurm job 949687. Not promoted because `slice=256` has a correctness failure and `slice=128` regresses.
- Round 2: Approved candidate `impl3_shape_dispatch_impl1_small_impl2_large`; executed as Slurm job 949703. All official slices passed correctness. Small slices dispatch to unchanged `impl=1` and are measurement-equivalent; large slices dispatch to unchanged `impl=2` and show valid per-slice dispatch-policy speedups.
