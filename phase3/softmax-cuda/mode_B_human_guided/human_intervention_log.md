# Human Intervention Log

- Robust baseline measurement only. Optimization is stopped pending human review.
- Round 1 initial proposal required revision. Human reviewer approved the revised compound candidate with paired same-job `impl=1` baseline requirements. Execution completed, but candidate is not eligible for full promotion due correctness failure on an official slice.
- Round 2 was requested after Round 1 rejection as full replacement. Human-approved direction was shape-aware dispatch: `slice=128/256 -> impl=1`, `slice=784/1024/2048 -> impl=2`. Execution completed through Slurm with all official slices PASS.
