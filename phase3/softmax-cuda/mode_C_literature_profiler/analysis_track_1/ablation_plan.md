# Ablation Plan For Submission 2

Submission 2 should be used only if the expected evidence value justifies
spending one of the remaining optimization submissions. All options below must
preserve `impl=0/1/2/3` and should add a new implementation ID.

## Option A: Reduction-Structure Ablation

Purpose:

- determine whether the Submission 1 improvement is associated with the
  warp/cross-warp reduction structure compared with `impl=2` / `softMax3`.

Required source change:

- add an ablation implementation that keeps the same cached-exp layout and
  block-per-slice launch shape, while isolating reduction structure differences
  as much as possible.

Hypothesis tested:

- if this ablation matches `impl=4`, reduction structure is a plausible
  contributor.
- if it matches `impl=3`, the observed gain may come from another resource or
  code-generation effect.

Target slices:

- 784
- 1024
- 2048

Expected benefit:

- high explanatory value for 784 and 1024.
- may clarify why 2048 is only measurement-equivalent.

Correctness risk:

- moderate, because reductions are numerically sensitive but tolerance is 1e-3.

Regression risk:

- moderate. A reduction-only ablation may perform worse than `impl=4`.

Should count as Submission 2:

- yes, if implemented and benchmarked on official cases.

Worth doing:

- yes, especially if profiler is unavailable or inconclusive.

## Option B: Cached-Exp Attribution Ablation

Purpose:

- determine whether cached exponentials independently contribute to large-slice
  improvement.

Required source change:

- add a candidate that keeps block-per-slice parallelism and the reduction
  structure under test, but avoids shared-memory cached exponentials by
  recomputing exponentials in the output pass.

Hypothesis tested:

- if no-cache is slower, cached exponentials may be helpful.
- if no-cache is similar or faster, shared-memory cache footprint or traffic may
  be limiting.

Target slices:

- 784
- 1024
- 2048

Expected benefit:

- strong attribution value.
- possible performance benefit if reduced shared-memory pressure outweighs
  duplicate `expf` cost.

Correctness risk:

- low to moderate. Same softmax formula, but two `expf` evaluations can produce
  small numeric differences.

Regression risk:

- high, because duplicate `expf` can be expensive.

Should count as Submission 2:

- yes, if implemented and benchmarked.

Worth doing:

- worth doing if final report needs causal attribution more than another speedup
  attempt. It may be less attractive if the main goal is immediate additional
  speedup.

## Option C: 2048-Specific Optimization

Purpose:

- investigate why `slice=2048` only reaches `speedup_vs_impl3=1.008x`, which is
  measurement-equivalent.

Required source change:

- add a 2048-specific path, likely with a different block size, shared-memory
  layout, or per-thread element count.

Hypothesis tested:

- 2048 may be limited by a different resource balance than 784 and 1024.

Target slices:

- primary: 2048
- guardrail: 784 and 1024 must not regress if dispatch policy includes them
- small slices must stay on `impl=1`

Expected benefit:

- possible additional speedup for 2048.
- could convert Submission 1's measurement-equivalent 2048 result into a valid
  improvement.

Correctness risk:

- moderate.

Regression risk:

- moderate to high if shared dispatch is changed. Lower if implemented as a
  strict 2048-only branch with fallback to existing `impl=4` for 784/1024.

Should count as Submission 2:

- yes.

Worth doing:

- worth doing only after profiler or ablation suggests a 2048-specific cause.
  Blind 2048 tuning risks spending Submission 2 with little explanatory value.

## Option D: Block-Size / Resource Tuning

Purpose:

- tune the large-slice path without changing small-slice behavior.

Required source change:

- add shape-aware block-size variants, such as separate large-slice kernels or
  launch policies for 784, 1024, and 2048.

Hypothesis tested:

- the fixed 256-thread block is not optimal for all large official slices.

Target slices:

- 784
- 1024
- 2048

Expected benefit:

- possible speedup on one or more large slices.

Correctness risk:

- low to moderate if the math is unchanged.

Regression risk:

- moderate. A tuned block size can improve one slice and regress another.

Should count as Submission 2:

- yes.

Worth doing:

- worth doing after profiler evidence shows occupancy, register, shared-memory,
  or memory-throughput constraints that block-size tuning can address. Without
  profiler evidence, this is close to blind tuning.

## Comparison

| option | evidence value | speedup potential | risk | recommendation |
|---|---|---|---|---|
| A: reduction-structure ablation | high | medium | moderate | best if profiler is unavailable or inconclusive |
| B: cached-exp attribution ablation | high | low to medium | high regression risk | useful for final explanation |
| C: 2048-specific optimization | medium | medium | moderate to high | wait for profiler evidence |
| D: block-size/resource tuning | medium | medium | moderate | profiler-informed only |

## Preferred Ablation Direction

If no profiler is approved, choose Option A for Submission 2. It best connects
Submission 1's accepted result to an evidence-backed explanation while avoiding
pure blind tuning.
